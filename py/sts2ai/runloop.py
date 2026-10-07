"""The inference side of the sim's threaded run loop (`sim::runloop`,
docs/run-env.md). Rust worker threads step every env on their own and
put the envs that wait for the network into a ready set; `Loop.step`
takes them, answers the combat rows with the combat policy (greedy, or
the pilot's turn search in the fights of `search_kinds`) and the run rows
through `decide`, and posts the answers. The workers step the envs
answered while the next rows go through the network, so the sim and the
GPU overlap instead of taking turns.

Runs take their seeds from a shared queue (`sim::env::Seeds`): an env
whose run ends takes the next unplayed seed, and leaves the batch when
none is left, so no env plays filler runs and an evaluation ends with its
last counted run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from sts2ai.env import Envs, Taken
from sts2ai.exactsearch import Hybrid
from sts2ai.forecast import Forecaster, Read
from sts2ai.model import Net, masked_logits
from sts2ai.search import choose

# Picks options for the envs waiting at run decisions, given their rows.
Decide = Callable[[list[int], np.ndarray, np.ndarray], np.ndarray]
# Sees a round's events (`Taken.ends`, `Taken.ended`, ...) before its decisions.
OnEvents = Callable[[Taken], None]


class Loop:
    """Drives `envs`' threaded loop (`Envs.start_loop`) with the combat
    policy. With `search` copies, fights of `search_kinds` play the
    pilot's turn search instead (`search.choose`, `groups` shuffles, the
    copies' dice drawn from `seed` and the step); with `hybrid` (top
    lines, playouts, race size or 0) the exact search's lines picked by
    playouts (`exactsearch.Hybrid`). With a `forecast`, each map step's
    row gets the forecast filled in before `decide` sees it
    (`Forecaster.fill`); `read` holds what the last step read, by row of
    that step's decision envs.

    `min_rows` is how many ready envs a step waits for (a quarter of the
    batch by default, fewer as envs leave), `timeout_ms` how long at most."""

    def __init__(
        self,
        combat: Net,
        device: torch.device,
        envs: Envs,
        search: int = 0,
        search_kinds: frozenset[str] = frozenset({"Elite", "Boss"}),
        groups: int = 4,
        seed: int = 0,
        hybrid: tuple[int, int, int] | None = None,
        forecast: Forecaster | None = None,
        min_rows: int | None = None,
        timeout_ms: int = 5,
        workers: int = 0,
    ):
        self.combat, self.device, self.envs = combat, device, envs
        self.forecast = forecast
        self.read: Read | None = None
        self.search, self.search_kinds, self.groups, self.seed = search, search_kinds, groups, seed
        self.hybrid = Hybrid(envs, *hybrid, seed=seed) if hybrid else None
        self.min_rows = max(1, envs.n // 4) if min_rows is None else min_rows
        self.timeout_ms = timeout_ms
        self.active = envs.n
        self.steps = 0
        self.decisions = 0
        self.searched = 0
        self.stopped = False
        self._combat_steps = 0
        envs.start_loop(workers)

    @property
    def searching(self) -> bool:
        return bool(self.search) or self.hybrid is not None

    @property
    def combat_steps(self) -> int:
        """Combat actions taken so far, lone legal ones included."""
        return self._combat_steps if self.stopped else self.envs.sim.loop_counts()[0]

    def stop(self) -> None:
        """Ends the sim's loop, keeping the counts."""
        self._combat_steps = self.combat_steps
        self.stopped = True
        self.envs.sim.stop_loop()

    @torch.no_grad()
    def step(self, decide: Decide, events: OnEvents | None = None) -> Taken:
        """One round: take the ready envs, answer them, post. Returns what
        was taken, with the events since the last round; `active` 0 there
        means every env has left and the last events are in. `events` sees
        them before `decide` does: a worker plays an env whose run ended on
        to the next run's first decision, so the end and that decision come
        in the same round."""
        envs = self.envs
        t = envs.take(max(1, min(self.min_rows, self.active // 2)), self.timeout_ms)
        self.active = t.active
        if events is not None:
            events(t)
        actions = options = np.empty(0, dtype=np.int64)
        if t.combat:
            floats = torch.from_numpy(t.floats).to(self.device, non_blocking=True)
            ids = torch.from_numpy(t.ids).to(self.device, non_blocking=True)
            mask = torch.from_numpy(t.mask).to(self.device, non_blocking=True)
            logits, _ = self.combat(floats, ids)
            masked = masked_logits(logits.float(), mask)
            actions = masked.argmax(dim=1).cpu().numpy()
            if self.searching:
                self.overrule(t, actions, masked)
        if t.decision:
            if self.forecast is not None:
                self.read = self.forecast.fill(t.run_floats, envs.forecast(t.decision))
            options = decide(t.decision, t.run_floats, t.run_ids)
            self.decisions += len(t.decision)
        if t.combat or t.decision:
            envs.post(t.combat, actions, t.decision, options)
        self.steps += 1
        return t

    def overrule(self, t: Taken, actions: np.ndarray, masked: torch.Tensor) -> None:
        """Replace the greedy `actions` by the turn search's picks in the
        rows fighting a searched kind; `masked` holds the policy's masked
        logits for every row. Every row taken has more than one legal
        action (the workers take lone ones themselves)."""
        sim = self.envs.sim
        rows = [k for k, env in enumerate(t.combat) if sim.fight(env)[1] in self.search_kinds]
        if not rows:
            return
        self.searched += len(rows)
        roots = [t.combat[k] for k in rows]
        if self.hybrid is not None:
            # The hybrid indexes by env, as its other callers do.
            probs = np.zeros((self.envs.n, masked.shape[1]), dtype=np.float32)
            probs[roots] = masked[rows].softmax(dim=1).cpu().numpy()
            by_env = np.zeros(self.envs.n, dtype=np.int64)
            by_env[roots] = actions[rows]
            self.hybrid.choose(self.combat, self.device, roots, probs, by_env)
            actions[rows] = by_env[roots]
            # Its per-search stats are for exactsearch's report; a run keeps none.
            self.hybrid.planner.searched.clear()
            return
        seed = self.seed << 32 | self.steps
        actions[rows] = choose(self.combat, self.device, sim, roots, t.mask[rows], actions[rows], self.search, self.groups, seed)
