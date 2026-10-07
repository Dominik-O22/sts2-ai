"""The inference side of the sim's threaded run loop (`sim::runloop`,
docs/run-env.md). Rust worker threads step every env on their own and
put the envs that wait for the network into a ready set; `Loop.step`
takes them, answers the combat rows with the combat policy (greedy, or
the pilot's turn search in the fights of `search_kinds`) and the run rows
through `decide`, and posts the answers. The workers step the envs
answered while the next rows go through the network, so the sim and the
GPU overlap instead of taking turns.

A turn search is a job the loop carries over its next rounds (`Search`):
the root envs stay held while their copies' rollouts advance one step a
round, every job's rows forwarded together after the round's live rows,
so a search holds nothing else up and the live envs keep stepping while
it runs. The roots are posted when the job's picks are in.

Runs take their seeds from a shared queue (`sim::env::Seeds`): an env
whose run ends takes the next unplayed seed, and leaves the batch when
none is left, so no env plays filler runs and an evaluation ends with its
last counted run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from sts2ai.env import Envs, Layout, Taken
from sts2ai.exactsearch import Hybrid
from sts2ai.forecast import Forecaster, Read
from sts2ai.model import Net, masked_logits
from sts2ai.search import CHOOSE_CHUNK, Rollout, forward, outputs, picks, spread

# Picks options for the envs waiting at run decisions, given their rows.
Decide = Callable[[list[int], np.ndarray, np.ndarray], np.ndarray]
# Sees a round's events (`Taken.ends`, `Taken.ended`, ...) before its decisions.
OnEvents = Callable[[Taken], None]

# Copies the searches in flight may hold between them: their rows share
# one pinned pool, about 17 KB a row.
MAX_COPIES = 65536


class Search:
    """One turn search over `roots` (envs the loop holds): `n` copies a
    root (`Envs.sim.fork`), every legal first action with its share, the
    policy's own `own` action per root to beat (`search.picks`)."""

    def __init__(self, sim, roots: list[int], mask: np.ndarray, own: np.ndarray, n: int, groups: int, seed: int):
        self.roots, self.own, self.n = roots, own, n
        self.forks = sim.fork(roots, n, groups, seed)
        self.first = np.concatenate([spread(np.flatnonzero(m), n) for m in mask])
        self.second = (np.full(len(self.first), -1), np.zeros(len(self.first), np.int64))
        self.rollout = Rollout(self.forks, self.first, second=self.second)

    def __len__(self) -> int:
        return len(self.forks)

    def picks(self) -> np.ndarray:
        return picks(self.first, self.rollout.score, self.second, self.own, self.n)


class Loop:
    """Drives `envs`' threaded loop (`Envs.start_loop`) with the combat
    policy. With `search` copies, fights of `search_kinds` play the
    pilot's turn search instead (`Search`, `groups` shuffles, the copies'
    dice drawn from `seed` and the round); with `hybrid` (top lines,
    playouts, race size or 0) the exact search's lines picked by
    playouts (`exactsearch.Hybrid`), which runs within the round. With a
    `forecast`, each map step's row gets the forecast filled in before
    `decide` sees it (`Forecaster.fill`); `read` holds what the last
    round read, by row of that round's decision envs.

    `min_rows` is how many ready envs a round waits for (a quarter of the
    batch by default, fewer as envs leave), `timeout_ms` how long at most;
    a round with searches in flight does not wait."""

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
        # Searches in flight, and roots waiting for room in the pool:
        # (env, its mask row, its own action).
        self.searches: list[Search] = []
        self.waiting: list[tuple[int, np.ndarray, int]] = []
        self.pool: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
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
        """One round: take the ready envs, answer them, post; advance the
        searches in flight. Returns what was taken, with the events since
        the last round; `active` 0 there means every env has left and
        the last events are in. `events` sees them before `decide` does:
        a worker plays an env whose run ended on to the next run's first
        decision, so the end and that decision come in the same round."""
        envs = self.envs
        busy = bool(self.searches or self.waiting)
        t = envs.take(1 if busy else max(1, min(self.min_rows, self.active // 2)), 0 if busy else self.timeout_ms)
        self.active = t.active
        if events is not None:
            events(t)
        actions = options = np.empty(0, dtype=np.int64)
        combat = t.combat
        if t.combat:
            floats = torch.from_numpy(t.floats).to(self.device, non_blocking=True)
            ids = torch.from_numpy(t.ids).to(self.device, non_blocking=True)
            mask = torch.from_numpy(t.mask).to(self.device, non_blocking=True)
            logits, _ = self.combat(floats, ids)
            masked = masked_logits(logits.float(), mask)
            actions = masked.argmax(dim=1).cpu().numpy()
            if self.hybrid is not None:
                self.overrule(t, actions, masked)
            elif self.search:
                combat, actions = self.hold(t, actions)
        if t.decision:
            if self.forecast is not None:
                self.read = self.forecast.fill(t.run_floats, envs.forecast(t.decision))
            options = decide(t.decision, t.run_floats, t.run_ids)
            self.decisions += len(t.decision)
        if combat or t.decision:
            envs.post(combat, actions, t.decision, options)
        if self.search:
            self.advance()
        self.steps += 1
        return t

    def hold(self, t: Taken, actions: np.ndarray) -> tuple[list[int], np.ndarray]:
        """Keeps the rows fighting a searched kind out of the post, as
        roots of a search (every row taken has more than one legal
        action, the workers take lone ones themselves); returns the rows
        posted now."""
        sim = self.envs.sim
        rows = [k for k, env in enumerate(t.combat) if sim.fight(env)[1] in self.search_kinds]
        if not rows:
            return t.combat, actions
        self.searched += len(rows)
        self.waiting += [(t.combat[k], t.mask[k].copy(), int(actions[k])) for k in rows]
        keep = np.ones(len(t.combat), bool)
        keep[rows] = False
        return [env for env, k in zip(t.combat, keep) if k], actions[keep]

    def advance(self) -> None:
        """Starts the searches the pool has room for, advances every search
        in flight by one step, all their rows through the network together,
        and posts the roots of the ones done."""
        held = sum(len(s) for s in self.searches)
        # As many waiting roots as the pool has room for, and one at least
        # when nothing is in flight, so a backlog bigger than the pool drains.
        fit = min(len(self.waiting), max(0, (MAX_COPIES - held) // self.search))
        if fit or (self.waiting and not self.searches):
            fit = max(fit, 1)
            roots, masks, own = zip(*self.waiting[:fit])
            del self.waiting[:fit]
            self.searches.append(Search(self.envs.sim, list(roots), np.stack(masks), np.array(own), self.search, self.groups, self.seed << 32 | self.steps))
            held += len(self.searches[-1])
        if not self.searches:
            return
        floats, ids, mask = self.buffers(held)
        f, i, m = floats.numpy(), ids.numpy(), mask.numpy()
        # Each search's rows, packed one after the other: (search, first row, rows).
        spans, at = [], 0
        for s in self.searches:
            if n := s.rollout.rows(f[at:], i[at:], m[at:]):
                spans.append((s, at, n))
                at += n
        if at:
            logits, values = forward(self.combat, self.device, floats[:at], ids[:at], CHOOSE_CHUNK)
            picks = np.concatenate([s.rollout.picks + a for s, a, _ in spans])
            sampled, value = outputs(logits, values, mask[:at].to(self.device, non_blocking=True), picks)
            k = 0
            for s, a, n in spans:
                live = len(s.rollout.picks)
                s.rollout.apply(sampled[k : k + live], value[a : a + n], m[a : a + n])
                k += live
        done = [s for s in self.searches if s.rollout.done]
        self.searches = [s for s in self.searches if not s.rollout.done]
        for s in done:
            self.envs.post(s.roots, s.picks(), [], np.empty(0, dtype=np.int64))

    def buffers(self, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The searches' pinned row pool, grown to `rows`."""
        if self.pool is None or len(self.pool[0]) < rows:
            L = Layout.load()
            pin = self.device.type == "cuda"
            self.pool = (
                torch.empty((rows, L.n_floats), pin_memory=pin),
                torch.empty((rows, L.n_ids), dtype=torch.int64, pin_memory=pin),
                torch.empty((rows, L.n_actions), dtype=torch.bool, pin_memory=pin),
            )
        return self.pool

    def overrule(self, t: Taken, actions: np.ndarray, masked: torch.Tensor) -> None:
        """Replace the greedy `actions` by the hybrid search's picks in the
        rows fighting a searched kind; `masked` holds the policy's masked
        logits for every row."""
        sim = self.envs.sim
        rows = [k for k, env in enumerate(t.combat) if sim.fight(env)[1] in self.search_kinds]
        if not rows:
            return
        self.searched += len(rows)
        roots = [t.combat[k] for k in rows]
        # The hybrid indexes by env, as its other callers do.
        probs = np.zeros((self.envs.n, masked.shape[1]), dtype=np.float32)
        probs[roots] = masked[rows].softmax(dim=1).cpu().numpy()
        by_env = np.zeros(self.envs.n, dtype=np.int64)
        by_env[roots] = actions[rows]
        self.hybrid.choose(self.combat, self.device, roots, probs, by_env)
        actions[rows] = by_env[roots]
        # Its per-search stats are for exactsearch's report; a run keeps none.
        self.hybrid.planner.searched.clear()
