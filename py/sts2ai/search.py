"""Turn search: play copies of a fight to the end of the turn and score
them. Used by the advisor's plan, by `searcheval`, and by `runplay`, which
plays many fights' decisions as the pilot does (`choose`).

Each copy starts with a given first action; the policy samples the rest of
the turn. A copy's score is the shaped reward it collected plus the value
head where the next turn starts (a finished fight already paid its
terminal reward). The forks shape rewards from the fight's start, as the
value head's targets were, so a copy that won and one still going are
scored on one scale (`sim::env::Forks`).

An opening scored by the mean of its copies is scored by the policy's
average continuation, and the player will not play that: they search
again at the next decision. With `second`, copies that saw the same thing
after their first action (one hand, one board: `observe_unique`) also try
every legal second action, and `openings` scores an opening by its best
second action per thing seen, averaged over those: best over the player's
choices, mean over chance. A turn-1 Entomancer hand showed why: the policy
attacks after Defend and feeds Personal Hive, so Defend averaged below
passing, while Defend then passing was the best line by far.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from sts2ai.env import Layout
from sts2ai.model import Net, masked_logits

# Actions per player turn searched; a runaway plan is cut after this many
# per turn.
MAX_PLAN_STEPS = 40
# Copies per network call: a search over many fights at once is too big for
# one batch on the GPU.
CHUNK = 16384
# `choose`'s, smaller: runplay shares the GPU with training, and a quarter
# of `CHUNK` keeps its peak near a gigabyte.
CHOOSE_CHUNK = 4096
# Copies each second action needs before `second` tries them all for what a
# first action led to; fewer and the max over them picks luck, so those
# copies keep the policy's play.
SECOND_MIN = 8
# Plan scores closer than this are value-head noise: the policy's pick
# keeps the top line, and the search only overrules it by a clear margin.
# The unit is half an HP fraction, so 0.02 is about three HP.
PLAN_MARGIN = 0.02


def forward(policy: Net, device: torch.device, floats: torch.Tensor, ids: torch.Tensor, chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy over a batch of any size, in `chunk`-sized pieces."""
    parts = [
        policy(floats[i : i + chunk].to(device, non_blocking=True), ids[i : i + chunk].to(device, non_blocking=True)) for i in range(0, len(floats), chunk)
    ]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def buffers(n: int, L: Layout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observation buffers for at least `n` copies: pinned, so they copy to
    the GPU fast, and kept between searches, since pinning is slow and a
    distillation search's buffers are close to a gigabyte. Only one search
    runs at a time. The rows copied to the GPU must be done before the
    next observe overwrites them; every step waits on its sampled actions,
    which is later."""
    global _buffers
    if _buffers is None or len(_buffers[0]) < n:
        pin = torch.cuda.is_available()
        _buffers = (
            torch.empty((n, L.n_floats), pin_memory=pin),
            torch.empty((n, L.n_ids), dtype=torch.int64, pin_memory=pin),
            torch.empty((n, L.n_actions), dtype=torch.bool, pin_memory=pin),
        )
    return _buffers


def outputs(logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, picks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One sampled action per entry of `picks` (a row of the network's
    output, `mask` its legal actions) and every row's value, on the CPU:
    one sync for any number of rollouts' rows."""
    sampled = np.empty(0, np.int64)
    if len(picks):
        masked = masked_logits(logits.float(), mask)
        rows = masked[torch.from_numpy(picks).to(logits.device)]
        sampled = torch.distributions.Categorical(logits=rows, validate_args=False).sample().cpu().numpy()
    return sampled, values.float().cpu().numpy()


class Rollout:
    """`rollout` as steps the caller drives, so a loop can batch many
    rollouts' rows with everything else it forwards (`sts2ai.runloop`).
    `rows(floats, ids, mask)` packs the rows the network must see next
    at the front of those buffers (which hold `len(forks)` rows) and
    returns how many, or None once the copies are scored; `picks` names
    the row each live copy samples its action from (none once scoring),
    and `apply(sampled, values, legal_rows)` takes those actions, the rows'
    values and legal actions (`outputs`) and steps the copies. `score`
    holds each copy's score at the end."""

    def __init__(
        self,
        forks,
        first: np.ndarray,
        depth: int = 1,
        second: tuple[np.ndarray, np.ndarray] | None = None,
        on_step: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ):
        self.forks, self.depth, self.second, self.on_step = forks, depth, second, on_step
        n = len(forks)
        self.inverse = np.empty(n, np.int64)
        self.rewards = np.zeros(n, np.float32)
        self.score = np.zeros(n, np.float32)
        self.step = 0
        self.live = np.arange(n)
        # The copies waiting on the network: in their turn, or, once every
        # turn is over, where the value head scores them.
        self.scoring = False
        self.done = False
        self.advance(np.ascontiguousarray(first, dtype=np.int64))

    def advance(self, actions: np.ndarray) -> None:
        """Applies one step's actions to the live copies."""
        if self.on_step is not None:
            self.on_step(actions, self.live)
        self.forks.step(actions, self.rewards)
        self.score += self.rewards
        self.step += 1

    def rows(self, floats: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> int | None:
        if self.done:
            return None
        if not self.scoring:
            # Only the copies still in their turn, packed: most end it in a
            # few steps.
            self.live = np.array(self.forks.live(), dtype=np.int64)
            if len(self.live) == 0 or self.step >= MAX_PLAN_STEPS * self.depth:
                # Where the next turn starts, by the value head; a finished
                # fight already paid its terminal reward.
                self.scoring = True
                self.live = np.flatnonzero(~np.array(self.forks.is_over()))
                if len(self.live) == 0:
                    self.done = True
                    return None
        # The network sees each distinct observation once; every copy
        # still samples its own action.
        return self.forks.observe_unique(self.live.tolist(), floats, ids, mask, self.inverse)

    @property
    def picks(self) -> np.ndarray:
        """The row of the last `rows` each live copy samples from."""
        return np.empty(0, np.int64) if self.scoring else self.inverse[: len(self.live)]

    def apply(self, sampled: np.ndarray, values: np.ndarray, legal_rows: np.ndarray) -> None:
        """`sampled`, an action per `picks` entry; `values` and `legal_rows`
        (the legal actions, on the CPU) for the rows of the last `rows`."""
        live = self.live
        if self.scoring:
            self.score[live] += values[self.inverse[: len(live)]]
            self.done = True
            return
        actions = np.zeros(len(self.forks), np.int64)
        actions[live] = sampled
        if self.step == 1 and self.second is not None:
            groups, seconds = self.second
            groups[:] = -1
            seen = self.inverse[: len(live)]
            for g in np.unique(seen):
                members = live[seen == g]
                legal = np.flatnonzero(legal_rows[g])
                if len(members) >= SECOND_MIN * len(legal):
                    actions[members] = spread(legal, len(members))
                    groups[members] = g
            seconds[:] = actions
        self.advance(actions)


@torch.no_grad()
def rollout(
    policy: Net,
    device: torch.device,
    forks,
    first: np.ndarray,
    on_step: Callable[[np.ndarray, np.ndarray], None] | None = None,
    depth: int = 1,
    second: tuple[np.ndarray, np.ndarray] | None = None,
    chunk: int = CHUNK,
) -> np.ndarray:
    """Play every copy in `forks` (forked with this `depth`) to the end of its turn, `first[i]` as copy
    i's first action. `on_step(actions, live)` sees each step's actions and
    the copies they apply to before it is applied. Returns each copy's
    score. With `second = (groups, actions)`, two length-n arrays filled
    here, copies that saw the same observation after their first action
    spread their second action over its legal ones when there are
    `SECOND_MIN` copies for each; `groups[i]` is copy i's observation group,
    or -1 where the policy played on or the turn was already over, and
    `actions[i]` its second action. The network sees `chunk` rows a call."""
    floats, ids, mask = buffers(len(forks), Layout.load())
    r = Rollout(forks, first, depth, second, on_step)
    while (n := r.rows(floats.numpy(), ids.numpy(), mask.numpy())) is not None:
        logits, values = forward(policy, device, floats[:n], ids[:n], chunk)
        sampled, value = outputs(logits, values, mask[:n].to(device, non_blocking=True), r.picks)
        r.apply(sampled, value, mask[:n].numpy())
    return r.score


def openings(first: np.ndarray, score: np.ndarray, second: tuple[np.ndarray, np.ndarray]) -> dict[int, tuple[float, int | None]]:
    """Each first action's value, as `rollout(..., second=...)` left the
    copies, and its best second action where the copies tried them all:
    for each observation group it led to, the best second action's mean
    score (chosen on half the copies, scored on the other half), then the
    mean over groups by copies; copies with no group count by their own
    mean. The best second action is the one of the largest group."""
    groups, actions = second
    out: dict[int, tuple[float, int | None]] = {}
    for a in np.unique(first):
        mine = first == a
        total = weight = 0.0
        best, largest = None, 0
        for g in np.unique(groups[mine]):
            group = mine & (groups == g)
            # A group seen by too few of this opening's copies to split
            # (it shares the group with another opening) counts its mean.
            if g < 0 or np.bincount(actions[group]).max(initial=0) < 2:
                value = float(score[group].mean())
            else:
                # The best second action is picked on one half of its copies
                # and scored on the other, both ways round: picked and scored
                # on the same copies, the max is partly whichever got lucky.
                members = {int(b): np.flatnonzero(group & (actions == b)) for b in np.unique(actions[group])}
                means = [{b: float(score[idx[h::2]].mean()) for b, idx in members.items() if len(idx) > h} for h in (0, 1)]
                picks = [max(m, key=m.get) for m in means]
                value = 0.5 * (means[1].get(picks[0], means[0][picks[0]]) + means[0].get(picks[1], means[1][picks[1]]))
                pick = picks[0]
                if group.sum() > largest:
                    best, largest = pick, int(group.sum())
            total += value * group.sum()
            weight += group.sum()
        out[int(a)] = (total / weight, best)
    return out


def spread(legal: np.ndarray, n: int) -> np.ndarray:
    """First actions for `n` copies: every legal action gets an equal share."""
    return legal[np.arange(n) % len(legal)]


def picks(first: np.ndarray, score: np.ndarray, second: tuple[np.ndarray, np.ndarray], own: np.ndarray, n: int) -> np.ndarray:
    """The pilot's pick per root from a search of `n` copies a root
    (`Rollout` with `second`): openings ranked by their best second
    action (`openings`). The policy's own action `own[r]` stays unless
    another beats it by `PLAN_MARGIN`."""
    out = own.copy()
    for r in range(len(own)):
        part = slice(r * n, (r + 1) * n)
        value = openings(first[part], score[part], (second[0][part], second[1][part]))
        best = max(value, key=lambda a: value[a][0])
        if int(own[r]) not in value or value[best][0] - value[int(own[r])][0] >= PLAN_MARGIN:
            out[r] = best
    return out


@torch.no_grad()
def choose(policy: Net, device: torch.device, sim, roots: list[int], mask: np.ndarray, own: np.ndarray, n: int, groups: int, seed: int) -> np.ndarray:
    """The pilot's pick (`advise.Session.plan`) at the decisions of many
    fights at once: `n` copies of each env in `roots` of the VecEnv `sim`,
    over `groups` draw-pile shuffles, every legal first action (`mask[r]`)
    with its share, openings ranked by their best second action
    (`picks`). Returns the action per root. The copies roll their own dice
    from `seed`, never the env's."""
    forks = sim.fork(roots, n, groups, seed)
    first = np.concatenate([spread(np.flatnonzero(m), n) for m in mask])
    second = (np.full(len(first), -1), np.zeros(len(first), np.int64))
    score = rollout(policy, device, forks, first, second=second, chunk=CHOOSE_CHUNK)
    return picks(first, score, second, own, n)
