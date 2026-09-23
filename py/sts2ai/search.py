"""Turn search: play copies of a fight to the end of the turn and score
them. Used by the advisor's plan and by `searcheval`.

Each copy starts with a given first action; the policy samples the rest of
the turn. A copy's score is the shaped reward it collected plus the value
head where the next turn starts (a finished fight already paid its
terminal reward).

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
from sts2ai.model import Policy, masked_logits

# Actions per player turn searched; a runaway plan is cut after this many
# per turn.
MAX_PLAN_STEPS = 40
# Copies per network call: a search over many fights at once is too big for
# one batch on the GPU.
CHUNK = 16384
# Copies each second action needs before `second` tries them all for what a
# first action led to; fewer and the max over them picks luck, so those
# copies keep the policy's play.
SECOND_MIN = 8


def forward(policy: Policy, device: torch.device, floats: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy over a batch of any size, in `CHUNK`-sized pieces."""
    parts = [
        policy(floats[i : i + CHUNK].to(device, non_blocking=True), ids[i : i + CHUNK].to(device, non_blocking=True))
        for i in range(0, len(floats), CHUNK)
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


@torch.no_grad()
def rollout(
    policy: Policy,
    device: torch.device,
    forks,
    first: np.ndarray,
    on_step: Callable[[np.ndarray, np.ndarray], None] | None = None,
    depth: int = 1,
    second: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Play every copy in `forks` (forked with this `depth`) to the end of its turn, `first[i]` as copy
    i's first action. `on_step(actions, live)` sees each step's actions and
    the copies they apply to before it is applied. Returns each copy's
    score. With `second = (groups, actions)`, two length-n arrays filled
    here, copies that saw the same observation after their first action
    spread their second action over its legal ones when there are
    `SECOND_MIN` copies for each; `groups[i]` is copy i's observation group,
    or -1 where the policy played on or the turn was already over, and
    `actions[i]` its second action."""
    n = len(forks)
    floats, ids, mask = buffers(n, Layout.load())
    inverse = np.empty(n, np.int64)
    rewards = np.zeros(n, np.float32)
    score = np.zeros(n, np.float32)
    actions = np.ascontiguousarray(first, dtype=np.int64)
    live = np.arange(n)
    for step in range(MAX_PLAN_STEPS * depth):
        if step > 0:
            # Only the copies still in their turn, packed: most end it in a
            # few steps.
            live = np.array(forks.live(), dtype=np.int64)
            if len(live) == 0:
                break
            # The network sees each distinct observation once; every copy
            # still samples its own action.
            n_unique = forks.observe_unique(live.tolist(), floats.numpy(), ids.numpy(), mask.numpy(), inverse)
            logits, _ = forward(policy, device, floats[:n_unique], ids[:n_unique])
            masked = masked_logits(logits.float(), mask[:n_unique].to(device, non_blocking=True))
            per_copy = masked[torch.from_numpy(inverse[: len(live)]).to(device)]
            actions = np.zeros(n, np.int64)
            actions[live] = torch.distributions.Categorical(logits=per_copy, validate_args=False).sample().cpu().numpy()
            if step == 1 and second is not None:
                groups, seconds = second
                groups[:] = -1
                seen = inverse[: len(live)]
                legal_rows = mask[:n_unique].numpy()
                for g in np.unique(seen):
                    members = live[seen == g]
                    legal = np.flatnonzero(legal_rows[g])
                    if len(members) >= SECOND_MIN * len(legal):
                        actions[members] = spread(legal, len(members))
                        groups[members] = g
                seconds[:] = actions
        if on_step is not None:
            on_step(actions, live)
        forks.step(actions, rewards)
        score += rewards
    # Where the next turn starts, by the value head; a finished fight
    # already paid its terminal reward.
    rows = np.flatnonzero(~np.array(forks.is_over()))
    if len(rows):
        n_unique = forks.observe_unique(rows.tolist(), floats.numpy(), ids.numpy(), mask.numpy(), inverse)
        _, value = forward(policy, device, floats[:n_unique], ids[:n_unique])
        score[rows] += value.float().cpu().numpy()[inverse[: len(rows)]]
    return score


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
