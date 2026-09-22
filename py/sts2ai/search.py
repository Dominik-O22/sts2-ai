"""Turn search: play copies of a fight to the end of the turn and score
them. Used by the advisor's plan and by `searcheval`.

Each copy starts with a given first action; the policy samples the rest of
the turn. A copy's score is the shaped reward it collected plus the value
head where the next turn starts (a finished fight already paid its
terminal reward).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from sts2ai.env import Layout
from sts2ai.model import Policy, masked_logits

# A turn is over well within this many actions; a runaway plan is cut here.
MAX_PLAN_STEPS = 40
# Copies per network call: a search over many fights at once is too big for
# one batch on the GPU.
CHUNK = 16384


def forward(policy: Policy, device: torch.device, floats: np.ndarray, ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy over a batch of any size, in `CHUNK`-sized pieces."""
    parts = [
        policy(torch.from_numpy(floats[i : i + CHUNK]).to(device), torch.from_numpy(ids[i : i + CHUNK]).to(device))
        for i in range(0, len(floats), CHUNK)
    ]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


@torch.no_grad()
def rollout(
    policy: Policy,
    device: torch.device,
    forks,
    first: np.ndarray,
    on_step: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Play every copy in `forks` to the end of its turn, `first[i]` as copy
    i's first action. `on_step(actions, live)` sees each step's actions and
    the copies they apply to before it is applied. Returns each copy's
    score."""
    L, n = Layout.load(), len(forks)
    floats = np.empty((n, L.n_floats), np.float32)
    ids = np.empty((n, L.n_ids), np.int64)
    mask = np.empty((n, L.n_actions), np.bool_)
    rewards = np.zeros(n, np.float32)
    score = np.zeros(n, np.float32)
    actions = np.ascontiguousarray(first, dtype=np.int64)
    live = np.arange(n)
    for step in range(MAX_PLAN_STEPS):
        if step > 0:
            # Only the copies still in their turn, packed: most end it in a
            # few steps.
            live = np.array(forks.live(), dtype=np.int64)
            if len(live) == 0:
                break
            k = len(live)
            forks.observe_rows(live.tolist(), floats, ids, mask)
            logits, _ = forward(policy, device, floats[:k], ids[:k])
            masked = masked_logits(logits.float(), torch.from_numpy(mask[:k]).to(device))
            actions = np.zeros(n, np.int64)
            actions[live] = torch.distributions.Categorical(logits=masked).sample().cpu().numpy()
        if on_step is not None:
            on_step(actions, live)
        forks.step(actions, rewards)
        score += rewards
    # Where the next turn starts, by the value head; a finished fight
    # already paid its terminal reward.
    rows = np.flatnonzero(~np.array(forks.is_over()))
    if len(rows):
        forks.observe_rows(rows.tolist(), floats, ids, mask)
        _, value = forward(policy, device, floats[: len(rows)], ids[: len(rows)])
        score[rows] += value.float().cpu().numpy()
    return score


def spread(legal: np.ndarray, n: int) -> np.ndarray:
    """First actions for `n` copies: every legal action gets an equal share."""
    return legal[np.arange(n) % len(legal)]
