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
    i's first action. `on_step(actions, over)` sees each step before it is
    applied. Returns each copy's score."""
    L, n = Layout.load(), len(forks)
    floats = np.zeros((n, L.n_floats), np.float32)
    ids = np.zeros((n, L.n_ids), np.int64)
    mask = np.zeros((n, L.n_actions), np.bool_)
    rewards = np.zeros(n, np.float32)
    forks.observe(floats, ids, mask)
    score = np.zeros(n, np.float32)
    for step in range(MAX_PLAN_STEPS):
        over = np.array(forks.turn_over())
        if over.all():
            break
        if step == 0:
            actions = first
        else:
            logits, _ = forward(policy, device, floats, ids)
            masked = masked_logits(logits, torch.from_numpy(mask).to(device))
            actions = torch.distributions.Categorical(logits=masked).sample().cpu().numpy()
        if on_step is not None:
            on_step(actions, over)
        forks.step(np.ascontiguousarray(actions, dtype=np.int64), floats, ids, mask, rewards)
        score += rewards
    _, value = forward(policy, device, floats, ids)
    return score + np.where(forks.is_over(), 0.0, value.float().cpu().numpy())


def spread(legal: np.ndarray, n: int) -> np.ndarray:
    """First actions for `n` copies: every legal action gets an equal share."""
    return legal[np.arange(n) % len(legal)]
