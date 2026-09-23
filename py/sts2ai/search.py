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

# Actions per player turn searched; a runaway plan is cut after this many
# per turn.
MAX_PLAN_STEPS = 40
# Copies per network call: a search over many fights at once is too big for
# one batch on the GPU.
CHUNK = 16384


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
) -> np.ndarray:
    """Play every copy in `forks` (forked with this `depth`) to the end of its turn, `first[i]` as copy
    i's first action. `on_step(actions, live)` sees each step's actions and
    the copies they apply to before it is applied. Returns each copy's
    score."""
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


def spread(legal: np.ndarray, n: int) -> np.ndarray:
    """First actions for `n` copies: every legal action gets an equal share."""
    return legal[np.arange(n) % len(legal)]
