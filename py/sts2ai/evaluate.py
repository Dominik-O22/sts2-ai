"""Win rate of a policy on the recorded fights (the held-out set), greedy.

    uv run python -m sts2ai.evaluate runs/<name>/latest.pt [--episodes 200]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import DEFAULT_RECORDINGS, End, Envs, Layout
from sts2ai.model import Policy, load_policy, masked_logits


@torch.no_grad()
def evaluate(
    policy: Policy,
    device: torch.device,
    episodes: int = 200,
    recordings: Path = DEFAULT_RECORDINGS,
    n_envs: int = 64,
    seed: int = 12345,
) -> tuple[float, dict[str, tuple[int, int]]]:
    """Returns the overall win rate and per-encounter (wins, fights)."""
    envs = Envs(n_envs, seed=seed)
    envs.load_recordings(recordings)
    ends: list[End] = []
    while len(ends) < episodes:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = policy(floats, ids)
        actions = masked_logits(logits, mask).argmax(dim=1).cpu().numpy()
        ends.extend(envs.step(actions))
    ends = ends[:episodes]
    by_enc: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for e in ends:
        w, n = by_enc[e.encounter]
        by_enc[e.encounter] = (w + e.won, n + 1)
    return float(np.mean([e.won for e in ends])), dict(by_enc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()
    win, by_enc = evaluate(policy, device, args.episodes, args.recordings)
    for enc, (w, n) in sorted(by_enc.items()):
        print(f"{enc:32s} {w:4d}/{n:<4d} {w / n:6.1%}")
    print(f"overall {win:.1%} over {args.episodes} fights")


if __name__ == "__main__":
    main()
