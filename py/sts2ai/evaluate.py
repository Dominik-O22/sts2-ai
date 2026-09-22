"""Greedy win rate of a policy on a held-out set.

    uv run python -m sts2ai.evaluate runs/<name>/latest.pt [--source holdout|recordings]

`holdout` is a fixed generated set: ten fights per act 1 encounter on
floors that encounter appears on, the same decks every time. `recordings`
are the recorder files, which so far are dev-console test fights, not run
decks; useful as a sanity check, not as a measure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import DEFAULT_RECORDINGS, End, Envs, Layout
from sts2ai.model import Policy, load_policy, masked_logits

HOLDOUT_PER_ENCOUNTER = 10


@torch.no_grad()
def evaluate(
    policy: Policy,
    device: torch.device,
    repeats: int = 2,
    source: str = "holdout",
    recordings: Path = DEFAULT_RECORDINGS,
    seed: int = 12345,
) -> tuple[float, dict[str, tuple[int, int]], dict[str, float]]:
    """Plays every setup in the set `repeats` times (different shuffles),
    one env per setup so each gets exactly that many fights. Returns the
    overall win rate, per-encounter (wins, fights), and per-kind win rates."""
    probe = Envs(1, seed=seed)
    n = probe.use_holdout(seed, HOLDOUT_PER_ENCOUNTER) if source == "holdout" else probe.load_recordings(recordings)
    envs = Envs(n, seed=seed)
    if source == "holdout":
        envs.use_holdout(seed, HOLDOUT_PER_ENCOUNTER)
    else:
        envs.load_recordings(recordings)
    per_env = [0] * n
    ends: list[End] = []
    while min(per_env) < repeats:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = policy(floats, ids)
        actions = masked_logits(logits, mask).argmax(dim=1).cpu().numpy()
        for e in envs.step(actions):
            if per_env[e.env] < repeats:
                per_env[e.env] += 1
                ends.append(e)
    by_enc: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for e in ends:
        w, n = by_enc[e.encounter]
        by_enc[e.encounter] = (w + e.won, n + 1)
    by_kind = {
        k.lower(): float(np.mean(won)) for k in ("Weak", "Normal", "Elite", "Boss") if (won := [e.won for e in ends if e.kind == k])
    }
    return float(np.mean([e.won for e in ends])), dict(by_enc), by_kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--source", choices=["holdout", "recordings"], default="holdout")
    ap.add_argument("--repeats", type=int, default=2, help="fights per setup")
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()
    win, by_enc, by_kind = evaluate(policy, device, args.repeats, args.source, args.recordings)
    for enc, (w, n) in sorted(by_enc.items()):
        print(f"{enc:32s} {w:4d}/{n:<4d} {w / n:6.1%}")
    print("  ".join(f"{k} {v:.1%}" for k, v in by_kind.items()))
    print(f"overall {win:.1%} over {sum(n for _, n in by_enc.values())} {args.source} fights")


if __name__ == "__main__":
    main()
