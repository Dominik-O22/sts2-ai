"""Checkpoints side by side on the held-out set, by act and kind.

    uv run python -m sts2ai.compare runs/set-5/latest.pt runs/set-6/latest.pt --repeats 10

Every checkpoint plays the same setups the same number of times. Two
repeats (what training evaluates with) swing a boss number by about six
points; ten are the least to call a difference between checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sts2ai.env import Layout
from sts2ai.evaluate import evaluate
from sts2ai.model import Policy, load_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--acts", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: dict[str, dict[str, float]] = {}
    for path in args.checkpoints:
        policy = Policy(Layout.load()).to(device)
        load_policy(path, policy, device)
        policy.eval()
        win, _, by_kind = evaluate(policy, device, args.repeats, acts=args.acts)
        rows[str(path)] = {"overall": win, **by_kind}
    keys = list(next(iter(rows.values())))
    width = max(len(p) for p in rows)
    print(" " * width + "".join(f"{k:>10}" for k in keys))
    for path, r in rows.items():
        print(f"{path:<{width}}" + "".join(f"{r.get(k, float('nan')):10.1%}" for k in keys))


if __name__ == "__main__":
    main()
