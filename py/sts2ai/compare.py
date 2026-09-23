"""Checkpoints side by side on the held-out set, by act and kind.

    uv run python -m sts2ai.compare runs/set-5/latest.pt runs/set-6/latest.pt --repeats 10

Every checkpoint plays the same setups the same number of times; besides
win rates it prints HP lost per won fight and potions drunk per fight,
where a policy that wins as often can still differ. Two
repeats (what training evaluates with) swing a boss number by about six
points; ten are the least to call a difference between checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import Layout
from sts2ai.evaluate import by_kind, play
from sts2ai.model import Policy, load_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--acts", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables: dict[str, dict[str, dict[str, float]]] = {"win rate": {}, "HP lost per won fight (% of max)": {}, "potions per fight": {}}
    for path in args.checkpoints:
        policy = Policy(Layout.load()).to(device)
        load_policy(path, policy, device)
        policy.eval()
        ends = play(policy, device, args.repeats, acts=args.acts)
        tables["win rate"][str(path)] = {"overall": float(np.mean([e.won for e in ends])), **by_kind(ends, lambda e: e.won)}
        tables["HP lost per won fight (% of max)"][str(path)] = by_kind(ends, lambda e: e.hp_lost if e.won else None)
        tables["potions per fight"][str(path)] = by_kind(ends, lambda e: e.potions_used)
    width = max(len(str(p)) for p in args.checkpoints)
    for title, rows in tables.items():
        keys = list(next(iter(rows.values())))
        fmt = "{:10.2f}" if title.startswith("potions") else "{:10.1%}"
        print(f"\n{title}")
        print(" " * width + "".join(f"{k:>10}" for k in keys))
        for path, r in rows.items():
            print(f"{path:<{width}}" + "".join(fmt.format(r.get(k, float("nan"))) for k in keys))

if __name__ == "__main__":
    main()
