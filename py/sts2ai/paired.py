"""Two `runplay --runs-out` files from the same seeds, compared run by run.

    uv run python -m sts2ai.paired runs/ab/clone.jsonl runs/ab/afterstate.jsonl

Both arms play the same seeds, so each seed is a pair and the difference
is measured on the pairs: wins, act 1 deaths, act 3 reached and floor, B
minus A, each with a 95% interval from resampling the seeds. A seed stuck
in either arm is dropped from both. An interval that spans zero is noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[list[str], dict[int, dict]]:
    """A runs file's command line and its runs by seed."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return lines[0]["argv"], {r["seed"]: r for r in lines[1:]}


def metrics(r: dict) -> tuple[float, float, float, float]:
    """Won, died in act 1, reached act 3, floor."""
    won = r["end"] == "won"
    return float(won), float(r["end"] == "died" and r["act"] == 0), float(won or r["act"] >= 2), float(r["floor"])


NAMES = ("won", "died in act 1", "reached act 3", "floor")


def interval(d: np.ndarray, resamples: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """The 95% interval of the mean of paired differences `d` (one row a
    seed), from resampling the seeds."""
    rng = np.random.default_rng(0)
    boot = np.stack([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(resamples)])
    return np.percentile(boot, [2.5, 97.5], axis=0)


def compare(path_a: Path, path_b: Path, resamples: int = 10_000) -> None:
    """Prints B minus A on the seeds both played."""
    (argv_a, a), (argv_b, b) = load(path_a), load(path_b)
    ok = lambda r: not r["end"].startswith("stuck")
    seeds = sorted(s for s in a.keys() & b.keys() if ok(a[s]) and ok(b[s]))
    print(f"A: {' '.join(argv_a[1:])}\nB: {' '.join(argv_b[1:])}")
    print(f"{len(seeds)} paired seeds ({len(a)} in A, {len(b)} in B, {len(a.keys() ^ b.keys())} unpaired, stuck ones dropped)")
    x = np.array([metrics(a[s]) for s in seeds])
    y = np.array([metrics(b[s]) for s in seeds])
    d = y - x
    lo, hi = interval(d, resamples)
    print(f"{'':15s} {'A':>8s} {'B':>8s} {'B - A':>8s}   95% interval")
    for k, name in enumerate(NAMES):
        pct = k < 3
        f = (lambda v: f"{v:8.1%}") if pct else (lambda v: f"{v:8.2f}")
        verdict = "" if lo[k] > 0 or hi[k] < 0 else "  within noise"
        print(f"{name:15s} {f(x[:, k].mean())} {f(y[:, k].mean())} {f(d[:, k].mean())}   [{f(lo[k]).strip()}, {f(hi[k]).strip()}]{verdict}")
    only_a = int(((x[:, 0] == 1) & (y[:, 0] == 0)).sum())
    only_b = int(((x[:, 0] == 0) & (y[:, 0] == 1)).sum())
    print(f"wins in one arm only: A {only_a}, B {only_b}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()
    compare(args.a, args.b, args.resamples)


if __name__ == "__main__":
    main()
