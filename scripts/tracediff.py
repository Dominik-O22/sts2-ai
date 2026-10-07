"""Compare two `runplay --runs-out` traces seed by seed: whether each seed's
run ended the same way (end, act, floor, deck) and fought, decided and
carried the same (`sts2ai.runtrace` fields). Prints the seeds that differ
with the first field that does, and a count.

    uv run python scripts/tracediff.py a.jsonl b.jsonl
    uv run python scripts/tracediff.py a.jsonl b.jsonl --seeds 0-127   # only these

Exit code 1 when any paired seed differs or a seed is in one file only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["seed"]: r for r in lines[1:]}


def first_difference(a: dict, b: dict) -> str | None:
    """Where two runs of one seed first differ, in words; None if nowhere."""
    for k in ("end", "act", "floor", "deck", "start", "source"):
        if a.get(k) != b.get(k):
            return f"{k}: {a.get(k)!r} vs {b.get(k)!r}"
    for k in ("fights", "decisions", "decks"):
        xs, ys = a.get(k, []), b.get(k, [])
        for i, (x, y) in enumerate(zip(xs, ys)):
            if x != y:
                return f"{k}[{i}]: {json.dumps(x)[:160]} vs {json.dumps(y)[:160]}"
        if len(xs) != len(ys):
            return f"{k}: {len(xs)} vs {len(ys)} entries"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--seeds", default="", help="LO-HI, inclusive; empty: every seed in either file")
    ap.add_argument("--show", type=int, default=20, help="differing seeds printed")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    seeds = set(a) | set(b)
    if args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = {s for s in seeds if lo <= s <= hi}
    paired = sorted(s for s in seeds if s in a and s in b)
    only = sorted(seeds - set(paired))
    differ = [(s, d) for s in paired if (d := first_difference(a[s], b[s]))]
    for s, d in differ[: args.show]:
        print(f"seed {s}: {d}")
    if only:
        print(f"in one file only: {only[:20]}{' ...' if len(only) > 20 else ''}")
    same = len(paired) - len(differ)
    print(f"{len(paired)} paired seeds: {same} identical, {len(differ)} differ; {len(only)} unpaired")
    raise SystemExit(1 if differ or only else 0)


if __name__ == "__main__":
    main()
