"""Training runs side by side against the hours they trained: each
snapshot (`it<N>.pt`) evaluated on the generated held-out set, the
winners' elite and boss fights and their weak and normal fights, placed at
the time the run reached it.

    uv run python -m sts2ai.curves runs/ab-attn runs/gen2 [--repeats 2]

Compare runs at equal hours, not equal iterations: an iteration costs more
in some (a bigger search, a wider network), and what a change buys is how
far a run gets in the time it has, and where it levels off. Evaluations are
kept in each run's `curves.json`, so a rerun only plays new snapshots. Run
it with nothing else training: evaluations beside a training run fight it
for memory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sts2ai.evaluate import easy, evaluate
from sts2ai.model import load_policy

ACTS = ("", "a2_", "a3_")


def hours_at(run: Path) -> dict[int, float]:
    """Hours since the run's first logged iteration, by global step."""
    log = EventAccumulator(str(run), size_guidance={"scalars": 0})
    log.Reload()
    points = log.Scalars("perf/sps")
    start = points[0].wall_time
    return {p.step: (p.wall_time - start) / 3600 for p in points}


def scores(path: Path, device: torch.device, repeats: int) -> dict[str, float]:
    policy = load_policy(path, device).eval()
    held, _, held_kinds = evaluate(policy, device, repeats)
    real, _, real_kinds = evaluate(policy, device, repeats, "setups")
    cheap = easy(policy, device, repeats)
    mean = lambda d, kind: sum(d[f"{a}{kind}"] for a in ACTS) / len(ACTS)  # noqa: E731
    return {
        "holdout": held,
        "holdout_bosses": mean(held_kinds, "boss"),
        "winners": real,
        "winners_elites": mean(real_kinds, "elite"),
        "winners_bosses": mean(real_kinds, "boss"),
        "easy_hp_beyond": cheap["all"]["gap"],
        "easy_normal_won": sum(cheap[f"a{a}_normal"]["win"] for a in (1, 2, 3)) / 3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    device = torch.device("cuda")
    print(f"{'run':12s} {'iter':>6s} {'hours':>5s} {'holdout':>7s} {'bosses':>6s} | {'winners':>7s} {'elites':>6s} {'bosses':>6s} | {'easy HP+':>8s} {'normals':>7s}")
    for run in args.runs:
        cache_path = run / "curves.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        hours = hours_at(run)
        snaps = sorted(run.glob("it*.pt"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
        for snap in snaps:
            key = f"{snap.name} x{args.repeats}"
            if key not in cache:
                cache[key] = scores(snap, device, args.repeats)
                cache_path.write_text(json.dumps(cache, indent=1))
            s = cache[key]
            step = torch.load(snap, map_location="cpu", weights_only=False)["global_step"]
            at = hours.get(step, max((h for st, h in hours.items() if st <= step), default=float("nan")))
            print(
                f"{run.name:12s} {int(re.sub(r'[^0-9]', '', snap.stem)):6d} {at:5.2f} {s['holdout']:7.1%} {s['holdout_bosses']:6.1%} | "
                f"{s['winners']:7.1%} {s['winners_elites']:6.1%} {s['winners_bosses']:6.1%} | {s['easy_hp_beyond']:+8.1f} {s['easy_normal_won']:7.1%}"
            )


if __name__ == "__main__":
    main()
