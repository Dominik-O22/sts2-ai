"""Play whole runs: a combat checkpoint plays the fights greedily, the run
decisions (paths, rewards, rest sites, relics) are random, made in the sim
(docs/run-env.md, build order step 2).

    uv run python -m sts2ai.runplay runs/<run>/latest.pt --envs 256 --runs-per-env 2

Every env plays runs back to back. The numbers cover each env's first
`--runs-per-env` runs, so long runs count as often as short ones; the
envs that finish early keep playing, and those extra runs only count
toward throughput.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import End, Envs, Layout
from sts2ai.model import Policy, load_policy, masked_logits


@torch.no_grad()
def play(policy: Policy, device: torch.device, envs: Envs, seed: int, per_env: int, minutes: float) -> tuple[list[End], int, float]:
    """Steps `envs` greedily until each env has finished `per_env` runs or
    `minutes` pass. Returns every fight that ended, the batch steps taken
    and the seconds spent."""
    envs.use_runs(seed)
    last = seed + per_env * envs.n
    left = set(range(seed, last))
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    ends: list[End] = []
    steps, start = 0, time.perf_counter()
    while left and time.perf_counter() - start < minutes * 60:
        floats = torch.from_numpy(envs.floats).to(device, non_blocking=True)
        ids = torch.from_numpy(envs.ids).to(device, non_blocking=True)
        mask = torch.from_numpy(envs.mask).to(device, non_blocking=True)
        with autocast:
            logits, _ = policy(floats, ids)
        actions = masked_logits(logits.float(), mask).argmax(dim=1).cpu().numpy()
        for e in envs.step(actions):
            ends.append(e)
            if e.run and e.run.end:
                left.discard(e.run.seed)
        steps += 1
    return ends, steps, time.perf_counter() - start


def report(ends: list[End], seed: int, last: int, steps: int, n: int, seconds: float) -> None:
    runs = [e for e in ends if e.run and e.run.end]
    counted = [e for e in runs if e.run.seed < last]
    seeds = {r.run.seed for r in counted}
    fights = [e for e in ends if e.run and e.run.seed in seeds]
    print(f"{len(counted)} of {last - seed} runs finished ({len(runs)} in all), {len(fights)} fights in them")
    if not counted:
        return
    outcome = Counter(r.run.end if not r.run.end.startswith("stuck") else "stuck" for r in counted)
    print("  " + "  ".join(f"{k} {v}" for k, v in outcome.most_common()))
    for why, k in Counter(r.run.end for r in counted if r.run.end.startswith("stuck")).most_common(5):
        print(f"    {k:4d} {why}")

    floors = np.array([r.floor for r in counted])
    print(f"floor reached: mean {floors.mean():.1f}, median {np.median(floors):.0f}, max {floors.max()}")
    by_act = Counter("won" if r.run.end == "won" else f"act {r.run.act + 1}" for r in counted)
    print("  ended in: " + "  ".join(f"{k} {v} ({v / len(counted):.0%})" for k, v in sorted(by_act.items())))
    print("  floor quantiles: " + "  ".join(f"p{q} {np.percentile(floors, q):.0f}" for q in (10, 25, 50, 75, 90)))

    decks = np.array([r.run.deck for r in counted])
    print(f"deck at the last fight: mean {decks.mean():.1f}, median {np.median(decks):.0f}, max {decks.max()}")

    print("combat win rate by act and kind (wins/fights):")
    table: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for e in fights:
        table[(e.run.act, e.kind)].append(e.won)
    for act in sorted({a for a, _ in table}):
        cells = [(k, table[(act, k)]) for k in ("Weak", "Normal", "Elite", "Boss") if (act, k) in table]
        print(f"  act {act + 1}: " + "   ".join(f"{k.lower():6s} {np.mean(w):6.1%} {sum(w):5d}/{len(w):<5d}" for k, w in cells))
    losses = Counter(e.encounter for e in fights if not e.won)
    print("  most runs lost to: " + ", ".join(f"{enc} {k}" for enc, k in losses.most_common(8)))

    rate = f"{steps * n / seconds:,.0f} combat steps/s, {len(runs) / seconds * 3600:,.0f} runs/hour"
    print(f"throughput: {rate} ({seconds:.0f} s, {n} envs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--old-vocab", type=Path, default=None, help="vocab.txt the checkpoint was trained with, if it predates the current sim")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--runs-per-env", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=10.0, help="stop early after this long")
    ap.add_argument("--seed", type=int, default=0, help="the first run's seed index")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device, args.old_vocab)
    policy.eval()
    envs = Envs(args.envs, seed=args.seed)
    ends, steps, seconds = play(policy, device, envs, args.seed, args.runs_per_env, args.minutes)
    report(ends, args.seed, args.seed + args.runs_per_env * args.envs, steps, args.envs, seconds)


if __name__ == "__main__":
    main()
