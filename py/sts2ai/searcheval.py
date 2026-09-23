"""How much the turn search adds to a policy, on the held-out set.

    uv run python -m sts2ai.searcheval runs/<name>/latest.pt --copies 64

Plays the elite and boss fights of `evaluate`'s held-out set twice from
the same shuffles: once greedy, once with a turn search at every decision
(the advisor's `--search`, many fights per batch). A large gap means the
fights are winnable and the policy leaves wins on the table; no gap means
the ceiling is the fights, not the play.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.advise import PLAN_MARGIN
from sts2ai.env import Envs, Layout
from sts2ai.evaluate import HOLDOUT_PER_ENCOUNTER
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.search import rollout, spread


def pick(first: np.ndarray, score: np.ndarray, own: int, mean: bool) -> int:
    """The first action whose copies scored best: the best copy (as the
    advisor prints it) or the mean over copies. The policy's own pick
    stays unless beaten by `PLAN_MARGIN`."""
    by_first = {int(a): float(score[first == a].mean() if mean else score[first == a].max()) for a in np.unique(first)}
    best = max(by_first, key=by_first.get)
    if own in by_first and by_first[best] - by_first[own] < PLAN_MARGIN:
        return own
    return best


@torch.no_grad()
def play(
    policy: Policy,
    device: torch.device,
    copies: int,
    kinds: set[str],
    acts: int,
    seed: int,
    mean: bool,
    depth: int = 1,
    recordings: Path | None = None,
) -> dict[str, list[bool]]:
    """One fight per held-out elite and boss setup (or per recording of
    one). Returns wins by encounter."""
    probe = Envs(1, seed=seed)
    n = probe.load_recordings(recordings) if recordings else probe.use_holdout(seed, HOLDOUT_PER_ENCOUNTER, acts)
    envs = Envs(n, seed=seed)
    if recordings:
        envs.load_recordings(recordings)
    else:
        envs.use_holdout(seed, HOLDOUT_PER_ENCOUNTER, acts)
    fights = [envs.sim.fight(i) for i in range(n)]
    active = {i for i, (_, kind) in enumerate(fights) if kind in kinds}
    results: dict[str, list[bool]] = defaultdict(list)
    step = 0
    while active:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = policy(floats, ids)
        actions = masked_logits(logits, mask).argmax(dim=1).cpu().numpy()
        if copies:
            roots = sorted(active)
            forks = envs.sim.fork(roots, copies, seed=seed + step, depth=depth)
            first = np.concatenate([spread(np.flatnonzero(envs.mask[i]), copies) for i in roots])
            score = rollout(policy, device, forks, first, depth=depth)
            for r, i in enumerate(roots):
                part = slice(r * copies, (r + 1) * copies)
                actions[i] = pick(first[part], score[part], int(actions[i]), mean)
        for e in envs.step(actions):
            if e.env in active:
                active.discard(e.env)
                results[fights[e.env][0]].append(bool(e.won))
        step += 1
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--copies", type=int, default=64, help="sim copies per fight per decision")
    ap.add_argument("--kinds", default="Elite,Boss")
    ap.add_argument("--acts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mean", action="store_true", help="rank first actions by mean copy score, not best copy")
    ap.add_argument("--depth", type=int, default=1, help="player turns each copy plays (1: the rest of this one)")
    ap.add_argument("--recordings", type=Path, default=None, help="real-run recordings instead of the held-out set")
    ap.add_argument("--repeats", type=int, default=1, help="fights per setup, different seeds")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()
    kinds = set(args.kinds.split(","))
    greedy: dict[str, list[bool]] = defaultdict(list)
    search: dict[str, list[bool]] = defaultdict(list)
    t0 = time.time()
    for r in range(args.repeats):
        for enc, won in play(policy, device, 0, kinds, args.acts, args.seed + r, args.mean, recordings=args.recordings).items():
            greedy[enc] += won
    t1 = time.time()
    for r in range(args.repeats):
        for enc, won in play(policy, device, args.copies, kinds, args.acts, args.seed + r, args.mean, args.depth, args.recordings).items():
            search[enc] += won
    t2 = time.time()
    print(f"{'encounter':32s} greedy  search")
    for enc in sorted(greedy):
        g, s = greedy[enc], search[enc]
        print(f"{enc:32s} {sum(g):3d}/{len(g):<3d} {sum(s):3d}/{len(s):<3d}")
    g = [w for v in greedy.values() for w in v]
    s = [w for v in search.values() for w in v]
    print(f"overall greedy {np.mean(g):.1%}  search {np.mean(s):.1%}  over {len(g)} fights ({t1 - t0:.0f}s vs {t2 - t1:.0f}s)")


if __name__ == "__main__":
    main()
