"""Expert iteration on hard fights: the hybrid search (`exactsearch.Hybrid`)
plays played runs' fights, and its decisions become policy targets for
training (`ppo.Config.expert`).

    uv run python -m sts2ai.expert runs/<run>/latest.pt OUT.npz --setups A.jsonl,B.jsonl --seeds 1,2

The policy trains toward a one-turn search; the hybrid plays its best lines
out to the fight's end, so its picks carry what the fight's later turns
make of a choice. Of the decisions it searches afresh (not the next step
of a line it already picked), every one where it overrules the policy is
kept, and `--keep` of the rest: the overrules carry what the search adds, the
agreements keep the targets from being only the policy's hard cases.
`--check` reads a file instead: how often a policy picks what the hybrid
did, on all kept rows and on the overrules.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import Envs
from sts2ai.exactsearch import Hybrid
from sts2ai.model import Net, for_play, load_policy, masked_logits

# `exactsearch`'s defaults, as the pilot and the benchmarks run the hybrid.
SEARCH = {"max_states": 500, "quiesce_states": 1000, "end_samples": 8, "samples": 4, "draw_cap": 32}
# Fights per batch: the hybrid's trees and playouts for all of them at once
# stay near 5 GB.
CHUNK = 350


@torch.no_grad()
def collect(net: Net, device: torch.device, lines: list[str], seed: int, keep: float, top: int, playouts: int) -> tuple[dict[str, np.ndarray], int]:
    """Play `lines` once each with the hybrid. Returns the kept decisions
    (observation, mask, the hybrid's action, whether it overruled the
    policy's) and the fights won."""
    envs = Envs(len(lines), seed=seed)
    envs.sim.use_setups("\n".join(lines), 1, seed)
    envs.sim.observe(envs.floats, envs.ids, envs.mask)
    hybrid = Hybrid(envs, top, playouts, 0, seed, **SEARCH)
    rng = np.random.default_rng(seed)
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ("floats", "ids", "mask", "action", "overruled")}
    active, won = set(range(envs.n)), 0
    while active:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = net(floats, ids)
        masked = masked_logits(logits.float(), mask)
        own = masked.argmax(dim=1).cpu().numpy()
        actions = own.copy()
        roots = np.array(sorted(active))
        # A decision on a line the hybrid already picked is the line's next
        # step: it often plays the same cards in another order than the
        # policy would, to the same state, which is no target.
        fresh = np.array([hybrid.planner.inner.planned_action(envs.sim, int(i)) is None for i in roots])
        hybrid.choose(net, device, roots.tolist(), masked.softmax(dim=1).cpu().numpy(), actions)
        overruled = actions[roots] != own[roots]
        kept = fresh & (overruled | (rng.random(len(roots)) < keep))
        at = roots[kept]
        for name, rows_of in (("floats", envs.floats), ("ids", envs.ids), ("mask", envs.mask)):
            rows[name].append(rows_of[at].copy())
        rows["action"].append(actions[at])
        rows["overruled"].append(overruled[kept])
        for e in envs.step(actions):
            if e.env in active:
                active.discard(e.env)
                won += e.won
    return {k: np.concatenate(v) for k, v in rows.items()}, won


@torch.no_grad()
def check(net: Net, device: torch.device, path: Path) -> str:
    """How often `net`'s greedy pick is the hybrid's, on a collected file."""
    d = np.load(path)
    picks = []
    for s in range(0, len(d["action"]), 4096):
        f = torch.from_numpy(d["floats"][s : s + 4096]).to(device)
        i = torch.from_numpy(d["ids"][s : s + 4096].astype(np.int64)).to(device)
        m = torch.from_numpy(d["mask"][s : s + 4096]).to(device)
        picks.append(masked_logits(net(f, i)[0].float(), m).argmax(dim=1).cpu().numpy())
    same = np.concatenate(picks) == d["action"]
    over = d["overruled"]
    return f"picks the hybrid's action on {same.mean():.1%} of {len(same)} rows, {same[over].mean():.1%} of the {over.sum()} it overruled"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path, help="decisions file to write (.npz), or to read with --check")
    ap.add_argument("--setups", default="", help="played runs' fights (`sts2ai.setups` lines), comma separated")
    ap.add_argument("--seeds", default="1", help="one pass over the fights per seed, comma separated")
    ap.add_argument("--keep", type=float, default=0.2, help="share of the searched decisions where the hybrid agreed that are kept")
    ap.add_argument("--top", type=int, default=10, help="lines the hybrid plays out")
    ap.add_argument("--playouts", type=int, default=32, help="playouts per line")
    ap.add_argument("--check", action="store_true", help="score the checkpoint against OUT instead of collecting")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = for_play(load_policy(args.checkpoint, device).eval())
    if args.check:
        print(check(net, device, args.out))
        return
    lines = [line for path in args.setups.split(",") for line in Path(path).read_text().splitlines() if line.strip()]
    parts, fights, won, t0 = [], 0, 0, time.time()
    for seed in (int(s) for s in args.seeds.split(",")):
        for start in range(0, len(lines), CHUNK):
            part, w = collect(net, device, lines[start : start + CHUNK], seed, args.keep, args.top, args.playouts)
            parts.append(part)
            fights += len(lines[start : start + CHUNK])
            won += w
            rows = sum(len(p["action"]) for p in parts)
            print(f"seed {seed} fights {fights} won {won / fights:.1%}, {rows} rows kept, {time.time() - t0:.0f}s", flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["ids"] = out["ids"].astype(np.int32)
    np.savez_compressed(args.out, **out)
    print(f"wrote {len(out['action'])} rows ({out['overruled'].sum()} overruled) to {args.out}")


if __name__ == "__main__":
    main()
