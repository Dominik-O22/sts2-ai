"""Fight-level benchmark: a checkpoint against a base on fixed sets of
hard fights, paired fight by fight.

    uv run python -m sts2ai.bench runs/<a>/latest.pt --base runs/<b>/latest.pt [--mode greedy] [--seeds 8]

A run-level A/B on 2,048 seeds separates about a point at a 4% win rate,
so most changes land inside its noise. Here both checkpoints play every
fight of each set from the same seeds, the difference is taken per fight
(its mean over seeds) and the interval comes from how that difference
spreads across fights. Next to the win rate it reports the fight's
terminal reward (`env::terminal_reward`: 1 plus the HP kept on a win, -1
plus a share of the enemies' HP taken on a loss), which moves with a
smaller interval because it tells a close fight from a lopsided one.

Sets: `winners`, the elite and boss fights of held-out winners' runs
(`setups.HOLDOUT`, players split from training); `sts2fun`, the same from
held-out sts2.fun players, losses included; `ours`, the elite and
boss fights of 1,024 clone runs with gen8-as on seeds 500000+, which no
training fight came from (`runplay --fights-out`, then `--make-ours`).
Results are cached per checkpoint, set, mode and seed in runs/bench/cache,
so a base plays once.

Two tiers. Greedy (the default) plays every fight on 8 seeds in minutes
and resolves about 0.3 points overall: the screen for any change. The
hybrid (`--mode hybrid32`, the pilot's search) runs a turn search and
hundreds of playouts to the fight's end at every decision, about 1.7 s a
boss fight, so it defaults to the verdict preset: one seed on a fixed
sample of `--fights 300` boss fights per set, about 25 minutes a new
checkpoint and some 3 points of resolution. Use it for changes that could
just teach greedy what the search already does: an act 3 boss specialist
gained 3 points greedy and none with the hybrid. Every fight on 8 seeds
would take about 7 hours a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.exactsearch import play
from sts2ai.model import Net, for_play, load_policy
from sts2ai.setups import HOLDOUT
from sts2ai.sts2fun import DIR as STS2FUN

ROOT = Path(__file__).resolve().parents[2] / "runs" / "bench"
SETS = {"winners": HOLDOUT, "sts2fun": STS2FUN / "setups" / "holdout.jsonl", "ours": ROOT / "ours.jsonl"}
HARD = ("_ELITE", "_BOSS")
# `exactsearch`'s defaults, as the pilot runs the hybrid, and fights per
# batch: its trees and playouts for every fight at once outgrow memory.
SEARCH = {"max_states": 500, "quiesce_states": 1000, "end_samples": 8, "samples": 4, "draw_cap": 32}
CHUNK = 350


# The hybrid's defaults (module docstring).
VERDICT = {"seeds": 1, "groups": "a1 boss,a2 boss,a3 boss", "fights": 300}


def sample(lines: list[str], n: int) -> list[str]:
    """A fixed `n` of `lines` in their order, the same every run, so the
    base's results stay cached."""
    if n <= 0 or n >= len(lines):
        return lines
    keep = np.sort(np.random.default_rng(0).choice(len(lines), n, replace=False))
    return [lines[i] for i in keep]


def hard_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip() and json.loads(line)["encounter"].endswith(HARD)]


def group(line: dict) -> str:
    """`a<act> <kind>`: acts are 17 floors long, as `runplay` counts them."""
    act = 1 + (line["floor"] > 17) + (line["floor"] > 33)
    return f"a{act} {'boss' if line['encounter'].endswith('_BOSS') else 'elite'}"


def cache_key(checkpoint: Path, lines: list[str], mode: str, seed: int) -> Path:
    st = checkpoint.resolve().stat()
    h = hashlib.sha1(f"{checkpoint.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode())
    h.update("\n".join(lines).encode())
    return ROOT / "cache" / f"{h.hexdigest()[:16]}-{mode}-{seed}.npy"


def results(checkpoint: Path, net: Net | None, device: torch.device, lines: list[str], mode: str, seed: int) -> tuple[np.ndarray, Net | None]:
    """(won, reward) per fight, from the cache or played. Loads the
    checkpoint only when something must be played."""
    path = cache_key(checkpoint, lines, mode, seed)
    if path.exists():
        return np.load(path), net
    if net is None:
        net = for_play(load_policy(checkpoint, device).eval())
    out = np.zeros((len(lines), 2), np.float32)
    chunk = len(lines) if mode == "greedy" else CHUNK
    for start in range(0, len(lines), chunk):
        run = play(net, device, lines[start : start + chunk], mode, seed, 256, SEARCH)
        for k, e in run.ends.items():
            out[start + k] = (e.won, e.reward)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out, net


def interval(d: np.ndarray) -> str:
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.3f} [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}]"


def report(name: str, lines: list[str], a: np.ndarray, b: np.ndarray | None) -> None:
    """`a`, `b`: [fights, seeds, (won, reward)]."""
    groups = defaultdict(list)
    for i, line in enumerate(lines):
        groups[group(json.loads(line))].append(i)
    print(f"== {name}: {len(lines)} fights x {a.shape[1]} seeds")
    head = f"{'':10s} {'n':>5s} {'won':>6s}"
    print(head + (f" {'base':>6s}  {'won diff [95%]':>26s}  {'reward diff [95%]':>26s}" if b is not None else f"  {'reward':>7s}"))
    for g, idx in [("all", list(range(len(lines))))] + sorted(groups.items()):
        fa = a[idx].mean(axis=1)
        row = f"{g:10s} {len(idx):5d} {fa[:, 0].mean():6.1%}"
        if b is None:
            print(row + f"  {fa[:, 1].mean():7.3f}")
            continue
        fb = b[idx].mean(axis=1)
        print(row + f" {fb[:, 0].mean():6.1%}  {interval(fa[:, 0] - fb[:, 0]):>26s}  {interval(fa[:, 1] - fb[:, 1]):>26s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, nargs="?")
    ap.add_argument("--base", type=Path, default=None, help="checkpoint to compare against, fight by fight")
    ap.add_argument("--mode", default="greedy", help="greedy, or hybridP: the pilot's search with P playouts a line (hybrid32)")
    ap.add_argument("--seeds", type=int, default=None, help="8 greedy, 1 with the hybrid")
    ap.add_argument("--sets", default="winners,sts2fun,ours")
    ap.add_argument("--groups", default=None, help="only these act and kind groups, comma separated (`a3 boss,a1 boss`); every boss with the hybrid")
    ap.add_argument("--fights", type=int, default=None, help="a fixed sample of this many fights a set; every fight greedy, 300 with the hybrid")
    ap.add_argument("--make-ours", type=Path, default=None, help="write the `ours` set from a `runplay --fights-out` file and exit")
    args = ap.parse_args()
    if args.make_ours:
        lines = hard_lines(args.make_ours)
        SETS["ours"].write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} elite and boss fights to {SETS['ours']}")
        return
    if args.checkpoint is None:
        ap.error("a checkpoint is needed")
    preset = VERDICT if args.mode != "greedy" else {"seeds": 8, "groups": "", "fights": 0}
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [args.checkpoint, *([args.base] if args.base else [])]
    nets: dict[Path, Net] = {}
    for name in args.sets.split(","):
        lines = hard_lines(SETS[name])
        if args.groups:
            lines = [line for line in lines if group(json.loads(line)) in args.groups.split(",")]
        lines = sample(lines, args.fights)
        per = {}
        for ck in checkpoints:
            rows = []
            for seed in range(1, args.seeds + 1):
                r, net = results(ck, nets.get(ck), device, lines, args.mode, seed)
                if net is not None:
                    nets[ck] = net
                rows.append(r)
            per[ck] = np.stack(rows, axis=1)
        report(name, lines, per[args.checkpoint], per[args.base] if args.base else None)


if __name__ == "__main__":
    main()
