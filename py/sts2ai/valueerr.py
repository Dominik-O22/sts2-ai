"""How far a checkpoint's value head is from the truth, on states its own
play reaches, and with more checkpoints, which of them is closer.

    uv run python -m sts2ai.valueerr runs/<run>/latest.pt --setups FIGHTS.jsonl [--fights 1000] [--copies 128]
    uv run python -m sts2ai.valueerr BASE.pt NEW.pt [MORE.pt ...] --setups FIGHTS.jsonl --fights 8000

The first checkpoint's policy plays `--fights` fights drawn from `--setups`
(sampling, as in training). At random turn starts the state is forked
`--copies` times, each copy with its own draw-pile shuffle and dice, and
played to the fight's end, once per checkpoint by that checkpoint's own
policy, the same shuffles and dice for each. A copy's score is the shaped
return from that state (rewards so far against the fight's baseline, the
terminal reward at the end), which is what the value head estimates, so
the copies' mean is each checkpoint's truth up to its own sampling error.
Every checkpoint is judged on the same states with the same luck, so the
differences come paired, with a bootstrap interval over the states.

Reports the value head's error against that mean, with the part the mean's
sampling error explains taken out, and the spread of single outcomes the
head learns from in training, by fight kind, by the worst encounters, and
by situation: HP left, the unblocked damage coming at it, the turn, the
enemies' HP left, and how the fight truly stands. Lower is better: the
searches lean on this head at every leaf, and a bias in one situation
does not average out over a search's leaves the way random error does.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import Envs
from sts2ai.model import for_play, load_policy, masked_logits
from sts2ai.search import rollout

# Turns a fork plays, at most: to the fight's end.
DEPTH = 200
# Fights played at once; more forks at once than this runs out of memory.
BATCH = 1000


def act(policy, device: torch.device, floats: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The policy's action probabilities and value on these rows."""
    with torch.no_grad():
        logits, values = policy(torch.from_numpy(floats).to(device), torch.from_numpy(ids).to(device))
    probs = masked_logits(logits.float(), torch.from_numpy(mask).to(device)).softmax(1).cpu().numpy()
    return probs, values.float().cpu().numpy()


def measure(policies: list, device: torch.device, setups: list[str], per_fight: int, copies: int, seed: int) -> list[dict]:
    """Per measured state, a list with one entry per policy: the head's
    value, the copies' mean and its standard error, and the single
    outcomes' spread; then where the state was. The first policy plays the
    fights; each one plays its own copies."""
    rng = np.random.default_rng(seed)
    envs = Envs(len(setups), seed=seed)
    envs.sim.use_setups("".join(setups), 1, seed)
    envs.sim.observe(envs.floats, envs.ids, envs.mask)
    names = [envs.sim.fight(i) for i in range(envs.n)]
    active, last_turn, taken = set(range(envs.n)), {}, defaultdict(int)
    out: list[dict] = []
    while active:
        probs, _ = act(policies[0], device, envs.floats, envs.ids, envs.mask)
        due = []
        for i in active:
            turn = round(float(envs.floats[i, 6]) * 10)
            fresh = last_turn.get(i) != turn and not envs.floats[i, 12]
            last_turn[i] = turn
            if fresh and taken[i] < per_fight and rng.random() < 0.35:
                due.append(i)
                taken[i] += 1
        if due:
            # The same shuffles, dice and first-action draws for every
            # policy, so their differences are not luck.
            fork_seed, u = int(rng.integers(2**62)), rng.random((len(due), copies))
            scores, values = [], []
            for policy in policies:
                p, v = act(policy, device, envs.floats[due], envs.ids[due], envs.mask[due])
                cdf = np.cumsum(p / p.sum(1, keepdims=True), 1)
                first = np.minimum((cdf[:, None, :] <= u[:, :, None]).sum(2), p.shape[1] - 1).ravel()
                forks = envs.sim.fork(due, copies, copies, fork_seed, DEPTH)
                scores.append(rollout(policy, device, forks, first, depth=DEPTH).reshape(len(due), copies))
                values.append(v)
            L = envs.layout
            for k, i in enumerate(due):
                encounter, kind = names[i]
                f = envs.floats[i]
                enemies = f[L.f_enemies : L.f_enemies + L.max_enemies * L.enemy_feats].reshape(L.max_enemies, L.enemy_feats)
                alive = (enemies[:, 0] > 0) & (enemies[:, 1] > 0)
                out.append(
                    {
                        "value": [float(v[k]) for v in values],
                        "mean": [float(sc[k].mean()) for sc in scores],
                        "mean_se": [float(sc[k].std(ddof=1) / np.sqrt(copies)) for sc in scores],
                        "spread": [float(sc[k].std(ddof=1)) for sc in scores],
                        "turn": last_turn[i],
                        "encounter": encounter,
                        "kind": kind,
                        "hp_frac": float(f[2]),
                        # Unblocked incoming damage over the HP left: 1 is lethal.
                        "threat": float(f[21] * 50 / max(f[0] * 100, 1)),
                        "enemy_left": float(enemies[alive, 2].sum() / max(enemies[alive, 3].sum(), 1e-6)) if alive.any() else 0.0,
                    }
                )
        actions = np.array([rng.choice(len(p), p=p / p.sum()) for p in probs])
        for e in envs.step(actions):
            active.discard(e.env)
    return out


# Situations a row falls in: (heading, [(label, test)]); how the fight
# truly stands goes by every checkpoint's truth averaged: binning by one
# checkpoint's own noisy truth would bias its error in each bin.
SITUATIONS = [
    ("HP left", [("under 30%", lambda r: r["hp_frac"] < 0.3), ("30-60%", lambda r: 0.3 <= r["hp_frac"] < 0.6), ("60% and up", lambda r: r["hp_frac"] >= 0.6)]),
    (
        "unblocked damage coming",
        [
            ("none", lambda r: r["threat"] == 0),
            ("under half the HP", lambda r: 0 < r["threat"] < 0.5),
            ("half to all", lambda r: 0.5 <= r["threat"] < 1),
            ("lethal", lambda r: r["threat"] >= 1),
        ],
    ),
    (
        "turn",
        [
            ("1-2", lambda r: r["turn"] <= 2),
            ("3-4", lambda r: 3 <= r["turn"] <= 4),
            ("5-7", lambda r: 5 <= r["turn"] <= 7),
            ("8 and later", lambda r: r["turn"] >= 8),
        ],
    ),
    (
        "enemies' HP left",
        [
            ("over two thirds", lambda r: r["enemy_left"] > 2 / 3),
            ("a third to two thirds", lambda r: 1 / 3 < r["enemy_left"] <= 2 / 3),
            ("under a third", lambda r: r["enemy_left"] <= 1 / 3),
        ],
    ),
    (
        "how the fight truly stands",
        [
            ("likely lost", lambda r: np.mean(r["mean"]) < -0.3),
            ("close", lambda r: -0.3 <= np.mean(r["mean"]) <= 0.6),
            ("likely won", lambda r: np.mean(r["mean"]) > 0.6),
        ],
    ),
]


def errors(rs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """`[states, checkpoints]` each: the head's error against the copies'
    mean, and that mean's own sampling variance."""
    return np.array([r["value"] for r in rs]) - np.array([r["mean"] for r in rs]), np.array([r["mean_se"] for r in rs]) ** 2


def rmse(err: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Per checkpoint (the last axis), the error with the truth's own
    sampling error taken out."""
    return np.sqrt(np.maximum((err**2).mean(-2) - noise.mean(-2), 0.0))


def compare(rs: list[dict], k: int, resamples: int = 1000) -> tuple[float, float, float]:
    """Checkpoint k's error minus the first's on the same states, and a 95%
    bootstrap interval over the states. Below zero is better."""
    err, noise = errors(rs)
    pair = lambda e, n: rmse(e, n)[..., k] - rmse(e, n)[..., 0]
    rng = np.random.default_rng(0)
    boot = np.concatenate([pair(err[i], noise[i]) for i in np.array_split(rng.integers(0, len(rs), (resamples, len(rs))), 10)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(pair(err, noise)), float(lo), float(hi)


def report(rows: list[dict]) -> None:
    def line(name: str, rs: list[dict]) -> str:
        err, noise = errors(rs)
        if err.shape[1] == 1:
            return (
                f"{name:28s} {len(rs):5d}  error {rmse(err, noise)[0]:.3f}  bias {err.mean():+.3f}  "
                f"(single outcomes spread {np.mean([r['spread'] for r in rs]):.3f}, truth's own error {np.sqrt(noise.mean()):.3f})"
            )
        diffs = "  ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*compare(rs, k)) for k in range(1, err.shape[1]))
        return f"{name:28s} {len(rs):5d}  error {' '.join(f'{e:.3f}' for e in rmse(err, noise))}  bias {' '.join(f'{b:+.3f}' for b in err.mean(0))}  vs first {diffs}"

    print(line("all", rows))
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for kind, rs in sorted(by.items()):
        print(line(f"  {kind}", rs))
    by = defaultdict(list)
    for r in rows:
        by[r["encounter"]].append(r)
    worst = sorted(by.items(), key=lambda kv: -np.abs(errors(kv[1])[0][:, 0]).mean())
    for enc, rs in worst[:6]:
        if len(rs) >= 20:
            print(line(f"  {enc}", rs))
    for heading, bins in SITUATIONS:
        if heading != "how the fight truly stands" and not all(key in rows[0] for key in ("hp_frac", "threat", "enemy_left")):
            continue
        print(heading)
        for label, test in bins:
            rs = [r for r in rows if test(r)]
            if len(rs) >= 20:
                print(line(f"  {label}", rs))


def load_rows(path: Path) -> list[dict]:
    """An earlier `--out`; files from before the paired measurement held
    one checkpoint's numbers as plain values."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        for key in ("value", "mean", "mean_se", "spread"):
            if not isinstance(r[key], list):
                r[key] = [r[key]]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="*", help="the first plays the fights; the others are compared with it")
    ap.add_argument("--setups", type=Path, default=None, help="fights to play, as sts2ai.setups writes them (held out from the checkpoints' training)")
    ap.add_argument("--fights", type=int, default=1000, help="drawn at random from --setups")
    ap.add_argument("--per-fight", type=int, default=2, help="states measured per fight, at most")
    ap.add_argument("--copies", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="write each measured state here as a JSON line")
    ap.add_argument("--report", type=Path, default=None, help="only report on the states an earlier --out wrote")
    args = ap.parse_args()
    if args.report:
        report(load_rows(args.report))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = [for_play(load_policy(c, device).eval()) for c in args.checkpoints]
    for k, c in enumerate(args.checkpoints):
        print(f"checkpoint {k}: {c}")
    setups = args.setups.read_text().splitlines(keepends=True)
    drawn = np.random.default_rng(args.seed).choice(len(setups), min(args.fights, len(setups)), replace=False)
    rows = []
    for b, at in enumerate(range(0, len(drawn), BATCH)):
        rows += measure(policies, device, [setups[i] for i in drawn[at : at + BATCH]], args.per_fight, args.copies, args.seed + b)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(rows)


if __name__ == "__main__":
    main()
