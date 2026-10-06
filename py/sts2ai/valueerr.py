"""How far a checkpoint's value head is from the truth, on states its own
play reaches.

    uv run python -m sts2ai.valueerr runs/<run>/latest.pt --setups FIGHTS.jsonl [--fights 1000] [--copies 128]

The policy plays each fight of `--setups` (sampling, as in training). At
random turn starts the value head's estimate is recorded, and the state is
forked `--copies` times, each copy with its own draw-pile shuffle and dice,
and played to the fight's end by the same policy. A copy's score is the
shaped return from that state (rewards so far against the fight's
baseline, the terminal reward at the end), which is what the value head
estimates, so the copies' mean is the truth up to its own sampling error.

Reports the value head's error against that mean, with the part the mean's
sampling error explains taken out, and the spread of single outcomes the
head learns from in training, by act and fight kind. Lower is better: the
searches lean on this head at every leaf.
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


def measure(policy, device: torch.device, setups: Path, fights: int, per_fight: int, copies: int, seed: int) -> list[dict]:
    """Per measured state: the head's value, the copies' mean and its
    standard error, the single outcomes' spread, and where it was."""
    rng = np.random.default_rng(seed)
    envs = Envs(fights, seed=seed)
    envs.use_setups(setups, 1, seed)
    names = [envs.sim.fight(i) for i in range(envs.n)]
    active, last_turn, taken = set(range(envs.n)), {}, defaultdict(int)
    out: list[dict] = []
    while active:
        with torch.no_grad():
            logits, values = policy(torch.from_numpy(envs.floats).to(device), torch.from_numpy(envs.ids).to(device))
        probs = masked_logits(logits.float(), torch.from_numpy(envs.mask).to(device)).softmax(1).cpu().numpy()
        values = values.float().cpu().numpy()
        due = []
        for i in active:
            turn = round(float(envs.floats[i, 6]) * 10)
            fresh = last_turn.get(i) != turn and not envs.floats[i, 12]
            last_turn[i] = turn
            if fresh and taken[i] < per_fight and rng.random() < 0.35:
                due.append(i)
                taken[i] += 1
        if due:
            forks = envs.sim.fork(due, copies, copies, int(rng.integers(2**62)), DEPTH)
            first = np.concatenate([rng.choice(len(probs[i]), copies, p=probs[i] / probs[i].sum()) for i in due])
            score = rollout(policy, device, forks, first, depth=DEPTH).reshape(len(due), copies)
            for k, i in enumerate(due):
                s = score[k]
                encounter, kind = names[i]
                out.append(
                    {
                        "value": float(values[i]),
                        "mean": float(s.mean()),
                        "mean_se": float(s.std(ddof=1) / np.sqrt(copies)),
                        "spread": float(s.std(ddof=1)),
                        "turn": last_turn[i],
                        "encounter": encounter,
                        "kind": kind,
                    }
                )
        actions = np.array([rng.choice(len(p), p=p / p.sum()) for p in probs])
        for e in envs.step(actions):
            active.discard(e.env)
    return out


def report(rows: list[dict]) -> None:
    def line(name: str, rs: list[dict]) -> str:
        err = np.array([r["value"] - r["mean"] for r in rs])
        noise = np.mean([r["mean_se"] ** 2 for r in rs])
        rmse = np.sqrt(max(np.mean(err**2) - noise, 0.0))
        return (
            f"{name:28s} {len(rs):5d}  error {rmse:.3f}  bias {err.mean():+.3f}  "
            f"(single outcomes spread {np.mean([r['spread'] for r in rs]):.3f}, truth's own error {np.sqrt(noise):.3f})"
        )

    print(line("all", rows))
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for kind, rs in sorted(by.items()):
        print(line(f"  {kind}", rs))
    by = defaultdict(list)
    for r in rows:
        by[r["encounter"]].append(r)
    worst = sorted(by.items(), key=lambda kv: -np.mean([abs(r["value"] - r["mean"]) for r in kv[1]]))
    for enc, rs in worst[:6]:
        if len(rs) >= 20:
            print(line(f"  {enc}", rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--setups", type=Path, required=True, help="fights to play, as sts2ai.setups writes them (held out from the checkpoint's training)")
    ap.add_argument("--fights", type=int, default=1000)
    ap.add_argument("--per-fight", type=int, default=2, help="states measured per fight, at most")
    ap.add_argument("--copies", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="write each measured state here as a JSON line")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = for_play(load_policy(args.checkpoint, device).eval())
    rows = measure(policy, device, args.setups, args.fights, args.per_fight, args.copies, args.seed)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(rows)


if __name__ == "__main__":
    main()
