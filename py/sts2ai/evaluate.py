"""Greedy win rate of a policy on a held-out set.

    uv run python -m sts2ai.evaluate runs/<name>/latest.pt [--source holdout|recordings|setups|easy]

`holdout` is a fixed generated set: ten fights per encounter on floors
that encounter appears on, the same decks every time; `--acts 1` keeps it
to act 1, the set checkpoints before set-4 were measured on. `recordings`
are fights the recorder mod saw in real runs: the decks a person built,
which is the number that says whether the advisor can be trusted. The
generator's decks are not those decks (docs/training.md, Real decks).
`setups` are the elite and boss fights of other players' winning runs
held out of training (`sts2ai.setups`): the decks that beat the game.
`easy` are their weak and normal fights, which everyone wins: each is
scored by the HP the policy loses against what the winner lost in it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from sts2ai.env import DEFAULT_RECORDINGS, End, Envs, has_recordings
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.setups import EASY_HOLDOUT, HOLDOUT

HOLDOUT_PER_ENCOUNTER = 10


def act_of(floor: int) -> int:
    """The act (1-based) a generated fight's floor is in. Recordings carry
    floor 0 and count as act 1."""
    return max(0, floor - 1) // 16 + 1


@torch.no_grad()
def play(
    policy: Policy,
    device: torch.device,
    repeats: int = 2,
    source: str = "holdout",
    recordings: Path = DEFAULT_RECORDINGS,
    seed: int = 12345,
    acts: int = 3,
    setups: Path = HOLDOUT,
    ascension: int | None = None,
) -> list[End]:
    """Plays every setup in the set `repeats` times (different shuffles),
    one env per setup so each gets exactly that many fights, greedy."""
    if source == "recordings" and not has_recordings(recordings):
        raise SystemExit(f"no run recordings in {recordings}; play with the recorder mod on (dev-console fights sit in dev/)")

    def load(envs: Envs) -> int:
        if source == "holdout":
            return envs.use_holdout(seed, HOLDOUT_PER_ENCOUNTER, acts)
        if source == "setups":
            return envs.use_setups(setups, 1, seed)
        return envs.load_recordings(recordings, ascension)

    n = load(Envs(1, seed=seed))
    envs = Envs(n, seed=seed)
    load(envs)
    per_env = [0] * n
    ends: list[End] = []
    while min(per_env) < repeats:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = policy(floats, ids)
        actions = masked_logits(logits, mask).argmax(dim=1).cpu().numpy()
        for e in envs.step(actions):
            if per_env[e.env] < repeats:
                per_env[e.env] += 1
                ends.append(e)
    return ends


def by_kind(ends: list[End], value: Callable[[End], float | None]) -> dict[str, float]:
    """Mean of `value` per act and kind, over the fights it is not None
    for. Act 1 keys stay plain ("boss"), later acts get a prefix
    ("a2_boss")."""
    return {
        f"{'' if a == 1 else f'a{a}_'}{k.lower()}": float(np.mean(vs))
        for a in (1, 2, 3)
        for k in ("Weak", "Normal", "Elite", "Boss")
        if (vs := [v for e in ends if e.kind == k and act_of(e.floor) == a and (v := value(e)) is not None])
    }


def evaluate(
    policy: Policy,
    device: torch.device,
    repeats: int = 2,
    source: str = "holdout",
    recordings: Path = DEFAULT_RECORDINGS,
    seed: int = 12345,
    acts: int = 3,
    setups: Path = HOLDOUT,
    ascension: int | None = None,
) -> tuple[float, dict[str, tuple[int, int]], dict[str, float]]:
    """`play`, summed up: the overall win rate, per-encounter (wins,
    fights), and per-kind win rates. `ascension` keeps recordings played at it."""
    ends = play(policy, device, repeats, source, recordings, seed, acts, setups, ascension)
    by_enc: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for e in ends:
        w, n = by_enc[e.encounter]
        by_enc[e.encounter] = (w + e.won, n + 1)
    return float(np.mean([e.won for e in ends])), dict(by_enc), by_kind(ends, lambda e: e.won)


def easy(policy: Policy, device: torch.device, repeats: int = 2, setups: Path = EASY_HOLDOUT, seed: int = 12345) -> dict[str, dict[str, float]]:
    """Weak and normal fights of held-out winners, by act and kind
    (`a1_weak`, ...) and over all (`all`): the win rate, the HP the policy
    lost and the winner lost (both net of healing), the mean gap, the share
    of fights the policy lost more HP in, and potions drunk per fight."""
    lines = [json.loads(line) for line in setups.read_text().splitlines() if line.strip()]
    per_fight: dict[int, list[End]] = defaultdict(list)
    for e in play(policy, device, repeats, "setups", seed=seed, setups=setups):
        per_fight[e.env].append(e)
    groups: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for i, ends in per_fight.items():
        line = lines[i]
        lost = float(np.mean([e.hp_lost for e in ends])) * line["max_hp"]
        row = (float(np.mean([e.won for e in ends])), lost, float(line["winner_hp_lost"]), float(np.mean([e.potions_used for e in ends])))
        act = 1 if line["game_floor"] <= 17 else 2 if line["game_floor"] <= 33 else 3
        groups[f"a{act}_{ends[0].kind.lower()}"].append(row)
        groups["all"].append(row)
    out = {}
    for name, rows in sorted(groups.items()):
        won, ours, theirs, potions = (np.array(c) for c in zip(*rows))
        out[name] = {
            "fights": len(rows),
            "win": float(won.mean()),
            "hp_lost": float(ours.mean()),
            "winner_hp_lost": float(theirs.mean()),
            "gap": float((ours - theirs).mean()),
            "worse": float((ours > theirs + 0.5).mean()),
            "potions": float(potions.mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--old-vocab", type=Path, default=None, help="vocab.txt the checkpoint was trained with, if it predates the current sim")
    ap.add_argument("--source", choices=["holdout", "recordings", "setups", "easy"], default="holdout")
    ap.add_argument("--repeats", type=int, default=2, help="fights per setup")
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    ap.add_argument("--acts", type=int, default=3, help="acts the holdout covers")
    ap.add_argument("--setups", type=Path, default=None, help="played runs' fights, for --source setups (default the ststracker holdout) or easy (its easy holdout)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device, args.old_vocab)
    policy.eval()
    if args.source == "easy":
        print(f"{'':10s} {'fights':>6s} {'won':>6s} {'HP lost':>7s} {'winner':>6s} {'gap':>5s} {'worse':>6s} {'potions':>7s}")
        for name, r in easy(policy, device, args.repeats, args.setups or EASY_HOLDOUT).items():
            print(
                f"{name:10s} {r['fights']:6.0f} {r['win']:6.1%} {r['hp_lost']:7.1f} {r['winner_hp_lost']:6.1f} "
                f"{r['gap']:+5.1f} {r['worse']:6.0%} {r['potions']:7.2f}"
            )
        return
    win, by_enc, kinds = evaluate(policy, device, args.repeats, args.source, args.recordings, acts=args.acts, setups=args.setups or HOLDOUT)
    for enc, (w, n) in sorted(by_enc.items()):
        print(f"{enc:32s} {w:4d}/{n:<4d} {w / n:6.1%}")
    print("  ".join(f"{k} {v:.1%}" for k, v in kinds.items()))
    print(f"overall {win:.1%} over {sum(n for _, n in by_enc.values())} {args.source} fights")


if __name__ == "__main__":
    main()
