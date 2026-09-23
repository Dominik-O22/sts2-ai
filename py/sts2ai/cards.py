"""Card reward advice: which offered card, or none, makes the deck fight
best, found by playing the act's elites and bosses with the trained policy.

    uv run python -m sts2ai.cards runs/<run>/latest.pt              # watch the game, advise each reward
    uv run python -m sts2ai.cards runs/<run>/latest.pt --recording FILE --offer INFLAME,SHRUG_IT_OFF

Watching, it polls the mod's `run.json`, which carries the cards on the
reward screen while it is open (`card_reward`), and prints a ranking for
each new reward. With `--recording`, the run is a recorded fight's start.

For each option the deck (with the card added) plays `repeats` fights
against every elite of the act and the boss the map shows, greedy, and the options are ranked
by the mean fight reward (the training reward: a win, plus HP and potions
kept). Every option faces the same enemies. Elites are fought at the run's
current HP, bosses at full HP (a rest site comes first). The score looks at
the deck as it stands: it does not plan for the picks and upgrades ahead.
At 512 fights per encounter two options closer than about 0.03 in value
are a tie: that is how far a run with another seed moves them.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import DEFAULT_RECORDINGS, End, Envs, Layout
from sts2ai.model import Policy, load_policy, masked_logits

RUN_STATE = DEFAULT_RECORDINGS.parent / "run.json"
BOSS_FLOOR = 16
# Where in an act the generator puts elites; only the enemy roll reads it.
ELITE_FLOOR = 10


@dataclass(frozen=True)
class Verdict:
    """One option's fights: `card` is None for skipping the reward."""

    card: dict | None
    value: float
    win: float
    win_by_encounter: dict[str, float]


def upcoming(act: str, bosses: list[str] | None = None) -> list[tuple[str, int, str]]:
    """The elites of `act` (the sim's act name, "Overgrowth" to "Glory") and
    its bosses, as (game name, floor, kind). `bosses` are the ones the map
    shows (game names); without them, every boss of the act."""
    index = {"Overgrowth": 0, "Underdocks": 0, "Hive": 1, "Glory": 2}[act]
    return [
        (name, index * BOSS_FLOOR + (BOSS_FLOOR if kind == "Boss" else ELITE_FLOOR), kind)
        for name, enc_act, kind in _sim.encounters()
        if enc_act == act and (kind == "Elite" or (kind == "Boss" and (not bosses or name in bosses)))
    ]


@torch.no_grad()
def fights(policy: Policy, device: torch.device, start: dict, hp: int, max_hp: int, act: str, repeats: int, seed: int) -> list[End]:
    """`repeats` greedy fights of the run in `start` against each elite (at
    `hp`) and boss (at `max_hp`, the ones in `start["bosses"]` if given) of
    `act`, every env's first fight only."""
    ends: list[End] = []
    for kind, fight_hp in (("Elite", hp), ("Boss", max_hp)):
        encounters = [(name, floor) for name, floor, k in upcoming(act, start.get("bosses")) if k == kind]
        if not encounters:
            continue
        n = len(encounters) * repeats
        envs = Envs(n, seed=seed)
        envs.sim.use_run(json.dumps(start), fight_hp, max_hp, encounters, repeats, seed)
        envs.sim.observe(envs.floats, envs.ids, envs.mask)
        done = np.zeros(n, bool)
        while not done.all():
            logits, _ = policy(torch.from_numpy(envs.floats).to(device), torch.from_numpy(envs.ids).to(device))
            actions = masked_logits(logits.float(), torch.from_numpy(envs.mask).to(device)).argmax(1).cpu().numpy()
            for e in envs.step(actions):
                if not done[e.env]:
                    done[e.env] = True
                    ends.append(e)
    return ends


def rank(
    policy: Policy, device: torch.device, start: dict, hp: int, max_hp: int, act: str, options: list[dict], repeats: int = 512, seed: int = 1
) -> list[Verdict]:
    """Every option (and skipping), best first."""
    verdicts = []
    for card in [None, *options]:
        run = {**start, "deck": start["deck"] + ([card] if card else [])}
        ends = fights(policy, device, run, hp, max_hp, act, repeats, seed)
        by_enc: dict[str, list[bool]] = defaultdict(list)
        for e in ends:
            by_enc[e.encounter].append(e.won)
        verdicts.append(
            Verdict(
                card=card,
                value=float(np.mean([e.reward for e in ends])),
                win=float(np.mean([e.won for e in ends])),
                win_by_encounter={k: float(np.mean(v)) for k, v in by_enc.items()},
            )
        )
    return sorted(verdicts, key=lambda v: -v.value)


def describe(verdicts: list[Verdict]) -> str:
    """The ranking as text, each option against skipping."""
    skip = next(v for v in verdicts if v.card is None)
    lines = []
    for v in verdicts:
        name = "skip" if v.card is None else v.card["id"] + ("+" if v.card.get("up") else "")
        worst = min(v.win_by_encounter, key=v.win_by_encounter.get)
        lines.append(
            f"{name:24s} value {v.value:+.3f} ({v.value - skip.value:+.3f} vs skip)  wins {v.win:.0%}  worst {worst} {v.win_by_encounter[worst]:.0%}"
        )
    return "\n".join(lines)


def watch(policy: Policy, device: torch.device, repeats: int, poll: float = 0.5) -> None:
    """Advise every card reward the game shows, until interrupted."""
    print(f"watching {RUN_STATE} for card rewards")
    last = None
    while True:
        try:
            run = json.loads(RUN_STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            run = {}
        offer = run.get("card_reward")
        if offer and offer != last:
            last = offer
            print(f"\n{run['act']}, {run['hp']}/{run['max_hp']} HP, {len(run['deck'])} cards")
            print(describe(rank(policy, device, run, run["hp"], run["max_hp"], run["act"], offer, repeats)), flush=True)
        elif not offer:
            last = None
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--recording", type=Path, help="advise on this recorded fight's run instead of watching the game")
    ap.add_argument("--offer", help="with --recording: offered cards by game id, comma separated; a trailing + means upgraded")
    ap.add_argument("--repeats", type=int, default=512, help="fights per elite and boss per option")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device).eval()
    load_policy(args.checkpoint, policy, device)
    if args.recording is None:
        watch(policy, device, args.repeats)
        return
    lines = [json.loads(line) for line in args.recording.read_text().splitlines() if line.strip()]
    start = next(r for r in lines if r["t"] == "start")
    snap = next(r for r in lines if r["t"] == "snapshot")
    act = next(a for name, a, _ in _sim.encounters() if name == start["encounter"])
    options = [{"id": c.rstrip("+"), "up": c.endswith("+")} for c in args.offer.split(",")]
    print(f"{act}, {snap['hp']}/{snap['max_hp']} HP, {len(start['deck'])} cards")
    print(describe(rank(policy, device, start, snap["hp"], snap["max_hp"], act, options, args.repeats)))


if __name__ == "__main__":
    main()
