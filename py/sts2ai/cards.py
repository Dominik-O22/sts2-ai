"""Deck advice: which card to take, upgrade, remove or buy, found by
playing the act's elites and boss with the trained policy.

    uv run python -m sts2ai.cards runs/<run>/latest.pt              # watch the game, advise each choice
    uv run python -m sts2ai.cards runs/<run>/latest.pt --recording FILE --offer INFLAME,SHRUG_IT_OFF

Watching, it polls the mod's `run.json` and prints a ranking for each new
card reward (`card_reward`), deck pick outside combat (`deck_choice`: rest
site and event upgrades, shop and event removals, and event offers of new
cards, where each card is priced alone, so "pick 2" means the top two) and
shop (`shop`, the cards for sale with their prices). Picks from the deck
it cannot price (a transform is random) are named and left alone. With `--recording`, the run is a
recorded fight's start and the options are cards to add.

For each option the changed deck plays `repeats` fights against every
elite of the act and the boss the map shows, greedy, and the options are
ranked by the mean fight reward (the training reward: a win, plus HP and
potions kept) against keeping the deck as it is. Every option faces the same enemies. Elites are fought at the run's
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
from typing import Literal

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
class Change:
    """One way the deck can change. `card` is as the recorder writes it
    (`id`, `up`, maybe `ench`); for `upgrade` and `remove` it is a card in
    the deck, and the first copy of it changes."""

    kind: Literal["add", "upgrade", "remove"]
    card: dict
    note: str = ""

    def apply(self, deck: list[dict]) -> list[dict]:
        if self.kind == "add":
            return deck + [self.card]
        i = deck.index(self.card)
        rest = deck[:i] + deck[i + 1 :]
        return rest if self.kind == "remove" else rest[:i] + [{**self.card, "up": True}] + rest[i:]

    def label(self) -> str:
        name = self.card["id"] + ("+" if self.card.get("up") else "")
        return f"{self.kind} {name}{self.note}"


@dataclass(frozen=True)
class Verdict:
    """One option's fights: `change` is None for keeping the deck."""

    change: Change | None
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
    policy: Policy, device: torch.device, start: dict, hp: int, max_hp: int, act: str, changes: list[Change], repeats: int = 512, seed: int = 1
) -> list[Verdict]:
    """Every change, and keeping the deck, best first. Changes that do the
    same thing (upgrading one of five Strikes) are tried once."""
    unique = list({json.dumps([c.kind, c.card], sort_keys=True): c for c in changes}.values())
    verdicts = []
    for change in [None, *unique]:
        run = {**start, "deck": change.apply(start["deck"]) if change else start["deck"]}
        ends = fights(policy, device, run, hp, max_hp, act, repeats, seed)
        by_enc: dict[str, list[bool]] = defaultdict(list)
        for e in ends:
            by_enc[e.encounter].append(e.won)
        verdicts.append(
            Verdict(
                change=change,
                value=float(np.mean([e.reward for e in ends])),
                win=float(np.mean([e.won for e in ends])),
                win_by_encounter={k: float(np.mean(v)) for k, v in by_enc.items()},
            )
        )
    return sorted(verdicts, key=lambda v: -v.value)


def describe(verdicts: list[Verdict]) -> str:
    """The ranking as text, each option against keeping the deck."""
    keep = next(v for v in verdicts if v.change is None)
    lines = []
    for v in verdicts:
        name = "keep deck" if v.change is None else v.change.label()
        bosses = {k: w for k, w in v.win_by_encounter.items() if k.endswith("Boss")}
        elites = {k: w for k, w in v.win_by_encounter.items() if not k.endswith("Boss")}
        worst = min(elites, key=elites.get) if elites else None
        lines.append(
            f"{name:32s} value {v.value:+.3f} ({v.value - keep.value:+.3f})  wins {v.win:.0%}  "
            + "  ".join(f"{k.removesuffix('Boss')} {w:.0%}" for k, w in sorted(bosses.items()))
            + (f"  worst elite {worst.removesuffix('Elite')} {elites[worst]:.0%}" if worst else "")
        )
    return "\n".join(lines)


def in_deck(cards: list[dict], deck: list[dict]) -> bool:
    """Whether every one of `cards` is a card of `deck`, copies counted."""
    rest = [json.dumps(c, sort_keys=True) for c in deck]
    for c in cards:
        key = json.dumps(c, sort_keys=True)
        if key not in rest:
            return False
        rest.remove(key)
    return True


def choices(run: dict) -> dict[str, list[Change] | str]:
    """What `run.json` offers right now, by kind of choice: the changes to
    price, or why a choice is not priced."""
    out: dict[str, list[Change] | str] = {}
    if run.get("card_reward"):
        out["card reward"] = [Change("add", c) for c in run["card_reward"]]
    if pick := run.get("deck_choice"):
        kind = {"TO_UPGRADE": "upgrade", "TO_REMOVE": "remove"}.get(pick.get("prompt"))
        if kind is None and not in_deck(pick["options"], run["deck"]):
            # Cards from outside the deck (an event's offer, like Room Full
            # of Cheese): whatever is picked joins the deck.
            kind = "add"
        out[f"deck pick {pick.get('prompt')}"] = [Change(kind, c) for c in pick["options"]] if kind else "not priced"
    if shop := run.get("shop"):
        if shop["cards"]:
            out["shop"] = [Change("add", e["card"], f" ({e['cost']}g)") for e in shop["cards"]]
    return out


def watch(policy: Policy, device: torch.device, repeats: int, poll: float = 0.5) -> None:
    """Advise every choice the game shows, until interrupted."""
    print(f"watching {RUN_STATE} for card rewards, deck picks and shops")
    seen: dict[str, str] = {}
    while True:
        try:
            run = json.loads(RUN_STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            run = {}
        now = choices(run) if run.get("active") else {}
        for what, changes in now.items():
            key = json.dumps(changes if isinstance(changes, str) else [[c.kind, c.card] for c in changes])
            if seen.get(what) == key:
                continue
            seen[what] = key
            print(f"\n{what}: {run['act']}, {run['hp']}/{run['max_hp']} HP, {len(run['deck'])} cards")
            if isinstance(changes, str):
                print(changes, flush=True)
                continue
            n = repeats if len(changes) <= 4 else repeats // 2
            print(describe(rank(policy, device, run, run["hp"], run["max_hp"], run["act"], changes, n)), flush=True)
        for what in set(seen) - set(now):
            del seen[what]
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
    changes = [Change("add", {"id": c.rstrip("+"), "up": c.endswith("+")}) for c in args.offer.split(",")]
    print(f"{act}, {snap['hp']}/{snap['max_hp']} HP, {len(start['deck'])} cards")
    print(describe(rank(policy, device, start, snap["hp"], snap["max_hp"], act, changes, args.repeats)))


if __name__ == "__main__":
    main()
