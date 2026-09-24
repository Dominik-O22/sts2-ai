"""Fights of played runs, taken from ststracker run pages
(`scripts/tracker.py` keeps each page's data), for training
(`--real-setups`) and evaluation (`evaluate --source setups`).

    uv run python -m sts2ai.setups                 # pages in the tracker dir -> setups/*.jsonl

One line per fight: the run as it stood when the fight began (`start`:
deck, relics, potions, gold, ascension), `hp` and `max_hp`, the
`encounter`, the `floor` in the generator's numbering (`game_floor` is the
page's), whether it is the last act's `second` boss, and the HP the winner
lost in it (`winner_hp_lost`). Elite and boss fights go to `train.jsonl`
and `holdout.jsonl`, weak and normal ones to `easy-train.jsonl` and
`easy-holdout.jsonl`. A page lists the final deck with the floor each card
joined it, and per floor the cards gained, removed and
upgraded, so the deck at a floor is the final one with the later changes
undone. What the page leaves out stays approximate: potions are the ones
the fight used, enchantments are the final ones, and a relic traded away
is missing. Setups the sim cannot build (a card it does not have) are
left out and counted.

Runs are split by player, so no player's style is in both files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sts2ai.env import Envs

TRACKER = Path.home() / ".local/share/SlayTheSpire2/sts2ai/tracker"
TRAIN = TRACKER / "setups" / "train.jsonl"
HOLDOUT = TRACKER / "setups" / "holdout.jsonl"
# The weak and normal fights, kept apart: they are won all but always, and
# what tells play apart there is the HP it costs (`evaluate --source easy`).
EASY_TRAIN = TRACKER / "setups" / "easy-train.jsonl"
EASY_HOLDOUT = TRACKER / "setups" / "easy-holdout.jsonl"
BUILD = "v0.107.1"


def strip(name: str) -> str:
    return name.split(".", 1)[1]


def walkable(run: dict) -> bool:
    player = run["players"]
    return len(player) == 1 and player[0]["characterId"] == "CHARACTER.IRONCLAD" and run["buildId"] == BUILD and not run["modifiers"]


# `TinkerTime.RiderEffect` in order. Each rider belongs to one card type
# (`TinkerTime.cs`), so the rider alone names the sim's Mad Science card.
RIDERS = ["NONE", "SAPPING", "VIOLENCE", "CHOKING", "ENERGIZED", "WISDOM", "CHAOS", "EXPERTISE", "CURIOUS", "IMPROVEMENT"]


def card(c: dict, upgraded: bool) -> dict:
    name = strip(c["id"])
    props = {p["name"]: p["value"] for p in (c.get("props") or {}).get("ints", [])}
    if name == "MAD_SCIENCE" and props.get("TinkerTimeRider"):
        name = f"MAD_SCIENCE_{RIDERS[props['TinkerTimeRider']]}"
    out = {"id": name, "up": upgraded}
    if ench := c.get("enchantment"):
        out["ench"] = [strip(ench["id"]), ench["amount"], False]
    return out


def deck_at(page: dict, floor: int) -> list[dict]:
    """The deck as the fight on `floor` began: the final deck without the
    cards that joined from `floor` on, with the cards removed from `floor`
    on put back and the upgrades made from `floor` on undone."""
    timeline = page["floorTimeline"]
    held = [dict(c) for c in page["runDetail"]["players"][0]["deck"] if c.get("floor_added_to_deck", 0) < floor]
    later = [f for f in timeline if f["floor"] >= floor]
    for f in later:
        held += [dict(c) for c in f["cardsRemoved"] if c.get("floor_added_to_deck", 0) < floor]
    upgraded = [c.get("current_upgrade_level", 0) > 0 for c in held]
    for f in later:
        for name in f["upgradedCards"]:
            i = next((i for i, c in enumerate(held) if c["id"] == name and upgraded[i]), None)
            if i is not None:
                upgraded[i] = False
    return [card(c, up) for c, up in zip(held, upgraded)]


def fights(page: dict, kinds: tuple[str, ...]) -> list[dict]:
    """A line per fight of the run whose encounter ends in one of `kinds`
    (`_ELITE`, `_BOSS`, `_WEAK`, `_NORMAL`)."""
    run, timeline = page["runDetail"], page["floorTimeline"]
    player = run["players"][0]
    by_floor = {f["floor"]: f for f in timeline}
    out = []
    for f in timeline:
        model = f["modelId"] or ""
        if not (model.startswith("ENCOUNTER.") and model.endswith(kinds)) or f["floor"] - 1 not in by_floor:
            continue
        before = by_floor[f["floor"] - 1]
        relics = [strip(r["id"]) for r in player["relics"] if r["floorAdded"] < f["floor"]]
        slots = 4 if "POTION_BELT" in relics else 2
        potions = [strip(p) for p in f["potionsUsed"]][:slots]
        start = {
            "ascension": run["ascension"],
            "deck": deck_at(page, f["floor"]),
            "relics": relics,
            "potions": potions + [None] * (slots - len(potions)),
            "gold": before["currentGold"],
        }
        boss = model.endswith("_BOSS")
        out.append(
            {
                "start": start,
                "hp": before["currentHp"],
                "max_hp": before["maxHp"],
                "encounter": strip(model),
                "floor": sim_floor(f["floor"], boss),
                "second": boss and f["floor"] == 49,
                "game_floor": f["floor"],
                # HP in minus HP out, after Burning Blood and any healing,
                # as `End.hp_lost` counts it for the model.
                "winner_hp_lost": before["currentHp"] - f["currentHp"],
            }
        )
    return out


def sim_floor(floor: int, boss: bool) -> int:
    """A game floor in the generator's numbering (`gen::act_floor`: acts of
    16 floors, the boss on the 16th). The game's act 1 has Neow's floor in
    front (its boss is floor 17), and act 3 has two bosses (48 and 49)."""
    act = 0 if floor <= 17 else 1 if floor <= 33 else 2
    if boss:
        return act * 16 + 16
    first = (1, 18, 34)[act]
    return act * 16 + min(15, floor - first + 1)


def holdout_player(player: str, share: float) -> bool:
    """A fixed `share` of players, by a hash of their id."""
    return int(hashlib.sha256(player.encode()).hexdigest(), 16) % 1000 < share * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=TRACKER / "pages")
    ap.add_argument("--out", type=Path, default=HOLDOUT.parent)
    ap.add_argument("--holdout", type=float, default=0.15, help="share of players held out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probe = Envs(1, seed=0)
    # (file prefix, encounter kinds): elites and bosses to win, the rest to win cheaply.
    groups = [("", ("_ELITE", "_BOSS")), ("easy-", ("_WEAK", "_NORMAL"))]
    kept: dict[str, list[str]] = {f"{prefix}{split}": [] for prefix, _ in groups for split in ("train", "holdout")}
    dropped: Counter[str] = Counter()
    runs = 0
    for path in sorted(args.pages.glob("*.json")):
        page = json.loads(path.read_text())
        if not walkable(page["runDetail"]):
            continue
        runs += 1
        split = "holdout" if holdout_player(path.name.rsplit("-", 1)[0], args.holdout) else "train"
        for prefix, kinds in groups:
            for fight in fights(page, kinds):
                line = json.dumps({**fight, "run": path.stem})
                try:
                    probe.sim.use_setups(line, 1)
                except ValueError as e:
                    dropped[str(e).split(": ", 1)[-1].split(":")[0]] += 1
                    continue
                kept[f"{prefix}{split}"].append(line)
    for name, lines in kept.items():
        (args.out / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{runs} runs: " + ", ".join(f"{len(lines)} {name}" for name, lines in kept.items()) + f" fights in {args.out}")
    print(f"{sum(dropped.values())} left out: " + ", ".join(f"{k} {v}" for k, v in dropped.most_common(8)))


if __name__ == "__main__":
    main()
