"""Turn a run page on ststracker.app into the game's .run history format,
as far as the page keeps it, so the run layer can walk it
(`sim/examples/runcheck.rs`).

    uv run python scripts/tracker.py URL OUT.run      # e.g. https://ststracker.app/<steam id>/runs/<start time>.run

The page (SvelteKit) serves its data at `URL/__data.json`, flattened by
devalue. It drops potion offers, the ancients' unchosen options,
transforms, enchantments made on the way, and the page of an event an
option sat on. Event pages are read off this machine's own history files;
every floor is marked `potions_unrecorded`, which the history check takes
as "the potions drawn were offered".
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HISTORY = Path.home() / ".local/share/SlayTheSpire2/steam"


def unflatten(values: list) -> object:
    """devalue's flat form back into values: each array or object holds
    indexes into `values`, -1 is undefined, and a tagged array (`["Date",
    ...]`) is kept as its payload."""
    cache: dict[int, object] = {}

    def get(i):
        if i == -1:
            return None
        if i in cache:
            return cache[i]
        v = values[i]
        if isinstance(v, list):
            if v and isinstance(v[0], str) and v[0] in ("Date", "Set", "Map", "BigInt", "RegExp"):
                out = v[1] if len(v) > 1 else None
                cache[i] = out
                return out
            out = []
            cache[i] = out
            out.extend(get(j) for j in v)
            return out
        if isinstance(v, dict):
            out = {}
            cache[i] = out
            for k, j in v.items():
                out[k] = get(j)
            return out
        cache[i] = v
        return v

    return get(0)


def event_pages(dirs: list[Path]) -> dict[tuple[str, str], str]:
    """(event, option) to the page it sits on, from the `event_choices`
    title keys of the .run files in `dirs`."""
    pages = {}
    for d in dirs:
        for path in d.glob("*.run"):
            for act in json.loads(path.read_text()).get("map_point_history", []):
                for floor in act:
                    for stats in floor.get("player_stats", []):
                        for c in stats.get("event_choices", []):
                            parts = c["title"]["key"].split(".")
                            if len(parts) == 6 and parts[1] == "pages":
                                pages[(parts[0], parts[4])] = parts[2]
    return pages


def room_type(point: str, model: str | None) -> str:
    """The history's `room_type`: an unknown point is the room it became."""
    if model and model.startswith("EVENT."):
        return "event"
    if model and model.startswith("ENCOUNTER."):
        return "boss" if model.endswith("_BOSS") else "elite" if model.endswith("_ELITE") else "monster"
    return point


def card(c: dict) -> dict:
    """A card as the history lists it: the page writes floor 0 where the
    record has no `floor_added_to_deck`."""
    return {k: v for k, v in c.items() if k != "floor_added_to_deck" or v}


def convert(page: dict, pages: dict[tuple[str, str], str]) -> dict:
    """The page's run (`runDetail`, `floorTimeline`) as a .run record."""
    run, timeline = page["runDetail"], page["floorTimeline"]
    player = run["players"][0]
    history = [[] for _ in run["acts"]]
    for f in timeline:
        model = f["modelId"]
        room = {"room_type": room_type(f["mapPointType"], model), "turns_taken": f["turnsTaken"]}
        if model:
            room["model_id"] = model
        if f["monsterIds"]:
            room["monster_ids"] = f["monsterIds"]
        stats = {
            "player_id": 1,
            "current_hp": f["currentHp"],
            "max_hp": f["maxHp"],
            "current_gold": f["currentGold"],
            "damage_taken": f["damageTaken"],
            "gold_gained": f["goldGained"],
            "gold_spent": f["goldSpent"],
            "hp_healed": f["hpHealed"],
            "potions_unrecorded": True,
        }
        choices = [{"card": card(c), "was_picked": True} for c in f["cardsPicked"]]
        choices += [{"card": card(c), "was_picked": False} for c in f["cardsSkipped"]]
        if choices:
            stats["card_choices"] = choices
        relics = [{"choice": r, "was_picked": True} for r in f["relicsPicked"]]
        relics += [{"choice": r, "was_picked": False} for r in f["relicsSkipped"]]
        if f["mapPointType"] == "ancient" and relics:
            stats["ancient_choice"] = [{"TextKey": r.split(".", 1)[1], "was_chosen": True} for r in f["relicsPicked"]]
        if relics:
            stats["relic_choices"] = relics
        for key, cards in [("cards_gained", f["cardsGained"]), ("cards_removed", f["cardsRemoved"])]:
            if cards:
                stats[key] = [card(c) for c in cards]
        if f["upgradedCards"]:
            stats["upgraded_cards"] = f["upgradedCards"]
        if f["restSiteChoices"]:
            stats["rest_site_choices"] = f["restSiteChoices"]
        if f["potionsUsed"]:
            stats["potion_used"] = f["potionsUsed"]
        if f["eventChoices"]:
            stats["event_choices"] = [
                {"title": {"key": f"{c['eventId']}.pages.{pages.get((c['eventId'], c['choiceId']), 'INITIAL')}.options.{c['choiceId']}.title", "table": "events"}}
                for c in f["eventChoices"]
            ]
        history[f["actIndex"]].append({"map_point_type": f["mapPointType"], "player_stats": [stats], "rooms": [room]})
    return {
        "acts": [a["actName"] for a in run["acts"]],
        "ascension": run["ascension"],
        "build_id": run["buildId"],
        "game_mode": run["gameMode"],
        "killed_by_encounter": run["killedByEncounter"],
        "killed_by_event": run["killedByEvent"],
        "map_point_history": history,
        "modifiers": run["modifiers"],
        "players": [
            {
                "character": player["characterId"],
                "deck": player["deck"],
                "relics": [{"id": r["id"], "floor_added_to_deck": r["floorAdded"]} for r in player["relics"]],
                "potions": [{"id": p["id"], "slot_index": p["slotIndex"]} for p in player["potions"]],
                "id": 1,
            }
        ],
        "seed": run["seed"],
        "start_time": int(run["startTime"]),
        "was_abandoned": run["wasAbandoned"],
        "win": run["win"],
    }


def fetch(url: str) -> dict:
    """The run page's data: the last node of its SvelteKit load."""
    with urllib.request.urlopen(url.rstrip("/") + "/__data.json") as r:
        nodes = json.load(r)["nodes"]
    return unflatten(nodes[-1]["data"])


if __name__ == "__main__":
    url, out = sys.argv[1:]
    pages = event_pages(list(HISTORY.glob("**/saves/history")))
    page = fetch(url)
    missing = sorted({(c["eventId"], c["choiceId"]) for f in page["floorTimeline"] for c in f["eventChoices"]} - set(pages))
    if missing:
        print(f"event page unknown, INITIAL assumed: {missing}")
    Path(out).write_text(json.dumps(convert(page, pages)))
