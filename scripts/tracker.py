"""Turn a run page on ststracker.app into the game's .run history format,
as far as the page keeps it, so the run layer can walk it
(`sim/examples/runcheck.rs`).

    python3 scripts/tracker.py            # paste run URLs one at a time, Enter on an empty line to stop
    python3 scripts/tracker.py URL...     # the same for the URLs given
    python3 scripts/tracker.py --reconvert  # rewrite the saved runs after a change here

Run URLs look like https://ststracker.app/<steam id>/runs/<start time>.run.
Each run is saved as `<steam id>-<start time>.run` in OUT (`--out`, default
~/.local/share/SlayTheSpire2/sts2ai/tracker), where runcheck reads it:

    cd sim && cargo run --release --example runcheck -- --effects ~/.local/share/SlayTheSpire2/sts2ai/tracker/*.run

The page (SvelteKit) serves its data at `URL/__data.json`, flattened by
devalue. It drops potion offers, the ancients' unchosen options,
transforms, enchantments made on the way, and the page of an event an
option sat on. Event pages are read off this machine's own history files;
every floor is marked `potions_unrecorded`, which the history check takes
as "the potions drawn were offered".
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

GAME_DATA = Path.home() / ".local/share/SlayTheSpire2"
HISTORY = GAME_DATA / "steam"


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


def room_type(floor: dict) -> str:
    """The history's `room_type`: an unknown point is the room it became,
    which the page names only for events and fights. A shop lays out cards
    or relics to buy; a chest gives one relic."""
    point, model = floor["mapPointType"], floor["modelId"]
    if model and model.startswith("EVENT."):
        return "event"
    if model and model.startswith("ENCOUNTER."):
        return "boss" if model.endswith("_BOSS") else "elite" if model.endswith("_ELITE") else "monster"
    if point == "unknown":
        shelf = floor["cardsSkipped"] or floor["relicsSkipped"] or floor["goldSpent"]
        return "shop" if shelf else "treasure"
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
        room = {"room_type": room_type(f), "turns_taken": f["turnsTaken"]}
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


def write(page: dict, path: Path, pages: dict[tuple[str, str], str]) -> None:
    """Writes the page's run to `path` and says what it is."""
    run = page["runDetail"]
    path.write_text(json.dumps(convert(page, pages)))
    character = run["players"][0]["characterId"].split(".")[-1].lower()
    outcome = "won" if run["win"] else f"lost to {run['killedByEncounter'].split('.')[-1].lower()}"
    print(f"  {path.name}: {character} A{run['ascension']} {run['buildId']}, {outcome} on floor {len(page['floorTimeline'])}, seed {run['seed']}")
    if run["buildId"] != "v0.107.1" or character != "ironclad" or len(run["players"]) != 1 or run["modifiers"]:
        print("  (runcheck skips it: only solo Ironclad runs on v0.107.1 without modifiers)")
    missing = sorted({f"{c['eventId']}.{c['choiceId']}" for f in page["floorTimeline"] for c in f["eventChoices"]} - {f"{e}.{o}" for e, o in pages})
    if missing:
        print(f"  event page unknown, INITIAL assumed: {', '.join(missing)}")


def save(url: str, out: Path, pages: dict[tuple[str, str], str]) -> None:
    """Fetches the run at `url` and writes it into `out`, keeping the page's
    data in `out/pages` so `--reconvert` can redo it."""
    parts = url.strip().rstrip("/").split("/")
    if len(parts) < 3 or parts[-2] != "runs" or not parts[-1].endswith(".run"):
        raise ValueError("not a run URL (…/<steam id>/runs/<start time>.run)")
    name = f"{parts[-3]}-{parts[-1]}"
    page = fetch(url.strip())
    (out / "pages" / f"{name}.json").write_text(json.dumps(page))
    write(page, out / name, pages)


def main() -> None:
    ap = argparse.ArgumentParser(description="Save ststracker.app runs as .run history files.")
    ap.add_argument("urls", nargs="*", help="run pages; without any, read them from the prompt")
    ap.add_argument("--out", type=Path, default=GAME_DATA / "sts2ai/tracker")
    ap.add_argument("--reconvert", action="store_true", help="rewrite every saved run from its kept page, fetching nothing")
    args = ap.parse_args()
    (args.out / "pages").mkdir(parents=True, exist_ok=True)
    pages = event_pages(list(HISTORY.glob("**/saves/history")))
    if args.reconvert:
        for kept in sorted((args.out / "pages").glob("*.json")):
            write(json.loads(kept.read_text()), args.out / kept.stem, pages)
        return
    urls = iter(args.urls) if args.urls else iter(lambda: input("run URL (Enter to stop): "), "")
    for url in urls:
        try:
            save(url, args.out, pages)
        except Exception as e:  # a bad paste or a page that moved: say so and take the next
            print(f"  failed: {e}")
    print(f"{len(list(args.out.glob('*.run')))} runs in {args.out}")


if __name__ == "__main__":
    main()
