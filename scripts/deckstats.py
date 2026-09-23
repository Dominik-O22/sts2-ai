#!/usr/bin/env python3
"""What the decks in recorded fights look like, so the generator in
`sim/src/gen.rs` can be set from real runs instead of guesses.

    uv run python scripts/deckstats.py                # ~/.local/share/SlayTheSpire2/sts2ai/recordings
    uv run python scripts/deckstats.py DIR [DIR ...]  # other folders (dev/ holds console fights)
    uv run python scripts/deckstats.py --gen GEN.jsonl   # beside generated fights, by act and kind

`GEN.jsonl` comes from `cd sim && cargo run --release --example gendump >
GEN.jsonl`: generated fights in the same `start` format.

Reads the `start` record of every recording: deck, relics, potions, HP, and
the enemies. Prints per fight one line, then the averages the generator
needs: picks beyond the starter deck, upgrades, removals, enchanted cards,
relics, potions, and starting HP, split by encounter kind.

Only played runs count: a start record carries its run's seed, and runs
the mod saw console lines change (`scripted`, which is how record.py sets
fights up) are left out. Averages weigh every run the same, however many
fights it has, so one long run does not stand in for all of them. Older
recordings carry no seed; `--all` counts them too, split into runs by time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

RECORDINGS = Path.home() / ".local/share/SlayTheSpire2/sts2ai/recordings"
STARTER = Counter({"STRIKE_IRONCLAD": 5, "DEFEND_IRONCLAD": 4, "BASH": 1})
IRONCLAD_HP = 80


def runs_by_time(rows: list[dict]) -> None:
    """Give seedless rows (sorted by time) a run: a new one after an hour's
    gap or when the deck or relics shrink, which a run never does."""
    run, last = 0, None
    for r in rows:
        if last and (r["time"] - last["time"] > 3600 or r["cards"] < last["cards"] - 3 or r["relics"] < last["relics"] - 1):
            run += 1
        r["run"], last = f"time-{run}", r


def start_record(path: Path) -> dict | None:
    """The `start` record with the player's HP from the first snapshot
    folded in, which is how `FightSetup::from_recording` reads it too."""
    start = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        if v.get("t") == "start" and start is None:
            start = v
        elif v.get("t") == "snapshot" and start is not None:
            start["hp"], start["max_hp"] = v["hp"], v["max_hp"]
            return start
    return None


def card_id(ref: dict | str) -> str:
    return ref["id"] if isinstance(ref, dict) else ref


def stats(start: dict) -> dict[str, float | int | str]:
    deck = start["deck"]
    ids = Counter(card_id(c) for c in deck)
    basics = sum(min(ids[k], n) for k, n in STARTER.items())
    removed = sum(STARTER.values()) - basics
    kind = start.get("room", "?")
    return {
        "encounter": start["encounter"],
        "kind": kind,
        "cards": len(deck),
        "picks": len(deck) - basics - ids["ASCENDERS_BANE"],
        "removed": removed,
        "upgraded": sum(1 for c in deck if isinstance(c, dict) and c.get("up")),
        "enchanted": sum(1 for c in deck if isinstance(c, dict) and c.get("ench")),
        "relics": len(start["relics"]),
        "potions": sum(1 for p in start["potions"] if p),
        "hp": start.get("hp", 0),
        "max_hp": start.get("max_hp", IRONCLAD_HP),
    }


KEYS = ["cards", "picks", "removed", "upgraded", "enchanted", "relics", "potions"]


def averages(rows: list[dict]) -> str:
    """The averages of `KEYS` and HP kept, as table cells: each run's mean
    first, then the mean over runs."""
    runs: dict[str, list[dict]] = {}
    for r in rows:
        runs.setdefault(r.get("run", r["encounter"] + str(id(r))), []).append(r)

    def over_runs(value) -> float:
        return mean(mean(value(r) for r in rs) for rs in runs.values())

    hp = over_runs(lambda r: r["hp"] / max(1, r["max_hp"])) if all(r["hp"] for r in rows) else float("nan")
    return " ".join(f"{over_runs(lambda r, k=k: r[k]):5.1f}" for k in KEYS) + f" {hp:5.0%}"


def compare(real: list[dict], generated: list[dict]) -> None:
    """Real and generated averages side by side, per act and room kind."""
    from sts2ai import _sim

    act = {name: a for name, a, _ in _sim.encounters()}
    for rows in (real, generated):
        for r in rows:
            r["act"] = {"Overgrowth": 1, "Underdocks": 1, "Hive": 2, "Glory": 3}.get(act.get(r["encounter"], ""), 0)
    print(f"{'act kind':14s} {'source':9s} {'runs':>4s} {'n':>5s} " + " ".join(f"{k[:5]:>5s}" for k in KEYS) + "   hp%")
    for a in (1, 2, 3):
        for kind in ("Monster", "Elite", "Boss"):
            for source, rows in (("real", real), ("generated", generated)):
                sub = [r for r in rows if r["act"] == a and r["kind"] == kind]
                if sub:
                    runs = len({r.get("run", id(r)) for r in sub})
                    print(f"{f'{a} {kind}':14s} {source:9s} {runs:4d} {len(sub):5d} " + averages(sub))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path, default=[RECORDINGS])
    ap.add_argument("--gen", type=Path, help="generated fights (gendump) to set beside the recordings")
    ap.add_argument("--all", action="store_true", help="count recordings from before runs carried a seed, split into runs by time")
    args = ap.parse_args()
    rows, seedless = [], []
    for d in args.dirs:
        for path in sorted(d.glob("*.jsonl")):
            if (start := start_record(path)) is None or start.get("scripted"):
                continue
            r = stats(start)
            r["time"] = path.stat().st_mtime
            if start.get("seed"):
                r["run"] = start["seed"]
                rows.append(r)
            else:
                seedless.append(r)
    if args.all:
        runs_by_time(sorted(seedless, key=lambda r: r["time"]))
        rows += seedless
    if not rows:
        raise SystemExit(f"no played-run recordings in {', '.join(map(str, args.dirs))} (older ones need --all)")
    if args.gen:
        compare(rows, [stats(json.loads(line)) for line in args.gen.read_text().splitlines() if line.strip()])
        return
    print(f"{'encounter':32s} {'kind':8s} cards picks rm  up  ench relics pots   hp")
    for r in rows:
        print(
            f"{r['encounter']:32s} {r['kind']:8s} {r['cards']:5d} {r['picks']:5d} {r['removed']:2d} {r['upgraded']:3d} "
            f"{r['enchanted']:5d} {r['relics']:6d} {r['potions']:4d} {r['hp']:3d}/{r['max_hp']}"
        )
    print()
    print(f"{'average':41s} " + " ".join(f"{k[:5]:>5s}" for k in KEYS) + "   hp%")
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"{kind:32s} n={len(sub):<6d} " + averages(sub))
    print("\npicks = cards beyond the starter deck (plus Ascender's Bane); rm = starter cards removed")


if __name__ == "__main__":
    main()
