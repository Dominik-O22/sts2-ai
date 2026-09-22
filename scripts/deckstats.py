#!/usr/bin/env python3
"""What the decks in recorded fights look like, so the generator in
`sim/src/gen.rs` can be set from real runs instead of guesses.

    uv run python scripts/deckstats.py                # ~/.local/share/SlayTheSpire2/sts2ai/recordings
    uv run python scripts/deckstats.py DIR [DIR ...]  # other folders (dev/ holds console fights)

Reads the `start` record of every recording: deck, relics, potions, HP, and
the enemies. Prints per fight one line, then the averages the generator
needs: picks beyond the starter deck, upgrades, removals, enchanted cards,
relics, potions, and starting HP, split by encounter kind.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

RECORDINGS = Path.home() / ".local/share/SlayTheSpire2/sts2ai/recordings"
STARTER = Counter({"STRIKE_IRONCLAD": 5, "DEFEND_IRONCLAD": 4, "BASH": 1})
IRONCLAD_HP = 80


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


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [RECORDINGS]
    rows = []
    for d in dirs:
        for path in sorted(d.glob("*.jsonl")):
            if (start := start_record(path)) is not None:
                rows.append(stats(start))
    if not rows:
        sys.exit(f"no recordings with a start record in {', '.join(map(str, dirs))}")
    print(f"{'encounter':32s} {'kind':8s} cards picks rm  up  ench relics pots   hp")
    for r in rows:
        print(
            f"{r['encounter']:32s} {r['kind']:8s} {r['cards']:5d} {r['picks']:5d} {r['removed']:2d} {r['upgraded']:3d} "
            f"{r['enchanted']:5d} {r['relics']:6d} {r['potions']:4d} {r['hp']:3d}/{r['max_hp']}"
        )
    print()
    keys = ["cards", "picks", "removed", "upgraded", "enchanted", "relics", "potions"]
    print(f"{'average':41s} " + " ".join(f"{k[:5]:>5s}" for k in keys) + "   hp%")
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        hp = mean(r["hp"] / max(1, r["max_hp"]) for r in sub) if all(r["hp"] for r in sub) else float("nan")
        print(f"{kind:32s} n={len(sub):<6d} " + " ".join(f"{mean(r[k] for r in sub):5.1f}" for k in keys) + f" {hp:5.0%}")
    print("\npicks = cards beyond the starter deck (plus Ascender's Bane); rm = starter cards removed")


if __name__ == "__main__":
    main()
