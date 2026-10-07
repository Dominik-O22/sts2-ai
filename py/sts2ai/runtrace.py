"""Per-run traces: what each run fought, decided and carried, one JSON line
a run (`runplay --runs-out`, written by the sim as each run ends:
`sim::runtrace`), for `sts2ai.runreport` and `sts2ai.paired`.

    uv run python -m sts2ai.runtrace runs/ab/NAME/a.jsonl   # check a trace's invariants

The file's first line is `{"argv": [...]}`; each other line is a run:

    seed, end, act, floor, deck   how it ended ("won", "died", "stuck: ..."),
                                  in which act (0-based), on which floor, and
                                  the cards in the last fight's deck
    fights     one per fight in order: floor, act, kind (Weak, Normal, Elite,
               Boss), enc, won, hp and max_hp going in, lost (HP, net of
               heals up to the fight's end: Burning Blood's +6 can make it
               negative), potions held going in, used (potions), steps
               (combat actions), cards (deck size), gold
    decisions  one per run decision the run policy made, in order, each
               [floor, kind, pick, options]: kind is the decision's (Path,
               Card, Shop, ...), options the distinct options offered, each
               its option kind and what it names ("Card INFLAME+", "Path
               Elite", "ShopRelic ANCHOR 150g"), and pick indexes them
    decks      the deck at each boss fight's start and at the last fight's
               (when that is no boss): floor, act, enc, hp, max_hp, gold, cards
               ("BASH+", "STRIKE~SHARP2" for an enchantment), relics, potions

A run whose decisions the sim makes (`--choices`) has no decisions. Event
options carry only a hash of their key (`Event#417`) and what they name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sts2ai import _sim
from sts2ai.env import RunLayout

NAMES = _sim.run_names()
CARDS = ["-"] + _sim.game_ids()["card"]
POTIONS = ["-"] + _sim.game_ids()["potion"]
ENCHANTS = ["-"] + _sim.game_ids()["enchant"]
# Each encounter's kind (Weak, Normal, Elite, Boss) by its game id.
KINDS = {name: kind for name, _, kind in _sim.encounters()}


def option_text(L: RunLayout, floats: np.ndarray, ids: np.ndarray, k: int, paths: bool = True) -> str:
    """Option token `k` of a run row in words: its kind and what it names;
    a map step with `paths` also the elites, rests and unknowns ahead."""
    i, f = L.i_options + k * L.option_ids, L.f_options + k * L.option_floats
    C = L.option_cards
    kind = NAMES["option"][ids[i]]
    parts = [kind]
    for c in range(C):
        if ids[i + 1 + c]:
            parts.append(CARDS[ids[i + 1 + c]] + ("+" if floats[f + 1 + c] else ""))
    if ids[i + 1 + C]:
        parts.append(ENCHANTS[ids[i + 1 + C]])
    if ids[i + 2 + C]:
        parts.append(NAMES["relic"][ids[i + 2 + C]])
    if ids[i + 3 + C]:
        parts.append(POTIONS[ids[i + 3 + C]])
    if ids[i + 4 + C]:
        parts.append(NAMES["room"][ids[i + 4 + C]])
    if kind == "Event":
        parts[0] = f"Event#{ids[i + 5 + C]}"
    if price := floats[f + 2 + C]:
        parts.append(f"{price * 100:.0f}g")
    if paths and kind == "Path":
        m = floats[f + 3 + C : f + L.option_floats] * 8
        parts.append(f"elites {m[10]:.0f}-{m[11]:.0f} rests {m[6]:.0f}-{m[7]:.0f} ?{m[0]:.0f}-{m[1]:.0f}")
    return " ".join(parts)


def offered(L: RunLayout, floats: np.ndarray) -> list[int]:
    """The option tokens a run row offers."""
    return np.flatnonzero(floats[L.f_options : L.f_options + L.max_options * L.option_floats : L.option_floats]).tolist()


def problems(run: dict) -> list[str]:
    """What is wrong with a run's trace, by invariants that hold for every
    run: each fight has an outcome, a won fight leaves HP and the lost one
    none, only the last can be lost and a run that died lost it, floors
    never go back, a run that ended in a fight has that fight's deck size,
    every pick is an offered option, and every boss fight has its deck."""
    out = []
    fights = run["fights"]
    for f in fights:
        if "won" not in f:
            out.append(f"floor {f['floor']}: no outcome")
        elif (f["lost"] < f["hp"]) != f["won"]:
            out.append(f"floor {f['floor']}: {'won' if f['won'] else 'lost'} losing {f['lost']} of {f['hp']} HP")
    if any(not f.get("won", True) for f in fights[:-1]):
        out.append("a fight lost before the last")
    if run["end"] == "died" and (not fights or fights[-1].get("won", True)):
        out.append("died without losing its last fight")
    if run["end"] == "won" and (not fights or fights[-1].get("kind") != "Boss" or not fights[-1].get("won")):
        out.append("won without beating a boss last")
    if [f["floor"] for f in fights] != sorted(f["floor"] for f in fights):
        out.append("fight floors go back")
    if run["end"] in ("won", "died") and fights and fights[-1]["cards"] != run["deck"]:
        out.append(f"last fight's deck has {fights[-1]['cards']} cards, the run {run['deck']}")
    for floor, kind, pick, options in run["decisions"]:
        if not 0 <= pick < len(options):
            out.append(f"floor {floor} {kind}: pick {pick} of {len(options)} options")
    bosses = [f["floor"] for f in fights if f.get("kind") == "Boss"]
    decks = {d["floor"] for d in run["decks"]}
    if missing := [b for b in bosses if b not in decks]:
        out.append(f"no deck at the boss fights on floors {missing}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", type=Path, nargs="+")
    args = ap.parse_args()
    bad = 0
    for path in args.traces:
        runs = [json.loads(line) for line in path.read_text().splitlines()[1:] if line.strip()]
        for r in runs:
            for p in problems(r):
                bad += 1
                print(f"{path} seed {r['seed']}: {p}")
        fights = sum(len(r["fights"]) for r in runs)
        print(f"{path}: {len(runs)} runs, {fights} fights, {sum(len(r['decisions']) for r in runs)} decisions")
    if bad:
        raise SystemExit(f"{bad} problems")


if __name__ == "__main__":
    main()
