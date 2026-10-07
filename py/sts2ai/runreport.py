"""What happened in the runs of `runplay --runs-out` traces
(`sts2ai.runtrace`), each file a column, B, C... paired against A by seed.

    uv run python -m sts2ai.runreport runs/ab/NAME/a.jsonl runs/ab/NAME/b.jsonl

Sections: outcomes (with `sts2ai.paired`'s intervals), fights by act and
kind, each boss, act 3's two bosses (with the conversion on the seeds both
arms reach), the decision mix, and the decks at the act 1 boss and the
first act 3 boss. Runs that got stuck are left out.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sts2ai import _sim, paired
from sts2ai.runtrace import KINDS

LETTERS = "ABCDEFGH"
TYPES = dict(zip(_sim.game_ids()["card"], _sim.card_types()))
# The Ironclad's starting deck at A10, left out of the most common cards.
STARTER = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE"}


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def row(label: str, cells: list[str], width: int = 22) -> None:
    print(f"  {label:{width}s}" + "".join(f"{c:>12s}" for c in cells))


def pct(x: float) -> str:
    return "-" if np.isnan(x) else f"{x:.1%}"


def num(x: float, digits: int = 1) -> str:
    return "-" if np.isnan(x) else f"{x:.{digits}f}"


def outcomes(arms: list[dict[int, dict]], paths: list[Path]) -> None:
    print("outcomes:")
    row("", [f"{LETTERS[k]}" for k in range(len(arms))])
    row("runs", [str(len(a)) for a in arms])
    for k, name in enumerate(paired.NAMES):
        values = [mean([paired.metrics(r)[k] for r in a.values()]) for a in arms]
        row(name, [pct(v) if k < 3 else num(v) for v in values])
    for k in range(1, len(arms)):
        print(f"\n{LETTERS[k]} against A, paired:")
        paired.compare(paths[0], paths[k])


def fights(arms: list[dict[int, dict]]) -> None:
    print("\nfights by act and kind: fights, won, HP going in, HP lost in won fights (net of end-of-fight heals), potions used per fight")
    for k in range(len(arms)):
        row(LETTERS[k], ["fights", "won", "HP in", "HP lost", "potions"])
        table: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for r in arms[k].values():
            for f in r["fights"]:
                if "won" in f:
                    table[(f["act"], f["kind"])].append(f)
        for act in sorted({a for a, _ in table}):
            for kind in ("Weak", "Normal", "Elite", "Boss"):
                if fs := table.get((act, kind)):
                    won = [f for f in fs if f["won"]]
                    cells = [
                        str(len(fs)),
                        pct(len(won) / len(fs)),
                        num(mean([f["hp"] for f in fs])),
                        num(mean([f["lost"] for f in won])),
                        num(mean([f["used"] for f in fs]), 2),
                    ]
                    row(f"act {act + 1} {kind.lower()}", cells)


def bosses(arms: list[dict[int, dict]]) -> None:
    print("\nbosses: fights, won, HP going in, per arm")
    by: list[dict[tuple[int, str], list[dict]]] = []
    for a in arms:
        t: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for r in a.values():
            for f in r["fights"]:
                if f["kind"] == "Boss" and "won" in f:
                    t[(f["act"], f["enc"])].append(f)
        by.append(t)
    row("", [f"{LETTERS[k]} {c}" for k in range(len(arms)) for c in ("n", "won", "HP in")], 34)
    for act, enc in sorted({key for t in by for key in t}):
        cells = []
        for t in by:
            fs = t.get((act, enc), [])
            cells += [str(len(fs)), pct(mean([f["won"] for f in fs])), num(mean([f["hp"] for f in fs]))]
        row(f"act {act + 1} {enc}", cells, 34)


def act3(r: dict) -> tuple[bool, bool, bool]:
    """Reached act 3's first boss, beat it, beat the second."""
    b = [f for f in r["fights"] if f["act"] == 2 and f["kind"] == "Boss" and "won" in f]
    return bool(b), bool(b) and b[0]["won"], len(b) > 1 and b[1]["won"]


def act3_bosses(arms: list[dict[int, dict]]) -> None:
    print("\nact 3 bosses:")
    row("", [LETTERS[k] for k in range(len(arms))])
    for i, name in enumerate(("reached the first", "beat the first", "beat the second")):
        row(name, [str(sum(act3(r)[i] for r in a.values())) for a in arms])
    for k in range(1, len(arms)):
        seeds = sorted(s for s in arms[0].keys() & arms[k].keys() if act3(arms[0][s])[0] and act3(arms[k][s])[0])
        if not seeds:
            continue
        x = np.array([act3(arms[0][s])[1:] for s in seeds], dtype=float)
        y = np.array([act3(arms[k][s])[1:] for s in seeds], dtype=float)
        lo, hi = paired.interval(y - x)
        print(f"  on the {len(seeds)} seeds both A and {LETTERS[k]} reach, {LETTERS[k]} minus A:")
        for i, name in enumerate(("beat the first", "beat both")):
            d = (y - x)[:, i].mean()
            noise = "" if lo[i] > 0 or hi[i] < 0 else "  within noise"
            print(f"    {name:15s} A {x[:, i].mean():6.1%}  {LETTERS[k]} {y[:, i].mean():6.1%}  {d:+6.1%}  [{lo[i]:+.1%}, {hi[i]:+.1%}]{noise}")


def picked(d: list) -> str:
    """A decision's pick by option kind; a map step by its room."""
    _, kind, pick, options = d
    text = options[pick]
    return " ".join(text.split()[:2]) if kind == "Path" else text.split()[0].split("#")[0]


def decisions(arms: list[dict[int, dict]]) -> None:
    print("\ndecision mix: share of each kind of pick, per decision kind")
    mixes: list[dict[str, Counter[str]]] = []
    for a in arms:
        m: dict[str, Counter[str]] = defaultdict(Counter)
        for r in a.values():
            for d in r["decisions"]:
                m[d[1]][picked(d)] += 1
        mixes.append(m)
    if not any(mixes):
        print("  no decisions traced (the sim made them)")
        return
    row("", [LETTERS[k] for k in range(len(arms))], 30)
    kinds = Counter[str]()
    for m in mixes:
        kinds.update({k: c.total() for k, c in m.items()})
    for kind, _ in kinds.most_common():
        row(kind, [str(m[kind].total()) for m in mixes], 30)
        options = Counter[str]()
        for m in mixes:
            options.update(m[kind])
        for option, _ in options.most_common():
            row("  " + option, [pct(m[kind][option] / max(m[kind].total(), 1)) for m in mixes], 30)
    elite = [m["Path"]["Path Elite"] / max(m["Path"].total(), 1) for m in mixes]
    heal = [m["Rest"]["RestHeal"] / max(m["Rest"]["RestHeal"] + m["Rest"]["RestSmith"], 1) for m in mixes]
    row("map steps into an elite", [pct(v) for v in elite], 30)
    row("rests healed (of heal, smith)", [pct(v) for v in heal], 30)


def base(card: str) -> str:
    return card.split("~")[0].rstrip("+")


def deck_at(r: dict, act: int) -> dict | None:
    """The deck at the run's first boss fight of `act` (0-based)."""
    return next((d for d in r["decks"] if d["act"] == act and KINDS[d["enc"]] == "Boss"), None)


def removals(r: dict, floor: int) -> int:
    return sum(1 for d in r["decisions"] if d[0] < floor and picked(d) == "DeckRemove")


def decks(arms: list[dict[int, dict]], act: int, name: str) -> None:
    held = [[(r, d) for r in a.values() if (d := deck_at(r, act)) is not None] for a in arms]
    print(f"\ndecks at {name}:")
    row("", [LETTERS[k] for k in range(len(arms))])
    row("decks", [str(len(h)) for h in held])
    stat: dict[str, Callable[[dict, dict], float]] = {
        "cards": lambda r, d: len(d["cards"]),
        "upgraded": lambda r, d: mean([c.split("~")[0].endswith("+") for c in d["cards"]]),
        "attacks": lambda r, d: mean([TYPES[base(c)] == "Attack" for c in d["cards"]]),
        "skills": lambda r, d: mean([TYPES[base(c)] == "Skill" for c in d["cards"]]),
        "powers": lambda r, d: mean([TYPES[base(c)] == "Power" for c in d["cards"]]),
        "relics": lambda r, d: len(d["relics"]),
        "removed so far": lambda r, d: removals(r, d["floor"]),
        "HP going in": lambda r, d: d["hp"],
        "max HP": lambda r, d: d["max_hp"],
        "gold": lambda r, d: d["gold"],
    }
    for label, f in stat.items():
        values = [mean([f(r, d) for r, d in h]) for h in held]
        share = label in ("upgraded", "attacks", "skills", "powers")
        row(label, [pct(v) if share else num(v) for v in values])
    holding = [Counter(c for _, d in h for c in {base(c) for c in d["cards"]} - STARTER) for h in held]
    shares = [{c: n / max(len(h), 1) for c, n in counts.items()} for counts, h in zip(holding, held)]
    top = sorted({c for s in shares for c in s}, key=lambda c: -max(s.get(c, 0) for s in shares))[:15]
    print("  most held cards, share of decks (starter cards left out):")
    row("", [LETTERS[k] for k in range(len(arms))] + [f"{LETTERS[k]} - A" for k in range(1, len(arms))])
    for c in top:
        row(c, [pct(s.get(c, 0)) for s in shares] + [f"{s.get(c, 0) - shares[0].get(c, 0):+.1%}" for s in shares[1:]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", type=Path, nargs="+", help="runplay --runs-out files, A first")
    args = ap.parse_args()
    arms = []
    for k, path in enumerate(args.traces):
        argv, runs = paired.load(path)
        stuck = [s for s, r in runs.items() if r["end"].startswith("stuck")]
        print(f"{LETTERS[k]}: {path} ({len(stuck)} stuck runs left out)\n   {' '.join(argv[1:])}")
        arms.append({s: r for s, r in runs.items() if s not in stuck})
    if any("fights" not in r for a in arms for r in a.values()):
        raise SystemExit("a file without traces: it predates runplay's run traces")
    print()
    outcomes(arms, args.traces)
    fights(arms)
    bosses(arms)
    act3_bosses(arms)
    decisions(arms)
    decks(arms, 0, "the act 1 boss")
    decks(arms, 2, "the first act 3 boss")


if __name__ == "__main__":
    main()
