#!/usr/bin/env python3
"""Walk the encounters that have no clean recording yet.

For each one it builds a deck through the dev console, tells you what the
monsters do and what you have to do to make it show, waits for you to fight
it, then replays the recording against the sim and reports.

    uv run python scripts/record.py                 # everything missing
    uv run python scripts/record.py --list          # just the status table
    uv run python scripts/record.py --act underdocks
    uv run python scripts/record.py --only WATERFALL_GIANT_BOSS
    uv run python scripts/record.py --repeat 2      # each fight twice
    uv run python scripts/record.py --redo          # include clean ones

The encounter list, the acts, and the description of every fight all come
from the sim, so an act that gets ported shows up here with nothing written
by hand. Only `ADVICE` is hand-written, and only where playing a fight
straight would not show the mechanic.

Start the game first, load a run at the ascension you want (ascension is
fixed at run start), and leave it sitting anywhere outside combat. If the
run is carrying something the sim has never heard of, a colorless card or a
relic from an unported act, it says so and stops rather than sending you
into fights that would fail on the setup.

The deck is the point. A fast deck kills a boss in three turns and proves
nothing: the long move cycles are five and six turns. So the deck is mostly
block and barely any damage, and it is built from the plainest cards in the
pool, so a divergence points at the monster rather than at some card port.

Everything is set before `fight`. Powers or block handed out mid-combat by
the console are not in the recording's `start` record, so the sim never
sees them and the replay diverges on the next snapshot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = Path.home() / ".local/share/SlayTheSpire2/sts2ai"
RECORDINGS = GAME_DIR / "recordings"

# Block, a trickle of damage, and nothing with a fiddly port. Applied once
# per session: `card ... Deck` edits the run deck, so it accumulates.
DECK = [
    "remove_card ASCENDERS_BANE Deck",
    "remove_card STRIKE_IRONCLAD Deck",
    "remove_card STRIKE_IRONCLAD Deck",
    *["card SHRUG_IT_OFF Deck"] * 4,
    *["card BLOOD_WALL Deck"] * 2,
    *["card IMPERVIOUS Deck"] * 2,
    *["card IRON_WAVE Deck"] * 2,
    "card FEEL_NO_PAIN Deck",
    "relic add ANCHOR",
    "relic add BAG_OF_PREPARATION",
]

# Topped up before every fight. Fairy in a Bottle buys one death, which is
# the difference between seeing a boss cycle and seeing half of one.
PER_FIGHT = ["heal 999", "potion FAIRY_IN_A_BOTTLE", "potion BLOCK_POTION"]

# Fresnel Lens enchants every block card we just added with Nimble 2, which
# is free enchantment coverage. `--plain` drops it when a divergence needs
# isolating.
PLAIN = ["relic remove FRESNEL_LENS"]


def describe(enc: str, asc: int = 10) -> str:
    """What the sim says this encounter is: its monsters, the powers they
    start with, and the moves they can show. Generated, so a new act needs
    nothing written by hand."""
    from sts2ai import _sim

    pretty = lambda name: name.replace("_", " ").title()
    lines = []
    for monster, powers, moves in _sim.encounter_brief(enc, asc):
        kit = ", ".join(f"{pretty(name)} {n}" for name, n in powers)
        cycle = ", ".join(pretty(m.removesuffix("_MOVE")) for m in moves)
        lines.append(f"{pretty(monster)}{' [' + kit + ']' if kit else ''}: {cycle}")
    return "\n             ".join(lines)


# What you have to do to make a mechanic show, for the fights where playing
# it straight would not. Everything else is described from the sim.
ADVICE = {
    "CORPSE_SLUGS_NORMAL": "Kill one slug well before the others, so the survivors' Ravenous shows.",
    "GREMLIN_MERC_NORMAL": "Kill it. Two gremlins take its place, both idle their first turn, and the fat one flees.",
    "HAUNTED_SHIP_NORMAL": "Let Haunt land and keep the statuses rather than exhausting them.",
    "LIVING_FOG_NORMAL": "Play a skill on a Smoggy turn, and let a Gas Bomb reach its own Explode.",
    "PUNCH_CONSTRUCT_NORMAL": "Land a debuff early so Artifact eats it, then land a second one.",
    "SEAPUNK_NORMAL": "Leave the cultist alive several turns: its Ritual is the thing worth watching.",
    "SEWER_CLAM_NORMAL": "Hit it and watch Plating come back each turn.",
    "TWO_TAILED_RATS_NORMAL": "Do not kill any of them early. Let the board fill so the summon cap shows.",
    "PHANTASMAL_GARDENERS_ELITE": "Spread damage rather than focusing, so all four stay in the cycle.",
    "SKULKING_COLONY_ELITE": "Dump everything into one turn to hit the Hardened Shell cap, then again next turn to see it refill.",
    "TERROR_EEL_ELITE": "Push it under the Shriek threshold and keep playing past the interrupt.",
    "LAGAVULIN_MATRIARCH_BOSS": "Worth twice: once letting it wake on its timer, once hitting through the Plating to wake it early.",
    "SOUL_FYSH_BOSS": "Survive six turns to close the loop, and do not exhaust the Beckons.",
    "WATERFALL_GIANT_BOSS": "Drop it to zero and keep playing: About To Blow then Explode is what ends the fight.",
    "INKLETS_NORMAL": "Three inklets, the middle one opening on Whirlwind. Slippery caps what each hit takes off.",
}


def send(commands: list[str], quiet: bool = False) -> None:
    """Run dev console lines through the mod and show what it said."""
    if not commands:
        return
    out = subprocess.run([str(ROOT / "scripts/game.sh"), *commands], capture_output=True, text=True)
    if not quiet:
        for line in out.stdout.strip().splitlines():
            print(f"    {line}")


def encounters() -> list[tuple[str, str, str]]:
    from sts2ai import _sim

    return _sim.encounters()


def replay(path: Path) -> tuple[bool, str]:
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--bin", "replay", str(path)],
        cwd=ROOT / "sim",
        capture_output=True,
        text=True,
    )
    line = next((l for l in out.stdout.splitlines() if l.startswith(("ok", "DIFF", "ERR"))), out.stderr.strip())
    return line.startswith("ok"), line


def recordings_for(enc: str) -> list[Path]:
    return sorted(RECORDINGS.glob(f"*-{enc}.jsonl"))


def is_finished(path: Path) -> bool:
    """A recording is done once the recorder wrote its `end` record."""
    try:
        return any(json.loads(l).get("t") == "end" for l in path.read_text().splitlines() if l.strip())
    except (OSError, json.JSONDecodeError):
        return False


def status() -> dict[str, str]:
    """ok, DIFF, or missing, per encounter."""
    out = {}
    for enc, _, _ in encounters():
        files = recordings_for(enc)
        out[enc] = "missing" if not files else ("ok" if any(replay(f)[0] for f in files) else "DIFF")
    return out


def newest_start() -> str | None:
    """The `start` record of the most recent recording, which describes the
    run as it stands: deck, enchantments, relics, potions."""
    files = sorted(RECORDINGS.glob("*.jsonl"))
    for path in reversed(files):
        for line in path.read_text().splitlines():
            if '"t":"start"' in line or '"t": "start"' in line:
                return line
    return None


def blocker() -> str | None:
    """Something the run is carrying that the sim has never heard of, so every
    fight will fail on the setup rather than on the rules. A colorless card, a
    relic from an act that is not ported yet, a potion like Colorless Potion.
    The console can add things to a run but not take them away, so the answer
    is usually a different run rather than a fix."""
    from sts2ai import _sim

    start = newest_start()
    return _sim.start_blocker(start) if start else None


def wait_for_fight(enc: str, before: set[Path]) -> Path | None:
    """Block until a new finished recording for `enc` shows up."""
    print("    fight it. ctrl-c to skip.")
    try:
        while True:
            new = [p for p in recordings_for(enc) if p not in before]
            if new and is_finished(new[-1]):
                return new[-1]
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n    skipped")
        return None


def run_one(enc: str) -> tuple[bool, str]:
    print(f"\n=== {enc}")
    print(f"    monster: {describe(enc)}")
    if enc in ADVICE:
        print(f"    you:     {ADVICE[enc]}")
    before = set(recordings_for(enc))
    send(PER_FIGHT, quiet=True)
    send([f"fight {enc}"])
    path = wait_for_fight(enc, before)
    if path is None:
        return False, "skipped"
    ok, line = replay(path)
    print(f"    {line}")
    return ok, line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the status table and stop")
    # The acts and tiers the sim carries, so a ported act needs no edit here.
    acts = sorted({act.lower() for _, act, _ in encounters()})
    kinds = sorted({kind.lower() for _, _, kind in encounters()})
    ap.add_argument("--act", choices=acts, help="only this act")
    ap.add_argument("--kind", choices=kinds, help="only this tier")
    ap.add_argument("--only", action="append", default=[], metavar="ENCOUNTER")
    ap.add_argument("--redo", action="store_true", help="include encounters that already replay clean")
    ap.add_argument("--no-deck", action="store_true", help="leave the run's deck alone")
    ap.add_argument("--plain", action="store_true", help="drop Fresnel Lens, so no card arrives enchanted")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="record each encounter N times, all of which must be clean")
    args = ap.parse_args()

    if not RECORDINGS.is_dir():
        sys.exit(f"no recordings folder at {RECORDINGS}; build and load the mod first")

    encs = encounters()
    if args.act:
        encs = [e for e in encs if e[1].lower() == args.act]
    if args.kind:
        encs = [e for e in encs if e[2].lower() == args.kind]
    if args.only:
        wanted = {o.upper() for o in args.only}
        encs = [e for e in encs if e[0] in wanted]

    print("checking what is already recorded...")
    state = status()
    for enc, act, kind in encs:
        print(f"  {state[enc]:>7}  {enc:<32} {act} {kind}")
    if args.list:
        return

    if (why := blocker()) is not None:
        sys.exit(
            f"\nthe run is carrying something the sim does not know: {why}\n"
            "every fight would fail on the setup, not on the rules. the dev console cannot\n"
            "remove potions or cards, so this needs a run without it."
        )

    todo = [e[0] for e in encs if args.redo or state[e[0]] != "ok"]
    if not todo:
        print("\nnothing to record.")
        return
    print(f"\n{len(todo)} to go. start the game, open a run, and stay out of combat.")
    input("enter when ready: ")

    if args.plain:
        send(PLAIN)
    if not args.no_deck:
        print("building the deck...")
        send(DECK, quiet=True)

    done, failed = [], []
    for enc in todo:
        runs = 0
        while runs < args.repeat:
            if args.repeat > 1:
                print(f"    run {runs + 1} of {args.repeat}")
            ok, line = run_one(enc)
            if ok:
                runs += 1
                continue
            # A setup the sim cannot build is not this fight's fault, and
            # every later fight would hit it too.
            if line.startswith("ERR") and (why := blocker()) is not None:
                print(f"\nstopping: the run is carrying {why}")
                failed.append(enc)
                break
            choice = input("    [enter] next, r retry, q quit: ").strip().lower()
            if choice == "r":
                continue
            failed.append(enc)
            if choice == "q":
                print(f"\nclean: {len(done)}  left: {len(todo) - len(done) - len(failed)}")
                return
            break
        else:
            done.append(enc)

    print(f"\nclean: {len(done)}")
    if failed:
        print("not clean: " + ", ".join(failed))


if __name__ == "__main__":
    main()
