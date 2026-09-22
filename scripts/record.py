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
    uv run python scripts/record.py --relics        # the relic fights instead
    uv run python scripts/record.py --relics --only paels_eye

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

`--relics` walks `RELIC_FIGHTS` instead: each group of relics goes on for
one fight and comes off again with `relic remove`, so every group starts
from the same run. Removing a relic does not undo what it did on pickup,
so a group whose pickup leaves a card behind takes it out again.

The recorder writes every fight into one folder. Fights this script drives
are moved into `recordings/dev/` once they are done, so the top folder
holds only fights from real runs, which is what evaluate.py and the deck
statistics read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = Path.home() / ".local/share/SlayTheSpire2/sts2ai"
RECORDINGS = GAME_DIR / "recordings"
DEV = RECORDINGS / "dev"

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


@dataclass
class RelicFight:
    """Relics that go on together for one fight, and what to do to make
    them show. Grouped by the hook they touch, so a divergence still points
    at a handful; the ones with their own turn flow fight alone."""

    name: str
    relics: list[str]
    advice: str
    encounter: str = "SEWER_CLAM_NORMAL"
    # Console lines before the fight and after it, beyond adding and
    # removing the relics themselves.
    setup: list[str] = field(default_factory=list)
    teardown: list[str] = field(default_factory=list)


# Sewer Clam by default: one monster, Plating to break, and a fight the
# block deck drags out long enough for the per-turn relics to cycle.
# Not here: Fur Coat, which needs a room its own map marked, and Delicate
# Frond, whose potions land before the start record, so the replay
# cannot see it work.
RELIC_FIGHTS = [
    RelicFight("paels_eye", ["PAELS_EYE"],
               "End turn 1 without playing anything: the hand exhausts, the clam does not move, and you go again."),
    RelicFight("whispering_earring", ["WHISPERING_EARRING"],
               "Nothing to do on turn 1: it plays your hand for you, leftmost card first, at the clam."),
    RelicFight("snecko_eye", ["SNECKO_EYE", "FAKE_SNECKO_EYE"],
               "Play normally. Every card you draw rolls a new cost for the fight."),
    RelicFight("choices_paradox", ["CHOICES_PARADOX"],
               "Turn 1 offers five cards. Take one and hold it past the end of the turn to see it retained."),
    RelicFight("history_course", ["HISTORY_COURSE"],
               "End a few turns on different attacks and skills: each turn opens by replaying the last one."),
    RelicFight("first_plays", ["THROWING_AXE", "MUSIC_BOX"],
               "Open the fight with an attack (Throwing Axe plays it twice), and play an attack every turn "
               "(Music Box hands back an ethereal copy of the first)."),
    RelicFight("play_limits", ["BRILLIANT_SCARF", "VELVET_CHOKER"],
               "Play five cards in one turn, the Angers help: the fifth is free. Then try for a seventh, "
               "which Velvet Choker refuses.",
               setup=["card ANGER Deck"] * 3, teardown=["remove_card ANGER Deck"] * 3),
    RelicFight("draw_and_hand", ["FIDDLE", "RUNIC_PYRAMID"],
               "Play Shrug It Off: Fiddle eats its draw. Leave cards in hand; they stay for next turn."),
    RelicFight("turn_one", ["BONE_TEA", "BLESSED_ANTLER", "RADIANT_PEARL", "JEWELED_MASK", "BIG_MUSHROOM",
                            "ROYAL_POISON", "FAKE_BLOOD_VIAL", "TEA_OF_DISCOURTESY", "EMBER_TEA", "SWORD_OF_JADE",
                            "FAKE_ANCHOR"],
               "All of it lands on turn 1. Play the Luminesce at some point."),
    RelicFight("every_turn", ["CROSSBOW", "SAI", "MR_STRUGGLES", "FAKE_HAPPY_FLOWER", "POLLINOUS_CORE",
                              "TOASTY_MITTENS", "PAELS_BLOOD", "IRON_CLUB", "SEAL_OF_GOLD"],
               "Go at least six turns, so the five-turn flower and the four-turn core both fire.",
               setup=["gold 100"]),
    RelicFight("energy", ["PRISMATIC_GEM", "ECTOPLASM", "SOZU", "BLOOD_SOAKED_ROSE", "PHILOSOPHERS_STONE",
                          "PUMPKIN_CANDLE", "SPIKED_GAUNTLETS", "PAELS_TEARS"],
               "End a turn with energy left (Pael's Tears pays it back), and play Feel No Pain at its "
               "raised cost. Sozu refuses the fight's potions, so no Fairy this time.",
               teardown=["remove_card ENTHRALLED Deck"]),
    RelicFight("card_hooks", ["DAUGHTER_OF_THE_WIND", "LOST_WISP", "FORGOTTEN_SOUL", "HAND_DRILL",
                              "FAKE_STRIKE_DUMMY", "PAELS_LEGION", "DIAMOND_DIADEM", "FAKE_ORICHALCUM"],
               "Break the clam's block with an attack (Hand Drill), play Feel No Pain (Lost Wisp), exhaust "
               "something (Forgotten Soul), and have one turn of two cards or fewer (Diamond Diadem)."),
    RelicFight("biiig_hug", ["BIIIG_HUG"],
               "On pickup it asks for four cards to remove: take Defends. Every reshuffle adds a Soot."),
    RelicFight("elite", ["BOOMING_CONCH", "BLACK_BLOOD"],
               "Elites only for the conch. Win it: Black Blood heals 12 afterwards.",
               encounter="TERROR_EEL_ELITE"),
]


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
    return sorted([*DEV.glob(f"*-{enc}.jsonl"), *RECORDINGS.glob(f"*-{enc}.jsonl")])


def park(path: Path) -> Path:
    """Move a finished dev-console fight out of the run recordings."""
    DEV.mkdir(exist_ok=True)
    return path.replace(DEV / path.name)


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
    files = sorted([*RECORDINGS.glob("*.jsonl"), *DEV.glob("*.jsonl")], key=lambda p: p.name)
    for path in reversed(files):
        for line in path.read_text().splitlines():
            if '"t":"start"' in line or '"t": "start"' in line:
                return line
    return None


def blocker() -> str | None:
    """Something the run is carrying that the sim has never heard of, so every
    fight will fail on the setup rather than on the rules. A colorless card, a
    relic from an act that is not ported yet, a potion like Colorless Potion.
    Cards and relics come off with `remove_card` and `relic remove`; a potion
    has no remove command, so it has to be used up."""
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


def run_one(enc: str, fight: RelicFight | None = None) -> tuple[bool, str]:
    """One recorded fight against `enc`, with a relic group on for it if
    given. The group comes off again whether or not the fight finished."""
    print(f"\n=== {fight.name if fight else enc}")
    print(f"    monster: {describe(enc)}")
    advice = fight.advice if fight else ADVICE.get(enc)
    if fight:
        print(f"    relics:  {', '.join(fight.relics)}")
    if advice:
        print(f"    you:     {advice}")
    before = set(recordings_for(enc))
    if fight:
        send([f"relic add {r}" for r in fight.relics] + fight.setup)
    send(PER_FIGHT, quiet=True)
    send([f"fight {enc}"])
    path = wait_for_fight(enc, before)
    if fight:
        send([f"relic remove {r}" for r in fight.relics] + fight.teardown, quiet=True)
    if path is None:
        return False, "skipped"
    # A run reloaded after an autosave gets back relics a teardown removed.
    held = set(start_record(path).get("relics", []))
    strays = sorted(held & {r for f in RELIC_FIGHTS for r in f.relics} - set(fight.relics if fight else []))
    if strays:
        print(f"    warning: also carried {', '.join(strays)}; take them off with `relic remove`")
    ok, line = replay(park(path))
    print(f"    {line}")
    return ok, line


def start_record(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            if v.get("t") == "start":
                return v
    return {}


def relic_recordings(fight: RelicFight) -> list[Path]:
    """Dev fights against the group's encounter with every relic of the
    group on."""
    want = set(fight.relics)
    return [p for p in sorted(DEV.glob(f"*-{fight.encounter}.jsonl")) if want <= set(start_record(p).get("relics", []))]


def relic_status() -> dict[str, str]:
    out = {}
    for fight in RELIC_FIGHTS:
        files = relic_recordings(fight)
        out[fight.name] = "missing" if not files else ("ok" if any(replay(f)[0] for f in files) else "DIFF")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the status table and stop")
    # The acts and tiers the sim carries, so a ported act needs no edit here.
    acts = sorted({act.lower() for _, act, _ in encounters()})
    kinds = sorted({kind.lower() for _, _, kind in encounters()})
    ap.add_argument("--act", choices=acts, help="only this act")
    ap.add_argument("--kind", choices=kinds, help="only this tier")
    ap.add_argument("--only", action="append", default=[], metavar="NAME", help="an encounter, or a relic fight with --relics")
    ap.add_argument("--redo", action="store_true", help="include encounters that already replay clean")
    ap.add_argument("--no-deck", action="store_true", help="leave the run's deck alone")
    ap.add_argument("--plain", action="store_true", help="drop Fresnel Lens, so no card arrives enchanted")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="record each encounter N times, all of which must be clean")
    ap.add_argument("--relics", action="store_true", help="walk the relic fights instead of the encounters")
    args = ap.parse_args()

    if not RECORDINGS.is_dir():
        sys.exit(f"no recordings folder at {RECORDINGS}; build and load the mod first")

    if args.relics:
        fights = RELIC_FIGHTS
        if args.only:
            wanted = {o.lower() for o in args.only}
            fights = [f for f in fights if f.name in wanted]
        print("checking what is already recorded...")
        state = relic_status()
        for f in fights:
            print(f"  {state[f.name]:>7}  {f.name:<20} {f.encounter}")
        if args.list:
            return
        walk([(f.encounter, f) for f in fights if args.redo or state[f.name] != "ok"], args)
        return

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
    walk([(e[0], None) for e in encs if args.redo or state[e[0]] != "ok"], args)


def walk(todo: list[tuple[str, RelicFight | None]], args: argparse.Namespace) -> None:
    """Set up and record each fight in turn, replaying as they finish."""
    if (why := blocker()) is not None:
        sys.exit(
            f"\nthe run is carrying something the sim does not know: {why}\n"
            "every fight would fail on the setup, not on the rules. a card or relic can come\n"
            "off with `remove_card` or `relic remove`; a potion needs using or a new run."
        )
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
    for enc, fight in todo:
        label = fight.name if fight else enc
        runs = 0
        while runs < args.repeat:
            if args.repeat > 1:
                print(f"    run {runs + 1} of {args.repeat}")
            ok, line = run_one(enc, fight)
            if ok:
                runs += 1
                continue
            # A setup the sim cannot build is not this fight's fault, and
            # every later fight would hit it too.
            if line.startswith("ERR") and (why := blocker()) is not None:
                print(f"\nstopping: the run is carrying {why}")
                failed.append(label)
                break
            choice = input("    [enter] next, r retry, q quit: ").strip().lower()
            if choice == "r":
                continue
            failed.append(label)
            if choice == "q":
                print(f"\nclean: {len(done)}  left: {len(todo) - len(done) - len(failed)}")
                return
            break
        else:
            done.append(label)

    print(f"\nclean: {len(done)}")
    if failed:
        print("not clean: " + ", ".join(failed))


if __name__ == "__main__":
    main()
