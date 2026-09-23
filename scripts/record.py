#!/usr/bin/env python3
"""Record the fights the sim has no clean replay of yet.

Everything to record is a job: console setup, one or more fights, and the
console lines that undo the setup. It builds each job's setup through the
dev console, says what the monsters do and what to do to make a mechanic
show, waits for the fight, then replays the recording against the sim.

    uv run python scripts/record.py                    # every job not yet clean
    uv run python scripts/record.py --list             # just the status table
    uv run python scripts/record.py hive elite         # jobs tagged both
    uv run python scripts/record.py relics             # the relic groups
    uv run python scripts/record.py waterfall_giant_boss
    uv run python scripts/record.py --redo --repeat 2  # clean ones too, twice each
    uv run python scripts/record.py --pilot runs/set-3/latest.pt --queue

Terms pick jobs by name or tag; a job needs to match every term. Encounter
jobs are named after the encounter and tagged with its act and kind; relic
groups are tagged `relics`. The encounters, acts and monster descriptions
all come from the sim, so a newly ported act shows up with nothing written
by hand. Only `ADVICE` and `RELIC_FIGHTS` are hand-written.

`--pilot CKPT` lets the policy play the fights through the bridge
(`sts2ai.play --record`): it samples its moves, and ends a fight with `win`
once the sim loses track or HP gets low, so a night of fights runs
unattended. Jobs marked `human` need your hands (a first turn left idle, a
pickup screen); the pilot steps aside for their fights. `--queue` keeps
running after the selected jobs and takes more from
`sts2ai/queue.jsonl`, one JSON object per line, appended while it runs:

    {"run": "kaiser_crab_boss"}
    {"job": {"name": "tf", "relics": ["TUNING_FORK"], "fights": ["SEWER_CLAM_NORMAL"]}}

Every fight's replay result goes to `sts2ai/results.jsonl`.

Start the game first, load a run at the ascension you want (ascension is
fixed at run start), and leave it sitting anywhere outside combat. If the
run is carrying something the sim cannot play, a card in `UNSUPPORTED_CARDS`
or a relic it does not know, it says so and stops rather than sending you
into fights that would fail on the setup.

The deck is the point. A fast deck kills a boss in three turns and proves
nothing: the long move cycles are five and six turns. So the deck is mostly
block and barely any damage, and it is built from the plainest cards in the
pool, so a divergence points at the monster rather than at some card port.
Each job has a loadout (`STANDARD`, or `POWERED` for acts 2 and 3, which
need damage as well as block): the exact deck, setup relics and a floor on
max HP. Before a
job, only the difference from what the run carries goes to the console, so
sessions no longer pile cards and relics on top of each other.

Everything is set before `fight`. Powers or block handed out mid-combat by
the console are not in the recording's `start` record, so the sim never
sees them and the replay diverges on the next snapshot. Relics go on before
a job's first fight and come off after its last; removing a relic does not
undo its pickup, so a job whose pickup leaves a card behind takes it out.

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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import game  # scripts/game.py: the game window, for what the console cannot do

ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = Path.home() / ".local/share/SlayTheSpire2/sts2ai"
RECORDINGS = GAME_DIR / "recordings"
DEV = RECORDINGS / "dev"

@dataclass
class Loadout:
    """What the run should carry into a job's fights: the exact deck, the
    setup relics, and at least this much max HP."""

    deck: dict[str, int]
    relics: dict[str, int]
    max_hp: int


# Block, a trickle of damage, and nothing with a fiddly port.
STANDARD = Loadout(
    {"STRIKE_IRONCLAD": 3, "DEFEND_IRONCLAD": 4, "BASH": 1, "SHRUG_IT_OFF": 4, "BLOOD_WALL": 2,
     "IMPERVIOUS": 2, "IRON_WAVE": 2, "FEEL_NO_PAIN": 1},
    {"ANCHOR": 1, "BAG_OF_PREPARATION": 1},
    80,
)
# For act 2 and 3, where block alone loses: a board of four bots out-hits
# any pile of it. Area damage for crowds (Thunderclap, Breakthrough),
# Strength that scales (Inflame, Demon Form), plain hits, and still enough
# block. Bosses have hundreds of HP, so their cycles show all the same.
POWERED = Loadout(
    {"STRIKE_IRONCLAD": 2, "DEFEND_IRONCLAD": 3, "BASH": 1, "SHRUG_IT_OFF": 3, "IMPERVIOUS": 2,
     "FLAME_BARRIER": 1, "FEEL_NO_PAIN": 1, "INFLAME": 2, "DEMON_FORM": 1, "THUNDERCLAP": 2,
     "BREAKTHROUGH": 1, "POMMEL_STRIKE": 2, "UPPERCUT": 1, "IRON_WAVE": 2},
    STANDARD.relics,
    150,
)
# Relics that raise max HP on pickup, and keep it when they come off again.
MAX_HP_RELICS = [("MANGO", 14), ("PEAR", 10), ("STRAWBERRY", 7)]


# Card jobs: a lean deck so the cards under test come up every few turns,
# and the HP to keep a long fight going.
CARD_BASE = {"STRIKE_IRONCLAD": 3, "DEFEND_IRONCLAD": 3, "BASH": 1, "SHRUG_IT_OFF": 2}


def loadout_for(job: "Job") -> Loadout:
    if job.cards:
        return Loadout({**CARD_BASE, **{c: 2 for c in job.cards}}, STANDARD.relics, 150)
    return POWERED if job.tags & {"hive", "glory"} else STANDARD


class Run:
    """Brings the run to a loadout, reading what it carries from the mod's
    `run.json` each time, so `reach` only sends the difference."""

    def reach(self, target: Loadout) -> None:
        state = run_state()
        self.deck = Counter(c["id"] for c in state.get("deck", []))
        self.relics = Counter(state.get("relics", []))
        self.max_hp = state.get("max_hp", 80)
        commands = []
        for card in sorted(set(self.deck) | set(target.deck)):
            diff = target.deck.get(card, 0) - self.deck[card]
            commands += [f"card {card} Deck"] * diff + [f"remove_card {card} Deck"] * -diff
            self.deck[card] += diff
        for relic, want in target.relics.items():
            diff = want - self.relics[relic]
            commands += [f"relic add {relic}"] * diff + [f"relic remove {relic}"] * -diff
            self.relics[relic] += diff
        for relic, gain in MAX_HP_RELICS:
            while self.max_hp + gain <= target.max_hp or (self.max_hp < target.max_hp and relic == "STRAWBERRY"):
                commands += [f"relic add {relic}", f"relic remove {relic}"]
                self.max_hp += gain
        if commands:
            print(f"    loadout: {len(commands)} console lines")
            send(commands, quiet=True)


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
    "BOWLBUGS_WEAK": "Fully block a Bowlbug Rock headbutt once, so it staggers the turn after.",
    "BOWLBUGS_NORMAL": "Fully block a Bowlbug Rock headbutt once, so it staggers the turn after.",
    "EXOSKELETONS_WEAK": "Hit one with a single big attack: Hard to Kill caps each hit at 9.",
    "HUNTER_KILLER_NORMAL": "Play several cards on a turn under Tender, and watch Strength and Dexterity come back at the end.",
    "LOUSE_PROGENITOR_NORMAL": "Hit it with a multi-hit attack card: Curl Up blocks only once the whole card is done.",
    "MYTES_NORMAL": "End a turn with a Toxic still in hand once.",
    "OVICOPTER_NORMAL": "Let an egg hatch, then kill eggs so it lays again rather than feeding.",
    "SLUMBERING_BEETLE_NORMAL": "Hit the beetle through its Plating to wake it before Slumber runs out.",
    "THE_OBSCURA_NORMAL": "Kill the Parafright once so it revives, and let Sail buff both.",
    "THIEVING_HOPPER_WEAK": "Hit it hard while it flutters so it gets knocked down; it flees after Nab otherwise.",
    "TUNNELER_WEAK": "Break its block after it burrows, so it is stunned back to Bite.",
    "DECIMILLIPEDE_ELITE": "Kill one segment early and keep the others alive two turns, so it reattaches.",
    "ENTOMANCER_ELITE": "Attack it several times a turn: every hit shuffles Dazed into your draw pile.",
    "INFESTED_PRISMS_ELITE": "Play skills: each one gives you Tainted until the enemy turn ends.",
    "KAISER_CRAB_BOSS": "Target one half, then the other, so Surrounded turns; kill one half to see Crab Rage.",
    "KNOWLEDGE_DEMON_BOSS": "Survive all three Curses of Knowledge, taking a different curse each time.",
    "THE_INSATIABLE_BOSS": "Play Frantic Escape to push the Sandpit back, and let it run down a turn or two.",
    "AXEBOTS_NORMAL": "Kill it twice: each death sends in a fresh Axebot on Boot Up. The third one stays dead.",
    "FABRICATOR_NORMAL": "Leave the bots alive so the board fills up and it switches to Disintegrate.",
    "FROG_KNIGHT_NORMAL": "Get it below half HP so its one Beetle Charge shows.",
    "GLOBE_HEAD_NORMAL": "Carry a power card and play it: Galvanized cards hurt you when played.",
    "THE_LOST_AND_FORGOTTEN_NORMAL": "Let both steal once, then kill one while the other lives so its stat comes back.",
    "TURRET_OPERATOR_WEAK": "Kill the turret first so the Living Shield switches to Smash.",
    "KNIGHTS_ELITE": "Carry an upgraded card for Dampen. Kill the Spectral Knight to lift Hex, the Magi Knight to lift Dampen.",
    "AEONGLASS_BOSS": "Hold Withers through two Increasing Intensity turns, and play six cards in a turn for Withering Presence.",
    "QUEEN_BOSS": "Play a Bound card, then kill the Amalgam while the Queen shows Burn Bright for Me.",
    "TEST_SUBJECT_BOSS": "Kill it three times: it respawns with Painful Stabs, then with Nemesis.",
}


@dataclass
class Send:
    """Console lines."""

    commands: list[str]


@dataclass
class Say:
    """Something for you to do; `wait` holds the job until you press enter."""

    text: str
    wait: bool = False


@dataclass
class Fight:
    """Start an encounter, wait for its recording, replay it."""

    encounter: str


Step = Send | Say | Fight


@dataclass
class Job:
    """One thing to record. `relics` go on before the first step and come
    off after the last, with `teardown`, whether or not the fights ran."""

    name: str
    tags: set[str]
    steps: list[Step]
    relics: list[str] = field(default_factory=list)
    teardown: list[str] = field(default_factory=list)
    advice: str | None = None
    # Its fights need a person, not the pilot.
    human: bool = False
    # Cards under test (`cards` jobs): the deck carries two of each, and the
    # job is done once each has been played in a clean recording.
    cards: list[str] = field(default_factory=list)

    def fights(self) -> list[str]:
        return [s.encounter for s in self.steps if isinstance(s, Fight)]

    def matches(self, terms: list[str]) -> bool:
        return all(t == self.name or t in self.tags for t in terms)


def relic_job(name: str, relics: list[str], advice: str, encounter: str = "SEWER_CLAM_NORMAL", *,
              setup: list[str] = [], teardown: list[str] = [], human: bool = False) -> Job:
    """A relic group: on for one fight, off again after."""
    return Job(name, {"relics"}, [Send(setup), Fight(encounter)], relics, teardown, advice, human)


# Sewer Clam by default: one monster, Plating to break, and a fight the
# block deck drags out long enough for the per-turn relics to cycle.
# Not here: Fur Coat, which needs a room its own map marked, Delicate Frond,
# whose potions land before the start record so the replay cannot see it
# work, and Lizard Tail, which needs a death.
RELIC_FIGHTS = [
    relic_job("paels_eye", ["PAELS_EYE"],
              "End turn 1 without playing anything: the hand exhausts, the clam does not move, and you go again.",
              human=True),
    relic_job("whispering_earring", ["WHISPERING_EARRING"],
              "Nothing to do on turn 1: it plays your hand for you, leftmost card first, at the clam."),
    relic_job("snecko_eye", ["SNECKO_EYE", "FAKE_SNECKO_EYE"],
              "Play normally. Every card you draw rolls a new cost for the fight."),
    relic_job("choices_paradox", ["CHOICES_PARADOX"],
              "Turn 1 offers five cards. Take one and hold it past the end of the turn to see it retained."),
    relic_job("history_course", ["HISTORY_COURSE"],
              "End a few turns on different attacks and skills: each turn opens by replaying the last one."),
    relic_job("first_plays", ["THROWING_AXE", "MUSIC_BOX"],
              "Open the fight with an attack (Throwing Axe plays it twice), and play an attack every turn "
              "(Music Box hands back an ethereal copy of the first)."),
    relic_job("play_limits", ["BRILLIANT_SCARF", "VELVET_CHOKER"],
              "Play five cards in one turn, the Angers help: the fifth is free. Then try for a seventh, "
              "which Velvet Choker refuses.",
              setup=["card ANGER Deck"] * 3, teardown=["remove_card ANGER Deck"] * 3, human=True),
    relic_job("draw_and_hand", ["FIDDLE", "RUNIC_PYRAMID"],
              "Play Shrug It Off: Fiddle eats its draw. Leave cards in hand; they stay for next turn."),
    relic_job("turn_one", ["BONE_TEA", "BLESSED_ANTLER", "RADIANT_PEARL", "JEWELED_MASK", "BIG_MUSHROOM",
                           "ROYAL_POISON", "FAKE_BLOOD_VIAL", "TEA_OF_DISCOURTESY", "EMBER_TEA", "SWORD_OF_JADE",
                           "FAKE_ANCHOR"],
              "All of it lands on turn 1. Play the Luminesce at some point."),
    relic_job("every_turn", ["CROSSBOW", "SAI", "MR_STRUGGLES", "FAKE_HAPPY_FLOWER", "POLLINOUS_CORE",
                             "TOASTY_MITTENS", "PAELS_BLOOD", "IRON_CLUB", "SEAL_OF_GOLD"],
              "Go at least six turns, so the five-turn flower and the four-turn core both fire.",
              setup=["gold 100"]),
    relic_job("energy", ["PRISMATIC_GEM", "ECTOPLASM", "SOZU", "BLOOD_SOAKED_ROSE", "PHILOSOPHERS_STONE",
                         "PUMPKIN_CANDLE", "SPIKED_GAUNTLETS", "PAELS_TEARS"],
              "End a turn with energy left (Pael's Tears pays it back), and play Feel No Pain at its "
              "raised cost. Sozu refuses the fight's potions, so no Fairy this time.",
              teardown=["remove_card ENTHRALLED Deck"], human=True),
    relic_job("card_hooks", ["DAUGHTER_OF_THE_WIND", "LOST_WISP", "FORGOTTEN_SOUL", "HAND_DRILL",
                             "FAKE_STRIKE_DUMMY", "PAELS_LEGION", "DIAMOND_DIADEM", "FAKE_ORICHALCUM"],
              "Break the clam's block with an attack (Hand Drill), play Feel No Pain (Lost Wisp), exhaust "
              "something (Forgotten Soul), and have one turn of two cards or fewer (Diamond Diadem).",
              human=True),
    relic_job("biiig_hug", ["BIIIG_HUG"],
              "On pickup it asks for four cards to remove: take Defends. Every reshuffle adds a Soot.",
              human=True),
    relic_job("very_hot_cocoa", ["VERY_HOT_COCOA"], "Spend the four extra energy on turn 1."),
    relic_job("elite", ["BOOMING_CONCH", "BLACK_BLOOD"],
              "Elites only for the conch. Win it: Black Blood heals 12 afterwards.",
              encounter="TERROR_EEL_ELITE"),
    # Relics that carry a count or a flag from one fight to the next. The
    # second fight starts from what the first and the rest site left, which
    # is what the recorder's `relic_state` has to get across.
    Job("carried_state", {"relics"},
        [Fight("SEWER_CLAM_NORMAL"), Send(["room RestSite"]),
         Say("At the rest site take Lift (Girya), then go back to the map.", wait=True),
         Fight("CULTISTS_NORMAL")],
        ["PEN_NIB", "NUNCHAKU", "TUNING_FORK", "JOSS_PAPER", "HAPPY_FLOWER", "PENDULUM", "GIRYA",
         "VENERABLE_TEA_SET"],
        advice="Play plenty of attacks and skills and exhaust a few cards in the first fight, so the "
        "counters end part-way into a cycle."),
]


# The Ruby Raiders are three at once, for area and random targeting; the
# clam is one long fight for everything else.
CARD_ENCOUNTERS = ["SEWER_CLAM_NORMAL", "RUBY_RAIDERS_NORMAL"]


def card_jobs() -> list[Job]:
    """The cards from outside the Ironclad pool, five to a job, in the order
    the sim lists them. Cards the sim refuses to play are left out."""
    from sts2ai import _sim

    names = _sim.card_ids()
    new = names[names.index("ALCHEMIZE"):]
    refused = set(_sim.unsupported_cards())
    new = [c for c in new if c not in refused]
    return [
        Job(f"cards_{i // 5 + 1:02d}", {"cards"}, [Fight(e) for e in CARD_ENCOUNTERS], cards=new[i : i + 5],
            advice="Play the new cards whenever they come up: " + ", ".join(new[i : i + 5]) + ".")
        for i in range(0, len(new), 5)
    ]


def encounter_jobs() -> list[Job]:
    """One job per encounter the sim models, tagged with act and kind."""
    return [
        Job(enc.lower(), {act.lower(), kind.lower(), "encounters"}, [Fight(enc)], advice=ADVICE.get(enc))
        for enc, act, kind in encounters()
    ]


def all_jobs() -> list[Job]:
    return encounter_jobs() + RELIC_FIGHTS + card_jobs()


def send(commands: list[str], quiet: bool = False) -> list[str]:
    """Run dev console lines through the mod; returns what it said."""
    if not commands:
        return []
    out = subprocess.run([str(ROOT / "scripts/game.sh"), *commands], capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if not quiet:
        for line in lines:
            print(f"    {line}")
    return lines


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


def run_state() -> dict:
    """The run as the game has it right now: `sts2ai/run.json`, which the
    mod rewrites whenever it changes (deck, relics, potions, HP, room)."""
    try:
        return json.loads((GAME_DIR / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"active": False}


def blocker(starting: bool = True) -> str | None:
    """Why fights cannot be set up from the run as it stands: no run, a dead
    one, a fight in progress, or something the sim cannot play (a card in
    `UNSUPPORTED_CARDS`, a relic or potion it does not know), which would make every fight fail on the setup rather than on
    the rules. Cards and relics come off with `remove_card` and `relic
    remove`; a potion has no remove command, so it has to be used up."""
    from sts2ai import _sim

    run = run_state()
    if not run.get("active"):
        return "no run in progress (start or continue one)"
    if run.get("dead"):
        return "a run that is over"
    # Between jobs the game still counts the fight just won as in progress
    # until the rewards are left behind, so only a session start checks.
    if starting and run.get("in_combat"):
        return "a fight in progress (finish it first)"
    # The sim checks names the way it reads a recording's start record,
    # which also names an encounter; any one it knows will do.
    return _sim.start_blocker(json.dumps({**run, "t": "start", "encounter": "NIBBITS_WEAK", "enemies": [{"id": "NIBBIT"}]}))


def wait_for_fight(enc: str, before: set[Path], yours: bool) -> Path | None:
    """Block until a new finished recording for `enc` shows up."""
    print("    fight it. ctrl-c to skip." if yours else "    the pilot is on it. ctrl-c to skip.")
    try:
        while True:
            new = [p for p in recordings_for(enc) if p not in before]
            if new and is_finished(new[-1]):
                return new[-1]
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n    skipped")
        return None


def start_record(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            if v.get("t") == "start":
                return v
    return {}


_replays: dict[Path, tuple[bool, str]] = {}


def replay_once(path: Path) -> tuple[bool, str]:
    """`replay`, remembered for the session: the status table replays every
    recording, and a job run only adds new ones."""
    if path not in _replays:
        _replays[path] = replay(path)
    return _replays[path]


# Cards that target a teammate: single player has none, so the game and the
# sim both refuse them, and holding one is all a solo fight can show.
SOLO_UNPLAYABLE = {"BELIEVE_IN_YOU", "COORDINATE", "INTERCEPT", "LIFT", "MIMIC"}


def card_coverage() -> dict[str, bool]:
    """Every card seen in a recording, True if some clean recording shows it
    played (or, for a card that cannot be played, held in hand)."""
    seen: dict[str, bool] = {}
    for path in sorted([*DEV.glob("*.jsonl"), *RECORDINGS.glob("*.jsonl")]):
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        played = {r["id"] for r in records if r.get("t") == "play"}
        held = {c["id"] for r in records if r.get("t") == "snapshot" for c in r.get("hand", [])}
        clean = None
        for card in played | held:
            if seen.get(card):
                continue
            if clean is None:
                clean = replay_once(path)[0]
            unplayable = card in held and all(c.get("cost", 0) < 0 for r in records if r.get("t") == "snapshot"
                                              for c in r.get("hand", []) if c["id"] == card)
            seen[card] = clean and (card in played or unplayable or card in SOLO_UNPLAYABLE)
    return seen


_coverage: dict[str, bool] | None = None


def status(job: Job) -> str:
    """ok when every fight of the job has a clean recording with the job's
    relics on, missing when one has none, DIFF otherwise. A card job counts
    its cards instead: ok once each is verified, else how many are left."""
    global _coverage
    if job.cards:
        if _coverage is None:
            _coverage = card_coverage()
        left = [c for c in job.cards if not _coverage.get(c)]
        return "ok" if not left else f"{len(left)}/{len(job.cards)} left"
    want = set(job.relics)
    worst = "ok"
    for enc in job.fights():
        files = [p for p in recordings_for(enc) if want <= set(start_record(p).get("relics", []))]
        if not files:
            return "missing"
        if not any(replay_once(p)[0] for p in files):
            worst = "DIFF"
    return worst


class Pilot:
    """The policy playing fights through the bridge (`sts2ai.play --record`),
    run in the background and logged to `sts2ai/pilot.log`. It steps
    aside for jobs that need a person."""

    def __init__(self, checkpoint: Path, search: int = 0):
        self.checkpoint = checkpoint
        self.search = search
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.proc is None:
            log = open(GAME_DIR / "pilot.log", "a")
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "sts2ai.play", str(self.checkpoint), "--record", "--search", str(self.search)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None


def fight(enc: str, job: Job, pilot: Pilot | None) -> tuple[bool, str, Path | None]:
    """One recorded fight: top up, start it, wait for the recording, replay."""
    print(f"    monster: {describe(enc)}")
    before = set(recordings_for(enc))
    send(PER_FIGHT, quiet=True)
    if pilot:
        pilot.stop() if job.human else pilot.start()
    send([f"fight {enc}"])
    path = wait_for_fight(enc, before, pilot is None or job.human)
    if path is None:
        return False, "skipped", None
    # A run reloaded after an autosave gets back relics a teardown removed.
    held = set(start_record(path).get("relics", []))
    strays = sorted(held & {r for j in RELIC_FIGHTS for r in j.relics} - set(job.relics))
    if strays:
        print(f"    warning: also carried {', '.join(strays)}; take them off with `relic remove`")
    path = park(path)
    ok, line = replay_once(path)
    print(f"    {line}")
    return ok, line, path


def run_job(job: Job, pilot: Pilot | None, run: Run | None) -> list[tuple[bool, str]]:
    """Set up, run every step, tear down. Returns each fight's result."""
    global _coverage
    _coverage = None  # new recordings change what is verified
    print(f"\n=== {job.name}" + (" (yours)" if job.human and pilot else ""))
    if job.relics:
        print(f"    relics:  {', '.join(job.relics)}")
    if job.advice:
        print(f"    you:     {job.advice}")
    results = []
    if run is not None:
        run.reach(loadout_for(job))
    send([f"relic add {r}" for r in job.relics], quiet=True)
    try:
        for step in job.steps:
            match step:
                case Send(commands):
                    send(commands, quiet=True)
                case Say(text, wait):
                    print(f"    you:     {text}")
                    if wait:
                        input("    enter when done: ")
                case Fight(enc):
                    ok, line, path = fight(enc, job, pilot)
                    results.append((ok, line))
                    log_result(job, enc, ok, line, path)
                    if not ok:
                        break
    finally:
        send([f"relic remove {r}" for r in job.relics] + job.teardown, quiet=True)
    return results


def log_result(job: Job, enc: str, ok: bool, line: str, path: Path | None) -> None:
    with open(GAME_DIR / "results.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "job": job.name, "fight": enc,
                            "ok": ok, "line": line, "file": path.name if path else None}) + "\n")


def queued_jobs(jobs: list[Job]) -> list[Job]:
    """Jobs appended to `sts2ai/queue.jsonl` since last time. A line is
    `{"run": "terms"}` for known jobs, or `{"job": {...}}` for a one-off:
    name, relics, fights, setup, teardown, advice, human."""
    queue, pos = GAME_DIR / "queue.jsonl", GAME_DIR / "queue.pos"
    if not queue.exists():
        return []
    lines = queue.read_text().splitlines()
    done = int(pos.read_text()) if pos.exists() else 0
    pos.write_text(str(len(lines)))
    out = []
    for line in lines[done:]:
        if not line.strip():
            continue
        entry = json.loads(line)
        if "run" in entry:
            out += [j for j in jobs if j.matches(entry["run"].lower().split())]
        elif (spec := entry.get("job")) is not None:
            steps: list[Step] = [Send(spec.get("setup", []))] + [Fight(e) for e in spec["fights"]]
            out.append(Job(spec["name"], {"queued"}, steps, spec.get("relics", []), spec.get("teardown", []),
                           spec.get("advice"), spec.get("human", False)))
    return out


def instant_mode(on: bool) -> bool:
    """Set the game's instant mode (the console only toggles it); returns
    whether it was on before."""
    active = any("ACTIVE" in line for line in send(["instant"], quiet=True))
    if active != on:
        send(["instant"], quiet=True)
    return not active


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terms", nargs="*", help="job names or tags; a job must match all")
    ap.add_argument("--list", action="store_true", help="print the status table and stop")
    ap.add_argument("--redo", action="store_true", help="include jobs that already replay clean")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="run each job N times")
    ap.add_argument("--pilot", type=Path, metavar="CKPT", help="let this checkpoint play the fights")
    ap.add_argument("--search", type=int, default=0, metavar="N", help="the pilot runs a turn search with N sim copies (stronger, slower)")
    ap.add_argument("--queue", action="store_true", help="keep running, taking jobs from sts2ai/queue.jsonl")
    ap.add_argument("--no-deck", action="store_true", help="leave the run's deck alone")
    ap.add_argument("--plain", action="store_true", help="drop Fresnel Lens, so no card arrives enchanted")
    args = ap.parse_args()

    if not RECORDINGS.is_dir():
        sys.exit(f"no recordings folder at {RECORDINGS}; build and load the mod first")

    jobs = all_jobs()
    terms = [t.lower() for t in args.terms]
    picked = [j for j in jobs if j.matches(terms)]
    # With --queue and no terms, only the queue decides what runs.
    if not (args.queue and not terms):
        print("checking what is already recorded...")
        state = {j.name: status(j) for j in picked}
        for j in picked:
            print(f"  {state[j.name]:>7}  {j.name:<32} {' '.join(sorted(j.tags))}")
        if args.list:
            return
        picked = [j for j in picked if args.redo or state[j.name] != "ok"]
    else:
        picked = []

    if not picked and not args.queue:
        print("\nnothing to record.")
        return
    # Unattended, the session opens the game and the saved run itself.
    if args.pilot and not run_state().get("active"):
        print("\nopening the game and continuing the saved run...")
        game.open_run("continue")
    if (why := blocker()) is not None:
        sys.exit(f"\ncannot set fights up: the game has {why}.")
    if not args.pilot:
        print(f"\n{len(picked)} to go. start the game, open a run, and stay out of combat.")
        input("enter when ready: ")
    if args.plain:
        send(PLAIN)
    run = None if args.no_deck else Run()

    pilot = Pilot(args.pilot, args.search) if args.pilot else None
    # Unattended, the animations are only time: instant mode, put back after.
    instant_before = instant_mode(True) if pilot else None
    clean, failed = [], []
    try:
        while True:
            for job in picked:
                for _ in range(args.repeat):
                    results = run_job(job, pilot, run)
                (clean if results and all(ok for ok, _ in results) else failed).append(job.name)
                if (why := blocker(starting=False)) is not None:
                    print(f"\nstopping: the game has {why}")
                    return
            if not args.queue:
                break
            picked = queued_jobs(jobs)
            if not picked:
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if pilot:
            pilot.stop()
        if instant_before is False:
            instant_mode(False)
        print(f"\nclean: {len(clean)}" + (f"\nnot clean: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
