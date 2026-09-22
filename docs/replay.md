# Recording and replaying real combats

The recorder mod logs every combat the game plays. The replay tool feeds
those logs through the sim and reports the first place the two disagree.

## Record

1. Build and install the mod: `scripts/build-mod.sh`. It compiles
   `mod/Recorder.cs` against the game's assemblies and copies the DLL and
   manifest to `<game>/mods/sts2ai/`.
2. Start the game. The first time, open the Mods menu and accept mod
   loading; the loader skips every mod until that flag is set.
3. Play combats. Each one lands in
   `~/.local/share/SlayTheSpire2/sts2ai/recordings/<time>-<encounter>.jsonl`.
   The dev console (`fight <encounter>`, `card`, `relic`, `potion`) builds
   arbitrary setups without needing a run to produce them.

The game log (`~/.local/share/SlayTheSpire2/logs/godot.log`) prints
`[sts2ai] recorder ready` when the mod loaded and `[sts2ai] recording ...`
per combat.

## Replay

```
cd sim && cargo run --release --bin replay            # everything recorded
cd sim && cargo run --release --bin replay FILE.jsonl # one file
```

Each line reports `ok`, `DIFF` with the record index and the first differing
field, or `ERR` for an id the sim does not know yet.

## What the log holds

One JSON object per line:

- `start`: encounter, room kind, ascension, deck, relics, potion slots,
  enemies with their rolled HP.
- `snapshot`: the full state at each point the player could act, written by
  a per-frame poll while it is the play phase and nothing is resolving.
  Includes hand with costs, draw pile in order, discard, exhaust, powers,
  potions, and each enemy's HP, block, powers, and next move.
- `play` (card, hand index, target), `potion` (id, target), `shuffle` (the
  resulting draw order), `turn_start`, `end` (won, HP).

## What the replay forces versus checks

Forced from the log: the opening draw order, every later shuffle, enemy
starting HP, and each enemy's next move. Everything else is the sim's own
work and is diffed at every snapshot. Card choices (Armaments, exhaust
picks) are not logged; the replay tries each option and keeps the one whose
result matches the next snapshot.

Card random targets are scripted from the recorded hits, random exhausts
from the recorded exhausts. Random outcomes with no record (Aggression's
pull, the target of a power hit like Juggernaut) are re-rolled: when a
snapshot does not match, the replay rewinds to the last matching one,
reseeds the card-selection and target streams, and tries again, up to 64
times. The report shows how many reseeds a clean
replay needed. Cards created mid-combat are logged as `gen` records and
forced like the shuffles.

A snapshot immediately followed by another snapshot, with no action in
between, caught the game mid-resolution and is skipped.

## Recording what is missing

`scripts/record.py` walks the encounters with no clean recording, builds a
deck for each through the dev console, says what the monster does and what
you have to do to make it show, waits for the fight, then replays it.

```
uv run python scripts/record.py --list          # status per encounter
uv run python scripts/record.py --act underdocks
uv run python scripts/record.py --only WATERFALL_GIANT_BOSS
uv run python scripts/record.py --repeat 2      # each fight twice
```

The encounter list, the acts and tiers it filters on, and the description of
every fight all come from the sim, so a newly ported act appears here with
nothing written by hand. Only the one-line "what you have to do" advice is
hand-written, and only for fights where playing straight would not show the
mechanic.

Before it starts, it asks the sim whether it could build a fight from the
run as it stands. A colorless card in the deck, a relic from an act that is
not ported, a Colorless Potion in the belt: any of those makes every fight
fail on the setup rather than on the rules, and the dev console can add
things to a run but not take them away, so it says so and stops.

The deck it builds is mostly block with barely any damage. The long move
cycles are five and six turns, so a deck that kills a boss in three proves
nothing, and the cards are the plainest in the pool so a divergence points
at the monster rather than at a card port. Fairy in a Bottle goes in the
belt before every fight, which buys one death.

Everything is set before `fight`. Powers or block handed out mid-combat by
the console are not in the `start` record, so the sim never sees them and
the replay diverges at the next snapshot.

## Scripting setups from outside the game

The mod also runs dev console commands from a file, so fights can be set
up without touching the in-game console. With a run open:

```
scripts/game.sh "fight NIBBITS_NORMAL"
scripts/game.sh "card BODY_SLAM Deck" "relic add VAJRA" "potion FIRE_POTION"
```

Useful commands: `fight <ENCOUNTER>`, `card <CARD> [Hand|Deck|Draw|Discard]`,
`remove_card <CARD> [pile]`, `relic add|remove <RELIC>`, `potion <POTION>`,
`power <POWER> <amount> <target>`, `energy <n>`, `draw <n>`, `heal <n>`,
`upgrade <hand-index>`, `win`, `unlock all`. Ids are the class names in
screaming snake case. Encounter ids match `sim/src/encounter.rs`.
Ascension is fixed at run start, so start the run at the level you want.

## Advisor

The same harness, one record at a time, in front of a trained policy:

```
uv run python -m sts2ai.advise runs/<run>/latest.pt
uv run python -m sts2ai.advise runs/<run>/latest.pt --replay FILE.jsonl --delay 0.2
```

It follows the newest file in the recordings folder, picking up a new one
when a new combat starts, and prints the policy's pick with two
alternatives at every decision point:

```
turn 1 | 3 energy | HP 80/80 (10 block) | Nibbit 46/46 Butt
  -> Flame Barrier                          86.7%
     Pyre                                    6.6%
     Drum Of Battle                          2.4%
```

You play the moves yourself. Nothing is sent back to the game.

`sim::replay::Replayer` holds the state: `feed` takes one record,
`flush` releases a snapshot it is holding, and `at_decision` says whether
the game is waiting on the player. The holding matters because a snapshot
is only known to be a real decision point once the next record arrives, or
once the recorder goes quiet, which is what `flush` means. A snapshot the
sim cannot settle stays held, so a mid-resolution poll never diverges; the
cost is that a decision point needing a reseed gets no advice until the
next record lands. A card choice open in the sim (Armaments, an exhaust
pick) is a decision point too, and the next snapshot settles which card the
game actually applied.

`--replay` feeds an existing recording as if it were being written, which
is how the advisor is tested. Divergences print and the advisor keeps
going.
