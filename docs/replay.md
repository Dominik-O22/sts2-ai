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
  enemies with their rolled HP. `relic_state` holds the charge or count of
  the relics that carry one between fights (Ember Tea's combats, Iron Club's
  plays, whether Fur Coat marked this room), read at combat setup before
  any of them fires. `opening` is the draw pile the opening shuffle made,
  and `early` holds the records written between setup and the first
  decision point (Crossbow's turn 1 card, Whispering Earring's plays and
  what they exhaust), which the file would otherwise miss.
- `snapshot`: the full state at each point the player could act, written by
  a per-frame poll while it is the play phase and nothing is resolving.
  Includes hand with costs, draw pile in order, discard, exhaust, powers,
  potions, and each enemy's HP, block, powers, and next move.
- `play` (card, hand index, target), `potion` (id, target), `shuffle` (the
  resulting draw order), `turn_start`, `end` (won, HP).
- `choice`: a card selection opened. Names the card in the play pile or the
  potion that just left the belt, and the options. With the bridge
  answering, it also carries `min` and `max`.
- `picked`: one card the bridge took from the open selection, then one with
  `card: null` when it closed. Hand-played recordings have none, and the
  replay infers the pick from the next snapshot instead.

## What the replay forces versus checks

Forced from the log: the opening draw order, every later shuffle, enemy
and player HP at the first decision point, and each enemy's next move.
Adopted from the snapshots where the game rolls what nothing records: the
depth of a card shuffled in at random (Beckon, Soot, Dazed), the costs
Confused rolls, a card taken from a choose-a-card screen whose `gen`
record only follows the pick, what a card Entropy transformed became, and
the potion Alchemize put in the belt. Everything else is the sim's own
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

`scripts/record.py` is the way to set up a fight, including one that is already
recorded. It knows the deck that makes a long fight, what to carry, and that
everything has to be set before `fight`.

`scripts/record.py` runs jobs with no clean recording: one per encounter,
the relic groups, and setups that span fights (relics carried through a rest
site). It builds each through the dev console, says what the monster does
and what you have to do to make it show, waits for the fight, then replays
it.

```
uv run python scripts/record.py --list              # status per job
uv run python scripts/record.py underdocks          # terms match names and tags
uv run python scripts/record.py waterfall_giant_boss
uv run python scripts/record.py --redo --repeat 2   # clean ones too, twice each
uv run python scripts/record.py --pilot runs/set-3/latest.pt --queue
```

The encounter list, the acts and kinds it tags them with, and the
description of every fight all come from the sim, so a newly ported act
appears here with nothing written by hand. Only the one-line "what you have
to do" advice and the relic jobs are hand-written.

With `--pilot`, the policy plays through the bridge (`sts2ai.play --record`).
It samples its moves so repeats differ, and ends a fight with `win` once the
sim loses track (the recording up to there is what the replay checks) or HP
in place of ending a turn the sim says would kill you, Fairies included (so the run survives). The game goes to instant mode
for the session. Jobs marked `human` hand their fights back to you.
`--queue` keeps it running on `sts2ai/queue.jsonl`: `{"run": "terms"}`
for known jobs, `{"job": {"name", "relics", "fights", "setup", "teardown",
"advice", "human"}}` for a one-off. Each fight's result is appended to
`sts2ai/results.jsonl`.

Before it starts, it asks the sim whether it could build a fight from the
run as it stands. A card the sim refuses (`UNSUPPORTED_CARDS`: Splash, Mad
Science), a relic or potion it does not know: any of those makes every fight
fail on the setup rather than on the rules, so it says so and stops. A
card or relic can come off with `remove_card` or `relic remove`; a potion
has no remove command and has to be used up.

`--relics` walks the relic fights in `RELIC_FIGHTS` instead of the
encounters. Each group of relics goes on before its fight and comes off
after, along with any card its pickup left in the deck, so every group
starts from the same run.

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

The game writes the `play` record only once the card has resolved, which
for a card with a choice is after you picked. So the recorder also polls
the hand's select mode and writes a `choice` record the moment the screen
opens, naming the card in the play pile (with its hand index) and the
options. The replayer plays that card early, which opens the same choice
in the sim while you are still looking at it, and then skips the `play`
record that follows. A choice whose card the sim cannot place (a target
it would need) waits for the real record, and the advice is late for it
as before.

A potion that opens a choice (Attack Potion) works the same way through
the `potion` field. The options also replace the sim's own roll for a
choose-a-card screen, so the pick is made from the cards the game shows.

`--replay` feeds an existing recording as if it were being written, which
is how the advisor is tested. Divergences print and the advisor keeps
going.

To let the policy play instead of advising, see `docs/bridge.md`.
