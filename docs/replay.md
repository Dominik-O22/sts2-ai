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

Not forced yet: random targets, random exhausts, and random card
generation. Decks using those can diverge without a sim bug. Logging those
outcomes is the next step for the harness.
