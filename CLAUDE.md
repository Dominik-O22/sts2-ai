# Working in this repo

`DESIGN.md` holds the decisions. This file holds the things an agent gets
wrong on its first day.

## Rules that are not negotiable

- Port from `decompiled/`, never from a wiki or another simulator, and cite
  the source class in a comment. If the port disagrees with a recording, the
  port is usually reading the wrong function. Check before inventing a rule
  that fits the data.
- Ids are append-only. `CardId`, `PowerId`, `MonsterId`, `RelicId`, `PotionId`
  and `EnchantmentId` are indexed by position, and `sim/vocab.txt` pins the
  order, so a new entry goes at the end of both the enum and the `ALL_*` list.
  Moving one invalidates every trained checkpoint. A test catches it.
- Regenerate `sim/vocab.txt` after adding any id, and commit it:
  `cd sim && cargo run --release --example vocab > vocab.txt`.

## Setting up game states to record

Fidelity comes from real fights, not from tests written by whoever wrote the
rule. `scripts/record.py` is how you get them: it runs jobs (an encounter, a
relic group, a multi-fight setup) with no clean recording, builds each one
through the dev console, says what the monsters do and what to do to make it
show, waits for the fight, then replays it.

```
uv run python scripts/record.py --list                  # status per job
uv run python scripts/record.py hive elite              # jobs matching every term
uv run python scripts/record.py soul_fysh_boss --redo --repeat 2
uv run python scripts/record.py --pilot runs/set-3/latest.pt --queue
```

With `--pilot` the policy plays the fights and the run goes unattended; with
`--queue` it keeps taking jobs appended to `sts2ai/queue.jsonl`, which is
how you set up the next fight while a session is running. Every result lands
in `sts2ai/results.jsonl`.

Reach for it whenever a fight needs setting up, not just when recording a gap.
It already knows the deck that makes a long fight, the potions worth carrying,
and that everything must be set before `fight`, because the console cannot add
to a combat that is already running without the replay diverging.

`scripts/game.py` covers what the dev console cannot: `continue` and
`new_run ASC` launch the game if needed and load a run through the mod's
own `sts2ai ...` commands (a `--pilot` session continues by itself when no
run is loaded), `menu` goes back to the main menu, `shot` screenshots the
game window alone, `click FX FY` and `key` send input to it.

Dom or the pilot plays the fights. You set them up and read the diffs. Jobs
marked `human` (an idle first turn, a pickup screen) need Dom.

Most of what it prints comes from the sim, so a newly ported act shows up with
nothing written by hand. `ADVICE` and `RELIC_FIGHTS` in that file are the
exception: one line per fight where playing straight would not show the
mechanic, and the relic jobs. Add one when you port a monster whose
interesting behaviour needs steering toward, or a relic.

`docs/replay.md` has the detail: what the recording holds, what the replay
forces versus checks, and the dev console commands.

## Commands

```
cd sim && cargo test --release              # 53 tests
cd sim && cargo run --release --bin replay  # every recording against the sim
uv sync --reinstall-package sts2ai          # rebuild the Python extension
uv run python -m sts2ai.vocab               # checkpoint remap self-check
./scripts/build-mod.sh                      # recorder mod + bridge, needs a game restart
uv run python -m sts2ai.play runs/<run>/latest.pt --search 256   # policy plays combats
uv run python -m sts2ai.train --iters 500
```

Run the replay suite after any change to the sim. It is the regression suite
that matters; the unit tests only cover mechanics a replay diff would not
pin down.
