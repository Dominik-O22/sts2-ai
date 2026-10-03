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
uv run python scripts/record.py --pilot runs/set-11/latest.pt --search 128 cards   # verify ported cards
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
interesting behaviour needs steering toward, or a relic. `EVENT_STARTS`
holds the console line for an event fight `fight` cannot build.

`docs/replay.md` has the detail: what the recording holds, what the replay
forces versus checks, and the dev console commands.

## Worktrees

A new worktree gets `decompiled/` copied in (`.worktreeinclude`). It does
not get `runs/`: read checkpoints from the main tree by absolute path. Its
`.venv` and `target/` are its own, so the first build there takes a while.
The Python extension builds into `py/sts2ai/_sim*.so`, untracked, so run
`uv sync --reinstall-package sts2ai` in a new worktree before any Python.
Copying the main tree's `.venv` (`cp --reflink`) saves the download, but its
`sts2ai.pth` still points at the main tree's `py/` until that sync.

## Commands

```
cd sim && cargo test --release              # check its exit code, not a pipe's
cd sim && cargo run --release --bin replay  # every recording against the sim
cd sim && cargo run --release --example runcheck -- --effects ~/.local/share/SlayTheSpire2/steam/*/modded/profile1/saves/history/*.run   # real runs against the run layer
uv sync --reinstall-package sts2ai          # rebuild the Python extension
./scripts/build-pgo.sh                      # the same with profile-guided optimization, 8-20% faster sim
uv run ruff check && uv run ruff format     # Python lint; never trade speed for a lint
uv run python -m sts2ai.vocab               # checkpoint remap self-check
./scripts/build-mod.sh                      # recorder mod + bridge, needs a game restart
uv run python -m sts2ai.play runs/<run>/latest.pt --search 256   # policy plays combats
uv run python -m sts2ai.train --iters 500
```

Run the replay suite after any change to the sim. It is the regression suite
that matters; the unit tests only cover mechanics a replay diff would not
pin down.

## Judging a change to run decisions

`scripts/ab.sh NAME "ARGS A" "ARGS B"` plays both arms side by side on the
same 2,048 seeds with the standard settings (gen8 fights, turn search in
elites and bosses) and prints `sts2ai.paired`: each difference with a 95%
interval. Use it for every run-policy change, and change one thing per arm.
A single arm against a number from an older log is not a result: seeds,
combat checkpoint and search settings drift, and at a 4% win rate 2,048 runs
only separate differences of about a point. An interval that spans zero is
noise; say so rather than reading a trend into it.

## Judging a combat checkpoint

`uv run python -m sts2ai.bench NEW.pt --base OLD.pt` plays both on about
8,850 held-out elite and boss fights (winners', sts2.fun players', and the
clone's own on fresh seeds) and 4,800 weak and normal ones (winners' and
sts2.fun players') from the same seeds, and reports per-fight differences
in wins and HP lost by act and kind; greedy at 8 seeds takes minutes and
resolves about 0.3 points. HP lost on the easy fights is the number that
compounds over a run; check it on every change. Greedy is the screen, and enough for a training recipe
change (epochs, lr, data mix). `--mode hybrid32` is the verdict for a
change that could just teach greedy what the pilot's search already does
(an act 3 boss specialist gained 3 points greedy and nothing with the
hybrid): it defaults to one seed on a fixed 300 boss fights a set, about
25 minutes a new checkpoint, the base cached. Do not widen it to every
fight and 8 seeds: that is about 7 hours a checkpoint. A paired `ab.sh` on
fresh seeds gives the run-level number.

Long jobs (evals, training) run as their own systemd units, never as plain
children of the session: `ab.sh` puts each arm in a capped scope, and a
job that must outlive the session starts with `systemd-run --user
--unit=NAME --same-dir ...`. On 2026-09-29 two eval arms outgrew the
memory left beside the desktop and systemd-oomd killed the whole terminal
scope, the session and both arms with it. An `--afterstate` arm needs about
8 GB, a plain one about 4.3 GB, and the desktop holds about 14 of the 30.

Work on a branch off current master and land it before starting the next
thing on top. Evals run from long-lived side branches missed the sim
speedup and the newest combat model for days.
