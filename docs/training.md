# Training the combat policy

Step 4 of the order of work in DESIGN.md: the Rust sim exposed to Python,
a fight generator, and a PPO baseline.

## Setup

```
uv sync                                  # venv, torch, and the Rust extension
uv sync --reinstall-package sts2ai       # after editing Rust (cache-keys usually catch it)
uv run python -m sts2ai.train --iters 500
uv run tensorboard --logdir runs
uv run python -m sts2ai.evaluate runs/<time>/latest.pt
uv run python -m sts2ai.train --resume runs/<time>/latest.pt --iters 2000   # continue a run
```

Python 3.13 is pinned in `.python-version`; the extension is built by
maturin from `sim-py/` into the package `sts2ai._sim`.

## Pieces

- `sim/src/gen.rs`: `generate(rng, floor, asc)` rolls a run state for a
  fight on that floor: starter deck plus about two picks per three floors,
  occasional upgrades and removals, a relic every four floors, potions in a
  third of the slots, HP between 40% and 100% (70% and up at the boss,
  which follows a rest site). Floors 1-3 draw weak
  encounters, 5-15 add elites, 16 is the boss. `FightSetup::from_recording`
  turns a recorder file into the same struct, which is how the recordings
  become the held-out set. Encounters from both act 1 variants (Overgrowth
  and Underdocks) are drawn from one pool: each fight is independent, so
  there is nothing to gain from keeping them apart.
- `sim/src/encode.rs`: the fixed observation and action space. Hand and
  choice slots are sorted by (card, upgraded, cost) so the policy sees a
  multiset; each hand slot carries its card id, its enchantment id, and
  the enchantment's amount and spent flag. Draw, discard, and exhaust
  piles are count vectors over (card, upgraded). Enemies sit in the
  game's slot order but nothing reads the order: each slot has its
  creature fields, a one-hot over intent kinds (a vocabulary, so act 2
  kinds append), the intent numbers, and powers. Actions: 10 hand slots x
  7 targets, 4 potion slots x 7 targets, end turn, 20 choice slots, skip.
  Target 0 is "no target", then the six enemy slots, so a bigger board
  appends targets. `mask` marks the legal ones and `decode` maps an index
  back to a `combat::Action`.
- `sim/src/env.rs`: `VecEnv` steps `n` combats in parallel with rayon,
  resets each one as it ends, and writes observations into caller buffers.
  Terminal reward: win is `1 + 0.5 * hp_frac + 0.05 * potions_left`, loss
  or a 500-step timeout is `-1`. This is the stopgap price table from
  DESIGN.md; the run value network replaces it later.
- `py/sts2ai/`: `Envs` owns the numpy buffers. `Policy` embeds card,
  enchantment, monster, move, and potion ids. Enemies are a set: one
  small encoder reads each enemy and the sum joins the MLP input, so no
  weight belongs to a slot. Cards and potions are scored per target from
  their own embedding, the encoded enemy, and the MLP state (`PairHead`),
  so "hit the low one that is about to attack" is learned once. `ppo.py`
  is a plain PPO with GAE. `evaluate.py` runs the policy greedily on the
  held-out set: ten generated fights per encounter from a fixed seed
  (`gen::holdout`), or on run recordings (Real decks, below).
- Reward (`env.rs`): the terminal reward is a win at `1 + 0.5 * hp_frac +
  0.05 * potions_left`, a loss or a 500-step timeout at -1. On top of it,
  potential-based shaping: each step pays the change in half the enemy HP
  fraction taken minus half the player HP fraction lost, measured from the
  fight's own start. A fight's rewards sum to its terminal reward (the
  batch test checks it), so the optimal policy is unchanged; the credit
  for playing Armaments before the Strikes just lands at the play, not
  fifty steps later.
- Checkpoints record the vocabulary and the layout. Vocabulary growth
  (new cards, powers, monsters, relics, potions, moves, enchantments,
  intent kinds) is remapped by name on load. A layout change (a slot count
  or a per-slot feature count, `model.SHAPE_FIELDS`) is not: that is a
  retrain, and the loader says so.

## Curriculum

`--floor-start 4 --floor-ramp 500`: fights come from floors 1 to
`max_floor`, and `max_floor` grows from 4 to 16 over the first 500
iterations. Weak fights stay in the mix so the policy keeps them. Once the
ramp is done, `--hard-frac 0.4` forces that share of fights onto an elite or
the boss, since normal fights are nearly always won by then.

## Advisor turn search

`advise.py --search 256` runs a search at every decision on top of the
policy's pick. `Advisor.fork` makes N copies of the sim state
(`env::Forks`), each with the recording's script dropped, its own dice,
and a reshuffled draw pile (four shuffle groups by default, since the
plan must not know the draw). Every legal first action gets an equal share
of copies; the policy samples the rest of the turn; each copy is scored by
the shaped reward it collected plus the value head where the next turn
starts. The best copy per first action is printed as a whole line of
plays. It runs in well under a second on the GPU. When every plan prints
-1.00, no line found survives the enemy's turn.

## Real decks

The generator's decks are wide but not run decks: picks are random by
rarity, so the pieces of any plan rarely co-occur, and it removes more
Strikes and Defends than a run does. The held-out set comes from the same
generator, so it measures in-distribution play, not the advisor. Fights
from real runs are the measure. The recorder mod writes every fight to
`~/.local/share/SlayTheSpire2/sts2ai/recordings/`; `scripts/record.py`
parks the dev-console fights it drives in `recordings/dev/`, so the top
folder holds run fights only.

```
uv run python -m sts2ai.evaluate runs/<time>/latest.pt --source recordings   # win rate on run decks
uv run python scripts/deckstats.py                                          # what run decks look like
```

`deckstats.py` prints picks, upgrades, removals, enchantments, relics,
potions and starting HP per fight and on average by encounter kind. The
constants in `gen.rs` (`generate_against`) should follow those numbers
once there are a few dozen run fights.

## Speed

Measured 2026-09-22 on the RTX 5070 Ti with 1024 envs x 32 steps: the sim
alone steps a million env-steps a second; one training iteration is about
0.16 s, two thirds of it the PPO update, which is memory-bound (the pair
head's `[batch, hand, targets, hidden]` tensor). CUDA graphs and fewer,
larger minibatches change nothing. About 200k steps/s, so a 2000-iteration
run is under six minutes and 800M steps is about an hour.

## What to watch

`episode/win_rate` and `episode/win_boss` (also `win_elite`) in TensorBoard, and
`eval/holdout_win_rate` with its per-kind splits every `--eval-every` iterations. The milestone
in DESIGN.md is 80% on the act 1 boss from real-run decks.
