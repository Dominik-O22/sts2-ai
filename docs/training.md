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
  third of the slots, HP between 40% and 100%. Floors 1-3 draw weak
  encounters, 5-15 add elites, 16 is the boss. `FightSetup::from_recording`
  turns a recorder file into the same struct, which is how the recordings
  become the held-out set.
- `sim/src/encode.rs`: the fixed observation and action space. Hand and
  choice slots are sorted by (card, upgraded, cost) so the policy sees a
  multiset. Draw, discard, and exhaust piles are count vectors over
  (card, upgraded). Enemies sit in the game's slot order, with intents and
  powers. Actions: 10 hand slots x 6 targets, 4 potion slots x 6 targets,
  end turn, 20 choice slots, skip. `mask` marks the legal ones and `decode`
  maps an index back to a `combat::Action`.
- `sim/src/env.rs`: `VecEnv` steps `n` combats in parallel with rayon,
  resets each one as it ends, and writes observations into caller buffers.
  Terminal reward: win is `1 + 0.5 * hp_frac + 0.05 * potions_left`, loss
  or a 500-step timeout is `-1`. This is the stopgap price table from
  DESIGN.md; the run value network replaces it later.
- `py/sts2ai/`: `Envs` owns the numpy buffers, `Policy` embeds card, monster,
  and potion ids and runs an MLP with masked policy and value heads,
  `ppo.py` is a plain PPO with GAE, `evaluate.py` runs the policy greedily
  on the recordings.

## Curriculum

`--floor-start 4 --floor-ramp 500`: fights come from floors 1 to
`max_floor`, and `max_floor` grows from 4 to 16 over the first 500
iterations. Weak fights stay in the mix so the policy keeps them.

## What to watch

`episode/win_rate` and `episode/win_boss` in TensorBoard, and
`eval/recorded_win_rate` every `--eval-every` iterations. The milestone
in DESIGN.md is 80% on the act 1 boss from real-run decks.
