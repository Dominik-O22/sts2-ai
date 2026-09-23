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
  fight on that floor: starter deck plus about two card rewards per three
  floors, each three offers of which four runs in five take the one that
  fits a deck plan (`PLANS`: Strength, exhaust, block, Vulnerable, HP loss)
  and skip once the deck has 20 cards and nothing fits. Upgrades go to
  non-basic cards, removals take Strikes first, and both come faster after
  act 1, as does max HP; a relic every four floors, potions in a
  third of the slots, an Ancient relic per act after the first, HP between
  40% and 100% (70% and up at the boss, which follows a rest site). Floors
  run 1 to 48, sixteen per act: in each act the first floors draw weak
  encounters (3 in act 1, 2 later), 5-15 add elites, the 16th is the boss. `FightSetup::from_recording`
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
  Rewards are described under Reward below; they are the stopgap price
  table from DESIGN.md, which the run value network replaces later.
- `py/sts2ai/`: `Envs` owns the numpy buffers. `Policy` embeds card,
  enchantment, monster, move, and potion ids. Enemies are a set: one
  small encoder reads each enemy and the sum joins the MLP input, so no
  weight belongs to a slot. Cards and potions are scored per target from
  their own embedding, the encoded enemy, and the MLP state (`PairHead`),
  so "hit the low one that is about to attack" is learned once. `ppo.py`
  is a plain PPO with GAE. `evaluate.py` runs the policy greedily on the
  held-out set: ten generated fights per encounter from a fixed seed
  (`gen::holdout`), or on run recordings (Real decks, below).
- Reward (`env.rs`): the terminal reward is a win at `1 + w * hp_frac +
  0.1 * potions_left`, a loss or a 500-step timeout at -1. `w` is 0.5,
  except after an act boss, where the next act's Ancient heals 80% of the
  missing HP at A10 and only the other 20% counts (0.1). A potion at 0.1 is
  about 16 HP. On top of it, potential-based shaping: each step pays the
  change in half the enemy HP lost (a running count over the fight, as a
  fraction of what the enemies started with) minus `w` times the player HP
  fraction lost, plus 0.1 per potion gained (a drink counts as one lost),
  measured from the fight's own start. Before the potion term a drink cost
  nothing until the fight ended, and the policy drank combat potions in
  weak fights it lost 5% HP in; with it, over 600 iterations from set-11,
  potions per fight fell by a third everywhere (weak 0.18 to 0.12, act 3
  boss 0.73 to 0.47) at the same win rates and about 2 points more HP lost
  in elites, the price the terminal reward sets. The count matters:
  read off the current HP bars, a monster that revives at full HP (Test
  Subject, the Waterfall Giant's blast turn) took the potential back, the
  killing blow cost half a fight's reward, and the policy learned to leave
  the Test Subject at 21 HP for five turns. A fight's rewards
  sum to its terminal reward (the batch test checks it), and PPO runs
  undiscounted (`gamma` 1), so the optimal policy is unchanged; the credit
  for playing Armaments before the Strikes just lands at the play, not
  fifty steps later. With a discount the policy was paid for finishing
  sooner, and traded HP and potions for it.
- Checkpoints record the vocabulary and the layout. Vocabulary growth
  (new cards, powers, monsters, relics, potions, moves, enchantments,
  intent kinds) is remapped by name on load. A layout change (a slot count
  or a per-slot feature count, `model.SHAPE_FIELDS`) is not: that is a
  retrain, and the loader says so.

## Curriculum

`--floor-start 4 --floor-ramp 500`: fights come from floors 1 to
`max_floor`, and `max_floor` grows from 4 to the last boss floor
(16 x `--acts`, 48 by default) over the first 500 iterations. Weak fights stay in the mix so the policy keeps them. Once the
ramp is done, `--hard-frac 0.4` forces that share of fights onto an elite or
the boss, since normal fights are nearly always won by then. With `--focus`
those forced fights are drawn by how often the policy recently lost each
elite and boss instead of evenly. `--lr-final` anneals the learning rate
linearly to that value over the run.

## Turn search

`py/sts2ai/search.py` plays copies of a fight (`env::Forks`) to the end of
the turn: each copy has its own dice and a reshuffled draw pile (shuffle
groups share one, since a plan must not know the draw), starts with a
given first action, and the policy samples the rest. A copy scores the
shaped reward it collected plus the value head where the next turn starts.
Copies of many fights go through the network in one batch, and only the
ones still playing each step. Copies that look the same share a row
(`Forks::observe_unique`): copies of a root that took the same first move
in the same shuffle group mostly do, so the network sees about a quarter
of the rows, and each copy still samples its own action.

`advise.py --search 256` (and `play.py`, and `record.py --pilot CKPT
--search N`) runs it at every decision: every legal first action gets an
equal share of copies, openings are ranked by their copies' mean score,
and the best copy of each is printed as the whole line. When every plan
prints -1.00, no line found survives the enemy's turn.

`uv run python -m sts2ai.searcheval CKPT --copies 128 --mean` plays the
held-out elite and boss fights greedy and again with the search from the
same shuffles. On set-6: 66.7% greedy, 85.4% with the search. `--depth 2`
plays copies one more player turn: 86.7%, at twice the cost.

## Search distillation

A plateaued PPO run gains nothing from more iterations; the search's
choices are better than the policy's. `--search-states N` runs the search
each iteration on N envs over the policy's `--search-top` favourite first
actions, and a cross-entropy term pulls the policy toward the search's
improved distribution: its own logits, each searched action moved by
(its mean copy score - the policy's expected score) / `--search-temp`
(Gumbel AlphaZero's completed-Q target). An action the search cannot tell
apart from the rest keeps the policy's probability, which matters: a
softmax over raw scores instead spread the policy over near-ties and
wrecked it in 30 iterations. Targets go to a buffer and the loss waits for
`--search-warmup` of them. From set-5, 2000 iterations at 10 repeats:
plain PPO 89.1% held-out and 58% on act 2 bosses, distillation 90.2% and
66%.

Roots come only from envs where the policy's favourite first move leads
the runner-up by at most `--search-margin` (0.8). Where it leads by more,
the target is the policy's own and trains nothing: offline, over 40% of
random roots were like that and carried 8 to 16% of the targets'
divergence from the policy. Filtered roots carry about 1.5x the signal
each, so `--search-states 352` matches 512 random ones. The search does
not hold up training: one still running when the rollout ends keeps
going, and that iteration starts none (`--search-sync` waits instead).
From set-11, 600 iterations at 10 repeats: 512 random roots with
`--search-sync` 91.8% at 92k steps/s, 352 filtered roots without it
91.9% at 112k. Entropy fell faster with the filtered roots, whose
targets pull harder.

`uv run python -m sts2ai.compare A.pt B.pt --repeats 10` puts checkpoints
side by side by act and kind; two repeats swing a boss number six points.

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

Measured 2026-09-23 on the RTX 5070 Ti (8-core Ryzen 9850X3D, SMT off in
the BIOS) with 1024 envs x 32 steps, resuming set-11, the same machine
and hour for both numbers: plain PPO runs about 178k steps/s; with the
search (`--search-states 352`, 128 copies, the defaults otherwise) about
130k, where the set-11 recipe on the code before (512 random roots, one
search per iteration) ran 41k.

What the search costs depends on how it shares the machine with the rest
of the iteration:

- It runs in a thread beside the update, pauses while the rollout runs
  (`search_go`), and never holds the iteration up (Search distillation,
  above). The rollout waits on the GPU and the sim every step and ran
  three to four times slower beside it.
- Its network is compiled too, and every graph compiles in the first
  iteration, with the search waited for. After that `torch.compile` may
  not compile again (`eager_on_recompile`): compiling in one thread while
  the other runs broke Adam once and hung once.
- Nothing in the update reads a value back per minibatch, so the update
  does not stall on each GPU round trip.
- Observations cross to the GPU from pinned buffers.
- `Forks` keeps copies that are in the same state on one node, with
  their own dice, and steps a (node, action) once unless the step rolls
  any; about half the search's sim steps were such repeats. That pays
  only because a clone is cheap: monster move graphs and the shuffle log
  sit behind `Arc`, and the extension allocates with mimalloc (glibc's
  malloc spent a fifth of the search waiting on locks).
- A node is hashed (for the row sharing) inside `Forks::step`, while its
  state is in cache, skipping the all-zero blocks that make up 97% of an
  encoding.

To profile the sim, `samply record` works once
`/proc/sys/kernel/perf_event_paranoid` is 1 or lower.

The search at play time (`advise.py --search`, `searcheval`) gets the
same row sharing and allocator; its search on set-11 went from 10 s to 4 s.

## What to watch

`episode/win_rate` and `episode/win_boss` (also `win_elite`) in TensorBoard, and
`eval/holdout_win_rate` with its per-kind splits every `--eval-every` iterations. The milestone
in DESIGN.md is 80% on the act 1 boss from real-run decks.
