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
- `py/sts2ai/`: `Envs` owns the numpy buffers. The policy (`SlotMLP`,
  `--arch slots`) embeds card, enchantment, monster, move, and potion ids.
  Enemies are a set: one small encoder reads each enemy and the sum joins
  the MLP input, so no weight belongs to a slot. Cards and potions are
  scored per target from their own embedding, the encoded enemy, and the
  MLP state (`PairHead`), so "hit the low one that is about to attack" is
  learned once. `--hidden` and `--depth` size its torso (512 and 2 through
  set-12). `--arch attn` (`SlotAttention`) turns the same slots into
  tokens (a global one for the rest of the observation, then hand cards,
  enemies, potions and choices) and runs a transformer over them before
  the same heads, so a card's encoding has seen the board; give it
  `--warmup`. The checkpoint records the arch, and every tool builds the
  right network from it (`model.load_policy`). `ppo.py`
  is a plain PPO with GAE. `evaluate.py` runs the policy greedily on the
  held-out set: ten generated fights per encounter from a fixed seed
  (`gen::holdout`), or on run recordings (Real decks, below).
- Reward (`env.rs`): the terminal reward is a win at `1 + w * hp_frac +
  0.1 * potions_left`, a loss or a 500-step timeout at `-1 + 0.2 *
  enemy_hp_taken` (capped at -0.8). Under a flat -1 every line of a lost
  fight paid the same, so the policy folded once its value read a fight as
  lost, and search tied every option there. `w` is 0.5,
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
- Checkpoints record the vocabulary, the layout and the arch. Vocabulary growth
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

Other players' winning runs come from ststracker.app (`scripts/tracker.py
--crawl`, 1,193 A10 Ironclad wins on 2026-09-24). `sts2ai.setups` turns
each run's elite and boss fights into setups, the deck rebuilt at that
floor from the final one, and splits them by player: 7,560 to train on and
1,649 held out (2,801 more hold a card the sim lacks, Mad Science and Splash
mostly).

```
uv run python -m sts2ai.setups                                               # pages -> setups/train.jsonl, holdout.jsonl
uv run python -m sts2ai.evaluate runs/<time>/latest.pt --source setups       # win rate on held-out winners' fights
uv run python -m sts2ai.evaluate runs/<time>/latest.pt --source easy         # HP lost in weak and normal fights, against the winners
uv run python -m sts2ai.train ... --real-setups $S/train.jsonl:0.25,$S/easy-train.jsonl:0.1   # S=~/.local/share/SlayTheSpire2/sts2ai/tracker/setups
```

With `--real-setups`, each file takes its share of the resets (after the
colon, `--real-frac` where none is given), one of its fights drawn with its
enemies rolled afresh; training evaluates on the held-out ones beside
the generated set. They are all wins, so they only show decks that got
through; the generator keeps the weak ones. set-14 on them, greedy: act 1
elites 92%, bosses 69%; act 2 88%, 67%; act 3 86%, 48%, against 15% on the
generated act 3 bosses, whose decks are far weaker than a winner's.

Winners alone flatter the HP a fight costs: the runs that went badly are
not there. sts2.fun keeps every run a player uploads, so `sts2ai.sts2fun`
takes the players who win at least half of 15+ A10 Ironclad runs, with all
their runs, and turns their fights into setups the same way (its own
`setups/` directory, split by player). The pages give less than
ststracker's: no patch, no card removals or purchases, no potion use, so
the deck is rebuilt forward from the starter deck and reconciled with the
final one at the first shop, fights carry no potions, and each line counts
the changes it could not place (`unexplained`). They show the fights these
players died in (`died`) and whether the run was won (`run_won`).

```
uv run python -m sts2ai.sts2fun crawl       # players, then their runs, 2 s between requests; resumes
uv run python -m sts2ai.sts2fun setups      # -> ~/.local/share/SlayTheSpire2/sts2ai/sts2fun/setups/*.jsonl
uv run python -m sts2ai.evaluate runs/<time>/latest.pt --source easy --setups <sts2fun easy file>
```

On 2026-09-25, 26 players (806 runs, 54% won; runs that meet Doormaker,
a boss the game has since removed, are left out as another version):
keeping only their won runs lowers their HP lost per weak or normal fight
by 0.4 to 1.5 and per elite or boss by 1 to 3. For the 14 players at 60%+
it is half that: the survivorship bias is small, and smaller the better
the player. Those 14 lose less than ststracker's winners (1.7 HP per easy
fight against 3.1), and gen5's gap to them is +6.4 HP a fight (+7.2 on
runs since June, +3.5 on the near-exact decks, which are mostly act 1).

## Generations from scratch

Once a line of fine-tunes plateaus (set-13 to set-14 at about 85.6% on
the generated set), a change to the reward, the data or the network is
tested from scratch, not bolted onto the plateaued checkpoint: at entropy
0.4 the old policy hardly explores what a new reward pays for. Runs are
compared at equal training hours with `sts2ai.curves` (snapshots every
1,000 iterations), judged on winners' elite and boss fights and on the HP
winners' easy fights cost, not on the generated set.

What the 2026-09-24/25 runs found, each from scratch with ab-attn's recipe
(lr 3e-4, warmup 30) and 704x256 search:

- gen2 (dot-product heads, pile tokens, choice attention; winners' elites
  and bosses at 30%): +1.7 points on winners' fights over ab-attn at 2 h,
  the rest level. Choice attention costs 19% of a training step for the
  4.5% of states with a choice.
- gen3 (HP priced per point at four times the old weight): fewer
  self-damage cards but kills a turn slower, so the easy-fight HP gap did
  not move and winners' fights fell 3 points. Reverted (#32).
- gen4 (`--search-margin 1.0 --search-kinds Weak,Normal,Elite,Boss
  --search-coef 1.0`, winners' fights at 55%: `train.jsonl:0.35,
  easy-train.jsonl:0.2`): winners' fights 77.1% at 2.17 h against gen2's
  75.1%, bosses 61.4 against 58.1, easy-fight HP beyond winners +4.9
  against +5.6. Search overrules its confident choices as often as
  before (19%), so the gain is the data more than absorbed search.
- gen5 (gen4 with `--incoming`: a head on the global token learning the
  damage the coming enemy phase deals, MSE at 0.5): at or above gen4 at
  every snapshot; at 6,000 iterations 84.4% generated, 77.2% winners',
  +4.7 HP, 95.0% of normals won.

The easy-fight HP gap (the policy losing 13.8 HP greedy in act 2 and 3
normals where winners lost 4.1) is play, not the measurement: rebuilt
fights cost the same HP as the exact recorded ones (141 of Dom's fights),
and the sim matches the pilot's real games. Search halves it; neither
2,048 copies, a two-turn search nor full-fight rollouts do better, so the
search is not what limits it. It is spread over encounters (the top 10 of
55 carry 45%). The worst, four Scrolls of Biting, shows the pattern:
winners block the opening 28 damage and clear the board in one or two
turns (110 of 262 fights), where the policy chips at one scroll.
Survivorship inflates the winners' side: their worst fights are in runs
that did not win.

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
