# The run environment

Status: design, not built. The run layer under it (`game_rng.rs`, `map.rs`,
`plan.rs`, `run.rs`, `rewards.rs`, `shop.rs`, `pools.rs`, `events.rs`)
replays real runs floor for floor; this turns it into something a policy
plays.

## What it is for

A whole-run policy: paths, card picks, shops, rest sites, events, Neow. The
combat policy we already train plays the fights. A run is won by beating the
last act's bosses (two at A10; 49 floors).

## What the run layer does not do yet

It is stream-exact and effect-free: it rolls what the game rolls, and
`history.rs` then overwrites HP, gold, deck and potions from the recording
after every floor. Forward play needs the effect layer, which exists nowhere
yet:

- relic pickup effects (`AfterObtained`: Strawberry, Old Coin, Potion Belt,
  Whetstone, War Paint, the `UPON_PICKUP` list in `run.rs`, Neow's), not
  just the stream draws `obtain()` ports for six relics;
- rest sites: the 30% heal, Regal Pillow, smith; ancients' heal on entry
  (`AncientEventModel`, 80% of missing HP at A2+);
- taking rewards: gold, potions into slots, the card, the relic, and the
  reward-screen logic that now lives in the checker (Lasting Candy's
  count, eggs upgrading the open card reward when a relic is taken first,
  relic pickup order, Lucky Fysh gold, Petrified Toad's rock) moved into
  `RunState` so the history check and the env share it;
- shops: prices (drawn and thrown away today), base costs, card removal
  and its rising price, restocking;
- event effects: `events.rs` ports the Rewards-stream draws of about 14
  events, and 33 more are known to draw nothing. No event's effect (HP,
  gold, curses, transforms) is ported;
- the map position (`PointId`), which makes the shop blacklist a direct
  check instead of the checker's path inference.

The room functions take the recorded choice and return what was drawn
(`event_option`, `rest_option`, `ancient_options`). Forward play needs the
options before the choice, and some draws happen while options are laid
out (Wongo's featured item, Darv's tome), so each splits into `offer()`
(the draws) and `take(choice)` (the effects), the shape `combat_rewards()`
already has.

## Numbers that shape it

- A run is about 1500 combat decisions and about 100 run decisions. Combat
  is where the compute goes, so fights from many runs batch on the GPU as
  `VecEnv` batches them now, and the combat batch should stay full.
- A run's 1500 combat steps cost 60 to 90 ms of CPU at 130k to 200k steps/s;
  three maps at 15 ms each are 45 ms, comparable. Generate an act's map on
  arrival (most early runs die in act 1); cache maps by seed if it shows.
- Combat plays greedy in the first cut. A search per decision is 10x to 100x
  and would turn ~130 runs/s into a few.

## Shape

### A fight source in `VecEnv`, not a new env

`VecEnv` already has two fight sources (generated and fixed). A run is a
third: a slot that owns a `RunState`, the act's `ActMap` and its position.
In `Slot::reset` it writes the finished fight back, advances the run to its
next fight, and builds the `FightSetup`. `observe`, `step`, `fork`, `fight`
and `combat(i)` carry over unchanged, so the combat policy, search and PPO
code keep working.

Between fights the run waits at run decisions. `VecEnv` gains
`step_run(indices, actions)`, which does no combat work: Python calls it in
a loop until no env sits at a run decision, then makes one combat step for
the whole batch. A reward screen is several run decisions in a row (relic,
potion, card, map step, rest, which card to smith), and none of them costs
an idle combat step or changes the combat batch's shape under
`torch.compile`.

### Run decisions

Every run decision is "pick one of these". Sub-decisions (which card to
smith, which card to remove) are separate decisions, not a product action
space.

| Phase | Options |
|---|---|
| Ancient (Neow, act ancients) | the relic and boon options; heal on entry |
| Map | the current point's children |
| Rewards | relics are taken first, in the game's order (eggs then upgrade the open card reward); then the potion if a slot is free; then one of the offered cards or skip. Gold is automatic |
| Shop | buy a card, relic or potion (with its price), remove a card, leave |
| Rest | heal, smith, relic options (Dig, Lift, ...) |
| Event | the event's options, then any card pick it opens |
| Treasure | none (take the relic) |
| Deck pick | a card from the deck (upgrade, remove, transform, enchant) |
| Done | terminal |

Unsupported cards (Splash, Mad Science: `UNSUPPORTED_CARDS`) are masked out
of offers and purchases.

### Observation and scoring

Tokens, shared with the token combat model and the deck-value network
(`sts2ai.deckvalue` already encodes deck, relics and potions this way):

- the run: a token per deck card (id, upgraded, enchantment and amount),
  relic (id, counter), potion; a global token (HP, max HP, gold, floor, act,
  ascension, the act's boss, the phase, the four `UnknownOdds`);
- afterstate scoring for the options whose outcome is a known state (take
  this card, buy this, smith that, remove that, skip): the run encoder
  values each resulting state and the policy is a softmax over those
  values. That is DESIGN.md's "argmax over offered options by value", and
  it needs no option tokens;
- option tokens only where the outcome is not known: map steps (each
  carrying the min and max count of every room type on paths through the
  point, and the distance to the next rest and shop) and event options.

New closed vocabularies (option kinds, events, room types, encounters as
the act's boss) go into `vocab.txt` under the append-only rule, like every
other id, so run checkpoints stay remappable.

### Fights

- Built directly from the run state: game ids to sim ids (`pools::sim_card`
  and friends) into a `FightSetup`. Not through the recorder's `start`
  format: `RunParts::of` carries a recorder fix that deletes the first
  Potion-Shaped Rock when Petrified Toad is held, and in forward play that
  rock is real.
- `RunState` holds what a fight needs as the fight needs it: relics with
  their persistent `counter` and `flag` (Ember Tea and Pumpkin Candle
  charges live there), potion slots with their count (four with Potion
  Belt), enchantments with their amount.
- Writeback after a fight: HP, max HP, gold (thieves, Hand of Greed, gold
  relics), potions, the relics whole, and which monsters escaped (the gold
  reward's proportion). The sim never changes the master deck, and
  post-victory heals (Burning Blood) already land on the fight's HP.
- Combat uses the combat sim's own RNG. The run streams the game also draws
  in combat are not advanced.

### Exactness

The run rolls what the game would roll for the same choices until the first
unported relic effect or event draw. From there the Rewards stream diverges
from the game's for the rest of that run (as `history.rs` models with
`rewards_live`). Fine for training; the env counts it per run.

### Unported content

An event whose effects are not ported is entered and left with no effect,
counted per event. A relic whose pickup is not ported returns an `Unported`
marker, not an error, and is counted. The counts, weighted by how often a
thing comes up and whether it draws on the Rewards stream, say what to port
next.

### Rewards and value

- Run policy: +1 for a win, otherwise floors cleared / 49 - 1. Run decisions
  are stored per env, not in the combat `Rollout`, and paid at run end
  (gamma 1) or bootstrapped from the run value head.
- Combat policy, first cut: frozen, with its per-fight reward. That reward
  (a win, plus HP and potions at hand-set weights) optimizes the single
  fight.
- Combat policy, next: the fight's reward becomes V_run(after) -
  V_run(before), the run value head's change. HP is then worth what the
  rest of the run pays for it (more before an elite, less before a rest),
  a potion is worth keeping for the boss next floor, and max HP from Feed
  or upgrades from Lesson Learned count. The combat observation gains run
  context (act, floor, the upcoming boss, distance to the next rest), and
  combat fine-tunes inside runs, on decks the run policy builds. This is
  the shared value function DESIGN.md keeps run and combat state apart
  for.

## Build order

1. Plumbing, no policy: the effect layer for what step 2 needs (relic
   pickups, rest heal and smith, ancient heal, reward taking with the
   checker's logic moved into `RunState`, map position, offer/take split),
   the direct `RunState` to `FightSetup` conversion and the writeback. A
   Rust test drives a seeded run to floor 49 with a stub fight result (won,
   HP x 0.7) and a fixed choice rule, deterministically. `runcheck` still
   matches the real runs it matches today.
2. The loop: the `VecEnv` run source, with run decisions made in Rust by a
   random policy, played from Python with the current combat checkpoint
   through the unchanged `Envs` wrapper (episode ends gain the run floor and
   whether the run ended). Floors reached, deck size and throughput are then
   real numbers.
3. Run decisions exposed to Python (`step_run`, token rows), the run policy
   with afterstate scoring, PPO over run decisions.
4. Shops (prices, removal), Neow and the act ancients' options, events by
   frequency.
5. The run value as the combat reward; combat fine-tuned inside runs.

## Open questions

- The combat policy was trained on generated decks; runs from a weak run
  policy make odd decks. Watch combat win rates by act.
- Whether map options need the full map as tokens or the per-option summary
  is enough.
