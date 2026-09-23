# The run environment

Status: step 1 built (the effect layer, direct fights, a forward run);
steps 2 to 5 are design. The run layer under it (`game_rng.rs`, `map.rs`,
`plan.rs`, `run.rs`, `rewards.rs`, `shop.rs`, `pools.rs`, `events.rs`)
replays real runs floor for floor; `effects.rs`, `rooms.rs` and
`forward.rs` make it something a policy plays.

## What it is for

A whole-run policy: paths, card picks, shops, rest sites, events, Neow. The
combat policy we already train plays the fights. A run is won by beating the
last act's bosses (two at A10; 49 floors).

## The effect layer

`effects.rs` is what rooms and relics do to the player between fights:
gold (Bowler Hat, Ectoplasm, Dragon Fruit), HP and max HP, cards joining
the deck with the hooks on them (the eggs, Fresnel Lens, Lucky Fysh),
potions into slots (three, two at Tight Belt), each relic's pickup, the
rest site options (heal with Regal Pillow and Stone Humidifier, smith,
Lift, Dig, Kindle, Cook), and what entering a room does (an ancient's heal,
Meal Ticket, Eternal Feather, Planisphere, Maw Bank). Nothing there asks
the player: what needs a choice comes back as an `Offered` (cards to take,
relic rewards, cards to pick from the deck), and a pickup the port lacks
comes back as `Offered::Unported`, with the relic held and its effect
skipped.

`rooms.rs` lays each room out and puts every choice to a `Chooser`: the
rewards screen, a treasure chest, a rest site, an ancient's relic, and
what pickups offer on the way. `RunState` holds what a fight needs as the
fight needs it: relics as `RunRelic` with the sim's `counter` and `flag`
(Lasting Candy, Silver Crucible and Lava Rock keep their run counts there
too), potion slots, enchantments with their amount, and the map point, so
`enter(map, point)` works the shop blacklist out itself.

Rooms whose options draw while they are laid out split into the draws and
the choice: `event_offer` then `event_option` (Wongo's featured item, the
Relic Trader's relics, the Fake Merchant's prices), `rest_options` then
`rest`, `ancient_options` then the relic taken in `rooms::ancient`.

Every pool relic's pickup is ported, apart from those whose pickup or
hooks draw on the Rewards stream in ways not ported (`UNPORTED_RELICS`:
Cauldron, Orrery, Calling Bell, Toy Box and others). Of Neow's and the
ancients' relics these are not: Leafy Poultice and New Leaf (transforms),
Nutritious Soup, Pandora's Box, Beautiful Bracelet, Touch of Orobas,
Pael's Claw, Growth, Horn and Legion, Archaic Tooth, Astrolabe, Tanx's
Whistle, Tri-Boomerang, Storybook, Signet Ring, Preserved Fog, Jewelry
Box, Alchemical Coffer, Claws, Fur Coat, Fragrant Mushroom, Golden
Compass, Byrdpip, Dusty Tome.

Still not here, and step 4: shops' prices, buying and card removal (a
forward run stocks the shop, which draws, and leaves); event effects (a
forward run lays the options out, which draws, and leaves); Neow's and
the ancients' options, drawn on the event's own stream (a forward run
takes the heal and leaves).

### Checked against real runs

`history::check(run, live)` walks a real run through the same room flows
with the player's recorded choices as the `Chooser`, so the checker and
the forward run share them. It compares what the flows draw with the
record (unchanged: 395 of 462 floors on the Rewards stream, rooms in all
23 runs). With `live` it also compares the player the effects leave (HP,
max HP, gold, deck, relics, potions) with each floor's `player_stats`,
and sets the player back to the record after a floor it does not
compare. Fights are not simulated there: what a fight did (gold stolen,
potions used, Petrified Toad's rock, a card a thief took and gave back)
comes from the record, and so do HP and max HP after a fight, since the
record does not split the fight's healing from the rewards'.
`runcheck --effects` prints it. On the modded profile: 348 floors
compared (207 fights, 63 rest sites, 27 ancients, 27 treasure rooms, 24
shops with their purchases read off the record), none differ. Not
compared: 21 fights that ended a run, 55 events, 10 floors that picked
up a relic whose pickup is not ported (Leafy Poultice, New Leaf,
Nutritious Soup, Beautiful Bracelet, Pandora's Box, Dingy Rug), and 28
floors of one run whose draws follow a Rewards stream already lost.

What no compared floor exercises yet: Whetstone, War Paint and Sand
Castle's upgrades, which shuffle on the Niche stream.

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
| Rewards | the relics first, one at a time in the order chosen, or leave them (a relic taken first works on the open card reward: the eggs); then keep or leave the potion; then one of the offered cards or skip. Gold is automatic |
| Shop | buy a card, relic or potion (with its price), remove a card, leave |
| Rest | heal, smith, relic options (Dig, Lift, Kindle, Cook); more than one with Miniature Tent |
| Event | the event's options, then any card pick it opens |
| Treasure | take the relic or leave it |
| Deck pick | a card from the deck (upgrade, remove, transform, enchant) |
| Done | terminal |

`rooms::Decision` is this table in code, and a `Chooser` answers it with
an index; `rooms::First` takes the first option every time. Unsupported
cards (Splash, Mad Science: `UNSUPPORTED_CARDS`) are masked out of card
and bundle offers before the chooser sees them (`forward.rs`), and will
be out of purchases.

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

- `RunState::fight_setup` builds the fight directly from the run state:
  game ids to sim ids (`pools::sim_card` and friends) into a `FightSetup`,
  relics the sim leaves out (`gen::INERT_RELICS`) dropped. Not through the
  recorder's `start` format: `RunParts::of` carries a recorder fix that
  deletes the first Potion-Shaped Rock when Petrified Toad is held, and in
  forward play that rock is real. A test builds recorded starts both ways.
- `RunState` holds what a fight needs as the fight needs it: relics with
  their persistent `counter` and `flag` (Ember Tea and Pumpkin Candle
  charges live there), potion slots with their count (four with Potion
  Belt), enchantments with their amount.
- `RunState::end_fight` writes back HP, max HP, gold (thieves, Hand of
  Greed, gold relics), the potion slots and each relic's counter and flag,
  and returns the gold reward's proportion from the monsters that escaped.
  The sim never changes the master deck, and post-victory heals (Burning
  Blood) already land on the fight's HP.
- The forward run rolls a fight's enemies from the run's seed and the
  floor, and hands the setup to a `forward::Fights`: `stub_fight` (won at
  70% HP) or a closure; step 2 plugs the combat sim in there.
- Combat uses the combat sim's own RNG. The run streams the game also draws
  in combat are not advanced.

### Exactness

The run rolls what the game would roll for the same choices until the first
unported relic effect or event draw. From there the Rewards stream diverges
from the game's for the rest of that run (as `history.rs` models with
`rewards_live`). Fine for training; the env counts it per run.

### Unported content

An event whose effects are not ported is entered and left with no effect
(after what laying its options out draws, `event_offer`), counted per
event; so are shops and the ancients' options. A relic whose pickup is not
ported returns an `Unported` marker, not an error, and is counted. The
counts, weighted by how often a thing comes up and whether it draws on the
Rewards stream, say what to port next. `examples/forward.rs` prints them:
over 200 seeds taking the first option, every run reaches floor 49 in
about 41 ms (the three maps are most of it), and what it meets unported is
shops, the ancients' options and events, and The Courier and White Star
among the relics.

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

1. Done. Plumbing, no policy: the effect layer for what step 2 needs (relic
   pickups, rest heal and smith, ancient heal, reward taking with the
   checker's logic moved into shared room flows, map position, offer/take
   split), the direct `RunState` to `FightSetup` conversion and the
   writeback. A Rust test drives a seeded run to floor 49 with a stub fight
   result (won, HP x 0.7) and a fixed choice rule, deterministically.
   `runcheck` still matches the real runs it matched, and `runcheck
   --effects` checks the effects against them.
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
