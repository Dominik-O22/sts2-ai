# The run environment

Status: steps 1 to 3 built (the effect layer, direct fights, a forward
run; runs as a `VecEnv` fight source; run decisions exposed to Python as
tokens and a run policy trained on them with PPO); of step 4, Neow, the
act ancients, shops and all events but three are built; step 5 is
design. The run layer under it (`game_rng.rs`, `map.rs`,
`plan.rs`, `run.rs`, `rewards.rs`, `shop.rs`, `pools.rs`, `events.rs`,
`ancients.rs`) replays real runs floor for floor; `effects.rs`,
`rooms.rs` and `forward.rs` make it something a policy plays.

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
rewards screen, a treasure chest, a rest site, an ancient's options, a
shop, and what pickups offer on the way. `RunState` holds what a fight needs as the
fight needs it: relics as `RunRelic` with the sim's `counter` and `flag`
(Lasting Candy, Silver Crucible and Lava Rock keep their run counts there
too), potion slots, enchantments with their amount, and the map point, so
`enter(map, point)` works the shop blacklist out itself.

Rooms whose options draw while they are laid out split into the draws and
the choice: `rest_options` then `rest`, `ancient_offer` then the relic
taken in `rooms::ancient`, `shop` then what `rooms::shop_room` buys. An
event does both in one flow (`RunState::event`, below).

Ancients (`ancients.rs`) lay their options out on the event's own stream,
seeded from the run's seed and the ancient's id, with the conditions each
reads off the deck and relics (Pael's Claw needs three cards Goopy takes,
Tri-Boomerang three Instinct takes, and so on). Every option is a relic,
so taking one is `obtain`. Darv's Dusty Tome readies its card on the
Rewards stream as it is laid out, and taking the tome adds it.

A shop (`shop.rs`) keeps each entry's price as rolled on the Shops
stream. The chooser buys one ware at a time (`Decision::Shop`, only what
it can pay for and take) and leaves past the end. The card removal asks
which card and costs 75, 25 more per removal bought this run (100 and 50
at Inflation); `RunState.shop_removals` counts them. The Courier restocks
an entry bought and takes a fifth off, Membership Card halves, Lord's
Parasol buys everything on entry, Maw Bank stops paying on a purchase, and
an egg bought upgrades the cards left on the shelf.

Transforms draw the new card from the original's pool, on the stream the
relic names: Leafy Poultice on Transformations, New Leaf, Astrolabe and
Pandora's Box on Niche. A curse or status is never offered for a
transform, since the port lacks their pools. A transform can still roll a
card the sim cannot play (`UNSUPPORTED_CARDS`), and the forward run then
ends stuck at its next fight.

Every pool relic's pickup is ported, apart from those whose pickup or
hooks draw on the Rewards stream in ways not ported (`UNPORTED_RELICS`:
Cauldron, Orrery, Calling Bell, Toy Box and others). Of Neow's and the
ancients' relics, these are not: Kaleidoscope (it offers other
characters' cards, and the port has only the Ironclad's and colorless
pools), Golden Compass and Fur Coat (they change the act's map), Glass
Eye, Driftwood, Sea Glass, Prismatic Gem, Pael's Wing and Tooth, Toy Box,
Black Star, Calling Bell, Delicate Frond and Glitter (`UNPORTED_RELICS`).
The event relic Byrdpip is not either.

### Events

`events.rs` has one function per event, written as the game's
`Models/Events/<Name>.cs` is, over the effect layer's commands (`damage`,
`gain_gold`, `lose_max_hp`, `add_card`, `obtain`, `transform`) and a few of
its own on `Ev`: `ask` lays a page out and puts it to the chooser as
`Decision::Event` (the open options only, each by its page and key, with
the potion, relic or card the layout drew for it), `pick` and the
enchant, upgrade, remove and transform helpers put deck picks, `grid` a
card grid. `EVENTS` maps class names to those functions; an event not in
it is entered and left. The event's own stream is seeded from the run's
seed and the hash of its id, as the ancients' is. The relics'
`AfterRoomEntered` (Maw Bank, Planisphere) runs once the first page is
laid out, as `EventRoom.EnterInternal` does, so a layout that reads gold
or HP reads it before them.

An event that starts a fight returns it (`EventFight`): the encounter,
the rewards it adds (`CombatRoom.ExtraRewards`: Punch Off's relic and
potion, the Fake Merchant's rug and shelf, the Lantern Key card), the
gold an encounter fixes, and Battleworn Dummy's setting, whose fight gives
no rewards and whose event pays out once the dummy is beaten
(`event_fight_won`). The run hands the fight out like a combat room's.
Punch Off and The Lantern Key lay out as a combat, which creates their
monsters on the run's Niche stream as the room is entered, fight or not.

What the events need of the effect layer is there too: damage out of
combat goes through Tungsten Rod, and a death out of combat is prevented
by a Fairy in a Bottle, then an unused Lizard Tail; a card joining the
deck triggers Darkstone Periapt, Bing Bong and Book of Five Rings;
Fragrant Mushroom's pickup and Dream Catcher's card reward at a rest heal
are ported. The combat sim has the Foul Potion and the Glowwater Potion
the events hand out.

Not ported: Tinker Time (its Mad Science carries a type and rider the deck
card has no field for, and the sim cannot play it), Colorful Philosophers
(the other characters' card pools) and Crystal Sphere (a grid minigame).
Where the port differs on purpose:

- The Fake Merchant lays out no options in the game, only a shop and a
  Foul Potion to throw. The port asks first whether to throw it (`THROW`
  or `SHOP`, no page), then sells the shelf through `Decision::Shop`.
- Welcome to Wongo's never gives the Customer Appreciation Badge, which
  counts Wongo points across the profile's runs.
- Trial's Double Down abandons the run; the port ends it as a death.
- A Spoils Map (The Legends Were True) pays its 600 gold and leaves the
  deck at the second act's first treasure room. The game generates that
  act's map as an hourglass through a single treasure (`SpoilsActMap`),
  which the port does not: it keeps the act's usual map.
- A shop does not take a Foul Potion for 100 gold, which the game allows.

### Checked against the game's code

`tools/oracle` runs the game's own code outside the game, and an example
per command diffs it with the port over many seeds:

- `ancients` (`examples/ancientcheck.rs`): every ancient's options, in
  the acts it can meet, with decks that turn each condition on and off.
  3000 of 3000 match.
- `obtain` (`examples/pickupcheck.rs`): `RelicCmd.Obtain` for every
  ported pickup but Sere Talon and Neow's Bones (their pickups crash the
  oracle outside the game), alone and mixed with the relics that change
  them (the eggs, Fresnel Lens, Lucky Fysh, Bowler Hat, Sozu, Silver
  Crucible), and Fishing Rod and War Hammer over fights won. The oracle's
  selector takes from the front, as `rooms::First` does; the player left
  and the Rewards, Niche, Transformations and CombatPotionGeneration
  counters match in 1999 of 1999 (one more the game itself refuses).
- `rewards` (`examples/rewardcheck.rs`): walks through three acts of
  fights, shops and unknown rooms, now with each shop's prices, removals
  bought and The Courier's restocks. 1000 of 1000 match.
- `events` (`examples/eventcheck.rs`): every ported event from
  `BeginEvent` on, in any act, with decks, gold, HP, relics (Tungsten Rod,
  the eggs, Bowler Hat, Lucky Fysh, Silver Crucible) and potions that turn
  each option's condition on and off, walked through its pages by a
  scripted choice per page, deck picks from the front. It prints each page
  laid out, the fight an option starts, the player left and the Rewards,
  Niche, Transformations, CombatPotionGeneration and event stream
  counters. The oracle stubs out what needs a running game (Godot nodes,
  hover tip icons, the reward synchronizer) and cannot finish a death or
  Trial's Double Down; those runs are left out. 4949 of 4949 match.

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
`runcheck --effects` prints it. On the modded profile as of 2026-09-24
(25 runs, 528 floors): 496 floors compared, none differ, the 60 event
floors among them; the Rewards stream is followed through 496.
Not compared: 23 fights that ended a run, 5 floors whose draws follow a
Rewards stream already lost, 2 that drew on a Niche stream already lost,
Dingy Rug's pickup and one Crystal Sphere. Before the events, the same
files gave 411 floors compared, none differing, 61 event floors not
compared, and the Rewards stream followed through 455.

The history check walks an event with the record's `event_choices` as
the chooser's answers, by page and key; a page with one option that the
record does not keep (`ThatWontSaveToChoiceHistory`) takes it. Three
limits of the record shape it. The record does not say which copy of a
card an upgrade went to, nor the deck's order, which an event's random
pick reads, so after a floor it does not compare the check keeps the
port's deck if it holds the same cards. The events that throw a potion
away themselves (Ranwid, The Future of Potions, Stone of All Time, the
Fake Merchant) pick it off the potions held, so the record's discards on
those floors wait for the event. And a chest's relic left behind is not
recorded at all.

runcheck also compares every ancient's options with the record's apart
from the effects, the dev console's runs included up to their first
console fight: 51 of 51 match.

Other players' runs come from ststracker.app: `scripts/tracker.py URL
OUT.run` turns a run page into this format, minus what the page drops
(potion offers, the ancients' unchosen options, transforms, enchantments
made on the way). Its first run, an A10 win (seed ZQY94FARS9), walks
from its seed with the rooms and the Rewards stream matching on all 49
floors; the effects differ only where the page is silent.

Fights draw on the run's Niche stream: `CombatState.CreateCreature`
rolls each enemy's max HP there. The run draws once per enemy a fight
starts with (`RunState::enemies_created`), which is what makes Whetstone
and Pandora's Box match after fights. Summons and Tough Egg's hatchlings
draw too, and the run layer does not see them, so the history check
stops comparing Niche draws after a fight whose monsters can summon.

## Numbers that shape it

- A run is about 1500 combat decisions and about 100 run decisions. Combat
  is where the compute goes, so fights from many runs batch on the GPU as
  `VecEnv` batches them now, and the combat batch should stay full.
- A run's 1500 combat steps cost 60 to 90 ms of CPU at 130k to 200k steps/s.
  An act's map costs about 0.4 ms (`ActMap::generate`): the port walks
  each path segment once where the game lists every path each pruning
  round, and `mapcheck` holds it to the game's maps (4800 of 4800). It was
  20 ms and 90% of run mode's env time, since most runs die in act 1 and
  need a fresh map; caching cannot help, as training seeds are fresh.
- Combat plays greedy in the first cut. A search per decision is 10x to 100x
  and would turn ~130 runs/s into a few.

## Shape

### A fight source in `VecEnv`, not a new env

`VecEnv` had two fight sources (generated and fixed). Runs are a third
(`VecEnv::set_runs`, `Envs.use_runs` in Python): a slot owns a
`forward::Run`, which holds the `RunState`, the act's `ActMap` and the map
point. `Run::next` plays the run to its next fight and hands the
`FightSetup` out, so `forward::play` is a loop over it with a `Fights`, and
a slot is the same loop spread over many steps. In `Slot::reset` the slot
writes the finished fight back (`end_fight`), calls `next` with how it
went, and starts the fight it gets. A run that ends starts a fresh one: a
slot's `k`-th run is seed index `base + slot + k * n`, game seed
`SIM<index>`. `observe`, `step`, `fork`, `fight` and `combat(i)` carry over
unchanged, so the combat policy, search and PPO code keep working.

`set_runs` takes who makes the run decisions (`RunChoices`, `use_runs(
choices=...)` in Python). `Random` is `rooms::Random` in Rust, on an RNG
seeded from the seed index, never the run's streams: uniform over the
options, a skip counting as one option for card and bundle rewards and
optional deck picks; relics are all taken, in random order, and potions
kept. `First` is `rooms::First`. `EpisodeEnd.run` (`End.run` in Python)
carries the run's seed index, act, floor, deck size and, on the fight that
ended the run, how it ended (won, died, or stuck on a fight the sim cannot
build). `End.floor` is the run's floor in run mode.

Under `Caller` the run waits at its decisions. `run_waiting` lists the
envs waiting, `observe_run` encodes their decisions (below), and
`step_run(envs, options)` answers them, playing each run on to its next
decision or fight without combat work; a fight reached starts and writes
its combat row, and a run that ends is returned. Python answers until no
env waits, then makes one combat step for the whole batch
(`sts2ai.runtrain.RunLoop`). A reward screen is several run decisions in a
row (relic, potion, card, map step, rest, which card to smith), and none
of them costs an idle combat step or changes the combat batch's shape. An
env that still waits sits a combat step out (reward 0, not done), so the
loop may also answer one round per step (`--no-drain`).

`Run::next` plays a whole segment between fights and cannot stop halfway:
the rooms call the chooser from deep inside their flows. So a `Caller`
slot keeps the run as it stood after the last fight, and each answer plays
the segment again on a copy with the answers so far (`env::Segment`). At
the first decision with no answer the chooser encodes it and unwinds
(`std::panic::resume_unwind`, which skips the panic hook), and the copy is
dropped; once the answers carry the copy to a fight or the run's end, the
copy is kept. A run with the same seed and choices plays the same way, so
the replays agree, and a test plays runs under random caller choices and
replays each through `forward::play` with the same answers and fights.
A segment of k decisions costs k replays, a few µs each apart from a new
act's map (0.4 ms), which the replays of the ancient that opens the act
repeat. A decision with one option or none is taken without asking.

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

`sim::runobs` encodes a decision as tokens, each segment a fixed number of
fixed-width tokens with a presence flag, so a batch is one array
(`run_layout()` in Python):

- a global token: the decision, the room the player is in, the act, the
  act's boss and second boss, the event or ancient being played; HP, max
  HP, gold, floor, ascension, the four `UnknownOdds`, the card rarity
  offset, the potion drop odds, deck size, potion slots and empty ones,
  card removals bought;
- a token per deck card (id, upgraded, enchantment and amount, 64 at
  most), relic (id, counter, flag, 40 at most) and potion held;
- a token per option (72 at most: a deck pick lists the deck, and a skip):
  its kind (`OptionKind`: take a card, skip, keep a potion, heal, smith,
  buy a relic, remove through a deck pick, leave the shop, and so on), the
  cards, enchantment, relic or potion it names, the price, and for a map
  step the point's type, the fewest and most of each point type on paths
  through it to the boss, and the rows to the nearest rest site and shop.
  An event option carries its event, page and key hashed into 1024
  buckets, which stays stable as events are ported, and the items its
  layout drew.

Ids index as the combat model and `sts2ai.deckvalue` do (sim id + 1), and
the relics the combat sim leaves out come after the sim's (`RUN_RELICS`).
The run's closed vocabularies (decisions, option kinds, rooms, acts,
events, bosses, those relics) are appended to `vocab.txt`; tests fail
when a ported event, a boss or an inert relic is missing from them.

Every option has a token in this first cut, including those whose
outcome is a known state (take this card, buy this, smith that). Scoring
those by the value of the state they lead to (afterstates: the run
encoder values each resulting state and the policy is a softmax over the
values, DESIGN.md's "argmax over offered options by value") can come
later; it needs the run value head to be good first.

The run policy (`sts2ai.runmodel`) is a small transformer over the tokens
(128 wide, 2 layers), a pointer head scoring each option token against
the global token, and a value head on the global token. Its card
embedding starts from the combat checkpoint's. A run checkpoint records
its arch, the run layout and the vocabulary, and refuses to load when
either moved: there is no remap for run checkpoints yet.

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
- Combat uses the combat sim's own RNG. Of the run streams the game also
  draws in combat, only the Niche stream moves, by one draw per enemy the
  fight starts with, whatever happens in it.

### Hidden information

The run layer is exact to the game, so a run is deterministic given its
seed and choices. The policies must not use that: they play on what a
skilled human could know.

- Allowed: what the game shows, and what a human could track from what they
  have seen and the public rules (card counting): the draw pile as a
  multiset, the unknown-room odds, the card rarity offset, the potion drop
  odds.
- Not allowed: the seed, stream positions, the run plan (upcoming
  encounters, events, the relic bags' order), the true draw order, or any
  lookahead that clones the exact streams.
- How a fight goes never moves the game's run streams (the Niche draws a
  fight makes depend on its encounter alone), so combat play cannot steer
  a run's rolls. Training seeds are fresh every run.
- Any search (combat's turn search, a future run-level one) resamples: the
  combat `Forks` already reshuffle the draw pile and roll their own dice.
- The run observation has a test
  (`runobs::tests::hidden_information_does_not_reach_the_encoding`): two
  runs with different seeds, streams and plans but the same visible state
  encode identically at a map step, a card reward, a relic reward and a
  deck pick.

The exact streams are for checking the port against real runs.

### Exactness

The run rolls what the game would roll for the same choices until the first
unported relic effect or event draw. From there the Rewards stream diverges
from the game's for the rest of that run (as `history.rs` models with
`rewards_live`). Fine for training; the env counts it per run.

### Unported content

An event that is not ported is entered and left with no effect, counted
per event. A relic whose pickup is not ported returns an `Unported` marker,
not an error, and is counted. The counts, weighted by how often a thing
comes up and whether it draws on the Rewards stream, say what to port
next. `examples/forward.rs` prints them. Over 200 seeds taking the first option,
which now also lingers in the Abyssal Baths and deciphers the Tablet of
Truth until they kill, 176 runs reach floor 49, 23 die on the way and
one is stuck on a Splash an event's transform rolled. The events they
meet unported are Tinker Time (22 times), Colorful Philosophers (18) and
Crystal Sphere (16), 56 visits over 200 runs where all 1186 were before;
then Kaleidoscope (18), Sea Glass (12), Glass Eye (9) and a handful of
other ancient relics.

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

### Potions need the run value

The combat reward prices every potion alike (`POTION_VALUE`), which is
wrong both ways: a Fairy in a Bottle kept for a boss is worth more than a
Block Potion, and Delicate Frond's refills are random, so drinking a strong
potion under it is a loss the flat price calls free. Rarity tiers are no
fix; potions are too situational. The run value is: V(run with this
potion) against V(run with a random refill). The act 3 bosses at A10 need
it too: HP and potions after the first carry into the second, and after
the second they are worth nothing.

## Build order

1. Done. Plumbing, no policy: the effect layer for what step 2 needs (relic
   pickups, rest heal and smith, ancient heal, reward taking with the
   checker's logic moved into shared room flows, map position, offer/take
   split), the direct `RunState` to `FightSetup` conversion and the
   writeback. A Rust test drives a seeded run to floor 49 with a stub fight
   result (won, HP x 0.7) and a fixed choice rule, deterministically.
   `runcheck` still matches the real runs it matched, and `runcheck
   --effects` checks the effects against them.
2. Done. The loop: the `VecEnv` run source, with run decisions made in
   Rust by a random policy, played from Python with the current combat
   checkpoint through the unchanged `Envs` wrapper. A Rust test plays runs
   through the env with the first legal action and replays each with
   `forward::play` and the same fights: the same setup at every fight and
   the same end. `sts2ai.runplay` reports the numbers. set-12 greedy, 256
   envs, each env's first 8 runs (2048), 3 rayon threads beside a training
   run:
   - Every run died, 99% in act 1: floor 10.5 on average, median 9, p90 17
     (the act 1 boss), best 30. Deck 15 cards at the last fight.
   - Combat win rate in act 1: weak 100%, normal 85%, elite 39% (1959
     fights), boss 9% (256). Six act 1 elites end most runs.
   - The deck is not the main reason. Run fights start at about half HP
     (elites at 48% on average), since random paths walk into elites and
     random rest sites smith half the time; elites entered above 70% HP are
     won about 70% of the time, near the 78% of generated elites on floors 6
     to 10. The run also lacks Neow's options, shops and events, so the deck
     is the starter plus a few random picks.
   - 7.6k combat steps/s, 270k runs/hour. The env step is 90% of the wall
     time, and nearly all of that is making maps (Numbers that shape it).
     Without maps the rest of a run between fights costs about 6 µs a
     fight.
3. Done, but for afterstate scoring and a long training run. Run
   decisions exposed to Python (`step_run`, token rows), the run policy
   with option tokens for every decision, PPO over run decisions
   (`sts2ai.runtrain`). ab-attn greedy plays the fights; 256 envs, each
   env's first 4 runs (1024), on 2026-09-24:

   | Run decisions | floor mean (p10 / p90) | ended in act 1 / 2 / 3 | act 1 elite / boss won | act 2 elite / boss | deck |
   |---|---|---|---|---|---|
   | random (`rooms::Random`) | 12.1 (7 / 17) | 94% / 6% / 0% | 57% / 22% | 42% / 0% | 16.9 |
   | first option (`rooms::First`) | 14.0 (7 / 24) | 81% / 18% / 0% | 71% / 49% | 45% / 19% | 20.7 |
   | run policy, untrained, greedy | 11.1 (7 / 17) | 99% / 1% / 0% | 46% / 8% | 25% / - | 17.3 |
   | run policy after 90 s of PPO (64 envs), greedy | 22.0 (13 / 33) | 49% / 49% / 2% | 91% / 59% | 59% / 14% | 24.1 |

   No run won. Three of 1024 ended stuck on a Splash a transform rolled.
   The 90-second policy already heals at every rest site, never walks
   into an elite, takes every card offered, keeps 93% of potions and
   rarely leaves a shop without buying; most of its gain is HP: act 1
   elites won 91% against 57% at random. A long training run is still to
   be done.

   Throughput, same machine, nothing else training (runs this short make
   the rates rough):

   | Envs | Run decisions | combat steps/s | run decisions/s | runs/hour |
   |---|---|---|---|---|
   | 256 | random, in Rust | 110k | - | 3.1M |
   | 256 | policy, drained before each combat step | 22k | 4.5k | 0.76M |
   | 256 | policy, one round per combat step | 60k | 10.5k | 1.8M |
   | 1024 | random, in Rust | 211k | - | 5.4M |
   | 1024 | policy, drained | 55k | 11.3k | 1.7M |
   | 1024 | policy, one round per step | 104k | 18.6k | 2.8M |

   Draining costs half the combat rate or more: each drain round is a
   run-policy forward on a small batch, and a reward screen, shop or deck
   pick chains several. Answering one round per step leaves about one env
   in twenty idle for a combat step instead, and plays the same runs.
4. Shops (prices, removal), Neow and the act ancients' options: done.
   Events: done but Tinker Time, Colorful Philosophers and Crystal
   Sphere.
5. The run value as the combat reward; combat fine-tuned inside runs.

## Open questions

- The combat policy was trained on generated decks; runs from a weak run
  policy make odd decks. Watch combat win rates by act. With random run
  decisions the first answer is HP, not decks: fights start far lower
  than the generator's.
- Whether map options need the full map as tokens or the per-option summary
  is enough.
