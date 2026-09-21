# Code survey: what the game's structure means for the Rust sim

Written 2026-09-21 against v0.107.1. Detail lives in the four reports:

- [cards.md](cards.md): card model, effects, costs, targeting, pools
- [combat-loop.md](combat-loop.md): turn order, action system, damage and block math, hooks, RNG, multiplayer
- [monsters.md](monsters.md): monster AI, act 1 encounters, ascension table, HP rolls
- [powers-relics-potions-run.md](powers-relics-potions-run.md): powers, relics, potions, run state, enchantments, modding

This file is the short version plus the decisions the survey forces.

## How the game is built

**One class per thing, canonical plus mutable clone.** `CardModel`, `PowerModel`, `RelicModel`, `PotionModel`, `MonsterModel` all derive from `AbstractModel`. A singleton canonical instance holds the definition; `ToMutable()` clones it per run or per combat. Cards in the deck are cloned again into the draw pile at combat start, so combat mutations never touch the deck.

**Effects are direct async calls, not a queue.** A card's `OnPlay` awaits `DamageCmd`, `PowerCmd`, `CreatureCmd`, and so on. Nested effects resolve depth first in source order via the call stack. The only queue is one level up, for player inputs (play card, use potion, end turn). Player choices mid-effect (Armaments, Headbutt) suspend the async chain via `PlayerChoiceContext`.

**Everything listens to everything.** `AbstractModel` has about 185 virtual hook methods. `Hook.cs` dispatches each to every listener in a fixed order: allies then enemies; per creature its powers first, then the monster model or, for the player, relics in pickup order, potions, orbs, then every card in every pile. Value hooks compose as one full additive pass, then one full multiplicative pass, then a cap pass.

**Numbers are decimal, truncated on write.** Damage is `(base + sum of additives) * product of multipliers`, kept as C# `decimal`, truncated to int once when subtracted from block and again from HP. Strength 3, base 6, Vulnerable, Weak gives 10.125, so 10.

**Durations tick at end of the enemy turn.** Vulnerable, Weak, Frail, Intangible decrement in `AfterSideTurnEnd` when the side is Enemy. A debuff a monster applies to the player skips its first tick.

**Monster AI is a state graph.** `MoveState` nodes with fixed follow-ups, `RandomBranchState` with weights and repeat or cooldown limits, `ConditionalBranchState` with predicates. Most act 1 monsters are deterministic loops. Only eight act 1 monster types consume the `MonsterAi` RNG stream.

**RNG is 12 named xoshiro streams.** Shuffle, MonsterAi, CombatTargets, Niche (monster HP), and others. Draw itself uses no RNG, it takes the top card. Shuffle sorts the pile by a stable key first, then Fisher-Yates.

**Multiplayer is everywhere but shallow.** Player lists, per-player turn numbers, two-phase turn end, scaling multipliers. All of it collapses at player count 1.

**Ascension is a `>=` ladder.** Every check is `HasLevel(x)` meaning level at least x. Act 1 at A10 is: 8 elites on the map, Neow heal at 0.8, gold at 0.75, one fewer potion slot, Ascender's Bane, worse card rarity odds, and per-monster HP and damage bumps from levels 8 and 9. Level 10's double boss only affects the last act.

## Scope numbers for the act 1 slice

| What | Count | Note |
|---|---|---|
| Ironclad cards | 87 | plus 64 colorless, 18 curses, 12 statuses |
| Ironclad relic pool | 8 | shared pool holds the rest of the 298 |
| Potions | 64 total | 3 Ironclad-specific |
| Act 1 monster types | 29 | including summons |
| Act 1 encounters | 22 | 4 weak, 12 normal, 3 elite, 3 boss |
| Powers in the game | 260 | act 1 plus Ironclad uses a small subset |

## Decisions the survey forces

**1. Effect queue with front insertion, not recursion.** The game uses recursive async calls, which we can't suspend in Rust without async machinery. An explicit queue where nested effects push to the front reproduces depth-first ordering exactly and gives suspension for free: a choice effect returns `Pending(choices)` from `step`, the rest of the card's effects wait in the queue, and the next `step` supplies the answer. This is the one place the sim's structure deliberately differs from the game's, and every port comment should note where a card's choice point lands.

**2. Listener order is data.** Hooks iterate creatures in side order, and per creature in the order above. The sim keeps powers, relics, and piles in insertion-ordered vectors and dispatches in that same order. No hash maps on the hot path.

**3. Numbers as f64, truncated at the same two write points.** Every multiplier in the act 1 and Ironclad scope is a dyadic rational (1.5, 0.75, 1.75, 0.5) so f64 products of integers are exact. Audit that claim when adding a multiplier. If a non-dyadic one appears, switch to a fixed-point type.

**4. Monster move graphs are ported verbatim.** Each monster is a small struct holding its state graph, current state, state log, and the encounter flags (`IsFront`, `IsAlone`, `MiddleInklet`). The graph walker is shared code.

**5. Own RNG, but keep the stream split.** The sim has separate streams for shuffle, monster AI, targets, and HP rolls so that replay injection can override one without disturbing the others. Injection replaces the stream with a recorded sequence.

**6. Run state is a separate struct from combat state.** The game's `Player` holds deck, relics, potions, gold, HP across the run; `PlayerCombatState` holds piles and energy. Combat clones deck cards in and writes only HP and relic counters back. Mirror that split.

**7. Player count is 1, hard-coded.** No player lists. The three scaling call sites become no-ops.

**8. Stars, orbs, Osty, enchantments are out of scope for the Ironclad slice.** Represent the cost path so it doesn't need a rewrite later, but don't implement them.

## What this changes about the bridge mod

The recording bridge doesn't need Harmony. `ModHelper.SubscribeForCombatStateHooks` registers a custom `AbstractModel` that receives all 185 hooks, and `CombatManager`, `Creature`, and `Player` expose plain C# events for state changes. That is enough to log every transition.

The dev console (`fight`, `card`, `relic`, `potion`, `applypower`, `energy`, `draw`) can construct an arbitrary combat setup in the real game. That means replay tests don't depend on what a run happens to produce. We can script a setup, record the fight, and diff it against the sim.

## Checked after the survey

- `fight <encounterId>` jumps the current run to any encounter, so it inherits the run's ascension. It randomizes the encounter's own RNG, which only affects monster composition for the few encounters that roll it.
- `unlock all` in the dev console marks every epoch as discovered. Six of the eight Ironclad relics are epoch-gated (Ironclad3Epoch: RedSkull, PaperPhrog, RuinedHelmet; Ironclad6Epoch: SelfFormingClay, CharonsAshes, DemonTongue), so run that before recording.
- Draw is a per-card loop. Each iteration reshuffles the discard into the draw pile if the draw pile is empty and the discard is not, then takes the top card. It stops at 10 cards in hand. Requested count is rounded up. The injection format is one shuffle order per reshuffle event, in order.
