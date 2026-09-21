# StS2 v0.107.1 — Act 1 (Overgrowth) monsters, encounters, ascension

Root: `decompiled/MegaCrit/Sts2/Core`

## 1. Monster model

### Base class — `Models/MonsterModel.cs`
Abstract, inherits `AbstractModel` (canonical singleton ↔ `ToMutable()` clone per combat instance). Key members:

- `public abstract int MinInitialHp { get; }` / `MaxInitialHp` — **properties, not fields**, so they re-evaluate ascension at access time.
- `protected abstract MonsterMoveStateMachine GenerateMoveStateMachine();` — built in `SetUpForCombat()`.
- `public MoveState NextMove { get; private set; }`
- `RollMove(targets)` → `NextMove = MoveStateMachine.RollMove(targets, Creature, RunRng.MonsterAi)`
- `PerformMove()` → `await move.PerformMove(combatState.PlayerCreatures)` then `MoveStateMachine.OnMovePerformed(move)`
- `SetMoveImmediate(state, forceTransition)` — used by stuns/mode-shifts; respects `NextMove.CanTransitionAway`.
- `Rng` (cosmetic/desync-safe, seeded per creature), `RunRng` (the shared `RunRngSet`).
- Hooks: `AfterAddedToRoom()` (where innate powers are applied), `BeforeRemovedFromRoom()`, `AfterDeath(...)`, `OnDieToDoom()`.
- `const string stunnedMoveId = "STUNNED"`.

### Move selection — `MonsterMoves/MonsterMoveStateMachine/`
It is a **graph of states**, not a weighted-bag-with-history like StS1. Three state kinds, all `MonsterState` with `GetNextState(owner, rng)`:

- `MoveState` — a real move. Holds `IReadOnlyList<AbstractIntent> Intents`, a `Func<IReadOnlyList<Creature>, Task> _onPerform`, and `FollowUpState`. `GetNextState` returns the follow-up unconditionally. `MustPerformOnceBeforeTransitioning` pins the monster (used by Ceremonial Beast's STUN_MOVE).
- `RandomBranchState` — weighted random with repeat/cooldown constraints. `ShouldAppearInLogs => false`.
- `ConditionalBranchState` — first branch whose `Func<bool>` is true wins (used by Nibbit front/back/alone).

`MonsterMoveStateMachine.FindNextMoveState` loops `GetNextState` until it lands on a state with `IsMove`, so branch states are transparent:

```csharp
if (!_currentState.CanTransitionAway || (!_performedFirstMove && _currentState.IsMove)) return;
do {
    string nextState = _currentState.GetNextState(owner, rng);
    ...
    SetCurrentState(string.IsNullOrEmpty(nextState) ? _initialState : States[nextState]);
} while (!_currentState.IsMove);
```
Note the second clause: **the initial move is never re-rolled before it is performed once**.

Repeat constraints (`MoveRepeatType` in `MonsterMoves/MoveRepeatType.cs`): `CanRepeatForever`, `CanRepeatXTimes`, `CannotRepeat`, `UseOnlyOnce`. `RandomBranchState.GetStateWeight` zeroes the weight if the last N entries of `StateLog` are all this state (N = 1 for `CannotRepeat`, `maxTimes` otherwise), or if the state appears at all in `StateLog` for `UseOnlyOnce`. A separate `cooldown` field zeroes the weight if the move appears in the last `cooldown` *move* log entries. Selection is linear scan over `rng.NextFloat(sumOfWeights)`.

### Intents — `MonsterMoves/Intents/`
`IntentType`: `Attack, Buff, Debuff, DebuffStrong, Defend, Escape, Heal, Hidden, Summon, Sleep, Stun, StatusCard, CardDebuff, DeathBlow, Unknown`. Classes: `SingleAttackIntent(damage)`, `MultiAttackIntent(damage, repeat)`, `BuffIntent`, `DebuffIntent(strong:bool)`, `CardDebuffIntent`, `DefendIntent`, `EscapeIntent`, `HealIntent`, `HiddenIntent`, `SleepIntent`, `StatusIntent(count)`, `StunIntent`, `SummonIntent`, `DeathBlowIntent`, `UnknownIntent`. A `MoveState` carries **multiple** intents (e.g. attack + defend + status), which is how compound intents are displayed. `AttackIntent.GetSingleDamage` runs the displayed number through `Hook.ModifyDamage(... ModifyDamageHookType.All ...)` from the *local player's* perspective, so intent numbers are post-Strength/Weak/Vulnerable.

### Monster A: `Models/Monsters/Nibbit.cs` (act-1 weak/normal)
HP 42–46 (A≥8: 44–48). Moves: `BUTT_MOVE` 12 dmg (A≥9: 13); `SLICE_MOVE` 6 dmg (A≥9: 7) + 5 block (A≥8: 6); `HISS_MOVE` +2 Strength (A≥9: 3). Entry is a `ConditionalBranchState("INIT_MOVE")`:

```csharp
if (_isAlone) conditional.AddState(butt, () => ((Nibbit)Creature.Monster).IsAlone);
else {
    conditional.AddState(hiss,  () => !((Nibbit)Creature.Monster).IsFront);
    conditional.AddState(slice, () =>  ((Nibbit)Creature.Monster).IsFront);
}
slice.FollowUpState = hiss; butt.FollowUpState = slice; hiss.FollowUpState = butt;
```
Pure deterministic cycle Butt → Slice → Hiss → Butt …, entered at a different point depending on front/back/alone flags set by the encounter.

### Monster B: `Models/Monsters/Byrdonis.cs` (act-1 elite)
HP 81–84 (A≥8: flat 90). `AfterAddedToRoom` applies `TerritorialPower` 1. Two-move alternation, starts on Swoop:
```csharp
MoveState peck  = new MoveState("PECK_MOVE",  PeckMove,  new MultiAttackIntent(PeckDamage, PeckRepeat)); // 3x3 (A≥9: 3x4)
MoveState swoop = new MoveState("SWOOP_MOVE", SwoopMove, new SingleAttackIntent(SwoopDamage));           // 17 (A≥9: 19)
swoop.FollowUpState = peck; peck.FollowUpState = swoop;
return new MonsterMoveStateMachine(list, swoop);
```
`TerritorialPower` (`Models/Powers/TerritorialPower.cs`) grants +Amount Strength in `AfterSideTurnEnd` — i.e. Ritual, renamed.

### Monster C: `Models/Monsters/Vantom.cs` (act-1 boss)
HP flat 173 (A≥8: 183). `AfterAddedToRoom` applies `SlipperyPower` 8 (A≥8: 9). Strict 4-move loop:
```
INK_BLOT_MOVE    SingleAttackIntent(7 / A9:8)
INKY_LANCE_MOVE  MultiAttackIntent(6 / A9:7, 2)
DISMEMBER_MOVE   SingleAttackIntent(26 / A9:30) + StatusIntent(3)  -> 3 Wounds to discard
PREPARE_MOVE     BuffIntent -> +2 Strength
```
`SlipperyPower`: every damage instance with `UnblockedDamage >= 1` is decremented by 1, and `ModifyHpLostAfterOsty` caps HP loss at 1 while it's up — i.e. Vantom takes **1 damage per hit** for its first 8 (9) hits. It scales ×playerCount in multiplayer.

## 2. Encounters

### Model — `Models/EncounterModel.cs`
`RoomType` (Monster/Elite/Boss), `IsWeak`, `Tags` (`EncounterTag`), `Slots` (named positions), `AllPossibleMonsters`, `GenerateMonsters()`. Each encounter gets its **own deterministic Rng**:
```csharp
uint seed = (uint)((int)runState.Rng.Seed + runState.TotalFloor + StringHelper.GetDeterministicHashCode(Id.Entry));
_rng = new Rng(seed);
```
Gold: Monster 10–20, Elite 35–45, Boss flat 100; ×0.75 if `AscensionLevel.Poverty`.

### Act identity — `Models/Acts/`
Acts are named, not numbered: **Overgrowth (Index 0 = Act 1, `IsDefault => true`)**, Hive (1), Glory (2), plus `DeprecatedAct` and `Underdocks` (an alt act, not default). `Overgrowth`: `BaseNumberOfRooms => 15`, `NumberOfWeakEncounters => 3`, floors = rooms + 2. Elites on map: `Math.Round(5 * (SwarmingElites ? 1.6 : 1))` = 5, or **8 at A≥1**. Rests: `mapRng.NextGaussianInt(7,1,6,7)`; unknowns: `NextGaussianInt(12,1,10,14)`.

### Pool generation — `ActModel.GenerateRooms` (no weights, grab-bag + no-repeat-tag)
All entries are added with weight `1.0`. Three independent `GrabBag<EncounterModel>`s: 3 weak, then `15 - 3 = 12` normal, then 15 elites pre-rolled. Bags refill when empty. Anti-repeat rule:
```csharp
EncounterModel e = grabBag.GrabAndRemove(rng, e => !e.SharesTagsWith(encounters.LastOrDefault()) && e != encounters.LastOrDefault());
if (e == null) e = grabBag.GrabAndRemove(rng);
```
Boss: `_rooms.Boss = rng.NextItem(AllBossEncounters)` — one of three. `SecondBoss` is only set on the **last** act under `DoubleBoss`, so Act 1 is unaffected. First-ever-run order is forced in `ApplyActDiscoveryOrderModifications` (NibbitsWeak, SlimesWeak, ShrinkerBeetleWeak, InkletsNormal, MawlerNormal, RubyRaidersNormal, NibbitsNormal; elites ByrdonisElite then PhrogParasiteElite).

### Every Act 1 encounter (22, from `Overgrowth.GenerateAllEncounters`)

**Weak (3 of these are drawn)**
| Encounter | Monsters | Tags |
|---|---|---|
| `FuzzyWurmCrawlerWeak` | 1× FuzzyWurmCrawler | Crawler |
| `NibbitsWeak` | 1× Nibbit (`IsAlone = true`) | Nibbit |
| `ShrinkerBeetleWeak` | 1× ShrinkerBeetle | Shrinker |
| `SlimesWeak` | 1 small slime + 1 random medium slime + the other small slime | Slimes |

**Normal (12 drawn)**
| Encounter | Monsters | Tags |
|---|---|---|
| `CubexConstructNormal` | 1× CubexConstruct | — |
| `FlyconidNormal` | random medium slime (Leaf/Twig M) + Flyconid | Mushroom, Slimes |
| `FogmogNormal` | 1× Fogmog (slot "fogmog"); summons EyeWithTeeth into "illusion" | — |
| `InkletsNormal` | 3× Inklet, middle one `MiddleInklet = true` | — |
| `MawlerNormal` | 1× Mawler | — |
| `NibbitsNormal` | Nibbit(`IsFront`) in "front" + Nibbit in "back" | — |
| `OvergrowthCrawlers` | ShrinkerBeetle + FuzzyWurmCrawler | Shrinker, Crawler |
| `RubyRaidersNormal` | 3 distinct raiders drawn from {Axe, Assassin, Brute, Crossbow, Tracker}, max 1 each | — |
| `SlimesNormal` | TwigSlimeM + LeafSlimeM + LeafSlimeS + TwigSlimeS (small order coin-flipped) | Slimes |
| `SlitheringStranglerNormal` | SlitheringStrangler + one of {SnappingJaxfruit} \| {1 random medium slime} \| {2 random small slimes} | — |
| `SnappingJaxfruitNormal` | SnappingJaxfruit + Flyconid | Mushroom |
| `VineShamblerNormal` | 1× VineShambler | — |

**Elite (3, pool cycled 15×)**
| Encounter | Monsters |
|---|---|
| `BygoneEffigyElite` | 1× BygoneEffigy |
| `ByrdonisElite` | 1× Byrdonis |
| `PhrogParasiteElite` | 1× PhrogParasite (slot "phrog"); on death spawns 4× Wriggler into "wriggler1..4" |

**Boss (1 of 3)** — `VantomBoss` (Vantom), `CeremonialBeastBoss` (CeremonialBeast), `TheKinBoss` (KinFollower ×2 — one `StartsWithDance` — + KinPriest in "leaderSlot"). `BossDiscoveryOrder` = Vantom, CeremonialBeast, TheKin (first unseen is forced during tutorial runs).

## 3. Ascension

`Entities/Ascension/AscensionLevel.cs` is an enum whose ordinal **is** the level; `AscensionManager.HasLevel(level) => _level >= (int)level`. **Every check in the codebase is `>=`** — there are no `==` ascension checks. `maxAscensionAllowed = 10`.

| Lvl | Enum | Effect | Where |
|---|---|---|---|
| 1 | `SwarmingElites` | Elite map points ×1.6 → 5 becomes 8 per act | `Map/MapPointTypeCounts.cs:14` |
| 2 | `WearyTraveler` | Ancient (Neow) pre-event heal multiplied by `0.8` | `Models/AncientEventModel.cs:180` |
| 3 | `Poverty` | All combat gold ×`PovertyAscensionGoldMultiplier = 0.75` (min and max) | `Models/EncounterModel.cs:75,94`; `Multiplayer/Game/OneOffSynchronizer.cs:134` |
| 4 | `TightBelt` | `player.SubtractFromMaxPotionCount(1)` | `AscensionManager.ApplyEffectsTo` |
| 5 | `AscendersBane` | Adds `AscendersBane` curse to deck, `FloorAddedToDeck = 1` | `AscensionManager.ApplyEffectsTo` |
| 6 | `Inflation` | Card-removal base cost 75→100, per-removal increase 25→50 | `Entities/Merchant/MerchantCardRemovalEntry.cs` |
| 7 | `Scarcity` | Worse card rewards: common odds 0.60→0.615, rare 0.03→0.0149, elite common 0.50→0.549, elite rare 0.10→0.05, shop common 0.54→0.585, shop rare 0.09→0.045, rarity growth 0.01→0.005, upgraded-card odd scaling 0.25→0.125 | `Odds/CardRarityOdds.cs`, `Factories/CardFactory.cs:23` |
| 8 | `ToughEnemies` | Per-monster HP bump (and a few block/utility values, e.g. Nibbit `SliceBlock` 5→6, Vantom `SlipperyAmt` 8→9, GremlinMerc damage) | every `Models/Monsters/*.cs` |
| 9 | `DeadlyEnemies` | Per-monster damage / buff-amount bump | every `Models/Monsters/*.cs` |
| 10 | `DoubleBoss` | A second boss encounter is rolled for the **last act only** (`i == State.Acts.Count - 1`); Act 1 unaffected | `Runs/RunManager.cs:686` |

**Not present**: no ascension modifies starting max HP, no A≥10 "elites get more HP", no per-act boss-count change for act 1. At A10 the Act-1-relevant deltas are exactly: 8 elites on the map, Neow heal ×0.8, gold ×0.75, −1 potion slot, Ascender's Bane in deck, worse card rarities, and the per-monster ToughEnemies/DeadlyEnemies values (both active, since 10 ≥ 8 and 10 ≥ 9).

Act 1 monster stat table at A10 (min–max HP; "flat" means `MaxInitialHp => MinInitialHp`):
Nibbit 44–48 · FuzzyWurmCrawler 58–59 · ShrinkerBeetle 40–42 · LeafSlimeS 12–16 · LeafSlimeM 33–36 · TwigSlimeS 8–12 · TwigSlimeM 27–29 · Inklet 12–18 · Mawler 76 flat · Fogmog 78 flat · EyeWithTeeth 6 flat (no ascension scaling) · Flyconid 51–53 · SnappingJaxfruit 34–36 · SlitheringStrangler 54–56 · VineShambler 64 flat · CubexConstruct 70 flat · AxeRubyRaider 21–23 · AssassinRubyRaider 19–24 · BruteRubyRaider 31–34 · CrossbowRubyRaider 19–22 · TrackerRubyRaider 22–26 · Byrdonis 90 flat · BygoneEffigy 132 flat · PhrogParasite 66–68 · Wriggler 18–22 · Vantom 183 flat · CeremonialBeast 262 flat · KinFollower 62–63 · KinPriest 199 flat.

## 4. Monster-side statuses/powers (act 1), all in `Models/Powers/`

- `TerritorialPower` — **Ritual**. `AfterSideTurnEnd` → `PowerCmd.Apply<StrengthPower>(owner, Amount)`. Byrdonis (1).
- `RitualPower.cs` exists but is used by cultist-type monsters (`DampCultist`, `CalcifiedCultist`, `DevotedSculptor`) — **not in act 1**.
- `CurlUpPower.cs` exists; used by `LouseProgenitor` — **not in act 1**.
- There is **no Split power and no Split-ing monster** anywhere (`rg Split Models/Monsters` → no hits). Slimes are fixed S/M variants, no splitting.
- `PlowPower` — the **Mode Shift** equivalent, Ceremonial Beast only. `PowerStackType.Counter`, `Type => Debuff`. On `AfterDamageReceived`, if `result.UnblockedDamage > 0 && target.CurrentHp <= Amount`, it strips all `TemporaryStrengthPower` and `StrengthPower`, then `CeremonialBeast.SetStunned()` + `CreatureCmd.Stun(owner, StunnedMove, BeastCryState.StateId)`, then removes itself. Amount = 150 (A≥9: 160). Note it triggers on **remaining HP ≤ Amount**, not on accumulated damage.
- `SlipperyPower` — Vantom, Inklet(1). Counter; decrements per unblocked hit; caps HP loss at 1. `ShouldScaleInMultiplayer`, ×playerCount.
- `InfestedPower` — PhrogParasite (4). `AfterDeath` spawns 4 stunned Wrigglers; `ShouldStopCombatFromEnding() => true` so the fight continues past the elite's death.
- `MinionPower` — KinFollower. `OwnerIsSecondaryEnemy`, `ShouldOwnerDeathTriggerFatal() => false`, power survives owner death.
- `IllusionPower` — EyeWithTeeth (Fogmog's summon). Illusions keep buffs through death and can revive; has `FollowUpStateId`.
- `ArtifactPower` — CubexConstruct (1). `SlowPower` — BygoneEffigy (1). `AsleepPower`/`SleepIntent` — BygoneEffigy sleeps until woken, then `+10 Strength`.
- Applied to players by act-1 monsters: `VulnerablePower` (Flyconid 2, Mawler 3), `FrailPower` (Flyconid 2, TrackerRubyRaider 2, KinPriest 1), `WeakPower` (KinPriest 1), `ConstrictPower` (SlitheringStrangler 3), `ShrinkPower` (ShrinkerBeetle, `-1` = infinite; −30 damage), `TangledPower` (VineShambler — afflicts all Attack cards with `Entangled`), `RingingPower` (CeremonialBeast), plus status cards (Wound/Slimed via `StatusIntent`).

## 5. HP rolling, minions, multi-phase

HP is rolled in `Combat/CombatState.CreateCreature` → `Creature.SetUniqueMonsterHpValue(creaturesOnSide, RunState.Rng.Niche)`. Stream = **`RunRngType.Niche`**, not a dedicated monster-HP stream (`Runs/RunRngSet.cs`; `MonsterAi` is a separate stream used only for move rolls). The roll is *uniform over the range minus values already taken by other enemies on the same side*:

```csharp
HashSet<int> hashSet = Enumerable.Range(min, max + 1 - min).ToHashSet();
hashSet.ExceptWith(creaturesOnSide.Except([this]).Select(e => e.MaxHp));
MonsterMaxHpBeforeModification = (_currentHp = (_maxHp = hashSet.Count <= 0 ? rng.NextInt(min, max+1) : rng.NextItem(hashSet)));
```
So duplicate monsters in one encounter get **distinct** max HP when the range allows. `Creature`'s constructor initially sets HP to `maxInitialHp`; the unique roll then overwrites it. Multiplayer applies `ScaleMonsterHpForMultiplayer(encounter, playerCount, actIndex)` afterwards.

Minions/summons: added mid-combat with `CreatureCmd.Add<T>(combatState, slotName)` or `CreatureCmd.Add(model, combatState, side, slot)`, which routes through `CreateCreature` (same HP roll, new `CombatId`) and `Encounter.OnCreatureSpawned` (tracked in `SpawnedEnemies`, used for gold proportion). Slots come from `EncounterModel.Slots` / `GetNextSlot`. `MonsterModel.SpawnedThisTurn` is true until the first side-switch. Examples: Fogmog's `ILLUSION_MOVE` → `CreatureCmd.Add<EyeWithTeeth>(CombatState, "illusion")`; PhrogParasite's `InfestedPower` → 4 Wrigglers with `StartStunned = true` (their `SPAWNED_MOVE` has a `StunIntent`).

Multi-phase bosses: **not** modelled as separate creatures in act 1. Phases are encoded in the move graph plus a gating power. Ceremonial Beast:
```
STAMP (BuffIntent, applies PlowPower 150/160) -> PLOW (self-loop: 18/20 dmg + 2 Str)
PlowPower breaks at HP <= Amount => strips Strength, forces STUN_MOVE (MustPerformOnceBeforeTransitioning)
STUN -> BEAST_CRY (Debuff) -> STOMP (15/17) -> CRUSH (17/19, +3/+4 Str) -> BEAST_CRY -> ...
```
Elsewhere in the game, true multi-HP-bar phases exist via distinct HP properties on one model (`TestSubject.FirstFormHp/SecondFormHp/ThirdFormHp`), but no act-1 monster uses that.

## 6. Surprises for an StS1 veteran

- **Acts have names, not numbers.** Act 1 = `Overgrowth`; the first act is identified by `ActModel.Index == 0` + `IsDefault`. Alt acts exist per index (`Underdocks`), so "Act 1" isn't guaranteed to be Overgrowth in a general run, though Overgrowth is the only Index-0 act shipped.
- **Move AI is a state graph, not an AI-id + roll-with-history.** Many act-1 monsters (Nibbit, Byrdonis, Vantom, Ruby Raiders, Ceremonial Beast, Vine Shambler) are fully deterministic loops with zero RNG. Only `RandomBranchState` monsters (Flyconid, Fogmog, Inklet, Mawler, PhrogParasite, small slimes, SlitheringStrangler, TwigSlimeM) consume `MonsterAi` RNG.
- **A move carries several intents at once** (`new MoveState(id, fn, new SingleAttackIntent(d), new DefendIntent())`), so attack+block and attack+status render as compound intents natively.
- **New intent types**: `DeathBlow`, `Sleep`, `Stun`, `StatusCard`, `CardDebuff`, `Hidden`, `DebuffStrong`. `IntendsToAttack` counts both `Attack` and `DeathBlow`.
- **Monster attacks target all opponents by default.** `AttackCommand.FromMonster(monster)` ends with `return TargetingAllOpponents(monster.Creature.CombatState);` — the game is co-op multiplayer, and moves receive `IReadOnlyList<Creature> targets = combatState.PlayerCreatures`. Powers declare `ShouldScaleInMultiplayer` / `GetScaledAmountForMultiplayer`, and monster HP is scaled by `ScaleMonsterHpForMultiplayer`. For a single-player sim all of this collapses to one target, but the plumbing assumes a list.
- **Intent numbers are already player-relative**: `AttackIntent.GetSingleDamage` resolves through `Hook.ModifyDamage` for `LocalContext.GetMe(...)`.
- **Slot names** matter — encounters declare named positions (`"front"/"back"`, `"phrog"/"wriggler1..4"`, `"illusion"/"fogmog"`), and monster behaviour branches on flags the encounter sets (`Nibbit.IsFront/IsAlone`, `Inklet.MiddleInklet`, `KinFollower.StartsWithDance`).
- **No Split mechanic exists.** Slimes are fixed S/M variants. Curl Up and Ritual exist as powers but are used outside act 1; Byrdonis's `TerritorialPower` is the act-1 Ritual analogue.
- **Cards can be "afflicted"** (`CardCmd.Afflict<Entangled>`), a card-level status distinct from statuses/powers — `VineShambler`'s `TangledPower` and `CardDebuffIntent` use it.
- **Encounter RNG is separate and derived from `(seed + totalFloor + hash(encounterId))`**, so which monsters spawn within an encounter does not consume run RNG streams.
- **Escape is reward-relevant**: `CalculateGoldProportion` = `1 - escaped/spawned`.
