# StS2 v0.107.1 — Powers, relics, potions, run state, enchantments, modding

Root: `decompiled/MegaCrit/Sts2/Core`

---

## 1. Powers

### Base class
`Models/PowerModel.cs` (650 lines), `abstract class PowerModel : AbstractModel`.

Core state fields: `_amount`, `_amountOnTurnStart`, `_skipNextDurationTick`, `_owner` (Creature), `_applier`, `_target`, `_dynamicVars` (DynamicVarSet — public, cloned, usable in loc strings), `_internalData` (private, reset on clone), `_canonicalInstance`.

Abstract/virtual classification knobs:
- `abstract PowerType Type` — `None|Buff|Debuff` (`Entities/Powers/PowerType.cs`).
- `abstract PowerStackType StackType` — `None|Counter` (visible amount, manually changed) `|Single` (hidden, always 1) (`Entities/Powers/PowerStackType.cs`).
- `virtual PowerInstanceType InstanceType` — `None|Instanced|InstancedPerApplier` (`Entities/Powers/PowerInstanceType.cs`). Instanced powers create a new instance per application (TheBomb); InstancedPerApplier one per applier (Oblivion).
- `virtual bool AllowNegative => false` (true for Strength/Dexterity/Focus).
- `virtual bool ShouldScaleInMultiplayer`, `GetScaledAmountForMultiplayer(...)` = `amount * playerCount * MultiplayerScalingModel.GetMultiplayerScaling(encounter, actIndex)`.

Removal rule (verbatim):
```csharp
public bool ShouldRemoveDueToAmount()
{
    if (AllowNegative || Amount > 0) { if (AllowNegative) return Amount == 0; return false; }
    return true;
}
```
i.e. non-negative powers are removed at ≤0, negative-capable powers removed at exactly 0. `SetAmount` clamps to ±999,999,999 and calls `Owner.InvokePowerModified(this, delta, silent)`.

`GetTypeForAmount(decimal)` flips buff/debuff display for negative counters. `AmountOnTurnStart` is refreshed at the very start of each turn before hooks so mid-turn applications don't trigger same-turn effects.

Duration ticking: `bool SkipNextDurationTick` — "enables the behavior of duration-type powers (Vulnerable, Weak, etc.) ticking down at the end of the monster side turn, but skipping the first tick if a monster applied the power to the player." Implemented in `Commands/PowerCmd.cs:190`:
```csharp
public static async Task TickDownDuration(PowerModel power)
{
    if (power.SkipNextDurationTick) power.SkipNextDurationTick = false;
    else await Decrement(power);
}
```

Power-local lifecycle hooks: `BeforeApplied(target, amount, applier, cardSource)`, `AfterApplied(applier, cardSource)`, `AfterRemoved(oldOwner)`, `ShouldPowerBeRemovedAfterOwnerDeath()`, `ShouldOwnerDeathTriggerFatal()`. `ToMutable(initialAmount)` clones from canonical; `ApplyInternal(owner, amount, silent)` / `RemoveInternal()`.

### How a power hooks into combat
Powers inherit the *global* hook surface from `Models/AbstractModel.cs` (2442 lines, ~185 virtual hook methods) and set `ShouldReceiveCombatHooks => true`. Dispatch is `Hooks/Hook.cs` (static class), which iterates `combatState.IterateHookListeners()` — gated by `IterateCombatHookListeners`, which yields nothing if `CombatManager.Instance.IsOverOrEnding && !IsStarting`.

Hook families (same surface for powers, relics, potions, cards, enchantments, modifiers):
- `Task`-returning event hooks: `AfterCardPlayed/Late`, `BeforeCardPlayed`, `AfterCardDrawn/Early`, `AfterCardExhausted`, `AfterCardDiscarded`, `Before/AfterSideTurnStart(+Late)`, `BeforeSideTurnEndVeryEarly/Early/…`, `AfterSideTurnEnd(+Late)`, `AfterPlayerTurnStart(+Early/Late)`, `Before/AfterDamageReceived(+Late)`, `AfterDamageGiven`, `Before/AfterBlockGained`, `AfterBlockCleared/Broken`, `Before/AfterDeath`, `Before/AfterPowerAmountChanged`, `Before/AfterPotionUsed`, `AfterOrbChanneled/Evoked`, `BeforeCombatStart(+Late)`, `AfterCombatEnd`, `AfterCombatVictory(+Early)`, `AfterRoomEntered`, `AfterActEntered`, `AfterRestSiteHeal/Smith`, `AfterItemPurchased`, `AfterMapGenerated`, `AfterRewardTaken`, …
- `decimal`-returning value modifiers, summed/multiplied across listeners: `ModifyDamageAdditive`, `ModifyDamageMultiplicative`, `ModifyDamageCap`, `ModifyBlockAdditive/Multiplicative`, `ModifyHpLostBeforeOsty(+Late)`, `ModifyHpLostAfterOsty(+Late)`, `ModifyEnergyGain`, `ModifyMaxEnergy`, `ModifyHandDraw(+Late)`, `ModifyGoldGained`, `ModifyMerchantPrice`, `ModifyOrbValue`, `ModifyPowerAmountGivenAdditive/Multiplicative`, `ModifyCardRewardUpgradeOdds`.

Damage pipeline signature: `ModifyDamageAdditive(Creature? target, decimal amount, ValueProp props, Creature? dealer, CardModel? cardSource)` — `props.IsPoweredAttack()` is the gate that separates attack damage from HP-loss/unpowered damage (`ValueProp.Unpowered`).

### Full examples

**Strength** (`Models/Powers/StrengthPower.cs`) — buff, Counter, `AllowNegative => true`, adds `Amount` to outgoing powered attacks when `Owner == dealer`. Body is exactly a four-line `ModifyDamageAdditive`.

**Vulnerable** (`Models/Powers/VulnerablePower.cs`) — debuff, Counter, `DamageIncrease` DynamicVar = `1.5m`:
```csharp
public override decimal ModifyDamageMultiplicative(Creature? target, decimal amount, ValueProp props, Creature? dealer, CardModel? cardSource)
{
    if (target != base.Owner) return 1m;
    if (!props.IsPoweredAttack()) return 1m;
    decimal num = base.DynamicVars["DamageIncrease"].BaseValue;
    // PaperPhrog (dealer relic), CrueltyPower (dealer), DebilitatePower (target) each get a chance to rewrite num
    return num;
}
public override async Task AfterSideTurnEnd(PlayerChoiceContext choiceContext, CombatSide side, IEnumerable<Creature> participants)
{ if (side == CombatSide.Enemy) await PowerCmd.TickDownDuration(this); }
```

**Weak** (`Models/Powers/WeakPower.cs`) — mirror image: `DamageDecrease = 0.75m`, applies when `dealer == Owner`, modifiers from `PaperKrane` (target's relic) and `DebilitatePower`; same `AfterSideTurnEnd → TickDownDuration` on `CombatSide.Enemy`. Both Vulnerable and Weak are the canonical turn-decrementing pattern: **duration ticks at the end of the ENEMY side turn only**.

**Feel No Pain** (`Models/Powers/FeelNoPainPower.cs`) — buff, Counter, `AfterCardExhausted` → `CreatureCmd.GainBlock(Owner, Amount, ValueProp.Unpowered, null)` when `card.Owner.Creature == Owner`.

**Demon Form** (`Models/Powers/DemonFormPower.cs`) — buff, Counter, `AfterSideTurnStart` → if `participants.Contains(Owner)`, `PowerCmd.Apply<StrengthPower>(…, Owner, Amount, Owner, null)`.

**Rage** (`Models/Powers/RagePower.cs`) — buff, Counter; `AfterCardPlayed` → gain `Amount` block on Attack plays; `AfterSideTurnEnd` → `PowerCmd.Remove(this)` (self-removing at end of turn).

`ITemporaryPower` (`Models/ITemporaryPower.cs`): powers like `TemporaryStrengthPower`/`FlexPotionPower` wrap an `InternallyAppliedPower` (StrengthPower) plus an `OriginModel`, with `IgnoreNextInstance()` for debuff-copy effects (Misery).

### All power names (260 files; `Models/Powers/*Power.cs`, suffix stripped)
Accelerant Accuracy Adaptable Afterimage Aggression Anticipate Arsenal Artifact Asleep Automation BackAttackLeft BackAttackRight Barricade BattlewornDummyTimeLimit BeaconOfHope BiasedCognition BlackHole BlockNextTurn Blur BorrowedTime Buffer Burrowed Burst Calamity Calcify CallOfTheVoid ChainsOfBinding ChildOfTheStars Clarity Colossus Confused Conqueror Constrict ConsumingShadow Coolant Coordinate CorrosiveWave Corruption Countdown Covered CrabRage CreativeAi CrimsonMantle Cruelty CrushUnder Curious CurlUp Dampen DanseMacabre DarkEmbrace DarkShackles Debilitate Demesne Demise DemonForm DevourLife Dexterity DiamondDiadem DieForYou Disintegration Doom DoubleDamage DrawCardsNextTurn Duplication DyingStar EchoForm EnergyNextTurn EnfeeblingTouch Enrage Entropy Envenom EscapeArtist FanOfKnives Fasten FeedingFrenzy FeelNoPain Feral FlameBarrier Flanking FlexPotion Flutter FocusedStrike Focus ForbiddenGrimoire ForegoneConclusion Frail FreeAttack FreePower FreeSkill Friendship Furnace Galvanic Genesis Gigantification Gravity Guarded Hailstorm HammerTime Hang HardenedShell HardToKill Hatch Haunt Heist HelicalDart HelloWorld Hellraiser Hex HighVoltage Hotfix Illusion Imbalanced Improvement Inferno Infested InfiniteBlades Intangible Intercept Iteration Juggernaut Juggling Knockdown Leadership Lethality LightningRod Loop MachineLearning MagicBomb Mangle MasterPlanner Mayhem MindRot Minion MonarchsGaze MonarchsGazeStrengthDown Monologue NecroMastery Nemesis Neurosurge Nightmare NoBlock NoDraw NoEnergyGain Nostalgia NoxiousFumes Oblivion OneTwoPunch Orbit Outbreak Pagestorm PainfulStabs PaleBlueDot Panache PaperCuts Parry PersonalHive PhantomBlades PiercingWail PillarOfCreation Plating Plow Poison PossessSpeed PossessStrength PrepTime Pyre Radiance Rage Rampart Ravenous ReaperForm Reattach Rebound Reflect Regen ReptileTrinket RetainHand Ringing Ritual RollingBoulder Royalties Rupture Sandpit SeekingEdge SelfFormingClay SentryMode SerpentForm SetupStrike ShacklingPotion Shadowmeld ShadowStep Shriek Shrink Shroud SicEm SignalBoost Skittish SleightOfFlesh Slippery Sloth Slow Slumber Smoggy Smokestack Sneaky Soar SpectrumShift SpeedPotion Speedster Spinner SpiritOfAsh Stampede StarNextTurn SteamEruption Stock Storm Strangle Stratagem Strength Subroutine Suck SummonNextTurn Surprise Surrounded Swipe SwordSage Synchronize TagTeam Tainted Tangled Tank TemporaryDexterity TemporaryFocus TemporaryStrength Tender Territorial TheBomb TheGambit TheHunt TheSealedThrone Thievery Thorns Thunder ToolsOfTheTrade ToricToughness Tracking TrashToTreasure Tyranny Unmovable Veilpiercer Vicious Vigor VitalSpark VoidForm Vulnerable WasteAway Weak WellLaidPlans WitheringPresence WraithForm.

---

## 2. Relics

### Base class
`Models/RelicModel.cs` (590 lines), `abstract class RelicModel : AbstractModel`, `ShouldReceiveCombatHooks => true`. Uses the same 185-hook `AbstractModel` surface as powers, plus relic-only: `AfterObtained()`, `AfterRemoved()`.

State: `Owner` (Player, not Creature), `IsWax`/`IsMelted` (`[SavedProperty(SerializationCondition.SaveIfNotTypeDefault)]`), `FloorAddedToDeck`, `StackCount` (with `IsStackable`/`IncrementStackCount`), `Status` (`RelicStatus.Normal|Active|Disabled`, drives the pulse shader), `DynamicVars`, `CanonicalInstance`, `HasBeenRemovedFromState`.

Flags: `abstract RelicRarity Rarity` (`None, Starter, Common, Uncommon, Rare, Shop, Event, Ancient` — `Entities/Relics/RelicRarity.cs`), `IsUsedUp`, `HasUponPickupEffect`, `SpawnsPets`, `IsStackable`, `IsAllowedInShops`, `ShowCounter`/`DisplayAmount`, `IsTradable` (false if used up / pickup-effect / melted / pet-spawner / Starter|Event|Ancient).

Prices: `MerchantCost` = Common 175, Uncommon 225, Rare 275, Shop 200; Starter/Event/Ancient 999999999.

Spawn gating: `virtual bool IsAllowed(IRunState runState)`, `IsAllowedAtNeow(Player)`, and helper `IsBeforeAct3TreasureChest(runState)` (`TotalFloor < 41`, or `< 38` in multiplayer).

Persistence: `ToSerializable()` → `SerializableRelic { Id, Props (SavedProperties), FloorAddedToDeck }`; `FromSerializable`. Any `[SavedProperty]` field on a relic round-trips (e.g. `Nunchaku.AttacksPlayed`, `MawBank.HasItemBeenBought`).

### Pools
`Models/RelicPoolModel.cs`: `AllRelics` (cached, `ModHelper.ConcatModelsFromMods(this, _relics)` appended), `AllRelicIds`, `abstract GenerateAllRelics()`, `virtual GetUnlockedRelics(UnlockState)`. `RelicModel.Pool => ModelDb.AllRelicPools.First(p => p.AllRelicIds.Contains(Id))` — so pool membership defines character-affiliation, and rarity is a per-relic property, not a pool axis.

Pools (`Models/RelicPools/`): `SharedRelicPool`, `IroncladRelicPool`, `SilentRelicPool`, `DefectRelicPool`, `RegentRelicPool`, `NecrobinderRelicPool`, `EventRelicPool`, `FallbackRelicPool`, `DeprecatedRelicPool`.

`IroncladRelicPool` (8 relics, verbatim list): `Brimstone, BurningBlood, CharonsAshes, DemonTongue, PaperPhrog, RedSkull, RuinedHelmet, SelfFormingClay`. `GetUnlockedRelics` strips relics belonging to `Ironclad3Epoch` / `Ironclad6Epoch` until those epochs are revealed (`Timeline/Epochs/`).

Total relic files: **298** in `Models/Relics/` (includes `DeprecatedRelic`, 10 `Pael*` boss relics, and ~11 `Fake*` shop-forgery relics).

### Three full relics
**Ironclad starter — BurningBlood** (`Models/Relics/BurningBlood.cs`): `Rarity => Starter`, `CanonicalVars = new HealVar(6m)`, and
```csharp
public override async Task AfterCombatVictory(CombatRoom _)
{ if (!base.Owner.Creature.IsDead) { Flash(); await CreatureCmd.Heal(base.Owner.Creature, base.DynamicVars.Heal.BaseValue); } }
```

**Simple counter — Nunchaku** (`Models/Relics/Nunchaku.cs`): Uncommon, `ShowCounter => true`, `[SavedProperty] int AttacksPlayed`, vars `CardsVar(10)` + `EnergyVar(1)`; `AfterCardPlayed` increments on Attack plays and, `if (CombatManager.Instance.IsInProgress && AttacksPlayed % 10 == 0)` calls `PlayerCmd.GainEnergy(1, Owner)`; `UpdateDisplay()` sets `RelicStatus.Active` when `AttacksPlayed % 10 == 9`. Note the counter persists across combats (saved), and increments even out of combat.

**Combat trigger — Vajra** (`Models/Relics/Vajra.cs`): Common, `PowerVar<StrengthPower>(1m)`, `AfterRoomEntered(room)` → `if (room is CombatRoom) { Flash(); PowerCmd.Apply<StrengthPower>(…, Owner.Creature, 1, Owner.Creature, null); }`.

**Out-of-combat — MawBank** (`Models/Relics/MawBank.cs`): Event rarity, `GoldVar(12)`, `[SavedProperty] bool HasItemBeenBought` (sets `Status = RelicStatus.Disabled`, `IsUsedUp => HasItemBeenBought`); `AfterRoomEntered` gives 12 gold if `Owner.RunState.BaseRoom == room && !HasItemBeenBought`; `AfterItemPurchased` disables it.

---

## 3. Potions

### Base class
`Models/PotionModel.cs` (389 lines), `abstract class PotionModel : AbstractModel`, `ShouldReceiveCombatHooks => true`.

- `abstract PotionRarity Rarity` — `None, Common, Uncommon, Rare, Event, Token`.
- `abstract PotionUsage Usage` — `None, CombatOnly, AnyTime, Automatic`.
- `abstract TargetType TargetType` — routed through `IsValidTarget(Creature?)`: `AnyEnemy` requires opposite side; `AnyAlly` same side but not self; `AnyPlayer` any player creature; `Self` == owner; `TargetedNoCreature` accepts null. Comment: "This operates differently than cards! … CardModel's TargetType.Self does not pass a target, whereas potions do."
- `virtual bool CanBeGeneratedInCombat` (excludes heal/revive potions from random in-combat generation), `virtual bool PassesCustomUsabilityCheck`.
- `protected virtual Task OnUse(PlayerChoiceContext, Creature? target)`.

Use flow — `OnUseWrapper`: `RemoveBeforeUse()` → `Hook.BeforePotionUsed` → throw VFX → `CombatManager.BeginCardOrPotionEffect` → `OnUse` → `CombatManager.History.PotionUsed(...)` → `Hook.AfterPotionUsed` → logs into `Owner.RunState.CurrentMapPointHistoryEntry.GetEntry(netId).PotionUsed` → `CheckForEmptyHand`. Queuing goes through `EnqueueManualUse(target)` → `UsePotionAction` → `RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue`.

Serialization: `SerializablePotion { Id, SlotIndex }`.

### Slots
`Entities/Players/Player.cs`: `public const int initialMaxPotionSlotCount = 3;`, `List<PotionModel?> _potionSlots`, `MaxPotionCount => _potionSlots.Count`, `PotionSlots`, `HasOpenPotionSlots`, `GetPotionSlotIndex`, `GetPotionAtSlotIndex`. Slot count is per-player and saved (`SerializablePlayer.MaxPotionSlotCount`, default 3). `PotionBelt` relic has `HasUponPickupEffect => true` and `AfterObtained` → `PlayerCmd.GainMaxPotionCount(2, Owner)`. Procurement returns `PotionProcureResult` / `PotionProcureFailureReason.TooFull` (`Entities/Potions/`).

### Pools
`Models/PotionPoolModel.cs` mirrors the relic pool. Pools: `SharedPotionPool`, `IroncladPotionPool`, `SilentPotionPool`, `DefectPotionPool`, `RegentPotionPool`, `NecrobinderPotionPool`, `EventPotionPool`, `TokenPotionPool`, `DeprecatedPotionPool`. `IroncladPotionPool` returns `Ironclad4Epoch.Potions` = `BloodPotion, SoldiersStew, Ashwater`, gated entirely on `unlockState.IsEpochRevealed<Ironclad4Epoch>()`. 64 potion files total.

### Two potions
**FirePotion** (`Models/Potions/FirePotion.cs`): Common, `CombatOnly`, `TargetType.AnyEnemy`, `DamageVar(20m, ValueProp.Unpowered)`; `OnUse` asserts target, `CreatureCmd.Damage(choiceContext, target, 20, Unpowered, Owner.Creature, null)`. (Unpowered ⇒ Strength/Vulnerable do NOT apply.)

**StrengthPotion / FlexPotion** — see `Models/Potions/StrengthPotion.cs` and `FlexPotion.cs` for the `ITemporaryPower` route (FlexPotion applies `FlexPotionPower`, which internally applies `StrengthPower` and strips it at end of turn).

### Drop chance
`Odds/PotionRewardOdds.cs` — pity system, not a flat roll:
```csharp
public const float targetOdds = 0.5f;     // long-run convergence target
public const float eliteBonus = 0.25f;
private const float _basePotionRewardOdds = 0.4f;   // starting CurrentValue

public bool Roll(Player player, AscensionManager ascensionManager, RoomType roomType)
{
    float currentValue = base.CurrentValue;
    bool flag = Hook.ShouldForcePotionReward(player.RunState, player, roomType);
    float num = ((roomType != RoomType.Elite) ? 0f : 0.25f);
    float num3 = currentValue + num * 0.5f;      // elite adds +0.125 effective
    if (flag || _rng.NextFloat() < num3) { base.CurrentValue -= 0.1f; return true; }
    base.CurrentValue += 0.1f; return false;
}
```
Per-player state (`PlayerOddsSet.PotionReward`, saved via `SerializablePlayerOddsSet`). Called from `RewardsSet.RollForPotionAndAddTo` (`Rewards/RewardsSet.cs:248`).

**Ascension note:** `ascensionManager` is a parameter of `Roll` but is *not read* in this build — potion drop rate is currently ascension-independent. Ascension does affect card rarity odds: `Odds/CardRarityOdds.cs` uses `AscensionHelper.GetValueIfAscension(AscensionLevel.Scarcity, …)` for every rarity constant.

---

## 4. Run state

### The class holding the run
`Runs/RunState.cs` implementing `Runs/IRunState.cs` (`: ICardScope, IPlayerCollection`). Owned by singleton `Runs/RunManager.cs`. Null-object fallback `NullRunState`.

`IRunState` surface: `Acts` / `CurrentActIndex` (setter clears `_visitedMapCoords`, resets `ActFloor` and `NextRoomId`) / `Act`; `ActMap Map`; `CurrentMapCoord`, `CurrentMapPoint`, `RunLocation`, `MapLocation`; `ActFloor`, `TotalFloor`; `CurrentRoomCount`, `CurrentRoom`, `BaseRoom` (rooms are a *stack* — an event that starts a fight pushes a `CombatRoom`); `IsGameOver`; `int AscensionLevel`; `RunRngSet Rng`; `RunOddsSet Odds`; `SharedRelicGrabBag`; `UnlockState`; `IReadOnlyList<ModifierModel> Modifiers`; `BadgeModels`; `MultiplayerScalingModel`; `MapPointHistory` (per-act lists of `MapPointHistoryEntry`), `CurrentMapPointHistoryEntry`; `ExtraRunFields ExtraFields`; `GameMode`; plus `IterateHookListeners(ICombatState? childCombatState)` and `GetAndIncrementNextRoomId()`.

Per-player data lives on `Entities/Players/Player.cs` (from `IPlayerCollection.Players`): `Deck`, `Relics`, `PotionSlots`/`MaxPotionCount`, `Gold`, `MaxEnergy`, `BaseOrbSlotCount`, `Creature` (which holds `CurrentHp`/`MaxHp`/`Block`/powers), `PlayerOdds`, `Rng`, `RelicGrabBag`, `UnlockState`, `Discovered*` lists, `MaxAscensionWhenRunStarted`.

Save shape: `Saves/SerializableRun.cs` — `SchemaVersion, Acts, Modifiers, DailyTime, CurrentActIndex, EventsSeen, PreFinishedRoom, SerializableOdds, SerializableSharedRelicGrabBag, Players, SerializableRng, VisitedMapCoords, MapPointHistory, SaveTime/StartTime/RunTime/WinTime, Ascension, NumReloads, PlatformType, MapDrawings, ExtraFields, GameMode`; `FloorReached => MapPointHistory.Sum(c => c.Count)`. Per player `Saves/Runs/SerializablePlayer.cs`: `CharacterId, CurrentHp, MaxHp, MaxEnergy, MaxPotionSlotCount (=3), Gold, BaseOrbSlotCount, NetId, Deck, Relics, Potions, Rng, Odds, RelicGrabBag, ExtraFields, UnlockState, Discovered*`. **Seed is not a single field** — RNG is a set of independent streams, `SerializableRunRngSet` / `SerializablePlayerRngSet`.

### Combat ↔ run state
`Combat/CombatState.cs` holds `IRunState RunState { get; }`, `_allies`/`_enemies` lists of `Creature`, `_nextCreatureId`, `EncounterModel`, `_escapedCreatures`, `_allCards`, `Modifiers`, `BadgeModels`, `MultiplayerScalingModel`, `int RoundNumber` (starts at 1), active side.

Reads from run state: players list and their `Creature` HP/max HP/block, deck (`Player.PopulateCombatState(rng, state)` clones every `Deck.Cards` entry into the draw pile with `cardModel.DeckVersion = item`, then `RandomizeOrderInternal(this, rng, state)` using `RunState.Rng.Shuffle`), relics (via hook listener iteration), potions, `RunState.CurrentActIndex`, `AscensionLevel`, `Modifiers`, `Encounter.GenerateMonstersWithSlots(CombatState.RunState)`.

Writes back: HP/block persist on the `Creature` owned by `Player` (so HP change is immediately run state); `Player.AfterCombatEnd()` does `Creature.RemoveAllPowersInternalExcept(); PlayerCombatState?.AfterCombatEnd(); Creature.LoseBlockInternal(Creature.Block);` — all powers and block are wiped at combat end, combat card clones are discarded (only `DeckVersion` originals survive). Rewards flow through `CombatRoom.OfferRoomEndRewards()` → `RewardsCmd.GenerateForRoomEnd(player, this)`. Relic `[SavedProperty]` counters mutate in place. History is recorded into `RunState.CurrentMapPointHistoryEntry` (monster IDs, cards played, potions used, card/relic choices).

### Map
`Map/`: `ActMap` (abstract; `startMapPoints`, `BossMapPoint`, `StartingMapPoint`, optional `SecondBossMapPoint`, `GetPoint(MapCoord)`, `GetPointsInRow`, `GetColumnCount/GetRowCount`). Concrete: `StandardActMap`, `GoldenPathActMap`, `SpoilsActMap`, `SavedActMap`, mocks, `NullActMap`. `MapPoint` = `MapCoord coord {col,row}`, `HashSet<MapPoint> parents` / `Children`, `MapPointType PointType`, `CanBeModified`, `Quests`. Generation/post-processing: `MapPathPruning.cs`, `MapPostProcessing.cs`, `MapPointTypeCounts.cs`, `MapTravel.cs`, `MapPointState.cs`.

`MapPointType`: `Unassigned, Unknown, Shop, Treasure, RestSite, Monster, Elite, Boss, Ancient` (`Ancient` is new). Distinct from `RoomType` (`Rooms/RoomType.cs`) because one map point can host multiple rooms; `RoomSet.cs` + `AbstractRoom` / `CombatRoom` / `EventRoom` / `MerchantRoom` / `RestSiteRoom` / `TreasureRoom` / `MapRoom`. `UnknownMapPointOdds.cs` governs `?` resolution.

Rewards (`Rewards/`): `RewardType` enum, `Reward` base; `CardReward`, `GoldReward`, `RelicReward`, `CardRemovalReward`, `SpecialCardReward`, `LinkedRewardSet`, `RewardsSet` (the roll logic).

---

## 5. Enchantments and Modifiers

**Enchantments** (`Models/EnchantmentModel.cs`, 460 lines; 23 in `Models/Enchantments/`) — a new StS2 system: persistent, stackable *card attachments* (a card can be enchanted; the enchantment is a model attached to a specific `CardModel`, serialized as `SerializableEnchantment`, previewable outside combat via `PreviewOutsideOfCombat => true`, and `ShouldReceiveCombatHooks => Card?.ShouldReceiveCombatHooks ?? false`). Fields: `Card`, `Amount`, `IsStackable`, `ShowAmount`, `Status`, `ShouldGlowGold/Red`, `ShouldStartAtBottomOfDrawPile`, `Props`. Gating: `CanEnchantCardType(CardType)`, `CanEnchant(CardModel)`. Effect API is a dedicated value pipeline separate from the global hooks: `EnchantDamageAdditive/Multiplicative(originalDamage, props)`, `EnchantBlockAdditive/Multiplicative`, `EnchantPlayCount(int)`, plus `OnPlay(choiceContext, cardPlay)`, `OnEnchant()`, `ModifyCard()`, `RecalculateValues()`.

Example, `Models/Enchantments/Sharp.cs`: `ShowAmount => true`, `CanEnchantCardType(t) => t == CardType.Attack`, `EnchantDamageAdditive => props.IsPoweredAttack() ? Amount : 0`.

Full list: Adroit, Clone, Corrupted, Glam, Goopy, Imbued, Inky, Instinct, Momentum, Nimble, PerfectFit, RoyallyApproved, Sharp, Slither, SlumberingEssence, SoulsPower, Sown, Spiral, Steady, Swift, TezcatarasEmber, Vigorous (+ `DeprecatedEnchantment`).

**Modifiers** (`Models/ModifierModel.cs`, 158 lines; 16 in `Models/Modifiers/`) — "Run-lifetime model which alters the run for daily and custom runs." `ShouldReceiveCombatHooks => true`. Lifecycle: `OnRunCreated(runState)` / `OnRunLoaded(runState)`; `GenerateNeowOption(EventModel)`; serialized as `SerializableModifier { Id, Props }`. Example `Midas`: doubles every `GoldReward`, removes smith options. Not relevant to standard A10 runs.

---

## 6. Modding

`Modding/`: `ModManager.cs`, `ModHelper.cs`, `Mod.cs`, `ModManifest.cs`, `ModInitializerAttribute.cs`, `ModDependency.cs`, `ModLoadState.cs`, `ModManagerState.cs`, `ModSettings.cs`, `ModSource.cs`, `SettingsSaveMod.cs`, `IModManagerFileIo.cs` / `ModManagerFileIo.cs`, `CombatHookSubscriptionDelegate.cs`, `RunHookSubscriptionDelegate.cs`.

**Entry point / Harmony.** `ModInitializerAttribute(string initializerMethod)` on a class: "If this is present, then upon loading the mod, we'll call the method named `initializerMethod` within the class. Otherwise, we'll create a harmony instance for the mod and call `Harmony.PatchAll`." Harmony is applied in `ModManager.cs:764`:
```csharp
Log.Info($"No ModInitializerAttribute detected. Calling Harmony.PatchAll for {assembly}");
Harmony harmony = new Harmony((mod.manifest.author ?? "unknown") + "." + modId);
```
`ModManager` also exposes `HasHarmonyPatches()`, `IsRunningModded()`, `Mods`, `GetLoadedMods()`, `State`, `PlayerAgreedToModLoading`, `GetGameplayRelevantModNameList()`, events `OnModDetected`, and `MetricsUploadHook OnMetricsUpload(SerializableRun run, bool isVictory, ulong localPlayerId)`.

**Hooks a bridge mod would use** (`ModHelper.cs`):
```csharp
public static void AddModelToPool<TPoolType, TModelType>()      // must run before pools freeze
public static void SubscribeForRunStateHooks(string id, RunHookSubscriptionDelegate del)
public static void SubscribeForCombatStateHooks(string id, CombatHookSubscriptionDelegate del)
public static IEnumerable<AbstractModel> IterateAllRunStateSubscribers(RunState runState)
public static IEnumerable<AbstractModel> IterateAllCombatStateSubscribers(CombatState combatState)
```
Registering a custom `AbstractModel` through these gets it every one of the ~185 hooks — this is the clean, Harmony-free way to build a state-export bridge: subscribe a dummy model, override `AfterSideTurnStart`/`AfterCardPlayed`/`AfterDamageReceived`/etc. and serialize on each.

**C# events for state changes** (no Harmony needed):
- `Runs/RunManager.cs`: `RunStarted(RunState)`, `RoomEntered`, `RoomExited`, `ActEntered`.
- `Combat/CombatManager.cs`: `CombatSetUp(CombatState)`, `CombatEnded(CombatRoom)`, `CombatWon(CombatRoom)`, `CreaturesChanged`, `TurnStarted`, `TurnEnded`, `PlayerEndedTurn(Player,bool)`, `PlayerUnendedTurn`, `AboutToSwitchToEnemyTurn`, `PlayerActionsDisabledChanged`.
- `Entities/Creatures/Creature.cs`: `BlockChanged(int,int)`, `CurrentHpChanged`, `MaxHpChanged`, `PowerApplied/PowerIncreased/PowerDecreased/PowerRemoved`, `Died`, `Revived`.
- `Entities/Players/Player.cs`: `RelicObtained/RelicRemoved`, `PotionProcured/PotionDiscarded/UsedPotionRemoved`, `MaxPotionCountChanged`, `GoldChanged`.
- Power-level: `PowerModel.Flashed/Removed/DisplayAmountChanged`; `RelicModel.Flashed/StatusChanged`.

**Serializable state already exists**: `SerializableRun` / `SerializablePlayer` (JSON via `Saves/MegaCritSerializerContext.cs` + `JsonSerializationUtility.cs`, and binary via `IPacketSerializable`). Combat state itself is not save-serialized as a whole, but `Combat/History/CombatHistory.cs` + `CombatHistoryEntry.cs` + `Entries/` record a structured, per-action combat log, and `Multiplayer/Serialization/` serializes the full action stream — the easiest existing hook for a deterministic export. `Multiplayer/Replay/` holds replay plumbing.

**Dev console** (`DevConsole/DevConsole.cs`, commands in `DevConsole/ConsoleCommands/`): `AbstractConsoleCmd` with `CmdName`, `Args`, `Description`, `bool IsNetworked`, `CmdResult Process(Player? issuingPlayer, string[] args)`. Commands: achievement, act, afflict, ancient, applypower, art, bestiary, block, card, cloud, damage, die, draw, dump, enchant, energy, event, fight, getlogs, godmode, gold, heal, instant, kill, leaderboard, log, multiplayer, open, potion, relic, removecard, room, sentry, stars, trailer, travel, unlock, upgradecard, win. `fight`, `card`, `relic`, `potion`, `applypower`, `energy`, `draw` are enough to construct arbitrary combat setups in the real game for replay tests.
