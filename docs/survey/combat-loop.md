# StS2 v0.107.1 — Combat loop, action system, damage, hooks, RNG

Root: `decompiled/MegaCrit/Sts2/Core`

## 1. Combat state objects

**`Combat/CombatState.cs`** — `class CombatState : ICombatState, ICardScope`. Interface in `Combat/ICombatState.cs`, null-object in `Combat/NullCombatState.cs`.
Fields: `List<Creature> _allies`, `_enemies`, `uint _nextCreatureId` (monotonic `Creature.CombatId`), `EncounterModel? _encounter`, `_escapedCreatures`, `_allCards`.
Properties: `IRunState RunState`, `Allies`, `Enemies`, `Creatures` (allies then enemies), `PlayerCreatures`, `Players`, `HittableEnemies`, `Modifiers`, `BadgeModels`, `MultiplayerScalingModel`, `int RoundNumber` (starts 1), `CombatSide CurrentSide` (`Combat/CombatSide.cs`: None/Player/Enemy).

**Creature-side: `Entities/Creatures/Creature.cs`** — `int Block`, `int CurrentHp`, `int MaxHp`, `List<PowerModel> _powers`, `uint? CombatId`, `MonsterModel? Monster`, `Player? Player`, `CombatSide Side`, `ICombatState? CombatState`, `Player? PetOwner`. `IsPlayer => Player != null`. Raw mutators are the `*Internal` methods (`DamageBlockInternal`, `LoseHpInternal`, `GainBlockInternal`, `LoseBlockInternal`, `HealInternal`, `ApplyPowerInternal`, `RemovePowerInternal`). Powers live **on the Creature**, not on the player.

**Player-side: `Entities/Players/PlayerCombatState.cs`** — the piles and resources:
```csharp
public CardPile Hand  { get; } = new CardPile(PileType.Hand);
public CardPile DrawPile { get; } = new CardPile(PileType.Draw);
public CardPile DiscardPile { get; } = new CardPile(PileType.Discard);
public CardPile ExhaustPile { get; } = new CardPile(PileType.Exhaust);
public CardPile PlayPile { get; } = new CardPile(PileType.Play);
```
plus `int Energy`, `int MaxEnergy => (int)Hook.ModifyMaxEnergy(...)`, `int Stars` (a second resource STS1 lacks), `int TurnNumber` (per-player, distinct from `RoundNumber`), `PlayerTurnPhase Phase`, `OrbQueue OrbQueue`, `List<Creature> _pets`.

**`Entities/Players/Player.cs`** — run-level: `Creature Creature`, `CardPile Deck`, `IReadOnlyList<RelicModel> Relics`, `PotionSlots`, `int MaxEnergy`, `int Gold`, `PlayerRngSet PlayerRng`, `ExtraPlayerFields ExtraFields`, `ulong NetId`, `bool IsActiveForHooks`, `PlayerCombatState? PlayerCombatState`.

HP/block are `int`; all *computation* is `decimal` and truncated on write.

## 2. Turn structure — `Combat/CombatManager.cs` (singleton, 1500 lines)

Phase enum `Combat/PlayerTurnPhase.cs`: `None, Start, AutoPrePlay, Play, AutoPostPlay, End`.

**Combat start**: `SetUpCombat(state)` → `player.ResetCombatState()`, `player.PopulateCombatState(RunState.Rng.Shuffle, state)` (initial shuffle), `AddCreature` for each. Then `StartCombatInternal()` → `AfterCreatureAdded` per creature → `IsInProgress = true` → `Hook.BeforeCombatStart` → `StartTurn()`.

**`StartTurn()` order** (line 422):
1. `SetPhaseForAllPlayers(None)`
2. `Creature.BeforeTurnStart` per creature (snapshots `power.AmountOnTurnStart`)
3. `Hook.BeforeSideTurnStart`
4. Player side: phase → `Start`, clear ready-sets; if not an extra turn, `enemy.PrepareForNextTurn(...)` (rolls next monster move)
5. `Creature.AfterTurnStart(side)` → **block clear** (`ClearBlock()` guarded by `Hook.ShouldClearBlock`; skipped entirely on player turn 1)
6. `Hook.AfterBlockCleared`
7. `SetupPlayerTurn(player, ctx)` per player (see below)
8. `Hook.AfterSideTurnStart`
9. `OrbQueue.AfterTurnStart` per player
10. `RunAutoPrePlayPhase`: phase `AutoPrePlay` → `Hook.AfterAutoPrePlayPhaseEntered` → phase `Play`
11. `CheckWinCondition`, unpause `ActionExecutor`, fire `TurnStarted`

**`SetupPlayerTurn` (line 629)** — energy then draw:
```csharp
if (Hook.ShouldPlayerResetEnergy(state, player)) player.PlayerCombatState.ResetEnergy();
else player.PlayerCombatState.AddMaxEnergyToCurrent();
await Hook.AfterEnergyReset(state, player);
await Hook.BeforeHandDraw(state, player, playerChoiceContext);
decimal handDraw = Hook.ModifyHandDraw(state, player, 5m, out var modifiers);
await Hook.AfterModifyingHandDraw(state, modifiers);
// turn 1 only: bottom-sort ShouldStartAtBottomOfDrawPile, top-sort Innate,
//   handDraw = Max(handDraw, innateCount); handDraw = Min(handDraw, CardPile.MaxCardsInHand);
await CardPileCmd.Draw(playerChoiceContext, handDraw, player, fromHandDraw: true);
await Hook.AfterPlayerTurnStart(state, playerChoiceContext, player);
```
Base draw is `5m` (also `CombatManager.baseHandDrawCount = 5`).

**End of player turn — split into two phases.**
`SetReadyToEndTurn` → `AfterAllPlayersReadyToEndTurn` → `WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction()` → `EndPlayerTurnPhaseOneInternal()` (line 1143):
1. phase → `AutoPostPlay`, `Hook.AfterAutoPostPlayPhaseEntered` per player
2. phase → `End`
3. `Hook.BeforeTurnEnd`
4. `CheckWinCondition`
5. `DoTurnEnd(player, ctx)` per player: `OrbQueue.BeforeTurnEnd`; then partition hand into cards with `HasTurnEndInHandEffect` and `CardKeyword.Ethereal` (gated by `Hook.ShouldEtherealTrigger`); **ethereals exhaust first** (`CardCmd.Exhaust(..., causedByEthereal: true)`), then `OnTurnEndInHandWrapper` for the turn-end cards
6. `Hook.BeforeFlush` per player

Then `ReadyToBeginEnemyTurnAction` → `AfterAllPlayersReadyToBeginEnemyTurn` → `EndPlayerTurnPhaseTwoInternal()` (line 1279):
1. `FlushPlayerHand` per player: if `Hook.ShouldFlush(state, player)`, every hand card that is not `card.ShouldRetainThisTurn` goes to `PileType.Discard`; then `Hook.AfterFlush(state, player, ctx, cardsToFlush, cardsToRetain)`; then `EndOfTurnCleanup()`
2. `Hook.AfterTurnEnd(state, Player, ...)`

Then `SwitchFromPlayerToEnemySide()`: collects `Hook.ShouldTakeExtraTurn` players → `SwitchSides()` → `Hook.AfterTakingExtraTurn` → `StartTurn()`.

`SwitchSides()` (line 1387): if side==Player and no extra turns → side=Enemy. Else side=Player, and if no extra turns `_state.RoundNumber++` and every player's `IncrementTurnNumber()`. Then `creature.OnSideSwitch()` for all; fires `TurnEnded`.

**Monster turn**: `ExecuteEnemyTurn` (line 1061) iterates `_state.Enemies.ToList()`, `PerformIntent()` (visual) → `enemy.TakeTurn()` → `Monster.PerformMove()`, then `CheckWinCondition` after each. Then `EndEnemyTurn` → `EndEnemyTurnInternal()`:
```csharp
await Hook.BeforeTurnEnd(_state, _state.CurrentSide, enemies);
foreach (Player p in _state.Players) p.PlayerCombatState.EndOfTurnCleanup();
await Hook.AfterTurnEnd(_state, _state.CurrentSide, enemies);
```
then `SwitchSides` + `StartTurn`.

**Duration ticks happen at end of the ENEMY side turn, not at end of the owner's turn.** `Models/Powers/VulnerablePower.cs`, `WeakPower.cs`, `FrailPower.cs`, `IntangiblePower.cs` all do `if (side == CombatSide.Enemy) await PowerCmd.TickDownDuration(this);`. `PowerCmd.TickDownDuration` honors `power.SkipNextDurationTick`, which `PowerCmd.Apply` sets when a debuff lands on a player-side creature (so a debuff applied this round doesn't tick the same round).

`Hook.AfterTurnEnd` dispatches `AfterSideTurnEnd` for all listeners, **then a second pass of `AfterSideTurnEndLate`**.

## 3. Action system

Explicitly *not* StS1. From `GameActions/GameAction.cs`:

> A GameAction is a thin wrapper around an async task that should be run in response to player input. THIS IS DIFFERENT from GameActions in STS1 (and the original Unity version of STS2). […] In STS2, these small units of logic are handled by Commands (see the MegaCrit.Sts2.Core.Commands namespace). A GameAction WRAPS these commands, and should ONLY be used for player input.

- **`GameActions/`** — only ~12 real actions: `PlayCardAction`, `UsePotionAction`, `DiscardPotionGameAction`, `EndPlayerTurnAction`, `UndoEndPlayerTurnAction`, `ReadyToBeginEnemyTurnAction`, `PickRelicAction`, `MoveToMapCoordAction`, vote actions, plus `Net*` mirrors. Each has an `ExecuteAction()` returning `Task` and a `ToNetAction()`.
- **`GameActions/ActionExecutor.cs`** — pulls from `ActionQueueSet.GetReadyAction()` and awaits one at a time; outside `NonInteractiveMode` it spins on Godot `ProcessFrame` signals until the task completes. Calls `CombatManager.Instance.CheckWinCondition()` after each action (except `EndPlayerTurnAction`/`ReadyToBeginEnemyTurnAction`).
- **`GameActions/Multiplayer/ActionQueueSet.cs`, `ActionQueueSynchronizer.cs`** — per-player queues + network ordering.
- **`Commands/`** — the actual rules: `CreatureCmd` (damage/block/heal/kill/stun), `CardCmd`, `CardPileCmd` (draw/shuffle/exhaust/discard), `PowerCmd`, `PlayerCmd`, `OrbCmd`, plus builders `Commands/Builders/AttackCommand.cs`. All `static async Task`.

**Nesting/resolution model**: there is no queue of effects. A card's `OnPlay` awaits commands directly, so effects resolve by **recursive async call stack**, depth-first, in source order. `Models/CardModel.cs:1867 OnPlayWrapper` shows the play sequence: move to play pile → `Hook.ModifyCardPlayResultPileTypeAndPosition` → `GeneratePlayCount` → loop `playCount` times { `Hook.BeforeCardPlayed` → `OnPlay` → `Enchantment.OnPlay` → `Affliction.OnPlay` → `Hook.AfterCardPlayed` }.

**Async/await matters a lot for a sim port, but only for two reasons**: (a) `Cmd.Wait` / `Cmd.CustomScaledWait` are pure animation delays and can be no-ops (`NonInteractiveMode.IsActive` already short-circuits them — see `Commands/Cmd.cs`); (b) `PlayerChoiceContext` (`GameActions/Multiplayer/PlayerChoiceContext.cs`) genuinely suspends mid-effect to gather a player decision. `HookPlayerChoiceContext` pauses one player's chain while others continue; `BlockingPlayerChoiceContext` blocks. For a single-player sim, model everything as a synchronous call stack and expose the choice points as sim decision points; use `BlockingPlayerChoiceContext` semantics.

**Epochs are not combat.** `Timeline/Epochs/*` (`Ironclad3Epoch`, `Relic2Epoch`, `NeowEpoch`, …) plus `Timeline/EpochModel.cs`, `EpochEra.cs`, `StoryPool.cs` are the unlock/progression-content system. Ignore them for a combat sim.

## 4. Damage and block resolution

**`Commands/CreatureCmd.cs:240` is the single damage function.** Core sequence per target:
```csharp
decimal modifiedAmount = Hook.ModifyDamage(runState, combatState, originalTarget, dealer, amount, props,
                                           cardSource, ModifyDamageHookType.All, CardPreviewMode.None, out modifiers);
await Hook.AfterModifyingDamageAmount(...);
await Hook.BeforeDamageReceived(...);
Creature creature = originalTarget.PetOwner?.Creature ?? originalTarget;
decimal blockedDamage = creature.DamageBlockInternal(modifiedAmount, props);
decimal unblockedDamage = Hook.ModifyHpLost(..., Math.Max(modifiedAmount - blockedDamage, 0m), ..., HpLossHookPhase.BeforeOsty, out modifiers);
Creature unblockedDamageTarget = Hook.ModifyUnblockedDamageTarget(combatState, originalTarget, unblockedDamage, props, dealer);
unblockedDamage = Hook.ModifyHpLost(..., unblockedDamageTarget, unblockedDamage, ..., HpLossHookPhase.AfterOsty, out modifiers);
DamageResult unblockedDamageResult = unblockedDamageTarget.LoseHpInternal(unblockedDamage, props);
```
Then `wasFullyBlocked = !Unblockable && (blockedDamage > 0 || target.Block > 0) && (int)unblockedDamage == 0`, then `AfterBlockBroken` → `AfterCurrentHpChanged` → `AfterDamageGiven` → `AfterDamageReceived` → `Kill(killedCreatures)`.

**`Hook.ModifyDamageInternal` (`Hooks/Hook.cs:2511`) — the exact order**: enchantment additive then multiplicative first (in `ModifyDamage`), then
1. full pass of `ModifyDamageAdditive` over all listeners: `num += item.ModifyDamageAdditive(...)` — Strength returns `Amount`
2. full pass of `ModifyDamageMultiplicative`: `num *= item.ModifyDamageMultiplicative(...)` — Vulnerable `1.5m`, Weak `0.75m`
3. `ModifyDamageCap` pass: takes the minimum cap; Intangible returns `1m`
4. `ModifyDamage` returns `Math.Max(0m, num)`

**No rounding until the write.** Everything stays `decimal`. Truncation happens in:
```csharp
// Creature.DamageBlockInternal
decimal num = props.HasFlag(ValueProp.Unblockable) ? 0m : Math.Min(Block, amount);
Block -= (int)num;   // truncates toward zero
return num;          // returns the un-truncated decimal

// Creature.LoseHpInternal
bool flag = CurrentHp > 0 && amount >= (decimal)CurrentHp;
int num = (int)Math.Min(amount, 999999999m);   // truncates
CurrentHp = Math.Max(CurrentHp - num, 0);
```
So Strength(+N) → ×1.5 Vulnerable → ×0.75 Weak, all as exact decimals, truncated once at HP subtraction. `Strength 3 + base 6 = 9 × 1.5 × 0.75 = 10.125 → 10 HP`. Note the asymmetry: `blockedDamage` is subtracted from block truncated but from `modifiedAmount` untruncated, so a fractional hit into block can lose a fractional point.

`Commands/DamageCmd.cs` is a façade: `DamageCmd.Attack(decimal)` returns an `AttackCommand`. `Commands/Builders/AttackCommand.cs:~536` is the hit loop: `Hook.BeforeAttack` → `decimal attackCount = Hook.ModifyAttackHitCount(combatState, this, _hitCount)` → per hit, recompute alive targets, pick random target via `RunState.Rng.CombatTargets.NextItem(validTargets)` if randomly targeted → `CreatureCmd.Damage(...)` → `Hook.AfterAttack`.

**Block gain — `CreatureCmd.GainBlock` (line 635)**:
```csharp
await Hook.BeforeBlockGained(combatState, creature, amount, props, cardPlay?.Card);
modifiedAmount = Hook.ModifyBlock(combatState, creature, modifiedAmount, props, cardPlay?.Card, cardPlay, out modifiers);
modifiedAmount = Math.Max(modifiedAmount, 0m);
await Hook.AfterModifyingBlockAmount(...);
if (modifiedAmount > 0m) { creature.GainBlockInternal(modifiedAmount); ... }
await Hook.AfterBlockGained(...);
```
`Hook.ModifyBlock` (`Hooks/Hook.cs:1310`) mirrors damage: enchantment additive then multiplicative, then a full `ModifyBlockAdditive` pass (`num += ...`, Dexterity returns `Amount`), then a full `ModifyBlockMultiplicative` pass (`num *= ...`, Frail returns `0.75m`, `MultiplayerScalingModel` returns `playerCount * scaling`), return `Math.Max(0m, num)`. `GainBlockInternal` does `Block = (int)Math.Min(Block + amount, 999999999m)` — truncated.

**`ValueProps/ValueProp.cs`** (flags): `Unblockable = 2` (HP loss, e.g. poison), `Unpowered = 4` (relic/potion/power damage — ignored by Strength etc.), `Move = 8` (attack-card / monster-move damage), `SkipHurtAnim = 0x10`.

## 5. Hooks

Two layers. **Dispatchers** are the ~147 `public static` methods on `Hooks/Hook.cs`; **subscriber overrides** are ~185 `public virtual` methods on `Models/AbstractModel.cs`, which `PowerModel`, `RelicModel`, `CardModel`, `PotionModel`, `MonsterModel`, `EnchantmentModel`, `AfflictionModel`, `ModifierModel`, `BadgeModel`, `SingletonModel` all inherit. Note the naming skew: `Hook.AfterTurnEnd` dispatches `AbstractModel.AfterSideTurnEnd` + `AfterSideTurnEndLate`.

Combat-relevant dispatchers (full list of 147 obtainable via `grep -n "public static" Hooks/Hook.cs`):
- Attack/damage: `BeforeAttack`, `AfterAttack`, `BeforeDamageReceived`, `AfterDamageReceived`, `AfterDamageGiven`, `AfterCurrentHpChanged`, `ModifyDamage`, `ModifyHpLost`, `ModifyAttackHitCount`, `ModifyUnblockedDamageTarget`, `AfterModifyingDamageAmount`, `AfterModifyingHpLostBeforeOsty`/`AfterOsty`
- Block: `BeforeBlockGained`, `AfterBlockGained`, `AfterBlockBroken`, `AfterBlockCleared`, `AfterPreventingBlockClear`, `ModifyBlock`, `ShouldClearBlock`, `AfterModifyingBlockAmount`
- Cards: `BeforeCardPlayed`, `AfterCardPlayed`, `BeforeCardAutoPlayed`, `AfterCardDrawn`, `AfterCardDiscarded`, `AfterCardExhausted`, `AfterCardChangedPiles`, `AfterCardEnteredCombat`, `AfterCardGeneratedForCombat`, `AfterShuffle`, `ModifyShuffleOrder`, `ModifyCardPlayCount`, `ModifyEnergyCostInCombat`, `ModifyStarCost`, `ModifyKeywordsInCombat`, `ModifyXValue`, `ShouldPlay`, `ShouldDraw`, `AfterPreventingDraw`, `AfterHandEmptied`, `ShouldEtherealTrigger`
- Turn: `BeforeSideTurnStart`, `AfterSideTurnStart`, `AfterPlayerTurnStart`, `AfterAutoPrePlayPhaseEntered`, `AfterAutoPostPlayPhaseEntered`, `BeforeTurnEnd`, `AfterTurnEnd`, `BeforeFlush`, `AfterFlush`, `ShouldFlush`, `ShouldTakeExtraTurn`, `AfterTakingExtraTurn`, `BeforeHandDraw`, `ModifyHandDraw`
- Resources: `AfterEnergyReset`, `AfterEnergySpent`, `ModifyMaxEnergy`, `ModifyEnergyGain`, `ShouldPlayerResetEnergy`, `AfterStarsGained`, `AfterStarsSpent`, `ShouldGainStars`, `ShouldPayExcessEnergyCostWithStars`
- Powers/death: `BeforePowerAmountChanged`, `AfterPowerAmountChanged`, `ModifyPowerAmountGiven`, `ModifyPowerAmountReceived`, `ShouldAfflict`, `BeforeDeath`, `AfterDeath`, `ShouldDie`, `AfterPreventingDeath`, `ShouldPowerBeRemovedOnDeath`, `ShouldCreatureBeRemovedFromCombatAfterDeath`, `ShouldStopCombatFromEnding`
- Orbs/summons/combat lifecycle: `AfterOrbChanneled`, `AfterOrbEvoked`, `ModifyOrbValue`, `ModifyOrbPassiveTriggerCount`, `AfterSummon`, `ModifySummonAmount`, `BeforeCombatStart`, `AfterCombatEnd`, `AfterCombatVictory`, `AfterCreatureAddedToCombat`
- Potions: `BeforePotionUsed`, `AfterPotionUsed`, `AfterPotionDiscarded`, `AfterPotionProcured`, `ShouldProcurePotion`

**Listener order matters and is deterministic.** `Hook.IterateCombatHookListeners` yields nothing when `CombatManager.IsOverOrEnding && !IsStarting` (checked once at enumeration start). `RunState.IterateHookListeners(combatState)` (`Runs/RunState.cs:545`) yields: per player — all deck cards + their enchantments; then, **only if `childCombatState == null`**, relics/potions/modifiers/badges/`MultiplayerScalingModel`; then mod subscribers; then `childCombatState.IterateHookListeners()`.
`CombatState.IterateHookListeners()` (`Combat/CombatState.cs:410`) iterates allies then enemies, and for each creature: **its powers first**, then (monster → the `MonsterModel`; player → relics in order, non-null potion slots, orbs, then all cards in all 5 piles with their afflictions/enchantments), then Modifiers/Badges.

## 6. RNG

`Random/MegaRandom.cs` — **Xoshiro256\*\*** seeded through Splitmix64. `Random/Rng.cs` wraps it with a `Counter` (number of draws) so state serializes as `(Seed, Counter)` and replays via `FastForwardCounter`. `Rng.Chaotic` is a non-deterministic instance used only for cosmetics (skins, screen shake).

Two named sets:
- **`Runs/RunRngSet.cs`** — `string StringSeed` (display) hashed to `uint Seed`. 12 streams (`Entities/Rngs/RunRngType.cs`): `UpFront, Shuffle, UnknownMapPoint, CombatCardGeneration, CombatPotionGeneration, CombatCardSelection, CombatEnergyCosts, CombatTargets, MonsterAi, Niche, CombatOrbs, TreasureRoomRelics`. Each is `new Rng(Seed + hash(SnakeCase(name)))`.
- **`Random/PlayerRngSet.cs`** — per-player, 3 streams (`Entities/Rngs/PlayerRngType.cs`): `Rewards, Shops, Transformations`.

Combat-relevant assignments:
- **Shuffle**: `RunState.Rng.Shuffle`. Initial deck shuffle at `CombatManager.cs:370`; reshuffle in `CardPileCmd.Shuffle` (`list.StableShuffle(player.RunState.Rng.Shuffle)` then `Hook.ModifyShuffleOrder`); also random pile insertion at `CardPileCmd.cs:514`.
- **Card draw**: no RNG — `CardPileCmd.Draw` takes `drawPile.Cards.FirstOrDefault()`, calling `ShuffleIfNecessary` when the draw pile is empty.
- **Monster moves**: `RunRng.MonsterAi` — `Models/MonsterModel.cs:417`: `NextMove = MoveStateMachine.RollMove(targets, Creature, RunRng.MonsterAi);`
- **Random attack targets**: `RunState.Rng.CombatTargets` (`AttackCommand.cs:601`, `CardCmd.cs:77/90`).
- **Random card selection** (True Grit etc.): `CombatCardSelection`. **Random costs** (Snecko): `CombatEnergyCosts`. Generated cards/potions: `CombatCardGeneration` / `CombatPotionGeneration`.
- Per-monster cosmetic/HP roll: `CombatState.cs:244` builds a bespoke `Rng` from `RunState.Rng.Seed + mapCoord`; unique monster HP uses `Rng.Niche`.

`StableShuffle` (`Extensions/ListExtensions.cs:22`) sorts the list by `IComparable<T>` **first**, then runs a Fisher–Yates `UnstableShuffle` (descending `i`, `swap(i, rng.NextInt(i+1))`). Port both steps exactly or draw order will diverge.

## 7. Multiplayer

Combat code is multiplayer-aware throughout, but the coupling is shallow and can be reduced to a `playerCount == 1` fast path:

- **State**: `CombatState.Players` is a list; `CombatState.Allies` holds all player creatures. `PlayerCombatState.TurnNumber` exists per player precisely because turns can desync. With one player, `TurnNumber == RoundNumber` unless an extra-turn effect fires.
- **Turn machinery**: `_playersReadyToEndTurn`, `_playersReadyToBeginEnemyTurn`, the two-phase turn end (`EndPlayerTurnPhaseOneInternal` / `PhaseTwoInternal`), `ReadyToBeginEnemyTurnAction`, and `ActionQueueSynchronizer` all exist to rendezvous multiple players. Single-player collapses this to: phase-one → phase-two → switch sides.
- **PlayerChoiceContext**: `HookPlayerChoiceContext` exists so one player's paused effect doesn't block others. Single-player can use `BlockingPlayerChoiceContext` semantics everywhere (it is already the fallback in `AttackCommand`).
- **Net types**: `GameActions/Net*Action.cs`, `Entities/Multiplayer/`, `Multiplayer/` (transport, replay, serialization, `CombatStateSynchronizer`), `NetCombatCardDb`. All ignorable.
- **Actual rules differences** (must not be ignored if you ever sim >1 player, but are no-ops at 1 player):
  - `Models/Singleton/MultiplayerScalingModel.cs` — `ModifyBlockMultiplicative` returns `1m` when `_runState.Players.Count == 1`, else `count * GetMultiplayerScaling(encounter, actIndex)` (1.1 / 1.2 / …).
  - `PowerCmd.Apply` — `if (combatState.Players.Count > 1 && (target.IsPrimaryEnemy || target.IsSecondaryEnemy) && power.ShouldScaleInMultiplayer) modifiedAmount = power.GetScaledAmountForMultiplayer(...)`.
  - `Creature.ScaleMonsterHpForMultiplayer` / `ScaleHpForMultiplayer(hp, encounter, playerCount, actIndex)`.
  - `CardMultiplayerConstraint` on cards.
  - `MultiplayerScalingModel` is only in the listener list when `childCombatState == null`, i.e. out-of-combat dispatches — but it sets `ShouldReceiveCombatHooks => true` and is reachable via `CombatState.IterateHookListeners`'s singleton/modifier tail.

Verdict: a single-player sim can drop `Multiplayer/`, `Entities/Multiplayer/`, all `Net*Action`, `ActionQueueSynchronizer`, `PlayerChoiceSynchronizer`, and `ChecksumTracker` entirely, and hard-code `playerCount = 1` in the three scaling call sites.
