# Card model survey — Slay the Spire 2 (v0.107.1)

Root: `decompiled/MegaCrit/Sts2/Core`

---

## 1. Class hierarchy: model vs. runtime instance

**There is no separate runtime card entity class.** `CardModel` is both the definition and the instance. The split is *canonical (immutable) vs. mutable clone*:

- `Core/Models/CardModel.cs` — `public abstract class CardModel : AbstractModel` (2265 lines). Every card is a `sealed class X : CardModel` in `Core/Models/Cards/` (~800 files).
- `ModelDb.Card<StrikeIronclad>()` returns the singleton **canonical** instance. `CardModel.ToMutable()` (line 1187) calls `MutableClone()` to produce a per-run/per-combat instance. `AssertMutable()` guards every mutating setter; `IsCanonical` short-circuits cost/keyword computation.
- `CanonicalInstance` (line 990) gives a mutable card back its definition. `DeckVersion` links a combat card back to the deck card it was cloned from.
- `Core/Entities/Cards/` holds only **enums and helper value types** (no card class): `CardType`, `CardRarity`, `TargetType`, `CardKeyword`, `CardTag`, `PileType`, `CardPile`, `CardEnergyCost`, `CardPlay`, `LocalCostModifier`, `ResourceInfo`, `UnplayableReason`, `CardScope`.

Constructor signature (line 1086) carries the four core static fields:

```csharp
protected CardModel(int canonicalEnergyCost, CardType type, CardRarity rarity,
                    TargetType targetType, bool shouldShowInCardLibrary = true)
```

Fields of interest:

| Concept | Where |
|---|---|
| cost | `EnergyCost` (`CardEnergyCost`), `CanonicalStarCost`/`BaseStarCost`/`CurrentStarCost` |
| type / rarity / target | `Type`, `Rarity`, `TargetType` (virtual, set via ctor) |
| keywords | `CanonicalKeywords` (virtual) → `LocalKeywords` set → `Keywords` (local + global) |
| tags (behaviorless) | `CanonicalTags` → `Tags` |
| upgrade | `CurrentUpgradeLevel`, `MaxUpgradeLevel` (default 1, some cards >1), `IsUpgraded`, `IsUpgradable`, `UpgradeInternal()`/`FinalizeUpgradeInternal()`/`DowngradeInternal()`, `OnUpgrade()` override |
| dynamic numbers | `DynamicVars` (`DynamicVarSet`), `CanonicalVars` |
| enchantments | `Enchantment` (`EnchantmentModel?`), `Affliction` (`AfflictionModel?`) |
| replay | `BaseReplayCount`, `GetEnchantedReplayCount()` |
| identity | `Id` (`ModelId` = `"category.entry"`), `CloneOf`/`IsClone`, `DupeOf`/`IsDupe`, `FloorAddedToDeck` |
| misc flags | `GainsBlock`, `OrbEvokeType`, `IsBasicStrikeOrDefend`, `IsRemovable` (= not Eternal), `CanBeGeneratedInCombat`, `HasTurnEndInHandEffect` |

`Pile` is derived, not stored: `_owner?.Piles.FirstOrDefault(p => p.Cards.Contains(this))`. `CardPile` (`Entities/Cards/CardPile.cs`) holds `IReadOnlyList<CardModel> Cards`; `MaxCardsInHand => 10`.

**Deck → combat:** `PileType.Deck` doc: *"When a new combat starts, all cards in here are cloned into your draw pile, so modifications to cards in combat won't modify cards in here."* Scope is tracked by `CardScope` enum (`None` / `Run` / `Combat`) and `ICardScope`.

**Dynamic vars** (`Core/Localization/DynamicVars/`): `DynamicVar` is a named `decimal` with four layers — `BaseValue` (authoritative for game logic), `EnchantedValue`, `PreviewValue` (display only), plus `WasJustUpgraded`. Subclasses: `DamageVar`, `BlockVar`, `PowerVar<T>`, `RepeatVar`, `EnergyVar`, `StarsVar`, `HealVar`, `GoldVar`, `MaxHpVar`, `HpLossVar`, `CardsVar`, `SummonVar`, `OstyDamageVar`, `CalculatedDamageVar`, `IfUpgradedVar`, `BoolVar`, `StringVar`, `ForgeVar`. `DynamicVarSet` exposes `.Damage`, `.Block`, `.Repeat`, and arbitrary `["Name"]` lookup. These double as localization format arguments — the card text is generated from the same numbers.

**Serialization:** `ToSerializable()` / `FromSerializable()` persist `{ Id, CurrentUpgradeLevel, Props, Enchantment, FloorAddedToDeck }` only — everything else is recomputed.

---

## 2. How effects are expressed

`protected virtual Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay)` (line 1643). It is **async/await against command helpers**, not an action queue that cards push into. The queue exists one level up (`RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(this, target))`), and `OnPlay` runs inside the dequeued action. Commands: `DamageCmd`, `PowerCmd`, `CreatureCmd`, `CardCmd`, `CardPileCmd`, `CardSelectCmd`, `SfxCmd`.

`OnPlayWrapper` (line ~1855) is the full pipeline: move to `PileType.Play` → `Hook.ModifyCardPlayResultPileTypeAndPosition` → `GeneratePlayCount` → loop { `Hook.BeforeCardPlayed`, `OnPlay`, `Enchantment.OnPlay`, `Affliction.OnPlay`, `Hook.AfterCardPlayed` } → move to result pile → `EnergyCost.AfterCardPlayedCleanup()`.

### a) Strike (`Models/Cards/StrikeIronclad.cs`) — simplest

```csharp
public sealed class StrikeIronclad : CardModel
{
	protected override HashSet<CardTag> CanonicalTags => new HashSet<CardTag> { CardTag.Strike };
	protected override IEnumerable<DynamicVar> CanonicalVars => ...(new DamageVar(6m, ValueProp.Move));
	public StrikeIronclad() : base(1, CardType.Attack, CardRarity.Basic, TargetType.AnyEnemy) { }

	protected override async Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		ArgumentNullException.ThrowIfNull(cardPlay.Target, "cardPlay.Target");
		await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this).Targeting(cardPlay.Target)
			.WithHitFx("vfx/vfx_attack_slash").Execute(choiceContext);
	}
	protected override void OnUpgrade() { base.DynamicVars.Damage.UpgradeValueBy(3m); }
}
```

### b) Bash (`Models/Cards/Bash.cs`) — damage + debuff, two vars

```csharp
protected override IEnumerable<DynamicVar> CanonicalVars => new DynamicVar[2] {
	new DamageVar(8m, ValueProp.Move), new PowerVar<VulnerablePower>(2m) };
public Bash() : base(2, CardType.Attack, CardRarity.Basic, TargetType.AnyEnemy) { }

protected override async Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay) {
	await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this).Targeting(cardPlay.Target)
		.WithHitFx("vfx/vfx_attack_blunt", null, "blunt_attack.mp3").Execute(choiceContext);
	await PowerCmd.Apply<VulnerablePower>(choiceContext, cardPlay.Target,
		base.DynamicVars.Vulnerable.BaseValue, base.Owner.Creature, this);
}
protected override void OnUpgrade() {
	base.DynamicVars.Damage.UpgradeValueBy(2m); base.DynamicVars.Vulnerable.UpgradeValueBy(1m); }
```

Note `DynamicVars.Vulnerable` — `PowerVar<VulnerablePower>` auto-names itself `"VulnerablePower"`; `Dominate` reads it as `DynamicVars["VulnerablePower"]`.

### c) Whirlwind (`Models/Cards/Whirlwind.cs`) — **X-cost + AoE**

```csharp
protected override bool HasEnergyCostX => true;
protected override IEnumerable<DynamicVar> CanonicalVars => ...(new DamageVar(5m, ValueProp.Move));
public Whirlwind() : base(0, CardType.Attack, CardRarity.Uncommon, TargetType.AllEnemies) { }

protected override async Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay) {
	int num = ResolveEnergyXValue();
	if (num > 0) { /* vfx, sfx */ }
	await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).WithHitCount(num).FromCard(this)
		.TargetingAllOpponents(base.CombatState).WithHitFx("vfx/vfx_giant_horizontal_slash")
		.Execute(choiceContext);
}
```

### d) Feel No Pain (`Models/Cards/FeelNoPain.cs`) — power card, custom-named var

```csharp
protected override IEnumerable<DynamicVar> CanonicalVars => ...(new DynamicVar("Power", 3m));
public FeelNoPain() : base(1, CardType.Power, CardRarity.Uncommon, TargetType.Self) { }
protected override async Task OnPlay(...) {
	await PowerCmd.Apply<FeelNoPainPower>(choiceContext, base.Owner.Creature,
		base.DynamicVars["Power"].BaseValue, base.Owner.Creature, this);
}
```

Power cards need no `Exhaust`: `GetResultPileTypeForCardPlay()` returns `PileType.None` for `Type == CardType.Power`.

### e) Armaments (`Models/Cards/Armaments.cs`) — **player choice**

```csharp
public override bool GainsBlock => true;
protected override async Task OnPlay(PlayerChoiceContext choiceContext, CardPlay cardPlay) {
	await CreatureCmd.GainBlock(base.Owner.Creature, base.DynamicVars.Block, cardPlay);
	if (base.IsUpgraded) {
		foreach (CardModel item in PileType.Hand.GetPile(base.Owner).Cards.Where(c => c.IsUpgradable))
			CardCmd.Upgrade(item);
		return;
	}
	CardModel cardModel = await CardSelectCmd.FromHandForUpgrade(choiceContext, base.Owner, this);
	if (cardModel != null) CardCmd.Upgrade(cardModel);
}
```

### f) Headbutt (`Models/Cards/Headbutt.cs`) — targeted attack + discard-pile selection

```csharp
CardModel cardModel = (await CardSelectCmd.FromCombatPile(
	prefs: new CardSelectorPrefs(base.SelectionScreenPrompt, 1), context: choiceContext,
	pile: PileType.Discard.GetPile(base.Owner), player: base.Owner)).FirstOrDefault();
if (cardModel != null) await CardPileCmd.Add(cardModel, PileType.Draw, CardPilePosition.Top);
```

Player choice is **awaited inline** via `PlayerChoiceContext` — a Rust port needs the equivalent of a suspendable effect or a pre-resolved choice list.

---

## 3. Registration, ids, pools

- Id: `ModelId { Category, Entry }`, serialized as `"category.entry"` (`Models/ModelId.cs`). Category derived from the C# type name; `Entry` drives localization keys (`cards/<entry>.title`) and art paths.
- `ModelDb` (`Models/ModelDb.cs`): `ModelDb.Card<T>()` (line 545), `ModelDb.AllCards` (line 92) = union of all pools + all character starting decks, `ModelDb.AllCardPools` (line 98) = character pools + shared pools.
- `CardPoolModel` (`Models/CardPoolModel.cs`): abstract `Title`, `EnergyColorName`, `CardFrameMaterialPath`, `DeckEntryCardColor`, `IsColorless`; `protected abstract CardModel[] GenerateAllCards()`; `AllCardIds`; `GetUnlockedCards(UnlockState, CardMultiplayerConstraint)` and `FilterThroughEpochs(...)` — pools are filtered by unlock state and by "epoch".
- `CardModel.Pool` resolves lazily by scanning `ModelDb.AllCardPools` for one whose `AllCardIds` contains this id; throws if none. `VisualCardPool` can differ (e.g. Trash Heap event cards rendered in their original character colors).

Pool sizes (`grep 'new CardModel\[' Models/CardPools/*.cs`):

| Pool | Count |
|---|---|
| **Ironclad** | **87** |
| Silent / Defect / Regent / Necrobinder | 88 each |
| Colorless | 64 |
| Event | 27 · Curse 18 · Token 14 · Status 12 · Quest 3 · Mock 12 · Deprecated 1 |

Characters: Ironclad, Silent, Defect, Regent, Necrobinder (no Watcher pool present).

List the Ironclad cards: `sed -n '26,120p' Models/CardPools/IroncladCardPool.cs`, or
`rg -o 'ModelDb\.Card<(\w+)>' -r '$1' Models/CardPools/IroncladCardPool.cs`. Rarity/cost/type per card live in each `Models/Cards/<Name>.cs` constructor, so a scraper can regex `: base\((\d+), CardType\.(\w+), CardRarity\.(\w+), TargetType\.(\w+)` per file.

---

## 4. Cost mechanics

**Two resources: energy and stars.** `PlayerCombatState.Energy` and `.Stars`.

`CardEnergyCost` (`Entities/Cards/CardEnergyCost.cs`) is the whole energy story:
- `Canonical` (printed cost), `_base` (canonical after permanent upgrades), `CostsX`.
- `GetWithModifiers(CostModifiers)` — `None` / `Local` (list of `LocalCostModifier` on the card) / `Global` (`Hook.ModifyEnergyCostInCombat`, e.g. Enthralled) / `All`. Early-outs: canonical instance, `_base < 0` (unplayable/no-cost), `CostsX`. Final `Math.Max(0, …)`.
- `LocalCostModifier { Amount, Type (Absolute|Relative), Expiration (EndOfCombat|EndOfTurn|WhenPlayed, flags), IsReduceOnly }`, applied **in insertion order**. `IsReduceOnly` = the modifier is skipped if it would raise the cost (used for any effect worded "Reduce"). See the worked example in the file header.
- Setters: `SetUntilPlayed`, `SetThisTurnOrUntilPlayed` (the usual "costs 0 this turn"), `SetThisTurn`, `SetThisCombat`, and `Add*` equivalents. `CardModel.SetToFreeThisTurn()` / `SetToFreeThisCombat()` zero both energy and stars.
- Cleanup: `EndOfTurnCleanup()` and `AfterCardPlayedCleanup()` remove by expiration flag.
- `GetAmountToSpend()`: for X cards returns owner's whole `Energy`; else clamped modified cost. `GetResolved()`: for X cards returns `CapturedXValue`.
- `UpgradeBy(addend)` clamps existing Absolute local modifiers down to the new base.

X-cost: `HasEnergyCostX => true`, canonical cost written as 0. On play `SpendEnergy` sets `EnergyCost.CapturedXValue = amount`; effects call `ResolveEnergyXValue()` = `Hook.ModifyXValue(CombatState, this, CapturedXValue)` (Chemical X). Star analogue: `HasStarCostX`, `LastStarsSpent`, `ResolveStarXValue()` — `Models/Cards/Stardust.cs`.

Stars use a simpler model: `BaseStarCost` + a **stack** of `TemporaryCardCost` (`UntilPlayed` / `ThisTurn` / `ThisCombat`), where `CurrentStarCost` takes the *last* one. Negative `CanonicalStarCost` (default `-1`) means "no star cost" and a temporary 0 does not make a star cost appear.

Playability, `CardModel.CanPlay(out UnplayableReason, out AbstractModel? preventer)` (line 1734) ORs flags:
`HasUnplayableKeyword` (keyword `Unplayable`), resource shortfall from `PlayerCombatState.HasEnoughResourcesFor` (`EnergyCostTooHigh` | `StarCostTooHigh`), `NoLivingAllies` (when `TargetType.AnyAlly` and ≤1 living player creature), `BlockedByHook` (`Hook.ShouldPlay`, e.g. Normality), `BlockedByCardLogic` (card overrides `protected virtual bool IsPlayable => true`, e.g. Grand Finale).

`HasEnoughResourcesFor` includes a conversion rule: if `Hook.ShouldPayExcessEnergyCostWithStars`, excess energy is paid **2 stars per 1 energy**.

Unplayable cards attempted anyway go through `MoveToResultPileWithoutPlaying` (power cards go to Discard rather than limbo).

---

## 5. Targeting

`Entities/Cards/TargetType.cs`: `None, Self, AnyEnemy, AllEnemies, RandomEnemy, AnyPlayer, AnyAlly, AllAllies, TargetedNoCreature, Osty`.

Only `AnyEnemy` and `AnyAlly` involve UI target selection (`AnyPlayer`/`AnyAlly` select only in multiplayer). `IsValidTarget(Creature?)`: null target is valid unless `AnyEnemy`/`AnyAlly`; otherwise target must be alive and on the opposite side (`AnyEnemy`) or same side (`AnyAlly`). Everything else resolves with `cardPlay.Target == null`.

Multi/random targeting is expressed in the **command builder**, not the enum: `DamageCmd.Attack(x).Targeting(creature)` / `.TargetingAllOpponents(combatState)` / `.TargetingRandomOpponents(combatState)`, plus `.WithHitCount(n)` for multi-hit (Whirlwind, Sword Boomerang, Stardust). `TargetType` mainly drives highlight/UI and playability.

---

## 6. Surprises for a StS1 veteran

- **Stars**: a second per-card resource alongside energy, with its own X-cost, its own temporary-cost stack, and a 1 energy → 2 stars fallback conversion.
- **New keywords**: `Sly` — the card is **auto-played when discarded** (`CardCmd` line 203: discarded Sly cards get `AutoPlay(..., AutoPlayType.SlyDiscard)`); `Eternal` — cannot be removed from the deck (`IsRemovable => !Keywords.Contains(Eternal)`). Full set: `Exhaust, Ethereal, Innate, Unplayable, Retain, Sly, Eternal`. `Retain` and `Sly` also have single-turn variants (`GiveSingleTurnRetain`, `GiveSingleTurnSly`), cleared in `EndOfTurnCleanup`.
- **Local vs. global keywords** (`KeywordSources`), mirroring cost modifiers: Music Box adds Ethereal *locally* (persists); Hex grants it *globally* (computed on demand, disappears with the power). Global keywords are never stored.
- **Replay**: `BaseReplayCount` + `Hook.ModifyCardPlayCount` means a single card play can execute `OnPlay` N times with `CardPlay.PlayIndex/PlayCount/IsFirstInSeries/IsLastInSeries`. Card code must handle being re-entered.
- **Enchantments and Afflictions** attached to individual cards (`EnchantmentModel`, `AfflictionModel`), each with their own `OnPlay`, dynamic vars, and `ModifyCard()` — persistent per-card modifiers with their own serialization.
- **Clones vs. dupes**: `CreateDupe()` returns to `PileType.None` (limbo) after play, strips Exhaust, cannot be re-duped, and inherits the original's X value.
- **Multi-upgrade cards**: `MaxUpgradeLevel` can exceed 1; titles render as `Name+2`.
- **`PileType.Play`** is a real pile (limbo) a card occupies mid-resolution; `PileType.None` means removed from combat.
- **Multiplayer is pervasive**: `CardMultiplayerConstraint { None, MultiplayerOnly, SingleplayerOnly }` gates pool membership; `PlayerChoiceContext` and `ActionQueueSynchronizer` exist for networked choice arbitration; `TargetType.AnyPlayer/AnyAlly/AllAllies`; `Owner` is a `Player`, so all pile lookups are per-player.
- **Osty**: a pet creature with its own targeting (`TargetType.Osty`, `CardTag.OstyAttack`, `OstyDamageVar`) and auto-targeting driven by `GainsBlock`.
- **Orbs** survive (`OrbEvokeType`), and there's a `CardType.Quest` plus `CardRarity.Ancient/Event/Token/Quest`.
- Decimal, not int: `DynamicVar` values are `decimal` clamped to 999,999,999.
- `CardPile.MaxCardsInHand => 10` is unchanged.
