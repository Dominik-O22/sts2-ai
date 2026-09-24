//! Powers (statuses) on a creature. `Models/PowerModel.cs` plus the individual
//! `Models/Powers/*.cs`. Each hook below is one `AbstractModel` virtual,
//! implemented as a match over `PowerId` so dispatch is a jump table.
//! Hooks that only read state return effects; hooks that keep per-power
//! counters (`InitInternalData` in the game) mutate `data`.

use crate::effect::{Effect, Pile};
use crate::ids::PowerId;
use crate::types::{CardType, CreatureRef, Side, ValueProp};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Power {
    pub id: PowerId,
    pub amount: i32,
    /// `PowerModel.SkipNextDurationTick`. Set when a debuff lands on the
    /// player so it does not tick down the same round it was applied.
    pub skip_next_tick: bool,
    /// Per-power counter: Dark Embrace ethereal count, Juggling attacks this
    /// turn, Crimson Mantle and Inferno self-damage, Slow cards played,
    /// Panache's cards left, The Bomb's damage.
    pub data: i32,
    /// `PowerModel.Applier`. Constrict and Shrink vanish when it dies.
    pub applier: Option<CreatureRef>,
}

impl Power {
    pub fn new(id: PowerId, amount: i32) -> Self {
        Self { id, amount, skip_next_tick: false, data: 0, applier: None }
    }

    /// The number on the power's icon (`PowerModel.DisplayAmount`): the
    /// amount, but for the powers that show a count kept in `data`.
    pub fn display_amount(&self) -> i32 {
        match self.id {
            // AutomationPower.cs: cardsLeft, 10 down to 1; `data` counts up.
            PowerId::Automation => 10 - self.data,
            // HardenedShellPower.cs: what it still blocks this turn.
            PowerId::HardenedShell => (self.amount - self.data).max(0),
            // PanachePower.cs: CardsLeft, 5 until the Panache is played.
            PowerId::Panache if self.data == 0 => PANACHE_CARDS,
            PowerId::Panache => self.data,
            // SlothPower.cs, TenderPower.cs: cards played this turn.
            PowerId::Sloth | PowerId::Tender => self.data,
            // SlowPower.cs: SlowAmount * 10, the percent more damage taken.
            PowerId::Slow => self.data * 10,
            // TagTeamPower.cs shows 1 whatever it holds.
            PowerId::TagTeam => 1,
            // WitheringPresencePower.cs: CardsLeft to the next Wither.
            PowerId::WitheringPresence => self.amount - self.data,
            _ => self.amount,
        }
    }
}

/// `PowerModel.Type == Debuff`.
pub fn is_debuff(id: PowerId) -> bool {
    matches!(
        id,
        PowerId::Vulnerable
            | PowerId::Weak
            | PowerId::Frail
            | PowerId::NoDraw
            | PowerId::NoEnergyGain
            | PowerId::Mangle
            | PowerId::Tangled
            | PowerId::Slow
            | PowerId::Shrink
            | PowerId::Ringing
            | PowerId::Plow
            | PowerId::Constrict
            | PowerId::Demise
            | PowerId::ShacklingPotion
            | PowerId::Shriek
            | PowerId::Smoggy
            | PowerId::Confused
            | PowerId::Imbalanced
            | PowerId::Tender
            | PowerId::Tainted
            | PowerId::Surrounded
            | PowerId::Disintegration
            | PowerId::MindRot
            | PowerId::Sloth
            | PowerId::WasteAway
            | PowerId::Hex
            | PowerId::Dampen
            | PowerId::ChainsOfBinding
            | PowerId::NoBlock
            | PowerId::TagTeam
            | PowerId::TheGambit
            | PowerId::DarkShackles
            | PowerId::Knockdown
    )
}

/// `PowerInstanceType.Instanced`: every application is a separate instance
/// with its own amount and counter, so the combat never stacks these.
pub fn instanced(id: PowerId) -> bool {
    matches!(
        id,
        PowerId::Panache | PowerId::RollingBoulder | PowerId::TheBomb | PowerId::TagTeam | PowerId::Automation | PowerId::Knockdown
    )
}

/// `PanachePower._baseCardsLeft`.
const PANACHE_CARDS: i32 = 5;

/// `PowerModel.AllowNegative`.
pub fn allow_negative(id: PowerId) -> bool {
    matches!(id, PowerId::Strength | PowerId::Dexterity | PowerId::Shrink | PowerId::Shriek)
}

/// `PowerModel.GetTypeForAmount == Debuff`: a negative counter that allows
/// negatives (Strength loss, the permanent Shrink) counts as a debuff, and a
/// negative amount of a plain debuff counts as a buff. All three
/// `allow_negative` powers are `Counter` stacks on their canonical model.
pub fn is_debuff_for_amount(id: PowerId, amount: i32) -> bool {
    if allow_negative(id) && amount < 0 {
        return true;
    }
    if !allow_negative(id) && is_debuff(id) && amount < 0 {
        return false;
    }
    is_debuff(id)
}

/// `TemporaryStrengthPower` / `TemporaryDexterityPower` subclasses: the
/// real power they apply and the sign.
pub fn temp_power(id: PowerId) -> Option<(PowerId, i32)> {
    match id {
        PowerId::SetupStrike | PowerId::FlexPotion | PowerId::FeedingFrenzy | PowerId::ReptileTrinket => {
            Some((PowerId::Strength, 1))
        }
        PowerId::Mangle | PowerId::ShacklingPotion | PowerId::DarkShackles => Some((PowerId::Strength, -1)),
        PowerId::SpeedPotion => Some((PowerId::Dexterity, 1)),
        _ => None,
    }
}

/// `TemporaryStrengthPower` subclasses only, with their sign.
pub fn temp_strength_sign(id: PowerId) -> Option<i32> {
    match temp_power(id) {
        Some((PowerId::Strength, sign)) => Some(sign),
        _ => None,
    }
}

impl Power {
    /// `PowerModel.ShouldRemoveDueToAmount`.
    pub fn should_remove(&self) -> bool {
        if allow_negative(self.id) {
            self.amount == 0
        } else {
            self.amount <= 0
        }
    }

    /// `ModifyDamageAdditive`. `owner` is the creature this power sits on.
    pub fn modify_damage_additive(&self, owner: CreatureRef, target: CreatureRef, dealer: Option<CreatureRef>, props: ValueProp) -> f64 {
        match self.id {
            // TaintedPower.cs: powered attacks on the owner hit harder.
            PowerId::Tainted if target == owner && props.is_powered() => self.amount as f64,
            // StrengthPower.cs
            PowerId::Strength if dealer == Some(owner) && props.is_powered() => self.amount as f64,
            // VigorPower.cs: the combat loop spends it after the attack.
            PowerId::Vigor if dealer == Some(owner) && props.is_powered() => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `ModifyDamageMultiplicative`. `dealer_vulnerable` and `dealer_cruelty`
    /// are the dealer's power amounts, read by Colossus and Vulnerable.
    pub fn modify_damage_multiplicative(
        &self,
        owner: CreatureRef,
        target: CreatureRef,
        dealer: Option<CreatureRef>,
        props: ValueProp,
        dealer_vulnerable: i32,
        dealer_cruelty: i32,
    ) -> f64 {
        if !props.is_powered() {
            return 1.0;
        }
        match self.id {
            // VulnerablePower.cs: 1.5 on the target, plus Cruelty/100 from the dealer.
            PowerId::Vulnerable if target == owner => 1.5 + dealer_cruelty as f64 / 100.0,
            // WeakPower.cs: 0.75 on the dealer.
            PowerId::Weak if dealer == Some(owner) => 0.75,
            // ColossusPower.cs: halve damage from Vulnerable dealers.
            PowerId::Colossus if target == owner && dealer.is_some() && dealer_vulnerable > 0 => 0.5,
            // SlowPower.cs: +10% per card played this turn.
            PowerId::Slow if target == owner => 1.0 + 0.1 * self.data as f64,
            // ShrinkPower.cs: the owner deals 30% less.
            PowerId::Shrink if dealer == Some(owner) => 0.7,
            // DiamondDiademPower.cs: halve powered attacks on the owner.
            PowerId::DiamondDiadem if target == owner => 0.5,
            // FlutterPower.cs: DamageDecrease 50.
            PowerId::Flutter if target == owner => 0.5,
            // SoarPower.cs: DamageDecrease 50.
            PowerId::Soar if target == owner => 0.5,
            // KnockdownPower.cs: everyone but whoever knocked it down hits
            // it `amount` times as hard. Alone that is nobody.
            PowerId::Knockdown if target == owner && dealer != self.applier => self.amount as f64,
            _ => 1.0,
        }
    }

    /// `ModifyBlockAdditive`. `source_owner` is the owner of the card that
    /// grants the block, or the target itself for monster moves.
    /// `defend_source` is false only for block from a card without the
    /// Defend tag.
    pub fn modify_block_additive(&self, owner: CreatureRef, source_owner: CreatureRef, props: ValueProp, defend_source: bool) -> f64 {
        match self.id {
            // DexterityPower.cs
            PowerId::Dexterity if source_owner == owner && props.is_powered() => self.amount as f64,
            // FastenPower.cs: the owner's powered block, when it comes from
            // a Defend or from no card at all.
            PowerId::Fasten if source_owner == owner && props.is_powered() && defend_source => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `ModifyBlockMultiplicative`. `block_plays_this_turn` counts the
    /// player's card plays that already gained block this turn (Unmovable);
    /// `from_card` is whether a card is the source.
    pub fn modify_block_multiplicative(
        &self,
        owner: CreatureRef,
        target: CreatureRef,
        props: ValueProp,
        block_plays_this_turn: usize,
        from_card: bool,
    ) -> f64 {
        match self.id {
            // NoBlockPower.cs: no block from cards, unless unpowered.
            PowerId::NoBlock if target == owner && !props.has(ValueProp::UNPOWERED) && from_card => 0.0,
            // FrailPower.cs
            PowerId::Frail if target == owner && props.is_powered() => 0.75,
            // UnmovablePower.cs: the first `amount` block gains from cards or
            // moves each turn are doubled.
            PowerId::Unmovable
                if target == CreatureRef::Player
                    && props.has(ValueProp::MOVE)
                    && block_plays_this_turn < self.amount as usize =>
            {
                2.0
            }
            _ => 1.0,
        }
    }

    /// `ModifyEnergyGain`.
    pub fn modify_energy_gain(&self, amount: i32) -> i32 {
        match self.id {
            PowerId::NoEnergyGain => 0,
            _ => amount,
        }
    }

    /// `ModifyMaxEnergy`.
    pub fn modify_max_energy(&self, amount: i32) -> i32 {
        match self.id {
            PowerId::Pyre => amount + self.amount,
            PowerId::WasteAway => amount - self.amount,
            _ => amount,
        }
    }

    /// `ModifyHandDraw`: Clarity draws one more each turn, Mind Rot fewer.
    pub fn modify_hand_draw(&self, count: u32) -> u32 {
        match self.id {
            PowerId::Clarity => count + 1,
            PowerId::MindRot => count.saturating_sub(self.amount.max(0) as u32),
            // DrawCardsNextTurnPower.cs: only once a turn has started with it.
            PowerId::DrawCardsNextTurn if self.data != 0 => count + self.amount.max(0) as u32,
            _ => count,
        }
    }

    /// `AfterEnergyReset`: Radiance grants energy, then decrements.
    pub fn after_energy_reset(&self, owner: CreatureRef) -> Vec<Effect> {
        match self.id {
            PowerId::Radiance => vec![Effect::GainEnergy { amount: 1 }, Effect::DecrementPower { target: owner, id: self.id }],
            // EnergyNextTurnPower.cs
            PowerId::EnergyNextTurn => vec![Effect::GainEnergy { amount: self.amount }, Effect::RemovePower { target: owner, id: self.id }],
            _ => vec![],
        }
    }

    /// `AfterBlockCleared` on the owner: Self-Forming Clay and Block Next Turn.
    pub fn after_block_cleared(&self, owner: CreatureRef) -> Vec<Effect> {
        match self.id {
            PowerId::SelfFormingClay | PowerId::BlockNextTurn => vec![
                Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None },
                Effect::RemovePower { target: owner, id: self.id },
            ],
            // ToricToughnessPower.cs: this instance's block. Its decrement is
            // per instance, so `Combat::tick_toric` does it.
            PowerId::ToricToughness => {
                vec![Effect::GainBlock { target: owner, amount: self.data as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            _ => vec![],
        }
    }

    /// `ShouldFlush`: Retain Hand keeps the hand.
    pub fn should_flush(&self) -> bool {
        self.id != PowerId::RetainHand
    }

    /// `ShouldClearBlock`.
    pub fn should_clear_block(&self, owner: CreatureRef, creature: CreatureRef) -> bool {
        !(matches!(self.id, PowerId::Barricade | PowerId::Burrowed) && owner == creature)
    }

    /// `ShouldDraw`: NoDraw blocks everything but the turn-start hand draw.
    pub fn should_draw(&self, from_hand_draw: bool) -> bool {
        !(self.id == PowerId::NoDraw && !from_hand_draw)
    }

    /// `SmoggyPower.ShouldPlay`: a smogged card cannot be played. The rest of
    /// the hook lives in `Combat::hook_allows_play`.
    pub fn blocks_smogged(&self) -> bool {
        self.id == PowerId::Smoggy
    }

    /// `TryModifyEnergyCostInCombatLate`: Corruption makes skills free, Free
    /// Attack makes attacks in hand free.
    pub fn free_card(&self, ty: CardType) -> bool {
        match self.id {
            PowerId::Corruption => ty == CardType::Skill,
            PowerId::FreeAttack => ty == CardType::Attack,
            _ => false,
        }
    }

    /// `ModifyCardPlayCount`. Powers that add plays are decremented once
    /// per modified play (`AfterModifyingCardPlayCount`).
    pub fn extra_plays(&self, ty: CardType) -> u32 {
        match self.id {
            PowerId::OneTwoPunch if ty == CardType::Attack => 1,
            PowerId::Duplication => 1,
            _ => 0,
        }
    }

    /// `Creature.BeforeTurnStart` records `AmountOnTurnStart`; `data` holds
    /// it for the two powers that read it.
    pub fn before_turn_start(&mut self) {
        if matches!(self.id, PowerId::DrawCardsNextTurn | PowerId::HelloWorld) {
            self.data = self.amount;
        }
    }

    /// `BeforeHandDraw`, player powers.
    pub fn before_hand_draw(&self) -> Vec<Effect> {
        match self.id {
            // HelloWorldPower.cs: `AmountOnTurnStart` distinct Commons.
            PowerId::HelloWorld if self.data >= 1 => vec![Effect::GenerateRandom {
                pool: crate::effect::GenPool::IroncladCommon,
                count: self.data as u32,
                to: Pile::Hand,
                free_this_turn: false,
                distinct: true,
                upgraded: false,
            }],
            _ => vec![],
        }
    }

    /// `BeforeSideTurnStart`.
    pub fn before_side_turn_start(&self, owner: CreatureRef, side: Side, round: u32) -> Vec<Effect> {
        match self.id {
            // AggressionPower.cs
            PowerId::Aggression if owner.side() == side => vec![Effect::AggressionPull { count: self.amount as u32 }],
            // PlatingPower.cs: enemy plating blocks before the first player turn.
            PowerId::Plating if side == Side::Player && owner != CreatureRef::Player && round == 1 => {
                vec![Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            _ => vec![],
        }
    }

    /// `AfterSideTurnStart`. `turn` is the player's turn number.
    pub fn after_side_turn_start(&self, owner: CreatureRef, side: Side, turn: u32, round: u32) -> Vec<Effect> {
        // RampartPower.cs: the Living Shield blocks for its turret as your
        // turn starts.
        if self.id == PowerId::Rampart && side == Side::Player {
            return vec![Effect::BlockMonsters { id: crate::ids::MonsterId::TurretOperator, amount: self.amount }];
        }
        if owner.side() != side {
            return vec![];
        }
        match self.id {
            // DemonFormPower.cs
            PowerId::DemonForm => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            // ClarityPower.cs
            PowerId::Clarity => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // DrawCardsNextTurnPower.cs: spent by the hand draw it grew.
            PowerId::DrawCardsNextTurn if self.data != 0 => vec![Effect::RemovePower { target: owner, id: self.id }],
            // PrepTimePower.cs
            PowerId::PrepTime => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Vigor,
                amount: self.amount,
                applier: Some(owner),
            }],
            // PlatingPower.cs: decrement each turn after the first.
            // SandpitPower.AfterSideTurnStartLate: one turn closer to being
            // eaten; its AfterRemoved kills the player outright.
            PowerId::Sandpit if side == Side::Enemy => {
                if self.amount <= 1 {
                    vec![Effect::RemovePower { target: owner, id: self.id }, Effect::Kill { target: CreatureRef::Player }]
                } else {
                    vec![Effect::DecrementPower { target: owner, id: self.id }]
                }
            }
            PowerId::Plating => {
                let first = match owner {
                    CreatureRef::Player => turn == 1,
                    _ => round == 1,
                };
                if first {
                    vec![]
                } else {
                    vec![Effect::DecrementPower { target: owner, id: PowerId::Plating }]
                }
            }
            _ => vec![],
        }
    }

    /// `AfterSideTurnStart` for powers that reset counters (Slow), plus
    /// `HardenedShellPower.BeforeSideTurnStart`, which clears its damage
    /// tally whichever side is starting.
    pub fn reset_at_side_turn_start(&mut self, owner: CreatureRef, side: Side) {
        if self.id == PowerId::Slow && owner.side() == side {
            self.data = 0;
        }
        if self.id == PowerId::HardenedShell {
            self.data = 0;
        }
        // SlothPower.BeforeSideTurnStart: a fresh allowance of plays.
        if self.id == PowerId::Sloth && owner.side() == side {
            self.data = 0;
        }
    }

    /// `AfterPlayerTurnStart`, player-side powers only.
    pub fn after_player_turn_start(&self, owner: CreatureRef) -> Vec<Effect> {
        let self_damage = || Effect::Damage {
            target: owner,
            amount: self.data as f64,
            props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED),
            dealer: Some(owner),
            card: None,
        };
        match self.id {
            // CrimsonMantlePower.cs: self damage grows per play, then block.
            PowerId::CrimsonMantle => vec![
                self_damage(),
                Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None },
            ],
            // InfernoPower.cs
            PowerId::Inferno => vec![self_damage()],
            // RollingBoulderPower.cs: the combat grows it by 5 once this is queued.
            PowerId::RollingBoulder => {
                vec![Effect::DamageAllEnemies { amount: self.amount as f64, props: ValueProp::UNPOWERED, dealer: owner }]
            }
            // EntropyPower.cs
            PowerId::Entropy => vec![Effect::TransformFromHand { count: self.amount.max(0) as u32 }],
            _ => vec![],
        }
    }

    /// `AfterAutoPrePlayPhaseEntered`, player powers: the turn is set up and
    /// the player has not acted yet.
    pub fn after_auto_pre_play(&self) -> Vec<Effect> {
        match self.id {
            // MayhemPower.cs
            PowerId::Mayhem => vec![Effect::AutoPlayFromDrawTop { count: self.amount.max(0) as u32, force_exhaust: false }],
            _ => vec![],
        }
    }

    /// `AfterCardDrawn` for the player's draws. `data` counts the draws.
    pub fn after_card_drawn(&mut self) -> Vec<Effect> {
        match self.id {
            // AutomationPower.cs: every tenth card drawn since this instance
            // arrived pays out `amount` energy and the count starts over.
            PowerId::Automation => {
                self.data += 1;
                if self.data >= 10 {
                    self.data = 0;
                    vec![Effect::GainEnergy { amount: self.amount }]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }

    /// `AfterAutoPostPlayPhaseEntered`: end of the player's turn, before the
    /// hand is discarded.
    pub fn after_auto_post_play(&self) -> Vec<Effect> {
        match self.id {
            // StampedePower.cs
            PowerId::Stampede => (0..self.amount).map(|_| Effect::AutoPlayRandomAttack).collect(),
            _ => vec![],
        }
    }

    /// `BeforeSideTurnEndVeryEarly`, which runs before the `Early` pass so
    /// Lagavulin's shell is gone before Plating can top it back up.
    pub fn before_side_turn_end_very_early(&self, owner: CreatureRef, side: Side) -> Vec<Effect> {
        match self.id {
            // AsleepPower: on the last sleeping turn, shed the Plating.
            PowerId::Asleep if owner.side() == side && self.amount <= 1 => {
                vec![Effect::RemovePower { target: owner, id: PowerId::Plating }]
            }
            _ => vec![],
        }
    }

    /// `BeforeSideTurnEndEarly`.
    pub fn before_side_turn_end_early(&self, owner: CreatureRef, side: Side) -> Vec<Effect> {
        match self.id {
            // PlatingPower.cs
            PowerId::Plating if owner.side() == side => {
                vec![Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            _ => vec![],
        }
    }

    /// `AfterSideTurnEnd`. Returns the effects the power queues. May reset
    /// per-turn counters.
    pub fn after_side_turn_end(&mut self, owner: CreatureRef, side: Side) -> Vec<Effect> {
        let own_side = owner.side() == side;
        let remove = || vec![Effect::RemovePower { target: owner, id: self.id }];
        match self.id {
            // Vulnerable/Weak/Frail: tick at end of the enemy side turn only.
            PowerId::Vulnerable | PowerId::Weak | PowerId::Frail if side == Side::Enemy => {
                vec![Effect::TickDuration { target: owner, id: self.id }]
            }
            // NoBlockPower.cs: plain decrement once the enemy turn is over.
            PowerId::NoBlock if side == Side::Enemy => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // PanachePower.cs: the count starts over each turn.
            PowerId::Panache if own_side => {
                if self.data != 0 {
                    self.data = PANACHE_CARDS;
                }
                vec![]
            }
            // ColossusPower.cs: plain decrement, no skip flag.
            PowerId::Colossus if side == Side::Enemy => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // IntangiblePower.cs: ticks on the enemy side's turn, whoever owns it.
            PowerId::Intangible if side == Side::Enemy => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // AsleepPower.cs: the nap runs out and the sleeper wakes; the move
            // graph's sleep branch reads the power being gone.
            PowerId::Asleep if own_side => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // SkittishPower.cs: the once-a-turn block is available again once
            // the player's turn is over.
            PowerId::Skittish if !own_side => {
                self.data = 0;
                vec![]
            }
            // TerritorialPower.cs: when the owner's side ends, gain Strength.
            PowerId::Territorial if own_side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            // Self-removing at the end of the owner's turn.
            PowerId::NoDraw
            | PowerId::NoEnergyGain
            | PowerId::OneTwoPunch
            | PowerId::Rage
            | PowerId::Tangled
            | PowerId::Ringing
            | PowerId::Rebound
                if own_side =>
            {
                remove()
            }
            // TenderPower.cs: the turn's Strength and Dexterity come back.
            PowerId::Tender if own_side => {
                let n = std::mem::take(&mut self.data);
                vec![
                    Effect::ApplyPower { target: owner, id: PowerId::Strength, amount: n, applier: self.applier },
                    Effect::ApplyPower { target: owner, id: PowerId::Dexterity, amount: n, applier: self.applier },
                ]
            }
            PowerId::Hatch if own_side => vec![Effect::DecrementPower { target: owner, id: self.id }],
            PowerId::EscapeArtist if own_side && self.amount > 1 => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // BattlewornDummyTimeLimitPower.cs: counts down, then the dummy escapes.
            PowerId::BattlewornDummyTimeLimit if own_side => {
                if self.amount > 1 {
                    vec![Effect::DecrementPower { target: owner, id: self.id }]
                } else {
                    vec![Effect::Escape { target: owner }]
                }
            }
            PowerId::Tainted if side == Side::Enemy => remove(),
            // SlumberPower.AfterSideTurnEnd: the nap runs out on its own and
            // the beetle wakes (WakeUpMove sheds the Plating).
            PowerId::Slumber if own_side => {
                let mut e = vec![Effect::DecrementPower { target: owner, id: self.id }];
                if self.amount <= 1 {
                    e.push(Effect::RemovePower { target: owner, id: PowerId::Plating });
                }
                e
            }
            // DisintegrationPower.AfterSideTurnEndLate.
            PowerId::Disintegration if own_side => vec![Effect::Damage {
                target: owner,
                amount: self.amount as f64,
                props: ValueProp::UNPOWERED,
                dealer: Some(owner),
                card: None,
            }],
            // ShrinkPower.cs: counts down unless permanent (-1).
            PowerId::Shrink if own_side && self.amount > 0 => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // ConstrictPower.cs: HP loss at the end of the owner's turn.
            PowerId::Constrict if own_side => vec![Effect::Damage {
                target: owner,
                amount: self.amount as f64,
                props: ValueProp::UNPOWERED,
                dealer: Some(owner),
                card: None,
            }],
            // FlameBarrierPower.cs: removed when the *other* side's turn ends.
            PowerId::FlameBarrier if !own_side => remove(),
            // DiamondDiademPower.cs: gone once the enemy turn is over.
            PowerId::DiamondDiadem if side == Side::Enemy => remove(),
            // TemporaryStrengthPower / TemporaryDexterityPower: remove self
            // and undo the real power.
            PowerId::SetupStrike
            | PowerId::Mangle
            | PowerId::FlexPotion
            | PowerId::ShacklingPotion
            | PowerId::SpeedPotion
            | PowerId::FeedingFrenzy
            | PowerId::DarkShackles
            | PowerId::ReptileTrinket
                if own_side =>
            {
                let (real, sign) = temp_power(self.id).unwrap();
                vec![
                    Effect::RemovePower { target: owner, id: self.id },
                    Effect::ApplyPower { target: owner, id: real, amount: -sign * self.amount, applier: Some(owner) },
                ]
            }
            // RegenPower.cs
            PowerId::Regen if own_side => vec![
                Effect::Heal { target: owner, amount: self.amount as f64 },
                Effect::DecrementPower { target: owner, id: self.id },
            ],
            // DemisePower.cs
            PowerId::Demise if own_side => vec![Effect::Damage {
                target: owner,
                amount: self.amount as f64,
                props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED),
                dealer: None,
                card: None,
            }],
            // RitualPower.cs: `WasJustAppliedByEnemy` costs the cultists their
            // first tick. Mazaleth's Gift puts it on the player, which does not.
            PowerId::Ritual if own_side && self.data == 1 => {
                self.data = 0;
                vec![]
            }
            PowerId::Ritual if own_side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            PowerId::RetainHand if own_side => vec![Effect::DecrementPower { target: owner, id: self.id }],
            PowerId::Duplication if own_side => remove(),
            // KnockdownPower.cs: gone once its owner's turn is over.
            PowerId::Knockdown if own_side => remove(),
            // DarkEmbracePower.cs: draw for ethereal exhausts at end of turn.
            PowerId::DarkEmbrace if own_side => {
                let n = self.amount * self.data;
                self.data = 0;
                if n > 0 {
                    vec![Effect::Draw { count: n as u32, from_hand_draw: false }]
                } else {
                    vec![]
                }
            }
            PowerId::Juggling if own_side => {
                self.data = 0;
                vec![]
            }
            // HighVoltagePower.cs
            PowerId::HighVoltage if own_side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            // NemesisPower.cs: Intangible on every other turn end; `data`
            // is `_shouldApplyIntangible`.
            PowerId::Nemesis if own_side => {
                self.data ^= 1;
                if self.data == 1 {
                    vec![Effect::ApplyPower { target: owner, id: PowerId::Intangible, amount: 1, applier: Some(owner) }]
                } else {
                    vec![Effect::RemovePower { target: owner, id: PowerId::Intangible }]
                }
            }
            _ => vec![],
        }
    }

    /// `BeforeApplied` for temporary powers: apply the real one.
    pub fn on_applied(&self, owner: CreatureRef, applied_amount: i32) -> Vec<Effect> {
        match temp_power(self.id) {
            Some((real, sign)) => vec![Effect::ApplyPower {
                target: owner,
                id: real,
                amount: sign * applied_amount,
                applier: Some(owner),
            }],
            None => vec![],
        }
    }

    /// `AfterCardExhausted`.
    pub fn after_card_exhausted(&mut self, owner: CreatureRef, ethereal: bool) -> Vec<Effect> {
        if owner != CreatureRef::Player {
            return vec![];
        }
        match self.id {
            // FeelNoPainPower.cs
            PowerId::FeelNoPain => {
                vec![Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            // DarkEmbracePower.cs
            PowerId::DarkEmbrace => {
                if ethereal {
                    self.data += 1;
                    vec![]
                } else {
                    vec![Effect::Draw { count: self.amount as u32, from_hand_draw: false }]
                }
            }
            _ => vec![],
        }
    }

    /// `AfterCardPlayed`. `ty` is the played card's type, `id` its id.
    pub fn after_card_played(&mut self, owner: CreatureRef, ty: CardType, card_id: crate::ids::CardId, upgraded: bool) -> Vec<Effect> {
        // SlowPower.cs sits on a monster and counts the player's plays.
        if self.id == PowerId::Slow {
            self.data += 1;
            return vec![];
        }
        // EnragePower.cs: Strength for every Skill you play.
        if self.id == PowerId::Enrage {
            if ty != CardType::Skill {
                return vec![];
            }
            return vec![Effect::ApplyPower { target: owner, id: PowerId::Strength, amount: self.amount, applier: Some(owner) }];
        }
        // WitheringPresencePower.cs: every `amount`th card you play (the
        // CardsLeft count) puts a Wither in your hand.
        if self.id == PowerId::WitheringPresence {
            self.data += 1;
            if self.data < self.amount {
                return vec![];
            }
            self.data = 0;
            return vec![Effect::GenerateCard { id: crate::ids::CardId::Wither, upgraded: false, to: Pile::Hand, free_this_turn: false }];
        }
        if owner != CreatureRef::Player {
            return vec![];
        }
        match self.id {
            // PanachePower.cs: `data` is CardsLeft, 0 until the Panache that
            // applied it has itself been played (`alreadyApplied`).
            PowerId::Panache => {
                if self.data == 0 {
                    self.data = PANACHE_CARDS;
                    return vec![];
                }
                self.data -= 1;
                if self.data > 0 {
                    return vec![];
                }
                self.data = PANACHE_CARDS;
                vec![Effect::DamageAllEnemies { amount: self.amount as f64, props: ValueProp::UNPOWERED, dealer: owner }]
            }
            // RagePower.cs
            PowerId::Rage if ty == CardType::Attack => {
                vec![Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            // JugglingPower.cs: on the third attack this turn, clone it.
            // TenderPower.cs: every card played costs 1 Strength and Dexterity
            // until the turn ends.
            PowerId::Tender => {
                self.data += 1;
                vec![
                    Effect::ApplyPower { target: owner, id: PowerId::Strength, amount: -1, applier: self.applier },
                    Effect::ApplyPower { target: owner, id: PowerId::Dexterity, amount: -1, applier: self.applier },
                ]
            }
            // CalamityPower.cs: every attack you play (each play of it) puts
            // `amount` random Ironclad attacks into your hand.
            PowerId::Calamity if ty == CardType::Attack => vec![Effect::GenerateRandom {
                pool: crate::effect::GenPool::IroncladAttacks,
                count: self.amount.max(0) as u32,
                to: Pile::Hand,
                free_this_turn: false,
                distinct: false,
                upgraded: false,
            }],
            PowerId::Juggling if ty == CardType::Attack => {
                self.data += 1;
                if self.data == 3 {
                    (0..self.amount)
                        .map(|_| Effect::GenerateCard { id: card_id, upgraded, to: Pile::Hand, free_this_turn: false })
                        .collect()
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }

    /// `BeforeDamageReceived` on the owner: ThornsPower hits the attacker
    /// back, killing blow or not.
    pub fn before_damage_received(&self, owner: CreatureRef, props: ValueProp, dealer: Option<CreatureRef>) -> Vec<Effect> {
        match (self.id, dealer) {
            (PowerId::Thorns, Some(d)) if props.is_powered() => vec![Effect::Damage {
                target: d,
                amount: self.amount as f64,
                props: ValueProp::UNPOWERED,
                dealer: Some(owner),
                card: None,
            }],
            _ => vec![],
        }
    }

    /// `AfterDamageReceived` on the owner. `own_turn` is whether the owner's
    /// side is acting.
    pub fn after_damage_received(
        &self,
        owner: CreatureRef,
        unblocked: i32,
        props: ValueProp,
        dealer: Option<CreatureRef>,
        own_turn: bool,
        hp_after: i32,
    ) -> Vec<Effect> {
        match self.id {
            // FlameBarrierPower.cs: hit the attacker back.
            PowerId::FlameBarrier if props.is_powered() => match dealer {
                Some(d) => vec![Effect::Damage {
                    target: d,
                    amount: self.amount as f64,
                    props: ValueProp::UNPOWERED,
                    dealer: Some(owner),
                    card: None,
                }],
                None => vec![],
            },
            // InfernoPower.cs: HP loss on your own turn burns every enemy.
            PowerId::Inferno if unblocked > 0 && own_turn => {
                vec![Effect::DamageAllEnemies { amount: self.amount as f64, props: ValueProp::UNPOWERED, dealer: owner }]
            }
            // TheGambitPower.cs: a powered attack that gets through kills you.
            PowerId::TheGambit if unblocked > 0 && props.is_powered() => {
                vec![Effect::RemovePower { target: owner, id: self.id }, Effect::Die { target: owner }]
            }
            // SlipperyPower.cs: one charge per unblocked hit.
            PowerId::Slippery if unblocked >= 1 => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // ShriekPower.cs: dropping to the threshold stuns it into Terror.
            PowerId::Shriek if unblocked > 0 && hp_after <= self.amount => vec![
                Effect::Stun { target: owner, next: Some("TERROR_MOVE") },
                Effect::RemovePower { target: owner, id: self.id },
            ],
            // AsleepPower.cs: one hit through the shell and it is up, shell
            // gone, this turn's move replaced by the wake-up into Slash.
            PowerId::Asleep if unblocked > 0 => vec![
                Effect::RemovePower { target: owner, id: PowerId::Plating },
                Effect::Stun { target: owner, next: Some("SLASH_MOVE") },
                Effect::RemovePower { target: owner, id: self.id },
            ],
            // SlumberPower.cs: each hit that gets through shortens the nap;
            // the last one stuns it awake into Roll Out.
            PowerId::Slumber if unblocked > 0 => {
                let mut e = vec![Effect::DecrementPower { target: owner, id: self.id }];
                if self.amount <= 1 {
                    e.push(Effect::Stun { target: owner, next: Some("ROLL_OUT_MOVE") });
                }
                e
            }
            // FlutterPower.cs: a powered attack that gets through spends a
            // charge; the last one knocks the hopper out of the air.
            PowerId::Flutter if unblocked > 0 && props.is_powered() => {
                let mut e = vec![Effect::DecrementPower { target: owner, id: self.id }];
                if self.amount <= 1 {
                    e.push(Effect::MonsterStep { me: owner, step: crate::monster::STEP_FLUTTER_DOWN });
                }
                e
            }
            // PersonalHivePower.cs: every powered hit on it puts Dazed into
            // the attacker's draw pile.
            PowerId::PersonalHive if props.is_powered() && dealer == Some(CreatureRef::Player) => (0..self.amount)
                .map(|_| Effect::GenerateCard { id: crate::ids::CardId::Dazed, upgraded: false, to: Pile::DrawRandom, free_this_turn: false })
                .collect(),
            // PlowPower.cs: `hp_after` is the owner's HP after the hit.
            PowerId::Plow if unblocked > 0 && hp_after <= self.amount => vec![
                Effect::RemoveStrength { target: owner },
                Effect::Stun { target: owner, next: Some("BEAST_CRY_MOVE") },
                Effect::RemovePower { target: owner, id: self.id },
            ],
            _ => vec![],
        }
    }

    /// `AfterBlockGained`.
    pub fn after_block_gained(&self, owner: CreatureRef, amount: f64) -> Vec<Effect> {
        match self.id {
            // JuggernautPower.cs: damage a random enemy.
            PowerId::Juggernaut if amount > 0.0 && owner == CreatureRef::Player => vec![Effect::Attack {
                dealer: owner,
                base: self.amount as f64,
                hits: 1,
                targets: crate::effect::AttackTargets::RandomOpponent,
                props: ValueProp::UNPOWERED,
                card: None,
            }],
            _ => vec![],
        }
    }

    /// `AfterPowerAmountChanged` for powers the owner applied.
    pub fn after_power_applied_by_owner(&self, applied: PowerId, amount: i32) -> Vec<Effect> {
        match self.id {
            // ViciousPower.cs
            PowerId::Vicious if applied == PowerId::Vulnerable && amount > 0 => {
                vec![Effect::Draw { count: self.amount as u32, from_hand_draw: false }]
            }
            _ => vec![],
        }
    }
}
