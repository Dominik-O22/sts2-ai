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
    /// turn, Crimson Mantle and Inferno self-damage, Slow cards played.
    pub data: i32,
    /// `PowerModel.Applier`. Constrict and Shrink vanish when it dies.
    pub applier: Option<CreatureRef>,
}

impl Power {
    pub fn new(id: PowerId, amount: i32) -> Self {
        Self { id, amount, skip_next_tick: false, data: 0, applier: None }
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
    )
}

/// `PowerModel.AllowNegative`.
pub fn allow_negative(id: PowerId) -> bool {
    matches!(id, PowerId::Strength | PowerId::Dexterity | PowerId::Shrink)
}

/// `PowerStackType.Single`: hidden amount, never stacks above 1.
pub fn is_single(id: PowerId) -> bool {
    matches!(
        id,
        PowerId::NoDraw
            | PowerId::Barricade
            | PowerId::Corruption
            | PowerId::Hellraiser
            | PowerId::NoEnergyGain
            | PowerId::Ringing
            | PowerId::Minion
            | PowerId::Infested
            | PowerId::Illusion
    )
}

/// `TemporaryStrengthPower` / `TemporaryDexterityPower` subclasses: the
/// real power they apply and the sign.
pub fn temp_power(id: PowerId) -> Option<(PowerId, i32)> {
    match id {
        PowerId::SetupStrike | PowerId::FlexPotion => Some((PowerId::Strength, 1)),
        PowerId::Mangle | PowerId::ShacklingPotion => Some((PowerId::Strength, -1)),
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
    pub fn modify_damage_additive(&self, owner: CreatureRef, dealer: Option<CreatureRef>, props: ValueProp) -> f64 {
        match self.id {
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
            _ => 1.0,
        }
    }

    /// `ModifyBlockAdditive`. `source_owner` is the owner of the card that
    /// grants the block, or the target itself for monster moves.
    pub fn modify_block_additive(&self, owner: CreatureRef, source_owner: CreatureRef, props: ValueProp) -> f64 {
        match self.id {
            // DexterityPower.cs
            PowerId::Dexterity if source_owner == owner && props.is_powered() => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `ModifyBlockMultiplicative`. `block_plays_this_turn` counts the
    /// player's card plays that already gained block this turn (Unmovable).
    pub fn modify_block_multiplicative(
        &self,
        owner: CreatureRef,
        target: CreatureRef,
        props: ValueProp,
        block_plays_this_turn: usize,
    ) -> f64 {
        match self.id {
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
            _ => amount,
        }
    }

    /// `ModifyHandDraw`: Clarity draws one more each turn.
    pub fn modify_hand_draw(&self, count: u32) -> u32 {
        match self.id {
            PowerId::Clarity => count + 1,
            _ => count,
        }
    }

    /// `AfterEnergyReset`: Radiance grants energy, then decrements.
    pub fn after_energy_reset(&self, owner: CreatureRef) -> Vec<Effect> {
        match self.id {
            PowerId::Radiance => vec![Effect::GainEnergy { amount: 1 }, Effect::DecrementPower { target: owner, id: self.id }],
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
            _ => vec![],
        }
    }

    /// `ShouldFlush`: Retain Hand keeps the hand.
    pub fn should_flush(&self) -> bool {
        self.id != PowerId::RetainHand
    }

    /// `ShouldClearBlock`.
    pub fn should_clear_block(&self, owner: CreatureRef, creature: CreatureRef) -> bool {
        !(self.id == PowerId::Barricade && owner == creature)
    }

    /// `ShouldDraw`: NoDraw blocks everything but the turn-start hand draw.
    pub fn should_draw(&self, from_hand_draw: bool) -> bool {
        !(self.id == PowerId::NoDraw && !from_hand_draw)
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
            // PlatingPower.cs: decrement each turn after the first.
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

    /// `AfterSideTurnStart` for powers that reset counters (Slow).
    pub fn reset_at_side_turn_start(&mut self, owner: CreatureRef, side: Side) {
        if self.id == PowerId::Slow && owner.side() == side {
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
            // ColossusPower.cs: plain decrement, no skip flag.
            PowerId::Colossus if side == Side::Enemy => vec![Effect::DecrementPower { target: owner, id: self.id }],
            // TerritorialPower.cs: when the owner's side ends, gain Strength.
            PowerId::Territorial if own_side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            // Self-removing at the end of the owner's turn.
            PowerId::NoDraw | PowerId::NoEnergyGain | PowerId::OneTwoPunch | PowerId::Rage | PowerId::Tangled | PowerId::Ringing
                if own_side =>
            {
                remove()
            }
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
            // TemporaryStrengthPower / TemporaryDexterityPower: remove self
            // and undo the real power.
            PowerId::SetupStrike | PowerId::Mangle | PowerId::FlexPotion | PowerId::ShacklingPotion | PowerId::SpeedPotion
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
            // RitualPower.cs (the enemy-applied skip does not apply to potions).
            PowerId::Ritual if own_side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            PowerId::RetainHand if own_side => vec![Effect::DecrementPower { target: owner, id: self.id }],
            PowerId::Duplication if own_side => remove(),
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
        if owner != CreatureRef::Player {
            return vec![];
        }
        match self.id {
            // RagePower.cs
            PowerId::Rage if ty == CardType::Attack => {
                vec![Effect::GainBlock { target: owner, amount: self.amount as f64, props: ValueProp::UNPOWERED, card: None }]
            }
            // JugglingPower.cs: on the third attack this turn, clone it.
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
            // ThornsPower.cs: hit the attacker back.
            PowerId::Thorns if props.is_powered() && dealer.is_some() => vec![Effect::Damage {
                target: dealer.unwrap(),
                amount: self.amount as f64,
                props: ValueProp::UNPOWERED,
                dealer: Some(owner),
                card: None,
            }],
            // SlipperyPower.cs: one charge per unblocked hit.
            PowerId::Slippery if unblocked >= 1 => vec![Effect::DecrementPower { target: owner, id: self.id }],
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
