//! Powers (statuses) on a creature. `Models/PowerModel.cs` plus the individual
//! `Models/Powers/*.cs`. Each hook below is one `AbstractModel` virtual,
//! implemented as a match over `PowerId` so dispatch is a jump table.

use crate::effect::Effect;
use crate::ids::PowerId;
use crate::types::{CreatureRef, Side, ValueProp};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Power {
    pub id: PowerId,
    pub amount: i32,
    /// `PowerModel.SkipNextDurationTick`. Set when a debuff lands on the
    /// player so it does not tick down the same round it was applied.
    pub skip_next_tick: bool,
}

/// `PowerModel.Type == Debuff`.
pub fn is_debuff(id: PowerId) -> bool {
    matches!(id, PowerId::Vulnerable | PowerId::Weak | PowerId::Frail)
}

/// `PowerModel.AllowNegative`.
pub fn allow_negative(id: PowerId) -> bool {
    matches!(id, PowerId::Strength | PowerId::Dexterity)
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
    pub fn modify_damage_additive(
        &self,
        owner: CreatureRef,
        dealer: Option<CreatureRef>,
        props: ValueProp,
    ) -> f64 {
        match self.id {
            // StrengthPower.cs
            PowerId::Strength if dealer == Some(owner) && props.is_powered() => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `ModifyDamageMultiplicative`.
    pub fn modify_damage_multiplicative(
        &self,
        owner: CreatureRef,
        target: CreatureRef,
        dealer: Option<CreatureRef>,
        props: ValueProp,
    ) -> f64 {
        match self.id {
            // VulnerablePower.cs: DamageIncrease = 1.5 on the target.
            PowerId::Vulnerable if target == owner && props.is_powered() => 1.5,
            // WeakPower.cs: DamageDecrease = 0.75 on the dealer.
            PowerId::Weak if dealer == Some(owner) && props.is_powered() => 0.75,
            _ => 1.0,
        }
    }

    /// `ModifyBlockAdditive`. `source_owner` is the owner of the card that
    /// grants the block, or the target itself for monster moves.
    pub fn modify_block_additive(
        &self,
        owner: CreatureRef,
        source_owner: CreatureRef,
        props: ValueProp,
    ) -> f64 {
        match self.id {
            // DexterityPower.cs
            PowerId::Dexterity if source_owner == owner && props.is_powered() => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `ModifyBlockMultiplicative`.
    pub fn modify_block_multiplicative(
        &self,
        owner: CreatureRef,
        target: CreatureRef,
        props: ValueProp,
    ) -> f64 {
        match self.id {
            // FrailPower.cs
            PowerId::Frail if target == owner && props.is_powered() => 0.75,
            _ => 1.0,
        }
    }

    /// `AfterSideTurnEnd`. Returns the effects the power queues.
    pub fn after_side_turn_end(&self, owner: CreatureRef, side: Side) -> Vec<Effect> {
        match self.id {
            // Vulnerable/Weak/Frail: tick at end of the enemy side turn only.
            PowerId::Vulnerable | PowerId::Weak | PowerId::Frail if side == Side::Enemy => {
                vec![Effect::TickDuration { target: owner, id: self.id }]
            }
            // TerritorialPower.cs: when the owner's side ends, gain Strength.
            PowerId::Territorial if owner.side() == side => vec![Effect::ApplyPower {
                target: owner,
                id: PowerId::Strength,
                amount: self.amount,
                applier: Some(owner),
            }],
            _ => vec![],
        }
    }
}
