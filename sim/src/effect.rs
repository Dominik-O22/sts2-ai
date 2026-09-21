//! The effect queue.
//!
//! The game resolves effects as nested async calls: a card's `OnPlay` awaits
//! `DamageCmd`, which awaits hooks, and so on, depth first in source order.
//! We can't suspend that call stack in Rust for player choices, so effects
//! are data in a `VecDeque`. When resolving an effect produces sub-effects,
//! they are pushed to the *front*, in order, which yields the same depth-first
//! order the call stack would. A choice effect can then leave the rest of the
//! queue waiting for the answer.

use crate::ids::PowerId;
use crate::types::{CreatureRef, Side, ValueProp};

#[derive(Clone, Debug, PartialEq)]
pub enum AttackTargets {
    One(CreatureRef),
    /// `TargetingAllOpponents`. Every living creature on the other side.
    AllOpponents,
    /// `TargetingRandomOpponents`. Re-rolled per hit.
    RandomOpponent,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// `Commands/Builders/AttackCommand.cs`. One or more hits of `base`
    /// damage, each resolved through `Damage`.
    Attack {
        dealer: CreatureRef,
        base: f64,
        hits: u32,
        targets: AttackTargets,
        props: ValueProp,
        /// Uid of the card that dealt it, if any.
        card: Option<u32>,
    },
    /// `CreatureCmd.Damage` for a single target.
    Damage {
        target: CreatureRef,
        amount: f64,
        props: ValueProp,
        dealer: Option<CreatureRef>,
    },
    /// `CreatureCmd.GainBlock`.
    GainBlock {
        target: CreatureRef,
        amount: f64,
        props: ValueProp,
        card: Option<u32>,
    },
    /// `PowerCmd.Apply`.
    ApplyPower {
        target: CreatureRef,
        id: PowerId,
        amount: i32,
        applier: Option<CreatureRef>,
    },
    /// `PowerCmd.TickDownDuration`.
    TickDuration { target: CreatureRef, id: PowerId },
    /// `CardPileCmd.Draw(count)`.
    Draw { count: u32 },
    /// `CardCmd.Exhaust`.
    Exhaust { uid: u32 },
    /// Card play pipeline, `CardModel.OnPlayWrapper`. Energy is already spent.
    PlayCard { uid: u32, target: Option<CreatureRef> },
    /// Tail of the pipeline: move the card to its result pile.
    FinishCardPlay { uid: u32 },

    // Turn flow. `Combat/CombatManager.cs`.
    StartTurn(Side),
    EndPlayerTurn,
    EnemyAct(usize),
    EndEnemyTurn,
}
