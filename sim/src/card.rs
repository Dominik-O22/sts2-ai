//! Cards. `CardDef` is the canonical definition (`ModelDb.Card<T>()`), `Card`
//! is the mutable per-combat clone (`CardModel.ToMutable()`). Effects are
//! expressed as a list of `Effect`s, which the combat loop resolves depth
//! first (see effect.rs for why that matches the game's async call order).

use crate::effect::{AttackTargets, Effect};
use crate::ids::{CardId, PowerId};
use crate::types::{CardRarity, CardType, CreatureRef, Keyword, TargetType, ValueProp};

#[derive(Debug)]
pub struct CardDef {
    pub id: CardId,
    /// Printed energy cost. `-1` means no cost (unplayable, statuses).
    pub cost: i32,
    pub ty: CardType,
    pub rarity: CardRarity,
    pub target: TargetType,
    pub keywords: &'static [Keyword],
}

pub fn def(id: CardId) -> &'static CardDef {
    use CardId::*;
    match id {
        StrikeIronclad => &CardDef {
            id: StrikeIronclad,
            cost: 1,
            ty: CardType::Attack,
            rarity: CardRarity::Basic,
            target: TargetType::AnyEnemy,
            keywords: &[],
        },
        DefendIronclad => &CardDef {
            id: DefendIronclad,
            cost: 1,
            ty: CardType::Skill,
            rarity: CardRarity::Basic,
            target: TargetType::Self_,
            keywords: &[],
        },
        Bash => &CardDef {
            id: Bash,
            cost: 2,
            ty: CardType::Attack,
            rarity: CardRarity::Basic,
            target: TargetType::AnyEnemy,
            keywords: &[],
        },
    }
}

/// A card instance in a deck or a combat pile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    /// Unique within a combat. Effects refer to cards by uid because piles move.
    pub uid: u32,
    pub id: CardId,
    pub upgraded: bool,
    /// `LocalCostModifier` with `Expiration.EndOfTurn`, absolute. `None` = unmodified.
    pub cost_this_turn: Option<i32>,
    /// `LocalCostModifier` with `Expiration.EndOfCombat`, absolute.
    pub cost_this_combat: Option<i32>,
}

impl Card {
    pub fn new(uid: u32, id: CardId, upgraded: bool) -> Self {
        Self { uid, id, upgraded, cost_this_turn: None, cost_this_combat: None }
    }

    pub fn def(&self) -> &'static CardDef {
        def(self.id)
    }

    /// `CardEnergyCost.GetWithModifiers(All)` for the modifier kinds we model.
    /// Later modifiers win, and the result is clamped at 0.
    pub fn cost(&self) -> i32 {
        let base = self.def().cost;
        if base < 0 {
            return base;
        }
        self.cost_this_turn.or(self.cost_this_combat).unwrap_or(base).max(0)
    }

    pub fn has(&self, k: Keyword) -> bool {
        self.def().keywords.contains(&k)
    }

    /// `CardModel.EndOfTurnCleanup`.
    pub fn end_of_turn_cleanup(&mut self) {
        self.cost_this_turn = None;
    }

    /// The card's dynamic vars after upgrade. Ported per card from
    /// `CanonicalVars` plus `OnUpgrade`.
    pub fn vars(&self) -> Vars {
        use CardId::*;
        let up = self.upgraded;
        match self.id {
            StrikeIronclad => Vars { damage: if up { 9.0 } else { 6.0 }, ..Vars::NONE },
            DefendIronclad => Vars { block: if up { 8.0 } else { 5.0 }, ..Vars::NONE },
            Bash => Vars {
                damage: if up { 10.0 } else { 8.0 },
                magic: if up { 3.0 } else { 2.0 },
                ..Vars::NONE
            },
        }
    }

    /// `CardModel.OnPlay`, ported per card. `target` is the chosen enemy for
    /// `TargetType::AnyEnemy` cards and `None` otherwise.
    pub fn on_play(&self, target: Option<CreatureRef>) -> Vec<Effect> {
        use CardId::*;
        let v = self.vars();
        let me = CreatureRef::Player;
        match self.id {
            // Models/Cards/StrikeIronclad.cs
            StrikeIronclad => vec![Effect::Attack {
                dealer: me,
                base: v.damage,
                hits: 1,
                targets: AttackTargets::One(target.expect("Strike needs a target")),
                props: ValueProp::MOVE,
                card: Some(self.uid),
            }],
            // Models/Cards/DefendIronclad.cs
            DefendIronclad => vec![Effect::GainBlock {
                target: me,
                amount: v.block,
                props: ValueProp::MOVE,
                card: Some(self.uid),
            }],
            // Models/Cards/Bash.cs: attack, then Vulnerable.
            Bash => {
                let t = target.expect("Bash needs a target");
                vec![
                    Effect::Attack {
                        dealer: me,
                        base: v.damage,
                        hits: 1,
                        targets: AttackTargets::One(t),
                        props: ValueProp::MOVE,
                        card: Some(self.uid),
                    },
                    Effect::ApplyPower {
                        target: t,
                        id: PowerId::Vulnerable,
                        amount: v.magic as i32,
                        applier: Some(me),
                    },
                ]
            }
        }
    }
}

/// `DynamicVarSet` flattened to the three numbers most cards use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vars {
    pub damage: f64,
    pub block: f64,
    pub magic: f64,
}

impl Vars {
    pub const NONE: Vars = Vars { damage: 0.0, block: 0.0, magic: 0.0 };
}
