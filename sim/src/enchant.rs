//! Card enchantments. `Models/Enchantments/*.cs` over `EnchantmentModel.cs`.
//!
//! A card carries at most one enchantment (`CardModel.Enchantment`), with an
//! amount and a Normal/Disabled status. Relics and events attach them outside
//! combat, so a card arrives already enchanted and the sim never creates one.
//!
//! The value hooks run before every other modifier: `Hook.ModifyDamage` folds
//! `EnchantDamageAdditive` and `EnchantDamageMultiplicative` in at the top,
//! ahead of powers and relics.

use crate::card::{Card, Tag};
use crate::combat::Combat;
use crate::effect::Effect;
use crate::ids::{CardId, PowerId};
use crate::types::{CardRarity, CardType, CreatureRef, Keyword, TargetType, ValueProp};

/// `Models/Enchantments/<Name>.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnchantmentId {
    Adroit,
    Clone,
    Corrupted,
    Glam,
    Goopy,
    Imbued,
    Inky,
    Instinct,
    Momentum,
    Nimble,
    PerfectFit,
    RoyallyApproved,
    Sharp,
    Slither,
    SlumberingEssence,
    SoulsPower,
    Sown,
    Spiral,
    Steady,
    Swift,
    TezcatarasEmber,
    Vigorous,
}

/// Every variant, for id lookups by name and for the policy's vocabulary.
pub const ALL: &[EnchantmentId] = &[
    EnchantmentId::Adroit,
    EnchantmentId::Clone,
    EnchantmentId::Corrupted,
    EnchantmentId::Glam,
    EnchantmentId::Goopy,
    EnchantmentId::Imbued,
    EnchantmentId::Inky,
    EnchantmentId::Instinct,
    EnchantmentId::Momentum,
    EnchantmentId::Nimble,
    EnchantmentId::PerfectFit,
    EnchantmentId::RoyallyApproved,
    EnchantmentId::Sharp,
    EnchantmentId::Slither,
    EnchantmentId::SlumberingEssence,
    EnchantmentId::SoulsPower,
    EnchantmentId::Sown,
    EnchantmentId::Spiral,
    EnchantmentId::Steady,
    EnchantmentId::Swift,
    EnchantmentId::TezcatarasEmber,
    EnchantmentId::Vigorous,
];

/// One enchantment on one card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Enchantment {
    pub id: EnchantmentId,
    pub amount: i32,
    /// `EnchantmentStatus.Disabled`: spent, and silent for the rest of combat.
    pub disabled: bool,
    /// Per-enchantment counter. Momentum's accumulated damage is the only one.
    pub data: i32,
}

impl Enchantment {
    pub fn new(id: EnchantmentId, amount: i32) -> Self {
        Self { id, amount, disabled: false, data: 0 }
    }

    /// `EnchantDamageAdditive`. Every damage enchantment ignores unpowered
    /// damage, so relic and power chip damage is never boosted.
    pub fn damage_additive(&self, props: ValueProp) -> f64 {
        use EnchantmentId::*;
        if !props.is_powered() {
            return 0.0;
        }
        match self.id {
            Sharp => self.amount as f64,
            // Inky's DamageVar is a flat 1 on top of the Weak it applies.
            Inky => 1.0,
            TezcatarasEmber => 3.0,
            // Momentum banks `amount` per play, so the first play gets nothing.
            Momentum => self.data as f64,
            Vigorous if !self.disabled => self.amount as f64,
            _ => 0.0,
        }
    }

    /// `EnchantDamageMultiplicative`.
    pub fn damage_multiplicative(&self, props: ValueProp) -> f64 {
        if !props.is_powered() {
            return 1.0;
        }
        match self.id {
            EnchantmentId::Corrupted => 1.5,
            EnchantmentId::Instinct => 2.0,
            _ => 1.0,
        }
    }

    /// `EnchantBlockAdditive`.
    pub fn block_additive(&self) -> f64 {
        match self.id {
            EnchantmentId::Nimble => self.amount as f64,
            // Goopy starts at 1 and grows a point per play, so its first
            // play is worth nothing.
            EnchantmentId::Goopy => (self.amount - 1) as f64,
            _ => 0.0,
        }
    }

    /// `EnchantPlayCount`, applied to the card's own replay count.
    pub fn play_count(&self, replays: u32) -> u32 {
        match self.id {
            EnchantmentId::Spiral => replays + 1,
            EnchantmentId::Glam if !self.disabled => replays + 1,
            _ => replays,
        }
    }

    /// `EnchantmentModel.OnPlay`, which runs after the card's own effects.
    /// Inky follows its card's targeting, so the whole combat is in scope.
    pub fn on_play(&self, c: &Combat, uid: u32, target: Option<CreatureRef>, card_target: TargetType) -> Vec<Effect> {
        use EnchantmentId::*;
        let me = CreatureRef::Player;
        match self.id {
            Adroit => vec![Effect::GainBlock {
                target: me,
                amount: self.amount as f64,
                props: ValueProp::MOVE,
                card: Some(uid),
            }],
            // The card is already paid for, so the bite lands either way.
            Corrupted => vec![Effect::Damage {
                target: me,
                amount: 2.0,
                props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED).or(ValueProp::MOVE),
                dealer: Some(me),
                card: Some(uid),
            }],
            Inky => {
                let weak = |t| Effect::ApplyPower { target: t, id: PowerId::Weak, amount: 1, applier: Some(me) };
                match card_target {
                    TargetType::AllEnemies => c.living_enemies().map(CreatureRef::Enemy).map(weak).collect(),
                    _ => target.map(weak).into_iter().collect(),
                }
            }
            Sown if !self.disabled => vec![Effect::GainEnergy { amount: self.amount }],
            Swift if !self.disabled => vec![Effect::Draw { count: self.amount as u32, from_hand_draw: false }],
            _ => vec![],
        }
    }

    /// The bookkeeping `OnPlay` and `AfterCardPlayed` do to the enchantment
    /// itself, once the play is over.
    pub fn after_played(&mut self) {
        match self.id {
            // Spent for the combat.
            EnchantmentId::Sown | EnchantmentId::Swift | EnchantmentId::Glam | EnchantmentId::Vigorous => {
                self.disabled = true
            }
            EnchantmentId::Momentum => self.data += self.amount,
            EnchantmentId::Goopy => self.amount += 1,
            _ => {}
        }
    }

    /// `ShouldStartAtBottomOfDrawPile`.
    pub fn starts_at_bottom(&self) -> bool {
        self.id == EnchantmentId::Imbued
    }

    /// `PerfectFit.ModifyShuffleOrder`: to the top, but not on the shuffle
    /// that opens the combat.
    pub fn shuffles_to_top(&self) -> bool {
        self.id == EnchantmentId::PerfectFit
    }

    /// `Slither.AfterCardDrawn`: a fresh random cost every time it is drawn.
    pub fn randomizes_cost_on_draw(&self) -> bool {
        self.id == EnchantmentId::Slither
    }

    /// `SlumberingEssence.BeforeFlush`: a point cheaper for every turn it sat
    /// in hand unplayed.
    pub fn cheapens_in_hand(&self) -> bool {
        self.id == EnchantmentId::SlumberingEssence
    }

    /// `Imbued.AfterAutoPrePlayPhaseEntered`: plays itself on turn one.
    pub fn autoplays_on_first_turn(&self) -> bool {
        self.id == EnchantmentId::Imbued
    }

    /// `EnchantmentModel.OnEnchant`: the keywords an enchantment stamps onto
    /// the card when it is attached, outside combat.
    pub fn keywords_added(&self) -> &'static [Keyword] {
        match self.id {
            EnchantmentId::Goopy => &[Keyword::Exhaust],
            EnchantmentId::RoyallyApproved => &[Keyword::Innate, Keyword::Retain],
            EnchantmentId::Steady => &[Keyword::Retain],
            _ => &[],
        }
    }

    /// `SoulsPower.OnEnchant` takes Exhaust off the card.
    pub fn removes_exhaust(&self) -> bool {
        self.id == EnchantmentId::SoulsPower
    }

    /// `TezcatarasEmber.OnEnchant` makes the card free for the rest of the run.
    /// Its Eternal keyword only matters to deck editing, which combat never does.
    pub fn makes_free(&self) -> bool {
        self.id == EnchantmentId::TezcatarasEmber
    }

    /// `CanEnchantCardType` / `CanEnchant`, for the setup generator. Rolling
    /// an enchantment onto a card it could never sit on would make fights the
    /// game cannot deal.
    pub fn can_enchant(&self, card: &Card) -> bool {
        use EnchantmentId::*;
        let d = card.def();
        // The base `CanEnchant`: statuses, curses, and quests take nothing,
        // and neither does an unplayable card sitting in the deck.
        if matches!(d.ty, CardType::Status | CardType::Curse) || card.has(Keyword::Unplayable) {
            return false;
        }
        // `CardTag.Defend`, which in the Ironclad slice is Defend itself.
        let defend = card.id == CardId::DefendIronclad;
        match self.id {
            Corrupted | Instinct | Momentum | Sharp | Vigorous => d.ty == CardType::Attack,
            Imbued => d.ty == CardType::Skill,
            RoyallyApproved => matches!(d.ty, CardType::Attack | CardType::Skill),
            Nimble => card.gains_block(),
            Goopy => defend,
            Spiral => d.rarity == CardRarity::Basic && (card.has_tag(Tag::Strike) || defend),
            SoulsPower => card.has(Keyword::Exhaust),
            Slither => !d.x_cost,
            _ => true,
        }
    }
}
