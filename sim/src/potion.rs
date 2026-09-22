//! Potions. `Models/PotionModel.cs` plus `Models/Potions/*.cs`: the shared
//! pool (`SharedPotionPool`) and the three Ironclad potions
//! (`Ironclad4Epoch`). Colorless Potion is left out until colorless cards
//! are ported. Potion targeting differs from cards: `Self` and `AnyPlayer`
//! potions receive the player's own creature as the target.

use crate::combat::Combat;
use crate::effect::{CardFilter, Effect, GenPool, Pile, Then};
use crate::ids::PowerId;
use crate::types::{CreatureRef, ValueProp};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionId {
    // Shared pool.
    AttackPotion,
    BeetleJuice,
    BlessingOfTheForge,
    BlockPotion,
    BottledPotential,
    Clarity,
    CureAll,
    DexterityPotion,
    DistilledChaos,
    DropletOfPrecognition,
    Duplicator,
    EnergyPotion,
    EntropicBrew,
    ExplosiveAmpoule,
    FairyInABottle,
    FirePotion,
    FlexPotion,
    Fortifier,
    FruitJuice,
    FyshOil,
    GamblersBrew,
    GigantificationPotion,
    HeartOfIron,
    LiquidBronze,
    LiquidMemories,
    LuckyTonic,
    MazalethsGift,
    OrobicAcid,
    PotionOfBinding,
    PowderedDemise,
    PowerPotion,
    RadiantTincture,
    RegenPotion,
    ShacklingPotion,
    ShipInABottle,
    SkillPotion,
    SneckoOil,
    SpeedPotion,
    StableSerum,
    StrengthPotion,
    SwiftPotion,
    TouchOfInsanity,
    VulnerablePotion,
    WeakPotion,
    // Ironclad pool.
    BloodPotion,
    SoldiersStew,
    Ashwater,
    // Token rarity: handed out, never offered as a reward. Named for its
    // class, `PotionShapedRock`, so the recorder's id matches.
    PotionShapedRock,
}

pub const ALL: &[PotionId] = &[
    PotionId::AttackPotion,
    PotionId::BeetleJuice,
    PotionId::BlessingOfTheForge,
    PotionId::BlockPotion,
    PotionId::BottledPotential,
    PotionId::Clarity,
    PotionId::CureAll,
    PotionId::DexterityPotion,
    PotionId::DistilledChaos,
    PotionId::DropletOfPrecognition,
    PotionId::Duplicator,
    PotionId::EnergyPotion,
    PotionId::EntropicBrew,
    PotionId::ExplosiveAmpoule,
    PotionId::FairyInABottle,
    PotionId::FirePotion,
    PotionId::FlexPotion,
    PotionId::Fortifier,
    PotionId::FruitJuice,
    PotionId::FyshOil,
    PotionId::GamblersBrew,
    PotionId::GigantificationPotion,
    PotionId::HeartOfIron,
    PotionId::LiquidBronze,
    PotionId::LiquidMemories,
    PotionId::LuckyTonic,
    PotionId::MazalethsGift,
    PotionId::OrobicAcid,
    PotionId::PotionOfBinding,
    PotionId::PowderedDemise,
    PotionId::PowerPotion,
    PotionId::RadiantTincture,
    PotionId::RegenPotion,
    PotionId::ShacklingPotion,
    PotionId::ShipInABottle,
    PotionId::SkillPotion,
    PotionId::SneckoOil,
    PotionId::SpeedPotion,
    PotionId::StableSerum,
    PotionId::StrengthPotion,
    PotionId::SwiftPotion,
    PotionId::TouchOfInsanity,
    PotionId::VulnerablePotion,
    PotionId::WeakPotion,
    PotionId::BloodPotion,
    PotionId::SoldiersStew,
    PotionId::Ashwater,
    PotionId::PotionShapedRock,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rarity {
    Common,
    Uncommon,
    Rare,
    /// `PotionRarity.Token`: never in a reward pool.
    Token,
}

/// Who the player picks when throwing it. `TargetType` collapsed to what the
/// action space needs: `Self`, `AnyPlayer`, and `AllEnemies` need no pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    None,
    Enemy,
}

impl PotionId {
    pub fn rarity(self) -> Rarity {
        use PotionId::*;
        match self {
            AttackPotion | BlockPotion | BloodPotion | DexterityPotion | EnergyPotion | ExplosiveAmpoule | FirePotion
            | FlexPotion | PowerPotion | SkillPotion | SpeedPotion | StrengthPotion | SwiftPotion | VulnerablePotion
            | WeakPotion => Rarity::Common,
            Ashwater | BlessingOfTheForge | Clarity | CureAll | Duplicator | Fortifier | FyshOil | GamblersBrew
            | HeartOfIron | LiquidBronze | PotionOfBinding | PowderedDemise | RadiantTincture | RegenPotion
            | StableSerum | TouchOfInsanity => Rarity::Uncommon,
            BeetleJuice | BottledPotential | DistilledChaos | DropletOfPrecognition | EntropicBrew | FairyInABottle
            | FruitJuice | GigantificationPotion | LiquidMemories | LuckyTonic | MazalethsGift | OrobicAcid
            | ShacklingPotion | ShipInABottle | SneckoOil | SoldiersStew => Rarity::Rare,
            PotionShapedRock => Rarity::Token,
        }
    }

    pub fn target(self) -> Target {
        use PotionId::*;
        match self {
            BeetleJuice | FirePotion | PotionShapedRock | PowderedDemise | VulnerablePotion | WeakPotion => Target::Enemy,
            _ => Target::None,
        }
    }

    /// `PotionUsage.CombatOnly` or `AnyTime`. Entropic Brew is out of combat
    /// only and Fairy in a Bottle is automatic.
    pub fn usable_in_combat(self) -> bool {
        !matches!(self, PotionId::EntropicBrew | PotionId::FairyInABottle)
    }

    /// `CanBeGeneratedInCombat`.
    pub fn generatable_in_combat(self) -> bool {
        !matches!(self, PotionId::FairyInABottle | PotionId::FruitJuice | PotionId::RegenPotion)
    }

    /// `OnUse`. `target` is the picked enemy for `Target::Enemy` potions.
    pub fn on_use(self, c: &Combat, target: Option<CreatureRef>) -> Vec<Effect> {
        use PotionId::*;
        let me = CreatureRef::Player;
        let enemy = || target.expect("enemy target required");
        let power = |t: CreatureRef, id: PowerId, amount: i32| Effect::ApplyPower { target: t, id, amount, applier: Some(me) };
        let self_power = |id: PowerId, amount: i32| power(me, id, amount);
        let all_enemies = |id: PowerId, amount: i32| -> Vec<Effect> {
            c.living_enemies().map(|i| power(CreatureRef::Enemy(i), id, amount)).collect()
        };
        let draw = |n: u32| Effect::Draw { count: n, from_hand_draw: false };
        let block = |n: f64| Effect::GainBlock { target: me, amount: n, props: ValueProp::UNPOWERED, card: None };
        let choose = |from: Pile, filter: CardFilter, then: Then, can_skip: bool| Effect::Choose { from, filter, then, can_skip };
        let offer = |pool: GenPool| Effect::OfferRandom { pool, count: 3, free: true, retain: false };
        match self {
            AttackPotion => vec![offer(GenPool::IroncladAttacks)],
            SkillPotion => vec![offer(GenPool::IroncladSkills)],
            PowerPotion => vec![offer(GenPool::IroncladPowers)],
            BeetleJuice => vec![power(enemy(), PowerId::Shrink, 4)],
            BlessingOfTheForge => vec![Effect::UpgradeHand],
            BlockPotion => vec![block(12.0)],
            BottledPotential => {
                let mut e: Vec<Effect> = c
                    .player
                    .hand
                    .iter()
                    .map(|k| Effect::MoveCard { uid: k.uid, to: Pile::DrawBottom })
                    .collect();
                e.push(Effect::Shuffle);
                e.push(draw(5));
                e
            }
            Clarity => vec![draw(1), self_power(PowerId::Clarity, 3)],
            CureAll => vec![Effect::GainEnergy { amount: 1 }, draw(2)],
            DexterityPotion => vec![self_power(PowerId::Dexterity, 2)],
            DistilledChaos => vec![Effect::AutoPlayFromDrawTop { count: 3, force_exhaust: false }],
            DropletOfPrecognition => vec![choose(Pile::DrawTop, CardFilter::Any, Then::MoveTo(Pile::Hand), false)],
            Duplicator => vec![self_power(PowerId::Duplication, 1)],
            EnergyPotion => vec![Effect::GainEnergy { amount: 2 }],
            EntropicBrew | FairyInABottle => vec![],
            ExplosiveAmpoule => vec![Effect::DamageAllEnemies { amount: 10.0, props: ValueProp::UNPOWERED, dealer: me }],
            FirePotion => vec![Effect::Damage {
                target: enemy(),
                amount: 20.0,
                props: ValueProp::UNPOWERED,
                dealer: Some(me),
                card: None,
            }],
            PotionShapedRock => vec![Effect::Damage {
                target: enemy(),
                amount: 15.0,
                props: ValueProp::UNPOWERED,
                dealer: Some(me),
                card: None,
            }],
            FlexPotion => vec![self_power(PowerId::FlexPotion, 5)],
            Fortifier => vec![block(c.player.creature.block as f64 * 2.0)],
            FruitJuice => vec![Effect::GainMaxHp { target: me, amount: 5 }],
            FyshOil => vec![self_power(PowerId::Strength, 1), self_power(PowerId::Dexterity, 1)],
            GamblersBrew => vec![choose(Pile::Hand, CardFilter::Any, Then::DiscardThenDraw { picked: 0 }, true)],
            GigantificationPotion => vec![self_power(PowerId::Gigantification, 1)],
            HeartOfIron => vec![self_power(PowerId::Plating, 7)],
            LiquidBronze => vec![self_power(PowerId::Thorns, 3)],
            LiquidMemories => vec![choose(Pile::Discard, CardFilter::Any, Then::ToHandFreeThisTurn, false)],
            LuckyTonic => vec![self_power(PowerId::Buffer, 1)],
            MazalethsGift => vec![self_power(PowerId::Ritual, 1)],
            OrobicAcid => [GenPool::IroncladAttacks, GenPool::IroncladSkills, GenPool::IroncladPowers]
                .into_iter()
                .map(|pool| Effect::GenerateRandom { pool, count: 1, to: Pile::Hand, free_this_turn: true, distinct: true })
                .collect(),
            PotionOfBinding => {
                let mut e = all_enemies(PowerId::Weak, 1);
                e.extend(all_enemies(PowerId::Vulnerable, 1));
                e
            }
            PowderedDemise => vec![power(enemy(), PowerId::Demise, 9)],
            RadiantTincture => vec![Effect::GainEnergy { amount: 1 }, self_power(PowerId::Radiance, 3)],
            RegenPotion => vec![self_power(PowerId::Regen, 5)],
            ShacklingPotion => all_enemies(PowerId::ShacklingPotion, 7),
            ShipInABottle => vec![block(10.0), self_power(PowerId::BlockNextTurn, 10)],
            SneckoOil => vec![draw(7), Effect::SneckoCosts],
            SpeedPotion => vec![self_power(PowerId::SpeedPotion, 5)],
            StableSerum => vec![self_power(PowerId::RetainHand, 2)],
            StrengthPotion => vec![self_power(PowerId::Strength, 2)],
            SwiftPotion => vec![draw(3)],
            TouchOfInsanity => vec![choose(Pile::Hand, CardFilter::CostsEnergy, Then::FreeThisCombat, false)],
            VulnerablePotion => vec![power(enemy(), PowerId::Vulnerable, 3)],
            WeakPotion => vec![power(enemy(), PowerId::Weak, 3)],
            // BloodPotion.cs: 20% of max HP, decimal division.
            BloodPotion => vec![Effect::Heal { target: me, amount: c.player.creature.max_hp as f64 * 20.0 / 100.0 }],
            SoldiersStew => vec![Effect::StrikeReplay],
            Ashwater => vec![choose(Pile::Hand, CardFilter::Any, Then::ExhaustMany, true)],
        }
    }
}
