//! Cards. `CardDef` is the canonical definition (`ModelDb.Card<T>()`), `Card`
//! is the mutable per-combat clone (`CardModel.ToMutable()`). Effects are
//! expressed as a list of `Effect`s, which the combat loop resolves depth
//! first (see effect.rs for why that matches the game's async call order).
//!
//! Each card's `on_play` is ported from `Models/Cards/<Name>.cs`. Cards that
//! need to observe the result of their own earlier effects continue in
//! `step`, reached through `Effect::CardStep`.

use crate::enchant::{Enchantment, EnchantmentId};
use crate::combat::Combat;
use crate::effect::{AttackTargets, CardFilter, Effect, GenPool, Pile, Then};
use crate::ids::{CardId, PowerId};
use crate::types::{CardRarity, CardType, CreatureRef, Keyword, TargetType, ValueProp};

/// `Entities/Cards/CardTag.cs`, the tags that have logic attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Strike,
}

#[derive(Debug)]
pub struct CardDef {
    pub id: CardId,
    /// Printed energy cost. `-1` means no cost (unplayable, statuses).
    pub cost: i32,
    /// `HasEnergyCostX`.
    pub x_cost: bool,
    pub ty: CardType,
    pub rarity: CardRarity,
    pub target: TargetType,
    pub keywords: &'static [Keyword],
    pub tags: &'static [Tag],
    /// `CanBeGeneratedInCombat`.
    pub generatable: bool,
}

macro_rules! defs {
    ($( $id:ident : $cost:expr, $ty:ident, $rar:ident, $tgt:ident
        $(, kw = [$($kw:ident),*])? $(, tags = [$($tag:ident),*])? $(, x = $x:expr)? $(, gen = $gen:expr)? ; )*) => {
        pub fn def(id: CardId) -> &'static CardDef {
            match id {
                $( CardId::$id => {
                    const D: CardDef = CardDef {
                        id: CardId::$id,
                        cost: $cost,
                        x_cost: false $(|| $x)?,
                        ty: CardType::$ty,
                        rarity: CardRarity::$rar,
                        target: TargetType::$tgt,
                        keywords: &[$($(Keyword::$kw),*)?],
                        tags: &[$($(Tag::$tag),*)?],
                        generatable: true $(&& $gen)?,
                    };
                    &D
                } )*
            }
        }
    };
}

defs! {
    Aggression: 1, Power, Rare, Self_;
    Anger: 0, Attack, Common, AnyEnemy;
    Armaments: 1, Skill, Common, Self_;
    AshenStrike: 1, Attack, Uncommon, AnyEnemy, tags = [Strike];
    Barricade: 3, Power, Rare, Self_;
    Bash: 2, Attack, Basic, AnyEnemy;
    BattleTrance: 0, Skill, Uncommon, Self_;
    Bloodletting: 0, Skill, Common, Self_;
    BloodWall: 2, Skill, Common, Self_;
    Bludgeon: 3, Attack, Uncommon, AnyEnemy;
    BodySlam: 1, Attack, Common, AnyEnemy;
    Brand: 0, Skill, Rare, Self_;
    Break: 1, Attack, Special, AnyEnemy;
    Breakthrough: 1, Attack, Common, AllEnemies;
    Bully: 0, Attack, Uncommon, AnyEnemy;
    BurningPact: 1, Skill, Uncommon, Self_;
    Cascade: -1, Skill, Rare, Self_, x = true;
    Cinder: 2, Attack, Common, AnyEnemy;
    Colossus: 1, Skill, Uncommon, Self_;
    Conflagration: 1, Attack, Rare, AllEnemies;
    Corruption: 3, Power, Special, Self_;
    CrimsonMantle: 1, Power, Rare, Self_;
    Cruelty: 1, Power, Rare, Self_;
    DarkEmbrace: 2, Power, Rare, Self_;
    DefendIronclad: 1, Skill, Basic, Self_;
    DemonForm: 3, Power, Rare, Self_;
    Dismantle: 1, Attack, Uncommon, AnyEnemy;
    Dominate: 1, Skill, Uncommon, AnyEnemy, kw = [Exhaust];
    DrumOfBattle: 1, Skill, Uncommon, Self_;
    EvilEye: 1, Skill, Uncommon, Self_;
    ExpectAFight: 2, Skill, Uncommon, Self_;
    Feed: 1, Attack, Rare, AnyEnemy, kw = [Exhaust], gen = false;
    FeelNoPain: 1, Power, Uncommon, Self_;
    FiendFire: 2, Attack, Rare, AnyEnemy, kw = [Exhaust];
    FightMe: 2, Attack, Uncommon, AnyEnemy;
    FlameBarrier: 2, Skill, Uncommon, Self_;
    ForgottenRitual: 1, Skill, Uncommon, Self_, kw = [Exhaust];
    Havoc: 1, Skill, Common, Self_;
    Headbutt: 1, Attack, Common, AnyEnemy;
    Hellraiser: 2, Power, Rare, Self_;
    Hemokinesis: 1, Attack, Uncommon, AnyEnemy;
    HowlFromBeyond: 3, Attack, Uncommon, AllEnemies;
    Impervious: 2, Skill, Rare, Self_, kw = [Exhaust];
    InfernalBlade: 1, Skill, Uncommon, Self_, kw = [Exhaust];
    Inferno: 1, Power, Uncommon, Self_;
    Inflame: 1, Power, Uncommon, Self_;
    IronWave: 1, Attack, Common, AnyEnemy;
    Juggernaut: 2, Power, Rare, Self_;
    Juggling: 1, Power, Uncommon, Self_;
    Mangle: 3, Attack, Rare, AnyEnemy;
    MoltenFist: 1, Attack, Common, AnyEnemy, kw = [Exhaust];
    NotYet: 2, Skill, Rare, Self_, kw = [Exhaust], gen = false;
    Offering: 0, Skill, Rare, Self_, kw = [Exhaust];
    OneTwoPunch: 1, Skill, Rare, Self_;
    PactsEnd: 0, Attack, Rare, AllEnemies;
    PerfectedStrike: 2, Attack, Common, AnyEnemy, tags = [Strike];
    Pillage: 1, Attack, Uncommon, AnyEnemy;
    PommelStrike: 1, Attack, Common, AnyEnemy, tags = [Strike];
    PrimalForce: 0, Skill, Rare, Self_;
    Pyre: 2, Power, Rare, Self_;
    Rage: 0, Skill, Uncommon, Self_;
    Rampage: 1, Attack, Uncommon, AnyEnemy;
    Rupture: 1, Power, Uncommon, Self_;
    SecondWind: 1, Skill, Uncommon, Self_;
    SetupStrike: 1, Attack, Common, AnyEnemy, tags = [Strike];
    ShrugItOff: 1, Skill, Common, Self_;
    Spite: 0, Attack, Uncommon, AnyEnemy;
    Stampede: 2, Power, Uncommon, Self_;
    Stoke: 1, Skill, Rare, Self_;
    Stomp: 3, Attack, Uncommon, AllEnemies;
    StoneArmor: 1, Power, Uncommon, Self_;
    StrikeIronclad: 1, Attack, Basic, AnyEnemy, tags = [Strike];
    SwordBoomerang: 1, Attack, Common, RandomEnemy;
    Taunt: 1, Skill, Uncommon, AnyEnemy;
    TearAsunder: 2, Attack, Rare, AnyEnemy;
    Thrash: 1, Attack, Rare, AnyEnemy;
    Thunderclap: 1, Attack, Common, AllEnemies;
    Tremble: 1, Skill, Common, AnyEnemy, kw = [Exhaust];
    TrueGrit: 1, Skill, Common, Self_;
    TwinStrike: 1, Attack, Common, AnyEnemy, tags = [Strike];
    Unmovable: 2, Power, Rare, Self_;
    Unrelenting: 2, Attack, Uncommon, AnyEnemy;
    Uppercut: 2, Attack, Uncommon, AnyEnemy;
    Vicious: 1, Power, Uncommon, Self_;
    Whirlwind: 0, Attack, Uncommon, AllEnemies, x = true;
    Wound: -1, Status, Special, None, kw = [Unplayable], gen = false;
    Slimed: 1, Status, Special, None, kw = [Exhaust], gen = false;
    Dazed: -1, Status, Special, None, kw = [Ethereal, Unplayable], gen = false;
    Burn: -1, Status, Special, None, kw = [Unplayable], gen = false;
    Infection: -1, Status, Special, None, kw = [Unplayable], gen = false;
    AscendersBane: -1, Curse, Special, None, kw = [Unplayable, Ethereal], gen = false;
    GiantRock: 1, Attack, Special, AnyEnemy, gen = false;
    MindBlast: 1, Attack, Uncommon, AnyEnemy, kw = [Innate], gen = false;
    Beckon: 1, Status, Special, None, gen = false;
    BadLuck: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Clumsy: -1, Curse, Special, None, kw = [Unplayable, Ethereal], gen = false;
    CurseOfTheBell: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Debt: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Decay: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Doubt: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Enthralled: 2, Curse, Special, None, gen = false;
    Folly: -1, Curse, Special, None, kw = [Unplayable, Ethereal, Innate], gen = false;
    Greed: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Guilty: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Injury: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Normality: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    PoorSleep: -1, Curse, Special, None, kw = [Unplayable, Retain], gen = false;
    Regret: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    Shame: -1, Curse, Special, None, kw = [Unplayable], gen = false;
    SporeMind: 1, Curse, Special, None, kw = [Exhaust], gen = false;
    Writhe: -1, Curse, Special, None, kw = [Unplayable, Innate], gen = false;
    Soot: -1, Status, Special, None, kw = [Unplayable], gen = false;
    Luminesce: 0, Skill, Special, Self_, kw = [Exhaust, Retain], gen = false;
    Toxic: 1, Status, Special, None, kw = [Exhaust], gen = false;
    FranticEscape: 1, Status, Special, Self_, gen = false;
    Disintegration: -1, Status, Special, None, kw = [Unplayable], gen = false;
    MindRot: -1, Status, Special, None, kw = [Unplayable], gen = false;
    Sloth: -1, Status, Special, None, kw = [Unplayable], gen = false;
    WasteAway: -1, Status, Special, None, kw = [Unplayable], gen = false;
    Wither: -1, Status, Special, None, kw = [Unplayable], gen = false;
}

/// The Ironclad card pool in `IroncladCardPool.cs` order, for generation.
pub const IRONCLAD_POOL: &[CardId] = &[
    CardId::Aggression, CardId::Anger, CardId::Armaments, CardId::AshenStrike, CardId::Barricade,
    CardId::Bash, CardId::BattleTrance, CardId::Bloodletting, CardId::BloodWall, CardId::Bludgeon,
    CardId::BodySlam, CardId::Brand, CardId::Break, CardId::Breakthrough, CardId::Bully,
    CardId::BurningPact, CardId::Cascade, CardId::Cinder, CardId::Colossus, CardId::Conflagration,
    CardId::Corruption, CardId::CrimsonMantle, CardId::Cruelty, CardId::DarkEmbrace,
    CardId::DefendIronclad, CardId::DemonForm, CardId::Dismantle, CardId::Dominate,
    CardId::DrumOfBattle, CardId::EvilEye, CardId::ExpectAFight, CardId::Feed, CardId::FeelNoPain,
    CardId::FiendFire, CardId::FightMe, CardId::FlameBarrier, CardId::ForgottenRitual, CardId::Havoc,
    CardId::Headbutt, CardId::Hellraiser, CardId::Hemokinesis, CardId::HowlFromBeyond,
    CardId::Impervious, CardId::InfernalBlade, CardId::Inferno, CardId::Inflame, CardId::IronWave,
    CardId::Juggernaut, CardId::Juggling, CardId::Mangle, CardId::MoltenFist, CardId::NotYet,
    CardId::Offering, CardId::OneTwoPunch, CardId::PactsEnd, CardId::PerfectedStrike, CardId::Pillage,
    CardId::PommelStrike, CardId::PrimalForce, CardId::Pyre, CardId::Rage, CardId::Rampage,
    CardId::Rupture, CardId::SecondWind, CardId::SetupStrike, CardId::ShrugItOff, CardId::Spite,
    CardId::Stampede, CardId::Stoke, CardId::Stomp, CardId::StoneArmor, CardId::StrikeIronclad,
    CardId::SwordBoomerang, CardId::Taunt, CardId::TearAsunder, CardId::Thrash, CardId::Thunderclap,
    CardId::Tremble, CardId::TrueGrit, CardId::TwinStrike, CardId::Unmovable, CardId::Unrelenting,
    CardId::Uppercut, CardId::Vicious, CardId::Whirlwind,
];

/// `Models/Afflictions/*.cs` from act 3. Smog predates this and lives in
/// `Card::smogged`; a card never holds more than one affliction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Affliction {
    /// Playing it hurts you for Globe Head's Galvanic (`Galvanized.cs`).
    Galvanized,
    /// Ethereal while Hex is on you (`Hexed.cs`).
    Hexed,
    /// Only one Bound card per turn (`Bound.cs`, Queen's Chains of Binding).
    Bound,
}

/// `DynamicVarSet` flattened to the numbers cards use.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vars {
    pub damage: f64,
    pub block: f64,
    /// Power amounts, "magic numbers".
    pub magic: f64,
    pub hits: u32,
    pub cards: u32,
    pub energy: i32,
    pub hp_loss: f64,
}

/// A card instance in a deck or a combat pile.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    /// Unique within a combat. Effects refer to cards by uid because piles move.
    pub uid: u32,
    pub id: CardId,
    pub upgraded: bool,
    /// `LocalCostModifier` with `Expiration.EndOfTurn`, absolute. `None` = unmodified.
    pub cost_this_turn: Option<i32>,
    /// `LocalCostModifier` with `Expiration.EndOfCombat`, absolute.
    pub cost_this_combat: Option<i32>,
    /// `CapturedXValue` for X-cost cards, set when played.
    pub captured_x: i32,
    /// `ExhaustOnNextPlay` (Havoc).
    pub exhaust_on_next_play: bool,
    /// Damage added by the card's own plays this combat (Rampage, Thrash).
    pub extra_damage: f64,
    /// Ethereal granted locally by Ghost Seed.
    pub ethereal_added: bool,
    /// `BaseReplayCount`: extra plays per play (Soldier's Stew).
    pub replay: u32,
    /// `CardModel.Affliction is Smog` (Living Fog). Cleared each turn end.
    pub smogged: bool,
    /// `CardModel.Enchantment`. Attached outside combat and carried in.
    pub enchantment: Option<Enchantment>,
    /// Retain granted by `CardCmd.ApplyKeyword` (Choices Paradox).
    pub retain_added: bool,
    /// `CardModel.IsDupe` (History Course): loses Exhaust and leaves combat
    /// once played.
    pub dupe: bool,
    /// `CardModel.Affliction`, the act 3 ones.
    pub affliction: Option<Affliction>,
    /// Downgraded by Magi Knight's Dampen; upgraded again when it lifts.
    pub dampened: bool,
}

impl Card {
    pub fn new(uid: u32, id: CardId, upgraded: bool) -> Self {
        Self {
            uid,
            id,
            upgraded,
            cost_this_turn: None,
            cost_this_combat: None,
            captured_x: 0,
            exhaust_on_next_play: false,
            extra_damage: 0.0,
            ethereal_added: false,
            replay: 0,
            smogged: false,
            enchantment: None,
            retain_added: false,
            dupe: false,
            affliction: None,
            dampened: false,
        }
    }

    pub fn def(&self) -> &'static CardDef {
        def(self.id)
    }

    pub fn ty(&self) -> CardType {
        self.def().ty
    }

    /// `EnergyCost._base`: printed cost after permanent upgrades.
    pub fn base_cost(&self) -> i32 {
        use CardId::*;
        let c = self.def().cost;
        let cheaper_when_upgraded = matches!(
            self.id,
            Barricade | BodySlam | Corruption | DarkEmbrace | ExpectAFight | Havoc | Hellraiser | MindBlast
                | InfernalBlade | Stampede | Unmovable
        );
        if self.upgraded && cheaper_when_upgraded {
            c - 1
        } else {
            c
        }
    }

    /// `CardEnergyCost.GetWithModifiers(Local)`. Later modifiers win; clamped at 0.
    /// Global modifiers (Corruption, Free Attack) are applied by `Combat::cost`.
    pub fn local_cost(&self) -> i32 {
        let base = self.base_cost();
        if base < 0 || self.def().x_cost {
            return base;
        }
        self.cost_this_turn.or(self.cost_this_combat).unwrap_or(base).max(0)
    }

    /// `CardCmd.Enchant`: attach one, with the keyword and cost changes its
    /// `OnEnchant` makes. Enchanting happens outside combat, so this is called
    /// while the deck is being built, not mid-fight.
    pub fn enchant(&mut self, id: EnchantmentId, amount: i32) {
        let e = Enchantment::new(id, amount);
        if e.makes_free() {
            self.cost_this_combat = Some(0);
        }
        self.enchantment = Some(e);
    }

    /// Keywords, including the ones upgrades and enchantments add.
    pub fn has(&self, k: Keyword) -> bool {
        if let Some(e) = &self.enchantment {
            if e.keywords_added().contains(&k) {
                return true;
            }
            if k == Keyword::Exhaust && e.removes_exhaust() {
                return false;
            }
        }
        if k == Keyword::Exhaust && self.dupe {
            return false;
        }
        if k == Keyword::Retain && self.retain_added {
            return true;
        }
        if self.def().keywords.contains(&k) {
            return true;
        }
        if k == Keyword::Ethereal && (self.ethereal_added || self.affliction == Some(Affliction::Hexed)) {
            return true;
        }
        k == Keyword::Innate && self.upgraded && matches!(self.id, CardId::Aggression | CardId::Juggling)
    }

    /// `CardModel.GainsBlock`: whether the printed card ever grants block,
    /// which is what Nimble and Goopy check before they can sit on it.
    pub fn gains_block(&self) -> bool {
        self.vars().block > 0.0
    }

    pub fn has_tag(&self, t: Tag) -> bool {
        self.def().tags.contains(&t)
    }

    /// `IsUpgradable`: statuses and curses have `MaxUpgradeLevel => 0`.
    pub fn upgradable(&self) -> bool {
        !self.upgraded && !matches!(self.ty(), CardType::Status | CardType::Curse)
    }

    /// `CardModel.EndOfTurnCleanup`.
    pub fn end_of_turn_cleanup(&mut self) {
        self.cost_this_turn = None;
        self.exhaust_on_next_play = false;
    }

    /// The card's dynamic vars after upgrade. Ported per card from
    /// `CanonicalVars` plus `OnUpgrade`. `damage` includes `extra_damage`.
    pub fn vars(&self) -> Vars {
        use CardId::*;
        let up = self.upgraded;
        let pick = |base: f64, upg: f64| if up { upg } else { base };
        let picku = |base: u32, upg: u32| if up { upg } else { base };
        let d = Vars::default;
        let mut v = match self.id {
            Aggression | Barricade | Cascade | Havoc | Hellraiser | InfernalBlade | PrimalForce
            | Stoke | Unmovable | Wound | Dazed | AscendersBane | DarkEmbrace | Juggling | Corruption => d(),
            // Curses. Regret reads the hand it ends the turn in, so its
            // damage is not a card number.
            Clumsy | CurseOfTheBell | Debt | Doubt | Enthralled
            | Folly | Greed | Guilty | Injury | Normality | PoorSleep | Regret | Shame | SporeMind | Writhe | Soot => d(),
            FranticEscape | Disintegration | MindRot | Sloth | WasteAway => d(),
            Anger => Vars { damage: pick(6.0, 8.0), ..d() },
            Armaments => Vars { block: 5.0, ..d() },
            AshenStrike => Vars { damage: 6.0, magic: pick(3.0, 4.0), ..d() },
            Bash => Vars { damage: pick(8.0, 10.0), magic: pick(2.0, 3.0), ..d() },
            BattleTrance => Vars { cards: picku(3, 4), ..d() },
            Bloodletting => Vars { hp_loss: 3.0, energy: if up { 3 } else { 2 }, ..d() },
            BloodWall => Vars { hp_loss: 2.0, block: pick(16.0, 20.0), ..d() },
            Bludgeon => Vars { damage: pick(32.0, 42.0), ..d() },
            BodySlam => Vars { magic: 1.0, ..d() },
            Brand => Vars { hp_loss: 1.0, magic: pick(1.0, 2.0), ..d() },
            Break => Vars { damage: pick(20.0, 30.0), magic: pick(5.0, 7.0), ..d() },
            Breakthrough => Vars { damage: pick(9.0, 13.0), hp_loss: 1.0, ..d() },
            Bully => Vars { damage: 4.0, magic: pick(2.0, 3.0), ..d() },
            BurningPact => Vars { cards: picku(2, 3), ..d() },
            Cinder => Vars { damage: pick(18.0, 24.0), ..d() },
            Colossus => Vars { block: pick(5.0, 8.0), magic: 1.0, ..d() },
            Conflagration => Vars { damage: 2.0, hits: picku(4, 5), ..d() },
            CrimsonMantle => Vars { magic: pick(8.0, 10.0), ..d() },
            Cruelty => Vars { magic: pick(25.0, 50.0), ..d() },
            DefendIronclad => Vars { block: pick(5.0, 8.0), ..d() },
            DemonForm => Vars { magic: pick(2.0, 3.0), ..d() },
            Dismantle => Vars { damage: pick(8.0, 10.0), ..d() },
            Dominate => Vars { magic: pick(1.0, 2.0), ..d() },
            DrumOfBattle => Vars { cards: 2, energy: if up { 3 } else { 2 }, ..d() },
            EvilEye => Vars { block: pick(8.0, 11.0), ..d() },
            ExpectAFight => Vars { magic: 1.0, ..d() },
            Feed => Vars { damage: pick(10.0, 12.0), magic: pick(3.0, 4.0), ..d() },
            FeelNoPain => Vars { magic: pick(3.0, 4.0), ..d() },
            FiendFire => Vars { damage: pick(7.0, 10.0), ..d() },
            FightMe => Vars { damage: pick(5.0, 6.0), hits: 2, magic: pick(3.0, 4.0), ..d() },
            FlameBarrier => Vars { block: pick(12.0, 16.0), magic: pick(4.0, 6.0), ..d() },
            ForgottenRitual => Vars { energy: if up { 4 } else { 3 }, ..d() },
            Headbutt => Vars { damage: pick(9.0, 12.0), ..d() },
            Hemokinesis => Vars { hp_loss: 2.0, damage: pick(15.0, 20.0), ..d() },
            HowlFromBeyond => Vars { damage: pick(16.0, 21.0), ..d() },
            Impervious => Vars { block: pick(30.0, 40.0), ..d() },
            Inferno => Vars { magic: pick(6.0, 9.0), ..d() },
            Inflame => Vars { magic: pick(2.0, 3.0), ..d() },
            IronWave => Vars { damage: pick(5.0, 7.0), block: pick(5.0, 7.0), ..d() },
            Juggernaut => Vars { magic: pick(6.0, 8.0), ..d() },
            Mangle => Vars { damage: pick(15.0, 20.0), magic: pick(10.0, 15.0), ..d() },
            MoltenFist => Vars { damage: pick(10.0, 14.0), ..d() },
            NotYet => Vars { magic: pick(10.0, 13.0), ..d() },
            Offering => Vars { hp_loss: 6.0, energy: 2, cards: picku(3, 5), ..d() },
            OneTwoPunch => Vars { magic: pick(1.0, 2.0), ..d() },
            PactsEnd => Vars { damage: pick(17.0, 23.0), cards: 3, ..d() },
            PerfectedStrike => Vars { damage: 6.0, magic: pick(2.0, 3.0), ..d() },
            Pillage => Vars { damage: pick(6.0, 9.0), ..d() },
            PommelStrike => Vars { damage: pick(9.0, 10.0), cards: picku(1, 2), ..d() },
            Pyre => Vars { energy: if up { 2 } else { 1 }, ..d() },
            Rage => Vars { magic: pick(3.0, 5.0), ..d() },
            Rampage => Vars { damage: 9.0, magic: pick(5.0, 9.0), ..d() },
            Rupture => Vars { magic: pick(1.0, 2.0), ..d() },
            SecondWind => Vars { block: pick(5.0, 7.0), ..d() },
            SetupStrike => Vars { damage: pick(7.0, 9.0), magic: pick(2.0, 3.0), ..d() },
            ShrugItOff => Vars { block: pick(8.0, 11.0), cards: 1, ..d() },
            Spite => Vars { damage: 5.0, hits: picku(2, 3), ..d() },
            Stampede => Vars { magic: 1.0, ..d() },
            Stomp => Vars { damage: pick(12.0, 15.0), ..d() },
            StoneArmor => Vars { magic: pick(4.0, 6.0), ..d() },
            StrikeIronclad => Vars { damage: pick(6.0, 9.0), ..d() },
            SwordBoomerang => Vars { damage: 3.0, hits: picku(3, 4), ..d() },
            Taunt => Vars { block: pick(7.0, 8.0), magic: pick(1.0, 2.0), ..d() },
            TearAsunder => Vars { damage: pick(5.0, 7.0), ..d() },
            Thrash => Vars { damage: pick(4.0, 6.0), hits: 2, ..d() },
            Thunderclap => Vars { damage: pick(4.0, 7.0), magic: 1.0, ..d() },
            Tremble => Vars { magic: pick(3.0, 4.0), ..d() },
            TrueGrit => Vars { block: pick(7.0, 9.0), ..d() },
            TwinStrike => Vars { damage: pick(5.0, 7.0), hits: 2, ..d() },
            Unrelenting => Vars { damage: pick(14.0, 20.0), ..d() },
            Uppercut => Vars { damage: 13.0, magic: pick(1.0, 2.0), ..d() },
            Vicious => Vars { cards: picku(1, 2), ..d() },
            Whirlwind => Vars { damage: pick(5.0, 8.0), ..d() },
            Slimed => Vars { cards: 1, ..d() },
            Burn => Vars { damage: 2.0, ..d() },
            Infection => Vars { damage: 3.0, ..d() },
            GiantRock => Vars { damage: pick(16.0, 20.0), ..d() },
            MindBlast => Vars { magic: 1.0, ..d() },
            Beckon => Vars { hp_loss: 6.0, ..d() },
            BadLuck => Vars { hp_loss: 13.0, ..d() },
            Decay => Vars { damage: 2.0, ..d() },
            Luminesce => Vars { energy: if up { 3 } else { 2 }, ..d() },
            Toxic => Vars { damage: 5.0, ..d() },
            // Aeonglass's upgrades land in `extra_damage`, 3 at a time.
            Wither => Vars { damage: 3.0, ..d() },
        };
        v.damage += self.extra_damage;
        v
    }

    /// `CardModel.OnPlay`, ported per card. `target` is the chosen enemy for
    /// `TargetType::AnyEnemy` cards and `None` otherwise. `c` is read-only
    /// combat state for cards whose numbers depend on it.
    pub fn on_play(&self, c: &Combat, target: Option<CreatureRef>) -> Vec<Effect> {
        use CardId::*;
        let v = self.vars();
        let me = CreatureRef::Player;
        let uid = self.uid;
        let t = || target.expect("card needs a target");
        let hit = |base: f64, hits: u32, targets: AttackTargets| Effect::Attack {
            dealer: me,
            base,
            hits,
            targets,
            props: ValueProp::MOVE,
            card: Some(uid),
        };
        let attack = |hits: u32| hit(v.damage, hits, AttackTargets::One(t()));
        let aoe = |hits: u32| hit(v.damage, hits, AttackTargets::AllOpponents);
        let block = || Effect::GainBlock { target: me, amount: v.block, props: ValueProp::MOVE, card: Some(uid) };
        let power = |target: CreatureRef, id: PowerId, amount: i32| Effect::ApplyPower {
            target,
            id,
            amount,
            applier: Some(me),
        };
        let self_power = |id: PowerId, amount: i32| power(me, id, amount);
        // `CreatureCmd.Damage(..., Unblockable | Unpowered | Move, this)`: HP loss.
        let lose_hp = |amount: f64| Effect::Damage {
            target: me,
            amount,
            props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED).or(ValueProp::MOVE),
            dealer: None,
            card: Some(uid),
        };
        let draw = |n: u32| Effect::Draw { count: n, from_hand_draw: false };
        let step = |s: u8| Effect::CardStep { uid, target, step: s };
        let m = v.magic as i32;

        match self.id {
            Aggression => vec![self_power(PowerId::Aggression, 1)],
            // Anger.cs: `CreateClone()` into the discard, so an enchanted
            // Anger breeds enchanted Angers.
            Anger => vec![attack(1), Effect::CloneCard { uid: self.uid, to: Pile::Discard }],
            Armaments => {
                let mut e = vec![block()];
                if self.upgraded {
                    e.push(Effect::UpgradeHand);
                } else {
                    e.push(Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then: Then::Upgrade, can_skip: false });
                }
                e
            }
            // 6 + 3 per card in the exhaust pile.
            AshenStrike => {
                let dmg = v.damage + v.magic * c.player.exhaust.len() as f64;
                vec![hit(dmg, 1, AttackTargets::One(t()))]
            }
            Barricade => vec![self_power(PowerId::Barricade, 1)],
            Bash => vec![attack(1), power(t(), PowerId::Vulnerable, m)],
            BattleTrance => vec![draw(v.cards), self_power(PowerId::NoDraw, 1)],
            Bloodletting => vec![lose_hp(v.hp_loss), Effect::GainEnergy { amount: v.energy }],
            BloodWall => vec![lose_hp(v.hp_loss), block()],
            Bludgeon => vec![attack(1)],
            // Damage equal to current block (0 base + 1 x block).
            BodySlam => vec![hit(v.magic * c.player.creature.block as f64, 1, AttackTargets::One(t()))],
            // MindBlast.cs: damage per card in the draw pile.
            MindBlast => vec![hit(v.magic * c.player.draw.len() as f64, 1, AttackTargets::One(t()))],
            Brand => vec![
                lose_hp(v.hp_loss),
                Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then: Then::Exhaust, can_skip: false },
                self_power(PowerId::Strength, m),
            ],
            Break => vec![attack(1), power(t(), PowerId::Vulnerable, m)],
            Breakthrough => vec![lose_hp(v.hp_loss), aoe(1)],
            // 4 + 2 per Vulnerable on the target.
            Bully => {
                let vuln = c.creature(t()).power_amount(PowerId::Vulnerable) as f64;
                vec![hit(v.damage + v.magic * vuln, 1, AttackTargets::One(t()))]
            }
            BurningPact => vec![
                Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then: Then::Exhaust, can_skip: false },
                draw(v.cards),
            ],
            Cascade => {
                let n = self.captured_x + if self.upgraded { 1 } else { 0 };
                vec![Effect::AutoPlayFromDrawTop { count: n.max(0) as u32, force_exhaust: false }]
            }
            Cinder => vec![
                attack(1),
                Effect::ExhaustRandomFromHand { filter: CardFilter::Any, then_add_damage_to: None },
            ],
            Colossus => vec![block(), self_power(PowerId::Colossus, m)],
            Conflagration => vec![aoe(v.hits)],
            Corruption => vec![self_power(PowerId::Corruption, 1)],
            CrimsonMantle => vec![self_power(PowerId::CrimsonMantle, m)],
            Cruelty => vec![self_power(PowerId::Cruelty, m)],
            DarkEmbrace => vec![self_power(PowerId::DarkEmbrace, 1)],
            DefendIronclad => vec![block()],
            DemonForm => vec![self_power(PowerId::DemonForm, m)],
            Dismantle => {
                let hits = if c.creature(t()).power_amount(PowerId::Vulnerable) > 0 { 2 } else { 1 };
                vec![attack(hits)]
            }
            Dominate => vec![power(t(), PowerId::Vulnerable, m), step(1)],
            DrumOfBattle => vec![draw(v.cards)],
            EvilEye => {
                let n = if c.stats.exhausted_this_turn > 0 { 2 } else { 1 };
                (0..n).map(|_| block()).collect()
            }
            // Energy per attack in hand.
            ExpectAFight => {
                let attacks = c.player.hand.iter().filter(|k| k.ty() == CardType::Attack).count() as i32;
                vec![Effect::GainEnergy { amount: attacks }, self_power(PowerId::NoEnergyGain, 1)]
            }
            Feed => vec![attack(1), step(1)],
            FeelNoPain => vec![self_power(PowerId::FeelNoPain, m)],
            FiendFire => {
                let n = c.player.hand.len() as u32;
                vec![Effect::ExhaustHand { filter: CardFilter::Any }, attack(n)]
            }
            FightMe => vec![attack(v.hits), self_power(PowerId::Strength, m), power(t(), PowerId::Strength, 1)],
            FlameBarrier => vec![block(), self_power(PowerId::FlameBarrier, m)],
            ForgottenRitual => {
                if c.stats.exhausted_this_turn > 0 {
                    vec![Effect::GainEnergy { amount: v.energy }]
                } else {
                    vec![]
                }
            }
            Havoc => vec![Effect::AutoPlayFromDrawTop { count: 1, force_exhaust: true }],
            Headbutt => vec![
                attack(1),
                Effect::Choose { from: Pile::Discard, filter: CardFilter::Any, then: Then::MoveTo(Pile::DrawTop), can_skip: false },
            ],
            Hellraiser => vec![self_power(PowerId::Hellraiser, 1)],
            Hemokinesis => vec![lose_hp(v.hp_loss), attack(1)],
            HowlFromBeyond => vec![aoe(1)],
            Impervious => vec![block()],
            InfernalBlade => vec![Effect::GenerateRandom {
                pool: GenPool::IroncladAttacks,
                count: 1,
                to: Pile::Hand,
                free_this_turn: true,
                distinct: true,
            }],
            Inferno => vec![self_power(PowerId::Inferno, m)],
            Inflame => vec![self_power(PowerId::Strength, m)],
            IronWave => vec![block(), attack(1)],
            Juggernaut => vec![self_power(PowerId::Juggernaut, m)],
            Juggling => vec![self_power(PowerId::Juggling, 1)],
            Mangle => vec![attack(1), power(t(), PowerId::Mangle, m)],
            MoltenFist => vec![attack(1), step(1)],
            NotYet => vec![Effect::Heal { target: me, amount: v.magic }],
            Offering => vec![lose_hp(v.hp_loss), Effect::GainEnergy { amount: v.energy }, draw(v.cards)],
            OneTwoPunch => vec![self_power(PowerId::OneTwoPunch, m)],
            PactsEnd => {
                if c.player.exhaust.len() as u32 >= v.cards {
                    vec![aoe(1)]
                } else {
                    vec![]
                }
            }
            // 6 + 2 per Strike-tagged card the player has in combat.
            PerfectedStrike => {
                let strikes = c.all_cards().filter(|k| k.has_tag(Tag::Strike)).count() as f64;
                vec![hit(v.damage + v.magic * strikes, 1, AttackTargets::One(t()))]
            }
            Pillage => vec![attack(1), step(1)],
            PommelStrike => vec![attack(1), draw(v.cards)],
            PrimalForce => vec![Effect::TransformHand {
                filter: CardFilter::Type(CardType::Attack),
                into: GiantRock,
                upgraded: self.upgraded,
            }],
            Pyre => vec![self_power(PowerId::Pyre, v.energy)],
            Rage => vec![self_power(PowerId::Rage, m)],
            Rampage => vec![attack(1), step(1)],
            Rupture => vec![self_power(PowerId::Rupture, m)],
            // Per non-attack in hand: exhaust it, then gain block.
            SecondWind => {
                let uids: Vec<u32> =
                    c.player.hand.iter().filter(|k| k.ty() != CardType::Attack).map(|k| k.uid).collect();
                uids.into_iter().flat_map(|u| [Effect::Exhaust { uid: u, ethereal: false }, block()]).collect()
            }
            SetupStrike => vec![attack(1), self_power(PowerId::SetupStrike, m)],
            ShrugItOff => vec![block(), draw(v.cards)],
            Spite => {
                let hits = if c.stats.hp_lost_this_turn { v.hits } else { 1 };
                vec![attack(hits)]
            }
            Stampede => vec![self_power(PowerId::Stampede, m)],
            Stoke => {
                let n = c.player.hand.len() as u32;
                vec![
                    Effect::ExhaustHand { filter: CardFilter::Any },
                    Effect::GenerateRandom {
                        pool: GenPool::Ironclad,
                        count: n,
                        to: Pile::Hand,
                        free_this_turn: false,
                        distinct: false,
                    },
                ]
            }
            Stomp => vec![aoe(1)],
            StoneArmor => vec![self_power(PowerId::Plating, m)],
            StrikeIronclad => vec![attack(1)],
            SwordBoomerang => vec![hit(v.damage, v.hits, AttackTargets::RandomOpponent)],
            Taunt => vec![block(), power(t(), PowerId::Vulnerable, m)],
            // 1 hit + 1 per unblocked hit the player has taken this combat.
            TearAsunder => vec![attack(1 + c.stats.unblocked_hits_taken)],
            Thrash => vec![
                attack(v.hits),
                Effect::ExhaustRandomFromHand {
                    filter: CardFilter::Type(CardType::Attack),
                    then_add_damage_to: Some(uid),
                },
            ],
            Thunderclap => {
                let mut e = vec![aoe(1)];
                for i in c.living_enemies() {
                    e.push(power(CreatureRef::Enemy(i), PowerId::Vulnerable, m));
                }
                e
            }
            Tremble => vec![power(t(), PowerId::Vulnerable, m)],
            TrueGrit => {
                let mut e = vec![block()];
                if self.upgraded {
                    e.push(Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then: Then::Exhaust, can_skip: false });
                } else {
                    e.push(Effect::ExhaustRandomFromHand { filter: CardFilter::Any, then_add_damage_to: None });
                }
                e
            }
            TwinStrike => vec![attack(v.hits)],
            Unmovable => vec![self_power(PowerId::Unmovable, 1)],
            Unrelenting => vec![attack(1), self_power(PowerId::FreeAttack, 1)],
            Uppercut => vec![attack(1), power(t(), PowerId::Weak, m), power(t(), PowerId::Vulnerable, m)],
            Vicious => vec![self_power(PowerId::Vicious, v.cards as i32)],
            Whirlwind => vec![aoe(self.captured_x.max(0) as u32)],
            Slimed => vec![draw(v.cards)],
            GiantRock => vec![attack(1)],
            Wound | Dazed | Burn | Infection | AscendersBane | Beckon | Soot | Wither => vec![],
            Luminesce => vec![Effect::GainEnergy { amount: v.energy }],
            Toxic | Disintegration | MindRot | Sloth | WasteAway => vec![],
            // FranticEscape.cs: the Sandpit gets a turn longer, and the card
            // costs 1 more for the rest of the combat.
            FranticEscape => {
                let pit = c.living_enemies().find(|&i| c.enemies[i].creature.power(PowerId::Sandpit).is_some());
                let mut e: Vec<Effect> = pit
                    .map(|i| Effect::ApplyPower { target: CreatureRef::Enemy(i), id: PowerId::Sandpit, amount: 1, applier: None })
                    .into_iter()
                    .collect();
                e.push(Effect::CostThisCombat { uid, delta: 1 });
                e
            }
            // Enthralled and Spore Mind are the only playable curses and
            // neither does anything; the rest are unplayable.
            BadLuck | Clumsy | CurseOfTheBell | Debt | Decay | Doubt | Enthralled | Folly | Greed | Guilty
            | Injury | Normality | PoorSleep | Regret | Shame | SporeMind | Writhe => vec![],
        }
    }

    /// Continuations for cards that read state their earlier effects changed.
    pub fn step(&self, c: &Combat, target: Option<CreatureRef>, step: u8) -> Vec<Effect> {
        use CardId::*;
        let me = CreatureRef::Player;
        let v = self.vars();
        match (self.id, step) {
            // Strength equal to the target's Vulnerable after applying it.
            (Dominate, 1) => {
                let n = c.creature(target.unwrap()).power_amount(PowerId::Vulnerable);
                vec![Effect::ApplyPower { target: me, id: PowerId::Strength, amount: n, applier: Some(me) }]
            }
            // Max HP if the attack killed the target. Minions never count as
            // Fatal (`MinionPower.ShouldOwnerDeathTriggerFatal`).
            (Feed, 1) => {
                let tgt = target.unwrap();
                if !c.creature(tgt).alive() && c.creature(tgt).power(PowerId::Minion).is_none() {
                    vec![Effect::GainMaxHp { target: me, amount: v.magic as i32 }]
                } else {
                    vec![]
                }
            }
            // Double the target's Vulnerable if it has any.
            (MoltenFist, 1) => {
                let tgt = target.unwrap();
                let n = if c.creature(tgt).alive() { c.creature(tgt).power_amount(PowerId::Vulnerable) } else { 0 };
                if n > 0 {
                    vec![Effect::ApplyPower { target: tgt, id: PowerId::Vulnerable, amount: n, applier: Some(me) }]
                } else {
                    vec![]
                }
            }
            // Draw; keep drawing while the drawn card is an attack.
            (Pillage, 1) => vec![Effect::Draw { count: 1, from_hand_draw: false }, Effect::CardStep { uid: self.uid, target, step: 2 }],
            (Pillage, 2) => {
                let again = c.stats.last_drawn.and_then(|u| c.find_card(u)).is_some_and(|k| k.ty() == CardType::Attack)
                    && c.player.hand.len() < crate::combat::MAX_HAND;
                if again {
                    self.step(c, target, 1)
                } else {
                    vec![]
                }
            }
            // Rampage: the growth into `extra_damage` is applied by the
            // combat loop as this step starts.
            (Rampage, 1) => vec![],
            _ => vec![],
        }
    }
}
