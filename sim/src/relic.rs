//! Relics. `Models/RelicModel.cs` plus `Models/Relics/*.cs` for every
//! Common, Uncommon, Rare, Shop, and Ironclad relic, and the event pool
//! (`RelicPools/EventRelicPool.cs`) where it reaches a fight. Relics live on the run
//! and persist counters across combats (`Nunchaku.AttacksPlayed`, Pen Nib's
//! count, Girya's lifts, Lizard Tail's use). Combat clones them in and the
//! caller reads them back from `Combat::relics` afterwards.
//!
//! Hooks are `impl Combat` methods here so they can read combat state and
//! return effects, dispatched from the matching points in combat.rs.

use crate::card::{Card, Tag};
use crate::combat::{Combat, RoomKind};
use crate::effect::{AttackTargets, CardFilter, Effect, GenPool, Pile};
use crate::ids::{CardId, PowerId};
use crate::types::{CardRarity, CardType, CreatureRef, Keyword, Side, TargetType, ValueProp};

/// `Models/Relics/<Name>.cs`. Event pool relics that never reach a fight
/// are not here but in `gen::INERT_RELICS`, and the other characters'
/// starter upgrades (Infused Core, Phylactery Unbound, Divine Destiny, Ring
/// of the Drake) are left out with those characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelicId {
    // Starter
    BurningBlood,
    // Common
    AmethystAubergine, Anchor, BagOfMarbles, BagOfPreparation, BloodVial, BookOfFiveRings,
    BronzeScales, CentennialPuzzle, FestivePopper, Gorget, HappyFlower, JuzuBracelet, Lantern,
    MealTicket, OddlySmoothStone, Pendulum, PotionBelt, RedMask, RedSkull, RegalPillow,
    Strawberry, StrikeDummy, Vajra, VenerableTeaSet, WarPaint, Whetstone,
    // Uncommon
    Akabeko, BowlerHat, Candelabra, EternalFeather, GremlinHorn, HornCleat, JossPaper,
    Kusarigama, LastingCandy, LetterOpener, LuckyFysh, MercuryHourglass, MiniatureCannon,
    Nunchaku, Orichalcum, OrnamentalFan, Pantograph, PaperPhrog, ParryingShield, Pear, PenNib,
    Permafrost, PetrifiedToad, Planisphere, ReptileTrinket, RippleBasin, SelfFormingClay,
    SparklingRouge, StoneCracker, TinyMailbox, TuningFork, Vambrace,
    // Rare
    ArtOfWar, BeatingRemnant, Bellows, CaptainsWheel, Chandelier, CharonsAshes, CloakClasp,
    DemonTongue, FrozenEgg, GamblingChip, GamePiece, Girya, IceCream, IntimidatingHelmet, Kunai,
    LizardTail, Mango, MeatOnTheBone, MoltenEgg, MummifiedHand, OldCoin, Pocketwatch,
    PrayerWheel, RainbowRing, RazorTooth, RuinedHelmet, Shovel, Shuriken, StoneCalendar,
    SturdyClamp, TheCourier, ToxicEgg, TungstenRod, UnceasingTop, UnsettlingLamp,
    VexingPuzzlebox, WhiteBeastStatue, WhiteStar,
    // Shop
    BeltBuckle, Bread, Brimstone, BurningSticks, Cauldron, ChemicalX, DingyRug, DollysMirror,
    DragonFruit, GhostSeed, GnarledHammer, Kifuda, LavaLamp, LeesWaffle, MembershipCard,
    MiniatureTent, MysticLighter, Orrery, PunchDagger, RingingTriangle, RoyalStamp,
    ScreamingFlagon, SlingOfCourage, TheAbacus, Toolbox, WingCharm,
    // Ancient
    PaelsFlesh,
    /// Run-time only (a card pick when obtained); nothing in combat.
    LeadPaperweight,
    // Event. Appended after the vocabulary was pinned; new ids go at the end.
    TheBoot,
    // The rest of the event pool, event and ancient rarity alike.
    BigMushroom, BiiigHug, BlackBlood, BlessedAntler, BloodSoakedRose, BoneTea, BoomingConch,
    BrilliantScarf, ChoicesParadox, Crossbow, DaughterOfTheWind, DelicateFrond, DiamondDiadem,
    Ectoplasm, EmberTea, FakeAnchor, FakeBloodVial, FakeHappyFlower, FakeOrichalcum, FakeSneckoEye,
    FakeStrikeDummy, FakeVenerableTeaSet, Fiddle, ForgottenSoul, FurCoat, HandDrill, HistoryCourse,
    IronClub, JeweledMask, LostWisp, MrStruggles, MusicBox, PaelsBlood, PaelsEye, PaelsLegion,
    PaelsTears, PhilosophersStone, PollinousCore, PrismaticGem, PumpkinCandle, RadiantPearl,
    RoyalPoison, RunicPyramid, Sai, SealOfGold, SneckoEye, Sozu, SpikedGauntlets, SwordOfJade,
    TeaOfDiscourtesy, ThrowingAxe, ToastyMittens, VelvetChoker, WhisperingEarring,
    // Act 2 (Hive): an ancient's relic met in a recorded act 2 run.
    VeryHotCocoa,
}

/// Every relic, for generators and tests.
pub const ALL: &[RelicId] = &[
    RelicId::BurningBlood, RelicId::AmethystAubergine, RelicId::Anchor, RelicId::BagOfMarbles, RelicId::BagOfPreparation,
    RelicId::BloodVial, RelicId::BookOfFiveRings, RelicId::BronzeScales, RelicId::CentennialPuzzle, RelicId::FestivePopper,
    RelicId::Gorget, RelicId::HappyFlower, RelicId::JuzuBracelet, RelicId::Lantern, RelicId::MealTicket,
    RelicId::OddlySmoothStone, RelicId::Pendulum, RelicId::PotionBelt, RelicId::RedMask, RelicId::RedSkull,
    RelicId::RegalPillow, RelicId::Strawberry, RelicId::StrikeDummy, RelicId::Vajra, RelicId::VenerableTeaSet,
    RelicId::WarPaint, RelicId::Whetstone, RelicId::Akabeko, RelicId::BowlerHat, RelicId::Candelabra,
    RelicId::EternalFeather, RelicId::GremlinHorn, RelicId::HornCleat, RelicId::JossPaper, RelicId::Kusarigama,
    RelicId::LastingCandy, RelicId::LetterOpener, RelicId::LuckyFysh, RelicId::MercuryHourglass, RelicId::MiniatureCannon,
    RelicId::Nunchaku, RelicId::Orichalcum, RelicId::OrnamentalFan, RelicId::Pantograph, RelicId::PaperPhrog,
    RelicId::ParryingShield, RelicId::Pear, RelicId::PenNib, RelicId::Permafrost, RelicId::PetrifiedToad,
    RelicId::Planisphere, RelicId::ReptileTrinket, RelicId::RippleBasin, RelicId::SelfFormingClay, RelicId::SparklingRouge,
    RelicId::StoneCracker, RelicId::TinyMailbox, RelicId::TuningFork, RelicId::Vambrace, RelicId::ArtOfWar,
    RelicId::BeatingRemnant, RelicId::Bellows, RelicId::CaptainsWheel, RelicId::Chandelier, RelicId::CharonsAshes,
    RelicId::CloakClasp, RelicId::DemonTongue, RelicId::FrozenEgg, RelicId::GamblingChip, RelicId::GamePiece,
    RelicId::Girya, RelicId::IceCream, RelicId::IntimidatingHelmet, RelicId::Kunai, RelicId::LizardTail, RelicId::Mango,
    RelicId::MeatOnTheBone, RelicId::MoltenEgg, RelicId::MummifiedHand, RelicId::OldCoin, RelicId::Pocketwatch,
    RelicId::PrayerWheel, RelicId::RainbowRing, RelicId::RazorTooth, RelicId::RuinedHelmet, RelicId::Shovel,
    RelicId::Shuriken, RelicId::StoneCalendar, RelicId::SturdyClamp, RelicId::TheCourier, RelicId::ToxicEgg,
    RelicId::TungstenRod, RelicId::UnceasingTop, RelicId::UnsettlingLamp, RelicId::VexingPuzzlebox,
    RelicId::WhiteBeastStatue, RelicId::WhiteStar, RelicId::BeltBuckle, RelicId::Bread, RelicId::Brimstone,
    RelicId::BurningSticks, RelicId::Cauldron, RelicId::ChemicalX, RelicId::DingyRug, RelicId::DollysMirror,
    RelicId::DragonFruit, RelicId::GhostSeed, RelicId::GnarledHammer, RelicId::Kifuda, RelicId::LavaLamp,
    RelicId::LeesWaffle, RelicId::MembershipCard, RelicId::MiniatureTent, RelicId::MysticLighter, RelicId::Orrery,
    RelicId::PunchDagger, RelicId::RingingTriangle, RelicId::RoyalStamp, RelicId::ScreamingFlagon,
    RelicId::SlingOfCourage, RelicId::TheAbacus, RelicId::Toolbox, RelicId::WingCharm,
    RelicId::PaelsFlesh, RelicId::LeadPaperweight, RelicId::TheBoot, RelicId::BigMushroom, RelicId::BiiigHug,
    RelicId::BlackBlood, RelicId::BlessedAntler, RelicId::BloodSoakedRose, RelicId::BoneTea, RelicId::BoomingConch,
    RelicId::BrilliantScarf, RelicId::ChoicesParadox, RelicId::Crossbow, RelicId::DaughterOfTheWind,
    RelicId::DelicateFrond, RelicId::DiamondDiadem, RelicId::Ectoplasm, RelicId::EmberTea, RelicId::FakeAnchor,
    RelicId::FakeBloodVial, RelicId::FakeHappyFlower, RelicId::FakeOrichalcum, RelicId::FakeSneckoEye,
    RelicId::FakeStrikeDummy, RelicId::FakeVenerableTeaSet, RelicId::Fiddle, RelicId::ForgottenSoul, RelicId::FurCoat,
    RelicId::HandDrill, RelicId::HistoryCourse, RelicId::IronClub, RelicId::JeweledMask, RelicId::LostWisp,
    RelicId::MrStruggles, RelicId::MusicBox, RelicId::PaelsBlood, RelicId::PaelsEye, RelicId::PaelsLegion,
    RelicId::PaelsTears, RelicId::PhilosophersStone, RelicId::PollinousCore, RelicId::PrismaticGem,
    RelicId::PumpkinCandle, RelicId::RadiantPearl, RelicId::RoyalPoison, RelicId::RunicPyramid, RelicId::Sai,
    RelicId::SealOfGold, RelicId::SneckoEye, RelicId::Sozu, RelicId::SpikedGauntlets, RelicId::SwordOfJade,
    RelicId::TeaOfDiscourtesy, RelicId::ThrowingAxe, RelicId::ToastyMittens, RelicId::VelvetChoker,
    RelicId::WhisperingEarring, RelicId::VeryHotCocoa,
];

/// Ancient rarity, handed out by the Ancients that open each act, and the
/// starter upgrade Touch of Orobas swaps in. Act 1 only sees the ones Neow
/// offers (`Events/Neow.cs`), which is why Booming Conch is missing here.
pub const ANCIENT: &[RelicId] = &[
    RelicId::PaelsFlesh, RelicId::BiiigHug, RelicId::BlackBlood, RelicId::BlessedAntler, RelicId::BloodSoakedRose,
    RelicId::BrilliantScarf, RelicId::ChoicesParadox, RelicId::Crossbow, RelicId::DelicateFrond,
    RelicId::DiamondDiadem, RelicId::Ectoplasm, RelicId::Fiddle, RelicId::FurCoat, RelicId::IronClub,
    RelicId::JeweledMask, RelicId::MusicBox, RelicId::PaelsBlood, RelicId::PaelsEye, RelicId::PaelsLegion,
    RelicId::PaelsTears, RelicId::PhilosophersStone, RelicId::PrismaticGem, RelicId::PumpkinCandle,
    RelicId::RadiantPearl, RelicId::RunicPyramid, RelicId::Sai, RelicId::SealOfGold, RelicId::SneckoEye,
    RelicId::Sozu, RelicId::SpikedGauntlets, RelicId::ThrowingAxe, RelicId::ToastyMittens, RelicId::VelvetChoker,
    RelicId::WhisperingEarring, RelicId::VeryHotCocoa,
];

/// A relic instance on the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relic {
    pub id: RelicId,
    /// Persistent counter: Nunchaku attacks, Pen Nib attacks, Tuning Fork
    /// skills, Joss Paper exhausts, Happy Flower / Pendulum turns, Girya lifts,
    /// Iron Club plays, Pollinous Core turns. Charged relics count what is
    /// left (Ember Tea combats, Pumpkin Candle kindling), a primed one holds
    /// 1 (Fake Venerable Tea Set), and Fur Coat holds 1 in a marked fight.
    /// `gen::from_start` reads the game's value from the recording.
    pub counter: i32,
    /// Per-combat counter, reset at combat start.
    pub combat_counter: i32,
    /// Per-combat scratch: Pen Nib's doubled card uid, Vambrace's triggering
    /// card, Unsettling Lamp's triggering card.
    pub scratch: i32,
    /// Per-combat flag (Centennial Puzzle used, Permafrost used, etc.).
    pub used: bool,
    /// Persistent flag: Lizard Tail spent, Venerable Tea Set primed.
    pub flag: bool,
}

impl Relic {
    pub fn new(id: RelicId) -> Self {
        // The charges a fresh one comes with.
        let counter = match id {
            RelicId::EmberTea | RelicId::PumpkinCandle => 5,
            RelicId::BoneTea | RelicId::TeaOfDiscourtesy => 1,
            _ => 0,
        };
        Self { id, counter, combat_counter: 0, scratch: 0, used: false, flag: false }
    }
}

fn gain_block(amount: i32) -> Effect {
    Effect::GainBlock { target: CreatureRef::Player, amount: amount as f64, props: ValueProp::UNPOWERED, card: None }
}

fn self_power(id: PowerId, amount: i32) -> Effect {
    Effect::ApplyPower { target: CreatureRef::Player, id, amount, applier: Some(CreatureRef::Player) }
}

fn all_enemies_damage(amount: i32) -> Effect {
    Effect::DamageAllEnemies { amount: amount as f64, props: ValueProp::UNPOWERED, dealer: CreatureRef::Player }
}

fn random_enemy_damage(amount: i32) -> Effect {
    Effect::Attack {
        dealer: CreatureRef::Player,
        base: amount as f64,
        hits: 1,
        targets: AttackTargets::RandomOpponent,
        props: ValueProp::UNPOWERED,
        card: None,
    }
}

fn draw(n: u32) -> Effect {
    Effect::Draw { count: n, from_hand_draw: false }
}

fn generate(id: CardId, to: Pile) -> Effect {
    Effect::GenerateCard { id, upgraded: false, to, free_this_turn: false }
}

impl Combat {
    pub fn has_relic(&self, id: RelicId) -> bool {
        self.relics.iter().any(|r| r.id == id)
    }

    fn relic_mut(&mut self, id: RelicId) -> Option<&mut Relic> {
        self.relics.iter_mut().find(|r| r.id == id)
    }

    /// `AfterRoomEntered(CombatRoom)` + `BeforeCombatStart` for relics, in
    /// relic order. Runs after monsters are placed, before the first turn.
    pub(crate) fn relic_before_combat_start(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let mut out = vec![];
        let room = self.room;
        let enemies: Vec<usize> = self.living_enemies().collect();
        // AfterRoomEntered, which `CombatRoom.StartCombat` runs before
        // anything reaches BeforeCombatStart.
        for r in &mut self.relics {
            match r.id {
                EmberTea if r.counter > 0 => {
                    r.counter -= 1;
                    out.push(self_power(PowerId::Strength, 2));
                }
                SwordOfJade => out.push(self_power(PowerId::Strength, 3)),
                PhilosophersStone => out.extend(enemies.iter().map(|&e| Effect::ApplyPower {
                    target: CreatureRef::Enemy(e),
                    id: PowerId::Strength,
                    amount: 1,
                    applier: None,
                })),
                _ => {}
            }
        }
        // FurCoat.BeforeCombatStart: a marked fight starts every enemy at 1 HP.
        if self.relics.iter().any(|r| r.id == FurCoat && r.counter > 0) {
            for &e in &enemies {
                self.enemies[e].creature.hp = 1;
            }
        }
        self.delicate_frond();
        let no_potions = self.potions.iter().all(|p| p.is_none());
        let hp_low = self.player.creature.hp * 2 <= self.player.creature.max_hp;
        for i in 0..self.relics.len() {
            let r = &mut self.relics[i];
            r.combat_counter = 0;
            r.scratch = 0;
            r.used = false;
            match r.id {
                Anchor => out.push(gain_block(10)),
                BronzeScales => out.push(self_power(PowerId::Thorns, 3)),
                Gorget => out.push(self_power(PowerId::Plating, 4)),
                OddlySmoothStone => out.push(self_power(PowerId::Dexterity, 1)),
                Vajra => out.push(self_power(PowerId::Strength, 1)),
                Girya if r.counter > 0 => out.push(self_power(PowerId::Strength, r.counter)),
                SlingOfCourage if room == RoomKind::Elite => out.push(self_power(PowerId::Strength, 2)),
                Pantograph if room == RoomKind::Boss => out.push(Effect::Heal { target: CreatureRef::Player, amount: 25.0 }),
                BeltBuckle if no_potions => {
                    r.used = true;
                    out.push(self_power(PowerId::Dexterity, 2));
                }
                // RedSkull: +3 Strength while at or below half HP.
                RedSkull if hp_low => {
                    r.used = true;
                    out.push(self_power(PowerId::Strength, 3));
                }
                FakeAnchor => out.push(gain_block(4)),
                FakeSneckoEye | SneckoEye => out.push(self_power(PowerId::Confused, 1)),
                TeaOfDiscourtesy if r.counter > 0 => {
                    r.counter -= 1;
                    out.extend((0..2).map(|_| generate(CardId::Dazed, Pile::DrawRandom)));
                }
                _ => {}
            }
        }
        if self.has_relic(GhostSeed) {
            for c in self.player.draw.iter_mut() {
                ghost_seed_mark(c);
            }
        }
        out
    }

    /// `Hook.BeforeSideTurnStart` for relics (player side).
    pub(crate) fn relic_before_side_turn_start(&mut self, side: Side) -> Vec<Effect> {
        use RelicId::*;
        if side != Side::Player {
            return vec![];
        }
        let turn = self.player.turn;
        let mut out = vec![];
        let enemies: Vec<usize> = self.living_enemies().collect();
        for r in &mut self.relics {
            match r.id {
                BagOfMarbles if turn <= 1 => {
                    for &e in &enemies {
                        out.push(Effect::ApplyPower {
                            target: CreatureRef::Enemy(e),
                            id: PowerId::Vulnerable,
                            amount: 1,
                            applier: Some(CreatureRef::Player),
                        });
                    }
                }
                RedMask if turn <= 1 => {
                    for &e in &enemies {
                        out.push(Effect::ApplyPower {
                            target: CreatureRef::Enemy(e),
                            id: PowerId::Weak,
                            amount: 1,
                            applier: Some(CreatureRef::Player),
                        });
                    }
                }
                // Per-turn counters.
                Kunai | Shuriken | OrnamentalFan | RainbowRing | BeatingRemnant | DemonTongue | Orichalcum => {
                    r.combat_counter = 0;
                    r.used = false;
                }
                // Pocketwatch: cards played last turn = this turn's count.
                Pocketwatch => {
                    r.scratch = r.combat_counter;
                    r.combat_counter = 0;
                }
                MusicBox => r.used = false,
                VelvetChoker => r.combat_counter = 0,
                PollinousCore => r.counter += 1,
                _ => {}
            }
        }
        out
    }

    /// `Hook.ShouldPlayerResetEnergy`: Ice Cream keeps leftover energy.
    pub(crate) fn relic_should_reset_energy(&self) -> bool {
        !(self.has_relic(RelicId::IceCream) && self.player.turn > 1)
    }

    /// `Hook.AfterEnergyReset`.
    pub(crate) fn relic_after_energy_reset(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                VenerableTeaSet if r.flag => {
                    r.flag = false;
                    out.push(Effect::GainEnergy { amount: 2 });
                }
                FakeVenerableTeaSet if r.counter > 0 => {
                    r.counter = 0;
                    out.push(Effect::GainEnergy { amount: 1 });
                }
                // ArtOfWar: `scratch` = attacks played last turn.
                ArtOfWar if turn > 1 => {
                    if r.scratch == 0 {
                        out.push(Effect::GainEnergy { amount: 1 });
                    }
                    r.scratch = 0;
                    r.combat_counter = 0;
                }
                _ => {}
            }
        }
        out
    }

    /// `Hook.ModifyHandDraw`, `AfterModifyingHandDraw`, then the `Late` pass.
    pub(crate) fn relic_modify_hand_draw(&mut self, count: u32) -> u32 {
        use RelicId::*;
        let turn = self.player.turn;
        let elite = self.room == RoomKind::Elite;
        let mut n = count;
        for r in &mut self.relics {
            match r.id {
                BagOfPreparation if turn <= 1 => n += 2,
                Pocketwatch if turn > 1 && r.scratch <= 3 => n += 3,
                BigMushroom if turn == 1 => n = n.saturating_sub(2),
                BoomingConch if turn <= 1 && elite => n += 2,
                PaelsBlood => n += 1,
                SneckoEye => n += 2,
                PollinousCore if r.counter == 4 => {
                    r.counter = 0;
                    n += 2;
                }
                _ => {}
            }
        }
        if self.has_relic(Fiddle) {
            n += 2;
        }
        n
    }

    /// `Hook.ModifyMaxEnergy` for relics: Bread after turn 1, Pael's Flesh
    /// from turn 3, a flat +1 from the ancient energy relics.
    pub(crate) fn relic_modify_max_energy(&self, amount: i32) -> i32 {
        let mut a = amount;
        if self.has_relic(RelicId::Bread) && self.player.turn > 1 {
            a += 1;
        }
        if self.has_relic(RelicId::PaelsFlesh) && self.player.turn >= 3 {
            a += 1;
        }
        for r in &self.relics {
            use RelicId::*;
            match r.id {
                BlessedAntler | BloodSoakedRose | Ectoplasm | PhilosophersStone | PrismaticGem | Sozu
                | SpikedGauntlets | VelvetChoker | WhisperingEarring => a += 1,
                PumpkinCandle if r.counter > 0 => a += 1,
                _ => {}
            }
        }
        a
    }

    /// `Hook.AfterSideTurnStart` for relics (player side).
    pub(crate) fn relic_after_side_turn_start(&mut self, side: Side) -> Vec<Effect> {
        use RelicId::*;
        if side != Side::Player {
            return vec![];
        }
        let turn = self.player.turn;
        let elite = self.room == RoomKind::Elite;
        let enemies: Vec<usize> = self.living_enemies().collect();
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                FakeHappyFlower => {
                    r.counter = (r.counter + 1) % 5;
                    if r.counter == 0 {
                        out.push(Effect::GainEnergy { amount: 1 });
                    }
                }
                BoneTea if turn <= 1 && r.counter > 0 => {
                    r.counter -= 1;
                    out.push(Effect::UpgradeHand);
                }
                BoomingConch if turn <= 1 && elite => out.push(Effect::GainEnergy { amount: 1 }),
                VeryHotCocoa if turn <= 1 => out.push(Effect::GainEnergy { amount: 4 }),
                Crossbow => out.push(Effect::GenerateRandom {
                    pool: GenPool::IroncladAttacks,
                    count: 1,
                    to: Pile::Hand,
                    free_this_turn: true,
                    distinct: true,
                    upgraded: false,
                }),
                PaelsLegion => r.combat_counter -= 1,
                // PaelsTears: `used` is HadLeftoverEnergy from last turn's end.
                PaelsTears if r.used => out.push(Effect::GainEnergy { amount: 2 }),
                Sai => out.push(gain_block(7)),
                SealOfGold if self.gold >= 5 => {
                    self.gold -= 5;
                    out.push(Effect::GainEnergy { amount: 1 });
                }
                HappyFlower => {
                    r.counter = (r.counter + 1) % 3;
                    if r.counter == 0 {
                        out.push(Effect::GainEnergy { amount: 1 });
                    }
                }
                Lantern if turn <= 1 => out.push(Effect::GainEnergy { amount: 1 }),
                Chandelier if turn == 3 => out.push(Effect::GainEnergy { amount: 3 }),
                Candelabra if turn == 2 => out.push(Effect::GainEnergy { amount: 2 }),
                Bread if turn == 1 => out.push(Effect::LoseEnergy { amount: 2 }),
                Brimstone => {
                    out.push(self_power(PowerId::Strength, 2));
                    for &e in &enemies {
                        out.push(Effect::ApplyPower {
                            target: CreatureRef::Enemy(e),
                            id: PowerId::Strength,
                            amount: 1,
                            applier: None,
                        });
                    }
                }
                Akabeko if turn <= 1 => out.push(self_power(PowerId::Vigor, 8)),
                LetterOpener if turn > 1 => r.combat_counter = 0,
                _ => {}
            }
        }
        out
    }

    /// `Hook.AfterPlayerTurnStart` for relics.
    pub(crate) fn relic_after_player_turn_start(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                BloodVial if turn <= 1 => out.push(Effect::Heal { target: CreatureRef::Player, amount: 2.0 }),
                FestivePopper if turn == 1 => out.push(all_enemies_damage(9)),
                Pendulum => {
                    r.counter = (r.counter + 1) % 3;
                    if r.counter == 0 {
                        out.push(draw(1));
                    }
                }
                MercuryHourglass => out.push(all_enemies_damage(3)),
                VexingPuzzlebox if turn == 1 => out.push(Effect::GenerateRandom {
                    pool: GenPool::Ironclad,
                    count: 1,
                    to: Pile::Hand,
                    free_this_turn: true,
                    distinct: true,
                    upgraded: false,
                }),
                MrStruggles => out.push(all_enemies_damage(turn as i32)),
                RoyalPoison if turn <= 1 => out.push(Effect::Damage {
                    target: CreatureRef::Player,
                    amount: 4.0,
                    props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED),
                    dealer: None,
                    card: None,
                }),
                ChoicesParadox if turn == 1 => {
                    out.push(Effect::OfferRandom { pool: GenPool::Ironclad, count: 5, free: false, retain: true })
                }
                _ => {}
            }
        }
        // AfterPlayerTurnStartLate.
        if turn <= 1 && self.has_relic(FakeBloodVial) {
            out.push(Effect::Heal { target: CreatureRef::Player, amount: 1.0 });
        }
        out
    }

    /// `Hook.BeforeHandDraw`, after the energy reset.
    pub(crate) fn relic_before_hand_draw(&self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let mut out = vec![];
        for r in &self.relics {
            match r.id {
                BlessedAntler if turn == 1 => out.extend((0..3).map(|_| generate(CardId::Dazed, Pile::DrawRandom))),
                JeweledMask if turn <= 1 => out.push(Effect::RelicStep { id: JeweledMask, step: 0 }),
                RadiantPearl if turn == 1 => out.push(generate(CardId::Luminesce, Pile::Hand)),
                ToastyMittens => out.push(Effect::RelicStep { id: ToastyMittens, step: 0 }),
                _ => {}
            }
        }
        out
    }

    /// `Hook.AfterAutoPrePlayPhaseEntered`, then its `Late` pass: once the
    /// turn is fully set up, before the player acts.
    pub(crate) fn relic_auto_pre_play(&self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let mut out = vec![];
        if turn > 1 && self.has_relic(HistoryCourse) {
            out.push(Effect::RelicStep { id: HistoryCourse, step: 0 });
        }
        if turn <= 1 && self.has_relic(WhisperingEarring) {
            out.push(Effect::RelicStep { id: WhisperingEarring, step: 0 });
        }
        out
    }

    /// `Hook.BeforeCardPlayed` for relics. `energy_paid` is the card's cost.
    pub(crate) fn relic_before_card_played(&mut self, card: &Card, energy_paid: i32) -> Vec<Effect> {
        use RelicId::*;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                IntimidatingHelmet if energy_paid >= 2 => out.push(gain_block(4)),
                // PenNib: every 10th attack deals double; `scratch` holds its uid.
                PenNib if card.ty() == CardType::Attack => {
                    r.counter = (r.counter + 1) % 10;
                    if r.counter == 0 {
                        r.scratch = card.uid as i32;
                    }
                }
                // MusicBox: the turn's first attack is the one it copies.
                MusicBox if r.scratch == 0 && !r.used && card.ty() == CardType::Attack => r.scratch = card.uid as i32,
                _ => {}
            }
        }
        out
    }

    /// `Hook.AfterCardPlayed` for relics.
    pub(crate) fn relic_after_card_played(&mut self, card: &Card) -> Vec<Effect> {
        use RelicId::*;
        let ty = card.ty();
        let hand_costing: Vec<u32> = self
            .player
            .hand
            .iter()
            .filter(|c| c.uid != card.uid && self.cost(c) > 0)
            .map(|c| c.uid)
            .collect();
        let mut out = vec![];
        let mut mummified: Option<u32> = None;
        for r in &mut self.relics {
            match r.id {
                GamePiece if ty == CardType::Power => out.push(draw(1)),
                MummifiedHand if ty == CardType::Power => {
                    if let Some(&u) = self.rngs.card_selection.pick(&hand_costing) {
                        mummified = Some(u);
                    }
                }
                RazorTooth if matches!(ty, CardType::Attack | CardType::Skill) => {
                    out.push(Effect::Upgrade { uid: card.uid });
                }
                RainbowRing if !r.used => {
                    // bits: 1 attack, 2 skill, 4 power
                    r.combat_counter |= match ty {
                        CardType::Attack => 1,
                        CardType::Skill => 2,
                        CardType::Power => 4,
                        _ => 0,
                    };
                    if r.combat_counter == 7 {
                        r.used = true;
                        out.push(self_power(PowerId::Strength, 1));
                        out.push(self_power(PowerId::Dexterity, 1));
                    }
                }
                Kunai if ty == CardType::Attack => {
                    r.combat_counter += 1;
                    if r.combat_counter % 3 == 0 {
                        out.push(self_power(PowerId::Dexterity, 1));
                    }
                }
                Shuriken if ty == CardType::Attack => {
                    r.combat_counter += 1;
                    if r.combat_counter % 3 == 0 {
                        out.push(self_power(PowerId::Strength, 1));
                    }
                }
                Kusarigama if ty == CardType::Attack => {
                    r.combat_counter += 1;
                    if r.combat_counter % 3 == 0 {
                        out.push(random_enemy_damage(6));
                    }
                }
                OrnamentalFan if ty == CardType::Attack => {
                    r.combat_counter += 1;
                    if r.combat_counter % 3 == 0 {
                        out.push(gain_block(4));
                    }
                }
                LetterOpener if ty == CardType::Skill => {
                    r.combat_counter += 1;
                    if r.combat_counter % 3 == 0 {
                        out.push(all_enemies_damage(5));
                    }
                }
                Nunchaku if ty == CardType::Attack => {
                    r.counter += 1;
                    if r.counter % 10 == 0 {
                        out.push(Effect::GainEnergy { amount: 1 });
                    }
                }
                TuningFork if ty == CardType::Skill => {
                    r.counter += 1;
                    if r.counter >= 10 {
                        r.counter -= 10;
                        out.push(gain_block(7));
                    }
                }
                ArtOfWar if ty == CardType::Attack => r.combat_counter = 1,
                Pocketwatch => r.combat_counter += 1,
                Permafrost if ty == CardType::Power && !r.used => {
                    r.used = true;
                    out.push(gain_block(7));
                }
                RippleBasin if ty == CardType::Attack => r.used = true,
                // Vambrace: the triggering card's play is over; doubling is spent.
                Vambrace if r.scratch == card.uid as i32 => r.used = true,
                PenNib if r.scratch == card.uid as i32 => r.scratch = 0,
                UnsettlingLamp if r.scratch == card.uid as i32 => r.used = true,
                DaughterOfTheWind if ty == CardType::Attack => out.push(gain_block(1)),
                LostWisp if ty == CardType::Power => out.push(all_enemies_damage(8)),
                IronClub => {
                    r.counter = (r.counter + 1) % 4;
                    if r.counter == 0 {
                        out.push(draw(1));
                    }
                }
                DiamondDiadem | VelvetChoker => r.combat_counter += 1,
                MusicBox if r.scratch == card.uid as i32 => {
                    r.scratch = 0;
                    r.used = true;
                    out.push(Effect::CloneToHand { uid: card.uid, ethereal: true });
                }
                // PaelsLegion: the play that got doubled puts it to sleep for two turns.
                PaelsLegion if r.scratch == card.uid as i32 => {
                    r.scratch = 0;
                    r.combat_counter = 2;
                }
                _ => {}
            }
        }
        if let Some(u) = mummified {
            if let Some(c) = self.player.hand.iter_mut().find(|c| c.uid == u) {
                c.cost_this_turn = Some(0);
            }
        }
        out
    }

    /// `Hook.AfterCardExhausted` for relics.
    pub(crate) fn relic_after_card_exhausted(&mut self, card: &Card, ethereal: bool) -> Vec<Effect> {
        use RelicId::*;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                CharonsAshes => out.push(all_enemies_damage(3)),
                JossPaper => {
                    if ethereal {
                        r.combat_counter += 1;
                    } else {
                        r.counter += 1;
                        if r.counter >= 5 {
                            out.push(draw((r.counter / 5) as u32));
                            r.counter %= 5;
                        }
                    }
                }
                ForgottenSoul => out.push(random_enemy_damage(1)),
                BurningSticks if !r.used && card.ty() == CardType::Skill => {
                    r.used = true;
                    out.push(Effect::GenerateCard { id: card.id, upgraded: card.upgraded, to: Pile::Hand, free_this_turn: false });
                }
                _ => {}
            }
        }
        out
    }

    /// `Hook.ModifyHpLost` for relics on the player: Tungsten Rod, Beating Remnant.
    pub(crate) fn relic_modify_hp_lost(&self, amount: f64) -> f64 {
        use RelicId::*;
        let mut a = amount;
        for r in &self.relics {
            match r.id {
                TungstenRod => a = (a - 1.0).max(0.0),
                BeatingRemnant => a = a.min((20 - r.combat_counter).max(0) as f64),
                _ => {}
            }
        }
        a
    }

    /// `Hook.ShouldDie` + `AfterPreventingDeath`: Lizard Tail. Returns the HP
    /// to restore to, if death was prevented.
    pub(crate) fn relic_prevent_death(&mut self) -> Option<i32> {
        let max_hp = self.player.creature.max_hp;
        let r = self.relic_mut(RelicId::LizardTail)?;
        if r.flag {
            return None;
        }
        r.flag = true;
        Some(((max_hp as f64 * 0.5) as i32).max(1))
    }

    /// `Hook.AfterDamageReceived` for relics when the player is hit.
    pub(crate) fn relic_after_damage_received(&mut self, lost: i32, props: ValueProp, own_turn: bool) -> Vec<Effect> {
        use RelicId::*;
        let hp_low = self.player.creature.hp * 2 <= self.player.creature.max_hp;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                CentennialPuzzle if lost > 0 && !r.used => {
                    r.used = true;
                    out.push(draw(3));
                }
                DemonTongue if lost > 0 && own_turn && !r.used => {
                    r.used = true;
                    out.push(Effect::Heal { target: CreatureRef::Player, amount: lost as f64 });
                }
                SelfFormingClay if lost > 0 => out.push(self_power(PowerId::SelfFormingClay, 3)),
                BeatingRemnant => r.combat_counter += lost,
                RedSkull => {
                    if hp_low && !r.used {
                        r.used = true;
                        out.push(self_power(PowerId::Strength, 3));
                    } else if !hp_low && r.used {
                        r.used = false;
                        out.push(self_power(PowerId::Strength, -3));
                    }
                }
                _ => {}
            }
        }
        let _ = props;
        out
    }

    /// Red Skull after a heal: drop the Strength if back above half.
    pub(crate) fn relic_after_heal(&mut self) -> Vec<Effect> {
        self.relic_after_damage_received(0, ValueProp::NONE, false)
    }

    /// `ModifyDamageAdditive` for relics (player as dealer, card source).
    pub(crate) fn relic_damage_additive(&self, card: Option<&Card>, props: ValueProp) -> f64 {
        use RelicId::*;
        let Some(card) = card else { return 0.0 };
        if !props.is_powered() {
            return 0.0;
        }
        let mut add = 0.0;
        for r in &self.relics {
            match r.id {
                StrikeDummy if card.has_tag(Tag::Strike) => add += 3.0,
                FakeStrikeDummy if card.has_tag(Tag::Strike) => add += 1.0,
                MiniatureCannon if card.upgraded => add += 3.0,
                _ => {}
            }
        }
        add
    }

    /// `ModifyDamageMultiplicative` for relics: Pen Nib's doubled attack.
    pub(crate) fn relic_damage_multiplicative(&self, card: Option<&Card>, props: ValueProp) -> f64 {
        let Some(card) = card else { return 1.0 };
        if !props.is_powered() {
            return 1.0;
        }
        let mut m = 1.0;
        for r in &self.relics {
            if r.id == RelicId::PenNib && r.scratch == card.uid as i32 {
                m *= 2.0;
            }
        }
        m
    }

    /// `ModifyBlockMultiplicative` for relics: Vambrace doubles the first
    /// card's block this combat, Pael's Legion every card's block while it is
    /// awake. Both record the card that triggered them.
    pub(crate) fn relic_block_multiplicative(&mut self, card: Option<u32>, props: ValueProp) -> f64 {
        let Some(uid) = card else { return 1.0 };
        if !props.has(ValueProp::MOVE) {
            return 1.0;
        }
        let uid = uid as i32;
        let mut m = 1.0;
        for r in &mut self.relics {
            match r.id {
                RelicId::Vambrace if !r.used && (r.scratch == 0 || r.scratch == uid) => {
                    r.scratch = uid;
                    m *= 2.0;
                }
                RelicId::PaelsLegion if r.combat_counter <= 0 => {
                    if r.scratch == 0 {
                        r.scratch = uid;
                    }
                    m *= 2.0;
                }
                _ => {}
            }
        }
        m
    }

    /// `BeforeSideTurnEndVeryEarly` + `BeforeSideTurnEnd` for relics.
    pub(crate) fn relic_before_side_turn_end(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let hand = self.player.hand.len() as i32;
        let block = self.player.creature.block;
        let energy = self.player.energy;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                CloakClasp if hand > 0 => out.push(gain_block(hand)),
                StoneCalendar if turn == 7 => out.push(all_enemies_damage(52)),
                ScreamingFlagon if hand == 0 => out.push(all_enemies_damage(20)),
                RippleBasin if !r.used => out.push(gain_block(4)),
                Orichalcum if block <= 0 => out.push(gain_block(6)),
                FakeOrichalcum if block <= 0 => out.push(gain_block(3)),
                DiamondDiadem => {
                    if r.combat_counter <= 2 {
                        out.push(self_power(PowerId::DiamondDiadem, 1));
                    }
                    r.combat_counter = 0;
                }
                PaelsTears => r.used = energy > 0,
                _ => {}
            }
        }
        out
    }

    /// `BeforeSideTurnEndEarly` for relics: Pael's Eye burns a hand that went
    /// unplayed, on the turn it is about to hand back.
    pub(crate) fn relic_before_side_turn_end_early(&self) -> Vec<Effect> {
        if self.pael_eye_ready() {
            vec![Effect::ExhaustHand { filter: CardFilter::Any }]
        } else {
            vec![]
        }
    }

    /// `PaelsEye.ShouldTakeExtraTurn`: once a combat, a turn with no card
    /// played by hand is taken again. Whispering Earring's opening turn
    /// counts as played.
    fn pael_eye_ready(&self) -> bool {
        let played = self.stats.manual_plays_this_turn > 0
            || (self.player.turn == 1 && self.has_relic(RelicId::WhisperingEarring));
        self.relics.iter().any(|r| r.id == RelicId::PaelsEye && !r.used) && !played
    }

    /// `Hook.ShouldTakeExtraTurn` + `AfterTakingExtraTurn`: true when the
    /// player goes again, which spends Pael's Eye.
    pub(crate) fn relic_take_extra_turn(&mut self) -> bool {
        if !self.pael_eye_ready() {
            return false;
        }
        if let Some(r) = self.relic_mut(RelicId::PaelsEye) {
            r.used = true;
        }
        true
    }

    /// `AfterSideTurnEnd` for relics (player side).
    pub(crate) fn relic_after_side_turn_end(&mut self, side: Side) -> Vec<Effect> {
        use RelicId::*;
        if side != Side::Player {
            return vec![];
        }
        let block = self.player.creature.block;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                ParryingShield if block >= 10 => out.push(random_enemy_damage(6)),
                JossPaper => {
                    r.counter += r.combat_counter;
                    r.combat_counter = 0;
                    if r.counter >= 5 {
                        out.push(draw((r.counter / 5) as u32));
                        r.counter %= 5;
                    }
                }
                ArtOfWar => {
                    r.scratch = r.combat_counter;
                    r.combat_counter = 0;
                }
                Kusarigama => r.combat_counter = 0,
                _ => {}
            }
        }
        out
    }

    /// `ShouldClearBlock` for relics: Sturdy Clamp keeps up to 10.
    pub(crate) fn relic_keeps_block(&self) -> bool {
        self.has_relic(RelicId::SturdyClamp)
    }

    /// `AfterBlockCleared` for relics on the player.
    pub(crate) fn relic_after_block_cleared(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let mut out = vec![];
        for r in &self.relics {
            match r.id {
                CaptainsWheel if turn == 3 => out.push(gain_block(18)),
                HornCleat if turn == 2 => out.push(gain_block(14)),
                SparklingRouge if turn == 3 => {
                    out.push(self_power(PowerId::Strength, 1));
                    out.push(self_power(PowerId::Dexterity, 1));
                }
                _ => {}
            }
        }
        out
    }

    /// `AfterDeath` of an enemy for relics.
    pub(crate) fn relic_after_enemy_death(&self) -> Vec<Effect> {
        if self.has_relic(RelicId::GremlinHorn) {
            vec![Effect::GainEnergy { amount: 1 }, draw(1)]
        } else {
            vec![]
        }
    }

    /// `TryModifyPowerAmountReceived` / `ModifyPowerAmountGiven` for relics.
    /// Ruined Helmet doubles the first Strength gain; Unsettling Lamp doubles
    /// every debuff from the first debuffing card.
    pub(crate) fn relic_modify_power_amount(&mut self, target: CreatureRef, id: PowerId, amount: i32, card: Option<u32>) -> i32 {
        use RelicId::*;
        let mut a = amount;
        let debuff = crate::power::is_debuff(id);
        for r in &mut self.relics {
            match r.id {
                RuinedHelmet if target == CreatureRef::Player && id == PowerId::Strength && amount > 0 && !r.used => {
                    r.used = true;
                    a *= 2;
                }
                UnsettlingLamp if debuff && amount > 0 && target != CreatureRef::Player && !r.used => {
                    if let Some(uid) = card {
                        if r.scratch == 0 {
                            r.scratch = uid as i32;
                        }
                        if r.scratch == uid as i32 {
                            a *= 2;
                        }
                    }
                }
                _ => {}
            }
        }
        a
    }

    /// `AfterShuffle` for relics: The Abacus, Biiig Hug.
    pub(crate) fn relic_after_shuffle(&self) -> Vec<Effect> {
        let mut out = vec![];
        for r in &self.relics {
            match r.id {
                RelicId::TheAbacus => out.push(gain_block(6)),
                // BiiigHug.AfterShuffle: a Soot shuffled into the new draw pile.
                RelicId::BiiigHug => out.push(generate(CardId::Soot, Pile::DrawRandom)),
                _ => {}
            }
        }
        out
    }

    /// `AfterHandEmptied` for relics: Unceasing Top.
    pub(crate) fn relic_after_hand_emptied(&self) -> Vec<Effect> {
        if self.has_relic(RelicId::UnceasingTop) && self.player.hand.is_empty() && self.side == Side::Player {
            vec![draw(1)]
        } else {
            vec![]
        }
    }

    /// `AfterPotionUsed`: Belt Buckle grants its Dexterity once the last
    /// potion is gone.
    pub(crate) fn relic_after_potion_used(&mut self) -> Vec<Effect> {
        let no_potions = self.potions.iter().all(|p| p.is_none());
        match self.relic_mut(RelicId::BeltBuckle) {
            Some(r) if no_potions && !r.used => {
                r.used = true;
                vec![self_power(PowerId::Dexterity, 2)]
            }
            _ => vec![],
        }
    }

    /// `AfterPotionProcured`: Belt Buckle takes its Dexterity back while a
    /// potion is carried (`used` is its `DexterityApplied`).
    pub(crate) fn relic_after_potion_procured(&mut self) -> Vec<Effect> {
        let carrying = self.potions.iter().any(Option::is_some);
        match self.relic_mut(RelicId::BeltBuckle) {
            Some(r) if carrying && r.used => {
                r.used = false;
                vec![self_power(PowerId::Dexterity, -2)]
            }
            _ => vec![],
        }
    }

    /// `ShouldFlush`: Ringing Triangle keeps the hand on turn 1, Runic Pyramid
    /// every turn.
    pub(crate) fn relic_should_flush(&self) -> bool {
        !(self.has_relic(RelicId::RingingTriangle) && self.player.turn <= 1) && !self.has_relic(RelicId::RunicPyramid)
    }

    /// `ModifyXValue`: Chemical X.
    pub(crate) fn relic_x_bonus(&self) -> i32 {
        if self.has_relic(RelicId::ChemicalX) {
            2
        } else {
            0
        }
    }

    /// `AfterCombatVictory` for relics: heals applied to the combat HP so the
    /// caller reads the post-fight value, and the charges a fight spends.
    pub(crate) fn relic_after_victory(&mut self) {
        use RelicId::*;
        let c = &mut self.player.creature;
        if c.hp <= 0 {
            return;
        }
        for r in &mut self.relics {
            match r.id {
                BurningBlood => c.hp = (c.hp + 6).min(c.max_hp),
                BlackBlood => c.hp = (c.hp + 12).min(c.max_hp),
                MeatOnTheBone if c.hp * 2 <= c.max_hp => c.hp = (c.hp + 12).min(c.max_hp),
                // PumpkinCandle.AfterCombatEnd burns one kindling.
                PumpkinCandle => r.counter = (r.counter - 1).max(0),
                _ => {}
            }
        }
    }

    /// `AfterCardEnteredCombat` for Ghost Seed.
    pub(crate) fn relic_card_entered_combat(&self, card: &mut Card) {
        if self.has_relic(RelicId::GhostSeed) {
            ghost_seed_mark(card);
        }
    }

    /// `TryModifyEnergyCostInCombat` for relics: Spiked Gauntlets taxes powers.
    pub(crate) fn relic_cost_additive(&self, card: &Card) -> i32 {
        i32::from(card.ty() == CardType::Power && self.has_relic(RelicId::SpikedGauntlets))
    }

    /// `TryModifyEnergyCostInCombatLate` for relics: Brilliant Scarf makes the
    /// fifth card played by hand each turn free, in hand or being played.
    pub(crate) fn relic_makes_free(&self, card: &Card) -> bool {
        let p = &self.player;
        self.has_relic(RelicId::BrilliantScarf)
            && self.stats.manual_plays_this_turn == 4
            && p.hand.iter().chain(&p.play).any(|c| c.uid == card.uid)
    }

    /// `ShouldPlay` for relics: Velvet Choker stops the seventh card a turn.
    pub(crate) fn relic_allows_play(&self) -> bool {
        !self.relics.iter().any(|r| r.id == RelicId::VelvetChoker && r.combat_counter >= 6)
    }

    /// `ShouldDraw` for relics: Fiddle refuses every draw on your own turn
    /// but the hand draw.
    pub(crate) fn relic_should_draw(&self, from_hand_draw: bool) -> bool {
        from_hand_draw || self.side != Side::Player || !self.has_relic(RelicId::Fiddle)
    }

    /// `AfterCreatureAddedToCombat` for an enemy that joins mid-fight: Fur
    /// Coat bites it down to 1 HP, Philosopher's Stone hands it Strength.
    pub(crate) fn relic_after_enemy_added(&mut self, i: usize) -> Vec<Effect> {
        let mut out = vec![];
        if self.relics.iter().any(|r| r.id == RelicId::FurCoat && r.counter > 0) {
            self.enemies[i].creature.hp = 1;
        }
        if self.has_relic(RelicId::PhilosophersStone) {
            out.push(Effect::ApplyPower { target: CreatureRef::Enemy(i), id: PowerId::Strength, amount: 1, applier: None });
        }
        out
    }

    /// `DelicateFrond.BeforeCombatStart`: fill every empty slot with a random
    /// potion (`PotionFactory.CreateRandomPotionOutOfCombat`: a rarity roll,
    /// then a pick). Sozu refuses the first one, which stops the loop.
    fn delicate_frond(&mut self) {
        if !self.has_relic(RelicId::DelicateFrond) || self.has_relic(RelicId::Sozu) {
            return;
        }
        for slot in 0..self.potions.len() {
            if self.potions[slot].is_some() {
                continue;
            }
            let roll = self.rngs.potion_generation.next_float(1.0);
            let rarity = if roll <= 0.1 {
                crate::potion::Rarity::Rare
            } else if roll <= 0.35 {
                crate::potion::Rarity::Uncommon
            } else {
                crate::potion::Rarity::Common
            };
            let options: Vec<crate::potion::PotionId> =
                crate::potion::ALL.iter().copied().filter(|p| p.rarity() == rarity).collect();
            self.potions[slot] = self.rngs.potion_generation.pick(&options).copied();
        }
    }

    /// `Effect::RelicStep`: the parts of a relic hook that have to see the
    /// state as it is when they run.
    pub(crate) fn relic_step(&mut self, id: RelicId, step: u8) -> Vec<Effect> {
        use RelicId::*;
        match (id, step) {
            // ToastyMittens.BeforeHandDraw: shuffle if the draw pile is dry,
            // then burn its top card (on turn 1 the first that is not
            // Innate) and take the Strength.
            (ToastyMittens, 0) => {
                let mut out = vec![];
                if self.reshuffle_if_needed() {
                    out.extend(self.after_shuffle());
                }
                out.push(Effect::RelicStep { id, step: 1 });
                out
            }
            (ToastyMittens, _) => {
                let draw = &self.player.draw;
                let not_innate = if self.player.turn == 1 { draw.iter().find(|c| !c.has(Keyword::Innate)) } else { None };
                let mut out: Vec<Effect> =
                    not_innate.or(draw.first()).map(|c| Effect::Exhaust { uid: c.uid, ethereal: false }).into_iter().collect();
                out.push(self_power(PowerId::Strength, 1));
                out
            }
            // JeweledMask.BeforeHandDraw: a random power from the draw pile
            // to the hand, free this turn.
            (JeweledMask, _) => {
                let powers: Vec<u32> =
                    self.player.draw.iter().filter(|c| c.ty() == CardType::Power).map(|c| c.uid).collect();
                if let Some(&uid) = self.rngs.card_selection.pick(&powers) {
                    if let Some(mut c) = self.take_card(uid) {
                        c.cost_this_turn = Some(0);
                        self.put_card(c, Pile::Hand);
                    }
                }
                vec![]
            }
            // HistoryCourse: auto-play a dupe of last turn's last attack or
            // skill (`CardModel.CreateDupe`).
            (HistoryCourse, _) => {
                let Some(mut c) = self.stats.last_turn_card.clone() else { return vec![] };
                c.uid = self.new_uid();
                c.dupe = true;
                c.cost_this_turn = None;
                c.exhaust_on_next_play = false;
                let uid = c.uid;
                self.player.play.push(c);
                vec![Effect::AutoPlay { uid, force_exhaust: false }]
            }
            // WhisperingEarring: on turn 1, play the leftmost playable card at
            // the leftmost enemy, paying for it, up to 13 cards.
            (WhisperingEarring, n) => {
                if n >= 13 || self.is_over() {
                    return vec![];
                }
                let Some(i) = self.player.hand.iter().position(|c| self.can_play(c)) else { return vec![] };
                let target = match self.player.hand[i].def().target {
                    TargetType::AnyEnemy => match self.living_enemies().next() {
                        Some(e) => Some(CreatureRef::Enemy(e)),
                        None => return vec![],
                    },
                    _ => None,
                };
                let paid = self.pay_for(i);
                let card = self.player.hand.remove(i);
                let uid = card.uid;
                self.player.play.push(card);
                vec![Effect::PlayCard { uid, target, paid }, Effect::RelicStep { id, step: n + 1 }]
            }
            _ => vec![],
        }
    }
}

/// `GhostSeed.CanAffect`: basic Strikes and Defends become Ethereal.
fn ghost_seed_mark(card: &mut Card) {
    if card.def().rarity == CardRarity::Basic && (card.has_tag(Tag::Strike) || card.id == crate::ids::CardId::DefendIronclad) {
        card.ethereal_added = true;
    }
}
