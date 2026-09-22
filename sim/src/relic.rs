//! Relics. `Models/RelicModel.cs` plus `Models/Relics/*.cs` for every
//! Common, Uncommon, Rare, Shop, and Ironclad relic. Relics live on the run
//! and persist counters across combats (`Nunchaku.AttacksPlayed`, Pen Nib's
//! count, Girya's lifts, Lizard Tail's use). Combat clones them in and the
//! caller reads them back from `Combat::relics` afterwards.
//!
//! Hooks are `impl Combat` methods here so they can read combat state and
//! return effects, dispatched from the matching points in combat.rs.

use crate::card::{Card, Tag};
use crate::combat::{Combat, RoomKind};
use crate::effect::{AttackTargets, Effect, GenPool, Pile};
use crate::ids::PowerId;
use crate::types::{CardRarity, CardType, CreatureRef, Side, ValueProp};

/// `Models/Relics/<Name>.cs`. Ancient (boss) and Event relics are not yet ported.
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
    RelicId::PaelsFlesh, RelicId::LeadPaperweight,
];

/// A relic instance on the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relic {
    pub id: RelicId,
    /// Persistent counter: Nunchaku attacks, Pen Nib attacks, Tuning Fork
    /// skills, Joss Paper exhausts, Happy Flower / Pendulum turns, Girya lifts.
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
        Self { id, counter: 0, combat_counter: 0, scratch: 0, used: false, flag: false }
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
        let no_potions = self.potions.iter().all(|p| p.is_none());
        let hp_low = self.player.creature.hp * 2 <= self.player.creature.max_hp;
        let deck_upgradable: Vec<u32> = vec![];
        let _ = deck_upgradable;
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

    /// `Hook.ModifyHandDraw`.
    pub(crate) fn relic_modify_hand_draw(&self, count: u32) -> u32 {
        use RelicId::*;
        let turn = self.player.turn;
        let mut n = count;
        for r in &self.relics {
            match r.id {
                BagOfPreparation if turn <= 1 => n += 2,
                Pocketwatch if turn > 1 && r.scratch <= 3 => n += 3,
                _ => {}
            }
        }
        n
    }

    /// `Hook.ModifyMaxEnergy` for relics: Bread after turn 1, Pael's Flesh
    /// from turn 3.
    pub(crate) fn relic_modify_max_energy(&self, amount: i32) -> i32 {
        let mut a = amount;
        if self.has_relic(RelicId::Bread) && self.player.turn > 1 {
            a += 1;
        }
        if self.has_relic(RelicId::PaelsFlesh) && self.player.turn >= 3 {
            a += 1;
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
        let enemies: Vec<usize> = self.living_enemies().collect();
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
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
                }),
                _ => {}
            }
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
    /// card's block this combat. Also records the triggering card.
    pub(crate) fn relic_block_multiplicative(&mut self, card: Option<u32>, props: ValueProp) -> f64 {
        let Some(uid) = card else { return 1.0 };
        if !props.has(ValueProp::MOVE) {
            return 1.0;
        }
        let Some(r) = self.relic_mut(RelicId::Vambrace) else { return 1.0 };
        if r.used || (r.scratch != 0 && r.scratch != uid as i32) {
            return 1.0;
        }
        r.scratch = uid as i32;
        2.0
    }

    /// `BeforeSideTurnEndVeryEarly` + `BeforeSideTurnEnd` for relics.
    pub(crate) fn relic_before_side_turn_end(&mut self) -> Vec<Effect> {
        use RelicId::*;
        let turn = self.player.turn;
        let hand = self.player.hand.len() as i32;
        let block = self.player.creature.block;
        let mut out = vec![];
        for r in &mut self.relics {
            match r.id {
                CloakClasp if hand > 0 => out.push(gain_block(hand)),
                StoneCalendar if turn == 7 => out.push(all_enemies_damage(52)),
                ScreamingFlagon if hand == 0 => out.push(all_enemies_damage(20)),
                RippleBasin if !r.used => out.push(gain_block(4)),
                Orichalcum if block <= 0 => out.push(gain_block(6)),
                _ => {}
            }
        }
        out
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

    /// `AfterShuffle` for relics: The Abacus.
    pub(crate) fn relic_after_shuffle(&self) -> Vec<Effect> {
        if self.has_relic(RelicId::TheAbacus) {
            vec![gain_block(6)]
        } else {
            vec![]
        }
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

    /// `ShouldFlush`: Ringing Triangle keeps the hand on turn 1.
    pub(crate) fn relic_should_flush(&self) -> bool {
        !(self.has_relic(RelicId::RingingTriangle) && self.player.turn <= 1)
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
    /// caller reads the post-fight value.
    pub(crate) fn relic_after_victory(&mut self) {
        use RelicId::*;
        let c = &mut self.player.creature;
        if c.hp <= 0 {
            return;
        }
        for r in &self.relics {
            match r.id {
                BurningBlood => c.hp = (c.hp + 6).min(c.max_hp),
                MeatOnTheBone if c.hp * 2 <= c.max_hp => c.hp = (c.hp + 12).min(c.max_hp),
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
}

/// `GhostSeed.CanAffect`: basic Strikes and Defends become Ethereal.
fn ghost_seed_mark(card: &mut Card) {
    if card.def().rarity == CardRarity::Basic && (card.has_tag(Tag::Strike) || card.id == crate::ids::CardId::DefendIronclad) {
        card.ethereal_added = true;
    }
}
