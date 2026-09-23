//! The pools rewards, shops and events draw cards and potions from, in the
//! game's order, which is what a draw's index points into:
//! `CardPools/IroncladCardPool.cs` and `CardPools/ColorlessCardPool.cs`
//! (`GenerateAllCards`), `PotionPools/IroncladPotionPool.cs` and
//! `PotionPools/SharedPotionPool.cs` (`GenerateAllPotions`), each entry
//! with its class's rarity and type. A fully unlocked profile has all of
//! them, and every card is upgradable. `tools/oracle pools` prints them
//! (`testdata/oracle-pools.txt`); a test holds these tables to it.
//!
//! Entries are game ids, so a card or potion the combat sim lacks can still
//! be offered; `sim_card`, `sim_potion` and `sim_relic` map to the sim's
//! where it has one.

use std::fmt::Write;

use crate::ids::{CardId, ALL_CARDS};
use crate::potion::{PotionId, ALL as ALL_POTIONS};
use crate::relic::{RelicId, ALL as ALL_RELICS};
use crate::replay::slug;
use crate::types::CardType::{self, Attack, Power, Skill};

/// `Entities/Cards/CardRarity.cs`, the members the pools hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rarity {
    Basic,
    Common,
    Uncommon,
    Rare,
    Ancient,
}
use Rarity::{Ancient, Basic, Common, Rare, Uncommon};

/// `Entities/Potions/PotionRarity.cs`, the members the pools hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionRarity {
    Common,
    Uncommon,
    Rare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolCard {
    pub id: &'static str,
    pub rarity: Rarity,
    pub kind: CardType,
    /// `CardMultiplayerConstraint.MultiplayerOnly`: never offered alone.
    pub multiplayer_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPotion {
    pub id: &'static str,
    pub rarity: PotionRarity,
}

const fn card(id: &'static str, rarity: Rarity, kind: CardType, multiplayer_only: bool) -> PoolCard {
    PoolCard { id, rarity, kind, multiplayer_only }
}

const fn potion(id: &'static str, rarity: PotionRarity) -> PoolPotion {
    PoolPotion { id, rarity }
}

pub const IRONCLAD_CARDS: &[PoolCard] = &[
    card("AGGRESSION", Rare, Power, false),
    card("ANGER", Common, Attack, false),
    card("ARMAMENTS", Common, Skill, false),
    card("ASHEN_STRIKE", Uncommon, Attack, false),
    card("BARRICADE", Rare, Power, false),
    card("BASH", Basic, Attack, false),
    card("BATTLE_TRANCE", Uncommon, Skill, false),
    card("BLOOD_WALL", Common, Skill, false),
    card("BLOODLETTING", Common, Skill, false),
    card("BLUDGEON", Uncommon, Attack, false),
    card("BODY_SLAM", Common, Attack, false),
    card("BRAND", Rare, Skill, false),
    card("BREAK", Ancient, Attack, false),
    card("BREAKTHROUGH", Common, Attack, false),
    card("BULLY", Uncommon, Attack, false),
    card("BURNING_PACT", Uncommon, Skill, false),
    card("CASCADE", Rare, Skill, false),
    card("CINDER", Common, Attack, false),
    card("COLOSSUS", Uncommon, Skill, false),
    card("CONFLAGRATION", Rare, Attack, false),
    card("CORRUPTION", Ancient, Power, false),
    card("CRIMSON_MANTLE", Rare, Power, false),
    card("CRUELTY", Rare, Power, false),
    card("DARK_EMBRACE", Rare, Power, false),
    card("DEFEND_IRONCLAD", Basic, Skill, false),
    card("DEMON_FORM", Rare, Power, false),
    card("DEMONIC_SHIELD", Uncommon, Skill, true),
    card("DISMANTLE", Uncommon, Attack, false),
    card("DOMINATE", Uncommon, Skill, false),
    card("DRUM_OF_BATTLE", Uncommon, Skill, false),
    card("EVIL_EYE", Uncommon, Skill, false),
    card("EXPECT_A_FIGHT", Uncommon, Skill, false),
    card("FEED", Rare, Attack, false),
    card("FEEL_NO_PAIN", Uncommon, Power, false),
    card("FIEND_FIRE", Rare, Attack, false),
    card("FIGHT_ME", Uncommon, Attack, false),
    card("FLAME_BARRIER", Uncommon, Skill, false),
    card("FORGOTTEN_RITUAL", Uncommon, Skill, false),
    card("HAVOC", Common, Skill, false),
    card("HEADBUTT", Common, Attack, false),
    card("HELLRAISER", Rare, Power, false),
    card("HEMOKINESIS", Uncommon, Attack, false),
    card("HOWL_FROM_BEYOND", Uncommon, Attack, false),
    card("IMPERVIOUS", Rare, Skill, false),
    card("INFERNAL_BLADE", Uncommon, Skill, false),
    card("INFERNO", Uncommon, Power, false),
    card("INFLAME", Uncommon, Power, false),
    card("IRON_WAVE", Common, Attack, false),
    card("JUGGERNAUT", Rare, Power, false),
    card("JUGGLING", Uncommon, Power, false),
    card("MANGLE", Rare, Attack, false),
    card("MOLTEN_FIST", Common, Attack, false),
    card("NOT_YET", Rare, Skill, false),
    card("OFFERING", Rare, Skill, false),
    card("ONE_TWO_PUNCH", Rare, Skill, false),
    card("PACTS_END", Rare, Attack, false),
    card("PERFECTED_STRIKE", Common, Attack, false),
    card("PILLAGE", Uncommon, Attack, false),
    card("POMMEL_STRIKE", Common, Attack, false),
    card("PRIMAL_FORCE", Rare, Skill, false),
    card("PYRE", Rare, Power, false),
    card("RAGE", Uncommon, Skill, false),
    card("RAMPAGE", Uncommon, Attack, false),
    card("RUPTURE", Uncommon, Power, false),
    card("SECOND_WIND", Uncommon, Skill, false),
    card("SETUP_STRIKE", Common, Attack, false),
    card("SHRUG_IT_OFF", Common, Skill, false),
    card("SPITE", Uncommon, Attack, false),
    card("STAMPEDE", Uncommon, Power, false),
    card("STOKE", Rare, Skill, false),
    card("STOMP", Uncommon, Attack, false),
    card("STONE_ARMOR", Uncommon, Power, false),
    card("STRIKE_IRONCLAD", Basic, Attack, false),
    card("SWORD_BOOMERANG", Common, Attack, false),
    card("TANK", Rare, Power, true),
    card("TAUNT", Uncommon, Skill, false),
    card("TEAR_ASUNDER", Rare, Attack, false),
    card("THRASH", Rare, Attack, false),
    card("THUNDERCLAP", Common, Attack, false),
    card("TREMBLE", Common, Skill, false),
    card("TRUE_GRIT", Common, Skill, false),
    card("TWIN_STRIKE", Common, Attack, false),
    card("UNMOVABLE", Rare, Power, false),
    card("UNRELENTING", Uncommon, Attack, false),
    card("UPPERCUT", Uncommon, Attack, false),
    card("VICIOUS", Uncommon, Power, false),
    card("WHIRLWIND", Uncommon, Attack, false),
];

pub const COLORLESS_CARDS: &[PoolCard] = &[
    card("ALCHEMIZE", Rare, Skill, false),
    card("ANOINTED", Rare, Skill, false),
    card("AUTOMATION", Uncommon, Power, false),
    card("BEACON_OF_HOPE", Rare, Power, true),
    card("BEAT_DOWN", Rare, Skill, false),
    card("BELIEVE_IN_YOU", Uncommon, Skill, true),
    card("BOLAS", Rare, Attack, false),
    card("CALAMITY", Rare, Power, false),
    card("CATASTROPHE", Uncommon, Skill, false),
    card("COORDINATE", Uncommon, Skill, true),
    card("DARK_SHACKLES", Uncommon, Skill, false),
    card("DISCOVERY", Uncommon, Skill, false),
    card("DRAMATIC_ENTRANCE", Uncommon, Attack, false),
    card("ENTROPY", Rare, Power, false),
    card("EQUILIBRIUM", Uncommon, Skill, false),
    card("ETERNAL_ARMOR", Rare, Power, false),
    card("FASTEN", Uncommon, Power, false),
    card("FINESSE", Uncommon, Skill, false),
    card("FISTICUFFS", Uncommon, Attack, false),
    card("FLASH_OF_STEEL", Uncommon, Attack, false),
    card("GANG_UP", Uncommon, Attack, true),
    card("GOLD_AXE", Rare, Attack, false),
    card("HAND_OF_GREED", Rare, Attack, false),
    card("HIDDEN_GEM", Rare, Skill, false),
    card("HUDDLE_UP", Uncommon, Skill, true),
    card("IMPATIENCE", Uncommon, Skill, false),
    card("INTERCEPT", Uncommon, Skill, true),
    card("JACK_OF_ALL_TRADES", Uncommon, Skill, false),
    card("JACKPOT", Rare, Attack, false),
    card("KNOCKDOWN", Rare, Attack, true),
    card("LIFT", Uncommon, Skill, true),
    card("MASTER_OF_STRATEGY", Rare, Skill, false),
    card("MAYHEM", Rare, Power, false),
    card("MIMIC", Rare, Skill, true),
    card("MIND_BLAST", Uncommon, Attack, false),
    card("NOSTALGIA", Rare, Power, false),
    card("OMNISLICE", Uncommon, Attack, false),
    card("PANACHE", Uncommon, Power, false),
    card("PANIC_BUTTON", Uncommon, Skill, false),
    card("PREP_TIME", Uncommon, Power, false),
    card("PRODUCTION", Uncommon, Skill, false),
    card("PROLONG", Uncommon, Skill, false),
    card("PROWESS", Uncommon, Power, false),
    card("PURITY", Uncommon, Skill, false),
    card("RALLY", Rare, Skill, true),
    card("REND", Rare, Attack, false),
    card("RESTLESSNESS", Uncommon, Skill, false),
    card("ROLLING_BOULDER", Rare, Power, false),
    card("SALVO", Rare, Attack, false),
    card("SCRAWL", Rare, Skill, false),
    card("SECRET_TECHNIQUE", Rare, Skill, false),
    card("SECRET_WEAPON", Rare, Skill, false),
    card("SEEKER_STRIKE", Uncommon, Attack, false),
    card("SHOCKWAVE", Uncommon, Skill, false),
    card("SPLASH", Uncommon, Skill, false),
    card("STRATAGEM", Uncommon, Power, false),
    card("TAG_TEAM", Uncommon, Attack, true),
    card("THE_BOMB", Uncommon, Skill, false),
    card("THE_GAMBIT", Rare, Skill, false),
    card("THINKING_AHEAD", Uncommon, Skill, false),
    card("THRUMMING_HATCHET", Uncommon, Attack, false),
    card("ULTIMATE_DEFEND", Uncommon, Skill, false),
    card("ULTIMATE_STRIKE", Uncommon, Attack, false),
    card("VOLLEY", Uncommon, Attack, false),
];

/// The character's pool comes first where a draw takes both
/// (`PotionFactory.GetPotionOptions`).
pub const IRONCLAD_POTIONS: &[PoolPotion] = &[
    potion("BLOOD_POTION", PotionRarity::Common),
    potion("SOLDIERS_STEW", PotionRarity::Rare),
    potion("ASHWATER", PotionRarity::Uncommon),
];

pub const SHARED_POTIONS: &[PoolPotion] = &[
    potion("ATTACK_POTION", PotionRarity::Common),
    potion("BEETLE_JUICE", PotionRarity::Rare),
    potion("BLESSING_OF_THE_FORGE", PotionRarity::Uncommon),
    potion("BLOCK_POTION", PotionRarity::Common),
    potion("BOTTLED_POTENTIAL", PotionRarity::Rare),
    potion("CLARITY", PotionRarity::Uncommon),
    potion("COLORLESS_POTION", PotionRarity::Common),
    potion("CURE_ALL", PotionRarity::Uncommon),
    potion("DEXTERITY_POTION", PotionRarity::Common),
    potion("DISTILLED_CHAOS", PotionRarity::Rare),
    potion("DROPLET_OF_PRECOGNITION", PotionRarity::Rare),
    potion("DUPLICATOR", PotionRarity::Uncommon),
    potion("ENERGY_POTION", PotionRarity::Common),
    potion("ENTROPIC_BREW", PotionRarity::Rare),
    potion("EXPLOSIVE_AMPOULE", PotionRarity::Common),
    potion("FAIRY_IN_A_BOTTLE", PotionRarity::Rare),
    potion("FIRE_POTION", PotionRarity::Common),
    potion("FLEX_POTION", PotionRarity::Common),
    potion("FORTIFIER", PotionRarity::Uncommon),
    potion("FRUIT_JUICE", PotionRarity::Rare),
    potion("FYSH_OIL", PotionRarity::Uncommon),
    potion("GAMBLERS_BREW", PotionRarity::Uncommon),
    potion("GIGANTIFICATION_POTION", PotionRarity::Rare),
    potion("HEART_OF_IRON", PotionRarity::Uncommon),
    potion("LIQUID_BRONZE", PotionRarity::Uncommon),
    potion("LIQUID_MEMORIES", PotionRarity::Rare),
    potion("LUCKY_TONIC", PotionRarity::Rare),
    potion("MAZALETHS_GIFT", PotionRarity::Rare),
    potion("OROBIC_ACID", PotionRarity::Rare),
    potion("POTION_OF_BINDING", PotionRarity::Uncommon),
    potion("POWDERED_DEMISE", PotionRarity::Uncommon),
    potion("POWER_POTION", PotionRarity::Common),
    potion("RADIANT_TINCTURE", PotionRarity::Uncommon),
    potion("REGEN_POTION", PotionRarity::Uncommon),
    potion("SHACKLING_POTION", PotionRarity::Rare),
    potion("SHIP_IN_A_BOTTLE", PotionRarity::Rare),
    potion("SKILL_POTION", PotionRarity::Common),
    potion("SNECKO_OIL", PotionRarity::Rare),
    potion("SPEED_POTION", PotionRarity::Common),
    potion("STABLE_SERUM", PotionRarity::Uncommon),
    potion("STRENGTH_POTION", PotionRarity::Common),
    potion("SWIFT_POTION", PotionRarity::Common),
    potion("TOUCH_OF_INSANITY", PotionRarity::Uncommon),
    potion("VULNERABLE_POTION", PotionRarity::Common),
    potion("WEAK_POTION", PotionRarity::Common),
];

/// The sim's card for a game id, if it has one.
pub fn sim_card(id: &str) -> Option<CardId> {
    ALL_CARDS.iter().copied().find(|c| slug(&format!("{c:?}")) == id)
}

/// The sim's potion for a game id, if it has one.
pub fn sim_potion(id: &str) -> Option<PotionId> {
    ALL_POTIONS.iter().copied().find(|p| slug(&format!("{p:?}")) == id)
}

/// The sim's relic for a game id, if it has one.
pub fn sim_relic(id: &str) -> Option<RelicId> {
    ALL_RELICS.iter().copied().find(|r| slug(&format!("{r:?}")) == id)
}

/// The tables as `tools/oracle pools` prints them.
pub fn oracle_text() -> String {
    let mut out = String::new();
    for (pool, cards) in [("IRONCLAD_CARD_POOL", IRONCLAD_CARDS), ("COLORLESS_CARD_POOL", COLORLESS_CARDS)] {
        for c in cards {
            let constraint = if c.multiplayer_only { "MultiplayerOnly" } else { "None" };
            let _ = writeln!(out, "card {pool} {} {:?} {:?} {constraint} 1", c.id, c.rarity, c.kind);
        }
    }
    for (pool, potions) in [("IRONCLAD_POTION_POOL", IRONCLAD_POTIONS), ("SHARED_POTION_POOL", SHARED_POTIONS)] {
        for p in potions {
            let _ = writeln!(out, "potion {pool} {} {:?}", p.id, p.rarity);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_game() {
        assert_eq!(oracle_text(), include_str!("../testdata/oracle-pools.txt"));
        assert_eq!(sim_card("BODY_SLAM"), Some(CardId::BodySlam));
    }
}
