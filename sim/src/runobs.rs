//! The run policy's observation (docs/run-env.md, Observation and scoring):
//! a run decision as tokens. One global token (HP, gold, floor, act, the
//! act's bosses, the decision, the odds a player can count), one per deck
//! card, relic and potion, and one per option, each option carrying what it
//! names (a card, relic, potion, map point, event option) and, for a map
//! step, what the paths through the point hold. A map step also carries
//! the part of the act's map still ahead, point by point with its links
//! (`map_ahead`). Every segment has a fixed length and a presence flag per
//! token, so a batch is one array.
//!
//! Only what the player sees or could count goes in (docs/run-env.md,
//! Hidden information): nothing here reads the seed, the streams or the
//! run plan beyond the act's bosses, which the map shows. A test holds two
//! runs that differ only in those to the same encoding.
//!
//! Ids share the combat model's and `sts2ai.deckvalue`'s indexing: a card,
//! potion or enchantment is its sim id + 1, a relic too, and the relics
//! the combat sim leaves out follow the sim's (`RUN_RELICS`). 0 is none,
//! or an id the sim does not know on a present token.

use crate::effects::{DeckAction, RestOption};
use crate::encode::N_RELICS;
use crate::encounter::{Act, Encounter, Kind};
use crate::events::Shown;
use crate::pools::Rarity;
use crate::types::CardType;
use crate::gen::FightSetup;
use crate::rng::Rng;
use crate::map::{ActMap, PointId, PointType};
use crate::pools::{sim_card, sim_enchantment, sim_potion, sim_relic};
use crate::rewards::Offer;
use crate::rooms::Decision;
use crate::run::{Room, RunState};
use crate::shop::{Item, Slot};

pub const MAX_DECK: usize = 64;
pub const MAX_RELICS: usize = 40;
pub const MAX_POTIONS: usize = 5;
/// A deck pick lists the whole deck, and a skip.
pub const MAX_OPTIONS: usize = MAX_DECK + 8;
/// Cards an option can name: a bundle's, or an event's.
pub const OPTION_CARDS: usize = 3;

/// Ids per token of each segment, then floats. Each segment's first float
/// is its presence flag, except the global token's, which is always there.
pub const GLOBAL_IDS: usize = 6;
pub const GLOBAL_FLOATS: usize = F_FORECAST + FORECAST_FLOATS;
/// The forecast's slots in the global token, after the run's own floats:
/// the act's elites' win chance and HP kept, then its boss's (`forecast`).
/// The sim writes zeros; the caller fills them from the combat model
/// (`sts2ai.forecast`), at map steps only.
pub const F_FORECAST: usize = 16;
pub const FORECAST_FLOATS: usize = 4;
pub const DECK_IDS: usize = 2;
pub const DECK_FLOATS: usize = 3;
pub const RELIC_IDS: usize = 1;
pub const RELIC_FLOATS: usize = 3;
pub const POTION_IDS: usize = 1;
pub const POTION_FLOATS: usize = 1;
/// kind, cards (`OPTION_CARDS`), enchantment, relic, potion, room, event
/// option (`EVENT_OPTIONS`), and a map point's column + 1 (its node in the
/// first row of `map_ahead`).
pub const OPTION_IDS: usize = 5 + OPTION_CARDS + 2;
/// present, upgraded per card, enchantment amount, price, the map summary,
/// what an event option's text says it does (`shown_feats`).
pub const OPTION_FLOATS: usize = 3 + OPTION_CARDS + MAP_FEATS + EVENT_FEATS;
/// An event option's `events::Shown`: HP, max HP and gold; the cards it
/// adds (how many, among how many each, attack / skill / power, common /
/// uncommon / rare, colorless, upgraded); deck cards removed, transformed,
/// upgraded, downgraded, enchanted and duplicated, and whether the game
/// picks them; curses, relics and potions gained; a fight.
pub const EVENT_FEATS: usize = 24;
/// Per map point type a path can pass (`PATH_POINTS`), the fewest and the
/// most on paths through a point; then the rows to the nearest rest site
/// and shop; then the fewest and most elites before the next rest site.
pub const MAP_FEATS: usize = 2 * PATH_POINTS.len() + 4;

pub const F_DECK: usize = GLOBAL_FLOATS;
pub const F_RELICS: usize = F_DECK + MAX_DECK * DECK_FLOATS;
pub const F_POTIONS: usize = F_RELICS + MAX_RELICS * RELIC_FLOATS;
pub const F_OPTIONS: usize = F_POTIONS + MAX_POTIONS * POTION_FLOATS;
pub const RUN_FLOATS: usize = F_OPTIONS + MAX_OPTIONS * OPTION_FLOATS;
pub const I_DECK: usize = GLOBAL_IDS;
pub const I_RELICS: usize = I_DECK + MAX_DECK * DECK_IDS;
pub const I_POTIONS: usize = I_RELICS + MAX_RELICS * RELIC_IDS;
pub const I_OPTIONS: usize = I_POTIONS + MAX_POTIONS * POTION_IDS;
pub const I_MAP: usize = I_OPTIONS + MAX_OPTIONS * OPTION_IDS;
pub const RUN_IDS: usize = I_MAP + MAP_ROWS * MAP_COLS * MAP_NODE_IDS;

/// The map ahead at a map step (`map_ahead`), ids only: a grid of rows
/// from the options' row up, the most rooms an act has (`map::rooms`), by
/// the map's columns. Each node is its point's type (a `ROOMS` id, 0 for
/// no point) and its links, a bit per point in the row above it leads to:
/// bit 0 one column left, bit 1 the same column, bit 2 one right
/// (`ActMap::generate` links no further).
pub const MAP_ROWS: usize = 15;
pub const MAP_COLS: usize = crate::map::COLS;
pub const MAP_NODE_IDS: usize = 2;

/// `rooms::Decision`'s kinds, as the global token names them.
pub const DECISIONS: [&str; 10] = ["Path", "Relic", "Card", "Bundle", "Potion", "Rest", "Ancient", "Deck", "Shop", "Event"];

/// The map point types a path option's summary counts.
pub const PATH_POINTS: [PointType; 6] =
    [PointType::Unknown, PointType::Shop, PointType::Treasure, PointType::RestSite, PointType::Monster, PointType::Elite];

/// Rooms: a map point's type for a path option, the room the player
/// stands in for the global token. Append-only.
pub const ROOMS: [&str; 9] = ["Unknown", "Shop", "Treasure", "RestSite", "Monster", "Elite", "Boss", "Ancient", "Event"];

/// The acts, append-only.
pub const ACTS: [Act; 4] = [Act::Overgrowth, Act::Underdocks, Act::Hive, Act::Glory];

/// What an option does. Append-only: its index + 1 is the option token's
/// kind id, and `sim/vocab.txt` pins the order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Path,
    Relic,
    LeaveRelics,
    Card,
    SkipCard,
    Bundle,
    SkipBundle,
    KeepPotion,
    LeavePotion,
    RestHeal,
    RestSmith,
    RestLift,
    RestDig,
    RestKindle,
    RestCook,
    RestClone,
    Ancient,
    DeckUpgrade,
    DeckRemove,
    DeckDuplicate,
    DeckEnchant,
    DeckTransform,
    DeckTransformUpgraded,
    DeckMaul,
    DeckTransformInto,
    StopPicking,
    ShopCard,
    ShopColorless,
    ShopRelic,
    ShopPotion,
    ShopRemoval,
    LeaveShop,
    Event,
}

pub const OPTION_KINDS: [OptionKind; 33] = {
    use OptionKind::*;
    [
        Path, Relic, LeaveRelics, Card, SkipCard, Bundle, SkipBundle, KeepPotion, LeavePotion, RestHeal, RestSmith, RestLift,
        RestDig, RestKindle, RestCook, RestClone, Ancient, DeckUpgrade, DeckRemove, DeckDuplicate, DeckEnchant, DeckTransform,
        DeckTransformUpgraded, DeckMaul, DeckTransformInto, StopPicking, ShopCard, ShopColorless, ShopRelic, ShopPotion,
        ShopRemoval, LeaveShop, Event,
    ]
};

/// Relics a run can hold that the combat sim leaves out
/// (`gen::INERT_RELICS`), by game id. Append-only: relic id
/// `N_RELICS + 1 + i` is the `i`-th. A test holds `INERT_RELICS` to it.
pub const RUN_RELICS: &[&str] = &[
    "ALCHEMICAL_COFFER", "ARCANE_SCROLL", "ARCHAIC_TOOTH", "ASTROLABE", "BEAUTIFUL_BRACELET", "BING_BONG",
    "BLACK_STAR", "BYRDPIP", "CALLING_BELL", "CHOSEN_CHEESE", "CIRCLET", "CLAWS", "CURSED_PEARL", "DARKSTONE_PERIAPT",
    "DISTINGUISHED_CAPE", "DREAM_CATCHER", "DRIFTWOOD", "DUSTY_TOME", "ELECTRIC_SHRYMP", "EMPTY_CAGE",
    "FAKE_LEES_WAFFLE", "FAKE_MANGO", "FAKE_MERCHANTS_RUG", "FISHING_ROD", "FRAGRANT_MUSHROOM", "FRESNEL_LENS",
    "GLASS_EYE", "GLITTER", "GOLDEN_COMPASS", "GOLDEN_PEARL", "HEFTY_TABLET", "JEWELRY_BOX", "KALEIDOSCOPE",
    "LARGE_CAPSULE", "LAVA_ROCK", "LEAFY_POULTICE", "LOOMING_FRUIT", "LORDS_PARASOL", "LOST_COFFER",
    "MASSIVE_SCROLL", "MAW_BANK", "MEAT_CLEAVER", "NEOWS_BONES", "NEOWS_TALISMAN", "NEOWS_TORMENT", "NEW_LEAF",
    "NUTRITIOUS_OYSTER", "NUTRITIOUS_SOUP", "PAELS_CLAW", "PAELS_GROWTH", "PAELS_HORN", "PAELS_TOOTH",
    "PAELS_WING", "PANDORAS_BOX", "PAPER_KRANE", "PHIAL_HOLSTER", "POMANDER", "PRECARIOUS_SHEARS",
    "PRECISE_SCISSORS", "PRESERVED_FOG", "SAND_CASTLE", "SCROLL_BOXES", "SEA_GLASS", "SERE_TALON",
    "SIGNET_RING", "SILKEN_TRESS", "SILVER_CRUCIBLE", "SMALL_CAPSULE", "STONE_HUMIDIFIER", "STORYBOOK",
    "SWORD_OF_STONE", "TANXS_WHISTLE", "TOUCH_OF_OROBAS", "TOY_BOX", "TRI_BOOMERANG", "VAKUU_CARD_SELECTOR",
    "WAR_HAMMER", "WINGED_BOOTS", "WONGO_CUSTOMER_APPRECIATION_BADGE", "WONGOS_MYSTERY_TICKET", "YUMMY_COOKIE",
];

/// Events and ancients by class name, append-only. A test holds the ported
/// events (`events::ported`) and `ancients::ANCIENTS` to it.
pub const RUN_EVENTS: &[&str] = &[
    "AbyssalBaths", "Amalgamator", "AromaOfChaos", "BattlewornDummy", "BrainLeech", "Bugslayer", "ByrdonisNest",
    "ColossalFlower", "DenseVegetation", "DollRoom", "DoorsOfLightAndDark", "DrowningBeacon", "EndlessConveyor",
    "FakeMerchant", "FieldOfManSizedHoles", "GraveOfTheForgotten", "HungryForMushrooms", "InfestedAutomaton",
    "JungleMazeAdventure", "LostWisp", "LuminousChoir", "MorphicGrove", "PotionCourier", "PunchOff", "RanwidTheElder",
    "Reflections", "RelicTrader", "RoomFullOfCheese", "RoundTeaParty", "SapphireSeed", "SelfHelpBook", "SlipperyBridge",
    "SpiralingWhirlpool", "SpiritGrafter", "StoneOfAllTime", "SunkenStatue", "SunkenTreasury", "Symbiote",
    "TabletOfTruth", "TeaMaster", "TheFutureOfPotions", "TheLanternKey", "TheLegendsWereTrue", "ThisOrThat",
    "TrashHeap", "Trial", "UnrestSite", "WaterloggedScriptorium", "WelcomeToWongos", "Wellspring",
    "WhisperingHollow", "WoodCarvings", "ZenWeaver", "Neow", "Orobas", "Pael", "Tezcatara", "Vakuu", "Darv",
    "Nonupeipe", "Tanx",
];

/// Every option a ported event can ask, as (event, page, key) the way
/// `events::EventOption` keeps them, append-only: its index + 1 is the
/// option token's event option id, and `sim/vocab.txt` pins the order. A
/// test walks every event and holds each option it asks to this list.
pub const EVENT_OPTIONS: &[(&str, &str, &str)] = &[
    ("AbyssalBaths", "ALL", "EXIT_BATHS"),
    ("AbyssalBaths", "ALL", "LINGER"),
    ("AbyssalBaths", "INITIAL", "ABSTAIN"),
    ("AbyssalBaths", "INITIAL", "IMMERSE"),
    ("Amalgamator", "INITIAL", "COMBINE_DEFENDS"),
    ("Amalgamator", "INITIAL", "COMBINE_STRIKES"),
    ("AromaOfChaos", "INITIAL", "LET_GO"),
    ("AromaOfChaos", "INITIAL", "MAINTAIN_CONTROL"),
    ("BattlewornDummy", "INITIAL", "SETTING_1"),
    ("BattlewornDummy", "INITIAL", "SETTING_2"),
    ("BattlewornDummy", "INITIAL", "SETTING_3"),
    ("BrainLeech", "INITIAL", "RIP"),
    ("BrainLeech", "INITIAL", "SHARE_KNOWLEDGE"),
    ("Bugslayer", "INITIAL", "EXTERMINATION"),
    ("Bugslayer", "INITIAL", "SQUASH"),
    ("ByrdonisNest", "INITIAL", "EAT"),
    ("ByrdonisNest", "INITIAL", "TAKE"),
    ("ColossalFlower", "INITIAL", "EXTRACT_CURRENT_PRIZE_1"),
    ("ColossalFlower", "INITIAL", "REACH_DEEPER_1"),
    ("ColossalFlower", "REACH_DEEPER_1", "EXTRACT_CURRENT_PRIZE_2"),
    ("ColossalFlower", "REACH_DEEPER_1", "REACH_DEEPER_2"),
    ("ColossalFlower", "REACH_DEEPER_2", "EXTRACT_INSTEAD"),
    ("ColossalFlower", "REACH_DEEPER_2", "POLLINOUS_CORE"),
    ("DenseVegetation", "INITIAL", "REST"),
    ("DenseVegetation", "INITIAL", "TRUDGE_ON"),
    ("DenseVegetation", "REST", "FIGHT"),
    ("DollRoom", "", "BING_BONG"),
    ("DollRoom", "", "DAUGHTER_OF_THE_WIND"),
    ("DollRoom", "", "MR_STRUGGLES"),
    ("DollRoom", "INITIAL", "EXAMINE"),
    ("DollRoom", "INITIAL", "RANDOM"),
    ("DollRoom", "INITIAL", "TAKE_SOME_TIME"),
    ("DoorsOfLightAndDark", "INITIAL", "DARK"),
    ("DoorsOfLightAndDark", "INITIAL", "LIGHT"),
    ("DrowningBeacon", "INITIAL", "BOTTLE"),
    ("DrowningBeacon", "INITIAL", "CLIMB"),
    ("EndlessConveyor", "ALL", "CAVIAR"),
    ("EndlessConveyor", "ALL", "CLAM_ROLL"),
    ("EndlessConveyor", "ALL", "FRIED_EEL"),
    ("EndlessConveyor", "ALL", "GOLDEN_FYSH"),
    ("EndlessConveyor", "ALL", "JELLY_LIVER"),
    ("EndlessConveyor", "ALL", "SEAPUNK_SALAD"),
    ("EndlessConveyor", "ALL", "SPICY_SNAPPY"),
    ("EndlessConveyor", "ALL", "SUSPICIOUS_CONDIMENT"),
    ("EndlessConveyor", "GRAB_SOMETHING_OFF_THE_BELT", "LEAVE"),
    ("EndlessConveyor", "INITIAL", "OBSERVE_CHEF"),
    ("FakeMerchant", "", "SHOP"),
    ("FakeMerchant", "", "THROW"),
    ("FieldOfManSizedHoles", "INITIAL", "ENTER_YOUR_HOLE"),
    ("FieldOfManSizedHoles", "INITIAL", "RESIST"),
    ("GraveOfTheForgotten", "INITIAL", "ACCEPT"),
    ("GraveOfTheForgotten", "INITIAL", "CONFRONT"),
    ("HungryForMushrooms", "INITIAL", "BIG_MUSHROOM"),
    ("HungryForMushrooms", "INITIAL", "FRAGRANT_MUSHROOM"),
    ("InfestedAutomaton", "INITIAL", "STUDY"),
    ("InfestedAutomaton", "INITIAL", "TOUCH_CORE"),
    ("JungleMazeAdventure", "INITIAL", "JOIN_FORCES"),
    ("JungleMazeAdventure", "INITIAL", "SOLO_QUEST"),
    ("LostWisp", "INITIAL", "CLAIM"),
    ("LostWisp", "INITIAL", "SEARCH"),
    ("LuminousChoir", "INITIAL", "OFFER_TRIBUTE"),
    ("LuminousChoir", "INITIAL", "REACH_INTO_THE_FLESH"),
    ("MorphicGrove", "INITIAL", "GROUP"),
    ("MorphicGrove", "INITIAL", "LONER"),
    ("PotionCourier", "INITIAL", "GRAB_POTIONS"),
    ("PotionCourier", "INITIAL", "RANSACK"),
    ("PunchOff", "INITIAL", "I_CAN_TAKE_THEM"),
    ("PunchOff", "INITIAL", "NAB"),
    ("PunchOff", "I_CAN_TAKE_THEM", "FIGHT"),
    ("RanwidTheElder", "INITIAL", "GOLD"),
    ("RanwidTheElder", "INITIAL", "POTION"),
    ("RanwidTheElder", "INITIAL", "RELIC"),
    ("Reflections", "INITIAL", "SHATTER"),
    ("Reflections", "INITIAL", "TOUCH_A_MIRROR"),
    ("RelicTrader", "", "PROCEED"),
    ("RelicTrader", "INITIAL", "BOTTOM"),
    ("RelicTrader", "INITIAL", "MIDDLE"),
    ("RelicTrader", "INITIAL", "TOP"),
    ("RoomFullOfCheese", "INITIAL", "GORGE"),
    ("RoomFullOfCheese", "INITIAL", "SEARCH"),
    ("RoundTeaParty", "INITIAL", "ENJOY_TEA"),
    ("RoundTeaParty", "INITIAL", "PICK_FIGHT"),
    ("RoundTeaParty", "PICK_FIGHT", "CONTINUE_FIGHT"),
    ("SapphireSeed", "INITIAL", "EAT"),
    ("SapphireSeed", "INITIAL", "PLANT"),
    ("SelfHelpBook", "INITIAL", "NO_OPTIONS"),
    ("SelfHelpBook", "INITIAL", "READ_ENTIRE_BOOK"),
    ("SelfHelpBook", "INITIAL", "READ_PASSAGE"),
    ("SelfHelpBook", "INITIAL", "READ_THE_BACK"),
    ("SlipperyBridge", "HOLD_ON_0", "HOLD_ON_1"),
    ("SlipperyBridge", "HOLD_ON_1", "HOLD_ON_2"),
    ("SlipperyBridge", "HOLD_ON_2", "HOLD_ON_3"),
    ("SlipperyBridge", "HOLD_ON_3", "HOLD_ON_4"),
    ("SlipperyBridge", "HOLD_ON_4", "HOLD_ON_5"),
    ("SlipperyBridge", "HOLD_ON_5", "HOLD_ON_6"),
    ("SlipperyBridge", "HOLD_ON_6", "HOLD_ON_LOOP"),
    ("SlipperyBridge", "HOLD_ON_LOOP", "HOLD_ON_LOOP"),
    ("SlipperyBridge", "INITIAL", "HOLD_ON_0"),
    ("SlipperyBridge", "INITIAL", "OVERCOME"),
    ("SpiralingWhirlpool", "INITIAL", "DRINK"),
    ("SpiralingWhirlpool", "INITIAL", "OBSERVE"),
    ("SpiritGrafter", "INITIAL", "LET_IT_IN"),
    ("SpiritGrafter", "INITIAL", "REJECTION"),
    ("StoneOfAllTime", "INITIAL", "LIFT"),
    ("StoneOfAllTime", "INITIAL", "PUSH"),
    ("SunkenStatue", "INITIAL", "DIVE_INTO_WATER"),
    ("SunkenStatue", "INITIAL", "GRAB_SWORD"),
    ("SunkenTreasury", "INITIAL", "FIRST_CHEST"),
    ("SunkenTreasury", "INITIAL", "SECOND_CHEST"),
    ("Symbiote", "INITIAL", "APPROACH"),
    ("Symbiote", "INITIAL", "KILL_WITH_FIRE"),
    ("TabletOfTruth", "DECIPHER", "GIVE_UP"),
    ("TabletOfTruth", "DECIPHER_1", "DECIPHER"),
    ("TabletOfTruth", "DECIPHER_2", "DECIPHER"),
    ("TabletOfTruth", "DECIPHER_3", "DECIPHER"),
    ("TabletOfTruth", "DECIPHER_4", "DECIPHER"),
    ("TabletOfTruth", "INITIAL", "DECIPHER_1"),
    ("TabletOfTruth", "INITIAL", "SMASH"),
    ("TeaMaster", "INITIAL", "BONE_TEA"),
    ("TeaMaster", "INITIAL", "EMBER_TEA"),
    ("TeaMaster", "INITIAL", "TEA_OF_DISCOURTESY"),
    ("TheFutureOfPotions", "INITIAL", "POTION"),
    ("TheLanternKey", "INITIAL", "KEEP_THE_KEY"),
    ("TheLanternKey", "INITIAL", "RETURN_THE_KEY"),
    ("TheLanternKey", "KEEP_THE_KEY", "FIGHT"),
    ("TheLegendsWereTrue", "INITIAL", "NAB_THE_MAP"),
    ("TheLegendsWereTrue", "INITIAL", "SLOWLY_FIND_AN_EXIT"),
    ("ThisOrThat", "INITIAL", "ORNATE"),
    ("ThisOrThat", "INITIAL", "PLAIN"),
    ("TrashHeap", "INITIAL", "DIVE_IN"),
    ("TrashHeap", "INITIAL", "GRAB"),
    ("Trial", "INITIAL", "ACCEPT"),
    ("Trial", "INITIAL", "REJECT"),
    ("Trial", "MERCHANT", "GUILTY"),
    ("Trial", "MERCHANT", "INNOCENT"),
    ("Trial", "NOBLE", "GUILTY"),
    ("Trial", "NOBLE", "INNOCENT"),
    ("Trial", "NONDESCRIPT", "GUILTY"),
    ("Trial", "NONDESCRIPT", "INNOCENT"),
    ("Trial", "REJECT", "ACCEPT"),
    ("Trial", "REJECT", "DOUBLE_DOWN"),
    ("UnrestSite", "INITIAL", "KILL"),
    ("UnrestSite", "INITIAL", "REST"),
    ("WaterloggedScriptorium", "INITIAL", "BLOODY_INK"),
    ("WaterloggedScriptorium", "INITIAL", "PRICKLY_SPONGE"),
    ("WaterloggedScriptorium", "INITIAL", "TENTACLE_QUILL"),
    ("WelcomeToWongos", "INITIAL", "BARGAIN_BIN"),
    ("WelcomeToWongos", "INITIAL", "FEATURED_ITEM"),
    ("WelcomeToWongos", "INITIAL", "LEAVE"),
    ("WelcomeToWongos", "INITIAL", "MYSTERY_BOX"),
    ("Wellspring", "INITIAL", "BATHE"),
    ("Wellspring", "INITIAL", "BOTTLE"),
    ("WhisperingHollow", "INITIAL", "GOLD"),
    ("WhisperingHollow", "INITIAL", "HUG"),
    ("WoodCarvings", "INITIAL", "BIRD"),
    ("WoodCarvings", "INITIAL", "SNAKE"),
    ("WoodCarvings", "INITIAL", "TORUS"),
    ("ZenWeaver", "INITIAL", "ARACHNID_ACUPUNCTURE"),
    ("ZenWeaver", "INITIAL", "BREATHING_TECHNIQUES"),
    ("ZenWeaver", "INITIAL", "EMOTIONAL_AWARENESS"),
];

/// The encounters an act can end with, append-only. A test holds every boss
/// encounter to it.
pub const RUN_BOSSES: &[Encounter] = &[
    Encounter::VantomBoss,
    Encounter::CeremonialBeastBoss,
    Encounter::TheKinBoss,
    Encounter::LagavulinMatriarchBoss,
    Encounter::SoulFyshBoss,
    Encounter::WaterfallGiantBoss,
    Encounter::KaiserCrabBoss,
    Encounter::KnowledgeDemonBoss,
    Encounter::TheInsatiableBoss,
    Encounter::AeonglassBoss,
    Encounter::QueenBoss,
    Encounter::TestSubjectBoss,
];

/// Embedding sizes, the pad included.
pub const RELIC_VOCAB: usize = N_RELICS + 1 + RUN_RELICS.len();
pub const DECISION_VOCAB: usize = DECISIONS.len() + 1;
pub const OPTION_VOCAB: usize = OPTION_KINDS.len() + 1;
pub const ROOM_VOCAB: usize = ROOMS.len() + 1;
pub const ACT_VOCAB: usize = ACTS.len() + 1;
pub const EVENT_VOCAB: usize = RUN_EVENTS.len() + 1;
pub const BOSS_VOCAB: usize = RUN_BOSSES.len() + 1;
pub const EVENT_OPTION_VOCAB: usize = EVENT_OPTIONS.len() + 1;

/// A run decision encoded: one row of the run buffers, and the chooser's
/// answer for each option token (the index `rooms::Decision` reads, where a
/// skip is past the end).
#[derive(Clone, Debug, PartialEq)]
pub struct RunObs {
    pub floats: Vec<f32>,
    pub ids: Vec<i64>,
    pub answers: Vec<usize>,
    /// Each option in words (`option_names`).
    pub names: Vec<String>,
    /// At a map step, the fights the forecast values (`forecast`).
    pub forecast: Vec<FightSetup>,
}

fn position<T: PartialEq>(list: &[T], x: &T) -> i64 {
    list.iter().position(|y| y == x).map_or(0, |i| i as i64 + 1)
}

fn card_id(id: &str) -> i64 {
    sim_card(id).map_or(0, |c| c as i64 + 1)
}

fn enchant_id(id: &str) -> i64 {
    sim_enchantment(id).map_or(0, |e| e as i64 + 1)
}

fn relic_id(id: &str) -> i64 {
    match sim_relic(id) {
        Some(r) => r as i64 + 1,
        None => RUN_RELICS.iter().position(|&r| r == id).map_or(0, |i| (N_RELICS + 1 + i) as i64),
    }
}

fn potion_id(id: &str) -> i64 {
    sim_potion(id).map_or(0, |p| p as i64 + 1)
}

fn room_id(room: Room) -> i64 {
    let name = match room {
        Room::Ancient(_) => "Ancient",
        Room::Event(_) => "Event",
        Room::Treasure => "Treasure",
        Room::Shop => "Shop",
        Room::RestSite => "RestSite",
        Room::Combat(kind, _) => match kind {
            crate::run::RoomType::Elite => "Elite",
            crate::run::RoomType::Boss => "Boss",
            _ => "Monster",
        },
    };
    position(&ROOMS, &name)
}

fn point_id(kind: PointType) -> i64 {
    position(&ROOMS, &format!("{kind:?}").as_str())
}

/// An event option's id: its `EVENT_OPTIONS` index + 1, 0 if not listed.
fn event_option_id(event: &str, page: &str, key: &str) -> i64 {
    static INDEX: std::sync::OnceLock<std::collections::HashMap<(&str, &str, &str), i64>> = std::sync::OnceLock::new();
    let index = INDEX.get_or_init(|| EVENT_OPTIONS.iter().enumerate().map(|(i, &o)| (o, i as i64 + 1)).collect());
    index.get(&(event, page, key)).copied().unwrap_or(0)
}

/// An event option in words, as `sim/vocab.txt` and `option_names` give
/// it: "Reflections INITIAL.SHATTER", or "DollRoom BING_BONG" for an option
/// on no page.
pub fn event_option_name(event: &str, page: &str, key: &str) -> String {
    if page.is_empty() { format!("{event} {key}") } else { format!("{event} {page}.{key}") }
}

/// An event option's `Shown` as the option token's last `EVENT_FEATS`
/// floats.
fn shown_feats(s: &Shown) -> [f32; EVENT_FEATS] {
    let c = s.cards;
    let kind = |k: CardType| c.is_some_and(|c| c.kind == Some(k)) as u8 as f32;
    let rarity = |r: Rarity| c.is_some_and(|c| c.rarity == Some(r)) as u8 as f32;
    let flag = |b: bool| b as u8 as f32;
    [
        s.hp as f32 / 10.0,
        s.max_hp as f32 / 10.0,
        s.gold as f32 / 100.0,
        c.map_or(0.0, |c| c.count as f32 / 2.0),
        c.map_or(0.0, |c| c.from as f32 / 8.0),
        kind(CardType::Attack),
        kind(CardType::Skill),
        kind(CardType::Power),
        rarity(Rarity::Common),
        rarity(Rarity::Uncommon),
        rarity(Rarity::Rare),
        flag(c.is_some_and(|c| c.colorless)),
        flag(c.is_some_and(|c| c.upgraded)),
        s.remove as f32 / 4.0,
        s.transform as f32 / 4.0,
        s.upgrade as f32 / 4.0,
        s.downgrade as f32 / 4.0,
        s.enchant as f32 / 4.0,
        s.duplicate as f32 / 20.0,
        flag(s.random),
        s.curses as f32 / 2.0,
        s.relics as f32 / 2.0,
        s.potions as f32 / 2.0,
        flag(s.fight),
    ]
}

/// One option token, before it is written.
#[derive(Default)]
struct Opt {
    kind: i64,
    cards: [(i64, bool); OPTION_CARDS],
    enchant: (i64, i32),
    relic: i64,
    potion: i64,
    room: i64,
    event_option: i64,
    column: i64,
    price: i32,
    map: [f32; MAP_FEATS],
    shown: [f32; EVENT_FEATS],
    answer: usize,
}

impl Opt {
    fn new(kind: OptionKind, answer: usize) -> Self {
        Opt { kind: position(&OPTION_KINDS, &kind), answer, ..Default::default() }
    }

    fn card(mut self, id: &str, upgraded: bool, enchantment: Option<(&str, i32)>) -> Self {
        if let Some(slot) = self.cards.iter_mut().find(|c| c.0 == 0) {
            *slot = (card_id(id), upgraded);
        }
        if let Some((e, amount)) = enchantment {
            self.enchant = (enchant_id(e), amount);
        }
        self
    }

    fn offer(self, offer: &Offer) -> Self {
        self.card(offer.id, offer.upgraded, offer.enchantment)
    }
}

/// What the paths from a map point to the act's boss hold: per
/// `PATH_POINTS` type the fewest and the most, the point itself counted,
/// then the fewest rows to a rest site and to a shop (the point's own row
/// is 0), normalized by 8, none reachable reading 2; then the fewest and
/// most elites on those paths before the first rest site (or the boss),
/// the point itself counted.
pub fn path_summary(map: &ActMap, point: PointId) -> [f32; MAP_FEATS] {
    let n = map.all_points().map(PointId::index).max().map_or(0, |m| m + 1);
    let mut memo: Vec<Option<Summary>> = vec![None; n];
    let s = summarize(map, point, &mut memo);
    let mut out = [0f32; MAP_FEATS];
    for t in 0..PATH_POINTS.len() {
        out[2 * t] = s.min[t] as f32 / 8.0;
        out[2 * t + 1] = s.max[t] as f32 / 8.0;
    }
    let far = |d: u32| if d == u32::MAX { 2.0 } else { d as f32 / 8.0 };
    out[MAP_FEATS - 4] = far(s.rest);
    out[MAP_FEATS - 3] = far(s.shop);
    out[MAP_FEATS - 2] = s.elites_to_rest.0 as f32;
    out[MAP_FEATS - 1] = s.elites_to_rest.1 as f32;
    out
}

#[derive(Clone, Copy)]
struct Summary {
    min: [u32; PATH_POINTS.len()],
    max: [u32; PATH_POINTS.len()],
    rest: u32,
    shop: u32,
    /// Fewest and most elites before the first rest site.
    elites_to_rest: (u32, u32),
}

fn summarize(map: &ActMap, point: PointId, memo: &mut [Option<Summary>]) -> Summary {
    if let Some(s) = memo[point.index()] {
        return s;
    }
    let children: Vec<PointId> = map[point].children.iter().collect();
    let mut s = if children.is_empty() {
        Summary { min: [0; PATH_POINTS.len()], max: [0; PATH_POINTS.len()], rest: u32::MAX, shop: u32::MAX, elites_to_rest: (0, 0) }
    } else {
        let mut acc = Summary {
            min: [u32::MAX; PATH_POINTS.len()],
            max: [0; PATH_POINTS.len()],
            rest: u32::MAX,
            shop: u32::MAX,
            elites_to_rest: (u32::MAX, 0),
        };
        for c in children {
            let cs = summarize(map, c, memo);
            for t in 0..PATH_POINTS.len() {
                acc.min[t] = acc.min[t].min(cs.min[t]);
                acc.max[t] = acc.max[t].max(cs.max[t]);
            }
            acc.rest = acc.rest.min(cs.rest.saturating_add(1));
            acc.shop = acc.shop.min(cs.shop.saturating_add(1));
            acc.elites_to_rest = (acc.elites_to_rest.0.min(cs.elites_to_rest.0), acc.elites_to_rest.1.max(cs.elites_to_rest.1));
        }
        acc
    };
    let kind = map[point].kind;
    if let Some(t) = PATH_POINTS.iter().position(|&k| k == kind) {
        s.min[t] += 1;
        s.max[t] += 1;
    }
    if kind == PointType::RestSite {
        s.rest = 0;
        s.elites_to_rest = (0, 0);
    }
    if kind == PointType::Elite {
        s.elites_to_rest = (s.elites_to_rest.0 + 1, s.elites_to_rest.1 + 1);
    }
    if kind == PointType::Shop {
        s.shop = 0;
    }
    memo[point.index()] = Some(s);
    s
}

/// Writes the map a player can still walk from `points`, a map step's
/// options, into `out` (the `I_MAP` segment): every point reachable from
/// them short of the boss, at its row counted from the options' and its
/// column, with its type as the map shows it (a `?` stays `Unknown`) and
/// its links to the row above.
fn map_ahead(map: &ActMap, points: &[PointId], out: &mut [i64]) {
    let Some(&first) = points.first() else { return };
    let base = map[first].row;
    let mut stack: Vec<PointId> = points.to_vec();
    while let Some(p) = stack.pop() {
        let point = &map[p];
        if point.kind == PointType::Boss {
            continue;
        }
        let node = ((point.row - base) * MAP_COLS + point.col) * MAP_NODE_IDS;
        if out[node] != 0 {
            continue;
        }
        let mut links = 0;
        for c in point.children.iter().filter(|&c| map[c].kind != PointType::Boss) {
            links |= 1 << (map[c].col + 1 - point.col);
            stack.push(c);
        }
        out[node] = point_id(point.kind);
        out[node + 1] = links;
    }
}

fn rest_kind(option: RestOption) -> OptionKind {
    match option {
        RestOption::Heal => OptionKind::RestHeal,
        RestOption::Smith => OptionKind::RestSmith,
        RestOption::Lift => OptionKind::RestLift,
        RestOption::Dig => OptionKind::RestDig,
        RestOption::Kindle => OptionKind::RestKindle,
        RestOption::Cook => OptionKind::RestCook,
        RestOption::Clone => OptionKind::RestClone,
    }
}

fn deck_kind(action: DeckAction) -> OptionKind {
    match action {
        DeckAction::Upgrade => OptionKind::DeckUpgrade,
        DeckAction::Remove => OptionKind::DeckRemove,
        DeckAction::Duplicate => OptionKind::DeckDuplicate,
        DeckAction::Enchant(..) => OptionKind::DeckEnchant,
        DeckAction::Transform { upgrade: false } => OptionKind::DeckTransform,
        DeckAction::Transform { upgrade: true } => OptionKind::DeckTransformUpgraded,
        DeckAction::Maul => OptionKind::DeckMaul,
        DeckAction::TransformInto(_) => OptionKind::DeckTransformInto,
    }
}

/// The options of `decision` in the order it lists them, each with the
/// answer that takes it; a skip, where the decision has one, last.
fn options(run: &RunState, decision: Decision<'_>) -> (usize, Vec<Opt>) {
    use OptionKind as K;
    let skip = |kind: K, len: usize| Opt::new(kind, len);
    let (kind, opts): (&str, Vec<Opt>) = match decision {
        Decision::Path(map, points) => (
            "Path",
            points
                .iter()
                .enumerate()
                .map(|(i, &p)| Opt {
                    room: point_id(map[p].kind),
                    map: path_summary(map, p),
                    column: if map[p].kind == PointType::Boss { 0 } else { map[p].col as i64 + 1 },
                    ..Opt::new(K::Path, i)
                })
                .collect(),
        ),
        Decision::Relic(relics) => (
            "Relic",
            relics
                .iter()
                .enumerate()
                .map(|(i, r)| Opt { relic: relic_id(r), ..Opt::new(K::Relic, i) })
                .chain([skip(K::LeaveRelics, relics.len())])
                .collect(),
        ),
        Decision::Card(cards) => (
            "Card",
            cards.iter().enumerate().map(|(i, c)| Opt::new(K::Card, i).offer(c)).chain([skip(K::SkipCard, cards.len())]).collect(),
        ),
        Decision::Bundle(bundles) => (
            "Bundle",
            bundles
                .iter()
                .enumerate()
                .map(|(i, b)| b.iter().fold(Opt::new(K::Bundle, i), Opt::offer))
                .chain([skip(K::SkipBundle, bundles.len())])
                .collect(),
        ),
        Decision::Potion(potion) => (
            "Potion",
            vec![Opt { potion: potion_id(potion), ..Opt::new(K::KeepPotion, 0) }, Opt { potion: potion_id(potion), ..Opt::new(K::LeavePotion, 1) }],
        ),
        Decision::Rest(options) => ("Rest", options.iter().enumerate().map(|(i, &o)| Opt::new(rest_kind(o), i)).collect()),
        Decision::Ancient(relics) => {
            ("Ancient", relics.iter().enumerate().map(|(i, r)| Opt { relic: relic_id(r), ..Opt::new(K::Ancient, i) }).collect())
        }
        Decision::Deck { action, cards, optional } => (
            "Deck",
            cards
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let card = &run.deck[c];
                    let e = card.enchantment.as_ref().map(|e| (e.id.as_str(), e.amount));
                    Opt::new(deck_kind(action), i).card(&card.id, card.upgraded, e)
                })
                .chain(optional.then(|| skip(K::StopPicking, cards.len())))
                .collect(),
        ),
        Decision::Shop(wares) => (
            "Shop",
            wares
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let o = match (&w.item, w.slot) {
                        (Item::Card(c), Slot::Colorless(_)) => Opt::new(K::ShopColorless, i).offer(c),
                        (Item::Card(c), _) => Opt::new(K::ShopCard, i).offer(c),
                        (Item::Relic(r), _) => Opt { relic: relic_id(r), ..Opt::new(K::ShopRelic, i) },
                        (Item::Potion(p), _) => Opt { potion: potion_id(p), ..Opt::new(K::ShopPotion, i) },
                        (Item::Removal, _) => Opt::new(K::ShopRemoval, i),
                    };
                    Opt { price: w.price, ..o }
                })
                .chain([skip(K::LeaveShop, wares.len())])
                .collect(),
        ),
        Decision::Event { event, options } => (
            "Event",
            options
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    let mut opt = Opt { event_option: event_option_id(event, o.page, o.key), shown: shown_feats(&o.shown), ..Opt::new(K::Event, i) };
                    for item in &o.items {
                        if sim_card(item).is_some() {
                            opt = opt.card(item, false, None);
                        } else if let Some(p) = sim_potion(item) {
                            opt.potion = p as i64 + 1;
                        } else if relic_id(item) > 0 {
                            opt.relic = relic_id(item);
                        }
                    }
                    opt
                })
                .collect(),
        ),
    };
    (DECISIONS.iter().position(|&d| d == kind).expect("a decision kind") + 1, opts)
}

/// Each option of `decision` in words, aligned with its option tokens
/// (`options`, cut as `observe` cuts them): what an afterstate's path
/// reads as.
pub fn option_names(run: &RunState, decision: Decision<'_>) -> Vec<String> {
    let offer = |o: &Offer| format!("{}{}", o.id, if o.upgraded { "+" } else { "" });
    let mut names: Vec<String> = match decision {
        Decision::Path(map, points) => points.iter().map(|&p| format!("{:?}", map[p].kind)).collect(),
        Decision::Relic(relics) => relics.iter().cloned().chain(["leave".into()]).collect(),
        Decision::Card(cards) => cards.iter().map(offer).chain(["skip".into()]).collect(),
        Decision::Bundle(bundles) => bundles.iter().map(|b| b.iter().map(offer).collect::<Vec<_>>().join("+")).chain(["skip".into()]).collect(),
        Decision::Potion(potion) => vec![format!("keep {potion}"), format!("leave {potion}")],
        Decision::Rest(options) => options.iter().map(|o| format!("{o:?}")).collect(),
        Decision::Ancient(relics) => relics.to_vec(),
        Decision::Deck { action, cards, optional } => cards
            .iter()
            .map(|&c| format!("{action:?} {}{}", run.deck[c].id, if run.deck[c].upgraded { "+" } else { "" }))
            .chain(optional.then(|| "stop".into()))
            .collect(),
        Decision::Shop(wares) => wares
            .iter()
            .map(|w| match &w.item {
                Item::Card(c) => format!("{} {}g", offer(c), w.price),
                Item::Relic(r) => format!("{r} {}g", w.price),
                Item::Potion(p) => format!("{p} {}g", w.price),
                Item::Removal => format!("removal {}g", w.price),
            })
            .chain(["leave".into()])
            .collect(),
        Decision::Event { event, options } => {
            options.iter().map(|o| [event_option_name(event, o.page, o.key)].into_iter().chain(o.items.iter().cloned()).collect::<Vec<_>>().join(" ")).collect()
        }
    };
    cut_options(&mut names);
    names
}

/// Cuts a decision's options to `MAX_OPTIONS`, keeping the last: a skip
/// is the one way out of some decisions.
fn cut_options<T>(options: &mut Vec<T>) {
    if options.len() > MAX_OPTIONS {
        let last = options.pop().expect("options");
        options.truncate(MAX_OPTIONS - 1);
        options.push(last);
    }
}

/// How many option tokens `decision` has, before `MAX_OPTIONS` cuts it.
pub fn option_count(decision: Decision<'_>) -> usize {
    match decision {
        Decision::Path(_, points) => points.len(),
        Decision::Relic(relics) => relics.len() + 1,
        Decision::Card(cards) => cards.len() + 1,
        Decision::Bundle(bundles) => bundles.len() + 1,
        Decision::Potion(_) => 2,
        Decision::Rest(options) => options.len(),
        Decision::Ancient(relics) => relics.len(),
        Decision::Deck { cards, optional, .. } => cards.len() + optional as usize,
        Decision::Shop(wares) => wares.len() + 1,
        Decision::Event { options, .. } => options.len(),
    }
}

/// Encodes `decision` as `run` stands when it is put: the row the run
/// policy reads, and each option's answer.
pub fn observe(run: &RunState, decision: Decision<'_>) -> RunObs {
    let mut f = vec![0f32; RUN_FLOATS];
    let mut ids = vec![0i64; RUN_IDS];
    let (kind, opts) = options(run, decision);

    let plan = &run.plan.acts[run.act];
    ids[..GLOBAL_IDS].copy_from_slice(&[
        kind as i64,
        run.room.map_or(0, room_id),
        position(&ACTS, &plan.act),
        position(RUN_BOSSES, &plan.boss),
        plan.second_boss.map_or(0, |b| position(RUN_BOSSES, &b)),
        match run.room {
            Some(Room::Event(name) | Room::Ancient(name)) => position(RUN_EVENTS, &name),
            _ => 0,
        },
    ]);
    let odds = run.unknown_odds.odds();
    let slots = run.potions.len();
    let empty = run.potions.iter().filter(|p| p.is_none()).count();
    f[..F_FORECAST].copy_from_slice(&[
        run.hp.max(0) as f32 / run.max_hp.max(1) as f32,
        run.max_hp as f32 / 100.0,
        run.gold as f32 / 500.0,
        run.floor as f32 / 49.0,
        run.ascension.0 as f32 / 20.0,
        odds[0],
        odds[1],
        odds[2],
        odds[3],
        run.card_odds.0 * 2.0,
        run.potion_odds.0,
        run.deck.len() as f32 / 40.0,
        slots as f32 / 5.0,
        empty as f32 / 5.0,
        run.shop_removals as f32 / 5.0,
        // HP in points as well as the fraction above: with the fraction
        // alone a value head reads lost max HP (Paper Cuts) as HP kept.
        run.hp.max(0) as f32 / 100.0,
    ]);

    for (k, card) in run.deck.iter().take(MAX_DECK).enumerate() {
        let (i, x) = (I_DECK + k * DECK_IDS, F_DECK + k * DECK_FLOATS);
        ids[i] = card_id(&card.id);
        ids[i + 1] = card.enchantment.as_ref().map_or(0, |e| enchant_id(&e.id));
        f[x] = 1.0;
        f[x + 1] = card.upgraded as u8 as f32;
        f[x + 2] = card.enchantment.as_ref().map_or(0.0, |e| e.amount as f32 / 10.0);
    }
    for (k, relic) in run.relics.iter().take(MAX_RELICS).enumerate() {
        let x = F_RELICS + k * RELIC_FLOATS;
        ids[I_RELICS + k * RELIC_IDS] = relic_id(&relic.id);
        f[x] = 1.0;
        f[x + 1] = relic.counter as f32 / 10.0;
        f[x + 2] = relic.flag as u8 as f32;
    }
    for (k, potion) in run.held_potions().take(MAX_POTIONS).enumerate() {
        ids[I_POTIONS + k * POTION_IDS] = potion_id(potion);
        f[F_POTIONS + k * POTION_FLOATS] = 1.0;
    }

    let mut opts = opts;
    cut_options(&mut opts);
    let answers = opts.iter().map(|o| o.answer).collect();
    for (k, o) in opts.iter().enumerate() {
        let (i, x) = (I_OPTIONS + k * OPTION_IDS, F_OPTIONS + k * OPTION_FLOATS);
        let row = &mut ids[i..i + OPTION_IDS];
        row[0] = o.kind;
        for (c, &(id, _)) in o.cards.iter().enumerate() {
            row[1 + c] = id;
        }
        row[1 + OPTION_CARDS..].copy_from_slice(&[o.enchant.0, o.relic, o.potion, o.room, o.event_option, o.column]);
        let row = &mut f[x..x + OPTION_FLOATS];
        row[0] = 1.0;
        for (c, &(_, up)) in o.cards.iter().enumerate() {
            row[1 + c] = up as u8 as f32;
        }
        row[1 + OPTION_CARDS] = o.enchant.1 as f32 / 10.0;
        row[2 + OPTION_CARDS] = o.price as f32 / 100.0;
        row[3 + OPTION_CARDS..3 + OPTION_CARDS + MAP_FEATS].copy_from_slice(&o.map);
        row[3 + OPTION_CARDS + MAP_FEATS..].copy_from_slice(&o.shown);
    }
    let mut forecast = Vec::new();
    if let Decision::Path(map, points) = decision {
        map_ahead(map, points, &mut ids[I_MAP..]);
        forecast = forecast_fights(run);
    }
    RunObs { floats: f, ids, answers, names: option_names(run, decision), forecast }
}

/// Openings the forecast rolls per encounter.
pub const FORECAST_ROLLS: usize = 4;

/// The fights a map step's forecast values (docs/run-env.md, The
/// forecast): each elite the act can hold, as a pool, and the act's boss
/// and second boss, which the map shows, against the player as they stand,
/// `FORECAST_ROLLS` openings each. The enemies and each combat's seed come
/// from the roll's index alone, never the run's seed or streams, so the
/// forecast is a function of what the player sees; which elite a map point
/// will hold is the plan's and stays out. Cards, relics and potions the
/// combat sim lacks (other characters' cards from Splash or Kaleidoscope)
/// are left out of the fights, as the rest of the player is still worth
/// reading.
pub fn forecast_fights(run: &RunState) -> Vec<FightSetup> {
    let plan = &run.plan.acts[run.act];
    let elites = crate::encounter::ALL.iter().copied().filter(|e| e.kind() == Kind::Elite && e.act() == plan.act);
    rolled(run, elites.chain([plan.boss]).chain(plan.second_boss).collect())
}

/// The next act's elites and bosses, all of each, pooled the way
/// `forecast_fights` pools the elites: the map does not show the next
/// act's boss until the act begins. Against the player as they stand, so
/// a pick that only pays off later still shows. Empty in the last act.
pub fn lookahead_fights(run: &RunState) -> Vec<FightSetup> {
    let Some(next) = run.plan.acts.get(run.act + 1) else { return Vec::new() };
    let hard = |e: &Encounter| matches!(e.kind(), Kind::Elite | Kind::Boss) && e.act() == next.act;
    rolled(run, crate::encounter::ALL.iter().copied().filter(hard).collect())
}

/// `encounters` against `run`'s player, `FORECAST_ROLLS` openings each,
/// with the cards, relics, enchantments and potions the sim lacks left
/// out; empty when one cannot be built.
fn rolled(run: &RunState, encounters: Vec<Encounter>) -> Vec<FightSetup> {
    let mut player = run.clone();
    player.deck.retain(|c| sim_card(&c.id).is_some_and(|id| !crate::card::UNSUPPORTED_CARDS.iter().any(|(u, _)| *u == id)));
    for card in &mut player.deck {
        if card.enchantment.as_ref().is_some_and(|e| sim_enchantment(&e.id).is_none()) {
            card.enchantment = None;
        }
    }
    player.relics.retain(|r| sim_relic(&r.id).is_some() || crate::gen::INERT_RELICS.contains(&r.id.as_str()));
    for slot in &mut player.potions {
        if slot.as_deref().is_some_and(|p| sim_potion(p).is_none()) {
            *slot = None;
        }
    }
    let mut out = Vec::with_capacity(encounters.len() * FORECAST_ROLLS);
    for encounter in encounters {
        for roll in 0..FORECAST_ROLLS {
            let enemies = encounter.monsters(&mut Rng::new(0xF0CA_57 + roll as u64));
            match player.fight_setup(encounter, enemies) {
                Ok(setup) => out.push(setup),
                Err(_) => return Vec::new(),
            }
        }
    }
    out
}

/// `fights` (`forecast_fights`, one or more runs' back to back) encoded
/// at their openings as the combat model reads them, a row each, in
/// parallel: `floats [n * N_FLOATS]`, `ids [n * N_IDS]`. Roll `k` of an
/// encounter opens with combat seed `k`.
pub fn forecast_rows(fights: &[FightSetup]) -> (Vec<f32>, Vec<i64>) {
    use crate::encode::{encode, N_ACTIONS, N_FLOATS, N_IDS};
    use rayon::prelude::*;
    let mut floats = vec![0f32; fights.len() * N_FLOATS];
    let mut ids = vec![0i64; fights.len() * N_IDS];
    floats.par_chunks_mut(N_FLOATS).zip(ids.par_chunks_mut(N_IDS)).zip(fights.par_iter().enumerate()).for_each_init(
        || [false; N_ACTIONS],
        |mask, ((f, i), (k, setup))| encode(&setup.combat((k % FORECAST_ROLLS) as u64), f, i, mask),
    );
    (floats, ids)
}

/// The run vocabularies for `sim/vocab.txt`, after the combat ones.
pub fn vocab_text() -> String {
    let mut out = String::new();
    for d in DECISIONS {
        out += &format!("decision {d}\n");
    }
    for k in OPTION_KINDS {
        out += &format!("option {k:?}\n");
    }
    for r in ROOMS {
        out += &format!("room {r}\n");
    }
    for a in ACTS {
        out += &format!("act {a:?}\n");
    }
    for e in RUN_EVENTS {
        out += &format!("event {e}\n");
    }
    for b in RUN_BOSSES {
        out += &format!("boss {b:?}\n");
    }
    for r in RUN_RELICS {
        out += &format!("runrelic {r}\n");
    }
    for &(event, page, key) in EVENT_OPTIONS {
        out += &format!("eventoption {}\n", event_option_name(event, page, key));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{select_acts, Unlocks};
    use crate::run::{DeckCard, Enchant, RunRelic};
    use crate::types::Ascension;

    /// The run as the player sees it at the first act's start, with a deck,
    /// relics, potions, HP and gold set, whatever the seed drew.
    fn visible(seed: &str) -> RunState {
        let unlocks = Unlocks::default();
        let acts = select_acts(crate::game_rng::RunRngs::new("SAME").seed, &unlocks);
        let mut run = RunState::new(seed, acts, Ascension(10), &unlocks);
        run.enter_act(0);
        run.plan.acts[0].boss = Encounter::VantomBoss;
        run.deck.push(DeckCard { id: "BODY_SLAM".into(), upgraded: true, enchantment: Some(Enchant { id: "SHARP".into(), amount: 2 }) });
        run.relics.push(RunRelic::new("MAW_BANK"));
        run.potions[0] = Some("FIRE_POTION".into());
        (run.hp, run.gold, run.floor) = (41, 173, 6);
        run
    }

    /// Two runs whose seeds differ, and with them the streams and the plan
    /// (upcoming encounters, events, the relic bags), but whose visible
    /// state is the same encode the same at every kind of decision.
    #[test]
    fn hidden_information_does_not_reach_the_encoding() {
        let (a, b) = (visible("SEEDA"), visible("SEEDB"));
        assert_ne!(a.rngs.seed, b.rngs.seed);
        assert_ne!(a.plan.acts[0].normal, b.plan.acts[0].normal, "the plans should differ for the test to mean anything");
        let map = ActMap::generate(a.rngs.seed, a.plan.acts[0].act, a.ascension);
        let points: Vec<PointId> = map[map.start].children.iter().collect();
        let offers = [Offer::new("BASH"), Offer::new("ANGER")];
        let relics = ["ANCHOR".to_string(), "WINGED_BOOTS".to_string()];
        let decisions = [
            Decision::Path(&map, &points),
            Decision::Card(&offers),
            Decision::Relic(&relics),
            Decision::Deck { action: DeckAction::Upgrade, cards: &[0, 3, 10], optional: true },
        ];
        for d in decisions {
            let (ea, eb) = (observe(&a, d), observe(&b, d));
            assert_eq!(ea, eb, "{d:?}");
        }
        let mut c = visible("SEEDA");
        c.hp -= 1;
        assert_ne!(observe(&a, Decision::Card(&offers)), observe(&c, Decision::Card(&offers)), "HP is visible");
    }

    /// The next act's forecast holds every elite and boss that act can
    /// hold, whichever boss the run's plan rolled, so two runs whose plans
    /// differ look ahead to the same fights; the last act has none.
    #[test]
    fn the_lookahead_reads_no_hidden_information() {
        let (a, b) = (visible("SEEDA"), visible("SEEDB"));
        let fights = lookahead_fights(&a);
        assert_eq!(fights, lookahead_fights(&b));
        let next = a.plan.acts[1].act;
        let hard: Vec<Encounter> = crate::encounter::ALL.iter().copied().filter(|e| matches!(e.kind(), Kind::Elite | Kind::Boss) && e.act() == next).collect();
        assert_eq!(fights.len(), hard.len() * FORECAST_ROLLS);
        assert!(hard.iter().all(|e| fights.iter().any(|f| f.encounter == *e)), "every elite and boss of the next act");
        let mut last = a.clone();
        last.act = last.plan.acts.len() - 1;
        assert!(lookahead_fights(&last).is_empty());
    }

    /// The forecast fights are the act's elites as a pool and its boss,
    /// against the player as they stand: two runs whose seeds, streams and
    /// plans (the elites' order among them) differ roll the same fights and
    /// encode them the same, and HP reaches them.
    #[test]
    fn the_forecast_reads_no_hidden_information() {
        let (a, b) = (visible("SEEDA"), visible("SEEDB"));
        assert_ne!(a.plan.acts[0].elites, b.plan.acts[0].elites, "the plans' elites should differ for the test to mean anything");
        let fights = forecast_fights(&a);
        assert_eq!(fights, forecast_fights(&b));
        assert_eq!(forecast_rows(&fights), forecast_rows(&forecast_fights(&b)));
        let act = a.plan.acts[0].act;
        let elites = crate::encounter::ALL.iter().filter(|e| e.kind() == Kind::Elite && e.act() == act).count();
        assert_eq!(fights.len(), (elites + 1) * FORECAST_ROLLS, "every elite of the act and the boss");
        assert!(fights.iter().all(|f| f.hp == a.hp && f.deck.len() == a.deck.len()));
        assert_eq!(fights.last().map(|f| f.encounter), Some(Encounter::VantomBoss));
        let map = ActMap::generate(a.rngs.seed, act, a.ascension);
        let points: Vec<PointId> = map[map.start].children.iter().collect();
        assert_eq!(observe(&a, Decision::Path(&map, &points)).forecast, fights, "a map step carries them");
        assert!(observe(&a, Decision::Card(&[Offer::new("BASH")])).forecast.is_empty(), "other decisions do not");
        let mut c = visible("SEEDA");
        c.hp -= 10;
        assert_ne!(forecast_rows(&forecast_fights(&c)), forecast_rows(&fights), "HP is visible");
    }

    #[test]
    fn options_answer_as_the_decision_reads_them() {
        let run = visible("SEEDA");
        let offers = [Offer::new("BASH"), Offer::new("ANGER")];
        let obs = observe(&run, Decision::Card(&offers));
        assert_eq!(obs.answers, [0, 1, 2], "two cards and the skip past the end");
        assert_eq!(obs.ids[I_OPTIONS + 1], card_id("BASH"));
        assert_eq!(obs.floats[F_OPTIONS + 2 * OPTION_FLOATS], 1.0, "the skip is present");
        assert_eq!(obs.floats[F_OPTIONS + 3 * OPTION_FLOATS], 0.0, "nothing after it");
        let deck: Vec<usize> = (0..100).map(|i| i % run.deck.len()).collect();
        let obs = observe(&run, Decision::Deck { action: DeckAction::Remove, cards: &deck, optional: true });
        assert_eq!(obs.answers.len(), MAX_OPTIONS);
        assert_eq!(*obs.answers.last().unwrap(), deck.len(), "a long pick keeps its way out");
    }

    /// The summary counts the point itself and every path's rooms: from the
    /// start, the fewest rest sites on a path is at most the most, and some
    /// path reaches a rest site.
    #[test]
    fn path_summary_bounds_the_paths() {
        let map = ActMap::generate(7, Act::Overgrowth, Ascension(10));
        for p in map[map.start].children.iter() {
            let s = path_summary(&map, p);
            for t in 0..PATH_POINTS.len() {
                assert!(s[2 * t] <= s[2 * t + 1]);
            }
            assert!(s[MAP_FEATS - 4] < 2.0, "a rest site ahead");
            assert!(s[2 * 4 + 1] > 0.0, "monsters ahead");
        }
    }

    /// The map ahead is the map: every point the options reach sits at its
    /// row and column with its type and its links, the rest of the grid is
    /// empty, and each option names its own node.
    #[test]
    fn map_ahead_holds_the_reachable_map() {
        for act in ACTS {
            assert!(crate::map::rooms(act) <= MAP_ROWS, "{act:?} has more rows than the map segment");
        }
        let run = visible("SEEDA");
        let map = ActMap::generate(7, Act::Overgrowth, Ascension(10));
        let from = map.grid_points().find(|&p| map[p].row > 1 && map[p].children.len() > 1).expect("a fork");
        let points: Vec<PointId> = map[from].children.iter().collect();
        let base = map[from].row + 1;
        let obs = observe(&run, Decision::Path(&map, &points));
        let mut reached = points.clone();
        let mut k = 0;
        while k < reached.len() {
            for c in map[reached[k]].children.iter() {
                if map[c].kind != PointType::Boss && !reached.contains(&c) {
                    reached.push(c);
                }
            }
            k += 1;
        }
        let mut nodes = 0;
        for p in map.grid_points().filter(|&p| map[p].row >= base) {
            let node = I_MAP + ((map[p].row - base) * MAP_COLS + map[p].col) * MAP_NODE_IDS;
            if !reached.contains(&p) {
                assert_eq!(obs.ids[node], 0, "an unreachable point stays out");
                continue;
            }
            nodes += 1;
            assert_eq!(obs.ids[node], point_id(map[p].kind));
            for c in map[p].children.iter().filter(|&c| map[c].kind != PointType::Boss) {
                assert_ne!(obs.ids[node + 1] & 1 << (map[c].col + 1 - map[p].col), 0, "a link to each child");
            }
            assert_eq!(obs.ids[node + 1].count_ones() as usize, map[p].children.iter().filter(|&c| map[c].kind != PointType::Boss).count());
        }
        assert_eq!(obs.ids[I_MAP..].chunks(MAP_NODE_IDS).filter(|n| n[0] != 0).count(), nodes, "nothing else in the grid");
        for (k, &p) in points.iter().enumerate() {
            assert_eq!(obs.ids[I_OPTIONS + k * OPTION_IDS + OPTION_IDS - 1], map[p].col as i64 + 1);
        }
    }

    /// What a `?` holds is rolled as the player enters it, and the map
    /// shows only the `?`: two runs whose rolls for the next `?` differ
    /// encode the map step the same, with the `?` as `Unknown`.
    #[test]
    fn an_unknown_room_encodes_as_unknown() {
        let map = ActMap::generate(7, Act::Overgrowth, Ascension(10));
        let points: Vec<PointId> = map[map.start].children.iter().collect();
        let roll = |run: &RunState| run.clone().roll_unknown(false);
        let a = visible("SEEDA");
        let b = (0..).map(|i| visible(&format!("SEED{i}"))).find(|b| roll(b) != roll(&a)).expect("a seed that rolls otherwise");
        let (ea, eb) = (observe(&a, Decision::Path(&map, &points)), observe(&b, Decision::Path(&map, &points)));
        assert_eq!(ea, eb);
        let unknown = point_id(PointType::Unknown);
        assert!(ea.ids[I_MAP..].chunks(MAP_NODE_IDS).any(|n| n[0] == unknown), "the map ahead shows its ? rooms");
    }

    /// Takes a random option at every decision, leaning on one index per
    /// run so a long event (Slippery Bridge's holds, Tablet of Truth's
    /// deciphers) goes deep, and keeps every event option asked.
    struct Wander {
        rng: Rng,
        lean: usize,
        asked: std::collections::BTreeSet<(usize, &'static str, &'static str, &'static str)>,
    }

    impl crate::rooms::Chooser for Wander {
        fn choose(&mut self, _: &RunState, decision: Decision<'_>) -> usize {
            if let Decision::Event { event, options } = decision {
                let at = RUN_EVENTS.iter().position(|&e| e == event).unwrap_or(usize::MAX);
                self.asked.extend(options.iter().map(|o| (at, event, o.page, o.key)));
            }
            let n = option_count(decision).max(1);
            if self.rng.next_int(5) == 0 { self.rng.next_int(n) } else { self.lean.min(n - 1) }
        }
    }

    /// Every option an event asks has an id: each ported event walked many
    /// times over decks, gold, relics and potions that open and lock its
    /// options. A missing one fails with the lines to append.
    #[test]
    fn every_event_option_has_an_id() {
        let unlocks = Unlocks::default();
        let mut wander = Wander { rng: Rng::new(7), lean: 0, asked: Default::default() };
        for name in crate::events::ported() {
            for seed in 0..300 {
                let acts = select_acts(crate::game_rng::RunRngs::new("WANDER").seed, &unlocks);
                let mut run = RunState::new(&format!("WANDER{seed}"), acts, Ascension(10), &unlocks);
                run.enter_act(seed / 9 % 3);
                (run.max_hp, run.hp) = (300, 300 - (seed as i32 % 7) * 10);
                match seed % 3 {
                    0 => {
                        run.gold = 999;
                        run.deck.push(DeckCard { id: "INFLAME".into(), upgraded: false, enchantment: None });
                        run.deck.push(DeckCard { id: "BASH".into(), upgraded: true, enchantment: None });
                        run.deck.push(DeckCard { id: "IMPERVIOUS".into(), upgraded: false, enchantment: None });
                        run.relics.extend(["ANCHOR", "LANTERN", "VAJRA"].map(RunRelic::new));
                        for (slot, potion) in run.potions.iter_mut().zip(["FOUL_POTION", "FIRE_POTION", "FAIRY_IN_A_BOTTLE"]) {
                            *slot = Some(potion.into());
                        }
                    }
                    1 => run.gold = 0,
                    _ => {
                        // Nothing to enchant (Self-Help Book's NO_OPTIONS), and a
                        // potion so Stone of All Time has an open option.
                        run.gold = 60;
                        run.deck = vec![DeckCard { id: "CLUMSY".into(), upgraded: false, enchantment: None }];
                        run.potions[0] = Some("FIRE_POTION".into());
                    }
                }
                wander.lean = seed / 3 % 3;
                run.event(name, &mut wander, &mut Vec::new());
            }
        }
        let missing: Vec<String> = wander
            .asked
            .iter()
            .filter(|&&(_, event, page, key)| event_option_id(event, page, key) == 0)
            .map(|&(_, event, page, key)| format!("    (\"{event}\", \"{page}\", \"{key}\"),"))
            .collect();
        assert!(missing.is_empty(), "event options with no id; append to EVENT_OPTIONS:\n{}", missing.join("\n"));
    }

    /// Encodes the first event decision it is put, and takes option 0.
    struct Look(Option<RunObs>);

    impl crate::rooms::Chooser for Look {
        fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
            if matches!(decision, Decision::Event { .. }) && self.0.is_none() {
                self.0 = Some(observe(run, decision));
            }
            0
        }
    }

    /// Reflections' two options read as themselves: each its own id and
    /// name, Shatter duplicating the deck for a curse, Touch a Mirror
    /// changing cards the game picks.
    #[test]
    fn an_event_option_carries_its_id_and_what_it_shows() {
        let mut run = visible("SEEDA");
        let mut look = Look(None);
        run.event("Reflections", &mut look, &mut Vec::new());
        let obs = look.0.expect("an event decision");
        let id = |k: usize| obs.ids[I_OPTIONS + k * OPTION_IDS + 5 + OPTION_CARDS];
        assert_eq!([id(0), id(1)], [event_option_id("Reflections", "INITIAL", "TOUCH_A_MIRROR"), event_option_id("Reflections", "INITIAL", "SHATTER")]);
        assert!(id(0) > 0 && id(1) > 0 && id(0) != id(1));
        assert_eq!(obs.names, ["Reflections INITIAL.TOUCH_A_MIRROR", "Reflections INITIAL.SHATTER"]);
        let shown = |k: usize| &obs.floats[F_OPTIONS + (k + 1) * OPTION_FLOATS - EVENT_FEATS..F_OPTIONS + (k + 1) * OPTION_FLOATS];
        assert_eq!(shown(1)[18], run.deck.len() as f32 / 20.0, "Shatter duplicates the deck");
        assert_eq!((shown(1)[19], shown(1)[20]), (0.0, 0.5), "chosen by no one, one curse");
        assert_eq!((shown(0)[15], shown(0)[16], shown(0)[19]), (1.0, 0.5, 1.0), "four upgrades, two downgrades, at random");
    }

    #[test]
    fn run_vocabularies_cover_the_sim() {
        for name in crate::gen::INERT_RELICS {
            assert!(relic_id(name) > 0, "{name} needs appending to RUN_RELICS");
        }
        for name in crate::events::ported().chain(crate::ancients::ANCIENTS) {
            assert!(RUN_EVENTS.contains(&name), "{name} needs appending to RUN_EVENTS");
        }
        for e in crate::encounter::ALL.iter().filter(|e| e.kind() == crate::encounter::Kind::Boss) {
            assert!(RUN_BOSSES.contains(e), "{e:?} needs appending to RUN_BOSSES");
        }
    }
}
