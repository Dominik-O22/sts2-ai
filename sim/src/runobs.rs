//! The run policy's observation (docs/run-env.md, Observation and scoring):
//! a run decision as tokens. One global token (HP, gold, floor, act, the
//! act's bosses, the decision, the odds a player can count), one per deck
//! card, relic and potion, and one per option, each option carrying what it
//! names (a card, relic, potion, map point, event option) and, for a map
//! step, what the paths through the point hold. Every segment has a fixed
//! length and a presence flag per token, so a batch is one array.
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
use crate::encounter::{Act, Encounter};
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
/// Event options by a hash of event, page and key, which needs no
/// vocabulary to stay stable as events are ported.
pub const EVENT_KEY_BUCKETS: usize = 1024;

/// Ids per token of each segment, then floats. Each segment's first float
/// is its presence flag, except the global token's, which is always there.
pub const GLOBAL_IDS: usize = 6;
pub const GLOBAL_FLOATS: usize = 15;
pub const DECK_IDS: usize = 2;
pub const DECK_FLOATS: usize = 3;
pub const RELIC_IDS: usize = 1;
pub const RELIC_FLOATS: usize = 3;
pub const POTION_IDS: usize = 1;
pub const POTION_FLOATS: usize = 1;
/// kind, cards (`OPTION_CARDS`), enchantment, relic, potion, room, event key.
pub const OPTION_IDS: usize = 5 + OPTION_CARDS + 1;
/// present, upgraded per card, enchantment amount, price, the map summary.
pub const OPTION_FLOATS: usize = 3 + OPTION_CARDS + MAP_FEATS;
/// Per map point type a path can pass (`PATH_POINTS`), the fewest and the
/// most on paths through a point; then the rows to the nearest rest site
/// and shop.
pub const MAP_FEATS: usize = 2 * PATH_POINTS.len() + 2;

pub const F_DECK: usize = GLOBAL_FLOATS;
pub const F_RELICS: usize = F_DECK + MAX_DECK * DECK_FLOATS;
pub const F_POTIONS: usize = F_RELICS + MAX_RELICS * RELIC_FLOATS;
pub const F_OPTIONS: usize = F_POTIONS + MAX_POTIONS * POTION_FLOATS;
pub const RUN_FLOATS: usize = F_OPTIONS + MAX_OPTIONS * OPTION_FLOATS;
pub const I_DECK: usize = GLOBAL_IDS;
pub const I_RELICS: usize = I_DECK + MAX_DECK * DECK_IDS;
pub const I_POTIONS: usize = I_RELICS + MAX_RELICS * RELIC_IDS;
pub const I_OPTIONS: usize = I_POTIONS + MAX_POTIONS * POTION_IDS;
pub const RUN_IDS: usize = I_OPTIONS + MAX_OPTIONS * OPTION_IDS;

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
pub const EVENT_KEY_VOCAB: usize = EVENT_KEY_BUCKETS + 1;

/// A run decision encoded: one row of the run buffers, and the chooser's
/// answer for each option token (the index `rooms::Decision` reads, where a
/// skip is past the end).
#[derive(Clone, Debug, PartialEq)]
pub struct RunObs {
    pub floats: Vec<f32>,
    pub ids: Vec<i64>,
    pub answers: Vec<usize>,
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

/// FNV-1a of the option's event, page and key, into `EVENT_KEY_BUCKETS`.
fn event_key(event: &str, page: &str, key: &str) -> i64 {
    let mut h: u32 = 0x811C_9DC5;
    for b in event.bytes().chain(*b".").chain(page.bytes()).chain(*b".").chain(key.bytes()) {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    (h as usize % EVENT_KEY_BUCKETS) as i64 + 1
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
    event_key: i64,
    price: i32,
    map: [f32; MAP_FEATS],
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
/// is 0). Normalized by 8; none reachable reads 2.
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
    out[MAP_FEATS - 2] = far(s.rest);
    out[MAP_FEATS - 1] = far(s.shop);
    out
}

#[derive(Clone, Copy)]
struct Summary {
    min: [u32; PATH_POINTS.len()],
    max: [u32; PATH_POINTS.len()],
    rest: u32,
    shop: u32,
}

fn summarize(map: &ActMap, point: PointId, memo: &mut [Option<Summary>]) -> Summary {
    if let Some(s) = memo[point.index()] {
        return s;
    }
    let children: Vec<PointId> = map[point].children.iter().collect();
    let mut s = if children.is_empty() {
        Summary { min: [0; PATH_POINTS.len()], max: [0; PATH_POINTS.len()], rest: u32::MAX, shop: u32::MAX }
    } else {
        let mut acc = Summary { min: [u32::MAX; PATH_POINTS.len()], max: [0; PATH_POINTS.len()], rest: u32::MAX, shop: u32::MAX };
        for c in children {
            let cs = summarize(map, c, memo);
            for t in 0..PATH_POINTS.len() {
                acc.min[t] = acc.min[t].min(cs.min[t]);
                acc.max[t] = acc.max[t].max(cs.max[t]);
            }
            acc.rest = acc.rest.min(cs.rest.saturating_add(1));
            acc.shop = acc.shop.min(cs.shop.saturating_add(1));
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
    }
    if kind == PointType::Shop {
        s.shop = 0;
    }
    memo[point.index()] = Some(s);
    s
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
                .map(|(i, &p)| Opt { room: point_id(map[p].kind), map: path_summary(map, p), ..Opt::new(K::Path, i) })
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
                    let mut opt = Opt { event_key: event_key(event, o.page, o.key), ..Opt::new(K::Event, i) };
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
    f[..GLOBAL_FLOATS].copy_from_slice(&[
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

    // A skip is kept when the list is cut short: it is the one way out of
    // some decisions.
    let mut opts = opts;
    if opts.len() > MAX_OPTIONS {
        let last = opts.pop().expect("options");
        opts.truncate(MAX_OPTIONS - 1);
        opts.push(last);
    }
    let answers = opts.iter().map(|o| o.answer).collect();
    for (k, o) in opts.iter().enumerate() {
        let (i, x) = (I_OPTIONS + k * OPTION_IDS, F_OPTIONS + k * OPTION_FLOATS);
        let row = &mut ids[i..i + OPTION_IDS];
        row[0] = o.kind;
        for (c, &(id, _)) in o.cards.iter().enumerate() {
            row[1 + c] = id;
        }
        row[1 + OPTION_CARDS..].copy_from_slice(&[o.enchant.0, o.relic, o.potion, o.room, o.event_key]);
        let row = &mut f[x..x + OPTION_FLOATS];
        row[0] = 1.0;
        for (c, &(_, up)) in o.cards.iter().enumerate() {
            row[1 + c] = up as u8 as f32;
        }
        row[1 + OPTION_CARDS] = o.enchant.1 as f32 / 10.0;
        row[2 + OPTION_CARDS] = o.price as f32 / 100.0;
        row[3 + OPTION_CARDS..].copy_from_slice(&o.map);
    }
    RunObs { floats: f, ids, answers }
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
            assert!(s[MAP_FEATS - 2] < 2.0, "a rest site ahead");
            assert!(s[2 * 4 + 1] > 0.0, "monsters ahead");
        }
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
