//! Real runs against the run layer: walks a run the game played floor by
//! floor, from its history file (`saves/history/*.run`), through the same
//! room flows the forward run takes (`rooms.rs`), with the player's
//! recorded choices as the `Chooser`. `examples/runcheck.rs` runs it over a
//! profile's history. Each floor is checked twice over:
//!
//! - The Rewards stream: what the port draws (gold, potions, cards and
//!   relics after fights, shops, what relics offered when picked up,
//!   treasure chests, what events and rest sites drew) against what the
//!   run recorded. The stream is shared by everything a run does, so the
//!   first floor whose consumer is not ported ends this check; the rooms go
//!   on, since they draw on their own streams.
//! - The effects (with `live`): the player as the port's effects leave
//!   them (HP, max HP, gold, deck, relics, potions) against the floor's
//!   `player_stats`. Fights are not simulated: what a fight did (damage,
//!   gold stolen, potions used, a card stolen) comes from the record, and
//!   so do HP and max HP after one, since the record does not split the
//!   fight's healing from the rewards'. A floor whose content is not
//!   ported (events, unported pickups) or whose draws follow a stream
//!   already lost is listed and not compared. After any floor not compared
//!   or diverging the player is set back to the record.
//!
//! Without `live` the player is set to the record after every floor, which
//! is all the stream check needs.
//!
//! The map point a floor was entered from is not recorded, so the shop
//! blacklist of an unknown point (`RunManager.BuildRoomTypeBlacklist`) is
//! worked out from every path through the act's map that fits the point
//! types the run met.
use serde_json::Value;

use crate::effects::{DeckAction, Offered, RestOption};
use crate::game_rng::RunStream;
use crate::encounter::{Act, Encounter};
use crate::map::{ActMap, PointId, PointType};
use crate::plan::Unlocks;
use crate::replay::slug;
use crate::rewards::{Offer, UNPORTED_RELICS};
use crate::rooms::{Chooser, Decision};
use crate::run::{DeckCard, Enchant, Room, RoomType, RunRelic, RunState};
use crate::shop::{Item, Ware};
use crate::types::Ascension;

/// Whether the port can walk the run: a standard singleplayer Ironclad run
/// on the pinned game version.
pub fn eligible(run: &Value) -> bool {
    let character = run["players"][0]["character"].as_str().unwrap_or("");
    let solo = run["players"].as_array().map_or(0, Vec::len) == 1;
    run["build_id"] == "v0.107.1" && run["game_mode"] == "standard" && character == "CHARACTER.IRONCLAD" && solo
}

/// What walking a run found.
pub struct Report {
    pub floors: usize,
    /// Floors walked with the Rewards stream still followed and everything
    /// on it matching.
    pub matched: usize,
    /// Why the Rewards check stopped: a mismatch, an unported consumer, or
    /// the run's end.
    pub stop: String,
    /// The first floor whose room is not the port's, if any.
    pub room_problem: Option<String>,
    /// The effects check, when asked for.
    pub effects: Effects,
    /// The ancients walked whose options the port laid out as the record
    /// has them, and each one that differs.
    pub ancients: usize,
    pub ancient_mismatches: Vec<String>,
}

/// The effects check of one run.
#[derive(Debug, Default)]
pub struct Effects {
    /// Floors compared with the record.
    pub checked: usize,
    /// Each compared floor that differed, and how.
    pub divergences: Vec<String>,
    /// Each floor not compared, and why.
    pub skipped: Vec<String>,
}

impl Report {
    /// A run built through the dev console fights what its plan does not
    /// hold; the walk stops at the first such fight.
    pub fn off_plan(&self) -> bool {
        self.room_problem.as_deref().is_some_and(|p| p.contains("ENCOUNTER."))
    }
}

fn act(id: &str) -> Act {
    match id {
        "ACT.OVERGROWTH" => Act::Overgrowth,
        "ACT.UNDERDOCKS" => Act::Underdocks,
        "ACT.HIVE" => Act::Hive,
        "ACT.GLORY" => Act::Glory,
        _ => panic!("unknown act {id}"),
    }
}

fn point_type(name: &str) -> PointType {
    match name {
        "ancient" => PointType::Ancient,
        "monster" => PointType::Monster,
        "elite" => PointType::Elite,
        "boss" => PointType::Boss,
        "unknown" => PointType::Unknown,
        "shop" => PointType::Shop,
        "treasure" => PointType::Treasure,
        "rest_site" => PointType::RestSite,
        _ => panic!("unknown map point type {name}"),
    }
}

/// The history's `room_type` for a room.
fn room_type_name(room: RoomType) -> &'static str {
    match room {
        RoomType::Monster => "monster",
        RoomType::Elite => "elite",
        RoomType::Boss => "boss",
        RoomType::Treasure => "treasure",
        RoomType::Shop => "shop",
        RoomType::Event => "event",
        RoomType::RestSite => "rest_site",
    }
}

/// The history's `model_id` for a room, if it has one.
fn model_id(room: Room) -> Option<String> {
    match room {
        Room::Combat(_, e) => Some(format!("ENCOUNTER.{}", slug(&format!("{e:?}")))),
        Room::Event(e) | Room::Ancient(e) => Some(format!("EVENT.{}", slug(e))),
        _ => None,
    }
}

/// Events not ported that draw nothing on the Rewards stream, so the
/// stream is followed past them.
const DRAWS_NOTHING: &[&str] = &["TinkerTime"];

/// The curses events add to the deck, which a grid check leaves out.
const CURSES: &[&str] = &["CLUMSY", "DOUBT", "REGRET", "SHAME", "INJURY", "POOR_SLEEP", "DECAY", "WRITHE", "NORMALITY"];

/// Walks `run`, which must be `eligible`; with `live`, checks the effects
/// too.
pub fn check(run: &Value, live: bool) -> Report {
    let seed = run["seed"].as_str().unwrap();
    let ascension = Ascension(run["ascension"].as_u64().unwrap() as u8);
    let acts: Vec<Act> = run["acts"].as_array().unwrap().iter().map(|a| act(a.as_str().unwrap())).collect();
    let acts: [Act; 3] = acts.try_into().expect("three acts");
    let mut state = RunState::new(seed, acts, ascension, &Unlocks::default());
    let mut player = Player::of(&state);
    let mut report = Report {
        floors: 0,
        matched: 0,
        stop: "the run ended".into(),
        room_problem: None,
        effects: Effects::default(),
        ancients: 0,
        ancient_mismatches: Vec::new(),
    };
    let mut rewards_live = true;
    // The run's Niche stream, which fights draw on once per enemy created:
    // lost after a fight whose monsters can add more.
    let mut niche_live = true;
    let mut previous_had_shop = false;
    let history = run["map_point_history"].as_array().unwrap();
    for (a, floors) in history.iter().enumerate() {
        state.enter_act(a);
        let map = ActMap::generate(state.rngs.seed, acts[a], ascension);
        let floors = floors.as_array().unwrap();
        let kinds: Vec<PointType> = floors.iter().map(|f| point_type(f["map_point_type"].as_str().unwrap())).collect();
        let paths = points_on_paths(&map, &kinds);
        for (j, floor) in floors.iter().enumerate() {
            report.floors += 1;
            let last = a + 1 == history.len() && j + 1 == floors.len();
            let label = format!("floor {} (act {} {})", report.floors, a + 1, floor["map_point_type"].as_str().unwrap());
            let point = kinds[j];
            let all_shops = |p: PointId| !map[p].children.is_empty() && map[p].children.iter().all(|c| map[c].kind == PointType::Shop);
            let banned: Vec<bool> = paths[j].iter().map(|&p| previous_had_shop || all_shops(p)).collect();
            if point == PointType::Unknown && banned.windows(2).any(|w| w[0] != w[1]) && report.room_problem.is_none() {
                report.room_problem = Some(format!("{label}: the paths that fit disagree on the shop blacklist"));
            }
            let stats = &floor["player_stats"][0];
            let rooms = floor["rooms"].as_array().unwrap();
            previous_had_shop = rooms.iter().any(|r| r["room_type"] == "shop");

            state.point = paths[j].first().copied();
            let room = state.enter_point(point, banned.first().copied().unwrap_or(false));
            let recorded = &rooms[0];
            let want = (recorded["room_type"].as_str().unwrap(), recorded["model_id"].as_str().map(str::to_string));
            let got = (room_type_name(room.kind()), model_id(room));
            if want != got && report.room_problem.is_none() {
                report.room_problem = Some(format!("{label}: {got:?}, the run met {want:?}"));
            }
            if report.off_plan() {
                // Built through the dev console from here: nothing after is
                // the plan's.
                report.floors -= 1;
                return report;
            }

            let ended = last && run["win"] != true && stats["card_choices"].is_null() && matches!(room, Room::Combat(..));
            if rewards_live && ended {
                report.stop = format!("{label}: the run ended in this fight");
                rewards_live = false;
            }
            let streamed = rewards_live;
            let counter = state.rewards().counter;
            let niche_before = (niche_live, state.rngs.run(RunStream::Niche).counter);
            let monsters = rooms.iter().filter_map(|r| r["monster_ids"].as_array()).flatten();
            if monsters.filter_map(Value::as_str).any(|m| SUMMONERS.contains(&game_id(m))) {
                niche_live = false;
            }
            let walked = (!ended).then(|| follow(&mut state, room, rooms, stats));
            match walked.as_ref().and_then(|w| w.ancient.as_ref()) {
                Some(Ok(())) => report.ancients += 1,
                Some(Err(why)) => report.ancient_mismatches.push(format!("{label}: {why}")),
                None => {}
            }
            if let (true, Some(walked)) = (streamed, &walked) {
                match &walked.stream {
                    Ok(()) => report.matched += 1,
                    Err(why) => {
                        report.stop = format!("{label}: {why}");
                        rewards_live = false;
                    }
                }
            }
            player.follow(stats);
            if !live {
                player.restore(&mut state);
                continue;
            }
            let skip = match &walked {
                None => Some("the run ended in this fight".to_string()),
                Some(w) if w.unported.is_some() => w.unported.clone(),
                _ if !streamed && state.rewards().counter != counter => Some("drew on the lost Rewards stream".into()),
                _ if !niche_before.0 && state.rngs.run(RunStream::Niche).counter != niche_before.1 + fought(rooms) => {
                    Some("drew on the lost Niche stream".into())
                }
                _ => None,
            };
            if let Some(why) = skip {
                report.effects.skipped.push(format!("{label}: {why}"));
                player.restore(&mut state);
                continue;
            }
            report.effects.checked += 1;
            if matches!(room, Room::Combat(..)) || rooms.len() > 1 {
                (state.hp, state.max_hp) = (player.hp, player.max_hp);
            }
            let missing = walked.map(|w| w.missing).unwrap_or_default();
            if let Some(diff) = player.differs(&state, &missing) {
                report.effects.divergences.push(format!("{label}: {diff}"));
                player.restore(&mut state);
            }
        }
    }
    report
}

/// The player as the record leaves them after each floor: the state the
/// effects check compares the port's with, and sets it back to.
struct Player {
    hp: i32,
    max_hp: i32,
    gold: i32,
    deck: Vec<DeckCard>,
    relics: Vec<String>,
    potions: Vec<String>,
}

impl Player {
    fn of(state: &RunState) -> Self {
        Player {
            hp: state.hp,
            max_hp: state.max_hp,
            gold: state.gold,
            deck: state.deck.clone(),
            relics: state.relics.iter().map(|r| r.id.clone()).collect(),
            potions: state.held_potions().map(str::to_string).collect(),
        }
    }

    /// A floor's `player_stats` applied.
    fn follow(&mut self, stats: &Value) {
        self.hp = stats["current_hp"].as_i64().unwrap() as i32;
        self.max_hp = stats["max_hp"].as_i64().unwrap() as i32;
        self.gold = stats["current_gold"].as_i64().unwrap() as i32;
        for (id, picked) in choices(stats, "relic_choices", "choice") {
            if picked {
                self.relics.push(id);
            }
        }
        for relic in stats["relics_removed"].as_array().into_iter().flatten() {
            let relic = game_id(relic.as_str().unwrap());
            if let Some(i) = self.relics.iter().position(|r| r == relic) {
                self.relics.remove(i);
            }
        }
        // A potion can be used before or after one of its kind is taken on
        // the same floor (Petrified Toad's rock), so what was not held
        // before is taken out after.
        let mut later = Vec::new();
        for potion in used_potions(stats) {
            match self.potions.iter().position(|p| *p == potion) {
                Some(i) => drop(self.potions.remove(i)),
                None => later.push(potion),
            }
        }
        self.potions.extend(choices(stats, "potion_choices", "choice").into_iter().filter(|c| c.1).map(|c| c.0));
        for potion in later {
            if let Some(i) = self.potions.iter().position(|p| *p == potion) {
                self.potions.remove(i);
            }
        }
        self.follow_deck(stats);
    }

    /// The deck after a floor: cards removed, transformed, gained,
    /// enchanted, downgraded and upgraded, as the record lists them.
    fn follow_deck(&mut self, stats: &Value) {
        let list = |key: &str| stats[key].as_array().cloned().unwrap_or_default();
        for card in list("cards_removed") {
            remove_card(&mut self.deck, &recorded_card(&card));
        }
        for t in list("cards_transformed") {
            remove_card(&mut self.deck, &recorded_card(&t["original_card"]));
            self.deck.push(recorded_card(&t["final_card"]));
        }
        for card in list("cards_gained") {
            self.deck.push(recorded_card(&card));
        }
        for e in list("cards_enchanted") {
            let card = recorded_card(&e["card"]);
            if let Some(c) = self.deck.iter_mut().find(|c| c.id == card.id && c.enchantment.is_none()) {
                c.enchantment = card.enchantment;
            }
        }
        // Downgrades first: an event that downgrades (Reflections) may
        // upgrade the same card again after.
        for card in list("downgraded_cards") {
            let card = game_id(card.as_str().unwrap());
            if let Some(c) = self.deck.iter_mut().find(|c| c.id == card && c.upgraded) {
                c.upgraded = false;
            }
        }
        for card in list("upgraded_cards") {
            let card = game_id(card.as_str().unwrap());
            if let Some(c) = self.deck.iter_mut().find(|c| c.id == card && !c.upgraded) {
                c.upgraded = true;
            }
        }
    }

    /// Sets the port's player to this one. Relics keep their counters where
    /// the port holds them, and a deck that holds the same cards stays as
    /// the port has it: the record does not say which copy of a card an
    /// upgrade went to, nor the order, which later random picks read.
    fn restore(&self, state: &mut RunState) {
        state.hp = self.hp;
        state.max_hp = self.max_hp;
        state.gold = self.gold;
        let (only_port, only_game) = difference(state.deck.iter().map(card_text).collect(), self.deck.iter().map(card_text).collect());
        if !only_port.is_empty() || !only_game.is_empty() {
            state.deck = self.deck.clone();
        }
        let mut held = std::mem::take(&mut state.relics);
        state.relics = self
            .relics
            .iter()
            .map(|id| match held.iter().position(|r| r.id == *id) {
                Some(i) => held.remove(i),
                None => RunRelic::new(id),
            })
            .collect();
        let slots = state.potions.len().max(self.potions.len());
        state.potions = self.potions.iter().cloned().map(Some).chain(std::iter::repeat(None)).take(slots).collect();
    }

    /// How the port's player differs from this one, if it does. `missing`
    /// are where the port did not offer what the record shows.
    fn differs(&self, state: &RunState, missing: &[String]) -> Option<String> {
        let mut diffs: Vec<String> = missing.to_vec();
        for (what, port, game) in [("hp", state.hp, self.hp), ("max hp", state.max_hp, self.max_hp), ("gold", state.gold, self.gold)] {
            if port != game {
                diffs.push(format!("{what} {port}, the run had {game}"));
            }
        }
        let (only_port, only_game) = difference(state.deck.iter().map(card_text).collect(), self.deck.iter().map(card_text).collect());
        if !only_port.is_empty() || !only_game.is_empty() {
            diffs.push(format!("deck has {only_port:?}, the run had {only_game:?}"));
        }
        let relics: Vec<String> = state.relics.iter().map(|r| r.id.clone()).collect();
        if relics != self.relics {
            let (only_port, only_game) = difference(relics, self.relics.clone());
            diffs.push(format!("relics {only_port:?}, the run had {only_game:?}"));
        }
        let (only_port, only_game) = difference(state.held_potions().map(str::to_string).collect(), self.potions.clone());
        if !only_port.is_empty() || !only_game.is_empty() {
            diffs.push(format!("potions {only_port:?}, the run had {only_game:?}"));
        }
        (!diffs.is_empty()).then(|| diffs.join("; "))
    }
}

/// A deck card as the effects check compares it: id, `+` if upgraded, the
/// enchantment and its amount.
fn card_text(c: &DeckCard) -> String {
    let ench = c.enchantment.as_ref().map_or(String::new(), |e| format!(" {}:{}", e.id, e.amount));
    format!("{}{}{ench}", c.id, if c.upgraded { "+" } else { "" })
}

/// What each of two lists holds that the other does not, as multisets.
fn difference(mut a: Vec<String>, mut b: Vec<String>) -> (Vec<String>, Vec<String>) {
    a.retain(|x| match b.iter().position(|y| y == x) {
        Some(i) => {
            b.remove(i);
            false
        }
        None => true,
    });
    (a, b)
}

/// `CATEGORY.ID` to `ID`.
fn game_id(id: &str) -> &str {
    id.split_once('.').unwrap().1
}

/// A card as the record serializes it (`SerializableCard`).
fn recorded_card(v: &Value) -> DeckCard {
    let enchantment = v["enchantment"]["id"].as_str().map(|id| Enchant {
        id: game_id(id).to_string(),
        amount: v["enchantment"]["amount"].as_i64().unwrap_or(0) as i32,
    });
    DeckCard { id: game_id(v["id"].as_str().unwrap()).to_string(), upgraded: v["current_upgrade_level"].as_i64().unwrap_or(0) > 0, enchantment }
}

/// Takes `card` out of `deck`: the same card if there is one, else one of
/// its id.
fn remove_card(deck: &mut Vec<DeckCard>, card: &DeckCard) {
    let i = deck.iter().position(|c| c == card).or_else(|| deck.iter().position(|c| c.id == card.id));
    if let Some(i) = i {
        deck.remove(i);
    }
}

/// The monsters that can add enemies to a fight (`CreatureCmd.Add`, and
/// the powers that call it) or hatch one (Tough Egg), each drawing on the
/// run's Niche stream where the record does not show it.
const SUMMONERS: &[&str] = &[
    "AXEBOT", "FABRICATOR", "FOGMOG", "GREMLIN_MERC", "LIVING_FOG", "OVICOPTER", "PHROG_PARASITE", "THE_OBSCURA", "TOUGH_EGG",
    "TWO_TAILED_RAT",
];

/// The enemies a floor's fights started with, as the record lists them.
fn fought(rooms: &[Value]) -> u32 {
    rooms.iter().filter_map(|r| r["monster_ids"].as_array()).map(|m| m.len() as u32).sum()
}

/// An ancient's option as the record keys it: the relic's id, but the
/// character for Orobas' Sea Glass.
fn ancient_option(option: &Value) -> String {
    match option["TextKey"].as_str().unwrap() {
        "IRONCLAD" | "SILENT" | "DEFECT" | "NECROBINDER" | "REGENT" => "SEA_GLASS".to_string(),
        relic => relic.to_string(),
    }
}

/// An `event_choices` title key as the page and key of the option chosen:
/// `EVENT.pages.PAGE.options.KEY.title`, or `RELIC.title` for an option a
/// relic titles, which has no page.
fn event_option(key: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = key.split('.').collect();
    match parts[..] {
        [_, "pages", page, "options", key, "title"] => Some((page.to_string(), key.to_string())),
        [relic, "title"] => Some((String::new(), relic.to_string())),
        _ => None,
    }
}

/// The potions a floor used or threw away.
fn used_potions(stats: &Value) -> Vec<String> {
    names(stats, "potion_used").into_iter().chain(names(stats, "potion_discarded")).collect()
}

/// The ids a list of `CATEGORY.ID` strings holds.
fn names(stats: &Value, key: &str) -> Vec<String> {
    stats[key].as_array().into_iter().flatten().map(|p| game_id(p.as_str().unwrap()).to_string()).collect()
}

/// The player's choices on one floor as its record lists them, taken as
/// the room flows ask.
struct Recorded {
    /// `cards_gained`, the cards taken from offers among them.
    cards: Vec<DeckCard>,
    /// `relic_choices` picked, in the order they came.
    relics: Vec<String>,
    /// `potion_choices` picked.
    potions: Vec<String>,
    rest: Vec<String>,
    upgraded: Vec<String>,
    removed: Vec<DeckCard>,
    /// `cards_transformed`, the originals.
    transformed: Vec<String>,
    enchanted: Vec<String>,
    ancient: Option<String>,
    /// `event_choices`, as each option's page and key (`event_option`).
    events: Vec<(String, String)>,
    /// Where the port did not offer what the record shows: a choice the
    /// record made, or an ancient's options.
    missing: Vec<String>,
}

impl Recorded {
    fn of(stats: &Value) -> Self {
        let list = |key: &str| stats[key].as_array().cloned().unwrap_or_default();
        let ids = |key: &str| list(key).iter().map(|v| game_id(v.as_str().unwrap()).to_string()).collect();
        Recorded {
            cards: list("cards_gained").iter().map(recorded_card).collect(),
            relics: choices(stats, "relic_choices", "choice").into_iter().filter(|c| c.1).map(|c| c.0).collect(),
            potions: choices(stats, "potion_choices", "choice").into_iter().filter(|c| c.1).map(|c| c.0).collect(),
            rest: list("rest_site_choices").iter().map(|v| v.as_str().unwrap().to_string()).collect(),
            upgraded: ids("upgraded_cards"),
            removed: list("cards_removed").iter().map(recorded_card).collect(),
            transformed: list("cards_transformed").iter().map(|t| recorded_card(&t["original_card"]).id).collect(),
            enchanted: list("cards_enchanted").iter().map(|e| recorded_card(&e["card"]).id).collect(),
            ancient: list("ancient_choice").iter().find(|o| o["was_chosen"] == true).map(ancient_option),
            events: list("event_choices").iter().filter_map(|c| event_option(c["title"]["key"].as_str()?)).collect(),
            missing: Vec::new(),
        }
    }

    /// Takes the first of `listed` that `matches` a card on offer: its
    /// index among `offered`, or `offered` past the end.
    fn take<T, U>(listed: &mut Vec<T>, offered: &[U], matches: impl Fn(&T, &U) -> bool) -> usize {
        for (l, item) in listed.iter().enumerate() {
            if let Some(i) = offered.iter().position(|o| matches(item, o)) {
                listed.remove(l);
                return i;
            }
        }
        offered.len()
    }
}

impl Chooser for Recorded {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
        match decision {
            Decision::Path(..) => 0,
            Decision::Card(offers) => Self::take(&mut self.cards, offers, |c, o| c.id == o.id),
            Decision::Bundle(bundles) => {
                let i = bundles.iter().position(|b| b.iter().all(|o| self.cards.iter().any(|c| c.id == o.id))).unwrap_or(bundles.len());
                for o in bundles.get(i).into_iter().flatten() {
                    Self::take(&mut self.cards, &[o], |c, o| c.id == o.id);
                }
                i
            }
            // The earliest in the record of the relics on offer.
            Decision::Relic(relics) => {
                let first = self.relics.iter().enumerate().find(|(_, r)| relics.contains(r)).map(|(l, _)| l);
                match first {
                    Some(l) => {
                        let relic = self.relics.remove(l);
                        relics.iter().position(|r| *r == relic).unwrap()
                    }
                    None => relics.len(),
                }
            }
            Decision::Potion(potion) => match self.potions.iter().position(|p| p == potion) {
                Some(i) => {
                    self.potions.remove(i);
                    0
                }
                None => 1,
            },
            Decision::Rest(options) => {
                if self.rest.is_empty() {
                    return options.len();
                }
                let id = self.rest.remove(0);
                match options.iter().position(|&o| RestOption::from_id(&id) == Some(o)) {
                    Some(i) => i,
                    None => {
                        self.missing.push(format!("not offered rest option {id}"));
                        options.len()
                    }
                }
            }
            Decision::Ancient(options) => options.iter().position(|o| Some(o) == self.ancient.as_ref()).unwrap_or(options.len()),
            // The next option the record chose; an option alone on its page
            // may be one the record does not keep (`ThatWontSaveToChoiceHistory`).
            Decision::Event { options, .. } => {
                let next = self.events.first().and_then(|(page, key)| options.iter().position(|o| o.page == page && o.key == key));
                match next {
                    Some(i) => {
                        self.events.remove(0);
                        i
                    }
                    None if options.len() == 1 => 0,
                    None => {
                        let shown: Vec<String> = options.iter().map(|o| format!("{}.{}", o.page, o.key)).collect();
                        self.missing.push(format!("chose {:?}, the port offered {shown:?}", self.events.first()));
                        0
                    }
                }
            }
            // What the record bought, relics first, then cards, potions and
            // the removal: the record does not keep the order, which only a
            // discount or an egg bought on the way would show.
            Decision::Shop(wares) => {
                let bought = |w: &Ware| match &w.item {
                    Item::Relic(r) => self.relics.contains(r),
                    Item::Card(o) => self.cards.iter().any(|c| c.id == o.id),
                    Item::Potion(p) => self.potions.iter().any(|q| q == p),
                    Item::Removal => !self.removed.is_empty(),
                };
                let rank = |w: &Ware| match w.item {
                    Item::Relic(_) => 0,
                    Item::Card(_) => 1,
                    Item::Potion(_) => 2,
                    Item::Removal => 3,
                };
                let Some(i) = (0..wares.len()).filter(|&i| bought(&wares[i])).min_by_key(|&i| rank(&wares[i])) else { return wares.len() };
                match &wares[i].item {
                    Item::Relic(r) => drop(Self::take(&mut self.relics, std::slice::from_ref(r), |a, b| a == b)),
                    Item::Card(o) => drop(Self::take(&mut self.cards, std::slice::from_ref(o), |c, o| c.id == o.id)),
                    Item::Potion(p) => drop(Self::take(&mut self.potions, std::slice::from_ref(p), |a, b| a == b)),
                    Item::Removal => {}
                }
                i
            }
            Decision::Deck { action, cards, .. } => {
                let deck = |i: &usize| &run.deck[*i];
                match action {
                    DeckAction::Upgrade => Self::take(&mut self.upgraded, cards, |id, i| deck(i).id == *id),
                    DeckAction::Remove => {
                        let exact = Self::take(&mut self.removed, cards, |c, i| deck(i) == c);
                        if exact < cards.len() {
                            return exact;
                        }
                        Self::take(&mut self.removed, cards, |c, i| deck(i).id == c.id)
                    }
                    DeckAction::Enchant(..) => Self::take(&mut self.enchanted, cards, |id, i| deck(i).id == *id),
                    DeckAction::Duplicate => Self::take(&mut self.cards, cards, |c, i| deck(i).id == c.id),
                    DeckAction::Transform { .. } | DeckAction::Maul | DeckAction::TransformInto(_) => Self::take(&mut self.transformed, cards, |id, i| deck(i).id == *id),
                }
            }
        }
    }
}

/// One floor walked.
struct Walked {
    /// Why the Rewards stream cannot be followed past this floor, if it
    /// cannot: a draw that differs from the record, or an unported
    /// consumer.
    stream: Result<(), String>,
    /// Why the floor's effects are not the port's, if they are not.
    unported: Option<String>,
    /// Where the port did not offer what the record shows.
    missing: Vec<String>,
    /// An ancient's options against the record's, if the floor had one.
    ancient: Option<Result<(), String>>,
}

/// One floor through the room flows with the player's recorded choices:
/// the fight's outcome and the player's own potion use first, from the
/// record, then the room. What it drew is then checked against the record.
fn follow(state: &mut RunState, room: Room, rooms: &[Value], stats: &Value) -> Walked {
    let mut chooser = Recorded::of(stats);
    let mut log: Vec<Offered> = Vec::new();
    let mut stream: Option<String> = None;
    let mut unported: Option<String> = None;
    if let Some(relic) = state.relics.iter().find(|r| UNPORTED_RELICS.contains(&r.id.as_str())) {
        stream = Some(format!("holds {}, not ported", relic.id));
    }
    if rooms.len() > 1 && !matches!(room, Room::Event(_)) {
        let why = format!("a second room ({}) in the floor", rooms[1]["model_id"]);
        return Walked { stream: Err(stream.unwrap_or(why.clone())), unported: Some(why), missing: Vec::new(), ancient: None };
    }

    if matches!(room, Room::Combat(..)) || rooms.len() > 1 {
        // Petrified Toad hands a Potion-Shaped Rock over as each fight
        // starts (`BeforeCombatStartLate`), which the record lists with the
        // floor's potions.
        if state.has_relic("PETRIFIED_TOAD") && state.add_potion("POTION_SHAPED_ROCK") {
            if let Some(i) = chooser.potions.iter().position(|p| p == "POTION_SHAPED_ROCK") {
                chooser.potions.remove(i);
            }
        }
    }
    if matches!(room, Room::Combat(..)) {
        state.enemies_created(fought(rooms) as usize);
        let lost = ["gold_stolen", "gold_lost"].iter().map(|k| stats[k].as_i64().unwrap_or(0) as i32).sum();
        state.lose_gold(lost);
        // A card a thief stole, less those it gave back on dying, which the
        // record lists as gained again from the floor they first came.
        let floor_of = |v: &Value| (v["id"].clone(), v["floor_added_to_deck"].clone());
        let mut returned: Vec<(Value, Value)> = stats["cards_gained"].as_array().into_iter().flatten().filter(|c| c["floor_added_to_deck"].is_i64()).map(floor_of).collect();
        for card in stats["cards_removed"].as_array().into_iter().flatten() {
            match returned.iter().position(|r| *r == floor_of(card)) {
                Some(i) => {
                    let back = recorded_card(card);
                    returned.remove(i);
                    if let Some(j) = chooser.cards.iter().position(|c| c.id == back.id) {
                        chooser.cards.remove(j);
                    }
                }
                None => remove_card(&mut state.deck, &recorded_card(card)),
            }
        }
        chooser.removed.clear();
    }
    // The player's potion use, from the record; what was not held yet is
    // taken out once the room is done.
    // The events that throw a potion away themselves (`PotionCmd.Discard`,
    // the Fake Merchant's Foul Potion) pick it off the potions held, so what
    // the record used and discarded waits.
    let mut used_later = Vec::new();
    let discards_own = matches!(room, Room::Event("RanwidTheElder" | "TheFutureOfPotions" | "StoneOfAllTime" | "FakeMerchant"));
    let used = if discards_own { names(stats, "potion_used") } else { used_potions(stats) };
    for potion in used {
        match state.potions.iter().position(|p| p.as_deref() == Some(potion.as_str())) {
            Some(slot) => state.potions[slot] = None,
            None => used_later.push(potion),
        }
    }

    let mut gold: Option<i32> = None;
    let mut event_cards: Vec<String> = Vec::new();
    let mut ancient = None;
    match room {
        Room::Combat(kind, encounter) => {
            // `GremlinMercNormal.CalculateGoldProportion`: the Fat Gremlin
            // fled with the gold the Merc stole, which the record shows as
            // gold stolen and none gained.
            let stolen = stats["gold_stolen"].as_i64().unwrap_or(0) > 0 && stats["gold_gained"].as_i64() == Some(0);
            let proportion = if encounter == Encounter::GremlinMercNormal && stolen { 0.0 } else { 1.0 };
            state.fight_won(kind);
            let rewards = state.combat_rewards(kind, proportion);
            gold = (!rewards.gold.is_empty()).then(|| rewards.gold.iter().sum());
            state.take_rewards(rewards, &mut chooser, &mut log);
        }
        Room::Ancient(name) => {
            let offer = state.ancient(name, &mut chooser, &mut log);
            let recorded: Vec<String> = stats["ancient_choice"].as_array().into_iter().flatten().map(ancient_option).collect();
            let why = format!("options {:?}, the run had {recorded:?}", offer.relics);
            if offer.relics != recorded {
                chooser.missing.push(why.clone());
            }
            ancient = Some(if offer.relics == recorded { Ok(()) } else { Err(why) });
        }
        Room::Event(name) => match state.event(name, &mut chooser, &mut log) {
            None => {
                unported = Some(format!("event {name}"));
                if !DRAWS_NOTHING.contains(&name) {
                    stream.get_or_insert(format!("event {name} is not ported"));
                }
            }
            Some(visit) => {
                // What the event itself offered; a fight's rewards follow.
                for offer in std::mem::take(&mut log) {
                    match offer {
                        Offered::Cards(cards) => event_cards.extend(cards.iter().map(offer_text)),
                        other => log.push(other),
                    }
                }
                match (visit.fight, rooms.get(1)) {
                    (Some(fight), Some(_)) => {
                        if !fight.created {
                            state.enemies_created(fought(rooms) as usize);
                        }
                        gold = state.event_fight_won(&fight, 1.0, &mut chooser, &mut log);
                    }
                    (Some(fight), None) => chooser.missing.push(format!("the port fought {:?}, the run did not", fight.encounter)),
                    (None, Some(room)) => chooser.missing.push(format!("the run fought {}, the port did not", room["model_id"])),
                    (None, None) => {}
                }
                // The relics the record took that the event did not give.
                let mut took: Vec<String> = log.iter().filter_map(|o| if let Offered::Took(r) = o { Some(r.clone()) } else { None }).flatten().collect();
                for relic in std::mem::take(&mut chooser.relics) {
                    match took.iter().position(|t| *t == relic) {
                        Some(i) => drop(took.remove(i)),
                        None => chooser.missing.push(format!("the run took {relic}")),
                    }
                }
            }
        },
        Room::Treasure => gold = Some(state.treasure_room(&mut chooser, &mut log)),
        Room::Shop => {
            state.shop_room(&mut chooser, &mut log);
            // What the record bought that the port did not sell or the
            // player could not pay for.
            let unbought = chooser.relics.drain(..).chain(chooser.cards.drain(..).map(|c| c.id)).chain(chooser.potions.drain(..));
            let unbought: Vec<String> = unbought.chain(chooser.removed.drain(..).map(|c| format!("removing {}", c.id))).collect();
            if !unbought.is_empty() {
                chooser.missing.push(format!("not bought {unbought:?}"));
            }
        }
        Room::RestSite => {
            match chooser.rest.iter().find(|o| RestOption::from_id(o).is_none()) {
                Some(option) => {
                    stream.get_or_insert(format!("rest site option {option} is not ported"));
                    unported = Some(format!("rest site option {option}"));
                }
                None => state.rest_site(&mut chooser, &mut log),
            }
        }
    }
    for potion in used_later {
        if let Some(slot) = state.potions.iter().position(|p| p.as_deref() == Some(potion.as_str())) {
            state.potions[slot] = None;
        }
    }
    for offer in &log {
        if let Offered::Unported(relic) = offer {
            if UNPORTED_RELICS.contains(&relic.as_str()) {
                stream.get_or_insert(format!("{relic} is not ported"));
            }
            unported.get_or_insert(format!("{relic} is not ported"));
        }
    }
    let stream = match stream {
        Some(why) => Err(why),
        None => drawn_as_recorded(state, room, stats, &log, gold, rooms.len() > 1, &event_cards),
    };
    Walked { stream, unported, missing: chooser.missing, ancient }
}

/// What a floor drew against what its record offered: the cards, relics,
/// potions and gold in `log`, the cards an event laid out in
/// `event_cards`.
fn drawn_as_recorded(
    state: &RunState,
    room: Room,
    stats: &Value,
    log: &[Offered],
    gold: Option<i32>,
    event_fight: bool,
    event_cards: &[String],
) -> Result<(), String> {
    let mut cards_offered: Vec<String> = Vec::new();
    let mut relics_offered: Vec<String> = Vec::new();
    let mut potions_offered: Vec<String> = Vec::new();
    for offer in log {
        match offer {
            Offered::Cards(cards) | Offered::Gained(cards) => cards_offered.extend(cards.iter().map(offer_text)),
            Offered::Bundles(bundles) => cards_offered.extend(bundles.iter().flatten().map(offer_text)),
            Offered::Relics(relics) | Offered::Took(relics) => relics_offered.extend(relics.iter().cloned()),
            Offered::Potions(potions) => potions_offered.extend(potions.iter().cloned()),
            Offered::Pick(_) | Offered::Unported(_) => {}
        }
    }
    let relic_picks: Vec<String> = choices(stats, "relic_choices", "choice").into_iter().filter(|c| c.1).map(|c| c.0).collect();
    // A record rebuilt from a run page that lists no potion offers
    // (`potions_unrecorded`): the potions drawn are taken as offered.
    let potions_recorded = stats["potions_unrecorded"] != true;

    // An event's own gifts (a named relic, a curse) are read off the record,
    // not drawn. What it drew is checked as far as the record shows it: the
    // cards taken off a grid, the relic pulled, the potions offered.
    if matches!(room, Room::Event(_)) && !event_fight {
        let gained = stats["cards_gained"].as_array().into_iter().flatten();
        let gained: Vec<String> = gained.map(|c| game_id(c["id"].as_str().unwrap()).to_string()).collect();
        let listed: Vec<String> = choices(stats, "card_choices", "card").into_iter().map(|c| c.0).collect();
        if !listed.is_empty() || !event_cards.is_empty() {
            // Card rewards list what they offered; a grid lists nothing, and
            // only its picks show among the gains (with any curse the event
            // adds).
            let fits = if listed.is_empty() {
                gained.iter().filter(|c| !CURSES.contains(&c.as_str())).all(|c| event_cards.contains(c))
            } else {
                same_set(listed.clone(), event_cards.to_vec())
            };
            if !fits {
                return Err(format!("cards {event_cards:?}, the run was offered {listed:?} and took {gained:?}"));
            }
        }
        if !relics_offered.iter().all(|r| relic_picks.contains(r)) {
            return Err(format!("relics {relics_offered:?}, the run took {relic_picks:?}"));
        }
        let want_potions: Vec<String> = choices(stats, "potion_choices", "choice").into_iter().map(|c| c.0).collect();
        if potions_recorded && !potions_offered.is_empty() && !same_set(want_potions.clone(), potions_offered.clone()) {
            return Err(format!("potions {potions_offered:?}, the run was offered {want_potions:?}"));
        }
        return if cards_offered.is_empty() { Ok(()) } else { Err(format!("a relic taken at the event offered {cards_offered:?}")) };
    }
    let ancient = matches!(room, Room::Ancient(_));
    let mut want_cards: Vec<String> = choices(stats, "card_choices", "card").into_iter().map(|c| c.0).collect();
    if matches!(room, Room::Shop) {
        // The record lists the cards left on the shelf; the ones bought are
        // the floor's gains.
        let bought = stats["cards_gained"].as_array().into_iter().flatten();
        want_cards.extend(bought.map(|c| game_id(c["id"].as_str().unwrap()).to_string()));
    }
    // A bundle screen records the bundle taken, not the ones left.
    let bundles = ancient && want_cards.len() < cards_offered.len();
    let fits = if bundles { want_cards.iter().all(|c| cards_offered.contains(c)) } else { same_set(want_cards.clone(), cards_offered.clone()) };
    if !fits {
        return Err(format!("cards {cards_offered:?}, the run was offered {want_cards:?} (gold {gold:?}, potions {potions_offered:?})"));
    }
    let mut want_relics: Vec<String> = choices(stats, "relic_choices", "choice").into_iter().map(|c| c.0).collect();
    if ancient && !want_relics.is_empty() {
        // The ancient's own option, drawn on the event's stream.
        want_relics.remove(0);
    }
    // A chest's relic left behind is not in the record; it is drawn on the
    // TreasureRoomRelics stream, not the Rewards one.
    let chest_left = matches!(room, Room::Treasure) && want_relics.is_empty();
    if !chest_left && !same_set(want_relics.clone(), relics_offered.clone()) {
        return Err(format!("relics {relics_offered:?}, the run was offered {want_relics:?}"));
    }
    // An ancient's potions come from relics that draw on the run's
    // CombatPotionGeneration stream (Phial Holster), not the Rewards one.
    let mut want_potions: Vec<String> = choices(stats, "potion_choices", "choice").into_iter().map(|c| c.0).collect();
    // Petrified Toad's rock, handed over as the fight started.
    if state.has_relic("PETRIFIED_TOAD") && matches!(room, Room::Combat(..)) {
        if let Some(i) = want_potions.iter().position(|p| p == "POTION_SHAPED_ROCK") {
            want_potions.remove(i);
        }
    }
    if potions_recorded && !ancient && !same_set(want_potions.clone(), potions_offered.clone()) {
        return Err(format!("potions {potions_offered:?}, the run was offered {want_potions:?}"));
    }
    if let Some(gold) = gold {
        let gained = stats["gold_gained"].as_i64().unwrap() as i32;
        // Lucky Fysh adds 15 gold for each card that joins the deck, the
        // reward's card among them if the Fysh came first.
        let cards = stats["cards_gained"].as_array().map_or(0, Vec::len) as i32;
        let fysh = state.has_relic("LUCKY_FYSH") && (gained - gold) % 15 == 0 && (0..=cards).contains(&((gained - gold) / 15));
        // Bowler Hat adds a quarter, truncated (`RunState::gain_gold`), unless
        // the gold was taken before the Hat on the same screen.
        let hat = state.has_relic("BOWLER_HAT") && gained == (f64::from(gold) * 1.25) as i32;
        if gained != gold && !fysh && !hat {
            return Err(format!("gold {gold}, the run gained {gained}"));
        }
    }
    Ok(())
}

/// For each floor of an act, the map points it can have been: those on a
/// path from the start whose points have the types the run met, floor by
/// floor, to the last floor it reached.
fn points_on_paths(map: &ActMap, kinds: &[PointType]) -> Vec<Vec<PointId>> {
    let mut forward: Vec<Vec<PointId>> = Vec::new();
    for (j, &kind) in kinds.iter().enumerate() {
        let here: Vec<PointId> = if j == 0 {
            vec![map.start]
        } else {
            let mut next: Vec<PointId> = Vec::new();
            for &p in &forward[j - 1] {
                for c in map[p].children.iter().filter(|&c| map[c].kind == kind) {
                    if !next.contains(&c) {
                        next.push(c);
                    }
                }
            }
            next
        };
        forward.push(here);
    }
    for j in (0..kinds.len().saturating_sub(1)).rev() {
        let (now, later) = forward.split_at_mut(j + 1);
        now[j].retain(|&p| later[0].iter().any(|&c| map[p].children.contains(c)));
    }
    forward
}

/// The ids a history list holds under `key`, with `was_picked`.
fn choices(stats: &Value, list: &str, key: &str) -> Vec<(String, bool)> {
    stats[list]
        .as_array()
        .map(|l| {
            l.iter()
                .map(|c| {
                    let id = if key == "card" { c["card"]["id"].as_str() } else { c[key].as_str() };
                    let id = id.unwrap().split_once('.').unwrap().1.to_string();
                    let up = key == "card" && c["card"]["current_upgrade_level"].as_i64().unwrap_or(0) > 0;
                    (if up { format!("{id}+") } else { id }, c["was_picked"].as_bool().unwrap_or(false))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn offer_text(o: &Offer) -> String {
    if o.upgraded { format!("{}+", o.id) } else { o.id.to_string() }
}

fn same_set(mut a: Vec<String>, mut b: Vec<String>) -> bool {
    a.sort();
    b.sort();
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two runs from the modded profile: a win at A10 and a death in the
    /// act 3 boss fight, between them shops, treasure, ancients (Neow with
    /// Hefty Tablet, Tezcatara, Darv, Vakuu), and events that draw (Room
    /// Full of Cheese, Welcome to Wongo's, Dense Vegetation's fight).
    #[test]
    fn matches_real_runs() {
        for (text, floors) in [(include_str!("../testdata/run-TBL5VNYN4M.run"), 49), (include_str!("../testdata/run-5J5VMZX7UB.run"), 49)] {
            let run: Value = serde_json::from_str(text).unwrap();
            assert!(eligible(&run));
            let report = check(&run, false);
            assert_eq!(report.room_problem, None);
            assert_eq!(report.floors, floors);
            let won = run["win"] == true;
            assert_eq!(report.matched, if won { floors } else { floors - 1 }, "stopped at {}", report.stop);
        }
    }

    /// The same runs with the effects live: every floor leaves the player
    /// as the record has them (rest heals, Meal Ticket, Frozen Egg's
    /// upgrades, Potion Belt, Petrified Toad's rocks, a thief's card given
    /// back, shops bought from at the port's prices, the ancients' relics,
    /// the events' options, Dense Vegetation's fight), but the fight that
    /// ended the run and one that drew on the Niche stream after a fight
    /// with summoners had lost it.
    #[test]
    fn effects_match_real_runs() {
        for (text, compared) in [(include_str!("../testdata/run-TBL5VNYN4M.run"), 49), (include_str!("../testdata/run-5J5VMZX7UB.run"), 47)] {
            let run: Value = serde_json::from_str(text).unwrap();
            let effects = check(&run, true).effects;
            assert_eq!(effects.divergences, Vec::<String>::new());
            assert_eq!(effects.checked, compared, "not compared: {:?}", effects.skipped);
            assert_eq!(check(&run, true).matched, check(&run, false).matched);
        }
    }
}
