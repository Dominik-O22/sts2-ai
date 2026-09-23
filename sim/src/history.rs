//! Real runs against the run layer: walks a run the game played floor by
//! floor, from its history file (`saves/history/*.run`), and checks what the
//! port offers against what the run recorded: each floor's room
//! (`map_point_history`), then everything drawn on the player's Rewards
//! stream: gold, potions, cards and relics after fights, shops, what relics
//! offered when picked up, treasure chests, what events and rest sites drew.
//! The player's recorded choices (relics taken, cards gained and removed,
//! potions kept) are applied as the walk goes. `examples/runcheck.rs` runs
//! it over a profile's history.
//!
//! The Rewards stream is shared by everything a run does, so the first floor
//! whose consumer is not ported ends the check of that stream; the rooms go
//! on, since they draw on their own streams.
//!
//! The map point a floor was entered from is not recorded, so the shop
//! blacklist of an unknown point (`RunManager.BuildRoomTypeBlacklist`) is
//! worked out from every path through the act's map that fits the point
//! types the run met.
use serde_json::Value;

use crate::encounter::{Act, Encounter};
use crate::map::{ActMap, PointId, PointType};
use crate::plan::Unlocks;
use crate::replay::slug;
use crate::effects::{Offered, RestOption};
use crate::rewards::{Offer, UNPORTED_RELICS};
use crate::run::{DeckCard, Enchant, Room, RoomType, RunState};
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
}

impl Report {
    /// A run built through the dev console fights what its plan does not
    /// hold.
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

/// The curses events add to the deck, which a grid check leaves out.
const CURSES: &[&str] = &["CLUMSY", "DOUBT", "REGRET", "SHAME", "INJURY", "POOR_SLEEP", "DECAY", "WRITHE", "NORMALITY"];

/// The ancients whose options are drawn on the event's own stream; what
/// the chosen relic does is `RunState::obtain`'s.
const ANCIENTS: &[&str] = &["NEOW", "DARV", "OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "TANX", "VAKUU"];

/// Walks `run`, which must be `eligible`.
pub fn check(run: &Value) -> Report {
    let seed = run["seed"].as_str().unwrap();
    let ascension = Ascension(run["ascension"].as_u64().unwrap() as u8);
    let acts: Vec<Act> = run["acts"].as_array().unwrap().iter().map(|a| act(a.as_str().unwrap())).collect();
    let acts: [Act; 3] = acts.try_into().expect("three acts");
    let mut state = RunState::new(seed, acts, ascension, &Unlocks::default());
    let mut report = Report { floors: 0, matched: 0, stop: "the run ended".into(), room_problem: None };
    let mut rewards_live = true;
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

            let room = state.enter_point(point, banned.first().copied().unwrap_or(false));
            let recorded = &rooms[0];
            let want = (recorded["room_type"].as_str().unwrap(), recorded["model_id"].as_str().map(str::to_string));
            let got = (room_type_name(room.kind()), model_id(room));
            if want != got && report.room_problem.is_none() {
                report.room_problem = Some(format!("{label}: {got:?}, the run met {want:?}"));
            }

            if rewards_live && last && run["win"] != true && stats["card_choices"].is_null() && matches!(room, Room::Combat(..)) {
                report.stop = format!("{label}: the run ended in this fight");
                rewards_live = false;
            }
            let followed = rewards_live && match follow(&mut state, room, rooms, stats) {
                Ok(()) => {
                    report.matched += 1;
                    true
                }
                Err(why) => {
                    report.stop = format!("{label}: {why}");
                    rewards_live = false;
                    false
                }
            };
            if !followed {
                // Past the Rewards check the relics still count for the rooms.
                for (relic, picked) in choices(stats, "relic_choices", "choice") {
                    if picked && !state.has_relic(&relic) {
                        state.obtain_relic(&relic);
                    }
                }
            }
            // The player as the floor left them, for the next floor's events.
            state.gold = stats["current_gold"].as_i64().unwrap() as i32;
            state.hp = stats["current_hp"].as_i64().unwrap() as i32;
            state.max_hp = stats["max_hp"].as_i64().unwrap() as i32;
            track_potions(&mut state, stats);
            track_deck(&mut state, stats);
        }
    }
    report
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

/// One floor on the Rewards stream: what the room drew against what the
/// run recorded, then the player's choices. An error says why the stream
/// can no longer be followed.
fn follow(state: &mut RunState, room: Room, rooms: &[Value], stats: &Value) -> Result<(), String> {
    if let Some(relic) = state.relics.iter().find(|r| UNPORTED_RELICS.contains(&r.id.as_str())) {
        return Err(format!("holds {}, not ported", relic.id));
    }
    if rooms.len() > 1 && !matches!(room, Room::Event(_)) {
        return Err(format!("a second room ({}) in the floor", rooms[1]["model_id"]));
    }
    let relic_picks: Vec<String> = choices(stats, "relic_choices", "choice").into_iter().filter(|c| c.1).map(|c| c.0).collect();
    let mut cards_offered: Vec<String> = Vec::new();
    let mut relics_offered: Vec<String> = Vec::new();
    let mut potions_offered: Vec<String> = Vec::new();
    let mut gold: Option<i32> = None;
    // Relics taken without a choice, which the record lists as picks too.
    let mut took: Vec<String> = Vec::new();
    // The card rewards of a fight, still open while the relics are taken.
    let mut reward_cards: Vec<Offer> = Vec::new();
    let mut event_fight = false;
    // Cards an event offered on a grid, which the record does not list:
    // only the ones taken show, as gains.
    let mut event_cards: Vec<String> = Vec::new();

    match room {
        Room::Combat(kind, encounter) => {
            if let Some(candy) = state.relic_mut("LASTING_CANDY") {
                candy.counter += 1;
            }
            // `GremlinMercNormal.CalculateGoldProportion`: the Fat Gremlin
            // fled with the gold the Merc stole, which the record shows as
            // gold stolen and none gained.
            let stolen = stats["gold_stolen"].as_i64().unwrap_or(0) > 0 && stats["gold_gained"].as_i64() == Some(0);
            let proportion = if encounter == Encounter::GremlinMercNormal && stolen { 0.0 } else { 1.0 };
            let rewards = state.combat_rewards(kind, proportion);
            gold = (!rewards.gold.is_empty()).then(|| rewards.gold.iter().sum());
            potions_offered.extend(rewards.potion.map(str::to_string));
            relics_offered.extend(rewards.relics.iter().map(|r| r.game_id()));
            reward_cards = rewards.cards.concat();
        }
        Room::Ancient(name) => {
            if !ANCIENTS.contains(&slug(name).as_str()) {
                return Err(format!("ancient {name} is not ported"));
            }
            let options = stats["ancient_choice"].as_array().into_iter().flatten();
            let options: Vec<String> = options.map(|o| o["TextKey"].as_str().unwrap().to_string()).collect();
            state.ancient_options(name, &options);
        }
        Room::Event(name) => {
            let taken = stats["event_choices"].as_array().into_iter().flatten();
            let taken: Vec<(String, String)> = taken
                .filter_map(|c| {
                    let key: Vec<&str> = c["title"]["key"].as_str()?.split('.').collect();
                    (key.len() == 6).then(|| (key[2].to_string(), key[4].to_string()))
                })
                .collect();
            for offered in state.event_option(name, &taken)? {
                match offered {
                    Offered::Cards(cards) => event_cards.extend(cards.iter().map(offer_text)),
                    Offered::Took(relics) => {
                        relics_offered.extend(relics.clone());
                        took.extend(relics);
                    }
                    Offered::Potions(potions) => potions_offered.extend(potions),
                    Offered::Unported(relic) if UNPORTED_RELICS.contains(&relic.as_str()) => return Err(format!("{relic} is not ported")),
                    Offered::Unported(_) | Offered::Pick(_) => {}
                    other => return Err(format!("event {name} offered {other:?}")),
                }
            }
            // A fight the event started, with a fight's rewards.
            if let Some(fight) = rooms.get(1) {
                if fight["model_id"] == "ENCOUNTER.FAKE_MERCHANT_EVENT_ENCOUNTER" {
                    return Err("the Fake Merchant's fight is not ported".into());
                }
                let kind = match fight["room_type"].as_str() {
                    Some("monster") => RoomType::Monster,
                    Some("elite") => RoomType::Elite,
                    _ => return Err(format!("event {name} led to {}", fight["room_type"])),
                };
                if let Some(candy) = state.relic_mut("LASTING_CANDY") {
                    candy.counter += 1;
                }
                let rewards = state.combat_rewards(kind, 1.0);
                gold = (!rewards.gold.is_empty()).then(|| rewards.gold.iter().sum());
                potions_offered.extend(rewards.potion.map(str::to_string));
                reward_cards = rewards.cards.concat();
                event_fight = true;
            }
        }
        Room::Treasure => {
            let (g, relic) = state.treasure();
            gold = Some(g);
            relics_offered.push(relic.game_id());
        }
        Room::Shop => {
            let shop = state.shop();
            cards_offered.extend(shop.cards.iter().chain(&shop.colorless).map(offer_text));
            relics_offered.extend(shop.relics.iter().map(|r| r.game_id()));
            potions_offered.extend(shop.potions.iter().map(|p| p.to_string()));
        }
        Room::RestSite => {
            let options: Vec<&str> = stats["rest_site_choices"].as_array().map_or(Vec::new(), |c| c.iter().filter_map(Value::as_str).collect());
            for option in options {
                let option = RestOption::from_id(option).ok_or_else(|| format!("rest site option {option} is not ported"))?;
                for offered in state.rest(option) {
                    match offered {
                        Offered::Potions(potions) => potions_offered.extend(potions),
                        Offered::Took(relics) => {
                            relics_offered.extend(relics.clone());
                            took.extend(relics);
                        }
                        Offered::Unported(relic) if UNPORTED_RELICS.contains(&relic.as_str()) => return Err(format!("{relic} is not ported")),
                        Offered::Unported(_) | Offered::Pick(_) => {}
                        other => return Err(format!("rest site offered {other:?}")),
                    }
                }
            }
        }
    }

    // The relics the player took, in the order they came, with what each
    // one's pickup offered.
    for relic in &relic_picks {
        if let Some(i) = took.iter().position(|t| t == relic) {
            took.remove(i);
            continue;
        }
        if UNPORTED_RELICS.contains(&relic.as_str()) {
            return Err(format!("{relic} is not ported"));
        }
        for offered in state.obtain(relic) {
            match offered {
                Offered::Cards(cards) | Offered::Gained(cards) => cards_offered.extend(cards.iter().map(offer_text)),
                Offered::Relics(relics) => relics_offered.extend(relics),
                Offered::Took(relics) => {
                    relics_offered.extend(relics.clone());
                    took.extend(relics);
                }
                Offered::Bundles(bundles) => cards_offered.extend(bundles.iter().flatten().map(offer_text)),
                Offered::Potions(potions) => potions_offered.extend(potions),
                Offered::Unported(relic) if UNPORTED_RELICS.contains(&relic.as_str()) => return Err(format!("{relic} is not ported")),
                Offered::Unported(_) | Offered::Pick(_) => {}
            }
        }
    }

    // A relic taken off the rewards screen before the card reward works on
    // the open reward (`CardReward.OnRelicObtained`), as an egg does.
    state.upgrade_by_eggs(&mut reward_cards);
    cards_offered.extend(reward_cards.iter().map(offer_text));

    // An event's own gifts (a named relic, a curse) are read off the record,
    // not drawn. What it drew is checked as far as the record shows it: the
    // cards taken off a grid, the relic pulled, the potions offered.
    if matches!(room, Room::Event(_)) && !event_fight {
        let gained = stats["cards_gained"].as_array().into_iter().flatten();
        let gained: Vec<String> = gained.map(|c| c["id"].as_str().unwrap().split_once('.').unwrap().1.to_string()).collect();
        let listed: Vec<String> = choices(stats, "card_choices", "card").into_iter().map(|c| c.0).collect();
        if !listed.is_empty() || !event_cards.is_empty() {
            // Card rewards list what they offered; a grid lists nothing, and
            // only its picks show among the gains (with any curse the event
            // adds).
            let fits = if listed.is_empty() {
                gained.iter().filter(|c| !CURSES.contains(&c.as_str())).all(|c| event_cards.contains(c))
            } else {
                same_set(listed.clone(), event_cards.clone())
            };
            if !fits {
                return Err(format!("cards {event_cards:?}, the run was offered {listed:?} and took {gained:?}"));
            }
        }
        let picked: Vec<String> = relic_picks.clone();
        if !relics_offered.iter().all(|r| picked.contains(r)) {
            return Err(format!("relics {relics_offered:?}, the run took {picked:?}"));
        }
        let want_potions: Vec<String> = choices(stats, "potion_choices", "choice").into_iter().map(|c| c.0).collect();
        if !potions_offered.is_empty() && !same_set(want_potions.clone(), potions_offered.clone()) {
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
        want_cards.extend(bought.map(|c| c["id"].as_str().unwrap().split_once('.').unwrap().1.to_string()));
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
    if !same_set(want_relics.clone(), relics_offered.clone()) {
        return Err(format!("relics {relics_offered:?}, the run was offered {want_relics:?}"));
    }
    // An ancient's potions come from relics that draw on the run's
    // CombatPotionGeneration stream (Phial Holster), not the Rewards one.
    let mut want_potions: Vec<String> = choices(stats, "potion_choices", "choice").into_iter().map(|c| c.0).collect();
    // Petrified Toad hands a Potion-Shaped Rock over as each fight starts,
    // which the record lists with the floor's potions.
    if state.has_relic("PETRIFIED_TOAD") && matches!(room, Room::Combat(..)) {
        if let Some(i) = want_potions.iter().position(|p| p == "POTION_SHAPED_ROCK") {
            want_potions.remove(i);
        }
    }
    if !ancient && !same_set(want_potions.clone(), potions_offered.clone()) {
        return Err(format!("potions {potions_offered:?}, the run was offered {want_potions:?}"));
    }
    if let Some(gold) = gold {
        let gained = stats["gold_gained"].as_i64().unwrap() as i32;
        // Lucky Fysh adds 15 gold for each card that joins the deck, the
        // reward's card among them if the Fysh came first.
        let cards = stats["cards_gained"].as_array().map_or(0, Vec::len) as i32;
        let fysh = state.has_relic("LUCKY_FYSH") && (gained - gold) % 15 == 0 && (0..=cards).contains(&((gained - gold) / 15));
        if gained != gold && !fysh {
            return Err(format!("gold {gold}, the run gained {gained}"));
        }
    }
    Ok(())
}

/// The deck after a floor: cards gained, removed, transformed, enchanted
/// and upgraded, as the record lists them.
fn track_deck(state: &mut RunState, stats: &Value) {
    let id = |v: &Value| v["id"].as_str().unwrap().split_once('.').unwrap().1.to_string();
    let remove = |state: &mut RunState, card: &str| {
        if let Some(i) = state.deck.iter().position(|c| c.id == card) {
            state.deck.remove(i);
        }
    };
    for card in stats["cards_removed"].as_array().into_iter().flatten() {
        remove(state, &id(card));
    }
    for t in stats["cards_transformed"].as_array().into_iter().flatten() {
        remove(state, &id(&t["original_card"]));
        state.deck.push(DeckCard::new(&id(&t["final_card"])));
    }
    for card in stats["cards_gained"].as_array().into_iter().flatten() {
        state.deck.push(DeckCard::new(&id(card)));
    }
    for e in stats["cards_enchanted"].as_array().into_iter().flatten() {
        let card = id(&e["card"]);
        let id = e["enchantment"].as_str().unwrap().split_once('.').unwrap().1.to_string();
        let amount = e["card"]["enchantment"]["amount"].as_i64().unwrap_or(0) as i32;
        if let Some(c) = state.deck.iter_mut().find(|c| c.id == card && c.enchantment.is_none()) {
            c.enchantment = Some(Enchant { id, amount });
        }
    }
    for relic in stats["relics_removed"].as_array().into_iter().flatten() {
        let relic = relic.as_str().unwrap().split_once('.').unwrap().1;
        state.relics.retain(|r| r.id != relic);
    }
    for card in stats["upgraded_cards"].as_array().into_iter().flatten() {
        let card = card.as_str().unwrap().split_once('.').unwrap().1;
        if let Some(c) = state.deck.iter_mut().find(|c| c.id == card && !c.upgraded) {
            c.upgraded = true;
        }
    }
}

/// The potions held after a floor: less the used and discarded ones, then
/// the ones kept from its choices.
fn track_potions(state: &mut RunState, stats: &Value) {
    for list in ["potion_used", "potion_discarded"] {
        for p in stats[list].as_array().into_iter().flatten() {
            let id = p.as_str().unwrap().split_once('.').unwrap().1;
            if let Some(slot) = state.potions.iter().position(|q| q.as_deref() == Some(id)) {
                state.potions[slot] = None;
            }
        }
    }
    for (id, picked) in choices(stats, "potion_choices", "choice") {
        if picked {
            match state.potions.iter().position(Option::is_none) {
                Some(slot) => state.potions[slot] = Some(id),
                None => state.potions.push(Some(id)),
            }
        }
    }
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
            let report = check(&run);
            assert_eq!(report.room_problem, None);
            assert_eq!(report.floors, floors);
            let won = run["win"] == true;
            assert_eq!(report.matched, if won { floors } else { floors - 1 }, "stopped at {}", report.stop);
        }
    }
}
