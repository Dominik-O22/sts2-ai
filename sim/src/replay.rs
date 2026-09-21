//! Replay harness. Reads a recording written by the `mod/` recorder (one
//! JSON record per line), rebuilds the combat in the sim with the recorded
//! shuffles, enemy HP, and monster moves forced, applies the recorded
//! actions, and diffs the sim against every recorded decision point.
//!
//! Model ids are the game's: the class name in screaming snake case, so
//! `StrikeIronclad` is `STRIKE_IRONCLAD` and `StrengthPower` is
//! `STRENGTH_POWER`.

use std::collections::{HashMap, VecDeque};

use serde_json::{json, Value};

use crate::card::Card;
use crate::combat::{Action, Combat, EnemySpec, Outcome, RoomKind, Script, Setup};
use crate::encounter::Encounter;
use crate::ids::{CardId, MonsterId, ALL_CARDS, ALL_MONSTERS};
use crate::potion::PotionId;
use crate::relic::{Relic, RelicId};
use crate::rng::Rng;
use crate::types::{Ascension, CreatureRef};

/// `StringHelper.Slugify` on a Rust enum variant name.
pub fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

fn table<T: Copy + std::fmt::Debug>(all: &[T], suffix: &str) -> HashMap<String, T> {
    all.iter().map(|v| (format!("{}{suffix}", slug(&format!("{v:?}"))), *v)).collect()
}

/// Lookups from game id strings to sim ids.
pub struct Ids {
    cards: HashMap<String, CardId>,
    monsters: HashMap<String, MonsterId>,
    relics: HashMap<String, RelicId>,
    potions: HashMap<String, PotionId>,
    encounters: HashMap<String, Encounter>,
}

impl Ids {
    pub fn new() -> Self {
        Self {
            cards: table(ALL_CARDS, ""),
            monsters: table(ALL_MONSTERS, ""),
            relics: table(crate::relic::ALL, ""),
            potions: table(crate::potion::ALL, ""),
            encounters: table(crate::encounter::ALL, ""),
        }
    }
}

impl Default for Ids {
    fn default() -> Self {
        Self::new()
    }
}

/// What a replay found.
#[derive(Debug)]
pub struct Report {
    /// Recorded decision points that matched.
    pub snapshots: usize,
    /// Actions applied.
    pub actions: usize,
    /// First mismatch, with the record index it happened at.
    pub divergence: Option<(usize, String)>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.divergence.is_none()
    }
}

/// Sort JSON values by their text, the one order both sides of a diff use.
fn sort_values(mut v: Vec<Value>) -> Vec<Value> {
    v.sort_by_key(|x| x.to_string());
    v
}

/// The sim's view of a combat in the recorder's snapshot shape, piles in
/// their real order. `diff` normalizes both sides.
pub fn snapshot_of(c: &Combat) -> Value {
    let card = |k: &Card| json!({ "id": slug(&format!("{:?}", k.id)), "up": k.upgraded });
    let hand: Vec<Value> =
        c.player.hand.iter().map(|k| json!({ "id": slug(&format!("{:?}", k.id)), "up": k.upgraded, "cost": c.cost(k) })).collect();
    let pile = |cards: &[Card]| cards.iter().map(card).collect::<Vec<_>>();
    let powers = |r: CreatureRef| {
        c.creature(r).powers.iter().map(|p| json!([format!("{}_POWER", slug(&format!("{:?}", p.id))), p.amount])).collect::<Vec<_>>()
    };
    json!({
        "t": "snapshot",
        "turn": c.player.turn,
        "round": c.round,
        "hp": c.player.creature.hp,
        "max_hp": c.player.creature.max_hp,
        "block": c.player.creature.block,
        "energy": c.player.energy,
        "powers": powers(CreatureRef::Player),
        "hand": hand,
        "draw": c.player.draw.iter().map(card).collect::<Vec<_>>(),
        "discard": pile(&c.player.discard),
        "exhaust": pile(&c.player.exhaust),
        "potions": c.potions.iter().map(|p| p.map(|p| slug(&format!("{p:?}")))).collect::<Vec<_>>(),
        // The game drops dead creatures from its enemy list.
        "enemies": c.enemies.iter().enumerate().filter(|(_, e)| e.creature.alive()).map(|(i, e)| json!({
            "id": slug(&format!("{:?}", e.monster.id)),
            "hp": e.creature.hp,
            "max_hp": e.creature.max_hp,
            "block": e.creature.block,
            "powers": powers(CreatureRef::Enemy(i)),
            "move": e.monster.next_move_name(),
        })).collect::<Vec<_>>(),
    })
}

/// A snapshot with the order-free collections sorted: hand, discard,
/// exhaust, powers, and potion slots do not affect play by their order.
fn normalize(rec: &Value) -> Value {
    let mut v = rec.clone();
    let sort_arr = |a: &mut Value| {
        if let Some(arr) = a.as_array_mut() {
            *arr = sort_values(std::mem::take(arr));
        }
    };
    for key in ["hand", "discard", "exhaust", "powers", "potions"] {
        if let Some(a) = v.get_mut(key) {
            sort_arr(a);
        }
    }
    if let Some(enemies) = v.get_mut("enemies").and_then(Value::as_array_mut) {
        for e in enemies {
            if let Some(a) = e.get_mut("powers") {
                sort_arr(a);
            }
        }
    }
    v
}

/// First differing field between a recorded snapshot and the sim, if any.
fn diff(recorded: &Value, sim: &Value) -> Option<String> {
    let rec = normalize(recorded);
    let sim = &normalize(sim);
    for key in ["turn", "round", "hp", "max_hp", "block", "energy", "powers", "hand", "draw", "discard", "exhaust", "potions"] {
        if rec.get(key) != sim.get(key) {
            return Some(format!("{key}: game {} vs sim {}", rec.get(key).unwrap_or(&Value::Null), sim.get(key).unwrap_or(&Value::Null)));
        }
    }
    let empty = vec![];
    let re = rec["enemies"].as_array().unwrap_or(&empty);
    let se = sim["enemies"].as_array().unwrap_or(&empty);
    if re.len() != se.len() {
        return Some(format!("enemy count: game {} vs sim {}", re.len(), se.len()));
    }
    for (i, (r, s)) in re.iter().zip(se).enumerate() {
        for key in ["id", "hp", "max_hp", "block", "powers"] {
            if r.get(key) != s.get(key) {
                return Some(format!("enemy {i} {key}: game {} vs sim {}", r[key], s[key]));
            }
        }
    }
    None
}

fn card_ref(ids: &Ids, v: &Value) -> Result<(CardId, bool), String> {
    let id = v["id"].as_str().ok_or("card without id")?;
    let card = *ids.cards.get(id).ok_or_else(|| format!("unknown card {id}"))?;
    Ok((card, v["up"].as_bool().unwrap_or(false)))
}

/// Enemy specs for the recorded monster list: re-roll the encounter's own
/// composition until it matches, so positional flags come out right.
fn specs_for(enc: Encounter, monsters: &[MonsterId]) -> Vec<EnemySpec> {
    for seed in 0..2000u64 {
        let specs = enc.monsters(&mut Rng::new(seed));
        if specs.iter().map(|s| s.id).eq(monsters.iter().copied()) {
            return specs;
        }
    }
    monsters.iter().map(|&id| EnemySpec { id, flags: Default::default() }).collect()
}

/// After the sim has no pending choice, compare against `snap`. With a
/// pending choice, try every answer and keep the first whose result
/// matches: the recording does not say which card was picked.
fn settle(c: &Combat, snap: &Value, depth: u32) -> Result<Combat, String> {
    if c.pending.is_none() {
        return match diff(snap, &snapshot_of(c)) {
            None => Ok(c.clone()),
            Some(d) => Err(d),
        };
    }
    if depth > 12 {
        return Err("choice chain too deep".into());
    }
    let mut first_err = None;
    for a in c.legal_actions() {
        let mut k = c.clone();
        k.step(a);
        match settle(&k, snap, depth + 1) {
            Ok(k) => return Ok(k),
            Err(e) => first_err.get_or_insert(e),
        };
    }
    Err(first_err.map_or("no choice matched".into(), |e| format!("no choice matched; first branch: {e}")))
}

/// Force the recorded next moves onto the sim's enemies.
fn force_moves(c: &mut Combat, snap: &Value) -> Result<(), String> {
    let empty = vec![];
    let living: Vec<usize> = c.living_enemies().collect();
    for (e, &i) in snap["enemies"].as_array().unwrap_or(&empty).iter().zip(&living) {
        let Some(name) = e["move"].as_str() else { continue };
        if !c.set_enemy_move(i, name) {
            return Err(format!("enemy {i} has no move {name}"));
        }
    }
    Ok(())
}

fn failed(report: &Report, n: usize, msg: String) -> Report {
    Report { snapshots: report.snapshots, actions: report.actions, divergence: Some((n, msg)) }
}

/// Replay one recording. `text` is the JSONL file contents. Outcomes the
/// script does not force yet (random targets, random card picks and
/// generation) come from the sim's own RNG, so recordings that exercise
/// them can diverge for that reason alone.
pub fn replay(text: &str, ids: &Ids) -> Result<Report, String> {
    replay_seeded(text, ids, 0)
}

/// `replay` with a chosen seed for the unscripted RNG streams.
pub fn replay_seeded(text: &str, ids: &Ids, seed: u64) -> Result<Report, String> {
    let records: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("bad json: {e}")))
        .collect::<Result<_, _>>()?;
    let start = records.first().filter(|r| r["t"] == "start").ok_or("no start record")?;
    let first_snap = records.iter().find(|r| r["t"] == "snapshot").ok_or("no snapshot")?;

    let deck: Vec<Card> = start["deck"]
        .as_array()
        .ok_or("start without deck")?
        .iter()
        .map(|v| card_ref(ids, v).map(|(id, up)| Card::new(0, id, up)))
        .collect::<Result<_, _>>()?;
    let relics: Vec<Relic> = start["relics"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| ids.relics.get(v.as_str()?)).map(|&id| Relic::new(id)).collect())
        .unwrap_or_default();
    let potions: Vec<Option<PotionId>> = start["potions"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().and_then(|s| ids.potions.get(s).copied())).collect())
        .unwrap_or_default();
    let enc_name = start["encounter"].as_str().unwrap_or("");
    let enc = *ids.encounters.get(enc_name).ok_or_else(|| format!("unknown encounter {enc_name}"))?;
    let monsters: Vec<MonsterId> = start["enemies"]
        .as_array()
        .ok_or("start without enemies")?
        .iter()
        .map(|e| {
            let id = e["id"].as_str().unwrap_or("");
            ids.monsters.get(id).copied().ok_or_else(|| format!("unknown monster {id}"))
        })
        .collect::<Result<_, _>>()?;
    let specs = specs_for(enc, &monsters);
    let room = match start["room"].as_str() {
        Some("Elite") => RoomKind::Elite,
        Some("Boss") => RoomKind::Boss,
        _ => RoomKind::Monster,
    };

    // The first shuffle is the initial one: the opening hand in draw order,
    // then the rest of the draw pile.
    let mut opening: Vec<(CardId, bool)> = vec![];
    for key in ["hand", "draw"] {
        for v in first_snap[key].as_array().ok_or("snapshot without piles")? {
            opening.push(card_ref(ids, v)?);
        }
    }
    let script = Script {
        shuffles: VecDeque::from(vec![opening]),
        enemy_hp: start["enemies"].as_array().unwrap().iter().map(|e| e["max_hp"].as_i64().unwrap_or(1) as i32).collect(),
    };
    let mut c = Combat::with_script(
        &Setup {
            deck: &deck,
            hp: first_snap["hp"].as_i64().unwrap_or(1) as i32,
            max_hp: first_snap["max_hp"].as_i64().unwrap_or(1) as i32,
            max_energy: start["max_energy"].as_i64().unwrap_or(3) as i32,
            relics: &relics,
            potions: &potions,
            enemies: &specs,
            room,
            asc: Ascension(start["ascension"].as_i64().unwrap_or(0) as u8),
            seed,
        },
        script,
    );
    // Enemies that start damaged (the start record is taken at the first decision point).
    for (i, e) in start["enemies"].as_array().unwrap().iter().enumerate() {
        if let (Some(hp), true) = (e["hp"].as_i64(), i < c.enemies.len()) {
            c.enemies[i].creature.hp = hp as i32;
        }
    }

    let mut report = Report { snapshots: 0, actions: 0, divergence: None };
    for (n, rec) in records.iter().enumerate().skip(1) {
        match rec["t"].as_str().unwrap_or("") {
            "snapshot" => {
                // The game can snapshot once more after the last enemy dies.
                if c.is_over() && rec["enemies"].as_array().is_some_and(|a| a.is_empty()) {
                    continue;
                }
                match settle(&c, rec, 0) {
                    Ok(k) => c = k,
                    Err(e) => return Ok(failed(&report, n, e)),
                }
                if let Err(e) = force_moves(&mut c, rec) {
                    return Ok(failed(&report, n, e));
                }
                report.snapshots += 1;
            }
            "shuffle" => {
                let order = rec["cards"]
                    .as_array()
                    .ok_or("shuffle without cards")?
                    .iter()
                    .map(|v| card_ref(ids, v))
                    .collect::<Result<Vec<_>, _>>()?;
                c.script.shuffles.push_back(order);
            }
            "play" => {
                if c.is_over() {
                    return Ok(failed(&report, n, "card played after the sim ended combat".into()));
                }
                let (id, up) = card_ref(ids, rec)?;
                // Targets index the game's list of living enemies. A kill
                // shot loses its target before the hook fires; with one
                // enemy left that is unambiguous.
                let living: Vec<usize> = c.living_enemies().collect();
                let target = match rec["target"].as_u64() {
                    Some(t) => living.get(t as usize).copied(),
                    None if living.len() == 1 && crate::card::def(id).target == crate::types::TargetType::AnyEnemy => Some(living[0]),
                    None => None,
                };
                // Prefer the recorded hand index: identical cards are
                // interchangeable, but which one leaves the hand changes
                // the discard order and so later random picks.
                let want_idx = rec["hand_idx"].as_u64().map(|i| i as usize);
                let legal = c.legal_actions();
                let matches = |a: &Action| match *a {
                    Action::PlayCard { hand_idx, target: t } => {
                        let k = &c.player.hand[hand_idx];
                        k.id == id && k.upgraded == up && t == target
                    }
                    _ => false,
                };
                let action = legal
                    .iter()
                    .copied()
                    .find(|a| matches(a) && matches!(a, Action::PlayCard { hand_idx, .. } if Some(*hand_idx) == want_idx))
                    .or_else(|| legal.iter().copied().find(matches));
                let Some(action) = action else {
                    return Ok(failed(&report, n, format!("game played {id:?} (upgraded {up}) at {target:?}; sim has no such legal play")));
                };
                c.step(action);
                report.actions += 1;
            }
            "potion" => {
                let name = rec["id"].as_str().unwrap_or("");
                let Some(&pid) = ids.potions.get(name) else { return Ok(failed(&report, n, format!("unknown potion {name}"))) };
                let Some(slot) = c.potions.iter().position(|p| *p == Some(pid)) else {
                    return Ok(failed(&report, n, format!("game used {name}; sim has no such potion")));
                };
                let living: Vec<usize> = c.living_enemies().collect();
                let target = rec["target"].as_u64().and_then(|t| living.get(t as usize).copied());
                c.step(Action::UsePotion { slot, target });
                report.actions += 1;
            }
            "turn_start" => {
                if rec["turn"].as_u64().unwrap_or(1) > 1 {
                    if c.is_over() || c.pending.is_some() {
                        return Ok(failed(&report, n, "game started a new turn; sim is over or waiting on a choice".into()));
                    }
                    c.step(Action::EndTurn);
                    report.actions += 1;
                }
            }
            "end" => {
                // A loss during the enemy turn has no turn_start after it.
                if !c.is_over() && c.pending.is_none() {
                    c.step(Action::EndTurn);
                    report.actions += 1;
                }
                let won = rec["won"].as_bool().unwrap_or(false);
                let sim_won = c.outcome == Some(Outcome::Won);
                if won != sim_won || !c.is_over() {
                    return Ok(failed(&report, n, format!("game ended (won {won}); sim outcome {:?}", c.outcome)));
                }
            }
            other => return Err(format!("unknown record type {other}")),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::Setup;
    use crate::rng::Rng;

    fn card_json(id: CardId, up: bool) -> Value {
        json!({ "id": slug(&format!("{id:?}")), "up": up })
    }

    /// Record a sim playout in the recorder's format, then replay it in a
    /// sim with a different seed. The forced shuffles, HP, and moves must
    /// make the replay match at every decision point.
    #[test]
    fn sim_recording_round_trips() {
        let ids = Ids::new();
        for seed in 0..30u64 {
            let mut rng = Rng::new(seed);
            let enc = *rng.pick(crate::encounter::ALL).unwrap();
            let specs = enc.monsters(&mut rng);
            let mut deck = crate::ironclad_starter_deck();
            for _ in 0..6 {
                deck.push(Card::new(0, *rng.pick(crate::card::IRONCLAD_POOL).unwrap(), rng.next_int(2) == 0));
            }
            let potions = [Some(PotionId::FirePotion), Some(*rng.pick(crate::potion::ALL).unwrap())];
            let relics = [Relic::new(RelicId::Vajra)];
            let mut c = Combat::with_setup(&Setup {
                deck: &deck,
                hp: 70,
                max_hp: 80,
                max_energy: 3,
                relics: &relics,
                potions: &potions,
                enemies: &specs,
                room: RoomKind::Monster,
                asc: Ascension(10),
                seed: seed + 100,
            });
            let mut lines = vec![json!({
                "t": "start",
                "encounter": slug(&format!("{enc:?}")),
                "room": "Monster",
                "ascension": 10,
                "max_energy": 3,
                "deck": deck.iter().map(|k| card_json(k.id, k.upgraded)).collect::<Vec<_>>(),
                "relics": ["VAJRA"],
                "potions": potions.iter().map(|p| p.map(|p| slug(&format!("{p:?}")))).collect::<Vec<_>>(),
                "enemies": c.enemies.iter().map(|e| json!({
                    "id": slug(&format!("{:?}", e.monster.id)),
                    "hp": e.creature.hp,
                    "max_hp": e.creature.max_hp,
                })).collect::<Vec<_>>(),
            })];
            // The initial shuffle is implied by the first snapshot.
            let mut logged_shuffles = 1;
            let mut steps = 0;
            loop {
                if c.is_over() || steps > 400 {
                    break;
                }
                if c.pending.is_none() {
                    lines.push(snapshot_of(&c));
                }
                let acts = c.legal_actions();
                let a = acts[rng.next_int(acts.len())];
                let played = match a {
                    Action::PlayCard { hand_idx, .. } => Some((c.player.hand[hand_idx].id, c.player.hand[hand_idx].upgraded)),
                    _ => None,
                };
                // The recorder indexes targets into the living enemy list.
                let living: Vec<usize> = c.living_enemies().collect();
                let living_idx = |t: Option<usize>| t.and_then(|t| living.iter().position(|&l| l == t));
                let turn_before = c.player.turn;
                c.step(a);
                steps += 1;
                // The recorder logs shuffles as they happen, then the play
                // or potion that caused them.
                for s in &c.shuffle_log[logged_shuffles..] {
                    let cards: Vec<Value> = s.iter().map(|&(id, up)| card_json(id, up)).collect();
                    lines.push(json!({ "t": "shuffle", "cards": cards }));
                }
                logged_shuffles = c.shuffle_log.len();
                match a {
                    Action::PlayCard { hand_idx, target } => {
                        let (id, up) = played.unwrap();
                        let mut l = card_json(id, up);
                        l["t"] = json!("play");
                        l["hand_idx"] = json!(hand_idx);
                        l["target"] = json!(living_idx(target));
                        lines.push(l);
                    }
                    Action::UsePotion { slot, target } => {
                        let id = potions[slot].map(|p| slug(&format!("{p:?}")));
                        lines.push(json!({ "t": "potion", "id": id, "target": living_idx(target) }));
                    }
                    Action::EndTurn if c.player.turn > turn_before => {
                        lines.push(json!({ "t": "turn_start", "turn": c.player.turn }));
                    }
                    _ => {}
                }
            }
            lines.push(json!({ "t": "end", "won": c.outcome == Some(Outcome::Won), "hp": c.player.creature.hp }));
            let text: String = lines.iter().map(|l| l.to_string() + "\n").collect();
            // Same RNG seed: every decision point must match.
            let report = replay_seeded(&text, &ids, seed + 100).unwrap();
            assert!(report.ok(), "seed {seed} {enc:?}: {:?}", report.divergence);
            assert!(report.snapshots > 0);
            // Cold seed: the forced opening shuffle and enemy HP must still
            // carry the first decision point.
            let report = replay(&text, &ids).unwrap();
            assert!(report.snapshots >= 1, "seed {seed} {enc:?}: {:?}", report.divergence);
        }
    }
}
