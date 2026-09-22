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
#[derive(Clone, Debug)]
pub struct Report {
    /// Recorded decision points that matched.
    pub snapshots: usize,
    /// Actions applied.
    pub actions: usize,
    /// First mismatch, with the record index it happened at.
    pub divergence: Option<(usize, String)>,
    /// Times an unscripted random pick had to be re-rolled to match.
    pub reseeds: u32,
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
        c.player.hand.iter().map(|k| json!({ "id": slug(&format!("{:?}", k.id)), "up": k.upgraded, "cost": if k.def().x_cost { -1 } else { c.cost(k) } })).collect();
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
        // The game drops dead creatures from its enemy list (except reviving
        // ones) and keeps it in slot order.
        "enemies": c.present_enemies().map(|i| (i, &c.enemies[i])).map(|(i, e)| json!({
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

/// Enemies spawned since the last snapshot rolled their HP in the sim; the
/// recording only shows the game's roll now. Adopt it while they are still
/// undamaged, which is when the roll is the only difference.
fn adopt_spawn_hp(c: &mut Combat, snap: &Value, known: usize) -> Result<(), String> {
    let empty = vec![];
    let living: Vec<usize> = c.present_enemies().collect();
    for (e, &i) in snap["enemies"].as_array().unwrap_or(&empty).iter().zip(&living) {
        let asc = c.asc;
        let cr = &mut c.enemies[i].creature;
        if i >= known && cr.hp == cr.max_hp {
            if let (Some(hp), Some(max)) = (e["hp"].as_i64(), e["max_hp"].as_i64()) {
                check_hp_range(c.enemies[i].monster.id, max as i32, asc)?;
                let cr = &mut c.enemies[i].creature;
                cr.max_hp = max as i32;
                cr.hp = hp as i32;
            }
        }
    }
    Ok(())
}

/// Snecko Oil rolls each hand card's cost; the recording shows the
/// results in the next snapshot. Copy them onto matching cards.
fn adopt_hand_costs(c: &mut Combat, snap: &Value) {
    let empty = vec![];
    let mut taken = vec![false; c.player.hand.len()];
    for r in snap["hand"].as_array().unwrap_or(&empty) {
        let (Some(id), Some(cost)) = (r["id"].as_str(), r["cost"].as_i64()) else { continue };
        let up = r["up"].as_bool().unwrap_or(false);
        let found = c.player.hand.iter().enumerate().position(|(i, k)| {
            !taken[i] && slug(&format!("{:?}", k.id)) == id && k.upgraded == up && k.cost_this_turn.is_some()
        });
        if let Some(i) = found {
            taken[i] = true;
            if cost >= 0 {
                c.player.hand[i].cost_this_turn = Some(cost as i32);
            }
        }
    }
}

/// Forced rolls still have to be rolls the sim could make.
fn check_hp_range(id: MonsterId, max_hp: i32, asc: Ascension) -> Result<(), String> {
    let (lo, hi) = crate::monster::Monster::hp_range(id, asc);
    if (lo..=hi).contains(&max_hp) {
        Ok(())
    } else {
        Err(format!("{id:?} max HP {max_hp} outside the sim's range {lo}..={hi}"))
    }
}

/// Recorded random outcomes are logged before the play that caused them
/// and belong to it alone; anything left after the play is stale.
fn clear_per_play(c: &mut Combat) {
    c.script.random_targets.clear();
    c.script.generated.clear();
    c.script.random_exhausts.clear();
}

/// Force the recorded next moves onto the sim's enemies.
fn force_moves(c: &mut Combat, snap: &Value) -> Result<(), String> {
    let empty = vec![];
    let living: Vec<usize> = c.present_enemies().collect();
    for (e, &i) in snap["enemies"].as_array().unwrap_or(&empty).iter().zip(&living) {
        let Some(name) = e["move"].as_str() else { continue };
        if !c.set_enemy_move(i, name) {
            return Err(format!("enemy {i} has no move {name}"));
        }
    }
    Ok(())
}

fn failed(report: &Report, n: usize, msg: String) -> Report {
    Report { snapshots: report.snapshots, actions: report.actions, divergence: Some((n, msg)), reseeds: report.reseeds }
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
        .map(|a| {
            a.iter()
                .filter_map(|v| {
                    let name = v.as_str()?;
                    let id = ids.relics.get(name);
                    if id.is_none() {
                        eprintln!("note: unknown relic {name}, ignored");
                    }
                    id
                })
                .map(|&id| Relic::new(id))
                .collect()
        })
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
    let asc = Ascension(start["ascension"].as_i64().unwrap_or(0) as u8);
    for (e, &id) in start["enemies"].as_array().unwrap().iter().zip(&monsters) {
        check_hp_range(id, e["max_hp"].as_i64().unwrap_or(0) as i32, asc)?;
    }
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
        random_targets: VecDeque::new(),
        generated: VecDeque::new(),
        random_exhausts: vec![],
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
            asc,
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

    let mut report = Report { snapshots: 0, actions: 0, divergence: None, reseeds: 0 };
    let mut known_enemies = c.enemies.len();
    let mut snecko_pending = false;
    // A snapshot followed by another snapshot with no action in between caught
    // the game mid-resolution (the recorder polls between a card's exhaust and
    // the draw it triggers, for example). Only the last one is a decision point.
    let transient = |n: usize| {
        records[n + 1..]
            .iter()
            .find(|r| !matches!(r["t"].as_str(), Some("hit" | "exhaust")))
            .is_some_and(|r| r["t"] == "snapshot")
    };
    // Random picks among existing cards (Aggression's pull, an unrecorded
    // random exhaust) are not scripted. When a snapshot does not match, rewind
    // to the last matching one, re-roll that stream, and try again.
    const RESEED_LIMIT: u32 = 64;
    let mut checkpoint = (c.clone(), 0usize, known_enemies, report.clone());
    let mut tries = 0u32;
    let mut n = 1;
    while n < records.len() {
        let rec = &records[n];
        match rec["t"].as_str().unwrap_or("") {
            "snapshot" => {
                // The game can snapshot once more after the last enemy dies.
                if c.is_over() && rec["enemies"].as_array().is_some_and(|a| a.is_empty()) {
                    n += 1;
                    continue;
                }
                if transient(n) {
                    n += 1;
                    continue;
                }
                if std::mem::take(&mut snecko_pending) {
                    adopt_hand_costs(&mut c, rec);
                }
                if let Err(e) = adopt_spawn_hp(&mut c, rec, known_enemies) {
                    return Ok(failed(&report, n, e));
                }
                known_enemies = c.enemies.len();
                match settle(&c, rec, 0) {
                    Ok(k) => c = k,
                    Err(e) => {
                        if tries >= RESEED_LIMIT || c.rngs.card_selection == checkpoint.0.rngs.card_selection {
                            return Ok(failed(&report, n, e));
                        }
                        tries += 1;
                        let (ck, ck_n, ck_known, ck_report) = &checkpoint;
                        report = Report { reseeds: report.reseeds + 1, ..ck_report.clone() };
                        c = ck.clone();
                        c.rngs.card_selection = crate::rng::Rng::new(seed ^ 0x06 ^ (u64::from(tries) << 32));
                        known_enemies = *ck_known;
                        snecko_pending = false;
                        n = ck_n + 1;
                        continue;
                    }
                }
                tries = 0;
                checkpoint = (c.clone(), n, known_enemies, report.clone());
                if let Err(e) = force_moves(&mut c, rec) {
                    return Ok(failed(&report, n, e));
                }
                report.snapshots += 1;
            }
            "hit" => {
                // Hits precede the play that dealt them; queue the targets of
                // enemy-targeting cards for the sim's random picks (random-
                // target cards, and auto-played cards choosing a target).
                let targeted = rec["card"].as_str().and_then(|s| ids.cards.get(s)).is_some_and(|&id| {
                    use crate::types::TargetType::*;
                    matches!(crate::card::def(id).target, RandomEnemy | AnyEnemy)
                });
                if let (true, Some(t)) = (targeted, rec["target"].as_u64()) {
                    c.script.random_targets.push_back(t as usize);
                }
            }
            "gen" => c.script.generated.push_back(card_ref(ids, rec)?.0),
            "exhaust" => c.script.random_exhausts.push(card_ref(ids, rec)?),
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
                let living: Vec<usize> = c.present_enemies().collect();
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
                clear_per_play(&mut c);
            }
            "potion" => {
                let name = rec["id"].as_str().unwrap_or("");
                let Some(&pid) = ids.potions.get(name) else { return Ok(failed(&report, n, format!("unknown potion {name}"))) };
                let Some(slot) = c.potions.iter().position(|p| *p == Some(pid)) else {
                    return Ok(failed(&report, n, format!("game used {name}; sim has no such potion")));
                };
                let living: Vec<usize> = c.present_enemies().collect();
                let target = rec["target"].as_u64().and_then(|t| living.get(t as usize).copied());
                c.step(Action::UsePotion { slot, target });
                report.actions += 1;
                clear_per_play(&mut c);
                snecko_pending = pid == PotionId::SneckoOil;
            }
            "turn_start" => {
                if rec["turn"].as_u64().unwrap_or(1) > 1 {
                    if c.is_over() || c.pending.is_some() {
                        return Ok(failed(&report, n, "game started a new turn; sim is over or waiting on a choice".into()));
                    }
                    c.step(Action::EndTurn);
                    report.actions += 1;
                    clear_per_play(&mut c);
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
        n += 1;
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
                let living: Vec<usize> = c.present_enemies().collect();
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
