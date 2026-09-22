//! Replay harness. Reads a recording written by the `mod/` recorder (one
//! JSON record per line), rebuilds the combat in the sim with the recorded
//! shuffles, enemy HP, and monster moves forced, applies the recorded
//! actions, and diffs the sim against every recorded decision point.
//!
//! `Replayer` does that one record at a time, which is what lets the
//! advisor (`py/sts2ai/advise.py`) follow a combat as it is being played.
//! `replay` and `replay_seeded` push a whole file through one.
//!
//! Model ids are the game's: the class name in screaming snake case, so
//! `StrikeIronclad` is `STRIKE_IRONCLAD` and `StrengthPower` is
//! `STRENGTH_POWER`.

use std::collections::{HashMap, VecDeque};

use serde_json::{json, Value};

use crate::card::{Card, IRONCLAD_POOL};
use crate::enchant::{EnchantmentId, ALL as ALL_ENCHANTMENTS};
use crate::combat::{Action, Combat, Outcome, Script};
use crate::gen::{card_ref, FightSetup};
use crate::effect::Then;
use crate::encounter::Encounter;
use crate::ids::{CardId, MonsterId, PowerId, ALL_CARDS, ALL_MONSTERS};
use crate::potion::PotionId;
use crate::relic::RelicId;
use crate::types::{Ascension, CreatureRef, Side};

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
#[derive(Clone)]
pub struct Ids {
    pub(crate) cards: HashMap<String, CardId>,
    pub(crate) monsters: HashMap<String, MonsterId>,
    pub(crate) relics: HashMap<String, RelicId>,
    pub(crate) potions: HashMap<String, PotionId>,
    pub(crate) encounters: HashMap<String, Encounter>,
    pub(crate) enchantments: HashMap<String, EnchantmentId>,
}

impl Ids {
    pub fn new() -> Self {
        Self {
            cards: table(ALL_CARDS, ""),
            monsters: table(ALL_MONSTERS, ""),
            relics: table(crate::relic::ALL, ""),
            potions: table(crate::potion::ALL, ""),
            encounters: table(crate::encounter::ALL, ""),
            enchantments: table(ALL_ENCHANTMENTS, ""),
        }
    }
}

impl Ids {
    /// The encounter a recorder id names, if the sim has it.
    pub fn encounter(&self, name: &str) -> Option<&Encounter> {
        self.encounters.get(name)
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

/// `Recorder.CardRef`'s enchantment triple: id, amount, whether it is spent.
fn ench_json(e: &crate::enchant::Enchantment) -> Value {
    json!([slug(&format!("{:?}", e.id)), e.amount, e.disabled])
}

/// The sim's view of a combat in the recorder's snapshot shape, piles in
/// their real order. `diff` normalizes both sides.
pub fn snapshot_of(c: &Combat) -> Value {
    // `ench` is omitted for a plain card, matching the recorder, so a run
    // with no enchantments diffs exactly as it did before.
    let card = |k: &Card| {
        let mut v = json!({ "id": slug(&format!("{:?}", k.id)), "up": k.upgraded });
        if let Some(e) = &k.enchantment {
            v["ench"] = ench_json(e);
        }
        v
    };
    let hand: Vec<Value> = c
        .player
        .hand
        .iter()
        .map(|k| {
            let mut v = card(k);
            v["cost"] = json!(if k.def().x_cost { -1 } else { c.cost(k) });
            v
        })
        .collect();
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
/// A hatched Tough Egg re-rolls its HP the same way, so it is adopted too,
/// minus the range check, which is the egg's and not the hatchling's.
fn adopt_spawn_hp(c: &mut Combat, snap: &Value, known: usize) -> Result<(), String> {
    let empty = vec![];
    let living: Vec<usize> = c.present_enemies().collect();
    let rerolled = std::mem::take(&mut c.stats.hp_rerolled);
    for (e, &i) in snap["enemies"].as_array().unwrap_or(&empty).iter().zip(&living) {
        let asc = c.asc;
        let cr = &mut c.enemies[i].creature;
        let fresh = i >= known;
        if (fresh || rerolled.contains(&i)) && cr.hp == cr.max_hp {
            if let (Some(hp), Some(max)) = (e["hp"].as_i64(), e["max_hp"].as_i64()) {
                if fresh {
                    check_hp_range(c.enemies[i].monster.id, max as i32, asc)?;
                }
                let cr = &mut c.enemies[i].creature;
                cr.max_hp = max as i32;
                cr.hp = hp as i32;
            }
        }
    }
    Ok(())
}

/// A card dropped into the draw pile at a random depth lands somewhere
/// nothing records: Beckon from the Soul Fysh, Soot from Biiig Hug, the
/// Dazed from Tea of Discourtesy and Blessed Antler before the opening draw.
/// When the sim has done that since the last snapshot and holds the same
/// cards across hand and draw pile as the recording, take the recording's
/// layout of both, which also settles which of them a draw picked up.
///
/// `opening` also lets a card match on id alone and take the snapshot's
/// upgrade, once every exact match has been placed: Bone Tea upgrades
/// whichever cards the opening hand drew, and those differ between game
/// and sim.
fn adopt_layout(c: &mut Combat, snap: &Value, opening: bool) {
    if c.stats.random_draw_inserts == 0 {
        return;
    }
    c.stats.random_draw_inserts = 0;
    let empty = vec![];
    let want: Vec<(bool, &Value)> = snap["hand"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|r| (true, r))
        .chain(snap["draw"].as_array().unwrap_or(&empty).iter().map(|r| (false, r)))
        .collect();
    let mut pool: Vec<Option<Card>> =
        c.player.hand.iter().chain(&c.player.draw).cloned().map(Some).collect();
    let mut placed: Vec<Option<Card>> = vec![None; want.len()];
    let passes: &[bool] = if opening { &[true, false] } else { &[true] };
    for &exact in passes {
        for (slot, (_, r)) in want.iter().enumerate() {
            if placed[slot].is_some() {
                continue;
            }
            let (Some(id), up) = (r["id"].as_str(), r["up"].as_bool().unwrap_or(false)) else { return };
            let fits = |k: &Card| slug(&format!("{:?}", k.id)) == id && (!exact || k.upgraded == up);
            if let Some(i) = pool.iter().position(|k| k.as_ref().is_some_and(fits)) {
                let mut k = pool[i].take().expect("position found it");
                k.upgraded = up;
                placed[slot] = Some(k);
            }
        }
    }
    if pool.iter().any(Option::is_some) || placed.iter().any(Option::is_none) {
        return;
    }
    let (hand, draw): (Vec<_>, Vec<_>) = want.iter().zip(placed).partition(|((in_hand, _), _)| *in_hand);
    c.player.hand = hand.into_iter().filter_map(|(_, k)| k).collect();
    c.player.draw = draw.into_iter().filter_map(|(_, k)| k).collect();
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

/// Two copies of a card differ only by a cost change one of them carries
/// (Touch of Insanity's free card), and a recorded shuffle names cards by id
/// alone, so the replay may have drawn the other copy. When the hand holds
/// the wrong one and the right one sits in another pile, swap them.
fn adopt_copy_costs(c: &mut Combat, snap: &Value) {
    type Key = (String, bool, i64);
    let key = |c: &Combat, k: &Card| -> Key {
        let cost = if k.def().x_cost { -1 } else { i64::from(c.cost(k)) };
        (slug(&format!("{:?}", k.id)), k.upgraded, cost)
    };
    let empty = vec![];
    let mut have: Vec<Option<Key>> = c.player.hand.iter().map(|k| Some(key(c, k))).collect();
    let mut missing: Vec<Key> = vec![];
    for r in snap["hand"].as_array().unwrap_or(&empty) {
        let want: Key = (r["id"].as_str().unwrap_or("").to_string(), r["up"].as_bool().unwrap_or(false), r["cost"].as_i64().unwrap_or(0));
        match have.iter().position(|h| h.as_ref() == Some(&want)) {
            Some(i) => have[i] = None,
            None => missing.push(want),
        }
    }
    for want in missing {
        let Some(h) = have.iter().position(|k| k.as_ref().is_some_and(|k| k.0 == want.0 && k.1 == want.1)) else { continue };
        let p = &c.player;
        let found = [&p.draw, &p.discard, &p.exhaust]
            .iter()
            .enumerate()
            .find_map(|(n, pile)| pile.iter().position(|k| key(c, k) == want).map(|j| (n, j)));
        let pl = &mut c.player;
        match found {
            Some((0, j)) => std::mem::swap(&mut pl.hand[h], &mut pl.draw[j]),
            Some((1, j)) => std::mem::swap(&mut pl.hand[h], &mut pl.discard[j]),
            Some((_, j)) => std::mem::swap(&mut pl.hand[h], &mut pl.exhaust[j]),
            None => continue,
        }
        have[h] = None;
    }
}

/// Confused (Snecko Eye) rolls a cost for the combat onto every card drawn,
/// from a stream the sim cannot share with the game. While it is up, copy
/// each hand card's recorded cost onto the sim's matching card, backing out
/// whatever the global modifiers add on top.
fn adopt_confused_costs(c: &mut Combat, snap: &Value) {
    if c.player.creature.power(PowerId::Confused).is_none() {
        return;
    }
    let empty = vec![];
    let mut taken = vec![false; c.player.hand.len()];
    for r in snap["hand"].as_array().unwrap_or(&empty) {
        let (Some(id), Some(cost)) = (r["id"].as_str(), r["cost"].as_i64()) else { continue };
        let up = r["up"].as_bool().unwrap_or(false);
        let found = c.player.hand.iter().enumerate().position(|(i, k)| {
            !taken[i]
                && slug(&format!("{:?}", k.id)) == id
                && k.upgraded == up
                && k.cost_this_combat.is_some()
                && k.cost_this_turn.is_none()
        });
        if let Some(i) = found {
            taken[i] = true;
            let k = &c.player.hand[i];
            let global = c.cost(k) - k.local_cost();
            if cost >= 0 {
                c.player.hand[i].cost_this_combat = Some((cost as i32 - global).max(0));
            }
        }
    }
}

/// A choose-a-card screen open at the turn start (Choices Paradox) is logged
/// as a snapshot with the pick already in hand, and the `gen` record naming
/// the pick only follows it. While an offer is pending, put any hand card
/// the sim cannot account for on offer in place of one the sim rolled.
fn adopt_offer(c: &mut Combat, snap: &Value) {
    if !c.pending.as_ref().is_some_and(|p| p.then == Then::TakeOffer) {
        return;
    }
    let empty = vec![];
    let mut held: Vec<String> = c.player.hand.iter().map(|k| slug(&format!("{:?}", k.id))).collect();
    let ids = Ids::new();
    for r in snap["hand"].as_array().unwrap_or(&empty) {
        let Some(name) = r["id"].as_str() else { continue };
        if let Some(i) = held.iter().position(|h| h == name) {
            held.remove(i);
            continue;
        }
        let Some(&id) = ids.cards.get(name) else { continue };
        if c.player.offer.iter().any(|k| k.id == id) {
            continue;
        }
        if let Some(k) = c.player.offer.first_mut() {
            k.id = id;
            c.script.adopted_offers.push(id);
        }
    }
}

/// A choose-a-card screen (Attack Potion) shows the cards the game rolled;
/// the `choice` record lists them. Put those on the sim's offer in place of
/// its own roll, so the pick is made from what the game shows.
fn adopt_offer_options(c: &mut Combat, ids: &Ids, rec: &Value) {
    if !c.pending.as_ref().is_some_and(|p| p.then == Then::TakeOffer) {
        return;
    }
    let Some(options) = rec["options"].as_array() else { return };
    let Ok(cards) = options.iter().map(|v| card_ref(ids, v)).collect::<Result<Vec<_>, _>>() else { return };
    if cards.len() != c.player.offer.len() {
        return;
    }
    for (k, (id, up)) in c.player.offer.iter_mut().zip(cards) {
        k.id = id;
        k.upgraded = up;
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

/// Replays a recording record by record, keeping a sim `Combat` in sync
/// with the game. `replay_seeded` pushes a whole file through one; the
/// advisor pushes lines as the recorder writes them.
pub struct Replayer {
    ids: Ids,
    seed: u64,
    c: Combat,
    report: Report,
    known_enemies: usize,
    snecko_pending: bool,
    /// A card the sim played early because the game's card-select screen
    /// opened for it (a `choice` record); the `play` record for it, which
    /// the game only writes once the card resolves, is then skipped.
    pre_played: Option<(CardId, bool)>,
    /// The same for a potion whose effect opened the screen (Attack
    /// Potion): its `potion` record comes after the pick.
    pre_potion: Option<PotionId>,
    gate: Gate,
    queue: VecDeque<(usize, Value)>,
    /// Records applied since the last checkpoint, re-applied on a reseed.
    since: Vec<(usize, Value)>,
    checkpoint: Checkpoint,
    tries: u32,
    next_index: usize,
    at_decision: bool,
    /// The hand the last snapshot showed, with costs. Two copies of a card
    /// can differ only by a cost the sim set randomly (Touch of Insanity,
    /// Snecko Oil), and then only this says which one the game played.
    last_hand: Vec<Value>,
    /// A divergence was reported; `replay_seeded` stops there.
    stopped: bool,
}

struct Checkpoint {
    c: Combat,
    known_enemies: usize,
    report: Report,
}

/// What feeding a record did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// State advanced; the game is not waiting on the player.
    Ok,
    /// The sim is settled at a decision point: time to recommend an action.
    Decision,
    /// The sim no longer matches the game, with the first differing field.
    Diverged(String),
    /// The combat ended and the sim agreed on the outcome.
    Ended,
}

/// One record's effect inside the replayer.
enum Applied {
    Ok,
    Decision,
    Ended,
    Diverged(String),
    /// Re-roll the unscripted streams and re-apply from the checkpoint.
    Rewind,
}

/// Reorders the record stream the way the replay wants to see it. A
/// snapshot immediately followed by another snapshot, with no action in
/// between, caught the game mid-resolution (the recorder polls between a
/// card's exhaust and the draw it triggers, for example) and is dropped;
/// `hit` and `exhaust` records arriving while a snapshot is held keep
/// their place after it.
#[derive(Default)]
struct Gate {
    held: Option<(usize, Value)>,
    deferred: Vec<(usize, Value)>,
}

impl Gate {
    fn feed(&mut self, n: usize, rec: Value) -> Vec<(usize, Value)> {
        match rec["t"].as_str().unwrap_or("") {
            "hit" | "exhaust" if self.held.is_some() => {
                self.deferred.push((n, rec));
                vec![]
            }
            "snapshot" => {
                let mut out = vec![];
                if self.held.take().is_some() {
                    out.append(&mut self.deferred);
                }
                self.held = Some((n, rec));
                out
            }
            _ => {
                let mut out = self.flush();
                out.push((n, rec));
                out
            }
        }
    }

    /// Release the held snapshot: no more records are coming, or the game
    /// has gone quiet and it is a real decision point.
    fn flush(&mut self) -> Vec<(usize, Value)> {
        let mut out: Vec<(usize, Value)> = self.held.take().into_iter().collect();
        out.append(&mut self.deferred);
        out
    }
}

/// Random picks among existing cards (Aggression's pull) and random targets
/// of power hits (Juggernaut) are not scripted. When a snapshot does not
/// match, rewind to the last matching one, re-roll those streams, and try
/// again.
const RESEED_LIMIT: u32 = 64;

impl Replayer {
    /// Build the combat the `start` record describes. `first_snap` is the
    /// recording's first snapshot: it carries the player's HP and the
    /// opening draw order, which the start record does not.
    pub fn new(start: &Value, first_snap: &Value, ids: Ids, seed: u64) -> Result<Self, String> {
        let fs = FightSetup::from_start(start, first_snap, &ids)?;
        let enemies = start["enemies"].as_array().ok_or("start without enemies")?;
        for (e, spec) in enemies.iter().zip(&fs.enemies) {
            check_hp_range(spec.id, e["max_hp"].as_i64().unwrap_or(0) as i32, fs.asc)?;
        }

        // The first shuffle is the initial one. The recorder logs it as
        // `opening`; older recordings only have the first snapshot, whose
        // opening hand in draw order, then the rest of the draw pile, is the
        // same thing unless a card was played before the player could act
        // (Whispering Earring).
        let mut opening: Vec<(CardId, bool)> = vec![];
        match start["opening"].as_array() {
            Some(cards) => {
                for v in cards {
                    opening.push(card_ref(&ids, v)?);
                }
            }
            None => {
                for key in ["hand", "draw"] {
                    for v in first_snap[key].as_array().ok_or("snapshot without piles")? {
                        opening.push(card_ref(&ids, v)?);
                    }
                }
            }
        }
        // Records from before the first decision point (`early`): cards
        // made (Crossbow's turn 1 attack) and cards exhausted (True Grit
        // under Whispering Earring). Only cards random generation could have
        // rolled are forced; Luminesce and the Dazed are placed by name.
        let early = |t: &str| -> Vec<(CardId, bool)> {
            start["early"]
                .as_array()
                .map(|a| a.iter().filter(|v| v["t"] == t).filter_map(|v| card_ref(&ids, v).ok()).collect())
                .unwrap_or_default()
        };
        let generated = early("gen")
            .into_iter()
            .map(|(id, _)| id)
            .filter(|&id| IRONCLAD_POOL.contains(&id) && crate::card::def(id).generatable)
            .collect();
        let random_exhausts = early("exhaust");
        let script = Script {
            shuffles: VecDeque::from(vec![opening]),
            enemy_hp: enemies.iter().map(|e| e["max_hp"].as_i64().unwrap_or(1) as i32).collect(),
            random_targets: VecDeque::new(),
            generated,
            random_exhausts,
            adopted_offers: vec![],
            unforced: vec![],
            hit_cards: vec![],
        };
        let mut c = Combat::with_script(&fs.as_setup(seed), script);
        // Enemies that start damaged (the start record is taken at the first decision point).
        for (i, e) in enemies.iter().enumerate() {
            if let (Some(hp), true) = (e["hp"].as_i64(), i < c.enemies.len()) {
                c.enemies[i].creature.hp = hp as i32;
            }
        }
        // The player's HP is read from that same first decision point, after
        // turn 1 has healed (Blood Vial) or hurt (Royal Poison), so the sim
        // having just done so again is undone.
        c.player.creature.hp = fs.hp;
        adopt_layout(&mut c, first_snap, true);

        let report = Report { snapshots: 0, actions: 0, divergence: None, reseeds: 0 };
        let known_enemies = c.enemies.len();
        Ok(Self {
            ids,
            seed,
            checkpoint: Checkpoint { c: c.clone(), known_enemies, report: report.clone() },
            c,
            report,
            known_enemies,
            snecko_pending: false,
            pre_played: None,
            pre_potion: None,
            gate: Gate::default(),
            queue: VecDeque::new(),
            since: vec![],
            tries: 0,
            // Record 0 is the start record this was built from.
            next_index: 1,
            at_decision: false,
            last_hand: vec![],
            stopped: false,
        })
    }

    pub fn combat(&self) -> &Combat {
        &self.c
    }

    /// True when the game is waiting on the player: the sim is settled at a
    /// recorded decision point, or it has a card choice open.
    pub fn at_decision(&self) -> bool {
        self.at_decision && !self.c.is_over()
    }

    pub fn report(&self) -> &Report {
        &self.report
    }

    pub fn ids(&self) -> &Ids {
        &self.ids
    }

    /// Apply one record. Errors are the ones a replay cannot continue past
    /// (bad ids, unknown record types); a state mismatch comes back as
    /// `Step::Diverged` and leaves the replayer usable.
    pub fn feed(&mut self, rec: &Value) -> Result<Step, String> {
        let n = self.next_index;
        self.next_index += 1;
        for item in self.gate.feed(n, rec.clone()) {
            self.queue.push_back(item);
        }
        self.drain()
    }

    /// Release a snapshot the gate is still holding, if the sim is ready
    /// for it. At the end of a file the held snapshot is the last decision
    /// point; live, the recorder having gone quiet means the same.
    ///
    /// A snapshot the sim cannot match stays held: that is the shape of a
    /// mid-resolution poll, and the record after it says whether it was one.
    /// A decision point that needs a reseed to match looks the same, so it
    /// gets no advice until the next record arrives.
    pub fn flush(&mut self) -> Result<Step, String> {
        if let Some((_, snap)) = &self.gate.held {
            if settle(&self.c, snap, 0).is_err() {
                return Ok(self.step());
            }
        }
        self.finish()
    }

    /// Release the held snapshot whether or not it matches: the file has
    /// ended, so nothing more is coming to settle it.
    pub fn finish(&mut self) -> Result<Step, String> {
        for item in self.gate.flush() {
            self.queue.push_back(item);
        }
        self.drain()
    }

    fn step(&self) -> Step {
        if self.at_decision() {
            Step::Decision
        } else {
            Step::Ok
        }
    }

    fn drain(&mut self) -> Result<Step, String> {
        let mut ended = false;
        while let Some((n, rec)) = self.queue.pop_front() {
            self.since.push((n, rec.clone()));
            match self.apply(n, &rec)? {
                Applied::Ok | Applied::Decision => {}
                Applied::Ended => ended = true,
                Applied::Diverged(msg) => {
                    self.report.divergence.get_or_insert((n, msg.clone()));
                    self.stopped = true;
                    return Ok(Step::Diverged(msg));
                }
                Applied::Rewind => {
                    for item in self.since.drain(..).rev() {
                        self.queue.push_front(item);
                    }
                }
            }
        }
        // The step reports where the feed left the sim, not what happened
        // on the way: several records can arrive at once.
        Ok(if ended { Step::Ended } else { self.step() })
    }

    /// Restore the last matching state and re-roll the unscripted streams.
    fn rewind(&mut self) {
        self.tries += 1;
        self.report = Report { reseeds: self.report.reseeds + 1, ..self.checkpoint.report.clone() };
        self.c = self.checkpoint.c.clone();
        let salt = u64::from(self.tries) << 32;
        self.c.rngs.card_selection = crate::rng::Rng::new(self.seed ^ 0x06 ^ salt);
        self.c.rngs.targets = crate::rng::Rng::new(self.seed ^ 0x03 ^ salt);
        // Spawned enemies roll their own HP and a generated status lands at a
        // random depth in the draw pile. Neither is recorded.
        self.c.rngs.niche = crate::rng::Rng::new(self.seed ^ 0x04 ^ salt);
        self.c.rngs.card_generation = crate::rng::Rng::new(self.seed ^ 0x05 ^ salt);
        self.known_enemies = self.checkpoint.known_enemies;
        self.snecko_pending = false;
        self.at_decision = false;
    }

    fn apply(&mut self, _n: usize, rec: &Value) -> Result<Applied, String> {
        match rec["t"].as_str().unwrap_or("") {
            "snapshot" => {
                // The game can snapshot once more after the last enemy dies.
                if self.c.is_over() && rec["enemies"].as_array().is_some_and(|a| a.is_empty()) {
                    return Ok(Applied::Ok);
                }
                self.c.script.unforced.clear();
                self.c.script.hit_cards.clear();
                if std::mem::take(&mut self.snecko_pending) {
                    adopt_hand_costs(&mut self.c, rec);
                }
                adopt_confused_costs(&mut self.c, rec);
                adopt_copy_costs(&mut self.c, rec);
                adopt_offer(&mut self.c, rec);
                adopt_layout(&mut self.c, rec, false);
                if let Err(e) = adopt_spawn_hp(&mut self.c, rec, self.known_enemies) {
                    return Ok(Applied::Diverged(e));
                }
                self.known_enemies = self.c.enemies.len();
                match settle(&self.c, rec, 0) {
                    Ok(k) => self.c = k,
                    Err(e) => {
                        // Nothing random happened since the checkpoint, so
                        // another roll of the dice cannot change the answer.
                        let untouched = self.c.rngs.card_selection == self.checkpoint.c.rngs.card_selection
                            && self.c.rngs.targets == self.checkpoint.c.rngs.targets
                            && self.c.rngs.niche == self.checkpoint.c.rngs.niche
                            && self.c.rngs.card_generation == self.checkpoint.c.rngs.card_generation;
                        if self.tries >= RESEED_LIMIT || untouched {
                            return Ok(Applied::Diverged(e));
                        }
                        self.rewind();
                        return Ok(Applied::Rewind);
                    }
                }
                self.tries = 0;
                self.since.clear();
                self.checkpoint =
                    Checkpoint { c: self.c.clone(), known_enemies: self.known_enemies, report: self.report.clone() };
                if let Err(e) = force_moves(&mut self.c, rec) {
                    return Ok(Applied::Diverged(e));
                }
                self.last_hand = rec["hand"].as_array().cloned().unwrap_or_default();
                self.report.snapshots += 1;
                // A snapshot the game took after the killing blow is not a
                // decision point, and neither is one the sim ended on.
                self.at_decision = !self.c.is_over() && self.c.side == Side::Player;
                Ok(if self.at_decision { Applied::Decision } else { Applied::Ok })
            }
            "hit" => {
                // Hits precede the play that dealt them; queue the targets of
                // enemy-targeting cards for the sim's random picks (random-
                // target cards, and auto-played cards choosing a target).
                let targeted = rec["card"].as_str().and_then(|s| self.ids.cards.get(s)).is_some_and(|&id| {
                    use crate::types::TargetType::*;
                    matches!(crate::card::def(id).target, RandomEnemy | AnyEnemy)
                });
                if let (true, Some(t)) = (targeted, rec["target"].as_u64()) {
                    self.c.script.random_targets.push_back(t as usize);
                }
                if let Some(&id) = rec["card"].as_str().and_then(|s| self.ids.cards.get(s)) {
                    self.c.script.hit_cards.push(id);
                }
                Ok(Applied::Ok)
            }
            "gen" => {
                let id = card_ref(&self.ids, rec)?.0;
                if let Some(i) = self.c.script.adopted_offers.iter().position(|&a| a == id) {
                    self.c.script.adopted_offers.remove(i);
                    return Ok(Applied::Ok);
                }
                // The sim already rolled this card: make its roll the game's.
                if let Some(rolled) = self.c.script.unforced.pop() {
                    let newest = self.c.all_cards().filter(|k| k.id == rolled).map(|k| k.uid).max();
                    if let Some(k) = newest.and_then(|u| self.c.find_card_mut(u)) {
                        k.id = id;
                        return Ok(Applied::Ok);
                    }
                }
                self.c.script.generated.push_back(id);
                Ok(Applied::Ok)
            }
            "exhaust" => {
                let k = card_ref(&self.ids, rec)?;
                self.c.script.random_exhausts.push(k);
                Ok(Applied::Ok)
            }
            "shuffle" => {
                let order = rec["cards"]
                    .as_array()
                    .ok_or("shuffle without cards")?
                    .iter()
                    .map(|v| card_ref(&self.ids, v))
                    .collect::<Result<Vec<_>, _>>()?;
                self.c.script.shuffles.push_back(order);
                Ok(Applied::Ok)
            }
            "play" => {
                let key = card_ref(&self.ids, rec)?;
                // By id only: the play record shows the card after it
                // resolved, which Razor Tooth may have upgraded.
                if self.pre_played.take().map(|(id, _)| id) == Some(key.0) {
                    return Ok(self.after_action());
                }
                self.play(rec)
            }
            // The game opened its card-select screen mid-play. The play that
            // opened it has not been reported yet, so apply it now from the
            // card (or potion) the record names, which puts the choice in
            // front of the policy while the player is still looking at it.
            "choice" => {
                if self.c.is_over() {
                    return Ok(Applied::Ok);
                }
                if self.c.pending.is_none() {
                    if !rec["card"].is_null() && self.pre_played.is_none() {
                        let key = card_ref(&self.ids, &rec["card"])?;
                        match self.play(&rec["card"])? {
                            // A play the sim cannot place (a target it would
                            // need) waits for the real record; the advice is
                            // late then.
                            Applied::Diverged(_) => return Ok(Applied::Ok),
                            _ => self.pre_played = Some(key),
                        }
                    } else if let Some(name) = rec["potion"].as_str().filter(|_| self.pre_potion.is_none()) {
                        if let Applied::Diverged(_) = self.use_potion(name, &Value::Null) {
                            return Ok(Applied::Ok);
                        }
                        self.pre_potion = self.ids.potions.get(name).copied();
                    }
                }
                adopt_offer_options(&mut self.c, &self.ids, rec);
                Ok(self.after_action())
            }
            // One card taken from the open choice, or `card: null` when the
            // selection closed. Written by the bridge, which answers the
            // game's selections itself; a hand-played recording has none and
            // the next snapshot settles the choice instead.
            "picked" => {
                let Some(p) = self.c.pending.clone() else { return Ok(Applied::Ok) };
                if rec["card"].is_null() {
                    if p.can_skip {
                        self.c.step(Action::Skip);
                        self.report.actions += 1;
                    }
                    return Ok(self.after_action());
                }
                let (id, up) = card_ref(&self.ids, &rec["card"])?;
                let ench = crate::gen::card_ench(&self.ids, &rec["card"])?.map(|e| (e.id, e.amount));
                let card = |uid: u32| crate::encode::option_card(&self.c, uid).filter(|k| k.id == id && k.upgraded == up);
                // Copies can differ by enchantment alone.
                let found = p
                    .options
                    .iter()
                    .position(|&uid| card(uid).is_some_and(|k| k.enchantment.map(|e| (e.id, e.amount)) == ench))
                    .or_else(|| p.options.iter().position(|&uid| card(uid).is_some()));
                let Some(i) = found else {
                    return Ok(Applied::Diverged(format!("game picked {id:?} (upgraded {up}); sim does not offer it")));
                };
                // The taken card's `gen` record follows; it is this pick.
                if p.then == Then::TakeOffer {
                    self.c.script.adopted_offers.push(id);
                }
                self.c.step(Action::Choose(i));
                self.report.actions += 1;
                Ok(self.after_action())
            }
            "potion" => {
                let name = rec["id"].as_str().unwrap_or("");
                if self.pre_potion.is_some() && self.pre_potion == self.ids.potions.get(name).copied() {
                    self.pre_potion = None;
                    self.at_decision = false;
                    return Ok(Applied::Ok);
                }
                Ok(self.use_potion(name, &rec["target"]))
            }
            "turn_start" => {
                self.at_decision = false;
                if rec["turn"].as_u64().unwrap_or(1) > 1 {
                    if self.c.is_over() || self.c.pending.is_some() {
                        return Ok(Applied::Diverged(
                            "game started a new turn; sim is over or waiting on a choice".into(),
                        ));
                    }
                    self.c.step(Action::EndTurn);
                    self.report.actions += 1;
                    clear_per_play(&mut self.c);
                }
                Ok(Applied::Ok)
            }
            "end" => {
                self.at_decision = false;
                // A loss during the enemy turn has no turn_start after it.
                if !self.c.is_over() && self.c.pending.is_none() {
                    self.c.step(Action::EndTurn);
                    self.report.actions += 1;
                }
                let won = rec["won"].as_bool().unwrap_or(false);
                let sim_won = self.c.outcome == Some(Outcome::Won);
                if won != sim_won || !self.c.is_over() {
                    return Ok(Applied::Diverged(format!("game ended (won {won}); sim outcome {:?}", self.c.outcome)));
                }
                Ok(Applied::Ended)
            }
            other => Err(format!("unknown record type {other}")),
        }
    }

    /// Apply a potion use: a `potion` record, or the potion of a `choice`
    /// record. `target` indexes the game's living enemies.
    fn use_potion(&mut self, name: &str, target: &Value) -> Applied {
        self.at_decision = false;
        let Some(&pid) = self.ids.potions.get(name) else {
            return Applied::Diverged(format!("unknown potion {name}"));
        };
        // Fairy in a Bottle drinks itself when a hit would kill you.
        // The sim spends it inside the death check rather than as an
        // action, and the next snapshot checks the belt emptied.
        if !pid.usable_in_combat() {
            return Applied::Ok;
        }
        let Some(slot) = self.c.potions.iter().position(|p| *p == Some(pid)) else {
            return Applied::Diverged(format!("game used {name}; sim has no such potion"));
        };
        let living: Vec<usize> = self.c.present_enemies().collect();
        let target = target.as_u64().and_then(|t| living.get(t as usize).copied());
        self.c.step(Action::UsePotion { slot, target });
        self.report.actions += 1;
        clear_per_play(&mut self.c);
        self.snecko_pending = pid == PotionId::SneckoOil;
        self.after_action()
    }

    /// Apply a `play` record (or the card of a `choice` record).
    fn play(&mut self, rec: &Value) -> Result<Applied, String> {
        self.at_decision = false;
        if self.c.is_over() {
            return Ok(Applied::Diverged("card played after the sim ended combat".into()));
        }
        let (id, up) = card_ref(&self.ids, rec)?;
        // Targets index the game's list of living enemies. A kill
        // shot loses its target before the hook fires; with one
        // enemy left that is unambiguous.
        let living: Vec<usize> = self.c.present_enemies().collect();
        let target = match rec["target"].as_u64() {
            Some(t) => living.get(t as usize).copied(),
            None if living.len() == 1 && crate::card::def(id).target == crate::types::TargetType::AnyEnemy => {
                Some(living[0])
            }
            None => None,
        };
        // Prefer the recorded hand index: identical cards are
        // interchangeable, but which one leaves the hand changes
        // the discard order and so later random picks.
        let want_idx = rec["hand_idx"].as_u64().map(|i| i as usize);
        // The play record shows the card after it resolved, so a card its
        // own play upgraded (Razor Tooth) reads as upgraded. The snapshot
        // before it shows it as it sat in hand.
        let up = want_idx
            .and_then(|i| self.last_hand.get(i))
            .filter(|k| k["id"] == rec["id"])
            .and_then(|k| k["up"].as_bool())
            .unwrap_or(up);
        // The cost the recording showed at that index, which is the
        // only thing separating two otherwise identical cards.
        let want_cost = want_idx.and_then(|i| self.last_hand.get(i)).and_then(|k| k["cost"].as_i64());
        let c = &self.c;
        let legal = c.legal_actions();
        let matches = |a: &Action| match *a {
            Action::PlayCard { hand_idx, target: t } => {
                let k = &c.player.hand[hand_idx];
                let cost_ok = want_cost.is_none_or(|w| w == i64::from(if k.def().x_cost { -1 } else { c.cost(k) }));
                k.id == id && k.upgraded == up && t == target && cost_ok
            }
            _ => false,
        };
        let action = legal
            .iter()
            .copied()
            .find(|a| matches(a) && matches!(a, Action::PlayCard { hand_idx, .. } if Some(*hand_idx) == want_idx))
            .or_else(|| legal.iter().copied().find(matches));
        let Some(action) = action else {
            return Ok(Applied::Diverged(format!(
                "game played {id:?} (upgraded {up}) at {target:?}; sim has no such legal play"
            )));
        };
        self.c.step(action);
        self.report.actions += 1;
        clear_per_play(&mut self.c);
        Ok(self.after_action())
    }

    /// A card choice left open by a play or a potion is itself a decision
    /// point: the game is showing the grid and the recording will not say
    /// which card was taken until the next snapshot.
    fn after_action(&mut self) -> Applied {
        if self.c.pending.is_some() && !self.c.is_over() {
            self.at_decision = true;
            Applied::Decision
        } else {
            Applied::Ok
        }
    }
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

    let mut r = Replayer::new(start, first_snap, ids.clone(), seed)?;
    for rec in &records[1..] {
        if matches!(r.feed(rec)?, Step::Diverged(_)) {
            return Ok(r.report.clone());
        }
    }
    r.finish()?;
    Ok(r.report.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::{RoomKind, Setup};
    use crate::relic::Relic;
    use crate::rng::Rng;

    fn card_json(id: CardId, up: bool) -> Value {
        json!({ "id": slug(&format!("{id:?}")), "up": up })
    }

    /// Play a random combat and log it in the recorder's format. The sim
    /// uses seed + 100, which a replay has to be given to match exactly.
    fn recorded_playout(seed: u64) -> String {
        {
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
                gold: 0,
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
            lines.iter().map(|l| l.to_string() + "\n").collect()
        }
    }

    /// A bridge command names its card well enough that the game, picking
    /// by the mod's rule (id and upgrade, then enchantment and cost), takes
    /// a copy the policy cannot tell from the one it meant.
    #[test]
    fn commands_name_the_card_meant() {
        let key = |c: &Combat, k: &Card| {
            let ench = k.enchantment.map(|e| (e.id, e.amount, e.disabled));
            (k.id, k.upgraded, ench, if k.def().x_cost { -1 } else { c.cost(k) })
        };
        // `Bridge.Find` in the mod.
        let find = |c: &Combat, cards: &[&Card], want: &Value| -> Option<(CardId, bool, Option<(EnchantmentId, i32, bool)>, i32)> {
            let named = cards.iter().filter(|k| slug(&format!("{:?}", k.id)) == want["id"] && Value::Bool(k.upgraded) == want["up"]);
            let score = |k: &&&Card| {
                let ench = k.enchantment.as_ref().map_or(Value::Null, ench_json);
                let cost = if k.def().x_cost { -1 } else { c.cost(k) };
                (ench == want["ench"]) as u8 * 2 + (want["cost"].is_null() || want["cost"] == cost) as u8
            };
            // First best, as `OrderByDescending(..).FirstOrDefault()` takes it.
            let best = named.clone().map(|k| score(&k)).max()?;
            named.into_iter().find(|k| score(k) == best).map(|k| key(c, k))
        };
        let mut rng = Rng::new(11);
        for i in 0..200u64 {
            let mut c = crate::gen::generate(&mut rng, 1 + (i % 16) as u32, Ascension(10)).combat(i);
            let mut m = vec![false; crate::encode::N_ACTIONS];
            let mut steps = 0;
            while !c.is_over() && steps < 300 {
                crate::encode::mask(&c, &mut m);
                let legal: Vec<usize> = (0..m.len()).filter(|&i| m[i]).collect();
                for &index in &legal {
                    let cmd = command(&c, index).expect("a legal action has a command");
                    match crate::encode::decode(&c, index).unwrap() {
                        Action::PlayCard { hand_idx, target } => {
                            let hand: Vec<&Card> = c.player.hand.iter().collect();
                            assert_eq!(find(&c, &hand, &cmd["card"]), Some(key(&c, &c.player.hand[hand_idx])), "{cmd}");
                            let living: Vec<usize> = c.present_enemies().collect();
                            assert_eq!(cmd["target"].as_u64().map(|t| living[t as usize]), target);
                        }
                        Action::Choose(o) => {
                            let p = c.pending.as_ref().unwrap();
                            let options: Vec<&Card> =
                                p.options.iter().filter_map(|&u| crate::encode::option_card(&c, u)).collect();
                            let meant = crate::encode::option_card(&c, p.options[o]).unwrap();
                            assert_eq!(find(&c, &options, &cmd["card"]), Some(key(&c, meant)), "{cmd}");
                        }
                        _ => {}
                    }
                }
                let a = crate::encode::decode(&c, legal[rng.next_int(legal.len())]).unwrap();
                c.step(a);
                steps += 1;
            }
        }
    }

    /// Record a sim playout in the recorder's format, then replay it in a
    /// sim with a different seed. The forced shuffles, HP, and moves must
    /// make the replay match at every decision point.
    #[test]
    fn sim_recording_round_trips() {
        let ids = Ids::new();
        for seed in 0..30u64 {
            let text = recorded_playout(seed);
            // Same RNG seed: every decision point must match.
            let report = replay_seeded(&text, &ids, seed + 100).unwrap();
            assert!(report.ok(), "seed {seed}: {:?}", report.divergence);
            assert!(report.snapshots > 0);
            // Cold seed: the forced opening shuffle and enemy HP must still
            // carry the first decision point.
            let report = replay(&text, &ids).unwrap();
            assert!(report.snapshots >= 1, "seed {seed}: {:?}", report.divergence);
        }
    }

    /// The advisor's path: records fed one at a time reach the same result
    /// as replaying the whole file, and every decision point the replayer
    /// reports has a legal action to recommend.
    #[test]
    fn feeding_records_one_at_a_time_matches_replay() {
        let ids = Ids::new();
        for seed in 0..10u64 {
            let text = recorded_playout(seed);
            let records: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
            let first_snap = records.iter().find(|r| r["t"] == "snapshot").unwrap();
            let mut r = Replayer::new(&records[0], first_snap, ids.clone(), seed + 100).unwrap();
            let mut decisions = 0;
            let mut ended = false;
            for rec in &records[1..] {
                // Live, the advisor flushes when the recorder goes quiet,
                // which is what makes a held snapshot a decision point.
                for step in [r.feed(rec).unwrap(), r.flush().unwrap()] {
                    match step {
                        Step::Decision => {
                            decisions += 1;
                            assert!(r.at_decision());
                            let mut m = vec![false; crate::encode::N_ACTIONS];
                            crate::encode::mask(r.combat(), &mut m);
                            let i = m.iter().position(|&b| b).expect("a legal action at a decision point");
                            assert!(crate::encode::describe(r.combat(), i).is_some());
                        }
                        Step::Ended => ended = true,
                        Step::Diverged(e) => panic!("seed {seed}: {e}"),
                        Step::Ok => {}
                    }
                }
            }
            r.flush().unwrap();
            assert!(ended);
            let whole = replay_seeded(&text, &ids, seed + 100).unwrap();
            let fed = r.report();
            assert!(fed.ok(), "seed {seed}: {:?}", fed.divergence);
            assert_eq!((fed.snapshots, fed.actions, fed.reseeds), (whole.snapshots, whole.actions, whole.reseeds));
            assert!(decisions >= whole.snapshots);
        }
    }
}

/// The bridge command that makes the game take action `index` (an index
/// into the encoded action space), or None if it is not legal now. The
/// inverse of the records above: cards are named by id, upgrade, cost and
/// enchantment rather than hand slot, since the sim's hand order is its own, and
/// targets index the game's living enemies. The mod's `Bridge.cs` reads it.
pub fn command(c: &Combat, index: usize) -> Option<Value> {
    let name = |id: &dyn std::fmt::Debug| slug(&format!("{id:?}"));
    let card = |k: &Card| {
        let mut v = json!({ "id": name(&k.id), "up": k.upgraded, "cost": if k.def().x_cost { -1 } else { c.cost(k) } });
        if let Some(e) = &k.enchantment {
            v["ench"] = ench_json(e);
        }
        v
    };
    let target = |t: Option<usize>| t.and_then(|e| c.present_enemies().position(|i| i == e));
    Some(match crate::encode::decode(c, index)? {
        Action::PlayCard { hand_idx, target: t } => {
            json!({ "cmd": "play", "card": card(&c.player.hand[hand_idx]), "target": target(t) })
        }
        Action::UsePotion { slot, target: t } => {
            json!({ "cmd": "potion", "slot": slot, "id": name(&c.potions.get(slot).copied().flatten()?), "target": target(t) })
        }
        Action::EndTurn => json!({ "cmd": "end" }),
        Action::Choose(i) => {
            let uid = *c.pending.as_ref()?.options.get(i)?;
            json!({ "cmd": "pick", "card": card(crate::encode::option_card(c, uid)?) })
        }
        Action::Skip => json!({ "cmd": "done" }),
    })
}
