//! Exact turn search: every line of play from a decision to the end of the
//! player's turn, over distinct states instead of sampled sequences.
//!
//! `env::Forks` lets the policy finish each line, so a first action is
//! judged by how the policy would go on. Most of a turn is deterministic,
//! and identical cards and commuting plays reach the same state, so the
//! distinct states a turn can reach are few. This walks them all: a
//! decision node per distinct state (`state_key`), a max over the player's
//! actions, an expectation over chance. The turn's ends are leaves the
//! caller scores with the value head (`Search::leaves`, `Search::set_values`).
//!
//! The search knows only what the player could know. The key holds the
//! draw, discard and exhaust piles as multisets, never the draw order or
//! the dice, and no step follows the real RNG. How a step's chance is
//! handled, found by a probe step on fixed dice (`Rule`):
//! - no dice and no draws: one outcome;
//! - draws with no dice: every distinct multiset the draw can take, at its
//!   hypergeometric weight, up to `draw_cap` of them;
//! - dice, a draw past the cap, or a draw whose size depends on what came:
//!   `samples` reseeded outcomes, the draw pile shuffled for each;
//! - the end of the turn (the enemies' turn and the next hand):
//!   `end_samples` reseeded outcomes, the same seeds for every line, so
//!   lines are compared on the same enemy moves and the same shuffles.
//!
//! Every seed comes from the root's key and `Config::seed`, so the same
//! visible state searches the same whatever its real draw order or dice.
//!
//! States are expanded best first by the policy's prior of the line that
//! reaches them (`Search::run`'s `priors`), so when `max_states` bites the
//! lines cut are the ones the policy thinks least likely. A cut state is a
//! leaf the value head scores mid-turn. States where an enemy may still die
//! this turn go on past the cap, up to `quiesce_states`.

use std::collections::{BinaryHeap, HashMap};
use std::fmt::Write as _;
use std::time::Instant;

use rayon::prelude::*;

use crate::card::Card;
use crate::combat::{Action, Combat};
use crate::encode::{self, N_ACTIONS};
use crate::env::{potential, step_reward, Baseline};
use crate::ids::PowerId;
use crate::potion::Target as PotionTarget;
use crate::relic::RelicId;
use crate::rng::{CombatRngs, Rng};
use crate::types::CardType;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Distinct decision states found before expansion stops; the ones
    /// not expanded by then become leaves the value head scores mid-turn.
    pub max_states: usize,
    /// Past `max_states`, states where an enemy may die this turn
    /// (`lethal_possible`) are still expanded while fewer than this many
    /// states are found.
    pub quiesce_states: usize,
    /// Distinct multisets a draw is enumerated over; a draw with more is
    /// sampled.
    pub draw_cap: usize,
    /// Reseeded outcomes of a mid-turn step that rolls dice.
    pub samples: usize,
    /// Reseeded enemy turns (with the next hand) per end of turn.
    pub end_samples: usize,
    /// Wall-clock budget in microseconds, 0 for none. Past it the search
    /// stops expanding as if the state cap had bitten.
    pub max_micros: u64,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self { max_states: 500, quiesce_states: 1000, draw_cap: 32, samples: 4, end_samples: 8, max_micros: 0, seed: 0 }
    }
}

/// How a step's chance was handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Det,
    Drawn,
    Sampled,
    EndTurn,
}

/// Where an outcome lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    Node(usize),
    Leaf(usize),
    /// The fight ended: the reward on the way is all there is.
    Over,
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    p: f32,
    reward: f32,
    next: Next,
}

/// One of a state's actions, identical cards and potions merged.
#[derive(Clone, Debug)]
struct Edge {
    key: ActKey,
    outcomes: Vec<Outcome>,
}

struct Node {
    key: u64,
    /// The state until it is expanded (or cut).
    combat: Option<Box<Combat>>,
    /// Best prior of a line reaching it.
    prior: f64,
    /// Expansion order, for how many states a line needed.
    rank: Option<usize>,
    edges: Vec<Edge>,
    /// Left unexpanded: the leaf the value head scores it as.
    cut: Option<usize>,
}

/// A state the value head scores: the next turn's start, or a mid-turn
/// state the cap left unexpanded.
pub struct Leaf {
    pub combat: Box<Combat>,
    pub cut: bool,
}

/// What a search did, for the report.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Decision states expanded, and found in all (at most `max_states`,
    /// or `quiesce_states`, plus one expansion's worth).
    pub expanded: usize,
    pub nodes: usize,
    pub end_leaves: usize,
    pub cut_leaves: usize,
    /// Outcomes that landed on a state: arrivals over distinct states is
    /// the transposition merge.
    pub arrivals: usize,
    /// Action lines to the end of the turn the tree would hold without
    /// merging, chance outcomes counted apart.
    pub lines: f64,
    /// Actions stepped, by `Rule`.
    pub det: usize,
    pub drawn: usize,
    pub sampled: usize,
    pub end_turn: usize,
    /// The state cap (or the time budget) left states unexpanded.
    pub capped: bool,
    /// States expanded past `max_states` for a possible kill.
    pub quiesced: usize,
    pub micros: u64,
}

/// An action up to symmetry: two copies of a card that look alike play
/// alike, as do two slots holding the same potion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ActKey {
    Play(u64, Option<usize>),
    Potion(u64, Option<usize>),
    Choose(u64),
    Skip,
    End,
}

pub struct Search {
    /// Decision states, the root first.
    nodes: Vec<Node>,
    by_key: HashMap<u64, usize>,
    leaves: Vec<Leaf>,
    /// Filled by `set_values`.
    leaf_values: Vec<f32>,
    node_values: Vec<f32>,
    pub stats: Stats,
}

/// A seed derived from visible things only.
fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Dice for the probe step: any fixed seed. A step that leaves them as
/// they were rolled nothing.
const PROBE: u64 = 0x5EA2_C4ED;

/// FxHash-style hasher that also takes `Debug` output, so whole structs
/// hash without listing their fields (a field added later is covered).
#[derive(Default)]
struct Fx(u64);

impl Fx {
    fn word(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7C_C1_B7_27_22_0A_95);
    }
    fn finish(&self) -> u64 {
        mix(self.0, 0x2545_F491_4F6C_DD1D)
    }
}

impl std::fmt::Write for Fx {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for chunk in s.as_bytes().chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.word(u64::from_le_bytes(b));
        }
        self.word(0xFF);
        Ok(())
    }
}

/// Everything about a card but its uid.
fn card_fp(k: &Card) -> u64 {
    let mut k = k.clone();
    k.uid = 0;
    let mut h = Fx::default();
    let _ = write!(h, "{k:?}");
    h.finish()
}

/// The state as the player can know it: two states with one key play the
/// same from here. Piles are multisets (the hand too: the observation
/// sorts it), cards are known by content, not uid, and the dice, the draw
/// order and what only the replay reads are left out. Stats a card reads
/// in the same effect that set them are left out too, so commuting plays
/// meet. Not in it: the effect queue, which only holds something while a
/// choice is open (the card in play and the choice stand for it), and the
/// monsters' move-graph position beyond the next move.
pub fn state_key(c: &Combat) -> u64 {
    let mut h = Fx::default();
    let canon = |uid: u32| -> u32 {
        c.find_card(uid).or_else(|| c.player.offer.iter().find(|k| k.uid == uid)).map_or(uid | 1 << 31, |k| card_fp(k) as u32 & !(1 << 31))
    };
    let p = &c.player;
    let _ = write!(h, "{:?}", p.creature);
    for x in [p.energy as u64, p.base_max_energy as u64, p.turn as u64, c.round as u64, c.gold as u64] {
        h.word(x);
    }
    for pile in [&p.hand, &p.draw, &p.discard, &p.exhaust, &p.play, &p.offer] {
        let mut fps: Vec<u64> = pile.iter().map(card_fp).collect();
        fps.sort_unstable();
        h.word(0xC0FF_EE00 | fps.len() as u64);
        fps.into_iter().for_each(|f| h.word(f));
    }
    for e in &c.enemies {
        let m = &e.monster;
        let _ = write!(
            h,
            "{:?}{:?}{:?}{:?}{:?}{:?}{}{}{}{}",
            e.creature, m.id, m.flags, m.vars, m.next_move, m.next_move_name(), m.spawned_this_turn, e.slot, e.reviving, e.escaped
        );
    }
    let _ = write!(h, "{:?}{:?}{:?}{:?}{:?}{:?}{:?}", c.order, c.side, c.outcome, c.relics, c.potions, c.after, c.room);
    if let Some(q) = &c.pending {
        let mut opts: Vec<u32> = q.options.iter().map(|&u| canon(u)).collect();
        opts.sort_unstable();
        let _ = write!(h, "{opts:?}{:?}{}", q.then, q.can_skip);
    }
    let mut s = c.stats.clone();
    // Read in the effect that set them.
    s.last_drawn = None;
    s.card_dealt = (0, 0);
    s.last_card_hit = None;
    s.last_block_gained = 0.0;
    // Only the replay reads these.
    s.random_draw_inserts = 0;
    s.transformed.clear();
    s.transform_carried = false;
    s.procured_potions.clear();
    s.random_choice = false;
    s.gem_pick = None;
    s.cracked.clear();
    s.hp_rerolled.clear();
    // Only History Course replays the last card.
    if c.has_relic(RelicId::HistoryCourse) {
        for k in [&mut s.last_card, &mut s.last_turn_card].into_iter().flatten() {
            k.uid = 0;
        }
    } else {
        s.last_card = None;
        s.last_turn_card = None;
    }
    for v in [
        &mut s.block_plays_this_turn,
        &mut s.finished_this_turn,
        &mut s.finished_last_turn,
        &mut s.hatchets_played,
        &mut s.hatchets_played_last_turn,
        &mut s.rebound,
        &mut s.nostalgia_top,
        &mut s.selected,
        &mut s.transform_picks,
    ] {
        v.iter_mut().for_each(|u| *u = canon(*u));
        v.sort_unstable();
    }
    s.strangle_pending.iter_mut().for_each(|x| x.0 = canon(x.0));
    let _ = write!(h, "{s:?}");
    h.finish()
}

/// The draw pile in a fixed order by content: what a player who knows the
/// pile as a multiset would write down.
fn sort_draw(c: &mut Combat) {
    c.player.draw.sort_by_cached_key(|k| (card_fp(k), k.uid));
}

/// How many cards a step took off the top of the draw pile whose uids,
/// top first, are `order`: `None` when it took cards from further down.
fn drawn_prefix(order: &[u32], after: &Combat) -> Option<usize> {
    let left: std::collections::HashSet<u32> = after.player.draw.iter().map(|k| k.uid).collect();
    let m = order.iter().take_while(|u| !left.contains(u)).count();
    order[m..].iter().all(|u| left.contains(u)).then_some(m)
}

fn binom(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// Every way to take `m` cards from kinds of the sizes `sizes`, as counts
/// per kind; `None` past `cap` of them.
fn compositions(sizes: &[usize], m: usize, cap: usize) -> Option<Vec<Vec<usize>>> {
    fn go(sizes: &[usize], left: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>, cap: usize) -> bool {
        let Some((&n, rest)) = sizes.split_first() else {
            if left == 0 {
                out.push(cur.clone());
            }
            return out.len() <= cap;
        };
        let room: usize = rest.iter().sum();
        for x in left.saturating_sub(room)..=n.min(left) {
            cur.push(x);
            let ok = go(rest, left - x, cur, out, cap);
            cur.pop();
            if !ok {
                return false;
            }
        }
        true
    }
    let mut out = vec![];
    go(sizes, m, &mut vec![], &mut out, cap).then_some(out)
}

/// Whether an enemy may die this turn: some living enemy's HP and block
/// within twice what the attacks in hand deal on paper (Strength counted
/// per hit, X costs at the energy left), plus 30 per enemy-targeted potion.
/// Generous on purpose: it only decides what is searched past the cap.
fn lethal_possible(c: &Combat) -> bool {
    let p = &c.player;
    let strength = p.creature.power_amount(PowerId::Strength) as f64;
    let attacks: f64 = p
        .hand
        .iter()
        .filter(|k| k.ty() == CardType::Attack)
        .map(|k| {
            let v = k.vars();
            let times = if k.def().x_cost { p.energy.max(0) as f64 } else { 1.0 };
            (v.damage + strength).max(0.0) * v.hits.max(1) as f64 * times
        })
        .sum();
    let potions = c.potions.iter().flatten().filter(|id| id.usable_in_combat() && id.target() == PotionTarget::Enemy).count();
    let reach = 2.0 * attacks + 30.0 * potions as f64;
    c.living_enemies().any(|i| {
        let e = &c.enemies[i].creature;
        ((e.hp + e.block) as f64) <= reach
    })
}

fn act_key(c: &Combat, a: Action) -> ActKey {
    match a {
        Action::PlayCard { hand_idx, target } => ActKey::Play(card_fp(&c.player.hand[hand_idx]), target),
        Action::UsePotion { slot, target } => ActKey::Potion(c.potions[slot].map_or(u64::MAX, |id| id as u64), target),
        Action::Choose(i) => {
            let uid = c.pending.as_ref().map_or(0, |p| p.options[i]);
            ActKey::Choose(encode::option_card(c, uid).map_or(uid as u64, card_fp))
        }
        Action::Skip => ActKey::Skip,
        Action::EndTurn => ActKey::End,
    }
}

/// Prior of an action off the root: the root's prior of the same card,
/// potion or choice, or the root's mean prior where the root had none.
struct Priors {
    by_key: HashMap<ActKey, f64>,
    fallback: f64,
}

impl Priors {
    fn of(c: &Combat, probs: Option<&[f32]>) -> Self {
        let mut by_key = HashMap::new();
        let Some(probs) = probs else { return Self { by_key, fallback: 1.0 } };
        let hand = encode::hand_order(c);
        let choices = encode::choice_order(c);
        for a in c.legal_actions() {
            if let Some(i) = encode::index_of(c, &hand, &choices, a) {
                *by_key.entry(act_key(c, a)).or_insert(0.0) += probs[i].max(0.0) as f64;
            }
        }
        let fallback = if by_key.is_empty() { 1.0 } else { by_key.values().sum::<f64>() / by_key.len() as f64 };
        Self { by_key, fallback: fallback.max(1e-6) }
    }

    /// Normalized priors of a state's actions.
    fn of_actions(&self, keys: &[ActKey]) -> Vec<f64> {
        let raw: Vec<f64> = keys.iter().map(|k| self.by_key.get(k).copied().unwrap_or(self.fallback).max(1e-6)).collect();
        let total: f64 = raw.iter().sum();
        raw.into_iter().map(|x| x / total).collect()
    }
}

/// Mutable state of one run.
struct Builder<'a> {
    cfg: &'a Config,
    seed: u64,
    base: Baseline,
    turn: u32,
    priors: Priors,
    nodes: Vec<Node>,
    by_key: HashMap<u64, usize>,
    leaves: Vec<Leaf>,
    leaf_keys: HashMap<u64, usize>,
    heap: BinaryHeap<(u64, std::cmp::Reverse<usize>)>,
    stats: Stats,
}

impl Builder<'_> {
    /// The node for a state reached with `prior`, made if new.
    fn node(&mut self, c: Combat, prior: f64) -> usize {
        let key = state_key(&c);
        if let Some(&id) = self.by_key.get(&key) {
            let n = &mut self.nodes[id];
            if n.rank.is_none() && prior > n.prior {
                n.prior = prior;
                self.heap.push((prior.to_bits(), std::cmp::Reverse(id)));
            }
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node { key, combat: Some(Box::new(c)), prior, rank: None, edges: vec![], cut: None });
        self.by_key.insert(key, id);
        self.heap.push((prior.to_bits(), std::cmp::Reverse(id)));
        id
    }

    fn end_leaf(&mut self, c: Combat) -> usize {
        let key = state_key(&c);
        *self.leaf_keys.entry(key).or_insert_with(|| {
            self.leaves.push(Leaf { combat: Box::new(c), cut: false });
            self.stats.end_leaves += 1;
            self.leaves.len() - 1
        })
    }

    /// `c` after `a` with dice `seed` and, when `shuffle`, its draw pile
    /// sorted and shuffled by `seed` first.
    fn stepped(c: &Combat, a: Action, seed: u64, shuffle: bool) -> Combat {
        let mut k = c.clone();
        if shuffle {
            sort_draw(&mut k);
            Rng::new(seed).shuffle(&mut k.player.draw);
        }
        k.rngs = CombatRngs::new(seed ^ 0xD1CE);
        k.step(a);
        k
    }

    /// The outcomes of `a` in `c` with their probabilities, and the rule
    /// that found them.
    fn transition(&self, c: &Combat, a: Action, node_key: u64, act: ActKey) -> (Rule, Vec<(f64, Combat)>) {
        let mut probe = c.clone();
        probe.rngs = CombatRngs::new(PROBE);
        probe.step(a);
        if a == Action::EndTurn || probe.player.turn > c.player.turn {
            // The same seeds for every line: lines meet the same enemy
            // turns and the same next hands.
            let n = self.cfg.end_samples.max(1);
            let outs = (0..n).map(|s| (1.0 / n as f64, Self::stepped(c, a, mix(self.seed, 0xE4D0 + s as u64), true))).collect();
            return (Rule::EndTurn, outs);
        }
        let order: Vec<u32> = c.player.draw.iter().map(|k| k.uid).collect();
        let rolled = probe.rngs != CombatRngs::new(PROBE);
        match (rolled, drawn_prefix(&order, &probe)) {
            (false, Some(0)) => (Rule::Det, vec![(1.0, probe)]),
            (false, Some(m)) => match self.enumerate(c, a, m) {
                Some(outs) => (Rule::Drawn, outs),
                None => (Rule::Sampled, self.sample(c, a, node_key, act)),
            },
            _ => (Rule::Sampled, self.sample(c, a, node_key, act)),
        }
    }

    fn sample(&self, c: &Combat, a: Action, node_key: u64, act: ActKey) -> Vec<(f64, Combat)> {
        let n = self.cfg.samples.max(1);
        let mut h = Fx::default();
        let _ = write!(h, "{act:?}");
        let base = mix(mix(self.seed, node_key), h.finish());
        (0..n).map(|s| (1.0 / n as f64, Self::stepped(c, a, mix(base, s as u64), true))).collect()
    }

    /// A draw of `m` from the pile as a multiset: each distinct set of
    /// cards put on top in turn, weighted hypergeometrically. `None` when
    /// there are more than `draw_cap`, or a set drew some other count or
    /// rolled dice (a draw that depends on what it drew).
    fn enumerate(&self, c: &Combat, a: Action, m: usize) -> Option<Vec<(f64, Combat)>> {
        let mut draw: Vec<(u64, Card)> = c.player.draw.iter().map(|k| (card_fp(k), k.clone())).collect();
        draw.sort_by_key(|(f, k)| (*f, k.uid));
        let mut kinds: Vec<(usize, usize)> = vec![];
        for (i, (f, _)) in draw.iter().enumerate() {
            match kinds.last_mut() {
                Some((start, len)) if draw[*start].0 == *f => *len += 1,
                _ => kinds.push((i, 1)),
            }
        }
        let sizes: Vec<usize> = kinds.iter().map(|&(_, n)| n).collect();
        let combos = compositions(&sizes, m, self.cfg.draw_cap)?;
        let total = binom(draw.len(), m);
        let mut out = Vec::with_capacity(combos.len());
        for x in combos {
            let mut top = Vec::with_capacity(draw.len());
            let mut rest = vec![];
            for (&(start, n), &take) in kinds.iter().zip(&x) {
                top.extend(draw[start..start + take].iter().map(|(_, k)| k.clone()));
                rest.extend(draw[start + take..start + n].iter().map(|(_, k)| k.clone()));
            }
            top.extend(rest);
            let order: Vec<u32> = top.iter().map(|k| k.uid).collect();
            let mut k = c.clone();
            k.player.draw = top;
            k.rngs = CombatRngs::new(PROBE);
            k.step(a);
            if k.rngs != CombatRngs::new(PROBE) || drawn_prefix(&order, &k) != Some(m) {
                return None;
            }
            let p = sizes.iter().zip(&x).map(|(&n, &t)| binom(n, t)).product::<f64>() / total;
            out.push((p, k));
        }
        Some(out)
    }

    fn expand(&mut self, id: usize) {
        let c = self.nodes[id].combat.take().expect("expanded twice");
        self.nodes[id].rank = Some(self.stats.expanded);
        self.stats.expanded += 1;
        let node_key = self.nodes[id].key;
        let prior = self.nodes[id].prior;
        let mut seen = std::collections::HashSet::new();
        let acts: Vec<(Action, ActKey)> =
            c.legal_actions().into_iter().map(|a| (a, act_key(&c, a))).filter(|(_, k)| seen.insert(*k)).collect();
        let keys: Vec<ActKey> = acts.iter().map(|&(_, k)| k).collect();
        let pa = self.priors.of_actions(&keys);
        let before = potential(&c, self.base);
        let mut edges = Vec::with_capacity(acts.len());
        for (&(a, key), &pa) in acts.iter().zip(&pa) {
            let (rule, outs) = self.transition(&c, a, node_key, key);
            match rule {
                Rule::Det => self.stats.det += 1,
                Rule::Drawn => self.stats.drawn += 1,
                Rule::Sampled => self.stats.sampled += 1,
                Rule::EndTurn => self.stats.end_turn += 1,
            }
            let mut outcomes: Vec<Outcome> = vec![];
            for (p, child) in outs {
                let reward = step_reward(before, &child, self.base, child.is_over());
                let next = if child.is_over() {
                    Next::Over
                } else if child.player.turn > self.turn {
                    Next::Leaf(self.end_leaf(child))
                } else {
                    Next::Node(self.node(child, prior * pa * p))
                };
                if next != Next::Over {
                    self.stats.arrivals += 1;
                }
                match outcomes.iter_mut().find(|o| o.next == next && next != Next::Over) {
                    Some(o) => o.p += p as f32,
                    None => outcomes.push(Outcome { p: p as f32, reward, next }),
                }
            }
            edges.push(Edge { key, outcomes });
        }
        self.nodes[id].edges = edges;
    }

    fn run(mut self, start: Instant) -> Search {
        while let Some((bits, std::cmp::Reverse(id))) = self.heap.pop() {
            let n = &self.nodes[id];
            if n.rank.is_some() || bits != n.prior.to_bits() {
                continue;
            }
            let over_time = self.cfg.max_micros > 0 && start.elapsed().as_micros() as u64 > self.cfg.max_micros;
            if (self.nodes.len() >= self.cfg.max_states && id != 0) || over_time {
                self.stats.capped = true;
                let quiesce = !over_time
                    && self.nodes.len() < self.cfg.quiesce_states
                    && lethal_possible(n.combat.as_ref().expect("unexpanded node holds its state"));
                if !quiesce {
                    continue;
                }
                self.stats.quiesced += 1;
            }
            self.expand(id);
        }
        for n in &mut self.nodes {
            if let Some(c) = n.combat.take() {
                n.cut = Some(self.leaves.len());
                self.leaves.push(Leaf { combat: c, cut: true });
                self.stats.cut_leaves += 1;
            }
        }
        self.stats.nodes = self.nodes.len();
        self.stats.lines = lines(&self.nodes);
        self.stats.micros = start.elapsed().as_micros() as u64;
        Search { nodes: self.nodes, by_key: self.by_key, leaves: self.leaves, leaf_values: vec![], node_values: vec![], stats: self.stats }
    }
}

/// Lines to the end of the turn through each node, leaves counting one.
fn lines(nodes: &[Node]) -> f64 {
    fn go(nodes: &[Node], id: usize, memo: &mut [f64]) -> f64 {
        if memo[id] >= 0.0 {
            return memo[id];
        }
        memo[id] = 0.0;
        let n: f64 = nodes[id]
            .edges
            .iter()
            .flat_map(|e| &e.outcomes)
            .map(|o| match o.next {
                Next::Node(c) => go(nodes, c, memo),
                _ => 1.0,
            })
            .sum();
        memo[id] = if nodes[id].edges.is_empty() { 1.0 } else { n };
        memo[id]
    }
    let mut memo = vec![-1.0; nodes.len()];
    go(nodes, 0, &mut memo)
}

impl Search {
    /// Search the rest of `root`'s turn. `priors` are the policy's
    /// probabilities over the action space at the root, which order the
    /// expansion; without them it goes breadth first.
    pub fn run(root: &Combat, priors: Option<&[f32]>, cfg: &Config) -> Self {
        let start = Instant::now();
        let mut c = root.clone();
        c.script = Default::default();
        c.shuffle_log = Default::default();
        sort_draw(&mut c);
        let key = state_key(&c);
        if let Some(p) = priors {
            assert_eq!(p.len(), N_ACTIONS, "priors must cover the action space");
        }
        let mut b = Builder {
            cfg,
            seed: mix(cfg.seed, key),
            base: Baseline::of(root),
            turn: root.player.turn,
            priors: Priors::of(&c, priors),
            nodes: vec![],
            by_key: HashMap::new(),
            leaves: vec![],
            leaf_keys: HashMap::new(),
            heap: BinaryHeap::new(),
            stats: Stats::default(),
        };
        b.node(c, 1.0);
        b.run(start)
    }

    /// `run` over many roots at once, one thread each.
    pub fn run_all(roots: &[&Combat], priors: &[Option<&[f32]>], cfg: &Config) -> Vec<Self> {
        roots.par_iter().zip(priors.par_iter()).map(|(r, p)| Self::run(r, *p, cfg)).collect()
    }

    /// The states the value head scores, in the order `set_values` takes
    /// their values.
    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }

    /// Take the value head's value of every leaf and back them up: a
    /// state is worth its best action, an action the expectation over its
    /// outcomes of the shaped reward on the way plus what it leads to.
    pub fn set_values(&mut self, leaf_values: &[f32]) {
        assert_eq!(leaf_values.len(), self.leaves.len(), "one value per leaf");
        let mut memo = vec![f32::NAN; self.nodes.len()];
        let mut on_stack = vec![false; self.nodes.len()];
        for id in 0..self.nodes.len() {
            self.v(id, leaf_values, &mut memo, &mut on_stack);
        }
        self.node_values = memo;
        self.leaf_values = leaf_values.to_vec();
    }

    /// The expanded node for a state `c` could be in, if the search holds one.
    pub fn find(&self, c: &Combat) -> Option<usize> {
        self.by_key.get(&state_key(c)).copied().filter(|&id| self.nodes[id].rank.is_some())
    }

    /// Each of `c`'s legal actions that the action space holds, by index,
    /// with its value: `c` is the root or a state `find` found. `None`
    /// when the search holds no expanded node for it or has no values yet.
    pub fn action_values(&self, c: &Combat) -> Option<Vec<(usize, f32)>> {
        let id = self.find(c).filter(|_| !self.node_values.is_empty())?;
        let node = &self.nodes[id];
        let (hand, choices) = (encode::hand_order(c), encode::choice_order(c));
        Some(
            c.legal_actions()
                .into_iter()
                .filter_map(|a| {
                    let index = encode::index_of(c, &hand, &choices, a)?;
                    let key = act_key(c, a);
                    node.edges.iter().find(|e| e.key == key).map(|e| (index, self.q(e)))
                })
                .collect(),
        )
    }

    /// The best line from the root, following each chance node's likeliest
    /// outcome: the latest expansion rank a state on it had (how many
    /// states the line needed), and whether it ran into a cut state.
    pub fn best_line(&self) -> (usize, bool) {
        let mut id = 0;
        let mut rank = 0;
        loop {
            let n = &self.nodes[id];
            if n.cut.is_some() {
                return (rank, true);
            }
            rank = rank.max(n.rank.unwrap_or(0));
            let best = n.edges.iter().max_by(|a, b| self.q(a).total_cmp(&self.q(b)));
            match best.and_then(|e| e.outcomes.iter().max_by(|a, b| a.p.total_cmp(&b.p))).map(|o| o.next) {
                Some(Next::Node(next)) => id = next,
                _ => return (rank, false),
            }
        }
    }

    fn value_of(&self, next: Next) -> f32 {
        match next {
            Next::Over => 0.0,
            Next::Node(n) => self.node_values[n],
            Next::Leaf(l) => self.leaf_values[l],
        }
    }

    fn q(&self, e: &Edge) -> f32 {
        e.outcomes.iter().map(|o| o.p * (o.reward + self.value_of(o.next))).sum()
    }

    fn v(&self, id: usize, values: &[f32], memo: &mut [f32], on_stack: &mut [bool]) -> f32 {
        if !memo[id].is_nan() {
            return memo[id];
        }
        if let Some(l) = self.nodes[id].cut {
            memo[id] = values[l];
            return memo[id];
        }
        // A state that reaches itself: no line in the sim does, but a
        // loop must not recurse forever. It counts as ending there.
        if on_stack[id] {
            return 0.0;
        }
        on_stack[id] = true;
        let q = |o: &Outcome, memo: &mut [f32], on_stack: &mut [bool]| {
            o.p * (o.reward
                + match o.next {
                    Next::Over => 0.0,
                    Next::Leaf(l) => values[l],
                    Next::Node(n) => self.v(n, values, memo, on_stack),
                })
        };
        let mut best = f32::NEG_INFINITY;
        for e in &self.nodes[id].edges {
            let mut sum = 0.0;
            for o in &e.outcomes {
                sum += q(o, memo, on_stack);
            }
            best = best.max(sum);
        }
        on_stack[id] = false;
        memo[id] = if best.is_finite() { best } else { 0.0 };
        memo[id]
    }
}

/// Encode `combats` into consecutive rows of the buffers, in parallel.
pub fn encode_all(combats: &[&Combat], floats: &mut [f32], ids: &mut [i64], mask: &mut [bool]) {
    use crate::encode::{N_FLOATS, N_IDS};
    combats
        .par_iter()
        .zip(floats.par_chunks_mut(N_FLOATS))
        .zip(ids.par_chunks_mut(N_IDS))
        .zip(mask.par_chunks_mut(N_ACTIONS))
        .for_each(|(((c, f), i), m)| encode::encode(c, f, i, m));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::EnemySpec;
    use crate::gen::generate;
    use crate::ids::{CardId, MonsterId};
    use crate::monster::Flags;
    use crate::types::Ascension;

    /// A stand-in for the value head: a function of what the player sees.
    fn values(s: &Search) -> Vec<f32> {
        s.leaves()
            .iter()
            .map(|l| {
                let c = &l.combat;
                let enemies: i32 = c.enemies.iter().map(|e| e.creature.hp.max(0) + e.creature.block).sum();
                (c.player.creature.hp - enemies) as f32 / 100.0 + c.player.hand.len() as f32 / 50.0
            })
            .collect()
    }

    fn nibbit(hand: &[CardId], hp: i32) -> Combat {
        let enemy = EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } };
        let mut c = Combat::new(&crate::ironclad_starter_deck(), 80, 80, 3, &[enemy], Ascension(10), 1);
        c.player.hand = hand.iter().enumerate().map(|(i, &id)| Card::new(900 + i as u32, id, false)).collect();
        c.player.energy = 3;
        let e = &mut c.enemies[0].creature;
        (e.hp, e.max_hp) = (hp, hp);
        c
    }

    fn value_of(s: &mut Search, c: &Combat, hand_idx: usize) -> f32 {
        let hand = encode::hand_order(c);
        let a = Action::PlayCard { hand_idx, target: Some(0) };
        let index = encode::index_of(c, &hand, &[], a).unwrap();
        s.set_values(&vec![0.0; s.leaves().len()]);
        s.action_values(c).unwrap().into_iter().find(|(i, _)| *i == index).unwrap().1
    }

    /// Root action values with `values` at the leaves, as text to compare.
    fn root_values(mut s: Search, root: &Combat) -> String {
        s.set_values(&values(&s));
        format!("{:?}", s.action_values(root).unwrap())
    }

    /// Bash first makes the Strike after it hit Vulnerable: 8 + 9 against
    /// 6 + 8 the other way round. With nothing but shaped reward to go on,
    /// the search opens with Bash, by the three HP it gains.
    #[test]
    fn order_matters_and_the_search_sees_it() {
        let c = nibbit(&[CardId::StrikeIronclad, CardId::Bash], 60);
        let mut s = Search::run(&c, None, &Config::default());
        let (strike, bash) = (value_of(&mut s, &c, 0), value_of(&mut s, &c, 1));
        let three_hp = 0.5 * 3.0 / c.stats.enemy_start_hp as f32;
        assert!((bash - strike - three_hp).abs() < 1e-4, "bash {bash} strike {strike}, expected a gap of {three_hp}");
        assert!(!s.stats.capped);
    }

    /// Strike, Defend, Strike and Defend, Strike, Strike end in one state:
    /// three cards and 3 energy make six distinct states (what was played:
    /// none, S, D, SS, SD, SSD), not one per order.
    #[test]
    fn commuting_plays_meet() {
        let c = nibbit(&[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::StrikeIronclad], 60);
        let s = Search::run(&c, None, &Config::default());
        assert_eq!(s.stats.nodes, 6, "{:?}", s.stats);
        assert!(s.stats.arrivals > s.stats.nodes);
    }

    /// Fights at several floors, with the turn's first decision as root.
    fn roots() -> Vec<Combat> {
        let mut rng = Rng::new(21);
        [3, 8, 14, 20, 30, 40].iter().enumerate().map(|(k, &floor)| generate(&mut rng, floor, Ascension(10)).combat(k as u64 + 5)).collect()
    }

    fn small() -> Config {
        Config { max_states: 300, quiesce_states: 400, ..Default::default() }
    }

    #[test]
    fn same_root_same_search() {
        for c in roots() {
            let (a, b) = (Search::run(&c, None, &small()), Search::run(&c, None, &small()));
            let keys = |s: &Search| s.leaves().iter().map(|l| state_key(&l.combat)).collect::<Vec<_>>();
            assert_eq!(keys(&a), keys(&b));
            assert_eq!(root_values(a, &c), root_values(b, &c));
        }
    }

    /// What the player cannot see changes nothing: the draw pile in another
    /// order and other dice give the same leaves and the same values.
    #[test]
    fn draw_order_and_dice_stay_hidden() {
        let (mut drawn, mut sampled) = (0, 0);
        for c in roots() {
            let mut other = c.clone();
            other.player.draw.reverse();
            let n = other.player.draw.len();
            other.player.draw.rotate_left(3.min(n));
            other.rngs = CombatRngs::new(0xBAD5EED);
            let (a, b) = (Search::run(&c, None, &small()), Search::run(&other, None, &small()));
            let keys = |s: &Search| s.leaves().iter().map(|l| state_key(&l.combat)).collect::<Vec<_>>();
            assert_eq!(keys(&a), keys(&b));
            (drawn, sampled) = (drawn + a.stats.drawn, sampled + a.stats.sampled);
            assert_eq!(root_values(a, &c), root_values(b, &other));
        }
        assert!(drawn > 0 && sampled > 0, "the roots never drew ({drawn}) or rolled dice ({sampled}) mid-turn");
    }
}
