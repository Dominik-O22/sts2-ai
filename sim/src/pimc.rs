//! A turn planner that samples the hidden future and solves it (perfect
//! information Monte Carlo), as a candidate teacher stronger than the
//! searches built on the policy.
//!
//! It knows only what the player could know. Its worlds are the decision's
//! state with the draw pile reshuffled and the dice reseeded from seeds of
//! its own, the same K worlds for every candidate so they are compared on
//! the same luck. Candidates are the distinct ways to play out this turn,
//! enumerated in the first world and ranked as the solver ranks turn starts.
//! A candidate is scored in each world by replaying its line until the world
//! shows the player something the first world did not (a different draw):
//! from there, and from the next turn's start once the turn ends, the
//! clairvoyant solver (`solve`) plays on for `depth` turns. A win scores
//! highest, otherwise the best race standing it reached (enemy HP taken plus
//! the player's HP left; the shaped potential prices HP at a tenth in boss
//! fights, and a planner scoring by it raced and lost 60% of fights greedy
//! play wins). Depth stays short on
//! purpose: a solver that knows the future skips block the dice turn out
//! not to need, and that optimism grows with the turns it sees.
//!
//! The planner's line is followed while the real fight keeps showing what
//! the first world showed; at the first difference (a draw) it plans again.

use rayon::prelude::*;

use crate::combat::{Action, Combat, Outcome};
use crate::env::Baseline;
use crate::mcts::key_of;
use crate::rng::{CombatRngs, Rng};
use crate::solve::{self, solve_with};
use crate::turnsearch::state_key;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Sampled worlds each candidate is scored on.
    pub samples: usize,
    /// Candidates scored, the best ranked of those enumerated.
    pub candidates: usize,
    /// Distinct mid-turn states enumerated for candidates, and per turn
    /// start in the solver.
    pub turn_states: usize,
    /// Turns the solver plays on after the candidate's.
    pub depth: u32,
    pub beam: usize,
    pub seed: u64,
    /// Diagnostic only: every world is the real one (draw order and dice
    /// read, not sampled), to tell the planner's flaws from sampling's.
    pub peek: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { samples: 8, candidates: 12, turn_states: 1000, depth: 2, beam: 30, seed: 0, peek: false }
    }
}

/// A way to play out the turn: its actions from the decision, and the state
/// key the player sees after each in the first world.
#[derive(Clone, Debug)]
pub struct Plan {
    pub line: Vec<Action>,
    pub keys: Vec<u64>,
}

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The decision with its draw order and dice replaced: one possible future.
fn world(root: &Combat, seed: u64) -> Combat {
    let mut c = root.clone();
    c.player.draw.sort_by_key(|k| k.uid);
    Rng::new(seed).shuffle(&mut c.player.draw);
    c.rngs = CombatRngs::new(seed ^ 0xD1CE);
    c
}

/// Every distinct way to play out the turn in world `w`, as plans ending
/// with the step that ends the turn, ranked; or a line that wins outright.
fn candidates(w: &Combat, base: Baseline, cfg: &Config) -> Result<Vec<Plan>, Plan> {
    let t = w.player.turn;
    let mut seen = std::collections::HashSet::new();
    let mut ends: Vec<(Plan, Box<Combat>)> = vec![];
    let mut stack = vec![(Box::new(w.clone()), Plan { line: vec![], keys: vec![] })];
    while let Some((c, plan)) = stack.pop() {
        for a in c.legal_actions() {
            let mut k = c.clone();
            k.step(a);
            let mut p = plan.clone();
            p.line.push(a);
            p.keys.push(state_key(&k));
            match k.outcome {
                Some(Outcome::Won) => return Err(p),
                Some(Outcome::Lost) => continue,
                None => {}
            }
            if k.player.turn > t {
                ends.push((p, k));
            } else if seen.len() < cfg.turn_states && seen.insert(key_of(&k, true)) {
                stack.push((k, p));
            }
        }
    }
    let ranked = solve::rank_union(ends.iter().map(|(_, c)| &**c).collect(), base, cfg.candidates);
    Ok(ranked.into_iter().map(|i| ends[i].0.clone()).collect())
}

/// A win's score, above any race standing (`solve::race` is at most 2).
const WIN: f32 = 3.0;

/// `plan` in world `w`: followed while `w` shows what the first world did,
/// then the solver plays on. `WIN` for a win, else the best race standing
/// it reached (`solve::race`), -1 for a loss.
fn score(plan: &Plan, w: &Combat, base: Baseline, cfg: &Config) -> f32 {
    let mut c = w.clone();
    for (a, &key) in plan.line.iter().zip(&plan.keys) {
        if !c.legal_actions().contains(a) {
            break;
        }
        c.step(*a);
        match c.outcome {
            Some(Outcome::Won) => return WIN,
            Some(Outcome::Lost) => return -1.0,
            None => {}
        }
        if state_key(&c) != key {
            break;
        }
    }
    let s = solve_with(&c, base, &solve::Config { beam: cfg.beam, turn_states: cfg.turn_states, max_turns: cfg.depth });
    if s.line.is_some() {
        WIN
    } else {
        s.best_race.max(-1.0)
    }
}

/// The plan for the turn from `root`, rewards shaped against `base`: the
/// candidate with the best mean score over the sampled worlds.
pub fn plan(root: &Combat, base: Baseline, cfg: &Config) -> Option<Plan> {
    let seed = mix(cfg.seed, state_key(root));
    let worlds: Vec<Combat> = (0..cfg.samples.max(1)).map(|k| if cfg.peek { root.clone() } else { world(root, mix(seed, k as u64)) }).collect();
    let cands = match candidates(&worlds[0], base, cfg) {
        Ok(c) => c,
        Err(win) => return Some(win),
    };
    let scores: Vec<f32> = cands
        .par_iter()
        .map(|p| worlds.par_iter().map(|w| score(p, w, base, cfg)).sum::<f32>() / worlds.len() as f32)
        .collect();
    let best = (0..cands.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))?;
    Some(cands[best].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::combat::EnemySpec;
    use crate::ids::{CardId, MonsterId};
    use crate::monster::Flags;
    use crate::types::Ascension;

    /// The kill on the table is the plan, whatever the hidden future.
    #[test]
    fn plans_the_kill() {
        let deck: Vec<Card> = crate::ironclad_starter_deck();
        let enemies = [EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }];
        let mut c = Combat::new(&deck, 30, 80, 3, &enemies, Ascension(10), 3);
        c.player.hand = vec![Card::new(900, CardId::StrikeIronclad, false), Card::new(901, CardId::DefendIronclad, false)];
        c.enemies[0].creature.hp = 5;
        let p = plan(&c, Baseline::of(&c), &Config::default()).unwrap();
        let mut k = c.clone();
        for a in &p.line {
            k.step(*a);
        }
        assert_eq!(k.outcome, Some(Outcome::Won));
    }

    /// Two states the player cannot tell apart plan the same.
    #[test]
    fn draw_order_and_dice_stay_hidden() {
        let deck: Vec<Card> = crate::ironclad_starter_deck();
        let enemies = [EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }];
        let a = Combat::new(&deck, 40, 80, 3, &enemies, Ascension(10), 3);
        let mut b = a.clone();
        b.player.draw.reverse();
        b.rngs = CombatRngs::new(0xBAD5EED);
        let cfg = Config { samples: 4, candidates: 6, ..Config::default() };
        assert_eq!(plan(&a, Baseline::of(&a), &cfg).map(|p| p.line), plan(&b, Baseline::of(&b), &cfg).map(|p| p.line));
    }
}
