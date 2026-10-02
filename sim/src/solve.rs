//! A solver for one fight with its future known: the real draw order and
//! dice stay in the state, so the fight is a single-player puzzle with no
//! chance, and a line that wins is proof the fight was winnable on that
//! seed. A measuring instrument for benchmarks, never for play (it reads
//! what no player can).
//!
//! No network: within a turn every distinct state is enumerated (states
//! keyed by everything, hidden order and dice included, so play orders
//! that meet are one state); across turns a beam keeps the next-turn starts
//! that rank best by any of a few plain measures (progress by the shaped
//! potential, the player's HP, the enemies' HP left), so defensive and
//! aggressive lines both survive the cut. It can miss a win; it cannot
//! report one that is not there (`replays` checks).

use std::collections::{HashMap, HashSet};

use crate::combat::{Action, Combat, Outcome};
use crate::env::{enemy_hp_taken, potential, Baseline};
use crate::mcts::key_of;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Next-turn starts kept per turn.
    pub beam: usize,
    /// Distinct mid-turn states explored from one turn start before its
    /// enumeration stops.
    pub turn_states: usize,
    pub max_turns: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self { beam: 300, turn_states: 3000, max_turns: 40 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Solution {
    /// The winning line from the root, when one was found.
    pub line: Option<Vec<Action>>,
    pub turns: u32,
    pub states: u64,
    /// A turn start's enumeration hit `turn_states`.
    pub capped: bool,
}

struct Start {
    combat: Box<Combat>,
    line: Vec<Action>,
}

/// Rank keys, higher is better.
fn ranks(c: &Combat, base: Baseline) -> [f32; 3] {
    [potential(c, base), c.player.creature.hp as f32 + c.player.creature.block as f32 * 0.5, enemy_hp_taken(c)]
}

pub fn solve(root: &Combat, cfg: &Config) -> Solution {
    let base = Baseline::of(root);
    let mut out = Solution::default();
    let mut frontier = vec![Start { combat: Box::new(root.clone()), line: vec![] }];
    for turn in 0..cfg.max_turns {
        out.turns = turn + 1;
        let mut next: HashMap<u64, Start> = HashMap::new();
        for start in &frontier {
            let mut seen = HashSet::new();
            let mut stack = vec![(start.combat.clone(), start.line.clone())];
            while let Some((c, line)) = stack.pop() {
                let t = c.player.turn;
                for a in c.legal_actions() {
                    let mut k = c.clone();
                    k.step(a);
                    out.states += 1;
                    let mut l = line.clone();
                    l.push(a);
                    match k.outcome {
                        Some(Outcome::Won) => {
                            out.line = Some(l);
                            return out;
                        }
                        Some(Outcome::Lost) => continue,
                        None => {}
                    }
                    let key = key_of(&k, true);
                    if k.player.turn > t {
                        next.entry(key).or_insert(Start { combat: k, line: l });
                    } else if seen.len() < cfg.turn_states {
                        if seen.insert(key) {
                            stack.push((k, l));
                        }
                    } else {
                        out.capped = true;
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = beam(next.into_values().collect(), base, cfg.beam);
    }
    out
}

/// The union of the best `width / 3` by each rank key.
fn beam(mut starts: Vec<Start>, base: Baseline, width: usize) -> Vec<Start> {
    if starts.len() <= width {
        return starts;
    }
    let keys: Vec<[f32; 3]> = starts.iter().map(|s| ranks(&s.combat, base)).collect();
    let mut keep = vec![false; starts.len()];
    for r in 0..3 {
        let mut idx: Vec<usize> = (0..starts.len()).collect();
        idx.sort_by(|&a, &b| keys[b][r].total_cmp(&keys[a][r]));
        idx.iter().take(width / 3).for_each(|&i| keep[i] = true);
    }
    let mut k = keep.into_iter();
    starts.retain(|_| k.next().unwrap());
    starts
}

/// Whether `line` played from `root` wins: the check a found line must pass.
pub fn replays(root: &Combat, line: &[Action]) -> bool {
    let mut c = root.clone();
    for &a in line {
        if c.is_over() || !c.legal_actions().contains(&a) {
            return false;
        }
        c.step(a);
    }
    c.outcome == Some(Outcome::Won)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::combat::EnemySpec;
    use crate::ids::{CardId, MonsterId};
    use crate::monster::Flags;
    use crate::types::Ascension;

    #[test]
    fn solves_a_fight_and_the_line_replays() {
        let deck: Vec<Card> = crate::ironclad_starter_deck();
        let enemies = [EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }];
        let mut c = Combat::new(&deck, 30, 80, 3, &enemies, Ascension(10), 11);
        c.player.hand = vec![Card::new(900, CardId::StrikeIronclad, false), Card::new(901, CardId::Bash, false)];
        let s = solve(&c, &Config::default());
        let line = s.line.expect("a starter deck beats a lone Nibbit");
        assert!(replays(&c, &line));
    }
}
