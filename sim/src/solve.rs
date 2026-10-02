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
    /// Without a win: the best shaped potential among the turn starts the
    /// search reached at `max_turns`, negative infinity when every line lost.
    pub best: f32,
    /// The same for `race`: the enemies' HP taken plus the player's HP left.
    pub best_race: f32,
}

/// How a race stands: the share of the enemies' HP taken plus the share of
/// the player's HP left (of the fight's starting max). The shaped potential
/// prices HP at a tenth in boss fights, so a planner scoring by it races
/// and dies; here staying alive counts as much as hitting.
pub fn race(c: &Combat, base: Baseline) -> f32 {
    enemy_hp_taken(c) + c.player.creature.hp.max(0) as f32 / base.max_hp().max(1) as f32
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
    solve_with(root, Baseline::of(root), cfg)
}

/// `solve` with rewards shaped against `base`, the fight's start, so the
/// frontier scores of searches from different states compare.
pub fn solve_with(root: &Combat, base: Baseline, cfg: &Config) -> Solution {
    let mut out = Solution { best: f32::NEG_INFINITY, best_race: f32::NEG_INFINITY, ..Solution::default() };
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
                        // Next-turn starts pile up to millions of states over a
                        // wide beam; cut them as they come, by the same ranks.
                        if next.len() >= 8 * cfg.beam {
                            let kept = beam(std::mem::take(&mut next).into_values().collect(), base, 2 * cfg.beam);
                            next = kept.into_iter().map(|s| (key_of(&s.combat, true), s)).collect();
                        }
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
            return out;
        }
        frontier = beam(next.into_values().collect(), base, cfg.beam);
    }
    out.best = frontier.iter().map(|s| potential(&s.combat, base)).fold(f32::NEG_INFINITY, f32::max);
    out.best_race = frontier.iter().map(|s| race(&s.combat, base)).fold(f32::NEG_INFINITY, f32::max);
    out
}

/// The union of the best `width / 3` by each rank key.
fn beam(starts: Vec<Start>, base: Baseline, width: usize) -> Vec<Start> {
    if starts.len() <= width {
        return starts;
    }
    let keep = rank_union(starts.iter().map(|s| &*s.combat).collect(), base, width);
    let mut keep = keep.into_iter().peekable();
    starts.into_iter().enumerate().filter_map(|(i, s)| (keep.next_if_eq(&i).is_some()).then_some(s)).collect()
}

/// Indices, ascending, of the union of the best `width / 3` (at least one)
/// of `states` by each rank key.
pub(crate) fn rank_union(states: Vec<&Combat>, base: Baseline, width: usize) -> Vec<usize> {
    let keys: Vec<[f32; 3]> = states.iter().map(|c| ranks(c, base)).collect();
    let mut keep = vec![false; states.len()];
    for r in 0..3 {
        let mut idx: Vec<usize> = (0..states.len()).collect();
        idx.sort_by(|&a, &b| keys[b][r].total_cmp(&keys[a][r]));
        idx.iter().take((width / 3).max(1)).for_each(|&i| keep[i] = true);
    }
    (0..states.len()).filter(|&i| keep[i]).collect()
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
