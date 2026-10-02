//! Tree search over a whole fight, many turns deep (AlphaZero-style PUCT).
//!
//! `turnsearch` walks every line to the end of the current turn and lets
//! the value head judge the rest; the losses that cost runs are races lost
//! over several turns. Here each simulation walks down the tree, picking
//! actions by PUCT over the policy's priors, until it reaches a state the
//! tree has not seen; the caller scores that state with the network
//! (`Tree::leaf`, `Tree::expand`), many trees' leaves in one batch.
//!
//! The search knows only what the player could know. Nodes are keyed by
//! `turnsearch::state_key`, what the player can tell apart, and hold the
//! state with its draw pile sorted; a simulation entering a node plays from
//! a copy with the draw pile reshuffled and the dice reseeded from its own
//! seed, never the real ones. An action's children are the outcomes
//! simulations reached (chance: dice, draws, enemy moves). The next hand
//! after an end of turn is nearly always new, so without a limit every
//! simulation would stop at a fresh leaf there and the tree would never
//! see a second turn; progressive widening allows an action about
//! sqrt(visits) distinct outcomes, and past that a simulation goes on into
//! one already seen, drawn by how often it came up. An action's value is
//! the mean over the simulations through it, the expectation over chance.
//!
//! Returns are on the value head's scale: the shaped reward on the way
//! (`env::step_reward` against the fight's baseline) plus the leaf's value;
//! a simulation that ends the fight has collected its terminal reward.

use crate::combat::Combat;
use crate::encode::{self, N_ACTIONS};
use crate::env::{potential, step_reward, Baseline};
use crate::rng::{CombatRngs, Rng};
use crate::turnsearch::state_key;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Exploration weight in PUCT.
    pub c_puct: f32,
    /// Distinct outcomes an action may have: `widen * visits^widen_exp`,
    /// at least one. Off (infinite) by default: on held-out act 3 bosses
    /// widening at 1 searched worse than none (74.8% vs 77.5% won).
    pub widen: f32,
    pub widen_exp: f32,
    /// Steps a simulation may take before its state is scored as a leaf.
    pub max_depth: u32,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self { c_puct: 1.25, widen: f32::INFINITY, widen_exp: 0.5, max_depth: 200, seed: 0 }
    }
}

/// An outcome an action led to: its node, the shaped reward of getting
/// there (a function of what the player can see, so one per outcome), and
/// how often simulations reached it.
struct Child {
    key: u64,
    node: u32,
    reward: f32,
    count: u32,
}

struct Edge {
    action: u16,
    prior: f32,
    visits: u32,
    total: f64,
    children: Vec<Child>,
}

struct Node {
    /// The state, draw pile sorted: what the player knows, plus one hidden
    /// draw order and dice that simulations replace.
    state: Box<Combat>,
    /// The network's value of the state, set on expansion.
    value: f32,
    visits: u32,
    edges: Vec<Edge>,
    expanded: bool,
}

pub struct Tree {
    base: Baseline,
    cfg: Config,
    nodes: Vec<Node>,
    sims: u64,
    /// The node the last `descend` left for the network, and the path to
    /// it: (node, edge, reward of that step) from the root down.
    pending: Option<(u32, Vec<(u32, u32, f32)>)>,
    /// Scratch state simulations play in.
    scratch: Box<Combat>,
    /// Range of the action values seen, to put Q on PUCT's scale.
    q_min: f32,
    q_max: f32,
}

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sorted(c: &Combat) -> Box<Combat> {
    let mut c = Box::new(c.clone());
    c.player.draw.sort_by_key(|k| k.uid);
    c
}

impl Tree {
    /// A search from `root`, rewards shaped against the fight's `base`. The
    /// seed mixes in the root's key, so one visible state searches the same
    /// whatever its hidden draw order and dice.
    pub fn new(root: &Combat, base: Baseline, cfg: Config) -> Self {
        let state = sorted(root);
        let cfg = Config { seed: mix(cfg.seed, state_key(&state)), ..cfg };
        let scratch = state.clone();
        let root = Node { state, value: 0.0, visits: 0, edges: vec![], expanded: false };
        Self { base, cfg, nodes: vec![root], sims: 0, pending: None, scratch, q_min: f32::INFINITY, q_max: f32::NEG_INFINITY }
    }

    /// Q on [0, 1] over the values seen so far.
    fn scaled(&self, q: f32) -> f32 {
        if self.q_max > self.q_min {
            (q - self.q_min) / (self.q_max - self.q_min)
        } else {
            0.5
        }
    }

    /// PUCT's pick at node `id`: an unvisited action counts as worth the
    /// node's own value.
    fn select(&self, id: u32) -> u32 {
        let n = &self.nodes[id as usize];
        let sqrt_n = (n.visits.max(1) as f32).sqrt();
        let mut best = (f32::NEG_INFINITY, 0);
        for (k, e) in n.edges.iter().enumerate() {
            let q = if e.visits > 0 { (e.total / e.visits as f64) as f32 } else { n.value };
            let score = self.scaled(q) + self.cfg.c_puct * e.prior * sqrt_n / (1.0 + e.visits as f32);
            if score > best.0 {
                best = (score, k as u32);
            }
        }
        best.1
    }

    /// Run one simulation down to a state the tree has not scored. Returns
    /// whether it left one for the network (`leaf`, then `expand`); a
    /// simulation that ended the fight is backed up already.
    pub fn descend(&mut self) -> bool {
        debug_assert!(self.pending.is_none(), "expand the last leaf first");
        let seed = mix(self.cfg.seed, self.sims);
        self.sims += 1;
        let mut rng = Rng::new(seed);
        let mut node = 0u32;
        let mut path = vec![];
        for depth in 0..self.cfg.max_depth {
            if !self.nodes[node as usize].expanded {
                break;
            }
            let edge = self.select(node);
            let n = &self.nodes[node as usize];
            let index = n.edges[edge as usize].action;
            self.scratch.clone_from(&n.state);
            let world = mix(seed, depth as u64);
            Rng::new(world).shuffle(&mut self.scratch.player.draw);
            self.scratch.rngs = CombatRngs::new(world ^ 0xD1CE);
            let Some(action) = encode::decode(&self.scratch, index as usize) else {
                break;
            };
            let before = potential(&self.scratch, self.base);
            self.scratch.step(action);
            let over = self.scratch.is_over();
            let reward = step_reward(before, &self.scratch, self.base, over);
            if over {
                path.push((node, edge, reward));
                self.backup(&path, 0.0);
                return false;
            }
            let key = state_key(&self.scratch);
            let e = &self.nodes[node as usize].edges[edge as usize];
            let room = (self.cfg.widen * (e.visits as f32 + 1.0).powf(self.cfg.widen_exp)).ceil().max(1.0) as usize;
            let next = match e.children.iter().position(|ch| ch.key == key) {
                Some(k) => k,
                None if e.children.len() < room => {
                    let child = self.nodes.len() as u32;
                    let state = sorted(&self.scratch);
                    self.nodes.push(Node { state, value: 0.0, visits: 0, edges: vec![], expanded: false });
                    let e = &mut self.nodes[node as usize].edges[edge as usize];
                    e.children.push(Child { key, node: child, reward, count: 0 });
                    e.children.len() - 1
                }
                // Widened enough: go on into an outcome already seen, as
                // often as it came up.
                None => {
                    let total: u32 = e.children.iter().map(|ch| ch.count).sum();
                    let mut pick = rng.next_int(total.max(1) as usize) as u32;
                    e.children.iter().position(|ch| pick < ch.count || ch.count == 0 || { pick -= ch.count; false }).unwrap_or(0)
                }
            };
            let ch = &mut self.nodes[node as usize].edges[edge as usize].children[next];
            ch.count += 1;
            path.push((node, edge, ch.reward));
            node = ch.node;
        }
        self.pending = Some((node, path));
        true
    }

    /// The state the last `descend` left for the network.
    pub fn leaf(&self) -> Option<&Combat> {
        self.pending.as_ref().map(|(node, _)| &*self.nodes[*node as usize].state)
    }

    /// The network's read of the leaf: `priors` over the action space (the
    /// policy's probabilities) and `value`. Expands it and backs the value up.
    pub fn expand(&mut self, priors: &[f32], value: f32) {
        let (node, path) = self.pending.take().expect("descend left no leaf");
        let n = &mut self.nodes[node as usize];
        if !n.expanded {
            let mut legal = [false; N_ACTIONS];
            encode::mask(&n.state, &mut legal);
            n.edges = (0..N_ACTIONS)
                .filter(|&a| legal[a])
                .map(|a| Edge { action: a as u16, prior: priors[a], visits: 0, total: 0.0, children: vec![] })
                .collect();
            n.value = value;
            n.expanded = !n.edges.is_empty();
        }
        let leaf_value = n.value;
        self.backup(&path, leaf_value);
    }

    fn backup(&mut self, path: &[(u32, u32, f32)], leaf_value: f32) {
        let mut g = leaf_value as f64;
        for &(node, edge, reward) in path.iter().rev() {
            g += reward as f64;
            let n = &mut self.nodes[node as usize];
            n.visits += 1;
            let e = &mut n.edges[edge as usize];
            e.visits += 1;
            e.total += g;
            let q = (e.total / e.visits as f64) as f32;
            self.q_min = self.q_min.min(q);
            self.q_max = self.q_max.max(q);
        }
    }

    /// (action index, visits, mean return) for each of the root's actions.
    pub fn root_stats(&self) -> Vec<(usize, u32, f32)> {
        let root = &self.nodes[0];
        root.edges.iter().map(|e| (e.action as usize, e.visits, if e.visits > 0 { (e.total / e.visits as f64) as f32 } else { root.value })).collect()
    }

    /// The most visited action at the root, ties to the higher mean return.
    pub fn best_action(&self) -> Option<usize> {
        self.root_stats().into_iter().max_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2))).map(|(a, _, _)| a)
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Player turns past the root's that the tree reaches.
    pub fn turns_deep(&self) -> u32 {
        let root = self.nodes[0].state.player.turn;
        self.nodes.iter().map(|n| n.state.player.turn.saturating_sub(root)).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::combat::{Action, EnemySpec};
    use crate::ids::{CardId, MonsterId};
    use crate::monster::Flags;
    use crate::types::Ascension;

    fn nibbit(hand: &[CardId], hp: i32) -> Combat {
        let deck: Vec<Card> = crate::ironclad_starter_deck();
        let mut c = Combat::new(&deck, 80, 80, 3, &[EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }], Ascension(10), 7);
        c.player.hand = hand.iter().enumerate().map(|(i, &id)| Card::new(800 + i as u32, id, false)).collect();
        c.enemies[0].creature.hp = hp;
        c
    }

    /// Runs `sims` simulations with flat priors and a flat value.
    fn search(c: &Combat, sims: usize, seed: u64) -> Tree {
        let mut t = Tree::new(c, Baseline::of(c), Config { seed, ..Config::default() });
        for _ in 0..sims {
            if t.descend() {
                let mut legal = [false; N_ACTIONS];
                encode::mask(t.leaf().unwrap(), &mut legal);
                let k = legal.iter().filter(|&&l| l).count().max(1) as f32;
                let priors: Vec<f32> = legal.iter().map(|&l| if l { 1.0 / k } else { 0.0 }).collect();
                t.expand(&priors, 0.0);
            }
        }
        t
    }

    /// With the kill on the table, the search finds it through the shaped
    /// reward alone: the strike that ends the fight collects the win.
    #[test]
    fn finds_the_kill() {
        let c = nibbit(&[CardId::StrikeIronclad, CardId::DefendIronclad], 5);
        let t = search(&c, 200, 1);
        let strike = crate::turnsearch::index(&c, Action::PlayCard { hand_idx: 0, target: Some(0) }).unwrap();
        assert_eq!(t.best_action(), Some(strike), "{:?}", t.root_stats());
    }

    /// Two states the player cannot tell apart (draw order and dice
    /// differ) search identically.
    #[test]
    fn draw_order_and_dice_stay_hidden() {
        let a = nibbit(&[CardId::StrikeIronclad, CardId::DefendIronclad], 30);
        let mut b = a.clone();
        b.player.draw.reverse();
        b.rngs = CombatRngs::new(0xBAD5EED);
        let (ta, tb) = (search(&a, 100, 3), search(&b, 100, 3));
        assert_eq!(ta.root_stats(), tb.root_stats());
        assert_eq!(ta.n_nodes(), tb.n_nodes());
    }

    /// Widening at chance lets the tree past the next hand: without it
    /// every end of turn reaches a fresh leaf and the tree stays in one turn.
    #[test]
    fn widening_reaches_later_turns() {
        // A starter deck has few distinct hands, so its next turns repeat
        // anyway; a big deck of distinct cards almost never does.
        let deck: Vec<Card> = crate::card::IRONCLAD_POOL.iter().take(30).map(|&id| Card::new(0, id, false)).collect();
        let enemies = [EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }];
        let mut c = Combat::new(&deck, 80, 80, 3, &enemies, Ascension(10), 7);
        c.enemies[0].creature.hp = 300;
        let widened = |widen| {
            let mut t = Tree::new(&c, Baseline::of(&c), Config { widen, seed: 5, ..Config::default() });
            for _ in 0..400 {
                if t.descend() {
                    t.expand(&[1.0 / N_ACTIONS as f32; N_ACTIONS], 0.0);
                }
            }
            t.turns_deep()
        };
        assert!(widened(1.0) >= 2, "the widened tree stayed within {} turns", widened(1.0));
        let mut t = Tree::new(&c, Baseline::of(&c), Config { seed: 5, ..Config::default() });
        for _ in 0..400 {
            if t.descend() {
                t.expand(&[1.0 / N_ACTIONS as f32; N_ACTIONS], 0.0);
            }
        }
        assert!(t.turns_deep() <= 1, "unlimited widening went {} turns deep", t.turns_deep());
    }
}
