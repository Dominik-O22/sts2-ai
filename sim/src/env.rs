//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to a fresh generated fight (or the next
//! held-out setup) as soon as it ends. Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use rayon::prelude::*;

use crate::combat::{After, Combat, Outcome};
use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::encounter::{Encounter, Kind};
use crate::gen::{act_floor, encounter_of_kind, generate, generate_against, FightSetup, BOSS_FLOOR, LAST_FLOOR};
use crate::rng::{CombatRngs, Rng};
use crate::potion::PotionId;
use crate::relic::RelicId;
use crate::types::{Ascension, AscensionLevel};

#[derive(Clone, Copy, Debug)]
pub struct EnvConfig {
    pub asc: Ascension,
    /// Generated fights sample a floor uniformly in `min_floor..=max_floor`.
    pub min_floor: u32,
    pub max_floor: u32,
    /// A fight still running after this many actions counts as lost.
    pub max_steps: u32,
    /// Fraction of resets that skip the floor roll and take an elite (on a
    /// floor from 5 up) or the boss instead, half each. The rest of the
    /// resets still roll elites and bosses at their natural rate.
    pub hard_frac: f32,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self { asc: Ascension(10), min_floor: 1, max_floor: LAST_FLOOR, max_steps: 500, hard_frac: 0.0 }
    }
}

/// What a finished fight looked like, for logging.
#[derive(Clone, Copy, Debug)]
pub struct EpisodeEnd {
    pub env: usize,
    pub won: bool,
    pub hp_frac: f32,
    /// HP lost over the fight, as a fraction of max HP.
    pub hp_lost: f32,
    /// Potions drunk (or thrown) over the fight.
    pub potions_used: u32,
    pub steps: u32,
    pub floor: u32,
    pub encounter: Encounter,
    pub kind: Kind,
    pub reward: f32,
}

/// What an HP point is worth when the run goes on: the same for every HP,
/// whatever the max, in any fight. Against a win's 1, losing 10 HP to a weak
/// fight costs 0.25. At 0.5 of the HP fraction (0.006 an HP at 80 max HP) the
/// policy won its weak and normal fights but lost 5 HP a fight more than the
/// winners who played the same ones (`evaluate --source easy`).
const HP_PRICE: f32 = 0.025;

/// What a potion kept is worth: about 12 HP to the fights ahead. At 16 HP
/// the policy drank 0.05 potions in an easy fight against the winners' 0.25.
/// Nothing after the run's last fight.
const POTION_VALUE: f32 = 12.0 * HP_PRICE;

fn potion_value(c: &Combat) -> f32 {
    if c.after == After::End {
        0.0
    } else {
        POTION_VALUE
    }
}

/// What a lost fight pays per fraction of the enemies' HP taken.
///
/// A flat -1 made every line of a lost fight worth the same, so once the
/// value head read a fight as lost the policy stopped trying (it ended turns
/// with Defends in hand against Aeonglass), and search, which scores lines
/// by the same rewards and value, tied every option there. The best loss
/// still pays -0.8, far under any win.
const LOSS_DAMAGE: f32 = 0.2;

/// Stopgap terminal reward (DESIGN.md, Decision engine): a win is worth 1
/// plus the HP kept at `hp_value` an HP and `POTION_VALUE` per unused
/// potion; a loss or a timed-out fight is -1 plus `LOSS_DAMAGE` per
/// fraction of the enemies' HP taken.
pub fn terminal_reward(c: &Combat) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => 1.0 + hp_value(c) * c.player.creature.hp.max(0) as f32 + potion_value(c) * potions_held(c) as f32,
        _ => -1.0 + LOSS_DAMAGE * enemy_hp_taken(c).min(1.0),
    }
}

/// The potions the reward prices: a potion the run gets back for free next
/// fight is worth nothing kept. Delicate Frond refills every empty slot
/// before each combat and Petrified Toad hands out a Potion-Shaped Rock
/// (`BeforeCombatStart`, `BeforeCombatStartLate`); Sozu stops both.
fn potions_held(c: &Combat) -> usize {
    if c.has_relic(RelicId::Sozu) {
        return c.potions.iter().flatten().count();
    }
    if c.has_relic(RelicId::DelicateFrond) {
        return 0;
    }
    let free_rocks = c.has_relic(RelicId::PetrifiedToad);
    c.potions.iter().flatten().filter(|&&p| !(free_rocks && p == PotionId::PotionShapedRock)).count()
}

/// What an HP point is worth, by what follows the fight: `HP_PRICE`.
/// After an act boss the Ancient that opens the next act heals all missing
/// HP, or 80% of it under Weary Traveler, so only the part it leaves
/// counts. Under Double Boss the last act's first boss is followed by the
/// second with no rest, so its HP counts in full; after the run's last
/// fight it counts for nothing.
fn hp_value(c: &Combat) -> f32 {
    match c.after {
        After::Act | After::Boss => HP_PRICE,
        After::Ancient if c.asc.has(AscensionLevel::WearyTraveler) => HP_PRICE * 0.2,
        After::Ancient | After::End => 0.0,
    }
}

/// Where a fight (or a search) started, so the potential is zero there.
/// Some fights open with damaged enemies or a relic heal, so this is read
/// off the combat, not the setup.
#[derive(Clone, Copy, Debug)]
pub struct Baseline {
    taken: f32,
    hp: i32,
    potions: usize,
}

/// Enemy HP lost so far, as a fraction of what the enemies started with.
/// Counted, not read off the current HP bars: a monster that revives at
/// full HP would otherwise make its killing blow cost reward, and the
/// policy learned to leave the Test Subject at 21 HP rather than kill it.
/// It can pass 1 in a fight with revives or summons.
fn enemy_hp_taken(c: &Combat) -> f32 {
    c.stats.enemy_hp_lost as f32 / c.stats.enemy_start_hp.max(1) as f32
}

impl Baseline {
    pub fn of(c: &Combat) -> Self {
        Self { taken: enemy_hp_taken(c), hp: c.player.creature.hp, potions: potions_held(c) }
    }
}

/// Potential for reward shaping: half the enemy HP taken since the
/// baseline (`enemy_hp_taken`), minus the player's HP lost and plus the
/// potions gained (a drink counts as one lost) at the prices
/// the terminal reward puts on them. Without the potion term a drink cost
/// nothing until the fight ended, and the policy drank combat potions in
/// weak fights it lost 5% HP in.
/// Zero at the baseline and, by convention, once the fight is over. Each
/// step is rewarded the change in potential, so a fight's rewards sum to
/// its terminal reward and no ordering of plays is preferred beyond what
/// the outcome says (potential-based shaping keeps the optimal policy);
/// the credit for extra damage just arrives at the play instead of at
/// the end.
pub fn potential(c: &Combat, base: Baseline) -> f32 {
    if c.is_over() {
        return 0.0;
    }
    let lost = (base.hp - c.player.creature.hp.max(0)) as f32;
    let potions = potions_held(c) as f32 - base.potions as f32;
    0.5 * (enemy_hp_taken(c) - base.taken) - hp_value(c) * lost + potion_value(c) * potions
}

/// The reward for a transition: the potential change, plus the terminal
/// reward when `over` (a timed-out fight is over without an outcome).
pub fn step_reward(before: f32, c: &Combat, base: Baseline, over: bool) -> f32 {
    if over {
        terminal_reward(c) - before
    } else {
        potential(c, base) - before
    }
}

struct Slot {
    combat: Combat,
    setup: FightSetup,
    base: Baseline,
    steps: u32,
    rng: Rng,
    resets: usize,
}

impl Slot {
    /// Start the next fight. One already over before the first decision
    /// (Whispering Earring can win turn 1 on its own) is skipped: there is
    /// nothing in it to act on.
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, pools: Pools) {
        loop {
            self.roll(index, n, cfg, pools);
            if !self.combat.is_over() {
                return;
            }
        }
    }

    fn roll(&mut self, index: usize, n: usize, cfg: &EnvConfig, Pools { fixed, hard, real }: Pools) {
        let acts = act_floor(cfg.min_floor).0..=act_floor(cfg.max_floor).0;
        let weighted: Vec<(Encounter, f32)> =
            hard.iter().copied().filter(|(e, _)| acts.contains(&(e.act().index() as u32))).collect();
        self.setup = if !fixed.is_empty() {
            fixed[(index + self.resets * n) % fixed.len()].clone()
        } else if let Some(pool) = self.real_pool(real) {
            pool[self.rng.next_int(pool.len())].rerolled(&mut self.rng)
        } else if self.rng.next_float(1.0) < cfg.hard_frac && !weighted.is_empty() {
            // Elites and bosses by weight: the ones the policy loses most.
            let total: f32 = weighted.iter().map(|(_, w)| w).sum();
            let mut x = self.rng.next_float(total);
            let enc = weighted.iter().find(|(_, w)| {
                x -= w;
                x < 0.0
            });
            let enc = enc.unwrap_or(weighted.last().unwrap()).0;
            let act = enc.act().index() as u32;
            let local = if enc.kind() == Kind::Boss { BOSS_FLOOR } else { 5 + self.rng.next_int((BOSS_FLOOR - 5) as usize) as u32 };
            generate_against(&mut self.rng, act * BOSS_FLOOR + local, cfg.asc, enc)
        } else if self.rng.next_float(1.0) < cfg.hard_frac {
            // An act the floor range reaches, then its boss or an elite.
            let act = act_floor(cfg.min_floor).0 + self.rng.next_int((act_floor(cfg.max_floor).0 - act_floor(cfg.min_floor).0 + 1) as usize) as u32;
            let (kind, local) = if self.rng.next_int(2) == 0 {
                (Kind::Boss, BOSS_FLOOR)
            } else {
                (Kind::Elite, 5 + self.rng.next_int((BOSS_FLOOR - 5) as usize) as u32)
            };
            let enc = encounter_of_kind(&mut self.rng, act, kind);
            generate_against(&mut self.rng, act * BOSS_FLOOR + local, cfg.asc, enc)
        } else {
            let floor = cfg.min_floor + self.rng.next_int((cfg.max_floor - cfg.min_floor + 1) as usize) as u32;
            generate(&mut self.rng, floor, cfg.asc)
        };
        self.resets += 1;
        self.steps = 0;
        self.combat = self.setup.combat(self.rng.next_u64());
        self.base = Baseline::of(&self.combat);
    }

    /// The pool of played fights this reset draws from, each taking its
    /// share of the resets; none for the rest. Draws nothing without pools.
    fn real_pool<'a>(&mut self, real: &'a [(Vec<FightSetup>, f32)]) -> Option<&'a [FightSetup]> {
        if real.is_empty() {
            return None;
        }
        let mut x = self.rng.next_float(1.0);
        real.iter().find(|(_, share)| {
            x -= share;
            x < 0.0
        })
        .map(|(pool, _)| pool.as_slice())
    }

    fn end(&self, index: usize) -> EpisodeEnd {
        let c = &self.combat;
        EpisodeEnd {
            env: index,
            won: c.outcome == Some(Outcome::Won),
            hp_frac: c.player.creature.hp.max(0) as f32 / c.player.creature.max_hp.max(1) as f32,
            hp_lost: (self.setup.hp - c.player.creature.hp.max(0)) as f32 / c.player.creature.max_hp.max(1) as f32,
            potions_used: (self.setup.potions.iter().flatten().count())
                .saturating_sub(c.potions.iter().flatten().count()) as u32,
            steps: self.steps,
            floor: self.setup.floor,
            encounter: self.setup.encounter,
            kind: self.setup.encounter.kind(),
            reward: terminal_reward(c),
        }
    }
}

/// Where a reset can take its fight from besides the generator.
#[derive(Clone, Copy)]
struct Pools<'a> {
    /// When set, resets cycle through these instead of generating.
    fixed: &'a [FightSetup],
    /// When set, the `hard_frac` share of fights draws its elite or boss
    /// by these weights instead of evenly.
    hard: &'a [(Encounter, f32)],
    /// Pools of played runs' fights, each drawn for its share of the
    /// resets with its enemies rolled afresh.
    real: &'a [(Vec<FightSetup>, f32)],
}

pub struct VecEnv {
    slots: Vec<Slot>,
    cfg: EnvConfig,
    fixed: Vec<FightSetup>,
    hard: Vec<(Encounter, f32)>,
    real: Vec<(Vec<FightSetup>, f32)>,
}

impl VecEnv {
    pub fn new(n: usize, seed: u64, cfg: EnvConfig) -> Self {
        let mut slots: Vec<Slot> = (0..n)
            .map(|i| {
                let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64));
                let setup = generate(&mut rng, 1, cfg.asc);
                let combat = setup.combat(0);
                Slot { base: Baseline::of(&combat), combat, setup, steps: 0, rng, resets: 0 }
            })
            .collect();
        for (i, s) in slots.iter_mut().enumerate() {
            s.reset(i, n, &cfg, Pools { fixed: &[], hard: &[], real: &[] });
        }
        Self { slots, cfg, fixed: vec![], hard: vec![], real: vec![] }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn asc(&self) -> Ascension {
        self.cfg.asc
    }

    /// Curriculum knob: which floors generated fights come from. Takes
    /// effect at each env's next reset.
    pub fn set_hard_frac(&mut self, frac: f32) {
        self.cfg.hard_frac = frac.clamp(0.0, 1.0);
    }

    pub fn set_floors(&mut self, min: u32, max: u32) {
        self.cfg.min_floor = min.clamp(1, LAST_FLOOR);
        self.cfg.max_floor = max.clamp(self.cfg.min_floor, LAST_FLOOR);
    }

    /// Evaluate on fixed setups (the recordings) instead of generated ones.
    /// Weights for the elites and bosses forced by `hard_frac`; empty
    /// goes back to drawing them evenly. Takes effect at each reset.
    pub fn set_hard_weights(&mut self, weights: Vec<(Encounter, f32)>) {
        self.hard = weights.into_iter().filter(|(e, w)| matches!(e.kind(), Kind::Elite | Kind::Boss) && *w > 0.0).collect();
    }

    /// Resets every env so the first `n` setups start immediately.
    pub fn set_fixed(&mut self, setups: Vec<FightSetup>) {
        self.fixed = setups;
        let n = self.slots.len();
        let cfg = self.cfg;
        let pools = Pools { fixed: &self.fixed, hard: &self.hard, real: &self.real };
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.resets = 0;
            s.reset(i, n, &cfg, pools);
        }
    }

    /// Pools of played runs' fights, each for its share of the resets from
    /// here on (the shares sum to at most 1), their enemies rolled afresh
    /// each time.
    pub fn set_real(&mut self, pools: Vec<(Vec<FightSetup>, f32)>) {
        assert!(pools.iter().map(|p| p.1).sum::<f32>() <= 1.0 + 1e-6, "played fights' shares sum past 1");
        self.real = pools.into_iter().filter(|(pool, share)| !pool.is_empty() && *share > 0.0).collect();
    }

    pub fn combat(&self, i: usize) -> &Combat {
        &self.slots[i].combat
    }

    pub fn setup(&self, i: usize) -> &FightSetup {
        &self.slots[i].setup
    }

    /// Encode every env's current state.
    pub fn observe(&self, floats: &mut [f32], ids: &mut [i64], mask: &mut [bool]) {
        self.check_buffers(floats, ids, mask);
        self.slots
            .par_iter()
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .for_each(|(((s, f), i), m)| encode::encode(&s.combat, f, i, m));
    }

    /// Apply one action index per env, reset the envs whose fight ended,
    /// and encode the states that follow. `rewards` and `dones` describe
    /// the transition; the returned list describes every fight that ended.
    pub fn step(
        &mut self,
        actions: &[i64],
        floats: &mut [f32],
        ids: &mut [i64],
        mask: &mut [bool],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Vec<EpisodeEnd> {
        self.check_buffers(floats, ids, mask);
        let n = self.slots.len();
        assert!(actions.len() == n && rewards.len() == n && dones.len() == n, "batch size mismatch");
        let cfg = self.cfg;
        let pools = Pools { fixed: &self.fixed, hard: &self.hard, real: &self.real };
        self.slots
            .par_iter_mut()
            .enumerate()
            .zip(actions.par_iter())
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .zip(rewards.par_iter_mut())
            .zip(dones.par_iter_mut())
            .map(|(((((((i, s), &a), f), ids), m), r), d)| {
                let action = encode::decode(&s.combat, a as usize)
                    .unwrap_or_else(|| panic!("env {i}: action {a} is not legal; legal: {:?}", s.combat.legal_actions()));
                let before = potential(&s.combat, s.base);
                s.combat.step(action);
                s.steps += 1;
                let over = s.combat.is_over() || s.steps >= cfg.max_steps;
                let end = over.then(|| s.end(i));
                *r = step_reward(before, &s.combat, s.base, over);
                *d = over;
                if over {
                    s.reset(i, n, &cfg, pools);
                }
                encode::encode(&s.combat, f, ids, m);
                end
            })
            .flatten()
            .collect()
    }

    fn check_buffers(&self, floats: &[f32], ids: &[i64], mask: &[bool]) {
        let n = self.slots.len();
        assert!(
            floats.len() == n * N_FLOATS && ids.len() == n * N_IDS && mask.len() == n * N_ACTIONS,
            "observation buffers do not match {n} envs"
        );
    }
}

/// A thread's scratch row, for encodings only hashed.
fn scratch() -> (Vec<f32>, Vec<i64>, Vec<bool>) {
    (vec![0.0; N_FLOATS], vec![0; N_IDS], vec![false; N_ACTIONS])
}

/// FxHash-style mix of an encoding, then murmur3's finalizer. The floats
/// are mostly zero, so they go in 32-byte chunks and all-zero chunks are
/// skipped; the others mix in with their position.
fn row_hash(floats: &[f32], ids: &[i64], mask: &[bool]) -> u64 {
    let mut h = 0u64;
    let mut mix = |w: u64| h = (h.rotate_left(5) ^ w).wrapping_mul(0x51_7C_C1_B7_27_22_0A_95);
    // A branchless OR, so zero blocks scan at memory speed.
    let any = |xs: &[f32]| xs.iter().fold(0, |a, x| a | x.to_bits()) != 0;
    for (b, block) in floats.chunks(64).enumerate() {
        if !any(block) {
            continue;
        }
        for (n, chunk) in block.chunks(8).enumerate() {
            if any(chunk) {
                mix((b * 8 + n) as u64);
                chunk.chunks(2).for_each(|p| mix(p.iter().fold(0, |a, x| a << 32 | x.to_bits() as u64)));
            }
        }
    }
    ids.iter().for_each(|&x| mix(x as u64));
    mask.chunks(8).for_each(|p| mix(p.iter().fold(0, |a, &x| a << 8 | x as u64)));
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ h >> 33
}

/// Copies of a combat stepped together, for a search over the rest of the
/// current turn (the advisor's plan, `searcheval`). Several roots can share
/// one batch, each with its own run of copies. Each fork forgets the recording's
/// script and rolls its own dice, and the draw pile is reshuffled: the
/// player does not know its order, so the plan must not either. Forks in
/// the same `group` share a shuffle, so plans in a group are compared on
/// the same hidden draws.
///
/// Copies are the unit of the API, but copies whose states are equal sit
/// on one shared node: copies of a shuffle group start on one node, and a
/// step that rolls no dice moves every copy that took it to one new node.
/// Only a step that rolls dice splits a group, each copy stepping on its
/// own. Rewards and encodings per copy are exactly those of independent
/// clones stepped one by one.
pub struct Forks {
    /// A node's combat is the state of every copy on it. Its `rngs` are
    /// whichever copy stepped it last, or the root's: the copy's own dice
    /// live in `Fork` and are swapped in for its steps.
    nodes: Vec<Box<Combat>>,
    /// Per node: the hash of its encoding, taken in `step` while its state
    /// is still in cache (`observe_unique` would fetch it all again). None
    /// until it first moves.
    hashes: Vec<Option<u64>>,
    copies: Vec<Fork>,
}

/// One copy of a search: the node holding its state, its own dice, the
/// last player turn it plays and the root's baseline.
struct Fork {
    node: usize,
    rngs: CombatRngs,
    last_turn: u32,
    base: Baseline,
}

/// Copies on one node that take the same action this step. Owned when the
/// group is all that refers to the node, so it steps in place.
struct Group<'a> {
    src: Src,
    action: i64,
    /// The copies, as `(node, action, copy)` keys.
    members: &'a [(usize, i64, usize)],
}

enum Src {
    Owned(Box<Combat>, Option<u64>),
    Shared(usize),
}

/// Where a group's copies ended up: nodes it made, one entry per copy, and
/// whether they stayed on the shared node instead (an illegal action).
struct GroupOut {
    nodes: Vec<(Box<Combat>, Option<u64>)>,
    moved: Vec<Moved>,
    stayed: Option<usize>,
}

struct Moved {
    copy: usize,
    /// Index into the group's `nodes`.
    node: usize,
    reward: f32,
    /// The copy's dice after the step, when the step rolled any.
    rngs: Option<CombatRngs>,
}

impl Forks {
    pub fn new(root: &Combat, n: usize, groups: usize, seed: u64) -> Self {
        Self::of(&[root], n, groups, seed, 1)
    }

    /// `n` copies of each root, root after root, each played for `depth`
    /// player turns: the rest of the current one, then `depth - 1` more.
    pub fn of(roots: &[&Combat], n: usize, groups: usize, seed: u64, depth: u32) -> Self {
        let per_group = n.div_ceil(groups.max(1)).max(1);
        let n_groups = n.div_ceil(per_group);
        let root_seed = |r: usize| seed ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let nodes = roots
            .par_iter()
            .enumerate()
            .flat_map_iter(|(r, root)| {
                (0..n_groups).map(move |group| {
                    let mut c = (*root).clone();
                    c.script = Default::default();
                    Rng::new(root_seed(r) ^ (group as u64 + 1) << 20).shuffle(&mut c.player.draw);
                    Box::new(c)
                })
            })
            .collect();
        let copies = (0..roots.len() * n)
            .into_par_iter()
            .map(|k| {
                let (r, i) = (k / n, k % n);
                Fork {
                    node: r * n_groups + i / per_group,
                    rngs: CombatRngs::new(root_seed(r) ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    last_turn: roots[r].player.turn + depth.max(1) - 1,
                    base: Baseline::of(roots[r]),
                }
            })
            .collect();
        let hashes = vec![None; roots.len() * n_groups];
        Self { nodes, hashes, copies }
    }

    pub fn len(&self) -> usize {
        self.copies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    /// Copy `i`'s state. Its `rngs` field is not copy `i`'s dice.
    pub fn combat(&self, i: usize) -> &Combat {
        &self.nodes[self.copies[i].node]
    }

    /// The player's turn is done: the sim moved on to the next one, or
    /// the fight ended.
    pub fn turn_over(&self, i: usize) -> bool {
        let c = self.combat(i);
        c.is_over() || c.player.turn > self.copies[i].last_turn
    }

    /// The forks still in their turn.
    pub fn live(&self) -> Vec<usize> {
        (0..self.copies.len()).filter(|&i| !self.turn_over(i)).collect()
    }

    /// Encode forks `rows`, one row per distinct observation: the distinct
    /// ones are packed at the front of the buffers, `inverse[k]` is the row
    /// fork `rows[k]` encodes to, and the count is returned. Copies on one
    /// node share a row outright; nodes that still encode the same (shuffle
    /// groups before a draw) are told apart by a 64-bit hash of their
    /// encoding.
    pub fn observe_unique(&self, rows: &[usize], floats: &mut [f32], ids: &mut [i64], mask: &mut [bool], inverse: &mut [usize]) -> usize {
        let mut slot_of_node = vec![usize::MAX; self.nodes.len()];
        let mut node_of_slot = vec![];
        let slots: Vec<usize> = rows
            .iter()
            .map(|&r| {
                let node = self.copies[r].node;
                if slot_of_node[node] == usize::MAX {
                    slot_of_node[node] = node_of_slot.len();
                    node_of_slot.push(node);
                }
                slot_of_node[node]
            })
            .collect();
        // Hash each node's encoding in a scratch row that stays in cache,
        // then write only the distinct ones to the (large) buffers.
        let hashes: Vec<u64> = node_of_slot
            .par_iter()
            .map_init(scratch, |(f, i, m), &node| {
                self.hashes[node].unwrap_or_else(|| {
                    encode::encode(&self.nodes[node], f, i, m);
                    row_hash(f, i, m)
                })
            })
            .collect();
        let mut distinct = std::collections::HashMap::with_capacity(node_of_slot.len());
        let mut first = vec![];
        let row_of_slot: Vec<usize> = hashes
            .into_iter()
            .zip(&node_of_slot)
            .map(|(h, &node)| {
                *distinct.entry(h).or_insert_with(|| {
                    first.push(node);
                    first.len() - 1
                })
            })
            .collect();
        for (k, slot) in slots.into_iter().enumerate() {
            inverse[k] = row_of_slot[slot];
        }
        first
            .par_iter()
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .for_each(|(((&node, f), i), m)| encode::encode(&self.nodes[node], f, i, m));
        first.len()
    }

    /// Step every fork whose turn is still running; the rest ignore their
    /// action. Writes the shaped reward of each transition, 0 for a fork
    /// that did not move.
    pub fn step(&mut self, actions: &[i64], rewards: &mut [f32]) {
        rewards.fill(0.0);
        let mut keyed: Vec<(usize, i64, usize)> =
            (0..self.copies.len()).filter(|&i| !self.turn_over(i)).map(|i| (self.copies[i].node, actions[i], i)).collect();
        keyed.par_sort_unstable();
        // Copies sitting a step out keep their node where it is.
        let mut held = vec![0u32; self.nodes.len()];
        self.copies.iter().for_each(|c| held[c.node] += 1);
        keyed.iter().for_each(|&(node, _, _)| held[node] -= 1);
        let runs: Vec<&[(usize, i64, usize)]> = keyed.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)).collect();
        let mut groups_on = vec![0u32; self.nodes.len()];
        runs.iter().for_each(|g| groups_on[g[0].0] += 1);
        let mut old: Vec<Option<Box<Combat>>> = std::mem::take(&mut self.nodes).into_iter().map(Some).collect();
        let old_hashes = std::mem::take(&mut self.hashes);
        let groups: Vec<Group> = runs
            .into_iter()
            .map(|members| {
                let node = members[0].0;
                let src = if held[node] == 0 && groups_on[node] == 1 {
                    Src::Owned(old[node].take().unwrap(), old_hashes[node])
                } else {
                    Src::Shared(node)
                };
                Group { src, action: members[0].1, members }
            })
            .collect();
        let outs: Vec<GroupOut> = groups.into_par_iter().map_init(scratch, |row, g| self.step_group(g, &old, row)).collect();

        let mut stayed = vec![false; old.len()];
        outs.iter().filter_map(|o| o.stayed).for_each(|node| stayed[node] = true);
        let mut remap = vec![usize::MAX; old.len()];
        for (k, c) in old.into_iter().enumerate() {
            let Some(c) = c.filter(|_| held[k] > 0 || stayed[k]) else { continue };
            remap[k] = self.nodes.len();
            self.nodes.push(c);
            self.hashes.push(old_hashes[k]);
        }
        self.copies.iter_mut().for_each(|c| c.node = remap[c.node]);
        for out in outs {
            let at = self.nodes.len();
            for (c, h) in out.nodes {
                self.nodes.push(c);
                self.hashes.push(h);
            }
            for m in out.moved {
                let copy = &mut self.copies[m.copy];
                copy.node = at + m.node;
                rewards[m.copy] = m.reward;
                if let Some(rngs) = m.rngs {
                    copy.rngs = rngs;
                }
            }
        }
    }

    /// Step one group: the first copy steps the node's state with its own
    /// dice, and if that rolled none the whole group lands on the result.
    /// Otherwise each other copy steps the pre-step state itself.
    fn step_group(&self, g: Group, old: &[Option<Box<Combat>>], (f, i, m): &mut (Vec<f32>, Vec<i64>, Vec<bool>)) -> GroupOut {
        let Group { src, action, members } = g;
        let (pre, hash, shared) = match src {
            Src::Owned(c, h) => (c, h, None),
            Src::Shared(node) => (old[node].as_ref().unwrap().clone(), None, Some(node)),
        };
        let Some(action) = encode::decode(&pre, action as usize) else {
            return match shared {
                Some(node) => GroupOut { nodes: vec![], moved: vec![], stayed: Some(node) },
                None => GroupOut {
                    nodes: vec![(pre, hash)],
                    moved: members.iter().map(|&(_, _, copy)| Moved { copy, node: 0, reward: 0.0, rngs: None }).collect(),
                    stayed: None,
                },
            };
        };
        let befores: Vec<f32> = members.iter().map(|&(_, _, copy)| potential(&pre, self.copies[copy].base)).collect();
        let reward = |j: usize, c: &Combat| step_reward(befores[j], c, self.copies[members[j].2].base, c.is_over());
        // The last copy to step takes the pre-step state instead of a clone.
        let mut pre = Some(pre);
        let take = |pre: &mut Option<Box<Combat>>, j: usize| if j + 1 == members.len() { pre.take().unwrap() } else { pre.as_ref().unwrap().clone() };
        let mut play = |mut c: Box<Combat>, copy: usize| {
            c.rngs = self.copies[copy].rngs.clone();
            c.step(action);
            let hash = (!c.is_over()).then(|| {
                encode::encode(&c, f, i, m);
                row_hash(f, i, m)
            });
            (c, hash)
        };
        let rep = members[0].2;
        let (c, hash) = play(take(&mut pre, 0), rep);
        // Xoshiro never returns to a state, so equal dice mean none rolled.
        if c.rngs == self.copies[rep].rngs {
            let moved = members.iter().enumerate().map(|(j, &(_, _, copy))| Moved { copy, node: 0, reward: reward(j, &c), rngs: None }).collect();
            return GroupOut { nodes: vec![(c, hash)], moved, stayed: None };
        }
        let mut out = GroupOut { nodes: vec![], moved: vec![], stayed: None };
        let mut land = |j: usize, (c, hash): (Box<Combat>, Option<u64>)| {
            out.moved.push(Moved { copy: members[j].2, node: out.nodes.len(), reward: reward(j, &c), rngs: Some(c.rngs.clone()) });
            out.nodes.push((c, hash));
        };
        land(0, (c, hash));
        for (j, &(_, _, copy)) in members.iter().enumerate().skip(1) {
            land(j, play(take(&mut pre, j), copy));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random masked actions through the batch API: every step returns a
    /// legal state, envs reset on their own, and episode reports match the
    /// done flags.
    #[test]
    fn batch_steps_reset_and_report() {
        let n = 64;
        let mut env = VecEnv::new(n, 1, EnvConfig::default());
        let mut floats = vec![0.0; n * N_FLOATS];
        let mut ids = vec![0; n * N_IDS];
        let mut mask = vec![false; n * N_ACTIONS];
        let mut rewards = vec![0.0; n];
        let mut dones = vec![false; n];
        let mut summed = vec![0.0f32; n];
        env.observe(&mut floats, &mut ids, &mut mask);
        let mut rng = Rng::new(9);
        let mut ended = 0;
        for _ in 0..400 {
            let actions: Vec<i64> = (0..n)
                .map(|i| {
                    let m = &mask[i * N_ACTIONS..][..N_ACTIONS];
                    let legal: Vec<usize> = (0..N_ACTIONS).filter(|&k| m[k]).collect();
                    assert!(!legal.is_empty(), "env {i} has no legal action");
                    legal[rng.next_int(legal.len())] as i64
                })
                .collect();
            let ends = env.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards, &mut dones);
            assert_eq!(ends.len(), dones.iter().filter(|&&d| d).count());
            for (i, r) in rewards.iter().enumerate() {
                summed[i] += r;
            }
            for e in &ends {
                assert!(dones[e.env]);
                // Shaping telescopes: a fight's rewards sum to its terminal reward.
                assert!((summed[e.env] - e.reward).abs() < 1e-4, "env {}: summed {} vs terminal {}", e.env, summed[e.env], e.reward);
                summed[e.env] = 0.0;
                assert!(e.won == (e.reward > 0.0));
            }
            ended += ends.len();
        }
        assert!(ended > n, "fights should have ended and restarted");
    }

    /// Forks of one combat run their turns to the end on their own dice,
    /// and each group shares a draw order.
    #[test]
    fn forks_run_out_the_turn() {
        let mut rng = Rng::new(4);
        let root = generate(&mut rng, 8, Ascension(10)).combat(3);
        let n = 16;
        let forks = Forks::new(&root, n, 4, 11);
        let draw = |c: &Combat| c.player.draw.iter().map(|k| k.id).collect::<Vec<_>>();
        assert_eq!(draw(forks.combat(0)), draw(forks.combat(1)), "same group, same shuffle");
        assert!((0..n).any(|i| draw(forks.combat(i)) != draw(&root)), "forks reshuffle the draw pile");
        let mut forks = forks;
        let mut floats = vec![0.0; n * N_FLOATS];
        let mut ids = vec![0; n * N_IDS];
        let mut mask = vec![false; n * N_ACTIONS];
        let mut rewards = vec![0.0f32; n];
        let mut inverse = vec![0; n];
        let all: Vec<usize> = (0..n).collect();
        for _ in 0..40 {
            forks.observe_unique(&all, &mut floats, &mut ids, &mut mask, &mut inverse);
            let actions: Vec<i64> = (0..n)
                .map(|i| {
                    let m = &mask[inverse[i] * N_ACTIONS..][..N_ACTIONS];
                    (0..N_ACTIONS).filter(|&k| m[k]).max_by_key(|&k| rng.next_int(1000).wrapping_add(k * 0)).unwrap_or(0) as i64
                })
                .collect();
            forks.step(&actions, &mut rewards);
            assert!(rewards.iter().all(|r| r.is_finite()));
            if (0..n).all(|i| forks.turn_over(i)) {
                return;
            }
        }
        panic!("some fork never ended its turn");
    }

    /// Forks that look the same share a row, and each row is its forks'
    /// own encoding: before anything happens only the shuffle group can
    /// set forks apart, and after a random move each still reads its own.
    #[test]
    fn observe_unique_keeps_one_row_per_observation() {
        let mut rng = Rng::new(4);
        let root = generate(&mut rng, 8, Ascension(10)).combat(3);
        let (n, groups) = (16, 4);
        let mut forks = Forks::new(&root, n, groups, 11);
        let mut floats = vec![0.0; n * N_FLOATS];
        let mut ids = vec![0; n * N_IDS];
        let mut mask = vec![false; n * N_ACTIONS];
        let mut inverse = vec![0; n];
        let mut rewards = vec![0.0f32; n];
        let all: Vec<usize> = (0..n).collect();
        let (mut f, mut i, mut m) = (vec![0.0; N_FLOATS], vec![0; N_IDS], vec![false; N_ACTIONS]);
        for round in 0..2 {
            let distinct = forks.observe_unique(&all, &mut floats, &mut ids, &mut mask, &mut inverse);
            if round == 0 {
                assert!(distinct <= groups, "{distinct} distinct rows before any move");
            }
            for (k, &row) in inverse.iter().enumerate() {
                assert!(row < distinct);
                encode::encode(forks.combat(k), &mut f, &mut i, &mut m);
                assert_eq!(f, floats[row * N_FLOATS..][..N_FLOATS], "fork {k}");
                assert_eq!(i, ids[row * N_IDS..][..N_IDS], "fork {k}");
                assert_eq!(m, mask[row * N_ACTIONS..][..N_ACTIONS], "fork {k}");
            }
            let actions: Vec<i64> = (0..n)
                .map(|k| {
                    let m = &mask[inverse[k] * N_ACTIONS..][..N_ACTIONS];
                    let legal: Vec<usize> = (0..N_ACTIONS).filter(|&a| m[a]).collect();
                    legal[rng.next_int(legal.len())] as i64
                })
                .collect();
            forks.step(&actions, &mut rewards);
        }
    }

    /// Independent clones stepped one by one: what shared nodes must match
    /// copy for copy.
    struct Naive {
        combats: Vec<Combat>,
        turns: Vec<u32>,
        bases: Vec<Baseline>,
    }

    impl Naive {
        fn of(roots: &[&Combat], n: usize, groups: usize, seed: u64, depth: u32) -> Self {
            let per_group = n.div_ceil(groups.max(1));
            let mut combats = vec![];
            for (r, root) in roots.iter().enumerate() {
                let seed = seed ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                for i in 0..n {
                    let mut c = (*root).clone();
                    c.script = Default::default();
                    c.rngs = CombatRngs::new(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let group = i / per_group;
                    Rng::new(seed ^ (group as u64 + 1) << 20).shuffle(&mut c.player.draw);
                    combats.push(c);
                }
            }
            let turns = roots.iter().flat_map(|root| std::iter::repeat_n(root.player.turn + depth - 1, n)).collect();
            let bases = roots.iter().flat_map(|root| std::iter::repeat_n(Baseline::of(root), n)).collect();
            Self { combats, turns, bases }
        }

        fn turn_over(&self, i: usize) -> bool {
            self.combats[i].is_over() || self.combats[i].player.turn > self.turns[i]
        }

        fn step(&mut self, actions: &[i64], rewards: &mut [f32]) {
            for (i, c) in self.combats.iter_mut().enumerate() {
                rewards[i] = 0.0;
                if c.is_over() || c.player.turn > self.turns[i] {
                    continue;
                }
                if let Some(action) = encode::decode(c, actions[i] as usize) {
                    let before = potential(c, self.bases[i]);
                    c.step(action);
                    rewards[i] = step_reward(before, c, self.bases[i], c.is_over());
                }
            }
        }
    }

    /// Shared nodes are invisible: over whole turns of random play on
    /// several fights, every copy's reward and encoding at every step are
    /// those of its own independent clone, and copies did share nodes.
    #[test]
    fn shared_nodes_match_independent_copies() {
        let mut rng = Rng::new(7);
        let asc = Ascension(10);
        let elite = encounter_of_kind(&mut rng, 0, Kind::Elite);
        let boss = encounter_of_kind(&mut rng, 1, Kind::Boss);
        let setups = [
            generate(&mut rng, 2, asc),
            generate(&mut rng, 9, asc),
            generate_against(&mut rng, 7, asc, elite),
            generate_against(&mut rng, BOSS_FLOOR + BOSS_FLOOR, asc, boss),
            generate(&mut rng, 2 * BOSS_FLOOR + 4, asc),
        ];
        let (n, groups) = (24, 4);
        let mut shared_at_some_point = false;
        for (case, (depth, seed)) in [(1, 3u64), (2, 5), (1, 8)].into_iter().enumerate() {
            let roots: Vec<Combat> = setups.iter().enumerate().map(|(k, s)| s.combat(seed + k as u64)).collect();
            let roots: Vec<&Combat> = roots.iter().collect();
            let mut forks = Forks::of(&roots, n, groups, seed, depth);
            let mut naive = Naive::of(&roots, n, groups, seed, depth);
            let total = forks.len();
            assert_eq!(total, naive.combats.len());
            let mut rewards = vec![0.0f32; total];
            let mut expected = vec![0.0f32; total];
            let (mut f, mut i, mut m) = scratch();
            let (mut f2, mut i2, mut m2) = scratch();
            for step in 0..300 {
                let mut actions = vec![0i64; total];
                for k in 0..total {
                    assert_eq!(forks.turn_over(k), naive.turn_over(k), "case {case} step {step} copy {k}");
                    encode::encode(forks.combat(k), &mut f, &mut i, &mut m);
                    encode::encode(&naive.combats[k], &mut f2, &mut i2, &mut m2);
                    assert!(f == f2 && i == i2 && m == m2, "case {case} step {step} copy {k}: encodings differ");
                    let legal: Vec<usize> = (0..N_ACTIONS).filter(|&a| m[a]).collect();
                    if !legal.is_empty() {
                        actions[k] = legal[rng.next_int(legal.len())] as i64;
                    }
                }
                if (0..total).all(|k| naive.turn_over(k)) {
                    break;
                }
                forks.step(&actions, &mut rewards);
                naive.step(&actions, &mut expected);
                for k in 0..total {
                    assert_eq!(rewards[k].to_bits(), expected[k].to_bits(), "case {case} step {step} copy {k}: reward {} vs {}", rewards[k], expected[k]);
                }
                let live = forks.live();
                let mut nodes: Vec<usize> = live.iter().map(|&k| forks.copies[k].node).collect();
                nodes.sort_unstable();
                nodes.dedup();
                shared_at_some_point |= nodes.len() < live.len();
            }
            assert!((0..total).all(|k| naive.turn_over(k)), "case {case}: a turn never ended");
        }
        assert!(shared_at_some_point, "no step shared a node between live copies");
    }

    /// A potion the run gets back next fight is not priced: Petrified
    /// Toad's rock, and anything under Delicate Frond, unless Sozu stops
    /// the refill.
    /// What a won fight leaves is priced by what follows it: in full before
    /// the last act's second boss, for nothing after the run's last fight.
    #[test]
    fn what_follows_prices_what_is_left() {
        let mut c = generate(&mut Rng::new(4), 8, Ascension(10)).combat(3);
        c.relics = vec![];
        c.potions = vec![Some(PotionId::FirePotion), None];
        c.player.creature.hp = c.player.creature.max_hp / 2;
        c.outcome = Some(Outcome::Won);
        let reward = |c: &mut Combat, after| {
            c.after = after;
            terminal_reward(c)
        };
        let hp = HP_PRICE * c.player.creature.hp as f32;
        assert_eq!(reward(&mut c, After::Act), 1.0 + hp + POTION_VALUE);
        assert_eq!(reward(&mut c, After::Boss), 1.0 + hp + POTION_VALUE, "the second boss follows with no rest");
        assert_eq!(reward(&mut c, After::Ancient), 1.0 + 0.2 * hp + POTION_VALUE, "the Ancient heals 80% at A10");
        assert_eq!(reward(&mut c, After::End), 1.0, "nothing is left to spend");
        // An HP point is worth the same at any max HP.
        c.player.creature.max_hp *= 2;
        assert_eq!(reward(&mut c, After::Act), 1.0 + hp + POTION_VALUE);
    }

    #[test]
    fn refilled_potions_are_free() {
        use crate::relic::Relic;
        let mut c = generate(&mut Rng::new(4), 8, Ascension(10)).combat(3);
        c.relics = vec![];
        c.potions = vec![Some(PotionId::PotionShapedRock), Some(PotionId::FirePotion)];
        assert_eq!(potions_held(&c), 2);
        c.relics.push(Relic::new(RelicId::PetrifiedToad));
        assert_eq!(potions_held(&c), 1, "the Toad's rock comes back");
        c.relics.push(Relic::new(RelicId::Sozu));
        assert_eq!(potions_held(&c), 2, "Sozu stops the Toad");
        c.relics = vec![Relic::new(RelicId::DelicateFrond)];
        assert_eq!(potions_held(&c), 0, "the Frond refills every slot");
    }

    /// Killing a monster that revives at full HP is progress, not a loss.
    #[test]
    fn killing_a_reviver_raises_the_potential() {
        let mut rng = Rng::new(2);
        for seed in 0..50 {
            let setup = generate_against(&mut rng, 2 * BOSS_FLOOR + BOSS_FLOOR, Ascension(10), Encounter::TestSubjectBoss);
            let mut c = setup.combat(seed);
            let base = Baseline::of(&c);
            c.enemies[0].creature.hp = 1;
            c.enemies[0].creature.block = 0;
            let Some(hit) = c.legal_actions().into_iter().find(|a| matches!(a, crate::combat::Action::PlayCard { target: Some(0), .. })) else {
                continue;
            };
            let before = potential(&c, base);
            c.step(hit);
            if c.enemies[0].creature.hp > 0 || c.is_over() {
                continue;
            }
            // It gets back up on its own turn, at full HP.
            while !c.is_over() && c.legal_actions().contains(&crate::combat::Action::EndTurn) && c.enemies[0].creature.hp == 0 {
                c.step(crate::combat::Action::EndTurn);
            }
            if c.enemies[0].creature.hp > 1 {
                assert!(potential(&c, base) > before, "the kill cost reward");
                return;
            }
        }
        panic!("no seed revived the Test Subject");
    }

    #[test]
    fn hard_weights_pick_the_forced_fight() {
        let cfg = EnvConfig { hard_frac: 1.0, ..Default::default() };
        let mut env = VecEnv::new(4, 3, cfg);
        env.set_hard_weights(vec![(Encounter::KnowledgeDemonBoss, 1.0), (Encounter::VantomBoss, 0.0)]);
        for s in env.slots.iter_mut() {
            for _ in 0..20 {
                s.reset(0, 4, &cfg, Pools { fixed: &[], hard: &env.hard, real: &[] });
                assert_eq!(s.setup.encounter, Encounter::KnowledgeDemonBoss);
                assert_eq!(act_floor(s.setup.floor), (1, BOSS_FLOOR));
            }
        }
    }

    #[test]
    fn hard_frac_forces_elites_and_bosses() {
        let cfg = EnvConfig { hard_frac: 1.0, ..Default::default() };
        let env = VecEnv::new(200, 3, cfg);
        assert!(env.slots.iter().all(|s| matches!(s.setup.encounter.kind(), Kind::Elite | Kind::Boss)));
        assert!(env.slots.iter().any(|s| s.setup.encounter.kind() == Kind::Boss));
        assert!(env.slots.iter().any(|s| s.setup.encounter.kind() == Kind::Elite));
    }

    #[test]
    fn fixed_setups_cycle_in_order() {
        let mut rng = Rng::new(2);
        let setups: Vec<FightSetup> = (1..=3).map(|f| generate(&mut rng, f, Ascension(10))).collect();
        let mut env = VecEnv::new(2, 5, EnvConfig::default());
        env.set_fixed(setups.clone());
        assert_eq!(env.slots[0].setup.encounter, setups[0].encounter);
        assert_eq!(env.slots[1].setup.encounter, setups[1].encounter);
        env.slots[0].reset(0, 2, &env.cfg, Pools { fixed: &env.fixed, hard: &[], real: &[] });
        assert_eq!(env.slots[0].setup.encounter, setups[2].encounter);
    }

    /// Each pool of played fights takes its share of the resets, and the
    /// generator the rest.
    #[test]
    fn real_pools_take_their_shares() {
        let ids = crate::replay::Ids::new();
        let line = |enc: &str| format!(r#"{{"start": {{"ascension": 10, "deck": [{{"id": "BASH"}}], "relics": [], "potions": [null, null]}}, "hp": 50, "max_hp": 80, "encounter": "{enc}", "floor": 48}}"#);
        let pool = |enc: &str| crate::gen::run_setups(&line(enc), &ids, 1).unwrap();
        let mut env = VecEnv::new(1, 5, EnvConfig::default());
        env.set_real(vec![(pool("QUEEN_BOSS"), 0.25), (pool("AEONGLASS_BOSS"), 0.5)]);
        let pools = Pools { fixed: &[], hard: &[], real: &env.real };
        let mut counts = [0; 3];
        for _ in 0..4000 {
            env.slots[0].roll(0, 1, &env.cfg, pools);
            counts[match env.slots[0].setup.encounter {
                Encounter::QueenBoss if env.slots[0].setup.hp == 50 => 0,
                Encounter::AeonglassBoss if env.slots[0].setup.hp == 50 => 1,
                _ => 2,
            }] += 1;
        }
        for (n, share) in counts.iter().zip([0.25, 0.5, 0.25]) {
            assert!((*n as f32 / 4000.0 - share).abs() < 0.03, "{counts:?}");
        }
    }

    /// A played run's fight keeps its deck, HP, encounter and floor each
    /// time it is drawn.
    #[test]
    fn real_setups_keep_the_run() {
        let ids = crate::replay::Ids::new();
        let line = r#"{"start": {"ascension": 10, "deck": [{"id": "BASH", "up": true}, {"id": "STRIKE_IRONCLAD"}], "relics": ["BURNING_BLOOD"], "potions": [null, null]}, "hp": 41, "max_hp": 80, "encounter": "QUEEN_BOSS", "floor": 48}"#;
        let real = crate::gen::run_setups(line, &ids, 1).unwrap();
        let mut env = VecEnv::new(8, 5, EnvConfig::default());
        env.set_real(vec![(real, 1.0)]);
        let pools = Pools { fixed: &[], hard: &[], real: &env.real };
        for s in env.slots.iter_mut() {
            s.reset(0, 8, &env.cfg, pools);
            assert_eq!((s.setup.encounter, s.setup.hp, s.setup.max_hp, s.setup.floor), (Encounter::QueenBoss, 41, 80, 48));
            assert_eq!(s.setup.deck.len(), 2);
            assert!(s.setup.deck[0].upgraded);
        }
    }
}
