//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to a fresh generated fight (or the next
//! held-out setup) as soon as it ends. Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use rayon::prelude::*;

use crate::combat::{Combat, Outcome, RoomKind};
use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::encounter::{Encounter, Kind};
use crate::gen::{act_floor, encounter_of_kind, generate, generate_against, FightSetup, BOSS_FLOOR, LAST_FLOOR};
use crate::rng::{CombatRngs, Rng};
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

/// Stopgap terminal reward (DESIGN.md, Decision engine): a win is worth 1
/// plus the HP fraction kept at `hp_weight` and 0.1 per unused potion, about
/// the 16 HP a potion is worth to the fights ahead; a loss or a timed-out
/// fight is -1.
pub fn terminal_reward(c: &Combat) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => {
            let hp = c.player.creature.hp as f32 / c.player.creature.max_hp.max(1) as f32;
            1.0 + hp_weight(c) * hp + 0.1 * c.potions.iter().flatten().count() as f32
        }
        _ => -1.0,
    }
}

/// What the HP fraction is worth: half a win, except after an act boss.
/// The Ancient that opens the next act heals all missing HP, or 80% of it
/// under Weary Traveler (`AncientEventModel.BeforeEventStarted`), so only
/// the part it leaves counts.
fn hp_weight(c: &Combat) -> f32 {
    match c.room {
        RoomKind::Boss if c.asc.has(AscensionLevel::WearyTraveler) => 0.5 * 0.2,
        RoomKind::Boss => 0.0,
        _ => 0.5,
    }
}

/// Where a fight (or a search) started, so the potential is zero there.
/// Some fights open with damaged enemies or a relic heal, so this is read
/// off the combat, not the setup.
#[derive(Clone, Copy, Debug)]
pub struct Baseline {
    taken: f32,
    hp: i32,
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
        Self { taken: enemy_hp_taken(c), hp: c.player.creature.hp }
    }
}

/// Potential for reward shaping: half the enemy HP taken since the
/// baseline (`enemy_hp_taken`), minus the fraction of the player's HP lost at the
/// price the terminal reward puts on it.
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
    let lost = (base.hp - c.player.creature.hp.max(0)) as f32 / c.player.creature.max_hp.max(1) as f32;
    0.5 * (enemy_hp_taken(c) - base.taken) - hp_weight(c) * lost
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
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup], hard: &[(Encounter, f32)]) {
        loop {
            self.roll(index, n, cfg, fixed, hard);
            if !self.combat.is_over() {
                return;
            }
        }
    }

    fn roll(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup], hard: &[(Encounter, f32)]) {
        let acts = act_floor(cfg.min_floor).0..=act_floor(cfg.max_floor).0;
        let weighted: Vec<(Encounter, f32)> =
            hard.iter().copied().filter(|(e, _)| acts.contains(&(e.act().index() as u32))).collect();
        self.setup = if !fixed.is_empty() {
            fixed[(index + self.resets * n) % fixed.len()].clone()
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

pub struct VecEnv {
    slots: Vec<Slot>,
    cfg: EnvConfig,
    /// When set, resets cycle through these instead of generating.
    fixed: Vec<FightSetup>,
    /// When set, the `hard_frac` share of fights draws its elite or boss
    /// by these weights instead of evenly.
    hard: Vec<(Encounter, f32)>,
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
            s.reset(i, n, &cfg, &[], &[]);
        }
        Self { slots, cfg, fixed: vec![], hard: vec![] }
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
        let (cfg, fixed, hard) = (self.cfg, &self.fixed, &self.hard);
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.resets = 0;
            s.reset(i, n, &cfg, fixed, hard);
        }
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
        let (cfg, fixed, hard) = (self.cfg, &self.fixed, &self.hard);
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
                    s.reset(i, n, &cfg, fixed, hard);
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

/// FxHash-style mix of an encoding's words in four independent lanes (a
/// single chain waits on each multiply), then murmur3's finalizer.
fn row_hash(floats: &[f32], ids: &[i64], mask: &[bool]) -> u64 {
    const K: u64 = 0x51_7C_C1_B7_27_22_0A_95;
    let mut lanes = [1u64, 2, 3, 4];
    let mut mix = |lane: usize, w: u64| lanes[lane] = (lanes[lane].rotate_left(5) ^ w).wrapping_mul(K);
    for q in floats.chunks(8) {
        for (lane, p) in q.chunks(2).enumerate() {
            mix(lane, p.iter().fold(0, |a, x| a << 32 | x.to_bits() as u64));
        }
    }
    ids.iter().for_each(|&x| mix(0, x as u64));
    mask.chunks(8).for_each(|p| mix(1, p.iter().fold(0, |a, &x| a << 8 | x as u64)));
    let mut h = lanes.iter().fold(0u64, |h, &l| (h.rotate_left(5) ^ l).wrapping_mul(K));
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
pub struct Forks {
    combats: Vec<Combat>,
    /// Per copy: the last player turn it plays, and the root's baseline.
    turns: Vec<u32>,
    bases: Vec<Baseline>,
    /// Per copy: the hash of its encoding, taken in `step` while its state
    /// is still in cache (`observe_unique` would fetch it all again). None
    /// until it first moves.
    hashes: Vec<Option<u64>>,
}

impl Forks {
    pub fn new(root: &Combat, n: usize, groups: usize, seed: u64) -> Self {
        Self::of(&[root], n, groups, seed, 1)
    }

    /// `n` copies of each root, root after root, each played for `depth`
    /// player turns: the rest of the current one, then `depth - 1` more.
    pub fn of(roots: &[&Combat], n: usize, groups: usize, seed: u64, depth: u32) -> Self {
        let per_group = n.div_ceil(groups.max(1));
        let combats = roots
            .par_iter()
            .enumerate()
            .flat_map_iter(|(r, root)| {
                let seed = seed ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                (0..n).map(move |i| {
                    let group = i / per_group;
                    let mut c = (*root).clone();
                    c.script = Default::default();
                    c.rngs = CombatRngs::new(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    Rng::new(seed ^ (group as u64 + 1) << 20).shuffle(&mut c.player.draw);
                    c
                })
            })
            .collect();
        let turns = roots.iter().flat_map(|root| std::iter::repeat_n(root.player.turn + depth.max(1) - 1, n)).collect();
        let bases = roots.iter().flat_map(|root| std::iter::repeat_n(Baseline::of(root), n)).collect();
        let hashes = vec![None; roots.len() * n];
        Self { combats, turns, bases, hashes }
    }

    pub fn len(&self) -> usize {
        self.combats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.combats.is_empty()
    }

    pub fn combat(&self, i: usize) -> &Combat {
        &self.combats[i]
    }

    /// The player's turn is done: the sim moved on to the next one, or
    /// the fight ended.
    pub fn turn_over(&self, i: usize) -> bool {
        let c = &self.combats[i];
        c.is_over() || c.player.turn > self.turns[i]
    }

    /// The forks still in their turn.
    pub fn live(&self) -> Vec<usize> {
        (0..self.combats.len()).filter(|&i| !self.turn_over(i)).collect()
    }

    /// Encode forks `rows`, one row per distinct observation: the distinct
    /// ones are packed at the front of the buffers, `inverse[k]` is the row
    /// fork `rows[k]` encodes to, and the count is returned. Copies of a
    /// root that took the same first action in the same shuffle group
    /// mostly still look the same, so the network sees about a quarter of
    /// the rows (a fourteenth on the first step). Forks are told apart by a
    /// 64-bit hash of their encoding.
    pub fn observe_unique(&self, rows: &[usize], floats: &mut [f32], ids: &mut [i64], mask: &mut [bool], inverse: &mut [usize]) -> usize {
        // Hash each encoding in a scratch row that stays in cache, then
        // write only the distinct ones to the (large) buffers.
        let hashes: Vec<u64> = rows
            .par_iter()
            .map_init(scratch, |(f, i, m), &r| {
                self.hashes[r].unwrap_or_else(|| {
                    encode::encode(&self.combats[r], f, i, m);
                    row_hash(f, i, m)
                })
            })
            .collect();
        let mut distinct = std::collections::HashMap::with_capacity(rows.len());
        let mut first = vec![];
        for (k, h) in hashes.into_iter().enumerate() {
            inverse[k] = *distinct.entry(h).or_insert_with(|| {
                first.push(rows[k]);
                first.len() - 1
            });
        }
        first
            .par_iter()
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .for_each(|(((&r, f), i), m)| encode::encode(&self.combats[r], f, i, m));
        first.len()
    }

    /// Step every fork whose turn is still running; the rest ignore their
    /// action. Writes the shaped reward of each transition, 0 for a fork
    /// that did not move.
    pub fn step(&mut self, actions: &[i64], rewards: &mut [f32]) {
        self.combats
            .par_iter_mut()
            .zip(self.turns.par_iter())
            .zip(self.bases.par_iter())
            .zip(actions.par_iter())
            .zip(rewards.par_iter_mut())
            .zip(self.hashes.par_iter_mut())
            .for_each_init(scratch, |(f, i, m), (((((c, &turn), &base), &a), r), h)| {
                *r = 0.0;
                if !(c.is_over() || c.player.turn > turn) {
                    if let Some(action) = encode::decode(c, a as usize) {
                        let before = potential(c, base);
                        c.step(action);
                        *r = step_reward(before, c, base, c.is_over());
                        *h = (!c.is_over()).then(|| {
                            encode::encode(c, f, i, m);
                            row_hash(f, i, m)
                        });
                    }
                }
            });
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
        let fixed = vec![];
        for s in env.slots.iter_mut() {
            for _ in 0..20 {
                s.reset(0, 4, &cfg, &fixed, &env.hard);
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
        env.slots[0].reset(0, 2, &env.cfg, &env.fixed, &[]);
        assert_eq!(env.slots[0].setup.encounter, setups[2].encounter);
    }
}
