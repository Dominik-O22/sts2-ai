//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to a fresh generated fight (or the next
//! held-out setup) as soon as it ends. Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use rayon::prelude::*;

use crate::combat::{Combat, Outcome};
use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::encounter::{Encounter, Kind};
use crate::gen::{encounter_of_kind, generate, generate_against, FightSetup, BOSS_FLOOR};
use crate::rng::{CombatRngs, Rng};
use crate::types::Ascension;

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
        Self { asc: Ascension(10), min_floor: 1, max_floor: BOSS_FLOOR, max_steps: 500, hard_frac: 0.0 }
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
    pub steps: u32,
    pub floor: u32,
    pub encounter: Encounter,
    pub kind: Kind,
    pub reward: f32,
}

/// Stopgap terminal reward (DESIGN.md, Decision engine): a win is worth 1
/// plus half the HP fraction kept and a little per unused potion; a loss
/// or a timed-out fight is -1.
pub fn terminal_reward(c: &Combat) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => {
            let hp = c.player.creature.hp as f32 / c.player.creature.max_hp.max(1) as f32;
            1.0 + 0.5 * hp + 0.05 * c.potions.iter().flatten().count() as f32
        }
        _ => -1.0,
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

fn enemy_hp_taken(c: &Combat) -> f32 {
    let (hp, max) = c.enemies.iter().fold((0, 0), |(h, m), e| (h + e.creature.hp.max(0), m + e.creature.max_hp));
    1.0 - hp as f32 / max.max(1) as f32
}

impl Baseline {
    pub fn of(c: &Combat) -> Self {
        Self { taken: enemy_hp_taken(c), hp: c.player.creature.hp }
    }
}

/// Potential for reward shaping: half the fraction of enemy HP taken
/// since the baseline, minus half the fraction of the player's HP lost.
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
    0.5 * (enemy_hp_taken(c) - base.taken) - 0.5 * lost
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
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup]) {
        self.setup = if !fixed.is_empty() {
            fixed[(index + self.resets * n) % fixed.len()].clone()
        } else if self.rng.next_float(1.0) < cfg.hard_frac {
            let (kind, floor) = if self.rng.next_int(2) == 0 {
                (Kind::Boss, BOSS_FLOOR)
            } else {
                (Kind::Elite, 5 + self.rng.next_int((BOSS_FLOOR - 5) as usize) as u32)
            };
            let enc = encounter_of_kind(&mut self.rng, kind);
            generate_against(&mut self.rng, floor, cfg.asc, enc)
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
            s.reset(i, n, &cfg, &[]);
        }
        Self { slots, cfg, fixed: vec![] }
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
        self.cfg.min_floor = min.clamp(1, BOSS_FLOOR);
        self.cfg.max_floor = max.clamp(self.cfg.min_floor, BOSS_FLOOR);
    }

    /// Evaluate on fixed setups (the recordings) instead of generated ones.
    /// Resets every env so the first `n` setups start immediately.
    pub fn set_fixed(&mut self, setups: Vec<FightSetup>) {
        self.fixed = setups;
        let n = self.slots.len();
        let (cfg, fixed) = (self.cfg, &self.fixed);
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.resets = 0;
            s.reset(i, n, &cfg, fixed);
        }
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
        let (cfg, fixed) = (self.cfg, &self.fixed);
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
                    s.reset(i, n, &cfg, fixed);
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

/// Copies of one combat stepped together, for a search over the rest of
/// the current turn (the advisor's plan). Each fork forgets the recording's
/// script and rolls its own dice, and the draw pile is reshuffled: the
/// player does not know its order, so the plan must not either. Forks in
/// the same `group` share a shuffle, so plans in a group are compared on
/// the same hidden draws.
pub struct Forks {
    combats: Vec<Combat>,
    turn: u32,
    base: Baseline,
}

impl Forks {
    pub fn new(root: &Combat, n: usize, groups: usize, seed: u64) -> Self {
        let per_group = n.div_ceil(groups.max(1));
        let combats = (0..n)
            .map(|i| {
                let group = i / per_group;
                let mut c = root.clone();
                c.script = Default::default();
                c.rngs = CombatRngs::new(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                Rng::new(seed ^ (group as u64 + 1) << 20).shuffle(&mut c.player.draw);
                c
            })
            .collect();
        Self { combats, turn: root.player.turn, base: Baseline::of(root) }
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
        c.is_over() || c.player.turn > self.turn
    }

    pub fn observe(&self, floats: &mut [f32], ids: &mut [i64], mask: &mut [bool]) {
        self.combats
            .par_iter()
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .for_each(|(((c, f), i), m)| encode::encode(c, f, i, m));
    }

    /// Step every fork whose turn is still running; the rest ignore their
    /// action. Writes the shaped reward of each transition (0 for a fork
    /// that did not move) and the next observation.
    pub fn step(&mut self, actions: &[i64], floats: &mut [f32], ids: &mut [i64], mask: &mut [bool], rewards: &mut [f32]) {
        let (turn, base) = (self.turn, self.base);
        self.combats
            .par_iter_mut()
            .zip(actions.par_iter())
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .zip(rewards.par_iter_mut())
            .for_each(|(((((c, &a), f), i), m), r)| {
                *r = 0.0;
                if !(c.is_over() || c.player.turn > turn) {
                    if let Some(action) = encode::decode(c, a as usize) {
                        let before = potential(c, base);
                        c.step(action);
                        *r = step_reward(before, c, base, c.is_over());
                    }
                }
                encode::encode(c, f, i, m);
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
        let mut rewards = vec![0.0; n];
        forks.observe(&mut floats, &mut ids, &mut mask);
        for _ in 0..40 {
            let actions: Vec<i64> = (0..n)
                .map(|i| {
                    let m = &mask[i * N_ACTIONS..][..N_ACTIONS];
                    (0..N_ACTIONS).filter(|&k| m[k]).max_by_key(|&k| rng.next_int(1000).wrapping_add(k * 0)).unwrap_or(0) as i64
                })
                .collect();
            forks.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards);
            assert!(rewards.iter().all(|r| r.is_finite()));
            if (0..n).all(|i| forks.turn_over(i)) {
                return;
            }
        }
        panic!("some fork never ended its turn");
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
        env.slots[0].reset(0, 2, &env.cfg, &env.fixed);
        assert_eq!(env.slots[0].setup.encounter, setups[2].encounter);
    }
}
