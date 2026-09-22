//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to a fresh generated fight (or the next
//! held-out setup) as soon as it ends. Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use rayon::prelude::*;

use crate::combat::{Combat, Outcome};
use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::encounter::Encounter;
use crate::gen::{generate, FightSetup, BOSS_FLOOR};
use crate::rng::Rng;
use crate::types::Ascension;

#[derive(Clone, Copy, Debug)]
pub struct EnvConfig {
    pub asc: Ascension,
    /// Generated fights sample a floor uniformly in `min_floor..=max_floor`.
    pub min_floor: u32,
    pub max_floor: u32,
    /// A fight still running after this many actions counts as lost.
    pub max_steps: u32,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self { asc: Ascension(10), min_floor: 1, max_floor: BOSS_FLOOR, max_steps: 500 }
    }
}

/// What a finished fight looked like, for logging.
#[derive(Clone, Copy, Debug)]
pub struct EpisodeEnd {
    pub env: usize,
    pub won: bool,
    pub hp_frac: f32,
    pub steps: u32,
    pub floor: u32,
    pub encounter: Encounter,
    pub reward: f32,
}

/// Stopgap terminal reward (DESIGN.md, Decision engine): a win is worth 1
/// plus half the HP fraction kept and a little per unused potion; a loss
/// or a timed-out fight is -1.
fn terminal_reward(c: &Combat) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => {
            let hp = c.player.creature.hp as f32 / c.player.creature.max_hp.max(1) as f32;
            1.0 + 0.5 * hp + 0.05 * c.potions.iter().flatten().count() as f32
        }
        _ => -1.0,
    }
}

struct Slot {
    combat: Combat,
    setup: FightSetup,
    steps: u32,
    rng: Rng,
    resets: usize,
}

impl Slot {
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup]) {
        self.setup = if fixed.is_empty() {
            let floor = cfg.min_floor + self.rng.next_int((cfg.max_floor - cfg.min_floor + 1) as usize) as u32;
            generate(&mut self.rng, floor, cfg.asc)
        } else {
            fixed[(index + self.resets * n) % fixed.len()].clone()
        };
        self.resets += 1;
        self.steps = 0;
        self.combat = self.setup.combat(self.rng.next_u64());
    }

    fn end(&self, index: usize) -> EpisodeEnd {
        let c = &self.combat;
        EpisodeEnd {
            env: index,
            won: c.outcome == Some(Outcome::Won),
            hp_frac: c.player.creature.hp.max(0) as f32 / c.player.creature.max_hp.max(1) as f32,
            steps: self.steps,
            floor: self.setup.floor,
            encounter: self.setup.encounter,
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
                Slot { combat: setup.combat(0), setup, steps: 0, rng, resets: 0 }
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

    /// Curriculum knob: which floors generated fights come from. Takes
    /// effect at each env's next reset.
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
                s.combat.step(action);
                s.steps += 1;
                let over = s.combat.is_over() || s.steps >= cfg.max_steps;
                let end = over.then(|| s.end(i));
                *r = end.map_or(0.0, |e| e.reward);
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
            for e in &ends {
                assert!(dones[e.env]);
                assert_eq!(rewards[e.env], e.reward);
                assert!(e.won == (e.reward > 0.0));
            }
            ended += ends.len();
        }
        assert!(ended > n, "fights should have ended and restarted");
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
