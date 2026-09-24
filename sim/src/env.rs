//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to its next fight as soon as one ends: a fresh
//! generated fight, the next held-out setup, or, in run mode, the next
//! fight of a run the slot plays (`forward::Run`). Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::combat::{After, Combat, Outcome};
use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::encounter::{Encounter, Kind};
use crate::forward::{self, Fought, Next, Run, StartPoint, START_POINTS};
use crate::gen::{act_floor, encounter_of_kind, generate, generate_against, FightSetup, BOSS_FLOOR, LAST_FLOOR};
use crate::rng::{CombatRngs, Rng};
use crate::rooms::{Chooser, Decision, First, Random};
use crate::run::{Carried, RunState};
use crate::runobs::{self, RunObs, RUN_FLOATS, RUN_IDS};
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
#[derive(Clone, Debug)]
pub struct EpisodeEnd {
    pub env: usize,
    pub won: bool,
    pub hp_frac: f32,
    /// HP lost over the fight, as a fraction of max HP.
    pub hp_lost: f32,
    /// Potions drunk (or thrown) over the fight.
    pub potions_used: u32,
    pub steps: u32,
    /// The floor generated for; in run mode, the run's floor.
    pub floor: u32,
    pub encounter: Encounter,
    pub kind: Kind,
    pub reward: f32,
    /// The run the fight was in, in run mode.
    pub run: Option<RunFight>,
}

/// A run fight's place in its run, or, reported by `step_run`, where a
/// run ended between fights.
#[derive(Clone, Debug, PartialEq)]
pub struct RunFight {
    /// The run's seed index (`RunSlot`).
    pub seed: u64,
    /// The act, 0-based.
    pub act: u32,
    /// The run's floor.
    pub floor: u32,
    /// Cards in the deck the fight was fought with.
    pub deck: u32,
    /// How the run ended, when it ended with this fight: the fight lost,
    /// the last boss beaten, or the next fight one the sim cannot build.
    pub end: Option<forward::End>,
    /// Where the run started.
    pub began: Began,
}

/// Where a run started: floor 1, or a start point with a generated player
/// or one the env's own runs carried there (`Starts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Began {
    Floor1,
    Generated(StartPoint),
    Own(StartPoint),
}

/// States kept per start point for `Starts`; past this the oldest go.
pub const START_POOL: usize = 2048;

/// Where run-mode runs start (docs/training.md, The run policy): floor 1
/// with chance `full`, else at a start point drawn by `weights` (in
/// `START_POINTS` order), from a state the env's runs passed there with
/// chance `own` when the pool holds any, else from a generated one. The
/// caller sets the chances; the runs fill the pools.
pub struct Starts {
    pub full: f32,
    pub weights: [f32; START_POINTS.len()],
    pub own: [f32; START_POINTS.len()],
    pool: [Vec<Carried>; START_POINTS.len()],
    /// Where each full pool's next state goes.
    next: [usize; START_POINTS.len()],
}

impl Default for Starts {
    fn default() -> Self {
        Self { full: 1.0, weights: [0.0; START_POINTS.len()], own: [0.0; START_POINTS.len()], pool: Default::default(), next: [0; START_POINTS.len()] }
    }
}

impl Starts {
    /// Where the next run starts, and the state it starts with when it is
    /// the env's own. Draws nothing while every run starts at floor 1.
    fn pick(&self, rng: &mut Rng) -> (Began, Option<Carried>) {
        let total: f32 = self.weights.iter().sum();
        if self.full >= 1.0 || total <= 0.0 || rng.next_float(1.0) < self.full {
            return (Began::Floor1, None);
        }
        let mut x = rng.next_float(total);
        let i = (0..START_POINTS.len()).find(|&i| {
            x -= self.weights[i];
            x < 0.0
        });
        let i = i.unwrap_or(START_POINTS.len() - 1);
        let pool = &self.pool[i];
        if !pool.is_empty() && rng.next_float(1.0) < self.own[i] {
            return (Began::Own(START_POINTS[i]), Some(pool[rng.next_int(pool.len())].clone()));
        }
        (Began::Generated(START_POINTS[i]), None)
    }

    fn keep(&mut self, at: StartPoint, carried: Carried) {
        let i = at.index();
        if self.pool[i].len() < START_POOL {
            self.pool[i].push(carried);
        } else {
            self.pool[i][self.next[i]] = carried;
            self.next[i] = (self.next[i] + 1) % START_POOL;
        }
    }

    pub fn pool_sizes(&self) -> [usize; START_POINTS.len()] {
        self.pool.each_ref().map(Vec::len)
    }
}

/// What a potion kept is worth: about the 16 HP it is worth to the fights
/// ahead, at `hp_weight`. Nothing after the run's last fight.
const POTION_VALUE: f32 = 0.1;

fn potion_value(c: &Combat) -> f32 {
    if c.after == After::End {
        0.0
    } else {
        POTION_VALUE
    }
}

/// Stopgap terminal reward (DESIGN.md, Decision engine): a win is worth 1
/// plus the HP fraction kept at `hp_weight` and `POTION_VALUE` per unused
/// potion; a loss or a timed-out fight is -1.
pub fn terminal_reward(c: &Combat) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => {
            let hp = c.player.creature.hp as f32 / c.player.creature.max_hp.max(1) as f32;
            1.0 + hp_weight(c) * hp + potion_value(c) * potions_held(c) as f32
        }
        _ => -1.0,
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

/// What the HP fraction is worth: half a win, by what follows the fight.
/// After an act boss the Ancient that opens the next act heals all missing
/// HP, or 80% of it under Weary Traveler, so only the part it leaves
/// counts. Under Double Boss the last act's first boss is followed by the
/// second with no rest, so its HP counts in full; after the run's last
/// fight it counts for nothing.
fn hp_weight(c: &Combat) -> f32 {
    match c.after {
        After::Act | After::Boss => 0.5,
        After::Ancient if c.asc.has(AscensionLevel::WearyTraveler) => 0.5 * 0.2,
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
/// baseline (`enemy_hp_taken`), minus the fraction of the player's HP lost
/// and plus the potions gained (a drink counts as one lost) at the prices
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
    let lost = (base.hp - c.player.creature.hp.max(0)) as f32 / c.player.creature.max_hp.max(1) as f32;
    let potions = potions_held(c) as f32 - base.potions as f32;
    0.5 * (enemy_hp_taken(c) - base.taken) - hp_weight(c) * lost + potion_value(c) * potions
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
    /// In run mode, the run whose fight `combat` is.
    run: Option<RunSlot>,
}

/// Who makes a run-mode slot's run decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunChoices {
    /// `rooms::Random`, on an RNG of the run's own (`run_chooser`).
    Random,
    /// `rooms::First`.
    First,
    /// The caller: the run stops at each decision until `step_run`
    /// answers it.
    Caller,
}

enum Choosing {
    Random(Random),
    First,
    Caller(Segment),
}

impl Choosing {
    fn of(choices: RunChoices, seed: u64) -> Self {
        match choices {
            RunChoices::Random => Choosing::Random(run_chooser(seed)),
            RunChoices::First => Choosing::First,
            RunChoices::Caller => Choosing::Caller(Segment::default()),
        }
    }
}

/// A run between fights under the caller's choices: the fight it came
/// from and the answers given since. `Run::next` cannot stop halfway, so
/// each answer plays the segment again from the last fight on a copy of
/// the run, with the answers so far, up to the first decision not yet
/// answered (`Replay`), and the copy is kept once it reaches a fight or
/// the run's end. The same run and answers play the same way, so the
/// replays agree.
#[derive(Default)]
struct Segment {
    fought: Option<Fought>,
    answers: Vec<usize>,
    /// The decision the run waits at.
    waiting: Option<RunObs>,
}

/// Unwinds out of `Run::next` at the first decision the caller has not
/// answered, carrying it.
struct Unanswered(RunObs);

/// Answers from a segment's list, then stops the run at the next decision.
/// A decision with one option or none takes the first without asking;
/// nothing is learned from it.
struct Replay<'a> {
    answers: &'a [usize],
    next: usize,
}

impl Chooser for Replay<'_> {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
        if runobs::option_count(decision) <= 1 {
            return 0;
        }
        if let Some(&answer) = self.answers.get(self.next) {
            self.next += 1;
            return answer;
        }
        std::panic::resume_unwind(Box::new(Unanswered(runobs::observe(run, decision))))
    }
}

impl Segment {
    /// Plays `run`'s copy through the answers: to the next fight or end
    /// with the copy, or the decision it stops at.
    fn replay(&self, run: &Run) -> Result<(Run, Next), RunObs> {
        let mut run = run.clone();
        let mut chooser = Replay { answers: &self.answers, next: 0 };
        let played = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.next(self.fought, &mut chooser)));
        match played {
            Ok(next) => Ok((run, next)),
            Err(payload) => match payload.downcast::<Unanswered>() {
                Ok(stop) => Err(stop.0),
                Err(other) => std::panic::resume_unwind(other),
            },
        }
    }
}

/// A slot's run, with the chooser making its run decisions. A slot's
/// `k`-th run is seed index `base + slot + k * n`, so a base seed and a
/// batch size name every run the batch plays.
struct RunSlot {
    run: Run,
    chooser: Choosing,
    choices: RunChoices,
    asc: Ascension,
    base: u64,
    /// The current run's seed index.
    seed: u64,
    /// Runs the slot has started.
    started: u64,
    began: Began,
    /// Draws where runs start, never on a run's streams.
    rng: Rng,
    starts: Arc<Mutex<Starts>>,
}

impl RunSlot {
    fn new(asc: Ascension, base: u64, index: usize, choices: RunChoices, starts: Arc<Mutex<Starts>>) -> Self {
        let seed = base + index as u64;
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x57A7);
        let (run, began) = begin(&starts, &mut rng, seed, asc);
        Self { run, chooser: Choosing::of(choices, seed), choices, asc, base, seed, started: 1, began, rng, starts }
    }

    /// Hands the start points the run passed to the pools.
    fn keep_passed(&mut self) {
        if !self.run.passed.is_empty() {
            let mut starts = self.starts.lock().expect("start pools");
            for (at, carried) in self.run.passed.drain(..) {
                starts.keep(at, carried);
            }
        }
    }

    /// The decision the run waits at for the caller, if it does.
    fn waiting(&self) -> Option<&RunObs> {
        match &self.chooser {
            Choosing::Caller(segment) => segment.waiting.as_ref(),
            _ => None,
        }
    }

    fn report(&self, deck: usize) -> RunFight {
        RunFight { seed: self.seed, act: self.run.state.act as u32, floor: self.run.state.floor as u32, deck: deck as u32, end: None, began: self.began }
    }

    /// Writes the fight `setup` started back into the run, now that it is
    /// over as `combat`, and plays on (`advance`). The report is the
    /// finished fight's place in its run (none for the first fight of a
    /// slot's first run, which follows no fight).
    fn next_fight(&mut self, setup: &FightSetup, combat: &Combat, index: usize, n: usize) -> (Option<FightSetup>, Option<RunFight>) {
        let fought = self.run.fighting().then(|| Fought {
            won: combat.outcome == Some(Outcome::Won),
            gold_proportion: self.run.state.end_fight(setup, combat),
        });
        let report = fought.map(|_| self.report(setup.deck.len()));
        self.advance(fought, report, index, n)
    }

    /// Answers the decision the run waits at with option `option` of its
    /// encoding, and plays on (`advance`). The report says where the run
    /// ended, if it did.
    fn answer(&mut self, option: usize, index: usize, n: usize) -> (Option<FightSetup>, Option<RunFight>) {
        let report = self.report(self.run.state.deck.len());
        let Choosing::Caller(segment) = &mut self.chooser else { panic!("env {index}: no run decision to answer") };
        let waiting = segment.waiting.take().unwrap_or_else(|| panic!("env {index}: not at a run decision"));
        let answer = *waiting.answers.get(option).unwrap_or_else(|| panic!("env {index}: option {option} of {}", waiting.answers.len()));
        segment.answers.push(answer);
        self.advance(None, Some(report), index, n)
    }

    /// Plays on to the next fight, through as many fresh runs as it takes,
    /// or to a decision the caller answers (None). The first run to end
    /// sets `report`'s end and floor; later ones are fresh runs that ended
    /// before a fight.
    fn advance(&mut self, mut fought: Option<Fought>, mut report: Option<RunFight>, index: usize, n: usize) -> (Option<FightSetup>, Option<RunFight>) {
        loop {
            let next = match &mut self.chooser {
                Choosing::Random(chooser) => self.run.next(fought.take(), chooser),
                Choosing::First => self.run.next(fought.take(), &mut First),
                Choosing::Caller(segment) => {
                    segment.fought = fought.take().or(segment.fought);
                    match segment.replay(&self.run) {
                        Err(decision) => {
                            segment.waiting = Some(decision);
                            return (None, report);
                        }
                        Ok((run, next)) => {
                            self.run = run;
                            *segment = Segment::default();
                            next
                        }
                    }
                }
            };
            self.keep_passed();
            match next {
                Next::Fight(setup) => return (Some(setup), report),
                Next::End(end) => {
                    if let Some(report) = report.as_mut().filter(|r| r.end.is_none()) {
                        report.floor = self.run.state.floor as u32;
                        report.end = Some(end);
                    }
                    self.seed = self.base + index as u64 + self.started * n as u64;
                    self.started += 1;
                    (self.run, self.began) = begin(&self.starts, &mut self.rng, self.seed, self.asc);
                    self.chooser = Choosing::of(self.choices, self.seed);
                }
            }
        }
    }
}

/// Run seed index `seed`, started where `starts` picks: a generated
/// player is rolled for the floor the start point stands at, as the
/// generator counts floors (sixteen an act).
fn begin(starts: &Mutex<Starts>, rng: &mut Rng, seed: u64, asc: Ascension) -> (Run, Began) {
    let (began, own) = starts.lock().expect("start pools").pick(rng);
    let run = match (began, own) {
        (Began::Own(at), Some(carried)) => Run::start_at(&run_seed(seed), asc, at, carried),
        (Began::Generated(at), _) => {
            let floor = match at {
                StartPoint::Entrance(act) => BOSS_FLOOR * act as u32,
                StartPoint::BossDoor(act) => BOSS_FLOOR * act as u32 + BOSS_FLOOR - 1,
            };
            Run::start_at(&run_seed(seed), asc, at, Carried::generated(&generate(rng, floor, asc)))
        }
        _ => Run::new(&run_seed(seed), asc),
    };
    (run, began)
}

/// The game seed of run seed index `seed`.
fn run_seed(seed: u64) -> String {
    format!("SIM{seed}")
}

/// The chooser for run seed index `seed`, on an RNG of its own.
fn run_chooser(seed: u64) -> Random {
    Random(Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED))
}

impl Slot {
    /// Start the next fight. One already over before the first decision
    /// (Whispering Earring can win turn 1 on its own) is skipped: there is
    /// nothing in it to act on. In run mode, returns the finished fight's
    /// place in its run; a run that waits at a decision for the caller
    /// starts no fight until `step_run` answers it.
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup], hard: &[(Encounter, f32)]) -> Option<RunFight> {
        let mut report = None;
        loop {
            let rolled = self.roll(index, n, cfg, fixed, hard);
            report = report.or(rolled);
            if !self.combat.is_over() || self.waiting() {
                return report;
            }
        }
    }

    /// Whether the slot's run waits at a decision for the caller.
    fn waiting(&self) -> bool {
        self.run.as_ref().is_some_and(|r| r.waiting().is_some())
    }

    /// Answers the run decision the slot waits at and starts the fight the
    /// run reaches, if it reaches one; a fight over at once is skipped as
    /// in `reset`. Returns where the run ended, if it did.
    fn answer(&mut self, option: usize, index: usize, n: usize, cfg: &EnvConfig) -> Option<RunFight> {
        let run = self.run.as_mut().expect("run mode");
        let (setup, report) = run.answer(option, index, n);
        if let Some(setup) = setup {
            self.start(setup);
            if self.combat.is_over() {
                let later = self.reset(index, n, cfg, &[], &[]);
                return report.filter(|r| r.end.is_some()).or(later.filter(|r| r.end.is_some()));
            }
        }
        report.filter(|r| r.end.is_some())
    }

    fn roll(&mut self, index: usize, n: usize, cfg: &EnvConfig, fixed: &[FightSetup], hard: &[(Encounter, f32)]) -> Option<RunFight> {
        let acts = act_floor(cfg.min_floor).0..=act_floor(cfg.max_floor).0;
        let weighted: Vec<(Encounter, f32)> =
            hard.iter().copied().filter(|(e, _)| acts.contains(&(e.act().index() as u32))).collect();
        let mut report = None;
        let setup = if let Some(run) = &mut self.run {
            let (setup, fought) = run.next_fight(&self.setup, &self.combat, index, n);
            report = fought;
            match setup {
                Some(setup) => setup,
                None => return report,
            }
        } else if !fixed.is_empty() {
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
        self.start(setup);
        report
    }

    fn start(&mut self, setup: FightSetup) {
        self.setup = setup;
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
            run: None,
        }
    }
}

pub struct VecEnv {
    slots: Vec<Slot>,
    /// Where run-mode runs start, shared with every slot's run.
    starts: Arc<Mutex<Starts>>,
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
                Slot { base: Baseline::of(&combat), combat, setup, steps: 0, rng, resets: 0, run: None }
            })
            .collect();
        for (i, s) in slots.iter_mut().enumerate() {
            s.reset(i, n, &cfg, &[], &[]);
        }
        Self { slots, cfg, fixed: vec![], hard: vec![], starts: Default::default() }
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
            s.run = None;
            s.reset(i, n, &cfg, fixed, hard);
        }
    }

    /// Run mode: every env plays runs at `asc`, fight after fight, its run
    /// decisions made by `choices`, a fresh run from the next seed index
    /// after each ends (`RunSlot`). Starts every env's first run; under
    /// `RunChoices::Caller` each then waits at its first decision.
    pub fn set_runs(&mut self, asc: Ascension, base: u64, choices: RunChoices) {
        self.fixed.clear();
        let n = self.slots.len();
        let (cfg, hard, starts) = (self.cfg, &self.hard, &self.starts);
        self.slots.par_iter_mut().enumerate().for_each(|(i, s)| {
            s.resets = 0;
            s.run = Some(RunSlot::new(asc, base, i, choices, starts.clone()));
            s.reset(i, n, &cfg, &[], hard);
        });
    }

    /// Where the runs that start from now on start (`Starts`).
    pub fn set_starts(&mut self, full: f32, weights: [f32; START_POINTS.len()], own: [f32; START_POINTS.len()]) {
        let mut starts = self.starts.lock().expect("start pools");
        (starts.full, starts.weights, starts.own) = (full, weights, own);
    }

    /// States kept per start point, in `START_POINTS` order.
    pub fn start_pools(&self) -> [usize; START_POINTS.len()] {
        self.starts.lock().expect("start pools").pool_sizes()
    }

    /// The envs whose run waits at a decision for the caller.
    pub fn run_waiting(&self) -> Vec<usize> {
        (0..self.slots.len()).filter(|&i| self.slots[i].waiting()).collect()
    }

    /// Encode the run decision each of `envs` waits at, a row each, laid
    /// out per `runobs`.
    pub fn observe_run(&self, envs: &[usize], floats: &mut [f32], ids: &mut [i64]) {
        assert!(floats.len() >= envs.len() * RUN_FLOATS && ids.len() >= envs.len() * RUN_IDS, "run buffers too small");
        floats.par_chunks_mut(RUN_FLOATS).zip(ids.par_chunks_mut(RUN_IDS)).zip(envs.par_iter()).for_each(|((f, i), &env)| {
            let obs = self.slots[env].run.as_ref().and_then(RunSlot::waiting).unwrap_or_else(|| panic!("env {env}: not at a run decision"));
            f.copy_from_slice(&obs.floats);
            i.copy_from_slice(&obs.ids);
        });
    }

    /// Answer the run decision each of `envs` waits at with its option
    /// token `options[k]`, and play each run on: to its next decision, or
    /// to a fight, which starts and whose combat row is written to the
    /// observation buffers. No combat steps. Returns the runs that ended,
    /// by env.
    pub fn step_run(&mut self, envs: &[usize], options: &[i64], floats: &mut [f32], ids: &mut [i64], mask: &mut [bool]) -> Vec<(usize, RunFight)> {
        self.check_buffers(floats, ids, mask);
        assert_eq!(envs.len(), options.len(), "an option per env");
        let n = self.slots.len();
        let mut chosen = vec![None; n];
        for (&env, &option) in envs.iter().zip(options) {
            assert!(chosen[env].replace(option as usize).is_none(), "env {env} answered twice");
        }
        let cfg = self.cfg;
        self.slots
            .par_iter_mut()
            .enumerate()
            .zip(chosen.par_iter())
            .zip(floats.par_chunks_mut(N_FLOATS))
            .zip(ids.par_chunks_mut(N_IDS))
            .zip(mask.par_chunks_mut(N_ACTIONS))
            .filter_map(|(((((i, s), option), f), ids), m)| {
                let ended = s.answer((*option)?, i, n, &cfg);
                encode::encode(&s.combat, f, ids, m);
                ended.map(|r| (i, r))
            })
            .collect()
    }

    pub fn combat(&self, i: usize) -> &Combat {
        &self.slots[i].combat
    }

    pub fn setup(&self, i: usize) -> &FightSetup {
        &self.slots[i].setup
    }

    /// Encode every env's current state. A run waiting at a decision
    /// shows the fight it last finished.
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
                // A run waiting at a decision sits the step out.
                if s.waiting() {
                    (*r, *d) = (0.0, false);
                    return None;
                }
                let action = encode::decode(&s.combat, a as usize)
                    .unwrap_or_else(|| panic!("env {i}: action {a} is not legal; legal: {:?}", s.combat.legal_actions()));
                let before = potential(&s.combat, s.base);
                s.combat.step(action);
                s.steps += 1;
                let over = s.combat.is_over() || s.steps >= cfg.max_steps;
                let mut end = over.then(|| s.end(i));
                *r = step_reward(before, &s.combat, s.base, over);
                *d = over;
                if let Some(end) = &mut end {
                    end.run = s.reset(i, n, &cfg, fixed, hard);
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
        let half = 0.5 * c.player.creature.hp as f32 / c.player.creature.max_hp as f32;
        assert_eq!(reward(&mut c, After::Act), 1.0 + half + 0.1);
        assert_eq!(reward(&mut c, After::Boss), 1.0 + half + 0.1, "the second boss follows with no rest");
        assert_eq!(reward(&mut c, After::Ancient), 1.0 + 0.2 * half + 0.1, "the Ancient heals 80% at A10");
        assert_eq!(reward(&mut c, After::End), 1.0, "nothing is left to spend");
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

    /// One run as a run-mode env played it: each fight's setup and the
    /// combat it ended as, and how the run ended.
    struct PlayedRun {
        seed: u64,
        fights: Vec<(FightSetup, Combat)>,
        end: forward::End,
    }

    /// Steps a run-mode env with the first legal action everywhere until
    /// every slot has finished `runs` runs; returns the runs finished and
    /// every fight's report.
    fn play_runs(n: usize, runs: usize) -> (Vec<PlayedRun>, Vec<String>) {
        let mut env = VecEnv::new(n, 5, EnvConfig::default());
        env.set_runs(Ascension(10), 100, RunChoices::Random);
        let (mut floats, mut ids, mut mask) = (vec![0.0; n * N_FLOATS], vec![0; n * N_IDS], vec![false; n * N_ACTIONS]);
        let (mut rewards, mut dones) = (vec![0.0; n], vec![false; n]);
        env.observe(&mut floats, &mut ids, &mut mask);
        let mut open: Vec<Vec<(FightSetup, Combat)>> = vec![vec![]; n];
        let mut finished = vec![0; n];
        let (mut played, mut reports) = (vec![], vec![]);
        while finished.iter().any(|&f| f < runs) {
            let actions: Vec<i64> = (0..n).map(|i| mask[i * N_ACTIONS..][..N_ACTIONS].iter().position(|&m| m).unwrap() as i64).collect();
            let before: Vec<(FightSetup, Combat)> = env.slots.iter().map(|s| (s.setup.clone(), s.combat.clone())).collect();
            for e in env.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards, &mut dones) {
                let (setup, mut combat) = before[e.env].clone();
                combat.step(encode::decode(&combat, actions[e.env] as usize).unwrap());
                open[e.env].push((setup, combat));
                let run = e.run.clone().expect("a run fight");
                assert_eq!((run.seed - 100) % n as u64, e.env as u64, "a slot plays its own seeds");
                if let Some(end) = run.end {
                    played.push(PlayedRun { seed: run.seed, fights: std::mem::take(&mut open[e.env]), end });
                    finished[e.env] += 1;
                }
                reports.push(format!("{e:?}"));
            }
        }
        (played, reports)
    }

    /// With every run sent to the last boss door, the runs after each
    /// env's first start there with a generated player and say so; a pool
    /// holding a state hands it out once `own` asks for it.
    #[test]
    fn runs_start_where_the_starts_say() {
        let n = 4;
        let mut env = VecEnv::new(n, 5, EnvConfig::default());
        env.set_runs(Ascension(10), 100, RunChoices::First);
        let door = StartPoint::BossDoor(2);
        let mut weights = [0.0; START_POINTS.len()];
        weights[door.index()] = 1.0;
        env.set_starts(0.0, weights, [1.0; START_POINTS.len()]);
        let (mut floats, mut ids, mut mask) = (vec![0.0; n * N_FLOATS], vec![0; n * N_IDS], vec![false; n * N_ACTIONS]);
        let (mut rewards, mut dones) = (vec![0.0; n], vec![false; n]);
        env.observe(&mut floats, &mut ids, &mut mask);
        let mut late = vec![];
        while late.len() < 8 {
            let actions: Vec<i64> = (0..n).map(|i| mask[i * N_ACTIONS..][..N_ACTIONS].iter().position(|&m| m).unwrap() as i64).collect();
            for e in env.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards, &mut dones) {
                let run = e.run.expect("a run fight");
                if run.began != Began::Floor1 {
                    assert_eq!(run.began, Began::Generated(door), "an empty pool falls back to a generated player");
                    assert!(run.floor >= 47, "a fight past the boss door on floor {}", run.floor);
                    late.push(run);
                }
            }
        }

        let mut starts = Starts { full: 0.0, weights, own: [1.0; START_POINTS.len()], ..Default::default() };
        let carried = Run::new("KEPT", Ascension(10)).state.carried();
        starts.keep(door, carried.clone());
        assert_eq!(starts.pick(&mut Rng::new(1)), (Began::Own(door), Some(carried)));
    }

    /// Run mode plays runs across fight boundaries as `forward::play` does
    /// given the same fights: the same run state at every fight (the setup
    /// carries the deck, relics with their counters, potions, HP, max HP,
    /// gold and floor) and the same end. A seed plays the same way twice.
    #[test]
    fn run_mode_plays_the_forward_run() {
        let (played, reports) = play_runs(3, 2);
        assert!(played.len() >= 6);
        assert!(played.iter().any(|r| r.fights.len() > 1), "some run went past its first fight");
        for run in &played {
            let mut fights = run.fights.iter();
            let forward = forward::play(&run_seed(run.seed), Ascension(10), &mut run_chooser(run.seed), &mut |state: &mut crate::run::RunState, setup: FightSetup| {
                let (recorded, combat) = fights.next().expect("forward played more fights");
                assert_eq!(format!("{setup:?}"), format!("{recorded:?}"), "seed {}: fight {}", run.seed, state.floor);
                Fought { won: combat.outcome == Some(Outcome::Won), gold_proportion: state.end_fight(&setup, combat) }
            });
            assert_eq!(forward.end, run.end, "seed {}", run.seed);
            assert_eq!(forward.fights, run.fights.len(), "seed {}", run.seed);
        }
        assert_eq!(reports, play_runs(3, 2).1);
    }

    /// Answers a run's decisions from a list, taking the only option of a
    /// decision with one as `Replay` does.
    struct Scripted(std::vec::IntoIter<usize>);

    impl Chooser for Scripted {
        fn choose(&mut self, _: &RunState, decision: Decision<'_>) -> usize {
            if runobs::option_count(decision) <= 1 {
                return 0;
            }
            self.0.next().expect("an answer for every decision")
        }
    }

    /// Runs under the caller's choices play as `forward::play` does with
    /// the same choices and fights: random options answered through
    /// `step_run` until none wait, then a combat step with the first legal
    /// action, and each finished run played forward with its answers.
    #[test]
    fn caller_choices_play_the_forward_run() {
        let n = 4;
        let mut env = VecEnv::new(n, 5, EnvConfig::default());
        env.set_runs(Ascension(10), 300, RunChoices::Caller);
        let (mut floats, mut ids, mut mask) = (vec![0.0; n * N_FLOATS], vec![0; n * N_IDS], vec![false; n * N_ACTIONS]);
        let (mut rewards, mut dones) = (vec![0.0; n], vec![false; n]);
        let (mut run_f, mut run_i) = (vec![0.0; n * RUN_FLOATS], vec![0; n * RUN_IDS]);
        let mut rng = Rng::new(3);
        let mut answers: std::collections::HashMap<u64, Vec<usize>> = Default::default();
        let mut fights: std::collections::HashMap<u64, Vec<(FightSetup, Combat)>> = Default::default();
        let mut ended: Vec<(u64, forward::End)> = vec![];
        let mut decisions = 0;
        while ended.len() < 8 {
            loop {
                let waiting = env.run_waiting();
                if waiting.is_empty() {
                    break;
                }
                env.observe_run(&waiting, &mut run_f, &mut run_i);
                let options: Vec<i64> = waiting
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let slot = env.slots[i].run.as_ref().unwrap();
                        let obs = slot.waiting().unwrap();
                        assert_eq!(obs.floats, run_f[k * RUN_FLOATS..][..RUN_FLOATS], "env {i}: the row is its decision");
                        let present = (0..runobs::MAX_OPTIONS).filter(|&o| run_f[k * RUN_FLOATS + runobs::F_OPTIONS + o * runobs::OPTION_FLOATS] == 1.0).count();
                        assert_eq!(present, obs.answers.len(), "env {i}: an option token per answer");
                        assert!(present > 1, "env {i}: a decision with one option is taken without asking");
                        let option = rng.next_int(present);
                        answers.entry(slot.seed).or_default().push(obs.answers[option]);
                        option as i64
                    })
                    .collect();
                decisions += waiting.len();
                for (_, run) in env.step_run(&waiting, &options, &mut floats, &mut ids, &mut mask) {
                    ended.push((run.seed, run.end.expect("an ended run")));
                }
            }
            let actions: Vec<i64> = (0..n).map(|i| mask[i * N_ACTIONS..][..N_ACTIONS].iter().position(|&m| m).unwrap() as i64).collect();
            let before: Vec<(FightSetup, Combat)> = env.slots.iter().map(|s| (s.setup.clone(), s.combat.clone())).collect();
            for e in env.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards, &mut dones) {
                let (setup, mut combat) = before[e.env].clone();
                combat.step(encode::decode(&combat, actions[e.env] as usize).unwrap());
                let run = e.run.expect("a run fight");
                fights.entry(run.seed).or_default().push((setup, combat));
                if let Some(end) = run.end {
                    ended.push((run.seed, end));
                }
            }
        }
        assert!(decisions > 50, "{decisions} decisions");
        for (seed, end) in ended {
            let mut recorded = fights.remove(&seed).unwrap_or_default().into_iter();
            let mut chooser = Scripted(answers.remove(&seed).unwrap_or_default().into_iter());
            let forward = forward::play(&run_seed(seed), Ascension(10), &mut chooser, &mut |state: &mut RunState, setup: FightSetup| {
                let (fought, combat) = recorded.next().expect("forward played more fights");
                assert_eq!(format!("{setup:?}"), format!("{fought:?}"), "seed {seed}: fight on floor {}", state.floor);
                Fought { won: combat.outcome == Some(Outcome::Won), gold_proportion: state.end_fight(&setup, &combat) }
            });
            assert_eq!(forward.end, end, "seed {seed}");
            assert!(chooser.0.next().is_none(), "seed {seed}: answers left over");
        }
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
