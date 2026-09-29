//! Vectorized training environments: `n` combats stepped together across
//! all cores, each resetting to its next fight as soon as one ends: a fresh
//! generated fight, the next held-out setup, or, in run mode, the next
//! fight of a run the slot plays (`forward::Run`). Observations are written into
//! caller-owned buffers laid out per `encode`, so the Python side can hand
//! over numpy arrays without copies.

use std::collections::{BTreeMap, BTreeSet, HashMap};
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
/// or one the env's own runs carried there, or a winner's player at an
/// act's entrance (`Starts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Began {
    Floor1,
    Generated(StartPoint),
    Own(StartPoint),
    Winner(StartPoint),
}

/// States kept per start point for `Starts`; past this the oldest go.
pub const START_POOL: usize = 2048;

/// Where run-mode runs start (docs/training.md, The run policy): with a
/// winner's player at an act's entrance with chance `winner` when there are
/// any (`winners`), else
/// floor 1 with chance `full`, else at a start point drawn by `weights` (in
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
    pub winner: f32,
    /// Winners' players by start point, as the history walk left them at
    /// an act's entrance (`history::entrances`).
    winners: [Vec<Carried>; START_POINTS.len()],
}

impl Default for Starts {
    fn default() -> Self {
        Self {
            full: 1.0,
            weights: [0.0; START_POINTS.len()],
            own: [0.0; START_POINTS.len()],
            pool: Default::default(),
            next: [0; START_POINTS.len()],
            winner: 0.0,
            winners: Default::default(),
        }
    }
}

impl Starts {
    /// A winner's player to start with, with chance `winner`: a start
    /// point holding any, evenly, then one of its players. Draws nothing
    /// while there are none or the chance is 0.
    fn pick_winner(&self, rng: &mut Rng) -> Option<(StartPoint, Carried)> {
        let points: Vec<usize> = (0..START_POINTS.len()).filter(|&i| !self.winners[i].is_empty()).collect();
        if points.is_empty() || self.winner <= 0.0 || rng.next_float(1.0) >= self.winner {
            return None;
        }
        let i = points[rng.next_int(points.len())];
        Some((START_POINTS[i], self.winners[i][rng.next_int(self.winners[i].len())].clone()))
    }

    /// Winners' players held per start point.
    pub fn winner_sizes(&self) -> [usize; START_POINTS.len()] {
        self.winners.each_ref().map(Vec::len)
    }

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
/// ahead, at `hp_weight`. Nothing after the run's last fight. The run value
/// heads of run-6 and run-7 price it nearer 4 HP in acts 1 and 2
/// (runs/scratch-keep/runvalueprobe.py); that is a separate experiment.
const POTION_VALUE: f32 = 0.1;

/// What max HP is worth per fraction of the fight's starting max HP: twice
/// HP at `hp_weight`. A point of max HP is a point of HP that every heal
/// after it restores: at A10 the Ancient opening the next act heals 80% of
/// what is missing and a rest 30% of max HP (`AncientEventModel`,
/// `HealRestSiteOption`), so it is worth about 3.4 HP points in act 1, 2 in
/// act 2 and 0.6 in act 3; the combat does not know its act, so this is
/// the middle. Paper Cuts takes it, Feed gives it.
const MAX_HP_VALUE: f32 = 1.0;

fn max_hp_value(c: &Combat) -> f32 {
    if c.after == After::End {
        0.0
    } else {
        MAX_HP_VALUE
    }
}

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
/// plus the HP fraction kept at `hp_weight`, the max HP gained or lost at
/// `max_hp_value` and `POTION_VALUE` per unused potion; a loss or a
/// timed-out fight is -1 plus `LOSS_DAMAGE` per fraction of the enemies'
/// HP taken. HP is counted against the max HP the fight (or search) began
/// with: against the current max, losing max HP raised the fraction and
/// paid the policy for every Paper Cuts hit.
pub fn terminal_reward(c: &Combat, base: Baseline) -> f32 {
    match c.outcome {
        Some(Outcome::Won) => {
            let start_max = base.max_hp.max(1) as f32;
            let hp = c.player.creature.hp as f32 / start_max;
            let max = (c.player.creature.max_hp - base.max_hp) as f32 / start_max;
            1.0 + hp_weight(c) * hp + max_hp_value(c) * max + potion_value(c) * potions_held(c) as f32
        }
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
    max_hp: i32,
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
        Self { taken: enemy_hp_taken(c), hp: c.player.creature.hp, max_hp: c.player.creature.max_hp, potions: potions_held(c) }
    }
}

/// Potential for reward shaping: half the enemy HP taken since the
/// baseline (`enemy_hp_taken`), minus the fraction of the player's HP lost,
/// plus the max HP gained (both against the max HP at the baseline) and
/// plus the potions gained (a drink counts as one lost) at the prices
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
    let start_max = base.max_hp.max(1) as f32;
    let lost = (base.hp - c.player.creature.hp.max(0)) as f32 / start_max;
    let max = (c.player.creature.max_hp - base.max_hp) as f32 / start_max;
    let potions = potions_held(c) as f32 - base.potions as f32;
    0.5 * (enemy_hp_taken(c) - base.taken) - hp_weight(c) * lost + max_hp_value(c) * max + potion_value(c) * potions
}

/// The reward for a transition: the potential change, plus the terminal
/// reward when `over` (a timed-out fight is over without an outcome).
pub fn step_reward(before: f32, c: &Combat, base: Baseline, over: bool) -> f32 {
    if over {
        terminal_reward(c, base) - before
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
    /// When on (`VecEnv::log_fights`), each run elite and boss fight as it
    /// starts, in the recorder's `start` format with the run's act.
    fights: Option<Vec<String>>,
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

/// Unwinds out of a branch at the first decision it has not answered,
/// carrying the run there and, at a decision short of a map step, the
/// decision encoded.
struct Halted(RunState, Option<RunObs>);

/// Answers from a list as `Replay` does, a segment's answers and then a
/// branch's beyond them, and asks for the reseed right after the branch's
/// first answer, the option under scrutiny, so that option takes effect
/// on streams drawn from `sample` alone. Stops the run at the next
/// decision not answered (`Halted`).
struct Branch<'a> {
    answers: &'a [usize],
    /// Index in `answers` of the option under scrutiny.
    option: usize,
    next: usize,
    /// Until the option's choice takes it.
    sample: Option<u64>,
}

impl Chooser for Branch<'_> {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
        let path = matches!(decision, Decision::Path(..));
        // A map step with one option is not asked, but it still ends the
        // room: the branch settles there rather than walk into the next.
        if runobs::option_count(decision) <= 1 && !(path && self.next >= self.answers.len()) {
            return 0;
        }
        if let Some(&answer) = self.answers.get(self.next) {
            self.next += 1;
            return answer;
        }
        std::panic::resume_unwind(Box::new(Halted(run.clone(), (!path).then(|| runobs::observe(run, decision)))))
    }

    fn reseed(&mut self) -> Option<u64> {
        if self.next == self.option + 1 { self.sample.take() } else { None }
    }
}

/// Where a branch stopped.
enum Stop {
    /// A decision it has not answered: a map step (None), where an
    /// afterstate settles, or a sub-decision in the same room.
    Decision(RunState, Option<RunObs>),
    /// It played through, to a fight or the run's end.
    Through(RunState, Option<forward::End>),
}

/// Plays `play` under `chooser`: what it returned, or where it halted.
/// Panics if the option was answered and nothing took the reseed, which a
/// chooser wrapper that drops `Chooser::reseed` would cause, and with it a
/// leak of the run's own draws.
fn halt<T>(chooser: &mut Branch, play: impl FnOnce(&mut Branch) -> T) -> Result<T, Halted> {
    let played = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| play(&mut *chooser)));
    let out = match played {
        Ok(t) => Ok(t),
        Err(payload) => match payload.downcast::<Halted>() {
            Ok(halted) => Err(*halted),
            Err(other) => std::panic::resume_unwind(other),
        },
    };
    assert!(chooser.next <= chooser.option || chooser.sample.is_none(), "the option's choice did not reseed");
    out
}

/// A segment's replay from the run as it stood at its last fight, as a
/// branch plays it.
fn play_segment(run: &Run, fought: Option<Fought>, chooser: &mut Branch) -> Stop {
    let mut run = run.clone();
    match halt(chooser, |c| run.next(fought, c)) {
        Ok(Next::Fight(_)) => Stop::Through(run.state, None),
        Ok(Next::End(end)) => Stop::Through(run.state, Some(end)),
        Err(Halted(state, at)) => Stop::Decision(state, at),
    }
}

/// Caps on an option's afterstate tree (`expand`).
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    /// Sub-decisions opened on the way, at most.
    pub depth: usize,
    /// Branches played per option and sample, at most.
    pub nodes: usize,
}

/// A leaf of an option's afterstate tree: where the run settled.
struct Settled {
    /// The sub-decisions' answers on the way, each an option token and
    /// its name.
    path: Vec<(usize, String)>,
    state: RunState,
    end: Option<forward::End>,
    /// Stopped at a sub-decision the caps did not open.
    capped: bool,
}

/// The afterstates of answering the decision a run waits at with
/// `answer` (its index for the decision), everything random reseeded from
/// `sample` once it is chosen: `prefix` are the segment's answers so far,
/// and `play` plays a branch of answers (`play_segment`). Stops at the
/// first map step, fight or end; at a sub-decision in the same room
/// (which card to smith, an event's next page, a shop after a purchase)
/// it opens every option, breadth first, while `caps` allow, since
/// otherwise a smith settles as doing nothing. Returns the leaves and the
/// caps that kept a sub-decision shut, as "kind cap".
fn expand(caps: Caps, prefix: &[usize], answer: usize, sample: u64, play: impl Fn(&mut Branch) -> Stop) -> (Vec<Settled>, BTreeSet<String>) {
    let mut queue = std::collections::VecDeque::from([(vec![answer], Vec::new())]);
    let (mut leaves, mut capped) = (Vec::new(), BTreeSet::new());
    let mut played = 0;
    while let Some((branch, path)) = queue.pop_front() {
        let answers: Vec<usize> = prefix.iter().chain(&branch).copied().collect();
        let mut chooser = Branch { answers: &answers, option: prefix.len(), next: 0, sample: Some(sample) };
        played += 1;
        match play(&mut chooser) {
            Stop::Through(state, end) => leaves.push(Settled { path, state, end, capped: false }),
            Stop::Decision(state, None) => leaves.push(Settled { path, state, end: None, capped: false }),
            Stop::Decision(state, Some(obs)) => {
                let kind = runobs::DECISIONS[obs.ids[0] as usize - 1];
                let shut = if path.len() >= caps.depth {
                    Some("depth")
                } else if played + queue.len() + obs.answers.len() > caps.nodes {
                    Some("nodes")
                } else {
                    None
                };
                match shut {
                    Some(cap) => {
                        capped.insert(format!("{kind} {cap}"));
                        leaves.push(Settled { path, state, end: None, capped: true });
                    }
                    None => {
                        for (j, (&a, name)) in obs.answers.iter().zip(obs.names).enumerate() {
                            let (mut branch, mut path) = (branch.clone(), path.clone());
                            branch.push(a);
                            path.push((j, name));
                            queue.push_back((branch, path));
                        }
                    }
                }
            }
        }
    }
    (leaves, capped)
}

/// One leaf of an option's afterstate tree (`VecEnv::afterstates`).
#[derive(Clone, Debug)]
pub struct Afterstate {
    /// Index into the envs asked about.
    pub row: usize,
    /// The option token.
    pub option: usize,
    pub sample: u64,
    /// The sub-decisions answered on the way: each an option token and
    /// its name.
    pub path: Vec<(usize, String)>,
    /// The HP fraction where the run settled.
    pub hp: f32,
    /// How the run ended, if it did in the option.
    pub end: Option<forward::End>,
    /// Settled at a sub-decision the caps did not open.
    pub capped: bool,
    /// Index into `Afterstates::fights`.
    pub fights: usize,
}

/// Every leaf of every option of the decisions asked about, with the
/// forecast fights of the states they settled at, each distinct state
/// once (none for a run that ended).
#[derive(Debug, Default)]
pub struct Afterstates {
    pub leaves: Vec<Afterstate>,
    pub fights: Vec<Vec<FightSetup>>,
    /// Each decision's options in words, by row.
    pub names: Vec<Vec<String>>,
    /// Decisions where a cap kept a sub-decision shut, by decision kind
    /// and cap ("Shop nodes").
    pub capped: BTreeMap<String, usize>,
}

/// What a job's leaves settled as, to tell a random option (its samples
/// settle differently) from one whose further samples would only repeat.
fn signature(leaves: &[Settled]) -> Vec<(Vec<usize>, String, Option<forward::End>, bool)> {
    leaves.iter().map(|l| (l.path.iter().map(|(j, _)| *j).collect(), settled_key(&l.state), l.end.clone(), l.capped)).collect()
}

/// What the forecast fights read of a settled state.
fn settled_key(state: &RunState) -> String {
    format!("{} {} {} {:?} {:?} {:?}", state.act, state.hp, state.max_hp, state.deck, state.relics, state.potions)
}

/// A node of an option's afterstate tree (`expand_tree`), the root first
/// and the rest breadth first: where a branch settled, or a sub-decision
/// it opened, whose options in token order are the nodes `children`.
#[derive(Clone, Debug)]
enum Node<L> {
    Settled(L),
    Opened { names: Vec<String>, children: std::ops::Range<usize> },
}

/// Where a branch settled, as `expand_tree` leaves it.
struct Settle {
    state: RunState,
    end: Option<forward::End>,
    /// Stopped at a sub-decision the caps did not open.
    capped: bool,
    /// What the forecast fights read of the state (`settled_key`).
    key: String,
}

/// `expand` as a tree (`Node`).
fn expand_tree(caps: Caps, prefix: &[usize], answer: usize, sample: u64, play: impl Fn(&mut Branch) -> Stop) -> (Vec<Node<Settle>>, BTreeSet<String>) {
    let settle = |state: RunState, end, capped| Node::Settled(Settle { key: settled_key(&state), state, end, capped });
    let mut queue = std::collections::VecDeque::from([vec![answer]]);
    let (mut tree, mut capped) = (Vec::new(), BTreeSet::new());
    // Nodes are played in the order they are queued, so the next child
    // queued is node `queued`.
    let mut queued = 1;
    while let Some(branch) = queue.pop_front() {
        let answers: Vec<usize> = prefix.iter().chain(&branch).copied().collect();
        let mut chooser = Branch { answers: &answers, option: prefix.len(), next: 0, sample: Some(sample) };
        let node = match play(&mut chooser) {
            Stop::Through(state, end) => settle(state, end, false),
            Stop::Decision(state, None) => settle(state, None, false),
            Stop::Decision(state, Some(obs)) => {
                let kind = runobs::DECISIONS[obs.ids[0] as usize - 1];
                let shut = if branch.len() > caps.depth {
                    Some("depth")
                } else if tree.len() + 1 + queue.len() + obs.answers.len() > caps.nodes {
                    Some("nodes")
                } else {
                    None
                };
                match shut {
                    Some(cap) => {
                        capped.insert(format!("{kind} {cap}"));
                        settle(state, None, true)
                    }
                    None => {
                        queue.extend(obs.answers.iter().map(|&a| branch.iter().copied().chain([a]).collect()));
                        let children = queued..queued + obs.answers.len();
                        queued = children.end;
                        Node::Opened { names: obs.names, children }
                    }
                }
            }
        };
        tree.push(node);
    }
    (tree, capped)
}

/// What a tree settled as, to tell a random option (its samples settle
/// differently) from one whose further samples would only repeat: each
/// node's settled state, or its option count.
fn tree_signature(tree: &[Node<Settle>]) -> Vec<(usize, Option<(&str, &Option<forward::End>, bool)>)> {
    tree.iter()
        .map(|n| match n {
            Node::Settled(s) => (0, Some((s.key.as_str(), &s.end, s.capped))),
            Node::Opened { children, .. } => (children.len(), None),
        })
        .collect()
}

/// A settled node as the score reads it.
#[derive(Clone, Copy, Debug)]
enum Leaf {
    /// The run ended in the option or has no HP left: 1 if won, else 0.
    Ended { won: bool },
    /// Alive, at `Pending`'s distinct state of this index.
    State(usize),
}

/// One option's tree under one sample.
struct Tree {
    row: usize,
    option: usize,
    sample: u64,
    nodes: Vec<Node<Leaf>>,
}

impl Tree {
    /// The value of node `i`: a sub-decision takes its best option.
    fn value(&self, i: usize, states: &[f64]) -> f64 {
        match &self.nodes[i] {
            Node::Settled(Leaf::Ended { won }) => *won as u8 as f64,
            Node::Settled(Leaf::State(s)) => states[*s],
            Node::Opened { children, .. } => children.clone().map(|c| self.value(c, states)).fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// The afterstates `VecEnv::afterstates` built last, kept for
/// `afterstate_scores` and `afterstate_option`.
struct Pending {
    /// Each decision's options in words, by row.
    names: Vec<Vec<String>>,
    /// Every option's trees, by row, option token and sample.
    trees: Vec<Tree>,
    /// Each distinct state's HP fraction.
    hp: Vec<f32>,
    /// Each distinct state's first forecast row, and the row count last.
    rows: Vec<usize>,
    /// Whether each forecast row is a boss fight.
    boss: Vec<bool>,
    /// Decisions where a cap kept a sub-decision shut, by decision kind
    /// and cap ("Shop nodes").
    capped: BTreeMap<String, usize>,
    /// Each distinct state's score, once `afterstate_scores` has run.
    scores: Option<Vec<f64>>,
}

/// What the value head reads for `VecEnv::afterstates`: every distinct
/// settled state's forecast fights at their openings
/// (`runobs::forecast_rows`), `floats [n * N_FLOATS]` and `ids [n *
/// N_IDS]`, and how many leaves and distinct states the trees hold.
pub struct AfterstateRows {
    pub floats: Vec<f32>,
    pub ids: Vec<i64>,
    pub leaves: usize,
    pub states: usize,
}

/// A decision's scores, `score [rows * MAX_OPTIONS]` (NaN where a row has
/// no such option), and the decisions where a cap kept a sub-decision
/// shut, by decision kind and cap.
pub struct AfterstateScores {
    pub score: Vec<f32>,
    pub capped: BTreeMap<String, usize>,
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
/// generator counts floors (sixteen an act). A winner's player, like the
/// env's own, goes into the fresh run `seed` names.
fn begin(starts: &Mutex<Starts>, rng: &mut Rng, seed: u64, asc: Ascension) -> (Run, Began) {
    let (began, own) = {
        let starts = starts.lock().expect("start pools");
        if let Some((at, carried)) = starts.pick_winner(rng) {
            return (Run::start_at(&run_seed(seed), asc, at, carried), Began::Winner(at));
        }
        starts.pick(rng)
    };
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
    fn reset(&mut self, index: usize, n: usize, cfg: &EnvConfig, pools: Pools) -> Option<RunFight> {
        let mut report = None;
        loop {
            let rolled = self.roll(index, n, cfg, pools);
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
                let later = self.reset(index, n, cfg, Pools { fixed: &[], hard: &[], real: &[] });
                return report.filter(|r| r.end.is_some()).or(later.filter(|r| r.end.is_some()));
            }
        }
        report.filter(|r| r.end.is_some())
    }

    fn roll(&mut self, index: usize, n: usize, cfg: &EnvConfig, Pools { fixed, hard, real }: Pools) -> Option<RunFight> {
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
        self.start(setup);
        report
    }

    fn start(&mut self, setup: FightSetup) {
        if let (Some(log), Some(run)) = (self.fights.as_mut(), self.run.as_ref()) {
            if matches!(setup.encounter.kind(), Kind::Elite | Kind::Boss) {
                let mut start = setup.run_json();
                start["act"] = (run.run.state.act as u32).into();
                log.push(start.to_string());
            }
        }
        self.setup = setup;
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
            reward: terminal_reward(c, self.base),
            run: None,
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
    /// Where run-mode runs start, shared with every slot's run.
    starts: Arc<Mutex<Starts>>,
    cfg: EnvConfig,
    fixed: Vec<FightSetup>,
    hard: Vec<(Encounter, f32)>,
    real: Vec<(Vec<FightSetup>, f32)>,
    /// The afterstates built last (`afterstates`).
    pending: Option<Pending>,
}

impl VecEnv {
    pub fn new(n: usize, seed: u64, cfg: EnvConfig) -> Self {
        let mut slots: Vec<Slot> = (0..n)
            .map(|i| {
                let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64));
                let setup = generate(&mut rng, 1, cfg.asc);
                let combat = setup.combat(0);
                Slot { base: Baseline::of(&combat), combat, setup, steps: 0, rng, resets: 0, run: None, fights: None }
            })
            .collect();
        for (i, s) in slots.iter_mut().enumerate() {
            s.reset(i, n, &cfg, Pools { fixed: &[], hard: &[], real: &[] });
        }
        Self { slots, cfg, fixed: vec![], hard: vec![], real: vec![], starts: Default::default(), pending: None }
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
            s.run = None;
            s.reset(i, n, &cfg, pools);
        }
    }

    /// Run mode: every env plays runs at `asc`, fight after fight, its run
    /// decisions made by `choices`, a fresh run from the next seed index
    /// after each ends (`RunSlot`). Starts every env's first run; under
    /// `RunChoices::Caller` each then waits at its first decision.
    /// Log each run elite and boss fight as it starts (`take_fights`), or
    /// stop and drop the log.
    pub fn log_fights(&mut self, on: bool) {
        for s in &mut self.slots {
            s.fights = on.then(Vec::new);
        }
    }

    /// The fights logged since the last call, drained.
    pub fn take_fights(&mut self) -> Vec<String> {
        self.slots.iter_mut().filter_map(|s| s.fights.as_mut()).flat_map(std::mem::take).collect()
    }

    pub fn set_runs(&mut self, asc: Ascension, base: u64, choices: RunChoices) {
        self.fixed.clear();
        let n = self.slots.len();
        let (cfg, hard, starts) = (self.cfg, &self.hard, &self.starts);
        self.slots.par_iter_mut().enumerate().for_each(|(i, s)| {
            s.resets = 0;
            s.run = Some(RunSlot::new(asc, base, i, choices, starts.clone()));
            s.reset(i, n, &cfg, Pools { fixed: &[], hard, real: &[] });
        });
    }

    /// Where the runs that start from now on start (`Starts`).
    pub fn set_starts(&mut self, full: f32, weights: [f32; START_POINTS.len()], own: [f32; START_POINTS.len()]) {
        let mut starts = self.starts.lock().expect("start pools");
        (starts.full, starts.weights, starts.own) = (full, weights, own);
    }

    /// Winners' players to start a `share` of the runs with, each at the
    /// start point it was kept at (`history::entrances`), in a fresh run;
    /// the others start as `set_starts` says. Returns how many each start
    /// point holds.
    pub fn set_winner_starts(&mut self, players: Vec<(StartPoint, Carried)>, share: f32) -> [usize; START_POINTS.len()] {
        let mut starts = self.starts.lock().expect("start pools");
        starts.winner = share;
        starts.winners = Default::default();
        for (at, carried) in players {
            starts.winners[at.index()].push(carried);
        }
        starts.winner_sizes()
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

    /// The forecast fights of the decisions `envs` wait at
    /// (`runobs::forecast_fights`, none but at map steps), in `envs` order.
    pub fn forecast(&self, envs: &[usize]) -> Vec<&[FightSetup]> {
        envs.iter()
            .map(|&env| {
                let obs = self.slots[env].run.as_ref().and_then(RunSlot::waiting).unwrap_or_else(|| panic!("env {env}: not at a run decision"));
                obs.forecast.as_slice()
            })
            .collect()
    }

    /// For each of `envs` waiting at a decision short of a map step, every
    /// option's afterstates under `samples` reseeds (`expand`), in `envs`
    /// order, then option token, then sample; an env at a map step gives
    /// none. An option whose first two samples settle the same is taken as
    /// deterministic and gets no more. Computed only when asked, so
    /// training pays nothing for it.
    pub fn afterstate_leaves(&self, envs: &[usize], samples: usize, caps: Caps) -> Afterstates {
        let path = runobs::DECISIONS.iter().position(|&d| d == "Path").expect("the map step") as i64 + 1;
        let segment = |env: usize| {
            let slot = self.slots[env].run.as_ref().expect("run mode");
            match &slot.chooser {
                Choosing::Caller(segment) if segment.waiting.is_some() => (&slot.run, segment),
                _ => panic!("env {env}: not at a run decision"),
            }
        };
        let run_jobs = |jobs: Vec<(usize, usize, u64)>| {
            jobs.into_par_iter()
                .map(|(row, option, sample)| {
                    let (run, seg) = segment(envs[row]);
                    let answer = seg.waiting.as_ref().expect("waiting").answers[option];
                    let (leaves, capped) = expand(caps, &seg.answers, answer, sample, |c| play_segment(run, seg.fought, c));
                    (row, option, sample, leaves, capped)
                })
                .collect::<Vec<_>>()
        };
        let mut out = Afterstates::default();
        let mut jobs = Vec::new();
        for (row, &env) in envs.iter().enumerate() {
            let obs = segment(env).1.waiting.as_ref().expect("waiting");
            out.names.push(obs.names.clone());
            if obs.ids[0] != path {
                jobs.extend((0..obs.answers.len()).flat_map(|option| (0..samples.min(2) as u64).map(move |sample| (row, option, sample))));
            }
        }
        let mut expanded = run_jobs(jobs);
        if samples > 2 {
            let mut first: HashMap<(usize, usize), Vec<_>> = HashMap::new();
            for (row, option, _, leaves, _) in &expanded {
                first.entry((*row, *option)).or_default().push(signature(leaves));
            }
            let mut random: Vec<(usize, usize)> = first.into_iter().filter(|(_, sigs)| sigs[0] != sigs[1]).map(|(k, _)| k).collect();
            random.sort();
            expanded.extend(run_jobs(random.into_iter().flat_map(|(row, option)| (2..samples as u64).map(move |s| (row, option, s))).collect()));
        }
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut shut: BTreeSet<(usize, String)> = BTreeSet::new();
        for (row, option, sample, leaves, capped) in expanded {
            shut.extend(capped.into_iter().map(|k| (row, k)));
            for leaf in leaves {
                let alive = leaf.end.is_none() && leaf.state.hp > 0;
                let fights = *seen.entry(settled_key(&leaf.state)).or_insert_with(|| {
                    out.fights.push(if alive { runobs::forecast_fights(&leaf.state) } else { Vec::new() });
                    out.fights.len() - 1
                });
                let hp = leaf.state.hp.max(0) as f32 / leaf.state.max_hp.max(1) as f32;
                out.leaves.push(Afterstate { row, option, sample, path: leaf.path, hp, end: leaf.end, capped: leaf.capped, fights });
            }
        }
        for (_, k) in shut {
            *out.capped.entry(k).or_default() += 1;
        }
        out
    }

    /// For each of `envs` waiting at a decision short of a map step, every
    /// option's afterstate trees under `samples` reseeds (`expand_tree`),
    /// kept until the next call: `afterstate_scores` scores them from the
    /// values of the rows returned, the distinct settled states' forecast
    /// fights. An env at a map step gets none. An option whose first two
    /// samples settle the same is taken as deterministic and gets no more.
    /// Computed only when asked, so training pays nothing for it.
    pub fn afterstates(&mut self, envs: &[usize], samples: usize, caps: Caps) -> AfterstateRows {
        let path = runobs::DECISIONS.iter().position(|&d| d == "Path").expect("the map step") as i64 + 1;
        let segment = |env: usize| {
            let slot = self.slots[env].run.as_ref().expect("run mode");
            match &slot.chooser {
                Choosing::Caller(segment) if segment.waiting.is_some() => (&slot.run, segment),
                _ => panic!("env {env}: not at a run decision"),
            }
        };
        let run_jobs = |jobs: Vec<(usize, usize, u64)>| {
            jobs.into_par_iter()
                .map(|(row, option, sample)| {
                    let (run, seg) = segment(envs[row]);
                    let answer = seg.waiting.as_ref().expect("waiting").answers[option];
                    let (tree, capped) = expand_tree(caps, &seg.answers, answer, sample, |c| play_segment(run, seg.fought, c));
                    (row, option, sample, tree, capped)
                })
                .collect::<Vec<_>>()
        };
        let mut names = Vec::with_capacity(envs.len());
        let mut jobs = Vec::new();
        for (row, &env) in envs.iter().enumerate() {
            let obs = segment(env).1.waiting.as_ref().expect("waiting");
            names.push(obs.names.clone());
            if obs.ids[0] != path {
                jobs.extend((0..obs.answers.len()).flat_map(|option| (0..samples.min(2) as u64).map(move |sample| (row, option, sample))));
            }
        }
        let mut expanded = run_jobs(jobs);
        if samples > 2 {
            let mut first: HashMap<(usize, usize), Vec<_>> = HashMap::new();
            for (row, option, _, tree, _) in &expanded {
                first.entry((*row, *option)).or_default().push(tree_signature(tree));
            }
            let mut random: Vec<(usize, usize)> = first.into_iter().filter(|(_, sigs)| sigs[0] != sigs[1]).map(|(k, _)| k).collect();
            random.sort();
            expanded.extend(run_jobs(random.into_iter().flat_map(|(row, option)| (2..samples as u64).map(move |s| (row, option, s))).collect()));
        }
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut states: Vec<RunState> = Vec::new();
        let mut shut: BTreeSet<(usize, String)> = BTreeSet::new();
        let mut leaves = 0;
        let mut trees = Vec::with_capacity(expanded.len());
        for (row, option, sample, tree, capped) in expanded {
            shut.extend(capped.into_iter().map(|k| (row, k)));
            let mut leaf = |s: Settle| {
                leaves += 1;
                if s.end.is_some() || s.state.hp <= 0 {
                    return Leaf::Ended { won: matches!(s.end, Some(forward::End::Won)) };
                }
                Leaf::State(*seen.entry(s.key).or_insert_with(|| {
                    states.push(s.state);
                    states.len() - 1
                }))
            };
            let nodes = tree
                .into_iter()
                .map(|n| match n {
                    Node::Settled(s) => Node::Settled(leaf(s)),
                    Node::Opened { names, children } => Node::Opened { names, children },
                })
                .collect();
            trees.push(Tree { row, option, sample, nodes });
        }
        trees.sort_by_key(|t| (t.row, t.option, t.sample));
        let mut capped = BTreeMap::new();
        for (_, k) in shut {
            *capped.entry(k).or_default() += 1;
        }

        let fights: Vec<Vec<FightSetup>> = states.par_iter().map(runobs::forecast_fights).collect();
        let all: Vec<FightSetup> = fights.iter().flatten().cloned().collect();
        let (floats, ids) = runobs::forecast_rows(&all);
        let mut rows = vec![0];
        rows.extend(fights.iter().scan(0, |n, f| {
            *n += f.len();
            Some(*n)
        }));
        let pending = Pending {
            names,
            trees,
            hp: states.iter().map(|s| s.hp.max(0) as f32 / s.max_hp.max(1) as f32).collect(),
            rows,
            boss: all.iter().map(|f| f.encounter.kind() == Kind::Boss).collect(),
            capped,
            scores: None,
        };
        self.pending = Some(pending);
        AfterstateRows { floats, ids, leaves, states: states.len() }
    }

    /// Scores every option of the decisions `afterstates` built last from
    /// `values`, the value head's read of its rows: a settled state's
    /// score is the calibrated win chance (`win . [V, hp, 1]` through a
    /// sigmoid, `sts2ai.forecast.Calibration`) of the act's elites, pooled,
    /// plus the boss's, each from the value `V` averaged over its
    /// openings and the state's HP fraction; a run won in the option
    /// scores 1, one that died or is stuck 0. A sub-decision takes its
    /// best option and an option the mean over its samples.
    pub fn afterstate_scores(&mut self, values: &[f32], win: [f64; 3]) -> AfterstateScores {
        let p = self.pending.as_mut().expect("afterstates to score");
        assert_eq!(values.len(), *p.rows.last().expect("rows"), "a value per forecast row");
        const R: usize = runobs::FORECAST_ROLLS;
        let scores: Vec<f64> = (0..p.hp.len())
            .map(|s| {
                let (mut sum, mut count) = ([0f64; 2], [0usize; 2]);
                for g in (p.rows[s]..p.rows[s + 1]).step_by(R) {
                    let v = values[g..g + R].iter().sum::<f32>() / R as f32;
                    let x = win[0] * v as f64 + win[1] * p.hp[s] as f64 + win[2];
                    let boss = p.boss[g] as usize;
                    sum[boss] += 1.0 / (1.0 + (-x).exp());
                    count[boss] += 1;
                }
                sum[0] / count[0].max(1) as f64 + sum[1] / count[1].max(1) as f64
            })
            .collect();
        let mut score = vec![f32::NAN; p.names.len() * runobs::MAX_OPTIONS];
        for trees in p.trees.chunk_by(|a, b| (a.row, a.option) == (b.row, b.option)) {
            let mean = trees.iter().map(|t| t.value(0, &scores)).sum::<f64>() / trees.len() as f64;
            score[trees[0].row * runobs::MAX_OPTIONS + trees[0].option] = mean as f32;
        }
        p.scores = Some(scores);
        AfterstateScores { score, capped: p.capped.clone() }
    }

    /// Option `option` of row `row` of the decisions `afterstate_scores`
    /// scored last, in words, and the sub-decisions' options on its best
    /// path under its best sample, the first on a tie.
    pub fn afterstate_option(&self, row: usize, option: usize) -> (String, Vec<String>) {
        let p = self.pending.as_ref().expect("afterstates");
        let scores = p.scores.as_deref().expect("afterstates scored");
        let first_best = |values: &mut dyn Iterator<Item = f64>| {
            values.enumerate().fold((0, f64::NEG_INFINITY), |best, (k, v)| if v > best.1 { (k, v) } else { best }).0
        };
        let trees: Vec<&Tree> = p.trees.iter().filter(|t| (t.row, t.option) == (row, option)).collect();
        let mut via = Vec::new();
        if let Some(tree) = trees.get(first_best(&mut trees.iter().map(|t| t.value(0, scores)))) {
            let mut i = 0;
            while let Node::Opened { names, children } = &tree.nodes[i] {
                let j = first_best(&mut children.clone().map(|c| tree.value(c, scores)));
                via.push(names[j].clone());
                i = children.start + j;
            }
        }
        (p.names[row][option].clone(), via)
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

    /// Where env `i`'s fight started: its rewards are shaped from here.
    pub fn base(&self, i: usize) -> Baseline {
        self.slots[i].base
    }

    /// Put `combat`, begun at `base`, in env `i`, so a search over envs
    /// (`turnsearch`) can search a fight followed elsewhere: the live
    /// pilot's. Env `i` is not meant to be stepped after.
    pub fn load(&mut self, i: usize, combat: &Combat, base: Baseline) {
        let s = &mut self.slots[i];
        s.combat.clone_from(combat);
        s.base = base;
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
                    end.run = s.reset(i, n, &cfg, pools);
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
/// Rewards are shaped from each root's fight baseline (`VecEnv::base`),
/// the one the value head's targets were shaped from. From a root, a copy
/// that ends the fight collects its terminal reward minus the root's
/// potential, and one that goes on collects its potential gain, to which
/// the value head adds the terminal it expects minus that potential: the
/// same scale. Shaped from the root instead, a finished copy scored the
/// root's potential (enemy HP taken, HP lost so far) above a going one.
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
/// last player turn it plays and the fight's baseline.
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
    pub fn new(root: &Combat, base: Baseline, n: usize, groups: usize, seed: u64) -> Self {
        Self::of(&[root], &[base], n, groups, seed, 1)
    }

    /// `n` copies of each root, root after root, each played for `depth`
    /// player turns: the rest of the current one, then `depth - 1` more.
    /// `bases[r]` is where root r's fight started.
    pub fn of(roots: &[&Combat], bases: &[Baseline], n: usize, groups: usize, seed: u64, depth: u32) -> Self {
        let seeds: Vec<u64> = (0..roots.len()).map(|r| seed ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)).collect();
        Self::with_seeds(roots, bases, &seeds, n, groups, depth)
    }

    /// `of` with each root's seed given: roots with one seed get the same
    /// dice and shuffles copy for copy, so states compared on them differ
    /// by what the states are, less by luck.
    pub fn with_seeds(roots: &[&Combat], bases: &[Baseline], seeds: &[u64], n: usize, groups: usize, depth: u32) -> Self {
        assert!(roots.len() == bases.len() && roots.len() == seeds.len(), "one baseline and seed per root");
        let per_group = n.div_ceil(groups.max(1)).max(1);
        let n_groups = n.div_ceil(per_group);
        let root_seed = |r: usize| seeds[r];
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
                    base: bases[r],
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

    /// Where copy `i`'s fight began.
    pub fn base(&self, i: usize) -> Baseline {
        self.copies[i].base
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
pub(crate) mod tests {
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
        let forks = Forks::new(&root, Baseline::of(&root), n, 4, 11);
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
        let mut forks = Forks::new(&root, Baseline::of(&root), n, groups, 11);
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

    /// A fight some turns in: the enemy at 5 HP with a Strike in hand, the
    /// player 20 HP down behind a wall of block, and where the fight began.
    pub(crate) fn fight_nearly_won() -> (Combat, Baseline) {
        use crate::card::Card;
        use crate::combat::EnemySpec;
        use crate::ids::{CardId, MonsterId};
        use crate::monster::Flags;
        let enemy = EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } };
        let start = Combat::new(&crate::ironclad_starter_deck(), 80, 80, 3, &[enemy], Ascension(10), 1);
        let base = Baseline::of(&start);
        let mut c = start;
        let e = &mut c.enemies[0].creature;
        c.stats.enemy_hp_lost += (e.hp - 5) as i64;
        (e.hp, e.block) = (5, 0);
        c.player.creature.hp -= 20;
        c.player.creature.block = 999;
        c.player.hand = vec![Card::new(900, CardId::StrikeIronclad, false)];
        c.player.energy = 3;
        (c, base)
    }

    /// What the value head learns a state is worth when the fight is sure
    /// to be won at its HP: the terminal reward less the potential so far.
    pub(crate) fn sure_win_value(c: &Combat, base: Baseline) -> f32 {
        let mut won = c.clone();
        won.outcome = Some(Outcome::Won);
        terminal_reward(&won, base) - potential(c, base)
    }

    /// Winning now and ending the turn into a fight sure to be won at the
    /// same HP score alike, the second with the value head's part added.
    #[test]
    fn winning_now_scores_like_a_sure_win_later() {
        let (root, base) = fight_nearly_won();
        assert!(potential(&root, base).abs() > 0.1, "the root must be far from the fight's start");
        let (hand, choices) = (encode::hand_order(&root), encode::choice_order(&root));
        let index = |a| encode::index_of(&root, &hand, &choices, a).unwrap() as i64;
        let strike = index(crate::combat::Action::PlayCard { hand_idx: 0, target: Some(0) });
        let end = index(crate::combat::Action::EndTurn);
        let mut forks = Forks::new(&root, base, 2, 1, 5);
        let mut rewards = [0.0f32; 2];
        forks.step(&[strike, end], &mut rewards);
        assert_eq!(forks.combat(0).outcome, Some(Outcome::Won));
        let later = forks.combat(1);
        assert!(!later.is_over() && later.player.creature.hp == root.player.creature.hp);
        let going = rewards[1] + sure_win_value(later, base);
        assert!((rewards[0] - going).abs() < 1e-5, "won now {} vs won later {going}", rewards[0]);
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
            let bases: Vec<Baseline> = roots.iter().map(|c| Baseline::of(c)).collect();
            let mut forks = Forks::of(&roots, &bases, n, groups, seed, depth);
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
        c.relics = vec![].into();
        c.potions = vec![Some(PotionId::FirePotion), None];
        c.player.creature.hp = c.player.creature.max_hp / 2;
        c.outcome = Some(Outcome::Won);
        let base = Baseline::of(&c);
        let reward = |c: &mut Combat, after| {
            c.after = after;
            terminal_reward(c, base)
        };
        let half = 0.5 * c.player.creature.hp as f32 / c.player.creature.max_hp as f32;
        assert_eq!(reward(&mut c, After::Act), 1.0 + half + POTION_VALUE);
        assert_eq!(reward(&mut c, After::Boss), 1.0 + half + POTION_VALUE, "the second boss follows with no rest");
        assert_eq!(reward(&mut c, After::Ancient), 1.0 + 0.2 * half + POTION_VALUE, "the Ancient heals 80% at A10");
        assert_eq!(reward(&mut c, After::End), 1.0, "nothing is left to spend");
    }

    /// Max HP lost (Paper Cuts) costs reward; counted against the current
    /// max, the HP fraction rose and paid for it.
    #[test]
    fn lost_max_hp_costs_reward() {
        let mut c = generate(&mut Rng::new(4), 8, Ascension(10)).combat(3);
        c.relics = vec![].into();
        c.potions = vec![None, None];
        c.player.creature.hp = c.player.creature.max_hp / 2;
        c.after = After::Act;
        c.outcome = Some(Outcome::Won);
        let base = Baseline::of(&c);
        let kept = terminal_reward(&c, base);
        c.player.creature.max_hp -= 4;
        assert!(terminal_reward(&c, base) < kept, "4 max HP lost with HP unchanged");
        c.outcome = None;
        assert!(potential(&c, base) < 0.0, "and costs it at the step it happens");
    }

    #[test]
    fn refilled_potions_are_free() {
        use crate::relic::Relic;
        let mut c = generate(&mut Rng::new(4), 8, Ascension(10)).combat(3);
        c.relics = vec![].into();
        c.potions = vec![Some(PotionId::PotionShapedRock), Some(PotionId::FirePotion)];
        assert_eq!(potions_held(&c), 2);
        c.relics.push(Relic::new(RelicId::PetrifiedToad));
        assert_eq!(potions_held(&c), 1, "the Toad's rock comes back");
        c.relics.push(Relic::new(RelicId::Sozu));
        assert_eq!(potions_held(&c), 2, "Sozu stops the Toad");
        c.relics = vec![Relic::new(RelicId::DelicateFrond)].into();
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

    /// A run played under the caller's choices: where it began, its
    /// answers, each fight's setup and the combat it ended as, and its end.
    struct CallerRun {
        seed: u64,
        began: Began,
        answers: Vec<usize>,
        fights: Vec<(FightSetup, Combat)>,
        end: forward::End,
    }

    /// Plays `env` (run mode, `RunChoices::Caller`) until `runs` runs have
    /// ended: random options answered through `step_run` until none wait,
    /// then a combat step with the first legal action. Checks each row is
    /// the decision its env waits at, with an option token per answer.
    fn caller_runs(env: &mut VecEnv, runs: usize) -> Vec<CallerRun> {
        let n = env.len();
        let (mut floats, mut ids, mut mask) = (vec![0.0; n * N_FLOATS], vec![0; n * N_IDS], vec![false; n * N_ACTIONS]);
        let (mut rewards, mut dones) = (vec![0.0; n], vec![false; n]);
        let (mut run_f, mut run_i) = (vec![0.0; n * RUN_FLOATS], vec![0; n * RUN_IDS]);
        env.observe(&mut floats, &mut ids, &mut mask);
        let mut rng = Rng::new(3);
        let mut answers: std::collections::HashMap<u64, Vec<usize>> = Default::default();
        let mut fights: std::collections::HashMap<u64, Vec<(FightSetup, Combat)>> = Default::default();
        let mut ended: Vec<RunFight> = vec![];
        while ended.len() < runs {
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
                ended.extend(env.step_run(&waiting, &options, &mut floats, &mut ids, &mut mask).into_iter().map(|(_, run)| run));
            }
            let actions: Vec<i64> = (0..n).map(|i| mask[i * N_ACTIONS..][..N_ACTIONS].iter().position(|&m| m).unwrap() as i64).collect();
            let before: Vec<(FightSetup, Combat)> = env.slots.iter().map(|s| (s.setup.clone(), s.combat.clone())).collect();
            for e in env.step(&actions, &mut floats, &mut ids, &mut mask, &mut rewards, &mut dones) {
                let (setup, mut combat) = before[e.env].clone();
                combat.step(encode::decode(&combat, actions[e.env] as usize).unwrap());
                let run = e.run.expect("a run fight");
                fights.entry(run.seed).or_default().push((setup, combat));
                if run.end.is_some() {
                    ended.push(run);
                }
            }
        }
        ended
            .into_iter()
            .map(|run| CallerRun {
                seed: run.seed,
                began: run.began,
                answers: answers.remove(&run.seed).unwrap_or_default(),
                fights: fights.remove(&run.seed).unwrap_or_default(),
                end: run.end.expect("an ended run"),
            })
            .collect()
    }

    /// Plays `start` on as `forward::play_on` does with the answers and
    /// fights `run` had in the env: the same setup at every fight, the same
    /// end, every answer used.
    fn replays_forward(start: Run, run: CallerRun) {
        let seed = run.seed;
        let mut recorded = run.fights.into_iter();
        let mut chooser = Scripted(run.answers.into_iter());
        let forward = forward::play_on(start, &mut chooser, &mut |state: &mut RunState, setup: FightSetup| {
            let (fought, combat) = recorded.next().expect("forward played more fights");
            assert_eq!(format!("{setup:?}"), format!("{fought:?}"), "seed {seed}: fight on floor {}", state.floor);
            Fought { won: combat.outcome == Some(Outcome::Won), gold_proportion: state.end_fight(&setup, &combat) }
        });
        assert_eq!(forward.end, run.end, "seed {seed}");
        assert!(recorded.next().is_none(), "seed {seed}: fights left over");
        assert!(chooser.0.next().is_none(), "seed {seed}: answers left over");
    }

    /// Runs under the caller's choices play as `forward::play` does with
    /// the same choices and fights.
    #[test]
    fn caller_choices_play_the_forward_run() {
        let mut env = VecEnv::new(4, 5, EnvConfig::default());
        env.set_runs(Ascension(10), 300, RunChoices::Caller);
        let runs = caller_runs(&mut env, 8);
        assert!(runs.iter().map(|r| r.answers.len()).sum::<usize>() > 50, "few decisions");
        for run in runs {
            replays_forward(Run::new(&run_seed(run.seed), Ascension(10)), run);
        }
    }

    /// Runs started with a winner's player at an act's entrance play as
    /// a fresh run of their own seed picked up there with that player
    /// (`Run::start_at`) does, given the same choices and fights. Every
    /// run starts so with the share at 1, the envs' first ones among them.
    #[test]
    fn winner_starts_play_the_forward_run() {
        let text = include_str!("../testdata/run-TBL5VNYN4M.run");
        let players = crate::history::entrances(&serde_json::from_str(text).unwrap());
        let mut env = VecEnv::new(4, 5, EnvConfig::default());
        assert_eq!(env.set_winner_starts(players.clone(), 1.0), [0, 1, 0, 1, 0]);
        env.set_runs(Ascension(10), 300, RunChoices::Caller);
        let runs = caller_runs(&mut env, 12);
        let mut began = std::collections::BTreeSet::new();
        for run in runs {
            let Began::Winner(at) = run.began else { panic!("seed {}: began {:?}", run.seed, run.began) };
            began.insert(at.act());
            let carried = players.iter().find(|(p, _)| *p == at).expect("a kept player").1.clone();
            replays_forward(Run::start_at(&run_seed(run.seed), Ascension(10), at, carried), run);
        }
        assert_eq!(began, [1, 2].into(), "runs start at both entrances");
    }

    /// An afterstate plays what could follow an option, never what the
    /// run's own draws hold. Two runs alike but for their streams and the
    /// grab bags' order (the same relics in them) deal different relics to
    /// Dig at a rest site (the Rewards stream and the bag) and to Trash
    /// Heap's Dive In (the event's own stream) on their own draws; under a
    /// reseed their afterstates are one player, forecast the same, and
    /// move with the sample. Smith opens the deck pick under it, and every
    /// option of a waiting env gets its leaves, a map step none.
    #[test]
    fn the_afterstate_reads_no_hidden_information() {
        use crate::game_rng::{GameRng, RunRngs};
        use std::collections::HashSet;
        let caps = Caps { depth: 3, nodes: 256 };
        let mut carried = Run::new("SEEDA", Ascension(10)).state.carried();
        carried.relics.push(crate::run::RunRelic::new("SHOVEL"));
        (carried.hp, carried.max_hp) = (40, 80);
        let a = Run::start_at("SEEDA", Ascension(10), StartPoint::BossDoor(0), carried.clone());
        let mut b = a.clone();
        b.state.rngs = RunRngs::new("SEEDB");
        let mut rng = GameRng::named(7, "another order");
        b.state.plan.player_bag.reshuffle(&mut rng);
        b.state.plan.shared_bag.reshuffle(&mut rng);
        assert_ne!(a.state.rngs.seed, b.state.rngs.seed);
        let contents = |run: &Run| {
            let mut bag = run.state.plan.player_bag.deques.clone();
            bag.iter_mut().for_each(|(_, d)| d.sort_by_cached_key(|r| r.game_id()));
            bag
        };
        assert_eq!(contents(&a), contents(&b), "the same relics in the bags");
        assert_ne!(a.state.plan.player_bag, b.state.plan.player_bag, "in another order");

        let Stop::Decision(_, Some(obs)) = play_segment(&a, None, &mut Branch { answers: &[], option: 0, next: 0, sample: None }) else {
            panic!("a rest site under the boss")
        };
        let names = obs.names.clone();
        assert_eq!(names, ["Heal", "Smith", "Dig"]);
        let own = |run: &Run, answer: usize| match play_segment(run, None, &mut Branch { answers: &[answer], option: 1, next: 0, sample: None }) {
            Stop::Decision(state, None) => state.carried(),
            _ => panic!("the map step to the boss follows the rest site"),
        };
        let dig = obs.answers[2];
        assert_ne!(own(&a, dig), own(&b, dig), "the runs' own draws deal Dig different relics, or the test means nothing");
        let after = |run: &Run, answer: usize, sample: u64| {
            let (leaves, capped) = expand(caps, &[], answer, sample, |c| play_segment(run, None, c));
            assert!(capped.is_empty(), "{capped:?}");
            leaves
                .into_iter()
                .map(|l| (l.path, l.state.carried(), l.end, l.capped, runobs::forecast_rows(&runobs::forecast_fights(&l.state))))
                .collect::<Vec<_>>()
        };
        for (j, name) in names.iter().enumerate() {
            let (la, lb) = (after(&a, obs.answers[j], 3), after(&b, obs.answers[j], 3));
            assert!(la == lb, "{name}: the afterstates differ");
            assert!(la.iter().all(|l| l.2.is_none() && !l.3));
            match name.as_str() {
                "Heal" => assert!(la.len() == 1 && la[0].0.is_empty() && la[0].1.hp > 40, "heal settles at the map step"),
                "Smith" => {
                    let upgraded = |c: &Carried| c.deck.iter().filter(|c| c.upgraded).count();
                    assert!(la.len() > 1, "smith opens the deck pick");
                    assert!(la.iter().all(|l| l.0.len() == 1 && upgraded(&l.1) == upgraded(&carried) + 1), "each leaf upgraded its card");
                }
                _ => {
                    assert!(la.len() == 1 && la[0].1.relics.len() == carried.relics.len() + 1);
                    let dealt: HashSet<String> = (3..9).map(|s| after(&a, dig, s)[0].1.relics.last().expect("a relic").id.clone()).collect();
                    assert!(dealt.len() > 1, "the sample deals the relic: {dealt:?}");
                }
            }
        }

        let dealt = |state: &RunState, answer: usize, sample: Option<u64>| {
            let mut run = state.clone();
            let mut chooser = Branch { answers: &[answer], option: sample.is_none() as usize, next: 0, sample };
            assert!(halt(&mut chooser, |c| run.event("TrashHeap", c, &mut Vec::new())).is_ok(), "the page settles the event");
            run.carried()
        };
        for answer in [0, 1] {
            assert_ne!(dealt(&a.state, answer, None), dealt(&b.state, answer, None), "Trash Heap {answer}: the runs' own streams deal differently, or the test means nothing");
            assert_eq!(dealt(&a.state, answer, Some(3)), dealt(&b.state, answer, Some(3)), "Trash Heap {answer}");
            let samples: HashSet<String> = (3..9).map(|s| format!("{:?}", dealt(&a.state, answer, Some(s)))).collect();
            assert!(samples.len() > 1, "Trash Heap {answer}: the sample deals it");
        }

        let n = 8;
        let mut env = VecEnv::new(n, 5, EnvConfig::default());
        env.set_runs(Ascension(10), 300, RunChoices::Caller);
        let (mut floats, mut ids, mut mask) = (vec![0.0; n * N_FLOATS], vec![0; n * N_IDS], vec![false; n * N_ACTIONS]);
        env.observe(&mut floats, &mut ids, &mut mask);
        let waiting = env.run_waiting();
        let options: Vec<usize> = waiting.iter().map(|&i| env.slots[i].run.as_ref().and_then(RunSlot::waiting).expect("waiting").answers.len()).collect();
        assert!(waiting.len() == n && options.iter().all(|&o| o > 1), "every run waits at its first ancient");
        let rows = env.afterstates(&waiting, 2, caps);
        let p = env.pending.as_ref().expect("afterstates");
        let covered: HashSet<(usize, usize, u64)> = p.trees.iter().map(|t| (t.row, t.option, t.sample)).collect();
        assert_eq!(covered.len(), options.iter().sum::<usize>() * 2, "every option, every sample");
        let live = |n: &Node<Leaf>| match n {
            Node::Settled(Leaf::State(s)) => Some(*s),
            _ => None,
        };
        assert!(p.trees.iter().flat_map(|t| t.nodes.iter().filter_map(live)).all(|s| p.rows[s + 1] > p.rows[s]), "a live state has its forecast fights");
        assert_eq!(rows.floats.len(), p.rows.last().expect("rows") * N_FLOATS);
        let zeros = vec![0; n];
        env.step_run(&waiting, &zeros.iter().map(|&z| z as i64).collect::<Vec<_>>(), &mut floats, &mut ids, &mut mask);
        let waiting = env.run_waiting();
        let at_path = |i: &usize| env.slots[*i].run.as_ref().and_then(RunSlot::waiting).is_some_and(|o| o.ids[0] == 1);
        let path: Vec<usize> = waiting.iter().copied().filter(at_path).collect();
        assert!(!path.is_empty(), "a map step follows the ancient's relic");
        assert_eq!(env.afterstates(&path, 2, caps).leaves, 0, "a map step is the policy's");
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
