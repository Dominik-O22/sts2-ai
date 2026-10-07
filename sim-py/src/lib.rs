//! Python bindings for the training environment (DESIGN.md, Training).
//! Two classes: `VecEnv` for training, `Advisor` for following a real
//! combat from the recorder's log. Plus the layout constants the model needs. Buffers
//! are numpy arrays the caller allocates once; `observe` and `step` fill
//! them in place with the GIL released.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use sim::encode::*;
use sim::env::{Baseline, EnvConfig, Forks as InnerForks, RunChoices, VecEnv as Inner};
use sim::runobs;
use sim::gen::{holdout, load_recordings, ACTS, LAST_FLOOR};
use sim::ids::{ALL_CARDS, ALL_MONSTERS};
use sim::replay::{command, Ids, Replayer, Step};
use sim::types::Ascension;

/// Forks clone combats with every `Vec` at capacity, so their first moves
/// reallocate, from every rayon thread at once; glibc's malloc spent a
/// fifth of the search's CPU on that, mostly waiting on its locks.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A batch of combats. See `sim::env::VecEnv`.
#[pyclass]
struct VecEnv {
    inner: Inner,
}

/// One finished fight: (env, won, hp_frac, hp_lost, potions_used, steps, floor, encounter, kind, reward, restart, run).
type End = (usize, bool, f32, f32, u32, u32, u32, String, String, f32, bool, Option<RunFight>);

/// A run fight's place in its run: (seed index, act, floor, deck size, how
/// the run ended with it: "won", "died", "stuck: <why>", or None; where the
/// run started: its place in `start_points()`, None for floor 1; with what
/// player there: "gen" generated, "own" the envs' own state, "win" a
/// winner's run, "" for floor 1).
type RunFight = (u64, u32, u32, u32, Option<String>, Option<usize>, &'static str);

fn end_text(end: sim::forward::End) -> String {
    use sim::forward::End;
    match end {
        End::Won => "won".into(),
        End::Died => "died".into(),
        End::Stuck(why) => format!("stuck: {why}"),
    }
}

fn run_fight(r: sim::env::RunFight) -> RunFight {
    use sim::env::Began;
    let end = r.end.map(end_text);
    let (start, source) = match r.began {
        Began::Floor1 => (None, ""),
        Began::Generated(at) => (Some(at.index()), "gen"),
        Began::Own(at) => (Some(at.index()), "own"),
        Began::Winner(at) => (Some(at.index()), "win"),
    };
    (r.seed, r.act, r.floor, r.deck, end, start, source)
}

impl VecEnv {
    fn inner_asc(&self) -> Ascension {
        self.inner.asc()
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed=0, asc=10, min_floor=1, max_floor=LAST_FLOOR, max_steps=500, hard_frac=0.0))]
    fn new(n: usize, seed: u64, asc: u8, min_floor: u32, max_floor: u32, max_steps: u32, hard_frac: f32) -> Self {
        let cfg = EnvConfig { asc: Ascension(asc), min_floor, max_floor, max_steps, hard_frac, restart_frac: 0.0 };
        Self { inner: Inner::new(n, seed, cfg) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Curriculum: fraction of resets forced onto an elite or boss.
    fn set_hard_frac(&mut self, frac: f32) {
        self.inner.set_hard_frac(frac);
    }

    /// Fraction of resets that resume a kept elite or boss turn start.
    fn set_restart_frac(&mut self, frac: f32) {
        self.inner.set_restart_frac(frac);
    }

    /// Weight of the next act's elites and bosses in afterstate scores.
    fn set_lookahead(&mut self, weight: f32) {
        self.inner.set_lookahead(weight);
    }

    /// Curriculum: floors generated fights are drawn from.
    fn set_floors(&mut self, min: u32, max: u32) {
        self.inner.set_floors(min, max);
    }

    /// Weights by encounter name (`ElitesName` as `End.encounter` reports
    /// it) for the elites and bosses `hard_frac` forces; empty resets to
    /// drawing them evenly.
    fn set_hard_weights(&mut self, weights: Vec<(String, f32)>) -> PyResult<()> {
        let by_name: std::collections::HashMap<String, sim::encounter::Encounter> =
            sim::encounter::ALL.iter().map(|&e| (format!("{e:?}"), e)).collect();
        let w = weights
            .into_iter()
            .map(|(name, w)| {
                by_name.get(&name).map(|&e| (e, w)).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown encounter {name}")))
            })
            .collect::<PyResult<_>>()?;
        self.inner.set_hard_weights(w);
        Ok(())
    }

    /// Put the advisor's current combat in env `i`, for `TurnPlanner` (and
    /// `exactsearch.Hybrid`) to search the fight the live pilot follows.
    fn load_combat(&mut self, i: usize, advisor: PyRef<'_, Advisor>) -> PyResult<()> {
        let base = advisor.base.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no combat yet"))?;
        self.inner.load(i, advisor.combat()?, base);
        Ok(())
    }

    /// Copies of the current fight in each of `envs`, `n` per env, for a
    /// turn search over many fights in one batch.
    /// `depth` player turns are played: the rest of this one, then more.
    #[pyo3(signature = (envs, n, groups=4, seed=0, depth=1))]
    fn fork(&self, envs: Vec<usize>, n: usize, groups: usize, seed: u64, depth: u32) -> Forks {
        let guards: Vec<_> = envs.iter().map(|&i| self.inner.combat(i)).collect();
        let roots: Vec<&sim::combat::Combat> = guards.iter().map(|g| &**g).collect();
        let bases: Vec<_> = envs.iter().map(|&i| self.inner.base(i)).collect();
        Forks { inner: InnerForks::of(&roots, &bases, n, groups, seed, depth) }
    }

    /// Solve each of `envs`' fights from its current state with its real
    /// future known (`sim::solve`, a measuring instrument: it reads the
    /// dice), in parallel. Per env: (a win found and replayed, turns
    /// searched, states, whether a turn's enumeration was capped).
    #[pyo3(signature = (envs, beam=300, turn_states=3000, max_turns=40))]
    fn solve(&self, py: Python<'_>, envs: Vec<usize>, beam: usize, turn_states: usize, max_turns: u32) -> Vec<(bool, u32, u64, bool)> {
        use rayon::prelude::*;
        let cfg = sim::solve::Config { beam, turn_states, max_turns };
        let guards: Vec<_> = envs.iter().map(|&i| self.inner.combat(i)).collect();
        let roots: Vec<&sim::combat::Combat> = guards.iter().map(|g| &**g).collect();
        py.detach(|| {
            roots
                .par_iter()
                .map(|&root| {
                    let s = sim::solve::solve(root, &cfg);
                    let won = s.line.as_ref().is_some_and(|l| sim::solve::replays(root, l));
                    (won, s.turns, s.states, s.capped)
                })
                .collect()
        })
    }

    /// `solve`'s winning line from each of `envs`' current state, replayed
    /// on its own copy: per step the observation (floats, ids) and the
    /// action's index, or `None` when no win was found. Clairvoyant: a
    /// measuring instrument, never a policy.
    #[pyo3(signature = (envs, beam=300, turn_states=3000, max_turns=40))]
    #[allow(clippy::type_complexity)]
    fn solve_trace(&self, py: Python<'_>, envs: Vec<usize>, beam: usize, turn_states: usize, max_turns: u32) -> Vec<Option<Vec<(Vec<f32>, Vec<i64>, usize)>>> {
        use rayon::prelude::*;
        let cfg = sim::solve::Config { beam, turn_states, max_turns };
        let guards: Vec<_> = envs.iter().map(|&i| self.inner.combat(i)).collect();
        let roots: Vec<&sim::combat::Combat> = guards.iter().map(|g| &**g).collect();
        py.detach(|| {
            roots
                .par_iter()
                .map(|&root| {
                    let line = sim::solve::solve(root, &cfg).line?;
                    let mut c = root.clone();
                    let (mut floats, mut ids, mut mask) = (vec![0.0; N_FLOATS], vec![0; N_IDS], vec![false; N_ACTIONS]);
                    let mut out = Vec::with_capacity(line.len());
                    for a in line {
                        encode(&c, &mut floats, &mut ids, &mut mask);
                        out.push((floats.clone(), ids.clone(), index_of(&c, &hand_order(&c), &choice_order(&c), a)?));
                        c.step(a);
                    }
                    Some(out)
                })
                .collect()
        })
    }

    /// (encounter, kind) of env `i`'s current fight.
    fn fight(&self, i: usize) -> (String, String) {
        let s = self.inner.setup(i);
        (format!("{:?}", s.encounter), format!("{:?}", s.encounter.kind()))
    }

    /// Switch to cycling through a fixed generated set: `per_encounter`
    /// fights against every encounter of the first `acts` acts, always the
    /// same for a seed.
    #[pyo3(signature = (seed=0, per_encounter=10, acts=ACTS))]
    fn use_holdout(&mut self, seed: u64, per_encounter: usize, acts: u32) -> usize {
        let setups = holdout(seed, per_encounter, self.inner_asc(), acts);
        let n = setups.len();
        self.inner.set_fixed(setups);
        n
    }

    /// Pools of played runs' fights (`sim::gen::run_setups`: one JSON
    /// object a line), each with the share of the resets it takes from here
    /// on, each fight drawn with its enemies rolled afresh. Returns each
    /// pool's size.
    #[pyo3(signature = (pools, seed=0))]
    fn use_real(&mut self, pools: Vec<(String, f32)>, seed: u64) -> PyResult<Vec<usize>> {
        let pools = pools
            .into_iter()
            .map(|(text, share)| Ok((sim::gen::run_setups(&text, &Ids::new(), seed).map_err(pyo3::exceptions::PyValueError::new_err)?, share)))
            .collect::<PyResult<Vec<_>>>()?;
        let sizes = pools.iter().map(|(p, _)| p.len()).collect();
        self.inner.set_real(pools);
        Ok(sizes)
    }

    /// Switch to cycling through fights of played runs (as `use_real`
    /// reads them), `repeats` each with enemies rolled from `seed`: the
    /// same file and seed give the same fights. Returns the number of fights.
    #[pyo3(signature = (setups, repeats, seed=0))]
    fn use_setups(&mut self, setups: &str, repeats: usize, seed: u64) -> PyResult<usize> {
        let runs = sim::gen::run_setups(setups, &Ids::new(), seed).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut rng = sim::rng::Rng::new(seed);
        let fights: Vec<_> = runs.iter().flat_map(|r| (0..repeats).map(|_| r.rerolled(&mut rng)).collect::<Vec<_>>()).collect();
        let n = fights.len();
        self.inner.set_fixed(fights);
        Ok(n)
    }

    /// Queue many runs' fights for `start_fights`, replacing the queue:
    /// each job (run JSON, max HP, encounters as (game name, floor,
    /// starting HP), seed) gets `repeats` fights per encounter, built by
    /// `sim::gen::run_fights` one encounter at a time. A fight's seed comes
    /// from its job's seed and its place in the job, so jobs with the same
    /// seed and encounters play the same shuffles. Returns the queue index
    /// each job starts at.
    fn queue_fight_jobs(&mut self, jobs: Vec<(String, i32, Vec<(String, u32, i32)>, u64)>, repeats: usize) -> PyResult<Vec<usize>> {
        let ids = Ids::new();
        let (mut fights, mut starts) = (vec![], vec![]);
        for (start, max_hp, encounters, seed) in jobs {
            let v: Value = serde_json::from_str(&start).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad run json: {e}")))?;
            starts.push(fights.len());
            let mut place = 0u64;
            for (name, floor, hp) in encounters {
                for setup in sim::gen::run_fights(&v, &ids, hp, max_hp, &[(name, floor)], repeats, seed).map_err(pyo3::exceptions::PyValueError::new_err)? {
                    fights.push((setup, seed.wrapping_mul(0x9E37_79B9).wrapping_add(place)));
                    place += 1;
                }
            }
        }
        self.inner.queue_fights(fights);
        Ok(starts)
    }

    /// Start queued fight `fights[i]` in env `envs[i]` and encode those
    /// envs' rows (`sim::env::VecEnv::start_fights`); returns the fights
    /// already over as they start, as `step` returns ended ones.
    fn start_fights(
        &mut self,
        py: Python<'_>,
        envs: Vec<usize>,
        fights: Vec<usize>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<Vec<End>> {
        let starts: Vec<(usize, usize)> = envs.into_iter().zip(fights).collect();
        let (f, i, m) = (floats.as_slice_mut()?, ids.as_slice_mut()?, mask.as_slice_mut()?);
        let ended = py.detach(|| self.inner.start_fights(&starts, f, i, m));
        Ok(ended.into_iter().map(episode_end).collect())
    }

    /// Switch to cycling through fights of the run in `start` (JSON with a
    /// recorder `start` record's run fields: deck, relics, potions, gold,
    /// max_energy, ascension) at `hp` of `max_hp`: `repeats` fights against
    /// each of `encounters` (game name, floor), in that order
    /// (`sim::gen::run_fights`: the same encounters and seed give the same
    /// enemies). Returns the number of fights.
    #[pyo3(signature = (start, hp, max_hp, encounters, repeats, seed=0))]
    fn use_run(&mut self, start: &str, hp: i32, max_hp: i32, encounters: Vec<(String, u32)>, repeats: usize, seed: u64) -> PyResult<usize> {
        let v: Value = serde_json::from_str(start).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad run json: {e}")))?;
        let setups = sim::gen::run_fights(&v, &Ids::new(), hp, max_hp, &encounters, repeats, seed).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let n = setups.len();
        self.inner.set_fixed(setups);
        Ok(n)
    }

    /// Switch to run mode: every env plays whole runs at `asc`, fight
    /// after fight; the envs' k-th runs are seed indices `seed + env + k *
    /// n` (`sim::env::VecEnv::set_runs`). `choices` makes the run
    /// decisions: "random", "first", or "caller" (`run_waiting`,
    /// `observe_run`, `step_run`).
    /// Log each run elite and boss fight as it starts, as a recorder
    /// `start` record with the run's act (`take_fights` drains them).
    #[pyo3(signature = (on, easy=false))]
    fn log_fights(&mut self, on: bool, easy: bool) {
        self.inner.log_fights(on, easy);
    }

    fn take_fights(&mut self) -> Vec<String> {
        self.inner.take_fights()
    }

    /// Run mode over seed indices `seed..last` (`sim::env::Seeds`; None:
    /// unbounded). With `static_seeds` env `i` plays `seed + i + k * n`.
    #[pyo3(signature = (seed=0, asc=10, choices="random", last=None, static_seeds=false))]
    fn use_runs(&mut self, py: Python<'_>, seed: u64, asc: u8, choices: &str, last: Option<u64>, static_seeds: bool) -> PyResult<()> {
        let choices = match choices {
            "random" => RunChoices::Random,
            "first" => RunChoices::First,
            "caller" => RunChoices::Caller,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown run choices {other:?}: random, first or caller"))),
        };
        py.detach(|| self.inner.set_runs(Ascension(asc), seed, last.unwrap_or(u64::MAX), choices, static_seeds));
        Ok(())
    }

    /// Trace each run (`sim::runtrace`): the lines come out of `take`.
    fn trace_runs(&mut self, on: bool) {
        self.inner.trace_runs(on);
    }

    /// Start the threaded run loop (`sim::runloop::Loop`) with `workers`
    /// threads; `take` and `post` drive it, `stop_loop` ends it.
    #[pyo3(signature = (workers=0))]
    fn start_loop(&mut self, workers: usize) {
        let workers = if workers == 0 { std::thread::available_parallelism().map_or(8, |n| n.get()) } else { workers };
        self.inner.start_loop(workers);
    }

    fn stop_loop(&mut self) {
        self.inner.stop_loop();
    }

    /// The envs ready for the network, their rows packed at the front of
    /// the buffers (`sim::runloop::Loop::take`): waits for `min_rows` of
    /// them, or until none is posted, or `timeout_ms`.
    #[allow(clippy::too_many_arguments)]
    fn take(
        &self,
        py: Python<'_>,
        min_rows: usize,
        timeout_ms: u64,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
        mut run_floats: PyReadwriteArray2<f32>,
        mut run_ids: PyReadwriteArray2<i64>,
    ) -> PyResult<Taken> {
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        let m = mask.as_slice_mut()?;
        let rf = run_floats.as_slice_mut()?;
        let ri = run_ids.as_slice_mut()?;
        let taken = py.detach(|| self.inner.run_loop().take(min_rows, std::time::Duration::from_millis(timeout_ms), f, i, m, rf, ri));
        let sim::env::Events { ends, runs, starts, traces } = taken.events;
        Ok(Taken {
            combat: taken.combat,
            decision: taken.decision,
            ends: ends.into_iter().map(episode_end).collect(),
            runs: runs.into_iter().map(run_fight).collect(),
            starts,
            traces,
            active: taken.active,
        })
    }

    /// Answer the envs taken: `actions[k]` for `combat[k]`, `options[k]`
    /// for `decision[k]`.
    fn post(&self, py: Python<'_>, combat: Vec<usize>, actions: PyReadonlyArray1<i64>, decision: Vec<usize>, options: PyReadonlyArray1<i64>) -> PyResult<()> {
        let a = actions.as_slice()?;
        let o = options.as_slice()?;
        py.detach(|| self.inner.run_loop().post(&combat, a, &decision, o));
        Ok(())
    }

    /// Combat steps taken and run decisions answered by the loop so far.
    fn loop_counts(&self) -> (u64, u64) {
        self.inner.run_loop().counts()
    }

    /// Where the runs that start from now on start: floor 1 with chance
    /// `full`, else a start point by `weights`, from the envs' own state
    /// there with chance `own` when they have one (`sim::env::Starts`;
    /// both lists in `start_points()` order).
    fn set_starts(&mut self, full: f32, weights: Vec<f32>, own: Vec<f32>) -> PyResult<()> {
        let per_point = |v: Vec<f32>| v.try_into().map_err(|_| pyo3::exceptions::PyValueError::new_err("a value per start point"));
        self.inner.set_starts(full, per_point(weights)?, per_point(own)?);
        Ok(())
    }

    /// Start a `share` of the runs that start from now on with winners'
    /// players, each in a fresh run: the player of each history file of
    /// `runs` at the entrances of acts 2 and 3, where its rooms so far
    /// match the record (`sim::history::entrances`). Returns how many
    /// players each start point holds, in `start_points()` order.
    fn use_winner_starts(&mut self, py: Python<'_>, runs: Vec<String>, share: f32) -> PyResult<Vec<usize>> {
        let runs = py.detach(|| sim::history::winner_starts(&runs)).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(self.inner.set_winner_starts(runs, share).to_vec())
    }

    /// States the envs' runs have kept per start point.
    fn start_pools(&self) -> Vec<usize> {
        self.inner.start_pools().to_vec()
    }

    /// The envs whose run waits at a decision for the caller.
    fn run_waiting(&self) -> Vec<usize> {
        self.inner.run_waiting()
    }

    /// Fill rows `0..len(envs)` of `floats [k, RUN_FLOATS]` and `ids [k,
    /// RUN_IDS]` with the decision each env waits at (`sim::runobs`).
    fn observe_run(&self, py: Python<'_>, envs: Vec<usize>, mut floats: PyReadwriteArray2<f32>, mut ids: PyReadwriteArray2<i64>) -> PyResult<()> {
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        py.detach(|| self.inner.observe_run(&envs, f, i));
        Ok(())
    }

    /// The forecast fights of the decisions `envs` wait at (`forecast_rows`).
    fn forecast<'py>(&self, py: Python<'py>, envs: Vec<usize>) -> Forecast<'py> {
        let fights = self.inner.forecast(&envs);
        let fights: Vec<&[sim::gen::FightSetup]> = fights.iter().map(Vec::as_slice).collect();
        forecast_rows(py, &fights)
    }

    /// Builds the afterstates of every option of the decisions `envs` wait
    /// at (`sim::env::VecEnv::afterstates`, none at a map step) and returns
    /// what the value head reads for them: the distinct settled states'
    /// forecast fights as combat rows `floats [n * N_FLOATS]` and `ids [n *
    /// N_IDS]`, and how many leaves and distinct states the trees hold.
    /// With `rows` off the rows come back empty (`afterstate_runs` scoring).
    #[pyo3(signature = (envs, samples, depth, nodes, rows=true))]
    fn afterstates<'py>(&mut self, py: Python<'py>, envs: Vec<usize>, samples: usize, depth: usize, nodes: usize, rows: bool) -> AfterstateRows<'py> {
        let caps = sim::env::Caps { depth, nodes };
        let rows = py.detach(|| self.inner.afterstates(&envs, samples, caps, rows));
        (rows.floats.into_pyarray(py), rows.ids.into_pyarray(py), rows.leaves, rows.states)
    }

    /// Scores the afterstates built last from the value head's `values` of
    /// their rows and the calibration's win coefficients
    /// (`sim::env::VecEnv::afterstate_scores`): `score [rows *
    /// MAX_OPTIONS]`, NaN where a row has no such option, and the decisions
    /// where a cap kept a sub-decision shut, by decision kind and cap.
    fn afterstate_scores<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<f32>,
        win: (f64, f64, f64),
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, std::collections::BTreeMap<String, usize>)> {
        let v = values.as_slice()?;
        let scored = py.detach(|| self.inner.afterstate_scores(v, [win.0, win.1, win.2]));
        Ok((scored.score.into_pyarray(py), scored.capped))
    }

    /// The afterstates' distinct settled states as run records
    /// (`sim::env::VecEnv::afterstate_runs`), for a scorer that reads runs.
    fn afterstate_runs(&self) -> Vec<String> {
        self.inner.afterstate_runs()
    }

    /// `afterstate_scores` from a score per distinct settled state
    /// (`sim::env::VecEnv::afterstate_scores_given`).
    fn afterstate_scores_given<'py>(&mut self, py: Python<'py>, scores: Vec<f64>) -> PyResult<(Bound<'py, PyArray1<f32>>, std::collections::BTreeMap<String, usize>)> {
        let scored = py.detach(|| self.inner.afterstate_scores_given(scores));
        Ok((scored.score.into_pyarray(py), scored.capped))
    }

    /// Option `option` of row `row` of the afterstates scored last, in
    /// words, and the sub-decisions' options on its best path.
    fn afterstate_option(&self, row: usize, option: usize) -> (String, Vec<String>) {
        self.inner.afterstate_option(row, option)
    }

    /// Answer each of `envs`' run decision with its option token
    /// `options[k]` and play on, writing the combat rows of the fights that
    /// start. Returns the runs that ended: (env, run).
    fn step_run(
        &mut self,
        py: Python<'_>,
        envs: Vec<usize>,
        options: PyReadonlyArray1<i64>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<Vec<(usize, RunFight)>> {
        let o = options.as_slice()?;
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        let m = mask.as_slice_mut()?;
        let ended = py.detach(|| self.inner.step_run(&envs, o, f, i, m));
        Ok(ended.into_iter().map(|(env, r)| (env, run_fight(r))).collect())
    }

    /// Switch to cycling through the recordings in `dir` (the held-out
    /// set). Returns the number loaded and the files that failed to parse.
    /// With `ascension`, only the fights recorded at it.
    #[pyo3(signature = (dir, ascension=None))]
    fn load_recordings(&mut self, dir: &str, ascension: Option<u8>) -> PyResult<(usize, Vec<String>)> {
        let (mut setups, errors) =
            load_recordings(std::path::Path::new(dir)).map_err(pyo3::exceptions::PyIOError::new_err)?;
        setups.retain(|s| ascension.is_none_or(|a| s.asc.0 == a));
        let n = setups.len();
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("no recordings loaded"));
        }
        self.inner.set_fixed(setups);
        Ok((n, errors))
    }

    /// Fill `floats [n, N_FLOATS]`, `ids [n, N_IDS]`, `mask [n, N_ACTIONS]`.
    fn observe(
        &self,
        py: Python<'_>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<()> {
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        let m = mask.as_slice_mut()?;
        py.detach(|| self.inner.observe(f, i, m));
        Ok(())
    }

    /// Apply `actions [n]`, write the next observation and the transition's
    /// `rewards [n]` and `dones [n]`, and return the fights that ended.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<i64>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
        mut rewards: PyReadwriteArray1<f32>,
        mut dones: PyReadwriteArray1<bool>,
    ) -> PyResult<Vec<End>> {
        let a = actions.as_slice()?;
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        let m = mask.as_slice_mut()?;
        let r = rewards.as_slice_mut()?;
        let d = dones.as_slice_mut()?;
        let ends = py.detach(|| self.inner.step(a, f, i, m, r, d));
        Ok(ends.into_iter().map(episode_end).collect())
    }
}

/// What one `take` handed out (`sim::runloop::Taken`): the envs at a
/// combat decision (row k of the combat buffers is `combat[k]`), the envs
/// at a run decision (row k of the run buffers), the fights and runs that
/// ended since the last take, the fight starts logged and the trace lines
/// written, and how many envs are still in the batch.
#[pyclass(get_all)]
struct Taken {
    combat: Vec<usize>,
    decision: Vec<usize>,
    ends: Vec<End>,
    runs: Vec<RunFight>,
    starts: Vec<String>,
    traces: Vec<String>,
    active: usize,
}

/// An ended fight as the tuple Python's `End` reads.
fn episode_end(e: sim::env::EpisodeEnd) -> End {
    let (encounter, kind) = (format!("{:?}", e.encounter), format!("{:?}", e.kind));
    (e.env, e.won, e.hp_frac, e.hp_lost, e.potions_used, e.steps, e.floor, encounter, kind, e.reward, e.restart, e.run.map(run_fight))
}

/// Live view of one real combat: recorder lines in, the sim's state and
/// plain-words actions out (`sim::replay::Replayer`).
#[pyclass]
struct Advisor {
    ids: Ids,
    seed: u64,
    start: Option<Value>,
    inner: Option<Replayer>,
    /// Where the fight began, for the forks' rewards.
    base: Option<Baseline>,
}

/// `Step` as the status string the advisor prints.
fn status(step: Result<Step, String>) -> String {
    match step {
        Ok(Step::Ok) => "ok".into(),
        Ok(Step::Decision) => "decision".into(),
        Ok(Step::Ended) => "ended".into(),
        Ok(Step::Diverged(e)) | Err(e) => format!("diverged: {e}"),
    }
}

#[pymethods]
impl Advisor {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self { ids: Ids::new(), seed, start: None, inner: None, base: None }
    }

    /// Feed one line of the recording. Returns "ok", "decision", "ended",
    /// "diverged: <why>", "unsupported: <why>" for a fight the sim cannot
    /// build (an Underdocks encounter, say), or "waiting" before the combat
    /// has started. A `start` record begins a new combat, dropping the old.
    fn feed_line(&mut self, line: &str) -> PyResult<String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok("ok".into());
        }
        let rec: Value = serde_json::from_str(line)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad json: {e}")))?;
        match rec["t"].as_str().unwrap_or("") {
            // A start record that carries the player's HP (written since
            // 2026-09-23) builds the combat at once, so a choice turn 1's
            // start opens (Gambling Chip) finds the sim waiting at it. An
            // older one waits for the first snapshot, which carried the HP
            // and the opening draw order instead.
            "start" => {
                self.inner = None;
                if rec["hp"].is_null() {
                    self.start = Some(rec);
                    return Ok("ok".into());
                }
                return Ok(match Replayer::new(&rec, None, self.ids.clone(), self.seed) {
                    Ok(r) => {
                        self.base = Some(Baseline::of(r.combat()));
                        self.inner = Some(r);
                        "ok".into()
                    }
                    Err(e) => format!("unsupported: {e}"),
                });
            }
            "snapshot" if self.inner.is_none() => {
                let Some(start) = self.start.clone() else { return Ok("waiting".into()) };
                match Replayer::new(&start, Some(&rec), self.ids.clone(), self.seed) {
                    Ok(r) => {
                        self.base = Some(Baseline::of(r.combat()));
                        self.inner = Some(r);
                    }
                    // Nothing about this fight is followable, so forget the
                    // start: later snapshots wait quietly for the next one.
                    Err(e) => {
                        self.start = None;
                        return Ok(format!("unsupported: {e}"));
                    }
                }
            }
            _ => {}
        }
        let Some(r) = self.inner.as_mut() else { return Ok("waiting".into()) };
        Ok(status(r.feed(&rec)))
    }

    /// Release a snapshot the replayer is holding back. Call this when the
    /// recorder has gone quiet: the game is waiting on the player, so the
    /// snapshot is a real decision point and not a mid-resolution poll.
    fn flush(&mut self) -> String {
        match self.inner.as_mut() {
            Some(r) => status(r.flush()),
            None => "waiting".into(),
        }
    }

    /// Release the held snapshot whether or not it settles at once. The
    /// bridge says when the game is waiting on the player, so the snapshot
    /// is a real decision point and may take a reseed to match.
    fn settle(&mut self) -> String {
        match self.inner.as_mut() {
            Some(r) => status(r.finish()),
            None => "waiting".into(),
        }
    }

    /// True when the sim is settled at a decision point.
    fn at_decision(&self) -> bool {
        self.inner.as_ref().is_some_and(Replayer::at_decision)
    }

    fn is_over(&self) -> bool {
        self.inner.as_ref().is_some_and(|r| r.combat().is_over())
    }

    /// True when the sim has a card choice open (a pick or a skip is legal).
    fn choosing(&self) -> bool {
        self.inner.as_ref().is_some_and(|r| r.combat().pending.is_some())
    }

    /// Fill `floats [1, N_FLOATS]`, `ids [1, N_IDS]`, `mask [1, N_ACTIONS]`
    /// with the current state, the same encoding training used.
    fn observe(
        &self,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<()> {
        let c = self.combat()?;
        encode(c, floats.as_slice_mut()?, ids.as_slice_mut()?, mask.as_slice_mut()?);
        Ok(())
    }

    /// An action index in plain words, or None if it is not legal now.
    fn describe(&self, index: usize) -> PyResult<Option<String>> {
        Ok(describe(self.combat()?, index))
    }

    /// The bridge command (a JSON line) that makes the game take action
    /// `index`, or None if it is not legal now (`sim::replay::command`).
    fn command(&self, index: usize) -> PyResult<Option<String>> {
        Ok(command(self.combat()?, index).map(|v| v.to_string()))
    }

    /// Turn, energy, HP, and the enemies with their intents.
    fn summary(&self) -> PyResult<String> {
        Ok(summary(self.combat()?))
    }

    /// Whether ending the turn now leaves the player alive once the enemy
    /// turn is over, played out on a copy (their rolled moves, the player's
    /// block, a Fairy in a Bottle catching a death). The recording pilot
    /// asks before it ends a turn.
    fn end_turn_survives(&self) -> PyResult<bool> {
        let mut c = self.combat()?.clone();
        if c.is_over() || c.pending.is_some() {
            return Ok(true);
        }
        c.step(sim::combat::Action::EndTurn);
        Ok(c.outcome != Some(sim::combat::Outcome::Lost))
    }

    /// Decision points matched, actions applied, reseeds needed.
    fn counts(&self) -> PyResult<(usize, usize, u32)> {
        let r = self.inner.as_ref().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no combat yet"))?;
        Ok((r.report().snapshots, r.report().actions, r.report().reseeds))
    }

    /// `n` copies of the current state for a search over the rest of the
    /// turn, in `groups` groups that each share a draw-pile shuffle.
    #[pyo3(signature = (n, groups=4, seed=0))]
    fn fork(&self, n: usize, groups: usize, seed: u64) -> PyResult<Forks> {
        let base = self.base.expect("a combat has its baseline");
        Ok(Forks { inner: InnerForks::new(self.combat()?, base, n, groups, seed) })
    }
}

/// Copies of one combat stepped together (`sim::env::Forks`): the
/// advisor's turn search plays each one to the end of the turn.
#[pyclass]
struct Forks {
    inner: InnerForks,
}

#[pymethods]
impl Forks {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Encode forks `rows`, one row per distinct observation, packed at
    /// the front of the buffers; `inverse[k]` is fork `rows[k]`'s row.
    /// Returns the number of rows (`sim::env::Forks::observe_unique`).
    fn observe_unique(
        &self,
        py: Python<'_>,
        rows: Vec<usize>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
        mut inverse: PyReadwriteArray1<i64>,
    ) -> PyResult<usize> {
        let k = rows.len();
        let f = floats.as_slice_mut()?;
        let i = ids.as_slice_mut()?;
        let m = mask.as_slice_mut()?;
        let inv = &mut inverse.as_slice_mut()?[..k];
        let mut rows_of = vec![0usize; k];
        let n = py.detach(|| self.inner.observe_unique(&rows, f, i, m, &mut rows_of));
        inv.iter_mut().zip(rows_of).for_each(|(o, r)| *o = r as i64);
        Ok(n)
    }

    /// Apply `actions [n]` to the forks still in their turn; write the
    /// shaped reward of each transition.
    fn step(&mut self, py: Python<'_>, actions: PyReadonlyArray1<i64>, mut rewards: PyReadwriteArray1<f32>) -> PyResult<()> {
        let a = actions.as_slice()?;
        let r = rewards.as_slice_mut()?;
        py.detach(|| self.inner.step(a, r));
        Ok(())
    }

    /// Copies of forks `rows` as new roots, `n` per row, with fresh dice
    /// and shuffles (`VecEnv.fork`): to value a state a search reached by
    /// playing it out many times. Rewards stay shaped from the fight's start.
    #[pyo3(signature = (rows, n, groups=4, seed=0, depth=1))]
    fn fork(&self, rows: Vec<usize>, n: usize, groups: usize, seed: u64, depth: u32) -> Forks {
        let roots: Vec<_> = rows.iter().map(|&i| self.inner.combat(i)).collect();
        let bases: Vec<_> = rows.iter().map(|&i| self.inner.base(i)).collect();
        Forks { inner: InnerForks::of(&roots, &bases, n, groups, seed, depth) }
    }

    /// The forks still in their turn.
    fn live(&self) -> Vec<usize> {
        self.inner.live()
    }

    fn is_over(&self) -> Vec<bool> {
        (0..self.inner.len()).map(|i| self.inner.combat(i).is_over()).collect()
    }

    /// An action index in fork `i` in plain words, or None if not legal there.
    fn describe(&self, i: usize, index: usize) -> Option<String> {
        describe(self.inner.combat(i), index)
    }
}

/// The exact turn search (`sim::turnsearch`) for the fights of a `VecEnv`,
/// one search per env kept between decisions: a later decision of the
/// same turn that the search already expanded reuses it, unless the state
/// cap cut it short.
///
/// Per decision: `prepare` searches where needed, `encode_leaves` and
/// `set_values` put the value head on the new searches' leaves, and
/// `action_values` reads each env's action values.
#[pyclass]
struct TurnPlanner {
    cfg: sim::turnsearch::Config,
    searches: Vec<Option<sim::turnsearch::Search>>,
    /// Envs searched afresh by the last `prepare`, awaiting values.
    fresh: Vec<usize>,
    /// Per env: the lines `lines` last listed, and the one committed to.
    lines: Vec<Vec<sim::turnsearch::Line>>,
    plans: Vec<Option<sim::turnsearch::Line>>,
}

impl TurnPlanner {
    fn fresh_leaves(&self) -> Vec<&sim::combat::Combat> {
        self.fresh.iter().flat_map(|&i| self.searches[i].as_ref().unwrap().leaves().iter().map(|l| &*l.combat)).collect()
    }
}

#[pymethods]
impl TurnPlanner {
    #[new]
    #[pyo3(signature = (n, max_states=500, quiesce_states=1000, draw_cap=32, samples=4, end_samples=8, max_micros=0, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, max_states: usize, quiesce_states: usize, draw_cap: usize, samples: usize, end_samples: usize, max_micros: u64, seed: u64) -> Self {
        let cfg = sim::turnsearch::Config { max_states, quiesce_states, draw_cap, samples, end_samples, max_micros, seed };
        Self { cfg, searches: (0..n).map(|_| None).collect(), fresh: vec![], lines: vec![vec![]; n], plans: vec![None; n] }
    }

    /// Get each of `envs` a search holding its current state: the one it
    /// has, when that expanded the state and was not cut short, or a new
    /// one ordered by `priors [len(envs), N_ACTIONS]` (the policy's
    /// probabilities). Returns the envs searched afresh.
    fn prepare(&mut self, py: Python<'_>, env: PyRef<'_, VecEnv>, envs: Vec<usize>, priors: PyReadonlyArray2<f32>) -> PyResult<Vec<usize>> {
        let priors = priors.as_slice()?;
        let inner = &env.inner;
        let (fresh, rows): (Vec<usize>, Vec<usize>) = envs
            .iter()
            .enumerate()
            .filter(|&(_, &i)| !self.searches[i].as_ref().is_some_and(|s| !s.stats.capped && s.find(&inner.combat(i)).is_some()))
            .map(|(row, &i)| (i, row))
            .unzip();
        let guards: Vec<_> = fresh.iter().map(|&i| inner.combat(i)).collect();
        let roots: Vec<&sim::combat::Combat> = guards.iter().map(|g| &**g).collect();
        let bases: Vec<Baseline> = fresh.iter().map(|&i| inner.base(i)).collect();
        let p: Vec<Option<&[f32]>> = rows.iter().map(|&r| Some(&priors[r * N_ACTIONS..][..N_ACTIONS])).collect();
        let cfg = self.cfg;
        let done = py.detach(|| sim::turnsearch::Search::run_all(&roots, &bases, &p, &cfg));
        for (&i, s) in fresh.iter().zip(done) {
            self.searches[i] = Some(s);
            self.lines[i].clear();
            self.plans[i] = None;
        }
        self.fresh = fresh.clone();
        Ok(fresh)
    }

    /// Leaves of the searches the last `prepare` made, all together.
    fn n_leaves(&self) -> usize {
        self.fresh.iter().map(|&i| self.searches[i].as_ref().unwrap().leaves().len()).sum()
    }

    /// Encode leaves `start..` into the buffers, as many as they hold.
    /// Returns how many were written.
    fn encode_leaves(
        &self,
        py: Python<'_>,
        start: usize,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<usize> {
        let (f, i, m) = (floats.as_slice_mut()?, ids.as_slice_mut()?, mask.as_slice_mut()?);
        let leaves = self.fresh_leaves();
        let n = (f.len() / N_FLOATS).min(leaves.len().saturating_sub(start));
        py.detach(|| sim::turnsearch::encode_all(&leaves[start..start + n], f, i, m));
        Ok(n)
    }

    /// The value head's value of every leaf `encode_leaves` wrote, in order.
    fn set_values(&mut self, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let mut v = values.as_slice()?;
        for &i in &self.fresh {
            let s = self.searches[i].as_mut().unwrap();
            let n = s.leaves().len();
            s.set_values(&v[..n]);
            v = &v[n..];
        }
        if !v.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("more values than leaves"));
        }
        Ok(())
    }

    /// (action index, value) for env `i`'s legal actions, or None when its
    /// search does not hold its state.
    fn action_values(&self, env: PyRef<'_, VecEnv>, i: usize) -> Option<Vec<(usize, f32)>> {
        self.searches[i].as_ref()?.action_values(&env.inner.combat(i))
    }

    /// Env `i`'s lines from its current state, best value first, as
    /// (first action index, value, reward on the way, how it ends); kept
    /// for `commit` and `fork_lines` to name by position. None when its
    /// search does not hold the state.
    fn lines(&mut self, env: PyRef<'_, VecEnv>, i: usize) -> Option<Vec<(usize, f32, f32, String)>> {
        let s = self.searches[i].as_ref()?;
        let guard = env.inner.combat(i);
        let c = &*guard;
        let lines = s.lines(c)?;
        let out = lines.iter().map(|l| s.line_action(c, l).map(|a| (a, l.value, l.way, format!("{:?}", l.end)))).collect::<Option<Vec<_>>>()?;
        self.lines[i] = lines;
        Some(out)
    }

    /// Follow env `i`'s line `k` (of the last `lines`) from here.
    fn commit(&mut self, i: usize, k: usize) {
        self.plans[i] = self.lines[i].get(k).cloned();
    }

    /// The committed line's action in env `i`'s current state, when the
    /// state is on it; None once it has left the line (chance, or the
    /// turn is over).
    fn planned_action(&self, env: PyRef<'_, VecEnv>, i: usize) -> Option<usize> {
        let (s, line) = (self.searches[i].as_ref()?, self.plans[i].as_ref()?);
        s.line_action(&env.inner.combat(i), line)
    }

    /// `n` copies of where each of `picks` (env, line) takes its env's
    /// state before the line's last step, the copies of pick p on dice and
    /// shuffles from `seeds[p]` (the same seed across one env's lines
    /// compares them on the same luck), each its own shuffle. Returns the
    /// forks, each forked pick's last step (-1 for a cut line) and which
    /// picks were forked: a line whose steps the fight's own state does
    /// not take (the search's draws or dice were not the fight's) is not.
    fn fork_lines(
        &self,
        env: PyRef<'_, VecEnv>,
        picks: Vec<(usize, usize)>,
        n: usize,
        seeds: Vec<u64>,
        depth: u32,
    ) -> PyResult<(Forks, Vec<i64>, Vec<bool>)> {
        let err = |m: &str| pyo3::exceptions::PyValueError::new_err(m.to_string());
        let (mut starts, mut last, mut ok, mut bases, mut kept_seeds) = (vec![], vec![], vec![], vec![], vec![]);
        for (&(i, k), &seed) in picks.iter().zip(&seeds) {
            let s = self.searches[i].as_ref().ok_or_else(|| err("no search for env"))?;
            let line = self.lines[i].get(k).ok_or_else(|| err("no such line"))?;
            let start = s.line_start(&env.inner.combat(i), line);
            ok.push(start.is_some());
            let Some((c, a)) = start else { continue };
            last.push(a.map_or(-1, |a| sim::turnsearch::index(&c, a).map_or(-1, |x| x as i64)));
            starts.push(c);
            bases.push(env.inner.base(i));
            kept_seeds.push(seed);
        }
        let roots: Vec<_> = starts.iter().collect();
        Ok((Forks { inner: InnerForks::with_seeds(&roots, &bases, &kept_seeds, n, n, depth) }, last, ok))
    }

    /// For each of `rows` of `forks`: the search's best action (by index)
    /// in env `owners[k]`'s search, when the copy is in a state of the
    /// searched turn the search expanded; -1 for another state of the
    /// turn, -2 once the copy is past it.
    fn tree_actions(&self, py: Python<'_>, forks: PyRef<'_, Forks>, rows: Vec<usize>, owners: Vec<usize>) -> Vec<i64> {
        use rayon::prelude::*;
        let f = &forks.inner;
        py.detach(|| {
            rows.par_iter()
                .zip(&owners)
                .map(|(&r, &o)| {
                    let (c, s) = (f.combat(r), self.searches[o].as_ref());
                    match s {
                        Some(s) if s.in_turn(c) => s.best_action(c).map_or(-1, |a| a as i64),
                        _ => -2,
                    }
                })
                .collect()
        })
    }

    /// Env `i`'s search as numbers: states found and expanded, leaves,
    /// steps by rule, whether the cap bit, time, and the best line's
    /// rank and cut (`Search::best_line`).
    fn stats<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(s) = self.searches[i].as_ref() else { return Ok(None) };
        let d = PyDict::new(py);
        let t = &s.stats;
        for (k, v) in [
            ("nodes", t.nodes),
            ("expanded", t.expanded),
            ("end_leaves", t.end_leaves),
            ("cut_leaves", t.cut_leaves),
            ("arrivals", t.arrivals),
            ("det", t.det),
            ("drawn", t.drawn),
            ("sampled", t.sampled),
            ("end_turn", t.end_turn),
            ("quiesced", t.quiesced),
        ] {
            d.set_item(k, v)?;
        }
        d.set_item("lines", t.lines)?;
        d.set_item("capped", t.capped)?;
        d.set_item("micros", t.micros)?;
        if !s.leaves().is_empty() {
            let (rank, cut) = s.best_line();
            d.set_item("pv_rank", rank)?;
            d.set_item("pv_cut", cut)?;
        }
        Ok(Some(d))
    }
}

impl Advisor {
    fn combat(&self) -> PyResult<&sim::combat::Combat> {
        self.inner.as_ref().map(Replayer::combat).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no combat yet"))
    }
}

/// Offsets into the float, id, and action vectors, by name.
#[pyfunction]
fn layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in [
        ("n_floats", N_FLOATS),
        ("n_ids", N_IDS),
        ("n_actions", N_ACTIONS),
        ("card_vocab", CARD_VOCAB),
        ("monster_vocab", MONSTER_VOCAB),
        ("potion_vocab", POTION_VOCAB),
        ("max_hand", MAX_HAND),
        ("max_enemies", MAX_ENEMIES),
        ("max_potions", MAX_POTIONS),
        ("max_choices", MAX_CHOICES),
        ("targets", TARGETS),
        ("a_play", A_PLAY),
        ("a_potion", A_POTION),
        ("a_end_turn", A_END_TURN),
        ("a_choose", A_CHOOSE),
        ("a_skip", A_SKIP),
        ("i_hand", I_HAND),
        ("i_enemies", I_ENEMIES),
        ("i_potions", I_POTIONS),
        ("i_choices", I_CHOICES),
        ("i_moves", I_MOVES),
        ("i_enchants", I_ENCHANTS),
        ("i_choice_enchants", I_CHOICE_ENCHANTS),
        ("i_piles", I_PILES),
        ("i_pile_enchants", I_PILE_ENCHANTS),
        ("i_resumes", I_RESUMES),
        ("max_pile_rows", MAX_PILE_ROWS),
        ("card_feats", CARD_FEATS),
        ("pile_feats", PILE_FEATS),
        ("relic_feats", RELIC_FEATS),
        ("move_vocab", move_vocab()),
        ("enchant_vocab", ENCHANT_VOCAB),
        ("global_len", GLOBAL_LEN),
        ("f_player_powers", F_PLAYER_POWERS),
        ("f_hand", F_HAND),
        ("hand_feats", HAND_FEATS),
        ("f_piles", F_PILES),
        ("f_enemies", F_ENEMIES),
        ("enemy_base", ENEMY_BASE),
        ("intent_nums", INTENT_NUMS),
        ("enemy_feats", ENEMY_FEATS),
        ("f_relics", F_RELICS),
        ("f_potions", F_POTIONS),
        ("f_choices", F_CHOICES),
        ("choice_feats", CHOICE_FEATS),
        ("n_cards", N_CARDS),
        ("n_powers", N_POWERS),
        ("n_relics", N_RELICS),
        ("n_intents", N_INTENTS),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// The points a run can start at besides floor 1, the latest first
/// (`sim::forward::START_POINTS`): "act 3 boss", "act 3 entrance", ...
#[pyfunction]
fn start_points() -> Vec<String> {
    use sim::forward::StartPoint;
    sim::forward::START_POINTS
        .iter()
        .map(|p| match p {
            StartPoint::Entrance(act) => format!("act {} entrance", act + 1),
            StartPoint::BossDoor(act) => format!("act {} boss", act + 1),
        })
        .collect()
}

/// Offsets and sizes of the run observation (`sim::runobs`), by name.
#[pyfunction]
fn run_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    use runobs::*;
    let d = PyDict::new(py);
    for (k, v) in [
        ("run_floats", RUN_FLOATS),
        ("run_ids", RUN_IDS),
        ("max_deck", MAX_DECK),
        ("max_relics", MAX_RELICS),
        ("max_potions", MAX_POTIONS),
        ("max_options", MAX_OPTIONS),
        ("option_cards", OPTION_CARDS),
        ("global_ids", GLOBAL_IDS),
        ("global_floats", GLOBAL_FLOATS),
        ("f_forecast", F_FORECAST),
        ("forecast_floats", FORECAST_FLOATS),
        ("forecast_rolls", FORECAST_ROLLS),
        ("map_feats", MAP_FEATS),
        ("deck_ids", DECK_IDS),
        ("deck_floats", DECK_FLOATS),
        ("relic_ids", RELIC_IDS),
        ("relic_floats", RELIC_FLOATS),
        ("potion_ids", POTION_IDS),
        ("potion_floats", POTION_FLOATS),
        ("option_ids", OPTION_IDS),
        ("option_floats", OPTION_FLOATS),
        ("f_deck", F_DECK),
        ("f_relics", F_RELICS),
        ("f_potions", F_POTIONS),
        ("f_options", F_OPTIONS),
        ("i_deck", I_DECK),
        ("i_relics", I_RELICS),
        ("i_potions", I_POTIONS),
        ("i_options", I_OPTIONS),
        ("i_map", I_MAP),
        ("map_rows", MAP_ROWS),
        ("map_cols", MAP_COLS),
        ("map_node_ids", MAP_NODE_IDS),
        ("card_vocab", CARD_VOCAB),
        ("enchant_vocab", ENCHANT_VOCAB),
        ("potion_vocab", POTION_VOCAB),
        ("relic_vocab", RELIC_VOCAB),
        ("decision_vocab", DECISION_VOCAB),
        ("option_vocab", OPTION_VOCAB),
        ("room_vocab", ROOM_VOCAB),
        ("act_vocab", ACT_VOCAB),
        ("event_vocab", EVENT_VOCAB),
        ("boss_vocab", BOSS_VOCAB),
        ("event_key_vocab", EVENT_KEY_VOCAB),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// The run vocabularies' names by id, id 0 the pad: "decision", "option",
/// "room", "act", "event", "boss", and "relic" with the relics the combat
/// sim leaves out after its own (`sim::runobs`).
#[pyfunction]
fn run_names() -> std::collections::HashMap<&'static str, Vec<String>> {
    use runobs::*;
    let named = |names: Vec<String>| std::iter::once("<pad>".to_string()).chain(names).collect::<Vec<_>>();
    std::collections::HashMap::from([
        ("decision", named(DECISIONS.iter().map(|d| d.to_string()).collect())),
        ("option", named(OPTION_KINDS.iter().map(|k| format!("{k:?}")).collect())),
        ("room", named(ROOMS.iter().map(|r| r.to_string()).collect())),
        ("act", named(ACTS.iter().map(|a| format!("{a:?}")).collect())),
        ("event", named(RUN_EVENTS.iter().map(|e| e.to_string()).collect())),
        ("boss", named(RUN_BOSSES.iter().map(|b| format!("{b:?}")).collect())),
        (
            "relic",
            named(sim::relic::ALL.iter().map(|r| sim::replay::slug(&format!("{r:?}"))).chain(RUN_RELICS.iter().map(|r| r.to_string())).collect()),
        ),
    ])
}

/// Card names by vocabulary index (index 0 is the pad).
#[pyfunction]
fn card_names() -> Vec<String> {
    std::iter::once("<pad>".to_string()).chain(ALL_CARDS.iter().map(|c| format!("{c:?}"))).collect()
}

/// Every card's game id (`BODY_SLAM`), in the sim's order.
#[pyfunction]
fn card_ids() -> Vec<String> {
    ALL_CARDS.iter().map(|c| sim::replay::slug(&format!("{c:?}"))).collect()
}

/// Every card's type (`Attack`, `Skill`, `Power`, `Status`, `Curse`), in
/// `game_ids()["card"]` order.
#[pyfunction]
fn card_types() -> Vec<String> {
    ALL_CARDS.iter().map(|&c| format!("{:?}", sim::card::def(c).ty)).collect()
}

/// Every card, relic, potion and enchantment's game id (`BODY_SLAM`), each
/// in the sim's order: index i here is id i + 1 in the policy's embeddings.
#[pyfunction]
fn game_ids() -> std::collections::HashMap<&'static str, Vec<String>> {
    let slugs = |names: Vec<String>| names.iter().map(|n| sim::replay::slug(n)).collect::<Vec<_>>();
    std::collections::HashMap::from([
        ("card", slugs(ALL_CARDS.iter().map(|c| format!("{c:?}")).collect())),
        ("relic", slugs(sim::relic::ALL.iter().map(|c| format!("{c:?}")).collect())),
        ("potion", slugs(sim::potion::ALL.iter().map(|c| format!("{c:?}")).collect())),
        ("enchant", slugs(sim::enchant::ALL.iter().map(|c| format!("{c:?}")).collect())),
    ])
}

/// A generated run state for a fight on `floor`, in the recorder's `start`
/// format as JSON (`FightSetup::run_json`); the same seed and floor give
/// the same run.
#[pyfunction]
fn generate_run(seed: u64, floor: u32) -> String {
    sim::gen::generate(&mut sim::rng::Rng::new(seed), floor, sim::types::Ascension(10)).run_json().to_string()
}

/// Rows, options taken, floors, streamed flags, what was left out and the
/// rows' forecast fights (`imitation`).
type Imitation<'py> =
    (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>, Vec<usize>, Vec<usize>, Vec<bool>, std::collections::BTreeMap<String, usize>, Forecast<'py>);

/// Afterstates' combat rows, leaves and distinct states (`afterstates`).
type AfterstateRows<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>, usize, usize);

/// Forecast fights at their openings (`sim::runobs::forecast_rows`):
/// combat rows `floats [n * N_FLOATS]` and `ids [n * N_IDS]`, the run row
/// each belongs to, and its encounter.
type Forecast<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>, Vec<usize>, Vec<String>);

/// The forecast fights of run rows, `fights[k]` those of row `k`.
fn forecast_rows<'py>(py: Python<'py>, fights: &[&[sim::gen::FightSetup]]) -> Forecast<'py> {
    let owner = fights.iter().enumerate().flat_map(|(k, f)| std::iter::repeat_n(k, f.len())).collect();
    let all: Vec<sim::gen::FightSetup> = fights.iter().flat_map(|f| f.iter().cloned()).collect();
    let encounters = all.iter().map(|f| format!("{:?}", f.encounter)).collect();
    let (floats, ids) = py.detach(|| runobs::forecast_rows(&all));
    (floats.into_pyarray(py), ids.into_pyarray(py), owner, encounters)
}

/// A run history's decisions as the run policy would see them, where the
/// walk is faithful to the record (`sim::history::imitate`), or None for a
/// run the port cannot walk: run rows `floats [n * RUN_FLOATS]` and `ids [n
/// * RUN_IDS]`, the option token taken, the floor, whether the Rewards
/// stream was still followed, and the decisions left out by why.
#[pyfunction]
fn imitation<'py>(py: Python<'py>, run: &str) -> PyResult<Option<Imitation<'py>>> {
    let run: Value = serde_json::from_str(run).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if !sim::history::eligible(&run) {
        return Ok(None);
    }
    let im = py.detach(|| sim::history::imitate(&run));
    let floats: Vec<f32> = im.rows.iter().flat_map(|r| r.obs.floats.iter().copied()).collect();
    let ids: Vec<i64> = im.rows.iter().flat_map(|r| r.obs.ids.iter().copied()).collect();
    let option = im.rows.iter().map(|r| r.option).collect();
    let floor = im.rows.iter().map(|r| r.floor).collect();
    let streamed = im.rows.iter().map(|r| r.streamed).collect();
    let fights: Vec<&[sim::gen::FightSetup]> = im.rows.iter().map(|r| r.obs.forecast.as_slice()).collect();
    let forecast = forecast_rows(py, &fights);
    Ok(Some((floats.into_pyarray(py), ids.into_pyarray(py), option, floor, streamed, im.left_out, forecast)))
}

/// Game ids of the cards the sim refuses to play (`card::UNSUPPORTED_CARDS`).
#[pyfunction]
fn unsupported_cards() -> Vec<String> {
    sim::card::UNSUPPORTED_CARDS.iter().map(|(c, _)| sim::replay::slug(&format!("{c:?}"))).collect()
}

#[pyfunction]
fn monster_names() -> Vec<String> {
    std::iter::once("<pad>".to_string()).chain(ALL_MONSTERS.iter().map(|c| format!("{c:?}"))).collect()
}

/// Every act 1 encounter as (id, act, kind), in the sim's own order. The
/// recording wizard walks this so it cannot drift from `encounter.rs`.
/// The pool a card (game name) transforms within and what it can become,
/// or None for a curse or status (`sim::gen::transform_options`).
#[pyfunction]
fn transform_options(name: &str) -> PyResult<Option<(&'static str, Vec<String>)>> {
    sim::gen::transform_options(name, &Ids::new()).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn encounters() -> Vec<(String, String, String)> {
    sim::encounter::ALL
        .iter()
        .map(|e| (sim::replay::slug(&format!("{e:?}")), format!("{:?}", e.act()), format!("{:?}", e.kind())))
        .collect()
}

/// Why the sim cannot build a fight from this recorder `start` record, if it
/// cannot. One check for every id a run can carry: a card, an enchantment, a
/// relic, a potion, the encounter, the monsters. A colorless card or an act 2
/// relic the sim has never heard of comes out here, before the fight rather
/// than halfway through it.
#[pyfunction]
fn start_blocker(start: &str) -> PyResult<Option<String>> {
    let v: Value = serde_json::from_str(start).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(sim::gen::FightSetup::from_start(&v, None, &Ids::new()).err())
}

/// What the sim knows about an encounter: each monster with the powers it
/// starts with and the moves it can show. Tooling describes a fight from
/// this, so a new act needs no notes written by hand.
#[pyfunction]
fn encounter_brief(name: &str, asc: u8) -> PyResult<Vec<(String, Vec<(String, i32)>, Vec<String>)>> {
    let ids = Ids::new();
    let enc = *ids.encounter(name).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown encounter {name}")))?;
    let asc = Ascension(asc);
    let mut seen: Vec<sim::ids::MonsterId> = vec![];
    for spec in enc.monsters(&mut sim::rng::Rng::new(0)) {
        if !seen.contains(&spec.id) {
            seen.push(spec.id);
        }
    }
    Ok(seen
        .into_iter()
        .map(|id| {
            let powers = sim::monster::Monster::innate_powers(id, asc)
                .into_iter()
                .map(|(p, n)| (sim::replay::slug(&format!("{p:?}")), n))
                .collect();
            let moves = sim::monster::move_names_of(id, asc).into_iter().map(str::to_string).collect();
            (sim::replay::slug(&format!("{id:?}")), powers, moves)
        })
        .collect())
}

/// Tree searches over whole fights (`sim::mcts`), one per env, run in
/// lockstep: each `descend` takes one simulation in every listed env and
/// writes the states they reached for the network, `expand` takes its read.
#[pyclass]
struct TreeSearch {
    cfg: sim::mcts::Config,
    trees: Vec<Option<sim::mcts::Tree>>,
    /// Envs whose last `descend` left a leaf, in row order.
    waiting: Vec<usize>,
}

#[pymethods]
impl TreeSearch {
    #[new]
    #[pyo3(signature = (n, c_puct=1.25, widen=f32::INFINITY, widen_exp=0.5, max_depth=200, seed=0, clairvoyant=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, c_puct: f32, widen: f32, widen_exp: f32, max_depth: u32, seed: u64, clairvoyant: bool) -> Self {
        let cfg = sim::mcts::Config { c_puct, widen, widen_exp, max_depth, seed, clairvoyant };
        Self { cfg, trees: (0..n).map(|_| None).collect(), waiting: vec![] }
    }

    /// Start a fresh search from each of `envs`' current states.
    fn start(&mut self, env: PyRef<'_, VecEnv>, envs: Vec<usize>) {
        for i in envs {
            self.trees[i] = Some(sim::mcts::Tree::new(&env.inner.combat(i), env.inner.base(i), self.cfg));
        }
    }

    /// One simulation in each of `envs`, in parallel. The states that need
    /// the network go into the buffers' first rows; returns their envs in
    /// row order (fewer than `envs` when simulations ended the fight).
    fn descend(
        &mut self,
        py: Python<'_>,
        envs: Vec<usize>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<Vec<usize>> {
        use rayon::prelude::*;
        let (f, i, m) = (floats.as_slice_mut()?, ids.as_slice_mut()?, mask.as_slice_mut()?);
        let mut wanted = vec![false; self.trees.len()];
        envs.iter().for_each(|&e| wanted[e] = true);
        let waiting: Vec<usize> = py.detach(|| {
            self.trees
                .par_iter_mut()
                .enumerate()
                .filter(|(e, t)| wanted[*e] && t.is_some())
                .filter_map(|(e, t)| t.as_mut().unwrap().descend().then_some(e))
                .collect()
        });
        if waiting.len() * N_FLOATS > f.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("buffers too small for the envs"));
        }
        let leaves: Vec<&sim::combat::Combat> = waiting.iter().map(|&e| self.trees[e].as_ref().unwrap().leaf().unwrap()).collect();
        py.detach(|| sim::turnsearch::encode_all(&leaves, f, i, m));
        self.waiting = waiting.clone();
        Ok(waiting)
    }

    /// The network's read of the last `descend`'s leaves, row for row:
    /// `priors [rows, N_ACTIONS]` (probabilities) and `values [rows]`.
    fn expand(&mut self, priors: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let (p, v) = (priors.as_slice()?, values.as_slice()?);
        if v.len() != self.waiting.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("one value per waiting leaf"));
        }
        for (row, &e) in self.waiting.iter().enumerate() {
            self.trees[e].as_mut().unwrap().expand(&p[row * N_ACTIONS..][..N_ACTIONS], v[row]);
        }
        self.waiting.clear();
        Ok(())
    }

    /// (action index, visits, mean return) for env `i`'s root actions.
    fn root_stats(&self, i: usize) -> Vec<(usize, u32, f32)> {
        self.trees[i].as_ref().map_or(vec![], |t| t.root_stats())
    }

    /// Env `i`'s most visited root action.
    fn best_action(&self, i: usize) -> Option<usize> {
        self.trees[i].as_ref()?.best_action()
    }

    fn n_nodes(&self, i: usize) -> usize {
        self.trees[i].as_ref().map_or(0, |t| t.n_nodes())
    }

    /// Player turns past the root's that env `i`'s tree reaches.
    fn turns_deep(&self, i: usize) -> u32 {
        self.trees[i].as_ref().map_or(0, |t| t.turns_deep())
    }
}

/// The share of the enemies' HP a lost fight pays back (`sim::env`), for
/// the whole process.
#[pyfunction]
fn set_loss_damage(w: f32) {
    sim::env::set_loss_damage(w);
}

#[pymodule]
fn _sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<Taken>()?;
    m.add_class::<Advisor>()?;
    m.add_class::<Forks>()?;
    m.add_class::<TurnPlanner>()?;
    m.add_class::<TreeSearch>()?;
    m.add_function(wrap_pyfunction!(set_loss_damage, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(run_layout, m)?)?;
    m.add_function(wrap_pyfunction!(run_names, m)?)?;
    m.add_function(wrap_pyfunction!(start_points, m)?)?;
    m.add_function(wrap_pyfunction!(card_names, m)?)?;
    m.add_function(wrap_pyfunction!(card_ids, m)?)?;
    m.add_function(wrap_pyfunction!(card_types, m)?)?;
    m.add_function(wrap_pyfunction!(game_ids, m)?)?;
    m.add_function(wrap_pyfunction!(generate_run, m)?)?;
    m.add_function(wrap_pyfunction!(unsupported_cards, m)?)?;
    m.add_function(wrap_pyfunction!(imitation, m)?)?;
    m.add_function(wrap_pyfunction!(monster_names, m)?)?;
    m.add_function(wrap_pyfunction!(encounters, m)?)?;
    m.add_function(wrap_pyfunction!(transform_options, m)?)?;
    m.add_function(wrap_pyfunction!(start_blocker, m)?)?;
    m.add_function(wrap_pyfunction!(encounter_brief, m)?)?;
    Ok(())
}
