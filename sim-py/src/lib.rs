//! Python bindings for the training environment (DESIGN.md, Training).
//! Two classes: `VecEnv` for training, `Advisor` for following a real
//! combat from the recorder's log. Plus the layout constants the model needs. Buffers
//! are numpy arrays the caller allocates once; `observe` and `step` fill
//! them in place with the GIL released.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use sim::encode::*;
use sim::env::{EnvConfig, Forks as InnerForks, RunChoices, VecEnv as Inner};
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

/// One finished fight: (env, won, hp_frac, hp_lost, potions_used, steps, floor, encounter, kind, reward, run).
type End = (usize, bool, f32, f32, u32, u32, u32, String, String, f32, Option<RunFight>);

/// A run fight's place in its run: (seed index, act, floor, deck size, how
/// the run ended with it: "won", "died", "stuck: <why>", or None; where the
/// run started: its place in `start_points()`, None for floor 1; whether
/// from the envs' own state there, not a generated one).
type RunFight = (u64, u32, u32, u32, Option<String>, Option<usize>, bool);

fn run_fight(r: sim::env::RunFight) -> RunFight {
    use sim::env::Began;
    use sim::forward::End;
    let end = r.end.map(|end| match end {
        End::Won => "won".into(),
        End::Died => "died".into(),
        End::Stuck(why) => format!("stuck: {why}"),
    });
    let (start, own) = match r.began {
        Began::Floor1 => (None, false),
        Began::Generated(at) => (Some(at.index()), false),
        Began::Own(at) => (Some(at.index()), true),
    };
    (r.seed, r.act, r.floor, r.deck, end, start, own)
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
        let cfg = EnvConfig { asc: Ascension(asc), min_floor, max_floor, max_steps, hard_frac };
        Self { inner: Inner::new(n, seed, cfg) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Curriculum: fraction of resets forced onto an elite or boss.
    fn set_hard_frac(&mut self, frac: f32) {
        self.inner.set_hard_frac(frac);
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

    /// Copies of the current fight in each of `envs`, `n` per env, for a
    /// turn search over many fights in one batch.
    /// `depth` player turns are played: the rest of this one, then more.
    #[pyo3(signature = (envs, n, groups=4, seed=0, depth=1))]
    fn fork(&self, envs: Vec<usize>, n: usize, groups: usize, seed: u64, depth: u32) -> Forks {
        let roots: Vec<_> = envs.iter().map(|&i| self.inner.combat(i)).collect();
        Forks { inner: InnerForks::of(&roots, n, groups, seed, depth) }
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
    #[pyo3(signature = (seed=0, asc=10, choices="random"))]
    fn use_runs(&mut self, py: Python<'_>, seed: u64, asc: u8, choices: &str) -> PyResult<()> {
        let choices = match choices {
            "random" => RunChoices::Random,
            "first" => RunChoices::First,
            "caller" => RunChoices::Caller,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown run choices {other:?}: random, first or caller"))),
        };
        py.detach(|| self.inner.set_runs(Ascension(asc), seed, choices));
        Ok(())
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
        Ok(ends
            .into_iter()
            .map(|e| {
                let (encounter, kind) = (format!("{:?}", e.encounter), format!("{:?}", e.kind));
                (e.env, e.won, e.hp_frac, e.hp_lost, e.potions_used, e.steps, e.floor, encounter, kind, e.reward, e.run.map(run_fight))
            })
            .collect())
    }
}

/// Live view of one real combat: recorder lines in, the sim's state and
/// plain-words actions out (`sim::replay::Replayer`).
#[pyclass]
struct Advisor {
    ids: Ids,
    seed: u64,
    start: Option<Value>,
    inner: Option<Replayer>,
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
        Self { ids: Ids::new(), seed, start: None, inner: None }
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
                        self.inner = Some(r);
                        "ok".into()
                    }
                    Err(e) => format!("unsupported: {e}"),
                });
            }
            "snapshot" if self.inner.is_none() => {
                let Some(start) = self.start.clone() else { return Ok("waiting".into()) };
                match Replayer::new(&start, Some(&rec), self.ids.clone(), self.seed) {
                    Ok(r) => self.inner = Some(r),
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
        Ok(Forks { inner: InnerForks::new(self.combat()?, n, groups, seed) })
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

/// Rows, options taken, floors, streamed flags and what was left out (`imitation`).
type Imitation<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>, Vec<usize>, Vec<usize>, Vec<bool>, std::collections::BTreeMap<String, usize>);

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
    Ok(Some((floats.into_pyarray(py), ids.into_pyarray(py), option, floor, streamed, im.left_out)))
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

#[pymodule]
fn _sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<Advisor>()?;
    m.add_class::<Forks>()?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(run_layout, m)?)?;
    m.add_function(wrap_pyfunction!(run_names, m)?)?;
    m.add_function(wrap_pyfunction!(start_points, m)?)?;
    m.add_function(wrap_pyfunction!(card_names, m)?)?;
    m.add_function(wrap_pyfunction!(card_ids, m)?)?;
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
