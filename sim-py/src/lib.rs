//! Python bindings for the training environment (DESIGN.md, Training).
//! Two classes: `VecEnv` for training, `Advisor` for following a real
//! combat from the recorder's log. Plus the layout constants the model needs. Buffers
//! are numpy arrays the caller allocates once; `observe` and `step` fill
//! them in place with the GIL released.

use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use sim::encode::*;
use sim::env::{EnvConfig, Forks as InnerForks, VecEnv as Inner};
use sim::gen::{holdout, load_recordings, ACTS, LAST_FLOOR};
use sim::ids::{ALL_CARDS, ALL_MONSTERS};
use sim::replay::{command, Ids, Replayer, Step};
use sim::types::Ascension;

/// A batch of combats. See `sim::env::VecEnv`.
#[pyclass]
struct VecEnv {
    inner: Inner,
}

/// One finished fight: (env, won, hp_frac, hp_lost, potions_used, steps, floor, encounter, kind, reward).
type End = (usize, bool, f32, f32, u32, u32, u32, String, String, f32);

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
    #[pyo3(signature = (envs, n, groups=4, seed=0))]
    fn fork(&self, envs: Vec<usize>, n: usize, groups: usize, seed: u64) -> Forks {
        let roots: Vec<_> = envs.iter().map(|&i| self.inner.combat(i)).collect();
        Forks { inner: InnerForks::of(&roots, n, groups, seed) }
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

    /// Switch to cycling through the recordings in `dir` (the held-out
    /// set). Returns the number loaded and the files that failed to parse.
    fn load_recordings(&mut self, dir: &str) -> PyResult<(usize, Vec<String>)> {
        let (setups, errors) =
            load_recordings(std::path::Path::new(dir)).map_err(pyo3::exceptions::PyIOError::new_err)?;
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
            .map(|e| (e.env, e.won, e.hp_frac, e.hp_lost, e.potions_used, e.steps, e.floor, format!("{:?}", e.encounter), format!("{:?}", e.kind), e.reward))
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
            "start" => {
                self.start = Some(rec);
                self.inner = None;
                return Ok("ok".into());
            }
            // The first snapshot carries the opening draw order and the
            // player's HP, so the combat is built from it and the start.
            "snapshot" if self.inner.is_none() => {
                let Some(start) = self.start.clone() else { return Ok("waiting".into()) };
                match Replayer::new(&start, &rec, self.ids.clone(), self.seed) {
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

    /// Encode forks `rows` into the first `len(rows)` rows of the buffers.
    fn observe_rows(
        &self,
        py: Python<'_>,
        rows: Vec<usize>,
        mut floats: PyReadwriteArray2<f32>,
        mut ids: PyReadwriteArray2<i64>,
        mut mask: PyReadwriteArray2<bool>,
    ) -> PyResult<()> {
        let k = rows.len();
        let f = &mut floats.as_slice_mut()?[..k * N_FLOATS];
        let i = &mut ids.as_slice_mut()?[..k * N_IDS];
        let m = &mut mask.as_slice_mut()?[..k * N_ACTIONS];
        py.detach(|| self.inner.observe_rows(&rows, f, i, m));
        Ok(())
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

/// Card names by vocabulary index (index 0 is the pad).
#[pyfunction]
fn card_names() -> Vec<String> {
    std::iter::once("<pad>".to_string()).chain(ALL_CARDS.iter().map(|c| format!("{c:?}"))).collect()
}

#[pyfunction]
fn monster_names() -> Vec<String> {
    std::iter::once("<pad>".to_string()).chain(ALL_MONSTERS.iter().map(|c| format!("{c:?}"))).collect()
}

/// Every act 1 encounter as (id, act, kind), in the sim's own order. The
/// recording wizard walks this so it cannot drift from `encounter.rs`.
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
    // Only `hp` and `max_hp` are read from the snapshot, so an empty one does.
    Ok(sim::gen::FightSetup::from_start(&v, &serde_json::json!({}), &Ids::new()).err())
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
    m.add_function(wrap_pyfunction!(card_names, m)?)?;
    m.add_function(wrap_pyfunction!(monster_names, m)?)?;
    m.add_function(wrap_pyfunction!(encounters, m)?)?;
    m.add_function(wrap_pyfunction!(start_blocker, m)?)?;
    m.add_function(wrap_pyfunction!(encounter_brief, m)?)?;
    Ok(())
}
