//! Python bindings for the training environment (DESIGN.md, Training).
//! One class, `VecEnv`, plus the layout constants the model needs. Buffers
//! are numpy arrays the caller allocates once; `observe` and `step` fill
//! them in place with the GIL released.

use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim::encode::*;
use sim::env::{EnvConfig, VecEnv as Inner};
use sim::gen::{load_recordings, BOSS_FLOOR};
use sim::ids::{ALL_CARDS, ALL_MONSTERS};
use sim::types::Ascension;

/// A batch of combats. See `sim::env::VecEnv`.
#[pyclass]
struct VecEnv {
    inner: Inner,
}

/// One finished fight: (env, won, hp_frac, hp_lost, steps, floor, encounter, reward).
type End = (usize, bool, f32, f32, u32, u32, String, f32);

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed=0, asc=10, min_floor=1, max_floor=BOSS_FLOOR, max_steps=500))]
    fn new(n: usize, seed: u64, asc: u8, min_floor: u32, max_floor: u32, max_steps: u32) -> Self {
        let cfg = EnvConfig { asc: Ascension(asc), min_floor, max_floor, max_steps };
        Self { inner: Inner::new(n, seed, cfg) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Curriculum: floors generated fights are drawn from.
    fn set_floors(&mut self, min: u32, max: u32) {
        self.inner.set_floors(min, max);
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
            .map(|e| (e.env, e.won, e.hp_frac, e.hp_lost, e.steps, e.floor, format!("{:?}", e.encounter), e.reward))
            .collect())
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

#[pymodule]
fn _sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(card_names, m)?)?;
    m.add_function(wrap_pyfunction!(monster_names, m)?)?;
    Ok(())
}
