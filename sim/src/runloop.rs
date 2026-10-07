//! The threaded run loop: worker threads step a batch's run-mode slots
//! (`env::Slot`) each on its own, and the caller (Python's inference
//! loop) takes the envs that wait for the network, answers them, and
//! posts the answers back. Nothing waits for a whole batch: an env that
//! needs an action goes to the ready set as soon as it does, and a worker
//! moves on to the next env with an answer. The caller's forward and the
//! workers' stepping overlap, which the batch loop (`VecEnv::step`) could
//! not do.
//!
//! An env is in exactly one place: posted (in `work`, or in a worker's
//! hands), ready (in `Ready`, waiting to be taken), taken (the caller
//! holds it), or done (out of seeds). `take` blocks until enough envs are
//! ready, or none is posted, so the caller never waits on an env it must
//! answer itself.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::encode::{self, N_ACTIONS, N_FLOATS, N_IDS};
use crate::env::{Answer, EnvConfig, Events, Phase, Slot};
use crate::runobs::{RUN_FLOATS, RUN_IDS};

/// The envs ready for the network and what happened on the way.
#[derive(Default)]
pub struct Ready {
    /// Envs at a combat decision, and envs at a run decision.
    pub combat: Vec<usize>,
    pub decision: Vec<usize>,
    pub events: Events,
}

struct Shared {
    slots: Arc<Vec<Mutex<Slot>>>,
    cfg: EnvConfig,
    /// Envs with an answer to apply, oldest first.
    work: Mutex<VecDeque<(usize, Answer)>>,
    work_cv: Condvar,
    ready: Mutex<Ready>,
    ready_cv: Condvar,
    /// Envs posted and not yet ready or done.
    posted: AtomicUsize,
    /// Envs out of seeds, for good.
    done: AtomicUsize,
    /// Combat steps taken, lone legal actions included.
    steps: AtomicU64,
    /// Run decisions answered.
    decisions: AtomicU64,
    stop: AtomicBool,
}

/// A running loop; dropping it stops the workers.
pub struct Loop {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// What `take` hands the caller: the envs whose rows it packed, in row
/// order, and the events since the last take.
pub struct Taken {
    pub combat: Vec<usize>,
    pub decision: Vec<usize>,
    pub events: Events,
    /// Envs still in the batch (not done) after this take.
    pub active: usize,
}

impl Loop {
    /// Starts `workers` threads over `slots`, every env posted with no
    /// answer so each lands in the ready set where it stands.
    pub fn start(slots: Arc<Vec<Mutex<Slot>>>, cfg: EnvConfig, workers: usize) -> Self {
        let n = slots.len();
        let shared = Arc::new(Shared {
            slots,
            cfg,
            work: Mutex::new((0..n).map(|i| (i, Answer::None)).collect()),
            work_cv: Condvar::new(),
            ready: Mutex::new(Ready::default()),
            ready_cv: Condvar::new(),
            posted: AtomicUsize::new(n),
            done: AtomicUsize::new(0),
            steps: AtomicU64::new(0),
            decisions: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        });
        let workers = (0..workers.max(1))
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || work(&shared))
            })
            .collect();
        Self { shared, workers }
    }

    /// Posts answers: `actions[k]` for `combat[k]`, `options[k]` for
    /// `decision[k]`. Each env must be one the caller holds.
    pub fn post(&self, combat: &[usize], actions: &[i64], decision: &[usize], options: &[i64]) {
        assert!(combat.len() == actions.len() && decision.len() == options.len(), "an answer per env");
        let s = &self.shared;
        let mut work = s.work.lock().expect("work");
        work.extend(combat.iter().zip(actions).map(|(&e, &a)| (e, Answer::Act(a))));
        work.extend(decision.iter().zip(options).map(|(&e, &o)| (e, Answer::Option(o as usize))));
        s.posted.fetch_add(combat.len() + decision.len(), Ordering::SeqCst);
        s.decisions.fetch_add(decision.len() as u64, Ordering::Relaxed);
        drop(work);
        s.work_cv.notify_all();
    }

    /// Waits until at least `min_rows` envs are ready, or no env is posted
    /// (nothing more will come until the caller posts), or `timeout`
    /// passes; then packs the ready envs' rows: combat rows into `floats`,
    /// `ids`, `mask` (row k is `combat[k]`), run rows into `run_floats`,
    /// `run_ids` (row k is `decision[k]`). The buffers hold a row per env.
    pub fn take(&self, min_rows: usize, timeout: Duration, floats: &mut [f32], ids: &mut [i64], mask: &mut [bool], run_floats: &mut [f32], run_ids: &mut [i64]) -> Taken {
        let s = &self.shared;
        let n = s.slots.len();
        assert!(floats.len() >= n * N_FLOATS && ids.len() >= n * N_IDS && mask.len() >= n * N_ACTIONS, "combat buffers hold a row per env");
        assert!(run_floats.len() >= n * RUN_FLOATS && run_ids.len() >= n * RUN_IDS, "run buffers hold a row per env");
        let deadline = Instant::now() + timeout;
        let mut ready = s.ready.lock().expect("ready");
        loop {
            let have = ready.combat.len() + ready.decision.len();
            if have >= min_rows.max(1) || s.posted.load(Ordering::SeqCst) == 0 {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            ready = s.ready_cv.wait_timeout(ready, deadline - now).expect("ready").0;
        }
        let taken = std::mem::take(&mut *ready);
        drop(ready);
        let slots = &s.slots;
        floats.par_chunks_mut(N_FLOATS).zip(ids.par_chunks_mut(N_IDS)).zip(mask.par_chunks_mut(N_ACTIONS)).zip(taken.combat.par_iter()).for_each(
            |(((f, i), m), &env)| {
                let slot = slots[env].lock().expect("slot");
                encode::encode(slot.combat(), f, i, m);
            },
        );
        run_floats.par_chunks_mut(RUN_FLOATS).zip(run_ids.par_chunks_mut(RUN_IDS)).zip(taken.decision.par_iter()).for_each(|((f, i), &env)| {
            let slot = slots[env].lock().expect("slot");
            let obs = slot.run_obs().unwrap_or_else(|| panic!("env {env}: not at a run decision"));
            f.copy_from_slice(&obs.floats);
            i.copy_from_slice(&obs.ids);
        });
        Taken { combat: taken.combat, decision: taken.decision, events: taken.events, active: n - s.done.load(Ordering::SeqCst) }
    }

    /// Combat steps taken and run decisions answered so far.
    pub fn counts(&self) -> (u64, u64) {
        (self.shared.steps.load(Ordering::Relaxed), self.shared.decisions.load(Ordering::Relaxed))
    }
}

impl Drop for Loop {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.work_cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A worker: takes posted envs and plays each on to where it needs the
/// caller, then hands it to the ready set with what happened.
fn work(s: &Shared) {
    let n = s.slots.len();
    let mut legal = Vec::with_capacity(32);
    loop {
        let (env, answer) = {
            let mut work = s.work.lock().expect("work");
            loop {
                if s.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(item) = work.pop_front() {
                    break item;
                }
                work = s.work_cv.wait(work).expect("work");
            }
        };
        let mut events = Events::default();
        let (phase, steps) = {
            let mut slot = s.slots[env].lock().expect("slot");
            let out = slot.play_on(answer, env, n, &s.cfg, &mut legal, &mut events);
            slot.take_logs(&mut events);
            out
        };
        s.steps.fetch_add(steps as u64, Ordering::Relaxed);
        let mut ready = s.ready.lock().expect("ready");
        match phase {
            Phase::Combat => ready.combat.push(env),
            Phase::Decision => ready.decision.push(env),
            Phase::Done => {
                s.done.fetch_add(1, Ordering::SeqCst);
            }
        }
        ready.events.append(&mut events);
        s.posted.fetch_sub(1, Ordering::SeqCst);
        drop(ready);
        s.ready_cv.notify_all();
    }
}
