//! Sim speed on played runs' fights: big decks, many relics, several enemies,
//! act 2 and 3 bosses. Random legal actions, single thread.
//!
//!     cargo run --release --example bench_real [setups.jsonl]
//!
//! Reports three numbers: steps per second over whole playouts (the setup's
//! `Combat` built outside the clock), what cloning a mid-fight state costs,
//! and what one action costs on a fresh clone (clone, legal actions, step),
//! the unit a tree search pays per node. Time is the thread's time on a
//! CPU (`/proc/thread-self/schedstat`), and each number is the best of five
//! rounds, which keeps a busy machine's preemptions out of the number.
use std::hint::black_box;
use std::time::Instant;

use sim::gen::{run_setups, FightSetup};
use sim::replay::Ids;
use sim::rng::Rng;
use sim::Combat;

const ROUNDS: usize = 5;

/// Seconds this thread has spent on a CPU, or wall time where Linux's
/// schedstat is missing.
fn cpu_secs() -> f64 {
    thread_local!(static START: Instant = Instant::now());
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
        .map_or_else(|| START.with(|t| t.elapsed().as_secs_f64()), |ns| ns as f64 * 1e-9)
}

/// The round with the best rate: units done over seconds taken.
fn best(mut round: impl FnMut(usize) -> (u64, f64)) -> (u64, f64) {
    (0..ROUNDS).map(&mut round).max_by(|a, b| (a.0 as f64 / a.1).total_cmp(&(b.0 as f64 / b.1))).unwrap()
}

fn playouts(setups: &[&FightSetup], passes: std::ops::Range<u64>) -> (u64, f64) {
    let mut steps = 0u64;
    let mut secs = 0.0;
    for pass in passes {
        let mut fights: Vec<(Combat, Rng)> = setups
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let seed = pass * 100_003 + i as u64;
                (s.combat(seed), Rng::new(seed))
            })
            .collect();
        let t = cpu_secs();
        for (c, rng) in &mut fights {
            while !c.is_over() {
                let acts = c.legal_actions();
                c.step(acts[rng.next_int(acts.len())]);
                steps += 1;
            }
        }
        secs += cpu_secs() - t;
        black_box(&fights);
    }
    (steps, secs)
}

/// A state partway into the fight: `n` random actions in, or the last one
/// before it ends.
fn mid_fight(s: &FightSetup, seed: u64, n: usize) -> Option<Combat> {
    let mut c = s.combat(seed);
    let mut rng = Rng::new(seed);
    for _ in 0..n {
        let acts = c.legal_actions();
        let mut next = c.clone();
        next.step(acts[rng.next_int(acts.len())]);
        if next.is_over() {
            break;
        }
        c = next;
    }
    (!c.is_over()).then_some(c)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.local/share/SlayTheSpire2/sts2ai/tracker/setups/holdout.jsonl")
    });
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let setups = run_setups(&text, &Ids::new(), 3).expect("setups");
    let all: Vec<&FightSetup> = setups.iter().collect();
    // Act 3 and every boss: the biggest decks and the busiest fights.
    let heavy: Vec<&FightSetup> =
        setups.iter().filter(|s| s.floor > 2 * sim::gen::BOSS_FLOOR || s.room == sim::RoomKind::Boss).collect();
    let deck = all.iter().map(|s| s.deck.len()).sum::<usize>() as f64 / all.len() as f64;
    let relics = all.iter().map(|s| s.relics.len()).sum::<usize>() as f64 / all.len() as f64;
    println!("{} fights ({} act 3 or boss), {deck:.1} cards and {relics:.1} relics on average", all.len(), heavy.len());

    for (name, set) in [("all", &all), ("act3+boss", &heavy)] {
        let (steps, secs) = best(|r| playouts(set, 3 * r as u64..3 * r as u64 + 3));
        println!("playouts {name:>9}: {:>9.0} steps/s ({:.0} ns/step, {steps} steps)", steps as f64 / secs, 1e9 * secs / steps as f64);
    }

    let t = cpu_secs();
    let n_new = 2000;
    for i in 0..n_new {
        black_box(all[i % all.len()].combat(i as u64));
    }
    println!("Combat::new       : {:>9.0} ns", 1e9 * (cpu_secs() - t) / n_new as f64);

    let roots: Vec<Combat> = all.iter().step_by(4).enumerate().filter_map(|(i, s)| mid_fight(s, i as u64, 12)).collect();
    let reps = 600;
    let (n, secs) = best(|_| {
        let t = cpu_secs();
        for _ in 0..reps {
            for r in &roots {
                black_box(r.clone());
            }
        }
        ((reps * roots.len()) as u64, cpu_secs() - t)
    });
    let clone_ns = 1e9 * secs / n as f64;
    println!("clone             : {clone_ns:>9.0} ns ({} mid-fight states, {:.0} clones/s)", roots.len(), 1e9 / clone_ns);

    let (n, secs) = best(|_| {
        let mut rng = Rng::new(1);
        let t = cpu_secs();
        for _ in 0..reps {
            for r in &roots {
                let mut c = r.clone();
                let acts = c.legal_actions();
                c.step(acts[rng.next_int(acts.len())]);
                black_box(&c);
            }
        }
        ((reps * roots.len()) as u64, cpu_secs() - t)
    });
    let node_ns = 1e9 * secs / n as f64;
    println!("clone+legal+step  : {node_ns:>9.0} ns ({:.0} nodes/s, step after clone {:.0} ns)", 1e9 / node_ns, node_ns - clone_ns);

    // A turn search's inner node: a card played, not the turn ended.
    let (n, secs) = best(|_| {
        let mut rng = Rng::new(1);
        let mut n = 0;
        let t = cpu_secs();
        for _ in 0..reps {
            for r in &roots {
                let mut c = r.clone();
                let mut acts = c.legal_actions();
                acts.retain(|a| matches!(a, sim::Action::PlayCard { .. }));
                if let Some(&a) = acts.get(rng.next_int(acts.len().max(1))) {
                    c.step(a);
                    n += 1;
                }
                black_box(&c);
            }
        }
        (n, cpu_secs() - t)
    });
    let play_ns = 1e9 * secs / n as f64;
    println!("clone+legal+play  : {play_ns:>9.0} ns ({:.0} nodes/s, card plays only)", 1e9 / play_ns);

    // A search that keeps one scratch state per thread and clones each node
    // into it (`clone_from`) reuses the scratch's buffers.
    let mut scratch = roots[0].clone();
    let (n, secs) = best(|_| {
        let t = cpu_secs();
        for _ in 0..reps {
            for r in &roots {
                scratch.clone_from(r);
                black_box(&scratch);
            }
        }
        ((reps * roots.len()) as u64, cpu_secs() - t)
    });
    println!("clone_from        : {:>9.0} ns", 1e9 * secs / n as f64);
    let (n, secs) = best(|_| {
        let mut rng = Rng::new(1);
        let mut acts = vec![];
        let mut n = 0;
        let t = cpu_secs();
        for _ in 0..reps {
            for r in &roots {
                scratch.clone_from(r);
                acts.clear();
                acts.extend(scratch.legal_actions().into_iter().filter(|a| matches!(a, sim::Action::PlayCard { .. })));
                if let Some(&a) = acts.get(rng.next_int(acts.len().max(1))) {
                    scratch.step(a);
                    n += 1;
                }
                black_box(&scratch);
            }
        }
        (n, cpu_secs() - t)
    });
    println!("clone_from+play   : {:>9.0} ns (card plays only)", 1e9 * secs / n as f64);

    // The same nodes on four threads expanding the same parents at once, as
    // a parallel search does: anything the clones share gets contended.
    let threads = 4;
    let (n, secs) = best(|_| {
        let per: Vec<(u64, f64)> = std::thread::scope(|sc| {
            let roots = &roots;
            let handles: Vec<_> = (0..threads)
                .map(|k| {
                    sc.spawn(move || {
                        let mut rng = Rng::new(k as u64);
                        let mut n = 0;
                        let t = cpu_secs();
                        for _ in 0..reps / 2 {
                            for r in roots {
                                let mut c = r.clone();
                                let mut acts = c.legal_actions();
                                acts.retain(|a| matches!(a, sim::Action::PlayCard { .. }));
                                if let Some(&a) = acts.get(rng.next_int(acts.len().max(1))) {
                                    c.step(a);
                                    n += 1;
                                }
                                black_box(&c);
                            }
                        }
                        (n, cpu_secs() - t)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        (per.iter().map(|p| p.0).sum(), per.iter().map(|p| p.1).sum())
    });
    let mt_ns = 1e9 * secs / n as f64;
    println!("4 threads, play   : {mt_ns:>9.0} ns per node per thread (CPU time)");
}
