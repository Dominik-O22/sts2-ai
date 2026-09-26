//! Sim speed on played runs' fights: big decks, many relics, several enemies,
//! act 2 and 3 bosses. Random legal actions, single thread.
//!
//!     cargo run --release --example bench_real [setups.jsonl]
//!
//! Reports three numbers: steps per second over whole playouts (the setup's
//! `Combat` built outside the clock), what cloning a mid-fight state costs,
//! and what one action costs on a fresh clone (clone, legal actions, step),
//! the unit a tree search pays per node.
use std::hint::black_box;
use std::time::Instant;

use sim::gen::{run_setups, FightSetup};
use sim::replay::Ids;
use sim::rng::Rng;
use sim::Combat;

fn playouts(setups: &[&FightSetup], passes: u64) -> (u64, f64) {
    let mut steps = 0u64;
    let mut secs = 0.0;
    for pass in 0..passes {
        for (i, s) in setups.iter().enumerate() {
            let seed = pass * 100_003 + i as u64;
            let mut c = s.combat(seed);
            let mut rng = Rng::new(seed);
            let t = Instant::now();
            while !c.is_over() {
                let acts = c.legal_actions();
                c.step(acts[rng.next_int(acts.len())]);
                steps += 1;
            }
            secs += t.elapsed().as_secs_f64();
            black_box(&c);
        }
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

    for (name, set, passes) in [("all", &all, 12), ("act3+boss", &heavy, 12)] {
        let (steps, secs) = playouts(set, passes);
        println!("playouts {name:>9}: {:>9.0} steps/s ({:.0} ns/step, {steps} steps)", steps as f64 / secs, 1e9 * secs / steps as f64);
    }

    let t = Instant::now();
    let n_new = 2000;
    for i in 0..n_new {
        black_box(all[i % all.len()].combat(i as u64));
    }
    println!("Combat::new       : {:>9.0} ns", 1e9 * t.elapsed().as_secs_f64() / n_new as f64);

    let roots: Vec<Combat> = all.iter().step_by(4).enumerate().filter_map(|(i, s)| mid_fight(s, i as u64, 12)).collect();
    let reps = 3000;
    let t = Instant::now();
    for _ in 0..reps {
        for r in &roots {
            black_box(r.clone());
        }
    }
    let clone_ns = 1e9 * t.elapsed().as_secs_f64() / (reps * roots.len()) as f64;
    println!("clone             : {clone_ns:>9.0} ns ({} mid-fight states, {:.0} clones/s)", roots.len(), 1e9 / clone_ns);

    let mut rng = Rng::new(1);
    let t = Instant::now();
    for _ in 0..reps {
        for r in &roots {
            let mut c = r.clone();
            let acts = c.legal_actions();
            c.step(acts[rng.next_int(acts.len())]);
            black_box(&c);
        }
    }
    let node_ns = 1e9 * t.elapsed().as_secs_f64() / (reps * roots.len()) as f64;
    println!("clone+legal+step  : {node_ns:>9.0} ns ({:.0} nodes/s, step after clone {:.0} ns)", 1e9 / node_ns, node_ns - clone_ns);
}
