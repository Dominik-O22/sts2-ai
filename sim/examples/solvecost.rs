//! What one solver state costs on real fights: clone, step, keys.
//!
//!     cargo run --release --example solvecost -- setups.jsonl
use std::hint::black_box;
use std::time::Instant;

use sim::gen::run_setups;
use sim::replay::Ids;

fn main() {
    let path = std::env::args().nth(1).expect("setups");
    let fights = run_setups(&std::fs::read_to_string(path).unwrap(), &Ids::new(), 1).unwrap();
    let mut states = vec![];
    for (f, setup) in fights.iter().enumerate().take(50) {
        let mut c = setup.combat(f as u64);
        for k in 0..30 {
            if c.is_over() {
                break;
            }
            states.push(c.clone());
            let acts = c.legal_actions();
            c.step(acts[k % acts.len()]);
        }
    }
    let n = states.len() as f64;
    let t = Instant::now();
    for c in &states {
        black_box(c.clone());
    }
    let clone = t.elapsed().as_secs_f64() / n;
    let t = Instant::now();
    for c in &states {
        let mut k = c.clone();
        let a = k.legal_actions()[0];
        k.step(a);
        black_box(k);
    }
    let step = t.elapsed().as_secs_f64() / n - clone;
    let t = Instant::now();
    for c in &states {
        black_box(sim::turnsearch::state_key(c));
    }
    let key = t.elapsed().as_secs_f64() / n;
    let t = Instant::now();
    for c in &states {
        black_box(format!("{:?}", c.rngs));
    }
    let rngs = t.elapsed().as_secs_f64() / n;
    let part = |name: &str, f: &dyn Fn(&sim::Combat) -> String| {
        let t = Instant::now();
        let mut len = 0;
        for c in &states {
            len += black_box(f(c)).len();
        }
        println!("  {name}: {:.2} us, {} bytes of Debug a state", t.elapsed().as_secs_f64() / n * 1e6, len / states.len());
    };
    part("player creature", &|c| format!("{:?}", c.player.creature));
    part("enemies", &|c| c.enemies.iter().map(|e| format!("{:?}{:?}{:?}{:?}{:?}", e.creature, e.monster.id, e.monster.flags, e.monster.vars, e.monster.next_move)).collect());
    part("order/side/outcome/relics/potions/after/room", &|c| format!("{:?}{:?}{:?}{:?}{:?}{:?}{:?}", c.order, c.side, c.outcome, c.relics, c.potions, c.after, c.room));
    part("stats", &|c| format!("{:?}", c.stats));
    part("card (one, hand[0])", &|c| c.player.hand.first().map_or(String::new(), |k| format!("{k:?}")));
    println!("{} states: clone {:.2} us, step {:.2} us, state_key {:.2} us, rngs Debug {:.2} us", states.len(), clone * 1e6, step * 1e6, key * 1e6, rngs * 1e6);
}
