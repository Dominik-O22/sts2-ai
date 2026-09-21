//! Random-policy playouts per second. `cargo run --release --example bench`.
use sim::ids::MonsterId;
use sim::monster::Flags;
use sim::rng::Rng;
use sim::*;
use std::time::Instant;

fn main() {
    let n = 20_000u64;
    let deck = ironclad_starter_deck();
    let spec = [EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }];
    let t = Instant::now();
    let (mut wins, mut steps) = (0u64, 0u64);
    for seed in 0..n {
        let mut c = Combat::new(&deck, IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, &spec, Ascension(10), seed);
        let mut rng = Rng::new(seed);
        while !c.is_over() {
            let acts = c.legal_actions();
            c.step(acts[rng.next_int(acts.len())]);
            steps += 1;
        }
        wins += (c.outcome == Some(Outcome::Won)) as u64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "{n} fights in {dt:.2}s: {:.0} fights/s, {:.0} steps/s, random win rate {:.1}%",
        n as f64 / dt,
        steps as f64 / dt,
        100.0 * wins as f64 / n as f64
    );
}
