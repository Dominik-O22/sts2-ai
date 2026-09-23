//! Generated fights in the recorder's `start` format, one JSON line each, so
//! `scripts/deckstats.py --gen` can set them beside real runs.
//! `cargo run --release --example gendump [fights_per_floor] > gen.jsonl`
use sim::gen::{generate, LAST_FLOOR};
use sim::rng::Rng;
use sim::types::Ascension;

fn main() {
    let per_floor: u32 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let mut rng = Rng::new(7);
    for floor in 1..=LAST_FLOOR {
        for _ in 0..per_floor {
            println!("{}", generate(&mut rng, floor, Ascension(10)).run_json());
        }
    }
}
