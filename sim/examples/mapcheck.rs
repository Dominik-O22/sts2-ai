//! Diffs the map port against the game's code for many seeds, and times it.
//!
//! `mapcheck seeds N` prints N game-style seeds for every act at A0 and A10,
//! one `SEED ACT ASCENSION` line each; `tools/oracle maps` turns those into
//! the game's maps; `mapcheck` with no arguments reads that and compares.
//! `mapcheck bench N` generates the maps for the same seeds and prints the
//! time each act takes.
//!
//! ```sh
//! cargo run --release --example mapcheck seeds 250 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll maps > maps.txt
//! cargo run --release --example mapcheck < maps.txt
//! cargo run --release --example mapcheck bench 250
//! ```
//!
//! A slice of that output is `testdata/oracle-maps.txt`, which the tests
//! check without dotnet.
use std::io::Read;
use std::time::Instant;

use sim::game_rng::{hash, GameRng};
use sim::map::{act_named, diff_oracle, ActMap};
use sim::Ascension;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

const ACTS: [&str; 4] = ["Overgrowth", "Underdocks", "Hive", "Glory"];

/// `n` seeds for every act at A0 and A10, the same ones on every call.
fn seeds(n: u32) -> Vec<(String, &'static str, u8)> {
    let mut rng = GameRng::new(0x5EED);
    let mut out = Vec::new();
    for ascension in [0, 10] {
        for act in ACTS {
            for _ in 0..n {
                let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
                out.push((seed, act, ascension));
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match &args[..] {
        [cmd, n] if cmd == "seeds" => {
            for (seed, act, ascension) in seeds(n.parse().expect("a seed count")) {
                println!("{seed} {act} {ascension}");
            }
        }
        [cmd, n] if cmd == "bench" => bench(n.parse().expect("a seed count")),
        [] => {
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text).unwrap();
            let (maps, mismatches) = diff_oracle(&text);
            for m in &mismatches {
                println!("{m}");
            }
            println!("{} of {maps} maps match the game", maps - mismatches.len());
            if !mismatches.is_empty() {
                std::process::exit(1);
            }
        }
        _ => panic!("usage: mapcheck [seeds N | bench N]"),
    }
}

/// Mean and slowest `ActMap::generate` per act, both ascensions together.
fn bench(n: u32) {
    let seeds = seeds(n);
    let mut total = 0.0;
    for act_name in ACTS {
        let act = act_named(act_name).unwrap();
        let times: Vec<f64> = seeds
            .iter()
            .filter(|(_, a, _)| *a == act_name)
            .map(|(seed, _, ascension)| {
                let t = Instant::now();
                std::hint::black_box(ActMap::generate(hash(seed) as u32, act, Ascension(*ascension)));
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        let sum: f64 = times.iter().sum();
        total += sum;
        let max = times.iter().cloned().fold(0.0, f64::max);
        println!("{act_name:>10}: {:.3} ms mean, {max:.3} ms max over {} maps", sum / times.len() as f64, times.len());
    }
    println!("{:>10}: {:.3} ms mean", "all", total / seeds.len() as f64);
}
