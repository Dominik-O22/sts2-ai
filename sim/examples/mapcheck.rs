//! Diffs the map port against the game's code for many seeds.
//!
//! `mapcheck seeds N` prints N game-style seeds for every act at A0 and A10,
//! one `SEED ACT ASCENSION` line each; `tools/oracle maps` turns those into
//! the game's maps; `mapcheck` with no arguments reads that and compares.
//!
//! ```sh
//! cargo run --release --example mapcheck seeds 250 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll maps > maps.txt
//! cargo run --release --example mapcheck < maps.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-maps.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::game_rng::GameRng;
use sim::map::diff_oracle;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "seeds", "usage: mapcheck [seeds N]");
        let n: u32 = n.parse().expect("a seed count");
        let mut rng = GameRng::new(0x5EED);
        for ascension in [0, 10] {
            for act in ["Overgrowth", "Underdocks", "Hive", "Glory"] {
                for _ in 0..n {
                    let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
                    println!("{seed} {act} {ascension}");
                }
            }
        }
        return;
    }
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
