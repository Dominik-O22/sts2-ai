//! Diffs rewards, shops and unknown rooms (`rewards`, `shop`, `run`)
//! against the game's code for many seeds.
//!
//! `rewardcheck walks N` prints N random walks, one `SEED ASCENSION
//! STEP...` line each, through all three acts at A0, A3, A6, A7 and A10:
//! fights, shops and unknown points; `tools/oracle rewards` turns those
//! into what the game offered; `rewardcheck` with no arguments reads that
//! and compares.
//!
//! ```sh
//! cargo run --release --example rewardcheck walks 500 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll rewards > rewards.txt
//! cargo run --release --example rewardcheck < rewards.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-rewards.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::game_rng::GameRng;
use sim::rewards::diff_oracle;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "walks", "usage: rewardcheck [walks N]");
        let n: usize = n.parse().expect("a walk count");
        let mut rng = GameRng::new(0x5EED);
        for i in 0..n {
            let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
            let ascension = [0, 3, 6, 7, 10][i % 5];
            let mut steps = Vec::new();
            for act in 0..3 {
                if act > 0 {
                    steps.push("R".to_string());
                }
                for _ in 0..rng.next_int_in(8, 17) {
                    let step = *rng.pick(&["M", "M", "M", "M", "M", "E", "E", "S", "?", "?", "?s"]).unwrap();
                    steps.push(if step.starts_with('?') { step.to_string() } else { format!("{step}{act}") });
                }
                steps.push(format!("B{act}"));
            }
            println!("{seed} {ascension} {}", steps.join(" "));
        }
        return;
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let (runs, mismatches) = diff_oracle(&text);
    for m in &mismatches {
        println!("{m}");
    }
    println!("{} of {runs} walks match", runs - mismatches.len());
}
