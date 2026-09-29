//! Diffs the run-start plan (`plan::RunPlan`) against the game's code for
//! many seeds.
//!
//! `roomcheck seeds N` prints N game-style seeds at A0 and at A10 with
//! everything unlocked, then N more with random epochs locked and bosses
//! unseen, one `SEED ASCENSION [LOCKED [UNSEEN]]` line each;
//! `tools/oracle rooms` turns those into the game's plans; `roomcheck` with
//! no arguments reads that and compares.
//!
//! ```sh
//! cargo run --release --example roomcheck seeds 500 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll rooms > rooms.txt
//! cargo run --release --example roomcheck < rooms.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-rooms.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::encounter::{Kind, ALL as ENCOUNTERS};
use sim::game_rng::GameRng;
use sim::plan::{diff_oracle, Epoch};
use sim::replay::slug;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "seeds", "usage: roomcheck [seeds N]");
        let n: u32 = n.parse().expect("a seed count");
        let mut rng = GameRng::new(0x5EED);
        let seed = |rng: &mut GameRng| (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect::<String>();
        for ascension in [0, 10] {
            for _ in 0..n {
                println!("{} {ascension}", seed(&mut rng));
            }
        }
        let bosses: Vec<String> =
            ENCOUNTERS.iter().filter(|e| e.kind() == Kind::Boss).map(|e| slug(&format!("{e:?}"))).collect();
        for _ in 0..n {
            let s = seed(&mut rng);
            let some = |rng: &mut GameRng, ids: &[String]| {
                let picked: Vec<&str> = ids.iter().filter(|_| rng.next_int(3) == 0).map(String::as_str).collect();
                if picked.is_empty() { "-".to_string() } else { picked.join(",") }
            };
            let epochs: Vec<String> = Epoch::ALL.iter().map(|e| e.id()).collect();
            let locked = some(&mut rng, &epochs);
            let unseen = some(&mut rng, &bosses);
            println!("{s} {} {locked} {unseen}", rng.next_int(11));
        }
        return;
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let (runs, mismatches) = diff_oracle(&text);
    for m in &mismatches {
        println!("{m}");
    }
    println!("{} of {runs} plans match the game", runs - mismatches.len());
    if !mismatches.is_empty() {
        std::process::exit(1);
    }
}
