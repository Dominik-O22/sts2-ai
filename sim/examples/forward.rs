//! Plays seeded runs forward (`sim::forward`) with every fight a stub win
//! at 70% HP and the first option always taken, and prints how each ended,
//! the deck and relics it finished with, then what the runs met that the
//! port lacks, and the time a run takes.
//!
//! ```sh
//! cargo run --release --example forward -- [RUNS] [ASCENSION]
//! ```
use std::collections::BTreeMap;
use std::time::Instant;

use sim::forward::{play, stub_fight};
use sim::rooms::First;
use sim::types::Ascension;

fn main() {
    let mut args = std::env::args().skip(1);
    let runs: usize = args.next().map_or(20, |a| a.parse().unwrap());
    let ascension = Ascension(args.next().map_or(10, |a| a.parse().unwrap()));
    let mut unported: BTreeMap<String, usize> = BTreeMap::new();
    let start = Instant::now();
    for i in 0..runs {
        let seed = format!("FWD{i:06}");
        let run = play(&seed, ascension, &mut First, &mut stub_fight);
        let s = &run.state;
        println!(
            "{seed}: {:?} on floor {}, {} fights, hp {}/{}, gold {}, deck {}, relics {}, potions {}",
            run.end,
            s.floor,
            run.fights,
            s.hp,
            s.max_hp,
            s.gold,
            s.deck.len(),
            s.relics.len(),
            s.held_potions().count()
        );
        for (what, n) in run.unported {
            *unported.entry(what).or_default() += n;
        }
    }
    let per_run = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;
    println!("{runs} runs, {per_run:.2} ms a run; not ported, times met:");
    let mut met: Vec<(String, usize)> = unported.into_iter().collect();
    met.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (what, n) in met {
        println!("  {n:5} {what}");
    }
}
