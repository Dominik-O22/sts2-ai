//! Walks runs the game played through the run layer (`sim::history`) and
//! prints, per run, how many floors matched on the Rewards stream before
//! the first consumer the port does not follow, and whether every room
//! matched. Runs built through the dev console, whose fights are not their
//! plan's, are listed and left out of the totals.
//!
//! ```sh
//! cargo run --release --example runcheck -- ~/.local/share/SlayTheSpire2/steam/*/modded/profile1/saves/history/*.run
//! ```
use serde_json::Value;

use sim::history::{check, eligible};

fn main() {
    let (mut checked, mut skipped) = (0, 0);
    let (mut floors_total, mut floors_matched, mut rooms_ok) = (0, 0, 0);
    for path in std::env::args().skip(1) {
        let run: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = path.rsplit('/').next().unwrap();
        if !eligible(&run) {
            skipped += 1;
            continue;
        }
        let report = check(&run);
        if report.off_plan() {
            // Built through the dev console: its fights are not its plan's.
            println!("{name} {}: not its plan's run ({})", run["seed"].as_str().unwrap(), report.room_problem.unwrap());
            skipped += 1;
            continue;
        }
        checked += 1;
        floors_total += report.floors;
        floors_matched += report.matched;
        rooms_ok += report.room_problem.is_none() as usize;
        let rooms = report.room_problem.as_deref().unwrap_or("all match");
        println!(
            "{name} {} A{}: rewards {}/{} floors, then {}; rooms {rooms}",
            run["seed"].as_str().unwrap(),
            run["ascension"],
            report.matched,
            report.floors,
            report.stop
        );
    }
    println!(
        "{checked} runs ({skipped} others skipped): {floors_matched} of {floors_total} floors matched before the first unported consumer; rooms match in {rooms_ok}"
    );
}

