//! Walks runs the game played through the run layer (`sim::history`) and
//! prints, per run, how many floors matched on the Rewards stream before
//! the first consumer the port does not follow, and whether every room
//! matched. Runs built through the dev console, whose fights are not their
//! plan's, are listed and left out of the totals, but for the ancients'
//! options they laid out before their first console fight (Neow, mostly).
//!
//! With `--effects` it checks the effect layer too: per run, the floors
//! whose effects were compared with the record, the first that differed
//! and why, then every floor not compared, and totals of why.
//!
//! ```sh
//! cargo run --release --example runcheck -- [--effects] ~/.local/share/SlayTheSpire2/steam/*/modded/profile1/saves/history/*.run
//! ```
use std::collections::BTreeMap;

use serde_json::Value;

use sim::history::{check, eligible};

fn main() {
    let effects = std::env::args().any(|a| a == "--effects");
    let (mut checked, mut skipped) = (0, 0);
    let (mut floors_total, mut floors_matched, mut rooms_ok) = (0, 0, 0);
    let (mut compared, mut diverged) = (0, 0);
    let (mut ancients, mut ancients_differ) = (0, 0);
    let mut not_compared: BTreeMap<String, usize> = BTreeMap::new();
    for path in std::env::args().skip(1).filter(|a| a != "--effects") {
        let run: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = path.rsplit('/').next().unwrap();
        if !eligible(&run) {
            skipped += 1;
            continue;
        }
        let report = check(&run, effects);
        ancients += report.ancients;
        for m in &report.ancient_mismatches {
            println!("{name}: ancient differs, {m}");
        }
        ancients_differ += report.ancient_mismatches.len();
        if report.off_plan() {
            // Built through the dev console: its fights are not its plan's,
            // and the walk stops at the first of them.
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
        if effects {
            let e = &report.effects;
            compared += e.checked;
            diverged += e.divergences.len();
            let first = e.divergences.first().map_or("none".to_string(), |d| d.clone());
            println!("  effects: {} floors compared, {} differ; first: {first}", e.checked, e.divergences.len());
            for d in e.divergences.iter().skip(1) {
                println!("    also {d}");
            }
            for s in &e.skipped {
                println!("    not compared: {s}");
                let why = s.split_once(": ").map_or(s.as_str(), |(_, why)| why);
                *not_compared.entry(why.to_string()).or_default() += 1;
            }
        }
    }
    println!(
        "{checked} runs ({skipped} others skipped): {floors_matched} of {floors_total} floors matched before the first unported consumer; rooms match in {rooms_ok}"
    );
    println!("ancients' options: {ancients} as the record has them, {ancients_differ} differ (the dev console's runs up to their first console fight included)");
    if effects {
        let skipped: usize = not_compared.values().sum();
        println!("effects: {compared} floors compared, {diverged} differ; {skipped} not compared:");
        let mut reasons: Vec<(&String, &usize)> = not_compared.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (why, n) in reasons {
            println!("  {n:4} {why}");
        }
    }
}
