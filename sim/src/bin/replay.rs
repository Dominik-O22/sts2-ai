//! Replay recordings against the sim and report divergences.
//!
//!     cargo run --release --bin replay [FILE_OR_DIR ...]
//!
//! With no arguments, replays everything in the game's recordings folder.

use std::path::{Path, PathBuf};

use sim::replay::{replay, Ids};

fn recordings(arg: &Path) -> Vec<PathBuf> {
    if arg.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(arg)
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect())
            .unwrap_or_default();
        v.sort();
        v
    } else {
        vec![arg.to_path_buf()]
    }
}

fn main() {
    let mut args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if args.is_empty() {
        let home = std::env::var("HOME").unwrap_or_default();
        args.push(PathBuf::from(home).join(".local/share/SlayTheSpire2/sts2ai/recordings"));
    }
    let ids = Ids::new();
    let mut failed = 0;
    let mut total = 0;
    for path in args.iter().flat_map(|a| recordings(a)) {
        total += 1;
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                failed += 1;
                println!("ERR  {name}: {e}");
                continue;
            }
        };
        match replay(&text, &ids) {
            Ok(r) if r.ok() => {
                let note = if r.reseeds > 0 { format!(", {} reseeds", r.reseeds) } else { String::new() };
                println!("ok   {name}: {} decision points, {} actions{note}", r.snapshots, r.actions)
            }
            Ok(r) => {
                failed += 1;
                let (n, msg) = r.divergence.unwrap();
                println!("DIFF {name}: after {} decision points, at record {n}: {msg}", r.snapshots);
            }
            Err(e) => {
                failed += 1;
                println!("ERR  {name}: {e}");
            }
        }
    }
    println!("{} of {total} recordings replayed cleanly", total - failed);
    if failed > 0 {
        std::process::exit(1);
    }
}
