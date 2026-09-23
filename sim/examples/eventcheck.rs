//! Diffs the events (`sim::events`) against the game's code for many seeds.
//!
//! `eventcheck lines N` prints N `tools/oracle events` lines: every ported
//! event in the acts it can meet, with decks, gold, HP, relics and potions
//! that turn each option's condition on and off, and a random walk through
//! its pages; `tools/oracle events` turns those into what the game did;
//! `eventcheck` with no arguments reads that and compares.
//!
//! ```sh
//! cargo run --release --example eventcheck lines 2000 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll events > events.txt
//! cargo run --release --example eventcheck < events.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-events.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::events::{diff_oracle, ported};
use sim::game_rng::GameRng;
use sim::replay::slug;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// Setup that moves the events' conditions and outcomes: cards of every
/// type in and out, upgraded ones, curses, gold, HP, relics that change
/// what an effect does, potions of every rarity.
const OPS: &[&str] = &[
    "-BASH", "-STRIKE_IRONCLAD", "-DEFEND_IRONCLAD", "+POMMEL_STRIKE", "+SHRUG_IT_OFF+", "+INFLAME", "+BASH+", "+IRON_WAVE+",
    "+TRUE_GRIT", "+DEMON_FORM", "+CLUMSY", "+INJURY", "+ANGER", "+HAVOC", "gold=0", "gold=60", "gold=130", "gold=320",
    "hp=9", "hp=20", "hp=45", "relic=TUNGSTEN_ROD", "relic=MOLTEN_EGG", "relic=TOXIC_EGG", "relic=BOWLER_HAT",
    "relic=LUCKY_FYSH", "relic=SILVER_CRUCIBLE", "relic=ANCHOR", "relic=VAJRA", "relic=BAG_OF_MARBLES",
    "relic=HAPPY_FLOWER", "relic=BRONZE_SCALES", "relic=REGAL_PILLOW", "potion=FIRE_POTION", "potion=FOUL_POTION",
    "potion=BLOCK_POTION", "potion=FAIRY_IN_A_BOTTLE", "potion=POTION_SHAPED_ROCK", "potion=GLOWWATER_POTION",
    "potion=DUPLICATOR", "potion=REGEN_POTION",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "lines", "usage: eventcheck [lines N]");
        let n: usize = n.parse().expect("a line count");
        let mut rng = GameRng::new(0xE7E7);
        let events: Vec<&str> = ported().collect();
        for i in 0..n {
            let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
            let event = events[i % events.len()];
            let act = rng.next_int(3);
            let ops: Vec<&str> = (0..rng.next_int(9)).map(|_| *rng.pick(OPS).unwrap()).collect();
            // A card goes out at most once, so the deck still holds one.
            let ops: Vec<&str> = ops.iter().enumerate().filter(|&(j, op)| !op.starts_with('-') || !ops[..j].contains(op)).map(|(_, o)| *o).collect();
            let choices: Vec<String> = (0..6).map(|_| rng.next_int(4).to_string()).collect();
            println!("{seed} {} {act} {} {} : {}", [0, 10][i % 2], slug(event), ops.join(" "), choices.join(" "));
        }
        return;
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let (runs, mismatches) = diff_oracle(&text);
    for m in &mismatches {
        println!("{m}");
    }
    println!("{} of {runs} events match", runs - mismatches.len());
}
