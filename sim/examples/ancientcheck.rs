//! Diffs the ancients' options (`sim::ancients`) against the game's code for
//! many seeds.
//!
//! `ancientcheck lines N` prints N `SEED ASCENSION ACT ANCIENT [+CARD|-CARD]...`
//! lines: every ancient in the acts it can meet, with decks that turn each
//! option's condition on and off (Bash, basic Strikes, cards Goopy, Swift
//! and Instinct take, Byrdonis' egg); `tools/oracle ancients` turns those
//! into what the game laid out; `ancientcheck` with no arguments reads that
//! and compares.
//!
//! ```sh
//! cargo run --release --example ancientcheck lines 2000 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll ancients > ancients.txt
//! cargo run --release --example ancientcheck < ancients.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-ancients.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::ancients::diff_oracle;
use sim::game_rng::GameRng;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// Deck changes that move the ancients' conditions: the starter's cards
/// out, attacks and skills in.
const OPS: &[&str] = &[
    "-BASH", "-STRIKE_IRONCLAD", "-STRIKE_IRONCLAD", "-DEFEND_IRONCLAD", "-DEFEND_IRONCLAD", "+POMMEL_STRIKE", "+SHRUG_IT_OFF",
    "+INFLAME", "+BYRDONIS_EGG", "+ASCENDERS_BANE",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "lines", "usage: ancientcheck [lines N]");
        let n: usize = n.parse().expect("a line count");
        let mut rng = GameRng::new(0xA7C1);
        let ancients = [("NEOW", 0), ("OROBAS", 1), ("PAEL", 1), ("TEZCATARA", 1), ("DARV", 1), ("DARV", 2), ("VAKUU", 2), ("NONUPEIPE", 2), ("TANX", 2)];
        for i in 0..n {
            let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
            let (ancient, act) = ancients[i % ancients.len()];
            let mut ops: Vec<&str> = Vec::new();
            for _ in 0..rng.next_int(12) {
                let op = *rng.pick(OPS).unwrap();
                // A card goes out only while the deck still holds one.
                let held = 1 + ops.iter().filter(|o| o[1..] == op[1..] && o.starts_with('+')).count();
                let out = ops.iter().filter(|o| **o == op).count();
                let starter = match &op[1..] {
                    "STRIKE_IRONCLAD" => 5,
                    "DEFEND_IRONCLAD" => 4,
                    "BASH" => 1,
                    _ => 0,
                };
                if op.starts_with('+') || out < starter + held - 1 {
                    ops.push(op);
                }
            }
            println!("{seed} {} {act} {ancient} {}", [0, 10][i % 2], ops.join(" "));
        }
        return;
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let (runs, mismatches) = diff_oracle(&text);
    for m in &mismatches {
        println!("{m}");
    }
    println!("{} of {runs} ancients match", runs - mismatches.len());
}
