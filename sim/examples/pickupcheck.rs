//! Diffs what relics do when picked up (`RunState::obtain`, settled by
//! `rooms::First`) against the game's `RelicCmd.Obtain` for many seeds.
//!
//! `pickupcheck lines N` prints N `SEED ASCENSION ACT RELIC...` lines: every
//! pickup the port has, alone and after others, some beside the relics
//! that change what a pickup does; `tools/oracle obtain` turns those into
//! the player each leaves; `pickupcheck` with no arguments reads that and
//! compares.
//!
//! ```sh
//! cargo run --release --example pickupcheck lines 2000 \
//!   | dotnet ../tools/oracle/bin/Debug/net9.0/oracle.dll obtain > obtain.txt
//! cargo run --release --example pickupcheck < obtain.txt
//! ```
//!
//! A slice of that output is `testdata/oracle-obtain.txt`, which the tests
//! check without dotnet.
use std::io::Read;

use sim::effects::diff_oracle;
use sim::game_rng::GameRng;

/// `SeedHelper._characters`.
const SEED_CHARS: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// Every relic whose pickup the port has, but Sere Talon and Neow's Bones,
/// whose pickups crash the oracle outside the game (the real runs check
/// Neow's Bones).
const PICKUPS: &[&str] = &[
    "STRAWBERRY", "PEAR", "MANGO", "FAKE_MANGO", "NUTRITIOUS_OYSTER", "BIG_MUSHROOM", "LOOMING_FRUIT", "LEES_WAFFLE",
    "FAKE_LEES_WAFFLE", "OLD_COIN", "GOLDEN_PEARL", "CURSED_PEARL", "POTION_BELT", "PHIAL_HOLSTER", "WHETSTONE", "WAR_PAINT",
    "SAND_CASTLE", "NEOWS_TALISMAN", "NEOWS_TORMENT", "BLOOD_SOAKED_ROSE", "DISTINGUISHED_CAPE", "DOLLYS_MIRROR",
    "PRECISE_SCISSORS", "EMPTY_CAGE", "BIIIG_HUG", "PRECARIOUS_SHEARS", "POMANDER", "YUMMY_COOKIE", "PUNCH_DAGGER",
    "ELECTRIC_SHRYMP", "GNARLED_HAMMER", "KIFUDA", "ROYAL_STAMP", "HEFTY_TABLET", "ARCANE_SCROLL", "LARGE_CAPSULE",
    "SMALL_CAPSULE", "SCROLL_BOXES", "LEAFY_POULTICE", "NEW_LEAF", "LOST_COFFER", "LEAD_PAPERWEIGHT",
    "SILKEN_TRESS", "ALCHEMICAL_COFFER", "TOUCH_OF_OROBAS", "ARCHAIC_TOOTH", "ASTROLABE", "PANDORAS_BOX", "PAELS_HORN",
    "PAELS_CLAW", "PAELS_GROWTH", "NUTRITIOUS_SOUP", "STORYBOOK", "JEWELRY_BOX", "TANXS_WHISTLE", "SIGNET_RING",
    "PRESERVED_FOG", "BEAUTIFUL_BRACELET", "TRI_BOOMERANG", "CLAWS", "PAELS_LEGION", "PUMPKIN_CANDLE",
];

/// Relics without a pickup that change what one does.
const CONTEXT: &[&str] = &[
    "MOLTEN_EGG", "TOXIC_EGG", "FROZEN_EGG", "FRESNEL_LENS", "LUCKY_FYSH", "BOWLER_HAT", "SOZU", "SILVER_CRUCIBLE", "DRAGON_FRUIT",
    "ECTOPLASM",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [cmd, n] = &args[..] {
        assert_eq!(cmd, "lines", "usage: pickupcheck [lines N]");
        let n: usize = n.parse().expect("a line count");
        let mut rng = GameRng::new(0x0B7A);
        for i in 0..n {
            let seed: String = (0..10).map(|_| *rng.pick(SEED_CHARS).unwrap() as char).collect();
            // Each pickup alone first, then mixes.
            let mut relics: Vec<&str> = if i < PICKUPS.len() { vec![PICKUPS[i]] } else { Vec::new() };
            while relics.len() < 1 + rng.next_int(4) as usize {
                let pool = if rng.next_int(3) == 0 { CONTEXT } else { PICKUPS };
                let relic = *rng.pick(pool).unwrap();
                if !relics.contains(&relic) {
                    relics.push(relic);
                }
            }
            println!("{seed} {} {} {}", [0, 10][i % 2], rng.next_int(3), relics.join(" "));
        }
        return;
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let (runs, mismatches) = diff_oracle(&text);
    for m in &mismatches {
        println!("{m}");
    }
    println!("{} of {runs} runs match", runs - mismatches.len());
}
