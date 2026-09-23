//! Generated fights in the recorder's `start` format, one JSON line each, so
//! `scripts/deckstats.py --gen` can set them beside real runs.
//! `cargo run --release --example gendump [fights_per_floor] > gen.jsonl`
use serde_json::{json, Value};
use sim::gen::{generate, LAST_FLOOR};
use sim::replay::slug;
use sim::rng::Rng;
use sim::types::Ascension;

fn main() {
    let per_floor: u32 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let mut rng = Rng::new(7);
    for floor in 1..=LAST_FLOOR {
        for _ in 0..per_floor {
            let s = generate(&mut rng, floor, Ascension(10));
            let deck: Vec<Value> = s
                .deck
                .iter()
                .map(|k| {
                    let mut c = json!({"id": slug(&format!("{:?}", k.id)), "up": k.upgraded});
                    if let Some(e) = k.enchantment {
                        c["ench"] = json!([slug(&format!("{:?}", e.id)), e.amount, false]);
                    }
                    c
                })
                .collect();
            let line = json!({
                "t": "start",
                "encounter": slug(&format!("{:?}", s.encounter)),
                "room": format!("{:?}", s.room),
                "floor": floor,
                "deck": deck,
                "relics": s.relics.iter().map(|r| slug(&format!("{:?}", r.id))).collect::<Vec<_>>(),
                "potions": s.potions.iter().map(|p| p.map(|id| slug(&format!("{id:?}")))).collect::<Vec<_>>(),
                "hp": s.hp,
                "max_hp": s.max_hp,
            });
            println!("{line}");
        }
    }
}
