//! Random legal play through played runs' fights, stopping at the first
//! hand over `MAX_HAND` with the actions that led there.
//!
//!     cargo run --release --example handcheck -- SEEDS setups.jsonl...
use sim::combat::MAX_HAND;
use sim::gen::run_setups;
use sim::replay::Ids;
use sim::rng::Rng;

fn main() {
    let mut args = std::env::args().skip(1);
    let seeds: u64 = args.next().expect("seeds").parse().unwrap();
    let ids = Ids::new();
    let mut fights = vec![];
    for path in args {
        fights.extend(run_setups(&std::fs::read_to_string(&path).unwrap(), &ids, 1).unwrap());
    }
    eprintln!("{} fights x {seeds} seeds", fights.len());
    let mut worst = 0;
    for seed in 0..seeds {
        for (f, setup) in fights.iter().enumerate() {
            let mut c = setup.combat(seed * 100_003 + f as u64);
            let mut rng = Rng::new(seed ^ (f as u64) << 20);
            let mut trail: Vec<String> = vec![];
            let mut steps = 0;
            while !c.is_over() && steps < 3000 {
                let acts = c.legal_actions();
                let a = acts[rng.next_int(acts.len())];
                let what = match a {
                    sim::combat::Action::PlayCard { hand_idx, .. } => format!("play {:?}", c.player.hand[hand_idx].id),
                    other => format!("{other:?}"),
                };
                let before = c.player.hand.len();
                c.step(a);
                steps += 1;
                trail.push(format!("T{} hand {before}->{} {what}", c.player.turn, c.player.hand.len()));
                worst = worst.max(c.player.hand.len());
                if c.player.hand.len() > MAX_HAND {
                    println!("hand {} in fight {f} ({:?}) seed {seed}", c.player.hand.len(), setup.encounter);
                    println!("relics {:?}", setup.relics.iter().map(|r| r.id).collect::<Vec<_>>());
                    println!("powers {:?}", c.player.creature.powers.iter().map(|p| (p.id, p.amount)).collect::<Vec<_>>());
                    println!("hand {:?}", c.player.hand.iter().map(|k| k.id).collect::<Vec<_>>());
                    for line in trail.iter().rev().take(25).rev() {
                        println!("  {line}");
                    }
                    return;
                }
            }
        }
    }
    println!("no hand over {MAX_HAND} (largest {worst})");
}
