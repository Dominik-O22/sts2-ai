//! Plays whole fights with the sampled-future planner, and with the same
//! planner peeking at the real future, to tell its flaws apart.
//!
//!     cargo run --release --example pimccheck -- setups.jsonl N
use rayon::prelude::*;
use sim::combat::Outcome;
use sim::env::Baseline;
use sim::gen::run_setups;
use sim::pimc::{plan, Config, Plan};
use sim::replay::Ids;
use sim::turnsearch::state_key;

fn play(setup: &sim::gen::FightSetup, seed: u64, cfg: &Config) -> bool {
    let mut c = setup.combat(seed);
    let base = Baseline::of(&c);
    let mut cur: Option<(Plan, usize)> = None;
    for _ in 0..400 {
        if c.is_over() {
            break;
        }
        let follow = cur.as_ref().is_some_and(|(p, pos)| {
            *pos > 0 && *pos < p.line.len() && state_key(&c) == p.keys[*pos - 1] && c.legal_actions().contains(&p.line[*pos])
        });
        if !follow {
            cur = plan(&c, base, cfg).map(|p| (p, 0));
        }
        let Some((p, pos)) = cur.as_mut() else { break };
        let a = p.line[*pos];
        *pos += 1;
        c.step(a);
    }
    c.outcome == Some(Outcome::Won)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let text = std::fs::read_to_string(args.next().expect("setups")).unwrap();
    let n: usize = args.next().map_or(40, |a| a.parse().unwrap());
    let fights = run_setups(&text, &Ids::new(), 1).unwrap();
    for (name, cfg) in [("peek", Config { samples: 1, peek: true, ..Config::default() }), ("sampled4", Config { samples: 4, ..Config::default() })] {
        let t = std::time::Instant::now();
        let won: usize = fights.par_iter().take(n).enumerate().map(|(i, s)| play(s, 1000 + i as u64, &cfg) as usize).sum();
        println!("{name}: won {won} of {n} ({:.0}s)", t.elapsed().as_secs_f64());
    }
}
