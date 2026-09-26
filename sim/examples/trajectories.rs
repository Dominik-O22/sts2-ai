//! Random playouts with a fingerprint of every state, one line per fight, for
//! checking that a change to the sim's speed changed no outcome. Record the
//! output before the change, rerun after, and diff:
//!
//!     cargo run --release --example trajectories > before.txt
//!     (change the sim)
//!     cargo run --release --example trajectories | diff before.txt -
//!
//! The fights: generated ones from every act (`gen::holdout`), plus the
//! played runs' fights in the tracker's setups when they are on disk. Each
//! step the state is cloned now and then and the clone plays on, so a clone
//! that differs from its original shows up too.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use sim::encode::{encode, N_ACTIONS, N_FLOATS, N_IDS};
use sim::gen::{holdout, run_setups, FightSetup};
use sim::replay::Ids;
use sim::rng::Rng;
use sim::types::Ascension;
use sim::Combat;

/// Every part of the state a later step or the policy can read, hashed.
fn fingerprint(c: &Combat, h: &mut DefaultHasher, buf: &mut (Vec<f32>, Vec<i64>, Vec<bool>)) {
    format!("{:?}", c.legal_actions()).hash(h);
    encode(c, &mut buf.0, &mut buf.1, &mut buf.2);
    buf.0.iter().for_each(|f| f.to_bits().hash(h));
    buf.1.hash(h);
    buf.2.hash(h);
    let creature = |cr: &sim::combat::Creature, h: &mut DefaultHasher| {
        (cr.hp, cr.max_hp, cr.block).hash(h);
        for p in &cr.powers {
            (p.id as usize, p.amount, p.skip_next_tick, p.data, format!("{:?}", p.applier)).hash(h);
        }
    };
    let p = &c.player;
    creature(&p.creature, h);
    for pile in [&p.hand, &p.draw, &p.discard, &p.exhaust, &p.play, &p.offer] {
        format!("{pile:?}").hash(h);
    }
    (p.energy, p.base_max_energy, p.turn).hash(h);
    for e in &c.enemies {
        creature(&e.creature, h);
        (e.monster.id as usize, e.monster.next_move_name(), e.slot, e.reviving, e.escaped).hash(h);
        format!("{:?}", e.monster.vars).hash(h);
    }
    format!("{:?} {} {:?} {:?} {}", c.order, c.round, c.side, c.outcome, c.gold).hash(h);
    format!("{:?} {:?} {:?} {:?} {:?}", c.pending, c.relics, c.potions, c.rngs, c.stats).hash(h);
}

fn main() {
    let ids = Ids::new();
    let mut sets: Vec<(String, Vec<FightSetup>)> = vec![("gen".into(), holdout(11, 6, Ascension(10), 3))];
    let home = std::env::var("HOME").unwrap_or_default();
    for (name, lines) in [("holdout", usize::MAX), ("train", 1500)] {
        let path = format!("{home}/.local/share/SlayTheSpire2/sts2ai/tracker/setups/{name}.jsonl");
        if let Ok(text) = std::fs::read_to_string(&path) {
            let text: String = text.lines().take(lines).map(|l| format!("{l}\n")).collect();
            sets.push((name.into(), run_setups(&text, &ids, 5).expect("setups")));
        }
    }
    let mut buf = (vec![0f32; N_FLOATS], vec![0i64; N_IDS], vec![false; N_ACTIONS]);
    for (name, setups) in &sets {
        for (i, setup) in setups.iter().enumerate() {
            for seed in 0..2u64 {
                let fight_seed = (i as u64) * 7919 + seed;
                let mut c = setup.combat(fight_seed);
                let mut rng = Rng::new(fight_seed ^ 0xABCD);
                let mut h = DefaultHasher::new();
                let mut steps = 0u32;
                fingerprint(&c, &mut h, &mut buf);
                while !c.is_over() && steps < 2000 {
                    let acts = c.legal_actions();
                    c.step(acts[rng.next_int(acts.len())]);
                    steps += 1;
                    if steps % 5 == 2 {
                        c = c.clone();
                    }
                    fingerprint(&c, &mut h, &mut buf);
                }
                println!("{name} {i} {seed} {steps} {:?} {} {:016x}", c.outcome, c.player.creature.hp, h.finish());
            }
        }
    }
}
