//! Checks the run-start plan against runs the game played: each run
//! history file (`saves/history/*.run`) names the seed, the acts, and the
//! rooms met floor by floor. The plan for the seed must hold them in order:
//! each act's ancient, its monster and elite fights as a prefix of the
//! planned lists, its bosses, and its events as a subsequence of the planned
//! events (a `?` room skips the ones the run does not allow).
//!
//! ```sh
//! cargo run --release --example historycheck -- ~/.local/share/SlayTheSpire2/steam/*/modded/profile1/saves/history/*.run
//! ```
//!
//! Plans assume a fully unlocked profile (`Unlocks::default`); runs from
//! before that, or built through the dev console, show up as mismatches.
//! Only standard single-player Ironclad runs on the pinned game version are
//! checked: another character's relic pool changes how many draws the
//! player's grab bag takes, and older builds drew differently.
use serde_json::Value;

use sim::encounter::Act;
use sim::game_rng::hash;
use sim::plan::{select_acts, ActPlan, RunPlan, Unlocks};
use sim::replay::slug;
use sim::types::Ascension;

fn main() {
    let (mut checked, mut matched, mut skipped) = (0, 0, 0);
    for path in std::env::args().skip(1) {
        let run: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = path.rsplit('/').next().unwrap();
        let character = run["players"][0]["character"].as_str().unwrap_or("");
        let solo = run["players"].as_array().map_or(0, Vec::len) == 1;
        if run["build_id"] != "v0.107.1" || run["game_mode"] != "standard" || character != "CHARACTER.IRONCLAD" || !solo {
            skipped += 1;
            continue;
        }
        let seed_text = run["seed"].as_str().unwrap();
        let seed = hash(seed_text) as u32;
        let ascension = Ascension(run["ascension"].as_u64().unwrap() as u8);
        let acts: Vec<Act> = run["acts"].as_array().unwrap().iter().map(|a| act(a.as_str().unwrap())).collect();
        let acts: [Act; 3] = acts.try_into().expect("three acts");
        let unlocks = Unlocks::default();
        let plan = RunPlan::generate(seed, acts, ascension, &unlocks);
        let mut problems = Vec::new();
        if select_acts(seed, &unlocks) != acts {
            problems.push(format!("acts {acts:?} are not the seed's {:?}", select_acts(seed, &unlocks)));
        }
        let floors = run["map_point_history"].as_array().unwrap();
        for (planned, floors) in plan.acts.iter().zip(floors) {
            if let Err(problem) = check_act(planned, floors.as_array().unwrap()) {
                problems.push(format!("{:?}: {problem}", planned.act));
            }
        }
        checked += 1;
        let what = format!("{name} {seed_text} A{} {:?}", ascension.0, acts);
        if problems.is_empty() {
            matched += 1;
            println!("{what}: ok");
        } else {
            println!("{what}: {}", problems.join("; "));
        }
    }
    println!("{matched} of {checked} standard Ironclad runs match their plan ({skipped} others skipped)");
}

fn act(id: &str) -> Act {
    match id {
        "ACT.OVERGROWTH" => Act::Overgrowth,
        "ACT.UNDERDOCKS" => Act::Underdocks,
        "ACT.HIVE" => Act::Hive,
        "ACT.GLORY" => Act::Glory,
        _ => panic!("unknown act {id}"),
    }
}

/// The rooms an act's floors met against its plan.
fn check_act(plan: &ActPlan, floors: &[Value]) -> Result<(), String> {
    let mut met: Vec<(&str, &str, &str)> = Vec::new();
    for floor in floors {
        let point = floor["map_point_type"].as_str().unwrap();
        for room in floor["rooms"].as_array().unwrap() {
            if let Some(id) = room["model_id"].as_str() {
                met.push((point, room["room_type"].as_str().unwrap(), id));
            }
        }
    }
    let ids = |room_type: &str| -> Vec<String> {
        met.iter()
            .filter(|&&(point, rt, id)| rt == room_type && point != "ancient" && !id.ends_with("_EVENT_ENCOUNTER"))
            .map(|&(_, _, id)| id.split_once('.').unwrap().1.to_string())
            .collect()
    };
    let encounter = |e: &sim::encounter::Encounter| slug(&format!("{e:?}"));
    let prefix = |kind: &str, got: Vec<String>, planned: Vec<String>| -> Result<(), String> {
        let want: Vec<String> = planned.iter().cycle().take(got.len()).cloned().collect();
        if got == want { Ok(()) } else { Err(format!("{kind} {got:?}, planned {want:?}")) }
    };

    if let Some(&(_, _, id)) = met.iter().find(|&&(point, _, _)| point == "ancient") {
        let want = plan.ancient.map(|a| format!("EVENT.{}", slug(a)));
        if want.as_deref() != Some(id) {
            return Err(format!("ancient {id}, planned {want:?}"));
        }
    }
    prefix("monsters", ids("monster"), plan.normal.iter().map(encounter).collect())?;
    prefix("elites", ids("elite"), plan.elites.iter().map(encounter).collect())?;
    let bosses: Vec<String> = [Some(plan.boss), plan.second_boss].iter().flatten().map(encounter).collect();
    let got = ids("boss");
    if !bosses.starts_with(&got) {
        return Err(format!("bosses {got:?}, planned {bosses:?}"));
    }
    let events = ids("event");
    let planned: Vec<String> = plan.events.iter().map(|e| slug(e)).collect();
    let mut rest = planned.iter();
    if let Some(missing) = events.iter().find(|e| !rest.any(|p| p == *e)) {
        return Err(format!("event {missing} is not next in {planned:?} after the events before it ({events:?})"));
    }
    Ok(())
}
