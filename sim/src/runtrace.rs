//! Per-run traces: what a run fought, decided and carried, one JSON line
//! a run, the format `sts2ai.runtrace` documents and `sts2ai.runreport`
//! and `sts2ai.paired` read. A `Tracer` sits in a run-mode slot
//! (`env::Slot`) and hears of each fight as it starts and ends, each
//! decision as it is answered, and the run's end, which finishes the
//! line.

use serde::Serialize;

use crate::combat::Combat;
use crate::encode::N_RELICS;
use crate::env::RunFight;
use crate::forward::End;
use crate::gen::FightSetup;
use crate::ids::ALL_CARDS;
use crate::replay::slug;
use crate::runobs::{self, RunObs, F_OPTIONS, I_OPTIONS, MAX_OPTIONS, OPTION_CARDS, OPTION_FLOATS, OPTION_IDS, RUN_RELICS};
use crate::{enchant, potion, relic};

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Fight {
    floor: u32,
    act: u32,
    kind: String,
    enc: String,
    hp: i32,
    max_hp: i32,
    potions: usize,
    cards: usize,
    gold: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    won: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lost: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    used: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Deck {
    floor: u32,
    act: u32,
    enc: String,
    hp: i32,
    max_hp: i32,
    gold: i32,
    cards: Vec<String>,
    relics: Vec<String>,
    potions: Vec<String>,
}

/// `[floor, kind, pick, options]`: the distinct options offered, in words,
/// and which of them was taken.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Decision(u32, &'static str, usize, Vec<String>);

#[derive(Serialize)]
struct Line<'a> {
    seed: u64,
    end: String,
    act: u32,
    floor: u32,
    deck: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'static str>,
    fights: &'a [Fight],
    decisions: &'a [Decision],
    decks: &'a [Deck],
}

/// One run's trace as it plays; `fights` ends with the fight being
/// played, whose outcome `fought` fills in.
#[derive(Default)]
pub struct Tracer {
    fights: Vec<Fight>,
    decisions: Vec<Decision>,
    decks: Vec<Deck>,
    /// The last fight's deck entry.
    last: Option<Deck>,
}

impl Tracer {
    /// A fight starts, `act` being the run's.
    pub fn started(&mut self, setup: &FightSetup, act: u32) {
        let enc = slug(&format!("{:?}", setup.encounter));
        self.fights.push(Fight {
            floor: setup.floor,
            act,
            kind: format!("{:?}", setup.encounter.kind()),
            enc: enc.clone(),
            hp: setup.hp,
            max_hp: setup.max_hp,
            potions: setup.potions.iter().flatten().count(),
            cards: setup.deck.len(),
            gold: setup.gold,
            won: None,
            lost: None,
            used: None,
            steps: None,
        });
        let last = Deck {
            floor: setup.floor,
            act,
            enc,
            hp: setup.hp,
            max_hp: setup.max_hp,
            gold: setup.gold,
            cards: setup.deck.iter().map(|c| card_text(&slug(&format!("{:?}", c.id)), c.upgraded, c.enchantment.as_ref().map(|e| (slug(&format!("{:?}", e.id)), e.amount)))).collect(),
            relics: setup.relics.iter().map(|r| slug(&format!("{:?}", r.id))).collect(),
            potions: setup.potions.iter().flatten().map(|p| slug(&format!("{p:?}"))).collect(),
        };
        if setup.encounter.kind() == crate::encounter::Kind::Boss {
            self.decks.push(last.clone());
        }
        self.last = Some(last);
    }

    /// The fight that started last is over as `combat`, after `steps`
    /// actions and `used` potions.
    pub fn fought(&mut self, setup: &FightSetup, combat: &Combat, used: u32, steps: u32) {
        if let Some(f) = self.fights.last_mut() {
            f.won = Some(combat.outcome == Some(crate::combat::Outcome::Won));
            f.lost = Some(setup.hp - combat.player.creature.hp.max(0));
            f.used = Some(used);
            f.steps = Some(steps);
        }
    }

    /// The decision `obs` is answered with its option token `option`.
    pub fn decided(&mut self, obs: &RunObs, option: usize) {
        let texts: Vec<(usize, String)> = offered(&obs.floats).into_iter().map(|j| (j, option_text(&obs.floats, &obs.ids, j))).collect();
        let mut distinct: Vec<String> = Vec::new();
        for (_, t) in &texts {
            if !distinct.contains(t) {
                distinct.push(t.clone());
            }
        }
        let picked = &texts.iter().find(|(j, _)| *j == option).expect("the pick is an offered option").1;
        let pick = distinct.iter().position(|t| t == picked).expect("picked text");
        let floor = (obs.floats[3] as f64 * 49.0).round() as u32;
        self.decisions.push(Decision(floor, runobs::DECISIONS[obs.ids[0] as usize - 1], pick, distinct));
    }

    /// The run ended as `run` says: its line, and the tracer is reset for
    /// the slot's next run.
    pub fn ended(&mut self, run: &RunFight) -> String {
        let mut me = std::mem::take(self);
        if let Some(last) = me.last.take() {
            if !me.decks.contains(&last) {
                me.decks.push(last);
            }
        }
        let (start, source) = match run.began {
            crate::env::Began::Floor1 => (None, None),
            crate::env::Began::Generated(at) => (Some(at.index()), Some("gen")),
            crate::env::Began::Own(at) => (Some(at.index()), Some("own")),
            crate::env::Began::Winner(at) => (Some(at.index()), Some("win")),
        };
        let end = match &run.end {
            Some(End::Won) => "won".to_string(),
            Some(End::Died) => "died".to_string(),
            Some(End::Stuck(why)) => format!("stuck: {why}"),
            None => String::new(),
        };
        let line = Line { seed: run.seed, end, act: run.act, floor: run.floor, deck: run.deck, start, source, fights: &me.fights, decisions: &me.decisions, decks: &me.decks };
        serde_json::to_string(&line).expect("a trace line")
    }
}

/// `BASH+`, `STRIKE~SHARP2`: a deck card as the trace writes it.
fn card_text(id: &str, upgraded: bool, enchantment: Option<(String, i32)>) -> String {
    let mut text = format!("{id}{}", if upgraded { "+" } else { "" });
    if let Some((e, amount)) = enchantment {
        text.push_str(&format!("~{e}{amount}"));
    }
    text
}

/// The option tokens a run row offers.
fn offered(floats: &[f32]) -> Vec<usize> {
    (0..MAX_OPTIONS).filter(|&k| floats[F_OPTIONS + k * OPTION_FLOATS] != 0.0).collect()
}

fn card_name(id: i64) -> String {
    if id == 0 { "-".into() } else { slug(&format!("{:?}", ALL_CARDS[id as usize - 1])) }
}

fn potion_name(id: i64) -> String {
    if id == 0 { "-".into() } else { slug(&format!("{:?}", potion::ALL[id as usize - 1])) }
}

fn enchant_name(id: i64) -> String {
    if id == 0 { "-".into() } else { slug(&format!("{:?}", enchant::ALL[id as usize - 1])) }
}

/// The relic vocabulary of `runobs`: the sim's relics, then the run-only
/// ones (`RUN_RELICS`); 0 is the pad.
fn relic_name(id: i64) -> String {
    let i = id as usize;
    if i == 0 {
        "<pad>".into()
    } else if i <= N_RELICS {
        slug(&format!("{:?}", relic::ALL[i - 1]))
    } else {
        RUN_RELICS[i - 1 - N_RELICS].to_string()
    }
}

/// Option token `k` of a run row in words: its kind and what it names,
/// as `sts2ai.runtrace.option_text(..., paths=False)` writes it.
pub fn option_text(floats: &[f32], ids: &[i64], k: usize) -> String {
    let (i, f) = (I_OPTIONS + k * OPTION_IDS, F_OPTIONS + k * OPTION_FLOATS);
    const C: usize = OPTION_CARDS;
    let kind = ids[i] as usize;
    let mut parts: Vec<String> = vec![if kind == 0 { "<pad>".into() } else { format!("{:?}", runobs::OPTION_KINDS[kind - 1]) }];
    for c in 0..C {
        if ids[i + 1 + c] != 0 {
            parts.push(format!("{}{}", card_name(ids[i + 1 + c]), if floats[f + 1 + c] != 0.0 { "+" } else { "" }));
        }
    }
    if ids[i + 1 + C] != 0 {
        parts.push(enchant_name(ids[i + 1 + C]));
    }
    if ids[i + 2 + C] != 0 {
        parts.push(relic_name(ids[i + 2 + C]));
    }
    if ids[i + 3 + C] != 0 {
        parts.push(potion_name(ids[i + 3 + C]));
    }
    if ids[i + 4 + C] != 0 {
        parts.push(runobs::ROOMS[ids[i + 4 + C] as usize - 1].to_string());
    }
    if parts[0] == "Event" {
        // The option's id is its `EVENT_OPTIONS` index + 1.
        if let Some(&(event, page, key)) = (ids[i + 5 + C] as usize).checked_sub(1).and_then(|k| runobs::EVENT_OPTIONS.get(k)) {
            parts[0] = format!("Event {}", runobs::event_option_name(event, page, key));
        }
    }
    let price = floats[f + 2 + C];
    if price != 0.0 {
        parts.push(format!("{}g", (price as f64 * 100.0).round() as i64));
    }
    parts.join(" ")
}
