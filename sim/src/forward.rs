//! A run played forward from its seed: each act's map walked point by
//! point, each room entered, fights fought by a `Fights`, the rewards
//! taken, rest sites used, treasure opened, shops and events passed
//! through, act after act, to the last boss or a death. Every decision goes
//! to a `Chooser` (`rooms.rs`), which is where a run policy plugs in.
//!
//! What the port lacks is counted, not an error: events (entered and left
//! after what laying them out draws), shops (stocked and left), the
//! ancients' options, and relic pickups that come back `Offered::Unported`.
//! From the first of them that draws on the Rewards stream the run no
//! longer rolls what the game would (docs/run-env.md, Exactness).

use std::collections::BTreeMap;

use crate::card::UNSUPPORTED_CARDS;
use crate::effects::Offered;
use crate::encounter::Encounter;
use crate::gen::FightSetup;
use crate::map::{ActMap, PointId};
use crate::plan::{select_acts, Unlocks};
use crate::pools::sim_card;
use crate::rewards::Offer;
use crate::rng::Rng;
use crate::rooms::{Chooser, Decision};
use crate::run::{Room, RunState};
use crate::types::Ascension;

/// How a fight went, once `Fights::fight` has written it into the run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fought {
    pub won: bool,
    /// `CombatRoom.GoldProportion` (`RunState::end_fight`).
    pub gold_proportion: f32,
}

/// Plays a fight out and writes it back into the run (`end_fight` for a
/// combat the sim played).
pub trait Fights {
    fn fight(&mut self, run: &mut RunState, setup: FightSetup) -> Fought;
}

impl<F: FnMut(&mut RunState, FightSetup) -> Fought> Fights for F {
    fn fight(&mut self, run: &mut RunState, setup: FightSetup) -> Fought {
        self(run, setup)
    }
}

/// Every fight won at seven tenths of the HP it started with, rounded up,
/// with nothing else changed.
pub fn stub_fight(run: &mut RunState, _: FightSetup) -> Fought {
    run.hp = (run.hp * 7 + 9) / 10;
    Fought { won: true, gold_proportion: 1.0 }
}

/// Whether the combat sim can play a card.
fn playable(offer: &Offer) -> bool {
    sim_card(offer.id).is_some_and(|id| !UNSUPPORTED_CARDS.iter().any(|(u, _)| *u == id))
}

/// A chooser that never sees a card the sim cannot play: those are left
/// out of card and bundle offers before `inner` chooses.
struct Playable<'a, C: Chooser>(&'a mut C);

impl<C: Chooser> Chooser for Playable<'_, C> {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
        match decision {
            Decision::Card(offers) => {
                let kept: Vec<usize> = (0..offers.len()).filter(|&i| playable(&offers[i])).collect();
                let cards: Vec<Offer> = kept.iter().map(|&i| offers[i]).collect();
                kept.get(self.0.choose(run, Decision::Card(&cards))).copied().unwrap_or(offers.len())
            }
            Decision::Bundle(bundles) => {
                let kept: Vec<usize> = (0..bundles.len()).filter(|&i| bundles[i].iter().all(playable)).collect();
                let offered: Vec<Vec<Offer>> = kept.iter().map(|&i| bundles[i].clone()).collect();
                kept.get(self.0.choose(run, Decision::Bundle(&offered))).copied().unwrap_or(bundles.len())
            }
            other => self.0.choose(run, other),
        }
    }
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq)]
pub enum End {
    /// Beat the last act's bosses.
    Won,
    /// Lost a fight, or HP ran out, on the run's last floor.
    Died,
    /// A fight the sim cannot build (a card, relic or potion it lacks).
    Stuck(String),
}

/// A run played to its end.
#[derive(Clone, Debug)]
pub struct Played {
    pub state: RunState,
    pub end: End,
    /// Fights fought, won or not.
    pub fights: usize,
    /// What the port lacks that the run met, by what, with how often.
    pub unported: BTreeMap<String, usize>,
}

/// Plays the run `seed` names at `ascension` on a fully unlocked profile.
/// The same seed, chooser and fights play the same run.
pub fn play(seed: &str, ascension: Ascension, chooser: &mut impl Chooser, fights: &mut impl Fights) -> Played {
    let unlocks = Unlocks::default();
    let acts = select_acts(crate::game_rng::RunRngs::new(seed).seed, &unlocks);
    let mut run = Played { state: RunState::new(seed, acts, ascension, &unlocks), end: End::Won, fights: 0, unported: BTreeMap::new() };
    let mut chooser = Playable(chooser);
    for act in 0..acts.len() {
        run.state.enter_act(act);
        let map = ActMap::generate(run.state.rngs.seed, acts[act], ascension);
        let mut point = map.start;
        loop {
            if let Some(end) = run.room(&map, point, &mut chooser, fights) {
                run.end = end;
                return run;
            }
            let children: Vec<PointId> = map[point].children.iter().collect();
            if children.is_empty() {
                break;
            }
            let i = chooser.choose(&run.state, Decision::Path(&children));
            point = children[i.min(children.len() - 1)];
        }
    }
    run
}

impl Played {
    fn count(&mut self, what: String) {
        *self.unported.entry(what).or_default() += 1;
    }

    /// Enters `point` and plays its room; `Some` if the run ended there.
    fn room(&mut self, map: &ActMap, point: PointId, chooser: &mut impl Chooser, fights: &mut impl Fights) -> Option<End> {
        let mut log = Vec::new();
        match self.state.enter(map, point) {
            Room::Combat(kind, encounter) => {
                let fought = match self.fight(encounter, fights) {
                    Ok(fought) => fought,
                    Err(why) => return Some(End::Stuck(why)),
                };
                if !fought.won || self.state.hp <= 0 {
                    return Some(End::Died);
                }
                let rewards = self.state.combat_rewards(kind, fought.gold_proportion);
                self.state.take_rewards(rewards, chooser, &mut log);
            }
            Room::Treasure => {
                self.state.treasure_room(chooser, &mut log);
            }
            Room::RestSite => self.state.rest_site(chooser, &mut log),
            Room::Shop => {
                self.state.shop();
                self.count("shop".into());
            }
            Room::Event(name) => {
                self.state.event_offer(name);
                self.count(format!("event {name}"));
            }
            Room::Ancient(name) => self.count(format!("ancient {name}'s options")),
        }
        for offer in log {
            if let Offered::Unported(relic) = offer {
                self.count(format!("relic {relic}"));
            }
        }
        (self.state.hp <= 0).then_some(End::Died)
    }

    /// Builds the fight, enemies rolled from the run's seed and the floor,
    /// and has `fights` play it.
    fn fight(&mut self, encounter: Encounter, fights: &mut impl Fights) -> Result<Fought, String> {
        let mut rng = Rng::new((self.state.rngs.seed as u64) << 8 | self.state.floor as u64);
        let setup = self.state.fight_setup(encounter, encounter.monsters(&mut rng))?;
        self.fights += 1;
        Ok(fights.fight(&mut self.state, setup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rooms::First;

    /// A seeded A10 run with every fight won at 70% HP and the first option
    /// always taken climbs all 49 floors, the same way twice: the same
    /// floors, deck, relics, potions, HP and gold, and the same draws on
    /// the Rewards stream.
    #[test]
    fn plays_a_seed_to_the_end_the_same_way_twice() {
        let run = || play("RUNSIM1", Ascension(10), &mut First, &mut stub_fight);
        let (a, b) = (run(), run());
        assert_eq!(a.end, End::Won, "unported: {:?}", a.unported);
        assert_eq!(a.state.floor, 49);
        assert_eq!(a.fights, b.fights);
        assert_eq!((a.state.hp, a.state.max_hp, a.state.gold), (b.state.hp, b.state.max_hp, b.state.gold));
        assert_eq!(a.state.deck, b.state.deck);
        assert_eq!(a.state.relics, b.state.relics);
        assert_eq!(a.state.potions, b.state.potions);
        assert_eq!(a.unported, b.unported);
        let mut a = a.state;
        assert_eq!(a.rewards().counter, b.state.clone().rewards().counter);
        assert!(a.deck.len() > 10 + 16, "took a card from every fight: {}", a.deck.len());
    }
}
