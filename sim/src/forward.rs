//! A run played forward from its seed: each act's map walked point by
//! point, each room entered, fights fought by a `Fights`, the rewards
//! taken, rest sites used, treasure opened, ancients' relics taken, shops
//! bought from, events passed through, act after act, to the last boss or
//! a death. Every decision goes to a `Chooser` (`rooms.rs`), which is where
//! a run policy plugs in.
//!
//! What the port lacks is counted, not an error: events not ported
//! (entered and left) and relic pickups that come back `Offered::Unported`.
//! From the first of them that draws on the Rewards stream the run no
//! longer rolls what the game would (docs/run-env.md, Exactness).

use std::collections::BTreeMap;

use crate::card::UNSUPPORTED_CARDS;
use crate::effects::Offered;
use crate::encounter::Encounter;
use crate::events::EventFight;
use crate::gen::FightSetup;
use crate::map::{ActMap, PointId};
use crate::plan::{select_acts, Unlocks};
use crate::pools::sim_card;
use crate::rewards::Offer;
use crate::rng::Rng;
use crate::rooms::{Chooser, Decision};
use crate::run::{Room, RoomType, RunState};
use crate::shop::{Item, Ware};
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
/// out of card and bundle offers and off the shop's shelves before `inner`
/// chooses.
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
            Decision::Shop(wares) => {
                let kept: Vec<usize> = (0..wares.len()).filter(|&i| !matches!(&wares[i].item, Item::Card(c) if !playable(c))).collect();
                let offered: Vec<Ware> = kept.iter().map(|&i| wares[i].clone()).collect();
                kept.get(self.0.choose(run, Decision::Shop(&offered))).copied().unwrap_or(wares.len())
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

/// Where a run stopped: at a fight to play, or at its end.
#[derive(Clone, Debug)]
pub enum Next {
    Fight(FightSetup),
    End(End),
}

/// A fight handed out and not yet fought: a combat room's, or one an event
/// started, whose rewards are the event's to give.
#[derive(Clone, Debug)]
enum Fighting {
    Room(RoomType),
    Event(EventFight),
}

/// A run in progress, stopped between rooms. `next` plays it to its next
/// fight and hands the fight out, so whoever plays fights (`play` with a
/// `Fights`, or a `VecEnv` slot over many steps) owns the loop.
#[derive(Clone, Debug)]
pub struct Run {
    pub state: RunState,
    map: ActMap,
    /// The map point to enter next, or, while `fighting`, the fight's.
    point: PointId,
    fighting: Option<Fighting>,
    /// Fights handed out, won or not.
    pub fights: usize,
    /// What the port lacks that the run met, by what, with how often.
    pub unported: BTreeMap<String, usize>,
}

impl Run {
    /// The run `seed` names at `ascension` on a fully unlocked profile, at
    /// the first act's start.
    pub fn new(seed: &str, ascension: Ascension) -> Self {
        let unlocks = Unlocks::default();
        let acts = select_acts(crate::game_rng::RunRngs::new(seed).seed, &unlocks);
        let mut state = RunState::new(seed, acts, ascension, &unlocks);
        state.enter_act(0);
        let map = ActMap::generate(state.rngs.seed, acts[0], ascension);
        let point = map.start;
        Self { state, map, point, fighting: None, fights: 0, unported: BTreeMap::new() }
    }

    /// Whether a fight has been handed out and not yet fought.
    pub fn fighting(&self) -> bool {
        self.fighting.is_some()
    }

    /// Plays the run on to its next fight or its end: first the rewards of
    /// the fight handed out last, `fought` (written into `state` already),
    /// then room after room. `fought` is `Some` exactly when `fighting`.
    /// The same seed, choices and fights play the same run.
    pub fn next(&mut self, fought: Option<Fought>, chooser: &mut impl Chooser) -> Next {
        assert_eq!(fought.is_some(), self.fighting(), "a fight's result goes with the fight");
        let mut chooser = Playable(chooser);
        if let (Some(fighting), Some(fought)) = (self.fighting.take(), fought) {
            if !fought.won || self.state.hp <= 0 {
                return Next::End(End::Died);
            }
            let mut log = Vec::new();
            match fighting {
                Fighting::Room(kind) => {
                    self.state.fight_won(kind);
                    let rewards = self.state.combat_rewards(kind, fought.gold_proportion);
                    self.state.take_rewards(rewards, &mut chooser, &mut log);
                }
                Fighting::Event(fight) => drop(self.state.event_fight_won(&fight, fought.gold_proportion, &mut chooser, &mut log)),
            }
            if let Some(end) = self.left(log) {
                return Next::End(end);
            }
            if !self.move_on(&mut chooser) {
                return Next::End(End::Won);
            }
        }
        loop {
            let fighting = match self.state.enter(&self.map, self.point) {
                Room::Combat(kind, encounter) => Some((Fighting::Room(kind), encounter)),
                room => match self.room(room, &mut chooser) {
                    Err(end) => return Next::End(end),
                    Ok(fight) => fight.map(|f| (f.encounter, Fighting::Event(f))).map(|(e, f)| (f, e)),
                },
            };
            if let Some((fighting, encounter)) = fighting {
                return self.hand_out(fighting, encounter);
            }
            if !self.move_on(&mut chooser) {
                return Next::End(End::Won);
            }
        }
    }

    /// Hands a fight out: its enemies rolled from the run's seed and the
    /// floor, not on the run's streams, and created on the run's Niche
    /// stream unless an event's layout created them already.
    fn hand_out(&mut self, fighting: Fighting, encounter: Encounter) -> Next {
        let mut rng = Rng::new((self.state.rngs.seed as u64) << 8 | self.state.floor as u64);
        let enemies = match &fighting {
            Fighting::Event(fight) => fight.enemies(&mut rng),
            Fighting::Room(_) => encounter.monsters(&mut rng),
        };
        match self.state.fight_setup(encounter, enemies) {
            Ok(setup) => {
                if !matches!(&fighting, Fighting::Event(f) if f.created) {
                    self.state.enemies_created(setup.enemies.len());
                }
                self.fighting = Some(fighting);
                self.fights += 1;
                Next::Fight(setup)
            }
            Err(why) => Next::End(End::Stuck(why)),
        }
    }

    /// Steps to the next map point: one of the current point's children,
    /// or the next act's start. False past the last act. Winged Boots let
    /// the player go to any point of the next row, three times
    /// (`MapTravel.GetTravelablePointsFrom`, `WingedBoots.AfterRoomEntered`).
    fn move_on(&mut self, chooser: &mut impl Chooser) -> bool {
        let children: Vec<PointId> = self.map[self.point].children.iter().collect();
        if children.is_empty() {
            let act = self.state.act + 1;
            let Some(plan) = self.state.plan.acts.get(act) else { return false };
            self.map = ActMap::generate(self.state.rngs.seed, plan.act, self.state.ascension);
            self.state.enter_act(act);
            self.point = self.map.start;
            return true;
        }
        let boots = self.state.relics.iter().any(|r| r.id == "WINGED_BOOTS" && r.counter < 3);
        let row = self.map[self.point].row + 1;
        let row: Vec<PointId> = self.map.grid_points().filter(|&p| self.map[p].row == row).collect();
        let options = if boots && !row.is_empty() { row } else { children };
        let i = chooser.choose(&self.state, Decision::Path(&self.map, &options));
        let next = options[i.min(options.len() - 1)];
        if !self.map[self.point].children.contains(next) {
            self.state.relic_mut("WINGED_BOOTS").expect("Winged Boots").counter += 1;
        }
        self.point = next;
        true
    }

    fn count(&mut self, what: String) {
        *self.unported.entry(what).or_default() += 1;
    }

    /// Plays a room that is not a combat room: the run's end if it ended
    /// there, else the fight an event started, if one did.
    fn room(&mut self, room: Room, chooser: &mut impl Chooser) -> Result<Option<EventFight>, End> {
        let mut log = Vec::new();
        let mut fight = None;
        match room {
            Room::Combat(..) => unreachable!("fights are handed out"),
            Room::Treasure => {
                self.state.treasure_room(chooser, &mut log);
            }
            Room::RestSite => self.state.rest_site(chooser, &mut log),
            Room::Shop => {
                self.state.shop_room(chooser, &mut log);
            }
            Room::Event(name) => match self.state.event(name, chooser, &mut log) {
                Some(visit) => fight = visit.fight,
                None => self.count(format!("event {name}")),
            },
            Room::Ancient(name) => {
                self.state.ancient(name, chooser, &mut log);
            }
        }
        match self.left(log) {
            Some(end) => Err(end),
            None => Ok(fight),
        }
    }

    /// Counts the unported relics a room's `log` met; `Some` if the player
    /// left the room dead.
    fn left(&mut self, log: Vec<Offered>) -> Option<End> {
        for offer in log {
            if let Offered::Unported(relic) = offer {
                self.count(format!("relic {relic}"));
            }
        }
        (self.state.hp <= 0).then_some(End::Died)
    }
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

/// Plays the run `seed` names at `ascension` on a fully unlocked profile,
/// each fight played by `fights`. The same seed, chooser and fights play
/// the same run.
pub fn play(seed: &str, ascension: Ascension, chooser: &mut impl Chooser, fights: &mut impl Fights) -> Played {
    let mut run = Run::new(seed, ascension);
    let mut fought = None;
    loop {
        match run.next(fought, chooser) {
            Next::Fight(setup) => fought = Some(fights.fight(&mut run.state, setup)),
            Next::End(end) => return Played { state: run.state, end, fights: run.fights, unported: run.unported },
        }
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
