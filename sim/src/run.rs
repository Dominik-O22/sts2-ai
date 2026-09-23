//! A run between fights, value for value with the game: the state the
//! run layer carries (`Runs/RunState.cs` and the player's side of it) and
//! the walk from map point to room (`RunManager.EnterMapPointInternal`,
//! `RollRoomTypeFor`, `CreateRoom`, `Odds/UnknownMapPointOdds.cs`,
//! `Rooms/RoomSet.cs`). Rewards live in `rewards.rs`.
//!
//! Cards, relics and potions are game ids here, so the run can hold what the
//! combat sim has no model for. A fight is built from the run in the sim's
//! ids (`fight_setup`) and written back once it is over (`end_fight`).
//!
//! Not here: the first run's tutorial rooms and rewards (a fully unlocked
//! profile never meets them), multiplayer, and the relic hooks on unknown
//! rooms besides Juzu Bracelet's (Golden Compass, the Lantern Key card).

use crate::card::{Card, UNSUPPORTED_CARDS};
use crate::combat::{Combat, EnemySpec, RoomKind};
use crate::enchant::Enchantment;
use crate::encounter::{Act, Encounter, Kind};
use crate::gen::{FightSetup, INERT_RELICS};
use crate::ids::MonsterId;
use crate::pools::{sim_card, sim_enchantment, sim_potion, sim_relic};
use crate::relic::Relic;
use crate::replay::slug;
use crate::game_rng::{GameRng, RunRngs, RunStream};
use crate::map::{ActMap, PointId, PointType};
use crate::plan::{RunPlan, Unlocks};
use crate::rewards::{CardOdds, PotionOdds};
use crate::types::{Ascension, AscensionLevel};

/// `Rooms/RoomType.cs`, the rooms a map point can become, in the game's
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RoomType {
    Monster,
    Elite,
    Boss,
    Treasure,
    Shop,
    Event,
    RestSite,
}

/// The room a map point became.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    /// A monster, elite or boss fight.
    Combat(RoomType, Encounter),
    /// An event's class name.
    Event(&'static str),
    /// The act's ancient, an event too.
    Ancient(&'static str),
    Treasure,
    Shop,
    RestSite,
}

impl Room {
    pub fn kind(self) -> RoomType {
        match self {
            Room::Combat(kind, _) => kind,
            Room::Event(_) | Room::Ancient(_) => RoomType::Event,
            Room::Treasure => RoomType::Treasure,
            Room::Shop => RoomType::Shop,
            Room::RestSite => RoomType::RestSite,
        }
    }
}

/// `UnknownMapPointOdds`: the chance a `?` holds each room besides an
/// event, in its dictionary's order (monster, elite, treasure, shop). Each
/// roll resets the room it lands on and raises the others by their base.
/// Elites start negative, which rules them out for good.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownOdds([f32; 4]);

impl UnknownOdds {
    const ROOMS: [RoomType; 4] = [RoomType::Monster, RoomType::Elite, RoomType::Treasure, RoomType::Shop];
    const BASE: [f32; 4] = [0.1, -1.0, 0.02, 0.03];

    /// `ResetToBase`, which every new act does.
    pub fn reset(&mut self) {
        self.0 = Self::BASE;
    }

    /// `Roll` with the rooms `allowed` leaves in: one draw, then the first
    /// room whose running sum of odds reaches it, else an event.
    fn roll(&mut self, allowed: impl Fn(RoomType) -> bool, rng: &mut GameRng) -> RoomType {
        let roll = rng.next_float(1.0);
        let mut sum = 0f32;
        let mut rolled = RoomType::Event;
        for (room, odds) in Self::ROOMS.into_iter().zip(self.0) {
            if allowed(room) && odds >= 0.0 {
                sum += odds;
                if roll <= sum {
                    rolled = room;
                    break;
                }
            }
        }
        for (i, room) in Self::ROOMS.into_iter().enumerate() {
            if room == rolled {
                self.0[i] = Self::BASE[i];
            } else if allowed(room) {
                self.0[i] += Self::BASE[i];
            }
        }
        rolled
    }
}

impl Default for UnknownOdds {
    fn default() -> Self {
        UnknownOdds(Self::BASE)
    }
}

/// `RoomSet`'s counters: how many rooms of each list an act has used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Visited {
    pub events: usize,
    pub normal: usize,
    pub elites: usize,
    pub bosses: usize,
}

/// A card in the deck, by game id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeckCard {
    pub id: String,
    pub upgraded: bool,
    pub enchantment: Option<Enchant>,
}

/// A card's enchantment (`EnchantmentModel`): its game id and `Amount`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enchant {
    pub id: String,
    pub amount: i32,
}

impl DeckCard {
    pub fn new(id: &str) -> Self {
        DeckCard { id: id.to_string(), upgraded: false, enchantment: None }
    }

    fn basic(&self) -> bool {
        matches!(self.id.as_str(), "STRIKE_IRONCLAD" | "DEFEND_IRONCLAD" | "BASH")
    }
}

/// A relic held, by game id, with what it keeps from room to room: the
/// combat sim's persistent `counter` and `flag` (`relic::Relic`, in the
/// sim's meaning), and the run layer's own (Lasting Candy's `CombatsSeen`
/// and Silver Crucible's `TimesUsed` as the counter, Lava Rock's
/// `HasTriggered` as the flag).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRelic {
    pub id: String,
    pub counter: i32,
    pub flag: bool,
}

impl RunRelic {
    /// A fresh one, with the charges the sim's comes with.
    pub fn new(id: &str) -> Self {
        let counter = crate::pools::sim_relic(id).map_or(0, |r| crate::relic::Relic::new(r).counter);
        RunRelic { id: id.to_string(), counter, flag: false }
    }
}

/// A singleplayer Ironclad run as the run layer sees it.
#[derive(Clone, Debug)]
pub struct RunState {
    /// The run's streams and the player's. The player's Rewards stream is
    /// the one rewards, shops and many relics and events draw on.
    pub rngs: RunRngs,
    pub ascension: Ascension,
    /// What run start drew; its grab bags empty as relics are pulled.
    pub plan: RunPlan,
    /// `CurrentActIndex`.
    pub act: usize,
    /// `RunState.TotalFloor`: the map points entered so far.
    pub floor: usize,
    pub visited: [Visited; 3],
    /// `VisitedEventIds`, class names.
    pub events_seen: Vec<&'static str>,
    pub unknown_odds: UnknownOdds,
    pub card_odds: CardOdds,
    pub potion_odds: PotionOdds,
    pub gold: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub deck: Vec<DeckCard>,
    /// In the order they came.
    pub relics: Vec<RunRelic>,
    /// `Player.PotionSlots`: one entry per slot, game ids.
    pub potions: Vec<Option<String>>,
    /// The map point the player stands on, once the act's first is entered.
    pub point: Option<PointId>,
    /// The room that point became, which the next point's shop blacklist
    /// reads.
    pub room: Option<Room>,
    /// `ExtraPlayerFields.CardShopRemovalsUsed`: card removals bought, which
    /// price the next.
    pub shop_removals: i32,
}

impl RunState {
    /// A new run of a fully unlocked profile, before Neow.
    pub fn new(seed: &str, acts: [Act; 3], ascension: Ascension, unlocks: &Unlocks) -> Self {
        let mut rngs = RunRngs::new(seed);
        let plan = RunPlan::generate(rngs.seed, acts, ascension, unlocks);
        rngs.run(RunStream::UpFront).fast_forward(plan.up_front);
        Self {
            rngs,
            ascension,
            plan,
            act: 0,
            floor: 0,
            visited: [Visited::default(); 3],
            events_seen: Vec::new(),
            unknown_odds: UnknownOdds::default(),
            card_odds: CardOdds::default(),
            potion_odds: PotionOdds::default(),
            gold: 99,
            hp: 80,
            max_hp: 80,
            deck: Self::starting_deck(ascension),
            relics: vec![RunRelic::new("BURNING_BLOOD")],
            potions: vec![None; if ascension.has(AscensionLevel::TightBelt) { 2 } else { 3 }],
            point: None,
            room: None,
            shop_removals: 0,
        }
    }

    /// `Ironclad.StartingDeck`, and Ascender's Bane from A5
    /// (`AscensionManager.ApplyEffectsTo`, which also takes a potion slot
    /// at Tight Belt).
    fn starting_deck(ascension: Ascension) -> Vec<DeckCard> {
        let mut deck: Vec<DeckCard> = ["STRIKE_IRONCLAD"; 5]
            .into_iter()
            .chain(["DEFEND_IRONCLAD"; 4])
            .chain(["BASH"])
            .map(DeckCard::new)
            .collect();
        if ascension.has(AscensionLevel::AscendersBane) {
            deck.push(DeckCard::new("ASCENDERS_BANE"));
        }
        deck
    }

    pub fn has_relic(&self, id: &str) -> bool {
        self.relics.iter().any(|r| r.id == id)
    }

    pub fn relic_mut(&mut self, id: &str) -> Option<&mut RunRelic> {
        self.relics.iter_mut().find(|r| r.id == id)
    }

    /// The potions held, slot order.
    pub fn held_potions(&self) -> impl Iterator<Item = &str> {
        self.potions.iter().flatten().map(String::as_str)
    }

    /// `RelicCmd.Obtain`'s bookkeeping: the relic leaves both bags. What
    /// it does on pickup is `obtain`'s.
    pub fn obtain_relic(&mut self, id: &str) {
        if let Some(relic) = crate::plan::BagRelic::from_game_id(id) {
            self.plan.player_bag.remove(relic);
            self.plan.shared_bag.remove(relic);
        }
        self.relics.push(RunRelic::new(id));
    }

    /// `RunManager.SetActInternal`: a new act starts its unknown odds over.
    pub fn enter_act(&mut self, act: usize) {
        self.act = act;
        self.unknown_odds.reset();
    }

    /// `UnknownMapPointOdds.Roll` on the run's UnknownMapPoint stream.
    /// `shop_banned` is `BuildRoomTypeBlacklist`'s answer; Juzu Bracelet
    /// takes monsters out (`ModifyUnknownMapPointRoomTypes`).
    pub fn roll_unknown(&mut self, shop_banned: bool) -> RoomType {
        let juzu = self.has_relic("JUZU_BRACELET");
        let allowed = |room: RoomType| !(shop_banned && room == RoomType::Shop) && !(juzu && room == RoomType::Monster);
        self.unknown_odds.roll(allowed, self.rngs.run(RunStream::UnknownMapPoint))
    }

    /// `RunManager.EnterMapCoord` onto `point` of the act's `map`: its shop
    /// blacklist (`BuildRoomTypeBlacklist`: the last room was a shop, or
    /// every child of the point is one), then `enter_point`.
    pub fn enter(&mut self, map: &ActMap, point: PointId) -> Room {
        let children = &map[point].children;
        let all_shops = !children.is_empty() && children.iter().all(|c| map[c].kind == PointType::Shop);
        let banned = self.room == Some(Room::Shop) || all_shops;
        self.point = Some(point);
        self.enter_point(map[point].kind, banned)
    }

    /// `EnterMapPointInternal` into a point of type `point`: the room type
    /// it resolves to, the room `CreateRoom` makes of it (pulling the act's
    /// next encounter or event), `MarkRoomVisited`, then what entering the
    /// room does (`room_entered`). The floor counts the point once the room
    /// exists, so an event's `IsAllowed` sees the floors before it.
    /// `shop_banned` is the point's shop blacklist; the history check,
    /// which does not know the point, works it out.
    pub fn enter_point(&mut self, point: PointType, shop_banned: bool) -> Room {
        let kind = match point {
            PointType::Unknown => self.roll_unknown(shop_banned),
            PointType::Shop => RoomType::Shop,
            PointType::Treasure => RoomType::Treasure,
            PointType::RestSite => RoomType::RestSite,
            PointType::Monster => RoomType::Monster,
            PointType::Elite => RoomType::Elite,
            PointType::Boss => RoomType::Boss,
            PointType::Ancient => RoomType::Event,
            PointType::Unassigned => panic!("an unassigned map point"),
        };
        let plan = &self.plan.acts[self.act];
        let visited = self.visited[self.act];
        let room = match kind {
            RoomType::Monster => Room::Combat(kind, plan.normal[visited.normal % plan.normal.len()]),
            RoomType::Elite => Room::Combat(kind, plan.elites[visited.elites % plan.elites.len()]),
            RoomType::Boss => {
                let boss = if visited.bosses > 0 { plan.second_boss.unwrap_or(plan.boss) } else { plan.boss };
                Room::Combat(kind, boss)
            }
            RoomType::Event if point == PointType::Ancient => Room::Ancient(plan.ancient.expect("an unlocked ancient")),
            RoomType::Event => Room::Event(self.pull_next_event()),
            RoomType::Treasure => Room::Treasure,
            RoomType::Shop => Room::Shop,
            RoomType::RestSite => Room::RestSite,
        };
        self.floor += 1;
        let visited = &mut self.visited[self.act];
        match kind {
            RoomType::Monster => visited.normal += 1,
            RoomType::Elite => visited.elites += 1,
            RoomType::Event => visited.events += 1,
            RoomType::Boss => visited.bosses += 1,
            _ => {}
        }
        self.room = Some(room);
        self.room_entered(room, point == PointType::Unknown);
        room
    }

    /// `ActModel.PullNextEvent`: `RoomSet.EnsureNextEventIsValid` skips, at
    /// most once round the list, the events the run does not allow or has
    /// seen, then the event counts as seen.
    fn pull_next_event(&mut self) -> &'static str {
        let events = &self.plan.acts[self.act].events;
        let next = |visited: usize| events[visited % events.len()];
        for _ in 0..events.len() {
            let event = next(self.visited[self.act].events);
            if self.event_allowed(event) && !self.events_seen.contains(&event) {
                break;
            }
            self.visited[self.act].events += 1;
        }
        let event = next(self.visited[self.act].events);
        self.events_seen.push(event);
        event
    }

    /// The relics held that `RelicModel.IsTradable`, as indices in the
    /// order they came: a Common, Uncommon, Rare or Shop relic with no
    /// pickup effect that is not used up (Lizard Tail's revive, Maw Bank
    /// once something is bought, Winged Boots' three flights). Neow's, the
    /// events' and the ancients' relics are neither rarity.
    pub(crate) fn tradable_relics(&self) -> Vec<usize> {
        const UPON_PICKUP: &[&str] = &[
            "CALLING_BELL", "CAULDRON", "DOLLYS_MIRROR", "GNARLED_HAMMER", "KIFUDA", "LEES_WAFFLE", "MANGO", "OLD_COIN", "PEAR",
            "POTION_BELT", "PUNCH_DAGGER", "ROYAL_STAMP", "STRAWBERRY", "TOY_BOX", "WAR_PAINT", "WHETSTONE", "ORRERY",
            "SEA_GLASS", "LOST_COFFER",
        ];
        use crate::types::RelicRarity::{Common, Rare, Shop, Uncommon};
        let used_up = |r: &RunRelic| match r.id.as_str() {
            "LIZARD_TAIL" | "MAW_BANK" => r.flag,
            "WINGED_BOOTS" => r.counter >= 3,
            _ => false,
        };
        (0..self.relics.len())
            .filter(|&i| {
                let r = &self.relics[i];
                let rarity = crate::plan::BagRelic::from_game_id(&r.id).map(|b| b.rarity());
                !UPON_PICKUP.contains(&r.id.as_str()) && !used_up(r) && matches!(rarity, Some(Common | Uncommon | Rare | Shop))
            })
            .collect()
    }

    /// `EventModel.IsAllowed` for a singleplayer run. Where it asks whether
    /// a card can take an enchantment (Grave of the Forgotten, Field of
    /// Man-Sized Holes, Spiraling Whirlpool) the answer is taken as yes: an
    /// Ironclad deck always holds a card that can.
    pub fn event_allowed(&self, event: &str) -> bool {
        let (act, gold, hp, floor) = (self.act, self.gold, self.hp, self.floor);
        let potions = self.held_potions().count();
        let removable = || self.deck.iter().filter(|c| c.removable());
        let basics = |id: &str| removable().filter(|c| c.id == id).count();
        match event {
            "ColossalFlower" => hp >= 19,
            "PunchOff" => floor >= 6,
            "SlipperyBridge" => floor > 6 && removable().next().is_some(),
            "WoodCarvings" => removable().any(DeckCard::basic),
            "Amalgamator" => basics("STRIKE_IRONCLAD") >= 2 && basics("DEFEND_IRONCLAD") >= 2,
            "ByrdonisNest" => !self.deck.iter().any(|c| c.id == "BYRDONIS_EGG") && !self.has_relic("BYRDPIP") && !self.has_relic("PAELS_LEGION"),
            "WelcomeToWongos" => act == 1 && gold >= 100,
            "PotionCourier" | "Symbiote" => act > 0,
            "TheLegendsWereTrue" => act == 0 && !self.deck.is_empty() && hp >= 10,
            "WaterloggedScriptorium" => gold >= 55,
            "DollRoom" => act == 1,
            "RoundTeaParty" => hp >= 12,
            "WarHistorianRepy" => false,
            "RoomFullOfCheese" | "BrainLeech" => act < 2,
            "TheFutureOfPotions" => potions >= 2,
            "RelicTrader" => act > 0 && self.tradable_relics().len() >= 5,
            "StoneOfAllTime" => act == 1 && potions > 0,
            "UnrestSite" => hp as f64 <= self.max_hp as f64 * 0.7,
            "ZenWeaver" => gold >= 125,
            "FakeMerchant" => act >= 1 && (gold >= 100 || self.held_potions().any(|p| p == "FOUL_POTION")),
            "MorphicGrove" => gold >= 100 && removable().count() >= 2,
            "TrashHeap" => hp > 5,
            "TeaMaster" => act < 2 && gold >= 150,
            "WhisperingHollow" => gold >= 44,
            "CrystalSphere" => gold >= 100 && act > 0,
            "LuminousChoir" => gold >= 149,
            "RanwidTheElder" => act > 0 && !self.tradable_relics().is_empty() && gold >= 100 && potions > 0,
            "EndlessConveyor" => gold >= 120,
            _ => true,
        }
    }
}

impl RunState {
    /// A fight against `enemies` of `encounter`, from the run as it stands:
    /// its deck, relics with their counters, potion slots, HP and gold in
    /// the combat sim's ids. Relics that do nothing in a fight
    /// (`gen::INERT_RELICS`) stay out. An error names what the sim lacks, or
    /// a card it cannot play (`UNSUPPORTED_CARDS`).
    pub fn fight_setup(&self, encounter: Encounter, enemies: Vec<EnemySpec>) -> Result<FightSetup, String> {
        let deck = self
            .deck
            .iter()
            .map(|c| {
                let id = sim_card(&c.id).ok_or_else(|| format!("unknown card {}", c.id))?;
                if let Some((_, why)) = UNSUPPORTED_CARDS.iter().find(|(u, _)| *u == id) {
                    return Err(format!("unsupported card {id:?}: {why}"));
                }
                let mut card = Card::new(0, id, c.upgraded);
                if let Some(e) = &c.enchantment {
                    let ench = sim_enchantment(&e.id).ok_or_else(|| format!("unknown enchantment {}", e.id))?;
                    card.attach(Enchantment::new(ench, e.amount));
                }
                Ok(card)
            })
            .collect::<Result<Vec<Card>, String>>()?;
        let relics = self
            .relics
            .iter()
            .filter(|r| !INERT_RELICS.contains(&r.id.as_str()))
            .map(|r| {
                let id = sim_relic(&r.id).ok_or_else(|| format!("unknown relic {}", r.id))?;
                Ok(Relic { counter: r.counter, flag: r.flag, ..Relic::new(id) })
            })
            .collect::<Result<Vec<Relic>, String>>()?;
        let potions = self
            .potions
            .iter()
            .map(|p| p.as_deref().map(|id| sim_potion(id).ok_or_else(|| format!("unknown potion {id}"))).transpose())
            .collect::<Result<_, String>>()?;
        let room = match encounter.kind() {
            Kind::Weak | Kind::Normal => RoomKind::Monster,
            Kind::Elite => RoomKind::Elite,
            Kind::Boss => RoomKind::Boss,
        };
        Ok(FightSetup {
            deck,
            hp: self.hp,
            max_hp: self.max_hp,
            max_energy: crate::IRONCLAD_ENERGY,
            relics,
            potions,
            enemies,
            encounter,
            room,
            // The plan drew the last act's second boss apart from its first.
            after: crate::gen::after(encounter, self.ascension, self.plan.acts[self.act].second_boss == Some(encounter)),
            asc: self.ascension,
            floor: self.floor as u32,
            gold: self.gold,
        })
    }

    /// Writes a fight `setup` started back into the run once it is over:
    /// HP, max HP, gold, the potion slots and the relics' counters. The sim
    /// never changes the deck, and its post-victory heals (Burning Blood)
    /// are already in the HP. Returns `CombatRoom.GoldProportion` for the
    /// rewards (`EncounterModel.CalculateGoldProportion`): the share of the
    /// monsters that did not escape; Gremlin Merc's none if its Fat Gremlin
    /// fled with stolen gold, half if with none.
    pub fn end_fight(&mut self, setup: &FightSetup, combat: &Combat) -> f32 {
        self.hp = combat.player.creature.hp.max(0);
        self.max_hp = combat.player.creature.max_hp;
        self.gold = combat.gold;
        self.potions = combat.potions.iter().map(|p| p.map(|id| slug(&format!("{id:?}")))).collect();
        for relic in &combat.relics {
            if let Some(held) = self.relic_mut(&slug(&format!("{:?}", relic.id))) {
                held.counter = relic.counter;
                held.flag = relic.flag;
            }
        }
        let escaped: Vec<MonsterId> = combat.enemies.iter().filter(|e| e.escaped).map(|e| e.monster.id).collect();
        if setup.encounter == Encounter::GremlinMercNormal {
            return match (escaped.contains(&MonsterId::FatGremlin), combat.gold < setup.gold) {
                (false, _) => 1.0,
                (true, false) => 0.5,
                (true, true) => 0.0,
            };
        }
        1.0 - escaped.len() as f32 / setup.enemies.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::Ids;
    use serde_json::Value;

    /// A recorder `start` as the run would hold it: the deck, the relics
    /// with the counters the recorder logs (read as `gen` reads them), the
    /// potion slots, HP and gold.
    fn run_of(start: &Value) -> RunState {
        let acts = [Act::Overgrowth, Act::Hive, Act::Glory];
        let mut run = RunState::new("SEED", acts, Ascension(start["ascension"].as_u64().unwrap() as u8), &Unlocks::default());
        run.hp = start["hp"].as_i64().unwrap() as i32;
        run.max_hp = start["max_hp"].as_i64().unwrap() as i32;
        run.gold = start["gold"].as_i64().unwrap() as i32;
        run.deck = start["deck"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| DeckCard {
                id: c["id"].as_str().unwrap().to_string(),
                upgraded: c["up"] == true,
                enchantment: c["ench"].as_array().map(|e| Enchant { id: e[0].as_str().unwrap().to_string(), amount: e[1].as_i64().unwrap() as i32 }),
            })
            .collect();
        run.relics = start["relics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let mut relic = RunRelic::new(r.as_str().unwrap());
                if let (Some(n), Some(id)) = (start["relic_state"][&relic.id].as_i64(), sim_relic(&relic.id)) {
                    relic.counter = crate::gen::relic_counter(id, n as i32);
                }
                relic
            })
            .collect();
        run.potions = start["potions"].as_array().unwrap().iter().map(|p| p.as_str().map(str::to_string)).collect();
        run
    }

    /// Three recorded starts (the older two with HP from their first
    /// snapshot): an enchanted card, inert relics, Happy Flower's count and
    /// Petrified Toad's rock among them. The fight the run builds holds what
    /// `FightSetup::from_start` builds from the record, except that the
    /// rock stays, since only the recorder logs it twice.
    #[test]
    fn builds_the_fight_a_recording_starts() {
        let ids = Ids::new();
        for line in include_str!("../testdata/fight-starts.jsonl").lines() {
            let start: Value = serde_json::from_str(line).unwrap();
            let recorded = FightSetup::from_start(&start, None, &ids).unwrap();
            let built = run_of(&start).fight_setup(recorded.encounter, recorded.enemies.clone()).unwrap();
            let cards = |s: &FightSetup| s.deck.iter().map(|c| (c.id, c.upgraded, c.enchantment)).collect::<Vec<_>>();
            assert_eq!(cards(&built), cards(&recorded));
            assert_eq!(built.relics, recorded.relics);
            assert_eq!((built.hp, built.max_hp, built.gold, built.max_energy, built.asc, built.room), (recorded.hp, recorded.max_hp, recorded.gold, recorded.max_energy, recorded.asc, recorded.room));
            let mut potions = built.potions.clone();
            if built.relics.iter().any(|r| r.id == crate::relic::RelicId::PetrifiedToad) {
                let rock = potions.iter().position(|&p| p == Some(crate::potion::PotionId::PotionShapedRock)).expect("the Toad's rock");
                potions[rock] = None;
            }
            assert_eq!(potions, recorded.potions);
        }
    }
}
