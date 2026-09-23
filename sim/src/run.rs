//! A run between fights, value for value with the game: the state the
//! run layer carries (`Runs/RunState.cs` and the player's side of it) and
//! the walk from map point to room (`RunManager.EnterMapPointInternal`,
//! `RollRoomTypeFor`, `CreateRoom`, `Odds/UnknownMapPointOdds.cs`,
//! `Rooms/RoomSet.cs`). Rewards live in `rewards.rs`.
//!
//! Cards, relics and potions are game ids here, so the run can hold what the
//! combat sim has no model for.
//!
//! Not here: the first run's tutorial rooms and rewards (a fully unlocked
//! profile never meets them), multiplayer, and the relic hooks on unknown
//! rooms besides Juzu Bracelet's (Golden Compass, the Lantern Key card).

use crate::encounter::{Act, Encounter};
use crate::game_rng::{GameRng, RunRngs, RunStream};
use crate::map::PointType;
use crate::plan::{RunPlan, Unlocks};
use crate::rewards::{CardOdds, PotionOdds};
use crate::types::Ascension;

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
    pub enchantment: Option<String>,
}

impl DeckCard {
    pub fn new(id: &str) -> Self {
        DeckCard { id: id.to_string(), upgraded: false, enchantment: None }
    }

    /// `CardModel.IsRemovable`: not Eternal, which Tezcatara's Ember makes
    /// a card.
    pub fn removable(&self) -> bool {
        self.enchantment.as_deref() != Some("TEZCATARAS_EMBER")
    }

    fn basic(&self) -> bool {
        matches!(self.id.as_str(), "STRIKE_IRONCLAD" | "DEFEND_IRONCLAD" | "BASH")
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
    /// Game ids, in the order they came.
    pub relics: Vec<String>,
    pub potions: Vec<String>,
    /// Lasting Candy's `CombatsSeen`, once it is held.
    pub lasting_candy_fights: Option<u32>,
    /// Lava Rock's `HasTriggered`.
    pub lava_rock_used: bool,
    /// Silver Crucible's `TimesUsed`.
    pub crucible_used: u32,
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
            relics: vec!["BURNING_BLOOD".into()],
            potions: Vec::new(),
            lasting_candy_fights: None,
            lava_rock_used: false,
            crucible_used: 0,
        }
    }

    /// `Ironclad.StartingDeck`, and Ascender's Bane from A5
    /// (`AscensionManager.ApplyEffectsTo`).
    fn starting_deck(ascension: Ascension) -> Vec<DeckCard> {
        let mut deck: Vec<DeckCard> = ["STRIKE_IRONCLAD"; 5]
            .into_iter()
            .chain(["DEFEND_IRONCLAD"; 4])
            .chain(["BASH"])
            .map(DeckCard::new)
            .collect();
        if ascension.has(crate::types::AscensionLevel::AscendersBane) {
            deck.push(DeckCard::new("ASCENDERS_BANE"));
        }
        deck
    }

    pub fn has_relic(&self, id: &str) -> bool {
        self.relics.iter().any(|r| r == id)
    }

    /// `RelicCmd.Obtain`'s bookkeeping: the relic leaves both bags. What
    /// it does on pickup is the caller's.
    pub fn obtain_relic(&mut self, id: &str) {
        if let Some(relic) = crate::plan::BagRelic::from_game_id(id) {
            self.plan.player_bag.remove(relic);
            self.plan.shared_bag.remove(relic);
        }
        if id == "LASTING_CANDY" {
            self.lasting_candy_fights = Some(0);
        }
        self.relics.push(id.to_string());
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

    /// `EnterMapPointInternal` up to entering the room: the room type the
    /// point resolves to, the room `CreateRoom` makes of it (pulling the
    /// act's next encounter or event), then `MarkRoomVisited`. The floor
    /// counts the point once the room exists, so an event's `IsAllowed`
    /// sees the floors before it.
    pub fn enter(&mut self, point: PointType, shop_banned: bool) -> Room {
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

    /// `RelicModel.IsTradable` over the relics held: a Common, Uncommon,
    /// Rare or Shop relic with no pickup effect. Neow's, the events' and the
    /// ancients' relics are neither. A Lizard Tail is taken as unused.
    fn tradable_relics(&self) -> usize {
        const UPON_PICKUP: &[&str] = &[
            "CALLING_BELL", "CAULDRON", "DOLLYS_MIRROR", "GNARLED_HAMMER", "KIFUDA", "LEES_WAFFLE", "MANGO", "OLD_COIN", "PEAR",
            "POTION_BELT", "PUNCH_DAGGER", "ROYAL_STAMP", "STRAWBERRY", "TOY_BOX", "WAR_PAINT", "WHETSTONE", "ORRERY",
            "SEA_GLASS", "LOST_COFFER",
        ];
        use crate::types::RelicRarity::{Common, Rare, Shop, Uncommon};
        self.relics
            .iter()
            .filter(|id| !UPON_PICKUP.contains(&id.as_str()))
            .filter_map(|id| crate::plan::BagRelic::from_game_id(id))
            .filter(|r| matches!(r.rarity(), Common | Uncommon | Rare | Shop))
            .count()
    }

    /// `EventModel.IsAllowed` for a singleplayer run. Where it asks whether
    /// a card can take an enchantment (Grave of the Forgotten, Field of
    /// Man-Sized Holes, Spiraling Whirlpool) the answer is taken as yes: an
    /// Ironclad deck always holds a card that can.
    pub fn event_allowed(&self, event: &str) -> bool {
        let (act, gold, hp, floor) = (self.act, self.gold, self.hp, self.floor);
        let potions = self.potions.len();
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
            "RelicTrader" => act > 0 && self.tradable_relics() >= 5,
            "StoneOfAllTime" => act == 1 && potions > 0,
            "UnrestSite" => hp as f64 <= self.max_hp as f64 * 0.7,
            "ZenWeaver" => gold >= 125,
            "FakeMerchant" => act >= 1 && (gold >= 100 || self.potions.iter().any(|p| p == "FOUL_POTION")),
            "MorphicGrove" => gold >= 100 && removable().count() >= 2,
            "TrashHeap" => hp > 5,
            "TeaMaster" => act < 2 && gold >= 150,
            "WhisperingHollow" => gold >= 44,
            "CrystalSphere" => gold >= 100 && act > 0,
            "LuminousChoir" => gold >= 149,
            "RanwidTheElder" => act > 0 && self.tradable_relics() > 0 && gold >= 100 && potions > 0,
            "EndlessConveyor" => gold >= 120,
            _ => true,
        }
    }
}
