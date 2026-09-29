//! Events (`Models/Events/<Name>.cs`) from entering to leaving: what each
//! lays out, on its own stream (`EventModel.Rng`, seeded from the run's seed
//! plus the hash of the event's id) or the player's, and what each option
//! does, with every choice put to a `Chooser` (`Decision::Event`, then the
//! deck picks and rewards an option opens). An event is written as the
//! game's is, one function per event, over the effect layer's commands
//! (`damage`, `gain_gold`, `take`, `add_card`, the deck picks) and a few of
//! its own (`Ev`).
//!
//! An event that starts a fight hands it back (`EventFight`) for the run to
//! play like any other; what winning gives is `event_fight_won`'s. An event
//! not ported is entered and left, and `event` says so.

use crate::combat::EnemySpec;
use crate::effects::{DeckAction, Offered, TransformRng};
use crate::encounter::Encounter;
use crate::ids::MonsterId;
use crate::rng::Rng;
use crate::game_rng::{hash, GameRng, PlayerStream};
use crate::plan::BagRelic;
use crate::pools::{PoolCard, PoolPotion, PotionRarity, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS, IRONCLAD_POTIONS, SHARED_POTIONS};
use crate::replay::slug;
use crate::card::def;
use crate::pools::sim_card;
use crate::rewards::{create_cards, create_potion, CardOptions, OddsType, Offer, Pulled};
use crate::shop::{Item, Slot, Ware};
use crate::rooms::{Chooser, Decision};
use crate::run::{DeckCard, Room, RunState};
use crate::types::{CardType, RelicRarity};

/// An option an event laid out (`EventOption`), by the text key the record
/// keeps: `ROOM_FULL_OF_CHEESE.pages.INITIAL.options.GORGE` is page
/// `INITIAL`, key `GORGE`. An option titled by a relic (the Doll Room's
/// dolls) has no page and the relic's id for its key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventOption {
    pub page: &'static str,
    pub key: &'static str,
    /// What the layout drew that the option names, as game ids: a potion,
    /// a relic held and the one offered for it, a card.
    pub items: Vec<String>,
    /// Shown but not choosable (`IsLocked`, no `OnChosen`).
    pub locked: bool,
}

impl EventOption {
    /// An option on the first page.
    fn new(key: &'static str) -> Self {
        Self::on("INITIAL", key)
    }

    fn on(page: &'static str, key: &'static str) -> Self {
        EventOption { page, key, items: Vec::new(), locked: false }
    }

    /// A locked option on the first page, by the key the game gives it
    /// (`APPROACH_LOCKED` in place of `APPROACH`).
    fn locked(key: &'static str) -> Self {
        EventOption { locked: true, ..Self::new(key) }
    }

    /// `key` if `open`, else the locked option `locked` in its place.
    fn or_locked(open: bool, key: &'static str, locked: &'static str) -> Self {
        if open { Self::new(key) } else { Self::locked(locked) }
    }

    fn naming(mut self, item: impl Into<String>) -> Self {
        self.items.push(item.into());
        self
    }
}

/// A fight an event started (`EventModel.EnterCombatWithoutExitingEvent`),
/// which the run plays as a monster room.
#[derive(Clone, Debug, PartialEq)]
pub struct EventFight {
    pub encounter: Encounter,
    /// Its monsters were created as the event was entered (a combat layout:
    /// Punch Off, The Lantern Key), which drew on the run's Niche stream
    /// then, so starting the fight draws nothing more.
    pub created: bool,
    /// `CombatRoom.ExtraRewards`, populated after the fight's own.
    pub extra: Vec<Extra>,
    /// Battleworn Dummy's setting (1 to 3): which dummy it fights and what
    /// beating it gives (`Resume`), in place of any rewards.
    pub dummy: Option<u8>,
    /// Gold the encounter gives in place of a monster room's
    /// (`FakeMerchantEventEncounter.MinGoldReward`).
    pub gold: Option<i32>,
}

impl EventFight {
    fn new(encounter: Encounter) -> Self {
        EventFight { encounter, created: false, extra: Vec::new(), dummy: None, gold: None }
    }

    /// The fight's enemies: the dummy the setting names
    /// (`BattlewornDummyEventEncounter.GenerateMonsters`), else the
    /// encounter's, rolled on the sim's `rng`.
    pub fn enemies(&self, rng: &mut Rng) -> Vec<EnemySpec> {
        match self.dummy {
            Some(setting) => {
                let dummy = [MonsterId::BattleFriendV1, MonsterId::BattleFriendV2, MonsterId::BattleFriendV3][setting as usize - 1];
                vec![EnemySpec { id: dummy, flags: Default::default() }]
            }
            None => self.encounter.monsters(rng),
        }
    }
}

/// A reward an event adds to its fight's (`Reward` subclasses).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Extra {
    /// `RelicReward(player)`: a rarity roll, then the front of the bag.
    Relic,
    /// `PotionReward(player)`: a random potion.
    Potion,
    /// A given relic as a reward.
    NamedRelic(String),
    /// `SpecialCardReward`: one given card to take or leave.
    Card(&'static str),
}

/// An event walked through: every page it laid out, in order, the fight
/// it started, if it did, and how many draws it made on its own stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Visit {
    pub pages: Vec<Vec<EventOption>>,
    pub fight: Option<EventFight>,
    pub draws: u32,
}

/// What an event does, from its layout to the fight it starts, if any.
type Play = fn(&mut RunState, &mut Ev) -> Option<EventFight>;

/// The events ported, by class name.
const EVENTS: &[(&str, Play)] = &[
    ("AbyssalBaths", abyssal_baths),
    ("Amalgamator", amalgamator),
    ("AromaOfChaos", aroma_of_chaos),
    ("BattlewornDummy", battleworn_dummy),
    ("BrainLeech", brain_leech),
    ("Bugslayer", bugslayer),
    ("ByrdonisNest", byrdonis_nest),
    ("ColossalFlower", colossal_flower),
    ("DenseVegetation", dense_vegetation),
    ("DollRoom", doll_room),
    ("DoorsOfLightAndDark", doors_of_light_and_dark),
    ("DrowningBeacon", drowning_beacon),
    ("EndlessConveyor", endless_conveyor),
    ("FakeMerchant", fake_merchant),
    ("FieldOfManSizedHoles", field_of_man_sized_holes),
    ("GraveOfTheForgotten", grave_of_the_forgotten),
    ("HungryForMushrooms", hungry_for_mushrooms),
    ("InfestedAutomaton", infested_automaton),
    ("JungleMazeAdventure", jungle_maze_adventure),
    ("LostWisp", lost_wisp),
    ("LuminousChoir", luminous_choir),
    ("MorphicGrove", morphic_grove),
    ("PotionCourier", potion_courier),
    ("PunchOff", punch_off),
    ("RanwidTheElder", ranwid_the_elder),
    ("Reflections", reflections),
    ("RelicTrader", relic_trader),
    ("RoomFullOfCheese", room_full_of_cheese),
    ("RoundTeaParty", round_tea_party),
    ("SapphireSeed", sapphire_seed),
    ("SelfHelpBook", self_help_book),
    ("SlipperyBridge", slippery_bridge),
    ("SpiralingWhirlpool", spiraling_whirlpool),
    ("SpiritGrafter", spirit_grafter),
    ("StoneOfAllTime", stone_of_all_time),
    ("SunkenStatue", sunken_statue),
    ("SunkenTreasury", sunken_treasury),
    ("Symbiote", symbiote),
    ("TabletOfTruth", tablet_of_truth),
    ("TeaMaster", tea_master),
    ("TheFutureOfPotions", the_future_of_potions),
    ("TheLanternKey", the_lantern_key),
    ("TheLegendsWereTrue", the_legends_were_true),
    ("ThisOrThat", this_or_that),
    ("TrashHeap", trash_heap),
    ("Trial", trial),
    ("UnrestSite", unrest_site),
    ("WaterloggedScriptorium", waterlogged_scriptorium),
    ("WelcomeToWongos", welcome_to_wongos),
    ("Wellspring", wellspring),
    ("WhisperingHollow", whispering_hollow),
    ("WoodCarvings", wood_carvings),
    ("ZenWeaver", zen_weaver),
];

/// The class names of the events ported.
pub fn ported() -> impl Iterator<Item = &'static str> {
    EVENTS.iter().map(|&(name, _)| name)
}

/// An event under way: its own stream, the pages laid out so far, and the
/// run's chooser and log, which every option's choices go through.
struct Ev<'a> {
    name: &'static str,
    rng: GameRng,
    pages: Vec<Vec<EventOption>>,
    chooser: &'a mut dyn Chooser,
    log: &'a mut Vec<Offered>,
}

impl Ev<'_> {
    /// Lays out a page and returns the index of the option chosen among
    /// all of them; only the open ones go to the chooser. The first page
    /// is when the relics' `AfterRoomEntered` runs (`EventRoom.EnterInternal`
    /// begins the event, which lays it out, then calls the hook).
    fn ask_at(&mut self, run: &mut RunState, options: Vec<EventOption>) -> usize {
        if self.pages.is_empty() {
            run.relics_entered(Room::Event(self.name), true);
        }
        let open: Vec<usize> = (0..options.len()).filter(|&i| !options[i].locked).collect();
        let shown: Vec<EventOption> = open.iter().map(|&i| options[i].clone()).collect();
        let i = self.chooser.choose(run, Decision::Event { event: self.name, options: &shown });
        self.pages.push(options);
        open[i.min(open.len() - 1)]
    }

    /// `ask_at`, returning the chosen option's key.
    fn ask(&mut self, run: &mut RunState, options: Vec<EventOption>) -> &'static str {
        let i = self.ask_at(run, options);
        self.pages.last().expect("a page")[i].key
    }

    fn settle(&mut self, run: &mut RunState, offered: Vec<Offered>) {
        run.settle(offered, &mut self.chooser, self.log);
    }

    /// `RelicCmd.Obtain` of a given relic, its pickup and all.
    fn take(&mut self, run: &mut RunState, relic: impl Into<String>) {
        let offered = run.take(relic.into());
        self.settle(run, offered);
    }

    /// `RelicFactory.PullNextRelicFromFront`, obtained.
    fn take_from_bag(&mut self, run: &mut RunState) {
        let relic = run.relic_reward().game_id();
        self.take(run, relic);
    }

    /// Cards picked out of `cards` (deck indices, in the order the screen
    /// shows them) for `action`, one at a time, at least `min` of them
    /// while any are left and at most `max`. Returns them, not acted on.
    fn pick(&mut self, run: &RunState, action: DeckAction, mut cards: Vec<usize>, min: usize, max: usize) -> Vec<usize> {
        let mut chosen = Vec::new();
        while chosen.len() < max && !cards.is_empty() {
            let optional = chosen.len() >= min;
            let i = self.chooser.choose(run, Decision::Deck { action, cards: &cards, optional });
            if i >= cards.len() {
                break;
            }
            chosen.push(cards.remove(i));
        }
        chosen
    }

    /// `CardSelectCmd.FromDeckForEnchantment` of `count` cards, then
    /// `CardCmd.Enchant` with `amount`.
    fn enchant(&mut self, run: &mut RunState, id: &'static str, amount: i32, count: usize) {
        let action = DeckAction::Enchant(id, amount);
        let chosen = self.pick(run, action, run.pickable(action), count, count);
        run.apply_pick(action, &chosen);
    }

    /// `CardSelectCmd.FromDeckForUpgrade`, then `CardCmd.Upgrade`.
    fn upgrade(&mut self, run: &mut RunState, count: usize) {
        let cards = run.pickable(DeckAction::Upgrade);
        let chosen = self.pick(run, DeckAction::Upgrade, cards, count, count);
        run.apply_pick(DeckAction::Upgrade, &chosen);
    }

    /// `CardSelectCmd.FromDeckForRemoval` among the cards `keep` passes,
    /// then `CardPileCmd.RemoveFromDeck`.
    fn remove(&mut self, run: &mut RunState, count: usize, keep: impl Fn(&DeckCard) -> bool) {
        let cards = run.pickable(DeckAction::Remove).into_iter().filter(|&i| keep(&run.deck[i])).collect();
        let chosen = self.pick(run, DeckAction::Remove, cards, count, count);
        run.apply_pick(DeckAction::Remove, &chosen);
    }

    /// `CardSelectCmd.FromDeckForTransformation`, then each card
    /// `CardCmd.TransformToRandom` on the event's stream.
    fn transform(&mut self, run: &mut RunState, count: usize) {
        let action = DeckAction::Transform { upgrade: false };
        let cards = run.pickable(action);
        let mut chosen = self.pick(run, action, cards, count, count);
        chosen.sort_unstable();
        run.transform(&chosen, false, TransformRng::Event(&mut self.rng));
    }

    /// `CardSelectCmd.FromSimpleGridForRewards`: `count` of `cards` taken,
    /// one at a time, each added to the deck. The screen cannot be left
    /// (`Cancelable` is off), so a skip takes the first card left.
    fn grid(&mut self, run: &mut RunState, mut cards: Vec<Offer>, count: usize) {
        self.log.push(Offered::Cards(cards.clone()));
        for _ in 0..count.min(cards.len()) {
            let i = self.chooser.choose(run, Decision::Card(&cards));
            let card = cards.remove(if i < cards.len() { i } else { 0 });
            run.add_card(DeckCard::from(card));
        }
    }
}

/// A pool card's `CardRarity`, if the pools hold it.
fn rarity(id: &str) -> Option<Rarity> {
    IRONCLAD_CARDS.iter().chain(COLORLESS_CARDS).find(|c| c.id == id).map(|c| c.rarity)
}

/// The character's and the shared potions, in the pools' order
/// (`GetUnlockedPotions`).
fn potions() -> impl Iterator<Item = &'static PoolPotion> {
    IRONCLAD_POTIONS.iter().chain(SHARED_POTIONS)
}

/// `PotionModel.Rarity` of a potion that can be held, as The Future of
/// Potions reads it. The event potions sit outside the pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeldRarity {
    Pool(PotionRarity),
    /// Foul Potion, Glowwater Potion.
    Event,
    /// Potion-Shaped Rock.
    Token,
}

fn potion_rarity(id: &str) -> HeldRarity {
    match potions().find(|p| p.id == id) {
        Some(p) => HeldRarity::Pool(p.rarity),
        None if id == "POTION_SHAPED_ROCK" => HeldRarity::Token,
        None => HeldRarity::Event,
    }
}

impl RunState {
    /// The event `name` (a class name) from entering to leaving, its
    /// choices put to `chooser` and what it offered appended to `log`.
    /// `None` if the event is not ported: it is then entered and left, with
    /// only the relics' `AfterRoomEntered`.
    pub fn event(&mut self, name: &'static str, chooser: &mut impl Chooser, log: &mut Vec<Offered>) -> Option<Visit> {
        let Some(&(_, play)) = EVENTS.iter().find(|(event, _)| *event == name) else {
            self.relics_entered(Room::Event(name), true);
            return None;
        };
        let rng = GameRng::new(self.rngs.seed.wrapping_add(hash(&slug(name)) as u32));
        let mut ev = Ev { name, rng, pages: Vec::new(), chooser, log };
        let fight = play(self, &mut ev);
        Some(Visit { pages: ev.pages, fight, draws: ev.rng.counter })
    }

    /// `RelicFactory.PullNextRelicFromFront` for a rarity, of the relics a
    /// shop may sell.
    fn pull_for_shop(&mut self, rarity: RelicRarity) -> Pulled {
        match self.plan.player_bag.pull_front(rarity, self.floor, BagRelic::allowed_in_shops) {
            Some(relic) => {
                self.plan.shared_bag.remove(relic);
                Pulled::Bag(relic)
            }
            None => Pulled::Circlet,
        }
    }

    /// A `CardReward` an event offers, of three cards with
    /// `ForNonCombatWithDefaultOdds`: as a card reward, Silver Crucible
    /// upgrades it too.
    fn event_card_reward(&mut self, pool: &'static [PoolCard]) -> Vec<Offer> {
        let mut cards = self.event_cards(3, pool, OddsType::Regular, |_| true);
        self.modify_card_reward(&mut cards);
        cards
    }

    /// `CardFactory.CreateForReward` with `ForNonCombatWithDefaultOdds`
    /// (Regular odds) or `ForNonCombatWithUniformOdds`, over the pool's
    /// cards `keep` takes: no source, so the odds do not move, and no
    /// upgrade roll. The eggs still upgrade.
    fn event_cards(&mut self, count: usize, pool: &'static [PoolCard], odds: OddsType, keep: impl Fn(&PoolCard) -> bool) -> Vec<Offer> {
        let options = CardOptions {
            cards: pool.iter().filter(|c| keep(c)).collect(),
            odds,
            encounter: false,
            upgrade_roll: false,
            card_reward: false,
        };
        let mut card_odds = self.card_odds;
        let mut cards = create_cards(count, &options, &mut card_odds, &mut self.roll_ctx());
        self.upgrade_by_eggs(&mut cards);
        cards
    }

    /// `BattlewornDummy.Resume` once the dummy of `setting` is beaten: a
    /// potion reward; two random upgrades, `StableShuffle` on the event's
    /// stream, which the layout left untouched; or a relic off the front of
    /// the bag.
    pub(crate) fn dummy_beaten(&mut self, setting: u8) -> Vec<Offered> {
        match setting {
            1 => vec![Offered::Potions(vec![self.any_potion().to_string()])],
            2 => {
                let mut rng = GameRng::new(self.rngs.seed.wrapping_add(hash("BATTLEWORN_DUMMY") as u32));
                for i in stable_shuffle(self, &mut rng, DeckCard::upgradable).into_iter().take(2) {
                    self.upgrade_card(i);
                }
                Vec::new()
            }
            _ => {
                let relic = self.relic_reward().game_id();
                self.take(relic)
            }
        }
    }

    /// `PotionReward` of `PlayerRng.Rewards.NextItem` over every potion the
    /// character can find (Wellspring, The Legends Were True, Battleworn
    /// Dummy): a uniform pick, no rarity roll.
    fn any_potion(&mut self) -> &'static str {
        let all: Vec<&PoolPotion> = potions().collect();
        self.rngs.player(PlayerStream::Rewards).pick(&all).expect("a potion").id
    }
}

/// `AbyssalBaths`: Immerse gains 2 max HP and deals 3 damage, one more each
/// time; Linger immerses again until the player leaves (or dies).
fn abyssal_baths(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    if ev.ask(run, vec![EventOption::new("IMMERSE"), EventOption::new("ABSTAIN")]) == "ABSTAIN" {
        run.heal(10);
        return None;
    }
    let mut damage = 3;
    loop {
        run.gain_max_hp(2);
        run.damage(damage);
        damage += 1;
        if run.hp <= 0 {
            return None;
        }
        let page = [EventOption::on("ALL", "LINGER"), EventOption::on("ALL", "EXIT_BATHS")];
        if ev.ask(run, page.to_vec()) == "EXIT_BATHS" {
            return None;
        }
    }
}

/// `BrainLeech`: Share Knowledge shows five of the character's cards to
/// take one of; Rip deals 5 damage for a colorless card reward.
fn brain_leech(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("SHARE_KNOWLEDGE"), EventOption::new("RIP")]) {
        "SHARE_KNOWLEDGE" => {
            let cards = run.event_cards(5, IRONCLAD_CARDS, OddsType::Regular, |_| true);
            ev.grid(run, cards, 1);
        }
        _ => {
            run.damage(5);
            let cards = run.event_card_reward(COLORLESS_CARDS);
            ev.settle(run, vec![Offered::Cards(cards)]);
        }
    }
    None
}

/// `DenseVegetation`: Trudge On deals 8 damage for 61 to 99 gold; Rest
/// heals as a rest site does (`MimicRestSiteHeal`), then wakes a fight.
fn dense_vegetation(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gold = ev.rng.next_int_in(61, 100);
    if ev.ask(run, vec![EventOption::new("TRUDGE_ON"), EventOption::new("REST")]) == "TRUDGE_ON" {
        run.damage(8);
        run.gain_gold(gold);
        return None;
    }
    let offered = run.rest_heal();
    ev.settle(run, offered);
    ev.ask(run, vec![EventOption::on("REST", "FIGHT")]);
    Some(EventFight::new(Encounter::DenseVegetationEventEncounter))
}

/// `LostWisp`: Claim gives the Lost Wisp and a Decay; Search 45 to 75 gold.
fn lost_wisp(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gold = 60 + ev.rng.next_int_in(-15, 16);
    match ev.ask(run, vec![EventOption::new("CLAIM"), EventOption::new("SEARCH")]) {
        "CLAIM" => {
            run.add_card(DeckCard::new("DECAY"));
            ev.take(run, "LOST_WISP");
        }
        _ => run.gain_gold(gold),
    }
    None
}

/// `PotionCourier`: Grab Potions offers three Foul Potions; Ransack an
/// uncommon potion off the Rewards stream.
fn potion_courier(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let potions = match ev.ask(run, vec![EventOption::new("GRAB_POTIONS"), EventOption::new("RANSACK")]) {
        "GRAB_POTIONS" => vec!["FOUL_POTION".to_string(); 3],
        _ => {
            let uncommon: Vec<&PoolPotion> = potions().filter(|p| p.rarity == PotionRarity::Uncommon).collect();
            vec![run.rewards().pick(&uncommon).expect("an uncommon potion").id.to_string()]
        }
    };
    ev.settle(run, vec![Offered::Potions(potions)]);
    None
}

/// `RanwidTheElder`: a random potion, 100 gold, or a random tradable relic
/// for relics off the front of the bag (two for the relic).
fn ranwid_the_elder(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let held: Vec<usize> = (0..run.potions.len()).filter(|&i| run.potions[i].is_some()).collect();
    let potion = ev.rng.pick(&held).copied();
    let tradable = run.tradable_relics();
    let relic = ev.rng.pick(&tradable).map(|&i| run.relics[i].id.clone());
    let options = vec![
        match potion {
            Some(slot) => EventOption::new("POTION").naming(run.potions[slot].clone().expect("a potion")),
            None => EventOption::locked("POTION_LOCKED"),
        },
        EventOption::new("GOLD"),
        match &relic {
            Some(id) => EventOption::new("RELIC").naming(id.clone()),
            None => EventOption::locked("RELIC_LOCKED"),
        },
    ];
    match ev.ask(run, options) {
        "POTION" => {
            run.discard_potion(potion.expect("the potion offered"));
            ev.take_from_bag(run);
        }
        "GOLD" => {
            run.lose_gold(100);
            ev.take_from_bag(run);
        }
        _ => {
            run.remove_relic(&relic.expect("the relic offered"));
            ev.take_from_bag(run);
            ev.take_from_bag(run);
        }
    }
    None
}

/// `RoomFullOfCheese`: Gorge shows eight commons to take two of; Search
/// deals 14 damage for Chosen Cheese.
fn room_full_of_cheese(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("GORGE"), EventOption::new("SEARCH")]) {
        "GORGE" => {
            let cards = run.event_cards(8, IRONCLAD_CARDS, OddsType::Uniform, |c| c.rarity == Rarity::Common);
            ev.grid(run, cards, 2);
        }
        _ => {
            run.damage(14);
            ev.take(run, "CHOSEN_CHEESE");
        }
    }
    None
}

/// `RoundTeaParty`: Enjoy Tea gives Royal Poison and heals to full; Pick
/// Fight deals 11 damage for a relic off the front of the bag.
fn round_tea_party(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("ENJOY_TEA"), EventOption::new("PICK_FIGHT")]) {
        "ENJOY_TEA" => {
            ev.take(run, "ROYAL_POISON");
            run.heal(run.max_hp - run.hp);
        }
        _ => {
            // `ThatWontSaveToChoiceHistory`: the record skips this page.
            ev.ask(run, vec![EventOption::on("PICK_FIGHT", "CONTINUE_FIGHT")]);
            run.damage(11);
            ev.take_from_bag(run);
        }
    }
    None
}

/// `SelfHelpBook`: Sharp 2 on an attack, Nimble 2 on a skill or Swift 2 on
/// a power, each locked without a card to take it.
fn self_help_book(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let books = [("READ_THE_BACK", "SHARP", CardType::Attack), ("READ_PASSAGE", "NIMBLE", CardType::Skill), ("READ_ENTIRE_BOOK", "SWIFT", CardType::Power)];
    let fits = |run: &RunState, id: &str, kind: CardType| run.deck.iter().any(|c| c.kind() == Some(kind) && c.can_enchant(id));
    let open: Vec<bool> = books.iter().map(|&(_, id, kind)| fits(run, id, kind)).collect();
    if !open.contains(&true) {
        ev.ask(run, vec![EventOption::new("NO_OPTIONS")]);
        return None;
    }
    let locked = ["READ_THE_BACK_LOCKED", "READ_PASSAGE_LOCKED", "READ_ENTIRE_BOOK_LOCKED"];
    let options = (0..3).map(|i| EventOption::or_locked(open[i], books[i].0, locked[i])).collect();
    let (_, id, kind) = books[ev.ask_at(run, options)];
    let action = DeckAction::Enchant(id, 2);
    let cards = run.pickable(action).into_iter().filter(|&i| run.deck[i].kind() == Some(kind)).collect();
    let chosen = ev.pick(run, action, cards, 1, 1);
    run.apply_pick(action, &chosen);
    None
}

/// `SlipperyBridge`: Overcome loses a random card; Hold On deals 3 damage,
/// one more each time, and draws another card, of a different kind, until
/// the player lets one go (or dies).
fn slippery_bridge(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    // `GetNewRandomCard`: first a card that is not Basic; then one not of
    // the kind just offered and never offered before; failing either, any
    // removable card.
    let mut skipped: Vec<usize> = Vec::new();
    let mut offered: Option<usize> = None;
    let mut holds = 0;
    // The pages and options Hold On goes through: after the n-th hold the
    // page is `HOLD_ON_{n-1}` and its option `HOLD_ON_{n}`, from the
    // seventh on `LOOP`.
    const HOLD: [&str; 8] = ["HOLD_ON_0", "HOLD_ON_1", "HOLD_ON_2", "HOLD_ON_3", "HOLD_ON_4", "HOLD_ON_5", "HOLD_ON_6", "HOLD_ON_LOOP"];
    loop {
        let deck = &run.deck;
        let mut cards: Vec<usize> = match offered {
            None => (0..deck.len()).filter(|&i| rarity(&deck[i].id) != Some(Rarity::Basic)).collect(),
            Some(last) => {
                skipped.push(last);
                (0..deck.len()).filter(|&i| deck[i].id != deck[last].id).collect()
            }
        };
        cards.retain(|&i| deck[i].removable() && !skipped.contains(&i));
        if cards.is_empty() {
            cards = (0..deck.len()).filter(|&i| deck[i].removable()).collect();
        }
        let lose = *ev.rng.pick(&cards).expect("a removable card");
        offered = Some(lose);
        let page = if holds == 0 { "INITIAL" } else { HOLD[(holds - 1).min(7)] };
        let options = vec![EventOption::new("OVERCOME").naming(run.deck[lose].id.clone()), EventOption::on(page, HOLD[holds.min(7)])];
        if ev.ask_at(run, options) == 0 {
            run.deck.remove(lose);
            return None;
        }
        run.damage(3 + holds as i32);
        holds += 1;
        if run.hp <= 0 {
            return None;
        }
    }
}

/// `SunkenStatue`: Grab Sword gives Sword of Stone; Dive Into Water 101 to
/// 121 gold for 7 damage.
fn sunken_statue(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gold = 111 + ev.rng.next_int_in(-10, 11);
    match ev.ask(run, vec![EventOption::new("GRAB_SWORD"), EventOption::new("DIVE_INTO_WATER")]) {
        "GRAB_SWORD" => ev.take(run, "SWORD_OF_STONE"),
        _ => {
            run.gain_gold(gold);
            run.damage(7);
        }
    }
    None
}

/// `Symbiote`: Approach puts Corrupted on a card (locked if none takes
/// it); Kill With Fire transforms one on the event's stream.
fn symbiote(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let open = run.deck.iter().any(|c| c.can_enchant("CORRUPTED"));
    let options = vec![EventOption::or_locked(open, "APPROACH", "APPROACH_LOCKED"), EventOption::new("KILL_WITH_FIRE")];
    match ev.ask(run, options) {
        "APPROACH" => ev.enchant(run, "CORRUPTED", 1, 1),
        _ => ev.transform(run, 1),
    }
    None
}

/// `TheFutureOfPotions`: each of the first three potions held trades for an
/// upgraded card reward of the rarity it maps to and a type drawn for it
/// (never a power for a common one).
fn the_future_of_potions(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    // `PotionToCardType`, drawn for every potion held on first use.
    let held: Vec<usize> = (0..run.potions.len()).filter(|&i| run.potions[i].is_some()).collect();
    let mut kinds = Vec::new();
    for &slot in &held {
        let id = run.potions[slot].as_deref().expect("a potion");
        let common = matches!(potion_rarity(id), HeldRarity::Pool(PotionRarity::Common) | HeldRarity::Token);
        let types: &[CardType] = if common { &[CardType::Attack, CardType::Skill] } else { &[CardType::Attack, CardType::Skill, CardType::Power] };
        kinds.push(*ev.rng.pick(types).expect("a card type"));
    }
    let rarity_of = |id: &str| match potion_rarity(id) {
        HeldRarity::Pool(PotionRarity::Rare) | HeldRarity::Event => Rarity::Rare,
        HeldRarity::Pool(PotionRarity::Uncommon) => Rarity::Uncommon,
        HeldRarity::Pool(PotionRarity::Common) | HeldRarity::Token => Rarity::Common,
    };
    let options: Vec<EventOption> = held.iter().take(3).map(|&slot| EventOption::new("POTION").naming(run.potions[slot].clone().expect("a potion"))).collect();
    if options.is_empty() {
        // No option: the event is over as it starts.
        run.relics_entered(Room::Event(ev.name), true);
        return None;
    }
    let i = ev.ask_at(run, options);
    let (slot, kind) = (held[i], kinds[i]);
    let target = rarity_of(run.potions[slot].as_deref().expect("a potion"));
    run.discard_potion(slot);
    let mut cards = run.event_cards(3, IRONCLAD_CARDS, OddsType::Uniform, |c| c.rarity == target && c.kind == kind);
    run.modify_card_reward(&mut cards);
    for c in cards.iter_mut().filter(|c| DeckCard::from(**c).upgradable()) {
        c.upgraded = true;
    }
    ev.settle(run, vec![Offered::Cards(cards)]);
    None
}

/// `ThisOrThat`: Plain deals 6 damage for 41 to 68 gold; Ornate gives a
/// relic off the front of the bag and a Clumsy.
fn this_or_that(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gold = ev.rng.next_int_in(41, 69);
    match ev.ask(run, vec![EventOption::new("PLAIN"), EventOption::new("ORNATE")]) {
        "PLAIN" => {
            run.damage(6);
            run.gain_gold(gold);
        }
        _ => {
            ev.take_from_bag(run);
            run.add_card(DeckCard::new("CLUMSY"));
        }
    }
    None
}

/// `Trial`: Accept draws the defendant, whose verdicts each add a curse and
/// give something; Reject asks again, and doubling down abandons the run.
fn trial(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    if ev.ask(run, vec![EventOption::new("ACCEPT"), EventOption::new("REJECT")]) == "REJECT"
        && ev.ask(run, vec![EventOption::on("REJECT", "ACCEPT"), EventOption::on("REJECT", "DOUBLE_DOWN")]) == "DOUBLE_DOWN"
    {
        // `NAbandonRunConfirmPopup`: the run ends there.
        run.hp = 0;
        return None;
    }
    let page = ["MERCHANT", "NOBLE", "NONDESCRIPT"][ev.rng.next_int(3) as usize];
    let verdict = ev.ask(run, vec![EventOption::on(page, "GUILTY"), EventOption::on(page, "INNOCENT")]);
    match (page, verdict) {
        ("MERCHANT", "GUILTY") => {
            run.add_card(DeckCard::new("REGRET"));
            ev.take_from_bag(run);
            ev.take_from_bag(run);
        }
        ("MERCHANT", _) => {
            run.add_card(DeckCard::new("SHAME"));
            ev.upgrade(run, 2);
        }
        ("NOBLE", "GUILTY") => run.heal(10),
        ("NOBLE", _) => {
            run.add_card(DeckCard::new("REGRET"));
            run.gain_gold(300);
        }
        (_, "GUILTY") => {
            run.add_card(DeckCard::new("DOUBT"));
            let rewards = (0..2).map(|_| Offered::Cards(run.event_card_reward(IRONCLAD_CARDS))).collect();
            ev.settle(run, rewards);
        }
        _ => {
            run.add_card(DeckCard::new("DOUBT"));
            ev.transform(run, 2);
        }
    }
    None
}

/// Takes the event options a `tools/oracle events` line scripts, each an
/// index among the open options modulo their count, the first past the
/// last; and the first of everything else, as the oracle's selector does.
struct Scripted {
    choices: Vec<usize>,
    next: usize,
}

impl Chooser for Scripted {
    fn choose(&mut self, _: &RunState, decision: Decision<'_>) -> usize {
        match decision {
            Decision::Event { options, .. } => {
                let i = self.choices.get(self.next).map_or(0, |&c| c % options.len());
                self.next += 1;
                i
            }
            _ => 0,
        }
    }
}

/// A `tools/oracle events` input line replayed on the port, printed as the
/// oracle prints it.
fn port_text(line: &str) -> String {
    use crate::encounter::Act;
    use crate::run::RunRelic;
    let (setup, choices) = line.split_once(':').unwrap_or((line, ""));
    let parts: Vec<&str> = setup.split_whitespace().collect();
    let ascension = crate::types::Ascension(parts[1].parse().unwrap());
    let mut run = RunState::new(parts[0], [Act::Overgrowth, Act::Hive, Act::Glory], ascension, &crate::plan::Unlocks::default());
    run.act = parts[2].parse().unwrap();
    for op in &parts[4..] {
        let (key, value) = op.split_once('=').unwrap_or(op.split_at(1));
        match key {
            "+" => {
                let id = value.trim_end_matches('+');
                run.deck.push(DeckCard { upgraded: id != value, ..DeckCard::new(id) });
            }
            "-" => {
                let i = run.deck.iter().position(|c| c.id == value).expect("a card to remove");
                run.deck.remove(i);
            }
            "gold" => run.gold = value.parse().unwrap(),
            "hp" => run.hp = value.parse().unwrap(),
            "maxhp" => run.max_hp = value.parse().unwrap(),
            "relic" => run.relics.push(RunRelic::new(value)),
            "potion" => {
                if let Some(slot) = run.potions.iter_mut().find(|p| p.is_none()) {
                    *slot = Some(value.to_string());
                }
            }
            _ => panic!("unknown setup {op}"),
        }
    }
    let name = ported().find(|e| slug(e) == parts[3]).expect("a ported event");
    let mut chooser = Scripted { choices: choices.split_whitespace().map(|c| c.parse().unwrap()).collect(), next: 0 };
    let visit = run.event(name, &mut chooser, &mut Vec::new()).expect("a ported event");
    let mut out = String::new();
    for page in &visit.pages {
        let options: Vec<String> = page
            .iter()
            .map(|o| {
                let key = if o.page.is_empty() { o.key.to_string() } else { format!("{}.{}", o.page, o.key) };
                if o.locked { format!("{key}!") } else { key }
            })
            .collect();
        out.push_str(&format!("page {}\n", options.join(" ")));
    }
    if let Some(shop) = visit.pages.first().and_then(|p| p.iter().find(|o| o.page.is_empty() && o.key == "SHOP")) {
        out.push_str(&format!("shelf {}\n", shop.items.join(" ")));
    }
    if let Some(fight) = &visit.fight {
        let extras = fight.extra.iter().map(|e| match e {
            Extra::Relic => "relic".to_string(),
            Extra::Potion => "potion".to_string(),
            Extra::NamedRelic(id) => format!("relic:{id}"),
            Extra::Card(_) => "card".to_string(),
        });
        let words: Vec<String> = std::iter::once(slug(&format!("{:?}", fight.encounter))).chain(extras).collect();
        out.push_str(&format!("fight {}\n", words.join(" ")));
    }
    out.push_str(&format!("end {} {}\n", run.oracle_text(), visit.draws));
    out
}

/// Checks `tools/oracle events` output against the port: returns how many
/// runs it held and each one that differs, with the first line that does.
pub fn diff_oracle(text: &str) -> (usize, Vec<String>) {
    let mut runs = 0;
    let mut mismatches = Vec::new();
    let mut header = "";
    let mut game = String::new();
    for line in text.lines().chain(["run end"]) {
        if let Some(next) = line.strip_prefix("run ") {
            // A run the game's code could not finish outside the game (a
            // death) is not the port's to match.
            if !header.is_empty() && !game.contains("error ") {
                runs += 1;
                let port = port_text(header);
                if port != game {
                    let first = port.lines().zip(game.lines()).find(|(p, g)| p != g);
                    mismatches.push(format!("{header}: {first:?}"));
                }
            }
            header = next;
            game.clear();
        } else {
            game.push_str(line);
            game.push('\n');
        }
    }
    (runs, mismatches)
}

/// `Amalgamator`: two basic Strikes (or Defends) out for an Ultimate Strike
/// (or Defend).
fn amalgamator(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let (basic, ultimate) = match ev.ask(run, vec![EventOption::new("COMBINE_STRIKES"), EventOption::new("COMBINE_DEFENDS")]) {
        "COMBINE_STRIKES" => ("STRIKE_IRONCLAD", "ULTIMATE_STRIKE"),
        _ => ("DEFEND_IRONCLAD", "ULTIMATE_DEFEND"),
    };
    ev.remove(run, 2, |c| c.id == basic);
    run.add_card(DeckCard::new(ultimate));
    None
}

/// `AromaOfChaos`: Let Go transforms a card on the event's stream; Maintain
/// Control upgrades one.
fn aroma_of_chaos(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("LET_GO"), EventOption::new("MAINTAIN_CONTROL")]) {
        "LET_GO" => ev.transform(run, 1),
        _ => ev.upgrade(run, 1),
    }
    None
}

/// `BattlewornDummy`: each setting fights its dummy, and beating it gives
/// what `dummy_beaten` says.
fn battleworn_dummy(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let options = ["SETTING_1", "SETTING_2", "SETTING_3"].map(EventOption::new).to_vec();
    let setting = ev.ask_at(run, options) as u8 + 1;
    Some(EventFight { dummy: Some(setting), ..EventFight::new(Encounter::BattlewornDummyEventEncounter) })
}

/// `Bugslayer`: an Exterminate or a Squash.
fn bugslayer(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gift = match ev.ask(run, vec![EventOption::new("EXTERMINATION"), EventOption::new("SQUASH")]) {
        "EXTERMINATION" => "EXTERMINATE",
        _ => "SQUASH",
    };
    run.add_card(DeckCard::new(gift));
    None
}

/// `ByrdonisNest`: 7 max HP, or Byrdonis' egg.
fn byrdonis_nest(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("EAT"), EventOption::new("TAKE")]) {
        "EAT" => run.gain_max_hp(7),
        _ => run.add_card(DeckCard::new("BYRDONIS_EGG")),
    }
    None
}

/// `ColossalFlower`: each dig deals 5, 6, then 7 damage for a bigger prize
/// (35, 75, 135 gold); past the second, Pollinous Core for 7 more.
fn colossal_flower(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    const PRIZE: [i32; 3] = [35, 75, 135];
    const DAMAGE: [i32; 3] = [5, 6, 7];
    const PAGES: [&str; 3] = ["INITIAL", "REACH_DEEPER_1", "REACH_DEEPER_2"];
    const EXTRACT: [&str; 2] = ["EXTRACT_CURRENT_PRIZE_1", "EXTRACT_CURRENT_PRIZE_2"];
    const DEEPER: [&str; 2] = ["REACH_DEEPER_1", "REACH_DEEPER_2"];
    for digs in 0..2 {
        let page = vec![EventOption::on(PAGES[digs], EXTRACT[digs]), EventOption::on(PAGES[digs], DEEPER[digs])];
        if ev.ask_at(run, page) == 0 {
            run.gain_gold(PRIZE[digs]);
            return None;
        }
        run.damage(DAMAGE[digs]);
    }
    let page = vec![EventOption::on("REACH_DEEPER_2", "EXTRACT_INSTEAD"), EventOption::on("REACH_DEEPER_2", "POLLINOUS_CORE")];
    match ev.ask(run, page) {
        "EXTRACT_INSTEAD" => run.gain_gold(PRIZE[2]),
        _ => {
            run.damage(DAMAGE[2]);
            ev.take(run, "POLLINOUS_CORE");
        }
    }
    None
}

/// `DollRoom`: a random doll; or, for 5 damage, two of the three to choose
/// from, or all three for 15. The dolls are options titled by their relic.
fn doll_room(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    // `_dolls`, and sorted as `StableShuffle` sorts them, by relic id.
    const DOLLS: [&str; 3] = ["DAUGHTER_OF_THE_WIND", "MR_STRUGGLES", "BING_BONG"];
    let options = vec![EventOption::new("RANDOM"), EventOption::new("TAKE_SOME_TIME"), EventOption::new("EXAMINE")];
    let doll = match ev.ask(run, options) {
        "RANDOM" => *ev.rng.pick(&DOLLS).expect("a doll"),
        way => {
            let (damage, count) = if way == "TAKE_SOME_TIME" { (5, 2) } else { (15, 3) };
            run.damage(damage);
            let mut dolls = DOLLS;
            dolls.sort_unstable();
            ev.rng.shuffle(&mut dolls);
            let options = dolls[..count].iter().map(|&d| EventOption::on("", d)).collect();
            ev.ask(run, options)
        }
    };
    ev.take(run, doll);
    None
}

/// `DoorsOfLightAndDark`: Light upgrades two random cards on the event's
/// stream; Dark removes one.
fn doors_of_light_and_dark(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("LIGHT"), EventOption::new("DARK")]) {
        "LIGHT" => {
            for i in stable_shuffle(run, &mut ev.rng, DeckCard::upgradable).into_iter().take(2) {
                run.upgrade_card(i);
            }
        }
        _ => ev.remove(run, 1, |_| true),
    }
    None
}

/// `StableShuffle` of the deck cards `keep` passes: sorted as `CardModel`
/// compares (id, then upgrade), then shuffled on `rng`.
fn stable_shuffle(run: &RunState, rng: &mut GameRng, keep: impl Fn(&DeckCard) -> bool) -> Vec<usize> {
    let deck = &run.deck;
    let mut cards: Vec<usize> = (0..deck.len()).filter(|&i| keep(&deck[i])).collect();
    cards.sort_by(|&a, &b| (&deck[a].id, deck[a].upgraded).cmp(&(&deck[b].id, deck[b].upgraded)));
    rng.shuffle(&mut cards);
    cards
}

/// `DrowningBeacon`: a Glowwater Potion, or Fresnel Lens for 13 max HP.
fn drowning_beacon(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("BOTTLE"), EventOption::new("CLIMB")]) {
        "BOTTLE" => ev.settle(run, vec![Offered::Potions(vec!["GLOWWATER_POTION".into()])]),
        _ => {
            run.lose_max_hp(13);
            ev.take(run, "FRESNEL_LENS");
        }
    }
    None
}

/// `EndlessConveyor`: dish after dish off the belt at 40 gold each (the
/// Golden Fysh is free and pays 75), each rolled by weight on the event's
/// stream, never the same twice running and a Seapunk Salad every fifth;
/// or Observe Chef upgrades a random card.
fn endless_conveyor(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let mut grabs = 0;
    let mut last: &str = "";
    let mut dish = roll_dish(run, &mut ev.rng, &mut grabs, &mut last);
    let grab = |run: &RunState, dish: &'static str| {
        if run.gold >= 40 { EventOption::on("ALL", dish) } else { EventOption { locked: true, ..EventOption::on("ALL", "LOCKED") } }
    };
    let options = vec![grab(run, dish), EventOption::new("OBSERVE_CHEF")];
    if ev.ask(run, options) == "OBSERVE_CHEF" {
        let upgradable: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgradable()).collect();
        if let Some(&i) = ev.rng.pick(&upgradable) {
            run.upgrade_card(i);
        }
        return None;
    }
    loop {
        if dish != "GOLDEN_FYSH" {
            run.lose_gold(40);
        }
        match dish {
            "CLAM_ROLL" => run.heal(10),
            "CAVIAR" => run.gain_max_hp(4),
            "SUSPICIOUS_CONDIMENT" => {
                let potion = run.any_potion().to_string();
                ev.settle(run, vec![Offered::Potions(vec![potion])]);
            }
            "JELLY_LIVER" => ev.transform(run, 1),
            "SEAPUNK_SALAD" => run.add_card(DeckCard::new("FEEDING_FRENZY")),
            "FRIED_EEL" => {
                let eel = run.event_cards(1, COLORLESS_CARDS, OddsType::Regular, |_| true);
                run.add_card(DeckCard::from(eel[0]));
            }
            "GOLDEN_FYSH" => run.gain_gold(75),
            _ => {
                let upgradable: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgradable()).collect();
                if let Some(&i) = ev.rng.pick(&upgradable) {
                    run.upgrade_card(i);
                }
            }
        }
        dish = roll_dish(run, &mut ev.rng, &mut grabs, &mut last);
        let options = vec![grab(run, dish), EventOption::on("GRAB_SOMETHING_OFF_THE_BELT", "LEAVE")];
        if ev.ask(run, options) == "LEAVE" {
            return None;
        }
    }
}

/// `EndlessConveyor.RollDish`.
fn roll_dish(run: &RunState, rng: &mut GameRng, grabs: &mut u32, last: &mut &'static str) -> &'static str {
    *grabs += 1;
    if *grabs % 5 == 0 {
        *last = "SEAPUNK_SALAD";
        return "SEAPUNK_SALAD";
    }
    let mut dishes: Vec<(&'static str, f32)> = vec![("CAVIAR", 6.0), ("SPICY_SNAPPY", 3.0), ("JELLY_LIVER", 3.0), ("FRIED_EEL", 3.0)];
    if run.potions.iter().any(Option::is_none) {
        dishes.push(("SUSPICIOUS_CONDIMENT", 3.0));
    }
    if run.hp != run.max_hp {
        dishes.push(("CLAM_ROLL", 6.0));
    }
    if *grabs > 1 {
        dishes.push(("GOLDEN_FYSH", 1.0));
    }
    dishes.retain(|(d, _)| d != last);
    let total: f32 = dishes.iter().map(|d| d.1).sum();
    let roll = rng.next_float(1.0) * total;
    let mut sum = 0f32;
    for (dish, weight) in dishes {
        sum += weight;
        if roll < sum {
            *last = dish;
            return dish;
        }
    }
    // Rounding left the roll past every weight: the dish stays as it was.
    last
}

/// `FakeMerchant`: six of nine fake relics at 50 gold each, rolled on the
/// Shops stream, to buy one at a time; a Foul Potion thrown at him starts a
/// fight whose rewards are his rug, the relics still on the shelf and 300
/// gold. The game lays out no options; the port asks whether to throw the
/// potion (`THROW`) before the shop opens.
fn fake_merchant(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    const RELICS: [&str; 9] = [
        "FAKE_ANCHOR", "FAKE_BLOOD_VIAL", "FAKE_HAPPY_FLOWER", "FAKE_LEES_WAFFLE", "FAKE_MANGO", "FAKE_ORICHALCUM",
        "FAKE_SNECKO_EYE", "FAKE_STRIKE_DUMMY", "FAKE_VENERABLE_TEA_SET",
    ];
    let mut relics = RELICS;
    ev.rng.shuffle(&mut relics);
    // `MerchantRelicEntry.CalcCost`; in an event room no relic discounts it.
    let mut shelf: Vec<Option<(&str, i32)>> = relics[..6]
        .iter()
        .map(|&r| Some((r, (50.0 * run.rngs.player(PlayerStream::Shops).next_float_in(0.85, 1.15)).round_ties_even() as i32)))
        .collect();
    let foul = run.potions.iter().position(|p| p.as_deref() == Some("FOUL_POTION"));
    let mut shop = EventOption::on("", "SHOP");
    shop.items = shelf.iter().flatten().map(|(r, price)| format!("{r}:{price}")).collect();
    let options = match foul {
        Some(_) => vec![EventOption::on("", "THROW"), shop],
        None => vec![shop],
    };
    if ev.ask(run, options) == "THROW" {
        run.discard_potion(foul.expect("a Foul Potion"));
        let mut extra = vec![Extra::NamedRelic("FAKE_MERCHANTS_RUG".into())];
        extra.extend(shelf.iter().flatten().map(|&(r, _)| Extra::NamedRelic(r.into())));
        return Some(EventFight { extra, gold: Some(300), ..EventFight::new(Encounter::FakeMerchantEventEncounter) });
    }
    loop {
        let wares: Vec<Ware> = (0..shelf.len())
            .filter_map(|i| shelf[i].filter(|&(_, price)| price <= run.gold).map(|(r, price)| Ware { slot: Slot::Relic(i), item: Item::Relic(r.into()), price }))
            .collect();
        let Some(ware) = wares.get(ev.chooser.choose(run, Decision::Shop(&wares))).cloned() else { return None };
        let Slot::Relic(i) = ware.slot else { unreachable!("only relics on the shelf") };
        let (relic, price) = shelf[i].take().expect("a stocked relic");
        run.lose_gold(price);
        ev.take(run, relic);
        // `Hook.AfterItemPurchased`: Maw Bank stops paying.
        if let Some(bank) = run.relic_mut("MAW_BANK") {
            bank.flag = true;
        }
    }
}

/// `FieldOfManSizedHoles`: Resist removes two cards and adds a Normality;
/// Enter Your Hole puts Perfect Fit on a card.
fn field_of_man_sized_holes(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("RESIST"), EventOption::new("ENTER_YOUR_HOLE")]) {
        "RESIST" => {
            ev.remove(run, 2, |_| true);
            run.add_card(DeckCard::new("NORMALITY"));
        }
        _ => ev.enchant(run, "PERFECT_FIT", 1, 1),
    }
    None
}

/// `GraveOfTheForgotten`: Confront adds a Decay and puts Soul's Power on a
/// card (locked if none takes it); Accept gives Forgotten Soul.
fn grave_of_the_forgotten(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let open = run.deck.iter().any(|c| c.can_enchant("SOULS_POWER"));
    match ev.ask(run, vec![EventOption::or_locked(open, "CONFRONT", "CONFRONT_LOCKED"), EventOption::new("ACCEPT")]) {
        "CONFRONT" => {
            run.add_card(DeckCard::new("DECAY"));
            ev.enchant(run, "SOULS_POWER", 1, 1);
        }
        _ => ev.take(run, "FORGOTTEN_SOUL"),
    }
    None
}

/// `HungryForMushrooms`: Big Mushroom or Fragrant Mushroom, options titled
/// by their relic (`RelicOption`).
fn hungry_for_mushrooms(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let relic = ev.ask(run, vec![EventOption::new("BIG_MUSHROOM"), EventOption::new("FRAGRANT_MUSHROOM")]);
    ev.take(run, relic);
    None
}

/// `InfestedAutomaton`: Study adds one of the character's powers; Touch
/// Core one of their cards that costs 0.
fn infested_automaton(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let free = |c: &PoolCard| sim_card(c.id).is_some_and(|id| def(id).cost == 0 && !def(id).x_cost);
    let cards = match ev.ask(run, vec![EventOption::new("STUDY"), EventOption::new("TOUCH_CORE")]) {
        "STUDY" => run.event_cards(1, IRONCLAD_CARDS, OddsType::Regular, |c| c.kind == CardType::Power),
        _ => run.event_cards(1, IRONCLAD_CARDS, OddsType::Regular, free),
    };
    cards.into_iter().for_each(|c| run.add_card(DeckCard::from(c)));
    None
}

/// `JungleMazeAdventure`: Solo Quest deals 18 damage for about 150 gold;
/// Join Forces about 50. Each amount is off by a float of up to 15, which
/// the game keeps as a `decimal` of seven digits.
fn jungle_maze_adventure(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let decimal = |f: f32| -> f64 { format!("{f:.6e}").parse().expect("a float") };
    let solo = 150.0 + decimal(ev.rng.next_float_in(-15.0, 15.0));
    let join = 50.0 + decimal(ev.rng.next_float_in(-15.0, 15.0));
    match ev.ask(run, vec![EventOption::new("SOLO_QUEST"), EventOption::new("JOIN_FORCES")]) {
        "SOLO_QUEST" => {
            // The three hit effects, `StableShuffle`d on the event's stream.
            ev.rng.shuffle(&mut [0, 1, 2]);
            run.damage(18);
            run.gain_gold(solo);
        }
        _ => run.gain_gold(join),
    }
    None
}

/// `LuminousChoir`: two cards out for a Spore Mind, or 100 to 149 gold for
/// a relic off the front of the bag.
fn luminous_choir(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let tribute = 149 - ev.rng.next_int_in(0, 50);
    let options = vec![EventOption::new("REACH_INTO_THE_FLESH"), EventOption::or_locked(run.gold >= tribute, "OFFER_TRIBUTE", "OFFER_TRIBUTE_LOCKED")];
    match ev.ask(run, options) {
        "REACH_INTO_THE_FLESH" => {
            ev.remove(run, 2, |_| true);
            run.add_card(DeckCard::new("SPORE_MIND"));
        }
        _ => {
            run.lose_gold(tribute);
            ev.take_from_bag(run);
        }
    }
    None
}

/// `MorphicGrove`: Group takes all the gold and transforms two cards on the
/// event's stream; Loner gives 5 max HP.
fn morphic_grove(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("GROUP"), EventOption::new("LONER")]) {
        "GROUP" => {
            run.lose_gold(run.gold);
            ev.transform(run, 2);
        }
        _ => run.gain_max_hp(5),
    }
    None
}

/// `PunchOff`: Nab adds an Injury for a relic reward; I Can Take Them
/// fights the two constructs for a relic and a potion besides. Its combat
/// layout creates them as the event is entered.
fn punch_off(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    run.enemies_created(2);
    ev.rng.next_int_in(91, 99);
    if ev.ask(run, vec![EventOption::new("NAB"), EventOption::new("I_CAN_TAKE_THEM")]) == "NAB" {
        run.add_card(DeckCard::new("INJURY"));
        let relic = run.relic_reward().game_id();
        ev.settle(run, vec![Offered::Relics(vec![relic])]);
        return None;
    }
    ev.ask(run, vec![EventOption::on("I_CAN_TAKE_THEM", "FIGHT")]);
    Some(EventFight { created: true, extra: vec![Extra::Relic, Extra::Potion], ..EventFight::new(Encounter::PunchOffEventEncounter) })
}

/// `Reflections`: Touch a Mirror downgrades two random upgraded cards, then
/// upgrades four random upgradable ones, on the event's stream; Shatter
/// copies the deck and adds Bad Luck.
fn reflections(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("TOUCH_A_MIRROR"), EventOption::new("SHATTER")]) {
        "TOUCH_A_MIRROR" => {
            let mut upgraded: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgraded).collect();
            for _ in 0..2 {
                let Some(&i) = ev.rng.pick(&upgraded) else { break };
                upgraded.retain(|&u| u != i);
                run.downgrade_card(i);
            }
            let mut upgradable: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgradable()).collect();
            for _ in 0..4 {
                let Some(&i) = ev.rng.pick(&upgradable) else { break };
                upgradable.retain(|&u| u != i);
                run.upgrade_card(i);
            }
        }
        _ => {
            // `RunState.CloneCard` of each card the deck held, in order.
            for i in 0..run.deck.len() {
                let copy = run.deck[i].clone();
                run.add_card(copy);
            }
            run.add_card(DeckCard::new("BAD_LUCK"));
        }
    }
    None
}

/// `RelicTrader`: up to three of the tradable relics held, `StableShuffle`d
/// on the event's stream, each against a relic off the front of the bag
/// (all three pulled as the event starts).
fn relic_trader(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let mut owned: Vec<String> = run.tradable_relics().into_iter().map(|i| run.relics[i].id.clone()).collect();
    owned.sort_unstable();
    ev.rng.shuffle(&mut owned);
    owned.truncate(3);
    if owned.is_empty() {
        ev.ask(run, vec![EventOption::on("", "PROCEED")]);
        return None;
    }
    let new: Vec<String> = (0..3).map(|_| run.relic_reward().game_id()).collect();
    let keys = ["TOP", "MIDDLE", "BOTTOM"];
    let options = (0..owned.len()).map(|i| EventOption::new(keys[i]).naming(owned[i].clone()).naming(new[i].clone())).collect();
    let i = ev.ask_at(run, options);
    run.remove_relic(&owned[i]);
    ev.take(run, new[i].clone());
    None
}

/// `SapphireSeed`: Eat heals 9 and upgrades a card; Plant puts Sown on one.
fn sapphire_seed(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("EAT"), EventOption::new("PLANT")]) {
        "EAT" => {
            run.heal(9);
            ev.upgrade(run, 1);
        }
        _ => ev.enchant(run, "SOWN", 1, 1),
    }
    None
}

/// `SpiralingWhirlpool`: Spiral on a card, or a third of max HP healed.
fn spiraling_whirlpool(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let heal = run.max_hp * 33 / 100;
    match ev.ask(run, vec![EventOption::new("OBSERVE"), EventOption::new("DRINK")]) {
        "OBSERVE" => ev.enchant(run, "SPIRAL", 1, 1),
        _ => run.heal(heal),
    }
    None
}

/// `SpiritGrafter`: Let It In heals 25 and adds a Metamorphosis; Rejection
/// upgrades a card for 10 damage.
fn spirit_grafter(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("LET_IT_IN"), EventOption::new("REJECTION")]) {
        "LET_IT_IN" => {
            run.heal(25);
            run.add_card(DeckCard::new("METAMORPHOSIS"));
        }
        _ => {
            ev.upgrade(run, 1);
            run.damage(10);
        }
    }
    None
}

/// `StoneOfAllTime`: Lift drinks a random potion for 10 max HP; Push puts
/// Vigorous 8 on a card for 6 damage. Each draws once more on the event's
/// stream after.
fn stone_of_all_time(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let held: Vec<usize> = (0..run.potions.len()).filter(|&i| run.potions[i].is_some()).collect();
    let potion = ev.rng.pick(&held).copied();
    let lift = match potion {
        Some(slot) => EventOption::new("LIFT").naming(run.potions[slot].clone().expect("a potion")),
        None => EventOption::locked("LIFT_LOCKED"),
    };
    let push = EventOption::or_locked(run.deck.iter().any(|c| c.can_enchant("VIGOROUS")), "PUSH", "PUSH_LOCKED");
    match ev.ask(run, vec![lift, push]) {
        "LIFT" => {
            run.discard_potion(potion.expect("the potion offered"));
            run.gain_max_hp(10);
        }
        _ => {
            run.damage(6);
            ev.enchant(run, "VIGOROUS", 8, 1);
        }
    }
    ev.rng.next_int(100);
    None
}

/// `SunkenTreasury`: 52 to 67 gold, or 303 to 363 and a Greed.
fn sunken_treasury(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let small = 60 + ev.rng.next_int(16) - 8;
    let large = 333 + ev.rng.next_int(61) - 30;
    match ev.ask(run, vec![EventOption::new("FIRST_CHEST"), EventOption::new("SECOND_CHEST")]) {
        "FIRST_CHEST" => run.gain_gold(small),
        _ => {
            run.gain_gold(large);
            run.add_card(DeckCard::new("GREED"));
        }
    }
    None
}

/// `TabletOfTruth`: each Decipher costs max HP (3, 6, 12, 24, then all but
/// one) and upgrades a random card on the event's stream, the fifth every
/// card; Smash heals 20. A decipher that would take the last max HP kills.
fn tablet_of_truth(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    if ev.ask(run, vec![EventOption::new("DECIPHER_1"), EventOption::new("SMASH")]) == "SMASH" {
        run.heal(20);
        return None;
    }
    const PAGES: [&str; 4] = ["DECIPHER_1", "DECIPHER_2", "DECIPHER_3", "DECIPHER_4"];
    let mut cost = 3;
    for count in 0..5 {
        if cost >= run.max_hp {
            run.lose_max_hp(run.max_hp - 1);
            run.lose_hp(run.hp);
            return None;
        }
        run.lose_max_hp(cost);
        let upgradable: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgradable()).collect();
        if count == 4 {
            upgradable.into_iter().for_each(|i| run.upgrade_card(i));
            return None;
        }
        if let Some(&i) = ev.rng.pick(&upgradable) {
            run.upgrade_card(i);
        }
        cost = [6, 12, 24, run.max_hp - 1][count];
        let page = vec![EventOption::on(PAGES[count], "DECIPHER"), EventOption::on("DECIPHER", "GIVE_UP")];
        if ev.ask(run, page) == "GIVE_UP" {
            return None;
        }
    }
    None
}

/// `TeaMaster`: Bone Tea for 50 gold, Ember Tea for 150 (each locked
/// without the gold), or Tea of Discourtesy for nothing.
fn tea_master(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let options = vec![
        EventOption::or_locked(run.gold >= 50, "BONE_TEA", "BONE_TEA_LOCKED"),
        EventOption::or_locked(run.gold >= 150, "EMBER_TEA", "EMBER_TEA_LOCKED"),
        EventOption::new("TEA_OF_DISCOURTESY"),
    ];
    let tea = ev.ask(run, options);
    run.lose_gold(match tea {
        "BONE_TEA" => 50,
        "EMBER_TEA" => 150,
        _ => 0,
    });
    ev.take(run, tea);
    None
}

/// `TheLanternKey`: 100 gold, or keep the key and fight its knight for the
/// Lantern Key card besides the rewards. Its combat layout creates the
/// knight as the event is entered.
fn the_lantern_key(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    run.enemies_created(1);
    if ev.ask(run, vec![EventOption::new("RETURN_THE_KEY"), EventOption::new("KEEP_THE_KEY")]) == "RETURN_THE_KEY" {
        run.gain_gold(100);
        return None;
    }
    ev.ask(run, vec![EventOption::on("KEEP_THE_KEY", "FIGHT")]);
    Some(EventFight { created: true, extra: vec![Extra::Card("LANTERN_KEY")], ..EventFight::new(Encounter::MysteriousKnightEventEncounter) })
}

/// `TheLegendsWereTrue`: Nab the Map adds Spoils Map; Slowly Find an Exit
/// deals 8 damage for a random potion.
fn the_legends_were_true(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("NAB_THE_MAP"), EventOption::new("SLOWLY_FIND_AN_EXIT")]) {
        "NAB_THE_MAP" => run.add_card(DeckCard::new("SPOILS_MAP")),
        _ => {
            run.damage(8);
            let potion = run.any_potion().to_string();
            ev.settle(run, vec![Offered::Potions(vec![potion])]);
        }
    }
    None
}

/// `TrashHeap`: Dive In deals 8 damage for one of five relics; Grab gives
/// 100 gold and one of ten cards; each picked on the event's stream.
fn trash_heap(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    const RELICS: [&str; 5] = ["DARKSTONE_PERIAPT", "DREAM_CATCHER", "HAND_DRILL", "MAW_BANK", "THE_BOOT"];
    const CARDS: [&str; 10] =
        ["CALTROPS", "CLASH", "DISTRACTION", "DUAL_WIELD", "ENTRENCH", "HELLO_WORLD", "OUTMANEUVER", "REBOUND", "RIP_AND_TEAR", "STACK"];
    match ev.ask(run, vec![EventOption::new("DIVE_IN"), EventOption::new("GRAB")]) {
        "DIVE_IN" => {
            run.damage(8);
            let relic = *ev.rng.pick(&RELICS).expect("a relic");
            ev.take(run, relic);
        }
        _ => {
            run.gain_gold(100);
            let gift = *ev.rng.pick(&CARDS).expect("a card");
            run.add_card(DeckCard::new(gift));
        }
    }
    None
}

/// `UnrestSite`: Rest heals what was missing as the event began and adds
/// Poor Sleep; Kill gives a relic off the front of the bag for 8 max HP.
fn unrest_site(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let missing = run.max_hp - run.hp;
    match ev.ask(run, vec![EventOption::new("REST"), EventOption::new("KILL")]) {
        "REST" => {
            run.heal(missing);
            run.add_card(DeckCard::new("POOR_SLEEP"));
        }
        _ => {
            run.lose_max_hp(8);
            ev.take_from_bag(run);
        }
    }
    None
}

/// `WaterloggedScriptorium`: 6 max HP; Steady on a card for 55 gold, or on
/// two for 99 (each locked without the gold).
fn waterlogged_scriptorium(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let options = vec![
        EventOption::new("BLOODY_INK"),
        EventOption::or_locked(run.gold >= 55, "TENTACLE_QUILL", "TENTACLE_QUILL_LOCKED"),
        EventOption::or_locked(run.gold >= 99, "PRICKLY_SPONGE", "PRICKLY_SPONGE_LOCKED"),
    ];
    match ev.ask(run, options) {
        "BLOODY_INK" => run.gain_max_hp(6),
        "TENTACLE_QUILL" => {
            run.lose_gold(55);
            ev.enchant(run, "STEADY", 1, 1);
        }
        _ => {
            run.lose_gold(99);
            ev.enchant(run, "STEADY", 1, 2);
        }
    }
    None
}

/// `WelcomeToWongos`: the featured item, a rare a shop could sell, pulled
/// as the event starts; a common one for 100 gold, the featured one for
/// 200, Wongo's Mystery Ticket for 300 (each locked without the gold); or
/// leave, and a random upgraded card is downgraded on the way out. The
/// Customer Appreciation Badge, which the profile's Wongo points give, is
/// left out.
fn welcome_to_wongos(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let featured = run.pull_for_shop(RelicRarity::Rare).game_id();
    let options = vec![
        EventOption::or_locked(run.gold >= 100, "BARGAIN_BIN", "BARGAIN_BIN_LOCKED"),
        EventOption::or_locked(run.gold >= 200, "FEATURED_ITEM", "FEATURED_ITEM_LOCKED").naming(featured.clone()),
        EventOption::or_locked(run.gold >= 300, "MYSTERY_BOX", "MYSTERY_BOX_LOCKED"),
        EventOption::new("LEAVE"),
    ];
    match ev.ask(run, options) {
        "BARGAIN_BIN" => {
            run.lose_gold(100);
            let relic = run.pull_for_shop(RelicRarity::Common).game_id();
            ev.take(run, relic);
        }
        "FEATURED_ITEM" => {
            run.lose_gold(200);
            ev.take(run, featured);
        }
        "MYSTERY_BOX" => {
            run.lose_gold(300);
            ev.take(run, "WONGOS_MYSTERY_TICKET");
        }
        _ => {
            let upgraded: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].upgraded).collect();
            if let Some(&i) = ev.rng.pick(&upgraded) {
                run.downgrade_card(i);
            }
        }
    }
    None
}

/// `Wellspring`: Bottle offers a random potion; Bathe removes a card and
/// adds a Guilty.
fn wellspring(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    match ev.ask(run, vec![EventOption::new("BOTTLE"), EventOption::new("BATHE")]) {
        "BOTTLE" => {
            let potion = run.any_potion().to_string();
            ev.settle(run, vec![Offered::Potions(vec![potion])]);
        }
        _ => {
            ev.remove(run, 1, |_| true);
            run.add_card(DeckCard::new("GUILTY"));
        }
    }
    None
}

/// `WhisperingHollow`: 26 to 44 gold for two potion rewards; Hug transforms
/// a card on the event's stream for 9 damage.
fn whispering_hollow(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let gold = 35 + ev.rng.next_int_in(-9, 10);
    match ev.ask(run, vec![EventOption::new("GOLD"), EventOption::new("HUG")]) {
        "GOLD" => {
            run.lose_gold(gold);
            let potions = (0..2).map(|_| create_potion(run.rewards()).to_string()).collect();
            ev.settle(run, vec![Offered::Potions(potions)]);
        }
        _ => {
            ev.transform(run, 1);
            run.damage(9);
        }
    }
    None
}

/// `WoodCarvings`: a basic card into a Peck or a Toric Toughness, or
/// Slither on a card (locked if none takes it).
fn wood_carvings(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let snake = EventOption::or_locked(run.deck.iter().any(|c| c.can_enchant("SLITHER")), "SNAKE", "SNAKE_LOCKED");
    let into = match ev.ask(run, vec![EventOption::new("BIRD"), snake, EventOption::new("TORUS")]) {
        "SNAKE" => {
            ev.enchant(run, "SLITHER", 1, 1);
            return None;
        }
        "BIRD" => "PECK",
        _ => "TORIC_TOUGHNESS",
    };
    // `CardSelectCmd.FromDeckGeneric` over the transformable basic cards,
    // then `CardCmd.TransformTo`.
    let action = DeckAction::TransformInto(into);
    let cards = (0..run.deck.len()).filter(|&i| run.deck[i].removable() && rarity(&run.deck[i].id) == Some(Rarity::Basic)).collect();
    let chosen = ev.pick(run, action, cards, 1, 1);
    run.apply_pick(action, &chosen);
    None
}

/// `ZenWeaver`: two Enlightenments for 50 gold; one card out for 125, or
/// two for 250 (each locked without the gold).
fn zen_weaver(run: &mut RunState, ev: &mut Ev) -> Option<EventFight> {
    let options = vec![
        EventOption::new("BREATHING_TECHNIQUES"),
        EventOption::or_locked(run.gold >= 125, "EMOTIONAL_AWARENESS", "LOCKED"),
        EventOption::or_locked(run.gold >= 250, "ARACHNID_ACUPUNCTURE", "LOCKED"),
    ];
    match ev.ask(run, options) {
        "BREATHING_TECHNIQUES" => {
            run.lose_gold(50);
            run.add_card(DeckCard::new("ENLIGHTENMENT"));
            run.add_card(DeckCard::new("ENLIGHTENMENT"));
        }
        "EMOTIONAL_AWARENESS" => {
            ev.remove(run, 1, |_| true);
            run.lose_gold(125);
        }
        _ => {
            ev.remove(run, 2, |_| true);
            run.lose_gold(250);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/oracle events` for every ported event, over seeds, acts and
    /// setups that turn each option's condition on and off, walked through
    /// its pages (`examples/eventcheck.rs` makes more): the pages laid out,
    /// the fight started, the player left and the streams drawn on.
    #[test]
    fn matches_the_game() {
        let (runs, mismatches) = diff_oracle(include_str!("../testdata/oracle-events.txt"));
        assert!(runs >= 300, "fixture holds {runs} runs");
        assert!(mismatches.is_empty(), "{} of {runs} differ:\n{}", mismatches.len(), mismatches.join("\n"));
    }

    /// Every ported event's fixture runs cover it.
    #[test]
    fn fixture_covers_every_event() {
        let text = include_str!("../testdata/oracle-events.txt");
        for name in ported() {
            assert!(text.contains(&format!(" {} ", slug(name))), "no fixture run for {name}");
        }
    }
}
