//! A room from what it lays out to what the player leaves with, with every
//! choice put to a `Chooser`: the rewards screen, a treasure chest, a rest
//! site, an ancient's relic, and what effects offer on the way (a card to
//! take, a relic reward, cards to pick from the deck). The forward run
//! (`forward.rs`) chooses with a rule or a policy, the history check
//! (`history.rs`) with what the player chose, so both walk the same code.
//!
//! Every flow appends what it laid out and what effects did to a log of
//! `Offered`, nested pickups included, in the order they happened: the
//! history check compares it with the record, the forward run counts what
//! was not ported.

use crate::ancients::AncientOffer;
use crate::effects::{DeckAction, DeckPick, Offered, RestOption};
use crate::events::{EventFight, EventOption};
use crate::map::{ActMap, PointId};
use crate::rewards::{Offer, Rewards};
use crate::rng::Rng;
use crate::run::{DeckCard, RoomType, RunState};
use crate::shop::{Item, Shop, Slot, Ware};

/// One decision, with what it chooses among.
#[derive(Clone, Copy, Debug)]
pub enum Decision<'a> {
    /// The next map point on the act's map, among those the current one
    /// leads to: its children, or the whole next row with Winged Boots.
    Path(&'a ActMap, &'a [PointId]),
    /// The next relic reward to take; past the end leaves the rest.
    Relic(&'a [String]),
    /// A card to take; past the end takes none.
    Card(&'a [Offer]),
    /// A bundle to take; past the end takes none.
    Bundle(&'a [Vec<Offer>]),
    /// 0 keeps the potion, anything else leaves it.
    Potion(&'a str),
    /// A rest site option; past the end leaves (only after the first, with
    /// Miniature Tent).
    Rest(&'a [RestOption]),
    /// The ancient option to take, by relic id.
    Ancient(&'a [String]),
    /// A card out of the deck for `action`, as deck indices; past the end
    /// stops picking, which `optional` says the screen allows.
    Deck { action: DeckAction, cards: &'a [usize], optional: bool },
    /// A ware to buy; past the end leaves the shop.
    Shop(&'a [Ware]),
    /// One of the options an event (a class name) lays out, the ones it
    /// lets the player choose.
    Event { event: &'static str, options: &'a [EventOption] },
}

/// Makes the run's decisions: an index into what the decision lists.
pub trait Chooser {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize;
}

impl<C: Chooser + ?Sized> Chooser for &mut C {
    fn choose(&mut self, run: &RunState, decision: Decision<'_>) -> usize {
        (**self).choose(run, decision)
    }
}

/// Takes the first option every time: the first path, relic, card, bundle
/// and rest option, keeps every potion, picks from the front of the deck,
/// and buys the first ware it can until it can buy none.
pub struct First;

impl Chooser for First {
    fn choose(&mut self, _: &RunState, _: Decision<'_>) -> usize {
        0
    }
}

/// Chooses at random on its own RNG, never the run's streams
/// (docs/run-env.md, Hidden information): uniformly among the options, a
/// skip counting as one where the decision has one (a card or bundle
/// reward, an optional deck pick, leaving a shop). Relics are all taken, in a random
/// order, and potions kept: leaving one only throws it away.
pub struct Random(pub Rng);

impl Chooser for Random {
    fn choose(&mut self, _: &RunState, decision: Decision<'_>) -> usize {
        let options = match decision {
            Decision::Path(_, points) => points.len(),
            Decision::Relic(relics) => relics.len(),
            Decision::Card(cards) => cards.len() + 1,
            Decision::Bundle(bundles) => bundles.len() + 1,
            Decision::Potion(_) => 1,
            Decision::Rest(options) => options.len(),
            Decision::Ancient(relics) => relics.len(),
            Decision::Deck { cards, optional, .. } => cards.len() + optional as usize,
            Decision::Shop(wares) => wares.len() + 1,
            Decision::Event { options, .. } => options.len(),
        };
        self.0.next_int(options.max(1))
    }
}

impl RunState {
    /// Resolves what effects offered, in order: each choice put to
    /// `chooser` and applied, a taken relic's pickup resolved in turn.
    pub fn settle(&mut self, offered: Vec<Offered>, chooser: &mut impl Chooser, log: &mut Vec<Offered>) {
        for offer in offered {
            log.push(offer.clone());
            match offer {
                Offered::Cards(cards) => {
                    if let Some(card) = cards.get(chooser.choose(self, Decision::Card(&cards))) {
                        self.add_card(DeckCard::from(*card));
                    }
                }
                Offered::Bundles(bundles) => {
                    if let Some(bundle) = bundles.get(chooser.choose(self, Decision::Bundle(&bundles))) {
                        bundle.iter().for_each(|&card| self.add_card(DeckCard::from(card)));
                    }
                }
                Offered::Relics(relics) => self.take_relics(relics, chooser, log),
                Offered::Potions(potions) => {
                    for potion in potions {
                        if chooser.choose(self, Decision::Potion(&potion)) == 0 {
                            self.add_potion(&potion);
                        }
                    }
                }
                Offered::Pick(pick) => self.pick_cards(pick, chooser),
                Offered::Gained(_) | Offered::Took(_) | Offered::Unported(_) => {}
            }
        }
    }

    /// Relic rewards taken one at a time, in the order the chooser takes
    /// them, each picked up before the next.
    fn take_relics(&mut self, mut relics: Vec<String>, chooser: &mut impl Chooser, log: &mut Vec<Offered>) {
        while !relics.is_empty() {
            let i = chooser.choose(self, Decision::Relic(&relics));
            if i >= relics.len() {
                break;
            }
            let relic = relics.remove(i);
            let pickup = self.obtain(&relic);
            self.settle(pickup, chooser, log);
        }
    }

    /// A deck pick, one card at a time, then its action on all of them.
    fn pick_cards(&mut self, pick: DeckPick, chooser: &mut impl Chooser) {
        let mut left = pick.cards;
        let mut chosen = Vec::new();
        while chosen.len() < pick.max && !left.is_empty() {
            let optional = chosen.len() >= pick.min;
            let i = chooser.choose(self, Decision::Deck { action: pick.action, cards: &left, optional });
            if i >= left.len() {
                break;
            }
            chosen.push(left.remove(i));
        }
        self.apply_pick(pick.action, &chosen);
    }

    /// The rewards screen after a won fight (`NRewardsScreen`): relics
    /// first, in the order the chooser takes them, since a relic taken
    /// before the card reward works on it (`CardReward.OnRelicObtained`:
    /// the eggs); then the potion; then each card reward; the gold is
    /// automatic.
    pub fn take_rewards(&mut self, rewards: Rewards, chooser: &mut impl Chooser, log: &mut Vec<Offered>) {
        if !rewards.relics.is_empty() {
            self.settle(vec![Offered::Relics(rewards.relics)], chooser, log);
        }
        let mut cards = rewards.cards;
        cards.iter_mut().for_each(|c| self.upgrade_by_eggs(c));
        let potions = (!rewards.potions.is_empty()).then(|| Offered::Potions(rewards.potions.iter().map(|p| p.to_string()).collect()));
        self.settle(potions.into_iter().chain(cards.into_iter().map(Offered::Cards)).collect(), chooser, log);
        rewards.gold.into_iter().for_each(|gold| self.gain_gold(gold));
    }

    /// A treasure room: the chest's gold, then its relic to take. Returns
    /// the gold. In the second act a Spoils Map pays out 600 gold and
    /// leaves the deck (`OneOffSynchronizer.TryHandleSpoilsMap`,
    /// `SpoilsMap.OnQuestComplete`): the game marks the treasure of that
    /// act's map, which it generates as an hourglass through a single
    /// treasure (`SpoilsActMap`); the port keeps the act's usual map and
    /// pays at its first treasure room.
    pub fn treasure_room(&mut self, chooser: &mut impl Chooser, log: &mut Vec<Offered>) -> i32 {
        let (mut gold, relic) = self.treasure();
        self.gain_gold(gold);
        if self.act == 1 {
            while let Some(i) = self.deck.iter().position(|c| c.id == "SPOILS_MAP") {
                self.gain_gold(600);
                gold += 600;
                self.deck.remove(i);
            }
        }
        self.settle(vec![Offered::Relics(vec![relic.game_id()])], chooser, log);
        gold
    }

    /// A rest site: one option, or as many as the chooser takes with
    /// Miniature Tent (`ShouldDisableRemainingRestSiteOptions`).
    pub fn rest_site(&mut self, chooser: &mut impl Chooser, log: &mut Vec<Offered>) {
        let mut options = self.rest_options();
        while !options.is_empty() {
            let i = chooser.choose(self, Decision::Rest(&options));
            if i >= options.len() {
                break;
            }
            let offered = self.rest(options.remove(i));
            self.settle(offered, chooser, log);
            if !self.has_relic("MINIATURE_TENT") {
                break;
            }
        }
    }

    /// A merchant's room: the stock laid out (`shop`), Lord's Parasol
    /// buying everything on entry, then what the chooser buys, one ware at
    /// a time, until it leaves. The card removal asks which card, and
    /// leaving that pick buys nothing. Returns the shop as left.
    pub fn shop_room(&mut self, chooser: &mut impl Chooser, log: &mut Vec<Offered>) -> Shop {
        let mut shop = self.shop();
        log.push(Offered::Cards(shop.cards.iter().chain(&shop.colorless).flatten().map(|e| e.item).collect()));
        log.push(Offered::Relics(shop.relics.iter().flatten().map(|e| e.item.game_id()).collect()));
        log.push(Offered::Potions(shop.potions.iter().flatten().map(|e| e.item.to_string()).collect()));
        if self.has_relic("LORDS_PARASOL") {
            self.buy_everything(&mut shop, chooser, log);
        }
        loop {
            let wares = self.wares(&shop);
            let Some(ware) = wares.get(chooser.choose(self, Decision::Shop(&wares))) else { break };
            if ware.item == Item::Removal {
                let cards = self.pickable(DeckAction::Remove);
                let pick = Decision::Deck { action: DeckAction::Remove, cards: &cards, optional: true };
                if let Some(&card) = cards.get(chooser.choose(self, pick)) {
                    self.remove_for(&mut shop, card, ware.price);
                }
                continue;
            }
            let pickup = self.buy(&mut shop, ware.slot, ware.price);
            self.settle(pickup, chooser, log);
        }
        shop
    }

    /// `LordsParasol.PurchaseEverything`: every stocked entry bought for
    /// nothing, the character's cards, the colorless ones, the relics and
    /// the potions in turn; a potion with no slot to go to stays.
    fn buy_everything(&mut self, shop: &mut Shop, chooser: &mut impl Chooser, log: &mut Vec<Offered>) {
        let slots = (0..shop.cards.len())
            .map(Slot::Card)
            .chain((0..shop.colorless.len()).map(Slot::Colorless))
            .chain((0..shop.relics.len()).map(Slot::Relic))
            .chain((0..shop.potions.len()).map(Slot::Potion));
        for slot in slots.collect::<Vec<_>>() {
            let stocked = match slot {
                Slot::Card(i) => shop.cards[i].is_some(),
                Slot::Colorless(i) => shop.colorless[i].is_some(),
                Slot::Relic(i) => shop.relics[i].is_some(),
                Slot::Potion(i) => shop.potions[i].is_some() && self.potions.iter().any(Option::is_none) && !self.has_relic("SOZU"),
                Slot::Removal => false,
            };
            if stocked {
                let pickup = self.buy(shop, slot, 0);
                self.settle(pickup, chooser, log);
            }
        }
    }

    /// What winning an event's fight gives (`gold_proportion` as for any
    /// fight): a monster room's rewards with the event's added
    /// (`CombatRoom.ExtraRewards`), or, for Battleworn Dummy, which gives
    /// none, what the event does as it resumes: a potion reward, two random
    /// upgrades on the event's stream, or a relic off the front of the bag,
    /// unless the dummy ran out the clock (escaped, so no gold share).
    /// Returns the gold rolled, if any.
    pub fn event_fight_won(&mut self, fight: &EventFight, gold_proportion: f32, chooser: &mut impl Chooser, log: &mut Vec<Offered>) -> Option<i32> {
        self.fight_won(RoomType::Monster);
        match fight.dummy {
            None => {
                let rewards = self.fight_rewards(RoomType::Monster, gold_proportion, fight.gold, &fight.extra);
                let gold = (!rewards.gold.is_empty()).then(|| rewards.gold.iter().sum());
                self.take_rewards(rewards, chooser, log);
                gold
            }
            Some(_) if gold_proportion <= 0.0 => None,
            Some(setting) => {
                let offered = self.dummy_beaten(setting);
                self.settle(offered, chooser, log);
                None
            }
        }
    }

    /// An ancient (`AncientEventModel`): its options laid out on its own
    /// stream (`ancient_offer`), then the one taken, picked up. Returns what
    /// it laid out.
    pub fn ancient(&mut self, name: &str, chooser: &mut impl Chooser, log: &mut Vec<Offered>) -> AncientOffer {
        let offer = self.ancient_offer(name);
        if let Some(relic) = offer.relics.get(chooser.choose(self, Decision::Ancient(&offer.relics))) {
            let pickup = self.take_ancient(&offer, relic);
            self.settle(pickup, chooser, log);
        }
        offer
    }
}
