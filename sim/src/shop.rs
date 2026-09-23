//! A merchant, value for value: the stock as the game rolls it on entry
//! (`Entities/Merchant/MerchantInventory.cs`, `CreateForNormalMerchant`),
//! each entry's price (`MerchantCardEntry.cs`, `MerchantRelicEntry.cs`,
//! `MerchantPotionEntry.cs`, `MerchantCardRemovalEntry.cs`), and buying:
//! what a purchase does (`MerchantEntry.OnTryPurchaseWrapper`), The
//! Courier's restock and discount, Membership Card's discount, Lord's
//! Parasol buying everything.
//!
//! What is sold comes off the player's Shops stream; the card rarities, the
//! upgrade rolls (which never upgrade) and the two relic rarities come off
//! the Rewards stream, so a shop moves it by fourteen draws. Prices draw on
//! the Shops stream.
//!
//! Not here: the modifiers that change the pool.

use crate::game_rng::PlayerStream;
use crate::pools::{PoolCard, PotionRarity, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS, IRONCLAD_POTIONS, SHARED_POTIONS};
use crate::rewards::{create_potions, next_allowed, pull_relic_from_back, roll_relic_rarity, CardOdds, Offer, OddsType, Pulled};
use crate::run::RunState;
use crate::types::{AscensionLevel, CardType, RelicRarity};

/// An entry's item and the price it rolled (`MerchantEntry._cost`), before
/// the relics' discounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Priced<T> {
    pub item: T,
    pub cost: i32,
}

/// A merchant's stock, in the order the screen lays it out; an entry sold
/// and not restocked is `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shop {
    /// Two attacks, two skills and a power of the character's.
    pub cards: Vec<Option<Priced<Offer>>>,
    /// `_characterCardEntries` index of the card on sale, at half price,
    /// until The Courier restocks it.
    pub on_sale: Option<usize>,
    /// An uncommon and a rare colorless card.
    pub colorless: Vec<Option<Priced<Offer>>>,
    pub relics: Vec<Option<Priced<Pulled>>>,
    pub potions: Vec<Option<Priced<&'static str>>>,
    /// The card removal, until it is used (`MerchantCardRemovalEntry.Used`).
    pub removal: bool,
}

/// Where a ware sits in the shop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Card(usize),
    Colorless(usize),
    Relic(usize),
    Potion(usize),
    Removal,
}

/// What a ware is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Card(Offer),
    /// A relic's game id.
    Relic(String),
    Potion(&'static str),
    /// A card out of the deck.
    Removal,
}

/// Something the player can buy now, at the price they would pay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ware {
    pub slot: Slot,
    pub item: Item,
    pub price: i32,
}

impl Shop {
    /// As `tools/oracle rewards` prints a shop step, after the step's name.
    pub fn oracle_text(&self) -> String {
        let cards = |entries: &[Option<Priced<Offer>>]| -> String {
            entries.iter().flatten().map(|e| format!("{}{}", e.item.id, if e.item.upgraded { "+" } else { "" })).collect::<Vec<_>>().join(" ")
        };
        let relics: Vec<String> = self.relics.iter().flatten().map(|r| r.item.game_id()).collect();
        let potions: Vec<&str> = self.potions.iter().flatten().map(|p| p.item).collect();
        format!(
            "cards {} sale {} colorless {} relics {} potions {}",
            cards(&self.cards),
            self.on_sale.map_or("none".to_string(), |i| i.to_string()),
            cards(&self.colorless),
            relics.join(" "),
            potions.join(" ")
        )
    }

    /// Every entry's price as rolled, in screen order, as `tools/oracle
    /// rewards` prints them; an empty entry's as 0.
    pub fn costs(&self) -> Vec<i32> {
        let cards = self.cards.iter().chain(&self.colorless).map(|e| e.map_or(0, |e| e.cost));
        let relics = self.relics.iter().map(|e| e.map_or(0, |e| e.cost));
        cards.chain(relics).chain(self.potions.iter().map(|e| e.map_or(0, |e| e.cost))).collect()
    }

    /// The slot at `i` counting across the screen: the character's cards,
    /// the colorless ones, the relics, the potions.
    pub fn slot(&self, i: usize) -> Slot {
        let (cards, colorless, relics) = (self.cards.len(), self.colorless.len(), self.relics.len());
        match i {
            _ if i < cards => Slot::Card(i),
            _ if i < cards + colorless => Slot::Colorless(i - cards),
            _ if i < cards + colorless + relics => Slot::Relic(i - cards - colorless),
            _ => Slot::Potion(i - cards - colorless - relics),
        }
    }

    /// What slot `slot` holds, as the oracle names it.
    pub fn item_text(&self, slot: Slot) -> String {
        let card = |e: &Option<Priced<Offer>>| e.map_or("-".into(), |e| format!("{}{}", e.item.id, if e.item.upgraded { "+" } else { "" }));
        match slot {
            Slot::Card(i) => card(&self.cards[i]),
            Slot::Colorless(i) => card(&self.colorless[i]),
            Slot::Relic(i) => self.relics[i].map_or("-".into(), |e| e.item.game_id()),
            Slot::Potion(i) => self.potions[i].map_or("-".into(), |e| e.item.to_string()),
            Slot::Removal => "removal".into(),
        }
    }
}

/// `MerchantInventory._coloredCardTypes`.
const COLORED_TYPES: [CardType; 5] = [CardType::Attack, CardType::Attack, CardType::Skill, CardType::Skill, CardType::Power];

/// `MerchantInventory._colorlessCardRarities`.
const COLORLESS_RARITIES: [Rarity; 2] = [Rarity::Uncommon, Rarity::Rare];

impl RunState {
    /// `MerchantInventory.CreateForNormalMerchant`: the character's cards,
    /// the colorless ones, the relics, the potions, in that order, each
    /// priced as it is stocked, and the card removal.
    pub fn shop(&mut self) -> Shop {
        let on_sale = self.rngs.player(PlayerStream::Shops).next_int(COLORED_TYPES.len() as i32) as usize;
        let mut shop = Shop { cards: Vec::new(), on_sale: Some(on_sale), colorless: Vec::new(), relics: Vec::new(), potions: Vec::new(), removal: true };
        for i in 0..COLORED_TYPES.len() {
            let mut entry = self.stock_card(&shop, Slot::Card(i));
            if i == on_sale {
                // `SetOnSale` prices the card again, and halves it.
                entry.cost = self.card_cost(&entry.item) / 2;
            }
            shop.cards.push(Some(entry));
        }
        for i in 0..COLORLESS_RARITIES.len() {
            let entry = self.stock_card(&shop, Slot::Colorless(i));
            shop.colorless.push(Some(entry));
        }
        let rarities = [roll_relic_rarity(self.rewards()), roll_relic_rarity(self.rewards()), RelicRarity::Shop];
        for rarity in rarities {
            let entry = self.stock_relic(rarity);
            shop.relics.push(Some(entry));
        }
        for potion in create_potions(3, self.rngs.player(PlayerStream::Shops)) {
            let cost = self.potion_cost(potion);
            shop.potions.push(Some(Priced { item: potion, cost }));
        }
        shop
    }

    /// One card entry's `Populate`: `CardFactory.CreateForMerchant` over
    /// the pool less every card on the shelf (a character entry rolls a
    /// Shop rarity offset like a reward's, then the next rarity its type
    /// holds; a colorless entry has its rarity), the card off the Shops
    /// stream, its upgrade roll on the Rewards stream (from a chance of
    /// minus a billion), the eggs' upgrades
    /// (`ModifyMerchantCardCreationResults`), then its price.
    fn stock_card(&mut self, shop: &Shop, slot: Slot) -> Priced<Offer> {
        let shelved: Vec<&str> = shop.cards.iter().chain(&shop.colorless).flatten().map(|e| e.item.id).collect();
        let items: Vec<&'static PoolCard> = match slot {
            Slot::Card(i) => {
                let kind = COLORED_TYPES[i];
                let options = merchant_options(IRONCLAD_CARDS, &shelved);
                let rolled = CardOdds::roll_with_offset(OddsType::Shop, self.card_odds.0, &mut self.roll_ctx());
                let rarity = next_allowed(rolled, |r| options.iter().any(|c| c.rarity == r && c.kind == kind)).expect("a rarity the type holds");
                options.into_iter().filter(|c| c.rarity == rarity && c.kind == kind).collect()
            }
            Slot::Colorless(i) => merchant_options(COLORLESS_CARDS, &shelved).into_iter().filter(|c| c.rarity == COLORLESS_RARITIES[i]).collect(),
            _ => unreachable!("a card slot"),
        };
        let card = *self.rngs.player(PlayerStream::Shops).pick(&items).expect("a card to stock");
        self.rewards().next_float(1.0);
        let mut offer = [Offer::new(card.id)];
        self.upgrade_by_eggs(&mut offer);
        let cost = self.card_cost(&offer[0]);
        Priced { item: offer[0], cost }
    }

    /// `MerchantRelicEntry.FillSlot`: the back of the bag for `rarity`, of
    /// what shops may sell, then its price.
    fn stock_relic(&mut self, rarity: RelicRarity) -> Priced<Pulled> {
        let relic = pull_relic_from_back(&mut self.plan.player_bag, &mut self.plan.shared_bag, rarity, self.floor);
        // `RelicModel.MerchantCost`; Circlet's rarity is None.
        let base = match relic {
            Pulled::Circlet => 1,
            Pulled::Bag(r) => match r.rarity() {
                RelicRarity::Common => 175,
                RelicRarity::Uncommon => 225,
                RelicRarity::Rare => 275,
                RelicRarity::Shop => 200,
                other => panic!("a {other:?} relic in a shop"),
            },
        };
        let roll = self.rngs.player(PlayerStream::Shops).next_float_in(0.85, 1.15);
        Priced { item: relic, cost: (base as f32 * roll).round_ties_even() as i32 }
    }

    /// `MerchantCardEntry.CalcCost`: 50, 75 or 150 by rarity, colorless a
    /// sixth or so more, times a roll on the Shops stream.
    fn card_cost(&mut self, card: &Offer) -> i32 {
        let colorless = COLORLESS_CARDS.iter().find(|c| c.id == card.id);
        let rarity = colorless.or_else(|| IRONCLAD_CARDS.iter().find(|c| c.id == card.id)).expect("a pool card").rarity;
        let mut base = match rarity {
            Rarity::Rare => 150,
            Rarity::Uncommon => 75,
            _ => 50,
        };
        if colorless.is_some() {
            base = (base as f32 * 1.15).round_ties_even() as i32;
        }
        let roll = self.rngs.player(PlayerStream::Shops).next_float_in(0.95, 1.05);
        (base as f32 * roll).round_ties_even() as i32
    }

    /// `MerchantPotionEntry.CalcCost`: 50, 75 or 100 by rarity, times a
    /// roll on the Shops stream.
    fn potion_cost(&mut self, potion: &str) -> i32 {
        let rarity = IRONCLAD_POTIONS.iter().chain(SHARED_POTIONS).find(|p| p.id == potion).expect("a pool potion").rarity;
        let base = match rarity {
            PotionRarity::Rare => 100,
            PotionRarity::Uncommon => 75,
            PotionRarity::Common => 50,
        };
        let roll = self.rngs.player(PlayerStream::Shops).next_float_in(0.95, 1.05);
        (base as f32 * roll).round_ties_even() as i32
    }

    /// `MerchantEntry.Cost` in a merchant's room: the relics' discounts
    /// (`Hook.ModifyMerchantPrice`: The Courier a fifth off, Membership Card
    /// half), exact, then truncated.
    pub fn merchant_price(&self, cost: i32) -> i32 {
        let (mut num, mut den) = (cost, 1);
        if self.has_relic("THE_COURIER") {
            (num, den) = (num * 4, den * 5);
        }
        if self.has_relic("MEMBERSHIP_CARD") {
            den *= 2;
        }
        num / den
    }

    /// `MerchantCardRemovalEntry.CalcCost`: 75 and 25 more for each removal
    /// bought this run, 100 and 50 at Inflation.
    pub fn removal_cost(&self) -> i32 {
        let (base, step) = if self.ascension.has(AscensionLevel::Inflation) { (100, 50) } else { (75, 25) };
        base + step * self.shop_removals
    }

    /// What the player can buy now: each stocked entry they can pay for
    /// and take (a potion needs a free slot and no Sozu), and the removal
    /// while it is unused and a card can go.
    pub fn wares(&self, shop: &Shop) -> Vec<Ware> {
        let mut wares = Vec::new();
        let mut offer = |slot: Slot, item: Item, cost: i32| {
            let price = self.merchant_price(cost);
            if price <= self.gold {
                wares.push(Ware { slot, item, price });
            }
        };
        for (i, e) in shop.cards.iter().enumerate() {
            if let Some(e) = e {
                offer(Slot::Card(i), Item::Card(e.item), e.cost);
            }
        }
        for (i, e) in shop.colorless.iter().enumerate() {
            if let Some(e) = e {
                offer(Slot::Colorless(i), Item::Card(e.item), e.cost);
            }
        }
        for (i, e) in shop.relics.iter().enumerate() {
            if let Some(e) = e {
                offer(Slot::Relic(i), Item::Relic(e.item.game_id()), e.cost);
            }
        }
        let slot_free = self.potions.iter().any(Option::is_none) && !self.has_relic("SOZU");
        for (i, e) in shop.potions.iter().enumerate() {
            if let (Some(e), true) = (e, slot_free) {
                offer(Slot::Potion(i), Item::Potion(e.item), e.cost);
            }
        }
        if shop.removal && self.deck.iter().any(|c| c.removable()) {
            offer(Slot::Removal, Item::Removal, self.removal_cost());
        }
        wares
    }

    /// `OnTryPurchaseWrapper` for a stocked entry, paid `price` (none for
    /// Lord's Parasol): the item taken (the card into the deck, the relic
    /// obtained, the potion into a slot) and paid for, then the entry
    /// restocked with The Courier (`RestockAfterPurchase`) or emptied, then
    /// `AfterItemPurchased` (Maw Bank) and the shelf's cards seen again by
    /// the eggs (`UpdateEntry`). Returns what a relic's pickup offered.
    pub(crate) fn buy(&mut self, shop: &mut Shop, slot: Slot, price: i32) -> Vec<crate::effects::Offered> {
        let mut offered = Vec::new();
        match slot {
            Slot::Card(i) | Slot::Colorless(i) => {
                let entries = if matches!(slot, Slot::Card(_)) { &shop.cards } else { &shop.colorless };
                let card = entries[i].expect("a stocked card").item;
                self.add_card(card.into());
                self.lose_gold(price);
            }
            Slot::Relic(i) => {
                self.lose_gold(price);
                offered = self.obtain(&shop.relics[i].expect("a stocked relic").item.game_id());
            }
            Slot::Potion(i) => {
                self.add_potion(shop.potions[i].expect("a stocked potion").item);
                self.lose_gold(price);
            }
            Slot::Removal => unreachable!("the removal is `remove_for`"),
        }
        if self.has_relic("THE_COURIER") {
            self.restock(shop, slot);
        } else {
            match slot {
                Slot::Card(i) => shop.cards[i] = None,
                Slot::Colorless(i) => shop.colorless[i] = None,
                Slot::Relic(i) => shop.relics[i] = None,
                Slot::Potion(i) => shop.potions[i] = None,
                Slot::Removal => {}
            }
        }
        self.after_purchase(shop, price);
        offered
    }

    /// The card removal bought for `price`, deck card `card` taken out
    /// (`OneOffSynchronizer.DoLocalMerchantCardRemoval`); the removal is
    /// used for this shop, and costs more in the next.
    pub(crate) fn remove_for(&mut self, shop: &mut Shop, card: usize, price: i32) {
        self.lose_gold(price);
        self.deck.remove(card);
        self.shop_removals += 1;
        shop.removal = false;
        self.after_purchase(shop, price);
    }

    /// `Hook.AfterItemPurchased` (Maw Bank stops paying once gold is
    /// spent), then `UpdateEntries`: the eggs upgrade the cards left on the
    /// shelf.
    fn after_purchase(&mut self, shop: &mut Shop, price: i32) {
        if price > 0 {
            if let Some(bank) = self.relic_mut("MAW_BANK") {
                bank.flag = true;
            }
        }
        for entry in shop.cards.iter_mut().chain(&mut shop.colorless).flatten() {
            self.upgrade_by_eggs(std::slice::from_mut(&mut entry.item));
        }
    }

    /// `RestockAfterPurchase`: a card entry populated again, off sale; a
    /// relic entry filled from a new rarity roll; a potion entry with a new
    /// potion; each priced.
    pub(crate) fn restock(&mut self, shop: &mut Shop, slot: Slot) {
        match slot {
            Slot::Card(i) => {
                if shop.on_sale == Some(i) {
                    shop.on_sale = None;
                }
                shop.cards[i] = Some(self.stock_card(shop, slot));
            }
            Slot::Colorless(i) => shop.colorless[i] = Some(self.stock_card(shop, slot)),
            Slot::Relic(i) => {
                let rarity = roll_relic_rarity(self.rewards());
                shop.relics[i] = Some(self.stock_relic(rarity));
            }
            Slot::Potion(i) => {
                let potion = create_potions(1, self.rngs.player(PlayerStream::Shops))[0];
                let cost = self.potion_cost(potion);
                shop.potions[i] = Some(Priced { item: potion, cost });
            }
            Slot::Removal => {}
        }
    }
}

/// `CreateForMerchant`'s options: the pool less what the shop already
/// shelves, its basic cards and the multiplayer-only ones.
fn merchant_options(pool: &'static [PoolCard], shelved: &[&str]) -> Vec<&'static PoolCard> {
    pool.iter().filter(|c| c.rarity != Rarity::Basic && !c.multiplayer_only && !shelved.contains(&c.id)).collect()
}
