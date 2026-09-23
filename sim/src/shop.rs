//! A merchant's stock as the game rolls it on entry, value for value:
//! `Entities/Merchant/MerchantInventory.cs` (`CreateForNormalMerchant`),
//! `MerchantCardEntry.cs`, `MerchantRelicEntry.cs`, `MerchantPotionEntry.cs`
//! and `CardFactory.CreateForMerchant`. What is sold comes off the player's
//! Shops stream; the card rarities, the upgrade rolls (which never upgrade)
//! and the two relic rarities come off the Rewards stream, so a shop moves
//! it by fourteen draws. Prices draw on the Shops stream too; they are drawn
//! here and not kept.
//!
//! Not here: restocking after a purchase (`Hook.ShouldRefillMerchantEntry`,
//! The Courier) and the modifiers that change the pool.

use crate::game_rng::PlayerStream;
use crate::pools::{PoolCard, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS};
use crate::rewards::{create_potions, next_allowed, pull_relic_from_back, roll_relic_rarity, CardOdds, Offer, OddsType, Pulled};
use crate::run::RunState;
use crate::types::{CardType, RelicRarity};

/// A merchant's stock, in the order the screen lays it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shop {
    /// Two attacks, two skills and a power of the character's.
    pub cards: Vec<Offer>,
    /// `_characterCardEntries` index of the card on sale, at half price.
    pub on_sale: usize,
    /// An uncommon and a rare colorless card.
    pub colorless: Vec<Offer>,
    pub relics: Vec<Pulled>,
    pub potions: Vec<&'static str>,
}

impl Shop {
    /// As `tools/oracle rewards` prints a shop step, after the step's name.
    pub fn oracle_text(&self) -> String {
        let cards = |offers: &[Offer]| -> String {
            offers.iter().map(|o| format!("{}{}", o.id, if o.upgraded { "+" } else { "" })).collect::<Vec<_>>().join(" ")
        };
        let relics: Vec<String> = self.relics.iter().map(|r| r.game_id()).collect();
        format!(
            "cards {} sale {} colorless {} relics {} potions {}",
            cards(&self.cards),
            self.on_sale,
            cards(&self.colorless),
            relics.join(" "),
            self.potions.join(" ")
        )
    }
}

/// `MerchantInventory._coloredCardTypes`.
const COLORED_TYPES: [CardType; 5] = [CardType::Attack, CardType::Attack, CardType::Skill, CardType::Skill, CardType::Power];

impl RunState {
    /// `MerchantInventory.CreateForNormalMerchant`: the character's cards,
    /// the colorless ones, the relics, the potions, in that order.
    pub fn shop(&mut self) -> Shop {
        let on_sale = self.rngs.player(PlayerStream::Shops).next_int(COLORED_TYPES.len() as i32) as usize;
        let mut stocked: Vec<&'static str> = Vec::new();
        let mut cards = Vec::new();
        for (i, kind) in COLORED_TYPES.into_iter().enumerate() {
            // `CreateForMerchant(type)`: a Shop rarity roll offset like a
            // reward's, then the next rarity the type holds.
            let options = merchant_options(IRONCLAD_CARDS, &stocked);
            let rolled = CardOdds::roll_with_offset(OddsType::Shop, self.card_odds.0, &mut self.roll_ctx());
            let rarity = next_allowed(rolled, |r| options.iter().any(|c| c.rarity == r && c.kind == kind)).expect("a rarity the type holds");
            let items: Vec<&PoolCard> = options.into_iter().filter(|c| c.rarity == rarity && c.kind == kind).collect();
            cards.push(self.stock_card(&items, &mut stocked));
            if i == on_sale {
                // `SetOnSale` prices the card again.
                self.price();
            }
        }
        let mut colorless = Vec::new();
        for rarity in [Rarity::Uncommon, Rarity::Rare] {
            let items: Vec<&PoolCard> = merchant_options(COLORLESS_CARDS, &stocked).into_iter().filter(|c| c.rarity == rarity).collect();
            colorless.push(self.stock_card(&items, &mut stocked));
        }
        let rarities = [roll_relic_rarity(self.rewards()), roll_relic_rarity(self.rewards()), RelicRarity::Shop];
        let relics = rarities
            .into_iter()
            .map(|rarity| {
                let relic = pull_relic_from_back(&mut self.plan.player_bag, &mut self.plan.shared_bag, rarity, self.floor);
                self.rngs.player(PlayerStream::Shops).next_float_in(0.85, 1.15);
                relic
            })
            .collect();
        let potions = create_potions(3, self.rngs.player(PlayerStream::Shops));
        for _ in &potions {
            self.price();
        }
        Shop { cards, on_sale, colorless, relics, potions }
    }

    /// One card entry's `Populate`: a card off the Shops stream, its upgrade
    /// roll on the Rewards stream (from a chance of minus a billion), the
    /// eggs' upgrades (`ModifyMerchantCardCreationResults`), then its price.
    fn stock_card(&mut self, items: &[&'static PoolCard], stocked: &mut Vec<&'static str>) -> Offer {
        let card = *self.rngs.player(PlayerStream::Shops).pick(items).expect("a card to stock");
        self.rewards().next_float(1.0);
        stocked.push(card.id);
        let mut offer = [Offer { id: card.id, upgraded: false }];
        self.upgrade_by_eggs(&mut offer);
        self.price();
        offer[0]
    }

    /// A card's or potion's `CalcCost`: one draw on the Shops stream.
    fn price(&mut self) {
        self.rngs.player(PlayerStream::Shops).next_float_in(0.95, 1.05);
    }
}

/// `CreateForMerchant`'s options: the pool less what the shop already
/// stocks, its basic cards and the multiplayer-only ones.
fn merchant_options(pool: &'static [PoolCard], stocked: &[&str]) -> Vec<&'static PoolCard> {
    pool.iter().filter(|c| c.rarity != Rarity::Basic && !c.multiplayer_only && !stocked.contains(&c.id)).collect()
}
