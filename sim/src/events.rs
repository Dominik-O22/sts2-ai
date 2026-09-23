//! What an event's chosen option draws on the player's Rewards stream or
//! pulls from a grab bag (`Models/Events/<Name>.cs`). An event's own choices
//! (its variables, which option does what) draw on its own stream and are
//! not here; what the option gives the player (cards, relics, gold) is the
//! caller's to read off the record.
//!
//! Only the events the checked runs met that draw are ported option by
//! option; the rest are listed as drawing nothing, or not ported.

use crate::pools::{PoolCard, PoolPotion, PotionRarity, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS, IRONCLAD_POTIONS, SHARED_POTIONS};
use crate::types::{CardType, RelicRarity};
use crate::game_rng::PlayerStream;
use crate::plan::BagRelic;
use crate::effects::Offered;
use crate::rewards::{create_cards, create_potion, CardOptions, OddsType, Offer, Pulled};
use crate::run::RunState;

/// Events none of whose options draw on the Rewards stream or pull from a
/// grab bag: the relics they give are named ones (Fresnel Lens, Sword of
/// Stone), and their other rolls use the event's own stream. The ones that
/// can start a fight are left out.
const QUIET: &[&str] = &[
    "AbyssalBaths", "Amalgamator", "AromaOfChaos", "Bugslayer", "ByrdonisNest", "ColossalFlower", "DollRoom",
    "DoorsOfLightAndDark", "DrowningBeacon", "FieldOfManSizedHoles", "GraveOfTheForgotten", "HungryForMushrooms",
    "JungleMazeAdventure", "LostWisp", "MorphicGrove", "Reflections", "SapphireSeed", "SelfHelpBook", "SlipperyBridge",
    "SpiralingWhirlpool", "SpiritGrafter", "StoneOfAllTime", "SunkenStatue", "SunkenTreasury", "Symbiote",
    "TabletOfTruth", "TeaMaster", "TinkerTime", "TrashHeap", "WaterloggedScriptorium", "WoodCarvings", "ZenWeaver",
];

impl RunState {
    /// The Rewards stream draws of an event's chosen options, with what they
    /// offered. `choices` are the options in the order they were taken, each
    /// as its page and name in the record (`INITIAL` and `GORGE` in
    /// `ROOM_FULL_OF_CHEESE.pages.INITIAL.options.GORGE.title`). A relic the
    /// event gives off a bag is obtained here, pickup and all
    /// (`Offered::Took`). An error names what is not ported.
    pub fn event_option(&mut self, event: &str, choices: &[(String, String)]) -> Result<Vec<Offered>, String> {
        if QUIET.contains(&event) {
            return Ok(Vec::new());
        }
        let first = choices.first().map_or("", |(_, option)| option.as_str());
        let last = choices.last().map(|(page, option)| (page.as_str(), option.as_str())).unwrap_or_default();
        let offered = match (event, first) {
            // `RoomFullOfCheese.Gorge`: eight commons to take two of.
            ("RoomFullOfCheese", "GORGE") => {
                vec![Offered::Cards(self.event_cards(8, IRONCLAD_CARDS, OddsType::Uniform, |c| c.rarity == Rarity::Common))]
            }
            ("RoomFullOfCheese", "SEARCH") => Vec::new(),
            // `BrainLeech.ShareKnowledge`: five of the character's to take
            // one of; `Rip`: a colorless card reward.
            ("BrainLeech", "SHARE_KNOWLEDGE") => vec![Offered::Cards(self.event_cards(5, IRONCLAD_CARDS, OddsType::Regular, |_| true))],
            ("BrainLeech", "RIP") => vec![Offered::Cards(self.event_card_reward(COLORLESS_CARDS))],
            // `ThisOrThat.Ornate`, `UnrestSite.Kill`: a relic off the front of
            // the bag, taken.
            ("ThisOrThat", "ORNATE") | ("UnrestSite", "KILL") => {
                let relic = self.relic_reward().game_id();
                self.take(relic)
            }
            ("ThisOrThat", "PLAIN") | ("UnrestSite", "REST") => Vec::new(),
            // `WhisperingHollow.Gold`: two potion rewards; `Hug` transforms
            // on the event's stream.
            ("WhisperingHollow", "GOLD") => {
                let potions = (0..2).map(|_| create_potion(self.rewards()).to_string()).collect();
                vec![Offered::Potions(potions)]
            }
            ("WhisperingHollow", "HUG") => Vec::new(),
            // `InfestedAutomaton.Study`: one of the character's powers.
            // `TouchCore` wants the cards' costs, which the pools lack.
            ("InfestedAutomaton", "STUDY") => {
                vec![Offered::Cards(self.event_cards(1, IRONCLAD_CARDS, OddsType::Regular, |c| c.kind == CardType::Power))]
            }
            // `PotionCourier.Ransack`: an uncommon potion as a reward;
            // `GrabPotions` offers named ones.
            ("PotionCourier", "RANSACK") => {
                let uncommon: Vec<&PoolPotion> =
                    IRONCLAD_POTIONS.iter().chain(SHARED_POTIONS).filter(|p| p.rarity == PotionRarity::Uncommon).collect();
                let potion = self.rewards().pick(&uncommon).expect("an uncommon potion").id;
                vec![Offered::Potions(vec![potion.to_string()])]
            }
            ("PotionCourier", "GRAB_POTIONS") => Vec::new(),
            // `WelcomeToWongos`: its featured item, a rare a shop could
            // sell, comes off the bag as the options are laid out; the
            // bargain bin sells the next such common. The mystery box sells
            // Wongo's Mystery Ticket, which is not ported.
            ("WelcomeToWongos", "BARGAIN_BIN" | "FEATURED_ITEM" | "LEAVE") => {
                let featured = self.pull_for_shop(RelicRarity::Rare).game_id();
                match first {
                    "BARGAIN_BIN" => {
                        let relic = self.pull_for_shop(RelicRarity::Common).game_id();
                        self.take(relic)
                    }
                    "FEATURED_ITEM" => self.take(featured),
                    _ => Vec::new(),
                }
            }
            // `FakeMerchant`: six of its fake relics on the shelf, each
            // priced on the Shops stream (`MerchantRelicEntry.CalcCost`);
            // which six is the event's own roll. The fight a thrown Foul
            // Potion starts is not ported.
            ("FakeMerchant", _) => {
                for _ in 0..6 {
                    self.rngs.player(PlayerStream::Shops).next_float_in(0.85, 1.15);
                }
                Vec::new()
            }
            // `DenseVegetation`: resting (`MimicRestSiteHeal`) leads to a
            // fight, whose rewards are a monster room's.
            ("DenseVegetation", "REST") => self.rest_heal(),
            ("DenseVegetation", "TRUDGE_ON") => Vec::new(),
            // `RelicTrader`: three relics off the front of the bag, one per
            // trade offered (`NewRelics`), the chosen one taken.
            ("RelicTrader", "TOP" | "MIDDLE" | "BOTTOM") => {
                let relics: Vec<String> = (0..3).map(|_| self.relic_reward().game_id()).collect();
                let i = ["TOP", "MIDDLE", "BOTTOM"].iter().position(|&o| o == first).expect("a trade");
                self.take(relics[i].clone())
            }
            // `Trial`: the defendant is the event's roll; the verdict decides.
            ("Trial", "ACCEPT") => match last {
                // `MerchantGuilty`: two relics off the front of the bag.
                ("MERCHANT", "GUILTY") => {
                    let mut offered = Vec::new();
                    for _ in 0..2 {
                        let relic = self.relic_reward().game_id();
                        offered.extend(self.take(relic));
                    }
                    offered
                }
                // `NondescriptGuilty`: two card rewards of the character's.
                ("NONDESCRIPT", "GUILTY") => {
                    (0..2).map(|_| Offered::Cards(self.event_card_reward(IRONCLAD_CARDS))).collect()
                }
                ("MERCHANT" | "NOBLE" | "NONDESCRIPT", "INNOCENT") | ("NOBLE", "GUILTY") => Vec::new(),
                _ => return Err(format!("Trial's {last:?} is not ported")),
            },
            _ => return Err(format!("event {event} ({first}) is not ported")),
        };
        Ok(offered)
    }

    /// What an ancient's options draw on the Rewards stream as they are
    /// laid out (`GenerateInitialOptions`), given the options it showed:
    /// Darv readies a Dusty Tome it offers (`DustyTome.SetupForPlayer`),
    /// picking the ancient card it will give among the character's, less
    /// the transcendence ones (Break, for the Ironclad).
    pub fn ancient_options(&mut self, ancient: &str, options: &[String]) {
        if ancient == "Darv" && options.iter().any(|o| o == "DUSTY_TOME") {
            let cards: Vec<&PoolCard> = IRONCLAD_CARDS.iter().filter(|c| c.rarity == Rarity::Ancient && c.id != "BREAK").collect();
            self.rewards().pick(&cards);
        }
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
        self.upgrade_by_crucible(&mut cards);
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
}
