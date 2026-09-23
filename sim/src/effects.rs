//! What the run's rooms and relics do to the player between fights, value
//! for value with the game: gold (`PlayerCmd.GainGold`), HP and max HP
//! (`CreatureCmd.Heal`, `GainMaxHp`, `LoseMaxHp`), cards joining and
//! leaving the deck (`CardPileCmd.Add`, `RemoveFromDeck`, `CardCmd.Upgrade`,
//! `Enchant`), potions (`PotionCmd.TryToProcure`), what a relic does when
//! picked up (`RelicCmd.Obtain`, each relic's `AfterObtained`), rest site
//! options (`Entities/RestSite/*RestSiteOption.cs`) and what entering a
//! room does (ancients' heal, `AfterRoomEntered`).
//!
//! Nothing here asks the player. What needs a choice comes back as an
//! `Offered` for `rooms.rs` to put to a `Chooser`; a pickup the port lacks
//! comes back as `Offered::Unported`, with the relic held and its effect
//! skipped.

use crate::card::{def, Card};
use crate::enchant::Enchantment;
use crate::game_rng::{PlayerStream, RunStream};
use crate::pools::{sim_card, sim_enchantment, PoolCard, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS};
use crate::rewards::{create_cards, create_potion, create_potions, CardOptions, Offer, UNPORTED_RELICS};
use crate::run::{DeckCard, Enchant, Room, RoomType, RunState};
use crate::types::{AscensionLevel, CardType};

/// What an effect put in front of the player, or did for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Offered {
    /// Cards to take one of, or none (a card reward, Hefty Tablet).
    Cards(Vec<Offer>),
    /// Cards put in the deck with no choice (Arcane Scroll's).
    Gained(Vec<Offer>),
    /// Relics offered as rewards to take (Neow's Bones, Small Capsule).
    Relics(Vec<String>),
    /// A relic handed over without a choice (Large Capsule, an event),
    /// already obtained.
    Took(Vec<String>),
    /// Bundles of cards to take one of (Scroll Boxes).
    Bundles(Vec<Vec<Offer>>),
    /// Potions offered as rewards.
    Potions(Vec<String>),
    /// Cards to pick out of the deck, and what happens to them.
    Pick(DeckPick),
    /// A relic whose pickup, or whose draws on the Rewards stream, the port
    /// lacks: held, with that skipped.
    Unported(String),
}

/// A choice of cards from the deck (`CardSelectCmd.FromDeck*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeckPick {
    pub action: DeckAction,
    /// The deck indices on offer, in the order the screen shows them.
    pub cards: Vec<usize>,
    pub min: usize,
    pub max: usize,
}

/// What happens to the cards a `DeckPick` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeckAction {
    Upgrade,
    Remove,
    /// Dolly's Mirror: a copy joins the deck.
    Duplicate,
    /// An enchantment by game id, with its amount.
    Enchant(&'static str, i32),
    /// New Leaf, Astrolabe: each to a random card on the run's Niche
    /// stream, upgraded with `upgrade` (`transform`).
    Transform { upgrade: bool },
}

/// The stream a transform draws its new card on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformRng {
    /// The player's Transformations stream (Leafy Poultice).
    Transformations,
    /// The run's Niche stream (New Leaf, Astrolabe, Pandora's Box).
    Niche,
}

/// `Entities/RestSite/*RestSiteOption.cs` by `OptionId`, the ones a
/// singleplayer run can be offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestOption {
    Heal,
    Smith,
    /// Girya.
    Lift,
    /// Shovel.
    Dig,
    /// Pumpkin Candle.
    Kindle,
    /// Meat Cleaver.
    Cook,
}

impl RestOption {
    /// The option a history file's `rest_site_choices` names.
    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "HEAL" => RestOption::Heal,
            "SMITH" => RestOption::Smith,
            "LIFT" => RestOption::Lift,
            "DIG" => RestOption::Dig,
            "KINDLE" => RestOption::Kindle,
            "COOK" => RestOption::Cook,
            _ => return None,
        })
    }
}

/// Every relic with an `AfterObtained`. The ones `RunState::obtain` has
/// no arm for are `Offered::Unported` when picked up.
const PICKUPS: &[&str] = &[
    "ALCHEMICAL_COFFER", "ARCANE_SCROLL", "ARCHAIC_TOOTH", "ASTROLABE", "BEAUTIFUL_BRACELET", "BELT_BUCKLE",
    "BIG_MUSHROOM", "BIIIG_HUG", "BLOOD_SOAKED_ROSE", "BYRDPIP", "CALLING_BELL", "CAULDRON", "CLAWS", "CURSED_PEARL",
    "DISTINGUISHED_CAPE", "DOLLYS_MIRROR", "DUSTY_TOME", "ELECTRIC_SHRYMP", "EMPTY_CAGE", "FAKE_LEES_WAFFLE",
    "FAKE_MANGO", "FAKE_SNECKO_EYE", "FRAGRANT_MUSHROOM", "FUR_COAT", "GLASS_EYE", "GNARLED_HAMMER", "GOLDEN_COMPASS",
    "GOLDEN_PEARL", "HEFTY_TABLET", "JEWELRY_BOX", "KALEIDOSCOPE", "KIFUDA", "LARGE_CAPSULE", "LEAD_PAPERWEIGHT",
    "LEAFY_POULTICE", "LEES_WAFFLE", "LOOMING_FRUIT", "LOST_COFFER", "MANGO", "MASSIVE_SCROLL", "NEOWS_BONES",
    "NEOWS_TALISMAN", "NEOWS_TORMENT", "NEW_LEAF", "NUTRITIOUS_OYSTER", "NUTRITIOUS_SOUP", "OLD_COIN", "ORRERY",
    "PAELS_CLAW", "PAELS_EYE", "PAELS_GROWTH", "PAELS_HORN", "PAELS_LEGION", "PAELS_TOOTH", "PANDORAS_BOX", "PEAR",
    "PHIAL_HOLSTER", "POMANDER", "POTION_BELT", "PRECARIOUS_SHEARS", "PRECISE_SCISSORS", "PRESERVED_FOG",
    "PUMPKIN_CANDLE", "PUNCH_DAGGER", "ROYAL_STAMP", "SAND_CASTLE", "SCROLL_BOXES", "SEA_GLASS", "SERE_TALON",
    "SIGNET_RING", "SILKEN_TRESS", "SMALL_CAPSULE", "SNECKO_EYE", "STORYBOOK", "STRAWBERRY", "TANXS_WHISTLE",
    "TOUCH_OF_OROBAS", "TOY_BOX", "TRI_BOOMERANG", "WAR_PAINT", "WHETSTONE", "YUMMY_COOKIE",
];

/// `Neow.AllPossibleOptions`' relics in order, as Neow's Bones lists them
/// (`GetValidRelics`): itself left out, and Massive Scroll, which
/// `IsAllowed` keeps to multiplayer. Kaleidoscope and Scroll Boxes pass
/// their `IsAllowedAtNeow` on a fully unlocked profile.
const NEOW_RELICS: &[&str] = &[
    "CURSED_PEARL", "HEFTY_TABLET", "LARGE_CAPSULE", "LEAFY_POULTICE", "PRECARIOUS_SHEARS", "SILKEN_TRESS",
    "SILVER_CRUCIBLE", "ARCANE_SCROLL", "BOOMING_CONCH", "FISHING_ROD", "GOLDEN_PEARL", "KALEIDOSCOPE",
    "LEAD_PAPERWEIGHT", "LOST_COFFER", "NEOWS_TORMENT", "NEW_LEAF", "PHIAL_HOLSTER", "PRECISE_SCISSORS",
    "SCROLL_BOXES", "WINGED_BOOTS", "LAVA_ROCK", "NEOWS_TALISMAN", "NUTRITIOUS_OYSTER", "POMANDER", "SMALL_CAPSULE",
    "STONE_HUMIDIFIER",
];

/// `CurseCardPool` less the curses that are not `CanBeGeneratedByModifiers`,
/// ordered by id, as Neow's Bones and Sere Talon draw from it.
const MODIFIER_CURSES: &[&str] = &["CLUMSY", "DEBT", "DECAY", "DOUBT", "GUILTY", "INJURY", "NORMALITY", "REGRET", "SHAME", "WRITHE"];

/// Cards with `CardKeyword.Eternal`, which no screen removes.
const ETERNAL: &[&str] = &["ASCENDERS_BANE", "BAD_LUCK", "CURSE_OF_THE_BELL", "ENTHRALLED", "FOLLY", "FORBIDDEN_GRIMOIRE", "GREED"];

/// `CardFactory.GetDefaultTransformationOptions` out of combat: the card's
/// pool, its common, uncommon and rare cards less the card itself and the
/// multiplayer-only ones. An ancient card, or one outside the pools (event
/// and token cards), draws from the colorless pool. `None` for a curse or a
/// status, whose pools the port does not have.
fn transform_options(card: &DeckCard) -> Option<Vec<&'static str>> {
    let own = IRONCLAD_CARDS.iter().find(|c| c.id == card.id);
    let pool = match own {
        Some(c) if c.rarity != Rarity::Ancient => IRONCLAD_CARDS,
        _ if matches!(card.kind(), Some(CardType::Curse | CardType::Status) | None) => return None,
        _ => COLORLESS_CARDS,
    };
    let kept = |c: &&PoolCard| matches!(c.rarity, Rarity::Common | Rarity::Uncommon | Rarity::Rare) && c.id != card.id && !c.multiplayer_only;
    Some(pool.iter().filter(kept).map(|c| c.id).collect())
}

impl From<Offer> for DeckCard {
    fn from(offer: Offer) -> Self {
        let enchantment = offer.enchantment.map(|(id, amount)| Enchant { id: id.to_string(), amount });
        DeckCard { id: offer.id.to_string(), upgraded: offer.upgraded, enchantment }
    }
}

impl DeckCard {
    /// Its `CardType`, if the combat sim knows the card.
    pub fn kind(&self) -> Option<CardType> {
        sim_card(&self.id).map(|c| def(c).ty)
    }

    /// `CardModel.IsUpgradable`: statuses and curses never are.
    pub fn upgradable(&self) -> bool {
        !self.upgraded && !matches!(self.kind(), Some(CardType::Status | CardType::Curse))
    }

    /// `CardModel.IsRemovable`: not Eternal, by its class or by Tezcatara's
    /// Ember.
    pub fn removable(&self) -> bool {
        !ETERNAL.contains(&self.id.as_str()) && self.enchantment.as_ref().is_none_or(|e| e.id != "TEZCATARAS_EMBER")
    }

    /// `EnchantmentModel.CanEnchant`: the enchantment's own rule on the
    /// card (`enchant.rs`), and no enchantment there already, since none
    /// stacks.
    pub fn can_enchant(&self, enchantment: &str) -> bool {
        let (Some(card), Some(id)) = (sim_card(&self.id), sim_enchantment(enchantment)) else { return false };
        self.enchantment.is_none() && Enchantment::new(id, 0).can_enchant(&Card::new(0, card, self.upgraded))
    }
}

impl RunState {
    /// `PlayerCmd.GainGold`: `Hook.ModifyGoldGained` (Bowler Hat a quarter
    /// more, Ectoplasm none), truncated, then `AfterGoldGained` (Dragon
    /// Fruit's max HP).
    pub fn gain_gold(&mut self, amount: i32) {
        let mut gold = amount as f64;
        for relic in &self.relics {
            match relic.id.as_str() {
                "BOWLER_HAT" => gold *= 1.25,
                "ECTOPLASM" => gold = 0.0,
                _ => {}
            }
        }
        if gold <= 0.0 {
            return;
        }
        self.gold += gold as i32;
        if self.has_relic("DRAGON_FRUIT") {
            self.gain_max_hp(1);
        }
    }

    /// `PlayerCmd.LoseGold`, which spends what there is.
    pub fn lose_gold(&mut self, amount: i32) {
        self.gold = (self.gold - amount).max(0);
    }

    /// `CreatureCmd.Heal`: `HealInternal` truncates HP after adding the
    /// decimal amount, which for a whole HP is truncating the amount.
    pub fn heal(&mut self, amount: i32) {
        self.hp = (self.hp + amount.max(0)).min(self.max_hp);
    }

    /// `CreatureCmd.GainMaxHp`: the new max, then a heal of what it added.
    pub fn gain_max_hp(&mut self, amount: i32) {
        self.max_hp += amount;
        self.heal(amount);
    }

    /// `CreatureCmd.LoseMaxHp`: HP above the new max is lost first; the max
    /// stays at least 1.
    pub fn lose_max_hp(&mut self, amount: i32) {
        let max = self.max_hp - amount;
        self.hp = self.hp.min(max);
        self.max_hp = max.max(1);
    }

    /// Unblockable damage out of combat (Precarious Shears).
    pub fn lose_hp(&mut self, amount: i32) {
        self.hp = (self.hp - amount).max(0);
    }

    /// `CardPileCmd.Add` to the deck: `Hook.ModifyCardBeingAddedToDeck`
    /// (the eggs upgrade their type, Fresnel Lens makes a block card
    /// Nimble), then `AfterCardChangedPiles` (Lucky Fysh's gold).
    pub fn add_card(&mut self, mut card: DeckCard) {
        let kind = card.kind();
        let eggs = [("MOLTEN_EGG", CardType::Attack), ("TOXIC_EGG", CardType::Skill), ("FROZEN_EGG", CardType::Power)];
        for relic in self.relics.iter().map(|r| r.id.as_str()) {
            match relic {
                "FRESNEL_LENS" if card.can_enchant("NIMBLE") => card.enchantment = Some(Enchant { id: "NIMBLE".into(), amount: 2 }),
                egg => {
                    if eggs.iter().any(|&(e, k)| e == egg && kind == Some(k)) && card.upgradable() {
                        card.upgraded = true;
                    }
                }
            }
        }
        self.deck.push(card);
        if self.has_relic("LUCKY_FYSH") {
            self.gain_gold(15);
        }
    }

    /// `CardCmd.Upgrade` on a deck card.
    pub fn upgrade_card(&mut self, i: usize) {
        if self.deck[i].upgradable() {
            self.deck[i].upgraded = true;
        }
    }

    /// `PotionCmd.TryToProcure`: Sozu refuses it, else the first empty
    /// slot takes it. False if it was not kept.
    pub fn add_potion(&mut self, id: &str) -> bool {
        if self.has_relic("SOZU") {
            return false;
        }
        match self.potions.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => {
                *slot = Some(id.to_string());
                true
            }
            None => false,
        }
    }

    /// Does what a `DeckPick` says to the cards taken, deck indices.
    pub fn apply_pick(&mut self, action: DeckAction, chosen: &[usize]) {
        let mut chosen = chosen.to_vec();
        chosen.sort_unstable();
        match action {
            DeckAction::Upgrade => chosen.iter().for_each(|&i| self.upgrade_card(i)),
            DeckAction::Remove => {
                for &i in chosen.iter().rev() {
                    self.deck.remove(i);
                }
            }
            // `RunState.CloneCard`: the copy keeps its upgrade and enchantment.
            DeckAction::Duplicate => {
                for &i in &chosen {
                    let copy = self.deck[i].clone();
                    self.add_card(copy);
                }
            }
            DeckAction::Enchant(id, amount) => {
                for &i in &chosen {
                    self.deck[i].enchantment = Some(Enchant { id: id.to_string(), amount });
                }
            }
            DeckAction::Transform { upgrade } => self.transform(&chosen, upgrade, TransformRng::Niche),
        }
    }

    /// `CardCmd.Transform` out of combat of the deck cards `cards`: for
    /// each in turn its new card drawn on `stream`
    /// (`CardFactory.CreateRandomCardForTransform`, `transform_options`) and
    /// `upgrade`d if asked; then the originals leave and the new cards join
    /// the deck through `add_card`, in the originals' deck order.
    pub fn transform(&mut self, cards: &[usize], upgrade: bool, stream: TransformRng) {
        let mut replaced: Vec<(usize, &'static str)> = Vec::new();
        for &i in cards {
            let Some(options) = transform_options(&self.deck[i]) else { continue };
            let rng = match stream {
                TransformRng::Transformations => self.rngs.player(PlayerStream::Transformations),
                TransformRng::Niche => self.rngs.run(RunStream::Niche),
            };
            replaced.push((i, *rng.pick(&options).expect("a card to transform into")));
        }
        replaced.sort_unstable();
        for &(i, _) in replaced.iter().rev() {
            self.deck.remove(i);
        }
        for (_, id) in replaced {
            self.add_card(DeckCard { upgraded: upgrade, ..DeckCard::new(id) });
        }
    }

    /// The deck indices a pick of `action` may take, in deck order.
    pub(crate) fn pickable(&self, action: DeckAction) -> Vec<usize> {
        let ok = |c: &DeckCard| match action {
            DeckAction::Upgrade => c.upgradable(),
            DeckAction::Remove => c.removable(),
            // Dolly's Mirror leaves out quest cards, which the Ironclad never holds.
            DeckAction::Duplicate => true,
            DeckAction::Enchant(id, _) => c.can_enchant(id),
            DeckAction::Transform { .. } => c.removable() && transform_options(c).is_some(),
        };
        (0..self.deck.len()).filter(|&i| ok(&self.deck[i])).collect()
    }

    fn pick(&self, action: DeckAction, min: usize, max: usize) -> Offered {
        Offered::Pick(DeckPick { action, cards: self.pickable(action), min, max })
    }

    /// `StableShuffle` on the run's Niche stream of the deck cards `keep`
    /// takes, then the first `count` upgraded (Whetstone, War Paint, Sand
    /// Castle). The sort is by id then upgrade level, as `CardModel`
    /// compares.
    fn upgrade_random(&mut self, count: usize, keep: impl Fn(&DeckCard) -> bool) {
        let mut cards: Vec<usize> = (0..self.deck.len()).filter(|&i| keep(&self.deck[i]) && self.deck[i].upgradable()).collect();
        cards.sort_by(|&a, &b| (&self.deck[a].id, self.deck[a].upgraded).cmp(&(&self.deck[b].id, self.deck[b].upgraded)));
        self.rngs.run(RunStream::Niche).shuffle(&mut cards);
        for &i in cards.iter().take(count) {
            self.upgrade_card(i);
        }
    }

    /// `RelicCmd.Obtain`, then the relic's `AfterObtained`: what it did,
    /// what it offers, and `Unported` for a pickup the port lacks or a
    /// relic whose Rewards stream draws it lacks (`UNPORTED_RELICS`).
    pub fn obtain(&mut self, id: &str) -> Vec<Offered> {
        self.obtain_relic(id);
        let mut offered = Vec::new();
        if UNPORTED_RELICS.contains(&id) {
            offered.push(Offered::Unported(id.to_string()));
            return offered;
        }
        let is_type = |kind: CardType| move |c: &DeckCard| c.kind() == Some(kind);
        match id {
            "STRAWBERRY" => self.gain_max_hp(7),
            "PEAR" => self.gain_max_hp(10),
            "MANGO" => self.gain_max_hp(14),
            "FAKE_MANGO" => self.gain_max_hp(3),
            "NUTRITIOUS_OYSTER" => self.gain_max_hp(11),
            "BIG_MUSHROOM" => self.gain_max_hp(20),
            "LOOMING_FRUIT" => self.gain_max_hp(31),
            "LEES_WAFFLE" => {
                self.gain_max_hp(7);
                self.heal(self.max_hp - self.hp);
            }
            // A tenth of max HP, truncated as `HealInternal` does.
            "FAKE_LEES_WAFFLE" => self.heal(self.max_hp / 10),
            "OLD_COIN" => self.gain_gold(300),
            "GOLDEN_PEARL" => self.gain_gold(150),
            "CURSED_PEARL" => {
                self.add_card(DeckCard::new("GREED"));
                self.gain_gold(333);
            }
            "POTION_BELT" => self.potions.extend([None, None]),
            // `PhialHolster`: a slot, then two potions on the run's
            // CombatPotionGeneration stream.
            "PHIAL_HOLSTER" => {
                self.potions.push(None);
                for potion in create_potions(2, self.rngs.run(RunStream::CombatPotionGeneration)) {
                    self.add_potion(potion);
                }
            }
            "WHETSTONE" => self.upgrade_random(2, is_type(CardType::Attack)),
            "WAR_PAINT" => self.upgrade_random(2, is_type(CardType::Skill)),
            "SAND_CASTLE" => self.upgrade_random(6, |_| true),
            // `NeowsTalisman`: the last basic Strike and Defend.
            "NEOWS_TALISMAN" => {
                for basic in ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"] {
                    if let Some(i) = self.deck.iter().rposition(|c| c.id == basic) {
                        self.upgrade_card(i);
                    }
                }
            }
            "NEOWS_TORMENT" => self.add_card(DeckCard::new("NEOWS_FURY")),
            // `LeafyPoultice`: 12 max HP, then the first basic Strike and
            // Defend transformed on the Transformations stream.
            "LEAFY_POULTICE" => {
                self.lose_max_hp(12);
                let basics = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"];
                let cards: Vec<usize> = basics.iter().filter_map(|b| self.deck.iter().position(|c| c.id == *b)).collect();
                self.transform(&cards, false, TransformRng::Transformations);
            }
            "NEW_LEAF" => offered.push(self.pick(DeckAction::Transform { upgrade: false }, 1, 1)),
            // `LostCoffer`: a card reward of the character's with a
            // combat's odds but no source, then a potion reward.
            "LOST_COFFER" => {
                let options = CardOptions { encounter: false, ..CardOptions::for_room(RoomType::Monster) };
                let cards = self.card_reward(&options);
                offered.push(Offered::Cards(cards));
                offered.push(Offered::Potions(vec![create_potion(self.rewards()).to_string()]));
            }
            // `LeadPaperweight`: two colorless cards, not a card reward, to
            // take one of.
            "LEAD_PAPERWEIGHT" => {
                let options = CardOptions {
                    cards: COLORLESS_CARDS.iter().collect(),
                    encounter: false,
                    card_reward: false,
                    ..CardOptions::for_room(RoomType::Monster)
                };
                let mut odds = self.card_odds;
                let mut cards = create_cards(2, &options, &mut odds, &mut self.roll_ctx());
                self.upgrade_by_eggs(&mut cards);
                offered.push(Offered::Cards(cards));
            }
            // `SilkenTress`: all the gold; its Glam is a card reward hook
            // (`modify_card_reward`).
            "SILKEN_TRESS" => self.lose_gold(self.gold),
            "BLOOD_SOAKED_ROSE" => self.add_card(DeckCard::new("ENTHRALLED")),
            // `SereTalon`: two different curses off the Niche stream, then
            // three Wishes.
            "SERE_TALON" => {
                let mut curses = MODIFIER_CURSES.to_vec();
                for _ in 0..2 {
                    let i = self.rngs.run(RunStream::Niche).next_int_in(0, curses.len() as i32) as usize;
                    self.add_card(DeckCard::new(curses.remove(i)));
                }
                (0..3).for_each(|_| self.add_card(DeckCard::new("WISH")));
            }
            "DISTINGUISHED_CAPE" => {
                self.lose_max_hp(9);
                (0..3).for_each(|_| self.add_card(DeckCard::new("APPARITION")));
            }
            "DOLLYS_MIRROR" => offered.push(self.pick(DeckAction::Duplicate, 1, 1)),
            "PRECISE_SCISSORS" => offered.push(self.pick(DeckAction::Remove, 1, 1)),
            "EMPTY_CAGE" => offered.push(self.pick(DeckAction::Remove, 2, 2)),
            "BIIIG_HUG" => offered.push(self.pick(DeckAction::Remove, 4, 4)),
            // `PrecariousShears`: the removal, then 16 unblockable damage.
            "PRECARIOUS_SHEARS" => {
                offered.push(self.pick(DeckAction::Remove, 2, 2));
                self.lose_hp(16);
            }
            "POMANDER" => offered.push(self.pick(DeckAction::Upgrade, 1, 1)),
            "YUMMY_COOKIE" => offered.push(self.pick(DeckAction::Upgrade, 4, 4)),
            "PUNCH_DAGGER" => offered.push(self.pick(DeckAction::Enchant("MOMENTUM", 5), 1, 1)),
            "ELECTRIC_SHRYMP" => offered.push(self.pick(DeckAction::Enchant("IMBUED", 1), 1, 1)),
            "GNARLED_HAMMER" => offered.push(self.pick(DeckAction::Enchant("SHARP", 3), 0, 3)),
            "KIFUDA" => offered.push(self.pick(DeckAction::Enchant("ADROIT", 3), 0, 3)),
            // `RoyalStamp`: the enchantable cards shuffled on the Niche
            // stream, which the screen then lays out in deck order.
            "ROYAL_STAMP" => {
                let action = DeckAction::Enchant("ROYALLY_APPROVED", 1);
                let mut cards = self.pickable(action);
                self.rngs.run(RunStream::Niche).shuffle(&mut cards);
                offered.push(self.pick(action, 1, 1));
            }
            // `HeftyTablet`, `ArcaneScroll`: rare cards, uniform odds, no
            // upgrade roll. The tablet adds an Injury whatever is taken.
            "HEFTY_TABLET" => {
                let mut odds = self.card_odds;
                let cards = create_cards(3, &CardOptions::uniform(Rarity::Rare), &mut odds, &mut self.roll_ctx());
                offered.push(Offered::Cards(cards));
                self.add_card(DeckCard::new("INJURY"));
            }
            "ARCANE_SCROLL" => {
                let mut odds = self.card_odds;
                let cards = create_cards(1, &CardOptions::uniform(Rarity::Rare), &mut odds, &mut self.roll_ctx());
                for &card in &cards {
                    self.add_card(DeckCard::from(card));
                }
                offered.push(Offered::Gained(cards));
            }
            // `LargeCapsule`: two relics off the front of the bag, each
            // obtained (and picked up) before the next is pulled, then a
            // Strike and a Defend.
            "LARGE_CAPSULE" => {
                for _ in 0..2 {
                    let relic = self.relic_reward().game_id();
                    offered.extend(self.take(relic));
                }
                self.add_card(DeckCard::new("STRIKE_IRONCLAD"));
                self.add_card(DeckCard::new("DEFEND_IRONCLAD"));
            }
            // `SmallCapsule`: a relic reward.
            "SMALL_CAPSULE" => {
                let relic = self.relic_reward().game_id();
                offered.push(Offered::Relics(vec![relic]));
            }
            // `NeowsBones`: two of Neow's other relics, shuffled on the
            // Rewards stream, as rewards, then a curse off the Niche stream.
            // The game draws the curse once the rewards are taken; none of
            // Neow's relics the port picks up draws on Niche, so drawing it
            // here comes to the same.
            "NEOWS_BONES" => {
                let mut relics: Vec<&str> = NEOW_RELICS.to_vec();
                self.rngs.player(PlayerStream::Rewards).shuffle(&mut relics);
                offered.push(Offered::Relics(relics[..2].iter().map(|r| r.to_string()).collect()));
                let curse = *self.rngs.run(RunStream::Niche).pick(MODIFIER_CURSES).expect("a curse");
                self.add_card(DeckCard::new(curse));
            }
            "SCROLL_BOXES" => {
                let bundles = self.scroll_boxes();
                offered.push(Offered::Bundles(bundles));
            }
            // The card it readies is added by the ancient that laid it out
            // (`RunState::take_ancient`).
            "DUSTY_TOME" => {}
            // Charged on creation (`RunRelic::new`), which is `Rekindle`.
            "PUMPKIN_CANDLE" => {}
            // Only in combat.
            "BELT_BUCKLE" | "SNECKO_EYE" | "FAKE_SNECKO_EYE" | "PAELS_EYE" => {}
            _ if PICKUPS.contains(&id) => offered.push(Offered::Unported(id.to_string())),
            _ => {}
        }
        offered
    }

    /// A relic handed over without a choice: obtained, with what its pickup
    /// offered after it.
    pub fn take(&mut self, relic: String) -> Vec<Offered> {
        let pickup = self.obtain(&relic);
        [Offered::Took(vec![relic])].into_iter().chain(pickup).collect()
    }

    /// `ScrollBoxes.GenerateRandomBundles` for the Ironclad: two bundles of
    /// two commons and an uncommon, no card in both.
    fn scroll_boxes(&mut self) -> Vec<Vec<Offer>> {
        let of = |rarity: Rarity| -> Vec<&'static PoolCard> {
            IRONCLAD_CARDS.iter().filter(|c| c.rarity == rarity && !c.multiplayer_only).collect()
        };
        let (commons, uncommons) = (of(Rarity::Common), of(Rarity::Uncommon));
        let rng = self.rngs.player(PlayerStream::Rewards);
        let mut used: Vec<&str> = Vec::new();
        let mut bundles = Vec::new();
        for _ in 0..2 {
            let mut bundle = Vec::new();
            let mut left: Vec<&PoolCard> = commons.iter().copied().filter(|c| !used.contains(&c.id)).collect();
            for _ in 0..2 {
                let i = rng.next_int_in(0, left.len() as i32) as usize;
                let card = left.remove(i);
                used.push(card.id);
                bundle.push(Offer::new(card.id));
            }
            let items: Vec<&PoolCard> = uncommons.iter().copied().filter(|c| !used.contains(&c.id)).collect();
            let card = *rng.pick(&items).expect("an uncommon");
            used.push(card.id);
            bundle.push(Offer::new(card.id));
            bundles.push(bundle);
        }
        bundles
    }

    /// What entering a room does before anything in it: an ancient's heal
    /// (`AncientEventModel.BeforeEventStarted`: Neow first empties HP, then
    /// the missing HP is healed, four fifths of it at Weary Traveler), then
    /// the relics' `AfterRoomEntered`, in the order they came.
    pub fn room_entered(&mut self, room: Room, unknown_point: bool) {
        if let Room::Ancient(name) = room {
            if name == "Neow" {
                self.hp = 0;
            }
            let missing = self.max_hp - self.hp;
            let amount = if self.ascension.has(AscensionLevel::WearyTraveler) { missing * 8 / 10 } else { missing };
            self.heal(amount);
        }
        for i in 0..self.relics.len() {
            let deck = self.deck.len() as i32;
            match (self.relics[i].id.as_str(), room) {
                ("MEAL_TICKET", Room::Shop) => self.heal(15),
                // Three HP for every five cards in the deck.
                ("ETERNAL_FEATHER", Room::RestSite) => self.heal(3 * (deck / 5)),
                ("PLANISPHERE", _) if unknown_point => self.heal(5),
                // Twelve gold a room until something is bought (the flag).
                ("MAW_BANK", _) if !self.relics[i].flag => self.gain_gold(12),
                _ => {}
            }
        }
    }

    /// `RestSiteOption.Generate`: Heal, Smith while a card can be upgraded,
    /// then what the relics add (`Hook.ModifyRestSiteOptions`), in the
    /// order they came.
    pub fn rest_options(&self) -> Vec<RestOption> {
        let mut options = vec![RestOption::Heal];
        if self.deck.iter().any(DeckCard::upgradable) {
            options.push(RestOption::Smith);
        }
        for relic in &self.relics {
            match relic.id.as_str() {
                "GIRYA" if relic.counter < 3 => options.push(RestOption::Lift),
                "SHOVEL" => options.push(RestOption::Dig),
                "PUMPKIN_CANDLE" => options.push(RestOption::Kindle),
                "MEAT_CLEAVER" => options.push(RestOption::Cook),
                _ => {}
            }
        }
        options
    }

    /// A rest site option's `OnSelect`.
    pub fn rest(&mut self, option: RestOption) -> Vec<Offered> {
        match option {
            RestOption::Heal => self.rest_heal(),
            RestOption::Smith => vec![self.pick(DeckAction::Upgrade, 1, 1)],
            RestOption::Lift => {
                self.relic_mut("GIRYA").expect("Girya").counter += 1;
                Vec::new()
            }
            // `DigRestSiteOption`: a relic off the front of the bag.
            RestOption::Dig => {
                let relic = self.relic_reward().game_id();
                self.take(relic)
            }
            // `PumpkinCandle.Rekindle`.
            RestOption::Kindle => {
                self.relic_mut("PUMPKIN_CANDLE").expect("Pumpkin Candle").counter += 5;
                Vec::new()
            }
            // `CookRestSiteOption`: two cards out, nine max HP.
            RestOption::Cook => {
                let pick = self.pick(DeckAction::Remove, 2, 2);
                self.gain_max_hp(9);
                vec![pick]
            }
        }
    }

    /// `HealRestSiteOption.ExecuteRestSiteHeal`, which Dense Vegetation
    /// mimics too: three tenths of max HP plus Regal Pillow's 15
    /// (`ModifyRestSiteHealAmount`), Stone Humidifier's max HP
    /// (`AfterRestSiteHeal`), then the rewards the relics add
    /// (`ModifyRestSiteHealRewards`): Tiny Mailbox's two potions. Dream
    /// Catcher's card reward is not ported (`UNPORTED_RELICS`).
    pub fn rest_heal(&mut self) -> Vec<Offered> {
        let pillow = if self.has_relic("REGAL_PILLOW") { 15 } else { 0 };
        self.heal(self.max_hp * 3 / 10 + pillow);
        if self.has_relic("STONE_HUMIDIFIER") {
            self.gain_max_hp(5);
        }
        let mut offered = Vec::new();
        if self.has_relic("TINY_MAILBOX") {
            offered.push(Offered::Potions((0..2).map(|_| create_potion(self.rewards()).to_string()).collect()));
        }
        if self.has_relic("DREAM_CATCHER") {
            offered.push(Offered::Unported("DREAM_CATCHER".into()));
        }
        offered
    }
}
