//! What a run offers after a fight, value for value with the game, and the
//! odds and factories every reward draws through: `Rewards/RewardsSet.cs`
//! (`GenerateRewardsFor`, then `GenerateWithoutOffering`),
//! `Odds/CardRarityOdds.cs`, `Odds/PotionRewardOdds.cs`,
//! `Factories/CardFactory.cs`, `Factories/PotionFactory.cs` and
//! `Factories/RelicFactory.cs`. All of it draws on the player's Rewards
//! stream, so one draw too many or too few shifts everything after it.
//!
//! `tools/oracle rewards` runs the game's own `RewardsSet` for a seed and a
//! list of rooms; the tests pin this port to it.
//!
//! Draw order, which the game spreads over two passes: `GenerateRewardsFor`
//! rolls the potion drop while it lists the rewards, then
//! `GenerateWithoutOffering` populates them in list order (gold, potion,
//! cards, relic), then relic hooks add rewards (`Hook.ModifyRewards`), which
//! populate after. The screen sorts them by `RewardsSetIndex` afterwards,
//! which changes nothing drawn.

use crate::encounter::Act;
use crate::game_rng::{GameRng, PlayerStream};
use crate::plan::{BagRelic, RelicBag, Unlocks};
use crate::pools::{PoolCard, PoolPotion, PotionRarity, Rarity, COLORLESS_CARDS, IRONCLAD_CARDS, IRONCLAD_POTIONS, SHARED_POTIONS};
use crate::run::{RoomType, RunState};
use crate::types::AscensionLevel::{Poverty, Scarcity};
use crate::types::{Ascension, CardType, RelicRarity};

/// `Runs/CardRarityOddsType.cs`, less `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OddsType {
    Regular,
    Elite,
    Boss,
    Shop,
    Uniform,
}

impl OddsType {
    /// The odds a combat room's card reward rolls with
    /// (`CardCreationOptions.ForRoom`).
    fn for_room(room: RoomType) -> Self {
        match room {
            RoomType::Elite => OddsType::Elite,
            RoomType::Boss => OddsType::Boss,
            RoomType::Shop => OddsType::Shop,
            _ => OddsType::Regular,
        }
    }

    /// `GetBaseOdds` for Rare and Uncommon. Common's never enters a roll,
    /// which matters: `regularCommonOdds` is a static field, so it keeps the
    /// ascension of the first run the process touched it in.
    fn rare_and_uncommon(self, ascension: Ascension) -> (f32, f32) {
        let scarce = |ascended: f32, base: f32| ascension.pick(Scarcity, ascended, base);
        match self {
            OddsType::Regular => (scarce(0.0149, 0.03), 0.37),
            OddsType::Elite => (scarce(0.05, 0.1), 0.4),
            OddsType::Boss => (1.0, 0.0),
            OddsType::Shop => (scarce(0.045, 0.09), 0.37),
            OddsType::Uniform => (0.33, 0.33),
        }
    }
}

/// `CardRarityOdds.CurrentValue`: the offset rare odds carry, which grows
/// with each card rolled short of rare and resets on a rare.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardOdds(pub f32);

impl Default for CardOdds {
    fn default() -> Self {
        CardOdds(CardOdds::BASE)
    }
}

impl CardOdds {
    /// `_baseRarityOffset`.
    const BASE: f32 = -0.05;

    /// `Roll`: a reward card's rarity, which moves the offset. Boss rewards
    /// roll with none.
    pub fn roll(&mut self, odds: OddsType, run: &mut RollCtx) -> Rarity {
        let offset = if odds == OddsType::Boss { 0.0 } else { self.0 };
        let rarity = Self::roll_with_offset(odds, offset, run);
        self.0 = if rarity == Rarity::Rare {
            Self::BASE
        } else {
            (self.0 + run.ascension.pick(Scarcity, 0.005, 0.01)).min(0.4)
        };
        rarity
    }

    /// `RollWithoutChangingFutureOdds`: the rare threshold is the base plus
    /// the offset, the uncommon one stacks on top of it.
    pub fn roll_with_offset(odds: OddsType, offset: f32, run: &mut RollCtx) -> Rarity {
        let roll = run.rng.next_float(1.0);
        let (rare, uncommon) = odds.rare_and_uncommon(run.ascension);
        let rare = rare + offset;
        if roll < rare {
            Rarity::Rare
        } else if roll < uncommon + rare {
            Rarity::Uncommon
        } else {
            Rarity::Common
        }
    }

    /// `RollWithBaseOdds`: no offset, and the thresholds do not stack.
    fn roll_with_base_odds(odds: OddsType, run: &mut RollCtx) -> Rarity {
        let roll = run.rng.next_float(1.0);
        let (rare, uncommon) = odds.rare_and_uncommon(run.ascension);
        if roll < rare {
            Rarity::Rare
        } else if roll < uncommon {
            Rarity::Uncommon
        } else {
            Rarity::Common
        }
    }
}

/// `PotionRewardOdds.CurrentValue`: the chance a fight drops a potion,
/// down a tenth after each drop and up a tenth after each miss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PotionOdds(pub f32);

impl Default for PotionOdds {
    fn default() -> Self {
        PotionOdds(0.4)
    }
}

impl PotionOdds {
    /// `Roll`. Elites add half their bonus; a forced drop (White Beast
    /// Statue) still draws.
    fn roll(&mut self, room: RoomType, forced: bool, rng: &mut GameRng) -> bool {
        let bonus = if room == RoomType::Elite { 0.25f32 } else { 0.0 };
        let chance = self.0 + bonus * 0.5;
        let roll = rng.next_float(1.0);
        if forced || roll < chance {
            self.0 -= 0.1;
            true
        } else {
            self.0 += 0.1;
            false
        }
    }
}

/// What a roll reads: the Rewards stream, the ascension and the act.
pub struct RollCtx<'a> {
    pub rng: &'a mut GameRng,
    pub ascension: Ascension,
    /// `RunState.CurrentActIndex`.
    pub act: usize,
}

/// A card as a reward offers it: its game id, upgraded or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offer {
    pub id: &'static str,
    pub upgraded: bool,
}

/// `CardCreationOptions` as the factories read it.
#[derive(Clone, Debug)]
pub struct CardOptions {
    /// `GetPossibleCards`: the pools, filtered, in order.
    pub cards: Vec<&'static PoolCard>,
    pub odds: OddsType,
    /// `CardCreationSource.Encounter`: a combat reward, whose rarity roll
    /// moves the odds.
    pub encounter: bool,
    /// Not `CardCreationFlags.NoUpgradeRoll`.
    pub upgrade_roll: bool,
    /// `CardCreationFlags.IsCardReward`, which the late hooks look for.
    pub card_reward: bool,
}

impl CardOptions {
    /// `CardCreationOptions.ForRoom`: the character's pool.
    pub fn for_room(room: RoomType) -> Self {
        CardOptions {
            cards: IRONCLAD_CARDS.iter().collect(),
            odds: OddsType::for_room(room),
            encounter: matches!(room, RoomType::Monster | RoomType::Elite | RoomType::Boss),
            upgrade_roll: true,
            card_reward: true,
        }
    }

    /// Ironclad's pool cut to one rarity, with uniform odds and no upgrade
    /// roll, as Hefty Tablet and Arcane Scroll ask for them.
    pub fn uniform(rarity: Rarity) -> Self {
        CardOptions {
            cards: IRONCLAD_CARDS.iter().filter(|c| c.rarity == rarity).collect(),
            odds: OddsType::Uniform,
            encounter: false,
            upgrade_roll: false,
            card_reward: false,
        }
    }
}

/// `CardFactory.CreateForReward`: `count` distinct cards, each followed by
/// its upgrade roll, before any hook sees them.
pub fn create_cards(count: usize, options: &CardOptions, odds: &mut CardOdds, run: &mut RollCtx) -> Vec<Offer> {
    let mut offered: Vec<Offer> = Vec::new();
    for _ in 0..count {
        // `FilterForPlayerCount`: alone, never the multiplayer-only cards.
        let cards: Vec<&PoolCard> = options
            .cards
            .iter()
            .copied()
            .filter(|c| !c.multiplayer_only && !offered.iter().any(|o| o.id == c.id))
            .collect();
        let items: Vec<&PoolCard> = if options.odds == OddsType::Uniform {
            cards.into_iter().filter(|c| !matches!(c.rarity, Rarity::Basic | Rarity::Ancient)).collect()
        } else {
            let rarity = if options.encounter && matches!(options.odds, OddsType::Regular | OddsType::Elite | OddsType::Boss) {
                odds.roll(options.odds, run)
            } else {
                CardOdds::roll_with_base_odds(options.odds, run)
            };
            let rarity = next_allowed(rarity, |r| cards.iter().any(|c| c.rarity == r)).expect("a rarity the pool holds");
            cards.into_iter().filter(|c| c.rarity == rarity).collect()
        };
        let card = *run.rng.pick(&items).expect("a card to offer");
        let mut offer = Offer { id: card.id, upgraded: false };
        if options.upgrade_roll {
            offer.upgraded = roll_for_upgrade(card, 0.0, run);
        }
        offered.push(offer);
    }
    offered
}

/// `CardFactory.GetNextAllowedRarity`: Common, Uncommon, Rare, round
/// again, from `rarity` to the first the pool holds.
pub(crate) fn next_allowed(start: Rarity, allowed: impl Fn(Rarity) -> bool) -> Option<Rarity> {
    let mut rarity = start;
    while !allowed(rarity) {
        rarity = match rarity {
            Rarity::Basic | Rarity::Rare => Rarity::Common,
            Rarity::Common => Rarity::Uncommon,
            Rarity::Uncommon => Rarity::Rare,
            Rarity::Ancient => return None,
        };
        if rarity == start {
            return None;
        }
    }
    Some(rarity)
}

/// `CardFactory.RollForUpgrade`: always one draw; a card short of rare
/// upgrades when it lands within a quarter per act (an eighth at
/// Scarcity). The game compares in `decimal`, converted from the `float`
/// roll, which keeps seven significant digits.
fn roll_for_upgrade(card: &PoolCard, base: f64, run: &mut RollCtx) -> bool {
    let roll: f64 = format!("{:.6e}", run.rng.next_float(1.0)).parse().expect("a float");
    let mut chance = base;
    if card.rarity != Rarity::Rare {
        chance += run.act as f64 * run.ascension.pick(Scarcity, 0.125, 0.25);
    }
    roll <= chance
}

/// `PotionFactory.CreateRandomPotionOutOfCombat`: a rarity, then one of
/// the character's and the shared potions of it.
pub fn create_potion(rng: &mut GameRng) -> &'static str {
    create_potions(1, rng)[0]
}

/// `PotionFactory.CreateRandomPotionsOutOfCombat`: `count` potions, none
/// twice.
pub fn create_potions(count: usize, rng: &mut GameRng) -> Vec<&'static str> {
    let mut options: Vec<&PoolPotion> = IRONCLAD_POTIONS.iter().chain(SHARED_POTIONS).collect();
    let mut potions = Vec::new();
    for _ in 0..count {
        let roll = rng.next_float(1.0);
        let rarity = if roll <= 0.1 {
            PotionRarity::Rare
        } else if roll <= 0.35 {
            PotionRarity::Uncommon
        } else {
            PotionRarity::Common
        };
        let of_rarity: Vec<&PoolPotion> = options.iter().copied().filter(|p| p.rarity == rarity).collect();
        let potion = *rng.pick(&of_rarity).expect("a potion of the rarity");
        options.retain(|p| p.id != potion.id);
        potions.push(potion.id);
    }
    potions
}

/// `RelicFactory.RollRarity`.
pub fn roll_relic_rarity(rng: &mut GameRng) -> RelicRarity {
    let roll = rng.next_float(1.0);
    if roll < 0.5 {
        RelicRarity::Common
    } else if roll < 0.83 {
        RelicRarity::Uncommon
    } else {
        RelicRarity::Rare
    }
}

/// A relic a reward pulled: from a bag, or `RelicFactory.FallbackRelic`
/// (Circlet) when every deque is dry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pulled {
    Bag(BagRelic),
    Circlet,
}

impl Pulled {
    pub fn game_id(self) -> String {
        match self {
            Pulled::Bag(r) => r.game_id(),
            Pulled::Circlet => "CIRCLET".into(),
        }
    }
}

/// `RelicFactory.PullNextRelicFromFront`: the player's bag, then the
/// relic leaves the shared bag too.
pub fn pull_relic(player_bag: &mut RelicBag, shared_bag: &mut RelicBag, rarity: RelicRarity, floor: usize) -> Pulled {
    match player_bag.pull_front(rarity, floor, |_| true) {
        Some(relic) => {
            shared_bag.remove(relic);
            Pulled::Bag(relic)
        }
        None => Pulled::Circlet,
    }
}

/// `RelicFactory.PullNextRelicFromBack` for a shop: the back of the
/// player's bag, skipping what shops may not sell.
pub fn pull_relic_from_back(player_bag: &mut RelicBag, shared_bag: &mut RelicBag, rarity: RelicRarity, floor: usize) -> Pulled {
    match player_bag.pull_back(rarity, floor, BagRelic::allowed_in_shops) {
        Some(relic) => {
            shared_bag.remove(relic);
            Pulled::Bag(relic)
        }
        None => Pulled::Circlet,
    }
}

/// A room's rewards as the screen lists them. `cards` holds one entry per
/// card reward (Prayer Wheel adds a second).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rewards {
    /// The rolled gold, then any a relic adds (Amethyst Aubergine).
    pub gold: Vec<i32>,
    pub potion: Option<&'static str>,
    pub relics: Vec<Pulled>,
    pub cards: Vec<Vec<Offer>>,
}

impl Rewards {
    /// As `tools/oracle rewards` prints a step, after the step's name.
    pub fn oracle_text(&self) -> String {
        let mut parts = Vec::new();
        for gold in &self.gold {
            parts.push(format!("gold {gold}"));
        }
        if let Some(potion) = self.potion {
            parts.push(format!("potion {potion}"));
        }
        for relic in &self.relics {
            parts.push(format!("relic {}", relic.game_id()));
        }
        for cards in &self.cards {
            let ids: Vec<String> = cards.iter().map(|o| format!("{}{}", o.id, if o.upgraded { "+" } else { "" })).collect();
            parts.push(format!("cards {}", ids.join(" ")));
        }
        parts.join(" ")
    }
}

/// `EncounterModel.MinGoldReward` and `MaxGoldReward` for a room, three
/// quarters of it at Poverty, truncated.
fn gold_range(room: RoomType, ascension: Ascension) -> (i32, i32) {
    let (min, max) = match room {
        RoomType::Monster => (10.0, 20.0),
        RoomType::Elite => (35.0, 45.0),
        _ => (100.0, 100.0),
    };
    let scale = if ascension.has(Poverty) { 0.75 } else { 1.0 };
    ((min * scale) as i32, (max * scale) as i32)
}

impl RunState {
    /// The player's Rewards stream.
    pub fn rewards(&mut self) -> &mut GameRng {
        self.rngs.player(PlayerStream::Rewards)
    }

    /// A roll's view of the run.
    pub fn roll_ctx(&mut self) -> RollCtx<'_> {
        RollCtx { rng: self.rngs.player(PlayerStream::Rewards), ascension: self.ascension, act: self.act }
    }

    /// `RewardsSet.WithRewardsFromRoom` and `GenerateWithoutOffering` for a
    /// fight the player won. `gold_proportion` is `CombatRoom.GoldProportion`,
    /// the share of monsters that did not escape (1 unless one fled); a
    /// monster room scales its gold range by it, and at 0 gives no gold and
    /// draws none. The final act's boss gives nothing.
    pub fn combat_rewards(&mut self, room: RoomType, gold_proportion: f32) -> Rewards {
        // Lasting Candy's `CombatsSeen` counts this fight.
        if let Some(candy) = self.relic_mut("LASTING_CANDY") {
            candy.counter += 1;
        }
        let mut rewards = Rewards::default();
        if room == RoomType::Boss && self.act + 1 >= self.plan.acts.len() {
            return rewards;
        }
        // `GenerateRewardsFor`.
        let forced = self.has_relic("WHITE_BEAST_STATUE");
        let potion = self.potion_odds.roll(room, forced, self.rngs.player(PlayerStream::Rewards));

        // `GenerateWithoutOffering`: `Populate` in list order.
        let (min, max) = gold_range(room, self.ascension);
        let scale = |g: i32| (g as f32 * gold_proportion).round_ties_even() as i32;
        let range = match room {
            RoomType::Monster if gold_proportion <= 0.0 => None,
            RoomType::Monster => Some((scale(min), scale(max))),
            _ => Some((min, max)),
        };
        if let Some((min, max)) = range {
            rewards.gold.push(self.rngs.player(PlayerStream::Rewards).next_int_in(min, max + 1));
        }
        if potion {
            rewards.potion = Some(create_potion(self.rngs.player(PlayerStream::Rewards)));
        }
        let cards = self.card_reward(&CardOptions::for_room(room));
        rewards.cards.push(cards);
        if room == RoomType::Elite {
            rewards.relics.push(self.relic_reward());
        }

        // `Hook.ModifyRewards`, over the relics in the order they came.
        let held: Vec<String> = self.relics.iter().map(|r| r.id.clone()).collect();
        for relic in held {
            match relic.as_str() {
                "PRAYER_WHEEL" if room == RoomType::Monster => {
                    let cards = self.card_reward(&CardOptions::for_room(RoomType::Monster));
                    rewards.cards.push(cards);
                }
                // A fixed amount, which draws nothing.
                "AMETHYST_AUBERGINE" => rewards.gold.push(15),
                "LAVA_ROCK" if room == RoomType::Boss && self.act == 0 && !self.relic_mut("LAVA_ROCK").unwrap().flag => {
                    self.relic_mut("LAVA_ROCK").unwrap().flag = true;
                    rewards.relics.push(self.relic_reward());
                    rewards.relics.push(self.relic_reward());
                }
                _ => {}
            }
        }
        rewards
    }

    /// `RelicReward.Populate` with no rarity given: `RollRarity`, then the
    /// front of the player's bag.
    pub fn relic_reward(&mut self) -> Pulled {
        let rarity = roll_relic_rarity(self.rngs.player(PlayerStream::Rewards));
        pull_relic(&mut self.plan.player_bag, &mut self.plan.shared_bag, rarity, self.floor)
    }

    /// `CardReward.Populate`: `CreateForReward` for three, then the relic
    /// hooks on the options (`Hook.TryModifyCardRewardOptions`, then the
    /// late ones).
    pub fn card_reward(&mut self, options: &CardOptions) -> Vec<Offer> {
        let mut odds = self.card_odds;
        let mut cards = create_cards(3, options, &mut odds, &mut self.roll_ctx());
        self.card_odds = odds;
        // Lasting Candy, on every second fight it has seen: a power card
        // besides, from a pool of powers with the reward's odds but no source.
        let candy = self.relics.iter().find(|r| r.id == "LASTING_CANDY").map(|r| r.counter);
        if options.encounter && candy.is_some_and(|n| n > 0 && n % 2 == 0) {
            let not_offered = |c: &&PoolCard| c.kind == CardType::Power && !cards.iter().any(|o| o.id == c.id);
            let mut powers: Vec<&'static PoolCard> = options.cards.iter().copied().filter(not_offered).collect();
            if powers.is_empty() {
                powers = options.cards.iter().copied().filter(|c| c.kind == CardType::Power).collect();
            }
            if !powers.is_empty() {
                let candy = CardOptions { cards: powers, encounter: false, card_reward: false, ..options.clone() };
                let mut odds = self.card_odds;
                cards.extend(create_cards(1, &candy, &mut odds, &mut self.roll_ctx()));
            }
        }
        if options.card_reward {
            self.upgrade_by_relics(&mut cards);
        }
        cards
    }

    /// `TryModifyCardRewardOptionsLate`: the eggs upgrade their type,
    /// Silver Crucible its first three rewards whole. Fresnel Lens only
    /// enchants, which the offer does not track.
    fn upgrade_by_relics(&mut self, cards: &mut [Offer]) {
        self.upgrade_by_eggs(cards);
        self.upgrade_by_crucible(cards);
    }

    /// Silver Crucible on a card reward (`CardCreationFlags.IsCardReward`).
    pub(crate) fn upgrade_by_crucible(&mut self, cards: &mut [Offer]) {
        if let Some(crucible) = self.relic_mut("SILVER_CRUCIBLE").filter(|r| r.counter < 3) {
            crucible.counter += 1;
            cards.iter_mut().for_each(|o| o.upgraded = true);
        }
    }

    /// `EggRelicHelper.UpgradeValidCards` for each egg held: Toxic Egg
    /// upgrades skills, Molten Egg attacks, Frozen Egg powers.
    pub fn upgrade_by_eggs(&self, cards: &mut [Offer]) {
        let eggs = [("TOXIC_EGG", CardType::Skill), ("MOLTEN_EGG", CardType::Attack), ("FROZEN_EGG", CardType::Power)];
        for offer in cards.iter_mut() {
            let kind = IRONCLAD_CARDS.iter().chain(COLORLESS_CARDS).find(|c| c.id == offer.id).map(|c| c.kind);
            if eggs.iter().any(|&(egg, k)| kind == Some(k) && self.has_relic(egg)) {
                offer.upgraded = true;
            }
        }
    }
}

/// The relics whose pickup or hooks draw on the Rewards stream (or add
/// rewards) in ways not ported yet. Holding or taking one ends what the
/// port can follow.
pub const UNPORTED_RELICS: &[&str] = &[
    "BIG_GAME_HUNTER", "BLACK_STAR", "CALLING_BELL", "CAULDRON", "DELICATE_FROND", "DINGY_RUG", "DREAM_CATCHER",
    "DRIFTWOOD", "GLASS_EYE", "GLITTER", "KALEIDOSCOPE", "LAVA_LAMP", "LEAD_PAPERWEIGHT", "LOST_COFFER", "MASSIVE_SCROLL",
    "ORRERY", "PAELS_TOOTH", "PAELS_WING", "PRISMATIC_GEM", "SEA_GLASS", "SILKEN_TRESS", "THE_COURIER", "TOY_BOX",
    "VAKUU_CARD_SELECTOR", "WHITE_STAR", "WING_CHARM", "WONGOS_MYSTERY_TICKET",
];

impl RunState {
    /// A treasure room's chest (`OneOffSynchronizer.DoTreasureRoomRewards`,
    /// `TreasureRoomRelicSynchronizer.BeginRelicPicking`): gold on the
    /// Rewards stream, three quarters of it at Poverty, and a relic off the
    /// front of the run's shared bag on the TreasureRoomRelics stream. The
    /// relic stays in the player's bag until it is taken.
    pub fn treasure(&mut self) -> (i32, Pulled) {
        let gold = self.rngs.player(PlayerStream::Rewards).next_int_in(42, 53) as f64;
        let gold = if self.ascension.has(Poverty) { gold * 0.75 } else { gold };
        let rarity = roll_relic_rarity(self.rngs.run(crate::game_rng::RunStream::TreasureRoomRelics));
        let relic = self.plan.shared_bag.pull_front(rarity, self.floor, |_| true).map_or(Pulled::Circlet, Pulled::Bag);
        (gold as i32, relic)
    }
}

/// A `tools/oracle rewards` input line replayed on the port, printed as
/// the oracle prints it. The oracle's run always has these acts; rewards do
/// not read which.
fn port_text(header: &str) -> String {
    let parts: Vec<&str> = header.split(' ').collect();
    let ascension = Ascension(parts[1].parse().unwrap());
    let acts = [Act::Overgrowth, Act::Hive, Act::Glory];
    let mut run = RunState::new(parts[0], acts, ascension, &Unlocks::default());
    let mut out = String::new();
    for step in &parts[2..] {
        let line = match *step {
            "R" => {
                run.unknown_odds.reset();
                "R".to_string()
            }
            "?" | "?s" => {
                let room = run.roll_unknown(*step == "?s");
                format!("{step} {room:?}")
            }
            _ if step.starts_with('S') => {
                run.act = (step.as_bytes()[1] - b'0') as usize;
                let shop = run.shop().oracle_text();
                let (rewards, shops) = (run.rewards().counter, run.rngs.player(PlayerStream::Shops).counter);
                format!("{step} {shop} counter {rewards} shops {shops}")
            }
            _ => {
                run.act = (step.as_bytes()[1] - b'0') as usize;
                let room = match step.as_bytes()[0] {
                    b'M' => RoomType::Monster,
                    b'E' => RoomType::Elite,
                    _ => RoomType::Boss,
                };
                let rewards = run.combat_rewards(room, 1.0).oracle_text();
                format!("{step} {rewards} counter {}", run.rngs.player(PlayerStream::Rewards).counter).replace("  ", " ")
            }
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// Checks `tools/oracle rewards` output against the port: returns how many
/// runs it held and the header of each one that differs, with the first
/// line that does.
pub fn diff_oracle(text: &str) -> (usize, Vec<String>) {
    let mut runs = 0;
    let mut game = String::new();
    let mut header = "";
    let mut mismatches = Vec::new();
    for line in text.lines().chain(["run end"]) {
        if let Some(next) = line.strip_prefix("run ") {
            if !header.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/oracle rewards` for runs at A0, A3, A6, A7 and A10: long walks
    /// through all three acts with shops among the fights, so the potion
    /// and card odds, the relic bag and the unknown odds all move far from
    /// where they start (`examples/rewardcheck.rs` makes more).
    #[test]
    fn matches_the_game() {
        let (runs, mismatches) = diff_oracle(include_str!("../testdata/oracle-rewards.txt"));
        assert!(runs >= 20, "fixture holds {runs} runs");
        assert!(mismatches.is_empty(), "{} of {runs} differ:\n{}", mismatches.len(), mismatches.join("\n"));
    }
}
