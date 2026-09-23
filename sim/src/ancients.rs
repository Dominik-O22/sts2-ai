//! The ancients' options, value for value: what each lays out as it starts
//! (`GenerateInitialOptions` in `Models/Events/Neow.cs`, `Orobas.cs`,
//! `Pael.cs`, `Tezcatara.cs`, `Vakuu.cs`, `Darv.cs`, `Nonupeipe.cs`,
//! `Tanx.cs`), drawn on the event's own stream (`EventModel.BeginEvent`:
//! the run's seed plus the hash of the ancient's id). Every option is a
//! relic; the one taken is obtained (`AncientEventModel.RelicOption`), and
//! what it does is `RunState::obtain`'s. `tools/oracle ancients` runs the
//! game's code for a seed; the tests pin it.

use crate::game_rng::{GameRng, PlayerStream};
use crate::pools::{Rarity, IRONCLAD_CARDS};
use crate::replay::slug;
use crate::run::{DeckCard, RunState};

/// What an ancient laid out: its relic options in the order shown, and the
/// card Dusty Tome readies if it is one of them (`DustyTome.SetupForPlayer`),
/// which taking the tome adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AncientOffer {
    pub relics: Vec<String>,
    pub tome: Option<&'static str>,
}

/// Every ancient, by class name.
pub const ANCIENTS: [&str; 8] = ["Neow", "Orobas", "Pael", "Tezcatara", "Vakuu", "Darv", "Nonupeipe", "Tanx"];

/// `Neow.CurseOptions`.
const NEOW_CURSED: [&str; 8] = [
    "CURSED_PEARL", "HEFTY_TABLET", "LARGE_CAPSULE", "LEAFY_POULTICE", "NEOWS_BONES", "PRECARIOUS_SHEARS", "SILKEN_TRESS",
    "SILVER_CRUCIBLE",
];

/// `Neow.PositiveOptions`.
const NEOW_POSITIVE: [&str; 14] = [
    "ARCANE_SCROLL", "BOOMING_CONCH", "FISHING_ROD", "GOLDEN_PEARL", "KALEIDOSCOPE", "LEAD_PAPERWEIGHT", "LOST_COFFER",
    "MASSIVE_SCROLL", "NEOWS_TORMENT", "NEW_LEAF", "PHIAL_HOLSTER", "PRECISE_SCISSORS", "SCROLL_BOXES", "WINGED_BOOTS",
];

/// The positive option a curse rules out, as `Neow.GenerateInitialOptions`
/// pairs them.
const NEOW_EXCLUDES: [(&str, &str); 4] = [
    ("CURSED_PEARL", "GOLDEN_PEARL"),
    ("HEFTY_TABLET", "ARCANE_SCROLL"),
    ("LEAFY_POULTICE", "NEW_LEAF"),
    ("PRECARIOUS_SHEARS", "PRECISE_SCISSORS"),
];

/// `Darv._validRelicSets`, less the filters: one relic is drawn from each
/// set that applies.
const DARV_SETS: [&[&str]; 9] = [
    &["ASTROLABE"],
    &["BLACK_STAR"],
    &["CALLING_BELL"],
    &["EMPTY_CAGE"],
    &["PANDORAS_BOX"],
    &["RUNIC_PYRAMID"],
    &["SNECKO_EYE"],
    &["ECTOPLASM", "SOZU"],
    &["PHILOSOPHERS_STONE", "VELVET_CHOKER"],
];

/// `Player.HasEventPet`: a relic that adds a pet, or Byrdonis' egg.
fn has_event_pet(run: &RunState) -> bool {
    run.has_relic("PAELS_LEGION") || run.has_relic("BYRDPIP") || run.deck.iter().any(|c| c.id == "BYRDONIS_EGG")
}

impl RunState {
    /// The options the ancient `name` (a class name) lays out as it starts,
    /// with what laying them out draws: the event's own stream, and Darv's
    /// Dusty Tome on the Rewards stream.
    pub fn ancient_offer(&mut self, name: &str) -> AncientOffer {
        let id = slug(name);
        let mut rng = GameRng::new(self.rngs.seed.wrapping_add(crate::game_rng::hash(&id) as u32));
        let rng = &mut rng;
        let enchantable = |id: &str| self.deck.iter().filter(|c| c.can_enchant(id)).count();
        let picked = |rng: &mut GameRng, items: &[&'static str]| *rng.pick(items).expect("an option");
        let mut tome = None;
        let relics: Vec<&'static str> = match id.as_str() {
            "NEOW" => self.neow_options(rng),
            // `Orobas`: a character other than the player's for Sea Glass,
            // then Prismatic Gem a third of the time instead; Touch of Orobas
            // needs a starter relic and Archaic Tooth a card it transcends,
            // and with neither the third option is a locked one, left out.
            "OROBAS" => {
                rng.next_int(4);
                let gem = rng.next_float(1.0) < 0.333_333_3;
                let first = ["ELECTRIC_SHRYMP", "GLASS_EYE", "SAND_CASTLE", if gem { "PRISMATIC_GEM" } else { "SEA_GLASS" }];
                let second = ["ALCHEMICAL_COFFER", "DRIFTWOOD", "RADIANT_PEARL"];
                let mut third = Vec::new();
                if self.has_relic("BURNING_BLOOD") || self.has_relic("BLACK_BLOOD") {
                    third.push("TOUCH_OF_OROBAS");
                }
                if self.deck.iter().any(|c| c.id == "BASH") {
                    third.push("ARCHAIC_TOOTH");
                }
                let mut options = vec![picked(rng, &first), picked(rng, &second)];
                match third.is_empty() {
                    true => drop(rng.pick(&["OPTION_POOL_3_LOCKED"])),
                    false => options.push(picked(rng, &third)),
                }
                options
            }
            // `Pael`: the second pool doubled before Growth joins it.
            "PAEL" => {
                let first = picked(rng, &["PAELS_FLESH", "PAELS_HORN", "PAELS_TEARS"]);
                let mut second = vec!["PAELS_WING"];
                if enchantable("GOOPY") >= 3 {
                    second.push("PAELS_CLAW");
                }
                if self.deck.iter().filter(|c| c.removable()).count() >= 5 {
                    second.push("PAELS_TOOTH");
                }
                second.extend(second.clone());
                second.push("PAELS_GROWTH");
                let second = picked(rng, &second);
                let mut third = vec!["PAELS_EYE", "PAELS_BLOOD"];
                if !has_event_pet(self) {
                    third.push("PAELS_LEGION");
                }
                vec![first, second, picked(rng, &third)]
            }
            // `Tezcatara`: Nutritious Soup while a basic Strike is left.
            "TEZCATARA" => {
                let mut first = vec!["VERY_HOT_COCOA", "YUMMY_COOKIE"];
                if self.deck.iter().any(|c| c.id == "STRIKE_IRONCLAD") {
                    first.push("NUTRITIOUS_SOUP");
                }
                vec![
                    picked(rng, &first),
                    picked(rng, &["BIIIG_HUG", "STORYBOOK", "TOASTY_MITTENS"]),
                    picked(rng, &["GOLDEN_COMPASS", "PUMPKIN_CANDLE", "TOY_BOX", "SEAL_OF_GOLD"]),
                ]
            }
            // `Vakuu`: each pool shuffled, its first taken.
            "VAKUU" => {
                let mut pools: [Vec<&'static str>; 3] = [
                    vec!["BLOOD_SOAKED_ROSE", "WHISPERING_EARRING", "FIDDLE"],
                    vec!["PRESERVED_FOG", "SERE_TALON", "DISTINGUISHED_CAPE"],
                    vec!["CHOICES_PARADOX", "MUSIC_BOX", "LORDS_PARASOL", "JEWELED_MASK"],
                ];
                pools.iter_mut().for_each(|pool| rng.shuffle(pool));
                pools.iter().map(|pool| pool[0]).collect()
            }
            // `Darv`: a relic from each set the act allows, shuffled; then,
            // on a coin, two of them and Dusty Tome, else three.
            "DARV" => {
                let sets = DARV_SETS.iter().enumerate().filter(|&(i, _)| match i {
                    7 => self.act == 1,
                    8 => self.act == 2,
                    _ => true,
                });
                let mut options: Vec<&'static str> = sets.map(|(_, set)| picked(rng, set)).collect();
                rng.shuffle(&mut options);
                if rng.next_bool() {
                    options.truncate(2);
                    options.push("DUSTY_TOME");
                    tome = Some(self.tome_card());
                } else {
                    options.truncate(3);
                }
                options
            }
            // `Nonupeipe`: Beautiful Bracelet joins with four cards Swift
            // can enchant.
            "NONUPEIPE" => {
                let mut pool = vec![
                    "BLESSED_ANTLER", "BRILLIANT_SCARF", "DELICATE_FROND", "DIAMOND_DIADEM", "FUR_COAT", "GLITTER", "JEWELRY_BOX",
                    "LOOMING_FRUIT", "SIGNET_RING",
                ];
                if enchantable("SWIFT") >= 4 {
                    pool.push("BEAUTIFUL_BRACELET");
                }
                rng.shuffle(&mut pool);
                pool.truncate(3);
                pool
            }
            // `Tanx`: Tri-Boomerang joins with three cards Instinct can
            // enchant.
            "TANX" => {
                let mut pool = vec![
                    "CLAWS", "CROSSBOW", "IRON_CLUB", "MEAT_CLEAVER", "SAI", "SPIKED_GAUNTLETS", "TANXS_WHISTLE", "THROWING_AXE",
                    "WAR_HAMMER",
                ];
                if enchantable("INSTINCT") >= 3 {
                    pool.push("TRI_BOOMERANG");
                }
                rng.shuffle(&mut pool);
                pool.truncate(3);
                pool
            }
            _ => panic!("unknown ancient {name}"),
        };
        AncientOffer { relics: relics.into_iter().map(str::to_string).collect(), tome }
    }

    /// `Neow.GenerateInitialOptions` without modifiers: a cursed option,
    /// the positive ones it leaves, three coins between pairs, then two of
    /// the positive ones shuffled and the cursed one last. Every option is
    /// `IsAllowedAtNeow` alone but Massive Scroll, kept to multiplayer.
    fn neow_options(&self, rng: &mut GameRng) -> Vec<&'static str> {
        let allowed = |relic: &&str| *relic != "MASSIVE_SCROLL";
        let cursed: Vec<&str> = NEOW_CURSED.into_iter().filter(allowed).collect();
        let curse = *rng.pick(&cursed).expect("a cursed option");
        let excluded = NEOW_EXCLUDES.iter().find(|(c, _)| *c == curse).map(|&(_, p)| p);
        let mut positive: Vec<&str> = NEOW_POSITIVE.into_iter().filter(|&p| Some(p) != excluded).collect();
        if curse != "LARGE_CAPSULE" {
            positive.push(if rng.next_bool() { "LAVA_ROCK" } else { "SMALL_CAPSULE" });
        }
        positive.push(if rng.next_bool() { "NUTRITIOUS_OYSTER" } else { "STONE_HUMIDIFIER" });
        positive.push(if rng.next_bool() { "NEOWS_TALISMAN" } else { "POMANDER" });
        positive.retain(allowed);
        rng.shuffle(&mut positive);
        positive.truncate(2);
        positive.push(curse);
        positive
    }

    /// `DustyTome.SetupForPlayer`: an ancient card of the character's on the
    /// Rewards stream, less the transcendence ones (Break, for the Ironclad).
    fn tome_card(&mut self) -> &'static str {
        let cards: Vec<&str> = IRONCLAD_CARDS.iter().filter(|c| c.rarity == Rarity::Ancient && c.id != "BREAK").map(|c| c.id).collect();
        *self.rngs.player(PlayerStream::Rewards).pick(&cards).expect("an ancient card")
    }

    /// Takes an ancient's relic: obtained, and Dusty Tome's card added
    /// upgraded (`DustyTome.AfterObtained`).
    pub fn take_ancient(&mut self, offer: &AncientOffer, relic: &str) -> Vec<crate::effects::Offered> {
        let pickup = self.obtain(relic);
        if let ("DUSTY_TOME", Some(card)) = (relic, offer.tome) {
            self.add_card(DeckCard { upgraded: true, ..DeckCard::new(card) });
        }
        pickup
    }
}

/// A `tools/oracle ancients` input line replayed on the port, printed as the
/// oracle prints it. The run's acts do not matter to the options.
fn port_text(header: &str) -> String {
    let parts: Vec<&str> = header.split_whitespace().collect();
    let acts = [crate::encounter::Act::Overgrowth, crate::encounter::Act::Hive, crate::encounter::Act::Glory];
    let ascension = crate::types::Ascension(parts[1].parse().unwrap());
    let mut run = RunState::new(parts[0], acts, ascension, &crate::plan::Unlocks::default());
    run.act = parts[2].parse().unwrap();
    for op in &parts[4..] {
        let (sign, id) = op.split_at(1);
        match sign {
            "+" => run.deck.push(DeckCard::new(id)),
            _ => {
                let i = run.deck.iter().position(|c| c.id == id).expect("a card to remove");
                run.deck.remove(i);
            }
        }
    }
    let name = ANCIENTS.iter().find(|a| slug(a) == parts[3]).expect("an ancient");
    let offer = run.ancient_offer(name);
    let options: Vec<String> = offer
        .relics
        .iter()
        .map(|r| match (r.as_str(), offer.tome) {
            ("DUSTY_TOME", Some(card)) => format!("DUSTY_TOME:{card}"),
            _ => r.clone(),
        })
        .collect();
    format!("{} {} counter {}\n", parts[3], options.join(" "), run.rewards().counter)
}

/// Checks `tools/oracle ancients` output against the port: returns how many
/// lines it held and each one that differs.
pub fn diff_oracle(text: &str) -> (usize, Vec<String>) {
    let mut runs = 0;
    let mut mismatches = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.strip_prefix("run ").expect("a run header");
        let game = format!("{}\n", lines.next().expect("the options"));
        runs += 1;
        let port = port_text(header);
        if port != game {
            mismatches.push(format!("{header}: port {}, game {}", port.trim(), game.trim()));
        }
    }
    (runs, mismatches)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/oracle ancients` for every ancient over many seeds, acts and
    /// decks that turn each condition on and off (`examples/ancientcheck.rs`
    /// makes more).
    #[test]
    fn matches_the_game() {
        let (runs, mismatches) = diff_oracle(include_str!("../testdata/oracle-ancients.txt"));
        assert!(runs >= 100, "fixture holds {runs} lines");
        assert!(mismatches.is_empty(), "{} of {runs} differ:\n{}", mismatches.len(), mismatches.join("\n"));
    }
}
