//! Fight setups for training. A generator rolls plausible run states keyed
//! on the floor the fight happens at, across all three acts (DESIGN.md, Training: "wide, not
//! clever"), and `FightSetup::from_recording` reads the recorder's `start`
//! record so real fights form the held-out set.

use serde_json::Value;

use crate::card::{def, Card, COLORLESS_POOL, IRONCLAD_POOL, UNSUPPORTED_CARDS};
use crate::enchant::{self, Enchantment};
use crate::combat::{Combat, EnemySpec, RoomKind, Setup};
use crate::encounter::{Encounter, Kind};
use crate::ids::{CardId, MonsterId};
use crate::potion::{self, PotionId};
use crate::relic::{self, Relic, RelicId};
use crate::replay::Ids;
use crate::rng::Rng;
use crate::types::{Ascension, AscensionLevel, CardRarity, CardType};
use crate::{ironclad_starter_deck, IRONCLAD_ENERGY, IRONCLAD_HP};

/// Relics the sim leaves out because nothing they do reaches a combat: they
/// act on pickup, on rewards, the map, rest sites and shops, or on the deck
/// once a fight is over. Byrdpip and Pael's Legion summon pets, but the pets
/// only stand there, and Pael's Legion's block is ported. Anything the
/// recorder names that is neither here nor ported is an error, not a silent
/// drop, because a missing relic replays clean while being wrong.
const INERT_RELICS: &[&str] = &[
    "ALCHEMICAL_COFFER", "ARCANE_SCROLL", "ARCHAIC_TOOTH", "ASTROLABE", "BEAUTIFUL_BRACELET", "BING_BONG",
    "BLACK_STAR", "BYRDPIP", "CALLING_BELL", "CHOSEN_CHEESE", "CLAWS", "CURSED_PEARL", "DARKSTONE_PERIAPT",
    "DISTINGUISHED_CAPE", "DREAM_CATCHER", "DRIFTWOOD", "DUSTY_TOME", "ELECTRIC_SHRYMP", "EMPTY_CAGE",
    "FAKE_LEES_WAFFLE", "FAKE_MANGO", "FAKE_MERCHANTS_RUG", "FISHING_ROD", "FRAGRANT_MUSHROOM", "FRESNEL_LENS",
    "GLASS_EYE", "GLITTER", "GOLDEN_COMPASS", "GOLDEN_PEARL", "HEFTY_TABLET", "JEWELRY_BOX", "KALEIDOSCOPE",
    "LARGE_CAPSULE", "LAVA_ROCK", "LEAFY_POULTICE", "LOOMING_FRUIT", "LORDS_PARASOL", "LOST_COFFER",
    "MASSIVE_SCROLL", "MAW_BANK", "MEAT_CLEAVER", "NEOWS_BONES", "NEOWS_TALISMAN", "NEOWS_TORMENT", "NEW_LEAF",
    "NUTRITIOUS_OYSTER", "NUTRITIOUS_SOUP", "PAELS_CLAW", "PAELS_GROWTH", "PAELS_HORN", "PAELS_TOOTH",
    "PAELS_WING", "PANDORAS_BOX", "PAPER_KRANE", "PHIAL_HOLSTER", "POMANDER", "PRECARIOUS_SHEARS",
    "PRECISE_SCISSORS", "PRESERVED_FOG", "SAND_CASTLE", "SCROLL_BOXES", "SEA_GLASS", "SERE_TALON",
    "SIGNET_RING", "SILKEN_TRESS", "SILVER_CRUCIBLE", "SMALL_CAPSULE", "STONE_HUMIDIFIER", "STORYBOOK",
    "SWORD_OF_STONE", "TANXS_WHISTLE", "TOUCH_OF_OROBAS", "TOY_BOX", "TRI_BOOMERANG", "VAKUU_CARD_SELECTOR",
    "WAR_HAMMER", "WINGED_BOOTS", "WONGO_CUSTOMER_APPRECIATION_BADGE", "WONGOS_MYSTERY_TICKET", "YUMMY_COOKIE",
];

/// Floors per act, the last one the boss room. Real acts run 13 to 15
/// rooms plus the Ancient and the boss; one length for all keeps a floor's
/// act a division.
pub const BOSS_FLOOR: u32 = 16;
/// Acts the generator rolls fights for.
pub const ACTS: u32 = 3;
/// The last boss room of the run.
pub const LAST_FLOOR: u32 = BOSS_FLOOR * ACTS;

/// Which act (0-based) a run floor is in, and the floor within that act.
pub fn act_floor(floor: u32) -> (u32, u32) {
    ((floor - 1) / BOSS_FLOOR, (floor - 1) % BOSS_FLOOR + 1)
}

/// An owned `Setup`: everything a combat needs from the run.
#[derive(Clone, Debug)]
pub struct FightSetup {
    pub deck: Vec<Card>,
    pub hp: i32,
    pub max_hp: i32,
    pub max_energy: i32,
    pub relics: Vec<Relic>,
    pub potions: Vec<Option<PotionId>>,
    pub enemies: Vec<EnemySpec>,
    pub encounter: Encounter,
    pub room: RoomKind,
    pub asc: Ascension,
    /// Floor the fight was generated for; 0 for recordings.
    pub floor: u32,
    /// Run gold, which only Gremlin Merc's Thievery reads.
    pub gold: i32,
}

impl FightSetup {
    pub fn as_setup(&self, seed: u64) -> Setup<'_> {
        Setup {
            deck: &self.deck,
            hp: self.hp,
            max_hp: self.max_hp,
            max_energy: self.max_energy,
            relics: &self.relics,
            potions: &self.potions,
            enemies: &self.enemies,
            room: self.room,
            asc: self.asc,
            seed,
            gold: self.gold,
        }
    }

    pub fn combat(&self, seed: u64) -> Combat {
        Combat::with_setup(&self.as_setup(seed))
    }

    /// Read a recorder file's `start` record (and the first snapshot for
    /// HP, which the start record does not carry). Relics the sim does not
    /// know are dropped; anything else unknown is an error.
    pub fn from_recording(text: &str, ids: &Ids) -> Result<Self, String> {
        let mut start = None;
        let mut first_snap = None;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).map_err(|e| format!("bad json: {e}"))?;
            match v["t"].as_str() {
                Some("start") if start.is_none() => start = Some(v),
                Some("snapshot") if first_snap.is_none() => first_snap = Some(v),
                _ => {}
            }
            if start.is_some() && first_snap.is_some() {
                break;
            }
        }
        let start = start.ok_or("no start record")?;
        let snap = first_snap.ok_or("no snapshot")?;
        Self::from_start(&start, &snap, ids)
    }

    /// The parsing shared with the replay harness.
    pub fn from_start(start: &Value, first_snap: &Value, ids: &Ids) -> Result<Self, String> {
        let RunParts { deck, relics, mut potions, gold, max_energy, asc } = RunParts::of(start, ids)?;
        // The belt is logged after Petrified Toad added its rock, which the
        // sim adds again. Taking out the first rock and letting the Toad fill
        // the first free slot gives back the logged belt, whichever rock was
        // the Toad's.
        let held = |id: RelicId| relics.iter().any(|r| r.id == id);
        if held(RelicId::PetrifiedToad) && !held(RelicId::Sozu) {
            if let Some(slot) = potions.iter().position(|&p| p == Some(PotionId::PotionShapedRock)) {
                potions[slot] = None;
            }
        }
        let enc_name = start["encounter"].as_str().unwrap_or("");
        let encounter = *ids.encounters.get(enc_name).ok_or_else(|| format!("unknown encounter {enc_name}"))?;
        let monsters: Vec<MonsterId> = start["enemies"]
            .as_array()
            .ok_or("start without enemies")?
            .iter()
            .map(|e| {
                let id = e["id"].as_str().unwrap_or("");
                ids.monsters.get(id).copied().ok_or_else(|| format!("unknown monster {id}"))
            })
            .collect::<Result<_, _>>()?;
        let room = match start["room"].as_str() {
            Some("Elite") => RoomKind::Elite,
            Some("Boss") => RoomKind::Boss,
            _ => RoomKind::Monster,
        };
        Ok(Self {
            deck,
            hp: first_snap["hp"].as_i64().unwrap_or(1) as i32,
            max_hp: first_snap["max_hp"].as_i64().unwrap_or(1) as i32,
            max_energy,
            relics,
            potions,
            enemies: specs_for(encounter, &monsters),
            encounter,
            room,
            asc,
            floor: 0,
            gold,
        })
    }

    /// The run in `start` (a recorder `start` record, or the same fields
    /// sent at a card reward) against `encounter` on `floor`, at `hp` of
    /// `max_hp`: the deck, relics, potions and gold are the run's, the
    /// enemies are rolled with `rng` as the generator would.
    pub fn run_against(start: &Value, ids: &Ids, hp: i32, max_hp: i32, encounter: Encounter, floor: u32, rng: &mut Rng) -> Result<Self, String> {
        let run = RunParts::of(start, ids)?;
        let rolled = generate_against(rng, floor, run.asc, encounter);
        Ok(Self {
            deck: run.deck,
            hp,
            max_hp,
            max_energy: run.max_energy,
            relics: run.relics,
            potions: run.potions,
            gold: run.gold,
            asc: run.asc,
            ..rolled
        })
    }
}

/// `repeats` fights of the run in `start` against each of `encounters`
/// (game name, floor), in that order (`FightSetup::run_against`). Fight k's
/// enemies are rolled from `seed` and k alone, so two runs given the same
/// encounters and seed face the same enemies.
pub fn run_fights(start: &Value, ids: &Ids, hp: i32, max_hp: i32, encounters: &[(String, u32)], repeats: usize, seed: u64) -> Result<Vec<FightSetup>, String> {
    let mut setups = vec![];
    for (name, floor) in encounters {
        let &enc = ids.encounters.get(name.as_str()).ok_or_else(|| format!("unknown encounter {name}"))?;
        for _ in 0..repeats {
            let mut rng = Rng::new(seed ^ (setups.len() as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            setups.push(FightSetup::run_against(start, ids, hp, max_hp, enc, *floor, &mut rng)?);
        }
    }
    Ok(setups)
}

/// What a card can become when transformed, by game name
/// (`CardFactory.GetDefaultTransformationOptions`): a Common, Uncommon or
/// Rare card of its pool other than itself, drawn uniformly. The pool is
/// the Ironclad's for its cards and the colorless one for the rest (quest,
/// event, ancient and token cards), named with the options. None for a
/// curse or status, which the game turns into another of its kind. Err for
/// a name the sim lacks.
pub fn transform_options(name: &str, ids: &Ids) -> Result<Option<(&'static str, Vec<String>)>, String> {
    let &id = ids.cards.get(name).ok_or_else(|| format!("unknown card {name}"))?;
    if matches!(def(id).ty, CardType::Curse | CardType::Status) {
        return Ok(None);
    }
    let (pool_name, pool) = if IRONCLAD_POOL.contains(&id) { ("Ironclad", IRONCLAD_POOL) } else { ("colorless", COLORLESS_POOL) };
    let options = pool
        .iter()
        .filter(|&&c| c != id && matches!(def(c).rarity, CardRarity::Common | CardRarity::Uncommon | CardRarity::Rare))
        .map(|c| crate::replay::slug(&format!("{c:?}")))
        .collect();
    Ok(Some((pool_name, options)))
}

/// What a fight takes from the run, read off a recorder `start` record.
struct RunParts {
    deck: Vec<Card>,
    relics: Vec<Relic>,
    potions: Vec<Option<PotionId>>,
    gold: i32,
    max_energy: i32,
    asc: Ascension,
}

impl RunParts {
    fn of(start: &Value, ids: &Ids) -> Result<Self, String> {
        let deck: Vec<Card> = start["deck"]
            .as_array()
            .ok_or("start without deck")?
            .iter()
            .map(|v| {
                let (id, up) = card_ref(ids, v)?;
                let mut k = Card::new(0, id, up);
                k.enchantment = card_ench(ids, v)?;
                Ok::<_, String>(k)
            })
            .collect::<Result<_, _>>()?;
        if let Some((id, why)) = UNSUPPORTED_CARDS.iter().find(|(id, _)| deck.iter().any(|k| k.id == *id)) {
            return Err(format!("unsupported card {id:?}: {why}"));
        }
        let relics: Vec<Relic> = start["relics"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|name| !INERT_RELICS.contains(name))
                    .map(|name| {
                        let &id = ids.relics.get(name).ok_or_else(|| format!("unknown relic {name}"))?;
                        let mut r = Relic::new(id);
                        if let Some(n) = start["relic_state"][name].as_i64() {
                            match id {
                                RelicId::LizardTail | RelicId::VenerableTeaSet => r.flag = n != 0,
                                _ => r.counter = relic_counter(id, n as i32),
                            }
                        }
                        Ok(r)
                    })
                    .collect::<Result<_, String>>()
            })
            .transpose()?
            .unwrap_or_default();
        let potions: Vec<Option<PotionId>> = start["potions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| match v.as_str() {
                        None => Ok(None),
                        Some(name) => ids.potions.get(name).copied().map(Some).ok_or_else(|| format!("unknown potion {name}")),
                    })
                    .collect::<Result<_, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            deck,
            relics,
            potions,
            gold: start["gold"].as_i64().unwrap_or(0) as i32,
            max_energy: start["max_energy"].as_i64().unwrap_or(3) as i32,
            asc: Ascension(start["ascension"].as_i64().unwrap_or(0) as u8),
        })
    }
}

/// A relic's `counter` from the value the recorder logs for it at combat
/// setup (`Recorder.RelicState`), which is the game's own field. Lizard
/// Tail and the Venerable Tea Set log a bool that goes into `flag`.
fn relic_counter(id: RelicId, game: i32) -> i32 {
    match id {
        // Counts that only matter modulo the play or turn they fire on,
        // which is how the sim keeps them.
        RelicId::IronClub => game % 4,
        RelicId::PenNib => game % 10,
        RelicId::HappyFlower | RelicId::Pendulum => game % 3,
        _ => game,
    }
}

pub(crate) fn card_ref(ids: &Ids, v: &Value) -> Result<(CardId, bool), String> {
    let id = v["id"].as_str().ok_or("card without id")?;
    let card = *ids.cards.get(id).ok_or_else(|| format!("unknown card {id}"))?;
    Ok((card, v["up"].as_bool().unwrap_or(false)))
}

/// The `ench` triple a recorded card carries, if it is enchanted.
pub(crate) fn card_ench(ids: &Ids, v: &Value) -> Result<Option<Enchantment>, String> {
    let Some(e) = v["ench"].as_array() else { return Ok(None) };
    let name = e.first().and_then(Value::as_str).ok_or("enchantment without id")?;
    let id = *ids.enchantments.get(name).ok_or_else(|| format!("unknown enchantment {name}"))?;
    Ok(Some(Enchantment {
        id,
        amount: e.get(1).and_then(Value::as_i64).unwrap_or(0) as i32,
        disabled: e.get(2).and_then(Value::as_bool).unwrap_or(false),
        data: 0,
    }))
}

/// Enemy specs for a recorded monster list: re-roll the encounter's own
/// composition until it matches, so positional flags come out right.
pub fn specs_for(enc: Encounter, monsters: &[MonsterId]) -> Vec<EnemySpec> {
    for seed in 0..2000u64 {
        let specs = enc.monsters(&mut Rng::new(seed));
        if specs.iter().map(|s| s.id).eq(monsters.iter().copied()) {
            return specs;
        }
    }
    monsters.iter().map(|&id| EnemySpec { id, flags: Default::default() }).collect()
}

/// Cards a reward screen can offer: the Ironclad pool minus basics and
/// specials (Break, Corruption come from other cards, not rewards).
fn reward_pool() -> Vec<CardId> {
    IRONCLAD_POOL.iter().copied().filter(|&id| !matches!(def(id).rarity, CardRarity::Basic | CardRarity::Special)).collect()
}

/// Deck plans a run can build toward. A run that commits to one takes
/// reward cards from it over others, the way a player does; grouped by what
/// each card's class does (`Models/Cards/<Name>.cs`), not by any tier list.
const PLANS: &[&[CardId]] = &[
    // Strength, and the multi-hits that scale with it.
    &[
        CardId::Inflame, CardId::DemonForm, CardId::FightMe, CardId::Brand, CardId::SetupStrike, CardId::Dominate,
        CardId::Rupture, CardId::SwordBoomerang, CardId::TwinStrike, CardId::Whirlwind, CardId::Thrash,
        CardId::PommelStrike, CardId::Uppercut,
    ],
    // Exhaust and the cards that pay off on it.
    &[
        CardId::Corruption, CardId::FeelNoPain, CardId::DarkEmbrace, CardId::BurningPact, CardId::SecondWind,
        CardId::TrueGrit, CardId::FiendFire, CardId::Havoc, CardId::EvilEye, CardId::ForgottenRitual,
        CardId::AshenStrike, CardId::DrumOfBattle, CardId::Stoke, CardId::Offering, CardId::Feed,
    ],
    // Block that stays, and the cards that hit with it.
    &[
        CardId::Barricade, CardId::BodySlam, CardId::Juggernaut, CardId::Unmovable, CardId::Impervious,
        CardId::FlameBarrier, CardId::ShrugItOff, CardId::StoneArmor, CardId::Colossus, CardId::IronWave,
        CardId::Armaments, CardId::Taunt,
    ],
    // Vulnerable and what it pays for.
    &[
        CardId::Cruelty, CardId::Vicious, CardId::Dominate, CardId::Tremble, CardId::Thunderclap, CardId::Uppercut,
        CardId::Bully, CardId::Dismantle, CardId::MoltenFist, CardId::Taunt, CardId::Colossus, CardId::Bludgeon,
    ],
    // Losing HP on purpose.
    &[
        CardId::Rupture, CardId::Bloodletting, CardId::Hemokinesis, CardId::Breakthrough, CardId::BloodWall,
        CardId::Offering, CardId::Brand, CardId::CrimsonMantle, CardId::Inferno, CardId::FeelNoPain,
    ],
];

/// One card reward: three offers by rarity, and the one a player on `plan`
/// would take, or none when the deck is big enough and nothing fits.
/// Rares show up more often in later acts, as the game's rare offset grows.
fn card_reward(rng: &mut Rng, pool: &[CardId], act: u32, deck: &[Card], plan: Option<&[CardId]>) -> Option<CardId> {
    let weight = |id: CardId| match def(id).rarity {
        CardRarity::Common => 60,
        CardRarity::Uncommon => 37,
        _ => 3 + 4 * act as usize,
    };
    let total: usize = pool.iter().map(|&id| weight(id)).sum();
    let mut offer = || {
        let mut n = rng.next_int(total);
        for &id in pool {
            if n < weight(id) {
                return id;
            }
            n -= weight(id);
        }
        pool[0]
    };
    let offers = [offer(), offer(), offer()];
    let Some(plan) = plan else { return Some(offers[rng.next_int(3)]) };
    let attacks = deck.iter().filter(|c| def(c.id).ty == CardType::Attack).count();
    let score = |id: CardId| -> f32 {
        let d = def(id);
        let mut s = match d.rarity {
            CardRarity::Rare => 1.0,
            CardRarity::Uncommon => 0.5,
            _ => 0.0,
        };
        if plan.contains(&id) {
            s += 2.0;
        }
        // A deck still needs to kill things.
        if d.ty == CardType::Attack && attacks * 3 < deck.len() {
            s += 1.5;
        }
        s
    };
    let best = offers.into_iter().max_by(|&a, &b| score(a).total_cmp(&score(b)))?;
    (deck.len() < 20 || score(best) >= 2.0).then_some(best)
}

/// Relics a run can hold besides the starter. Ancient relics are left out:
/// the Ancients that hand them out open acts 2 and 3, and Neow's are
/// either inert or not on the list.
fn relic_pool() -> Vec<RelicId> {
    relic::ALL
        .iter()
        .copied()
        .filter(|&id| !matches!(id, RelicId::BurningBlood | RelicId::LeadPaperweight) && !relic::ANCIENT.contains(&id))
        .collect()
}

/// Enchant a random unenchanted card with one the card can take. Amounts
/// are the 1 to 3 the act 1 sources hand out.
fn enchant_one(rng: &mut Rng, deck: &mut [Card]) {
    let plain: Vec<usize> = (0..deck.len()).filter(|&i| deck[i].enchantment.is_none()).collect();
    let Some(&i) = rng.pick(&plain) else { return };
    let k = &deck[i];
    let legal: Vec<_> =
        enchant::ALL.iter().copied().filter(|id| Enchantment::new(*id, 1).can_enchant(k)).collect();
    let Some(&id) = rng.pick(&legal) else { return };
    deck[i].enchant(id, rng.next_int(3) as i32 + 1);
}

/// Which encounters a floor can hold. The first floors of an act are the
/// weak pool (`NumberOfWeakEncounters`: 3 in act 1, 2 after), the act's
/// last floor is its boss, elites appear from its floor 5.
fn encounter_for(rng: &mut Rng, floor: u32) -> Encounter {
    let (act, local) = act_floor(floor);
    let weak = if act == 0 { 3 } else { 2 };
    let kind = if local == BOSS_FLOOR {
        Kind::Boss
    } else if local <= weak {
        Kind::Weak
    } else if local >= 5 && rng.next_int(4) == 0 {
        Kind::Elite
    } else {
        Kind::Normal
    };
    encounter_of_kind(rng, act, kind)
}

/// The map encounters of act `act` (0-based); act 1 is Overgrowth and
/// Underdocks together. Event fights are left out: no room rolls them.
fn act_encounters(act: u32) -> impl Iterator<Item = Encounter> {
    crate::encounter::ALL.iter().copied().filter(move |e| !e.is_event() && e.act().index() as u32 == act)
}

pub fn encounter_of_kind(rng: &mut Rng, act: u32, kind: Kind) -> Encounter {
    let pool: Vec<Encounter> = act_encounters(act).filter(|e| e.kind() == kind).collect();
    *rng.pick(&pool).unwrap()
}

/// Roll a run state for a fight on `floor` (1 to `LAST_FLOOR`), against an
/// encounter the floor can hold. Numbers are rough act 1 averages, kept for
/// the later acts: about two card picks per three floors, an upgrade every
/// six floors, a relic every four, potions used as fast as they come, and
/// an Ancient relic at the start of each act after the first.
pub fn generate(rng: &mut Rng, floor: u32, asc: Ascension) -> FightSetup {
    let floor = floor.clamp(1, LAST_FLOOR);
    let encounter = encounter_for(rng, floor);
    generate_against(rng, floor, asc, encounter)
}

/// `generate` for a chosen encounter, used to oversample elites and bosses.
pub fn generate_against(rng: &mut Rng, floor: u32, asc: Ascension, encounter: Encounter) -> FightSetup {
    let floor = floor.clamp(1, LAST_FLOOR);
    let mut deck = ironclad_starter_deck();
    if asc.has(AscensionLevel::AscendersBane) {
        deck.push(Card::new(0, CardId::AscendersBane, false));
    }
    let pool = reward_pool();
    // Most runs commit to a plan; the rest take whatever, so the policy
    // still sees unfocused decks.
    let plan = (rng.next_int(5) != 0).then(|| PLANS[rng.next_int(PLANS.len())]);
    // Act 1 rates per floor are set from the four played runs recorded by
    // 2026-09-23 (`scripts/deckstats.py --gen`): act 1 decks carry well
    // under one upgrade, enchantment and removal. Later acts had one played
    // run between them, too few to set anything from.
    for f in 1..floor {
        // Later acts have more rest sites spent on smithing, and more shops
        // and events that remove cards.
        let later = act_floor(f).0 > 0;
        if rng.next_int(3) < 2 {
            if let Some(id) = card_reward(rng, &pool, act_floor(f).0, &deck, plan) {
                deck.push(Card::new(0, id, rng.next_int(20) == 0));
            }
        }
        // Smiths go to the cards that matter, not to Strikes.
        if rng.next_int(if later { 4 } else { 20 }) == 0 {
            let good: Vec<usize> = (0..deck.len())
                .filter(|&i| !deck[i].upgraded && !matches!(def(deck[i].id).rarity, CardRarity::Basic | CardRarity::Special))
                .collect();
            let i = match rng.pick(&good) {
                Some(&i) => i,
                None => rng.next_int(deck.len()),
            };
            if def(deck[i].id).rarity != CardRarity::Special {
                deck[i].upgraded = true;
            }
        }
        // Removals take Strikes first, then Defends.
        if rng.next_int(if later { 5 } else { 16 }) == 0 {
            let strikes = deck.iter().filter(|c| c.id == CardId::StrikeIronclad).count();
            let first = if strikes > 0 { CardId::StrikeIronclad } else { CardId::DefendIronclad };
            if let Some(i) = deck.iter().position(|c| c.id == first) {
                deck.remove(i);
            }
        }
        // Relics and events hand out enchantments; act 1 rarely sees more
        // than one or two, and they only ever land on a card that accepts
        // them (`EnchantmentModel.CanEnchant`).
        if rng.next_int(30) == 0 {
            enchant_one(rng, &mut deck);
        }
    }

    let mut relics = vec![Relic::new(RelicId::BurningBlood)];
    let mut relic_pool = relic_pool();
    let n_relics = (floor / 4) as usize + usize::from(rng.next_int(2) == 0 && floor > 1);
    for _ in 0..n_relics.min(relic_pool.len()) {
        let i = rng.next_int(relic_pool.len());
        relics.push(Relic::new(relic_pool.swap_remove(i)));
    }
    let mut ancients = relic::ANCIENT.to_vec();
    for _ in 0..act_floor(floor).0 {
        let i = rng.next_int(ancients.len());
        relics.push(Relic::new(ancients.swap_remove(i)));
    }
    let slots = if relics.iter().any(|r| r.id == RelicId::PotionBelt) { 4 } else { 2 };
    // Played runs carry about one potion into an act 1 fight.
    let potions: Vec<Option<PotionId>> =
        (0..slots).map(|_| if rng.next_int(2) == 0 { Some(*rng.pick(potion::ALL).unwrap()) } else { None }).collect();

    let room = match encounter.kind() {
        Kind::Elite => RoomKind::Elite,
        Kind::Boss => RoomKind::Boss,
        _ => RoomKind::Monster,
    };
    // Events, relics and Ancients raise max HP by roughly this much an act.
    let max_hp = IRONCLAD_HP + (0..act_floor(floor).0).map(|_| 5 + rng.next_int(8) as i32).sum::<i32>();
    // The floor before the boss is a rest site, so boss fights start
    // rested. Played runs reach elites and other fights at about 65%.
    let (lo, span) = match encounter.kind() {
        Kind::Boss => (0.7, 0.3),
        _ if floor == 1 => (1.0, 0.0),
        _ => (0.35, 0.6),
    };
    let hp = (max_hp as f32 * (lo + rng.next_float(span))).round() as i32;
    FightSetup {
        deck,
        hp: hp.max(1),
        max_hp,
        max_energy: IRONCLAD_ENERGY,
        relics,
        potions,
        enemies: encounter.monsters(rng),
        encounter,
        room,
        asc,
        floor,
        // Roughly what a run is carrying by this floor, before it spends any.
        gold: 99 + (floor * 25) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_setups_run_to_completion() {
        let mut rng = Rng::new(42);
        for i in 0..300u64 {
            let floor = 1 + (i % LAST_FLOOR as u64) as u32;
            let s = generate(&mut rng, floor, Ascension(10));
            assert!(s.deck.len() >= 8 && s.hp >= 1 && s.hp <= s.max_hp);
            assert_eq!(s.relics[0].id, RelicId::BurningBlood);
            let mut c = s.combat(i);
            let mut steps = 0;
            while !c.is_over() && steps < 5000 {
                let acts = c.legal_actions();
                c.step(acts[rng.next_int(acts.len())]);
                steps += 1;
            }
            assert!(c.is_over(), "runaway fight on floor {floor}: {:?}", s.encounter);
        }
    }

    #[test]
    fn holdout_is_deterministic_and_covers_every_encounter() {
        let a = holdout(7, 3, Ascension(10), ACTS);
        let b = holdout(7, 3, Ascension(10), ACTS);
        assert_eq!(a.len(), 3 * crate::encounter::ALL.iter().filter(|e| !e.is_event()).count());
        assert!(a.iter().all(|s| !s.encounter.is_event()));
        assert!(a.iter().zip(&b).all(|(x, y)| x.encounter == y.encounter && x.deck.len() == y.deck.len() && x.hp == y.hp));
        assert!(a.iter().all(|s| act_floor(s.floor).0 == s.encounter.act().index() as u32));
    }

    #[test]
    fn boss_fights_start_rested() {
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            let s = generate(&mut rng, BOSS_FLOOR, Ascension(10));
            assert!(s.hp >= (s.max_hp as f32 * 0.7).round() as i32, "boss start hp {}", s.hp);
        }
    }

    #[test]
    fn floor_curriculum_orders_encounter_kinds() {
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            assert_eq!(generate(&mut rng, 1, Ascension(10)).encounter.kind(), Kind::Weak);
            assert_eq!(generate(&mut rng, BOSS_FLOOR, Ascension(10)).encounter.kind(), Kind::Boss);
            assert_ne!(generate(&mut rng, 4, Ascension(10)).encounter.kind(), Kind::Elite);
            let s = generate(&mut rng, LAST_FLOOR, Ascension(10));
            assert_eq!((s.encounter.act().index(), s.encounter.kind()), (2, Kind::Boss));
            assert_eq!(generate(&mut rng, BOSS_FLOOR + 1, Ascension(10)).encounter.act().index(), 1);
        }
    }
}

/// Every `*.jsonl` recording in `dir` (not recursive) as a setup, sorted by
/// file name. Files the sim cannot read are reported, not fatal.
pub fn load_recordings(dir: &std::path::Path) -> Result<(Vec<FightSetup>, Vec<String>), String> {
    let ids = Ids::new();
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    paths.sort();
    let mut setups = vec![];
    let mut errors = vec![];
    for p in paths {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| FightSetup::from_recording(&t, &ids)) {
            Ok(s) => setups.push(s),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    Ok((setups, errors))
}

/// A fixed held-out set: `per_encounter` generated fights against every
/// map encounter of `acts`, on floors that encounter can appear on. Seeded, so
/// every evaluation sees the same decks.
pub fn holdout(seed: u64, per_encounter: usize, asc: Ascension, acts: u32) -> Vec<FightSetup> {
    let mut rng = Rng::new(seed);
    let mut out = vec![];
    for act in 0..acts.clamp(1, ACTS) {
        let weak = if act == 0 { 3 } else { 2 };
        for enc in act_encounters(act) {
            for _ in 0..per_encounter {
                let local = match enc.kind() {
                    Kind::Weak => 1 + rng.next_int(weak as usize) as u32,
                    Kind::Normal => weak + 1 + rng.next_int((BOSS_FLOOR - weak - 1) as usize) as u32,
                    Kind::Elite => 5 + rng.next_int(11) as u32,
                    Kind::Boss => BOSS_FLOOR,
                };
                out.push(generate_against(&mut rng, act * BOSS_FLOOR + local, asc, enc));
            }
        }
    }
    out
}
