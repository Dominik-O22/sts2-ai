//! Fight setups for training. A generator rolls plausible act 1 run states
//! keyed on the floor the fight happens at (DESIGN.md, Training: "wide, not
//! clever"), and `FightSetup::from_recording` reads the recorder's `start`
//! record so real fights form the held-out set.

use serde_json::Value;

use crate::card::{def, Card, IRONCLAD_POOL};
use crate::combat::{Combat, EnemySpec, RoomKind, Setup};
use crate::encounter::{Encounter, Kind};
use crate::ids::{CardId, MonsterId};
use crate::potion::{self, PotionId};
use crate::relic::{self, Relic, RelicId};
use crate::replay::Ids;
use crate::rng::Rng;
use crate::types::{Ascension, AscensionLevel, CardRarity};
use crate::{ironclad_starter_deck, IRONCLAD_ENERGY, IRONCLAD_HP};

/// Last floor of act 1: the boss room.
pub const BOSS_FLOOR: u32 = 16;

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
        let deck: Vec<Card> = start["deck"]
            .as_array()
            .ok_or("start without deck")?
            .iter()
            .map(|v| card_ref(ids, v).map(|(id, up)| Card::new(0, id, up)))
            .collect::<Result<_, _>>()?;
        let relics: Vec<Relic> = start["relics"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| ids.relics.get(v.as_str()?)).map(|&id| Relic::new(id)).collect())
            .unwrap_or_default();
        let potions: Vec<Option<PotionId>> = start["potions"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().and_then(|s| ids.potions.get(s).copied())).collect())
            .unwrap_or_default();
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
            max_energy: start["max_energy"].as_i64().unwrap_or(3) as i32,
            relics,
            potions,
            enemies: specs_for(encounter, &monsters),
            encounter,
            room,
            asc: Ascension(start["ascension"].as_i64().unwrap_or(0) as u8),
            floor: 0,
        })
    }
}

pub(crate) fn card_ref(ids: &Ids, v: &Value) -> Result<(CardId, bool), String> {
    let id = v["id"].as_str().ok_or("card without id")?;
    let card = *ids.cards.get(id).ok_or_else(|| format!("unknown card {id}"))?;
    Ok((card, v["up"].as_bool().unwrap_or(false)))
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

/// Relics a run can hold besides the starter. Ancient (boss) relics are
/// left out: act 1 fights happen before the first boss chest.
fn relic_pool() -> Vec<RelicId> {
    relic::ALL
        .iter()
        .copied()
        .filter(|&id| !matches!(id, RelicId::BurningBlood | RelicId::PaelsFlesh | RelicId::LeadPaperweight))
        .collect()
}

/// Which encounters a floor can hold. Floors 1-3 are the weak pool, the
/// boss floor is a boss, elites appear from floor 5 (`Overgrowth.cs`
/// places the first elite after the weak stretch).
fn encounter_for(rng: &mut Rng, floor: u32) -> Encounter {
    let kind = if floor >= BOSS_FLOOR {
        Kind::Boss
    } else if floor <= 3 {
        Kind::Weak
    } else if floor >= 5 && rng.next_int(4) == 0 {
        Kind::Elite
    } else {
        Kind::Normal
    };
    encounter_of_kind(rng, kind)
}

pub fn encounter_of_kind(rng: &mut Rng, kind: Kind) -> Encounter {
    let pool: Vec<Encounter> = crate::encounter::ALL.iter().copied().filter(|e| e.kind() == kind).collect();
    *rng.pick(&pool).unwrap()
}

/// Roll a run state for a fight on `floor` (1 to `BOSS_FLOOR`), against an
/// encounter the floor can hold. Numbers
/// are rough act 1 averages: about two card picks per three floors, an
/// upgrade every six floors, a relic every four, potions used as fast as
/// they come.
pub fn generate(rng: &mut Rng, floor: u32, asc: Ascension) -> FightSetup {
    let floor = floor.clamp(1, BOSS_FLOOR);
    let encounter = encounter_for(rng, floor);
    generate_against(rng, floor, asc, encounter)
}

/// `generate` for a chosen encounter, used to oversample elites and bosses.
pub fn generate_against(rng: &mut Rng, floor: u32, asc: Ascension, encounter: Encounter) -> FightSetup {
    let floor = floor.clamp(1, BOSS_FLOOR);
    let mut deck = ironclad_starter_deck();
    if asc.has(AscensionLevel::AscendersBane) {
        deck.push(Card::new(0, CardId::AscendersBane, false));
    }
    let pool = reward_pool();
    let rarity_weight = |id: CardId| match def(id).rarity {
        CardRarity::Common => 60,
        CardRarity::Uncommon => 37,
        _ => 3,
    };
    let total: usize = pool.iter().map(|&id| rarity_weight(id)).sum();
    let pick_reward = |rng: &mut Rng| {
        let mut n = rng.next_int(total);
        for &id in &pool {
            let w = rarity_weight(id);
            if n < w {
                return id;
            }
            n -= w;
        }
        pool[0]
    };
    for _ in 1..floor {
        if rng.next_int(3) < 2 {
            let id = pick_reward(rng);
            deck.push(Card::new(0, id, rng.next_int(8) == 0));
        }
        // Smiths and events: upgrade a random card, or remove a basic.
        if rng.next_int(6) == 0 {
            let i = rng.next_int(deck.len());
            if def(deck[i].id).rarity != CardRarity::Special {
                deck[i].upgraded = true;
            }
        }
        if rng.next_int(8) == 0 {
            if let Some(i) = deck.iter().position(|c| matches!(c.id, CardId::StrikeIronclad | CardId::DefendIronclad)) {
                deck.remove(i);
            }
        }
    }

    let mut relics = vec![Relic::new(RelicId::BurningBlood)];
    let mut relic_pool = relic_pool();
    let n_relics = (floor / 4) as usize + usize::from(rng.next_int(2) == 0 && floor > 1);
    for _ in 0..n_relics.min(relic_pool.len()) {
        let i = rng.next_int(relic_pool.len());
        relics.push(Relic::new(relic_pool.swap_remove(i)));
    }
    let slots = if relics.iter().any(|r| r.id == RelicId::PotionBelt) { 4 } else { 2 };
    let potions: Vec<Option<PotionId>> =
        (0..slots).map(|_| if rng.next_int(3) == 0 { Some(*rng.pick(potion::ALL).unwrap()) } else { None }).collect();

    let room = match encounter.kind() {
        Kind::Elite => RoomKind::Elite,
        Kind::Boss => RoomKind::Boss,
        _ => RoomKind::Monster,
    };
    let max_hp = IRONCLAD_HP;
    let hp = if floor == 1 { max_hp } else { (max_hp as f32 * (0.4 + rng.next_float(0.6))).round() as i32 };
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_setups_run_to_completion() {
        let mut rng = Rng::new(42);
        for i in 0..300u64 {
            let floor = 1 + (i % BOSS_FLOOR as u64) as u32;
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
        let a = holdout(7, 3, Ascension(10));
        let b = holdout(7, 3, Ascension(10));
        assert_eq!(a.len(), 3 * crate::encounter::ALL.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.encounter == y.encounter && x.deck.len() == y.deck.len() && x.hp == y.hp));
        assert!(crate::encounter::ALL.iter().all(|e| a.iter().any(|s| s.encounter == *e)));
    }

    #[test]
    fn floor_curriculum_orders_encounter_kinds() {
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            assert_eq!(generate(&mut rng, 1, Ascension(10)).encounter.kind(), Kind::Weak);
            assert_eq!(generate(&mut rng, BOSS_FLOOR, Ascension(10)).encounter.kind(), Kind::Boss);
            assert_ne!(generate(&mut rng, 4, Ascension(10)).encounter.kind(), Kind::Elite);
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
/// act 1 encounter, on floors that encounter can appear on. Seeded, so
/// every evaluation sees the same decks.
pub fn holdout(seed: u64, per_encounter: usize, asc: Ascension) -> Vec<FightSetup> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(per_encounter * crate::encounter::ALL.len());
    for &enc in crate::encounter::ALL {
        for _ in 0..per_encounter {
            let floor = match enc.kind() {
                Kind::Weak => 1 + rng.next_int(3) as u32,
                Kind::Normal => 4 + rng.next_int(12) as u32,
                Kind::Elite => 5 + rng.next_int(11) as u32,
                Kind::Boss => BOSS_FLOOR,
            };
            out.push(generate_against(&mut rng, floor, asc, enc));
        }
    }
    out
}
