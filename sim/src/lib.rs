//! Slay the Spire 2 combat simulator, ported from the decompiled v0.107.1
//! assembly. See DESIGN.md and docs/survey/README.md for the decisions.

pub mod card;
pub mod combat;
pub mod effect;
pub mod encounter;
pub mod ids;
pub mod monster;
pub mod potion;
pub mod power;
pub mod relic;
pub mod replay;
pub mod rng;
pub mod types;

pub use combat::{Action, Combat, EnemySpec, Outcome, RoomKind, Setup};
pub use potion::PotionId;
pub use relic::{Relic, RelicId};
pub use types::Ascension;

/// The Ironclad's starting deck: `Models/Characters/Ironclad.cs`.
pub fn ironclad_starter_deck() -> Vec<card::Card> {
    use ids::CardId::*;
    [
        StrikeIronclad,
        StrikeIronclad,
        StrikeIronclad,
        StrikeIronclad,
        StrikeIronclad,
        DefendIronclad,
        DefendIronclad,
        DefendIronclad,
        DefendIronclad,
        Bash,
    ]
    .into_iter()
    .map(|id| card::Card::new(0, id, false))
    .collect()
}

/// Ironclad starting stats: `Models/Characters/Ironclad.cs`, `CharacterModel.MaxEnergy`.
pub const IRONCLAD_HP: i32 = 80;
pub const IRONCLAD_ENERGY: i32 = 3;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{MonsterId, PowerId};
    use crate::monster::Flags;
    use crate::power::Power;
    use crate::types::{CreatureRef, ValueProp};

    fn nibbit_fight(seed: u64) -> Combat {
        Combat::new(
            &ironclad_starter_deck(),
            IRONCLAD_HP,
            IRONCLAD_HP,
            IRONCLAD_ENERGY,
            &[EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }],
            Ascension(10),
            seed,
        )
    }

    #[test]
    fn damage_formula_matches_game_truncation() {
        // Strength 3 + base 6 = 9, x1.5 Vulnerable, x0.75 Weak = 10.125 -> 10.
        let mut c = nibbit_fight(1);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 3));
        c.player.creature.powers.push(Power::new(PowerId::Weak, 1));
        c.enemies[0].creature.powers.push(Power::new(PowerId::Vulnerable, 1));
        let d = c.modify_damage(CreatureRef::Enemy(0), Some(CreatureRef::Player), 6.0, ValueProp::MOVE);
        assert_eq!(d, 10.125);
        // Unpowered damage (potions) ignores all of it.
        let d = c.modify_damage(CreatureRef::Enemy(0), Some(CreatureRef::Player), 6.0, ValueProp::UNPOWERED);
        assert_eq!(d, 6.0);
    }

    #[test]
    fn first_turn_draws_five_with_three_energy() {
        let c = nibbit_fight(7);
        assert_eq!(c.player.hand.len(), 5);
        assert_eq!(c.player.draw.len(), 5);
        assert_eq!(c.player.energy, 3);
        assert!(c.enemies[0].monster.next_move_name().is_some());
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("BUTT_MOVE"));
    }

    #[test]
    fn alone_nibbit_cycles_butt_slice_hiss() {
        let mut c = nibbit_fight(3);
        let mut seen = vec![];
        for _ in 0..4 {
            seen.push(c.enemies[0].monster.next_move_name().unwrap());
            c.step(Action::EndTurn);
        }
        assert_eq!(seen, ["BUTT_MOVE", "SLICE_MOVE", "HISS_MOVE", "BUTT_MOVE"]);
        // A10 Nibbit: 13 from Butt, then 7 from Slice with +0 Str, Hiss gives 3 Str.
        assert_eq!(c.player.creature.hp, 80 - 13 - 7 - (13 + 3));
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Strength), 3);
    }

    #[test]
    fn vulnerable_from_bash_skips_no_tick_on_enemies_and_expires() {
        let mut c = nibbit_fight(11);
        // Force a Bash into hand slot 0.
        let bash = c.player.draw.iter().position(|k| k.id == ids::CardId::Bash);
        if let Some(i) = bash {
            let card = c.player.draw.remove(i);
            c.player.hand.insert(0, card);
        } else {
            let i = c.player.hand.iter().position(|k| k.id == ids::CardId::Bash).unwrap();
            c.player.hand.swap(0, i);
        }
        let hp0 = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: 0, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp0 - 8);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Vulnerable), 2);
        // Enemies get no skip flag: ticks at the end of each enemy turn.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Vulnerable), 1);
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Vulnerable), 0);
        assert!(c.enemies[0].creature.power(PowerId::Vulnerable).is_none());
    }

    /// Random decks from the whole Ironclad pool, random policy, many seeds.
    /// Exists to catch panics and runaway loops in card ports, not to check
    /// any specific rule.
    #[test]
    fn random_pool_decks_do_not_panic() {
        use crate::card::{Card, IRONCLAD_POOL};
        for seed in 0..400u64 {
            let mut rng = rng::Rng::new(seed);
            let mut deck = ironclad_starter_deck();
            for _ in 0..8 {
                let id = *rng.pick(IRONCLAD_POOL).unwrap();
                deck.push(Card::new(0, id, rng.next_int(3) == 0));
            }
            deck.push(Card::new(0, ids::CardId::AscendersBane, false));
            let mut c = Combat::new(
                &deck,
                IRONCLAD_HP,
                IRONCLAD_HP,
                IRONCLAD_ENERGY,
                &[EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } }],
                Ascension(10),
                seed,
            );
            let mut steps = 0;
            while !c.is_over() {
                let acts = c.legal_actions();
                assert!(!acts.is_empty(), "no legal actions at seed {seed}");
                let a = acts[rng.next_int(acts.len())];
                c.step(a);
                steps += 1;
                assert!(steps < 20_000, "runaway fight at seed {seed}");
                assert!(c.player.hand.len() <= combat::MAX_HAND);
            }
        }
    }

    fn fight(enemies: &[EnemySpec], seed: u64) -> Combat {
        Combat::new(&ironclad_starter_deck(), IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, enemies, Ascension(10), seed)
    }

    fn one(id: MonsterId) -> EnemySpec {
        EnemySpec { id, flags: Flags::default() }
    }

    /// Play until the combat ends or `max` steps, random policy.
    fn play_random(c: &mut Combat, rng: &mut rng::Rng, max: u32) {
        let mut steps = 0;
        while !c.is_over() && steps < max {
            let acts = c.legal_actions();
            assert!(!acts.is_empty(), "no legal actions");
            c.step(acts[rng.next_int(acts.len())]);
            steps += 1;
        }
    }

    #[test]
    fn slippery_caps_hp_loss_at_one_per_hit() {
        let mut c = fight(&[one(MonsterId::Vantom)], 5);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Slippery), 9);
        let hp0 = c.enemies[0].creature.hp;
        c.player.creature.powers.push(Power::new(PowerId::Strength, 50));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp0 - 1);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Slippery), 8);
    }

    #[test]
    fn phrog_death_spawns_four_wrigglers_and_combat_continues() {
        let mut c = fight(&[one(MonsterId::PhrogParasite)], 9);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert!(!c.enemies[0].creature.alive());
        assert_eq!(c.enemies.len(), 5);
        assert!(!c.is_over());
        assert!(c.enemies[1..].iter().all(|e| e.monster.id == MonsterId::Wriggler && e.creature.alive()));
        // Spawned wrigglers open stunned, then bite/wriggle by slot.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[1].monster.next_move_name(), Some("NASTY_BITE_MOVE"));
        assert_eq!(c.enemies[2].monster.next_move_name(), Some("WRIGGLE_MOVE"));
    }

    #[test]
    fn ceremonial_beast_plow_breaks_into_stun_then_cry() {
        let mut c = fight(&[one(MonsterId::CeremonialBeast)], 2);
        // Turn 1: Stamp applies Plow 160 at A10.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Plow), 160);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("PLOW_MOVE"));
        // Bring it to 170, then hit for 20: HP 150 <= 160 breaks the Plow.
        c.enemies[0].creature.hp = 170;
        c.player.creature.powers.push(Power::new(PowerId::Strength, 14));
        let i = c.player.hand.iter().position(|k| k.id == ids::CardId::StrikeIronclad).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, 150);
        assert!(c.enemies[0].creature.power(PowerId::Plow).is_none());
        assert!(c.enemies[0].creature.power(PowerId::Strength).is_none());
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("STUNNED"));
        let hp = c.player.creature.hp;
        c.step(Action::EndTurn);
        assert_eq!(c.player.creature.hp, hp, "stunned turn deals no damage");
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("BEAST_CRY_MOVE"));
        c.step(Action::EndTurn);
        assert!(c.player.creature.power(PowerId::Ringing).is_some());
        // Ringing: only one card play this turn.
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.legal_actions(), vec![Action::EndTurn]);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("STOMP_MOVE"));
    }

    #[test]
    fn eye_with_teeth_revives_and_fogmog_death_wins() {
        let mut c = fight(&[one(MonsterId::Fogmog)], 4);
        c.step(Action::EndTurn); // Illusion move: summon the Eye.
        assert_eq!(c.enemies.len(), 2);
        assert!(!c.enemies[1].primary());
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(1) });
        assert!(!c.enemies[1].creature.alive() && c.enemies[1].reviving);
        assert!(!c.is_over());
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[1].creature.hp, 6, "revived to full");
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.outcome, Some(Outcome::Won), "primary enemy dead ends combat");
    }

    #[test]
    fn kin_priest_death_ends_the_fight() {
        let mut rng = rng::Rng::new(1);
        let specs = encounter::Encounter::TheKinBoss.monsters(&mut rng);
        let mut c = fight(&specs, 8);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(1) });
        assert_eq!(c.outcome, Some(Outcome::Won));
    }

    #[test]
    fn cubex_artifact_blocks_first_debuff() {
        let mut c = fight(&[one(MonsterId::CubexConstruct)], 3);
        assert_eq!(c.enemies[0].creature.block, 13);
        let i = c.player.hand.iter().position(|k| k.id == ids::CardId::Bash).unwrap_or_else(|| {
            let j = c.player.draw.iter().position(|k| k.id == ids::CardId::Bash).unwrap();
            let card = c.player.draw.remove(j);
            c.player.hand.insert(0, card);
            0
        });
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert!(c.enemies[0].creature.power(PowerId::Vulnerable).is_none());
        assert!(c.enemies[0].creature.power(PowerId::Artifact).is_none());
    }

    fn with_relics(relics: &[Relic], enemies: &[EnemySpec], seed: u64) -> Combat {
        Combat::with_setup(&Setup {
            deck: &ironclad_starter_deck(),
            hp: IRONCLAD_HP,
            max_hp: IRONCLAD_HP,
            max_energy: IRONCLAD_ENERGY,
            relics,
            potions: &[],
            enemies,
            room: RoomKind::Monster,
            asc: Ascension(10),
            seed,
        })
    }

    fn with_potions(potions: &[Option<PotionId>], enemies: &[EnemySpec], seed: u64) -> Combat {
        Combat::with_setup(&Setup {
            deck: &ironclad_starter_deck(),
            hp: IRONCLAD_HP,
            max_hp: IRONCLAD_HP,
            max_energy: IRONCLAD_ENERGY,
            relics: &[],
            potions,
            enemies,
            room: RoomKind::Monster,
            asc: Ascension(10),
            seed,
        })
    }

    #[test]
    fn fire_potion_ignores_strength_and_empties_its_slot() {
        let mut c = with_potions(&[Some(PotionId::FirePotion), None], &[one(MonsterId::Nibbit)], 1);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 5));
        let hp = c.enemies[0].creature.hp;
        assert!(c.legal_actions().contains(&Action::UsePotion { slot: 0, target: Some(0) }));
        c.step(Action::UsePotion { slot: 0, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - 20);
        assert_eq!(c.potions, vec![None, None]);
        assert!(!c.legal_actions().iter().any(|a| matches!(a, Action::UsePotion { .. })));
    }

    #[test]
    fn fairy_in_a_bottle_is_automatic_and_belt_buckle_waits_for_it() {
        let alone = EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } };
        let mut c = Combat::with_setup(&Setup {
            deck: &ironclad_starter_deck(),
            hp: 5,
            max_hp: IRONCLAD_HP,
            max_energy: IRONCLAD_ENERGY,
            relics: &[Relic::new(RelicId::BeltBuckle)],
            potions: &[Some(PotionId::FairyInABottle)],
            enemies: &[alone],
            room: RoomKind::Monster,
            asc: Ascension(10),
            seed: 3,
        });
        assert!(!c.legal_actions().iter().any(|a| matches!(a, Action::UsePotion { .. })));
        assert_eq!(c.player.creature.power_amount(PowerId::Dexterity), 0);
        c.step(Action::EndTurn); // Butt for 13.
        assert_eq!(c.player.creature.hp, 24);
        assert_eq!(c.potions, vec![None]);
        assert_eq!(c.player.creature.power_amount(PowerId::Dexterity), 2, "Belt Buckle fires on the automatic use");
        assert!(!c.is_over());
    }

    #[test]
    fn gamblers_brew_draws_one_per_discard_on_skip() {
        let mut c = with_potions(&[Some(PotionId::GamblersBrew)], &[one(MonsterId::Nibbit)], 4);
        c.step(Action::UsePotion { slot: 0, target: None });
        assert!(c.legal_actions().contains(&Action::Skip));
        c.step(Action::Choose(0));
        c.step(Action::Choose(0));
        assert_eq!(c.player.hand.len(), 3);
        c.step(Action::Skip);
        assert_eq!(c.player.hand.len(), 5);
        assert_eq!(c.player.discard.len(), 2);
        assert!(c.pending.is_none());
    }

    #[test]
    fn gigantification_triples_one_attack_card() {
        let mut c = with_potions(&[Some(PotionId::GigantificationPotion)], &[one(MonsterId::Nibbit)], 5);
        c.step(Action::UsePotion { slot: 0, target: None });
        let strike = |c: &Combat| c.player.hand.iter().position(|k| k.id == ids::CardId::StrikeIronclad);
        for _ in 0..2 {
            if strike(&c).is_none() {
                let j = c.player.draw.iter().position(|k| k.id == ids::CardId::StrikeIronclad).unwrap();
                let card = c.player.draw.remove(j);
                c.player.hand.insert(0, card);
            }
        }
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: strike(&c).unwrap(), target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - 18);
        assert!(c.player.creature.power(PowerId::Gigantification).is_none());
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: strike(&c).unwrap(), target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - 6);
    }

    /// Three random potions on random decks against every encounter, used
    /// whenever the random policy feels like it.
    #[test]
    fn random_potions_do_not_panic() {
        use crate::card::{Card, IRONCLAD_POOL};
        for (n, enc) in encounter::ALL.iter().enumerate() {
            for seed in 0..25u64 {
                let seed = seed * 1000 + n as u64;
                let mut rng = rng::Rng::new(seed);
                let specs = enc.monsters(&mut rng);
                let mut deck = ironclad_starter_deck();
                for _ in 0..8 {
                    deck.push(Card::new(0, *rng.pick(IRONCLAD_POOL).unwrap(), rng.next_int(3) == 0));
                }
                let potions: Vec<Option<PotionId>> = (0..3).map(|_| Some(*rng.pick(potion::ALL).unwrap())).collect();
                let mut c = with_potions(&potions, &specs, seed);
                play_random(&mut c, &mut rng, 20_000);
                assert!(c.is_over(), "runaway fight in {enc:?} seed {seed} potions {potions:?}");
            }
        }
    }

    #[test]
    fn vajra_and_anchor_apply_at_combat_start() {
        let c = with_relics(&[Relic::new(RelicId::Vajra), Relic::new(RelicId::Anchor)], &[one(MonsterId::Nibbit)], 1);
        assert_eq!(c.player.creature.power_amount(PowerId::Strength), 1);
        assert_eq!(c.player.creature.block, 10);
    }

    #[test]
    fn pen_nib_doubles_the_tenth_attack_and_counter_persists() {
        let mut nib = Relic::new(RelicId::PenNib);
        nib.counter = 8;
        let mut c = with_relics(&[nib], &[one(MonsterId::Vantom)], 2);
        // Strip Slippery so damage is visible.
        c.enemies[0].creature.powers.clear();
        let strike = |c: &Combat| c.player.hand.iter().position(|k| k.id == ids::CardId::StrikeIronclad);
        if strike(&c).is_none() {
            let j = c.player.draw.iter().position(|k| k.id == ids::CardId::StrikeIronclad).unwrap();
            let card = c.player.draw.remove(j);
            c.player.hand.insert(0, card);
        }
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: strike(&c).unwrap(), target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - 6, "ninth attack is normal");
        if strike(&c).is_none() {
            let j = c.player.draw.iter().position(|k| k.id == ids::CardId::StrikeIronclad).unwrap();
            let card = c.player.draw.remove(j);
            c.player.hand.insert(0, card);
        }
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: strike(&c).unwrap(), target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - 12, "tenth attack is doubled");
        assert_eq!(c.relics[0].counter, 0, "counter wrapped and is readable after combat");
    }

    #[test]
    fn lizard_tail_prevents_one_death() {
        let alone = EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } };
        let mut c = with_relics(&[Relic::new(RelicId::LizardTail)], &[alone], 3);
        c.player.creature.hp = 5;
        c.step(Action::EndTurn); // Butt for 13.
        assert_eq!(c.player.creature.hp, 40);
        assert!(c.relics[0].flag);
        assert!(!c.is_over());
    }

    /// Random relic sets on random decks against every encounter.
    #[test]
    fn random_relic_sets_do_not_panic() {
        use crate::card::{Card, IRONCLAD_POOL};
        let all: Vec<RelicId> = relic::ALL.to_vec();
        for (n, enc) in encounter::ALL.iter().enumerate() {
            for seed in 0..25u64 {
                let seed = seed * 1000 + n as u64;
                let mut rng = rng::Rng::new(seed);
                let specs = enc.monsters(&mut rng);
                let mut deck = ironclad_starter_deck();
                for _ in 0..8 {
                    deck.push(Card::new(0, *rng.pick(IRONCLAD_POOL).unwrap(), rng.next_int(3) == 0));
                }
                let relics: Vec<Relic> = (0..6).map(|_| Relic::new(*rng.pick(&all).unwrap())).collect();
                let mut c = Combat::with_setup(&Setup {
                    deck: &deck,
                    hp: IRONCLAD_HP,
                    max_hp: IRONCLAD_HP,
                    max_energy: IRONCLAD_ENERGY,
                    relics: &relics,
                    potions: &[],
                    enemies: &specs,
                    room: RoomKind::Elite,
                    asc: Ascension(10),
                    seed,
                });
                play_random(&mut c, &mut rng, 20_000);
                assert!(c.is_over(), "runaway fight in {enc:?} seed {seed} relics {relics:?}");
            }
        }
    }

    /// Every act 1 encounter, random pool decks, random policy: no panics,
    /// no runaway fights.
    #[test]
    fn every_act1_encounter_runs() {
        use crate::card::{Card, IRONCLAD_POOL};
        for (n, enc) in encounter::ALL.iter().enumerate() {
            for seed in 0..60u64 {
                let seed = seed * 100 + n as u64;
                let mut rng = rng::Rng::new(seed);
                let specs = enc.monsters(&mut rng);
                let mut deck = ironclad_starter_deck();
                for _ in 0..10 {
                    let id = *rng.pick(IRONCLAD_POOL).unwrap();
                    deck.push(Card::new(0, id, rng.next_int(3) == 0));
                }
                let mut c = Combat::new(&deck, IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, &specs, Ascension(10), seed);
                play_random(&mut c, &mut rng, 20_000);
                assert!(c.is_over(), "runaway fight in {enc:?} seed {seed}");
            }
        }
    }

    #[test]
    fn random_playouts_terminate() {
        let mut wins = 0;
        for seed in 0..200u64 {
            let mut c = nibbit_fight(seed);
            let mut rng = rng::Rng::new(seed);
            let mut steps = 0;
            while !c.is_over() {
                let acts = c.legal_actions();
                let a = acts[rng.next_int(acts.len())];
                c.step(a);
                steps += 1;
                assert!(steps < 10_000, "runaway fight");
            }
            if c.outcome == Some(Outcome::Won) {
                wins += 1;
            }
        }
        assert!(wins > 0, "a random Ironclad should beat a lone Nibbit sometimes");
    }
}
