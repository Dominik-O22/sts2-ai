//! Slay the Spire 2 combat simulator, ported from the decompiled v0.107.1
//! assembly. See DESIGN.md and docs/survey/README.md for the decisions.

pub mod card;
pub mod combat;
pub mod effect;
pub mod enchant;
pub mod encode;
pub mod encounter;
pub mod env;
pub mod gen;
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

    /// A Nibbit fight whose deck holds one enchanted copy of `id`, in hand.
    fn enchanted(id: ids::CardId, ench: enchant::EnchantmentId, amount: i32, seed: u64) -> Combat {
        use crate::card::Card;
        let mut deck = ironclad_starter_deck();
        let mut k = Card::new(0, id, false);
        k.enchant(ench, amount);
        deck.push(k);
        let mut c = Combat::new(&deck, IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, &[one(MonsterId::Nibbit)], Ascension(10), seed);
        if let Some(i) = c.player.draw.iter().position(|k| k.enchantment.is_some()) {
            let card = c.player.draw.remove(i);
            c.player.hand.insert(0, card);
        }
        c
    }

    /// The enchanted card's slot in hand.
    fn ench_idx(c: &Combat) -> usize {
        c.player.hand.iter().position(|k| k.enchantment.is_some()).unwrap()
    }

    /// Anger.cs clones the card model, so the copy carries the enchantment.
    /// Seen in a run: a Sharp Anger whose copy the sim left plain.
    #[test]
    fn anger_clone_keeps_the_enchantment() {
        let mut c = enchanted(ids::CardId::Anger, enchant::EnchantmentId::Sharp, 2, 5);
        let idx = ench_idx(&c);
        c.step(Action::PlayCard { hand_idx: idx, target: Some(0) });
        let angers: Vec<_> = c.player.discard.iter().filter(|k| k.id == ids::CardId::Anger).collect();
        assert_eq!(angers.len(), 2);
        assert!(angers.iter().all(|k| k.enchantment.is_some_and(|e| e.id == enchant::EnchantmentId::Sharp && e.amount == 2)));
        assert_ne!(angers[0].uid, angers[1].uid);
    }

    /// Shame is the curse a relic handed Dom, and the one whose timing is
    /// easy to get wrong: the Frail it lands must survive the turn end it
    /// was applied on.
    #[test]
    fn shame_lands_frail_that_does_not_tick_the_turn_it_arrives() {
        use crate::card::Card;
        let mut c = fight(&[one(MonsterId::Nibbit)], 5);
        c.player.hand.push(Card::new(900, ids::CardId::Shame, false));
        let shame = c.player.hand.len() - 1;
        assert!(!c.legal_actions().contains(&Action::PlayCard { hand_idx: shame, target: None }));
        c.step(Action::EndTurn);
        assert_eq!(c.player.creature.power_amount(PowerId::Frail), 1);
        // A second turn holding it stacks, and the first point now ticks.
        c.player.hand.push(Card::new(901, ids::CardId::Shame, false));
        c.step(Action::EndTurn);
        assert_eq!(c.player.creature.power_amount(PowerId::Frail), 1);
    }

    /// Normality vetoes from hand, so it costs a play without ever being one.
    #[test]
    fn normality_stops_the_fourth_card_each_turn() {
        use crate::card::Card;
        let mut c = fight(&[one(MonsterId::Nibbit)], 5);
        c.player.energy = 99;
        c.player.hand.clear();
        for uid in 0..5 {
            c.player.hand.push(Card::new(910 + uid, ids::CardId::DefendIronclad, false));
        }
        c.player.hand.push(Card::new(920, ids::CardId::Normality, false));
        for _ in 0..3 {
            c.step(Action::PlayCard { hand_idx: 0, target: None });
        }
        assert_eq!(c.stats.cards_played_this_turn, 3);
        assert!(!c.legal_actions().iter().any(|a| matches!(a, Action::PlayCard { .. })));
    }

    #[test]
    fn sharp_lands_before_vulnerable_multiplies() {
        let mut c = enchanted(ids::CardId::StrikeIronclad, enchant::EnchantmentId::Sharp, 3, 5);
        c.enemies[0].creature.powers.push(Power::new(PowerId::Vulnerable, 2));
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: ench_idx(&c), target: Some(0) });
        // Hook.ModifyDamage asks the enchantment first: (6 + 3) * 1.5, not
        // 6 * 1.5 + 3, which would be 12.
        assert_eq!(c.enemies[0].creature.hp, hp - 13);
    }

    #[test]
    fn momentum_banks_its_damage_so_the_first_hit_is_plain() {
        let mut c = enchanted(ids::CardId::StrikeIronclad, enchant::EnchantmentId::Momentum, 4, 5);
        // Duplication buys the second play the banked damage shows up on.
        c.player.creature.powers.push(Power::new(PowerId::Duplication, 1));
        let hp = c.enemies[0].creature.hp;
        c.step(Action::PlayCard { hand_idx: ench_idx(&c), target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, hp - (6 + 6 + 4));
    }

    #[test]
    fn sown_pays_its_energy_once_and_goes_quiet() {
        let mut c = enchanted(ids::CardId::DefendIronclad, enchant::EnchantmentId::Sown, 2, 5);
        c.player.creature.powers.push(Power::new(PowerId::Duplication, 1));
        let energy = c.player.energy;
        c.step(Action::PlayCard { hand_idx: ench_idx(&c), target: None });
        // One energy for the Defend, two back, and nothing for the replay.
        assert_eq!(c.player.energy, energy - 1 + 2);
        assert!(c.player.discard.iter().any(|k| k.enchantment.is_some_and(|e| e.disabled)));
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

    /// Killing the giant only arms the blast; the fight ends when it lands.
    #[test]
    fn waterfall_giant_explodes_for_its_banked_pressure_instead_of_dying() {
        let mut c = fight(&[one(MonsterId::WaterfallGiant)], 3);
        // Turn 1 is Pressurize, worth 20 at A10.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::SteamEruption), 20);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        // Dead on paper, but Steam Eruption holds the combat open and the
        // giant comes back with the wind-up queued.
        assert!(!c.is_over());
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("ABOUT_TO_BLOW_MOVE"));
        c.step(Action::EndTurn);
        // The wind-up banks the pressure into the blast and drops the power.
        assert!(c.enemies[0].creature.power(PowerId::SteamEruption).is_none());
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("EXPLODE_MOVE"));
        let hp = c.player.creature.hp;
        c.step(Action::EndTurn);
        assert_eq!(c.player.creature.hp, hp - 20);
        assert_eq!(c.outcome, Some(Outcome::Won));
    }

    /// The matriarch naps behind Plating; one hit through it wakes her early.
    #[test]
    fn lagavulin_sleeps_behind_plating_until_a_hit_lands() {
        let mut c = fight(&[one(MonsterId::LagavulinMatriarch)], 4);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Asleep), 3);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("SLEEP_MOVE"));
        // Plating blocks the first turn's chip damage outright.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::Asleep), 2);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("SLEEP_MOVE"));
        // A hit big enough to get through the shell wakes her into Slash.
        c.enemies[0].creature.block = 0;
        c.player.creature.powers.push(Power::new(PowerId::Strength, 20));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert!(c.enemies[0].creature.power(PowerId::Asleep).is_none());
        assert!(c.enemies[0].creature.power(PowerId::Plating).is_none());
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("STUNNED"));
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("SLASH_MOVE"));
    }

    /// Living Fog's smog settles on every skill once you play one.
    #[test]
    fn smog_blocks_the_rest_of_the_turns_skills_and_lifts_after_it() {
        let mut c = fight(&[one(MonsterId::LivingFog)], 6);
        c.player.creature.powers.push(Power::new(PowerId::Smoggy, 1));
        let skill = |c: &Combat| c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Skill);
        let i = skill(&c).expect("a skill in the opening hand");
        c.step(Action::PlayCard { hand_idx: i, target: None });
        assert!(c.player.hand.iter().filter(|k| k.ty() == crate::types::CardType::Skill).all(|k| k.smogged));
        assert!(!c.legal_actions().iter().any(|a| matches!(a, Action::PlayCard { hand_idx, .. }
            if c.player.hand[*hand_idx].ty() == crate::types::CardType::Skill)));
        // Attacks are untouched, and the fog is gone next turn.
        c.step(Action::EndTurn);
        assert!(c.player.hand.iter().all(|k| !k.smogged));
    }

    /// The merc's death is a hand-off: a sneaky gremlin, and a fat one that
    /// runs off with the loot.
    #[test]
    fn gremlin_merc_death_brings_two_gremlins_and_the_fat_one_flees() {
        let mut c = fight(&[one(MonsterId::GremlinMerc)], 7);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert!(!c.is_over());
        assert_eq!(c.enemies.len(), 3);
        assert_eq!(c.enemies[1].monster.id, MonsterId::SneakyGremlin);
        assert_eq!(c.enemies[2].monster.id, MonsterId::FatGremlin);
        // Both idle the turn they arrive, then the fat one leaves.
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[2].monster.next_move_name(), Some("FLEE_MOVE"));
        c.step(Action::EndTurn);
        assert!(c.enemies[2].escaped);
        // An escape is not a kill: the sneaky gremlin still has to go.
        assert!(!c.is_over());
    }

    /// Hardened Shell is a per-turn HP budget, not a per-hit cap.
    #[test]
    fn hardened_shell_caps_hp_lost_per_turn_and_refills_next_turn() {
        let mut c = fight(&[one(MonsterId::SkulkingColony)], 8);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::HardenedShell), 20);
        let start = c.enemies[0].creature.hp;
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let attacks: Vec<usize> =
            (0..c.player.hand.len()).filter(|&i| c.player.hand[i].ty() == crate::types::CardType::Attack).collect();
        assert!(attacks.len() >= 2, "need two attacks to spend the budget twice");
        c.step(Action::PlayCard { hand_idx: attacks[0], target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, start - 20);
        // The second attack this turn finds the budget already spent.
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, start - 20);
        c.step(Action::EndTurn);
        let after = c.enemies[0].creature.hp;
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert_eq!(c.enemies[0].creature.hp, after - 20);
    }

    /// A slug that watches its neighbour die gorges: Strength, and no move.
    #[test]
    fn corpse_slug_goes_ravenous_when_its_neighbour_dies() {
        let mut c = fight(&[one(MonsterId::CorpseSlug), one(MonsterId::CorpseSlug)], 11);
        assert_eq!(c.enemies[1].creature.power_amount(PowerId::Ravenous), 5);
        c.player.creature.powers.push(Power::new(PowerId::Strength, 500));
        let i = c.player.hand.iter().position(|k| k.ty() == crate::types::CardType::Attack).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
        assert!(!c.enemies[0].creature.alive());
        assert_eq!(c.enemies[1].creature.power_amount(PowerId::Strength), 5);
        assert_eq!(c.enemies[1].monster.next_move_name(), Some("STUNNED"));
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
        // Its 13 starting block never lands in the real game (recorded); see spawn().
        assert_eq!(c.enemies[0].creature.block, 0);
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
            gold: 0,
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
            gold: 0,
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
            gold: 0,
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

    /// Decimillipede: a segment killed while the others live stays in the
    /// fight, plays dead for a turn, then reattaches with 25 HP.
    #[test]
    fn decimillipede_segment_reattaches() {
        let specs = encounter::Encounter::DecimillipedeElite.monsters(&mut rng::Rng::new(1));
        let mut c = with_relics(&[], &specs, 1);
        c.enemies[0].creature.hp = 1;
        c.player.hand.insert(0, card::Card::new(900, ids::CardId::StrikeIronclad, false));
        c.player.energy = 1;
        c.step(Action::PlayCard { hand_idx: 0, target: Some(0) });
        assert!(!c.is_over(), "the other segments still live");
        assert!(c.enemies[0].reviving && c.living_enemies().count() == 2);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("DEAD_MOVE"));
        c.player.creature.hp = 999;
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("REATTACH_MOVE"));
        assert_eq!(c.enemies[0].creature.hp, 0);
        c.step(Action::EndTurn);
        assert_eq!(c.enemies[0].creature.hp, 25);
        assert!(!c.enemies[0].reviving);
    }

    /// Pael's Eye: a turn with nothing played burns the hand and comes round
    /// again before the enemy moves, once a combat.
    #[test]
    fn paels_eye_takes_one_extra_turn() {
        let alone = EnemySpec { id: MonsterId::Nibbit, flags: Flags { is_alone: true, ..Default::default() } };
        let mut c = with_relics(&[Relic::new(RelicId::PaelsEye)], &[alone], 3);
        let intent = c.enemies[0].monster.next_move_name();
        c.step(Action::EndTurn);
        assert_eq!((c.side, c.player.turn, c.round), (types::Side::Player, 2, 1));
        assert_eq!(c.player.exhaust.len(), 5, "the unplayed hand is exhausted");
        assert_eq!(c.player.creature.hp, IRONCLAD_HP, "the enemy has not acted");
        assert_eq!(c.enemies[0].monster.next_move_name(), intent, "and still owes the same move");
        c.step(Action::EndTurn);
        assert_eq!((c.player.turn, c.round), (3, 2), "spent: the second idle turn passes to the enemy");
    }

    /// Whispering Earring plays the opening hand left to right until it runs
    /// out of energy, before the first decision.
    #[test]
    fn whispering_earring_plays_turn_one() {
        let c = with_relics(&[Relic::new(RelicId::WhisperingEarring)], &[one(MonsterId::Nibbit)], 1);
        assert_eq!(c.stats.manual_plays_this_turn, 0);
        assert!(c.stats.cards_played_this_turn > 0);
        assert!(c.player.hand.iter().all(|k| c.cost(k) > c.player.energy), "nothing playable is left");
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
                    gold: 0,
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

    /// Pillage keeps drawing while it draws attacks, Hellraiser auto-plays
    /// every Strike drawn, and with enough negative Strength the Strikes
    /// deal nothing: the game loops forever here. The sim scores it lost.
    #[test]
    fn unresolvable_effect_loop_is_a_loss_not_a_panic() {
        let mut deck: Vec<card::Card> = (0..8).map(|_| card::Card::new(0, ids::CardId::StrikeIronclad, false)).collect();
        deck.push(card::Card::new(0, ids::CardId::Pillage, false));
        let mut c = Combat::new(&deck, IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, &[one(MonsterId::Mawler)], Ascension(10), 1);
        c.player.creature.powers.push(Power::new(PowerId::Hellraiser, 1));
        c.player.creature.powers.push(Power::new(PowerId::Strength, -20));
        let pillage = match c.player.hand.iter().position(|k| k.id == ids::CardId::Pillage) {
            Some(i) => i,
            None => {
                let j = c.player.draw.iter().position(|k| k.id == ids::CardId::Pillage).unwrap();
                let card = c.player.draw.remove(j);
                c.player.hand.insert(0, card);
                0
            }
        };
        c.step(Action::PlayCard { hand_idx: pillage, target: Some(0) });
        assert_eq!(c.outcome, Some(Outcome::Lost));
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
