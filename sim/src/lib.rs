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
pub mod game_rng;
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

    /// A replay forces opening moves and HP from the log, so only this pins
    /// what `PunchOffEventEncounter` sets up.
    #[test]
    fn punch_off_constructs_start_hurt_and_one_opens_on_fast_punch() {
        for seed in 0..20 {
            let specs = encounter::Encounter::PunchOffEventEncounter.monsters(&mut rng::Rng::new(seed));
            let c = fight(&specs, seed);
            let moves: Vec<_> = c.enemies.iter().map(|e| e.monster.next_move_name().unwrap()).collect();
            assert_eq!(moves, ["FAST_PUNCH_MOVE", "READY_MOVE"]);
            for e in &c.enemies {
                let down = e.creature.max_hp - e.creature.hp;
                assert!((2..=9).contains(&down), "seed {seed}: down {down}");
            }
        }
    }

    /// The dummy never acts and leaves at the end of the third enemy turn.
    #[test]
    fn battleworn_dummy_escapes_after_three_enemy_turns() {
        let mut c = fight(&[one(MonsterId::BattleFriendV3)], 2);
        assert_eq!(c.enemies[0].creature.power_amount(PowerId::BattlewornDummyTimeLimit), 3);
        c.step(Action::EndTurn);
        c.step(Action::EndTurn);
        assert!(!c.is_over());
        assert_eq!(c.player.creature.hp, IRONCLAD_HP);
        c.step(Action::EndTurn);
        assert!(c.enemies[0].escaped);
        assert_eq!(c.outcome, Some(combat::Outcome::Won));
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

    /// Colorless Potion offers three distinct colorless cards, never one
    /// only a teammate could use, and the pick is free this turn.
    #[test]
    fn colorless_potion_offers_three_colorless_cards() {
        let mut c = with_potions(&[Some(PotionId::ColorlessPotion), None], &[one(MonsterId::Nibbit)], 1);
        c.step(Action::UsePotion { slot: 0, target: None });
        let offer: Vec<_> = c.player.offer.iter().map(|k| k.id).collect();
        assert_eq!(offer.len(), 3);
        assert!(offer.iter().all(|id| card::COLORLESS_POOL.contains(id) && !card::MULTIPLAYER_ONLY.contains(id)));
        assert!(offer.iter().enumerate().all(|(i, id)| !offer[..i].contains(id)));
        c.step(Action::Choose(0));
        let taken = c.player.hand.last().unwrap();
        assert_eq!(taken.id, offer[0]);
        assert_eq!(c.cost(taken), 0);
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

    /// Put a Strike in hand and play it at enemy 0.
    fn strike_first_enemy(c: &mut Combat) {
        if let Some(j) = c.player.draw.iter().position(|k| k.id == ids::CardId::StrikeIronclad) {
            let card = c.player.draw.remove(j);
            c.player.hand.insert(0, card);
        }
        let i = c.player.hand.iter().position(|k| k.id == ids::CardId::StrikeIronclad).unwrap();
        c.step(Action::PlayCard { hand_idx: i, target: Some(0) });
    }

    /// Test Subject: a kill puts it into Respawn instead of ending the fight,
    /// and its next turn brings it back at its second form's max HP.
    #[test]
    fn test_subject_respawns_instead_of_dying() {
        let mut c = fight(&[one(MonsterId::TestSubject)], 1);
        c.enemies[0].creature.hp = 1;
        strike_first_enemy(&mut c);
        assert!(!c.is_over(), "Adaptable keeps the fight going");
        assert!(c.enemies[0].reviving);
        assert_eq!(c.enemies[0].monster.next_move_name(), Some("RESPAWN_MOVE"));
        c.step(Action::EndTurn);
        let second = Ascension(10).pick(types::AscensionLevel::ToughEnemies, 212, 200);
        assert_eq!((c.enemies[0].creature.hp, c.enemies[0].creature.max_hp), (second, second));
        assert!(c.enemies[0].creature.power(PowerId::PainfulStabs).is_some());
        assert!(c.enemies[0].creature.power(PowerId::Enrage).is_none(), "death strips what does not outlive it");
    }

    /// Axebot: its Stock sends a fresh one into the same slot, one stock
    /// down, opening on Boot Up.
    #[test]
    fn axebot_respawns_from_stock() {
        let mut c = fight(&[one(MonsterId::Axebot)], 1);
        c.enemies[0].creature.hp = 1;
        strike_first_enemy(&mut c);
        assert!(!c.is_over());
        let fresh = &c.enemies[1];
        assert_eq!((fresh.monster.id, fresh.slot), (MonsterId::Axebot, c.enemies[0].slot));
        assert_eq!(fresh.creature.power_amount(PowerId::Stock), 1);
        assert_eq!(fresh.monster.next_move_name(), Some("BOOT_UP_MOVE"));
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

    /// Put a fresh card at the front of the hand.
    fn to_hand(c: &mut Combat, id: ids::CardId, up: bool) {
        let uid = 10_000 + c.player.hand.len() as u32 + c.player.discard.len() as u32 * 16;
        c.player.hand.insert(0, card::Card::new(uid, id, up));
    }

    fn automation_counts(c: &Combat) -> Vec<i32> {
        c.player.creature.powers.iter().filter(|p| p.id == PowerId::Automation).map(|p| p.data).collect()
    }

    /// AutomationPower is Instanced: a second copy is a new instance with
    /// its own count of draws, so the two pay out on different draws.
    #[test]
    fn automation_instances_count_their_own_draws() {
        use ids::CardId::{Automation, MasterOfStrategy};
        let mut c = fight(&[one(MonsterId::Nibbit)], 5);
        to_hand(&mut c, Automation, false);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        to_hand(&mut c, MasterOfStrategy, false);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        to_hand(&mut c, Automation, false);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        assert_eq!(automation_counts(&c), vec![3, 0]);
        c.step(Action::EndTurn);
        assert_eq!(automation_counts(&c), vec![8, 5]);
        let energy = c.player.energy;
        to_hand(&mut c, MasterOfStrategy, false);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        // The older one reached ten on the second draw and started over.
        assert_eq!(automation_counts(&c), vec![1, 8]);
        assert_eq!(c.player.energy, energy + 1);
    }

    /// Entropy asks for exactly its amount of hand cards, never the same
    /// one twice, and puts a different card of the original's pool in each
    /// place.
    #[test]
    fn entropy_transforms_the_picked_cards_in_place() {
        let mut c = fight(&[one(MonsterId::Nibbit)], 7);
        c.player.creature.powers.push(Power::new(PowerId::Entropy, 2));
        c.step(Action::EndTurn);
        let before: Vec<card::Card> = c.player.hand.clone();
        let first = c.pending.as_ref().expect("Entropy asks").options[0];
        c.step(Action::Choose(0));
        assert!(!c.pending.as_ref().expect("second pick").options.contains(&first));
        c.step(Action::Choose(0));
        assert!(c.pending.is_none());
        let changed: Vec<usize> = (0..before.len()).filter(|&i| c.player.hand[i].uid != before[i].uid).collect();
        assert_eq!(changed.len(), 2);
        for i in changed {
            let (old, new) = (&before[i], &c.player.hand[i]);
            assert_ne!(old.id, new.id);
            assert!(card::transform_options(old.id).contains(&new.id));
        }
    }

    /// Bolas comes back to hand before the next turn's draw.
    #[test]
    fn bolas_returns_to_hand_the_turn_after_it_was_played() {
        let mut c = fight(&[one(MonsterId::Nibbit)], 3);
        to_hand(&mut c, ids::CardId::Bolas, false);
        c.step(Action::PlayCard { hand_idx: 0, target: Some(0) });
        assert!(c.player.discard.iter().any(|k| k.id == ids::CardId::Bolas));
        c.step(Action::EndTurn);
        assert!(c.player.hand.iter().any(|k| k.id == ids::CardId::Bolas));
        assert_eq!(c.player.hand.len(), 6);
    }

    /// Fisticuffs blocks for everything its hit dealt, the part the enemy's
    /// block soaked up included.
    #[test]
    fn fisticuffs_blocks_for_blocked_damage_too() {
        let mut c = fight(&[one(MonsterId::Nibbit)], 3);
        c.enemies[0].creature.block = 3;
        to_hand(&mut c, ids::CardId::Fisticuffs, false);
        c.step(Action::PlayCard { hand_idx: 0, target: Some(0) });
        assert_eq!(c.player.creature.block, 7);
    }

    /// Random decks from the colorless pool, random policy: catches panics
    /// and runaway loops in the colorless ports.
    #[test]
    fn random_colorless_decks_do_not_panic() {
        use crate::card::{Card, COLORLESS_POOL};
        for seed in 0..300u64 {
            let mut rng = rng::Rng::new(seed);
            let mut deck = ironclad_starter_deck();
            for _ in 0..8 {
                let id = *rng.pick(COLORLESS_POOL).unwrap();
                deck.push(Card::new(0, id, rng.next_int(3) == 0));
            }
            let mut c = Combat::with_setup(&Setup {
                deck: &deck,
                hp: IRONCLAD_HP,
                max_hp: IRONCLAD_HP,
                max_energy: IRONCLAD_ENERGY,
                relics: &[],
                potions: &[None, None],
                enemies: &[one(MonsterId::Nibbit), one(MonsterId::Nibbit)],
                room: RoomKind::Monster,
                asc: Ascension(10),
                seed,
                gold: 0,
            });
            let mut steps = 0;
            while !c.is_over() {
                let acts = c.legal_actions();
                assert!(!acts.is_empty(), "no legal actions at seed {seed}");
                c.step(acts[rng.next_int(acts.len())]);
                steps += 1;
                assert!(steps < 20_000, "runaway fight at seed {seed}");
                assert!(c.player.hand.len() <= combat::MAX_HAND);
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

    /// A Nibbit fight with `cards` as the whole hand and energy to spare.
    fn holding(cards: &[ids::CardId]) -> Combat {
        use crate::card::Card;
        let mut c = fight(&[one(MonsterId::Nibbit)], 5);
        c.player.energy = 99;
        c.player.hand = cards.iter().enumerate().map(|(i, &id)| Card::new(900 + i as u32, id, false)).collect();
        c
    }

    /// ReboundPower picks the result pile as a play begins, so the Rebound
    /// that grants it is discarded and the next card goes on the draw pile.
    #[test]
    fn rebound_moves_the_next_card_not_itself() {
        use ids::CardId::{DefendIronclad, Rebound};
        let mut c = holding(&[Rebound, DefendIronclad]);
        c.step(Action::PlayCard { hand_idx: 0, target: Some(0) });
        assert!(c.player.discard.iter().any(|k| k.id == Rebound));
        assert_eq!(c.player.creature.power_amount(PowerId::Rebound), 1);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        assert_eq!(c.player.draw[0].id, DefendIronclad);
        assert!(c.player.creature.power(PowerId::Rebound).is_none());
    }

    /// ToricToughnessPower is Instanced: each play counts down its own turns.
    #[test]
    fn toric_toughness_instances_count_down_apart() {
        use ids::CardId::ToricToughness;
        let mut c = holding(&[ToricToughness, ToricToughness]);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        let keep = c.player.hand.pop().unwrap();
        c.step(Action::EndTurn);
        assert_eq!(c.player.creature.block, 5);
        c.player.energy = 99;
        c.player.hand.push(keep);
        let i = c.player.hand.len() - 1;
        c.step(Action::PlayCard { hand_idx: i, target: None });
        c.step(Action::EndTurn);
        // Both fire; the first is spent, the second has a turn left.
        assert_eq!(c.player.creature.block, 10);
        let left: Vec<i32> =
            c.player.creature.powers.iter().filter(|p| p.id == PowerId::ToricToughness).map(|p| p.amount).collect();
        assert_eq!(left, vec![1]);
    }

    /// Enlightenment+ is a reduce-only cost for the combat: a card free this
    /// turn stays free, and costs 1 once the turn is over.
    #[test]
    fn enlightenment_only_ever_lowers_a_cost() {
        use ids::CardId::{Bludgeon, Enlightenment, StrikeIronclad};
        let mut c = holding(&[Enlightenment, Bludgeon, StrikeIronclad]);
        c.player.hand[0].upgraded = true;
        c.player.hand[1].cost_this_turn = Some(0);
        c.step(Action::PlayCard { hand_idx: 0, target: None });
        let costs: Vec<i32> = c.player.hand.iter().map(|k| c.cost(k)).collect();
        assert_eq!(costs, vec![0, 1]);
        for k in &mut c.player.hand {
            k.end_of_turn_cleanup();
        }
        let costs: Vec<i32> = c.player.hand.iter().map(|k| c.cost(k)).collect();
        assert_eq!(costs, vec![1, 1]);
    }

    /// A Nibbit fight with `cards` put into hand (uids 900 up) and energy to
    /// spare.
    fn with_hand(enemies: &[EnemySpec], cards: &[(ids::CardId, bool)]) -> Combat {
        let mut c = fight(enemies, 3);
        for (i, &(id, up)) in cards.iter().enumerate() {
            c.player.hand.push(card::Card::new(900 + i as u32, id, up));
        }
        c.player.energy = 10;
        c
    }

    fn hand_idx(c: &Combat, uid: u32) -> usize {
        c.player.hand.iter().position(|k| k.uid == uid).unwrap()
    }

    /// NostalgiaPower picks the result pile before the play starts, so the
    /// first attack or skill of the turn is the one that goes back on top.
    #[test]
    fn nostalgia_returns_only_the_first_attack_or_skill() {
        use ids::CardId::*;
        let mut c = with_hand(&[one(MonsterId::Nibbit)], &[(Nostalgia, false), (StrikeIronclad, false), (DefendIronclad, false)]);
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 900), target: None });
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 901), target: Some(0) });
        assert_eq!(c.player.draw[0].uid, 901);
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 902), target: None });
        assert!(c.player.discard.iter().any(|k| k.uid == 902));
    }

    /// Each Bomb is its own instance: they count down side by side and go
    /// off for their own damage.
    #[test]
    fn two_bombs_are_separate_instances() {
        use ids::CardId::*;
        let mut c = with_hand(&[one(MonsterId::Nibbit)], &[(TheBomb, true), (TheBomb, false)]);
        let e = &mut c.enemies[0].creature;
        (e.hp, e.max_hp) = (500, 500);
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 900), target: None });
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 901), target: None });
        let bombs = |c: &Combat| c.player.creature.powers.iter().filter(|p| p.id == PowerId::TheBomb).map(|p| p.amount).collect::<Vec<_>>();
        assert_eq!(bombs(&c), [3, 3]);
        c.step(Action::EndTurn);
        c.step(Action::EndTurn);
        assert_eq!(bombs(&c), [1, 1]);
        c.step(Action::EndTurn);
        assert!(bombs(&c).is_empty());
        assert_eq!(c.enemies[0].creature.hp, 500 - 50 - 40);
    }

    /// Purity picks every card first and exhausts them together once the
    /// selection closes.
    #[test]
    fn purity_exhausts_its_picks_when_the_selection_closes() {
        use ids::CardId::*;
        let mut c = with_hand(&[one(MonsterId::Nibbit)], &[(Purity, false), (Wound, false), (Dazed, false)]);
        let exhausted = c.player.exhaust.len();
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 900), target: None });
        let pick = |c: &Combat, uid: u32| c.pending.as_ref().unwrap().options.iter().position(|&u| u == uid).unwrap();
        c.step(Action::Choose(pick(&c, 901)));
        assert!(!c.pending.as_ref().unwrap().options.contains(&901));
        assert_eq!(c.player.exhaust.len(), exhausted);
        c.step(Action::Choose(pick(&c, 902)));
        c.step(Action::Skip);
        let gone: Vec<u32> = c.player.exhaust.iter().map(|k| k.uid).collect();
        assert!(gone.contains(&901) && gone.contains(&902) && gone.contains(&900));
    }

    /// Omnislice passes on block eaten and overkill too, not just HP lost.
    #[test]
    fn omnislice_splashes_everything_the_hit_dealt() {
        use ids::CardId::*;
        let mut c = with_hand(&[one(MonsterId::Nibbit), one(MonsterId::Nibbit)], &[(Omnislice, false)]);
        c.enemies[0].creature.block = 5;
        c.enemies[0].creature.hp = 1;
        c.enemies[1].creature.block = 0;
        let hp1 = c.enemies[1].creature.hp;
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 900), target: Some(0) });
        assert!(!c.enemies[0].creature.alive());
        assert_eq!(c.enemies[1].creature.hp, hp1 - 8);
    }

    /// Stratagem's pick after a shuffle comes before the draw that caused
    /// the shuffle takes its card.
    #[test]
    fn stratagem_picks_before_the_draw_that_shuffled() {
        use ids::CardId::*;
        let mut c = with_hand(&[one(MonsterId::Nibbit)], &[(ShrugItOff, false)]);
        c.player.creature.powers.push(Power::new(PowerId::Stratagem, 1));
        let draw = std::mem::take(&mut c.player.draw);
        c.player.discard.extend(draw);
        let held = c.player.hand.len();
        c.step(Action::PlayCard { hand_idx: hand_idx(&c, 900), target: None });
        assert_eq!(c.player.hand.len(), held - 1, "nothing drawn while the pick is open");
        let picked = c.pending.as_ref().unwrap().options[0];
        c.step(Action::Choose(0));
        assert_eq!(c.player.hand.len(), held + 1);
        assert!(c.player.hand.iter().any(|k| k.uid == picked));
    }

    /// The colorless cards the Ironclad meets outside its pool, dealt into
    /// random decks and played at random: catches panics and loops.
    #[test]
    fn colorless_decks_do_not_panic() {
        use crate::card::{Card, UNSUPPORTED_CARDS};
        use ids::CardId::*;
        // Every card from outside the Ironclad pool (the colorless pool
        // onward), plus a few that shuffle and pick.
        let first = ids::ALL_CARDS.iter().position(|&id| id == Alchemize).unwrap();
        let mut pool: Vec<ids::CardId> = ids::ALL_CARDS[first..]
            .iter()
            .copied()
            .filter(|id| !UNSUPPORTED_CARDS.iter().any(|(u, _)| u == id))
            .collect();
        pool.extend([Havoc, BurningPact, ShrugItOff]);
        for seed in 0..300u64 {
            let mut rng = rng::Rng::new(seed);
            let mut deck = ironclad_starter_deck();
            for _ in 0..10 {
                deck.push(Card::new(0, *rng.pick(&pool).unwrap(), rng.next_int(2) == 0));
            }
            let enemies = [one(MonsterId::Nibbit), one(MonsterId::Nibbit)];
            let mut c = Combat::new(&deck, IRONCLAD_HP, IRONCLAD_HP, IRONCLAD_ENERGY, &enemies, Ascension(10), seed);
            let mut steps = 0;
            while !c.is_over() {
                let acts = c.legal_actions();
                assert!(!acts.is_empty(), "no legal actions at seed {seed}");
                c.step(acts[rng.next_int(acts.len())]);
                steps += 1;
                assert!(steps < 20_000, "runaway fight at seed {seed}");
                assert!(c.player.hand.len() <= combat::MAX_HAND);
            }
        }
    }
}
