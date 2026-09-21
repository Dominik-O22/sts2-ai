//! Slay the Spire 2 combat simulator, ported from the decompiled v0.107.1
//! assembly. See DESIGN.md and docs/survey/README.md for the decisions.

pub mod card;
pub mod combat;
pub mod effect;
pub mod ids;
pub mod monster;
pub mod power;
pub mod rng;
pub mod types;

pub use combat::{Action, Combat, EnemySpec, Outcome};
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
        c.player.creature.powers.push(Power { id: PowerId::Strength, amount: 3, skip_next_tick: false });
        c.player.creature.powers.push(Power { id: PowerId::Weak, amount: 1, skip_next_tick: true });
        c.enemies[0].creature.powers.push(Power { id: PowerId::Vulnerable, amount: 1, skip_next_tick: false });
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
