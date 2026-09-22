//! The fixed-size observation and action space the policy sees (DESIGN.md,
//! Decision engine). Every combat state encodes to `N_FLOATS` floats plus
//! `N_IDS` vocabulary ids for embeddings, and every legal `Action` maps to
//! one of `N_ACTIONS` indices with a mask over the rest.
//!
//! Hand and choice slots are shown sorted by (id, upgraded, cost), so the
//! policy sees a multiset, not the game's hand order. `decode` applies the
//! same sort to map a slot back to the real hand index.
//!
//! Enemy slots follow the game's order, but nothing the player does depends
//! on where an enemy stands, so the policy treats them as a set. Target 0
//! of every play and potion action is "no target"; the enemy slots follow,
//! so growing `MAX_ENEMIES` appends targets rather than moving them.

use crate::card::Card;
use crate::combat::{Action, Combat, RoomKind};
use crate::effect::{Pile, Then};
use crate::enchant::ALL as ALL_ENCHANTMENTS;
use crate::ids::{CardId, ALL_CARDS, ALL_MONSTERS, ALL_POWERS};
use crate::monster::{self, Intent};
use crate::potion;
use crate::relic;
use crate::types::CreatureRef;

pub const MAX_HAND: usize = crate::combat::MAX_HAND;
/// Phrog Parasite plus its four Wrigglers is the largest act 1 board.
pub const MAX_ENEMIES: usize = 6;
/// Two slots, four with Potion Belt.
pub const MAX_POTIONS: usize = 4;
/// Distinct (id, upgraded, cost) options a card choice can show.
pub const MAX_CHOICES: usize = 20;
/// "No target" first, then one per enemy slot.
pub const TARGETS: usize = MAX_ENEMIES + 1;

pub const N_CARDS: usize = ALL_CARDS.len();
pub const N_POWERS: usize = ALL_POWERS.len();
pub const N_MONSTERS: usize = ALL_MONSTERS.len();
pub const N_RELICS: usize = relic::ALL.len();
pub const N_POTIONS: usize = potion::ALL.len();
pub const N_ENCHANTMENTS: usize = ALL_ENCHANTMENTS.len();
pub const N_INTENTS: usize = monster::INTENT_KINDS.len();
/// Embedding vocabularies: 0 is the pad, ids are shifted by one.
pub const CARD_VOCAB: usize = N_CARDS + 1;
pub const MONSTER_VOCAB: usize = N_MONSTERS + 1;
pub const POTION_VOCAB: usize = N_POTIONS + 1;
pub const ENCHANT_VOCAB: usize = N_ENCHANTMENTS + 1;

// Action index layout.
pub const A_PLAY: usize = 0;
pub const A_POTION: usize = A_PLAY + MAX_HAND * TARGETS;
pub const A_END_TURN: usize = A_POTION + MAX_POTIONS * TARGETS;
pub const A_CHOOSE: usize = A_END_TURN + 1;
pub const A_SKIP: usize = A_CHOOSE + MAX_CHOICES;
pub const N_ACTIONS: usize = A_SKIP + 1;

// Float feature layout.
pub const F_GLOBAL: usize = 0;
/// Scalars, then a one-hot of the pending choice's kind (`THEN_KINDS`).
pub const GLOBAL_LEN: usize = 20 + THEN_KINDS;
const THEN_KINDS: usize = 10;
pub const F_PLAYER_POWERS: usize = F_GLOBAL + GLOBAL_LEN;
pub const F_HAND: usize = F_PLAYER_POWERS + N_POWERS;
/// Per hand slot: present, upgraded, cost, X cost, playable, exhausts
/// next, extra damage, enchantment amount, enchantment spent. The
/// enchantment itself is an id (`I_ENCHANTS`).
pub const HAND_FEATS: usize = 9;
pub const F_PILES: usize = F_HAND + MAX_HAND * HAND_FEATS;
/// Draw, discard, exhaust: counts per (card, upgraded).
pub const PILE_LEN: usize = N_CARDS * 2;
pub const F_ENEMIES: usize = F_PILES + 3 * PILE_LEN;
/// Per enemy slot: creature fields, a one-hot over intent kinds, the
/// intent numbers, then powers. The kinds are a vocabulary (`INTENT_KINDS`)
/// so a new one is an append, like a new power.
pub const ENEMY_BASE: usize = 7;
/// Damage, hits, total damage, strong debuff, status count.
pub const INTENT_NUMS: usize = 5;
pub const ENEMY_FEATS: usize = ENEMY_BASE + N_INTENTS + INTENT_NUMS + N_POWERS;
pub const F_RELICS: usize = F_ENEMIES + MAX_ENEMIES * ENEMY_FEATS;
pub const F_POTIONS: usize = F_RELICS + 2 * N_RELICS;
pub const F_CHOICES: usize = F_POTIONS + MAX_POTIONS;
pub const CHOICE_FEATS: usize = 3;
pub const N_FLOATS: usize = F_CHOICES + MAX_CHOICES * CHOICE_FEATS;

// Id layout.
pub const I_HAND: usize = 0;
pub const I_ENEMIES: usize = I_HAND + MAX_HAND;
pub const I_POTIONS: usize = I_ENEMIES + MAX_ENEMIES;
pub const I_CHOICES: usize = I_POTIONS + MAX_POTIONS;
/// Each enemy's next move, by `monster::all_move_names` index.
pub const I_MOVES: usize = I_CHOICES + MAX_CHOICES;
/// Each hand slot's enchantment, by `ALL_ENCHANTMENTS` index; 0 for none.
pub const I_ENCHANTS: usize = I_MOVES + MAX_ENEMIES;
pub const N_IDS: usize = I_ENCHANTS + MAX_HAND;

/// Move embedding vocabulary; 0 is the pad. Not a const: the names come
/// from the monster graphs.
pub fn move_vocab() -> usize {
    monster::all_move_names().len() + 1
}

/// What a pending choice does with the pick, as a small class index.
fn then_kind(then: Then) -> usize {
    match then {
        Then::Exhaust => 0,
        Then::Upgrade => 1,
        Then::MoveTo(Pile::Hand) => 2,
        Then::MoveTo(Pile::DrawTop | Pile::DrawBottom | Pile::DrawRandom) => 3,
        Then::MoveTo(Pile::Discard | Pile::Exhaust) => 4,
        Then::FreeThisCombat => 5,
        Then::ToHandFreeThisTurn => 6,
        Then::TakeOffer => 7,
        Then::ExhaustMany => 8,
        Then::DiscardThenDraw { .. } => 9,
    }
}

/// Sort key that makes hand and choice slots order-free.
/// Everything the observation shows about a card, so two hand positions
/// that look alike really are alike. Three copies of one card carrying
/// different enchantments sort apart, and the same way every turn.
/// `extra_damage` is never negative, so its bits order like the number.
fn card_key(c: &Combat, k: &Card) -> (usize, bool, i32, usize, i32, i32, bool, u64, bool) {
    let e = k.enchantment;
    (
        k.id as usize,
        k.upgraded,
        c.cost(k),
        e.map_or(0, |e| e.id as usize + 1),
        e.map_or(0, |e| e.amount),
        e.map_or(0, |e| e.data),
        e.is_some_and(|e| e.disabled),
        k.extra_damage.to_bits(),
        k.exhaust_on_next_play,
    )
}

/// Hand indices in slot order.
pub fn hand_order(c: &Combat) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..c.player.hand.len()).collect();
    idx.sort_by_key(|&i| card_key(c, &c.player.hand[i]));
    idx
}

/// A pending option's card: piles first, then the offer screen.
fn option_card(c: &Combat, uid: u32) -> Option<&Card> {
    c.find_card(uid).or_else(|| c.player.offer.iter().find(|k| k.uid == uid))
}

/// Distinct pending options in slot order, as (uid, card). Options past
/// `MAX_CHOICES` are unreachable; act 1 decks do not get there.
pub fn choice_order(c: &Combat) -> Vec<(u32, &Card)> {
    let Some(p) = &c.pending else { return vec![] };
    let mut opts: Vec<(u32, &Card)> = p.options.iter().filter_map(|&u| option_card(c, u).map(|k| (u, k))).collect();
    opts.sort_by_key(|(_, k)| card_key(c, k));
    opts.dedup_by_key(|(_, k)| card_key(c, k));
    opts.truncate(MAX_CHOICES);
    opts
}

/// Enemy indices by target slot: the game's slot order, dead ones included
/// so slots stay put when something dies. Corpses pile up past the slot count
/// in a long fight (Living Fog's bombs), and when they do they give up their
/// slots first, so every living enemy stays targetable.
fn enemy_slots(c: &Combat) -> Vec<usize> {
    if c.order.len() <= MAX_ENEMIES {
        return c.order.clone();
    }
    let mut slots: Vec<usize> = c.order.iter().copied().filter(|&i| c.enemies[i].acts()).collect();
    slots.extend(c.order.iter().copied().filter(|&i| !c.enemies[i].acts()));
    slots.truncate(MAX_ENEMIES);
    slots
}

fn target_slot(c: &Combat, target: Option<usize>) -> Option<usize> {
    match target {
        None => Some(0),
        Some(e) => enemy_slots(c).iter().position(|&i| i == e).map(|s| s + 1),
    }
}

/// Action index for a legal action, `None` if it does not fit the space.
pub fn index_of(c: &Combat, hand: &[usize], choices: &[(u32, &Card)], a: Action) -> Option<usize> {
    match a {
        Action::PlayCard { hand_idx, target } => {
            let slot = hand.iter().position(|&i| i == hand_idx)?;
            Some(A_PLAY + slot * TARGETS + target_slot(c, target)?)
        }
        Action::UsePotion { slot, target } => {
            (slot < MAX_POTIONS).then_some(())?;
            Some(A_POTION + slot * TARGETS + target_slot(c, target)?)
        }
        Action::EndTurn => Some(A_END_TURN),
        Action::Choose(i) => {
            let uid = c.pending.as_ref()?.options.get(i)?;
            // Duplicates collapse onto the first option with the same key.
            let key = card_key(c, option_card(c, *uid)?);
            let slot = choices.iter().position(|(_, k)| card_key(c, k) == key)?;
            Some(A_CHOOSE + slot)
        }
        Action::Skip => Some(A_SKIP),
    }
}

/// The `Action` behind an index, if it is legal right now.
pub fn decode(c: &Combat, index: usize) -> Option<Action> {
    let hand = hand_order(c);
    let choices = choice_order(c);
    c.legal_actions().into_iter().find(|&a| index_of(c, &hand, &choices, a) == Some(index))
}

/// Write the legal-action mask.
pub fn mask(c: &Combat, out: &mut [bool]) {
    out.fill(false);
    let hand = hand_order(c);
    let choices = choice_order(c);
    for a in c.legal_actions() {
        if let Some(i) = index_of(c, &hand, &choices, a) {
            out[i] = true;
        }
    }
}

fn powers_into(c: &Combat, r: CreatureRef, out: &mut [f32]) {
    for p in &c.creature(r).powers {
        out[p.id as usize] = (p.amount as f32 / 10.0).clamp(-10.0, 10.0);
    }
}

/// `out` is `N_INTENTS + INTENT_NUMS` long: the kind one-hot, then the numbers.
fn intent_into(intents: &[Intent], out: &mut [f32]) {
    let (kinds, nums) = out.split_at_mut(N_INTENTS);
    for i in intents {
        kinds[i.kind()] = 1.0;
        match *i {
            Intent::Attack { damage, hits } => {
                nums[0] = damage as f32 / 20.0;
                nums[1] = hits as f32 / 3.0;
                nums[2] = (damage * hits as i32) as f32 / 40.0;
            }
            Intent::Debuff { strong } => nums[3] = strong as u8 as f32,
            Intent::Status { count } => nums[4] = count as f32 / 3.0,
            // A blast that also kills the attacker; damage reads like an attack.
            Intent::DeathBlow { damage } => {
                nums[0] = damage as f32 / 20.0;
                nums[1] = 1.0 / 3.0;
                nums[2] = damage as f32 / 40.0;
            }
            _ => {}
        }
    }
}

/// Encode the state into `floats` (`N_FLOATS`), `ids` (`N_IDS`) and the
/// action `mask` (`N_ACTIONS`).
pub fn encode(c: &Combat, floats: &mut [f32], ids: &mut [i64], mask_out: &mut [bool]) {
    floats.fill(0.0);
    ids.fill(0);
    let p = &c.player;
    let living = c.living_enemies().count();
    let g = &mut floats[F_GLOBAL..F_GLOBAL + GLOBAL_LEN];
    g[0] = p.creature.hp as f32 / 100.0;
    g[1] = p.creature.max_hp as f32 / 100.0;
    g[2] = p.creature.hp as f32 / p.creature.max_hp.max(1) as f32;
    g[3] = p.creature.block as f32 / 50.0;
    g[4] = p.energy as f32 / 5.0;
    g[5] = c.max_energy() as f32 / 5.0;
    g[6] = p.turn as f32 / 10.0;
    g[7] = c.round as f32 / 10.0;
    g[8] = p.hand.len() as f32 / 10.0;
    g[9] = p.draw.len() as f32 / 30.0;
    g[10] = p.discard.len() as f32 / 30.0;
    g[11] = p.exhaust.len() as f32 / 30.0;
    g[12] = c.pending.is_some() as u8 as f32;
    g[13] = c.pending.as_ref().is_some_and(|q| q.can_skip) as u8 as f32;
    g[14 + match c.room {
        RoomKind::Monster => 0,
        RoomKind::Elite => 1,
        RoomKind::Boss => 2,
    }] = 1.0;
    g[17] = living as f32 / MAX_ENEMIES as f32;
    g[18] = c.potions.iter().flatten().count() as f32 / MAX_POTIONS as f32;
    g[19] = c.stats.cards_played_this_turn as f32 / 10.0;
    if let Some(p) = &c.pending {
        g[20 + then_kind(p.then)] = 1.0;
    }

    powers_into(c, CreatureRef::Player, &mut floats[F_PLAYER_POWERS..F_HAND]);

    for (slot, &i) in hand_order(c).iter().enumerate().take(MAX_HAND) {
        let k = &p.hand[i];
        let cost = c.cost(k);
        let f = &mut floats[F_HAND + slot * HAND_FEATS..][..HAND_FEATS];
        f[0] = 1.0;
        f[1] = k.upgraded as u8 as f32;
        f[2] = cost as f32 / 3.0;
        f[3] = k.def().x_cost as u8 as f32;
        f[4] = (cost >= 0 && cost <= p.energy) as u8 as f32;
        f[5] = k.exhaust_on_next_play as u8 as f32;
        // Flat extra damage the card carries: Rampage's growth and
        // Momentum's banked amount are the same thing to a policy.
        f[6] = (k.extra_damage as f32 + k.enchantment.map_or(0, |e| e.data) as f32) / 10.0;
        if let Some(e) = &k.enchantment {
            f[7] = e.amount as f32 / 3.0;
            f[8] = e.disabled as u8 as f32;
            ids[I_ENCHANTS + slot] = e.id as i64 + 1;
        }
        ids[I_HAND + slot] = k.id as i64 + 1;
    }

    for (n, pile) in [&p.draw, &p.discard, &p.exhaust].into_iter().enumerate() {
        let f = &mut floats[F_PILES + n * PILE_LEN..][..PILE_LEN];
        for k in pile {
            f[k.id as usize * 2 + k.upgraded as usize] += 1.0;
        }
    }

    for (slot, &i) in enemy_slots(c).iter().enumerate() {
        let e = &c.enemies[i];
        let f = &mut floats[F_ENEMIES + slot * ENEMY_FEATS..][..ENEMY_FEATS];
        f[0] = 1.0;
        f[1] = e.creature.alive() as u8 as f32;
        f[2] = e.creature.hp as f32 / 100.0;
        f[3] = e.creature.max_hp as f32 / 100.0;
        f[4] = e.creature.hp as f32 / e.creature.max_hp.max(1) as f32;
        f[5] = e.creature.block as f32 / 30.0;
        f[6] = e.reviving as u8 as f32;
        intent_into(e.monster.intents(), &mut f[ENEMY_BASE..ENEMY_BASE + N_INTENTS + INTENT_NUMS]);
        powers_into(c, CreatureRef::Enemy(i), &mut f[ENEMY_BASE + N_INTENTS + INTENT_NUMS..]);
        ids[I_ENEMIES + slot] = e.monster.id as i64 + 1;
        ids[I_MOVES + slot] = e.monster.next_move_name().and_then(monster::move_index).map_or(0, |m| m as i64 + 1);
    }

    for r in &c.relics {
        floats[F_RELICS + r.id as usize] = 1.0;
        floats[F_RELICS + N_RELICS + r.id as usize] = r.counter as f32 / 10.0;
    }

    for (slot, id) in c.potions.iter().enumerate().take(MAX_POTIONS) {
        if let Some(id) = id {
            floats[F_POTIONS + slot] = 1.0;
            ids[I_POTIONS + slot] = *id as i64 + 1;
        }
    }

    for (slot, (_, k)) in choice_order(c).iter().enumerate() {
        let f = &mut floats[F_CHOICES + slot * CHOICE_FEATS..][..CHOICE_FEATS];
        f[0] = 1.0;
        f[1] = k.upgraded as u8 as f32;
        f[2] = c.cost(k) as f32 / 3.0;
        ids[I_CHOICES + slot] = k.id as i64 + 1;
    }

    mask(c, mask_out);
}

/// Names for the Python side: card ids by vocabulary index.
pub fn card_name(index: usize) -> Option<String> {
    ALL_CARDS.get(index.checked_sub(1)?).map(|id: &CardId| format!("{id:?}"))
}

/// A `CamelCase` id as words, dropping the character suffix ids carry
/// (`StrikeIronclad` is just "Strike" to a player).
fn words(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out.strip_suffix(" Ironclad").map(str::to_string).unwrap_or(out)
}

/// An enemy as "Nibbit (left)": the name plus where it sits on screen,
/// which is how the player tells two of the same monster apart.
fn enemy_name(c: &Combat, enemy: usize) -> String {
    let present: Vec<usize> = c.present_enemies().collect();
    let name = words(&format!("{:?}", c.enemies[enemy].monster.id));
    let Some(slot) = present.iter().position(|&i| i == enemy) else { return name };
    let place = match (present.len(), slot) {
        (1, _) => return name,
        (_, 0) => "left".to_string(),
        (n, s) if s == n - 1 => "right".to_string(),
        (3, 1) => "middle".to_string(),
        (_, s) => format!("#{}", s + 1),
    };
    format!("{name} ({place})")
}

/// What a pending choice does with the card picked.
fn choice_verb(then: Then) -> &'static str {
    match then {
        Then::Exhaust | Then::ExhaustMany => "exhaust",
        Then::Upgrade => "upgrade",
        Then::MoveTo(Pile::Hand) | Then::ToHandFreeThisTurn => "take",
        Then::MoveTo(Pile::DrawTop) => "draw next",
        Then::MoveTo(Pile::DrawBottom) => "bury",
        Then::MoveTo(Pile::DrawRandom) => "shuffle in",
        Then::MoveTo(Pile::Discard) => "discard",
        Then::MoveTo(Pile::Exhaust) => "exhaust",
        Then::FreeThisCombat => "make free",
        Then::TakeOffer => "take",
        Then::DiscardThenDraw { .. } => "discard",
    }
}

/// Plain words for an action index, the way the advisor reads it out:
/// "Bash -> Nibbit (left)", "End turn", "Choose: exhaust Strike".
/// `None` when the index is not legal right now.
pub fn describe(c: &Combat, index: usize) -> Option<String> {
    let card = |k: &Card| {
        let name = words(&format!("{:?}", k.id));
        if k.upgraded {
            format!("{name}+")
        } else {
            name
        }
    };
    Some(match decode(c, index)? {
        Action::PlayCard { hand_idx, target } => {
            let name = card(&c.player.hand[hand_idx]);
            match target {
                Some(e) => format!("{name} -> {}", enemy_name(c, e)),
                None => name,
            }
        }
        Action::UsePotion { slot, target } => {
            let name = words(&format!("{:?}", c.potions.get(slot).copied().flatten()?));
            match target {
                Some(e) => format!("Use {name} -> {}", enemy_name(c, e)),
                None => format!("Use {name}"),
            }
        }
        Action::EndTurn => "End turn".to_string(),
        Action::Choose(i) => {
            let pending = c.pending.as_ref()?;
            let uid = *pending.options.get(i)?;
            format!("Choose: {} {}", choice_verb(pending.then), card(option_card(c, uid)?))
        }
        Action::Skip => "Skip".to_string(),
    })
}

/// A monster move name (`HISS_MOVE`) as words.
fn move_words(name: &str) -> String {
    let name = name.strip_suffix("_MOVE").unwrap_or(name);
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or(String::new(), |f| f.to_ascii_uppercase().to_string() + &c.as_str().to_ascii_lowercase())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Turn, energy, HP and the board, one line, for the advisor's header.
pub fn summary(c: &Combat) -> String {
    let p = &c.player;
    let block = |n: i32| if n > 0 { format!(" ({n} block)") } else { String::new() };
    let enemies: Vec<String> = c
        .present_enemies()
        .map(|i| {
            let e = &c.enemies[i];
            let intent = move_words(e.monster.next_move_name().unwrap_or("?"));
            format!("{} {}/{}{} {intent}", enemy_name(c, i), e.creature.hp, e.creature.max_hp, block(e.creature.block))
        })
        .collect();
    format!(
        "turn {} | {} energy | HP {}/{}{} | {}",
        p.turn,
        p.energy,
        p.creature.hp,
        p.creature.max_hp,
        block(p.creature.block),
        enemies.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::generate;
    use crate::rng::Rng;
    use crate::types::Ascension;

    /// The `as usize` casts index the ALL lists, so those must be in
    /// declaration order.
    #[test]
    fn vocab_lists_follow_declaration_order() {
        assert!(ALL_CARDS.iter().enumerate().all(|(i, &id)| id as usize == i));
        assert!(ALL_POWERS.iter().enumerate().all(|(i, &id)| id as usize == i));
        assert!(ALL_MONSTERS.iter().enumerate().all(|(i, &id)| id as usize == i));
        assert!(relic::ALL.iter().enumerate().all(|(i, &id)| id as usize == i));
        assert!(potion::ALL.iter().enumerate().all(|(i, &id)| id as usize == i));
    }

    /// Every legal action has exactly one index, the mask marks exactly the
    /// legal ones, and decoding the index gives back an equivalent action.
    #[test]
    fn mask_and_decode_agree_with_legal_actions() {
        let mut rng = Rng::new(7);
        let mut floats = vec![0.0; N_FLOATS];
        let mut ids = vec![0; N_IDS];
        let mut m = vec![false; N_ACTIONS];
        for i in 0..400u64 {
            let s = generate(&mut rng, 1 + (i % 16) as u32, Ascension(10));
            let mut c = s.combat(i);
            let mut steps = 0;
            while !c.is_over() && steps < 3000 {
                encode(&c, &mut floats, &mut ids, &mut m);
                let legal = c.legal_actions();
                let hand = hand_order(&c);
                let choices = choice_order(&c);
                let mut indices: Vec<usize> = legal.iter().filter_map(|&a| index_of(&c, &hand, &choices, a)).collect();
                indices.sort_unstable();
                indices.dedup();
                assert_eq!(m.iter().filter(|&&b| b).count(), indices.len());
                assert!(indices.iter().all(|&i| m[i]));
                // Only duplicate choice options may fail to get an index.
                let unindexed = legal.iter().filter(|&&a| index_of(&c, &hand, &choices, a).is_none()).count();
                assert!(unindexed == 0 || c.pending.is_some(), "unindexed action in {:?}", legal);
                let pick = indices[rng.next_int(indices.len())];
                let a = decode(&c, pick).expect("masked index decodes");
                for &e in &c.order {
                    if let Some(name) = c.enemies[e].monster.next_move_name() {
                        assert!(monster::move_index(name).is_some(), "move {name} missing from the vocabulary");
                    }
                }
                assert_eq!(index_of(&c, &hand, &choices, a), Some(pick));
                assert!(legal.contains(&a));
                c.step(a);
                steps += 1;
            }
        }
    }

    /// Three Strikes that differ only by their enchantment take three
    /// fixed positions, whatever order the game holds them in.
    #[test]
    fn copies_of_one_card_sort_by_their_enchantment() {
        use crate::enchant::EnchantmentId;
        let mut rng = Rng::new(3);
        let s = generate(&mut rng, 8, Ascension(10));
        let mut c = s.combat(1);
        c.player.hand.clear();
        for (uid, ench) in [(101, Some((EnchantmentId::Sharp, 3))), (102, None), (103, Some((EnchantmentId::Sharp, 1)))] {
            let mut k = Card::new(uid, CardId::StrikeIronclad, false);
            if let Some((id, amount)) = ench {
                k.enchant(id, amount);
            }
            c.player.hand.push(k);
        }
        let uids = |c: &Combat| hand_order(c).iter().map(|&i| c.player.hand[i].uid).collect::<Vec<_>>();
        assert_eq!(uids(&c), vec![102, 103, 101]);
        c.player.hand.reverse();
        assert_eq!(uids(&c), vec![102, 103, 101]);
    }

    #[test]
    fn sorted_hand_slots_map_back_to_hand_indices() {
        let mut rng = Rng::new(3);
        let s = generate(&mut rng, 8, Ascension(10));
        let c = s.combat(1);
        let order = hand_order(&c);
        let keys: Vec<_> = order.iter().map(|&i| card_key(&c, &c.player.hand[i])).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..c.player.hand.len()).collect::<Vec<_>>());
    }
}

/// Every vocabulary the policy embeds, in index order, one `kind name`
/// per line. `sim/vocab.txt` holds it and every checkpoint records it, so
/// a checkpoint from a smaller sim can be remapped by name (the Python
/// side does that). Regenerate with `cargo run --release --example vocab > vocab.txt`.
pub fn vocab_text() -> String {
    let mut out = String::new();
    for id in ALL_CARDS {
        out += &format!("card {id:?}\n");
    }
    for id in ALL_POWERS {
        out += &format!("power {id:?}\n");
    }
    for id in ALL_MONSTERS {
        out += &format!("monster {id:?}\n");
    }
    for id in relic::ALL {
        out += &format!("relic {id:?}\n");
    }
    for id in potion::ALL {
        out += &format!("potion {id:?}\n");
    }
    for name in monster::all_move_names() {
        out += &format!("move {name}\n");
    }
    for id in ALL_ENCHANTMENTS {
        out += &format!("enchant {id:?}\n");
    }
    for name in monster::INTENT_KINDS {
        out += &format!("intent {name}\n");
    }
    out
}

#[cfg(test)]
mod vocab_tests {
    /// `sim/vocab.txt` must describe the built sim: checkpoints record it
    /// and the Python loader remaps older checkpoints by name against it.
    #[test]
    fn vocabulary_file_matches_the_sim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/vocab.txt");
        let pinned = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            pinned == super::vocab_text(),
            "sim/vocab.txt is out of date; regenerate it with `cargo run --release --example vocab > vocab.txt` in sim/ and commit it"
        );
    }
}
