//! Combat state and the loop that resolves effects. `Combat/CombatState.cs`,
//! `Combat/CombatManager.cs`, `Commands/CreatureCmd.cs`, `Commands/CardPileCmd.cs`.

use std::collections::VecDeque;

use crate::card::Card;
use crate::effect::{AttackTargets, Effect};
use crate::ids::{MonsterId, PowerId};
use crate::monster::{Flags, Monster};
use crate::power::{is_debuff, Power};
use crate::rng::CombatRngs;
use crate::types::{Ascension, CardType, CreatureRef, Keyword, Side, TargetType};
use crate::types::ValueProp;

pub const MAX_HAND: usize = 10;
pub const BASE_HAND_DRAW: u32 = 5;
/// `decimal` clamp used at every HP/block write.
const CLAMP: f64 = 999_999_999.0;

/// `Entities/Creatures/Creature.cs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Creature {
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    /// Insertion ordered. Hook dispatch iterates in this order.
    pub powers: Vec<Power>,
}

impl Creature {
    pub fn alive(&self) -> bool {
        self.hp > 0
    }
    pub fn power(&self, id: PowerId) -> Option<&Power> {
        self.powers.iter().find(|p| p.id == id)
    }
    pub fn power_amount(&self, id: PowerId) -> i32 {
        self.power(id).map_or(0, |p| p.amount)
    }
}

/// `Entities/Players/PlayerCombatState.cs` plus the player's `Creature`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerCombat {
    pub creature: Creature,
    pub hand: Vec<Card>,
    /// Index 0 is the top of the draw pile.
    pub draw: Vec<Card>,
    pub discard: Vec<Card>,
    pub exhaust: Vec<Card>,
    /// `PileType.Play`: limbo for the card currently resolving.
    pub play: Vec<Card>,
    pub energy: i32,
    pub max_energy: i32,
    /// `PlayerCombatState.TurnNumber`, starts at 1.
    pub turn: u32,
}

#[derive(Clone, Debug)]
pub struct Enemy {
    pub creature: Creature,
    pub monster: Monster,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Won,
    Lost,
}

/// Player inputs. `GameActions/PlayCardAction.cs`, `EndPlayerTurnAction.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    PlayCard { hand_idx: usize, target: Option<usize> },
    EndTurn,
}

/// A monster to place in the encounter.
#[derive(Clone, Copy, Debug)]
pub struct EnemySpec {
    pub id: MonsterId,
    pub flags: Flags,
}

#[derive(Clone, Debug)]
pub struct Combat {
    pub player: PlayerCombat,
    pub enemies: Vec<Enemy>,
    /// `CombatState.RoundNumber`, starts at 1.
    pub round: u32,
    pub side: Side,
    pub asc: Ascension,
    pub rngs: CombatRngs,
    pub outcome: Option<Outcome>,
    queue: VecDeque<Effect>,
    next_uid: u32,
}

impl Combat {
    /// `CombatManager.SetUpCombat` + `StartCombatInternal`. Deck cards are
    /// cloned into the draw pile and shuffled; monsters get their HP roll and
    /// innate powers; then the first player turn starts.
    pub fn new(
        deck: &[Card],
        hp: i32,
        max_hp: i32,
        max_energy: i32,
        enemies: &[EnemySpec],
        asc: Ascension,
        seed: u64,
    ) -> Self {
        let mut rngs = CombatRngs::new(seed);
        let mut next_uid = 1;
        let mut draw: Vec<Card> = deck
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c.uid = next_uid;
                next_uid += 1;
                c
            })
            .collect();
        rngs.shuffle.shuffle(&mut draw);

        let mut placed: Vec<Enemy> = Vec::with_capacity(enemies.len());
        for spec in enemies {
            let (lo, hi) = Monster::hp_range(spec.id, asc);
            let hp = roll_unique_hp(lo, hi, &placed, &mut rngs.niche);
            let mut creature = Creature { hp, max_hp: hp, block: 0, powers: vec![] };
            for (id, amount) in Monster::innate_powers(spec.id, asc) {
                creature.powers.push(Power { id, amount, skip_next_tick: false });
            }
            placed.push(Enemy { creature, monster: Monster::new(spec.id, asc, spec.flags) });
        }

        let mut c = Self {
            player: PlayerCombat {
                creature: Creature { hp, max_hp, block: 0, powers: vec![] },
                hand: vec![],
                draw,
                discard: vec![],
                exhaust: vec![],
                play: vec![],
                energy: 0,
                max_energy,
                turn: 1,
            },
            enemies: placed,
            round: 1,
            side: Side::Player,
            asc,
            rngs,
            outcome: None,
            queue: VecDeque::new(),
            next_uid,
        };
        c.queue.push_back(Effect::StartTurn(Side::Player));
        c.run();
        c
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn creature(&self, r: CreatureRef) -> &Creature {
        match r {
            CreatureRef::Player => &self.player.creature,
            CreatureRef::Enemy(i) => &self.enemies[i].creature,
        }
    }

    fn creature_mut(&mut self, r: CreatureRef) -> &mut Creature {
        match r {
            CreatureRef::Player => &mut self.player.creature,
            CreatureRef::Enemy(i) => &mut self.enemies[i].creature,
        }
    }

    pub fn living_enemies(&self) -> impl Iterator<Item = usize> + '_ {
        self.enemies.iter().enumerate().filter(|(_, e)| e.creature.alive()).map(|(i, _)| i)
    }

    /// Every action the player may take right now.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = vec![];
        if self.is_over() || self.side != Side::Player {
            return out;
        }
        for (i, card) in self.player.hand.iter().enumerate() {
            if !self.can_play(card) {
                continue;
            }
            match card.def().target {
                TargetType::AnyEnemy => {
                    for e in self.living_enemies() {
                        out.push(Action::PlayCard { hand_idx: i, target: Some(e) });
                    }
                }
                _ => out.push(Action::PlayCard { hand_idx: i, target: None }),
            }
        }
        out.push(Action::EndTurn);
        out
    }

    /// `CardModel.CanPlay` for the reasons we model.
    fn can_play(&self, card: &Card) -> bool {
        if card.has(Keyword::Unplayable) || card.def().cost < 0 {
            return false;
        }
        self.player.energy >= card.cost()
    }

    /// Apply one player action and resolve everything it triggers.
    pub fn step(&mut self, action: Action) {
        assert!(!self.is_over(), "step after combat ended");
        assert_eq!(self.side, Side::Player, "not the player's turn");
        match action {
            Action::PlayCard { hand_idx, target } => {
                let card = self.player.hand[hand_idx].clone();
                assert!(self.can_play(&card), "illegal card play");
                let target = match card.def().target {
                    TargetType::AnyEnemy => {
                        let t = target.expect("target required");
                        assert!(self.enemies[t].creature.alive(), "dead target");
                        Some(CreatureRef::Enemy(t))
                    }
                    _ => None,
                };
                // PlayCardAction: spend energy, then run the play pipeline.
                self.player.energy -= card.cost();
                let card = self.player.hand.remove(hand_idx);
                self.player.play.push(card.clone());
                self.queue.push_back(Effect::PlayCard { uid: card.uid, target });
            }
            Action::EndTurn => self.queue.push_back(Effect::EndPlayerTurn),
        }
        self.run();
    }

    /// Drain the queue. Stops when combat ends.
    fn run(&mut self) {
        while let Some(e) = self.queue.pop_front() {
            if self.is_over() {
                self.queue.clear();
                break;
            }
            self.resolve(e);
        }
    }

    /// Push sub-effects to the front, preserving their order.
    fn push_front_all(&mut self, effects: Vec<Effect>) {
        for e in effects.into_iter().rev() {
            self.queue.push_front(e);
        }
    }

    fn resolve(&mut self, e: Effect) {
        match e {
            Effect::Attack { dealer, base, hits, targets, props, card } => {
                // AttackCommand.Execute: per hit, recompute living targets.
                let mut subs = Vec::new();
                for _ in 0..hits {
                    let ts: Vec<CreatureRef> = match &targets {
                        AttackTargets::One(t) => vec![*t],
                        AttackTargets::AllOpponents => self.opponents_of(dealer),
                        AttackTargets::RandomOpponent => {
                            let opts = self.opponents_of(dealer);
                            if opts.is_empty() {
                                vec![]
                            } else {
                                vec![opts[self.rngs.targets.next_int(opts.len())]]
                            }
                        }
                    };
                    for t in ts {
                        subs.push(Effect::Damage { target: t, amount: base, props, dealer: Some(dealer) });
                    }
                }
                let _ = card;
                self.push_front_all(subs);
            }
            Effect::Damage { target, amount, props, dealer } => {
                if !self.creature(target).alive() {
                    return;
                }
                self.damage(target, amount, props, dealer);
                self.check_win();
            }
            Effect::GainBlock { target, amount, props, card: _ } => {
                self.gain_block(target, amount, props, CreatureRef::Player);
            }
            Effect::ApplyPower { target, id, amount, applier } => {
                self.apply_power(target, id, amount, applier);
            }
            Effect::TickDuration { target, id } => {
                let c = self.creature_mut(target);
                if let Some(p) = c.powers.iter_mut().find(|p| p.id == id) {
                    if p.skip_next_tick {
                        p.skip_next_tick = false;
                    } else {
                        p.amount -= 1;
                        if p.should_remove() {
                            c.powers.retain(|p| p.id != id);
                        }
                    }
                }
            }
            Effect::Draw { count } => self.draw(count),
            Effect::Exhaust { uid } => {
                if let Some(card) = self.take_card(uid) {
                    self.player.exhaust.push(card);
                }
            }
            Effect::PlayCard { uid, target } => {
                let card = self.player.play.iter().find(|c| c.uid == uid).cloned().expect("card in play");
                let mut subs = card.on_play(target);
                subs.push(Effect::FinishCardPlay { uid });
                self.push_front_all(subs);
            }
            Effect::FinishCardPlay { uid } => {
                // CardModel.GetResultPileTypeForCardPlay.
                if let Some(card) = self.take_card(uid) {
                    if card.def().ty == CardType::Power {
                        // Powers leave combat (PileType.None).
                    } else if card.has(Keyword::Exhaust) {
                        self.player.exhaust.push(card);
                    } else {
                        self.player.discard.push(card);
                    }
                }
            }
            Effect::StartTurn(side) => self.start_turn(side),
            Effect::EndPlayerTurn => self.end_player_turn(),
            Effect::EnemyAct(i) => {
                if self.enemies[i].creature.alive() {
                    let subs = self.enemies[i].monster.perform(CreatureRef::Enemy(i), self.asc);
                    self.push_front_all(subs);
                }
            }
            Effect::EndEnemyTurn => self.end_enemy_turn(),
        }
    }

    fn opponents_of(&self, dealer: CreatureRef) -> Vec<CreatureRef> {
        match dealer.side() {
            Side::Player => self.living_enemies().map(CreatureRef::Enemy).collect(),
            Side::Enemy => {
                if self.player.creature.alive() {
                    vec![CreatureRef::Player]
                } else {
                    vec![]
                }
            }
        }
    }

    // ---- turn flow -------------------------------------------------------

    /// `CombatManager.StartTurn`.
    fn start_turn(&mut self, side: Side) {
        self.side = side;
        match side {
            Side::Player => {
                // Enemies roll their next move (PrepareForNextTurn).
                for i in 0..self.enemies.len() {
                    if self.enemies[i].creature.alive() {
                        self.enemies[i].monster.roll_move(&mut self.rngs.monster_ai);
                    }
                }
                // Creature.AfterTurnStart: block clears except on player turn 1.
                if self.player.turn != 1 {
                    self.player.creature.block = 0;
                }
                // SetupPlayerTurn: energy, then draw.
                self.player.energy = self.player.max_energy;
                self.queue.push_front(Effect::Draw { count: BASE_HAND_DRAW });
            }
            Side::Enemy => {
                for e in &mut self.enemies {
                    if e.creature.alive() {
                        e.creature.block = 0;
                    }
                }
                let mut subs: Vec<Effect> = (0..self.enemies.len()).map(Effect::EnemyAct).collect();
                subs.push(Effect::EndEnemyTurn);
                self.push_front_all(subs);
            }
        }
    }

    /// `EndPlayerTurnPhaseOneInternal` + `PhaseTwoInternal` + `SwitchSides`.
    fn end_player_turn(&mut self) {
        // Ethereal cards exhaust first.
        let ethereal: Vec<u32> =
            self.player.hand.iter().filter(|c| c.has(Keyword::Ethereal)).map(|c| c.uid).collect();
        for uid in ethereal {
            if let Some(card) = self.take_card(uid) {
                self.player.exhaust.push(card);
            }
        }
        // FlushPlayerHand: discard everything not retained.
        let (retain, flush): (Vec<Card>, Vec<Card>) =
            self.player.hand.drain(..).partition(|c| c.has(Keyword::Retain));
        self.player.hand = retain;
        self.player.discard.extend(flush);
        self.end_of_turn_cleanup();
        let hooks = self.after_side_turn_end(Side::Player);
        let mut subs = hooks;
        subs.push(Effect::StartTurn(Side::Enemy));
        self.push_front_all(subs);
    }

    /// `EndEnemyTurnInternal` + `SwitchSides` back to the player.
    fn end_enemy_turn(&mut self) {
        self.end_of_turn_cleanup();
        let mut subs = self.after_side_turn_end(Side::Enemy);
        self.round += 1;
        self.player.turn += 1;
        subs.push(Effect::StartTurn(Side::Player));
        self.push_front_all(subs);
    }

    /// `PlayerCombatState.EndOfTurnCleanup` over every card in combat.
    fn end_of_turn_cleanup(&mut self) {
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            for c in pile.iter_mut() {
                c.end_of_turn_cleanup();
            }
        }
    }

    /// `Hook.AfterTurnEnd` dispatching `AfterSideTurnEnd` to every listener
    /// in the game's iteration order: player powers, then each enemy's powers.
    fn after_side_turn_end(&self, side: Side) -> Vec<Effect> {
        let mut out = vec![];
        for p in &self.player.creature.powers {
            out.extend(p.after_side_turn_end(CreatureRef::Player, side));
        }
        for (i, e) in self.enemies.iter().enumerate() {
            if !e.creature.alive() {
                continue;
            }
            for p in &e.creature.powers {
                out.extend(p.after_side_turn_end(CreatureRef::Enemy(i), side));
            }
        }
        out
    }

    /// `CombatManager.CheckWinCondition` / `IsEnding` plus the loss branch of
    /// `CreatureCmd.Kill`.
    fn check_win(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        if !self.player.creature.alive() {
            self.outcome = Some(Outcome::Lost);
        } else if self.living_enemies().next().is_none() {
            self.outcome = Some(Outcome::Won);
        }
    }

    // ---- piles -------------------------------------------------------------

    /// `CardPileCmd.Draw`: per card, reshuffle if needed, take the top.
    fn draw(&mut self, count: u32) {
        for _ in 0..count {
            if self.player.hand.len() >= MAX_HAND {
                break;
            }
            if self.player.draw.is_empty() {
                if self.player.discard.is_empty() {
                    break;
                }
                // CardPileCmd.Shuffle: discard becomes the draw pile.
                let mut cards = std::mem::take(&mut self.player.discard);
                cards.sort_by_key(|c| (c.id, c.upgraded));
                self.rngs.shuffle.shuffle(&mut cards);
                self.player.draw = cards;
            }
            let card = self.player.draw.remove(0);
            self.player.hand.push(card);
        }
    }

    /// Remove a card from whichever combat pile holds it.
    fn take_card(&mut self, uid: u32) -> Option<Card> {
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            if let Some(i) = pile.iter().position(|c| c.uid == uid) {
                return Some(pile.remove(i));
            }
        }
        None
    }

    #[allow(dead_code)]
    fn new_uid(&mut self) -> u32 {
        let u = self.next_uid;
        self.next_uid += 1;
        u
    }

    // ---- value hooks --------------------------------------------------------

    /// Listener order for value hooks: allies then enemies, powers first.
    /// Relics slot in after the player's powers once they exist.
    fn listeners(&self) -> impl Iterator<Item = (CreatureRef, &Power)> {
        let player = self.player.creature.powers.iter().map(|p| (CreatureRef::Player, p));
        let enemies = self
            .enemies
            .iter()
            .enumerate()
            .flat_map(|(i, e)| e.creature.powers.iter().map(move |p| (CreatureRef::Enemy(i), p)));
        player.chain(enemies)
    }

    /// `Hook.ModifyDamage`: one full additive pass, one multiplicative pass,
    /// then a cap pass (none modelled yet), floored at 0.
    pub fn modify_damage(
        &self,
        target: CreatureRef,
        dealer: Option<CreatureRef>,
        amount: f64,
        props: ValueProp,
    ) -> f64 {
        let mut num = amount;
        for (owner, p) in self.listeners() {
            num += p.modify_damage_additive(owner, dealer, props);
        }
        for (owner, p) in self.listeners() {
            num *= p.modify_damage_multiplicative(owner, target, dealer, props);
        }
        num.max(0.0)
    }

    /// `Hook.ModifyBlock`.
    pub fn modify_block(
        &self,
        target: CreatureRef,
        source_owner: CreatureRef,
        amount: f64,
        props: ValueProp,
    ) -> f64 {
        let mut num = amount;
        for (owner, p) in self.listeners() {
            num += p.modify_block_additive(owner, source_owner, props);
        }
        for (owner, p) in self.listeners() {
            num *= p.modify_block_multiplicative(owner, target, props);
        }
        num.max(0.0)
    }

    // ---- commands ------------------------------------------------------------

    /// `CreatureCmd.Damage` core. Returns HP actually lost.
    fn damage(&mut self, target: CreatureRef, amount: f64, props: ValueProp, dealer: Option<CreatureRef>) -> i32 {
        let modified = self.modify_damage(target, dealer, amount, props);
        let c = self.creature_mut(target);
        // Creature.DamageBlockInternal: block absorbs min(block, amount),
        // truncated on the block write but not on the remainder.
        let blocked = if props.has(ValueProp::UNBLOCKABLE) { 0.0 } else { (c.block as f64).min(modified) };
        c.block -= blocked as i32;
        let unblocked = (modified - blocked).max(0.0);
        // Creature.LoseHpInternal: truncate once.
        let lost = unblocked.min(CLAMP) as i32;
        let before = c.hp;
        c.hp = (c.hp - lost).max(0);
        before - c.hp
    }

    /// `CreatureCmd.GainBlock`.
    fn gain_block(&mut self, target: CreatureRef, amount: f64, props: ValueProp, source_owner: CreatureRef) {
        let source_owner = match target {
            CreatureRef::Player => source_owner,
            e => e,
        };
        let modified = self.modify_block(target, source_owner, amount, props);
        if modified > 0.0 {
            let c = self.creature_mut(target);
            c.block = ((c.block as f64 + modified).min(CLAMP)) as i32;
        }
    }

    /// `PowerCmd.Apply` + `ModifyAmount`. Stacks onto an existing instance,
    /// otherwise creates one. Debuffs landing on the player skip their next tick.
    fn apply_power(&mut self, target: CreatureRef, id: PowerId, amount: i32, _applier: Option<CreatureRef>) {
        if amount == 0 || !self.creature(target).alive() {
            return;
        }
        let c = self.creature_mut(target);
        if let Some(p) = c.powers.iter_mut().find(|p| p.id == id) {
            p.amount += amount;
            if p.should_remove() {
                c.powers.retain(|p| p.id != id);
            }
        } else {
            let skip = target == CreatureRef::Player && is_debuff(id);
            c.powers.push(Power { id, amount, skip_next_tick: skip });
        }
    }
}

/// `Creature.SetUniqueMonsterHpValue`: uniform over the range minus the max HP
/// values already taken by other enemies, falling back to a plain roll.
fn roll_unique_hp(lo: i32, hi: i32, placed: &[Enemy], rng: &mut crate::rng::Rng) -> i32 {
    let mut options: Vec<i32> = (lo..=hi).filter(|v| !placed.iter().any(|e| e.creature.max_hp == *v)).collect();
    if options.is_empty() {
        options = (lo..=hi).collect();
    }
    options[rng.next_int(options.len())]
}
