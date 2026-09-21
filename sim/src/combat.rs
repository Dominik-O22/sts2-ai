//! Combat state and the loop that resolves effects. `Combat/CombatState.cs`,
//! `Combat/CombatManager.cs`, `Commands/CreatureCmd.cs`, `Commands/CardPileCmd.cs`,
//! `Commands/CardCmd.cs`, `Commands/PowerCmd.cs`.

use std::collections::VecDeque;

use crate::card::{Card, Tag, IRONCLAD_POOL};
use crate::effect::{AttackTargets, CardFilter, Effect, GenPool, Pile, Then};
use crate::ids::{CardId, MonsterId, PowerId};
use crate::monster::{Flags, Monster};
use crate::potion::{PotionId, Target as PotionTarget};
use crate::power::{is_debuff, is_single, Power};
use crate::relic::Relic;
use crate::rng::CombatRngs;
use crate::types::{Ascension, CardType, CreatureRef, Keyword, Side, TargetType, ValueProp};

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
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerCombat {
    pub creature: Creature,
    pub hand: Vec<Card>,
    /// Index 0 is the top of the draw pile.
    pub draw: Vec<Card>,
    pub discard: Vec<Card>,
    pub exhaust: Vec<Card>,
    /// `PileType.Play`: limbo for cards currently resolving.
    pub play: Vec<Card>,
    /// Generated cards on a choose-a-card screen (Attack/Skill/Power Potion).
    /// Not part of combat until taken.
    pub offer: Vec<Card>,
    pub energy: i32,
    /// `Player.MaxEnergy` before power modifiers.
    pub base_max_energy: i32,
    /// `PlayerCombatState.TurnNumber`, starts at 1.
    pub turn: u32,
}

#[derive(Clone, Debug)]
pub struct Enemy {
    pub creature: Creature,
    pub monster: Monster,
    /// Illusion: dead but will act (revive) on its next turn.
    pub reviving: bool,
}

impl Enemy {
    /// `Creature.IsPrimaryEnemy`: combat ends when no primary enemy lives.
    pub fn primary(&self) -> bool {
        self.creature.power(PowerId::Minion).is_none()
    }
    /// Takes part in the enemy turn.
    fn acts(&self) -> bool {
        self.creature.alive() || self.reviving
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Won,
    Lost,
}

/// Player inputs. `GameActions/PlayCardAction.cs`, `EndPlayerTurnAction.cs`,
/// `UsePotionAction.cs`, plus answering a pending card choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    PlayCard { hand_idx: usize, target: Option<usize> },
    EndTurn,
    UsePotion { slot: usize, target: Option<usize> },
    /// Index into `Pending::options`.
    Choose(usize),
    /// Close a skippable choice without picking.
    Skip,
}

/// A suspended card selection (`CardSelectCmd`), waiting for `Action::Choose`
/// or, when `can_skip`, `Action::Skip`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    pub options: Vec<u32>,
    pub then: Then,
    pub can_skip: bool,
}

/// A monster to place in the encounter.
#[derive(Clone, Copy, Debug)]
pub struct EnemySpec {
    pub id: MonsterId,
    pub flags: Flags,
}

/// `Rooms/RoomType.cs`, the combat kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomKind {
    Monster,
    Elite,
    Boss,
}

/// Everything a combat needs from the run.
#[derive(Clone, Debug)]
pub struct Setup<'a> {
    pub deck: &'a [Card],
    pub hp: i32,
    pub max_hp: i32,
    pub max_energy: i32,
    pub relics: &'a [Relic],
    /// Potion slots, `None` for empty ones.
    pub potions: &'a [Option<PotionId>],
    pub enemies: &'a [EnemySpec],
    pub room: RoomKind,
    pub asc: Ascension,
    pub seed: u64,
}

/// Recorded outcomes the replay harness forces instead of rolling:
/// shuffle results and starting enemy HP. Each shuffle consumes one entry;
/// once the queue is empty the RNG takes over again.
#[derive(Clone, Debug, Default)]
pub struct Script {
    /// Draw pile orders, top first, as (id, upgraded).
    pub shuffles: VecDeque<Vec<(CardId, bool)>>,
    /// Max HP per starting enemy, by index.
    pub enemy_hp: Vec<i32>,
    /// Targets for random-target hits, as indices into the living enemies.
    pub random_targets: VecDeque<usize>,
}

/// The parts of `CombatManager.History` that cards and powers read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub exhausted_this_turn: u32,
    /// Player took unblocked damage this turn.
    pub hp_lost_this_turn: bool,
    /// Unblocked hits the player has taken this combat (Tear Asunder).
    pub unblocked_hits_taken: u32,
    /// Uids of cards that gained the player block this turn (Unmovable).
    pub block_plays_this_turn: Vec<u32>,
    pub last_drawn: Option<u32>,
    /// Rupture: Strength owed once the current card play finishes.
    pub rupture_pending: i32,
    /// Card plays started this turn (Ringing).
    pub cards_played_this_turn: u32,
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
    pub pending: Option<Pending>,
    pub stats: Stats,
    /// The run's relics, with counters updated in place.
    pub relics: Vec<Relic>,
    pub room: RoomKind,
    /// Potion slots. Using a potion empties its slot.
    pub potions: Vec<Option<PotionId>>,
    pub script: Script,
    /// Every shuffle result this combat, top first (the recorder's view).
    pub shuffle_log: Vec<Vec<(CardId, bool)>>,
    queue: VecDeque<Effect>,
    next_uid: u32,
    /// False during setup, before the first turn starts.
    started: bool,
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
        Self::with_setup(&Setup {
            deck,
            hp,
            max_hp,
            max_energy,
            relics: &[],
            potions: &[],
            enemies,
            room: RoomKind::Monster,
            asc,
            seed,
        })
    }

    pub fn with_setup(setup: &Setup) -> Self {
        Self::with_script(setup, Script::default())
    }

    /// `with_setup` with forced shuffles and enemy HP.
    pub fn with_script(setup: &Setup, mut script: Script) -> Self {
        let Setup { deck, hp, max_hp, max_energy, enemies, asc, seed, .. } = *setup;
        let mut rngs = CombatRngs::new(seed);
        let mut next_uid = 1;
        let mut draw: Vec<Card> = deck
            .iter()
            .map(|c| {
                let mut c = Card::new(next_uid, c.id, c.upgraded);
                c.extra_damage = 0.0;
                next_uid += 1;
                c
            })
            .collect();
        let mut shuffle_log = vec![];
        shuffle_cards(&mut draw, &mut script, &mut rngs.shuffle, &mut shuffle_log);

        let mut c = Self {
            player: PlayerCombat {
                creature: Creature { hp, max_hp, block: 0, powers: vec![] },
                hand: vec![],
                draw,
                discard: vec![],
                exhaust: vec![],
                play: vec![],
                offer: vec![],
                energy: 0,
                base_max_energy: max_energy,
                turn: 1,
            },
            enemies: Vec::with_capacity(enemies.len()),
            round: 1,
            side: Side::Player,
            asc,
            rngs,
            outcome: None,
            pending: None,
            stats: Stats::default(),
            relics: setup.relics.to_vec(),
            room: setup.room,
            potions: setup.potions.to_vec(),
            script,
            shuffle_log,
            queue: VecDeque::new(),
            next_uid,
            started: false,
        };
        for (i, spec) in enemies.iter().enumerate() {
            c.spawn(spec.id, spec.flags);
            if let Some(&hp) = c.script.enemy_hp.get(i) {
                let e = &mut c.enemies[i].creature;
                e.max_hp = hp;
                e.hp = hp;
            }
        }
        c.started = true;
        let pre = c.relic_before_combat_start();
        c.queue.extend(pre);
        c.queue.push_back(Effect::StartTurn(Side::Player));
        c.run();
        c
    }

    /// `CombatState.CreateCreature` + `MonsterModel.AfterAddedToRoom`: roll
    /// HP, apply innate powers and block, add to the enemy list.
    fn spawn(&mut self, id: MonsterId, flags: Flags) {
        let (lo, hi) = Monster::hp_range(id, self.asc);
        let hp = roll_unique_hp(lo, hi, &self.enemies, &mut self.rngs.niche);
        let mut creature = Creature { hp, max_hp: hp, block: 0, powers: vec![] };
        let me = CreatureRef::Enemy(self.enemies.len());
        for (pid, amount) in Monster::innate_powers(id, self.asc) {
            let mut p = Power::new(pid, amount);
            p.applier = Some(me);
            creature.powers.push(p);
        }
        creature.block = Monster::innate_block(id);
        let mut monster = Monster::new(id, self.asc, flags);
        // CombatManager.AfterCreatureAdded: roll at once during the player's turn.
        if self.started && self.side == Side::Player {
            monster.roll_move(&mut self.rngs.monster_ai);
        }
        self.enemies.push(Enemy { creature, monster, reviving: false });
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// Replace an enemy's rolled move with a named one from its state graph.
    /// Returns false when the graph has no such move.
    pub fn set_enemy_move(&mut self, i: usize, name: &str) -> bool {
        self.enemies[i].monster.force_named_move(name)
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

    /// Every card the player has in combat, all piles.
    pub fn all_cards(&self) -> impl Iterator<Item = &Card> {
        let p = &self.player;
        p.hand.iter().chain(&p.draw).chain(&p.discard).chain(&p.exhaust).chain(&p.play)
    }

    pub fn find_card(&self, uid: u32) -> Option<&Card> {
        self.all_cards().find(|c| c.uid == uid)
    }

    fn find_card_mut(&mut self, uid: u32) -> Option<&mut Card> {
        let p = &mut self.player;
        p.hand
            .iter_mut()
            .chain(&mut p.draw)
            .chain(&mut p.discard)
            .chain(&mut p.exhaust)
            .chain(&mut p.play)
            .find(|c| c.uid == uid)
    }

    /// `PlayerCombatState.MaxEnergy` through `Hook.ModifyMaxEnergy`.
    pub fn max_energy(&self) -> i32 {
        let n = self.player.creature.powers.iter().fold(self.player.base_max_energy, |acc, p| p.modify_max_energy(acc));
        self.relic_modify_max_energy(n)
    }

    /// `CardEnergyCost.GetWithModifiers(All)`: local modifiers then global
    /// ones (Corruption, Free Attack). X-cost cards report `-1`.
    pub fn cost(&self, card: &Card) -> i32 {
        let local = card.local_cost();
        if local < 0 || card.def().x_cost {
            return local;
        }
        if self.player.creature.powers.iter().any(|p| p.free_card(card.ty())) {
            return 0;
        }
        // TangledPower.cs: every attack is afflicted with Entangled (+amount).
        if card.ty() == CardType::Attack {
            return local + self.player.creature.power_amount(PowerId::Tangled);
        }
        local
    }

    /// `Hook.ShouldPlay`: Ringing allows only the first card play each turn.
    fn hook_allows_play(&self) -> bool {
        !(self.player.creature.power(PowerId::Ringing).is_some() && self.stats.cards_played_this_turn > 0)
    }

    /// Every action the player may take right now.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = vec![];
        if self.is_over() {
            return out;
        }
        if let Some(p) = &self.pending {
            out.extend((0..p.options.len()).map(Action::Choose));
            if p.can_skip {
                out.push(Action::Skip);
            }
            return out;
        }
        if self.side != Side::Player {
            return out;
        }
        for (slot, id) in self.potions.iter().enumerate() {
            let Some(id) = id else { continue };
            if !id.usable_in_combat() {
                continue;
            }
            match id.target() {
                PotionTarget::Enemy => {
                    for e in self.living_enemies() {
                        out.push(Action::UsePotion { slot, target: Some(e) });
                    }
                }
                PotionTarget::None => out.push(Action::UsePotion { slot, target: None }),
            }
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
        if card.has(Keyword::Unplayable) || !self.hook_allows_play() {
            return false;
        }
        if card.def().x_cost {
            return true;
        }
        let cost = self.cost(card);
        cost >= 0 && self.player.energy >= cost
    }

    /// Apply one player action and resolve everything it triggers.
    pub fn step(&mut self, action: Action) {
        assert!(!self.is_over(), "step after combat ended");
        match action {
            Action::Choose(i) => {
                let p = self.pending.take().expect("no pending choice");
                let uid = p.options[i];
                let subs = self.choose(p.then, uid);
                self.push_front_all(subs);
            }
            Action::Skip => {
                let p = self.pending.take().expect("no pending choice");
                assert!(p.can_skip, "choice cannot be skipped");
                match p.then {
                    Then::DiscardThenDraw { picked } if picked > 0 => {
                        self.queue.push_front(Effect::Draw { count: picked, from_hand_draw: false });
                    }
                    Then::TakeOffer => self.player.offer.clear(),
                    _ => {}
                }
            }
            Action::UsePotion { slot, target } => {
                assert!(self.pending.is_none() && self.side == Side::Player, "not accepting potions");
                // PotionModel.RemoveBeforeUse, then OnUse, then the after hooks.
                let id = self.potions[slot].take().expect("empty potion slot");
                assert!(id.usable_in_combat(), "potion not usable in combat");
                let target = match id.target() {
                    PotionTarget::Enemy => {
                        let t = target.expect("target required");
                        assert!(self.enemies[t].creature.alive(), "dead target");
                        Some(CreatureRef::Enemy(t))
                    }
                    PotionTarget::None => None,
                };
                let mut subs = id.on_use(self, target);
                subs.push(Effect::AfterPotionUsed);
                self.push_front_all(subs);
            }
            Action::PlayCard { hand_idx, target } => {
                assert!(self.pending.is_none() && self.side == Side::Player, "not accepting card plays");
                let card = self.player.hand[hand_idx].clone();
                assert!(self.can_play(&card), "illegal card play");
                let target = self.resolve_target(&card, target);
                // PlayCardAction: spend energy (all of it for X cards), then play.
                let paid;
                if card.def().x_cost {
                    paid = self.player.energy;
                    self.player.hand[hand_idx].captured_x = self.player.energy + self.relic_x_bonus();
                    self.player.energy = 0;
                } else {
                    paid = self.cost(&card);
                    self.player.energy -= paid;
                }
                let card = self.player.hand.remove(hand_idx);
                let uid = card.uid;
                self.player.play.push(card);
                self.queue.push_back(Effect::PlayCard { uid, target, paid });
            }
            Action::EndTurn => {
                assert!(self.pending.is_none() && self.side == Side::Player, "not accepting end turn");
                self.queue.push_back(Effect::EndPlayerTurn);
            }
        }
        self.run();
    }

    fn resolve_target(&self, card: &Card, target: Option<usize>) -> Option<CreatureRef> {
        match card.def().target {
            TargetType::AnyEnemy => {
                let t = target.expect("target required");
                assert!(self.enemies[t].creature.alive(), "dead target");
                Some(CreatureRef::Enemy(t))
            }
            _ => None,
        }
    }

    /// Drain the queue. Stops when combat ends or a choice is pending.
    fn run(&mut self) {
        let mut resolved = 0u32;
        while self.pending.is_none() {
            let Some(e) = self.queue.pop_front() else { break };
            if self.is_over() {
                self.queue.clear();
                break;
            }
            resolved += 1;
            assert!(
                resolved < 100_000 && self.queue.len() < 100_000,
                "runaway effect loop: resolving {e:?}, queue front {:?}",
                self.queue.iter().take(5).collect::<Vec<_>>()
            );
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
                            match self.script.random_targets.pop_front() {
                                Some(t) if t < opts.len() => vec![opts[t]],
                                _ => self.rngs.targets.pick(&opts).copied().into_iter().collect(),
                            }
                        }
                    };
                    for t in ts {
                        subs.push(Effect::Damage { target: t, amount: base, props, dealer: Some(dealer), card });
                    }
                }
                // VigorPower.AfterAttack: spent by the first card attack.
                // GigantificationPower.AfterAttack: one charge per card attack.
                if dealer == CreatureRef::Player && card.is_some() && props.is_powered() {
                    let v = self.player.creature.power_amount(PowerId::Vigor);
                    if v > 0 {
                        subs.push(Effect::ApplyPower { target: dealer, id: PowerId::Vigor, amount: -v, applier: None });
                    }
                    if self.player.creature.power(PowerId::Gigantification).is_some() {
                        subs.push(Effect::DecrementPower { target: dealer, id: PowerId::Gigantification });
                    }
                }
                self.push_front_all(subs);
            }
            Effect::Damage { target, amount, props, dealer, card } => {
                if !self.creature(target).alive() {
                    return;
                }
                let subs = self.damage(target, amount, props, dealer, card);
                self.push_front_all(subs);
                self.check_win();
            }
            Effect::DamageAllEnemies { amount, props, dealer } => {
                let subs: Vec<Effect> = self
                    .living_enemies()
                    .map(|i| Effect::Damage { target: CreatureRef::Enemy(i), amount, props, dealer: Some(dealer), card: None })
                    .collect();
                self.push_front_all(subs);
            }
            Effect::GainBlock { target, amount, props, card } => {
                let subs = self.gain_block(target, amount, props, card);
                self.push_front_all(subs);
            }
            Effect::Heal { target, amount } => {
                let c = self.creature_mut(target);
                c.hp = (c.hp + amount as i32).min(c.max_hp);
                if target == CreatureRef::Player {
                    let subs = self.relic_after_heal();
                    self.push_front_all(subs);
                }
            }
            Effect::LoseEnergy { amount } => {
                self.player.energy = (self.player.energy - amount).max(0);
            }
            Effect::GainMaxHp { target, amount } => {
                let c = self.creature_mut(target);
                c.max_hp += amount;
                c.hp += amount;
            }
            Effect::GainEnergy { amount } => {
                // PlayerCmd.GainEnergy through Hook.ModifyEnergyGain.
                if amount > 0 {
                    let n = self.player.creature.powers.iter().fold(amount, |acc, p| p.modify_energy_gain(acc));
                    if n > 0 {
                        self.player.energy += n;
                    }
                }
            }
            Effect::ApplyPower { target, id, amount, applier } => {
                let subs = self.apply_power(target, id, amount, applier);
                self.push_front_all(subs);
            }
            Effect::RemovePower { target, id } => {
                self.creature_mut(target).powers.retain(|p| p.id != id);
            }
            Effect::TickDuration { target, id } => {
                let c = self.creature_mut(target);
                if let Some(p) = c.powers.iter_mut().find(|p| p.id == id) {
                    if p.skip_next_tick {
                        p.skip_next_tick = false;
                        return;
                    }
                }
                self.modify_power(target, id, -1);
            }
            Effect::DecrementPower { target, id } => self.modify_power(target, id, -1),
            Effect::Draw { count, from_hand_draw } => {
                let subs = self.draw(count, from_hand_draw);
                self.push_front_all(subs);
            }
            Effect::Exhaust { uid, ethereal } => {
                let subs = self.exhaust(uid, ethereal);
                self.push_front_all(subs);
            }
            Effect::ExhaustHand { filter } => {
                let uids: Vec<u32> = self.player.hand.iter().filter(|c| filter_ok(filter, c)).map(|c| c.uid).collect();
                self.push_front_all(uids.into_iter().map(|uid| Effect::Exhaust { uid, ethereal: false }).collect());
            }
            Effect::ExhaustRandomFromHand { filter, then_add_damage_to } => {
                let uids: Vec<u32> = self.player.hand.iter().filter(|c| filter_ok(filter, c)).map(|c| c.uid).collect();
                if let Some(&uid) = self.rngs.card_selection.pick(&uids) {
                    // Thrash: absorb the exhausted attack's damage first.
                    if let Some(thrash) = then_add_damage_to {
                        let dmg = self.find_card(uid).map_or(0.0, |k| k.vars().damage);
                        if let Some(t) = self.find_card_mut(thrash) {
                            t.extra_damage += dmg;
                        }
                    }
                    self.queue.push_front(Effect::Exhaust { uid, ethereal: false });
                }
            }
            Effect::MoveCard { uid, to } => {
                if let Some(card) = self.take_card(uid) {
                    self.put_card(card, to);
                }
            }
            Effect::Upgrade { uid } => {
                if let Some(c) = self.find_card_mut(uid) {
                    if c.upgradable() {
                        c.upgraded = true;
                    }
                }
            }
            Effect::UpgradeHand => {
                for c in &mut self.player.hand {
                    if c.upgradable() {
                        c.upgraded = true;
                    }
                }
            }
            Effect::TransformHand { filter, into, upgraded } => {
                for c in &mut self.player.hand {
                    if filter_ok(filter, c) {
                        *c = Card::new(c.uid, into, upgraded);
                    }
                }
            }
            Effect::GenerateCard { id, upgraded, to, free_this_turn } => {
                let mut card = Card::new(self.new_uid(), id, upgraded);
                if free_this_turn {
                    card.cost_this_turn = Some(0);
                }
                self.relic_card_entered_combat(&mut card);
                self.put_card(card, to);
            }
            Effect::GenerateRandom { pool, count, to, free_this_turn, distinct } => {
                let options = pool_cards(pool);
                let mut chosen = Vec::new();
                if distinct {
                    let mut opts = options.clone();
                    self.rngs.card_generation.shuffle(&mut opts);
                    chosen.extend(opts.into_iter().take(count as usize));
                } else {
                    for _ in 0..count {
                        if let Some(&id) = self.rngs.card_generation.pick(&options) {
                            chosen.push(id);
                        }
                    }
                }
                let subs = chosen
                    .into_iter()
                    .map(|id| Effect::GenerateCard { id, upgraded: false, to, free_this_turn })
                    .collect();
                self.push_front_all(subs);
            }
            Effect::Choose { from, filter, then, can_skip } => {
                let pile = match from {
                    Pile::Hand => &self.player.hand,
                    Pile::Discard => &self.player.discard,
                    Pile::Exhaust => &self.player.exhaust,
                    Pile::DrawTop | Pile::DrawBottom => &self.player.draw,
                };
                let options: Vec<u32> = pile.iter().filter(|c| filter_ok(filter, c)).map(|c| c.uid).collect();
                if !options.is_empty() {
                    self.pending = Some(Pending { options, then, can_skip });
                } else if let Then::DiscardThenDraw { picked } = then {
                    // Gambler's Brew with the hand emptied: draw what was discarded.
                    if picked > 0 {
                        self.queue.push_front(Effect::Draw { count: picked, from_hand_draw: false });
                    }
                }
            }
            Effect::OfferRandom { pool, count } => {
                let mut opts = pool_cards(pool);
                self.rngs.card_generation.shuffle(&mut opts);
                self.player.offer.clear();
                for id in opts.into_iter().take(count as usize) {
                    let uid = self.new_uid();
                    self.player.offer.push(Card::new(uid, id, false));
                }
                let options: Vec<u32> = self.player.offer.iter().map(|c| c.uid).collect();
                if !options.is_empty() {
                    self.pending = Some(Pending { options, then: Then::TakeOffer, can_skip: true });
                }
            }
            Effect::TakeOffer { uid } => {
                if let Some(i) = self.player.offer.iter().position(|c| c.uid == uid) {
                    let mut card = self.player.offer.remove(i);
                    card.cost_this_turn = Some(0);
                    self.relic_card_entered_combat(&mut card);
                    self.put_card(card, Pile::Hand);
                }
                self.player.offer.clear();
            }
            Effect::Shuffle => {
                // CardPileCmd.Shuffle: discard then draw, shuffled together.
                let mut cards = std::mem::take(&mut self.player.discard);
                cards.append(&mut self.player.draw);
                shuffle_cards(&mut cards, &mut self.script, &mut self.rngs.shuffle, &mut self.shuffle_log);
                self.player.draw = cards;
                let subs = self.relic_after_shuffle();
                self.push_front_all(subs);
            }
            Effect::SneckoCosts => {
                for i in 0..self.player.hand.len() {
                    let c = &self.player.hand[i];
                    if c.def().x_cost || c.local_cost() < 0 {
                        continue;
                    }
                    let cost = self.rngs.energy_costs.next_int(4) as i32;
                    self.player.hand[i].cost_this_turn = Some(cost);
                }
            }
            Effect::StrikeReplay => {
                let p = &mut self.player;
                for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
                    for c in pile.iter_mut().filter(|c| c.has_tag(Tag::Strike)) {
                        c.replay += 1;
                    }
                }
            }
            Effect::AfterPotionUsed => {
                let mut subs = self.relic_after_potion_used();
                subs.extend(self.relic_after_hand_emptied());
                self.push_front_all(subs);
            }
            Effect::PlayCard { uid, target, paid } => {
                let Some(card) = self.find_card(uid) else { return };
                // GeneratePlayCount: replays, then Hook.ModifyCardPlayCount, then
                // each modifying power is told (OneTwoPunch, Duplication decrement).
                let mut plays = 1 + card.replay;
                let modifiers: Vec<PowerId> = self
                    .player
                    .creature
                    .powers
                    .iter()
                    .filter(|p| p.extra_plays(card.ty()) > 0)
                    .inspect(|p| plays += p.extra_plays(card.ty()))
                    .map(|p| p.id)
                    .collect();
                for id in modifiers {
                    self.modify_power(CreatureRef::Player, id, -1);
                }
                let mut subs: Vec<Effect> = (0..plays).map(|_| Effect::CardPlayIter { uid, target, paid }).collect();
                subs.push(Effect::FinishCardPlay { uid });
                self.push_front_all(subs);
            }
            Effect::CardPlayIter { uid, target, paid } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                self.stats.cards_played_this_turn += 1;
                let mut subs = self.before_card_played(&card);
                subs.extend(self.relic_before_card_played(&card, paid));
                subs.extend(card.on_play(self, target));
                subs.push(Effect::AfterCardPlayed { uid });
                self.push_front_all(subs);
            }
            Effect::AfterCardPlayed { uid } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                let subs = self.after_card_played(&card);
                self.push_front_all(subs);
            }
            Effect::FinishCardPlay { uid } => {
                // Rampage/Thrash growth after the play, Rupture's deferred Strength.
                if let Some(c) = self.find_card_mut(uid) {
                    if c.id == CardId::Rampage {
                        c.extra_damage += c.vars().magic;
                    }
                }
                let owed = std::mem::take(&mut self.stats.rupture_pending);
                if owed > 0 {
                    self.queue.push_front(Effect::ApplyPower {
                        target: CreatureRef::Player,
                        id: PowerId::Strength,
                        amount: owed,
                        applier: Some(CreatureRef::Player),
                    });
                }
                // CardModel.GetResultPileTypeForCardPlay + Corruption.
                if let Some(card) = self.take_card(uid) {
                    let corruption = self.player.creature.power(PowerId::Corruption).is_some();
                    if card.ty() == CardType::Power {
                        // Powers leave combat (PileType.None).
                    } else if card.has(Keyword::Exhaust)
                        || card.exhaust_on_next_play
                        || (corruption && card.ty() == CardType::Skill)
                    {
                        self.player.exhaust.push(card);
                        let subs = self.after_card_exhausted(uid, false);
                        self.push_front_all(subs);
                    } else {
                        self.player.discard.push(card);
                    }
                }
                let subs = self.relic_after_hand_emptied();
                self.push_front_all(subs);
            }
            Effect::CardStep { uid, target, step } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                let subs = card.step(self, target, step);
                self.push_front_all(subs);
            }
            Effect::AutoPlay { uid, force_exhaust } => {
                let subs = self.auto_play(uid, force_exhaust);
                self.push_front_all(subs);
            }
            Effect::AutoPlayFromDrawTop { force_exhaust } => {
                if self.reshuffle_if_needed() {
                    let subs = self.relic_after_shuffle();
                    self.push_front_all(subs);
                }
                if !self.player.draw.is_empty() {
                    let card = self.player.draw.remove(0);
                    let uid = card.uid;
                    self.player.play.push(card);
                    self.queue.push_front(Effect::AutoPlay { uid, force_exhaust });
                }
            }
            Effect::AutoPlayRandomAttack => {
                let uids: Vec<u32> =
                    self.player.hand.iter().filter(|c| filter_ok(CardFilter::PlayableAttack, c)).map(|c| c.uid).collect();
                if let Some(&uid) = self.rngs.shuffle.pick(&uids) {
                    self.queue.push_front(Effect::AutoPlay { uid, force_exhaust: false });
                }
            }
            Effect::AggressionPull { count } => {
                let mut uids: Vec<u32> =
                    self.player.discard.iter().filter(|c| c.ty() == CardType::Attack).map(|c| c.uid).collect();
                self.rngs.card_selection.shuffle(&mut uids);
                for uid in uids.into_iter().take(count as usize) {
                    if let Some(mut card) = self.take_card(uid) {
                        if card.upgradable() {
                            card.upgraded = true;
                        }
                        self.player.hand.push(card);
                    }
                }
            }
            Effect::SpawnMonster { id, flags } => self.spawn(id, flags),
            Effect::Stun { target, next } => {
                if let CreatureRef::Enemy(i) = target {
                    self.enemies[i].monster.stun(next);
                }
            }
            Effect::RemoveStrength { target } => {
                self.creature_mut(target).powers.retain(|p| {
                    p.id != PowerId::Strength && crate::power::temp_strength_sign(p.id).is_none()
                });
            }
            Effect::Revive { target } => {
                if let CreatureRef::Enemy(i) = target {
                    let e = &mut self.enemies[i];
                    e.creature.hp = e.creature.max_hp;
                    e.reviving = false;
                }
            }
            Effect::StartTurn(side) => self.start_turn(side),
            Effect::EndPlayerTurn => self.end_player_turn(),
            Effect::TurnEndInHand => self.turn_end_in_hand(),
            Effect::FlushHand => self.flush_hand(),
            Effect::EnemyAct(i) => {
                // Creature.TakeTurn skips monsters spawned since the last side switch.
                if self.enemies[i].acts() && !self.enemies[i].monster.spawned_this_turn {
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
        // MonsterModel.OnSideSwitch.
        for e in &mut self.enemies {
            e.monster.spawned_this_turn = false;
        }
        let mut subs = Vec::new();
        // Hook.BeforeSideTurnStart.
        subs.extend(self.collect_powers(|p, owner, c| p.before_side_turn_start(owner, side, c.round)));
        subs.extend(self.relic_before_side_turn_start(side));
        match side {
            Side::Player => {
                // Enemies roll their next move (PrepareForNextTurn).
                for i in 0..self.enemies.len() {
                    if self.enemies[i].acts() {
                        self.enemies[i].monster.roll_move(&mut self.rngs.monster_ai);
                    }
                }
                // Creature.AfterTurnStart: block clears except on player turn 1,
                // and unless a power (Barricade) or Sturdy Clamp prevents it.
                if self.player.turn != 1 {
                    if self.relic_keeps_block() {
                        self.player.creature.block = self.player.creature.block.min(10);
                    } else if self.should_clear_block(CreatureRef::Player) {
                        self.player.creature.block = 0;
                        // Hook.AfterBlockCleared: player powers, then relics.
                        for p in &self.player.creature.powers {
                            subs.extend(p.after_block_cleared(CreatureRef::Player));
                        }
                        subs.extend(self.relic_after_block_cleared());
                    }
                }
                // SetupPlayerTurn: energy, then draw (innate on top on turn 1).
                if self.relic_should_reset_energy() {
                    self.player.energy = self.max_energy();
                } else {
                    self.player.energy += self.max_energy();
                }
                // Hook.AfterEnergyReset: player powers, then relics.
                for p in &self.player.creature.powers {
                    subs.extend(p.after_energy_reset(CreatureRef::Player));
                }
                subs.extend(self.relic_after_energy_reset());
                let draw = self.player.creature.powers.iter().fold(BASE_HAND_DRAW, |n, p| p.modify_hand_draw(n));
                let mut draw = self.relic_modify_hand_draw(draw);
                if self.player.turn == 1 {
                    let innate: Vec<usize> = self
                        .player
                        .draw
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.has(Keyword::Innate))
                        .map(|(i, _)| i)
                        .collect();
                    let n = innate.len() as u32;
                    for (k, i) in innate.into_iter().enumerate() {
                        let c = self.player.draw.remove(i);
                        self.player.draw.insert(k, c);
                    }
                    draw = draw.max(n).min(MAX_HAND as u32);
                }
                subs.push(Effect::Draw { count: draw, from_hand_draw: true });
                // Hook.AfterPlayerTurnStart.
                subs.extend(self.collect_powers(|p, owner, _| {
                    if owner == CreatureRef::Player {
                        p.after_player_turn_start(owner)
                    } else {
                        vec![]
                    }
                }));
                subs.extend(self.relic_after_player_turn_start());
            }
            Side::Enemy => {
                for i in 0..self.enemies.len() {
                    let r = CreatureRef::Enemy(i);
                    if self.enemies[i].creature.alive() && self.should_clear_block(r) {
                        self.enemies[i].creature.block = 0;
                    }
                }
            }
        }
        // Hook.AfterSideTurnStart.
        for p in &mut self.player.creature.powers {
            p.reset_at_side_turn_start(CreatureRef::Player, side);
        }
        for (i, e) in self.enemies.iter_mut().enumerate() {
            for p in &mut e.creature.powers {
                p.reset_at_side_turn_start(CreatureRef::Enemy(i), side);
            }
        }
        let (turn, round) = (self.player.turn, self.round);
        subs.extend(self.collect_powers(|p, owner, _| p.after_side_turn_start(owner, side, turn, round)));
        subs.extend(self.relic_after_side_turn_start(side));
        if side == Side::Enemy {
            subs.extend((0..self.enemies.len()).map(Effect::EnemyAct));
            subs.push(Effect::EndEnemyTurn);
        }
        self.push_front_all(subs);
    }

    /// `EndPlayerTurnPhaseOneInternal`, first half: the AutoPostPlay phase
    /// and pre-flush hooks. The rest continues in `TurnEndInHand` and
    /// `FlushHand`, queued after these effects so they fully resolve first.
    fn end_player_turn(&mut self) {
        let mut subs = Vec::new();
        // AutoPostPlay phase: Stampede, Howl From Beyond in the exhaust pile.
        subs.extend(self.collect_powers(|p, owner, _| if owner == CreatureRef::Player { p.after_auto_post_play() } else { vec![] }));
        let howls: Vec<u32> = self.player.exhaust.iter().filter(|c| c.id == CardId::HowlFromBeyond).map(|c| c.uid).collect();
        subs.extend(howls.into_iter().map(|uid| Effect::AutoPlay { uid, force_exhaust: false }));
        // BeforeSideTurnEndEarly (Plating), then BeforeSideTurnEnd relics.
        subs.extend(self.collect_powers(|p, owner, _| p.before_side_turn_end_early(owner, Side::Player)));
        subs.extend(self.relic_before_side_turn_end());
        subs.push(Effect::TurnEndInHand);
        self.push_front_all(subs);
    }

    /// `DoTurnEnd`: ethereal cards exhaust, then turn-end-in-hand effects.
    fn turn_end_in_hand(&mut self) {
        let ethereal: Vec<u32> = self.player.hand.iter().filter(|c| c.has(Keyword::Ethereal)).map(|c| c.uid).collect();
        let mut subs: Vec<Effect> = ethereal.into_iter().map(|uid| Effect::Exhaust { uid, ethereal: true }).collect();
        for c in &self.player.hand {
            if matches!(c.id, CardId::Burn | CardId::Infection) {
                subs.push(Effect::Damage {
                    target: CreatureRef::Player,
                    amount: c.vars().damage,
                    props: ValueProp::UNPOWERED.or(ValueProp::MOVE),
                    dealer: None,
                    card: Some(c.uid),
                });
            }
        }
        subs.push(Effect::FlushHand);
        self.push_front_all(subs);
    }

    /// `EndPlayerTurnPhaseTwoInternal` + `SwitchSides`: discard the hand,
    /// cleanup, end-of-turn hooks, then the enemy turn.
    fn flush_hand(&mut self) {
        if self.relic_should_flush() && self.player.creature.powers.iter().all(|p| p.should_flush()) {
            let (retain, flush): (Vec<Card>, Vec<Card>) = self.player.hand.drain(..).partition(|c| c.has(Keyword::Retain));
            self.player.hand = retain;
            self.player.discard.extend(flush);
        }
        self.end_of_turn_cleanup();
        let mut subs = self.after_side_turn_end(Side::Player);
        subs.push(Effect::StartTurn(Side::Enemy));
        self.push_front_all(subs);
    }

    /// `EndEnemyTurnInternal` + `SwitchSides` back to the player.
    fn end_enemy_turn(&mut self) {
        self.end_of_turn_cleanup();
        let mut subs = self.after_side_turn_end(Side::Enemy);
        self.round += 1;
        self.player.turn += 1;
        self.stats.exhausted_this_turn = 0;
        self.stats.hp_lost_this_turn = false;
        self.stats.block_plays_this_turn.clear();
        self.stats.cards_played_this_turn = 0;
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
    fn after_side_turn_end(&mut self, side: Side) -> Vec<Effect> {
        let mut out = vec![];
        for p in &mut self.player.creature.powers {
            out.extend(p.after_side_turn_end(CreatureRef::Player, side));
        }
        out.extend(self.relic_after_side_turn_end(side));
        for (i, e) in self.enemies.iter_mut().enumerate() {
            if !e.creature.alive() {
                continue;
            }
            for p in &mut e.creature.powers {
                out.extend(p.after_side_turn_end(CreatureRef::Enemy(i), side));
            }
        }
        out
    }

    /// Collect effects from a read-only power hook, in listener order.
    fn collect_powers(&self, f: impl Fn(&Power, CreatureRef, &Combat) -> Vec<Effect>) -> Vec<Effect> {
        let mut out = vec![];
        for p in &self.player.creature.powers {
            out.extend(f(p, CreatureRef::Player, self));
        }
        for (i, e) in self.enemies.iter().enumerate() {
            if e.creature.alive() {
                for p in &e.creature.powers {
                    out.extend(f(p, CreatureRef::Enemy(i), self));
                }
            }
        }
        out
    }

    fn should_clear_block(&self, r: CreatureRef) -> bool {
        self.creature(r).powers.iter().all(|p| p.should_clear_block(r, r))
    }

    /// `CombatManager.CheckWinCondition` / `IsEnding` plus the loss branch of
    /// `CreatureCmd.Kill`.
    fn check_win(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        if !self.player.creature.alive() {
            self.outcome = Some(Outcome::Lost);
        } else if !self.enemies.iter().any(|e| e.creature.alive() && e.primary()) {
            self.outcome = Some(Outcome::Won);
            self.relic_after_victory();
        }
    }

    // ---- card play hooks ----------------------------------------------------

    /// `Hook.BeforeCardPlayed`: Free Attack decrement, Stomp cost reduction.
    fn before_card_played(&mut self, card: &Card) -> Vec<Effect> {
        let mut out = vec![];
        if card.ty() == CardType::Attack {
            if self.player.creature.power(PowerId::FreeAttack).is_some() {
                out.push(Effect::DecrementPower { target: CreatureRef::Player, id: PowerId::FreeAttack });
            }
            for c in self.player.hand.iter_mut().filter(|c| c.id == CardId::Stomp) {
                let cur = c.cost_this_turn.unwrap_or(c.base_cost());
                c.cost_this_turn = Some((cur - 1).max(0));
            }
        }
        out
    }

    /// `Hook.AfterCardPlayed`.
    fn after_card_played(&mut self, card: &Card) -> Vec<Effect> {
        let mut out = vec![];
        for p in &mut self.player.creature.powers {
            out.extend(p.after_card_played(CreatureRef::Player, card.ty(), card.id, card.upgraded));
        }
        out.extend(self.relic_after_card_played(card));
        for (i, e) in self.enemies.iter_mut().enumerate() {
            for p in &mut e.creature.powers {
                out.extend(p.after_card_played(CreatureRef::Enemy(i), card.ty(), card.id, card.upgraded));
            }
        }
        out
    }

    /// `Hook.AfterCardExhausted`: powers, then cards with self-hooks (Drum of Battle).
    fn after_card_exhausted(&mut self, uid: u32, ethereal: bool) -> Vec<Effect> {
        self.stats.exhausted_this_turn += 1;
        let mut out = vec![];
        for p in &mut self.player.creature.powers {
            out.extend(p.after_card_exhausted(CreatureRef::Player, ethereal));
        }
        if let Some(c) = self.find_card(uid).cloned() {
            out.extend(self.relic_after_card_exhausted(&c, ethereal));
            if c.id == CardId::DrumOfBattle {
                out.push(Effect::GainEnergy { amount: c.vars().energy });
            }
        }
        out
    }

    /// `CardCmd.AutoPlay`.
    fn auto_play(&mut self, uid: u32, force_exhaust: bool) -> Vec<Effect> {
        let Some(card) = self.find_card(uid).cloned() else { return vec![] };
        if !self.player.creature.alive() {
            return vec![];
        }
        if card.has(Keyword::Unplayable) || !self.hook_allows_play() {
            // MoveToResultPileWithoutPlaying.
            if let Some(c) = self.take_card(uid) {
                if c.ty() == CardType::Power {
                    self.player.discard.push(c);
                } else if c.has(Keyword::Exhaust) {
                    self.player.exhaust.push(c);
                } else {
                    self.player.discard.push(c);
                }
            }
            return vec![];
        }
        let target = match card.def().target {
            TargetType::AnyEnemy => {
                let opts: Vec<usize> = self.living_enemies().collect();
                match self.rngs.targets.pick(&opts) {
                    Some(&i) => Some(CreatureRef::Enemy(i)),
                    None => return vec![],
                }
            }
            _ => None,
        };
        if let Some(c) = self.take_card(uid) {
            let mut c = c;
            if c.def().x_cost {
                c.captured_x = self.player.energy + self.relic_x_bonus();
            }
            c.exhaust_on_next_play = force_exhaust || c.exhaust_on_next_play;
            self.player.play.push(c);
        }
        vec![Effect::PlayCard { uid, target, paid: 0 }]
    }

    // ---- piles -------------------------------------------------------------

    /// Returns true when a shuffle happened.
    fn reshuffle_if_needed(&mut self) -> bool {
        if self.player.draw.is_empty() && !self.player.discard.is_empty() {
            // CardPileCmd.Shuffle: discard becomes the draw pile.
            let mut cards = std::mem::take(&mut self.player.discard);
            shuffle_cards(&mut cards, &mut self.script, &mut self.rngs.shuffle, &mut self.shuffle_log);
            self.player.draw = cards;
            return true;
        }
        false
    }

    /// `CardPileCmd.Draw`: per card, reshuffle if needed, take the top.
    /// Returns hook effects (Hellraiser auto-plays drawn Strikes).
    fn draw(&mut self, count: u32, from_hand_draw: bool) -> Vec<Effect> {
        // Pillage reads this after each draw; a draw that yields nothing must clear it.
        self.stats.last_drawn = None;
        if !self.player.creature.powers.iter().all(|p| p.should_draw(from_hand_draw)) {
            return vec![];
        }
        let hellraiser = self.player.creature.power(PowerId::Hellraiser).is_some();
        let mut out = vec![];
        for _ in 0..count {
            if self.player.hand.len() >= MAX_HAND {
                break;
            }
            if self.reshuffle_if_needed() {
                out.extend(self.relic_after_shuffle());
            }
            if self.player.draw.is_empty() {
                break;
            }
            let card = self.player.draw.remove(0);
            let uid = card.uid;
            let strike = card.has_tag(Tag::Strike);
            self.player.hand.push(card);
            self.stats.last_drawn = Some(uid);
            if hellraiser && strike {
                out.push(Effect::AutoPlay { uid, force_exhaust: false });
            }
        }
        out
    }

    /// `CardCmd.Exhaust`.
    fn exhaust(&mut self, uid: u32, ethereal: bool) -> Vec<Effect> {
        let Some(card) = self.take_card(uid) else { return vec![] };
        self.player.exhaust.push(card);
        self.after_card_exhausted(uid, ethereal)
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

    fn put_card(&mut self, card: Card, to: Pile) {
        match to {
            Pile::Hand => {
                if self.player.hand.len() < MAX_HAND {
                    self.player.hand.push(card);
                } else {
                    self.player.discard.push(card);
                }
            }
            Pile::DrawTop => self.player.draw.insert(0, card),
            Pile::DrawBottom => self.player.draw.push(card),
            Pile::Discard => self.player.discard.push(card),
            Pile::Exhaust => self.player.exhaust.push(card),
        }
    }

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
    pub fn modify_damage(&self, target: CreatureRef, dealer: Option<CreatureRef>, amount: f64, props: ValueProp) -> f64 {
        self.modify_damage_from(target, dealer, amount, props, None)
    }

    /// `modify_damage` with the source card, for relics that read it.
    fn modify_damage_from(&self, target: CreatureRef, dealer: Option<CreatureRef>, amount: f64, props: ValueProp, card: Option<u32>) -> f64 {
        let (mut dv, mut dc) = match dealer {
            Some(d) => (self.creature(d).power_amount(PowerId::Vulnerable), self.creature(d).power_amount(PowerId::Cruelty)),
            None => (0, 0),
        };
        // PaperPhrog.ModifyVulnerableMultiplier: +0.25 for the player's attacks.
        if dealer == Some(CreatureRef::Player) && self.has_relic(crate::relic::RelicId::PaperPhrog) {
            dc += 25;
        }
        let card = card.and_then(|u| self.find_card(u));
        let player_card = if dealer == Some(CreatureRef::Player) { card } else { None };
        let mut num = amount;
        for (owner, p) in self.listeners() {
            num += p.modify_damage_additive(owner, dealer, props);
        }
        num += self.relic_damage_additive(player_card, props);
        for (owner, p) in self.listeners() {
            num *= p.modify_damage_multiplicative(owner, target, dealer, props, dv, dc);
        }
        num *= self.relic_damage_multiplicative(player_card, props);
        // GigantificationPower.ModifyDamageMultiplicative: the player's card attacks.
        if player_card.is_some() && props.is_powered() && self.player.creature.power(PowerId::Gigantification).is_some() {
            num *= 3.0;
        }
        let _ = &mut dv;
        num.max(0.0)
    }

    /// `Hook.ModifyBlock`.
    pub fn modify_block(&self, target: CreatureRef, source_owner: CreatureRef, amount: f64, props: ValueProp, card: Option<u32>) -> f64 {
        let plays = self.stats.block_plays_this_turn.iter().filter(|&&u| Some(u) != card).count();
        let mut num = amount;
        for (owner, p) in self.listeners() {
            num += p.modify_block_additive(owner, source_owner, props);
        }
        for (owner, p) in self.listeners() {
            num *= p.modify_block_multiplicative(owner, target, props, plays);
        }
        num.max(0.0)
    }

    // ---- commands ------------------------------------------------------------

    /// `CreatureCmd.Damage` core. Returns the effects of `AfterDamageReceived` hooks.
    fn damage(&mut self, target: CreatureRef, amount: f64, props: ValueProp, dealer: Option<CreatureRef>, card: Option<u32>) -> Vec<Effect> {
        let modified = self.modify_damage_from(target, dealer, amount, props, card);
        let c = self.creature_mut(target);
        // Creature.DamageBlockInternal: block absorbs min(block, amount),
        // truncated on the block write but not on the remainder.
        let blocked = if props.has(ValueProp::UNBLOCKABLE) { 0.0 } else { (c.block as f64).min(modified) };
        c.block -= blocked as i32;
        let mut unblocked = (modified - blocked).max(0.0);
        // SlipperyPower.ModifyHpLostAfterOsty: at most 1 HP per hit.
        if c.power(PowerId::Slippery).is_some() && unblocked >= 1.0 {
            unblocked = 1.0;
        }
        let mut buffered = false;
        if target == CreatureRef::Player {
            unblocked = self.relic_modify_hp_lost(unblocked);
            // BufferPower.ModifyHpLostAfterOstyLate: only counts when it changed
            // the truncated value.
            if unblocked >= 1.0 && self.player.creature.power(PowerId::Buffer).is_some() {
                unblocked = 0.0;
                buffered = true;
            }
        }
        let c = self.creature_mut(target);
        // Creature.LoseHpInternal: truncate once.
        let mut lost = unblocked.min(CLAMP) as i32;
        let was_alive = c.alive();
        c.hp = (c.hp - lost).max(0);
        let mut fairy_used = false;
        // LizardTail: ShouldDie false once, then heal to half.
        if target == CreatureRef::Player && self.player.creature.hp <= 0 && was_alive {
            if let Some(hp) = self.relic_prevent_death() {
                lost = lost.min(self.player.creature.max_hp);
                self.player.creature.hp = hp;
            } else if let Some(slot) = self.potions.iter().position(|p| *p == Some(PotionId::FairyInABottle)) {
                // FairyInABottle.ShouldDie + AfterPreventingDeath: 30% max HP, at least 1.
                self.potions[slot] = None;
                lost = lost.min(self.player.creature.max_hp);
                let heal = (self.player.creature.max_hp as f64 * 0.3).max(1.0) as i32;
                self.player.creature.hp = heal.min(self.player.creature.max_hp);
                fairy_used = true;
            }
        }
        if buffered {
            self.modify_power(CreatureRef::Player, PowerId::Buffer, -1);
        }
        let hp_after = self.creature(target).hp;
        let own_turn = self.side == target.side();
        if target == CreatureRef::Player && lost > 0 {
            self.stats.hp_lost_this_turn = true;
            self.stats.unblocked_hits_taken += 1;
        }
        // Hook.AfterDamageReceived over the target's powers.
        let mut out = vec![];
        if fairy_used {
            // OnUseWrapper ran for the automatic use, so its after hooks fire.
            out.push(Effect::AfterPotionUsed);
        }
        for p in &self.creature(target).powers {
            out.extend(p.after_damage_received(target, lost, props, dealer, own_turn, hp_after));
        }
        if target == CreatureRef::Player {
            out.extend(self.relic_after_damage_received(lost, props, own_turn));
        }
        if was_alive && !self.creature(target).alive() {
            if let CreatureRef::Enemy(i) = target {
                out.extend(self.on_enemy_death(i));
            }
        }
        // Rupture: HP loss from a card you are playing is paid after the play.
        if target == CreatureRef::Player && lost > 0 && own_turn {
            if let Some(r) = self.player.creature.power(PowerId::Rupture) {
                let in_play = card.is_some_and(|u| self.player.play.iter().any(|c| c.uid == u));
                if in_play {
                    self.stats.rupture_pending += r.amount;
                } else {
                    out.push(Effect::ApplyPower {
                        target: CreatureRef::Player,
                        id: PowerId::Strength,
                        amount: r.amount,
                        applier: Some(CreatureRef::Player),
                    });
                }
            }
        }
        out
    }

    /// `Hook.AfterDeath` for an enemy: Infested spawns Wrigglers, Illusion
    /// schedules a revive, and powers the dead creature applied to the
    /// player with `RemoveOnApplierDeath` semantics (Constrict, Shrink) go.
    fn on_enemy_death(&mut self, i: usize) -> Vec<Effect> {
        let me = CreatureRef::Enemy(i);
        let mut out = vec![];
        // InfestedPower.AfterDeath spawns synchronously, before any win check.
        if self.enemies[i].creature.power(PowerId::Infested).is_some() {
            for slot in 1..=4u8 {
                self.spawn(MonsterId::Wriggler, Flags { start_stunned: true, slot, ..Default::default() });
            }
        }
        if self.enemies[i].creature.power(PowerId::Illusion).is_some() {
            self.enemies[i].reviving = true;
            self.enemies[i].monster.set_revive();
        }
        self.player
            .creature
            .powers
            .retain(|p| !(matches!(p.id, PowerId::Constrict | PowerId::Shrink) && p.applier == Some(me)));
        out.extend(self.relic_after_enemy_death());
        out
    }

    /// `CreatureCmd.GainBlock`. Returns `AfterBlockGained` hook effects.
    fn gain_block(&mut self, target: CreatureRef, amount: f64, props: ValueProp, card: Option<u32>) -> Vec<Effect> {
        // Cards are only played by the player, so the block source owner is
        // always the target itself.
        let mut modified = self.modify_block(target, target, amount, props, card);
        if target == CreatureRef::Player {
            modified *= self.relic_block_multiplicative(card, props);
        }
        let mut out = vec![];
        if modified > 0.0 {
            let c = self.creature_mut(target);
            c.block = ((c.block as f64 + modified).min(CLAMP)) as i32;
            if target == CreatureRef::Player && props.has(ValueProp::MOVE) {
                if let Some(u) = card {
                    if !self.stats.block_plays_this_turn.contains(&u) {
                        self.stats.block_plays_this_turn.push(u);
                    }
                }
            }
            for p in &self.creature(target).powers {
                out.extend(p.after_block_gained(target, modified));
            }
        }
        out
    }

    /// `PowerCmd.Apply`. Stacks onto an existing instance, otherwise creates
    /// one. Debuffs landing on the player skip their next tick. Returns
    /// `BeforeApplied` / `AfterPowerAmountChanged` hook effects.
    fn apply_power(&mut self, target: CreatureRef, id: PowerId, amount: i32, applier: Option<CreatureRef>) -> Vec<Effect> {
        if amount == 0 || !self.creature(target).alive() {
            return vec![];
        }
        let mut out = vec![];
        let in_play = self.player.play.last().map(|c| c.uid);
        let amount = self.relic_modify_power_amount(target, id, amount, in_play);
        // ArtifactPower.cs: negate a debuff and spend a charge.
        if is_debuff(id) && amount > 0 && self.creature(target).power(PowerId::Artifact).is_some() {
            self.modify_power(target, PowerId::Artifact, -1);
            return vec![];
        }
        let exists = self.creature(target).power(id).is_some();
        if exists {
            if is_single(id) {
                return vec![];
            }
            out.extend(Power::new(id, 0).on_applied(target, amount));
            self.modify_power(target, id, amount);
        } else {
            let mut p = Power::new(id, amount);
            p.skip_next_tick = target == CreatureRef::Player && is_debuff(id);
            p.applier = applier;
            out.extend(p.on_applied(target, amount));
            self.creature_mut(target).powers.push(p);
        }
        // Hook.AfterPowerAmountChanged: Vicious watches Vulnerable the player applied.
        if applier == Some(CreatureRef::Player) {
            for p in &self.player.creature.powers {
                out.extend(p.after_power_applied_by_owner(id, amount));
            }
        }
        out
    }

    /// `PowerCmd.ModifyAmount`, including removal at zero.
    fn modify_power(&mut self, target: CreatureRef, id: PowerId, delta: i32) {
        let c = self.creature_mut(target);
        if let Some(p) = c.powers.iter_mut().find(|p| p.id == id) {
            p.amount += delta;
            if p.should_remove() {
                c.powers.retain(|p| p.id != id);
            }
        }
    }
}

impl Combat {
    /// What to do with a chosen card.
    fn choose(&mut self, then: Then, uid: u32) -> Vec<Effect> {
        let again = |then: Then| Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then, can_skip: true };
        match then {
            Then::Exhaust => vec![Effect::Exhaust { uid, ethereal: false }],
            Then::Upgrade => vec![Effect::Upgrade { uid }],
            Then::MoveTo(p) => vec![Effect::MoveCard { uid, to: p }],
            Then::FreeThisCombat => {
                if let Some(c) = self.find_card_mut(uid) {
                    c.cost_this_combat = Some(0);
                }
                vec![]
            }
            Then::ToHandFreeThisTurn => {
                if let Some(c) = self.find_card_mut(uid) {
                    c.cost_this_turn = Some(0);
                }
                vec![Effect::MoveCard { uid, to: Pile::Hand }]
            }
            Then::TakeOffer => vec![Effect::TakeOffer { uid }],
            Then::ExhaustMany => vec![Effect::Exhaust { uid, ethereal: false }, again(Then::ExhaustMany)],
            Then::DiscardThenDraw { picked } => {
                vec![Effect::MoveCard { uid, to: Pile::Discard }, again(Then::DiscardThenDraw { picked: picked + 1 })]
            }
        }
    }
}

fn filter_ok(f: CardFilter, c: &Card) -> bool {
    match f {
        CardFilter::Any => true,
        CardFilter::Type(t) => c.ty() == t,
        CardFilter::NotType(t) => c.ty() != t,
        CardFilter::PlayableAttack => c.ty() == CardType::Attack && !c.has(Keyword::Unplayable),
        CardFilter::CostsEnergy => c.local_cost() > 0 || c.def().x_cost,
    }
}

/// Shuffle in place, or impose the next scripted order. Scripted orders
/// match cards by (id, upgraded); cards the script does not mention go to
/// the bottom in their existing order.
fn shuffle_cards(cards: &mut Vec<Card>, script: &mut Script, rng: &mut crate::rng::Rng, log: &mut Vec<Vec<(CardId, bool)>>) {
    cards.sort_by_key(|c| (c.id, c.upgraded));
    match script.shuffles.pop_front() {
        Some(order) => {
            // Advance the stream as a real shuffle would, so later draws from
            // it (Stampede's pick) stay aligned with an unscripted run.
            let mut scratch: Vec<usize> = (0..cards.len()).collect();
            rng.shuffle(&mut scratch);
            let mut rest = std::mem::take(cards);
            for (id, up) in order {
                if let Some(i) = rest.iter().position(|c| c.id == id && c.upgraded == up) {
                    cards.push(rest.remove(i));
                }
            }
            cards.append(&mut rest);
        }
        None => rng.shuffle(cards),
    }
    log.push(cards.iter().map(|c| (c.id, c.upgraded)).collect());
}

/// Generatable Ironclad cards for a `GenPool`.
fn pool_cards(pool: GenPool) -> Vec<CardId> {
    IRONCLAD_POOL
        .iter()
        .copied()
        .filter(|id| {
            let d = crate::card::def(*id);
            d.generatable
                && match pool {
                    GenPool::Ironclad => true,
                    GenPool::IroncladAttacks => d.ty == CardType::Attack,
                    GenPool::IroncladSkills => d.ty == CardType::Skill,
                    GenPool::IroncladPowers => d.ty == CardType::Power,
                }
        })
        .collect()
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
