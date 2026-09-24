//! Combat state and the loop that resolves effects. `Combat/CombatState.cs`,
//! `Combat/CombatManager.cs`, `Commands/CreatureCmd.cs`, `Commands/CardPileCmd.cs`,
//! `Commands/CardCmd.cs`, `Commands/PowerCmd.cs`.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::card::{Affliction, Card, Tag, IRONCLAD_POOL};
use crate::effect::{AttackTargets, CardFilter, Effect, GenPool, Picked, Pile, Then};
use crate::ids::{CardId, MonsterId, PowerId};
use crate::monster::{Flags, Monster, RollCtx};
use crate::potion::{PotionId, Target as PotionTarget};
use crate::power::{instanced, is_debuff, is_debuff_for_amount, Power};
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
    pub fn power_mut(&mut self, id: PowerId) -> Option<&mut Power> {
        self.powers.iter_mut().find(|p| p.id == id)
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
    /// 1-based index into `EncounterModel.Slots`. Slot order is the order the
    /// game lists enemies in, which recorded targets index.
    pub slot: u8,
    /// Illusion: dead but will act (revive) on its next turn.
    pub reviving: bool,
    /// Fled the fight (`CreatureCmd.Escape`). Neither alive nor a corpse.
    pub escaped: bool,
}

impl Enemy {
    /// `Creature.IsPrimaryEnemy`: combat ends when no primary enemy lives.
    pub fn primary(&self) -> bool {
        self.creature.power(PowerId::Minion).is_none()
    }
    /// Takes part in the enemy turn, and still holds its slot.
    pub fn acts(&self) -> bool {
        (self.creature.alive() || self.reviving) && !self.escaped
    }

    /// `PowerModel.ShouldStopCombatFromEnding` for the powers that use it.
    /// Steam Eruption survives its owner's death (`ShouldPowerBeRemovedAfter\
    /// OwnerDeath` is false), which is how the blast still goes off.
    /// Adaptable (the Test Subject) outlives its owner the same way, which
    /// keeps the fight going while it respawns.
    fn stops_combat_ending(&self) -> bool {
        self.creature.power(PowerId::SteamEruption).is_some() || self.creature.power(PowerId::Adaptable).is_some()
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

/// What follows a fight in the run, which sets what the HP and potions it
/// leaves are worth (`env::terminal_reward`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum After {
    /// More of the act: what is left carries into it.
    #[default]
    Act,
    /// An act boss: the next act's Ancient heals the missing HP, or 80% of
    /// it under Weary Traveler (`AncientEventModel.BeforeEventStarted`).
    Ancient,
    /// The first of the last act's two bosses under Double Boss (A10): the
    /// second follows with no rest in between (`RunManager`, `RoomSet`).
    Boss,
    /// The run's last fight.
    End,
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
    /// Only Gremlin Merc's Thievery reads it.
    pub gold: i32,
}

/// A card in a recorded draw order: id, upgraded, and its enchantment with
/// whether it is spent. The enchantment tells otherwise equal copies apart:
/// Anger's clone carries a fresh Vigorous, the played original a spent one.
pub type ShuffleCard = (CardId, bool, Option<(crate::enchant::EnchantmentId, bool)>);

/// Recorded outcomes the replay harness forces instead of rolling:
/// shuffle results and starting enemy HP. Each shuffle consumes one entry;
/// once the queue is empty the RNG takes over again.
#[derive(Clone, Debug, Default)]
pub struct Script {
    /// Draw pile orders, top first.
    pub shuffles: VecDeque<Vec<ShuffleCard>>,
    /// Shuffles rolled with no order queued, since the last snapshot. The
    /// replayer rewinds when the game's order arrives after the sim needed
    /// it (a choice card played early, or a reshuffle an after-play hook
    /// triggers once the `play` record is already written).
    pub unscripted_shuffles: u32,
    /// Max HP per starting enemy, by index.
    pub enemy_hp: Vec<i32>,
    /// The recording's first snapshot is still to come. It shows the
    /// player's HP after turn 1's start has healed (Blood Vial) or hurt
    /// (Royal Poison), and the opening layout; both are taken once the sim
    /// settles there, which a choice that start opens (Gambling Chip)
    /// holds up.
    pub first_snapshot_pending: bool,
    /// Targets for random-target hits, as indices into the living enemies.
    pub random_targets: VecDeque<usize>,
    /// Cards the recording saw generated, in order. Random generation takes
    /// its picks from here (`take_generated`).
    pub generated: VecDeque<CardId>,
    /// Cards a random exhaust may pick, by (id, upgraded); matched, not ordered.
    pub random_exhausts: Vec<(CardId, bool)>,
    /// Cards put on offer from a snapshot because the game logged the pick
    /// only after the choice (Choices Paradox). Their `gen` record, still to
    /// come, must not force the next random card.
    pub adopted_offers: Vec<CardId>,
    /// Cards random generation rolled with nothing scripted, since the last
    /// snapshot. A `gen` record that arrives after the roll (Crossbow's turn
    /// start card follows `turn_start`) rewrites the newest of them.
    pub unforced: Vec<CardId>,
    /// Cards the recording saw deal damage since the last snapshot. A
    /// random auto-play (Stampede) picks one of them when it can.
    pub hit_cards: Vec<CardId>,
    /// Monsters the recording saw join since the last snapshot. A random
    /// spawn (the Fabricator's bot) takes the first one it can make.
    pub spawns: Vec<MonsterId>,
}

impl Script {
    /// The first recorded card a random generation from `pool` could have
    /// made. The game logs every `AddGeneratedCardToCombat`, so cards the
    /// sim makes by name (Personal Hive's Dazed after Jackpot's hit) sit
    /// ahead of the roll's own picks; they belong to earlier generations and
    /// are dropped with it.
    fn take_generated(&mut self, pool: &[CardId]) -> Option<CardId> {
        let i = self.generated.iter().position(|id| pool.contains(id))?;
        self.generated.drain(..i);
        self.generated.pop_front()
    }
}

/// The parts of `CombatManager.History` that cards and powers read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// HP the enemies have lost to anything, summed over the fight, and
    /// their HP when it began. Training's reward shaping reads these: a
    /// monster that dies and comes back (Test Subject, the Waterfall
    /// Giant's blast) still counts the HP it lost the first time.
    pub enemy_hp_lost: i64,
    pub enemy_start_hp: i64,
    /// Cards dropped into the draw pile at a random depth (Beckon). Nothing
    /// records the depth, so the replay harness adopts it from the recording.
    pub random_draw_inserts: u32,
    /// Enemies whose Skittish has been triggered by the attack in flight and
    /// are owed their block once every hit of it has landed.
    pub skittish_pending: Vec<usize>,
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
    /// A Skill has been played this turn, so Smoggy fogs the rest (Living Fog).
    pub skill_played_this_turn: bool,
    /// Cards the player chose to play this turn, auto-plays left out
    /// (`CardPlay.IsAutoPlay`): Brilliant Scarf, Pael's Eye.
    pub manual_plays_this_turn: u32,
    /// The last attack or skill whose play finished this turn, and the one
    /// from the turn before (History Course). Dupes do not count.
    pub last_card: Option<Card>,
    pub last_turn_card: Option<Card>,
    /// The card offer on screen makes its pick free this turn (the potions).
    pub offer_free: bool,
    /// The coming player turn is an extra one (Pael's Eye): the enemies keep
    /// their moves.
    pub extra_turn: bool,
    /// Cards the combat started with have uids `1..=deck_size`; the rest were
    /// made during it (`CardModel.DeckVersion` is null for those).
    pub deck_size: u32,
    /// The Disintegration a Curse of Knowledge offer carries.
    pub offer_disintegration: i32,
    /// Enemies whose HP was rolled again mid-fight (a hatched Tough Egg), for
    /// the replay to adopt like a fresh spawn's.
    pub hp_rerolled: Vec<usize>,
    /// Wounds Painful Stabs owes once the attack in flight is over.
    pub wounds_pending: u32,
    /// A Bound card has been played this turn (`ChainsOfBindingPower`).
    pub bound_played: bool,
    /// What the last `CreatureCmd.GainBlock` returned (Toric Toughness).
    pub last_block_gained: f64,
    /// Cards in play whose result pile Rebound moved to the top of the draw
    /// pile, decided as their play began.
    pub rebound: Vec<u32>,
    /// Attack and skill plays started this turn (Nostalgia).
    pub attack_skill_plays_this_turn: u32,
    /// Cards whose play Nostalgia sends to the top of the draw pile instead
    /// of the discard; decided as the play starts, like the game's result pile.
    pub nostalgia_top: Vec<u32>,
    /// `StranglePower`'s `amountsForPlayedCards`: for each card play in
    /// flight, the enemy index and the Strangle it had as the play began.
    pub strangle_pending: Vec<(u32, usize, i32)>,
    /// Thrumming Hatchets whose play finished this turn, and last turn.
    pub hatchets_played: Vec<u32>,
    pub hatchets_played_last_turn: Vec<u32>,
    /// The last card damage call: the card's uid and `TotalDamage +
    /// OverkillDamage` (Omnislice). `None` when the target was already dead.
    pub last_card_hit: Option<(u32, i32)>,
    /// Cards picked so far in an open `Then::Select`.
    pub selected: Vec<u32>,
    /// `History.CardPlaysFinished.Count()`: every play of every card this
    /// combat, auto-plays and replays included (Gold Axe).
    pub card_plays_finished: u32,
    /// Uids whose play finished this player turn and the one before, the
    /// enemy turn in between counting as the earlier one (Bolas).
    pub finished_this_turn: Vec<u32>,
    pub finished_last_turn: Vec<u32>,
    /// Damage the card play in progress has dealt, blocked and overkill
    /// included (`TotalDamage + OverkillDamage`, Fisticuffs), with the
    /// uid of the card that dealt it.
    pub card_dealt: (u32, i32),
    /// Entropy's picks so far, transformed together once all are in.
    pub transform_picks: Vec<u32>,
    /// Cards transformed at random (new uid, the card it replaced), and
    /// potion slots filled at random, since the last snapshot: the replay
    /// adopts what the game rolled for them.
    pub transformed: Vec<(u32, CardId)>,
    /// A transform the last snapshot showed not done yet. The snapshot
    /// could not say which picked card it was, so the replay may move it.
    pub transform_carried: bool,
    pub procured_potions: Vec<usize>,
    /// The open choice's options were drawn at random from the draw pile
    /// (Seeker Strike); the replay takes the game's instead.
    pub random_choice: bool,
    /// Hidden Gem's random pick and the replays it gave, until the next
    /// snapshot shows which card the game chose.
    pub gem_pick: Option<(u32, u32)>,
    /// Draw pile cards Stone Cracker upgraded at random, until the first
    /// snapshot shows which ones the game picked.
    pub cracked: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Combat {
    pub player: PlayerCombat,
    /// Indexed by `CreatureRef::Enemy`; indices are stable for the combat.
    pub enemies: Vec<Enemy>,
    /// Enemy indices in the game's slot order (`SortEnemiesBySlotName`):
    /// turn order and the order the player sees them in.
    pub order: Vec<usize>,
    /// `CombatState.RoundNumber`, starts at 1.
    pub round: u32,
    pub side: Side,
    pub asc: Ascension,
    /// Run gold. Nothing in combat spends it except Gremlin Merc's Thievery.
    pub gold: i32,
    pub rngs: CombatRngs,
    pub outcome: Option<Outcome>,
    pub pending: Option<Pending>,
    pub stats: Stats,
    /// The run's relics, with counters updated in place.
    pub relics: Vec<Relic>,
    pub room: RoomKind,
    /// What follows the fight; `FightSetup::combat` sets it.
    pub after: After,
    /// Potion slots. Using a potion empties its slot.
    pub potions: Vec<Option<PotionId>>,
    pub script: Script,
    /// Every shuffle result this combat, top first (the recorder's view).
    /// Only the replay reads it, so clones share it until the next shuffle.
    pub shuffle_log: Arc<Vec<Vec<(CardId, bool)>>>,
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
            gold: 0,
        })
    }

    pub fn with_setup(setup: &Setup) -> Self {
        Self::with_script(setup, Script::default())
    }

    /// `with_setup` with forced shuffles and enemy HP.
    pub fn with_script(setup: &Setup, mut script: Script) -> Self {
        let Setup { deck, hp, max_hp, max_energy, enemies, asc, seed, gold, .. } = *setup;
        let mut rngs = CombatRngs::new(seed);
        let mut next_uid = 1;
        // A copy, not a fresh card: the deck's cards carry their
        // enchantment, and Tezcatara's Ember its free cost.
        let mut draw: Vec<Card> = deck
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c.uid = next_uid;
                c.extra_damage = 0.0;
                next_uid += 1;
                c
            })
            .collect();
        let mut shuffle_log = Arc::default();
        shuffle_cards(&mut draw, &mut script, &mut rngs.shuffle, &mut shuffle_log);
        apply_shuffle_order(&mut draw, true);

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
            order: vec![],
            round: 1,
            side: Side::Player,
            asc,
            gold,
            rngs,
            outcome: None,
            pending: None,
            stats: Stats::default(),
            relics: setup.relics.to_vec(),
            room: setup.room,
            after: After::Act,
            potions: setup.potions.to_vec(),
            script,
            shuffle_log,
            queue: VecDeque::new(),
            next_uid,
            started: false,
        };
        c.stats.deck_size = deck.len() as u32;
        for spec in enemies {
            c.spawn(spec.id, spec.flags);
        }
        // MonsterModel.AfterAddedToRoom, which runs once every starting
        // monster is in the room.
        for i in 0..c.enemies.len() {
            if is_segment(c.enemies[i].monster.id) {
                c.even_segment_hp(i);
            }
            if c.enemies[i].monster.id == MonsterId::Rocket {
                // Rocket.AfterAddedToRoom: the player stands between the halves.
                let mut p = Power::new(PowerId::Surrounded, 1);
                p.applier = Some(CreatureRef::Enemy(i));
                c.player.creature.powers.push(p);
            }
        }
        for (i, &hp) in c.script.enemy_hp.iter().enumerate().take(c.enemies.len()) {
            let e = &mut c.enemies[i].creature;
            e.max_hp = hp;
            e.hp = hp;
        }
        c.stats.enemy_start_hp = c.enemies.iter().map(|e| e.creature.hp as i64).sum();
        c.started = true;
        c.galvanize_deck();
        let pre = c.relic_before_combat_start();
        c.queue.extend(pre);
        c.queue.push_back(Effect::StartTurn(Side::Player));
        c.run();
        c
    }

    /// `DecimillipedeSegment.AfterAddedToRoom`: an even max HP no other segment
    /// has, stepping up by 2 and wrapping to the minimum past the maximum.
    fn even_segment_hp(&mut self, i: usize) {
        let (lo, hi) = Monster::hp_range(self.enemies[i].monster.id, self.asc);
        let mut hp = self.enemies[i].creature.max_hp;
        hp += hp % 2;
        let taken = |hp: i32| self.enemies.iter().enumerate().any(|(j, e)| j != i && e.creature.max_hp == hp);
        while taken(hp) {
            hp += 2;
            if hp > hi {
                hp = lo;
            }
        }
        let e = &mut self.enemies[i].creature;
        e.max_hp = hp;
        e.hp = hp;
    }

    /// `CombatState.CreateCreature` + `MonsterModel.AfterAddedToRoom`: roll
    /// HP, apply innate powers and block, add to the enemy list.
    fn spawn(&mut self, id: MonsterId, flags: Flags) {
        let (lo, hi) = Monster::hp_range(id, self.asc);
        let hp = roll_unique_hp(lo, hi, &self.enemies, &mut self.rngs.niche);
        let mut creature = Creature { hp: (hp - i32::from(flags.hp_reduction)).max(1), max_hp: hp, block: 0, powers: vec![] };
        let me = CreatureRef::Enemy(self.enemies.len());
        for (pid, amount) in Monster::innate_powers(id, self.asc) {
            let mut p = Power::new(pid, amount);
            p.applier = Some(me);
            creature.powers.push(p);
        }
        // ToughEgg.AfterAddedToRoom: Hatch counts one more turn when it is
        // laid on the enemy's turn. Ovicopter then makes it a Minion.
        if id == MonsterId::ToughEgg {
            let hatch = if self.side == Side::Player { 1 } else { 2 };
            creature.powers.push(Power::new(PowerId::Hatch, hatch));
            creature.powers.push(Power::new(PowerId::Minion, 1));
        }
        // Axebot.AfterAddedToRoom: a respawn carries only the stock it has left.
        if let Some(stock) = flags.stock {
            creature.powers.retain(|p| p.id != PowerId::Stock);
            if stock > 0 {
                let mut p = Power::new(PowerId::Stock, i32::from(stock));
                p.applier = Some(me);
                creature.powers.push(p);
            }
        }
        // AfterAddedToRoom block (Cubex) goes through CreatureCmd.GainBlock,
        // which returns early until IsInProgress is set, and the starting
        // monsters are added before that. Only mid-combat spawns get it.
        creature.block = if self.started { Monster::innate_block(id) } else { 0 };
        let mut monster = Monster::new(id, self.asc, flags);
        // CombatManager.AfterCreatureAdded: roll at once during the player's turn.
        if self.started && self.side == Side::Player {
            let ctx = self.roll_ctx_for(&monster, self.enemies.len());
            monster.roll_move(&mut self.rngs.monster_ai, ctx);
        }
        // `EncounterModel.GetNextSlot`: the lowest free slot, which is often
        // not the end. Fogmog holds "fogmog" with "illusion" ahead of it, and
        // Living Fog holds the last of six with five bomb slots ahead.
        // `TwoTailedRat.CallForBackup` and `Ovicopter.LayEggsMove` are the
        // exceptions: they inline `Slots.LastOrDefault`, so their summons
        // arrive at the back.
        let last = matches!(id, MonsterId::TwoTailedRat | MonsterId::ToughEgg);
        let slot = if flags.slot != 0 { flags.slot } else { self.free_slot(last) };
        self.enemies.push(Enemy { creature, monster, slot, reviving: false, escaped: false });
        let idx = self.enemies.len() - 1;
        let at = self.order.iter().position(|&j| self.enemies[j].slot > slot).unwrap_or(self.order.len());
        self.order.insert(at, idx);
        // CreatureCmd.Add runs Hook.AfterCreatureAddedToCombat; the starting
        // monsters never go through it.
        if self.started {
            let subs = self.relic_after_enemy_added(idx);
            self.push_front_all(subs);
        }
    }

    /// The free slot a spawn takes: the lowest, or the highest for the one
    /// summon that reaches for `Slots.LastOrDefault` instead.
    fn free_slot(&self, last: bool) -> u8 {
        let taken: Vec<u8> = self.present_enemies().map(|i| self.enemies[i].slot).collect();
        let mut free = (1..=self.encounter_slots() as u8).filter(|n| !taken.contains(n));
        if last { free.next_back().unwrap_or(1) } else { free.next().unwrap_or(1) }
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// What enemy `i`'s move graph may read about the rest of the fight.
    fn roll_ctx(&self, i: usize) -> RollCtx {
        self.roll_ctx_for(&self.enemies[i].monster, i)
    }

    /// `roll_ctx` for a monster not yet in the enemy list (a fresh spawn).
    fn roll_ctx_for(&self, m: &Monster, i: usize) -> RollCtx {
        let me = self.enemies.get(i).map(|e| &e.creature);
        RollCtx {
            asleep: me.is_some_and(|c| c.power(PowerId::Asleep).is_some()),
            can_summon: self.can_summon(m, i),
            slumbering: self.enemies.get(i).is_some_and(|e| e.creature.power(PowerId::Slumber).is_some()),
            // GetTeammatesOf counts the monster itself, even one not yet placed.
            living_allies: self.living_enemies().count() + usize::from(i >= self.enemies.len()),
            allies_alive: self.living_enemies().filter(|&j| j != i).count(),
            below_half: me.is_some_and(|c| c.hp < c.max_hp / 2),
        }
    }

    /// `TwoTailedRat.CanSummon`: off cooldown, under the three-call cap, a
    /// free slot left, and no peer already rolled the call this turn.
    fn can_summon(&self, m: &Monster, i: usize) -> bool {
        if m.id != MonsterId::TwoTailedRat {
            return false;
        }
        if m.vars.turns_until_summonable > 0 || m.vars.call_for_backup_count >= 3 {
            return false;
        }
        if self.free_slots() == 0 {
            return false;
        }
        !self.enemies.iter().enumerate().any(|(j, e)| {
            j != i && e.acts() && e.monster.next_move_name() == Some("CALL_FOR_BACKUP_MOVE")
        })
    }

    /// `EncounterModel.GetNextSlot`: room left for one more monster. A slot
    /// frees up when the monster in it leaves, which is why Living Fog keeps
    /// making bombs all fight.
    fn free_slots(&self) -> usize {
        self.encounter_slots().saturating_sub(self.present_enemies().count())
    }

    /// `EncounterModel.Slots`. Only the two encounters that summon into named
    /// slots have a list longer than their starting monsters, and each is
    /// identified by the monster doing the summoning. Everything else is held
    /// to what the observation can carry.
    fn encounter_slots(&self) -> usize {
        let named = self.enemies.iter().find_map(|e| match e.monster.id {
            // bomb1..bomb5 plus livingFog.
            MonsterId::LivingFog => Some(6),
            // first..fifth.
            MonsterId::TwoTailedRat => Some(5),
            // egg1..egg5 plus ovicopter.
            MonsterId::Ovicopter => Some(6),
            // illusion plus obscura.
            MonsterId::TheObscura => Some(2),
            // bot1, bot2, fabricator, bot3, bot4.
            MonsterId::Fabricator => Some(5),
            _ => None,
        });
        named.unwrap_or(crate::encode::MAX_ENEMIES).min(crate::encode::MAX_ENEMIES)
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

    /// Enemies still in the game's list, in slot order: alive, or dead but
    /// about to revive (Illusion).
    pub fn present_enemies(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied().filter(|&i| self.enemies[i].acts())
    }

    /// Living enemies in slot order.
    pub fn living_enemies(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied().filter(|&i| self.enemies[i].creature.alive())
    }

    /// Every card the player has in combat, all piles.
    pub fn all_cards(&self) -> impl Iterator<Item = &Card> {
        let p = &self.player;
        p.hand.iter().chain(&p.draw).chain(&p.discard).chain(&p.exhaust).chain(&p.play)
    }

    pub fn find_card(&self, uid: u32) -> Option<&Card> {
        self.all_cards().find(|c| c.uid == uid)
    }

    pub(crate) fn find_card_mut(&mut self, uid: u32) -> Option<&mut Card> {
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
    /// ones (Spiked Gauntlets, then the late Corruption, Free Attack,
    /// Brilliant Scarf). X-cost cards report `-1`.
    pub fn cost(&self, card: &Card) -> i32 {
        let local = card.local_cost();
        if local < 0 || card.def().x_cost {
            return local;
        }
        if self.player.creature.powers.iter().any(|p| p.free_card(card.ty())) || self.relic_makes_free(card) {
            return 0;
        }
        // CuriousPower.cs: a power, so its hook runs ahead of the relics'.
        let curious = self.player.creature.power_amount(PowerId::Curious);
        let local = if card.ty() == CardType::Power && local > 0 { (local - curious).max(0) } else { local };
        let local = local + self.relic_cost_additive(card);
        // TangledPower.cs: every attack is afflicted with Entangled (+amount).
        if card.ty() == CardType::Attack {
            return local + self.player.creature.power_amount(PowerId::Tangled);
        }
        local
    }

    /// `Hook.ShouldPlay` for the player's own card: Ringing allows only the
    /// first play each turn, Velvet Choker six, and Smoggy blocks anything the
    /// fog has settled on.
    fn hook_allows_play(&self, card: &Card) -> bool {
        if self.player.creature.power(PowerId::Ringing).is_some() && self.stats.cards_played_this_turn > 0 {
            return false;
        }
        // Two curses veto from hand: Enthralled until it is itself played,
        // Normality once three cards have gone down this turn.
        let vetoed = |c: &Card| match c.id {
            CardId::Enthralled => card.id != CardId::Enthralled,
            CardId::Normality => self.stats.cards_played_this_turn >= 3,
            _ => false,
        };
        if self.player.hand.iter().any(vetoed) || !self.relic_allows_play() {
            return false;
        }
        // SlothPower.ShouldPlay: only `amount` cards a turn.
        if self.player.creature.power(PowerId::Sloth).is_some_and(|p| p.data >= p.amount) {
            return false;
        }
        // ChainsOfBindingPower.ShouldPlay: one Bound card a turn.
        if card.affliction == Some(Affliction::Bound) && self.stats.bound_played {
            return false;
        }
        !(card.smogged && self.player.creature.powers.iter().any(|p| p.blocks_smogged()))
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
    pub(crate) fn can_play(&self, card: &Card) -> bool {
        if card.has(Keyword::Unplayable) || !self.hook_allows_play(card) {
            return false;
        }
        // Clash.IsPlayable: only with nothing but attacks in hand. NoLivingAllies:
        // a single-player fight has no one else for an AnyAlly card to target.
        if (card.id == CardId::Clash && self.player.hand.iter().any(|k| k.ty() != CardType::Attack))
            || card.def().target == TargetType::AnyAlly
        {
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
                    Then::Select { done, .. } => {
                        let subs = self.finish_select(done);
                        self.push_front_all(subs);
                    }
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
                // SurroundedPower.BeforePotionUsed.
                if let Some(t) = target {
                    self.face_crab(t);
                }
                let mut subs = id.on_use(self, target);
                subs.push(Effect::AfterPotionUsed);
                self.push_front_all(subs);
            }
            Action::PlayCard { hand_idx, target } => {
                assert!(self.pending.is_none() && self.side == Side::Player, "not accepting card plays");
                let card = self.player.hand[hand_idx].clone();
                assert!(self.can_play(&card), "illegal card play");
                let target = self.resolve_target(&card, target);
                let paid = self.pay_for(hand_idx);
                self.stats.manual_plays_this_turn += 1;
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

    /// `CardModel.SpendResources` for a hand card: its cost, or all the
    /// energy for an X card, which captures X. Returns what was paid.
    pub(crate) fn pay_for(&mut self, hand_idx: usize) -> i32 {
        let card = &self.player.hand[hand_idx];
        if card.def().x_cost {
            let paid = self.player.energy;
            self.player.hand[hand_idx].captured_x = paid + self.relic_x_bonus();
            self.player.energy = 0;
            paid
        } else {
            let paid = self.cost(card);
            self.player.energy -= paid;
            paid
        }
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
            // Some combos never terminate in the game either (Pillage
            // drawing Strikes that Hellraiser auto-plays for 0 damage). The
            // real game hangs; we score it as a loss so training never
            // dies on it.
            if resolved >= 100_000 || self.queue.len() >= 100_000 {
                self.queue.clear();
                self.outcome = Some(Outcome::Lost);
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
                self.push_front_all(vec![Effect::AttackHits { dealer, base, left: hits, targets, props, card }]);
            }
            // AttackCommand.Execute's hit loop: before each hit, stop if the
            // attacker died (Thorns can kill a monster mid-flurry) or no
            // target is left, and recompute the living targets.
            Effect::AttackHits { dealer, base, left, targets, props, card } => {
                let ts: Vec<CreatureRef> = if left == 0 || !self.creature(dealer).alive() {
                    vec![]
                } else {
                    match &targets {
                        AttackTargets::One(t) => vec![*t],
                        AttackTargets::AllOpponents => self.opponents_of(dealer),
                        AttackTargets::RandomOpponent => {
                            let opts = self.opponents_of(dealer);
                            match self.script.random_targets.pop_front() {
                                Some(t) if t < opts.len() => vec![opts[t]],
                                _ => self.rngs.targets.pick(&opts).copied().into_iter().collect(),
                            }
                        }
                    }
                };
                if ts.is_empty() {
                    self.push_front_all(vec![Effect::EndAttack { dealer, card, props }]);
                    return;
                }
                let next = Effect::AttackHits { dealer, base, left: left - 1, targets, props, card };
                if ts.len() > 1 {
                    let mut subs = self.damage_many(&ts, base, props, Some(dealer), card);
                    subs.push(next);
                    self.push_front_all(subs);
                    self.check_win();
                } else {
                    let mut subs: Vec<Effect> =
                        ts.into_iter().map(|t| Effect::Damage { target: t, amount: base, props, dealer: Some(dealer), card }).collect();
                    subs.push(next);
                    self.push_front_all(subs);
                }
            }
            Effect::EndAttack { dealer, card, props } => {
                let mut subs = vec![Effect::AfterAttack];
                // VigorPower.AfterAttack: whoever swung spends it, and the
                // Terror Eel swings with it too.
                if props.is_powered() {
                    let v = self.creature(dealer).power_amount(PowerId::Vigor);
                    if v > 0 {
                        subs.push(Effect::ApplyPower { target: dealer, id: PowerId::Vigor, amount: -v, applier: None });
                    }
                }
                // GigantificationPower.AfterAttack: one charge per card attack.
                if dealer == CreatureRef::Player && card.is_some() && props.is_powered() {
                    if self.player.creature.power(PowerId::Gigantification).is_some() {
                        subs.push(Effect::DecrementPower { target: dealer, id: PowerId::Gigantification });
                    }
                }
                self.push_front_all(subs);
            }
            Effect::Damage { target, amount, props, dealer, card } => {
                if !self.creature(target).alive() {
                    if card.is_some() {
                        self.stats.last_card_hit = None;
                    }
                    return;
                }
                let subs = self.damage(target, amount, props, dealer, card);
                self.push_front_all(subs);
                self.check_win();
            }
            Effect::DamageAllEnemies { amount, props, dealer } => {
                let targets: Vec<CreatureRef> = self.living_enemies().map(CreatureRef::Enemy).collect();
                let subs = self.damage_many(&targets, amount, props, Some(dealer), None);
                self.push_front_all(subs);
                self.check_win();
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
            Effect::ApplyPowerAllEnemies { id, amount, applier } => {
                let subs: Vec<Effect> = self
                    .living_enemies()
                    .map(|i| Effect::ApplyPower { target: CreatureRef::Enemy(i), id, amount, applier })
                    .collect();
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
                // Scripted: the first hand card matching a recorded exhaust.
                let scripted = self.player.hand.iter().find_map(|c| {
                    let i = self.script.random_exhausts.iter().position(|&(id, up)| c.id == id && c.upgraded == up)?;
                    uids.contains(&c.uid).then_some((c.uid, i))
                });
                let pick = match scripted {
                    Some((uid, i)) => {
                        self.script.random_exhausts.remove(i);
                        Some(uid)
                    }
                    None => self.rngs.card_selection.pick(&uids).copied(),
                };
                if let Some(uid) = pick {
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
                self.card_entered_combat(&mut card);
                self.put_card(card, to);
            }
            Effect::CloneCard { uid, to } => {
                // CardScope.CloneCard keeps everything the card carries
                // (upgrade, enchantment, cost changes, banked damage); only
                // the exhaust-on-next-play flag is reset.
                if let Some(src) = self.find_card(uid) {
                    let mut card = src.clone();
                    card.uid = self.new_uid();
                    card.exhaust_on_next_play = false;
                    let cost = card.cost_this_turn;
                    self.card_entered_combat(&mut card);
                    card.cost_this_turn = cost;
                    self.put_card(card, to);
                }
            }
            Effect::GenerateRandom { pool, count, to, free_this_turn, distinct, upgraded } => {
                let subs = self
                    .roll_cards(pool, count, distinct)
                    .into_iter()
                    .map(|id| Effect::GenerateCard { id, upgraded: upgraded && Card::new(0, id, false).upgradable(), to, free_this_turn })
                    .collect();
                self.push_front_all(subs);
            }
            Effect::GenerateRandomFreeThisCombat { pool, count, to, distinct } => {
                for id in self.roll_cards(pool, count, distinct) {
                    let mut card = Card::new(self.new_uid(), id, false);
                    card.cost_this_combat = Some(0);
                    self.card_entered_combat(&mut card);
                    self.put_card(card, to);
                }
            }
            Effect::UpgradeAll { except } => self.for_each_card(|c| {
                if c.uid != except && c.upgradable() {
                    c.upgraded = true;
                }
            }),
            // LocalCostModifier with IsReduceOnly: it only counts where it
            // lowers the cost the earlier modifiers left.
            Effect::CapHandCost { cost, this_combat } => {
                for c in &mut self.player.hand {
                    if c.def().x_cost || c.base_cost() < 0 {
                        continue;
                    }
                    if this_combat {
                        c.cost_this_combat = Some(c.cost_this_combat.unwrap_or(c.base_cost()).min(cost));
                        if let Some(t) = c.cost_this_turn {
                            c.cost_this_turn = Some(t.min(cost));
                        }
                    } else {
                        c.cost_this_turn = Some(c.local_cost().min(cost));
                    }
                }
            }
            Effect::GrowDamage { id, amount } => self.for_each_card(|c| {
                if c.id == id {
                    c.extra_damage += amount;
                }
            }),
            Effect::LoseMaxHp { target, amount, from_card } => {
                let c = self.creature(target);
                let new_max = c.max_hp - amount;
                let mut subs = vec![];
                if new_max < c.hp {
                    let mut props = ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED);
                    if from_card {
                        props = props.or(ValueProp::MOVE);
                    }
                    subs.push(Effect::Damage { target, amount: (c.hp - new_max) as f64, props, dealer: None, card: None });
                }
                subs.push(Effect::SetMaxHp { target, max_hp: new_max.max(1) });
                self.push_front_all(subs);
            }
            Effect::ApplyInstanced { target, id, amount, data } => {
                if amount != 0 && self.creature(target).alive() {
                    let mut p = Power::new(id, amount);
                    p.data = data;
                    p.applier = Some(target);
                    self.creature_mut(target).powers.push(p);
                }
            }
            Effect::Choose { from, filter, then, can_skip } => {
                let pile = match from {
                    Pile::Hand => &self.player.hand,
                    Pile::Discard => &self.player.discard,
                    Pile::Exhaust => &self.player.exhaust,
                    Pile::DrawTop | Pile::DrawBottom | Pile::DrawRandom => &self.player.draw,
                };
                let mut options: Vec<u32> = pile
                    .iter()
                    .filter(|c| filter_ok(filter, c) && !self.stats.selected.contains(&c.uid))
                    .map(|c| c.uid)
                    .collect();
                if let Then::TransformPick { .. } = then {
                    options.retain(|u| !self.stats.transform_picks.contains(u));
                }
                if !options.is_empty() {
                    self.pending = Some(Pending { options, then, can_skip });
                } else if let Then::DiscardThenDraw { picked } = then {
                    // Gambler's Brew with the hand emptied: draw what was discarded.
                    if picked > 0 {
                        self.queue.push_front(Effect::Draw { count: picked, from_hand_draw: false });
                    }
                } else if let Then::Select { done, .. } = then {
                    let subs = self.finish_select(done);
                    self.push_front_all(subs);
                }
            }
            Effect::ChooseFromRandomDraw { count } => {
                let mut uids: Vec<u32> = self.player.draw.iter().map(|c| c.uid).collect();
                self.rngs.card_selection.shuffle(&mut uids);
                uids.truncate(count as usize);
                if !uids.is_empty() {
                    self.pending = Some(Pending { options: uids, then: Then::MoveTo(Pile::Hand), can_skip: false });
                    self.stats.random_choice = true;
                }
            }
            Effect::OfferRandom { pool, count, free, retain } => {
                let mut opts = pool_cards(pool);
                self.rngs.card_generation.shuffle(&mut opts);
                // Scripted: the recording only shows the card taken, so make
                // sure it is on offer.
                if let Some(id) = self.script.take_generated(&opts) {
                    opts.retain(|&o| o != id);
                    opts.insert(0, id);
                }
                self.player.offer.clear();
                for id in opts.into_iter().take(count as usize) {
                    let mut card = Card::new(self.new_uid(), id, false);
                    card.retain_added = retain;
                    self.player.offer.push(card);
                }
                self.stats.offer_free = free;
                let options: Vec<u32> = self.player.offer.iter().map(|c| c.uid).collect();
                if !options.is_empty() {
                    self.pending = Some(Pending { options, then: Then::TakeOffer, can_skip: !retain });
                }
            }
            Effect::TakeOffer { uid } => {
                if let Some(i) = self.player.offer.iter().position(|c| c.uid == uid) {
                    let mut card = self.player.offer.remove(i);
                    // KnowledgeDemon.IChoosable.OnChosen: the pick's power
                    // lands and the card never enters a pile.
                    let curse = match card.id {
                        CardId::Disintegration => Some((PowerId::Disintegration, self.stats.offer_disintegration)),
                        CardId::MindRot => Some((PowerId::MindRot, 1)),
                        CardId::Sloth => Some((PowerId::Sloth, 3)),
                        CardId::WasteAway => Some((PowerId::WasteAway, 1)),
                        _ => None,
                    };
                    if let Some((id, amount)) = curse {
                        self.player.offer.clear();
                        let me = CreatureRef::Player;
                        self.queue.push_front(Effect::ApplyPower { target: me, id, amount, applier: Some(me) });
                        return;
                    }
                    if self.stats.offer_free {
                        card.cost_this_turn = Some(0);
                    }
                    self.card_entered_combat(&mut card);
                    self.put_card(card, Pile::Hand);
                }
                self.player.offer.clear();
            }
            Effect::OfferCurse { cards, disintegration } => {
                self.player.offer.clear();
                for id in cards {
                    let card = Card::new(self.new_uid(), id, false);
                    self.player.offer.push(card);
                }
                self.stats.offer_free = false;
                self.stats.offer_disintegration = disintegration;
                let options = self.player.offer.iter().map(|c| c.uid).collect();
                self.pending = Some(Pending { options, then: Then::TakeOffer, can_skip: false });
            }
            Effect::CostThisCombat { uid, delta } => {
                if let Some(c) = self.find_card_mut(uid) {
                    c.cost_this_combat = Some(c.cost_this_combat.unwrap_or(c.base_cost()) + delta);
                }
            }
            Effect::MonsterStep { me, step } => {
                if let CreatureRef::Enemy(i) = me {
                    let subs = self.monster_step(i, step);
                    self.push_front_all(subs);
                }
            }
            // ReattachPower.DoReattach: only while another segment still lives.
            Effect::Reattach { target, hp } => {
                if let CreatureRef::Enemy(i) = target {
                    if !self.other_segments_dead(i) {
                        let e = &mut self.enemies[i];
                        e.creature.hp = hp.min(e.creature.max_hp);
                        e.reviving = false;
                    }
                }
            }
            // ToughEgg.HatchMove: every power but Minion goes, then a fresh
            // HP roll (`Rng.NextInt` excludes the maximum).
            Effect::Hatch { target } => {
                if let CreatureRef::Enemy(i) = target {
                    let (lo, hi) = if self.asc.has(crate::types::AscensionLevel::ToughEnemies) { (20, 23) } else { (19, 22) };
                    let hp = lo + self.rngs.niche.next_int((hi - lo) as usize) as i32;
                    let c = &mut self.enemies[i].creature;
                    c.powers.retain(|p| p.id == PowerId::Minion);
                    c.max_hp = hp;
                    c.hp = hp;
                    self.stats.hp_rerolled.push(i);
                }
            }
            Effect::CloneToHand { uid, ethereal } => {
                if let Some(mut card) = self.find_card(uid).cloned() {
                    card.uid = self.new_uid();
                    card.ethereal_added |= ethereal;
                    card.dupe = false;
                    self.card_entered_combat(&mut card);
                    self.put_card(card, Pile::Hand);
                }
            }
            Effect::RelicStep { id, step } => {
                let subs = self.relic_step(id, step);
                self.push_front_all(subs);
            }
            Effect::TurnDraw => {
                let subs = self.turn_draw();
                self.push_front_all(subs);
            }
            Effect::ExtraPlayerTurn => {
                // SwitchSides with a player taking an extra turn: the turn
                // number moves on, the round does not.
                self.begin_player_turn();
                self.stats.extra_turn = true;
                self.queue.push_front(Effect::StartTurn(Side::Player));
            }
            Effect::Shuffle => {
                // CardPileCmd.Shuffle: discard then draw, shuffled together.
                let mut cards = std::mem::take(&mut self.player.discard);
                cards.append(&mut self.player.draw);
                shuffle_cards(&mut cards, &mut self.script, &mut self.rngs.shuffle, &mut self.shuffle_log);
                apply_shuffle_order(&mut cards, false);
                self.player.draw = cards;
                let subs = self.after_shuffle();
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
                self.decide_result_pile(uid);
                let Some(card) = self.find_card(uid) else { return };
                // GeneratePlayCount: replays, then Hook.ModifyCardPlayCount, then
                // each modifying power is told (OneTwoPunch, Duplication decrement).
                // GeneratePlayCount asks the enchantment first, then the powers.
                let replays = card.enchantment.map_or(card.replay, |e| e.play_count(card.replay));
                let mut plays = 1 + replays;
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
                // ThrowingAxe: the first card each combat is played twice.
                if let Some(r) = self.relics.iter_mut().find(|r| r.id == crate::relic::RelicId::ThrowingAxe && !r.used) {
                    r.used = true;
                    plays += 1;
                }
                let mut subs: Vec<Effect> = (0..plays).map(|_| Effect::CardPlayIter { uid, target, paid }).collect();
                subs.push(Effect::FinishCardPlay { uid });
                self.push_front_all(subs);
            }
            Effect::CardPlayIter { uid, target, paid } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                self.stats.cards_played_this_turn += 1;
                if matches!(card.ty(), CardType::Attack | CardType::Skill) {
                    self.stats.attack_skill_plays_this_turn += 1;
                }
                self.stats.card_dealt = (uid, 0);
                // SlothPower.BeforeCardPlayed, SurroundedPower.BeforeCardPlayed.
                if let Some(p) = self.player.creature.power_mut(PowerId::Sloth) {
                    p.data += 1;
                }
                if let Some(t) = target {
                    self.face_crab(t);
                }
                let mut subs = self.before_card_played(&card);
                subs.extend(self.relic_before_card_played(&card, paid));
                subs.extend(card.on_play(self, target));
                // CardModel.OnPlayWrapper runs the enchantment after the card.
                if card.enchantment.is_some() {
                    subs.push(Effect::EnchantOnPlay { uid, target });
                }
                subs.push(Effect::AfterCardPlayed { uid });
                self.push_front_all(subs);
            }
            Effect::EnchantOnPlay { uid, target } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                let Some(e) = card.enchantment else { return };
                let subs = e.on_play(self, uid, target, card.def().target);
                if let Some(c) = self.find_card_mut(uid) {
                    if let Some(e) = c.enchantment.as_mut() {
                        e.after_played();
                    }
                }
                self.push_front_all(subs);
            }
            Effect::AfterCardPlayed { uid } => {
                let Some(card) = self.find_card(uid).cloned() else { return };
                // CombatHistory.CardPlayFinished, which Thrumming Hatchet reads.
                if card.id == CardId::ThrummingHatchet && !self.stats.hatchets_played.contains(&uid) {
                    self.stats.hatchets_played.push(uid);
                }
                // History.CardPlayFinished, logged just before the hook.
                self.stats.card_plays_finished += 1;
                self.stats.finished_this_turn.push(uid);
                let subs = self.after_card_played(&card);
                self.push_front_all(subs);
            }
            Effect::FinishCardPlay { uid } => {
                // Rupture's deferred Strength.
                let owed = std::mem::take(&mut self.stats.rupture_pending);
                if owed > 0 {
                    self.queue.push_front(Effect::ApplyPower {
                        target: CreatureRef::Player,
                        id: PowerId::Strength,
                        amount: owed,
                        applier: Some(CreatureRef::Player),
                    });
                }
                // CardModel.GetResultPileTypeForCardPlay + Corruption, then
                // Nostalgia, which only redirects a discard.
                let nostalgia = self.stats.nostalgia_top.iter().position(|&u| u == uid).map(|i| self.stats.nostalgia_top.remove(i));
                if let Some(card) = self.take_card(uid) {
                    let corruption = self.player.creature.power(PowerId::Corruption).is_some();
                    if matches!(card.ty(), CardType::Attack | CardType::Skill) && !card.dupe {
                        self.stats.last_card = Some(card.clone());
                    }
                    if card.ty() == CardType::Power || card.dupe {
                        // Powers and dupes leave combat (PileType.None).
                    } else if card.has(Keyword::Exhaust)
                        || card.exhaust_on_next_play
                        || (corruption && card.ty() == CardType::Skill)
                    {
                        self.player.exhaust.push(card);
                        let subs = self.after_card_exhausted(uid, false);
                        self.push_front_all(subs);
                    } else if self.stats.rebound.contains(&uid) || nostalgia.is_some() {
                        self.player.draw.insert(0, card);
                    } else {
                        self.player.discard.push(card);
                    }
                }
                self.stats.rebound.retain(|&u| u != uid);
                let subs = self.relic_after_hand_emptied();
                self.push_front_all(subs);
            }
            Effect::CardStep { uid, target, step } => {
                // Rampage.OnPlay grows its damage right after the hit, once
                // per play and before any after-play hook (Razor Tooth) can
                // upgrade the increase.
                if let Some(c) = self.find_card_mut(uid).filter(|c| c.id == CardId::Rampage && step == 1) {
                    c.extra_damage += c.vars().magic;
                }
                let Some(card) = self.find_card(uid).cloned() else { return };
                let subs = card.step(self, target, step);
                self.push_front_all(subs);
            }
            Effect::AutoPlay { uid, force_exhaust } => {
                let subs = self.auto_play(uid, force_exhaust);
                self.push_front_all(subs);
            }
            Effect::AutoPlayFromDrawTop { count, force_exhaust } => {
                self.queue.push_front(Effect::AutoPlayTake { left: count, force_exhaust, taken: vec![] });
            }
            Effect::AutoPlayTake { left, force_exhaust, mut taken } => {
                // All cards leave the draw pile before any is played, so a
                // played card's draws do not eat the next one.
                let mut subs = vec![];
                for n in (0..left).rev() {
                    if self.reshuffle_if_needed() {
                        let hooks = self.after_shuffle();
                        // Stratagem's pick is a choice the game awaits
                        // before it takes the next card.
                        if self.player.creature.power(PowerId::Stratagem).is_some() {
                            subs.extend(hooks);
                            subs.push(Effect::AutoPlayTake { left: n + 1, force_exhaust, taken });
                            self.push_front_all(subs);
                            return;
                        }
                        subs.extend(hooks);
                    }
                    if self.player.draw.is_empty() {
                        break;
                    }
                    let card = self.player.draw.remove(0);
                    taken.push(card.uid);
                    self.player.play.push(card);
                }
                subs.extend(taken.into_iter().map(|uid| Effect::AutoPlay { uid, force_exhaust }));
                self.push_front_all(subs);
            }
            Effect::ApplyBomb { turns, damage } => {
                let me = CreatureRef::Player;
                let before = self.player.creature.powers.len();
                let subs = self.apply_power(me, PowerId::TheBomb, turns, Some(me));
                if self.player.creature.powers.len() > before {
                    if let Some(p) = self.player.creature.powers.last_mut() {
                        p.data = damage;
                    }
                }
                self.push_front_all(subs);
            }
            Effect::Die { target } => {
                if target == CreatureRef::Player && self.player.creature.alive() {
                    self.player.creature.hp = 0;
                    if self.save_player() == Some(true) {
                        self.queue.push_front(Effect::AfterPotionUsed);
                    }
                    self.check_win();
                } else {
                    self.queue.push_front(Effect::Kill { target });
                }
            }
            Effect::AutoPlayRandomAttack => {
                let uids: Vec<u32> =
                    self.player.hand.iter().filter(|c| filter_ok(CardFilter::PlayableAttack, c)).map(|c| c.uid).collect();
                let rolled = self.rngs.shuffle.pick(&uids).copied();
                let scripted = self.script.hit_cards.iter().enumerate().find_map(|(i, &id)| {
                    let uid = self.player.hand.iter().find(|c| c.id == id && uids.contains(&c.uid))?.uid;
                    Some((i, uid))
                });
                let pick = match scripted {
                    Some((i, uid)) => {
                        self.script.hit_cards.remove(i);
                        Some(uid)
                    }
                    None => rolled,
                };
                if let Some(uid) = pick {
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
            // BeatDown.OnPlay: `StableShuffle` on the Shuffle stream, then the
            // first `count`. Each is auto-played at a random enemy if it
            // needs one, which `auto_play` rolls.
            Effect::AutoPlayDiscardAttacks { count } => {
                let mut uids: Vec<u32> = self
                    .player
                    .discard
                    .iter()
                    .filter(|c| c.ty() == CardType::Attack && !c.has(Keyword::Unplayable))
                    .map(|c| c.uid)
                    .collect();
                let mut subs = vec![];
                for _ in 0..count {
                    let Some(uid) = self.pick_hit_or_random(&uids) else { break };
                    uids.retain(|&u| u != uid);
                    subs.push(Effect::AutoPlay { uid, force_exhaust: false });
                }
                self.push_front_all(subs);
            }
            // Catastrophe.OnPlay: one card at a time, a playable one if the
            // draw pile has any, and no reshuffle when it runs dry.
            Effect::AutoPlayFromDraw { count } => {
                if count == 0 {
                    return;
                }
                let playable: Vec<u32> =
                    self.player.draw.iter().filter(|c| !c.has(Keyword::Unplayable)).map(|c| c.uid).collect();
                let any: Vec<u32> = self.player.draw.iter().map(|c| c.uid).collect();
                let pick = if playable.is_empty() { self.pick_hit_or_random(&any) } else { self.pick_hit_or_random(&playable) };
                let mut subs: Vec<Effect> = pick.map(|uid| Effect::AutoPlay { uid, force_exhaust: false }).into_iter().collect();
                subs.push(Effect::AutoPlayFromDraw { count: count - 1 });
                self.push_front_all(subs);
            }
            // Anointed.OnPlay: `TakeRandom` on CombatCardSelection, as many
            // as the hand has room for.
            Effect::PullRaresToHand => {
                let room = MAX_HAND.saturating_sub(self.player.hand.len());
                let mut uids: Vec<u32> = self
                    .player
                    .draw
                    .iter()
                    .filter(|c| c.def().rarity == crate::types::CardRarity::Rare)
                    .map(|c| c.uid)
                    .collect();
                self.rngs.card_selection.shuffle(&mut uids);
                let subs = uids.into_iter().take(room).map(|uid| Effect::MoveCard { uid, to: Pile::Hand }).collect();
                self.push_front_all(subs);
            }
            // HiddenGem.OnPlay: a playable draw pile card that is not a
            // curse and not already replayed, an attack, skill or power if
            // there is one, picked on CombatCardSelection.
            Effect::ReplayRandomDrawCard { replays } => {
                let eligible = |c: &Card| {
                    !c.has(Keyword::Unplayable)
                        && c.ty() != CardType::Curse
                        && c.enchantment.map_or(c.replay, |e| e.play_count(c.replay)) < 1
                };
                let all: Vec<u32> = self.player.draw.iter().filter(|c| eligible(c)).map(|c| c.uid).collect();
                let main: Vec<u32> = self
                    .player
                    .draw
                    .iter()
                    .filter(|c| eligible(c) && matches!(c.ty(), CardType::Attack | CardType::Skill | CardType::Power))
                    .map(|c| c.uid)
                    .collect();
                let options = if main.is_empty() { all } else { main };
                if let Some(&uid) = self.rngs.card_selection.pick(&options) {
                    if let Some(c) = self.find_card_mut(uid) {
                        c.replay += replays;
                    }
                    self.stats.gem_pick = Some((uid, replays));
                }
            }
            // EntropyPower.AfterPlayerTurnStart: CardSelectCmd.FromHand for
            // exactly `count`, which takes the whole hand without asking
            // when that is no more than `count`.
            Effect::TransformFromHand { count } => {
                let hand: Vec<u32> = self.player.hand.iter().map(|c| c.uid).collect();
                if count == 0 || hand.is_empty() {
                    return;
                }
                if hand.len() <= count as usize {
                    let subs = self.transforms_in_hand_order(&hand);
                    self.push_front_all(subs);
                } else {
                    self.stats.transform_picks.clear();
                    self.pending = Some(Pending { options: hand, then: Then::TransformPick { left: count }, can_skip: false });
                }
            }
            // CardCmd.TransformToRandom: a fresh card from the original's
            // transform pool, in its place, rolled on CombatCardSelection.
            Effect::TransformRandom { uid } => {
                let Some(i) = self.player.hand.iter().position(|c| c.uid == uid) else { return };
                let options = crate::card::transform_options(self.player.hand[i].id);
                let Some(&id) = self.rngs.card_selection.pick(&options) else { return };
                let was = self.player.hand[i].id;
                let mut card = Card::new(self.new_uid(), id, false);
                self.card_entered_combat(&mut card);
                self.stats.transformed.push((card.uid, was));
                self.player.hand[i] = card;
            }
            Effect::FillPotionSlots => {
                let mut subs = vec![];
                while self.potions.contains(&None) {
                    let open = self.potions.iter().filter(|p| p.is_none()).count();
                    subs.extend(self.procure_random_potion(false));
                    if self.potions.iter().filter(|p| p.is_none()).count() == open {
                        break;
                    }
                }
                self.push_front_all(subs);
            }
            Effect::ProcureRandomPotion => {
                let subs = self.procure_random_potion(true);
                self.push_front_all(subs);
            }
            Effect::GainGold { amount } => {
                let subs = self.relic_gain_gold(amount);
                self.push_front_all(subs);
            }
            // EncounterModel.GetNextSlot: a summon with no slot left is
            // simply skipped, which is how Living Fog stops at five bombs.
            Effect::SpawnMonster { id, flags } => {
                if self.free_slots() > 0 {
                    self.spawn(id, flags);
                    // TwoTailedRat.CallForBackup keeps the cap in step across
                    // every rat, called ones included.
                    if id == MonsterId::TwoTailedRat {
                        let next = self.enemies.iter().map(|e| e.monster.vars.call_for_backup_count).max().unwrap_or(0) + 1;
                        for e in &mut self.enemies {
                            if e.monster.id == MonsterId::TwoTailedRat {
                                e.monster.vars.call_for_backup_count = next;
                            }
                        }
                    }
                }
            }
            Effect::FabricateBot { fabricator, aggro } => self.fabricate_bot(fabricator, aggro),
            Effect::BlockMonsters { id, amount } => {
                let subs: Vec<Effect> = self
                    .living_enemies()
                    .filter(|&j| self.enemies[j].monster.id == id)
                    .map(|j| Effect::GainBlock {
                        target: CreatureRef::Enemy(j),
                        amount: amount as f64,
                        props: ValueProp::UNPOWERED,
                        card: None,
                    })
                    .collect();
                self.push_front_all(subs);
            }
            Effect::ApplyPowerAllies { source, id, amount } => {
                let subs: Vec<Effect> = self
                    .living_enemies()
                    .filter(|&j| CreatureRef::Enemy(j) != source)
                    .map(|j| Effect::ApplyPower { target: CreatureRef::Enemy(j), id, amount, applier: Some(source) })
                    .collect();
                self.push_front_all(subs);
            }
            Effect::SetMaxHp { target, max_hp } => {
                let c = self.creature_mut(target);
                c.max_hp = max_hp.max(1);
                c.hp = c.hp.min(c.max_hp);
            }
            Effect::ReviveAt { target, max_hp } => {
                if let CreatureRef::Enemy(i) = target {
                    let e = &mut self.enemies[i];
                    e.creature.max_hp = max_hp;
                    e.creature.hp = max_hp;
                    e.reviving = false;
                }
            }
            Effect::UpgradeWithers => {
                let p = &mut self.player;
                for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
                    for c in pile.iter_mut().filter(|c| c.id == CardId::Wither) {
                        c.extra_damage += 3.0;
                    }
                }
            }
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
            // CreatureCmd.Kill: the bomb and the giant finish themselves off.
            Effect::Kill { target } => {
                self.creature_mut(target).hp = 0;
                if let CreatureRef::Enemy(i) = target {
                    let subs = self.on_enemy_death(i);
                    self.push_front_all(subs);
                }
                self.check_win();
            }
            // CreatureCmd.Escape: gone from the fight, but not dead, so no
            // death hooks and no reward.
            Effect::Steal { thief } => {
                // ThieveryPower.Steal: capped by what you are actually
                // carrying, and banked so Surprise can hand it on.
                let taken = self.creature(thief).power_amount(PowerId::Thievery).min(self.gold).max(0);
                self.gold -= taken;
                if let Some(p) = self.creature_mut(thief).power_mut(PowerId::Thievery) {
                    p.data += taken;
                }
            }
            Effect::Escape { target } => {
                if let CreatureRef::Enemy(i) = target {
                    self.enemies[i].escaped = true;
                    self.enemies[i].creature.hp = 0;
                }
                self.check_win();
            }
            // WaterfallGiant.AboutToBlowMove.
            Effect::ArmSteamEruption { target } => {
                if let CreatureRef::Enemy(i) = target {
                    let amount = self.enemies[i].creature.power_amount(PowerId::SteamEruption);
                    self.enemies[i].monster.arm_death_blow(amount);
                    self.enemies[i].creature.powers.retain(|p| p.id != PowerId::SteamEruption);
                }
            }
            Effect::StartTurn(side) => self.start_turn(side),
            Effect::EndPlayerTurn => self.end_player_turn(),
            Effect::TurnEndInHand => self.turn_end_in_hand(),
            Effect::FlushHand => self.flush_hand(),
            Effect::EnemyAct(i) => {
                // Creature.TakeTurn skips monsters spawned since the last side switch.
                if self.enemies[i].acts() && !self.enemies[i].monster.spawned_this_turn {
                    // The Forgotten's Dread reads its own Dexterity.
                    self.enemies[i].monster.vars.own_dex = self.enemies[i].creature.power_amount(PowerId::Dexterity);
                    let subs = self.enemies[i].monster.perform(CreatureRef::Enemy(i), self.asc);
                    self.push_front_all(subs);
                }
            }
            Effect::EndEnemyTurn => self.end_enemy_turn(),
            Effect::FinishEnemyTurn => self.finish_enemy_turn(),
            Effect::SideTurnEndEarly(side) => self.side_turn_end_early(side),
            Effect::AfterAttack => {
                let owed = std::mem::take(&mut self.stats.skittish_pending);
                // PainfulStabsPower.AfterAttack: the Wounds for the hits that got through.
                let wounds = std::mem::take(&mut self.stats.wounds_pending);
                let subs = (0..wounds)
                    .map(|_| Effect::GenerateCard { id: CardId::Wound, upgraded: false, to: Pile::Discard, free_this_turn: false })
                    .chain(owed
                    .into_iter()
                    .filter_map(|i| {
                        let amount = self.enemies.get(i)?.creature.power(PowerId::Skittish)?.amount as f64;
                        Some(Effect::GainBlock {
                            target: CreatureRef::Enemy(i),
                            amount,
                            props: ValueProp::UNPOWERED,
                            card: None,
                        })
                    }))
                    .collect();
                self.push_front_all(subs);
            }
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
        // Creature.BeforeTurnStart for the side starting its turn.
        if side == Side::Player {
            self.player.creature.powers.iter_mut().for_each(Power::before_turn_start);
        }
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
                // Enemies roll their next move (PrepareForNextTurn), unless
                // this is an extra turn and they still owe the last one.
                let extra = std::mem::take(&mut self.stats.extra_turn);
                for i in 0..self.enemies.len() {
                    if self.enemies[i].acts() && !extra {
                        let ctx = self.roll_ctx(i);
                        self.enemies[i].monster.roll_move(&mut self.rngs.monster_ai, ctx);
                    }
                }
                // Creature.AfterTurnStart: block clears except on player turn 1,
                // unless a listener prevents it. Hook.ShouldClearBlock takes the
                // first preventer in CombatState.IterateHookListeners order:
                // powers (Barricade keeps everything) before relics (Sturdy
                // Clamp trims to 10 in AfterPreventingBlockClear).
                if self.player.turn != 1 && self.should_clear_block(CreatureRef::Player) {
                    let kept = if self.relic_keeps_block() { 10 } else { 0 };
                    self.player.creature.block = self.player.creature.block.min(kept);
                }
                // Hook.AfterBlockCleared runs for every creature starting its
                // turn, even when nothing was cleared (turn 1, Barricade,
                // Sturdy Clamp): player powers, then relics.
                for p in &self.player.creature.powers {
                    subs.extend(p.after_block_cleared(CreatureRef::Player));
                }
                self.tick_toric();
                subs.extend(self.relic_after_block_cleared());
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
                // Hook.BeforeHandDraw: player powers, then relics.
                for p in &self.player.creature.powers {
                    subs.extend(p.before_hand_draw());
                }
                subs.extend(self.relic_before_hand_draw());
                // ThrummingHatchet.BeforeHandDraw, after the relics: back to
                // hand from wherever it went if it was played last turn.
                for &uid in &self.stats.hatchets_played_last_turn {
                    if !self.player.hand.iter().any(|c| c.uid == uid) {
                        subs.push(Effect::MoveCard { uid, to: Pile::Hand });
                    }
                }
                // Bolas.BeforeHandDraw, a card hook, so after the relics: a
                // Bolas played last turn comes back to hand.
                let bolas: Vec<u32> = self
                    .all_cards()
                    .filter(|k| k.id == CardId::Bolas && self.stats.finished_last_turn.contains(&k.uid))
                    .filter(|k| !self.player.hand.iter().any(|h| h.uid == k.uid))
                    .map(|k| k.uid)
                    .collect();
                subs.extend(bolas.into_iter().map(|uid| Effect::MoveCard { uid, to: Pile::Hand }));
                subs.push(Effect::TurnDraw);
                // Imbued.AfterAutoPrePlayPhaseEntered: on turn one it plays
                // itself out of hand, which the bottom-of-pile rule feeds.
                if self.player.turn <= 1 {
                    let imbued: Vec<u32> = self
                        .player
                        .draw
                        .iter()
                        .chain(&self.player.hand)
                        .filter(|c| c.enchantment.is_some_and(|e| e.autoplays_on_first_turn()))
                        .map(|c| c.uid)
                        .collect();
                    subs.extend(imbued.into_iter().map(|uid| Effect::AutoPlay { uid, force_exhaust: false }));
                }
                // Hook.AfterPlayerTurnStart.
                subs.extend(self.collect_powers(|p, owner, _| {
                    if owner == CreatureRef::Player {
                        p.after_player_turn_start(owner)
                    } else {
                        vec![]
                    }
                }));
                // RollingBoulderPower.AfterPlayerTurnStart: `SetAmount(Amount + 5)`
                // once the hit is out; the queued hit keeps the old amount.
                for p in self.player.creature.powers.iter_mut().filter(|p| p.id == PowerId::RollingBoulder) {
                    p.amount += 5;
                }
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
        if side == Side::Player {
            // Hook.AfterAutoPrePlayPhaseEntered: powers, then relics.
            subs.extend(self.player.creature.powers.iter().flat_map(|p| p.after_auto_pre_play()));
            subs.extend(self.relic_auto_pre_play());
        }
        if side == Side::Enemy {
            subs.extend(self.order.iter().map(|&i| Effect::EnemyAct(i)));
            subs.push(Effect::EndEnemyTurn);
        }
        self.push_front_all(subs);
    }

    /// The hand draw in `SetupPlayerTurn`, once the `BeforeHandDraw` hooks
    /// have resolved: `ModifyHandDraw`, Innate cards to the top on turn 1,
    /// then the draw.
    fn turn_draw(&mut self) -> Vec<Effect> {
        let draw = self.player.creature.powers.iter().fold(BASE_HAND_DRAW, |n, p| p.modify_hand_draw(n));
        let mut draw = self.relic_modify_hand_draw(draw);
        if self.player.turn == 1 {
            let innate: Vec<usize> =
                self.player.draw.iter().enumerate().filter(|(_, c)| c.has(Keyword::Innate)).map(|(i, _)| i).collect();
            let n = innate.len() as u32;
            for (k, i) in innate.into_iter().enumerate() {
                let c = self.player.draw.remove(i);
                self.player.draw.insert(k, c);
            }
            draw = draw.max(n).min(MAX_HAND as u32);
        }
        vec![Effect::Draw { count: draw, from_hand_draw: true }]
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
        subs.extend(self.collect_powers(|p, owner, _| p.before_side_turn_end_very_early(owner, Side::Player)));
        subs.push(Effect::SideTurnEndEarly(Side::Player));
        self.push_front_all(subs);
    }

    /// `DoTurnEnd`: ethereal cards exhaust, then turn-end-in-hand effects.
    fn turn_end_in_hand(&mut self) {
        // Regret.BeforeSideTurnEnd reads the hand before anything exhausts.
        let held = self.player.hand.len() as f64;
        let ethereal: Vec<u32> = self.player.hand.iter().filter(|c| c.has(Keyword::Ethereal)).map(|c| c.uid).collect();
        let mut subs: Vec<Effect> = ethereal.into_iter().map(|uid| Effect::Exhaust { uid, ethereal: true }).collect();
        let unblockable = ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED).or(ValueProp::MOVE);
        for c in &self.player.hand {
            // Burn and Infection deal damage; Beckon, Bad Luck and Regret
            // take HP through block.
            let hurt = match c.id {
                CardId::Burn | CardId::Infection | CardId::Decay | CardId::Toxic | CardId::Wither => {
                    Some((c.vars().damage, ValueProp::UNPOWERED.or(ValueProp::MOVE)))
                }
                CardId::Beckon | CardId::BadLuck => Some((c.vars().hp_loss, unblockable)),
                CardId::Regret => Some((held, unblockable)),
                _ => None,
            };
            if let Some((amount, props)) = hurt {
                subs.push(Effect::Damage { target: CreatureRef::Player, amount, props, dealer: None, card: Some(c.uid) });
            }
            // Doubt and Shame sit out the tick of the debuff they land, which
            // is what every debuff applied to the player does here.
            if let Some(id) = match c.id {
                CardId::Doubt => Some(PowerId::Weak),
                CardId::Shame => Some(PowerId::Frail),
                _ => None,
            } {
                subs.push(Effect::ApplyPower { target: CreatureRef::Player, id, amount: 1, applier: None });
            }
        }
        subs.push(Effect::FlushHand);
        self.push_front_all(subs);
    }

    /// `EndPlayerTurnPhaseTwoInternal` + `SwitchSides`: discard the hand,
    /// cleanup, end-of-turn hooks, then the enemy turn.
    fn flush_hand(&mut self) {
        // SlumberingEssence.BeforeFlush, on the cards still in hand.
        for c in self.player.hand.iter_mut().filter(|c| c.enchantment.is_some_and(|e| e.cheapens_in_hand())) {
            c.cost_this_combat = Some((c.cost_this_combat.unwrap_or(c.base_cost()) - 1).max(0));
        }
        if self.relic_should_flush() && self.player.creature.powers.iter().all(|p| p.should_flush()) {
            let (retain, flush): (Vec<Card>, Vec<Card>) = self.player.hand.drain(..).partition(|c| c.has(Keyword::Retain));
            self.player.hand = retain;
            self.player.discard.extend(flush);
        }
        self.end_of_turn_cleanup();
        let mut subs = self.after_side_turn_end(Side::Player);
        // SwitchFromPlayerToEnemySide asks Hook.ShouldTakeExtraTurn first.
        if self.relic_take_extra_turn() {
            subs.push(Effect::ExtraPlayerTurn);
        } else {
            subs.push(Effect::StartTurn(Side::Enemy));
        }
        self.push_front_all(subs);
    }

    /// `EndEnemyTurnInternal` + `SwitchSides` back to the player.
    /// `BeforeSideTurnEndVeryEarly` then `Early`, the same two passes the
    /// player's turn gets: Lagavulin sheds its shell before Plating tops it
    /// back up, and every plated enemy blocks up for the coming turn.
    fn end_enemy_turn(&mut self) {
        let mut subs = self.collect_powers(|p, owner, _| p.before_side_turn_end_very_early(owner, Side::Enemy));
        subs.push(Effect::SideTurnEndEarly(Side::Enemy));
        self.push_front_all(subs);
    }

    /// The `Early` pass, collected after `VeryEarly` has resolved: Lagavulin
    /// has already shed its shell by the time Plating would top it up.
    fn side_turn_end_early(&mut self, side: Side) {
        let mut subs = self.collect_powers(|p, owner, _| p.before_side_turn_end_early(owner, side));
        if side == Side::Player {
            subs.extend(self.relic_before_side_turn_end_early());
            subs.extend(self.bombs_before_turn_end());
            subs.extend(self.relic_before_side_turn_end());
            self.unbind();
            subs.push(Effect::TurnEndInHand);
        } else {
            subs.push(Effect::FinishEnemyTurn);
        }
        self.push_front_all(subs);
    }

    /// `TheBombPower.BeforeSideTurnEnd` on the player's turn, per instance:
    /// count down, or go off at 1. Each instance is its own power, so the
    /// countdown and removal happen here rather than through id-keyed
    /// effects; nothing reads the bomb between now and its blast.
    fn bombs_before_turn_end(&mut self) -> Vec<Effect> {
        let mut out = vec![];
        self.player.creature.powers.retain_mut(|p| {
            if p.id != PowerId::TheBomb {
                return true;
            }
            if p.amount > 1 {
                p.amount -= 1;
                return true;
            }
            out.push(Effect::DamageAllEnemies { amount: p.data as f64, props: ValueProp::UNPOWERED, dealer: CreatureRef::Player });
            false
        });
        out
    }

    /// `Hook.AfterShuffle`: the player's powers (Stratagem), then relics.
    pub(crate) fn after_shuffle(&self) -> Vec<Effect> {
        let mut out = vec![];
        // StratagemPower.AfterShuffle: pick `amount` cards from the draw pile
        // into hand (min and max both `amount`, so it cannot be skipped).
        if let Some(n) = self.player.creature.power(PowerId::Stratagem).map(|p| p.amount.clamp(0, u8::MAX as i32) as u8) {
            let (from, filter) = (Pile::DrawTop, CardFilter::Any);
            let then = Then::Select { from, filter, left: n, optional: false, done: Picked::ToHand };
            out.push(Effect::Choose { from, filter, then, can_skip: false });
        }
        out.extend(self.relic_after_shuffle());
        out
    }

    /// A `Then::Select` closing: act on every card picked, in pick order.
    fn finish_select(&mut self, done: Picked) -> Vec<Effect> {
        std::mem::take(&mut self.stats.selected)
            .into_iter()
            .map(|uid| match done {
                Picked::Exhaust => Effect::Exhaust { uid, ethereal: false },
                Picked::ToHand => Effect::MoveCard { uid, to: Pile::Hand },
            })
            .collect()
    }

    fn finish_enemy_turn(&mut self) {
        self.end_of_turn_cleanup();
        let mut subs = self.after_side_turn_end(Side::Enemy);
        self.round += 1;
        self.begin_player_turn();
        subs.push(Effect::StartTurn(Side::Player));
        self.push_front_all(subs);
    }

    /// `PlayerCombatState.IncrementTurnNumber` and the per-turn history.
    fn begin_player_turn(&mut self) {
        self.player.turn += 1;
        let s = &mut self.stats;
        s.exhausted_this_turn = 0;
        s.hp_lost_this_turn = false;
        s.block_plays_this_turn.clear();
        s.cards_played_this_turn = 0;
        s.skill_played_this_turn = false;
        s.manual_plays_this_turn = 0;
        s.last_turn_card = s.last_card.take();
        s.attack_skill_plays_this_turn = 0;
        s.hatchets_played_last_turn = std::mem::take(&mut s.hatchets_played);
        s.finished_last_turn = std::mem::take(&mut s.finished_this_turn);
    }

    /// `PlayerCombatState.EndOfTurnCleanup` over every card in combat.
    fn end_of_turn_cleanup(&mut self) {
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            for c in pile.iter_mut() {
                c.end_of_turn_cleanup();
                // SmoggyPower.AfterSideTurnEnd clears every Smog it laid down.
                c.smogged = false;
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
        for &i in &self.order {
            let e = &mut self.enemies[i];
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
        for &i in &self.order {
            let e = &self.enemies[i];
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
        } else if self.enemies.iter().any(|e| e.stops_combat_ending()) {
            // Hook.ShouldStopCombatFromEnding: the giant still owes a blast.
            // Surprise needs no such guard here, because its gremlins are
            // already in the list by the time this runs.
        } else if !self.enemies.iter().any(|e| e.creature.alive() && e.primary()) {
            self.outcome = Some(Outcome::Won);
            self.relic_after_victory();
        }
    }

    // ---- card play hooks ----------------------------------------------------

    /// `Hook.BeforeCardPlayed`: Free Attack decrement, Stomp cost reduction.
    fn before_card_played(&mut self, card: &Card) -> Vec<Effect> {
        let mut out = vec![];
        if card.affliction == Some(Affliction::Bound) && !card.dupe {
            self.stats.bound_played = true;
        }
        // StranglePower.BeforeCardPlayed: remember the amount as the card
        // begins, so the card that applies or stacks it is hit by the old one.
        for (i, e) in self.enemies.iter().enumerate() {
            let strangle = e.creature.power(PowerId::Strangle).filter(|p| p.applier == Some(CreatureRef::Player));
            if let Some(p) = strangle.filter(|_| e.creature.alive()) {
                self.stats.strangle_pending.push((card.uid, i, p.amount));
            }
        }
        if card.ty() == CardType::Attack {
            if self.player.creature.power(PowerId::FreeAttack).is_some() {
                out.push(Effect::DecrementPower { target: CreatureRef::Player, id: PowerId::FreeAttack });
            }
            // Stomp.BeforeCardPlayed is a hook on the Stomp itself, so it
            // fires wherever the card is sitting, not only in hand.
            let p = &mut self.player;
            for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
                for c in pile.iter_mut().filter(|c| c.id == CardId::Stomp) {
                    let cur = c.cost_this_turn.unwrap_or(c.base_cost());
                    c.cost_this_turn = Some((cur - 1).max(0));
                }
            }
        }
        out
    }

    /// `Hook.AfterCardPlayed`.
    /// `SmoggyPower.AfterCardPlayed` / `AfterCardEnteredCombat`: once a Skill
    /// has gone off this turn, every Skill in combat is fogged over, including
    /// ones drawn or made afterwards.
    fn spread_smog(&mut self) {
        if self.player.creature.powers.iter().all(|p| !p.blocks_smogged()) {
            return;
        }
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            for c in pile.iter_mut().filter(|c| c.ty() == CardType::Skill) {
                c.smogged = true;
            }
        }
    }

    fn after_card_played(&mut self, card: &Card) -> Vec<Effect> {
        let mut out = vec![];
        for p in &mut self.player.creature.powers {
            out.extend(p.after_card_played(CreatureRef::Player, card.ty(), card.id, card.upgraded));
        }
        out.extend(self.relic_after_card_played(card));
        if card.ty() == CardType::Skill {
            self.stats.skill_played_this_turn = true;
        }
        if self.stats.skill_played_this_turn {
            self.spread_smog();
        }
        let strangled: Vec<(usize, i32)> =
            self.stats.strangle_pending.iter().filter(|s| s.0 == card.uid).map(|&(_, i, n)| (i, n)).collect();
        self.stats.strangle_pending.retain(|s| s.0 != card.uid);
        for (i, e) in self.enemies.iter_mut().enumerate() {
            let alive = e.creature.alive();
            for p in &mut e.creature.powers {
                // StranglePower.AfterCardPlayed: unblockable, unpowered, for
                // the amount it had as this card began.
                if let Some(&(_, n)) = strangled.iter().find(|s| s.0 == i).filter(|_| p.id == PowerId::Strangle && alive) {
                    out.push(Effect::Damage {
                        target: CreatureRef::Enemy(i),
                        amount: n as f64,
                        props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED),
                        dealer: None,
                        card: None,
                    });
                }
                out.extend(p.after_card_played(CreatureRef::Enemy(i), card.ty(), card.id, card.upgraded));
            }
        }
        for i in self.living_enemies().collect::<Vec<_>>() {
            let me = CreatureRef::Enemy(i);
            let c = &self.enemies[i].creature;
            // CurlUpPower.AfterCardPlayed: once the attack that hit is done.
            if let Some(p) = c.power(PowerId::CurlUp).filter(|p| p.data == card.uid as i32) {
                out.push(Effect::GainBlock { target: me, amount: p.amount as f64, props: ValueProp::UNPOWERED, card: None });
                out.push(Effect::RemovePower { target: me, id: PowerId::CurlUp });
            }
            // VitalSparkPower.AfterCardPlayed: every skill is Tainted while it
            // lives, and playing one hands the player that much Tainted.
            if let Some(p) = c.power(PowerId::VitalSpark).filter(|_| card.ty() == CardType::Skill) {
                out.push(Effect::ApplyPower { target: CreatureRef::Player, id: PowerId::Tainted, amount: p.amount, applier: None });
            }
        }
        // GalvanicPower.AfterCardPlayed: each Galvanic hurts you for a
        // Galvanized card, by its own amount.
        if card.affliction == Some(Affliction::Galvanized) {
            for i in self.living_enemies().collect::<Vec<_>>() {
                if let Some(amount) = self.enemies[i].creature.power(PowerId::Galvanic).map(|p| p.amount) {
                    out.push(Effect::Damage {
                        target: CreatureRef::Player,
                        amount: amount as f64,
                        props: ValueProp::UNPOWERED.or(ValueProp::MOVE),
                        dealer: None,
                        card: None,
                    });
                }
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
        // An AnyAlly card finds no ally to aim at and is not played either.
        if card.has(Keyword::Unplayable) || !self.hook_allows_play(&card) || card.def().target == TargetType::AnyAlly {
            // MoveToResultPileWithoutPlaying. AutoPlayFromDrawPile sets
            // ExhaustOnNextPlay before it plays anything, so Havoc burns a
            // card it could not play rather than discarding it, and burning
            // it goes through CardCmd.Exhaust: Feel No Pain sees it.
            if card.dupe {
                self.take_card(uid);
                return vec![];
            }
            if force_exhaust || card.exhaust_on_next_play || card.has(Keyword::Exhaust) {
                return vec![Effect::Exhaust { uid, ethereal: false }];
            }
            if let Some(c) = self.take_card(uid) {
                self.player.discard.push(c);
            }
            return vec![];
        }
        let target = match card.def().target {
            TargetType::AnyEnemy => {
                let opts: Vec<usize> = self.living_enemies().collect();
                let scripted = self.script.random_targets.pop_front().filter(|&t| t < opts.len()).map(|t| opts[t]);
                match scripted.or_else(|| self.rngs.targets.pick(&opts).copied()) {
                    Some(i) => Some(CreatureRef::Enemy(i)),
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

    /// One card from `uids` for a random auto-play (Beat Down, Catastrophe).
    /// The recording names the cards that dealt damage since the snapshot,
    /// so the first candidate it names wins, and the run of hit records
    /// that card left (one per hit) is spent with it. Otherwise a roll on
    /// CombatCardSelection, which a replay can re-roll; the game uses the
    /// Shuffle stream, which the sim never shares with it.
    fn pick_hit_or_random(&mut self, uids: &[u32]) -> Option<u32> {
        let hit = self.script.hit_cards.iter().enumerate().find_map(|(i, &id)| {
            let uid = uids.iter().copied().find(|&u| self.find_card(u).is_some_and(|k| k.id == id))?;
            Some((i, id, uid))
        });
        match hit {
            Some((i, id, uid)) => {
                // One play's worth of hit records: its hits, once per enemy for
                // an attack on all of them. Two copies of a card played in a row
                // leave two runs of records, not one.
                let card = self.find_card(uid).cloned();
                let per_play = card.map_or(1, |k| {
                    let targets = if k.def().target == TargetType::AllEnemies { self.living_enemies().count() } else { 1 };
                    k.vars().hits.max(1) as usize * targets.max(1)
                });
                let run = self.script.hit_cards[i..].iter().take_while(|&&h| h == id).count().min(per_play);
                self.script.hit_cards.drain(i..i + run);
                Some(uid)
            }
            None => self.rngs.card_selection.pick(uids).copied(),
        }
    }

    /// `CardCmd.Transform` sorts its transformations by pile and index, so
    /// several picked cards are transformed left to right.
    fn transforms_in_hand_order(&self, uids: &[u32]) -> Vec<Effect> {
        self.player.hand.iter().filter(|c| uids.contains(&c.uid)).map(|c| Effect::TransformRandom { uid: c.uid }).collect()
    }

    /// `PotionCmd.TryToProcure` of a random potion
    /// (`PotionFactory.CreateRandomPotionInCombat`: a rarity roll, then a
    /// pick among what can be made in combat; `CreateRandomPotionOutOfCombat`
    /// without that filter, which Entropic Brew uses even in a fight). Sozu
    /// refuses it, a full belt has no room, and Belt Buckle takes back its
    /// Dexterity once a potion arrives.
    fn procure_random_potion(&mut self, in_combat: bool) -> Vec<Effect> {
        use crate::potion::{Rarity, ALL};
        let roll = self.rngs.potion_generation.next_float(1.0);
        let rarity = if roll <= 0.1 {
            Rarity::Rare
        } else if roll <= 0.35 {
            Rarity::Uncommon
        } else {
            Rarity::Common
        };
        let options: Vec<PotionId> =
            ALL.iter().copied().filter(|p| p.rarity() == rarity && (!in_combat || p.generatable_in_combat())).collect();
        let Some(&id) = self.rngs.potion_generation.pick(&options) else { return vec![] };
        if self.has_relic(crate::relic::RelicId::Sozu) {
            return vec![];
        }
        let Some(slot) = self.potions.iter().position(Option::is_none) else { return vec![] };
        self.potions[slot] = Some(id);
        self.stats.procured_potions.push(slot);
        self.relic_after_potion_procured()
    }

    // ---- piles -------------------------------------------------------------

    /// Returns true when a shuffle happened.
    pub(crate) fn reshuffle_if_needed(&mut self) -> bool {
        if self.player.draw.is_empty() && !self.player.discard.is_empty() {
            // CardPileCmd.Shuffle: discard becomes the draw pile.
            let mut cards = std::mem::take(&mut self.player.discard);
            shuffle_cards(&mut cards, &mut self.script, &mut self.rngs.shuffle, &mut self.shuffle_log);
            apply_shuffle_order(&mut cards, false);
            self.player.draw = cards;
            return true;
        }
        false
    }

    /// `CardPileCmd.Draw`, one card per call: reshuffle if needed, take the
    /// top, then the per-card hooks (Hellraiser auto-plays a drawn Strike).
    /// The game awaits those hooks inside its draw loop, so the auto-played
    /// card is already in the discard pile if the next draw reshuffles. The
    /// remaining draws are queued after the hooks for the same reason.
    fn draw(&mut self, count: u32, from_hand_draw: bool) -> Vec<Effect> {
        // Pillage reads this after each draw; a draw that yields nothing must clear it.
        self.stats.last_drawn = None;
        if count == 0
            || !self.player.creature.powers.iter().all(|p| p.should_draw(from_hand_draw))
            || !self.relic_should_draw(from_hand_draw)
        {
            return vec![];
        }
        if self.player.hand.len() >= MAX_HAND {
            return vec![];
        }
        let mut out = vec![];
        if self.reshuffle_if_needed() {
            let hooks = self.after_shuffle();
            // CardPileCmd.Draw awaits the shuffle's hooks before it takes the
            // card, so Stratagem's pick comes first and this draw waits.
            if self.player.creature.power(PowerId::Stratagem).is_some() {
                out.extend(hooks);
                out.push(Effect::Draw { count, from_hand_draw });
                return out;
            }
            out.extend(hooks);
        }
        if self.player.draw.is_empty() {
            return out;
        }
        let mut card = self.player.draw.remove(0);
        let uid = card.uid;
        let strike = card.has_tag(Tag::Strike);
        // Slither.AfterCardDrawn: reroll the cost, for this combat.
        if card.enchantment.is_some_and(|e| e.randomizes_cost_on_draw()) {
            card.cost_this_combat = Some(self.rngs.energy_costs.next_int(4) as i32);
        }
        // ConfusedPower.AfterCardDrawn (Snecko Eye), on the same stream.
        if self.player.creature.power(PowerId::Confused).is_some() && card.def().cost >= 0 {
            card.cost_this_combat = Some(self.rngs.energy_costs.next_int(4) as i32);
        }
        self.player.hand.push(card);
        self.stats.last_drawn = Some(uid);
        self.bind_drawn(uid);
        for p in &mut self.player.creature.powers {
            out.extend(p.after_card_drawn());
        }
        if strike && self.player.creature.power(PowerId::Hellraiser).is_some() {
            out.push(Effect::AutoPlay { uid, force_exhaust: false });
        }
        // Void.AfterCardDrawn: drawing it costs energy.
        if self.find_card(uid).is_some_and(|k| k.id == CardId::Void) {
            out.push(Effect::LoseEnergy { amount: 1 });
        }
        if count > 1 {
            out.push(Effect::Draw { count: count - 1, from_hand_draw });
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
    pub(crate) fn take_card(&mut self, uid: u32) -> Option<Card> {
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            if let Some(i) = pile.iter().position(|c| c.uid == uid) {
                return Some(pile.remove(i));
            }
        }
        None
    }

    pub(crate) fn put_card(&mut self, card: Card, to: Pile) {
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
            Pile::DrawRandom => {
                let at = self.rngs.card_generation.next_int(self.player.draw.len() + 1);
                self.player.draw.insert(at, card);
                self.stats.random_draw_inserts += 1;
            }
            Pile::Discard => self.player.discard.push(card),
            Pile::Exhaust => self.player.exhaust.push(card),
        }
    }

    pub(crate) fn new_uid(&mut self) -> u32 {
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
            .order
            .iter()
            .flat_map(|&i| self.enemies[i].creature.powers.iter().map(move |p| (CreatureRef::Enemy(i), p)));
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
        // Hook.ModifyDamage folds the card's enchantment in at the very top,
        // before any power or relic sees the number.
        if let Some(e) = player_card.and_then(|c| c.enchantment.as_ref()) {
            num += e.damage_additive(props);
            num *= e.damage_multiplicative(props);
        }
        for (owner, p) in self.listeners() {
            num += p.modify_damage_additive(owner, target, dealer, props);
        }
        num += self.relic_damage_additive(player_card, props);
        for (owner, p) in self.listeners() {
            num *= p.modify_damage_multiplicative(owner, target, dealer, props, dv, dc);
        }
        num *= self.relic_damage_multiplicative(player_card, props);
        // SurroundedPower.ModifyDamageMultiplicative: the crab half behind the
        // player hits for half again, powered or not.
        if let (CreatureRef::Player, Some(CreatureRef::Enemy(d))) = (target, dealer) {
            if let Some(facing) = self.player.creature.power(PowerId::Surrounded).map(|p| p.data) {
                let behind = if facing == 0 { PowerId::BackAttackLeft } else { PowerId::BackAttackRight };
                if self.enemies[d].creature.power(behind).is_some() {
                    num *= 1.5;
                }
            }
        }
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
        if let Some(e) = card.and_then(|u| self.find_card(u)).and_then(|c| c.enchantment.as_ref()) {
            num += e.block_additive();
        }
        // FastenPower only looks at block from Defends or from no card.
        let defend_source = card.and_then(|u| self.find_card(u)).is_none_or(|c| c.has_tag(Tag::Defend));
        for (owner, p) in self.listeners() {
            num += p.modify_block_additive(owner, source_owner, props, defend_source);
        }
        for (owner, p) in self.listeners() {
            num *= p.modify_block_multiplicative(owner, target, props, plays, card.is_some());
        }
        num.max(0.0)
    }

    // ---- commands ------------------------------------------------------------

    /// `CreatureCmd.Damage` core. Returns the effects of `AfterDamageReceived` hooks.
    /// `CreatureCmd.Damage` on several targets: every target takes its hit
    /// before the aftermath of any runs (a death and its `AfterDeath` hooks,
    /// `AfterDamageReceived`). A Conflagration pass that kills The Lost hits
    /// The Forgotten before Possess Strength hands the Strength back.
    fn damage_many(&mut self, targets: &[CreatureRef], amount: f64, props: ValueProp, dealer: Option<CreatureRef>, card: Option<u32>) -> Vec<Effect> {
        let mut out = vec![];
        for &target in targets {
            if !self.creature(target).alive() {
                if card.is_some() {
                    self.stats.last_card_hit = None;
                }
                continue;
            }
            out.extend(self.damage(target, amount, props, dealer, card));
        }
        out
    }

    fn damage(&mut self, target: CreatureRef, amount: f64, props: ValueProp, dealer: Option<CreatureRef>, card: Option<u32>) -> Vec<Effect> {
        let mut modified = self.modify_damage_from(target, dealer, amount, props, card);
        // IntangiblePower.ModifyDamageCap: the whole hit is capped at 1, so
        // block only ever eats 1 of it.
        if self.creature(target).power(PowerId::Intangible).is_some() {
            modified = modified.min(1.0);
        }
        // HardToKillPower.ModifyDamageCap: no hit takes more than the amount.
        if let Some(cap) = self.creature(target).power(PowerId::HardToKill).map(|p| p.amount) {
            modified = modified.min(cap as f64);
        }
        let c = self.creature_mut(target);
        // Creature.DamageBlockInternal: block absorbs min(block, amount),
        // truncated on the block write but not on the remainder.
        let blocked = if props.has(ValueProp::UNBLOCKABLE) { 0.0 } else { (c.block as f64).min(modified) };
        let had_block = c.block > 0;
        c.block -= blocked as i32;
        let block_broken = had_block && blocked > 0.0 && c.block == 0;
        let fully_blocked = !props.has(ValueProp::UNBLOCKABLE) && (blocked > 0.0 || c.block > 0) && modified - blocked < 1.0;
        // ImbalancedPower.AfterDamageGiven: a fully blocked hit throws the
        // Bowlbug Rock off balance.
        if let Some(CreatureRef::Enemy(d)) = dealer {
            if fully_blocked && self.enemies[d].creature.power(PowerId::Imbalanced).is_some() {
                self.enemies[d].monster.vars.off_balance = true;
            }
        }
        // CurlUpPower.AfterDamageReceived remembers the card attack that hit.
        if let (CreatureRef::Enemy(i), Some(uid)) = (target, card) {
            if props.is_powered() {
                if let Some(p) = self.enemies[i].creature.power_mut(PowerId::CurlUp) {
                    if p.data == 0 {
                        p.data = uid as i32;
                    }
                }
            }
        }
        let c = self.creature_mut(target);
        let mut unblocked = (modified - blocked).max(0.0);
        let through = unblocked > 0.0;
        // SlipperyPower.ModifyHpLostAfterOsty: at most 1 HP per hit.
        if c.power(PowerId::Slippery).is_some() && unblocked >= 1.0 {
            unblocked = 1.0;
        }
        // HardenedShellPower.ModifyHpLostBeforeOstyLate: a budget of HP per
        // turn, spent down by everything that gets through.
        if let Some(shell) = c.power(PowerId::HardenedShell) {
            let left = (shell.amount - shell.data).max(0) as f64;
            unblocked = unblocked.min(left);
        }
        // TheBoot.ModifyHpLostAfterOstyLate: your powered attacks that get
        // 1 to 4 HP through deal 5.
        if target != CreatureRef::Player
            && dealer == Some(CreatureRef::Player)
            && props.is_powered()
            && (1.0..5.0).contains(&unblocked)
            && self.has_relic(crate::relic::RelicId::TheBoot)
        {
            unblocked = 5.0;
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
        // Creature.LoseHpInternal: truncate once.
        let mut lost = unblocked.min(CLAMP) as i32;
        // The hit's TotalDamage + OverkillDamage, before any death save.
        if let Some(u) = card {
            if self.stats.card_dealt.0 == u {
                self.stats.card_dealt.1 += blocked as i32 + lost;
            }
        }
        let c = self.creature_mut(target);
        let was_alive = c.alive();
        let before = c.hp;
        c.hp = (c.hp - lost).max(0);
        // HP a monster really lost; an unkillable bar (the Giant's blast
        // turn) is not progress.
        if target != CreatureRef::Player && before < CLAMP as i32 {
            self.stats.enemy_hp_lost += (before - self.creature(target).hp) as i64;
        }
        // DamageResult.TotalDamage + OverkillDamage: the block it ate plus
        // everything past it, kill or not.
        if let Some(u) = card {
            self.stats.last_card_hit = Some((u, blocked as i32 + lost));
        }
        let mut fairy_used = false;
        // A monster's powered hit that got through, from a Paper Cuts owner.
        let paper_cuts = (target == CreatureRef::Player && through && props.is_powered())
            .then_some(dealer)
            .flatten()
            .filter(|d| d.side() == Side::Enemy)
            .and_then(|d| self.creature(d).power(PowerId::PaperCuts).map(|p| p.amount));
        let dying = target == CreatureRef::Player && self.player.creature.hp <= 0 && was_alive;
        // CreatureCmd.Damage runs AfterDamageGiven before it kills: a fatal
        // hit has already cost the max HP when a Fairy or Lizard Tail heals.
        if let (true, Some(cuts)) = (dying, paper_cuts) {
            self.player.creature.max_hp -= cuts;
        }
        if dying {
            if let Some(fairy) = self.save_player() {
                lost = lost.min(self.player.creature.max_hp);
                fairy_used = fairy;
            }
        }
        if buffered {
            self.modify_power(CreatureRef::Player, PowerId::Buffer, -1);
        }
        if let Some(shell) = self.creature_mut(target).power_mut(PowerId::HardenedShell) {
            shell.data += lost;
        }
        // SkittishPower.AfterAttack: the first card attack that lands each
        // turn makes it curl up. `AfterAttack` is once per attack, so the
        // block waits for the last hit instead of soaking up the rest.
        if lost > 0 && props.has(ValueProp::MOVE) && card.is_some() {
            if let Some(skittish) = self.creature_mut(target).power_mut(PowerId::Skittish) {
                if skittish.data == 0 {
                    skittish.data = 1;
                    if let CreatureRef::Enemy(i) = target {
                        self.stats.skittish_pending.push(i);
                    }
                }
            }
        }
        // SuckPower.AfterAttack: Strength for each of its own hits that landed.
        if lost > 0 && props.is_powered() {
            if let Some(d) = dealer.filter(|&d| d != target) {
                if let Some(amount) = self.creature(d).power(PowerId::Suck).map(|p| p.amount) {
                    self.queue.push_front(Effect::ApplyPower {
                        target: d,
                        id: PowerId::Strength,
                        amount,
                        applier: Some(d),
                    });
                }
            }
        }
        let hp_after = self.creature(target).hp;
        let own_turn = self.side == target.side();
        if target == CreatureRef::Player && lost > 0 {
            self.stats.hp_lost_this_turn = true;
            self.stats.unblocked_hits_taken += 1;
        }
        // Hook.AfterDamageReceived over the target's powers.
        let mut out = vec![];
        // A monster's powered hit that got through: PaperCutsPower costs max
        // HP (`CreatureCmd.LoseMaxHp`, a hit for whatever no longer fits,
        // then the new max), PainfulStabsPower owes Wounds after the attack.
        if target == CreatureRef::Player && through && props.is_powered() {
            if let Some(d) = dealer.filter(|d| d.side() == Side::Enemy) {
                if let Some(cuts) = paper_cuts.filter(|_| !dying) {
                    let new_max = self.player.creature.max_hp - cuts;
                    let over = self.player.creature.hp - new_max;
                    if over > 0 {
                        out.push(Effect::Damage {
                            target,
                            amount: over as f64,
                            props: ValueProp::UNBLOCKABLE.or(ValueProp::UNPOWERED),
                            dealer: None,
                            card: None,
                        });
                    }
                    out.push(Effect::SetMaxHp { target, max_hp: new_max });
                }
                if let Some(stabs) = self.creature(d).power(PowerId::PainfulStabs).map(|p| p.amount) {
                    self.stats.wounds_pending += stabs.max(0) as u32;
                }
            }
        }
        if fairy_used {
            // OnUseWrapper ran for the automatic use, so its after hooks fire.
            out.push(Effect::AfterPotionUsed);
        }
        // CreatureCmd.Damage skips Hook.AfterDamageReceived for a hit that
        // killed its target, and only then kills: a death a Fairy in a Bottle
        // or Lizard Tail prevents still gets no Flame Barrier or Thorns.
        let killed = dying || (was_alive && !self.creature(target).alive());
        for p in &self.creature(target).powers {
            out.extend(p.before_damage_received(target, props, dealer));
        }
        if !killed {
            for p in &self.creature(target).powers {
                out.extend(p.after_damage_received(target, lost, props, dealer, own_turn, hp_after));
            }
            if target == CreatureRef::Player {
                out.extend(self.relic_after_damage_received(lost, props, own_turn));
            }
        }
        // BurrowedPower.AfterBlockBroken: the Tunneler is stunned back to Bite
        // and loses the burrow.
        if block_broken && target != CreatureRef::Player && self.creature(target).power(PowerId::Burrowed).is_some() {
            out.push(Effect::Stun { target, next: Some("BITE_MOVE") });
            out.push(Effect::RemovePower { target, id: PowerId::Burrowed });
        }
        // HandDrill.AfterDamageGiven: breaking an enemy's block leaves it Vulnerable.
        if block_broken
            && dealer == Some(CreatureRef::Player)
            && target != CreatureRef::Player
            && self.has_relic(crate::relic::RelicId::HandDrill)
        {
            out.push(Effect::ApplyPower { target, id: PowerId::Vulnerable, amount: 2, applier: Some(CreatureRef::Player) });
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

    /// `Hook.ShouldDie` for the player at 0 HP. LizardTail: false once, then
    /// heal to half. FairyInABottle.ShouldDie + AfterPreventingDeath: 30% max
    /// HP, at least 1. Returns whether a save happened, and if so whether
    /// the Fairy was spent.
    fn save_player(&mut self) -> Option<bool> {
        if let Some(hp) = self.relic_prevent_death() {
            self.player.creature.hp = hp;
            return Some(false);
        }
        let slot = self.potions.iter().position(|p| *p == Some(PotionId::FairyInABottle))?;
        self.potions[slot] = None;
        let heal = (self.player.creature.max_hp as f64 * 0.3).max(1.0) as i32;
        self.player.creature.hp = heal.min(self.player.creature.max_hp);
        Some(true)
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
        // SurprisePower.AfterDeath: two gremlins take the merc's place. Their
        // slots are named, so the sneaky one always comes first.
        if self.enemies[i].creature.power(PowerId::Surprise).is_some() {
            let loot = self.enemies[i].creature.power(PowerId::Thievery).map_or(0, |p| p.data);
            self.spawn(MonsterId::SneakyGremlin, Flags::default());
            self.spawn(MonsterId::FatGremlin, Flags::default());
            // The fat one runs off with everything the merc took.
            if loot > 0 {
                if let Some(fat) = self.enemies.last_mut() {
                    fat.creature.powers.push(Power::new(PowerId::Heist, loot));
                }
            }
        }
        // WaterfallGiant.TriggerAboutToBlowState: the giant comes back with
        // an unkillable HP bar for exactly as long as the blast takes.
        if self.enemies[i].creature.power(PowerId::SteamEruption).is_some() {
            self.enemies[i].creature.max_hp = CLAMP as i32;
            self.enemies[i].creature.hp = CLAMP as i32;
            self.enemies[i].monster.force_about_to_blow();
        }
        // RavenousPower.AfterDeath: a living teammate gorges, gaining Strength
        // and losing the turn to it.
        for j in 0..self.enemies.len() {
            if j == i || !self.enemies[j].creature.alive() {
                continue;
            }
            if let Some(amount) = self.enemies[j].creature.power(PowerId::Ravenous).map(|p| p.amount) {
                let them = CreatureRef::Enemy(j);
                out.push(Effect::Stun { target: them, next: None });
                out.push(Effect::ApplyPower { target: them, id: PowerId::Strength, amount, applier: Some(them) });
            }
        }
        // ReattachPower.AfterDeath: a segment with another still alive plays
        // dead and comes back.
        if self.enemies[i].creature.power(PowerId::Reattach).is_some() && !self.other_segments_dead(i) {
            self.enemies[i].reviving = true;
            self.enemies[i].monster.force_named_move("DEAD_MOVE");
        }
        // CrabRagePower.AfterDeath: the other half gets angry.
        for j in 0..self.enemies.len() {
            if j != i && self.enemies[j].creature.alive() && self.enemies[j].creature.power(PowerId::CrabRage).is_some() {
                let them = CreatureRef::Enemy(j);
                out.push(Effect::ApplyPower { target: them, id: PowerId::Strength, amount: 6, applier: Some(them) });
                out.push(Effect::GainBlock { target: them, amount: 99.0, props: ValueProp::UNPOWERED, card: None });
                out.push(Effect::RemovePower { target: them, id: PowerId::CrabRage });
            }
        }
        // SurroundedPower.AfterDeath: with only one side left, face it.
        let left: Vec<usize> = self.living_enemies().collect();
        let all = |id: PowerId| left.iter().all(|&j| self.enemies[j].creature.power(id).is_some());
        if !left.is_empty() && (all(PowerId::BackAttackLeft) || all(PowerId::BackAttackRight)) {
            self.face_crab(CreatureRef::Enemy(left[0]));
        }
        out.extend(self.glory_after_death(i));
        self.player
            .creature
            .powers
            .retain(|p| !(matches!(p.id, PowerId::Constrict | PowerId::Shrink) && p.applier == Some(me)));
        out.extend(self.relic_after_enemy_death());
        // Creature.RemoveAllPowersAfterDeath, once every death hook has read
        // them: a power stays only if it outlives its owner
        // (`ShouldPowerBeRemovedAfterOwnerDeath`). IllusionPower's
        // `ShouldPowerBeRemovedOnDeath` also keeps an illusion's buffs and
        // temporary debuffs, which is what it revives with.
        let illusion = self.enemies[i].creature.power(PowerId::Illusion).is_some();
        self.enemies[i].creature.powers.retain(|p| {
            matches!(
                p.id,
                PowerId::SteamEruption | PowerId::Minion | PowerId::Adaptable | PowerId::PainfulStabs | PowerId::Reattach
            ) || (illusion && (!is_debuff(p.id) || crate::power::temp_power(p.id).is_some()))
        });
        out
    }

    /// The act 3 `AfterDeath` hooks, run while the dead monster still holds
    /// its powers (`CreatureCmd.Kill` strips them afterwards).
    fn glory_after_death(&mut self, i: usize) -> Vec<Effect> {
        let me = CreatureRef::Enemy(i);
        let mut out = vec![];
        let dead = self.enemies[i].creature.clone();
        // PossessStrength/SpeedPower.AfterDeath: what it stole comes back.
        for (possess, stat) in [(PowerId::PossessStrength, PowerId::Strength), (PowerId::PossessSpeed, PowerId::Dexterity)] {
            if let Some(stolen) = dead.power(possess).map(|p| p.data).filter(|&n| n != 0) {
                out.push(Effect::ApplyPower { target: CreatureRef::Player, id: stat, amount: -stolen, applier: None });
            }
        }
        // HexPower / DampenPower go when their caster dies, taking the
        // Hexed afflictions and the downgrades with them.
        let owned = |c: &Combat, id: PowerId| c.player.creature.power(id).is_some_and(|p| p.applier == Some(me));
        if owned(self, PowerId::Hex) {
            self.player.creature.powers.retain(|p| p.id != PowerId::Hex);
            self.for_each_card(|c| {
                if c.affliction == Some(Affliction::Hexed) {
                    c.affliction = None;
                }
            });
        }
        if owned(self, PowerId::Dampen) {
            self.player.creature.powers.retain(|p| p.id != PowerId::Dampen);
            self.for_each_card(|c| {
                if std::mem::take(&mut c.dampened) {
                    c.upgraded = true;
                }
            });
        }
        // StockPower.AfterDeath: a fresh Axebot takes its slot, one stock down.
        if let Some(stock) = dead.power(PowerId::Stock).map(|p| p.amount).filter(|&n| n > 0) {
            let slot = self.enemies[i].slot;
            self.spawn(MonsterId::Axebot, Flags { stock: Some((stock - 1) as u8), slot, ..Default::default() });
        }
        // AdaptablePower.AfterDeath: the Test Subject stays in the fight and
        // respawns on its next turn, keeping only what outlives a death.
        if dead.power(PowerId::Adaptable).is_some() {
            let e = &mut self.enemies[i];
            e.reviving = true;
            e.creature.powers.retain(|p| matches!(p.id, PowerId::Adaptable | PowerId::PainfulStabs));
            e.monster.force_to("RESPAWN_MOVE");
        }
        // Queen.AfterDeath: with the Amalgam gone she stops feeding it and,
        // if she was about to, enrages instead.
        if self.enemies[i].monster.id == MonsterId::TorchHeadAmalgam {
            for q in self.living_enemies().collect::<Vec<_>>() {
                let queen = &mut self.enemies[q].monster;
                if queen.id == MonsterId::Queen {
                    queen.vars.amalgam_died = true;
                    if queen.next_move_name() == Some("BURN_BRIGHT_FOR_ME_MOVE") {
                        queen.force_to("ENRAGE_MOVE");
                    }
                }
            }
        }
        out
    }

    /// `Fabricator.SpawnBot`: a bot from the pool other than the last one it
    /// made, as its Minion, in the next free slot.
    fn fabricate_bot(&mut self, fabricator: CreatureRef, aggro: bool) {
        let CreatureRef::Enemy(f) = fabricator else { return };
        let pool: &[MonsterId] =
            if aggro { &[MonsterId::Zapbot, MonsterId::Stabbot] } else { &[MonsterId::Guardbot, MonsterId::Noisebot] };
        let last = self.enemies[f].monster.vars.last_spawned;
        let options: Vec<MonsterId> = pool.iter().copied().filter(|&m| Some(m) != last).collect();
        let rolled = self.rngs.monster_ai.pick(&options).copied();
        let scripted = self.script.spawns.iter().position(|m| options.contains(m));
        let Some(id) = scripted.map(|i| self.script.spawns.remove(i)).or(rolled) else { return };
        self.enemies[f].monster.vars.last_spawned = Some(id);
        if self.free_slots() == 0 {
            return;
        }
        self.spawn(id, Flags::default());
        let bot = self.enemies.len() - 1;
        let mut minion = Power::new(PowerId::Minion, 1);
        minion.applier = Some(fabricator);
        self.enemies[bot].creature.powers.push(minion);
    }

    /// Every card the player has in combat, mutably.
    fn for_each_card(&mut self, mut f: impl FnMut(&mut Card)) {
        let p = &mut self.player;
        for pile in [&mut p.hand, &mut p.draw, &mut p.discard, &mut p.exhaust, &mut p.play] {
            pile.iter_mut().for_each(&mut f);
        }
    }

    /// `GalvanicPower.BeforeCombatStart`: every power card starts Galvanized.
    fn galvanize_deck(&mut self) {
        if !self.enemies.iter().any(|e| e.creature.power(PowerId::Galvanic).is_some()) {
            return;
        }
        self.for_each_card(|c| {
            if c.ty() == CardType::Power && c.affliction.is_none() {
                c.affliction = Some(Affliction::Galvanized);
            }
        });
    }

    /// `Hook.AfterCardEnteredCombat` for a card made mid-fight: relics, then
    /// the powers that afflict or match it (Galvanic, Hex, Aeonglass's Withers).
    fn card_entered_combat(&self, card: &mut Card) {
        self.relic_card_entered_combat(card);
        let living = || self.living_enemies().map(|i| &self.enemies[i]);
        if card.affliction.is_none() {
            if card.ty() == CardType::Power && living().any(|e| e.creature.power(PowerId::Galvanic).is_some()) {
                card.affliction = Some(Affliction::Galvanized);
            } else if self.player.creature.power(PowerId::Hex).is_some() {
                card.affliction = Some(Affliction::Hexed);
            }
        }
        // Stomp.AfterCardEnteredCombat: a new Stomp is 1 cheaper this turn
        // per attack play finished this turn. Clones skip it (`IsClone`);
        // `CloneCard` puts their cost back.
        if card.id == CardId::Stomp {
            let attacks = self
                .stats
                .finished_this_turn
                .iter()
                .filter(|&&u| self.find_card(u).is_some_and(|k| k.ty() == CardType::Attack))
                .count() as i32;
            if attacks > 0 {
                let cur = card.cost_this_turn.unwrap_or(card.base_cost());
                card.cost_this_turn = Some((cur - attacks).max(0));
            }
        }
        // Aeonglass.AfterCardGeneratedForCombat: a new Wither is upgraded as
        // often as the ones already out.
        if card.id == CardId::Wither {
            if let Some(a) = living().find(|e| e.monster.id == MonsterId::Aeonglass) {
                card.extra_damage = 3.0 * a.monster.vars.wither_upgrades as f64;
            }
        }
    }

    /// `ChainsOfBindingPower.AfterCardDrawn`: on your own turn, the first
    /// `amount` cards drawn each turn are Bound.
    fn bind_drawn(&mut self, uid: u32) {
        if self.side != Side::Player {
            return;
        }
        let Some(chains) = self.player.creature.power_mut(PowerId::ChainsOfBinding) else { return };
        if chains.data >= chains.amount {
            return;
        }
        let Some(card) = self.player.hand.iter_mut().find(|c| c.uid == uid) else { return };
        if card.affliction.is_some() || card.smogged {
            return;
        }
        card.affliction = Some(Affliction::Bound);
        if let Some(chains) = self.player.creature.power_mut(PowerId::ChainsOfBinding) {
            chains.data += 1;
        }
    }

    /// `ChainsOfBindingPower.BeforeSideTurnEnd`: the turn's Bound cards are
    /// set free and the count starts over.
    fn unbind(&mut self) {
        let Some(chains) = self.player.creature.power_mut(PowerId::ChainsOfBinding) else { return };
        chains.data = 0;
        self.stats.bound_played = false;
        self.for_each_card(|c| {
            if c.affliction == Some(Affliction::Bound) {
                c.affliction = None;
            }
        });
    }

    /// `CreatureCmd.GainBlock`. Returns `AfterBlockGained` hook effects.
    fn gain_block(&mut self, target: CreatureRef, amount: f64, props: ValueProp, card: Option<u32>) -> Vec<Effect> {
        // Cards are only played by the player, so the block source owner is
        // always the target itself.
        let mut modified = self.modify_block(target, target, amount, props, card);
        if target == CreatureRef::Player {
            modified *= self.relic_block_multiplicative(card, props);
        }
        self.stats.last_block_gained = modified;
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
        // ArtifactPower.cs: negate a debuff (by GetTypeForAmount, so Strength
        // loss and the permanent -1 Shrink count) and spend a charge.
        if is_debuff_for_amount(id, amount) && self.creature(target).power(PowerId::Artifact).is_some() {
            self.modify_power(target, PowerId::Artifact, -1);
            return vec![];
        }
        // PowerCmd.Apply finds no instance to stack onto for an Instanced power.
        let existing = if instanced(id) { None } else { self.creature(target).power(id).map(|p| p.amount) };
        // An existing instance takes the amount whatever its StackType, which
        // only decides whether the number is drawn: two Snecko relics make
        // Confused 2.
        if existing.is_some() {
            out.extend(Power::new(id, 0).on_applied(target, amount));
            self.modify_power(target, id, amount);
        } else {
            let mut p = Power::new(id, amount);
            p.skip_next_tick = target == CreatureRef::Player && is_debuff(id);
            // RitualPower.AfterApplied: an enemy's own Ritual sits out the
            // turn it lands on.
            p.data = i32::from(id == PowerId::Ritual && target != CreatureRef::Player);
            p.applier = applier;
            out.extend(p.on_applied(target, amount));
            self.creature_mut(target).powers.push(p);
        }
        // Inferno.OnPlay / CrimsonMantle.OnPlay: `IncrementSelfDamage` on the
        // power the card applied. `data` is the self-damage the power deals
        // at turn start.
        if matches!(id, PowerId::Inferno | PowerId::CrimsonMantle) {
            if let Some(p) = self.creature_mut(target).powers.iter_mut().find(|p| p.id == id) {
                p.data += 1;
            }
        }
        // HexPower / DampenPower.AfterApplied, on a fresh instance.
        if existing.is_none() && target == CreatureRef::Player {
            match id {
                PowerId::Hex => self.for_each_card(|c| {
                    if c.affliction.is_none() {
                        c.affliction = Some(Affliction::Hexed);
                    }
                }),
                PowerId::Dampen => self.for_each_card(|c| {
                    if c.upgraded {
                        c.upgraded = false;
                        c.dampened = true;
                    }
                }),
                _ => {}
            }
        }
        // PossessStrength/SpeedPower.AfterPowerAmountChanged: keep count of
        // what the owner has taken from the player.
        if let (CreatureRef::Player, Some(thief @ CreatureRef::Enemy(_))) = (target, applier) {
            let possess = match id {
                PowerId::Strength => Some(PowerId::PossessStrength),
                PowerId::Dexterity => Some(PowerId::PossessSpeed),
                _ => None,
            };
            if let Some(p) = possess.filter(|_| amount < 0).and_then(|pid| self.creature_mut(thief).power_mut(pid)) {
                p.data += amount;
            }
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
    /// `ReattachPower.AreAllOtherSegmentsDead`.
    fn other_segments_dead(&self, i: usize) -> bool {
        self.enemies
            .iter()
            .enumerate()
            .filter(|&(j, e)| j != i && e.creature.power(PowerId::Reattach).is_some())
            .all(|(_, e)| !e.creature.alive())
    }

    /// `SurroundedPower.UpdateDirection`: turn to face a crab half standing
    /// behind the player.
    fn face_crab(&mut self, target: CreatureRef) {
        let CreatureRef::Enemy(t) = target else { return };
        let left = self.enemies[t].creature.power(PowerId::BackAttackLeft).is_some();
        let right = self.enemies[t].creature.power(PowerId::BackAttackRight).is_some();
        if let Some(p) = self.player.creature.power_mut(PowerId::Surrounded) {
            if p.data == 0 && left {
                p.data = 1;
            } else if p.data == 1 && right {
                p.data = 0;
            }
        }
    }

    /// `Effect::MonsterStep`: the parts of a monster's move that read the
    /// combat as the move resolves.
    fn monster_step(&mut self, i: usize, step: u8) -> Vec<Effect> {
        use crate::monster::{STEP_FLUTTER_DOWN, STEP_PHEROMONE, STEP_SAIL, STEP_STAGGER, STEP_STEAL};
        let me = CreatureRef::Enemy(i);
        match step {
            // BowlbugRock.HeadbuttMove: `CreatureCmd.Stun(DizzyMove)`.
            STEP_STAGGER if self.enemies[i].monster.vars.off_balance => {
                self.enemies[i].monster.stun(None);
                vec![]
            }
            STEP_FLUTTER_DOWN => {
                self.enemies[i].monster.stun_past_next();
                vec![]
            }
            STEP_STEAL => self.steal_card(i),
            // Entomancer.SpitMove: the hive grows to three, then Strength.
            STEP_PHEROMONE => {
                if self.enemies[i].creature.power_amount(PowerId::PersonalHive) < 3 {
                    vec![
                        Effect::ApplyPower { target: me, id: PowerId::PersonalHive, amount: 1, applier: Some(me) },
                        Effect::ApplyPower { target: me, id: PowerId::Strength, amount: 1, applier: Some(me) },
                    ]
                } else {
                    vec![Effect::ApplyPower { target: me, id: PowerId::Strength, amount: 2, applier: Some(me) }]
                }
            }
            // TheObscura.WailMove: Strength to every teammate, itself included.
            STEP_SAIL => self
                .living_enemies()
                .map(|j| Effect::ApplyPower { target: CreatureRef::Enemy(j), id: PowerId::Strength, amount: 3, applier: Some(me) })
                .collect(),
            _ => vec![],
        }
    }

    /// `ThievingHopper.ThieveryMove`: take a card the combat started with
    /// from the draw or discard pile, the rarest first (uncommon, then
    /// common or rare, then basic, then the rest), and hold it in a Swipe.
    fn steal_card(&mut self, i: usize) -> Vec<Effect> {
        use crate::types::CardRarity;
        let deck = self.stats.deck_size;
        let imbued = |c: &Card| c.enchantment.is_some_and(|e| e.id == crate::enchant::EnchantmentId::Imbued);
        let cards: Vec<&Card> = self.player.draw.iter().chain(&self.player.discard).filter(|c| c.uid <= deck).collect();
        let tiers: [&dyn Fn(&Card) -> bool; 4] = [
            &|c| !imbued(c) && c.def().rarity == CardRarity::Uncommon,
            &|c| !imbued(c) && matches!(c.def().rarity, CardRarity::Common | CardRarity::Rare),
            &|c| !imbued(c) && c.def().rarity == CardRarity::Basic,
            &|c| imbued(c) || (c.def().rarity == CardRarity::Special && !matches!(c.ty(), CardType::Status | CardType::Curse)),
        ];
        let pool: Vec<u32> = tiers
            .iter()
            .map(|t| cards.iter().filter(|c| t(c)).map(|c| c.uid).collect::<Vec<_>>())
            .find(|v| !v.is_empty())
            .unwrap_or_else(|| cards.iter().map(|c| c.uid).collect());
        let Some(&uid) = self.rngs.card_generation.pick(&pool) else { return vec![] };
        self.take_card(uid);
        let me = CreatureRef::Enemy(i);
        vec![Effect::ApplyPower { target: me, id: PowerId::Swipe, amount: 1, applier: Some(me) }]
    }

    /// The cards a random generation makes, `CardFactory.GetForCombat` or
    /// `GetDistinctForCombat`. A recorded `gen` forces the pick; with none,
    /// the roll is kept in `unforced` for a later record to rewrite.
    fn roll_cards(&mut self, pool: GenPool, count: u32, distinct: bool) -> Vec<CardId> {
        let options = pool_cards(pool);
        let mut chosen = Vec::new();
        if distinct {
            let mut opts = options.clone();
            self.rngs.card_generation.shuffle(&mut opts);
            for _ in 0..count {
                let scripted = self.script.take_generated(&opts);
                let id = match scripted {
                    Some(id) => id,
                    None => match opts.first() {
                        Some(&id) => {
                            self.script.unforced.push(id);
                            id
                        }
                        None => break,
                    },
                };
                opts.retain(|&o| o != id);
                chosen.push(id);
            }
        } else {
            for _ in 0..count {
                let scripted = self.script.take_generated(&options);
                if scripted.is_none() {
                    self.script.unforced.extend(self.rngs.card_generation.pick(&options).copied());
                }
                if let Some(id) = scripted.or_else(|| self.script.unforced.last().copied()) {
                    chosen.push(id);
                }
            }
        }
        chosen
    }

    /// `Hook.ModifyCardPlayResultPileTypeAndPosition`, which the game asks
    /// once as the play begins, before `OnPlay`, so the card that grants
    /// Rebound is not moved by it. Player powers in order: Corruption sends
    /// a Skill to the exhaust pile, Rebound a card bound for the discard
    /// pile to the top of the draw pile and spends a charge. Where the card
    /// finally goes is still decided in `FinishCardPlay`; this only records
    /// Rebound's claim.
    fn decide_result_pile(&mut self, uid: u32) {
        let Some(card) = self.find_card(uid) else { return };
        if card.ty() == CardType::Power || card.dupe {
            return;
        }
        let ty = card.ty();
        let mut discard = !(card.has(Keyword::Exhaust) || card.exhaust_on_next_play);
        let mut rebound = false;
        for p in &self.player.creature.powers {
            match p.id {
                PowerId::Corruption if ty == CardType::Skill => discard = false,
                // NostalgiaPower: the first `amount` attacks and skills each
                // turn go back on top of the draw pile.
                PowerId::Nostalgia
                    if discard
                        && matches!(ty, CardType::Attack | CardType::Skill)
                        && p.amount > self.stats.attack_skill_plays_this_turn as i32 =>
                {
                    self.stats.nostalgia_top.push(uid);
                    discard = false;
                }
                PowerId::Rebound if discard => rebound = true,
                _ => {}
            }
        }
        if rebound {
            self.stats.rebound.push(uid);
            // ReboundPower.AfterModifyingCardPlayResultPileOrPosition.
            self.modify_power(CreatureRef::Player, PowerId::Rebound, -1);
        }
    }

    /// `ToricToughnessPower.AfterBlockCleared`'s decrement, instance by
    /// instance: each counts down its own turns. The block each gains is
    /// already queued, and nothing between the two reads the count.
    fn tick_toric(&mut self) {
        let powers = &mut self.player.creature.powers;
        for p in powers.iter_mut().filter(|p| p.id == PowerId::ToricToughness) {
            p.amount -= 1;
        }
        powers.retain(|p| p.id != PowerId::ToricToughness || p.amount > 0);
    }

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
            Then::CloneToHand { copies } => (0..copies).map(|_| Effect::CloneCard { uid, to: Pile::Hand }).collect(),
            Then::ToHandMany { left } => {
                let mut e = vec![Effect::MoveCard { uid, to: Pile::Hand }];
                if left > 1 {
                    e.push(Effect::Choose {
                        from: Pile::Discard,
                        filter: CardFilter::Any,
                        then: Then::ToHandMany { left: left - 1 },
                        can_skip: true,
                    });
                }
                e
            }
            Then::Select { from, filter, left, optional, done } => {
                self.stats.selected.push(uid);
                if left <= 1 {
                    return self.finish_select(done);
                }
                let then = Then::Select { from, filter, left: left - 1, optional, done };
                vec![Effect::Choose { from, filter, then, can_skip: optional }]
            }
            Then::TransformPick { left } => {
                self.stats.transform_picks.push(uid);
                if left > 1 {
                    vec![Effect::Choose { from: Pile::Hand, filter: CardFilter::Any, then: Then::TransformPick { left: left - 1 }, can_skip: false }]
                } else {
                    let picks = std::mem::take(&mut self.stats.transform_picks);
                    self.transforms_in_hand_order(&picks)
                }
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
        CardFilter::AttackOrPower => matches!(c.ty(), CardType::Attack | CardType::Power),
    }
}

/// Shuffle in place, or impose the next scripted order. Scripted orders
/// match cards by (id, upgraded); cards the script does not mention go to
/// the bottom in their existing order.
/// `Hook.ModifyShuffleOrder`, run after the shuffle itself. Imbued sinks to
/// the bottom every time; Perfect Fit rises to the top on every shuffle but
/// the one that opens the combat.
fn apply_shuffle_order(cards: &mut Vec<Card>, initial: bool) {
    let sinks = |c: &Card| c.enchantment.is_some_and(|e| e.starts_at_bottom());
    let rises = |c: &Card| !initial && c.enchantment.is_some_and(|e| e.shuffles_to_top());
    if !cards.iter().any(|c| sinks(c) || rises(c)) {
        return;
    }
    let mut top: Vec<Card> = vec![];
    let mut bottom: Vec<Card> = vec![];
    cards.retain(|c| {
        if rises(c) {
            top.push(c.clone());
            false
        } else if sinks(c) {
            bottom.push(c.clone());
            false
        } else {
            true
        }
    });
    top.append(cards);
    top.append(&mut bottom);
    *cards = top;
}

fn shuffle_cards(cards: &mut Vec<Card>, script: &mut Script, rng: &mut crate::rng::Rng, log: &mut Arc<Vec<Vec<(CardId, bool)>>>) {
    cards.sort_by_key(|c| (c.id, c.upgraded));
    match script.shuffles.pop_front() {
        Some(order) => {
            // Advance the stream as a real shuffle would, so later draws from
            // it (Stampede's pick) stay aligned with an unscripted run.
            let mut scratch: Vec<usize> = (0..cards.len()).collect();
            rng.shuffle(&mut scratch);
            let mut rest = std::mem::take(cards);
            for (id, up, ench) in order {
                let same = |c: &Card| c.id == id && c.upgraded == up;
                let exact = rest.iter().position(|c| same(c) && c.enchantment.map(|e| (e.id, e.disabled)) == ench);
                if let Some(i) = exact.or_else(|| rest.iter().position(same)) {
                    cards.push(rest.remove(i));
                }
            }
            cards.append(&mut rest);
        }
        None => {
            script.unscripted_shuffles += 1;
            rng.shuffle(cards);
        }
    }
    Arc::make_mut(log).push(cards.iter().map(|c| (c.id, c.upgraded)).collect());
}

/// The cards a `GenPool` draws from: `CardFactory.FilterForCombat` (can be
/// generated in combat, not Basic, not Ancient, which in the Ironclad pool
/// is every Special) and `FilterForPlayerCount` (no multiplayer-only cards).
fn pool_cards(pool: GenPool) -> Vec<CardId> {
    use crate::types::CardRarity::{Common, Rare, Uncommon};
    let source = match pool {
        GenPool::Colorless | GenPool::ColorlessAll => crate::card::COLORLESS_POOL,
        _ => IRONCLAD_POOL,
    };
    source
        .iter()
        .copied()
        .filter(|id| {
            let d = crate::card::def(*id);
            d.generatable
                && matches!(d.rarity, Common | Uncommon | Rare)
                && !crate::card::MULTIPLAYER_ONLY.contains(id)
                && match pool {
                    GenPool::Ironclad => true,
                    GenPool::IroncladAttacks => d.ty == CardType::Attack,
                    GenPool::IroncladSkills => d.ty == CardType::Skill,
                    GenPool::IroncladPowers => d.ty == CardType::Power,
                    GenPool::IroncladCommon => d.rarity == crate::types::CardRarity::Common,
                    GenPool::IroncladZeroCost => d.cost == 0 && !d.x_cost,
                    GenPool::Colorless => *id != CardId::JackOfAllTrades,
                    GenPool::ColorlessAll => true,
                }
        })
        .collect()
}

/// The three `DecimillipedeSegment`s.
fn is_segment(id: MonsterId) -> bool {
    matches!(id, MonsterId::DecimillipedeSegmentFront | MonsterId::DecimillipedeSegmentMiddle | MonsterId::DecimillipedeSegmentBack)
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
