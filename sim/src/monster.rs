//! Monsters and their move graphs. `Models/MonsterModel.cs` and
//! `MonsterMoves/MonsterMoveStateMachine/*.cs`. Each monster's graph is
//! ported verbatim from its `GenerateMoveStateMachine`, and each move body
//! from the matching `*Move` method.

use crate::effect::{AttackTargets, Effect, Pile};
use crate::ids::{CardId, MonsterId, PowerId};
use crate::rng::Rng;
use crate::types::{Ascension, AscensionLevel, CreatureRef, ValueProp};

/// What the intent display promises. `MonsterMoves/Intents/*.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// Base damage before the player's Strength/Vulnerable/Weak view.
    Attack { damage: i32, hits: u32 },
    Defend,
    Buff,
    Debuff { strong: bool },
    CardDebuff,
    Status { count: u32 },
    Summon,
    Sleep,
    Stun,
    Heal,
}

/// `MoveRepeatType.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    Forever,
    Times(u32),
    CannotRepeat,
    Once,
}

/// One weighted branch of a `RandomBranchState`.
#[derive(Clone, Debug)]
pub struct Branch {
    pub state: usize,
    pub cooldown: u32,
    pub repeat: Repeat,
    pub weight: f32,
}

/// Predicates used by `ConditionalBranchState`s, evaluated against the
/// monster's encounter flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    IsAlone,
    IsFront,
    NotFront,
    Slot(u8),
}

#[derive(Clone, Debug)]
pub enum State {
    Move {
        name: &'static str,
        intents: Vec<Intent>,
        follow_up: Option<usize>,
        must_perform_once: bool,
    },
    Random(Vec<Branch>),
    Conditional(Vec<(usize, Cond)>),
}

impl State {
    fn is_move(&self) -> bool {
        matches!(self, State::Move { .. })
    }
    /// `MonsterState.ShouldAppearInLogs`: only moves are logged.
    fn logged(&self) -> bool {
        self.is_move()
    }
    fn name(&self) -> &'static str {
        match self {
            State::Move { name, .. } => name,
            _ => "",
        }
    }
}

/// Encounter-set flags. `Nibbit.IsFront/IsAlone`, `Inklet.MiddleInklet`,
/// `KinFollower.StartsWithDance`, `Wriggler.StartStunned`, `Creature.SlotName`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    pub is_front: bool,
    pub is_alone: bool,
    pub middle: bool,
    pub starts_with_dance: bool,
    pub start_stunned: bool,
    /// 1-based slot number for monsters whose behaviour depends on it.
    pub slot: u8,
}

#[derive(Clone, Debug)]
pub struct Monster {
    pub id: MonsterId,
    pub flags: Flags,
    states: Vec<State>,
    initial: usize,
    current: usize,
    /// `MonsterMoveStateMachine.StateLog`, move states only.
    log: Vec<usize>,
    performed_first: bool,
    /// `MoveState._performedAtLeastOnce` for the current state.
    performed_current: bool,
    /// Index of the rolled `NextMove`, valid after `roll_move`.
    pub next_move: Option<usize>,
    /// `MonsterModel.SpawnedThisTurn`: no action until the next side switch.
    pub spawned_this_turn: bool,
}

impl Monster {
    pub fn new(id: MonsterId, asc: Ascension, flags: Flags) -> Self {
        let (states, initial) = graph(id, asc, flags);
        let mut m = Self {
            id,
            flags,
            states,
            initial,
            current: initial,
            log: vec![],
            performed_first: false,
            performed_current: false,
            next_move: None,
            spawned_this_turn: true,
        };
        if m.states[initial].logged() {
            m.log.push(initial);
        }
        m
    }

    /// `(MinInitialHp, MaxInitialHp)` at this ascension.
    pub fn hp_range(id: MonsterId, asc: Ascension) -> (i32, i32) {
        use AscensionLevel::ToughEnemies as T;
        use MonsterId::*;
        let r = |lo: i32, hi: i32, tlo: i32, thi: i32| (asc.pick(T, tlo, lo), asc.pick(T, thi, hi));
        let flat = |v: i32, tv: i32| (asc.pick(T, tv, v), asc.pick(T, tv, v));
        match id {
            Nibbit => r(42, 46, 44, 48),
            FuzzyWurmCrawler => r(55, 57, 58, 59),
            ShrinkerBeetle => r(38, 40, 40, 42),
            LeafSlimeS => r(11, 15, 12, 16),
            LeafSlimeM => r(32, 35, 33, 36),
            TwigSlimeS => r(7, 11, 8, 12),
            TwigSlimeM => r(26, 28, 27, 29),
            Inklet => r(11, 17, 12, 18),
            Mawler => flat(72, 76),
            Fogmog => flat(74, 78),
            EyeWithTeeth => (6, 6),
            Flyconid => r(47, 49, 51, 53),
            SnappingJaxfruit => r(31, 33, 34, 36),
            SlitheringStrangler => r(53, 55, 54, 56),
            VineShambler => flat(61, 64),
            CubexConstruct => flat(65, 70),
            AxeRubyRaider => r(20, 22, 21, 23),
            AssassinRubyRaider => r(18, 23, 19, 24),
            BruteRubyRaider => r(30, 33, 31, 34),
            CrossbowRubyRaider => r(18, 21, 19, 22),
            TrackerRubyRaider => r(21, 25, 22, 26),
            Byrdonis => r(81, 84, 90, 90),
            BygoneEffigy => flat(127, 132),
            PhrogParasite => r(61, 64, 66, 68),
            Wriggler => r(17, 21, 18, 22),
            Vantom => flat(173, 183),
            CeremonialBeast => flat(252, 262),
            KinFollower => r(58, 59, 62, 63),
            KinPriest => flat(190, 199),
        }
    }

    /// Powers applied in `AfterAddedToRoom`.
    pub fn innate_powers(id: MonsterId, asc: Ascension) -> Vec<(PowerId, i32)> {
        use MonsterId::*;
        match id {
            Inklet => vec![(PowerId::Slippery, 1)],
            // IllusionPower.AfterApplied also applies Minion.
            EyeWithTeeth => vec![(PowerId::Illusion, 1), (PowerId::Minion, 1)],
            CubexConstruct => vec![(PowerId::Artifact, 1)],
            Byrdonis => vec![(PowerId::Territorial, 1)],
            BygoneEffigy => vec![(PowerId::Slow, 1)],
            PhrogParasite => vec![(PowerId::Infested, 4)],
            Vantom => vec![(PowerId::Slippery, asc.pick(AscensionLevel::ToughEnemies, 9, 8))],
            KinFollower => vec![(PowerId::Minion, 1)],
            _ => vec![],
        }
    }

    /// Block gained in `AfterAddedToRoom` (Cubex Construct).
    pub fn innate_block(id: MonsterId) -> i32 {
        match id {
            MonsterId::CubexConstruct => 13,
            _ => 0,
        }
    }

    pub fn intents(&self) -> &[Intent] {
        match self.next_move.map(|i| &self.states[i]) {
            Some(State::Move { intents, .. }) => intents,
            _ => &[],
        }
    }

    pub fn next_move_name(&self) -> Option<&'static str> {
        self.next_move.map(|i| self.states[i].name())
    }

    /// `MonsterMoveStateMachine.RollMove` via `FindNextMoveState`.
    pub fn roll_move(&mut self, rng: &mut Rng) -> usize {
        if self.can_transition_away() && !(!self.performed_first && self.states[self.current].is_move()) {
            let mut first_logged = None;
            loop {
                let next = self.next_state(rng);
                self.current = next;
                self.performed_current = false;
                if first_logged.is_none() && self.states[next].logged() {
                    first_logged = Some(next);
                }
                if self.states[self.current].is_move() {
                    break;
                }
            }
            if let Some(s) = first_logged {
                self.log.push(s);
            }
        }
        assert!(self.states[self.current].is_move(), "rolled into a non-move state");
        self.next_move = Some(self.current);
        self.current
    }

    fn can_transition_away(&self) -> bool {
        match &self.states[self.current] {
            State::Move { must_perform_once, .. } => !must_perform_once || self.performed_current,
            _ => true,
        }
    }

    /// `Creature.StunInternal` + `MonsterModel.SetMoveImmediate`: replace the
    /// next move with a one-shot STUNNED move whose follow-up is `next`, or
    /// the last logged move when `None`.
    pub fn stun(&mut self, next: Option<&'static str>) {
        if !self.can_transition_away() {
            return;
        }
        let follow = match next {
            Some(n) => self.states.iter().position(|s| s.name() == n),
            None => self.log.last().copied(),
        };
        self.force_move("STUNNED", vec![Intent::Stun], follow);
    }

    /// `IllusionPower.AfterDeath`: a one-shot REVIVE move that heals to full.
    pub fn set_revive(&mut self) {
        let follow = self.log.last().copied();
        self.force_move("REVIVE_MOVE", vec![Intent::Heal], follow);
    }

    /// Replay: make `name` the next move as if the graph had rolled it.
    pub fn force_named_move(&mut self, name: &str) -> bool {
        let Some(idx) = self.states.iter().position(|s| s.is_move() && s.name() == name) else { return false };
        self.current = idx;
        self.performed_current = false;
        self.next_move = Some(idx);
        if self.states[idx].logged() && self.log.last() != Some(&idx) {
            self.log.push(idx);
        }
        true
    }

    fn force_move(&mut self, name: &'static str, intents: Vec<Intent>, follow_up: Option<usize>) {
        self.states.push(State::Move { name, intents, follow_up, must_perform_once: true });
        let idx = self.states.len() - 1;
        self.current = idx;
        self.performed_current = false;
        self.next_move = Some(idx);
    }

    /// `MonsterState.GetNextState` for the current state.
    fn next_state(&self, rng: &mut Rng) -> usize {
        match &self.states[self.current] {
            State::Move { follow_up, .. } => follow_up.unwrap_or(self.initial),
            State::Conditional(branches) => branches
                .iter()
                .find(|(_, c)| self.eval(*c))
                .map(|(s, _)| *s)
                .expect("no conditional branch matched"),
            State::Random(branches) => {
                let weights: Vec<f32> = branches.iter().map(|b| self.branch_weight(b)).collect();
                let total: f32 = weights.iter().sum();
                let mut roll = rng.next_float(total);
                for (b, w) in branches.iter().zip(&weights) {
                    roll -= w;
                    if roll <= 0.0 {
                        return b.state;
                    }
                }
                panic!("no random branch chosen")
            }
        }
    }

    fn eval(&self, c: Cond) -> bool {
        match c {
            Cond::IsAlone => self.flags.is_alone,
            Cond::IsFront => self.flags.is_front,
            Cond::NotFront => !self.flags.is_front,
            Cond::Slot(n) => self.flags.slot == n,
        }
    }

    /// `RandomBranchState.GetStateWeight`.
    fn branch_weight(&self, b: &Branch) -> f32 {
        let mut w = 1.0;
        match b.repeat {
            Repeat::Once => {
                if self.log.contains(&b.state) {
                    w = 0.0;
                }
            }
            Repeat::Forever => {}
            Repeat::CannotRepeat | Repeat::Times(_) => {
                let max = match b.repeat {
                    Repeat::CannotRepeat => 1,
                    Repeat::Times(n) => n as usize,
                    _ => unreachable!(),
                };
                // Zero unless one of the last `max` log entries is a different state.
                if self.log.len() >= max {
                    let tail = &self.log[self.log.len() - max..];
                    w = if tail.iter().all(|&s| s == b.state) { 0.0 } else { 1.0 };
                }
            }
        }
        if b.cooldown > 0 {
            let recent = self.log.iter().rev().take(b.cooldown as usize);
            if recent.into_iter().any(|&s| s == b.state) {
                return 0.0;
            }
        }
        w * b.weight
    }

    /// `MonsterModel.PerformMove`: the effects of `next_move`, then bookkeeping.
    pub fn perform(&mut self, me: CreatureRef, asc: Ascension) -> Vec<Effect> {
        let idx = self.next_move.expect("perform before roll");
        let name = self.states[idx].name();
        self.performed_first = true;
        self.performed_current = true;
        match name {
            "STUNNED" => vec![],
            "REVIVE_MOVE" => vec![Effect::Revive { target: me }],
            _ => moves(self.id, name, me, asc),
        }
    }
}

/// Monster attack from a move: `DamageCmd.Attack(x).FromMonster(this)`.
fn attack(me: CreatureRef, damage: i32, hits: u32) -> Effect {
    Effect::Attack {
        dealer: me,
        base: damage as f64,
        hits,
        targets: AttackTargets::AllOpponents,
        props: ValueProp::MOVE,
        card: None,
    }
}

fn block(me: CreatureRef, amount: i32) -> Effect {
    Effect::GainBlock { target: me, amount: amount as f64, props: ValueProp::MOVE, card: None }
}

fn buff(me: CreatureRef, id: PowerId, amount: i32) -> Effect {
    Effect::ApplyPower { target: me, id, amount, applier: Some(me) }
}

/// `PowerCmd.Apply<T>(targets, ...)` against every player creature.
fn debuff(me: CreatureRef, id: PowerId, amount: i32) -> Effect {
    Effect::ApplyPower { target: CreatureRef::Player, id, amount, applier: Some(me) }
}

/// `CardPileCmd.AddToCombatAndPreview<T>(targets, Discard, n)`.
fn statuses(id: CardId, n: u32) -> impl Iterator<Item = Effect> {
    (0..n).map(move |_| Effect::GenerateCard { id, upgraded: false, to: Pile::Discard, free_this_turn: false })
}

/// Per-monster move bodies. The `name` is the `MoveState` id string.
fn moves(id: MonsterId, name: &str, me: CreatureRef, asc: Ascension) -> Vec<Effect> {
    use AscensionLevel::{DeadlyEnemies as D, ToughEnemies as T};
    use MonsterId::*;
    let d = |a: i32, b: i32| asc.pick(D, b, a);
    match (id, name) {
        (Nibbit, "BUTT_MOVE") => vec![attack(me, d(12, 13), 1)],
        (Nibbit, "SLICE_MOVE") => vec![attack(me, d(6, 7), 1), block(me, asc.pick(T, 6, 5))],
        (Nibbit, "HISS_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],

        (FuzzyWurmCrawler, "FIRST_ACID_GOOP" | "ACID_GOOP") => vec![attack(me, d(4, 6), 1)],
        (FuzzyWurmCrawler, "INHALE") => vec![buff(me, PowerId::Strength, 7)],

        (ShrinkerBeetle, "SHRINKER_MOVE") => vec![debuff(me, PowerId::Shrink, -1)],
        (ShrinkerBeetle, "CHOMP_MOVE") => vec![attack(me, d(7, 8), 1)],
        (ShrinkerBeetle, "STOMP_MOVE") => vec![attack(me, d(13, 14), 1)],

        (LeafSlimeS, "TACKLE_MOVE") => vec![attack(me, d(3, 4), 1)],
        (LeafSlimeS, "GOOP_MOVE") => statuses(CardId::Slimed, 1).collect(),
        (LeafSlimeM, "CLUMP_SHOT") => vec![attack(me, d(8, 9), 1)],
        (LeafSlimeM, "STICKY_SHOT") => statuses(CardId::Slimed, 2).collect(),
        (TwigSlimeS, "TACKLE_MOVE") => vec![attack(me, d(4, 5), 1)],
        (TwigSlimeM, "POKEY_POUNCE_MOVE") => vec![attack(me, d(11, 12), 1)],
        (TwigSlimeM, "STICKY_SHOT_MOVE") => statuses(CardId::Slimed, 1).collect(),

        (Inklet, "JAB_MOVE") => vec![attack(me, d(3, 4), 1)],
        (Inklet, "WHIRLWIND_MOVE") => vec![attack(me, d(2, 3), 3)],
        (Inklet, "PIERCING_GAZE_MOVE") => vec![attack(me, d(10, 11), 1)],

        (Mawler, "RIP_AND_TEAR_MOVE") => vec![attack(me, d(14, 16), 1)],
        (Mawler, "ROAR_MOVE") => vec![debuff(me, PowerId::Vulnerable, 3)],
        (Mawler, "CLAW_MOVE") => vec![attack(me, d(4, 5), 2)],

        (Fogmog, "ILLUSION_MOVE") => vec![Effect::SpawnMonster { id: EyeWithTeeth, flags: Flags::default() }],
        (Fogmog, "SWIPE_MOVE" | "SWIPE_RANDOM_MOVE") => vec![attack(me, d(8, 9), 1), buff(me, PowerId::Strength, 1)],
        (Fogmog, "HEADBUTT_MOVE") => vec![attack(me, d(14, 16), 1)],
        (EyeWithTeeth, "DISTRACT_MOVE") => statuses(CardId::Dazed, 3).collect(),

        (Flyconid, "VULNERABLE_SPORES_MOVE") => vec![debuff(me, PowerId::Vulnerable, 2)],
        (Flyconid, "FRAIL_SPORES_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Frail, 2)],
        (Flyconid, "SMASH_MOVE") => vec![attack(me, d(11, 12), 1)],

        (SnappingJaxfruit, "ENERGY_ORB_MOVE") => vec![attack(me, d(3, 4), 1), buff(me, PowerId::Strength, 2)],

        (SlitheringStrangler, "CONSTRICT") => vec![debuff(me, PowerId::Constrict, 3)],
        (SlitheringStrangler, "THWACK") => vec![attack(me, d(7, 8), 1), block(me, 5)],
        (SlitheringStrangler, "LASH") => vec![attack(me, d(12, 13), 1)],

        (VineShambler, "GRASPING_VINES_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Tangled, 1)],
        (VineShambler, "SWIPE_MOVE") => vec![attack(me, d(6, 7), 2)],
        (VineShambler, "CHOMP_MOVE") => vec![attack(me, d(16, 18), 1)],

        (CubexConstruct, "CHARGE_UP_MOVE") => vec![buff(me, PowerId::Strength, 2)],
        (CubexConstruct, "REPEATER_BLAST_MOVE" | "REPEATER_BLAST_MOVE_2") => {
            vec![attack(me, d(7, 8), 1), buff(me, PowerId::Strength, 2)]
        }
        (CubexConstruct, "EXPEL_MOVE") => vec![attack(me, d(5, 6), 2)],

        (AxeRubyRaider, "SWING_1" | "SWING_2") => vec![attack(me, d(5, 6), 1), block(me, d(5, 6))],
        (AxeRubyRaider, "BIG_SWING") => vec![attack(me, d(12, 13), 1)],
        (AssassinRubyRaider, "KILLSHOT_MOVE") => vec![attack(me, d(10, 11), 1)],
        (BruteRubyRaider, "BEAT_MOVE") => vec![attack(me, d(7, 8), 1)],
        (BruteRubyRaider, "ROAR_MOVE") => vec![buff(me, PowerId::Strength, 3)],
        (CrossbowRubyRaider, "FIRE_MOVE") => vec![attack(me, d(14, 16), 1)],
        (CrossbowRubyRaider, "RELOAD_MOVE") => vec![block(me, 3)],
        (TrackerRubyRaider, "TRACK_MOVE") => vec![debuff(me, PowerId::Frail, 2)],
        (TrackerRubyRaider, "HOUNDS_MOVE") => vec![attack(me, 1, d(8, 9) as u32)],

        (Byrdonis, "PECK_MOVE") => vec![attack(me, d(3, 4), 3)],
        (Byrdonis, "SWOOP_MOVE") => vec![attack(me, d(17, 19), 1)],

        (BygoneEffigy, "SLEEP_MOVE" | "SLEEP_MOVE_2") => vec![],
        (BygoneEffigy, "WAKE_MOVE") => vec![buff(me, PowerId::Strength, 10)],
        (BygoneEffigy, "SLASHES_MOVE") => vec![attack(me, d(13, 15), 1)],

        (PhrogParasite, "INFECT_MOVE") => statuses(CardId::Infection, 3).collect(),
        (PhrogParasite, "LASH_MOVE") => vec![attack(me, d(4, 5), 4)],
        (Wriggler, "SPAWNED_MOVE") => vec![],
        (Wriggler, "NASTY_BITE_MOVE") => vec![attack(me, d(6, 7), 1)],
        (Wriggler, "WRIGGLE_MOVE") => statuses(CardId::Infection, 1).chain([buff(me, PowerId::Strength, 2)]).collect(),

        (Vantom, "INK_BLOT_MOVE") => vec![attack(me, d(7, 8), 1)],
        (Vantom, "INKY_LANCE_MOVE") => vec![attack(me, d(6, 7), 2)],
        (Vantom, "DISMEMBER_MOVE") => std::iter::once(attack(me, d(26, 30), 1)).chain(statuses(CardId::Wound, 3)).collect(),
        (Vantom, "PREPARE_MOVE") => vec![buff(me, PowerId::Strength, 2)],

        (CeremonialBeast, "STAMP_MOVE") => vec![buff(me, PowerId::Plow, d(150, 160))],
        (CeremonialBeast, "PLOW_MOVE") => vec![attack(me, d(18, 20), 1), buff(me, PowerId::Strength, 2)],
        (CeremonialBeast, "STUN_MOVE") => vec![],
        (CeremonialBeast, "BEAST_CRY_MOVE") => vec![debuff(me, PowerId::Ringing, 1)],
        (CeremonialBeast, "STOMP_MOVE") => vec![attack(me, d(15, 17), 1)],
        (CeremonialBeast, "CRUSH_MOVE") => vec![attack(me, d(17, 19), 1), buff(me, PowerId::Strength, d(3, 4))],

        (KinFollower, "QUICK_SLASH_MOVE") => vec![attack(me, 5, 1)],
        (KinFollower, "BOOMERANG_MOVE") => vec![attack(me, 2, 2)],
        (KinFollower, "POWER_DANCE_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],
        (KinPriest, "ORB_OF_FRAILTY_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Frail, 1)],
        (KinPriest, "ORB_OF_WEAKNESS_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Weak, 1)],
        (KinPriest, "BEAM_MOVE") => vec![attack(me, 3, 3)],
        (KinPriest, "RITUAL_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],

        _ => panic!("unknown move {name} for {id:?}"),
    }
}

/// Builder for move graphs. States are referenced by index; moves are
/// created first so branches can point at them.
struct G {
    states: Vec<State>,
}

impl G {
    fn new() -> Self {
        Self { states: vec![] }
    }
    fn mv(&mut self, name: &'static str, intents: Vec<Intent>) -> usize {
        self.states.push(State::Move { name, intents, follow_up: None, must_perform_once: false });
        self.states.len() - 1
    }
    fn mv_once(&mut self, name: &'static str, intents: Vec<Intent>) -> usize {
        let i = self.mv(name, intents);
        if let State::Move { must_perform_once, .. } = &mut self.states[i] {
            *must_perform_once = true;
        }
        i
    }
    fn random(&mut self, branches: Vec<Branch>) -> usize {
        self.states.push(State::Random(branches));
        self.states.len() - 1
    }
    fn cond(&mut self, branches: Vec<(usize, Cond)>) -> usize {
        self.states.push(State::Conditional(branches));
        self.states.len() - 1
    }
    fn follow(&mut self, from: usize, to: usize) {
        if let State::Move { follow_up, .. } = &mut self.states[from] {
            *follow_up = Some(to);
        }
    }
    fn done(self, initial: usize) -> (Vec<State>, usize) {
        (self.states, initial)
    }
}

fn br(state: usize, repeat: Repeat) -> Branch {
    Branch { state, cooldown: 0, repeat, weight: 1.0 }
}

fn atk(damage: i32) -> Intent {
    Intent::Attack { damage, hits: 1 }
}

fn multi(damage: i32, hits: u32) -> Intent {
    Intent::Attack { damage, hits }
}

/// Per-monster move graphs: `GenerateMoveStateMachine`.
fn graph(id: MonsterId, asc: Ascension, flags: Flags) -> (Vec<State>, usize) {
    use AscensionLevel::DeadlyEnemies as D;
    use Intent::*;
    use MonsterId::*;
    let d = |a: i32, b: i32| asc.pick(D, b, a);
    let mut g = G::new();
    match id {
        // Butt -> Slice -> Hiss -> Butt, entered by front/back/alone.
        Nibbit => {
            let butt = g.mv("BUTT_MOVE", vec![atk(d(12, 13))]);
            let slice = g.mv("SLICE_MOVE", vec![atk(d(6, 7)), Defend]);
            let hiss = g.mv("HISS_MOVE", vec![Buff]);
            let init = if flags.is_alone {
                g.cond(vec![(butt, Cond::IsAlone)])
            } else {
                g.cond(vec![(hiss, Cond::NotFront), (slice, Cond::IsFront)])
            };
            g.follow(slice, hiss);
            g.follow(butt, slice);
            g.follow(hiss, butt);
            g.done(init)
        }
        FuzzyWurmCrawler => {
            let first = g.mv("FIRST_ACID_GOOP", vec![atk(d(4, 6))]);
            let goop = g.mv("ACID_GOOP", vec![atk(d(4, 6))]);
            let inhale = g.mv("INHALE", vec![Buff]);
            g.follow(first, inhale);
            g.follow(inhale, goop);
            g.follow(goop, first);
            g.done(first)
        }
        ShrinkerBeetle => {
            let shrink = g.mv("SHRINKER_MOVE", vec![Debuff { strong: true }]);
            let chomp = g.mv("CHOMP_MOVE", vec![atk(d(7, 8))]);
            let stomp = g.mv("STOMP_MOVE", vec![atk(d(13, 14))]);
            g.follow(shrink, chomp);
            g.follow(chomp, stomp);
            g.follow(stomp, chomp);
            g.done(shrink)
        }
        LeafSlimeS => {
            let tackle = g.mv("TACKLE_MOVE", vec![atk(d(3, 4))]);
            let goop = g.mv("GOOP_MOVE", vec![Status { count: 1 }]);
            let rand = g.random(vec![br(tackle, Repeat::CannotRepeat), br(goop, Repeat::CannotRepeat)]);
            g.follow(tackle, rand);
            g.follow(goop, rand);
            g.done(rand)
        }
        LeafSlimeM => {
            let clump = g.mv("CLUMP_SHOT", vec![atk(d(8, 9))]);
            let sticky = g.mv("STICKY_SHOT", vec![Status { count: 2 }]);
            g.follow(sticky, clump);
            g.follow(clump, sticky);
            g.done(sticky)
        }
        TwigSlimeS => {
            let tackle = g.mv("TACKLE_MOVE", vec![atk(d(4, 5))]);
            g.follow(tackle, tackle);
            g.done(tackle)
        }
        TwigSlimeM => {
            let pounce = g.mv("POKEY_POUNCE_MOVE", vec![atk(d(11, 12))]);
            let sticky = g.mv("STICKY_SHOT_MOVE", vec![Status { count: 1 }]);
            let rand = g.random(vec![br(pounce, Repeat::Times(2)), br(sticky, Repeat::CannotRepeat)]);
            g.follow(pounce, rand);
            g.follow(sticky, rand);
            g.done(sticky)
        }
        // Jab -> RAND(gaze | whirlwind) -> Jab. Middle Inklet opens with Whirlwind.
        Inklet => {
            let jab = g.mv("JAB_MOVE", vec![atk(d(3, 4))]);
            let gaze = g.mv("PIERCING_GAZE_MOVE", vec![atk(d(10, 11))]);
            let whirl = g.mv("WHIRLWIND_MOVE", vec![multi(d(2, 3), 3)]);
            let rand = g.random(vec![br(gaze, Repeat::CannotRepeat), br(whirl, Repeat::CannotRepeat)]);
            g.follow(jab, rand);
            g.follow(whirl, jab);
            g.follow(gaze, jab);
            g.done(if flags.middle { whirl } else { jab })
        }
        Mawler => {
            let rip = g.mv("RIP_AND_TEAR_MOVE", vec![atk(d(14, 16))]);
            let roar = g.mv("ROAR_MOVE", vec![Debuff { strong: false }]);
            let claw = g.mv("CLAW_MOVE", vec![multi(d(4, 5), 2)]);
            let rand = g.random(vec![br(rip, Repeat::CannotRepeat), br(roar, Repeat::Once), br(claw, Repeat::CannotRepeat)]);
            g.follow(rip, rand);
            g.follow(roar, rand);
            g.follow(claw, rand);
            g.done(claw)
        }
        Fogmog => {
            let illusion = g.mv("ILLUSION_MOVE", vec![Summon]);
            let swipe = g.mv("SWIPE_MOVE", vec![atk(d(8, 9)), Buff]);
            let swipe_r = g.mv("SWIPE_RANDOM_MOVE", vec![atk(d(8, 9)), Buff]);
            let headbutt = g.mv("HEADBUTT_MOVE", vec![atk(d(14, 16))]);
            let branch = g.random(vec![
                Branch { state: swipe_r, cooldown: 0, repeat: Repeat::CannotRepeat, weight: 0.4 },
                Branch { state: headbutt, cooldown: 0, repeat: Repeat::CannotRepeat, weight: 0.6 },
            ]);
            g.follow(illusion, swipe);
            g.follow(swipe, branch);
            g.follow(swipe_r, headbutt);
            g.follow(headbutt, swipe);
            g.done(illusion)
        }
        EyeWithTeeth => {
            let distract = g.mv("DISTRACT_MOVE", vec![Status { count: 3 }]);
            g.follow(distract, distract);
            g.done(distract)
        }
        Flyconid => {
            let vuln = g.mv("VULNERABLE_SPORES_MOVE", vec![Debuff { strong: false }]);
            let frail = g.mv("FRAIL_SPORES_MOVE", vec![atk(d(8, 9)), Debuff { strong: false }]);
            let smash = g.mv("SMASH_MOVE", vec![atk(d(11, 12))]);
            let rand = g.random(vec![
                Branch { state: vuln, cooldown: 3, repeat: Repeat::CannotRepeat, weight: 1.0 },
                Branch { state: frail, cooldown: 2, repeat: Repeat::CannotRepeat, weight: 1.0 },
                br(smash, Repeat::CannotRepeat),
            ]);
            let initial = g.random(vec![
                Branch { state: frail, cooldown: 2, repeat: Repeat::CannotRepeat, weight: 1.0 },
                br(smash, Repeat::CannotRepeat),
            ]);
            g.follow(vuln, rand);
            g.follow(frail, rand);
            g.follow(smash, rand);
            g.done(initial)
        }
        SnappingJaxfruit => {
            let orb = g.mv("ENERGY_ORB_MOVE", vec![atk(d(3, 4)), Buff]);
            g.follow(orb, orb);
            g.done(orb)
        }
        SlitheringStrangler => {
            let constrict = g.mv("CONSTRICT", vec![Debuff { strong: false }]);
            let thwack = g.mv("THWACK", vec![atk(d(7, 8)), Defend]);
            let lash = g.mv("LASH", vec![atk(d(12, 13))]);
            let rand = g.random(vec![br(thwack, Repeat::Forever), br(lash, Repeat::Forever)]);
            g.follow(constrict, rand);
            g.follow(thwack, constrict);
            g.follow(lash, constrict);
            g.done(constrict)
        }
        VineShambler => {
            let vines = g.mv("GRASPING_VINES_MOVE", vec![atk(d(8, 9)), CardDebuff]);
            let swipe = g.mv("SWIPE_MOVE", vec![multi(d(6, 7), 2)]);
            let chomp = g.mv("CHOMP_MOVE", vec![atk(d(16, 18))]);
            g.follow(swipe, vines);
            g.follow(vines, chomp);
            g.follow(chomp, swipe);
            g.done(swipe)
        }
        CubexConstruct => {
            let charge = g.mv("CHARGE_UP_MOVE", vec![Buff]);
            let blast1 = g.mv("REPEATER_BLAST_MOVE", vec![atk(d(7, 8)), Buff]);
            let blast2 = g.mv("REPEATER_BLAST_MOVE_2", vec![atk(d(7, 8)), Buff]);
            let expel = g.mv("EXPEL_MOVE", vec![multi(d(5, 6), 2)]);
            g.follow(charge, blast1);
            g.follow(blast1, blast2);
            g.follow(blast2, expel);
            g.follow(expel, blast1);
            g.done(charge)
        }
        AxeRubyRaider => {
            let s1 = g.mv("SWING_1", vec![atk(d(5, 6)), Defend]);
            let s2 = g.mv("SWING_2", vec![atk(d(5, 6)), Defend]);
            let big = g.mv("BIG_SWING", vec![atk(d(12, 13))]);
            g.follow(s1, s2);
            g.follow(s2, big);
            g.follow(big, s1);
            g.done(s1)
        }
        AssassinRubyRaider => {
            let k = g.mv("KILLSHOT_MOVE", vec![atk(d(10, 11))]);
            g.follow(k, k);
            g.done(k)
        }
        BruteRubyRaider => {
            let beat = g.mv("BEAT_MOVE", vec![atk(d(7, 8))]);
            let roar = g.mv("ROAR_MOVE", vec![Buff]);
            g.follow(beat, roar);
            g.follow(roar, beat);
            g.done(beat)
        }
        CrossbowRubyRaider => {
            let fire = g.mv("FIRE_MOVE", vec![atk(d(14, 16))]);
            let reload = g.mv("RELOAD_MOVE", vec![Defend]);
            g.follow(fire, reload);
            g.follow(reload, fire);
            g.done(reload)
        }
        TrackerRubyRaider => {
            let track = g.mv("TRACK_MOVE", vec![Debuff { strong: false }]);
            let hounds = g.mv("HOUNDS_MOVE", vec![multi(1, d(8, 9) as u32)]);
            g.follow(track, hounds);
            g.follow(hounds, hounds);
            g.done(track)
        }
        Byrdonis => {
            let peck = g.mv("PECK_MOVE", vec![multi(d(3, 4), 3)]);
            let swoop = g.mv("SWOOP_MOVE", vec![atk(d(17, 19))]);
            g.follow(swoop, peck);
            g.follow(peck, swoop);
            g.done(swoop)
        }
        BygoneEffigy => {
            let sleep = g.mv("SLEEP_MOVE", vec![Sleep]);
            let wake = g.mv("WAKE_MOVE", vec![Buff]);
            let sleep2 = g.mv("SLEEP_MOVE_2", vec![Sleep]);
            let slash = g.mv("SLASHES_MOVE", vec![atk(d(13, 15))]);
            g.follow(sleep, wake);
            g.follow(wake, slash);
            g.follow(sleep2, slash);
            g.follow(slash, slash);
            g.done(sleep)
        }
        PhrogParasite => {
            let infect = g.mv("INFECT_MOVE", vec![Status { count: 3 }]);
            let lash = g.mv("LASH_MOVE", vec![multi(d(4, 5), 4)]);
            g.follow(infect, lash);
            g.follow(lash, infect);
            g.done(infect)
        }
        Wriggler => {
            let bite = g.mv("NASTY_BITE_MOVE", vec![atk(d(6, 7))]);
            let wriggle = g.mv("WRIGGLE_MOVE", vec![Buff, Status { count: 1 }]);
            let spawned = g.mv("SPAWNED_MOVE", vec![Stun]);
            let init = g.cond(vec![(bite, Cond::Slot(1)), (wriggle, Cond::Slot(2)), (bite, Cond::Slot(3)), (wriggle, Cond::Slot(4))]);
            g.follow(spawned, init);
            g.follow(bite, wriggle);
            g.follow(wriggle, bite);
            g.done(if flags.start_stunned { spawned } else { init })
        }
        Vantom => {
            let blot = g.mv("INK_BLOT_MOVE", vec![atk(d(7, 8))]);
            let lance = g.mv("INKY_LANCE_MOVE", vec![multi(d(6, 7), 2)]);
            let dismember = g.mv("DISMEMBER_MOVE", vec![atk(d(26, 30)), Status { count: 3 }]);
            let prepare = g.mv("PREPARE_MOVE", vec![Buff]);
            g.follow(blot, lance);
            g.follow(lance, dismember);
            g.follow(dismember, prepare);
            g.follow(prepare, blot);
            g.done(blot)
        }
        // Stamp -> Plow (loop) until Plow breaks, then Stun -> Cry -> Stomp -> Crush -> Cry.
        CeremonialBeast => {
            let stamp = g.mv("STAMP_MOVE", vec![Buff]);
            let plow = g.mv("PLOW_MOVE", vec![atk(d(18, 20)), Buff]);
            let stun = g.mv_once("STUN_MOVE", vec![Stun]);
            let cry = g.mv("BEAST_CRY_MOVE", vec![Debuff { strong: false }]);
            let stomp = g.mv("STOMP_MOVE", vec![atk(d(15, 17))]);
            let crush = g.mv("CRUSH_MOVE", vec![atk(d(17, 19)), Buff]);
            g.follow(stamp, plow);
            g.follow(plow, plow);
            g.follow(stun, cry);
            g.follow(cry, stomp);
            g.follow(stomp, crush);
            g.follow(crush, cry);
            g.done(stamp)
        }
        KinFollower => {
            let slash = g.mv("QUICK_SLASH_MOVE", vec![atk(5)]);
            let boomerang = g.mv("BOOMERANG_MOVE", vec![multi(2, 2)]);
            let dance = g.mv("POWER_DANCE_MOVE", vec![Buff]);
            g.follow(slash, boomerang);
            g.follow(boomerang, dance);
            g.follow(dance, slash);
            g.done(if flags.starts_with_dance { dance } else { slash })
        }
        KinPriest => {
            let frailty = g.mv("ORB_OF_FRAILTY_MOVE", vec![atk(d(8, 9)), Debuff { strong: false }]);
            let weakness = g.mv("ORB_OF_WEAKNESS_MOVE", vec![atk(d(8, 9)), Debuff { strong: false }]);
            let beam = g.mv("BEAM_MOVE", vec![multi(3, 3)]);
            let ritual = g.mv("RITUAL_MOVE", vec![Buff]);
            g.follow(frailty, weakness);
            g.follow(weakness, beam);
            g.follow(beam, ritual);
            g.follow(ritual, frailty);
            g.done(frailty)
        }
    }
}

/// Every move name any act 1 monster can show as its next move, for the
/// policy's move vocabulary. Built once from every graph variant.
pub fn all_move_names() -> &'static [&'static str] {
    static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names = vec!["STUNNED", "REVIVE_MOVE"];
        let variants = [
            Flags::default(),
            Flags { is_alone: true, ..Default::default() },
            Flags { is_front: true, ..Default::default() },
            Flags { middle: true, ..Default::default() },
            Flags { starts_with_dance: true, ..Default::default() },
            Flags { start_stunned: true, ..Default::default() },
        ];
        for &id in crate::ids::ALL_MONSTERS {
            for flags in variants {
                for asc in [Ascension(0), Ascension(10)] {
                    for s in graph(id, asc, flags).0 {
                        if let State::Move { name, .. } = s {
                            if !names.contains(&name) {
                                names.push(name);
                            }
                        }
                    }
                }
            }
        }
        names
    })
}

/// Index into `all_move_names`.
pub fn move_index(name: &str) -> Option<usize> {
    static INDEX: std::sync::OnceLock<std::collections::HashMap<&'static str, usize>> = std::sync::OnceLock::new();
    INDEX.get_or_init(|| all_move_names().iter().enumerate().map(|(i, &n)| (n, i)).collect()).get(name).copied()
}
