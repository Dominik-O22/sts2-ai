//! Monsters and their move graphs. `Models/MonsterModel.cs` and
//! `MonsterMoves/MonsterMoveStateMachine/*.cs`. Each monster's graph is
//! ported verbatim from its `GenerateMoveStateMachine`.

use crate::effect::{AttackTargets, Effect};
use crate::ids::{MonsterId, PowerId};
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
    Status { count: u32 },
    Stun,
    Unknown,
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
    Always,
    IsAlone,
    IsFront,
    NotFront,
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
}

/// Encounter-set flags. `Nibbit.IsFront/IsAlone`, `Inklet.MiddleInklet`, etc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    pub is_front: bool,
    pub is_alone: bool,
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
        };
        if m.states[initial].logged() {
            m.log.push(initial);
        }
        m
    }

    /// `(MinInitialHp, MaxInitialHp)` at this ascension.
    pub fn hp_range(id: MonsterId, asc: Ascension) -> (i32, i32) {
        use AscensionLevel::ToughEnemies as T;
        match id {
            MonsterId::Nibbit => (asc.pick(T, 44, 42), asc.pick(T, 48, 46)),
        }
    }

    /// Powers applied in `AfterAddedToRoom`.
    pub fn innate_powers(id: MonsterId, _asc: Ascension) -> Vec<(PowerId, i32)> {
        match id {
            MonsterId::Nibbit => vec![],
        }
    }

    pub fn intents(&self) -> &[Intent] {
        match self.next_move.map(|i| &self.states[i]) {
            Some(State::Move { intents, .. }) => intents,
            _ => &[],
        }
    }

    pub fn next_move_name(&self) -> Option<&'static str> {
        match self.next_move.map(|i| &self.states[i]) {
            Some(State::Move { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// `MonsterMoveStateMachine.RollMove` via `FindNextMoveState`.
    pub fn roll_move(&mut self, rng: &mut Rng) -> usize {
        let can_leave = match &self.states[self.current] {
            State::Move { must_perform_once, .. } => !must_perform_once || self.performed_current,
            _ => true,
        };
        let stuck_on_first = !self.performed_first && self.states[self.current].is_move();
        if can_leave && !stuck_on_first {
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
            Cond::Always => true,
            Cond::IsAlone => self.flags.is_alone,
            Cond::IsFront => self.flags.is_front,
            Cond::NotFront => !self.flags.is_front,
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
        let name = match &self.states[idx] {
            State::Move { name, .. } => *name,
            _ => unreachable!(),
        };
        self.performed_first = true;
        self.performed_current = true;
        moves(self.id, name, me, asc)
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

/// Per-monster move bodies. The `name` is the `MoveState` id string.
fn moves(id: MonsterId, name: &str, me: CreatureRef, asc: Ascension) -> Vec<Effect> {
    use AscensionLevel::{DeadlyEnemies as D, ToughEnemies as T};
    match (id, name) {
        // Models/Monsters/Nibbit.cs
        (MonsterId::Nibbit, "BUTT_MOVE") => vec![attack(me, asc.pick(D, 13, 12), 1)],
        (MonsterId::Nibbit, "SLICE_MOVE") => vec![
            attack(me, asc.pick(D, 7, 6), 1),
            Effect::GainBlock {
                target: me,
                amount: asc.pick(T, 6, 5) as f64,
                props: ValueProp::MOVE,
                card: None,
            },
        ],
        (MonsterId::Nibbit, "HISS_MOVE") => vec![Effect::ApplyPower {
            target: me,
            id: PowerId::Strength,
            amount: asc.pick(D, 3, 2),
            applier: Some(me),
        }],
        _ => panic!("unknown move {name} for {id:?}"),
    }
}

/// Per-monster move graphs: `GenerateMoveStateMachine`.
fn graph(id: MonsterId, asc: Ascension, flags: Flags) -> (Vec<State>, usize) {
    use AscensionLevel::DeadlyEnemies as D;
    match id {
        // Models/Monsters/Nibbit.cs. States: 0 INIT (conditional), 1 BUTT,
        // 2 SLICE, 3 HISS. Butt -> Slice -> Hiss -> Butt.
        MonsterId::Nibbit => {
            let init = if flags.is_alone {
                State::Conditional(vec![(1, Cond::IsAlone)])
            } else {
                State::Conditional(vec![(3, Cond::NotFront), (2, Cond::IsFront)])
            };
            let states = vec![
                init,
                State::Move {
                    name: "BUTT_MOVE",
                    intents: vec![Intent::Attack { damage: asc.pick(D, 13, 12), hits: 1 }],
                    follow_up: Some(2),
                    must_perform_once: false,
                },
                State::Move {
                    name: "SLICE_MOVE",
                    intents: vec![Intent::Attack { damage: asc.pick(D, 7, 6), hits: 1 }, Intent::Defend],
                    follow_up: Some(3),
                    must_perform_once: false,
                },
                State::Move {
                    name: "HISS_MOVE",
                    intents: vec![Intent::Buff],
                    follow_up: Some(1),
                    must_perform_once: false,
                },
            ];
            (states, 0)
        }
    }
}
