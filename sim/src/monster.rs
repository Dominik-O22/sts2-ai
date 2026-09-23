//! Monsters and their move graphs. `Models/MonsterModel.cs` and
//! `MonsterMoves/MonsterMoveStateMachine/*.cs`. Each monster's graph is
//! ported verbatim from its `GenerateMoveStateMachine`, and each move body
//! from the matching `*Move` method.

use std::sync::Arc;

use crate::effect::{AttackTargets, Effect, Pile};
use crate::ids::{CardId, MonsterId, PowerId};
use crate::rng::Rng;
use crate::types::{Ascension, AscensionLevel, CreatureRef, ValueProp};

/// `Effect::MonsterStep` codes: move continuations that need the combat state.
/// Bowlbug Rock staggers if its headbutt was fully blocked.
pub const STEP_STAGGER: u8 = 0;
/// Thieving Hopper's last Flutter charge is gone: it drops, skipping a move.
pub const STEP_FLUTTER_DOWN: u8 = 1;
/// Thieving Hopper takes a card from your draw or discard pile.
pub const STEP_STEAL: u8 = 2;
/// Entomancer grows its hive, or its Strength once the hive is full.
pub const STEP_PHEROMONE: u8 = 3;
/// The Obscura's Sail: Strength to its whole side.
pub const STEP_SAIL: u8 = 4;

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
    /// `EscapeIntent`: leaving combat without dying (Fat Gremlin).
    Escape,
    /// `DeathBlowIntent`: attack, then the attacker kills itself.
    DeathBlow { damage: i32 },
}

/// Intent kinds by index, the policy's intent vocabulary. `sim/vocab.txt`
/// pins the order, so a new kind goes at the end of both this list and
/// `Intent::kind`.
pub const INTENT_KINDS: &[&str] =
    &["Attack", "Defend", "Buff", "Debuff", "CardDebuff", "Status", "Summon", "Sleep", "Stun", "Heal", "Escape", "DeathBlow"];

impl Intent {
    /// Index into `INTENT_KINDS`.
    pub fn kind(&self) -> usize {
        match self {
            Intent::Attack { .. } => 0,
            Intent::Defend => 1,
            Intent::Buff => 2,
            Intent::Debuff { .. } => 3,
            Intent::CardDebuff => 4,
            Intent::Status { .. } => 5,
            Intent::Summon => 6,
            Intent::Sleep => 7,
            Intent::Stun => 8,
            Intent::Heal => 9,
            Intent::Escape => 10,
            Intent::DeathBlow { .. } => 11,
        }
    }
}

/// `MoveRepeatType.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    Forever,
    Times(u32),
    CannotRepeat,
    Once,
}

/// `RandomBranchState.AddBranch`'s weight argument. Most are constants; the
/// rest are `Func<float>` closures reading the combat state.
#[derive(Clone, Copy, Debug)]
pub enum Weight {
    Fixed(f32),
    /// `TwoTailedRat`: `CanSummon() ? yes : no`.
    IfCanSummon { yes: f32, no: f32 },
}

impl Weight {
    fn value(self, ctx: RollCtx) -> f32 {
        match self {
            Weight::Fixed(w) => w,
            Weight::IfCanSummon { yes, no } => if ctx.can_summon { yes } else { no },
        }
    }
}

/// One weighted branch of a `RandomBranchState`.
#[derive(Clone, Debug)]
pub struct Branch {
    pub state: usize,
    pub cooldown: u32,
    pub repeat: Repeat,
    pub weight: Weight,
}

/// Predicates used by `ConditionalBranchState`s, evaluated against the
/// monster's encounter flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    IsAlone,
    IsFront,
    NotFront,
    Slot(u8),
    /// `Creature.HasPower<AsleepPower>` (Lagavulin Matriarch's sleep branch).
    Asleep,
    NotAsleep,
    /// `BowlbugRock.IsOffBalance`.
    OffBalance,
    Balanced,
    /// `Creature.HasPower<SlumberPower>` (Slumbering Beetle).
    Slumbering,
    Awake,
    /// `Ovicopter.CanLay`: three or fewer of its side alive, itself included.
    CanLay,
    CannotLay,
    /// Knowledge Demon still has a Curse of Knowledge set left to hand out.
    CursesLeft,
    CursesDone,
    /// `LivingShield.GetAllyCount() > 0`: another enemy is still alive.
    HasAllies,
    NoAllies,
    /// `Fabricator.CanFabricate`: fewer than four living on its side.
    CanFabricate,
    CannotFabricate,
    /// `FrogKnight`'s HALF_HEALTH branch: the charge comes once, below half.
    BeetleCharge,
    NoBeetleCharge,
    /// `Queen.HasAmalgamDied`.
    AmalgamAlive,
    AmalgamDead,
    /// `TestSubject.Respawns < 2`: still in its second form.
    SecondForm,
    ThirdForm,
}

/// What a branch predicate may read outside the monster itself. The game's
/// closures reach into `Creature` and `CombatState`; these are the pieces
/// act 1's graphs actually ask for.
#[derive(Clone, Copy, Debug, Default)]
pub struct RollCtx {
    /// `Creature.HasPower<AsleepPower>()`.
    pub asleep: bool,
    /// `TwoTailedRat.CanSummon()`, which folds together its own counters, the
    /// free encounter slots, and whether a peer is already calling.
    pub can_summon: bool,
    /// `Creature.HasPower<SlumberPower>()`.
    pub slumbering: bool,
    /// Living creatures on the monster's side, itself included.
    pub living_allies: usize,
    /// Living enemies other than this one (Living Shield, Fabricator).
    pub allies_alive: usize,
    /// `CurrentHp < MaxHp / 2`, integer division (Frog Knight).
    pub below_half: bool,
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
    /// `CorpseSlug.StarterMoveIdx` / `TwoTailedRat.StarterMoveIndex`: which
    /// move of three the monster opens on. Also `DecimillipedeSegment.StarterMoveIdx`.
    pub starter_move: u8,
    /// `Chomper.ScreamFirst`: the second chomper opens on Screech.
    pub scream_first: bool,
    /// `Axebot._stockOverrideAmount`: set on an Axebot its Stock respawned,
    /// which opens on Boot Up with this many respawns left.
    pub stock: Option<u8>,
}

/// Counters a monster keeps across its own moves, each named after the field
/// it ports. Fields the move graph reads are also mirrored into `RollCtx`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vars {
    /// `TwoTailedRat.TurnsUntilSummonable`, starting at 2.
    pub turns_until_summonable: i32,
    /// `TwoTailedRat.CallForBackupCount`, kept in step across all the rats.
    pub call_for_backup_count: i32,
    /// `WaterfallGiant.CurrentPressureGunDamage`, growing by 5 a shot.
    pub pressure_gun_damage: i32,
    /// `WaterfallGiant.SteamEruptionDamage`, the pressure the blast carries.
    pub steam_eruption_damage: i32,
    /// `BowlbugRock.IsOffBalance`: its last headbutt was fully blocked.
    pub off_balance: bool,
    /// `KnowledgeDemon.CurseOfKnowledgeCounter`: sets handed out so far.
    pub curses_given: i32,
    /// `Axebot.StockAmount`: 2, or what a respawn was left with.
    pub stock: i32,
    /// `FrogKnight.HasBeetleCharged`.
    pub beetle_charged: bool,
    /// `Queen.HasAmalgamDied`.
    pub amalgam_died: bool,
    /// `TestSubject.Respawns` and `ExtraMultiClawCount`.
    pub respawns: i32,
    pub extra_claws: i32,
    /// `Aeonglass.WitherUpgradeCount` and `AdditionalStrength`.
    pub wither_upgrades: i32,
    pub extra_strength: i32,
    /// The monster's own Dexterity when it acts (The Forgotten's Dread).
    pub own_dex: i32,
    /// `Fabricator._lastSpawned`: the next bot is never the same one.
    pub last_spawned: Option<MonsterId>,
}

#[derive(Clone, Debug)]
pub struct Monster {
    pub id: MonsterId,
    pub flags: Flags,
    pub vars: Vars,
    /// The move graph, shared between clones until a stun or a death blow
    /// rewrites it.
    states: Arc<Vec<State>>,
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
            vars: Vars {
                turns_until_summonable: 2,
                pressure_gun_damage: Self::pressure_gun_base(id, asc),
                stock: flags.stock.map_or(2, i32::from),
                ..Vars::default()
            },
            states: Arc::new(states),
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

            Toadpole => r(21, 25, 22, 26),
            CorpseSlug => r(25, 27, 27, 29),
            DampCultist => r(51, 53, 52, 54),
            CalcifiedCultist => r(38, 41, 39, 42),
            FossilStalker => r(51, 53, 54, 56),
            GremlinMerc => r(47, 49, 51, 53),
            FatGremlin => r(13, 17, 14, 18),
            SneakyGremlin => r(10, 14, 11, 15),
            GasBomb => flat(7, 8),
            HauntedShip => flat(63, 67),
            LivingFog => flat(80, 82),
            PhantasmalGardener => r(26, 31, 27, 32),
            PunchConstruct => flat(55, 60),
            Seapunk => r(44, 46, 47, 49),
            SewerClam => flat(56, 58),
            SkulkingColony => flat(75, 80),
            SludgeSpinner => r(37, 39, 41, 42),
            TwoTailedRat => r(17, 21, 18, 22),
            TerrorEel => flat(140, 150),
            LagavulinMatriarch => flat(222, 233),
            SoulFysh => flat(211, 221),
            WaterfallGiant => flat(240, 250),

            BowlbugEgg => r(21, 22, 23, 24),
            BowlbugNectar => r(35, 38, 36, 39),
            BowlbugRock => r(45, 48, 46, 49),
            BowlbugSilk => r(40, 43, 41, 44),
            Chomper => r(60, 64, 63, 67),
            Crusher => flat(209, 219),
            Rocket => flat(199, 209),
            DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack => r(40, 46, 46, 52),
            Entomancer => flat(145, 155),
            Exoskeleton => r(24, 28, 25, 29),
            HunterKiller => flat(121, 126),
            InfestedPrism => flat(161, 171),
            KnowledgeDemon => flat(379, 399),
            LouseProgenitor => r(134, 136, 138, 141),
            Myte => r(61, 67, 64, 69),
            Ovicopter => r(124, 130, 126, 132),
            ToughEgg => r(14, 18, 15, 19),
            SlumberingBeetle => flat(86, 89),
            SpinyToad => r(116, 119, 121, 124),
            TheInsatiable => flat(321, 341),
            TheObscura => flat(123, 129),
            Parafright => (21, 21),
            ThievingHopper => flat(79, 84),
            Tunneler => flat(87, 92),
            Axebot => r(70, 78, 76, 86),
            DevotedSculptor => flat(162, 172),
            FrogKnight => flat(191, 199),
            GlobeHead => flat(148, 158),
            OwlMagistrate => flat(231, 247),
            ScrollOfBiting => r(30, 37, 33, 39),
            SlimedBerserker => flat(261, 281),
            LivingShield => flat(55, 65),
            TurretOperator => flat(41, 51),
            TheLost => flat(93, 99),
            TheForgotten => flat(106, 111),
            MechaKnight => flat(300, 320),
            FlailKnight => flat(101, 108),
            SpectralKnight => flat(93, 97),
            MagiKnight => flat(82, 89),
            SoulNexus => flat(234, 254),
            Fabricator => flat(150, 155),
            Zapbot | Stabbot | Noisebot => r(18, 23, 19, 24),
            Guardbot => r(16, 20, 17, 21),
            Queen => flat(400, 419),
            TorchHeadAmalgam => flat(199, 211),
            // The first of its three forms.
            TestSubject => flat(100, 111),
            Aeonglass => flat(512, 535),
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

            CorpseSlug => vec![(PowerId::Ravenous, asc.pick(AscensionLevel::DeadlyEnemies, 5, 4))],
            FossilStalker => vec![(PowerId::Suck, 3)],
            GremlinMerc => vec![(PowerId::Surprise, 1), (PowerId::Thievery, 20)],
            GasBomb => vec![(PowerId::Minion, 1)],
            PhantasmalGardener => vec![(PowerId::Skittish, asc.pick(AscensionLevel::ToughEnemies, 7, 6))],
            PunchConstruct => vec![(PowerId::Artifact, 1)],
            SewerClam => vec![(PowerId::Plating, asc.pick(AscensionLevel::ToughEnemies, 9, 8))],
            SkulkingColony => vec![(PowerId::HardenedShell, 20)],
            TerrorEel => vec![(PowerId::Shriek, asc.pick(AscensionLevel::ToughEnemies, 75, 70))],
            // LagavulinMatriarch.Sleep: shell first, then the nap counter.
            LagavulinMatriarch => vec![(PowerId::Plating, 12), (PowerId::Asleep, 3)],

            BowlbugRock => vec![(PowerId::Imbalanced, 1)],
            Chomper => vec![(PowerId::Artifact, 2)],
            Crusher => vec![(PowerId::BackAttackLeft, 1), (PowerId::CrabRage, 1)],
            Rocket => vec![(PowerId::BackAttackRight, 1), (PowerId::CrabRage, 1)],
            DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack => vec![(PowerId::Reattach, 25)],
            Entomancer => vec![(PowerId::PersonalHive, 1)],
            Exoskeleton => vec![(PowerId::HardToKill, 9)],
            InfestedPrism => vec![(PowerId::VitalSpark, asc.pick(AscensionLevel::DeadlyEnemies, 3, 2))],
            LouseProgenitor => vec![(PowerId::CurlUp, asc.pick(AscensionLevel::ToughEnemies, 18, 14))],
            SlumberingBeetle => vec![(PowerId::Plating, asc.pick(AscensionLevel::ToughEnemies, 18, 15)), (PowerId::Slumber, 3)],
            // IllusionPower.AfterApplied also applies Minion.
            Parafright => vec![(PowerId::Illusion, 1), (PowerId::Minion, 1)],
            ThievingHopper => vec![(PowerId::EscapeArtist, 5)],
            // A respawned Axebot carries fewer; `Combat::spawn` sets that.
            Axebot => vec![(PowerId::Stock, 2)],
            FrogKnight => vec![(PowerId::Plating, asc.pick(AscensionLevel::ToughEnemies, 19, 15))],
            GlobeHead => vec![(PowerId::Galvanic, 6)],
            ScrollOfBiting => vec![(PowerId::PaperCuts, 2)],
            LivingShield => vec![(PowerId::Rampart, 25)],
            TheLost => vec![(PowerId::PossessStrength, 1)],
            TheForgotten => vec![(PowerId::PossessSpeed, 1)],
            MechaKnight => vec![(PowerId::Artifact, 3)],
            Zapbot => vec![(PowerId::HighVoltage, 2)],
            TorchHeadAmalgam => vec![(PowerId::Minion, 1)],
            TestSubject => vec![(PowerId::Adaptable, 1), (PowerId::Enrage, asc.pick(AscensionLevel::DeadlyEnemies, 3, 2))],
            // WitheringPresencePower counts six cards; the amount is that count.
            Aeonglass => vec![(PowerId::WitheringPresence, 6), (PowerId::Artifact, 3)],
            _ => vec![],
        }
    }

    /// `WaterfallGiant.BasePressureGunDamage`, the opening value of the shot
    /// that grows by 5 each time it fires.
    pub fn pressure_gun_base(id: MonsterId, asc: Ascension) -> i32 {
        match id {
            MonsterId::WaterfallGiant => asc.pick(AscensionLevel::DeadlyEnemies, 23, 20),
            _ => 0,
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
    pub fn roll_move(&mut self, rng: &mut Rng, ctx: RollCtx) -> usize {
        if self.can_transition_away() && !(!self.performed_first && self.states[self.current].is_move()) {
            let mut first_logged = None;
            loop {
                let next = self.next_state(rng, ctx);
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

    /// `FlutterPower`'s knock-down: a stun whose follow-up is the move after
    /// the one it had queued (`StateLog.Last().GetNextState`).
    pub fn stun_past_next(&mut self) {
        if !self.can_transition_away() {
            return;
        }
        let follow = self.log.last().and_then(|&i| match &self.states[i] {
            State::Move { follow_up, .. } => Some(follow_up.unwrap_or(self.initial)),
            _ => None,
        });
        self.force_move("STUNNED", vec![Intent::Stun], follow);
    }

    /// `WaterfallGiant.TriggerAboutToBlowState`: jump straight to the wind-up,
    /// whatever the graph had queued (`SetMoveImmediate(forceTransition: true)`).
    pub fn force_about_to_blow(&mut self) {
        self.force_to("ABOUT_TO_BLOW_MOVE");
    }

    /// `MonsterModel.SetMoveImmediate`: `name` becomes the next move at once
    /// (Test Subject's death, the Queen enraging when her Amalgam falls).
    pub fn force_to(&mut self, name: &str) {
        let Some(idx) = self.states.iter().position(|s| s.name() == name) else { return };
        self.current = idx;
        self.performed_current = false;
        self.next_move = Some(idx);
    }

    /// `WaterfallGiant.SteamEruptionDamage`: bank the pressure, and show it on
    /// the blast's intent.
    pub fn arm_death_blow(&mut self, damage: i32) {
        self.vars.steam_eruption_damage = damage;
        for st in Arc::make_mut(&mut self.states) {
            if let State::Move { name: "EXPLODE_MOVE", intents, .. } = st {
                *intents = vec![Intent::DeathBlow { damage }];
            }
        }
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
        Arc::make_mut(&mut self.states).push(State::Move { name, intents, follow_up, must_perform_once: true });
        let idx = self.states.len() - 1;
        self.current = idx;
        self.performed_current = false;
        self.next_move = Some(idx);
    }

    /// `MonsterState.GetNextState` for the current state.
    fn next_state(&self, rng: &mut Rng, ctx: RollCtx) -> usize {
        match &self.states[self.current] {
            State::Move { follow_up, .. } => follow_up.unwrap_or(self.initial),
            State::Conditional(branches) => branches
                .iter()
                .find(|(_, c)| self.eval(*c, ctx))
                .map(|(s, _)| *s)
                .expect("no conditional branch matched"),
            State::Random(branches) => {
                let weights: Vec<f32> = branches.iter().map(|b| self.branch_weight(b, ctx)).collect();
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

    fn eval(&self, c: Cond, ctx: RollCtx) -> bool {
        match c {
            Cond::IsAlone => self.flags.is_alone,
            Cond::IsFront => self.flags.is_front,
            Cond::NotFront => !self.flags.is_front,
            Cond::Slot(n) => self.flags.slot == n,
            Cond::Asleep => ctx.asleep,
            Cond::NotAsleep => !ctx.asleep,
            Cond::OffBalance => self.vars.off_balance,
            Cond::Balanced => !self.vars.off_balance,
            Cond::Slumbering => ctx.slumbering,
            Cond::Awake => !ctx.slumbering,
            Cond::CanLay => ctx.living_allies <= 3,
            Cond::CannotLay => ctx.living_allies > 3,
            Cond::CursesLeft => self.vars.curses_given < 3,
            Cond::CursesDone => self.vars.curses_given >= 3,
            Cond::HasAllies => ctx.allies_alive > 0,
            Cond::NoAllies => ctx.allies_alive == 0,
            Cond::CanFabricate => ctx.allies_alive + 1 < 4,
            Cond::CannotFabricate => ctx.allies_alive + 1 >= 4,
            Cond::BeetleCharge => !self.vars.beetle_charged && ctx.below_half,
            Cond::NoBeetleCharge => self.vars.beetle_charged || !ctx.below_half,
            Cond::AmalgamAlive => !self.vars.amalgam_died,
            Cond::AmalgamDead => self.vars.amalgam_died,
            Cond::SecondForm => self.vars.respawns < 2,
            Cond::ThirdForm => self.vars.respawns >= 2,
        }
    }

    /// `RandomBranchState.GetStateWeight`.
    fn branch_weight(&self, b: &Branch, ctx: RollCtx) -> f32 {
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
        w * b.weight.value(ctx)
    }

    /// `MonsterModel.PerformMove`: the effects of `next_move`, then bookkeeping.
    pub fn perform(&mut self, me: CreatureRef, asc: Ascension) -> Vec<Effect> {
        let idx = self.next_move.expect("perform before roll");
        let name = self.states[idx].name();
        self.performed_first = true;
        self.performed_current = true;
        match name {
            "STUNNED" => stunned(self.id, me, &mut self.vars),
            "REVIVE_MOVE" => vec![Effect::Revive { target: me }],
            _ => moves(self.id, name, me, asc, &mut self.vars),
        }
    }
}

/// What a monster does on the turn a stun took: the `stunMove` its
/// `CreatureCmd.Stun` call passed.
fn stunned(id: MonsterId, me: CreatureRef, vars: &mut Vars) -> Vec<Effect> {
    match id {
        // BowlbugRock.DizzyMove.
        MonsterId::BowlbugRock => {
            vars.off_balance = false;
            vec![]
        }
        // SlumberingBeetle.WakeUpMove sheds the Plating.
        MonsterId::SlumberingBeetle => vec![Effect::RemovePower { target: me, id: PowerId::Plating }],
        _ => vec![],
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

/// `PowerCmd.Apply<T>(Creature, amount, null, null)`: no applier.
fn buff_unsourced(me: CreatureRef, id: PowerId, amount: i32) -> Effect {
    Effect::ApplyPower { target: me, id, amount, applier: None }
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
/// `vars` is the monster's own counter block, which a few moves read and bump.
fn moves(id: MonsterId, name: &str, me: CreatureRef, asc: Ascension, vars: &mut Vars) -> Vec<Effect> {
    use AscensionLevel::{DeadlyEnemies as D, ToughEnemies as T};
    use MonsterId::*;
    let d = |a: i32, b: i32| asc.pick(D, b, a);
    let t = |a: i32, b: i32| asc.pick(T, b, a);
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

        // Spiken stacks Thorns; Spike Spit spends the same 2 back.
        (Toadpole, "SPIKEN_MOVE") => vec![buff(me, PowerId::Thorns, 2)],
        (Toadpole, "SPIKE_SPIT_MOVE") => vec![buff(me, PowerId::Thorns, -2), attack(me, d(3, 4), 3)],
        (Toadpole, "WHIRL_MOVE") => vec![attack(me, d(7, 8), 1)],

        (CorpseSlug, "WHIP_SLAP_MOVE") => vec![attack(me, 3, 2)],
        (CorpseSlug, "GLOMP_MOVE") => vec![attack(me, d(8, 9), 1)],
        (CorpseSlug, "GOOP_MOVE") => vec![debuff(me, PowerId::Frail, 2)],

        (DampCultist, "INCANTATION_MOVE") => vec![buff(me, PowerId::Ritual, d(5, 6))],
        (DampCultist, "DARK_STRIKE_MOVE") => vec![attack(me, d(1, 3), 1)],
        (CalcifiedCultist, "INCANTATION_MOVE") => vec![buff(me, PowerId::Ritual, 2)],
        (CalcifiedCultist, "DARK_STRIKE_MOVE") => vec![attack(me, d(9, 11), 1)],

        (FossilStalker, "TACKLE_MOVE") => vec![attack(me, d(9, 11), 1), debuff(me, PowerId::Frail, 1)],
        (FossilStalker, "LATCH_MOVE") => vec![attack(me, d(12, 14), 1)],
        (FossilStalker, "LASH_MOVE") => vec![attack(me, d(3, 4), 2)],

        // The merc's damage scales on Tough, not Deadly. Every one of its
        // moves ends with a Thievery steal.
        (GremlinMerc, "GIMME_MOVE") => vec![attack(me, t(7, 8), 2), Effect::Steal { thief: me }],
        (GremlinMerc, "DOUBLE_SMASH_MOVE") => {
            vec![attack(me, t(6, 7), 2), Effect::Steal { thief: me }, debuff(me, PowerId::Weak, 2)]
        }
        (GremlinMerc, "HEHE_MOVE") => {
            vec![attack(me, t(8, 9), 1), Effect::Steal { thief: me }, buff(me, PowerId::Strength, 2)]
        }
        (FatGremlin, "SPAWNED_MOVE") => vec![],
        (FatGremlin, "FLEE_MOVE") => vec![Effect::Escape { target: me }],
        (SneakyGremlin, "SPAWNED_MOVE") => vec![],
        (SneakyGremlin, "TACKLE_MOVE") => vec![attack(me, d(9, 10), 1)],

        (GasBomb, "EXPLODE_MOVE") => vec![attack(me, d(8, 9), 1), Effect::Kill { target: me }],

        (HauntedShip, "SWIPE_MOVE") => vec![attack(me, d(13, 14), 1)],
        (HauntedShip, "STOMP_MOVE") => vec![attack(me, d(4, 5), 3)],
        (HauntedShip, "HAUNT_MOVE") => {
            std::iter::once(debuff(me, PowerId::Weak, 3)).chain(statuses(CardId::Dazed, 5)).collect()
        }

        (LivingFog, "ADVANCED_GAS_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Smoggy, 1)],
        // BloatAmount is 1, and the spawn is dropped when no bomb slot is free.
        (LivingFog, "BLOAT_MOVE") => vec![
            Effect::SpawnMonster { id: GasBomb, flags: Flags::default() },
            attack(me, d(5, 6), 1),
        ],
        (LivingFog, "SUPER_GAS_BLAST_MOVE") => vec![attack(me, d(8, 9), 1)],

        (PhantasmalGardener, "BITE_MOVE") => vec![attack(me, 5, 1)],
        (PhantasmalGardener, "LASH_MOVE") => vec![attack(me, 7, 1)],
        (PhantasmalGardener, "FLAIL_MOVE") => vec![attack(me, 1, 3)],
        (PhantasmalGardener, "ENLARGE_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],

        (PunchConstruct, "READY_MOVE") => vec![block(me, 10)],
        (PunchConstruct, "STRONG_PUNCH_MOVE") => vec![attack(me, d(14, 16), 1)],
        (PunchConstruct, "FAST_PUNCH_MOVE") => vec![attack(me, d(5, 6), 2), debuff(me, PowerId::Frail, 1)],

        (Seapunk, "SEA_KICK_MOVE") => vec![attack(me, d(11, 13), 1)],
        (Seapunk, "SPINNING_KICK_MOVE") => vec![attack(me, 2, 4)],
        (Seapunk, "BUBBLE_BURP_MOVE") => vec![block(me, t(7, 8)), buff(me, PowerId::Strength, d(1, 2))],

        (SewerClam, "PRESSURIZE_MOVE") => vec![buff(me, PowerId::Strength, 4)],
        (SewerClam, "JET_MOVE") => vec![attack(me, d(10, 11), 1)],

        (SkulkingColony, "ZOOM_MOVE" | "ZOOM_MOVE_2") => vec![attack(me, d(14, 16), 1)],
        (SkulkingColony, "INERTIA_MOVE") => vec![attack(me, d(9, 11), 1), buff(me, PowerId::Strength, d(2, 4))],
        (SkulkingColony, "PIERCING_STABS_MOVE") => vec![attack(me, d(7, 8), 2)],

        (SludgeSpinner, "OIL_SPRAY_MOVE") => vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Weak, 1)],
        (SludgeSpinner, "SLAM_MOVE") => vec![attack(me, d(11, 12), 1)],
        (SludgeSpinner, "RAGE_MOVE") => vec![attack(me, d(6, 7), 1), buff(me, PowerId::Strength, 3)],

        // Every move but the call brings the summon one turn closer.
        (TwoTailedRat, "SCRATCH_MOVE") => {
            vars.turns_until_summonable -= 1;
            vec![attack(me, d(8, 9), 1)]
        }
        (TwoTailedRat, "DISEASE_BITE_MOVE") => {
            vars.turns_until_summonable -= 1;
            vec![attack(me, d(6, 7), 1)]
        }
        (TwoTailedRat, "SCREECH_MOVE") => {
            vars.turns_until_summonable -= 1;
            vec![debuff(me, PowerId::Frail, 1)]
        }
        (TwoTailedRat, "CALL_FOR_BACKUP_MOVE") => vec![Effect::SpawnMonster { id: TwoTailedRat, flags: Flags::default() }],

        (TerrorEel, "CRASH_MOVE") => vec![attack(me, d(16, 18), 1)],
        (TerrorEel, "THRASH_MOVE") => vec![attack(me, d(3, 4), 3), buff(me, PowerId::Vigor, 6)],
        (TerrorEel, "STUN_MOVE") => vec![],
        (TerrorEel, "TERROR_MOVE") => vec![debuff(me, PowerId::Vulnerable, 99)],

        (LagavulinMatriarch, "SLEEP_MOVE") => vec![],
        (LagavulinMatriarch, "SLASH_MOVE") => vec![attack(me, d(19, 21), 1)],
        (LagavulinMatriarch, "SLASH2_MOVE") => vec![attack(me, d(12, 14), 1), block(me, t(12, 14))],
        (LagavulinMatriarch, "DISEMBOWEL_MOVE") => vec![attack(me, d(9, 10), 2)],
        (LagavulinMatriarch, "SOUL_SIPHON_MOVE") => vec![
            debuff(me, PowerId::Strength, -2),
            debuff(me, PowerId::Dexterity, -2),
            buff(me, PowerId::Strength, 2),
        ],

        // One Beckon shuffled into the draw pile, one into the discard.
        (SoulFysh, "BECKON_MOVE") => vec![
            Effect::GenerateCard { id: CardId::Beckon, upgraded: false, to: Pile::DrawRandom, free_this_turn: false },
            Effect::GenerateCard { id: CardId::Beckon, upgraded: false, to: Pile::Discard, free_this_turn: false },
        ],
        (SoulFysh, "DE_GAS_MOVE") => vec![attack(me, d(16, 17), 1)],
        (SoulFysh, "GAZE_MOVE") => std::iter::once(attack(me, d(7, 8), 1)).chain(statuses(CardId::Beckon, 1)).collect(),
        (SoulFysh, "FADE_MOVE") => vec![buff(me, PowerId::Intangible, 2)],
        (SoulFysh, "SCREAM_MOVE") => vec![attack(me, d(13, 15), 1), debuff(me, PowerId::Vulnerable, 3)],

        (WaterfallGiant, "PRESSURIZE_MOVE") => vec![buff(me, PowerId::SteamEruption, d(15, 20))],
        (WaterfallGiant, "STOMP_MOVE") => vec![
            attack(me, d(15, 16), 1),
            debuff(me, PowerId::Weak, 1),
            buff(me, PowerId::SteamEruption, 3),
        ],
        (WaterfallGiant, "RAM_MOVE") => vec![attack(me, d(10, 11), 1), buff(me, PowerId::SteamEruption, 3)],
        (WaterfallGiant, "SIPHON_MOVE") => vec![
            Effect::Heal { target: me, amount: t(10, 15) as f64 },
            buff(me, PowerId::SteamEruption, 3),
        ],
        (WaterfallGiant, "PRESSURE_GUN_MOVE") => {
            let shot = vars.pressure_gun_damage;
            vars.pressure_gun_damage += 5;
            vec![attack(me, shot, 1), buff(me, PowerId::SteamEruption, 3)]
        }
        (WaterfallGiant, "PRESSURE_UP_MOVE") => vec![attack(me, d(13, 14), 1), buff(me, PowerId::SteamEruption, 3)],
        // Banks the pressure into the blast and drops the power.
        (WaterfallGiant, "ABOUT_TO_BLOW_MOVE") => vec![Effect::ArmSteamEruption { target: me }],
        (WaterfallGiant, "EXPLODE_MOVE") => {
            vec![attack(me, vars.steam_eruption_damage, 1), Effect::Kill { target: me }]
        }

        // Act 2 (Hive).
        (BowlbugEgg, "BITE_MOVE") => vec![attack(me, d(7, 8), 1), block(me, d(7, 8))],
        (BowlbugNectar, "THRASH_MOVE" | "THRASH2_MOVE") => vec![attack(me, 3, 1)],
        (BowlbugNectar, "BUFF_MOVE") => vec![buff(me, PowerId::Strength, d(15, 16))],
        // Imbalanced may leave it off balance once the hit has landed.
        (BowlbugRock, "HEADBUTT_MOVE") => vec![attack(me, d(15, 16), 1), Effect::MonsterStep { me, step: STEP_STAGGER }],
        (BowlbugRock, "DIZZY_MOVE") => {
            vars.off_balance = false;
            vec![]
        }
        (BowlbugSilk, "THRASH_MOVE") => vec![attack(me, d(4, 5), 2)],
        (BowlbugSilk, "TOXIC_SPIT_MOVE") => vec![debuff(me, PowerId::Weak, 1)],

        (Chomper, "CLAMP_MOVE") => vec![attack(me, d(8, 9), 2)],
        (Chomper, "SCREECH_MOVE") => statuses(CardId::Dazed, 3).collect(),

        (Crusher, "THRASH_MOVE") => vec![attack(me, d(12, 14), 1)],
        (Crusher, "ENLARGING_STRIKE_MOVE") => vec![attack(me, 4, 1)],
        (Crusher, "BUG_STING_MOVE") => {
            vec![attack(me, d(6, 7), 2), debuff(me, PowerId::Weak, 2), debuff(me, PowerId::Frail, 2)]
        }
        (Crusher, "ADAPT_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],
        (Crusher, "GUARDED_STRIKE_MOVE") => vec![attack(me, d(12, 14), 1), block(me, 18)],
        (Rocket, "TARGETING_RETICLE_MOVE") => vec![attack(me, d(3, 4), 1)],
        (Rocket, "PRECISION_BEAM_MOVE") => vec![attack(me, d(18, 20), 1)],
        (Rocket, "CHARGE_UP_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],
        (Rocket, "LASER_MOVE") => vec![attack(me, d(31, 35), 1)],
        (Rocket, "RECHARGE_MOVE") => vec![],

        (DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack, "WRITHE_MOVE") => {
            vec![attack(me, d(5, 6), 2)]
        }
        (DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack, "BULK_MOVE") => {
            vec![attack(me, d(6, 7), 1), buff(me, PowerId::Strength, 2)]
        }
        (DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack, "CONSTRICT_MOVE") => {
            vec![attack(me, d(8, 9), 1), debuff(me, PowerId::Weak, 1)]
        }
        (DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack, "DEAD_MOVE") => vec![],
        // ReattachPower.DoReattach: back with the power's amount as HP.
        (DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack, "REATTACH_MOVE") => {
            vec![Effect::Reattach { target: me, hp: 25 }]
        }

        (Entomancer, "PHEROMONE_SPIT_MOVE") => vec![Effect::MonsterStep { me, step: STEP_PHEROMONE }],
        (Entomancer, "BEES_MOVE") => vec![attack(me, 3, d(7, 8) as u32)],
        (Entomancer, "SPEAR_MOVE") => vec![attack(me, d(18, 20), 1)],

        (Exoskeleton, "SKITTER_MOVE") => vec![attack(me, 1, d(3, 4) as u32)],
        (Exoskeleton, "MANDIBLES_MOVE") => vec![attack(me, d(8, 9), 1)],
        (Exoskeleton, "ENRAGE_MOVE") => vec![buff(me, PowerId::Strength, 2)],

        (HunterKiller, "TENDERIZING_GOOP_MOVE") => vec![debuff(me, PowerId::Tender, 1)],
        (HunterKiller, "BITE_MOVE") => vec![attack(me, d(17, 19), 1)],
        (HunterKiller, "PUNCTURE_MOVE") => vec![attack(me, d(7, 8), 3)],

        (InfestedPrism, "JAB_MOVE") => vec![attack(me, d(15, 17), 1)],
        (InfestedPrism, "RADIATE_MOVE") => vec![attack(me, d(11, 13), 1), block(me, d(11, 13))],
        (InfestedPrism, "WHIRLWIND_MOVE") => vec![attack(me, d(5, 6), 3)],
        (InfestedPrism, "PULSATE_MOVE") => {
            vec![attack(me, d(8, 10), 1), block(me, t(20, 22)), buff(me, PowerId::VitalSpark, d(2, 3))]
        }

        // Three sets, each Disintegration against a harsher alternative.
        (KnowledgeDemon, "CURSE_OF_KNOWLEDGE_MOVE") => {
            let n = vars.curses_given.clamp(0, 2) as usize;
            vars.curses_given += 1;
            let other = [CardId::MindRot, CardId::Sloth, CardId::WasteAway][n];
            vec![Effect::OfferCurse { cards: [CardId::Disintegration, other], disintegration: [6, 7, 8][n] }]
        }
        (KnowledgeDemon, "SLAP_MOVE") => vec![attack(me, d(17, 18), 1)],
        (KnowledgeDemon, "KNOWLEDGE_OVERWHELMING_MOVE") => vec![attack(me, d(8, 9), 3)],
        (KnowledgeDemon, "PONDER_MOVE") => vec![
            attack(me, d(11, 13), 1),
            Effect::Heal { target: me, amount: 30.0 },
            buff(me, PowerId::Strength, d(2, 3)),
        ],

        (LouseProgenitor, "WEB_CANNON_MOVE") => vec![attack(me, d(9, 10), 1), debuff(me, PowerId::Frail, 2)],
        (LouseProgenitor, "CURL_AND_GROW_MOVE") => vec![block(me, t(14, 18)), buff(me, PowerId::Strength, 5)],
        (LouseProgenitor, "POUNCE_MOVE") => vec![attack(me, d(14, 16), 1)],

        // Two Toxic straight into your hand.
        (Myte, "TOXIC_MOVE") => (0..2)
            .map(|_| Effect::GenerateCard { id: CardId::Toxic, upgraded: false, to: Pile::Hand, free_this_turn: false })
            .collect(),
        (Myte, "BITE_MOVE") => vec![attack(me, d(13, 15), 1)],
        (Myte, "SUCK_MOVE") => vec![attack(me, d(4, 6), 1), buff(me, PowerId::Strength, d(2, 3))],

        // Three eggs, each into the last free egg slot; a crowded board gets fewer.
        (Ovicopter, "LAY_EGGS_MOVE") => {
            (0..3).map(|_| Effect::SpawnMonster { id: ToughEgg, flags: Flags::default() }).collect()
        }
        (Ovicopter, "SMASH_MOVE") => vec![attack(me, d(16, 17), 1)],
        (Ovicopter, "TENDERIZER_MOVE") => vec![attack(me, d(7, 8), 1), debuff(me, PowerId::Vulnerable, 2)],
        (Ovicopter, "NUTRITIONAL_PASTE_MOVE") => vec![buff(me, PowerId::Strength, d(3, 4))],
        (ToughEgg, "HATCH_MOVE") => vec![Effect::Hatch { target: me }],
        (ToughEgg, "NIBBLE_MOVE") => vec![attack(me, d(4, 5), 1)],

        (SlumberingBeetle, "SNORE_MOVE") => vec![],
        (SlumberingBeetle, "ROLL_OUT_MOVE") => vec![attack(me, d(16, 18), 1), buff(me, PowerId::Strength, 2)],

        (SpinyToad, "PROTRUDING_SPIKES_MOVE") => vec![buff(me, PowerId::Thorns, 5)],
        (SpinyToad, "SPIKE_EXPLOSION_MOVE") => vec![attack(me, d(23, 25), 1), buff(me, PowerId::Thorns, -5)],
        (SpinyToad, "TONGUE_LASH_MOVE") => vec![attack(me, d(17, 19), 1)],

        // Liquify: the Sandpit, then three Frantic Escapes shuffled into the
        // draw pile and three into the discard.
        (TheInsatiable, "LIQUIFY_GROUND_MOVE") => std::iter::once(buff(me, PowerId::Sandpit, 4))
            .chain((0..3).map(|_| Effect::GenerateCard {
                id: CardId::FranticEscape,
                upgraded: false,
                to: Pile::DrawRandom,
                free_this_turn: false,
            }))
            .chain(statuses(CardId::FranticEscape, 3))
            .collect(),
        (TheInsatiable, "THRASH_MOVE" | "THRASH_MOVE_2") => vec![attack(me, d(8, 9), 2)],
        (TheInsatiable, "LUNGING_BITE_MOVE") => vec![attack(me, d(28, 31), 1)],
        (TheInsatiable, "SALIVATE_MOVE") => vec![buff(me, PowerId::Strength, d(2, 3))],

        // Slots are ["illusion", "obscura"]: the Parafright stands in front.
        (TheObscura, "ILLUSION_MOVE") => {
            vec![Effect::SpawnMonster { id: Parafright, flags: Flags { slot: 1, ..Default::default() } }]
        }
        (TheObscura, "PIERCING_GAZE_MOVE") => vec![attack(me, d(10, 11), 1)],
        (TheObscura, "SAIL_MOVE") => vec![Effect::MonsterStep { me, step: STEP_SAIL }],
        (TheObscura, "HARDENING_STRIKE_MOVE") => vec![attack(me, d(6, 7), 1), block(me, d(6, 7))],
        (Parafright, "SLAM_MOVE") => vec![attack(me, d(16, 17), 1)],

        (ThievingHopper, "THIEVERY_MOVE") => vec![Effect::MonsterStep { me, step: STEP_STEAL }, attack(me, d(17, 19), 1)],
        (ThievingHopper, "FLUTTER_MOVE") => vec![buff(me, PowerId::Flutter, 5)],
        (ThievingHopper, "HAT_TRICK_MOVE") => vec![attack(me, d(21, 23), 1)],
        (ThievingHopper, "NAB_MOVE") => vec![attack(me, d(14, 16), 1)],
        (ThievingHopper, "ESCAPE_MOVE") => vec![Effect::Escape { target: me }],

        (Tunneler, "BITE_MOVE") => vec![attack(me, d(13, 15), 1)],
        (Tunneler, "BURROW_MOVE") => vec![buff(me, PowerId::Burrowed, 1), block(me, t(32, 37))],
        (Tunneler, "BELOW_MOVE") => vec![attack(me, d(23, 26), 1)],
        (Tunneler, "DIZZY_MOVE") => vec![],
        // Act 3 (Glory).
        // Boot Up is where a respawned Axebot starts: Strength for every
        // stock it has spent.
        (Axebot, "BOOT_UP_MOVE") => vec![block(me, d(10, 15)), buff(me, PowerId::Strength, d(3, 4) * (2 - vars.stock))],
        (Axebot, "ONE_TWO_MOVE") => vec![attack(me, d(9, 10), 2)],
        (Axebot, "HAMMER_UPPERCUT_MOVE") => {
            vec![attack(me, d(12, 14), 1), debuff(me, PowerId::Weak, 2), debuff(me, PowerId::Frail, 2)]
        }

        (DevotedSculptor, "FORBIDDEN_INCANTATION_MOVE") => vec![buff_unsourced(me, PowerId::Ritual, 9)],
        (DevotedSculptor, "SAVAGE_MOVE") => vec![attack(me, d(12, 15), 1)],

        (FrogKnight, "FOR_THE_QUEEN") => vec![buff(me, PowerId::Strength, 5)],
        (FrogKnight, "STRIKE_DOWN_EVIL") => vec![attack(me, d(21, 23), 1)],
        (FrogKnight, "TONGUE_LASH") => vec![attack(me, d(13, 14), 1), debuff(me, PowerId::Frail, 2)],
        (FrogKnight, "BEETLE_CHARGE") => {
            vars.beetle_charged = true;
            vec![attack(me, d(35, 40), 1)]
        }

        (GlobeHead, "THUNDER_STRIKE") => vec![attack(me, d(6, 7), 3)],
        (GlobeHead, "SHOCKING_SLAP") => vec![attack(me, d(13, 14), 1), debuff(me, PowerId::Frail, 2)],
        (GlobeHead, "GALVANIC_BURST") => vec![attack(me, d(16, 17), 1), buff(me, PowerId::Strength, 2)],

        (OwlMagistrate, "MAGISTRATE_SCRUTINY") => vec![attack(me, d(16, 17), 1)],
        (OwlMagistrate, "PECK_ASSAULT") => vec![attack(me, 4, 6)],
        (OwlMagistrate, "JUDICIAL_FLIGHT") => vec![buff(me, PowerId::Soar, 1)],
        (OwlMagistrate, "VERDICT") => vec![
            attack(me, d(33, 36), 1),
            debuff(me, PowerId::Vulnerable, 4),
            Effect::RemovePower { target: me, id: PowerId::Soar },
        ],

        (ScrollOfBiting, "CHOMP") => vec![attack(me, d(14, 16), 1)],
        (ScrollOfBiting, "CHEW") => vec![attack(me, d(5, 6), 2)],
        (ScrollOfBiting, "MORE_TEETH") => vec![buff(me, PowerId::Strength, 2)],

        (SlimedBerserker, "VOMIT_ICHOR_MOVE") => statuses(CardId::Slimed, 10).collect(),
        // The Weak has no applier.
        (SlimedBerserker, "LEECHING_HUG_MOVE") => vec![
            Effect::ApplyPower { target: CreatureRef::Player, id: PowerId::Weak, amount: 3, applier: None },
            buff(me, PowerId::Strength, 3),
        ],
        (SlimedBerserker, "SMOTHER_MOVE") => vec![attack(me, d(30, 33), 1)],
        (SlimedBerserker, "FURIOUS_PUMMELING_MOVE") => vec![attack(me, d(4, 5), 4)],

        (LivingShield, "SHIELD_SLAM_MOVE") => vec![attack(me, 6, 1)],
        (LivingShield, "SMASH_MOVE") => vec![attack(me, d(16, 18), 1), buff(me, PowerId::Strength, 3)],
        (TurretOperator, "UNLOAD_MOVE" | "UNLOAD_MOVE_2") => vec![attack(me, d(3, 4), 5)],
        (TurretOperator, "RELOAD_MOVE") => vec![buff(me, PowerId::Strength, 1)],

        // What they steal comes back when they die (PossessStrength/SpeedPower).
        (TheLost, "DEBILITATING_SMOG") => vec![debuff(me, PowerId::Strength, -2), buff(me, PowerId::Strength, 2)],
        (TheLost, "EYE_LASERS") => vec![attack(me, d(4, 5), 2)],
        (TheForgotten, "MIASMA") => {
            vec![debuff(me, PowerId::Dexterity, -2), block(me, 8), buff(me, PowerId::Dexterity, 2)]
        }
        (TheForgotten, "DREAD") => vec![attack(me, d(13, 15) + vars.own_dex, 1)],

        (MechaKnight, "CHARGE_MOVE") => vec![attack(me, d(25, 30), 1)],
        (MechaKnight, "FLAMETHROWER_MOVE") => (0..4)
            .map(|_| Effect::GenerateCard { id: CardId::Burn, upgraded: false, to: Pile::Hand, free_this_turn: false })
            .collect(),
        (MechaKnight, "WINDUP_MOVE") => vec![block(me, 15), buff(me, PowerId::Strength, 5)],
        (MechaKnight, "HEAVY_CLEAVE_MOVE") => vec![attack(me, d(35, 40), 1)],

        (FlailKnight, "WAR_CHANT") => vec![buff(me, PowerId::Strength, 3)],
        (FlailKnight, "FLAIL_MOVE") => vec![attack(me, d(9, 10), 2)],
        (FlailKnight, "RAM_MOVE") => vec![attack(me, d(15, 17), 1)],
        (SpectralKnight, "HEX") => vec![debuff(me, PowerId::Hex, 2)],
        (SpectralKnight, "SOUL_SLASH") => vec![attack(me, d(15, 17), 1)],
        (SpectralKnight, "SOUL_FLAME") => vec![attack(me, d(3, 4), 3)],
        (MagiKnight, "POWER_SHIELD_MOVE") => vec![attack(me, d(6, 7), 1), block(me, t(5, 9))],
        (MagiKnight, "DAMPEN_MOVE") => vec![debuff(me, PowerId::Dampen, 1)],
        (MagiKnight, "PREP_MOVE") => vec![block(me, t(5, 9))],
        (MagiKnight, "MAGIC_BOMB") => vec![attack(me, d(35, 40), 1)],
        (MagiKnight, "RAM_MOVE") => vec![attack(me, d(10, 11), 1)],

        (SoulNexus, "SOUL_BURN_MOVE") => vec![attack(me, d(29, 31), 1)],
        (SoulNexus, "MAELSTROM_MOVE") => vec![attack(me, d(6, 7), 4)],
        (SoulNexus, "DRAIN_LIFE_MOVE") => {
            vec![attack(me, d(18, 19), 1), debuff(me, PowerId::Vulnerable, 2), debuff(me, PowerId::Weak, 2)]
        }

        (Fabricator, "FABRICATE_MOVE") => vec![
            Effect::FabricateBot { fabricator: me, aggro: false },
            Effect::FabricateBot { fabricator: me, aggro: true },
        ],
        (Fabricator, "FABRICATING_STRIKE_MOVE") => {
            vec![attack(me, d(18, 21), 1), Effect::FabricateBot { fabricator: me, aggro: true }]
        }
        (Fabricator, "DISINTEGRATE_MOVE") => vec![attack(me, d(11, 13), 1)],
        (Zapbot, "ZAP") => vec![attack(me, d(14, 15), 1)],
        (Stabbot, "STAB_MOVE") => vec![attack(me, d(11, 12), 1), debuff(me, PowerId::Frail, 1)],
        (Guardbot, "GUARD_MOVE") => vec![Effect::BlockMonsters { id: Fabricator, amount: 15 }],
        (Noisebot, "NOISE_MOVE") => vec![
            Effect::GenerateCard { id: CardId::Dazed, upgraded: false, to: Pile::Discard, free_this_turn: false },
            Effect::GenerateCard { id: CardId::Dazed, upgraded: false, to: Pile::DrawRandom, free_this_turn: false },
        ],

        (Queen, "PUPPET_STRINGS_MOVE") => vec![debuff(me, PowerId::ChainsOfBinding, 3)],
        (Queen, "YOU_ARE_MINE_MOVE") => vec![
            debuff(me, PowerId::Frail, 99),
            debuff(me, PowerId::Weak, 99),
            debuff(me, PowerId::Vulnerable, 99),
        ],
        (Queen, "BURN_BRIGHT_FOR_ME_MOVE") => {
            vec![Effect::ApplyPowerAllies { source: me, id: PowerId::Strength, amount: 1 }, block(me, 20)]
        }
        (Queen, "OFF_WITH_YOUR_HEAD_MOVE") => vec![attack(me, d(3, 4), 5)],
        (Queen, "EXECUTION_MOVE") => vec![attack(me, d(15, 18), 1)],
        (Queen, "ENRAGE_MOVE") => vec![buff(me, PowerId::Strength, 2)],
        (TorchHeadAmalgam, "TACKLE_MOVE" | "TACKLE_2_MOVE") => vec![attack(me, d(18, 19), 1)],
        (TorchHeadAmalgam, "BEAM_MOVE") => vec![attack(me, 8, 3)],
        (TorchHeadAmalgam, "TACKLE_3_MOVE" | "TACKLE_4_MOVE") => vec![attack(me, d(14, 15), 1)],

        // Back from the dead in the next form: second with Painful Stabs,
        // third with Nemesis and nothing left to revive it.
        (TestSubject, "RESPAWN_MOVE") => {
            vars.respawns += 1;
            if vars.respawns == 1 {
                vec![Effect::ReviveAt { target: me, max_hp: t(200, 212) }, buff(me, PowerId::PainfulStabs, 1)]
            } else {
                vec![
                    Effect::ReviveAt { target: me, max_hp: t(300, 313) },
                    buff(me, PowerId::Nemesis, 1),
                    Effect::RemovePower { target: me, id: PowerId::Adaptable },
                    Effect::RemovePower { target: me, id: PowerId::PainfulStabs },
                ]
            }
        }
        (TestSubject, "BITE_MOVE") => vec![attack(me, d(20, 22), 1)],
        (TestSubject, "SKULL_BASH_MOVE") => vec![attack(me, d(14, 16), 1), debuff(me, PowerId::Vulnerable, 1)],
        (TestSubject, "MULTI_CLAW_MOVE") => {
            let hits = 3 + vars.extra_claws as u32;
            vars.extra_claws += 1;
            vec![attack(me, d(10, 11), hits)]
        }
        (TestSubject, "PHASE3_LACERATE_MOVE") => vec![attack(me, d(10, 11), 3)],
        (TestSubject, "BIG_POUNCE") => vec![attack(me, 45, 1)],
        (TestSubject, "BURNING_GROWL_MOVE") => {
            statuses(CardId::Burn, d(3, 5) as u32).chain([buff(me, PowerId::Strength, d(2, 3))]).collect()
        }

        (Aeonglass, "EBB_MOVE") => vec![attack(me, d(26, 32), 1), block(me, 33)],
        (Aeonglass, "EYE_LASERS_MOVE") => vec![attack(me, d(11, 12), 2)],
        // Every Wither already in play hits 3 harder, and the new ones match.
        (Aeonglass, "INCREASING_INTENSITY_MOVE") => {
            vars.wither_upgrades += 1;
            let strength = d(3, 4) + vars.extra_strength;
            vars.extra_strength += 1;
            std::iter::once(Effect::UpgradeWithers)
                .chain(statuses(CardId::Wither, d(1, 2) as u32))
                .chain([buff(me, PowerId::Strength, strength)])
                .collect()
        }

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
    Branch { state, cooldown: 0, repeat, weight: Weight::Fixed(1.0) }
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
    let t = |a: i32, b: i32| asc.pick(AscensionLevel::ToughEnemies, b, a);
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
                Branch { state: swipe_r, cooldown: 0, repeat: Repeat::CannotRepeat, weight: Weight::Fixed(0.4) },
                Branch { state: headbutt, cooldown: 0, repeat: Repeat::CannotRepeat, weight: Weight::Fixed(0.6) },
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
                Branch { state: vuln, cooldown: 3, repeat: Repeat::CannotRepeat, weight: Weight::Fixed(1.0) },
                Branch { state: frail, cooldown: 2, repeat: Repeat::CannotRepeat, weight: Weight::Fixed(1.0) },
                br(smash, Repeat::CannotRepeat),
            ]);
            let initial = g.random(vec![
                Branch { state: frail, cooldown: 2, repeat: Repeat::CannotRepeat, weight: Weight::Fixed(1.0) },
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

        // Front toadpole opens on Spiken, the back one on Whirl.
        Toadpole => {
            let spit = g.mv("SPIKE_SPIT_MOVE", vec![multi(d(3, 4), 3)]);
            let whirl = g.mv("WHIRL_MOVE", vec![atk(d(7, 8))]);
            let spiken = g.mv("SPIKEN_MOVE", vec![Buff]);
            let init = g.cond(vec![(whirl, Cond::NotFront), (spiken, Cond::IsFront)]);
            g.follow(whirl, spiken);
            g.follow(spiken, spit);
            g.follow(spit, whirl);
            g.done(init)
        }
        // A fixed cycle entered at the slug's assigned starting move.
        CorpseSlug => {
            let whip = g.mv("WHIP_SLAP_MOVE", vec![multi(3, 2)]);
            let glomp = g.mv("GLOMP_MOVE", vec![atk(d(8, 9))]);
            let goop = g.mv("GOOP_MOVE", vec![Debuff { strong: false }]);
            g.follow(whip, glomp);
            g.follow(glomp, goop);
            g.follow(goop, whip);
            g.done(match flags.starter_move % 3 {
                0 => whip,
                1 => glomp,
                _ => goop,
            })
        }
        // Both cultists: one Incantation, then Dark Strike forever.
        DampCultist | CalcifiedCultist => {
            let dmg = if id == DampCultist { d(1, 3) } else { d(9, 11) };
            let incant = g.mv("INCANTATION_MOVE", vec![Buff]);
            let strike = g.mv("DARK_STRIKE_MOVE", vec![atk(dmg)]);
            g.follow(incant, strike);
            g.follow(strike, strike);
            g.done(incant)
        }
        FossilStalker => {
            let tackle = g.mv("TACKLE_MOVE", vec![atk(d(9, 11)), Debuff { strong: false }]);
            let latch = g.mv("LATCH_MOVE", vec![atk(d(12, 14))]);
            let lash = g.mv("LASH_MOVE", vec![multi(d(3, 4), 2)]);
            let rand = g.random(vec![
                br(latch, Repeat::Times(2)),
                br(tackle, Repeat::Times(2)),
                br(lash, Repeat::Times(2)),
            ]);
            g.follow(tackle, rand);
            g.follow(latch, rand);
            g.follow(lash, rand);
            g.done(latch)
        }
        GremlinMerc => {
            let gimme = g.mv("GIMME_MOVE", vec![multi(t(7, 8), 2)]);
            let smash = g.mv("DOUBLE_SMASH_MOVE", vec![multi(t(6, 7), 2), Debuff { strong: false }]);
            let hehe = g.mv("HEHE_MOVE", vec![atk(t(8, 9)), Buff]);
            g.follow(gimme, smash);
            g.follow(smash, hehe);
            g.follow(hehe, gimme);
            g.done(gimme)
        }
        // Both spawned gremlins idle the turn they arrive.
        FatGremlin => {
            let spawned = g.mv("SPAWNED_MOVE", vec![Stun]);
            let flee = g.mv("FLEE_MOVE", vec![Escape]);
            g.follow(spawned, flee);
            g.follow(flee, flee);
            g.done(spawned)
        }
        SneakyGremlin => {
            let spawned = g.mv("SPAWNED_MOVE", vec![Stun]);
            let tackle = g.mv("TACKLE_MOVE", vec![atk(d(9, 10))]);
            g.follow(spawned, tackle);
            g.follow(tackle, tackle);
            g.done(spawned)
        }
        GasBomb => {
            let boom = g.mv("EXPLODE_MOVE", vec![DeathBlow { damage: d(8, 9) }]);
            g.follow(boom, boom);
            g.done(boom)
        }
        HauntedShip => {
            let swipe = g.mv("SWIPE_MOVE", vec![atk(d(13, 14))]);
            let stomp = g.mv("STOMP_MOVE", vec![multi(d(4, 5), 3)]);
            let haunt = g.mv("HAUNT_MOVE", vec![Debuff { strong: false }, Status { count: 5 }]);
            g.follow(haunt, swipe);
            g.follow(swipe, stomp);
            g.follow(stomp, swipe);
            g.done(haunt)
        }
        LivingFog => {
            let gas = g.mv("ADVANCED_GAS_MOVE", vec![atk(d(8, 9)), CardDebuff]);
            let bloat = g.mv("BLOAT_MOVE", vec![atk(d(5, 6)), Summon]);
            let blast = g.mv("SUPER_GAS_BLAST_MOVE", vec![atk(d(8, 9))]);
            g.follow(gas, bloat);
            g.follow(bloat, blast);
            g.follow(blast, bloat);
            g.done(gas)
        }
        // One shared cycle; each gardener enters it at its own slot's move.
        PhantasmalGardener => {
            let bite = g.mv("BITE_MOVE", vec![atk(5)]);
            let lash = g.mv("LASH_MOVE", vec![atk(7)]);
            let enlarge = g.mv("ENLARGE_MOVE", vec![Buff]);
            let flail = g.mv("FLAIL_MOVE", vec![multi(1, 3)]);
            let init = g.cond(vec![
                (flail, Cond::Slot(1)),
                (bite, Cond::Slot(2)),
                (lash, Cond::Slot(3)),
                (enlarge, Cond::Slot(4)),
            ]);
            g.follow(bite, lash);
            g.follow(lash, flail);
            g.follow(flail, enlarge);
            g.follow(enlarge, bite);
            g.done(init)
        }
        PunchConstruct => {
            let ready = g.mv("READY_MOVE", vec![Defend]);
            let strong = g.mv("STRONG_PUNCH_MOVE", vec![atk(d(14, 16))]);
            let fast = g.mv("FAST_PUNCH_MOVE", vec![multi(d(5, 6), 2), Debuff { strong: false }]);
            g.follow(ready, fast);
            g.follow(fast, strong);
            g.follow(strong, ready);
            g.done(ready)
        }
        Seapunk => {
            let kick = g.mv("SEA_KICK_MOVE", vec![atk(d(11, 13))]);
            let spin = g.mv("SPINNING_KICK_MOVE", vec![multi(2, 4)]);
            let bubble = g.mv("BUBBLE_BURP_MOVE", vec![Buff, Defend]);
            g.follow(kick, spin);
            g.follow(spin, bubble);
            g.follow(bubble, kick);
            g.done(kick)
        }
        SewerClam => {
            let pressurize = g.mv("PRESSURIZE_MOVE", vec![Buff]);
            let jet = g.mv("JET_MOVE", vec![atk(d(10, 11))]);
            g.follow(pressurize, jet);
            g.follow(jet, pressurize);
            g.done(jet)
        }
        // Two Zooms in a row, then Inertia and Piercing Stabs.
        SkulkingColony => {
            let zoom = g.mv("ZOOM_MOVE", vec![atk(d(14, 16))]);
            let zoom2 = g.mv("ZOOM_MOVE_2", vec![atk(d(14, 16))]);
            let inertia = g.mv("INERTIA_MOVE", vec![atk(d(9, 11)), Buff]);
            let stabs = g.mv("PIERCING_STABS_MOVE", vec![multi(d(7, 8), 2)]);
            g.follow(zoom, zoom2);
            g.follow(zoom2, inertia);
            g.follow(inertia, stabs);
            g.follow(stabs, zoom);
            g.done(zoom)
        }
        SludgeSpinner => {
            let oil = g.mv("OIL_SPRAY_MOVE", vec![atk(d(8, 9)), Debuff { strong: false }]);
            let slam = g.mv("SLAM_MOVE", vec![atk(d(11, 12))]);
            let rage = g.mv("RAGE_MOVE", vec![atk(d(6, 7)), Buff]);
            let rand = g.random(vec![
                br(oil, Repeat::CannotRepeat),
                br(slam, Repeat::CannotRepeat),
                br(rage, Repeat::CannotRepeat),
            ]);
            g.follow(oil, rand);
            g.follow(slam, rand);
            g.follow(rage, rand);
            g.done(oil)
        }
        // While a call is possible it crowds out the other three 9 to 1.
        TwoTailedRat => {
            let scratch = g.mv("SCRATCH_MOVE", vec![atk(d(8, 9))]);
            let bite = g.mv("DISEASE_BITE_MOVE", vec![atk(d(6, 7))]);
            let screech = g.mv("SCREECH_MOVE", vec![Debuff { strong: false }]);
            let call = g.mv("CALL_FOR_BACKUP_MOVE", vec![Summon]);
            let ordinary = Weight::IfCanSummon { yes: 1.0 / 12.0, no: 1.0 };
            let rand = g.random(vec![
                Branch { state: scratch, cooldown: 0, repeat: Repeat::CannotRepeat, weight: ordinary },
                Branch { state: bite, cooldown: 0, repeat: Repeat::CannotRepeat, weight: ordinary },
                Branch { state: screech, cooldown: 3, repeat: Repeat::CannotRepeat, weight: ordinary },
                Branch {
                    state: call,
                    cooldown: 0,
                    repeat: Repeat::Once,
                    weight: Weight::IfCanSummon { yes: 0.75, no: 0.0 },
                },
            ]);
            g.follow(scratch, rand);
            g.follow(bite, rand);
            g.follow(screech, rand);
            g.follow(call, rand);
            // Rats placed by the encounter open on a set move; ones called in
            // later roll from the branch like any other turn.
            g.done(match flags.starter_move % 3 {
                _ if flags.slot == 0 => rand,
                0 => scratch,
                1 => bite,
                _ => screech,
            })
        }
        // Crash and Thrash trade off; Shriek interrupts into Stun then Terror.
        TerrorEel => {
            let crash = g.mv("CRASH_MOVE", vec![atk(d(16, 18))]);
            let thrash = g.mv("THRASH_MOVE", vec![multi(d(3, 4), 3), Buff]);
            let stun = g.mv("STUN_MOVE", vec![Stun]);
            let terror = g.mv("TERROR_MOVE", vec![Debuff { strong: true }]);
            g.follow(crash, thrash);
            g.follow(thrash, crash);
            g.follow(stun, terror);
            g.follow(terror, crash);
            g.done(crash)
        }
        // Sleeps until the nap counter runs out or a hit gets through.
        LagavulinMatriarch => {
            let sleep = g.mv("SLEEP_MOVE", vec![Sleep]);
            let slash = g.mv("SLASH_MOVE", vec![atk(d(19, 21))]);
            let slash2 = g.mv("SLASH2_MOVE", vec![atk(d(12, 14)), Defend]);
            let disembowel = g.mv("DISEMBOWEL_MOVE", vec![multi(d(9, 10), 2)]);
            let siphon = g.mv("SOUL_SIPHON_MOVE", vec![Debuff { strong: true }, Buff]);
            let branch = g.cond(vec![(sleep, Cond::Asleep), (slash, Cond::NotAsleep)]);
            g.follow(sleep, branch);
            g.follow(slash, disembowel);
            g.follow(disembowel, slash2);
            g.follow(slash2, siphon);
            g.follow(siphon, slash);
            g.done(sleep)
        }
        SoulFysh => {
            let beckon = g.mv("BECKON_MOVE", vec![Status { count: 2 }]);
            let degas = g.mv("DE_GAS_MOVE", vec![atk(d(16, 17))]);
            let gaze = g.mv("GAZE_MOVE", vec![atk(d(7, 8)), Status { count: 1 }]);
            let fade = g.mv("FADE_MOVE", vec![Buff]);
            let scream = g.mv("SCREAM_MOVE", vec![atk(d(13, 15)), Debuff { strong: false }]);
            g.follow(beckon, degas);
            g.follow(degas, gaze);
            g.follow(gaze, fade);
            g.follow(fade, scream);
            g.follow(scream, beckon);
            g.done(beckon)
        }
        // Every move but Pressurize also builds pressure. Killing it only
        // sets off the blast, which is what actually ends the fight.
        WaterfallGiant => {
            let pressurize = g.mv("PRESSURIZE_MOVE", vec![Buff]);
            let stomp = g.mv("STOMP_MOVE", vec![atk(d(15, 16)), Debuff { strong: false }, Buff]);
            let ram = g.mv("RAM_MOVE", vec![atk(d(10, 11)), Buff]);
            let siphon = g.mv("SIPHON_MOVE", vec![Heal, Buff]);
            let gun = g.mv("PRESSURE_GUN_MOVE", vec![atk(d(20, 23)), Buff]);
            let up = g.mv("PRESSURE_UP_MOVE", vec![atk(d(13, 14)), Buff]);
            let about = g.mv_once("ABOUT_TO_BLOW_MOVE", vec![Stun]);
            let explode = g.mv("EXPLODE_MOVE", vec![DeathBlow { damage: 0 }]);
            g.follow(pressurize, stomp);
            g.follow(stomp, ram);
            g.follow(ram, siphon);
            g.follow(siphon, gun);
            g.follow(gun, up);
            g.follow(up, stomp);
            g.follow(about, explode);
            g.follow(explode, explode);
            g.done(pressurize)
        }

        // Act 2 (Hive).
        BowlbugEgg => {
            let bite = g.mv("BITE_MOVE", vec![atk(d(7, 8)), Defend]);
            g.follow(bite, bite);
            g.done(bite)
        }
        BowlbugNectar => {
            let thrash = g.mv("THRASH_MOVE", vec![atk(3)]);
            let buff = g.mv("BUFF_MOVE", vec![Buff]);
            let thrash2 = g.mv("THRASH2_MOVE", vec![atk(3)]);
            g.follow(thrash, buff);
            g.follow(buff, thrash2);
            g.follow(thrash2, thrash2);
            g.done(thrash)
        }
        // Headbutt until one is fully blocked; the stagger that follows is a
        // stun, so the Dizzy branch only shows if the stun could not land.
        BowlbugRock => {
            let headbutt = g.mv("HEADBUTT_MOVE", vec![atk(d(15, 16))]);
            let dizzy = g.mv("DIZZY_MOVE", vec![Stun]);
            let post = g.cond(vec![(dizzy, Cond::OffBalance), (headbutt, Cond::Balanced)]);
            g.follow(headbutt, post);
            g.follow(dizzy, headbutt);
            g.done(headbutt)
        }
        BowlbugSilk => {
            let thrash = g.mv("THRASH_MOVE", vec![multi(d(4, 5), 2)]);
            let spit = g.mv("TOXIC_SPIT_MOVE", vec![Debuff { strong: false }]);
            g.follow(thrash, spit);
            g.follow(spit, thrash);
            g.done(spit)
        }
        Chomper => {
            let clamp = g.mv("CLAMP_MOVE", vec![multi(d(8, 9), 2)]);
            let screech = g.mv("SCREECH_MOVE", vec![Status { count: 3 }]);
            g.follow(clamp, screech);
            g.follow(screech, clamp);
            g.done(if flags.scream_first { screech } else { clamp })
        }
        Crusher => {
            let thrash = g.mv("THRASH_MOVE", vec![atk(d(12, 14))]);
            let enlarge = g.mv("ENLARGING_STRIKE_MOVE", vec![atk(4)]);
            let sting = g.mv("BUG_STING_MOVE", vec![multi(d(6, 7), 2), Debuff { strong: false }]);
            let adapt = g.mv("ADAPT_MOVE", vec![Buff]);
            let guarded = g.mv("GUARDED_STRIKE_MOVE", vec![atk(d(12, 14)), Defend]);
            g.follow(thrash, enlarge);
            g.follow(enlarge, sting);
            g.follow(sting, adapt);
            g.follow(adapt, guarded);
            g.follow(guarded, thrash);
            g.done(thrash)
        }
        Rocket => {
            let reticle = g.mv("TARGETING_RETICLE_MOVE", vec![atk(d(3, 4))]);
            let beam = g.mv("PRECISION_BEAM_MOVE", vec![atk(d(18, 20))]);
            let charge = g.mv("CHARGE_UP_MOVE", vec![Buff]);
            let laser = g.mv("LASER_MOVE", vec![atk(d(31, 35))]);
            let recharge = g.mv("RECHARGE_MOVE", vec![Sleep]);
            g.follow(reticle, beam);
            g.follow(beam, charge);
            g.follow(charge, laser);
            g.follow(laser, recharge);
            g.follow(recharge, reticle);
            g.done(reticle)
        }
        // A fixed cycle entered at the segment's assigned move; a segment
        // that dies plays dead, reattaches, then picks moves at random.
        DecimillipedeSegmentFront | DecimillipedeSegmentMiddle | DecimillipedeSegmentBack => {
            let writhe = g.mv("WRITHE_MOVE", vec![multi(d(5, 6), 2)]);
            let bulk = g.mv("BULK_MOVE", vec![atk(d(6, 7)), Buff]);
            let constrict = g.mv("CONSTRICT_MOVE", vec![atk(d(8, 9)), Debuff { strong: false }]);
            let dead = g.mv("DEAD_MOVE", vec![]);
            let reattach = g.mv_once("REATTACH_MOVE", vec![Heal]);
            let rand = g.random(vec![
                br(writhe, Repeat::CannotRepeat),
                br(bulk, Repeat::CannotRepeat),
                br(constrict, Repeat::CannotRepeat),
            ]);
            g.follow(constrict, bulk);
            g.follow(bulk, writhe);
            g.follow(writhe, constrict);
            g.follow(dead, reattach);
            g.follow(reattach, rand);
            g.done(match flags.starter_move % 3 {
                0 => writhe,
                1 => bulk,
                _ => constrict,
            })
        }
        Entomancer => {
            let spit = g.mv("PHEROMONE_SPIT_MOVE", vec![Buff]);
            let bees = g.mv("BEES_MOVE", vec![multi(3, d(7, 8) as u32)]);
            let spear = g.mv("SPEAR_MOVE", vec![atk(d(18, 20))]);
            g.follow(bees, spear);
            g.follow(spear, spit);
            g.follow(spit, bees);
            g.done(bees)
        }
        // Each exoskeleton opens on the move its slot names.
        Exoskeleton => {
            let skitter = g.mv("SKITTER_MOVE", vec![multi(1, d(3, 4) as u32)]);
            let mandibles = g.mv("MANDIBLES_MOVE", vec![atk(d(8, 9))]);
            let enrage = g.mv("ENRAGE_MOVE", vec![Buff]);
            let rand = g.random(vec![br(skitter, Repeat::CannotRepeat), br(mandibles, Repeat::CannotRepeat)]);
            let init = g.cond(vec![
                (skitter, Cond::Slot(1)),
                (mandibles, Cond::Slot(2)),
                (enrage, Cond::Slot(3)),
                (rand, Cond::Slot(4)),
            ]);
            g.follow(skitter, rand);
            g.follow(mandibles, enrage);
            g.follow(enrage, rand);
            g.done(init)
        }
        HunterKiller => {
            let goop = g.mv("TENDERIZING_GOOP_MOVE", vec![Debuff { strong: false }]);
            let bite = g.mv("BITE_MOVE", vec![atk(d(17, 19))]);
            let puncture = g.mv("PUNCTURE_MOVE", vec![multi(d(7, 8), 3)]);
            let rand = g.random(vec![br(bite, Repeat::CannotRepeat), br(puncture, Repeat::Times(2))]);
            g.follow(goop, rand);
            g.follow(bite, rand);
            g.follow(puncture, rand);
            g.done(goop)
        }
        InfestedPrism => {
            let jab = g.mv("JAB_MOVE", vec![atk(d(15, 17))]);
            let radiate = g.mv("RADIATE_MOVE", vec![atk(d(11, 13)), Defend]);
            let whirl = g.mv("WHIRLWIND_MOVE", vec![multi(d(5, 6), 3)]);
            let pulsate = g.mv("PULSATE_MOVE", vec![atk(d(8, 10)), Buff, Defend]);
            g.follow(jab, radiate);
            g.follow(radiate, whirl);
            g.follow(whirl, pulsate);
            g.follow(pulsate, jab);
            g.done(jab)
        }
        // Three rounds of Curse of Knowledge, then the attacks alone.
        KnowledgeDemon => {
            let curse = g.mv("CURSE_OF_KNOWLEDGE_MOVE", vec![Debuff { strong: false }]);
            let slap = g.mv("SLAP_MOVE", vec![atk(d(17, 18))]);
            let overwhelm = g.mv("KNOWLEDGE_OVERWHELMING_MOVE", vec![multi(d(8, 9), 3)]);
            let ponder = g.mv("PONDER_MOVE", vec![atk(d(11, 13)), Heal, Buff]);
            let branch = g.cond(vec![(curse, Cond::CursesLeft), (slap, Cond::CursesDone)]);
            g.follow(curse, slap);
            g.follow(slap, overwhelm);
            g.follow(overwhelm, ponder);
            g.follow(ponder, branch);
            g.done(curse)
        }
        LouseProgenitor => {
            let web = g.mv("WEB_CANNON_MOVE", vec![atk(d(9, 10)), Debuff { strong: false }]);
            let pounce = g.mv("POUNCE_MOVE", vec![atk(d(14, 16))]);
            let curl = g.mv("CURL_AND_GROW_MOVE", vec![Defend, Buff]);
            g.follow(web, curl);
            g.follow(curl, pounce);
            g.follow(pounce, web);
            g.done(web)
        }
        // Slots are ["first", "second"]: the first opens on Toxic, the
        // second on Suck.
        Myte => {
            let toxic = g.mv("TOXIC_MOVE", vec![Status { count: 2 }]);
            let bite = g.mv("BITE_MOVE", vec![atk(d(13, 15))]);
            let suck = g.mv("SUCK_MOVE", vec![atk(d(4, 6)), Buff]);
            let init = g.cond(vec![(toxic, Cond::Slot(1)), (suck, Cond::Slot(2))]);
            g.follow(toxic, bite);
            g.follow(bite, suck);
            g.follow(suck, toxic);
            g.done(init)
        }
        // Lays eggs while its side is small enough, feeds itself otherwise.
        Ovicopter => {
            let paste = g.mv("NUTRITIONAL_PASTE_MOVE", vec![Buff]);
            let lay = g.mv("LAY_EGGS_MOVE", vec![Summon]);
            let smash = g.mv("SMASH_MOVE", vec![atk(d(16, 17))]);
            let tenderizer = g.mv("TENDERIZER_MOVE", vec![atk(d(7, 8)), Debuff { strong: false }]);
            let branch = g.cond(vec![(lay, Cond::CanLay), (paste, Cond::CannotLay)]);
            g.follow(lay, smash);
            g.follow(paste, smash);
            g.follow(smash, tenderizer);
            g.follow(tenderizer, branch);
            g.done(lay)
        }
        ToughEgg => {
            let hatch = g.mv("HATCH_MOVE", vec![Summon]);
            let nibble = g.mv("NIBBLE_MOVE", vec![atk(d(4, 5))]);
            g.follow(hatch, nibble);
            g.follow(nibble, nibble);
            g.done(hatch)
        }
        SlumberingBeetle => {
            let snore = g.mv("SNORE_MOVE", vec![Sleep]);
            let rollout = g.mv("ROLL_OUT_MOVE", vec![atk(d(16, 18)), Buff]);
            let branch = g.cond(vec![(snore, Cond::Slumbering), (rollout, Cond::Awake)]);
            g.follow(snore, branch);
            g.follow(rollout, rollout);
            g.done(snore)
        }
        SpinyToad => {
            let spikes = g.mv("PROTRUDING_SPIKES_MOVE", vec![Buff]);
            let explosion = g.mv("SPIKE_EXPLOSION_MOVE", vec![atk(d(23, 25))]);
            let lash = g.mv("TONGUE_LASH_MOVE", vec![atk(d(17, 19))]);
            g.follow(spikes, explosion);
            g.follow(explosion, lash);
            g.follow(lash, spikes);
            g.done(spikes)
        }
        TheInsatiable => {
            let liquify = g.mv("LIQUIFY_GROUND_MOVE", vec![Buff, Status { count: 6 }]);
            let thrash = g.mv("THRASH_MOVE", vec![multi(d(8, 9), 2)]);
            let thrash2 = g.mv("THRASH_MOVE_2", vec![multi(d(8, 9), 2)]);
            let bite = g.mv("LUNGING_BITE_MOVE", vec![atk(d(28, 31))]);
            let salivate = g.mv("SALIVATE_MOVE", vec![Buff]);
            g.follow(liquify, thrash);
            g.follow(thrash, bite);
            g.follow(bite, salivate);
            g.follow(salivate, thrash2);
            g.follow(thrash2, thrash);
            g.done(liquify)
        }
        TheObscura => {
            let illusion = g.mv("ILLUSION_MOVE", vec![Summon]);
            let gaze = g.mv("PIERCING_GAZE_MOVE", vec![atk(d(10, 11))]);
            let sail = g.mv("SAIL_MOVE", vec![Buff]);
            let hardening = g.mv("HARDENING_STRIKE_MOVE", vec![atk(d(6, 7)), Defend]);
            let rand = g.random(vec![
                br(gaze, Repeat::CannotRepeat),
                br(sail, Repeat::CannotRepeat),
                br(hardening, Repeat::CannotRepeat),
            ]);
            g.follow(illusion, rand);
            g.follow(gaze, rand);
            g.follow(sail, rand);
            g.follow(hardening, rand);
            g.done(illusion)
        }
        Parafright => {
            let slam = g.mv("SLAM_MOVE", vec![atk(d(16, 17))]);
            g.follow(slam, slam);
            g.done(slam)
        }
        // Steals, takes to the air, attacks twice, then flees.
        ThievingHopper => {
            let thievery = g.mv("THIEVERY_MOVE", vec![atk(d(17, 19)), CardDebuff]);
            let nab = g.mv("NAB_MOVE", vec![atk(d(14, 16))]);
            let hat = g.mv("HAT_TRICK_MOVE", vec![atk(d(21, 23))]);
            let flutter = g.mv("FLUTTER_MOVE", vec![Buff]);
            let escape = g.mv("ESCAPE_MOVE", vec![Escape]);
            g.follow(thievery, flutter);
            g.follow(flutter, hat);
            g.follow(hat, nab);
            g.follow(nab, escape);
            g.follow(escape, escape);
            g.done(thievery)
        }
        // Bite, burrow, then Below forever until its block breaks, which
        // stuns it back to Bite.
        Tunneler => {
            let bite = g.mv("BITE_MOVE", vec![atk(d(13, 15))]);
            let burrow = g.mv("BURROW_MOVE", vec![Buff, Defend]);
            let below = g.mv("BELOW_MOVE", vec![atk(d(23, 26))]);
            let dizzy = g.mv("DIZZY_MOVE", vec![Stun]);
            g.follow(bite, burrow);
            g.follow(burrow, below);
            g.follow(below, below);
            g.follow(dizzy, bite);
            g.done(bite)
        }
        // Act 3 (Glory).
        // Uppercut and One-Two trade off; a respawned Axebot boots up first.
        Axebot => {
            let boot = g.mv("BOOT_UP_MOVE", vec![Defend, Buff]);
            let one_two = g.mv("ONE_TWO_MOVE", vec![multi(d(9, 10), 2)]);
            let uppercut = g.mv("HAMMER_UPPERCUT_MOVE", vec![atk(d(12, 14)), Debuff { strong: false }]);
            g.follow(boot, uppercut);
            g.follow(uppercut, one_two);
            g.follow(one_two, uppercut);
            g.done(if flags.stock.is_some() { boot } else { uppercut })
        }
        DevotedSculptor => {
            let incant = g.mv("FORBIDDEN_INCANTATION_MOVE", vec![Buff]);
            let savage = g.mv("SAVAGE_MOVE", vec![atk(d(12, 15))]);
            g.follow(incant, savage);
            g.follow(savage, savage);
            g.done(incant)
        }
        // Once below half, one Beetle Charge takes Tongue Lash's place.
        FrogKnight => {
            let queen = g.mv("FOR_THE_QUEEN", vec![Buff]);
            let strike = g.mv("STRIKE_DOWN_EVIL", vec![atk(d(21, 23))]);
            let lash = g.mv("TONGUE_LASH", vec![atk(d(13, 14)), Debuff { strong: false }]);
            let charge = g.mv("BEETLE_CHARGE", vec![atk(d(35, 40))]);
            let half = g.cond(vec![(lash, Cond::NoBeetleCharge), (charge, Cond::BeetleCharge)]);
            g.follow(queen, half);
            g.follow(strike, queen);
            g.follow(lash, strike);
            g.follow(charge, lash);
            g.done(lash)
        }
        GlobeHead => {
            let thunder = g.mv("THUNDER_STRIKE", vec![multi(d(6, 7), 3)]);
            let slap = g.mv("SHOCKING_SLAP", vec![atk(d(13, 14)), Debuff { strong: false }]);
            let burst = g.mv("GALVANIC_BURST", vec![atk(d(16, 17)), Buff]);
            g.follow(slap, thunder);
            g.follow(thunder, burst);
            g.follow(burst, slap);
            g.done(slap)
        }
        OwlMagistrate => {
            let scrutiny = g.mv("MAGISTRATE_SCRUTINY", vec![atk(d(16, 17))]);
            let peck = g.mv("PECK_ASSAULT", vec![multi(4, 6)]);
            let flight = g.mv("JUDICIAL_FLIGHT", vec![Buff]);
            let verdict = g.mv("VERDICT", vec![atk(d(33, 36)), Debuff { strong: false }]);
            g.follow(scrutiny, peck);
            g.follow(peck, flight);
            g.follow(flight, verdict);
            g.follow(verdict, scrutiny);
            g.done(scrutiny)
        }
        ScrollOfBiting => {
            let chomp = g.mv("CHOMP", vec![atk(d(14, 16))]);
            let chew = g.mv("CHEW", vec![multi(d(5, 6), 2)]);
            let teeth = g.mv("MORE_TEETH", vec![Buff]);
            let rand = g.random(vec![br(chomp, Repeat::CannotRepeat), br(chew, Repeat::Times(2))]);
            g.follow(chomp, teeth);
            g.follow(chew, rand);
            g.follow(teeth, chew);
            g.done(match flags.starter_move % 3 {
                0 => chomp,
                1 => chew,
                _ => teeth,
            })
        }
        SlimedBerserker => {
            let vomit = g.mv("VOMIT_ICHOR_MOVE", vec![Status { count: 10 }]);
            let hug = g.mv("LEECHING_HUG_MOVE", vec![Debuff { strong: false }, Buff]);
            let smother = g.mv("SMOTHER_MOVE", vec![atk(d(30, 33))]);
            let pummel = g.mv("FURIOUS_PUMMELING_MOVE", vec![multi(d(4, 5), 4)]);
            g.follow(vomit, pummel);
            g.follow(pummel, hug);
            g.follow(hug, smother);
            g.follow(smother, vomit);
            g.done(vomit)
        }
        // Shield Slam while the turret stands, Smash once it is alone.
        LivingShield => {
            let slam = g.mv("SHIELD_SLAM_MOVE", vec![atk(6)]);
            let smash = g.mv("SMASH_MOVE", vec![atk(d(16, 18)), Buff]);
            let branch = g.cond(vec![(slam, Cond::HasAllies), (smash, Cond::NoAllies)]);
            g.follow(slam, branch);
            g.follow(smash, smash);
            g.done(slam)
        }
        TurretOperator => {
            let unload = g.mv("UNLOAD_MOVE", vec![multi(d(3, 4), 5)]);
            let unload2 = g.mv("UNLOAD_MOVE_2", vec![multi(d(3, 4), 5)]);
            let reload = g.mv("RELOAD_MOVE", vec![Buff]);
            g.follow(unload, unload2);
            g.follow(unload2, reload);
            g.follow(reload, unload);
            g.done(unload)
        }
        TheLost => {
            let smog = g.mv("DEBILITATING_SMOG", vec![Debuff { strong: false }, Buff]);
            let lasers = g.mv("EYE_LASERS", vec![multi(d(4, 5), 2)]);
            g.follow(smog, lasers);
            g.follow(lasers, smog);
            g.done(smog)
        }
        // Dread's intent reads its Dexterity; the base shows here.
        TheForgotten => {
            let miasma = g.mv("MIASMA", vec![Debuff { strong: false }, Defend, Buff]);
            let dread = g.mv("DREAD", vec![atk(d(13, 15))]);
            g.follow(miasma, dread);
            g.follow(dread, miasma);
            g.done(miasma)
        }
        MechaKnight => {
            let charge = g.mv("CHARGE_MOVE", vec![atk(d(25, 30))]);
            let flame = g.mv("FLAMETHROWER_MOVE", vec![Status { count: 4 }]);
            let windup = g.mv("WINDUP_MOVE", vec![Defend, Buff]);
            let cleave = g.mv("HEAVY_CLEAVE_MOVE", vec![atk(d(35, 40))]);
            g.follow(charge, flame);
            g.follow(flame, windup);
            g.follow(windup, cleave);
            g.follow(cleave, flame);
            g.done(charge)
        }
        FlailKnight => {
            let chant = g.mv("WAR_CHANT", vec![Buff]);
            let flail = g.mv("FLAIL_MOVE", vec![multi(d(9, 10), 2)]);
            let ram = g.mv("RAM_MOVE", vec![atk(d(15, 17))]);
            let rand = g.random(vec![br(chant, Repeat::CannotRepeat), br(flail, Repeat::Times(2)), br(ram, Repeat::Times(2))]);
            g.follow(chant, rand);
            g.follow(flail, rand);
            g.follow(ram, rand);
            g.done(ram)
        }
        SpectralKnight => {
            let hex = g.mv("HEX", vec![Debuff { strong: false }]);
            let slash = g.mv("SOUL_SLASH", vec![atk(d(15, 17))]);
            let flame = g.mv("SOUL_FLAME", vec![multi(d(3, 4), 3)]);
            let rand = g.random(vec![br(slash, Repeat::Times(2)), br(flame, Repeat::CannotRepeat)]);
            g.follow(hex, slash);
            g.follow(slash, rand);
            g.follow(flame, rand);
            g.done(hex)
        }
        MagiKnight => {
            let shield = g.mv("POWER_SHIELD_MOVE", vec![atk(d(6, 7)), Defend]);
            let dampen = g.mv("DAMPEN_MOVE", vec![Debuff { strong: false }]);
            let prep = g.mv("PREP_MOVE", vec![Defend]);
            let bomb = g.mv("MAGIC_BOMB", vec![atk(d(35, 40))]);
            let spear = g.mv("RAM_MOVE", vec![atk(d(10, 11))]);
            g.follow(shield, dampen);
            g.follow(dampen, spear);
            g.follow(spear, prep);
            g.follow(prep, bomb);
            g.follow(bomb, spear);
            g.done(shield)
        }
        SoulNexus => {
            let burn = g.mv("SOUL_BURN_MOVE", vec![atk(d(29, 31))]);
            let maelstrom = g.mv("MAELSTROM_MOVE", vec![multi(d(6, 7), 4)]);
            let drain = g.mv("DRAIN_LIFE_MOVE", vec![atk(d(18, 19)), Debuff { strong: true }]);
            let rand = g.random(vec![
                br(burn, Repeat::CannotRepeat),
                br(maelstrom, Repeat::CannotRepeat),
                br(drain, Repeat::CannotRepeat),
            ]);
            g.follow(burn, rand);
            g.follow(maelstrom, rand);
            g.follow(drain, rand);
            g.done(burn)
        }
        // Builds bots while it has room for them, then just hits.
        Fabricator => {
            let fabricate = g.mv("FABRICATE_MOVE", vec![Summon]);
            let strike = g.mv("FABRICATING_STRIKE_MOVE", vec![atk(d(18, 21)), Summon]);
            let disintegrate = g.mv("DISINTEGRATE_MOVE", vec![atk(d(11, 13))]);
            let rand = g.random(vec![br(fabricate, Repeat::Forever), br(strike, Repeat::Forever)]);
            let branch = g.cond(vec![(rand, Cond::CanFabricate), (disintegrate, Cond::CannotFabricate)]);
            g.follow(fabricate, branch);
            g.follow(strike, branch);
            g.follow(disintegrate, branch);
            g.done(branch)
        }
        Zapbot => {
            let zap = g.mv("ZAP", vec![atk(d(14, 15))]);
            g.follow(zap, zap);
            g.done(zap)
        }
        Stabbot => {
            let stab = g.mv("STAB_MOVE", vec![atk(d(11, 12)), Debuff { strong: false }]);
            g.follow(stab, stab);
            g.done(stab)
        }
        Guardbot => {
            let guard = g.mv("GUARD_MOVE", vec![Defend]);
            g.follow(guard, guard);
            g.done(guard)
        }
        Noisebot => {
            let noise = g.mv("NOISE_MOVE", vec![Status { count: 2 }]);
            g.follow(noise, noise);
            g.done(noise)
        }
        // Burn Bright feeds the Amalgam until it falls; then the Queen fights.
        Queen => {
            let puppet = g.mv("PUPPET_STRINGS_MOVE", vec![CardDebuff]);
            let mine = g.mv("YOU_ARE_MINE_MOVE", vec![Debuff { strong: false }]);
            let burn = g.mv("BURN_BRIGHT_FOR_ME_MOVE", vec![Buff, Defend]);
            let off = g.mv("OFF_WITH_YOUR_HEAD_MOVE", vec![multi(d(3, 4), 5)]);
            let execution = g.mv("EXECUTION_MOVE", vec![atk(d(15, 18))]);
            let enrage = g.mv("ENRAGE_MOVE", vec![Buff]);
            let mine_branch = g.cond(vec![(burn, Cond::AmalgamAlive), (off, Cond::AmalgamDead)]);
            let burn_branch = g.cond(vec![(burn, Cond::AmalgamAlive), (off, Cond::AmalgamDead)]);
            g.follow(puppet, mine);
            g.follow(mine, mine_branch);
            g.follow(burn, burn_branch);
            g.follow(off, execution);
            g.follow(execution, enrage);
            g.follow(enrage, off);
            g.done(puppet)
        }
        TorchHeadAmalgam => {
            let t1 = g.mv("TACKLE_MOVE", vec![atk(d(18, 19))]);
            let t2 = g.mv("TACKLE_2_MOVE", vec![atk(d(18, 19))]);
            let beam = g.mv("BEAM_MOVE", vec![multi(8, 3)]);
            let t3 = g.mv("TACKLE_3_MOVE", vec![atk(d(14, 15))]);
            let t4 = g.mv("TACKLE_4_MOVE", vec![atk(d(14, 15))]);
            g.follow(t1, t2);
            g.follow(t2, beam);
            g.follow(beam, t3);
            g.follow(t3, t4);
            g.follow(t4, beam);
            g.done(t1)
        }
        // Three forms. Death forces Respawn (`TestSubject.TriggerDeadState`),
        // which leads into the next form's loop.
        TestSubject => {
            let respawn = g.mv_once("RESPAWN_MOVE", vec![Heal, Buff]);
            let bite = g.mv("BITE_MOVE", vec![atk(d(20, 22))]);
            let bash = g.mv("SKULL_BASH_MOVE", vec![atk(d(14, 16)), Debuff { strong: false }]);
            let claw = g.mv("MULTI_CLAW_MOVE", vec![multi(d(10, 11), 3)]);
            let lacerate = g.mv("PHASE3_LACERATE_MOVE", vec![multi(d(10, 11), 3)]);
            let pounce = g.mv("BIG_POUNCE", vec![atk(45)]);
            let growl = g.mv("BURNING_GROWL_MOVE", vec![Status { count: d(3, 5) as u32 }, Buff]);
            let revive = g.cond(vec![(claw, Cond::SecondForm), (lacerate, Cond::ThirdForm)]);
            g.follow(bite, bash);
            g.follow(bash, bite);
            g.follow(claw, claw);
            g.follow(lacerate, pounce);
            g.follow(pounce, growl);
            g.follow(growl, lacerate);
            g.follow(respawn, revive);
            g.done(bite)
        }
        Aeonglass => {
            let ebb = g.mv("EBB_MOVE", vec![atk(d(26, 32)), Defend]);
            let lasers = g.mv("EYE_LASERS_MOVE", vec![multi(d(11, 12), 2)]);
            let intensity = g.mv("INCREASING_INTENSITY_MOVE", vec![Status { count: d(1, 2) as u32 }, Buff]);
            g.follow(ebb, lasers);
            g.follow(lasers, intensity);
            g.follow(intensity, ebb);
            g.done(ebb)
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

/// The moves one monster can show, across every positional variant. Tooling
/// uses it to describe an encounter without anyone writing notes by hand.
pub fn move_names_of(id: MonsterId, asc: Ascension) -> Vec<&'static str> {
    let variants = [
        Flags::default(),
        Flags { is_alone: true, ..Default::default() },
        Flags { is_front: true, ..Default::default() },
        Flags { middle: true, ..Default::default() },
        Flags { starts_with_dance: true, ..Default::default() },
        Flags { start_stunned: true, ..Default::default() },
    ];
    let mut names: Vec<&'static str> = vec![];
    for flags in variants {
        for s in graph(id, asc, flags).0 {
            if let State::Move { name, .. } = s {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// Index into `all_move_names`.
pub fn move_index(name: &str) -> Option<usize> {
    static INDEX: std::sync::OnceLock<std::collections::HashMap<&'static str, usize>> = std::sync::OnceLock::new();
    INDEX.get_or_init(|| all_move_names().iter().enumerate().map(|(i, &n)| (n, i)).collect()).get(name).copied()
}
