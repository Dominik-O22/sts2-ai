//! What a run draws the moment it starts, value for value with the game:
//! the acts it visits (`StartRunLobby.BeginRunLocally`,
//! `ActModel.GetRandomList`), then on the run's UpFront stream the relic grab
//! bags (`RunManager.InitializeNewRun`, `Runs/RelicGrabBag.cs`) and every
//! act's rooms (`RunManager.GenerateRooms`, `ActModel.GenerateRooms`,
//! `Helpers/GrabBag.cs`): the order its events, weak, normal and elite
//! encounters come in, its boss (the last act a second one at A10) and its
//! ancient. The run pulls from these lists front to back as it enters
//! rooms (`RoomSet`), and relic rewards pull from the bags.
//!
//! What it draws depends on the profile's unlocks (`Unlocks`), which gate
//! acts, events, relics and ancients behind epochs. `tools/oracle rooms`
//! prints the game's plan for a seed and an unlock state;
//! `examples/roomcheck.rs` diffs this port against it, and
//! `examples/historycheck.rs` checks it against the rooms real runs met.
//!
//! Not here: the first run's fixed room order
//! (`Overgrowth.ApplyActDiscoveryOrderModifications` at `NumberOfRuns == 0`),
//! multiplayer (one room fewer, a bag per player), and the act 1 a lobby can
//! force or a modifier can change.

use std::fmt::Write;

use crate::encounter::{Act, Encounter, Kind, ALL as ENCOUNTERS};
use crate::game_rng::{hash, GameRng};
use crate::map::rooms;
use crate::relic::RelicId;
use crate::replay::slug;
use crate::types::{Ascension, AscensionLevel, RelicRarity};

/// `ModelDb.Acts`, in its order.
const ACTS: [Act; 4] = [Act::Overgrowth, Act::Underdocks, Act::Hive, Act::Glory];

/// The epochs (`Timeline/Epochs/*.cs`) whose reveal changes what a run
/// plans. The game's id is the name in screaming snake case plus `_EPOCH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epoch {
    Underdocks,
    Neow,
    Darv,
    Orobas,
    Event1,
    Event2,
    Event3,
    Relic1,
    Relic2,
    Relic3,
    Relic4,
    Relic5,
    Ironclad3,
    Ironclad6,
}

impl Epoch {
    pub const ALL: [Epoch; 14] = [
        Epoch::Underdocks,
        Epoch::Neow,
        Epoch::Darv,
        Epoch::Orobas,
        Epoch::Event1,
        Epoch::Event2,
        Epoch::Event3,
        Epoch::Relic1,
        Epoch::Relic2,
        Epoch::Relic3,
        Epoch::Relic4,
        Epoch::Relic5,
        Epoch::Ironclad3,
        Epoch::Ironclad6,
    ];

    /// `EpochModel.Id`.
    pub fn id(self) -> String {
        format!("{}_EPOCH", slug(&format!("{self:?}")))
    }

    /// The relics it unlocks, `Relic<N>Epoch.Relics` and
    /// `Ironclad<N>Epoch.Relics`. `SharedRelicPool.GetUnlockedRelics` and
    /// `IroncladRelicPool.GetUnlockedRelics` drop them while it is hidden.
    fn relics(self) -> &'static [RelicId] {
        use RelicId::*;
        match self {
            Epoch::Relic1 => &[UnsettlingLamp, IntimidatingHelmet, ReptileTrinket],
            Epoch::Relic2 => &[BookOfFiveRings, IceCream, Kusarigama],
            Epoch::Relic3 => &[VexingPuzzlebox, RippleBasin, FestivePopper],
            Epoch::Relic4 => &[MiniatureCannon, TungstenRod, WhiteStar],
            Epoch::Relic5 => &[TinyMailbox, JossPaper, BeatingRemnant],
            Epoch::Ironclad3 => &[RedSkull, PaperPhrog, RuinedHelmet],
            Epoch::Ironclad6 => &[SelfFormingClay, CharonsAshes, DemonTongue],
            _ => &[],
        }
    }

    /// The event it unlocks, `Event<N>Epoch.Events`, which
    /// `ActModel.GenerateRooms` drops while it is hidden.
    fn events(self) -> &'static [&'static str] {
        match self {
            Epoch::Event1 => &["TrashHeap"],
            Epoch::Event2 => &["Reflections"],
            Epoch::Event3 => &["ColorfulPhilosophers"],
            _ => &[],
        }
    }

    /// The ancient it unlocks: Neow in either act 1
    /// (`Overgrowth.GetUnlockedAncients`), Orobas in the Hive, Darv among the
    /// shared ones (`UnlockState.SharedAncients`).
    fn ancient(self) -> Option<&'static str> {
        match self {
            Epoch::Neow => Some("Neow"),
            Epoch::Darv => Some("Darv"),
            Epoch::Orobas => Some("Orobas"),
            _ => None,
        }
    }
}

/// The profile a run starts from, as far as planning reads it
/// (`Unlocks/UnlockState.cs` and the progress save's discovered acts). The
/// default has everything unlocked and seen.
#[derive(Clone, Debug, Default)]
pub struct Unlocks {
    /// Epochs not yet revealed on the timeline.
    pub locked: Vec<Epoch>,
    /// Underdocks is unlocked but was never entered
    /// (`ProgressState.DiscoveredActs`), so act 1 is Underdocks with no draw.
    pub underdocks_undiscovered: bool,
    /// Bosses never fought (`UnlockState.HasSeenEncounter`). Every standard
    /// run gets `ActModel.ApplyDiscoveryOrderModifications`, which makes the
    /// first of an act's `BossDiscoveryOrder` left in here its boss.
    pub unseen_bosses: Vec<Encounter>,
}

impl Unlocks {
    fn act(&self, act: Act) -> bool {
        act != Act::Underdocks || !self.locked.contains(&Epoch::Underdocks)
    }

    /// The pools' `GetUnlockedRelics`. Each epoch lists relics of one pool
    /// only, so checking every hidden epoch is the same as checking the
    /// pool's own.
    fn relic(&self, relic: BagRelic) -> bool {
        !self.locked.iter().any(|e| matches!(relic, BagRelic::Sim(id) if e.relics().contains(&id)))
    }

    fn event(&self, event: &str) -> bool {
        !self.locked.iter().any(|e| e.events().contains(&event))
    }

    fn ancient(&self, ancient: &str) -> bool {
        !self.locked.iter().any(|e| e.ancient() == Some(ancient))
    }
}

/// A relic in a grab bag: the sim's id, or the game's class name for the two
/// the sim has no id for, which only ever sit in the shared bag's event and
/// ancient deques.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BagRelic {
    Sim(RelicId),
    Game(&'static str),
}

impl BagRelic {
    /// The game's model id entry.
    pub fn game_id(self) -> String {
        match self {
            BagRelic::Sim(id) => slug(&format!("{id:?}")),
            BagRelic::Game(name) => slug(name),
        }
    }
}

/// `RelicGrabBag` once populated: a deque per rarity, in the order each
/// rarity first shows up in the relics it was given (a .NET `Dictionary`
/// enumerates in insertion order) and shuffled in that order. Rewards pull
/// from the front, shops from the back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelicBag {
    pub deques: Vec<(RelicRarity, Vec<BagRelic>)>,
}

impl RelicBag {
    /// `RelicGrabBag.Populate`.
    fn populate(relics: impl Iterator<Item = (BagRelic, RelicRarity)>, rng: &mut GameRng) -> Self {
        let mut deques: Vec<(RelicRarity, Vec<BagRelic>)> = Vec::new();
        for (relic, rarity) in relics {
            match deques.iter_mut().find(|(r, _)| *r == rarity) {
                Some((_, deque)) => deque.push(relic),
                None => deques.push((rarity, vec![relic])),
            }
        }
        for (_, deque) in &mut deques {
            rng.shuffle(deque);
        }
        Self { deques }
    }
}

/// One act's `RoomSet` as `ActModel.GenerateRooms` leaves it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActPlan {
    pub act: Act,
    /// Event class names (`Models/Events/<Name>.cs`). A `?` room takes the
    /// next one the run allows (`RoomSet.EnsureNextEventIsValid`).
    pub events: Vec<&'static str>,
    /// The weak encounters first, then the rest; a monster room takes the
    /// next one, wrapping around.
    pub normal: Vec<Encounter>,
    /// Fifteen, wrapping around like `normal`.
    pub elites: Vec<Encounter>,
    pub boss: Encounter,
    /// The last act's at Double Boss (A10).
    pub second_boss: Option<Encounter>,
    /// The ancient's class name. None when no ancient is unlocked, as for
    /// act 1 before Neow is.
    pub ancient: Option<&'static str>,
}

/// Everything drawn at run start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunPlan {
    pub acts: Vec<ActPlan>,
    /// `RunState.SharedRelicGrabBag`: every unlocked shared relic, any rarity.
    pub shared_bag: RelicBag,
    /// The player's own (`Player.RelicGrabBag`): the unlocked shared and
    /// Ironclad relics of the rarities rewards give.
    pub player_bag: RelicBag,
    /// The UpFront stream's counter after planning, which a save stores.
    pub up_front: u32,
}

impl RunPlan {
    /// A new singleplayer Ironclad run's plan, as `RunState.CreateForNewRun`
    /// and `RunManager.SetUpNewSingleplayer` make it. `seed` is the run's
    /// (`RunRngs::seed`); `acts` usually `select_acts`, unless the lobby
    /// forced act 1.
    pub fn generate(seed: u32, acts: [Act; 3], ascension: Ascension, unlocks: &Unlocks) -> Self {
        let mut rng = GameRng::named(seed, "up_front");

        // `InitializeNewRun`: the shared bag, then the player's
        // (`Player.PopulateRelicGrabBagIfNecessary`), whose list
        // `RelicGrabBag.Populate(Player, Rng)` builds.
        let shared = || SHARED_RELICS.iter().copied().filter(|&(r, _)| unlocks.relic(r));
        let shared_bag = RelicBag::populate(shared(), &mut rng);
        let ironclad = IRONCLAD_RELICS.iter().copied().filter(|&(r, _)| unlocks.relic(r));
        let rewardable = |&(_, rarity): &(BagRelic, RelicRarity)| {
            matches!(rarity, RelicRarity::Common | RelicRarity::Uncommon | RelicRarity::Rare | RelicRarity::Shop)
        };
        let player_bag = RelicBag::populate(shared().chain(ironclad).filter(rewardable), &mut rng);

        // `GenerateRooms`: the shared ancients go to acts 2 and 3, a random
        // number of them to each, before any act is filled.
        let mut ancients: Vec<&'static str> = SHARED_ANCIENTS.iter().copied().filter(|a| unlocks.ancient(a)).collect();
        rng.shuffle(&mut ancients);
        let mut subsets = vec![Vec::new()];
        for _ in 1..acts.len() {
            let count = rng.next_int(ancients.len() as i32 + 1) as usize;
            subsets.push(ancients.drain(..count).collect());
        }
        let mut plans = Vec::new();
        for (i, (&act, subset)) in acts.iter().zip(subsets).enumerate() {
            let mut plan = generate_rooms(act, subset, unlocks, &mut rng);
            // `ActModel.ApplyDiscoveryOrderModifications`.
            if let Some(&boss) = table(act).boss_discovery_order.iter().find(|b| unlocks.unseen_bosses.contains(b)) {
                plan.boss = boss;
            }
            if i == acts.len() - 1 && ascension.has(AscensionLevel::DoubleBoss) {
                let others: Vec<Encounter> = bosses(act).filter(|&b| b != plan.boss).collect();
                plan.second_boss = rng.pick(&others).copied();
            }
            plans.push(plan);
        }
        Self { acts: plans, shared_bag, player_bag, up_front: rng.counter }
    }

    /// The plan as `tools/oracle rooms` prints it, header aside.
    pub fn oracle_text(&self) -> String {
        let join = |ids: Vec<String>| ids.join(" ");
        let encounters = |list: &[Encounter]| join(list.iter().map(|e| slug(&format!("{e:?}"))).collect());
        let mut out = String::new();
        for (name, bag) in [("shared_bag", &self.shared_bag), ("player_bag", &self.player_bag)] {
            for (rarity, deque) in &bag.deques {
                let _ = writeln!(out, "{name} {rarity:?} {}", join(deque.iter().map(|r| r.game_id()).collect()));
            }
        }
        for act in &self.acts {
            let bosses: Vec<Encounter> = [Some(act.boss), act.second_boss].into_iter().flatten().collect();
            let _ = writeln!(out, "act {}", slug(&format!("{:?}", act.act)));
            let _ = writeln!(out, "events {}", join(act.events.iter().map(|e| slug(e)).collect()));
            let _ = writeln!(out, "normal {}", encounters(&act.normal));
            let _ = writeln!(out, "elite {}", encounters(&act.elites));
            let _ = writeln!(out, "boss {}", encounters(&bosses));
            let _ = writeln!(out, "ancient {}", act.ancient.map(slug).unwrap_or_default());
        }
        let _ = writeln!(out, "up_front {}", self.up_front);
        out
    }
}

/// `ActModel.GetRandomList` on the run's `act_selection` stream: for each act
/// index, one of the unlocked acts there in `ModelDb.Acts` order, except
/// that an unlocked alt act nobody has entered yet is taken without a draw.
pub fn select_acts(seed: u32, unlocks: &Unlocks) -> [Act; 3] {
    let mut rng = GameRng::named(seed, "act_selection");
    [0, 1, 2].map(|index| {
        let open: Vec<Act> = ACTS.iter().copied().filter(|&a| a.index() == index && unlocks.act(a)).collect();
        if open.contains(&Act::Underdocks) && unlocks.underdocks_undiscovered {
            return Act::Underdocks;
        }
        *rng.pick(&open).expect("an unlocked act at every index")
    })
}

/// `ActModel.GenerateRooms` for one player.
fn generate_rooms(act: Act, shared_ancients: Vec<&'static str>, unlocks: &Unlocks, rng: &mut GameRng) -> ActPlan {
    let table = table(act);
    let mut events: Vec<&'static str> =
        table.events.iter().chain(SHARED_EVENTS).copied().filter(|e| unlocks.event(e)).collect();
    rng.shuffle(&mut events);

    let of_kind = |kind: Kind| table.encounters.iter().copied().filter(move |e| e.kind() == kind);
    // The weak and the normal encounters share one list, so the first normal
    // one avoids the last weak one's tags.
    let mut normal = Vec::new();
    let mut bag = GrabBag::default();
    for _ in 0..table.weak {
        bag.refill_if_empty(of_kind(Kind::Weak));
        add_without_repeating_tags(&mut normal, &mut bag, rng);
    }
    let mut bag = GrabBag::default();
    for _ in table.weak..rooms(act) {
        bag.refill_if_empty(of_kind(Kind::Normal));
        add_without_repeating_tags(&mut normal, &mut bag, rng);
    }
    let mut elites = Vec::new();
    let mut bag = GrabBag::default();
    for _ in 0..15 {
        bag.refill_if_empty(of_kind(Kind::Elite));
        add_without_repeating_tags(&mut elites, &mut bag, rng);
    }

    let all_bosses: Vec<Encounter> = bosses(act).collect();
    let boss = *rng.pick(&all_bosses).expect("every act has bosses");
    let ancients: Vec<&'static str> =
        table.ancients.iter().copied().filter(|a| unlocks.ancient(a)).chain(shared_ancients).collect();
    let ancient = rng.pick(&ancients).copied();
    ActPlan { act, events, normal, elites, boss, second_boss: None, ancient }
}

/// `ActModel.AllBossEncounters`.
fn bosses(act: Act) -> impl Iterator<Item = Encounter> {
    table(act).encounters.iter().copied().filter(|e| e.kind() == Kind::Boss)
}

/// `ActModel.AddWithoutRepeatingTags`: an encounter that is neither the last
/// one nor shares a tag with it, or failing that any.
fn add_without_repeating_tags(list: &mut Vec<Encounter>, bag: &mut GrabBag, rng: &mut GameRng) {
    let last = list.last().copied();
    let fresh = |e: Encounter| last.is_none_or(|last| e != last && !e.shares_tags_with(last));
    if let Some(e) = bag.grab_and_remove_where(rng, fresh).or_else(|| bag.grab_and_remove(rng)) {
        list.push(e);
    }
}

/// `Helpers/GrabBag.cs` with every entry weighing 1, as `GenerateRooms`
/// fills it. One `NextDouble` scaled by the total weight picks an entry.
#[derive(Default)]
struct GrabBag(Vec<Encounter>);

impl GrabBag {
    fn refill_if_empty(&mut self, encounters: impl Iterator<Item = Encounter>) {
        if self.0.is_empty() {
            self.0.extend(encounters);
        }
    }

    /// `GrabIndex(rng)`: the first entry whose running weight passes the
    /// roll.
    fn index(&self, rng: &mut GameRng) -> Option<usize> {
        let roll = rng.next_double() * self.0.len() as f64;
        (0..self.0.len()).find(|&i| roll < (i + 1) as f64)
    }

    /// `GrabAndRemove(rng)`, which draws even from an empty bag.
    fn grab_and_remove(&mut self, rng: &mut GameRng) -> Option<Encounter> {
        self.index(rng).map(|i| self.0.remove(i))
    }

    /// `GrabAndRemove(rng, predicate)`: nothing, without a draw, if no entry
    /// passes, else rolls until one does.
    fn grab_and_remove_where(&mut self, rng: &mut GameRng, keep: impl Fn(Encounter) -> bool) -> Option<Encounter> {
        if !self.0.iter().any(|&e| keep(e)) {
            return None;
        }
        loop {
            let i = self.index(rng)?;
            if keep(self.0[i]) {
                return Some(self.0.remove(i));
            }
        }
    }
}

/// An act's lists as its `Models/Acts/<Act>.cs` gives them.
struct ActTable {
    /// `GenerateAllEncounters`, in its order.
    encounters: &'static [Encounter],
    /// `AllEvents`, class names.
    events: &'static [&'static str],
    /// `AllAncients`.
    ancients: &'static [&'static str],
    boss_discovery_order: [Encounter; 3],
    /// `NumberOfWeakEncounters`.
    weak: usize,
}

fn table(act: Act) -> &'static ActTable {
    match act {
        Act::Overgrowth => &OVERGROWTH,
        Act::Underdocks => &UNDERDOCKS,
        Act::Hive => &HIVE,
        Act::Glory => &GLORY,
    }
}

const OVERGROWTH: ActTable = {
    use Encounter::*;
    ActTable {
        encounters: &[
            BygoneEffigyElite, ByrdonisElite, CeremonialBeastBoss, CubexConstructNormal, FlyconidNormal,
            FogmogNormal, FuzzyWurmCrawlerWeak, InkletsNormal, MawlerNormal, NibbitsNormal, NibbitsWeak,
            OvergrowthCrawlers, PhrogParasiteElite, RubyRaidersNormal, ShrinkerBeetleWeak, SlimesNormal,
            SlimesWeak, SlitheringStranglerNormal, SnappingJaxfruitNormal, TheKinBoss, VantomBoss,
            VineShamblerNormal,
        ],
        events: &[
            "AromaOfChaos", "ByrdonisNest", "DenseVegetation", "JungleMazeAdventure", "LuminousChoir",
            "MorphicGrove", "SapphireSeed", "SunkenStatue", "TabletOfTruth", "UnrestSite", "Wellspring",
            "WhisperingHollow", "WoodCarvings",
        ],
        ancients: &["Neow"],
        boss_discovery_order: [VantomBoss, CeremonialBeastBoss, TheKinBoss],
        weak: 3,
    }
};

const UNDERDOCKS: ActTable = {
    use Encounter::*;
    ActTable {
        encounters: &[
            CorpseSlugsNormal, CorpseSlugsWeak, CultistsNormal, FossilStalkerNormal, GremlinMercNormal,
            HauntedShipNormal, LagavulinMatriarchBoss, LivingFogNormal, PhantasmalGardenersElite,
            PunchConstructNormal, SeapunkNormal, SeapunkWeak, SewerClamNormal, SkulkingColonyElite,
            SludgeSpinnerWeak, SoulFyshBoss, TerrorEelElite, ToadpolesWeak, TwoTailedRatsNormal,
            WaterfallGiantBoss,
        ],
        events: &[
            "AbyssalBaths", "DrowningBeacon", "EndlessConveyor", "PunchOff", "SpiralingWhirlpool", "SunkenStatue",
            "SunkenTreasury", "DoorsOfLightAndDark", "TrashHeap", "WaterloggedScriptorium",
        ],
        ancients: &["Neow"],
        boss_discovery_order: [WaterfallGiantBoss, SoulFyshBoss, LagavulinMatriarchBoss],
        weak: 3,
    }
};

const HIVE: ActTable = {
    use Encounter::*;
    ActTable {
        encounters: &[
            BowlbugsNormal, BowlbugsWeak, ChompersNormal, DecimillipedeElite, EntomancerElite, ExoskeletonsNormal,
            ExoskeletonsWeak, HunterKillerNormal, KaiserCrabBoss, InfestedPrismsElite, KnowledgeDemonBoss,
            LouseProgenitorNormal, MytesNormal, OvicopterNormal, SlumberingBeetleNormal, SpinyToadNormal,
            TheInsatiableBoss, TheObscuraNormal, ThievingHopperWeak, TunnelerWeak,
        ],
        events: &[
            "Amalgamator", "Bugslayer", "ColorfulPhilosophers", "ColossalFlower", "FieldOfManSizedHoles",
            "InfestedAutomaton", "LostWisp", "SpiritGrafter", "TheLanternKey", "ZenWeaver",
        ],
        ancients: &["Orobas", "Pael", "Tezcatara"],
        boss_discovery_order: [TheInsatiableBoss, KnowledgeDemonBoss, KaiserCrabBoss],
        weak: 2,
    }
};

const GLORY: ActTable = {
    use Encounter::*;
    ActTable {
        encounters: &[
            AxebotsNormal, ConstructMenagerieNormal, DevotedSculptorWeak, AeonglassBoss, FabricatorNormal,
            FrogKnightNormal, GlobeHeadNormal, KnightsElite, MechaKnightElite, OwlMagistrateNormal, QueenBoss,
            ScrollsOfBitingNormal, ScrollsOfBitingWeak, SlimedBerserkerNormal, SoulNexusElite, TestSubjectBoss,
            TheLostAndForgottenNormal, TurretOperatorWeak,
        ],
        events: &[
            "BattlewornDummy", "GraveOfTheForgotten", "HungryForMushrooms", "Reflections", "RoundTeaParty", "Trial",
            "TinkerTime",
        ],
        ancients: &["Nonupeipe", "Tanx", "Vakuu"],
        boss_discovery_order: [QueenBoss, TestSubjectBoss, AeonglassBoss],
        weak: 2,
    }
};

/// `ModelDb.AllSharedEvents`, open to every act after its own.
const SHARED_EVENTS: &[&str] = &[
    "BrainLeech", "CrystalSphere", "DollRoom", "FakeMerchant", "PotionCourier", "RanwidTheElder", "RelicTrader",
    "RoomFullOfCheese", "SelfHelpBook", "SlipperyBridge", "StoneOfAllTime", "Symbiote", "TeaMaster",
    "TheFutureOfPotions", "TheLegendsWereTrue", "ThisOrThat", "WarHistorianRepy", "WelcomeToWongos",
];

/// `ModelDb.AllSharedAncients`.
const SHARED_ANCIENTS: &[&str] = &["Darv"];

/// `RelicPools/SharedRelicPool.cs`, in its order, each with its class's
/// `Rarity`.
const SHARED_RELICS: &[(BagRelic, RelicRarity)] = {
    use BagRelic::{Game, Sim};
    use RelicId::*;
    use RelicRarity::{Ancient, Common, Event, Rare, Shop, Uncommon};
    &[
        (Sim(Akabeko), Uncommon), (Sim(AmethystAubergine), Common), (Sim(Anchor), Common), (Sim(ArtOfWar), Rare),
        (Sim(BagOfMarbles), Common), (Sim(BagOfPreparation), Common), (Sim(BeatingRemnant), Rare),
        (Sim(Bellows), Rare), (Sim(BeltBuckle), Shop), (Sim(BloodVial), Common), (Sim(BookOfFiveRings), Common),
        (Sim(BowlerHat), Uncommon), (Sim(Bread), Shop), (Sim(BronzeScales), Common), (Sim(BurningSticks), Shop),
        (Sim(Candelabra), Uncommon), (Sim(CaptainsWheel), Rare), (Sim(Cauldron), Shop),
        (Sim(CentennialPuzzle), Common), (Sim(Chandelier), Rare), (Sim(ChemicalX), Shop), (Sim(CloakClasp), Rare),
        (Sim(DingyRug), Shop), (Sim(DollysMirror), Shop), (Sim(DragonFruit), Shop), (Sim(EternalFeather), Uncommon),
        (Sim(FestivePopper), Common), (Game("FresnelLens"), Event), (Sim(FrozenEgg), Rare),
        (Sim(GamblingChip), Rare), (Sim(GamePiece), Rare), (Sim(GhostSeed), Shop), (Sim(Girya), Rare),
        (Sim(GnarledHammer), Shop), (Sim(Gorget), Common), (Sim(GremlinHorn), Uncommon), (Sim(HappyFlower), Common),
        (Sim(HornCleat), Uncommon), (Sim(IceCream), Rare), (Sim(IntimidatingHelmet), Rare),
        (Sim(JossPaper), Uncommon), (Sim(JuzuBracelet), Common), (Sim(Kifuda), Shop), (Sim(Kunai), Rare),
        (Sim(Kusarigama), Uncommon), (Sim(Lantern), Common), (Sim(LastingCandy), Uncommon), (Sim(LavaLamp), Shop),
        (Sim(LeesWaffle), Shop), (Sim(LetterOpener), Uncommon), (Sim(LizardTail), Rare),
        (Game("LoomingFruit"), Ancient), (Sim(LuckyFysh), Uncommon), (Sim(Mango), Rare), (Sim(MealTicket), Common),
        (Sim(MeatOnTheBone), Rare), (Sim(MembershipCard), Shop), (Sim(MercuryHourglass), Uncommon),
        (Sim(MiniatureCannon), Uncommon), (Sim(MiniatureTent), Shop), (Sim(MoltenEgg), Rare),
        (Sim(MummifiedHand), Rare), (Sim(MysticLighter), Shop), (Sim(Nunchaku), Uncommon),
        (Sim(OddlySmoothStone), Common), (Sim(OldCoin), Rare), (Sim(Orichalcum), Uncommon),
        (Sim(OrnamentalFan), Uncommon), (Sim(Orrery), Shop), (Sim(Pantograph), Uncommon),
        (Sim(ParryingShield), Uncommon), (Sim(Pear), Uncommon), (Sim(PenNib), Uncommon), (Sim(Pendulum), Common),
        (Sim(Permafrost), Uncommon), (Sim(PetrifiedToad), Uncommon), (Sim(Planisphere), Uncommon),
        (Sim(Pocketwatch), Rare), (Sim(PotionBelt), Common), (Sim(PrayerWheel), Rare), (Sim(PunchDagger), Shop),
        (Sim(RainbowRing), Rare), (Sim(RazorTooth), Rare), (Sim(RedMask), Common), (Sim(RegalPillow), Common),
        (Sim(ReptileTrinket), Uncommon), (Sim(RingingTriangle), Shop), (Sim(RippleBasin), Uncommon),
        (Sim(RoyalStamp), Shop), (Sim(ScreamingFlagon), Shop), (Sim(Shovel), Rare), (Sim(Shuriken), Rare),
        (Sim(SlingOfCourage), Shop), (Sim(SparklingRouge), Uncommon), (Sim(StoneCalendar), Rare),
        (Sim(StoneCracker), Uncommon), (Sim(Strawberry), Common), (Sim(StrikeDummy), Common),
        (Sim(SturdyClamp), Rare), (Sim(TheAbacus), Shop), (Sim(TheCourier), Rare), (Sim(TinyMailbox), Uncommon),
        (Sim(Toolbox), Shop), (Sim(ToxicEgg), Rare), (Sim(TungstenRod), Rare), (Sim(TuningFork), Uncommon),
        (Sim(UnceasingTop), Rare), (Sim(UnsettlingLamp), Rare), (Sim(Vajra), Common), (Sim(Vambrace), Uncommon),
        (Sim(VenerableTeaSet), Common), (Sim(VeryHotCocoa), Ancient), (Sim(VexingPuzzlebox), Rare),
        (Sim(WarPaint), Common), (Sim(Whetstone), Common), (Sim(WhiteBeastStatue), Rare), (Sim(WhiteStar), Rare),
        (Sim(WingCharm), Shop),
    ]
};

/// `RelicPools/IroncladRelicPool.cs`, in its order, each with its class's
/// `Rarity`.
const IRONCLAD_RELICS: &[(BagRelic, RelicRarity)] = {
    use BagRelic::Sim;
    use RelicId::*;
    use RelicRarity::{Common, Rare, Shop, Starter, Uncommon};
    &[
        (Sim(Brimstone), Shop), (Sim(BurningBlood), Starter), (Sim(CharonsAshes), Rare), (Sim(DemonTongue), Rare),
        (Sim(PaperPhrog), Uncommon), (Sim(RedSkull), Common), (Sim(RuinedHelmet), Rare),
        (Sim(SelfFormingClay), Uncommon),
    ]
};

/// Checks `tools/oracle rooms` output against `RunPlan::generate`: returns
/// how many runs it held and the header of each one that differs, with the
/// first line that does.
pub fn diff_oracle(text: &str) -> (usize, Vec<String>) {
    let mut runs: Vec<(&str, String)> = Vec::new();
    for line in text.lines() {
        match (line.strip_prefix("run "), runs.last_mut()) {
            (Some(header), _) => runs.push((header, String::new())),
            (None, Some((_, plan))) => {
                plan.push_str(line);
                plan.push('\n');
            }
            (None, None) => panic!("a plan before the first header: {line:?}"),
        }
    }
    let mut mismatches = Vec::new();
    for (header, want) in &runs {
        let (seed, ascension, unlocks) = parse_header(header);
        let seed = hash(seed) as u32;
        let got = RunPlan::generate(seed, select_acts(seed, &unlocks), ascension, &unlocks).oracle_text();
        if got != *want {
            let line = got
                .lines()
                .zip(want.lines())
                .find(|(g, w)| g != w)
                .map_or("different length".to_string(), |(g, w)| format!("got {g:?}, game {w:?}"));
            mismatches.push(format!("{header}: {line}"));
        }
    }
    (runs.len(), mismatches)
}

/// A `tools/oracle rooms` input line: `SEED ASCENSION [LOCKED [UNSEEN]]`.
fn parse_header(header: &str) -> (&str, Ascension, Unlocks) {
    let parts: Vec<&str> = header.split(' ').collect();
    let listed = |i: usize| parts.get(i).filter(|&&p| p != "-").map_or(Vec::new(), |p| p.split(',').collect());
    let locked = listed(2)
        .into_iter()
        .map(|id| *Epoch::ALL.iter().find(|e| e.id() == id).unwrap_or_else(|| panic!("unknown epoch {id}")))
        .collect();
    let unseen_bosses = listed(3)
        .into_iter()
        .map(|id| {
            *ENCOUNTERS
                .iter()
                .find(|e| slug(&format!("{e:?}")) == id)
                .unwrap_or_else(|| panic!("unknown encounter {id}"))
        })
        .collect();
    let unlocks = Unlocks { locked, unseen_bosses, ..Unlocks::default() };
    (parts[0], Ascension(parts[1].parse().expect("an ascension")), unlocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/oracle rooms` for seeds at A0 and A10, fully unlocked and with
    /// epochs locked and bosses unseen, after TBL5VNYN4M at A10, a real run
    /// whose history file lists the rooms it met.
    #[test]
    fn matches_the_game() {
        let (runs, mismatches) = diff_oracle(include_str!("../testdata/oracle-rooms.txt"));
        assert!(runs >= 30, "fixture holds {runs} runs");
        assert!(mismatches.is_empty(), "{} of {runs} differ:\n{}", mismatches.len(), mismatches.join("\n"));
    }

    /// Underdocks nobody has entered is act 1 whatever the seed, and the
    /// act 2 and 3 draws move up the stream.
    #[test]
    fn undiscovered_underdocks_is_forced() {
        let unlocks = Unlocks { underdocks_undiscovered: true, ..Unlocks::default() };
        for seed in 0..50 {
            assert_eq!(select_acts(seed, &unlocks), [Act::Underdocks, Act::Hive, Act::Glory]);
        }
        let locked = Unlocks { locked: vec![Epoch::Underdocks], underdocks_undiscovered: true, ..Unlocks::default() };
        assert!((0..50).all(|seed| select_acts(seed, &locked)[0] == Act::Overgrowth));
    }
}
