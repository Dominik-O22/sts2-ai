//! Encounters: `Models/Encounters/*.cs` via each act's `GenerateAllEncounters`.
//! Act 1 is not one fixed act. `ActModel.GetRandomList` picks per act index
//! from `ModelDb.ActsByIndex`, and both `Overgrowth` and `Underdocks` have
//! `Index => 0`, so a run rolls one of the two. Both are here, and act 2,
//! `Hive`, which is alone at `Index => 1`, and act 3 (`Glory`, `Index => 2`).
//!
//! Each encounter generates its monster list, rolling its own composition
//! where the game does.

use crate::combat::EnemySpec;
use crate::ids::MonsterId;
use crate::monster::Flags;
use crate::rng::Rng;

/// The act an encounter belongs to. `Acts/Overgrowth.cs`, `Acts/Underdocks.cs`
/// (the two act 1s), `Acts/Hive.cs` and `Acts/Glory.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Overgrowth,
    Underdocks,
    Hive,
    Glory,
}

impl Act {
    /// `ActModel.Index`: 0 for either act 1, 1 for act 2, 2 for act 3.
    pub fn index(self) -> u8 {
        match self {
            Act::Overgrowth | Act::Underdocks => 0,
            Act::Hive => 1,
            Act::Glory => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Weak,
    Normal,
    Elite,
    Boss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encounter {
    FuzzyWurmCrawlerWeak,
    NibbitsWeak,
    ShrinkerBeetleWeak,
    SlimesWeak,
    CubexConstructNormal,
    FlyconidNormal,
    FogmogNormal,
    InkletsNormal,
    MawlerNormal,
    NibbitsNormal,
    OvergrowthCrawlers,
    RubyRaidersNormal,
    SlimesNormal,
    SlitheringStranglerNormal,
    SnappingJaxfruitNormal,
    VineShamblerNormal,
    BygoneEffigyElite,
    ByrdonisElite,
    PhrogParasiteElite,
    VantomBoss,
    CeremonialBeastBoss,
    TheKinBoss,
    // Underdocks.
    CorpseSlugsWeak,
    SeapunkWeak,
    SludgeSpinnerWeak,
    ToadpolesWeak,
    CorpseSlugsNormal,
    CultistsNormal,
    FossilStalkerNormal,
    GremlinMercNormal,
    HauntedShipNormal,
    LivingFogNormal,
    PunchConstructNormal,
    SeapunkNormal,
    SewerClamNormal,
    TwoTailedRatsNormal,
    PhantasmalGardenersElite,
    SkulkingColonyElite,
    TerrorEelElite,
    LagavulinMatriarchBoss,
    SoulFyshBoss,
    WaterfallGiantBoss,
    // Hive.
    BowlbugsNormal,
    BowlbugsWeak,
    ChompersNormal,
    DecimillipedeElite,
    EntomancerElite,
    ExoskeletonsNormal,
    ExoskeletonsWeak,
    HunterKillerNormal,
    KaiserCrabBoss,
    InfestedPrismsElite,
    KnowledgeDemonBoss,
    LouseProgenitorNormal,
    MytesNormal,
    OvicopterNormal,
    SlumberingBeetleNormal,
    SpinyToadNormal,
    TheInsatiableBoss,
    TheObscuraNormal,
    ThievingHopperWeak,
    TunnelerWeak,
    // Act 3 (Glory).
    DevotedSculptorWeak,
    ScrollsOfBitingWeak,
    TurretOperatorWeak,
    AxebotsNormal,
    ConstructMenagerieNormal,
    FabricatorNormal,
    FrogKnightNormal,
    GlobeHeadNormal,
    OwlMagistrateNormal,
    ScrollsOfBitingNormal,
    SlimedBerserkerNormal,
    TheLostAndForgottenNormal,
    KnightsElite,
    MechaKnightElite,
    SoulNexusElite,
    AeonglassBoss,
    QueenBoss,
    TestSubjectBoss,
    // Event fights: an event's option starts them, never the map.
    DenseVegetationEventEncounter,
    PunchOffEventEncounter,
    MysteriousKnightEventEncounter,
    FakeMerchantEventEncounter,
    BattlewornDummyEventEncounter,
}

pub const ALL: &[Encounter] = &[
    Encounter::FuzzyWurmCrawlerWeak,
    Encounter::NibbitsWeak,
    Encounter::ShrinkerBeetleWeak,
    Encounter::SlimesWeak,
    Encounter::CubexConstructNormal,
    Encounter::FlyconidNormal,
    Encounter::FogmogNormal,
    Encounter::InkletsNormal,
    Encounter::MawlerNormal,
    Encounter::NibbitsNormal,
    Encounter::OvergrowthCrawlers,
    Encounter::RubyRaidersNormal,
    Encounter::SlimesNormal,
    Encounter::SlitheringStranglerNormal,
    Encounter::SnappingJaxfruitNormal,
    Encounter::VineShamblerNormal,
    Encounter::BygoneEffigyElite,
    Encounter::ByrdonisElite,
    Encounter::PhrogParasiteElite,
    Encounter::VantomBoss,
    Encounter::CeremonialBeastBoss,
    Encounter::TheKinBoss,
    Encounter::CorpseSlugsWeak,
    Encounter::SeapunkWeak,
    Encounter::SludgeSpinnerWeak,
    Encounter::ToadpolesWeak,
    Encounter::CorpseSlugsNormal,
    Encounter::CultistsNormal,
    Encounter::FossilStalkerNormal,
    Encounter::GremlinMercNormal,
    Encounter::HauntedShipNormal,
    Encounter::LivingFogNormal,
    Encounter::PunchConstructNormal,
    Encounter::SeapunkNormal,
    Encounter::SewerClamNormal,
    Encounter::TwoTailedRatsNormal,
    Encounter::PhantasmalGardenersElite,
    Encounter::SkulkingColonyElite,
    Encounter::TerrorEelElite,
    Encounter::LagavulinMatriarchBoss,
    Encounter::SoulFyshBoss,
    Encounter::WaterfallGiantBoss,
    Encounter::BowlbugsNormal,
    Encounter::BowlbugsWeak,
    Encounter::ChompersNormal,
    Encounter::DecimillipedeElite,
    Encounter::EntomancerElite,
    Encounter::ExoskeletonsNormal,
    Encounter::ExoskeletonsWeak,
    Encounter::HunterKillerNormal,
    Encounter::KaiserCrabBoss,
    Encounter::InfestedPrismsElite,
    Encounter::KnowledgeDemonBoss,
    Encounter::LouseProgenitorNormal,
    Encounter::MytesNormal,
    Encounter::OvicopterNormal,
    Encounter::SlumberingBeetleNormal,
    Encounter::SpinyToadNormal,
    Encounter::TheInsatiableBoss,
    Encounter::TheObscuraNormal,
    Encounter::ThievingHopperWeak,
    Encounter::TunnelerWeak,
    Encounter::DevotedSculptorWeak,
    Encounter::ScrollsOfBitingWeak,
    Encounter::TurretOperatorWeak,
    Encounter::AxebotsNormal,
    Encounter::ConstructMenagerieNormal,
    Encounter::FabricatorNormal,
    Encounter::FrogKnightNormal,
    Encounter::GlobeHeadNormal,
    Encounter::OwlMagistrateNormal,
    Encounter::ScrollsOfBitingNormal,
    Encounter::SlimedBerserkerNormal,
    Encounter::TheLostAndForgottenNormal,
    Encounter::KnightsElite,
    Encounter::MechaKnightElite,
    Encounter::SoulNexusElite,
    Encounter::AeonglassBoss,
    Encounter::QueenBoss,
    Encounter::TestSubjectBoss,
    Encounter::DenseVegetationEventEncounter,
    Encounter::PunchOffEventEncounter,
    Encounter::MysteriousKnightEventEncounter,
    Encounter::FakeMerchantEventEncounter,
    Encounter::BattlewornDummyEventEncounter,
];

fn one(id: MonsterId) -> EnemySpec {
    EnemySpec { id, flags: Flags::default() }
}

/// A monster in a named slot, by its 1-based position in `Slots`.
fn in_slot(id: MonsterId, slot: u8) -> EnemySpec {
    EnemySpec { id, flags: Flags { slot, ..Flags::default() } }
}

/// `CorpseSlug.StarterMoveIdx`: which of the three moves it opens on.
fn slug_with_move(idx: u8) -> EnemySpec {
    EnemySpec { id: MonsterId::CorpseSlug, flags: Flags { starter_move: idx, ..Flags::default() } }
}

/// `ScrollOfBiting.StarterMoveIdx`.
fn scroll_with_move(idx: u8) -> EnemySpec {
    EnemySpec { id: MonsterId::ScrollOfBiting, flags: Flags { starter_move: idx, ..Flags::default() } }
}

impl Encounter {
    /// Started by an event's option (`EventModel.EnterCombatWithoutExitingEvent`),
    /// never rolled for a map room. `kind` still follows the encounter's
    /// `RoomType`, which is `Monster` for all of them.
    pub fn is_event(self) -> bool {
        use Encounter::*;
        matches!(
            self,
            DenseVegetationEventEncounter
                | PunchOffEventEncounter
                | MysteriousKnightEventEncounter
                | FakeMerchantEventEncounter
                | BattlewornDummyEventEncounter
        )
    }

    pub fn kind(self) -> Kind {
        use Encounter::*;
        match self {
            FuzzyWurmCrawlerWeak | NibbitsWeak | ShrinkerBeetleWeak | SlimesWeak => Kind::Weak,
            CorpseSlugsWeak | SeapunkWeak | SludgeSpinnerWeak | ToadpolesWeak => Kind::Weak,
            DevotedSculptorWeak | ScrollsOfBitingWeak | TurretOperatorWeak => Kind::Weak,
            KnightsElite | MechaKnightElite | SoulNexusElite => Kind::Elite,
            AeonglassBoss | QueenBoss | TestSubjectBoss => Kind::Boss,
            BygoneEffigyElite | ByrdonisElite | PhrogParasiteElite => Kind::Elite,
            PhantasmalGardenersElite | SkulkingColonyElite | TerrorEelElite => Kind::Elite,
            VantomBoss | CeremonialBeastBoss | TheKinBoss => Kind::Boss,
            LagavulinMatriarchBoss | SoulFyshBoss | WaterfallGiantBoss => Kind::Boss,
            BowlbugsWeak | ExoskeletonsWeak | ThievingHopperWeak | TunnelerWeak => Kind::Weak,
            DecimillipedeElite | EntomancerElite | InfestedPrismsElite => Kind::Elite,
            KaiserCrabBoss | KnowledgeDemonBoss | TheInsatiableBoss => Kind::Boss,
            _ => Kind::Normal,
        }
    }

    /// Which act this encounter belongs to. The two act 1s never mix in a run.
    pub fn act(self) -> Act {
        use Encounter::*;
        match self {
            BowlbugsNormal | BowlbugsWeak | ChompersNormal | DecimillipedeElite | EntomancerElite
            | ExoskeletonsNormal | ExoskeletonsWeak | HunterKillerNormal | KaiserCrabBoss | InfestedPrismsElite
            | KnowledgeDemonBoss | LouseProgenitorNormal | MytesNormal | OvicopterNormal | SlumberingBeetleNormal
            | SpinyToadNormal | TheInsatiableBoss | TheObscuraNormal | ThievingHopperWeak | TunnelerWeak
            // `Acts/Hive.cs` lists `TheLanternKey`, whose fight this is.
            | MysteriousKnightEventEncounter => Act::Hive,
            DevotedSculptorWeak | ScrollsOfBitingWeak | TurretOperatorWeak | AxebotsNormal | ConstructMenagerieNormal
            | FabricatorNormal | FrogKnightNormal | GlobeHeadNormal | OwlMagistrateNormal | ScrollsOfBitingNormal
            | SlimedBerserkerNormal | TheLostAndForgottenNormal | KnightsElite | MechaKnightElite | SoulNexusElite
            | AeonglassBoss | QueenBoss | TestSubjectBoss
            // `Acts/Glory.cs` lists the `BattlewornDummy` event.
            | BattlewornDummyEventEncounter => Act::Glory,
            CorpseSlugsWeak | SeapunkWeak | SludgeSpinnerWeak | ToadpolesWeak | CorpseSlugsNormal | CultistsNormal
            | FossilStalkerNormal | GremlinMercNormal | HauntedShipNormal | LivingFogNormal | PunchConstructNormal
            | SeapunkNormal | SewerClamNormal | TwoTailedRatsNormal | PhantasmalGardenersElite | SkulkingColonyElite
            | TerrorEelElite | LagavulinMatriarchBoss | SoulFyshBoss | WaterfallGiantBoss
            // `Acts/Underdocks.cs` lists the `PunchOff` event.
            | PunchOffEventEncounter => Act::Underdocks,
            // `Acts/Overgrowth.cs` lists the `DenseVegetation` event.
            // `FakeMerchant` is one of `ModelDb.AllSharedEvents`, open to
            // every act; it sits here because an encounter names one.
            _ => Act::Overgrowth,
        }
    }

    /// `EncounterModel.GenerateMonsters`. `rng` stands in for the
    /// encounter's own seeded `Rng`.
    pub fn monsters(self, rng: &mut Rng) -> Vec<EnemySpec> {
        use Encounter::*;
        use MonsterId::*;
        let medium = |rng: &mut Rng| one(*rng.pick(&[LeafSlimeM, TwigSlimeM]).unwrap());
        match self {
            FuzzyWurmCrawlerWeak => vec![one(FuzzyWurmCrawler)],
            NibbitsWeak => vec![EnemySpec { id: Nibbit, flags: Flags { is_alone: true, ..Default::default() } }],
            ShrinkerBeetleWeak => vec![one(ShrinkerBeetle)],
            // One small slime, a random medium, then the other small slime.
            SlimesWeak => {
                let (a, b) = if rng.next_int(2) == 0 { (LeafSlimeS, TwigSlimeS) } else { (TwigSlimeS, LeafSlimeS) };
                vec![one(a), medium(rng), one(b)]
            }
            CubexConstructNormal => vec![one(CubexConstruct)],
            FlyconidNormal => vec![medium(rng), one(Flyconid)],
            // Slots are ["illusion", "fogmog"]: the Eye it summons stands
            // in front of it.
            FogmogNormal => vec![EnemySpec { id: Fogmog, flags: Flags { slot: 2, ..Default::default() } }],
            InkletsNormal => vec![
                one(Inklet),
                EnemySpec { id: Inklet, flags: Flags { middle: true, ..Default::default() } },
                one(Inklet),
            ],
            MawlerNormal => vec![one(Mawler)],
            NibbitsNormal => vec![
                EnemySpec { id: Nibbit, flags: Flags { is_front: true, ..Default::default() } },
                one(Nibbit),
            ],
            OvergrowthCrawlers => vec![one(ShrinkerBeetle), one(FuzzyWurmCrawler)],
            // Three distinct raiders from the five.
            RubyRaidersNormal => {
                let mut pool = vec![AxeRubyRaider, AssassinRubyRaider, BruteRubyRaider, CrossbowRubyRaider, TrackerRubyRaider];
                rng.shuffle(&mut pool);
                pool.into_iter().take(3).map(one).collect()
            }
            SlimesNormal => {
                let (a, b) = if rng.next_int(2) == 0 { (LeafSlimeS, TwigSlimeS) } else { (TwigSlimeS, LeafSlimeS) };
                vec![one(TwigSlimeM), one(LeafSlimeM), one(a), one(b)]
            }
            SlitheringStranglerNormal => {
                let mut v = vec![one(SlitheringStrangler)];
                match rng.next_int(3) {
                    0 => v.push(one(SnappingJaxfruit)),
                    1 => v.push(medium(rng)),
                    _ => {
                        v.push(one(*rng.pick(&[LeafSlimeS, TwigSlimeS]).unwrap()));
                        v.push(one(*rng.pick(&[LeafSlimeS, TwigSlimeS]).unwrap()));
                    }
                }
                v
            }
            SnappingJaxfruitNormal => vec![one(SnappingJaxfruit), one(Flyconid)],
            VineShamblerNormal => vec![one(VineShambler)],
            BygoneEffigyElite => vec![one(BygoneEffigy)],
            ByrdonisElite => vec![one(Byrdonis)],
            PhrogParasiteElite => vec![one(PhrogParasite)],
            VantomBoss => vec![one(Vantom)],
            CeremonialBeastBoss => vec![one(CeremonialBeast)],
            TheKinBoss => vec![
                EnemySpec { id: KinFollower, flags: Flags { starts_with_dance: true, ..Default::default() } },
                one(KinPriest),
                one(KinFollower),
            ],

            // `CorpseSlug.EnsureCorpseSlugsStartWithDifferentMoves`: one roll,
            // then consecutive starting moves down the line.
            CorpseSlugsWeak | CorpseSlugsNormal => {
                let n = if self == CorpseSlugsWeak { 2 } else { 3 };
                let first = rng.next_int(3) as u8;
                (0..n).map(|i| slug_with_move((first + i) % 3)).collect()
            }
            SeapunkWeak => vec![one(Seapunk)],
            SludgeSpinnerWeak => vec![one(SludgeSpinner)],
            // Front toadpole opens on Spiken, the back one on Whirl.
            ToadpolesWeak => vec![
                EnemySpec { id: Toadpole, flags: Flags { is_front: true, ..Default::default() } },
                one(Toadpole),
            ],
            CultistsNormal => vec![one(CalcifiedCultist), one(DampCultist)],
            FossilStalkerNormal => vec![one(FossilStalker)],
            // Only the merc starts; Surprise brings the other two when it dies.
            GremlinMercNormal => vec![one(GremlinMerc)],
            HauntedShipNormal => vec![one(HauntedShip)],
            // The five bomb slots stay empty until Bloat fills them.
            // Slots are bomb1..bomb5 then livingFog, so every bomb it
            // bloats out stands in front of it.
            LivingFogNormal => vec![EnemySpec { id: LivingFog, flags: Flags { slot: 6, ..Default::default() } }],
            PunchConstructNormal => vec![one(PunchConstruct)],
            SeapunkNormal => vec![one(CalcifiedCultist), one(Seapunk)],
            SewerClamNormal => vec![one(SewerClam)],
            // Three rats in the last three slots, starting moves offset by one.
            TwoTailedRatsNormal => {
                let first = rng.next_int(3) as u8;
                (0..3)
                    .map(|i| EnemySpec {
                        id: TwoTailedRat,
                        flags: Flags { starter_move: (first + i) % 3, slot: 3 + i as u8, ..Default::default() },
                    })
                    .collect()
            }
            // Each gardener opens on the move its slot names.
            PhantasmalGardenersElite => (1..=4)
                .map(|slot| EnemySpec { id: PhantasmalGardener, flags: Flags { slot, ..Default::default() } })
                .collect(),
            SkulkingColonyElite => vec![one(SkulkingColony)],
            TerrorEelElite => vec![one(TerrorEel)],
            LagavulinMatriarchBoss => vec![one(LagavulinMatriarch)],
            SoulFyshBoss => vec![one(SoulFysh)],
            WaterfallGiantBoss => vec![one(WaterfallGiant)],

            // BowlbugsNormal: the rock, then two workers drawn from egg, silk
            // and nectar, at most one of each.
            BowlbugsNormal => {
                let mut workers = vec![BowlbugEgg, BowlbugSilk, BowlbugNectar];
                let mut v = vec![one(BowlbugRock)];
                for _ in 0..2 {
                    let i = rng.next_int(workers.len());
                    v.push(one(workers.remove(i)));
                }
                v
            }
            BowlbugsWeak => vec![one(BowlbugRock), one(*rng.pick(&[BowlbugEgg, BowlbugNectar]).unwrap())],
            ChompersNormal => vec![one(Chomper), EnemySpec { id: Chomper, flags: Flags { scream_first: true, ..Flags::default() } }],
            // Consecutive starting moves down the body, from one roll.
            DecimillipedeElite => {
                let first = rng.next_int(3) as u8;
                [DecimillipedeSegmentFront, DecimillipedeSegmentMiddle, DecimillipedeSegmentBack]
                    .into_iter()
                    .enumerate()
                    .map(|(i, id)| EnemySpec {
                        id,
                        flags: Flags { starter_move: (first + i as u8) % 3, slot: i as u8 + 1, ..Flags::default() },
                    })
                    .collect()
            }
            EntomancerElite => vec![one(Entomancer)],
            ExoskeletonsNormal => (1..=4).map(|slot| in_slot(Exoskeleton, slot)).collect(),
            ExoskeletonsWeak => (1..=3).map(|slot| in_slot(Exoskeleton, slot)).collect(),
            HunterKillerNormal => vec![one(HunterKiller)],
            // Slots are ["crusher", "rocket"]: the player stands between them.
            KaiserCrabBoss => vec![in_slot(Crusher, 1), in_slot(Rocket, 2)],
            InfestedPrismsElite => vec![one(InfestedPrism)],
            KnowledgeDemonBoss => vec![one(KnowledgeDemon)],
            LouseProgenitorNormal => vec![one(LouseProgenitor)],
            MytesNormal => vec![in_slot(Myte, 1), in_slot(Myte, 2)],
            // Slots are egg1..egg5 then ovicopter; eggs fill from the back.
            OvicopterNormal => vec![in_slot(Ovicopter, 6)],
            SlumberingBeetleNormal => vec![one(BowlbugRock), one(BowlbugSilk), one(SlumberingBeetle)],
            SpinyToadNormal => vec![one(SpinyToad)],
            TheInsatiableBoss => vec![one(TheInsatiable)],
            // Slots are ["illusion", "obscura"]: its Parafright appears in front.
            TheObscuraNormal => vec![in_slot(TheObscura, 2)],
            ThievingHopperWeak => vec![one(ThievingHopper)],
            TunnelerWeak => vec![one(Tunneler)],
            DevotedSculptorWeak => vec![one(DevotedSculptor)],
            // Starting moves offset by one from a single roll, like the slugs.
            ScrollsOfBitingWeak | ScrollsOfBitingNormal => {
                let first = rng.next_int(3) as u8;
                let mut v: Vec<EnemySpec> = (0..3).map(|i| scroll_with_move((first + i) % 3)).collect();
                // The fourth scroll always opens on More Teeth.
                if self == ScrollsOfBitingNormal {
                    v.push(scroll_with_move(2));
                }
                v
            }
            TurretOperatorWeak => vec![one(LivingShield), one(TurretOperator)],
            AxebotsNormal => vec![one(Axebot)],
            ConstructMenagerieNormal => vec![one(PunchConstruct), one(CubexConstruct), one(CubexConstruct)],
            // Slots are bot1, bot2, fabricator, bot3, bot4: the bots it
            // builds fill in around it.
            FabricatorNormal => vec![EnemySpec { id: Fabricator, flags: Flags { slot: 3, ..Default::default() } }],
            FrogKnightNormal => vec![one(FrogKnight)],
            GlobeHeadNormal => vec![one(GlobeHead)],
            OwlMagistrateNormal => vec![one(OwlMagistrate)],
            SlimedBerserkerNormal => vec![one(SlimedBerserker)],
            TheLostAndForgottenNormal => vec![one(TheLost), one(TheForgotten)],
            KnightsElite => vec![one(FlailKnight), one(SpectralKnight), one(MagiKnight)],
            MechaKnightElite => vec![one(MechaKnight)],
            SoulNexusElite => vec![one(SoulNexus)],
            AeonglassBoss => vec![one(Aeonglass)],
            // Slots are amalgam, queen.
            QueenBoss => vec![one(TorchHeadAmalgam), one(Queen)],
            TestSubjectBoss => vec![one(TestSubject)],

            // `DenseVegetationEventEncounter`: a Wriggler in each of
            // wriggler1..4, none of them stunned, so each opens on its slot's move.
            DenseVegetationEventEncounter => (1..=4).map(|slot| in_slot(Wriggler, slot)).collect(),
            // `PunchOffEventEncounter`: two constructs, the first opening on
            // Fast Punch, each down 2 to 9 HP (`Rng.NextInt(2, 10)`).
            PunchOffEventEncounter => [1, 0]
                .into_iter()
                .map(|starter_move| EnemySpec {
                    id: PunchConstruct,
                    flags: Flags { starter_move, hp_reduction: 2 + rng.next_int(8) as u8, ..Flags::default() },
                })
                .collect(),
            MysteriousKnightEventEncounter => vec![one(MysteriousKnight)],
            FakeMerchantEventEncounter => vec![one(FakeMerchantMonster)],
            // The dummy the player's setting picked; the sim rolls the pick.
            BattlewornDummyEventEncounter => vec![one(*rng.pick(&[BattleFriendV1, BattleFriendV2, BattleFriendV3]).unwrap())],
        }
    }
}
