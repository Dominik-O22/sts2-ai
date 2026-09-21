//! Act 1 (Overgrowth) encounters: `Models/Encounters/*.cs` via
//! `Overgrowth.GenerateAllEncounters`. Each encounter generates its monster
//! list, rolling its own composition where the game does.

use crate::combat::EnemySpec;
use crate::ids::MonsterId;
use crate::monster::Flags;
use crate::rng::Rng;

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
];

fn one(id: MonsterId) -> EnemySpec {
    EnemySpec { id, flags: Flags::default() }
}

impl Encounter {
    pub fn kind(self) -> Kind {
        use Encounter::*;
        match self {
            FuzzyWurmCrawlerWeak | NibbitsWeak | ShrinkerBeetleWeak | SlimesWeak => Kind::Weak,
            BygoneEffigyElite | ByrdonisElite | PhrogParasiteElite => Kind::Elite,
            VantomBoss | CeremonialBeastBoss | TheKinBoss => Kind::Boss,
            _ => Kind::Normal,
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
            FogmogNormal => vec![one(Fogmog)],
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
        }
    }
}
