//! Closed vocabularies. One variant per game model class. Growing these is
//! the main way the sim's scope grows.

/// `Models/Cards/<Name>.cs`. The Ironclad pool, plus the statuses, curses,
/// and tokens the Ironclad slice can meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CardId {
    // Ironclad pool (Models/CardPools/IroncladCardPool.cs), multiplayer-only
    // cards (DemonicShield, Tank) excluded.
    Aggression,
    Anger,
    Armaments,
    AshenStrike,
    Barricade,
    Bash,
    BattleTrance,
    Bloodletting,
    BloodWall,
    Bludgeon,
    BodySlam,
    Brand,
    Break,
    Breakthrough,
    Bully,
    BurningPact,
    Cascade,
    Cinder,
    Colossus,
    Conflagration,
    Corruption,
    CrimsonMantle,
    Cruelty,
    DarkEmbrace,
    DefendIronclad,
    DemonForm,
    Dismantle,
    Dominate,
    DrumOfBattle,
    EvilEye,
    ExpectAFight,
    Feed,
    FeelNoPain,
    FiendFire,
    FightMe,
    FlameBarrier,
    ForgottenRitual,
    Havoc,
    Headbutt,
    Hellraiser,
    Hemokinesis,
    HowlFromBeyond,
    Impervious,
    InfernalBlade,
    Inferno,
    Inflame,
    IronWave,
    Juggernaut,
    Juggling,
    Mangle,
    MoltenFist,
    NotYet,
    Offering,
    OneTwoPunch,
    PactsEnd,
    PerfectedStrike,
    Pillage,
    PommelStrike,
    PrimalForce,
    Pyre,
    Rage,
    Rampage,
    Rupture,
    SecondWind,
    SetupStrike,
    ShrugItOff,
    Spite,
    Stampede,
    Stoke,
    Stomp,
    StoneArmor,
    StrikeIronclad,
    SwordBoomerang,
    Taunt,
    TearAsunder,
    Thrash,
    Thunderclap,
    Tremble,
    TrueGrit,
    TwinStrike,
    Unmovable,
    Unrelenting,
    Uppercut,
    Vicious,
    Whirlwind,
    // Statuses (Models/CardPools/StatusCardPool.cs), the act 1 subset.
    Wound,
    Slimed,
    Dazed,
    Burn,
    Infection,
    // Curses.
    AscendersBane,
    // Tokens.
    GiantRock,
    // Colorless (Models/CardPools/ColorlessCardPool.cs), as met in recordings.
    MindBlast,
    // Appended, never reordered: the embedding rows are indexed by position
    // and `sim/vocab.txt` pins the order.
    /// Soul Fysh status. 6 unblockable HP loss if it ends the turn in hand,
    /// and playable for 1 to be rid of it.
    Beckon,
    // Curses. Events and relics hand these out, so a real deck carries
    // them even though nothing in combat creates one.
    /// 13 unblockable if it ends the turn in hand.
    BadLuck,
    /// Dead card that exhausts itself at end of turn.
    Clumsy,
    /// Dead card. Eternal, which only matters to deck editing.
    CurseOfTheBell,
    /// Costs gold at end of turn, which combat does not track.
    Debt,
    /// 2 blockable damage if it ends the turn in hand.
    Decay,
    /// Weak 1 if it ends the turn in hand.
    Doubt,
    /// Blocks every other card until it is played.
    Enthralled,
    /// Starts in hand and exhausts itself.
    Folly,
    /// Dead card.
    Greed,
    /// Dead card. Its counter runs between combats.
    Guilty,
    /// Dead card.
    Injury,
    /// Blocks card plays once three have happened this turn.
    Normality,
    /// Dead card that will not leave your hand.
    PoorSleep,
    /// Unblockable damage equal to the hand it ends the turn in.
    Regret,
    /// Frail 1 if it ends the turn in hand.
    Shame,
    /// Does nothing; you pay 1 to be rid of it.
    SporeMind,
    /// Starts in hand and stays there.
    Writhe,
    // Relic cards. Appended, never reordered.
    /// Dead status Biiig Hug shuffles in with every reshuffle.
    Soot,
    /// Retained energy token from Radiant Pearl.
    Luminesce,
    // Act 3 (Glory).
    /// Unplayable status that hurts at the end of the turn, harder with every
    /// Increasing Intensity (Aeonglass).
    Wither,
}

/// `Models/Powers/<Name>Power.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PowerId {
    Strength,
    Dexterity,
    Vulnerable,
    Weak,
    Frail,
    /// Byrdonis's Ritual. Grants Strength at end of its side's turn.
    Territorial,
    // Ironclad card powers.
    Aggression,
    NoDraw,
    Barricade,
    Colossus,
    Corruption,
    CrimsonMantle,
    Cruelty,
    DarkEmbrace,
    DemonForm,
    FeelNoPain,
    FlameBarrier,
    Hellraiser,
    Inferno,
    Juggernaut,
    Juggling,
    /// `TemporaryStrengthPower` with `IsPositive => false` (Mangle).
    Mangle,
    NoEnergyGain,
    OneTwoPunch,
    Pyre,
    Rage,
    Rupture,
    /// `TemporaryStrengthPower`, positive (Setup Strike).
    SetupStrike,
    Stampede,
    Plating,
    FreeAttack,
    Vicious,
    Unmovable,
    // Monster-side powers used in act 1.
    /// Caps HP loss at 1 per hit, one charge per hit (Vantom, Inklet).
    Slippery,
    /// Attacks cost 1 more this turn (Vine Shambler).
    Tangled,
    /// Damage taken grows 10% per card played this turn (Bygone Effigy).
    Slow,
    /// Player deals 30% less; `-1` amount means permanent (Shrinker Beetle).
    Shrink,
    /// Only the first card each turn may be played (Ceremonial Beast).
    Ringing,
    /// Stunned and stripped of Strength when HP drops to the amount.
    Plow,
    /// Secondary enemy: its death does not end combat.
    Minion,
    /// On death, spawns four Wrigglers (Phrog Parasite).
    Infested,
    /// Secondary enemy that revives to full HP the turn after dying.
    Illusion,
    /// HP loss at the end of the owner's turn (Slithering Strangler).
    Constrict,
    /// Negates the next debuff (Cubex Construct, Punch Construct).
    Artifact,
    // Relic powers.
    /// Damage back to attackers (Bronze Scales).
    Thorns,
    /// Next attack card deals extra damage, then it is spent (Akabeko).
    Vigor,
    /// Block when block is cleared next turn (Self-Forming Clay).
    SelfFormingClay,
    // Potion powers.
    /// Heal at the end of the owner's turn, then decrement (Regen Potion).
    Regen,
    /// Next attack card deals triple damage (Gigantification Potion).
    Gigantification,
    /// Next card is played twice; gone at end of turn (Duplicator).
    Duplication,
    /// Negates the next HP loss (Lucky Tonic).
    Buffer,
    /// Draw one extra card per turn, decrementing (Clarity).
    Clarity,
    /// Unblockable HP loss at the end of the owner's turn (Powdered Demise).
    Demise,
    /// Energy after each energy reset, decrementing (Radiant Tincture).
    Radiance,
    /// Strength at the end of the owner's turn (Mazaleth's Gift).
    Ritual,
    /// Hand is not discarded at end of turn (Stable Serum).
    RetainHand,
    /// Block once block is cleared next turn (Ship in a Bottle).
    BlockNextTurn,
    /// `TemporaryStrengthPower` from Flex Potion.
    FlexPotion,
    /// `TemporaryStrengthPower`, negative, from Shackling Potion.
    ShacklingPotion,
    /// `TemporaryDexterityPower` from Speed Potion.
    SpeedPotion,
    // Appended, never reordered: see the note on `CardId`.
    // Monster-side powers used in the Underdocks.
    /// Caps damage received at 1, ticking down at the end of the enemy turn
    /// (Soul Fysh).
    Intangible,
    /// Caps the HP the owner can lose per turn at the amount (Skulking Colony).
    HardenedShell,
    /// Block once per turn, the first time a card hit lands (Phantasmal Gardener).
    Skittish,
    /// Strength per attack that landed, on the owner's own attacks (Fossil Stalker).
    Suck,
    /// Strength when a teammate dies, and the owner is stunned that turn
    /// (Corpse Slug).
    Ravenous,
    /// Asleep behind Plating: any unblocked hit wakes Lagavulin Matriarch, and
    /// so does the counter running out.
    Asleep,
    /// Stunned into TERROR_MOVE the first time the owner's HP drops to the
    /// amount (Terror Eel).
    Shriek,
    /// Playing a Skill smogs every Skill in combat until end of turn, making
    /// them unplayable (Living Fog).
    Smoggy,
    /// Pressure the Waterfall Giant explodes for. Its death is deferred until
    /// the blast lands.
    SteamEruption,
    /// Gremlin Merc's death brings a Sneaky and a Fat Gremlin, so it does not
    /// end the combat.
    Surprise,
    /// Gremlin Merc steals gold on every move it makes. `data` banks the
    /// running total, which is what the Fat Gremlin inherits.
    Thievery,
    /// The gold the Fat Gremlin is carrying off, returned if you kill it.
    /// Inert in combat, but the recorder logs it.
    Heist,
    // Relic powers, appended.
    /// Every drawn card costs a random 0 to 3 for the combat (Snecko Eye).
    Confused,
    /// Halves powered attack damage taken until the enemy turn ends
    /// (Diamond Diadem).
    DiamondDiadem,
    // Act 3 (Glory).
    /// Respawns the Axebot with one less stock when it dies.
    Stock,
    /// Power cards are Galvanized; playing one hurts you (Globe Head).
    Galvanic,
    /// Halves powered attack damage taken (Owl Magistrate in flight).
    Soar,
    /// Unblocked attack damage costs you max HP (Scroll of Biting).
    PaperCuts,
    /// Blocks every Turret Operator at the start of your turn (Living Shield).
    Rampart,
    /// Strength it steals returns when it dies (The Lost).
    PossessStrength,
    /// Dexterity it steals returns when it dies (The Forgotten).
    PossessSpeed,
    /// Every card is Hexed, and Hexed cards are Ethereal (Spectral Knight).
    Hex,
    /// Your upgraded cards are downgraded until the caster dies (Magi Knight).
    Dampen,
    /// Strength at the end of the owner's turn (Zapbot).
    HighVoltage,
    /// Binds the first cards drawn each turn; only one Bound card may be
    /// played per turn (Queen).
    ChainsOfBinding,
    /// Revives the Test Subject in a stronger form instead of dying.
    Adaptable,
    /// Strength whenever you play a Skill (Test Subject).
    Enrage,
    /// A Wound for every unblocked hit of its attacks (Test Subject).
    PainfulStabs,
    /// Intangible every other turn (Test Subject).
    Nemesis,
    /// A Wither into your hand every sixth card you play (Aeonglass).
    WitheringPresence,
}

/// `Models/Monsters/<Name>.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MonsterId {
    Nibbit,
    FuzzyWurmCrawler,
    ShrinkerBeetle,
    LeafSlimeS,
    LeafSlimeM,
    TwigSlimeS,
    TwigSlimeM,
    Inklet,
    Mawler,
    Fogmog,
    EyeWithTeeth,
    Flyconid,
    SnappingJaxfruit,
    SlitheringStrangler,
    VineShambler,
    CubexConstruct,
    AxeRubyRaider,
    AssassinRubyRaider,
    BruteRubyRaider,
    CrossbowRubyRaider,
    TrackerRubyRaider,
    Byrdonis,
    BygoneEffigy,
    PhrogParasite,
    Wriggler,
    Vantom,
    CeremonialBeast,
    KinFollower,
    KinPriest,
    // Underdocks (Acts/Underdocks.cs).
    Toadpole,
    CorpseSlug,
    DampCultist,
    CalcifiedCultist,
    FossilStalker,
    GremlinMerc,
    FatGremlin,
    SneakyGremlin,
    GasBomb,
    HauntedShip,
    LivingFog,
    PhantasmalGardener,
    PunchConstruct,
    Seapunk,
    SewerClam,
    SkulkingColony,
    SludgeSpinner,
    TwoTailedRat,
    TerrorEel,
    LagavulinMatriarch,
    SoulFysh,
    WaterfallGiant,
    // Act 3 (Glory).
    Axebot,
    DevotedSculptor,
    FrogKnight,
    GlobeHead,
    OwlMagistrate,
    ScrollOfBiting,
    SlimedBerserker,
    LivingShield,
    TurretOperator,
    TheLost,
    TheForgotten,
    MechaKnight,
    FlailKnight,
    SpectralKnight,
    MagiKnight,
    SoulNexus,
    Fabricator,
    Zapbot,
    Stabbot,
    Guardbot,
    Noisebot,
    Queen,
    TorchHeadAmalgam,
    TestSubject,
    Aeonglass,
}

/// Every variant, for id lookups by name.
pub const ALL_CARDS: &[CardId] = &[
    CardId::Aggression,
    CardId::Anger,
    CardId::Armaments,
    CardId::AshenStrike,
    CardId::Barricade,
    CardId::Bash,
    CardId::BattleTrance,
    CardId::Bloodletting,
    CardId::BloodWall,
    CardId::Bludgeon,
    CardId::BodySlam,
    CardId::Brand,
    CardId::Break,
    CardId::Breakthrough,
    CardId::Bully,
    CardId::BurningPact,
    CardId::Cascade,
    CardId::Cinder,
    CardId::Colossus,
    CardId::Conflagration,
    CardId::Corruption,
    CardId::CrimsonMantle,
    CardId::Cruelty,
    CardId::DarkEmbrace,
    CardId::DefendIronclad,
    CardId::DemonForm,
    CardId::Dismantle,
    CardId::Dominate,
    CardId::DrumOfBattle,
    CardId::EvilEye,
    CardId::ExpectAFight,
    CardId::Feed,
    CardId::FeelNoPain,
    CardId::FiendFire,
    CardId::FightMe,
    CardId::FlameBarrier,
    CardId::ForgottenRitual,
    CardId::Havoc,
    CardId::Headbutt,
    CardId::Hellraiser,
    CardId::Hemokinesis,
    CardId::HowlFromBeyond,
    CardId::Impervious,
    CardId::InfernalBlade,
    CardId::Inferno,
    CardId::Inflame,
    CardId::IronWave,
    CardId::Juggernaut,
    CardId::Juggling,
    CardId::Mangle,
    CardId::MoltenFist,
    CardId::NotYet,
    CardId::Offering,
    CardId::OneTwoPunch,
    CardId::PactsEnd,
    CardId::PerfectedStrike,
    CardId::Pillage,
    CardId::PommelStrike,
    CardId::PrimalForce,
    CardId::Pyre,
    CardId::Rage,
    CardId::Rampage,
    CardId::Rupture,
    CardId::SecondWind,
    CardId::SetupStrike,
    CardId::ShrugItOff,
    CardId::Spite,
    CardId::Stampede,
    CardId::Stoke,
    CardId::Stomp,
    CardId::StoneArmor,
    CardId::StrikeIronclad,
    CardId::SwordBoomerang,
    CardId::Taunt,
    CardId::TearAsunder,
    CardId::Thrash,
    CardId::Thunderclap,
    CardId::Tremble,
    CardId::TrueGrit,
    CardId::TwinStrike,
    CardId::Unmovable,
    CardId::Unrelenting,
    CardId::Uppercut,
    CardId::Vicious,
    CardId::Whirlwind,
    CardId::Wound,
    CardId::Slimed,
    CardId::Dazed,
    CardId::Burn,
    CardId::Infection,
    CardId::AscendersBane,
    CardId::GiantRock,
    CardId::MindBlast,
    CardId::Beckon,
    CardId::BadLuck,
    CardId::Clumsy,
    CardId::CurseOfTheBell,
    CardId::Debt,
    CardId::Decay,
    CardId::Doubt,
    CardId::Enthralled,
    CardId::Folly,
    CardId::Greed,
    CardId::Guilty,
    CardId::Injury,
    CardId::Normality,
    CardId::PoorSleep,
    CardId::Regret,
    CardId::Shame,
    CardId::SporeMind,
    CardId::Writhe,
    CardId::Soot,
    CardId::Luminesce,
    CardId::Wither,
];

pub const ALL_POWERS: &[PowerId] = &[
    PowerId::Strength,
    PowerId::Dexterity,
    PowerId::Vulnerable,
    PowerId::Weak,
    PowerId::Frail,
    PowerId::Territorial,
    PowerId::Aggression,
    PowerId::NoDraw,
    PowerId::Barricade,
    PowerId::Colossus,
    PowerId::Corruption,
    PowerId::CrimsonMantle,
    PowerId::Cruelty,
    PowerId::DarkEmbrace,
    PowerId::DemonForm,
    PowerId::FeelNoPain,
    PowerId::FlameBarrier,
    PowerId::Hellraiser,
    PowerId::Inferno,
    PowerId::Juggernaut,
    PowerId::Juggling,
    PowerId::Mangle,
    PowerId::NoEnergyGain,
    PowerId::OneTwoPunch,
    PowerId::Pyre,
    PowerId::Rage,
    PowerId::Rupture,
    PowerId::SetupStrike,
    PowerId::Stampede,
    PowerId::Plating,
    PowerId::FreeAttack,
    PowerId::Vicious,
    PowerId::Unmovable,
    PowerId::Slippery,
    PowerId::Tangled,
    PowerId::Slow,
    PowerId::Shrink,
    PowerId::Ringing,
    PowerId::Plow,
    PowerId::Minion,
    PowerId::Infested,
    PowerId::Illusion,
    PowerId::Constrict,
    PowerId::Artifact,
    PowerId::Thorns,
    PowerId::Vigor,
    PowerId::SelfFormingClay,
    PowerId::Regen,
    PowerId::Gigantification,
    PowerId::Duplication,
    PowerId::Buffer,
    PowerId::Clarity,
    PowerId::Demise,
    PowerId::Radiance,
    PowerId::Ritual,
    PowerId::RetainHand,
    PowerId::BlockNextTurn,
    PowerId::FlexPotion,
    PowerId::ShacklingPotion,
    PowerId::SpeedPotion,
    PowerId::Intangible,
    PowerId::HardenedShell,
    PowerId::Skittish,
    PowerId::Suck,
    PowerId::Ravenous,
    PowerId::Asleep,
    PowerId::Shriek,
    PowerId::Smoggy,
    PowerId::SteamEruption,
    PowerId::Surprise,
    PowerId::Thievery,
    PowerId::Heist,
    PowerId::Confused,
    PowerId::DiamondDiadem,
    PowerId::Stock,
    PowerId::Galvanic,
    PowerId::Soar,
    PowerId::PaperCuts,
    PowerId::Rampart,
    PowerId::PossessStrength,
    PowerId::PossessSpeed,
    PowerId::Hex,
    PowerId::Dampen,
    PowerId::HighVoltage,
    PowerId::ChainsOfBinding,
    PowerId::Adaptable,
    PowerId::Enrage,
    PowerId::PainfulStabs,
    PowerId::Nemesis,
    PowerId::WitheringPresence,
];

pub const ALL_MONSTERS: &[MonsterId] = &[
    MonsterId::Nibbit,
    MonsterId::FuzzyWurmCrawler,
    MonsterId::ShrinkerBeetle,
    MonsterId::LeafSlimeS,
    MonsterId::LeafSlimeM,
    MonsterId::TwigSlimeS,
    MonsterId::TwigSlimeM,
    MonsterId::Inklet,
    MonsterId::Mawler,
    MonsterId::Fogmog,
    MonsterId::EyeWithTeeth,
    MonsterId::Flyconid,
    MonsterId::SnappingJaxfruit,
    MonsterId::SlitheringStrangler,
    MonsterId::VineShambler,
    MonsterId::CubexConstruct,
    MonsterId::AxeRubyRaider,
    MonsterId::AssassinRubyRaider,
    MonsterId::BruteRubyRaider,
    MonsterId::CrossbowRubyRaider,
    MonsterId::TrackerRubyRaider,
    MonsterId::Byrdonis,
    MonsterId::BygoneEffigy,
    MonsterId::PhrogParasite,
    MonsterId::Wriggler,
    MonsterId::Vantom,
    MonsterId::CeremonialBeast,
    MonsterId::KinFollower,
    MonsterId::KinPriest,
    MonsterId::Toadpole,
    MonsterId::CorpseSlug,
    MonsterId::DampCultist,
    MonsterId::CalcifiedCultist,
    MonsterId::FossilStalker,
    MonsterId::GremlinMerc,
    MonsterId::FatGremlin,
    MonsterId::SneakyGremlin,
    MonsterId::GasBomb,
    MonsterId::HauntedShip,
    MonsterId::LivingFog,
    MonsterId::PhantasmalGardener,
    MonsterId::PunchConstruct,
    MonsterId::Seapunk,
    MonsterId::SewerClam,
    MonsterId::SkulkingColony,
    MonsterId::SludgeSpinner,
    MonsterId::TwoTailedRat,
    MonsterId::TerrorEel,
    MonsterId::LagavulinMatriarch,
    MonsterId::SoulFysh,
    MonsterId::WaterfallGiant,
    MonsterId::Axebot,
    MonsterId::DevotedSculptor,
    MonsterId::FrogKnight,
    MonsterId::GlobeHead,
    MonsterId::OwlMagistrate,
    MonsterId::ScrollOfBiting,
    MonsterId::SlimedBerserker,
    MonsterId::LivingShield,
    MonsterId::TurretOperator,
    MonsterId::TheLost,
    MonsterId::TheForgotten,
    MonsterId::MechaKnight,
    MonsterId::FlailKnight,
    MonsterId::SpectralKnight,
    MonsterId::MagiKnight,
    MonsterId::SoulNexus,
    MonsterId::Fabricator,
    MonsterId::Zapbot,
    MonsterId::Stabbot,
    MonsterId::Guardbot,
    MonsterId::Noisebot,
    MonsterId::Queen,
    MonsterId::TorchHeadAmalgam,
    MonsterId::TestSubject,
    MonsterId::Aeonglass,
];
