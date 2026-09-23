//! Small shared enums and value types. Ported from `Entities/Cards/*.cs`,
//! `ValueProps/ValueProp.cs`, `Entities/Ascension/AscensionLevel.cs`,
//! `Entities/Relics/RelicRarity.cs`.

/// `ValueProps/ValueProp.cs`. Flags on a damage or block amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueProp(u8);

impl ValueProp {
    pub const NONE: ValueProp = ValueProp(0);
    /// HP loss that ignores block.
    pub const UNBLOCKABLE: ValueProp = ValueProp(2);
    /// Relic, potion, or power damage. Strength, Vulnerable etc. ignore it.
    pub const UNPOWERED: ValueProp = ValueProp(4);
    /// Attack-card or monster-move damage / block.
    pub const MOVE: ValueProp = ValueProp(8);

    pub const fn has(self, other: ValueProp) -> bool {
        self.0 & other.0 != 0
    }
    pub const fn or(self, other: ValueProp) -> ValueProp {
        ValueProp(self.0 | other.0)
    }
    /// `ValuePropExtensions.IsPoweredAttack`. Same predicate is used for block.
    pub const fn is_powered(self) -> bool {
        self.has(ValueProp::MOVE) && !self.has(ValueProp::UNPOWERED)
    }
}

/// `Entities/Cards/CardType.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardType {
    Attack,
    Skill,
    Power,
    Status,
    Curse,
}

/// `Entities/Cards/CardRarity.cs`, the members the Ironclad slice meets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardRarity {
    Basic,
    Common,
    Uncommon,
    Rare,
    Special,
}

/// `Entities/Relics/RelicRarity.cs`, less `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelicRarity {
    Starter,
    Common,
    Uncommon,
    Rare,
    Shop,
    Event,
    Ancient,
}

/// `Entities/Cards/TargetType.cs`, single-player subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetType {
    None,
    Self_,
    AnyEnemy,
    AllEnemies,
    RandomEnemy,
    /// Another living player. Alone there is none, so `CardModel.CanPlay`
    /// refuses the card (`NoLivingAllies`) and an auto-play skips it.
    AnyAlly,
    /// Every player on your side, you included.
    AllAllies,
}

/// `Entities/Cards/CardKeyword.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Keyword {
    Exhaust,
    Ethereal,
    Innate,
    Unplayable,
    Retain,
}

/// `Combat/CombatSide.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Player,
    Enemy,
}

/// Identifies a creature in combat. Enemies keep their index for the whole
/// fight; dead enemies stay in the list with 0 HP, as in the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CreatureRef {
    Player,
    Enemy(usize),
}

impl CreatureRef {
    pub fn side(self) -> Side {
        match self {
            CreatureRef::Player => Side::Player,
            CreatureRef::Enemy(_) => Side::Enemy,
        }
    }
}

/// `Entities/Ascension/AscensionLevel.cs`. The ordinal is the level, and every
/// check in the game is "level >= x" (`AscensionManager.HasLevel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AscensionLevel {
    None = 0,
    SwarmingElites = 1,
    WearyTraveler = 2,
    Poverty = 3,
    TightBelt = 4,
    AscendersBane = 5,
    Inflation = 6,
    Scarcity = 7,
    ToughEnemies = 8,
    DeadlyEnemies = 9,
    DoubleBoss = 10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ascension(pub u8);

impl Ascension {
    /// `AscensionManager.HasLevel`.
    pub fn has(self, level: AscensionLevel) -> bool {
        self.0 >= level as u8
    }
    /// `AscensionHelper.GetValueIfAscension(level, ifAscended, otherwise)`.
    pub fn pick<T>(self, level: AscensionLevel, ascended: T, base: T) -> T {
        if self.has(level) {
            ascended
        } else {
            base
        }
    }
}
