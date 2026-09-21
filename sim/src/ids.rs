//! Closed vocabularies. One variant per game model class. Growing these is
//! the main way the sim's scope grows.

/// `Models/Cards/<Name>.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CardId {
    StrikeIronclad,
    DefendIronclad,
    Bash,
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
}

/// `Models/Monsters/<Name>.cs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MonsterId {
    Nibbit,
}
