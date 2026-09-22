//! The effect queue.
//!
//! The game resolves effects as nested async calls: a card's `OnPlay` awaits
//! `DamageCmd`, which awaits hooks, and so on, depth first in source order.
//! We can't suspend that call stack in Rust for player choices, so effects
//! are data in a `VecDeque`. When resolving an effect produces sub-effects,
//! they are pushed to the *front*, in order, which yields the same depth-first
//! order the call stack would. A choice effect leaves the rest of the queue
//! waiting for the answer.

use crate::ids::{CardId, MonsterId, PowerId};
use crate::monster::Flags;
use crate::types::{CardType, CreatureRef, Side, ValueProp};

#[derive(Clone, Debug, PartialEq)]
pub enum AttackTargets {
    One(CreatureRef),
    /// `TargetingAllOpponents`. Every living creature on the other side.
    AllOpponents,
    /// `TargetingRandomOpponents`. Re-rolled per hit.
    RandomOpponent,
}

/// Where a generated or moved card goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pile {
    Hand,
    DrawTop,
    DrawBottom,
    /// `CardPilePosition.Random`: shuffled in anywhere (Soul Fysh's Beckon).
    DrawRandom,
    Discard,
    Exhaust,
}

/// Which cards a hand filter keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardFilter {
    Any,
    Type(CardType),
    NotType(CardType),
    /// Playable attacks (Stampede).
    PlayableAttack,
    /// Cards whose cost is above zero (Touch of Insanity).
    CostsEnergy,
}

/// What to do with the card the player picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Then {
    Exhaust,
    Upgrade,
    MoveTo(Pile),
    /// Cost 0 for the rest of combat (Touch of Insanity).
    FreeThisCombat,
    /// To hand at cost 0 this turn (Liquid Memories).
    ToHandFreeThisTurn,
    /// Take one card from the offer pile into hand, free this turn; the
    /// rest of the offer vanishes (Attack/Skill/Power Potion).
    TakeOffer,
    /// Exhaust it and ask again (Ashwater).
    ExhaustMany,
    /// Discard it and ask again; on skip, draw one per discarded card
    /// (Gambler's Brew).
    DiscardThenDraw { picked: u32 },
}

/// What a random generator draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenPool {
    /// Any Ironclad card that can be generated in combat.
    Ironclad,
    /// Ironclad attacks only.
    IroncladAttacks,
    IroncladSkills,
    IroncladPowers,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// `Commands/Builders/AttackCommand.cs`. One or more hits of `base`
    /// damage, each resolved through `Damage`.
    Attack {
        dealer: CreatureRef,
        base: f64,
        hits: u32,
        targets: AttackTargets,
        props: ValueProp,
        /// Uid of the card that dealt it, if any.
        card: Option<u32>,
    },
    /// `CreatureCmd.Damage` for a single target.
    Damage {
        target: CreatureRef,
        amount: f64,
        props: ValueProp,
        dealer: Option<CreatureRef>,
        card: Option<u32>,
    },
    /// `CreatureCmd.Damage` against every living enemy (Inferno).
    DamageAllEnemies { amount: f64, props: ValueProp, dealer: CreatureRef },
    /// `CreatureCmd.GainBlock`.
    GainBlock {
        target: CreatureRef,
        amount: f64,
        props: ValueProp,
        card: Option<u32>,
    },
    /// `CreatureCmd.Heal`.
    Heal { target: CreatureRef, amount: f64 },
    /// `CreatureCmd.GainMaxHp`.
    GainMaxHp { target: CreatureRef, amount: i32 },
    /// `PlayerCmd.GainEnergy`.
    GainEnergy { amount: i32 },
    /// `PlayerCmd.LoseEnergy`.
    LoseEnergy { amount: i32 },
    /// `PowerCmd.Apply`.
    ApplyPower {
        target: CreatureRef,
        id: PowerId,
        amount: i32,
        applier: Option<CreatureRef>,
    },
    /// `PowerCmd.Remove`.
    RemovePower { target: CreatureRef, id: PowerId },
    /// `PowerCmd.TickDownDuration`.
    TickDuration { target: CreatureRef, id: PowerId },
    /// `PowerCmd.Decrement`.
    DecrementPower { target: CreatureRef, id: PowerId },
    /// `CardPileCmd.Draw(count)`. `from_hand_draw` marks the turn-start draw.
    Draw { count: u32, from_hand_draw: bool },
    /// `CardCmd.Exhaust`.
    Exhaust { uid: u32, ethereal: bool },
    /// Exhaust every hand card matching the filter (FiendFire, Stoke, SecondWind).
    ExhaustHand { filter: CardFilter },
    /// Exhaust one random hand card matching the filter (Cinder, TrueGrit, Thrash).
    ExhaustRandomFromHand { filter: CardFilter, then_add_damage_to: Option<u32> },
    /// Move a card between piles (`CardPileCmd.Add`).
    MoveCard { uid: u32, to: Pile },
    /// `CardCmd.Upgrade`.
    Upgrade { uid: u32 },
    /// Upgrade every upgradable card in hand (Armaments+).
    UpgradeHand,
    /// `CardCmd.Transform` every hand card matching the filter into `into`.
    TransformHand { filter: CardFilter, into: CardId, upgraded: bool },
    /// Put a fresh card into combat (`AddGeneratedCardToCombat`, `CreateClone`).
    GenerateCard { id: CardId, upgraded: bool, to: Pile, free_this_turn: bool },
    /// Random cards from a pool (`CardFactory.GetForCombat` / `GetDistinctForCombat`).
    GenerateRandom { pool: GenPool, count: u32, to: Pile, free_this_turn: bool, distinct: bool },
    /// Ask the player to pick one card from those matching the filter in the
    /// given pile, then do `then` with it. Empty option lists are skipped.
    /// `can_skip` adds `Action::Skip` to the choice.
    Choose { from: Pile, filter: CardFilter, then: Then, can_skip: bool },
    /// Generate `count` distinct cards into the offer pile and let the player
    /// take one (`CardSelectCmd.FromChooseACardScreen`).
    OfferRandom { pool: GenPool, count: u32 },
    /// `Then::TakeOffer` for the picked uid.
    TakeOffer { uid: u32 },
    /// `CardPileCmd.Shuffle`: discard and draw piles merged and shuffled.
    Shuffle,
    /// Snecko Oil: every non-X hand card costs a random 0-3 this turn.
    SneckoCosts,
    /// Soldier's Stew: every Strike in combat gains one replay.
    StrikeReplay,
    /// `Hook.AfterPotionUsed` plus the empty-hand check.
    AfterPotionUsed,
    /// Card play pipeline, `CardModel.OnPlayWrapper`. Energy is already spent;
    /// `paid` is how much (Intimidating Helmet reads it).
    PlayCard { uid: u32, target: Option<CreatureRef>, paid: i32 },
    /// One iteration of the play loop (`GeneratePlayCount` may make several).
    CardPlayIter { uid: u32, target: Option<CreatureRef>, paid: i32 },
    /// `EnchantmentModel.OnPlay`, after the card's own effects.
    EnchantOnPlay { uid: u32, target: Option<CreatureRef> },
    /// `Hook.AfterCardPlayed` for one iteration.
    AfterCardPlayed { uid: u32 },
    /// Tail of the pipeline: move the card to its result pile.
    FinishCardPlay { uid: u32 },
    /// A card's continuation after earlier effects resolved (see `Card::step`).
    CardStep { uid: u32, target: Option<CreatureRef>, step: u8 },
    /// `CardCmd.AutoPlay`: play a card from wherever it is, no energy cost,
    /// random target if needed.
    AutoPlay { uid: u32, force_exhaust: bool },
    /// `CardPileCmd.AutoPlayFromDrawPile`: move `count` cards from the top
    /// to the play pile first, then auto-play them in order.
    AutoPlayFromDrawTop { count: u32, force_exhaust: bool },
    /// Auto-play `count` random playable attacks from hand (Stampede).
    AutoPlayRandomAttack,
    /// Aggression: move up to `count` random attacks from discard to hand, upgraded.
    AggressionPull { count: u32 },

    /// `CreatureCmd.Add`: a monster joins the enemy side mid-combat.
    SpawnMonster { id: MonsterId, flags: Flags },
    /// `CreatureCmd.Stun`: replace the next move with a STUNNED move.
    Stun { target: CreatureRef, next: Option<&'static str> },
    /// Plow: strip every Strength-type power from the target.
    RemoveStrength { target: CreatureRef },
    /// Illusion's REVIVE move: heal to full.
    Revive { target: CreatureRef },
    /// `CreatureCmd.Kill`: the blast monsters set off as they land it.
    Kill { target: CreatureRef },
    /// `CreatureCmd.Escape`: out of the fight without dying (Fat Gremlin).
    Escape { target: CreatureRef },
    /// `ThieveryPower.Steal`, after each of the thief's moves.
    Steal { thief: CreatureRef },
    /// The enemy turn's cleanup, after its `BeforeSideTurnEnd` hooks.
    FinishEnemyTurn,
    /// `BeforeSideTurnEndEarly`, collected only once the `VeryEarly` pass has
    /// finished, so the two see each other's work.
    SideTurnEndEarly(Side),
    /// `AttackCommand`'s `AfterAttack`, once every hit of one attack has
    /// landed: Skittish curls up after the whole card, not between its hits.
    AfterAttack,
    /// `WaterfallGiant.AboutToBlowMove`: bank the Steam Eruption amount into
    /// the blast that follows, then drop the power.
    ArmSteamEruption { target: CreatureRef },

    // Turn flow. `Combat/CombatManager.cs`.
    StartTurn(Side),
    EndPlayerTurn,
    /// `DoTurnEnd`: ethereal exhausts and turn-end-in-hand card effects.
    TurnEndInHand,
    /// `FlushPlayerHand` and the switch to the enemy side.
    FlushHand,
    EnemyAct(usize),
    EndEnemyTurn,
}
