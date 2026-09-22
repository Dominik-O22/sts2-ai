// Combat recorder. Writes one JSONL file per combat under
// <user data>/sts2ai/recordings/, for the simulator's replay suite.
//
// Two sources feed the log:
//   - A per-frame poll that snapshots the full combat state whenever it is
//     the player's turn to act and nothing is resolving. Consecutive
//     identical snapshots are dropped, so each written snapshot is a
//     decision point.
//   - A model subscribed through ModHelper.SubscribeForCombatStateHooks,
//     which receives the same hooks as relics and powers, for the events
//     the sim must replay verbatim: card plays, potion uses, shuffles (the
//     resulting order), turn starts, and combat end. No Harmony: its native
//     patch helper does not load on every Linux setup.

using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.ValueProps;

namespace Sts2Ai;

[ModInitializer(nameof(Initialize))]
public static class Recorder
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };
    private static StreamWriter? _file;
    private static string _lastSnapshot = "";
    /// Hand at the last snapshot, so a play can be logged with its hand index
    /// (the card has already left the hand when the hook fires).
    private static List<CardModel> _lastHand = new();
    /// Enemies at the last snapshot; a killed target has already left the
    /// combat state when the play hook fires.
    private static List<Creature> _lastEnemies = new();
    /// Potion slots when the last potion event was logged (or the file
    /// opened). A slot emptied since then means a potion is in flight: it
    /// leaves the belt a frame before its effect runs.
    private static List<PotionModel?> _potionBaseline = new();

    public static void Initialize()
    {
        try
        {
            var model = new RecorderModel();
            ModHelper.SubscribeForCombatStateHooks("sts2ai", _ => new[] { model });
            // The end-of-combat hook is not dispatched to mod subscribers
            // once the combat state is gone; the event fires regardless.
            CombatManager.Instance.CombatEnded += room => Guard(() => OnCombatEnd(room));
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.Connect(SceneTree.SignalName.ProcessFrame, Callable.From(Poll));
            GD.Print("[sts2ai] recorder ready");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[sts2ai] init failed: {ex}");
        }
    }

    // ---- polling --------------------------------------------------------

    private static void Poll()
    {
        try
        {
            Commands.Poll();
            var cm = CombatManager.Instance;
            if (!cm.IsInProgress) return;
            var sync = RunManager.Instance.ActionQueueSynchronizer;
            if (sync == null || sync.CombatState != ActionSynchronizerCombatState.PlayPhase) return;
            var state = cm.DebugOnlyGetState();
            var me = LocalContext.GetMe(state);
            if (state == null || me == null || cm.IsExecutingCardOrPotionEffect(me)) return;
            // A played card sits in the play pile until its result-pile move; not a decision point yet.
            if (me.PlayerCombatState == null || me.PlayerCombatState.PlayPile.Cards.Count > 0) return;
            if (_file != null && PotionInFlight(me)) return;

            string snap = JsonSerializer.Serialize(Snapshot(state, me), Json);
            if (snap == _lastSnapshot) return;
            if (_file == null) Open(state, me);
            _lastSnapshot = snap;
            _lastHand = me.PlayerCombatState!.Hand.Cards.ToList();
            _lastEnemies = state.Enemies.ToList();
            Write(snap);
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[sts2ai] poll failed: {ex.Message}");
        }
    }

    private static void Open(CombatState state, Player me)
    {
        var run = RunManager.Instance.DebugOnlyGetState();
        string encounter = state.Encounter!.Id.Entry;
        string dir = Path.Combine(OS.GetUserDataDir(), "sts2ai", "recordings");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{DateTime.Now:yyyyMMdd-HHmmss}-{encounter}.jsonl");
        _file = new StreamWriter(path, false);
        _lastSnapshot = "";
        _potionBaseline = me.PotionSlots.ToList();
        GD.Print($"[sts2ai] recording {path}");
        Write(JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["t"] = "start",
            ["encounter"] = encounter,
            ["room"] = state.Encounter.RoomType.ToString(),
            ["ascension"] = run?.AscensionLevel ?? 0,
            ["max_energy"] = me.MaxEnergy,
            // Gremlin Merc steals it and hands what it took to the Fat
            // Gremlin as Heist, which shows up in the enemy powers.
            ["gold"] = me.Gold,
            ["deck"] = me.Deck.Cards.Select(CardRef).ToList(),
            ["relics"] = me.Relics.Select(r => r.Id.Entry).ToList(),
            ["potions"] = me.PotionSlots.Select(p => p?.Id.Entry).ToList(),
            ["enemies"] = state.Enemies.Select(e => new Dictionary<string, object?>
            {
                ["id"] = e.Monster?.Id.Entry,
                ["hp"] = e.CurrentHp,
                ["max_hp"] = e.MaxHp,
            }).ToList(),
        }, Json));
    }

    private static void Close()
    {
        _file?.Flush();
        _file?.Dispose();
        _file = null;
        _lastSnapshot = "";
    }

    private static void Write(string line)
    {
        if (_file == null) return;
        _file.WriteLine(line);
        _file.Flush();
    }

    private static void Event(Dictionary<string, object?> e)
    {
        if (_file == null) return;
        Write(JsonSerializer.Serialize(e, Json));
    }

    // ---- state ------------------------------------------------------------

    // An enchantment rides along with the card, so every pile logs it: the
    // sim folds Sharp and Nimble into damage and block before any power or
    // relic sees the number, and cannot infer either from the card alone.
    private static Dictionary<string, object?> CardRef(CardModel c)
    {
        var d = new Dictionary<string, object?>
        {
            ["id"] = c.Id.Entry,
            ["up"] = c.IsUpgraded,
        };
        if (c.Enchantment is { } e)
        {
            d["ench"] = new object?[] { e.Id.Entry, e.Amount, e.Status == EnchantmentStatus.Disabled };
        }
        return d;
    }

    private static Dictionary<string, object?> HandCard(CardModel c)
    {
        var d = CardRef(c);
        d["cost"] = c.EnergyCost.CostsX ? -1 : c.EnergyCost.GetWithModifiers(CostModifiers.All);
        return d;
    }

    private static List<object?[]> Powers(Creature c) =>
        c.Powers.Select(p => new object?[] { p.Id.Entry, p.Amount }).ToList();

    private static Dictionary<string, object?> Snapshot(CombatState state, Player me)
    {
        var pcs = me.PlayerCombatState!;
        var c = me.Creature;
        return new Dictionary<string, object?>
        {
            ["t"] = "snapshot",
            ["turn"] = pcs.TurnNumber,
            ["round"] = state.RoundNumber,
            ["hp"] = c.CurrentHp,
            ["max_hp"] = c.MaxHp,
            ["block"] = c.Block,
            ["energy"] = pcs.Energy,
            ["powers"] = Powers(c),
            ["hand"] = pcs.Hand.Cards.Select(HandCard).ToList(),
            ["draw"] = pcs.DrawPile.Cards.Select(CardRef).ToList(),
            ["discard"] = pcs.DiscardPile.Cards.Select(CardRef).ToList(),
            ["exhaust"] = pcs.ExhaustPile.Cards.Select(CardRef).ToList(),
            ["potions"] = me.PotionSlots.Select(p => p?.Id.Entry).ToList(),
            ["enemies"] = state.Enemies.Select(e => new Dictionary<string, object?>
            {
                ["id"] = e.Monster?.Id.Entry,
                ["hp"] = e.CurrentHp,
                ["max_hp"] = e.MaxHp,
                ["block"] = e.Block,
                ["powers"] = Powers(e),
                ["move"] = e.Monster?.NextMove.Id,
            }).ToList(),
        };
    }

    private static bool PotionInFlight(Player me)
    {
        var now = me.PotionSlots;
        for (int i = 0; i < _potionBaseline.Count && i < now.Count; i++)
            if (_potionBaseline[i] != null && now[i] == null) return true;
        return false;
    }

    internal static void Guard(Action a)
    {
        try { a(); }
        catch (Exception ex) { GD.PrintErr($"[sts2ai] hook failed: {ex.Message}"); }
    }

    private static int? EnemyIndex(Creature? target)
    {
        if (target == null || target.IsPlayer) return null;
        int i = _lastEnemies.FindIndex(e => ReferenceEquals(e, target));
        return i < 0 ? null : i;
    }

    // ---- hooks --------------------------------------------------------------

    internal static void OnCardPlayed(CardPlay cardPlay)
    {
        if (cardPlay.IsAutoPlay || cardPlay.PlayIndex != 0) return;
        int idx = _lastHand.FindIndex(c => ReferenceEquals(c, cardPlay.Card));
        Event(new()
        {
            ["t"] = "play",
            ["id"] = cardPlay.Card.Id.Entry,
            ["up"] = cardPlay.Card.IsUpgraded,
            ["hand_idx"] = idx < 0 ? null : idx,
            ["target"] = EnemyIndex(cardPlay.Target),
        });
    }

    /// Damage the player dealt to an enemy, so the replay can script random
    /// targets (Sword Boomerang). Index is into the live enemy list.
    internal static void OnDamage(Creature target, DamageResult result, Creature? dealer, CardModel? card)
    {
        if (dealer == null || !dealer.IsPlayer || target.IsPlayer) return;
        var enemies = target.CombatState?.Enemies;
        if (enemies == null) return;
        int idx = -1;
        for (int i = 0; i < enemies.Count; i++)
            if (ReferenceEquals(enemies[i], target)) { idx = i; break; }
        if (idx < 0) return;
        Event(new() { ["t"] = "hit", ["target"] = idx, ["amount"] = result.TotalDamage, ["card"] = card?.Id.Entry });
    }

    /// A card created mid-combat (Infernal Blade), so random generation can be scripted.
    internal static void OnGenerated(CardModel card)
    {
        Event(new() { ["t"] = "gen", ["id"] = card.Id.Entry, ["up"] = card.IsUpgraded });
    }

    /// Any exhaust, with the hand index at the last snapshot when known, so
    /// random exhausts (Thrash) can be scripted.
    internal static void OnExhausted(CardModel card, bool ethereal)
    {
        int idx = _lastHand.FindIndex(c => ReferenceEquals(c, card));
        Event(new() { ["t"] = "exhaust", ["id"] = card.Id.Entry, ["up"] = card.IsUpgraded, ["hand_idx"] = idx < 0 ? null : idx, ["ethereal"] = ethereal });
    }

    internal static void OnPotionUsed(PotionModel potion, Creature? target)
    {
        Event(new() { ["t"] = "potion", ["id"] = potion.Id.Entry, ["target"] = EnemyIndex(target) });
        _potionBaseline = potion.Owner.PotionSlots.ToList();
    }

    internal static void OnShuffle(List<CardModel> cards, bool isInitialShuffle)
    {
        // The initial shuffle happens before the file opens; the first
        // snapshot carries that order in its draw pile.
        if (isInitialShuffle) return;
        Event(new() { ["t"] = "shuffle", ["cards"] = cards.Select(CardRef).ToList() });
    }

    internal static void OnTurnStart(Player player)
    {
        Event(new() { ["t"] = "turn_start", ["turn"] = player.PlayerCombatState?.TurnNumber });
    }

    internal static void OnCombatEnd(CombatRoom room)
    {
        if (_file == null) return;
        var me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        Event(new() { ["t"] = "end", ["won"] = me?.Creature.IsAlive ?? false, ["hp"] = me?.Creature.CurrentHp });
        Close();
    }
}

/// The hook listener. Registered once; the game calls these after its own
/// listeners for every combat.
public sealed class RecorderModel : AbstractModel
{
    public override bool ShouldReceiveCombatHooks => true;

    public override Task AfterCardPlayed(PlayerChoiceContext choiceContext, CardPlay cardPlay)
    {
        Guard(() => Recorder.OnCardPlayed(cardPlay));
        return Task.CompletedTask;
    }

    public override Task AfterDamageReceived(PlayerChoiceContext choiceContext, Creature target, DamageResult result, ValueProp props, Creature? dealer, CardModel? cardSource)
    {
        Guard(() => Recorder.OnDamage(target, result, dealer, cardSource));
        return Task.CompletedTask;
    }

    public override Task AfterCardGeneratedForCombat(CardModel card, Player? creator)
    {
        Guard(() => Recorder.OnGenerated(card));
        return Task.CompletedTask;
    }

    public override Task AfterCardExhausted(PlayerChoiceContext choiceContext, CardModel card, bool causedByEthereal)
    {
        Guard(() => Recorder.OnExhausted(card, causedByEthereal));
        return Task.CompletedTask;
    }

    public override Task AfterPotionUsed(PotionModel potion, Creature? target)
    {
        Guard(() => Recorder.OnPotionUsed(potion, target));
        return Task.CompletedTask;
    }

    public override void ModifyShuffleOrder(Player player, List<CardModel> cards, bool isInitialShuffle)
    {
        Guard(() => Recorder.OnShuffle(cards, isInitialShuffle));
    }

    public override Task AfterPlayerTurnStart(PlayerChoiceContext choiceContext, Player player)
    {
        Guard(() => Recorder.OnTurnStart(player));
        return Task.CompletedTask;
    }

    private static void Guard(Action a) => Recorder.Guard(a);
}
