// Combat recorder. Writes one JSONL file per combat under
// <user data>/sts2ai/recordings/, for the simulator's replay suite.
//
// Two sources feed the log:
//   - A per-frame poll that snapshots the full combat state whenever it is
//     the player's turn to act and nothing is resolving. Consecutive
//     identical snapshots are dropped, so each written snapshot is a
//     decision point.
//   - Harmony postfixes on the game's static Hook dispatchers for the
//     events the sim must replay verbatim: card plays, potion uses,
//     shuffles (the resulting order), turn starts, and combat end.

using System.Text.Json;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

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

    public static void Initialize()
    {
        try
        {
            new Harmony("sts2ai").PatchAll();
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
            var cm = CombatManager.Instance;
            if (!cm.IsInProgress) return;
            var sync = RunManager.Instance.ActionQueueSynchronizer;
            if (sync == null || sync.CombatState != ActionSynchronizerCombatState.PlayPhase) return;
            var state = cm.DebugOnlyGetState();
            var me = LocalContext.GetMe(state);
            if (state == null || me == null || cm.IsExecutingCardOrPotionEffect(me)) return;

            string snap = JsonSerializer.Serialize(Snapshot(state, me), Json);
            if (snap == _lastSnapshot) return;
            if (_file == null) Open(state, me);
            _lastSnapshot = snap;
            _lastHand = me.PlayerCombatState!.Hand.Cards.ToList();
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
        GD.Print($"[sts2ai] recording {path}");
        Write(JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["t"] = "start",
            ["encounter"] = encounter,
            ["room"] = state.Encounter.RoomType.ToString(),
            ["ascension"] = run?.AscensionLevel ?? 0,
            ["max_energy"] = me.MaxEnergy,
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

    private static Dictionary<string, object?> CardRef(CardModel c) => new()
    {
        ["id"] = c.Id.Entry,
        ["up"] = c.IsUpgraded,
    };

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

    private static int? EnemyIndex(Creature? target)
    {
        if (target == null || target.IsPlayer) return null;
        var enemies = target.CombatState?.Enemies;
        if (enemies == null) return null;
        for (int i = 0; i < enemies.Count; i++)
            if (ReferenceEquals(enemies[i], target)) return i;
        return null;
    }

    // ---- hook patches -----------------------------------------------------

    [HarmonyPatch(typeof(Hook), nameof(Hook.AfterCardPlayed))]
    private static class AfterCardPlayedPatch
    {
        private static void Postfix(CardPlay cardPlay)
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
    }

    [HarmonyPatch(typeof(Hook), nameof(Hook.AfterPotionUsed))]
    private static class AfterPotionUsedPatch
    {
        private static void Postfix(PotionModel potion, Creature? target)
        {
            Event(new() { ["t"] = "potion", ["id"] = potion.Id.Entry, ["target"] = EnemyIndex(target) });
        }
    }

    [HarmonyPatch(typeof(Hook), nameof(Hook.ModifyShuffleOrder))]
    private static class ShufflePatch
    {
        private static void Postfix(List<CardModel> cards, bool isInitialShuffle)
        {
            // The initial shuffle happens before the file opens; the first
            // snapshot carries that order in its draw pile.
            if (isInitialShuffle) return;
            Event(new() { ["t"] = "shuffle", ["cards"] = cards.Select(CardRef).ToList() });
        }
    }

    [HarmonyPatch(typeof(Hook), nameof(Hook.AfterPlayerTurnStart))]
    private static class TurnStartPatch
    {
        private static void Postfix(Player player)
        {
            Event(new() { ["t"] = "turn_start", ["turn"] = player.PlayerCombatState?.TurnNumber });
        }
    }

    [HarmonyPatch(typeof(Hook), nameof(Hook.AfterCombatEnd))]
    private static class CombatEndPatch
    {
        private static void Postfix(CombatRoom room)
        {
            var me = LocalContext.GetMe(room.CombatState);
            Event(new() { ["t"] = "end", ["won"] = me?.Creature.IsAlive ?? false, ["hp"] = me?.Creature.CurrentHp });
            Close();
        }
    }
}
