// Localhost bridge for a Python player (`sts2ai.play`). One client at a
// time on 127.0.0.1:47474 (STS2AI_PORT overrides), JSON lines both ways.
//
// Out: every record the recorder writes for the current combat (a client
// joining mid-fight gets the fight so far), plus
//   {"t":"ready"}                  play phase, settled, waiting on a command
//   {"t":"select","min","max","picked"}  a card selection waits on a pick
//   {"t":"error","msg","wait"}     the last command was refused; with
//                                  `wait` the game was busy, so wait for
//                                  the next `ready`, else it is still waiting
// In:
//   {"cmd":"play","card":{"id","up","cost","ench"},"target":k}
//   {"cmd":"potion","slot":s,"id":"...","target":k}
//   {"cmd":"end"}
//   {"cmd":"pick","card":{"id","up"}}   add a card to the open selection
//   {"cmd":"done"}                      close it with what was picked
//   {"cmd":"manual"}                    show it to the player instead
// `target` indexes the living enemies, as in the recording. The sim writes
// these commands (`sim::replay::command`).
//
// While a client is connected and a combat runs, the bridge is the game's
// card selector (`CardSelectCmd.PushSelector`, the hook tests and AutoSlay
// use), so every in-combat card choice comes here instead of to a screen.
// Outside combat it steps aside and the player picks rewards as usual.

using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Ai;

public static class Bridge
{
    private const int DefaultPort = 47474;
    /// Frames the decision point must hold still before `ready` goes out:
    /// the recorder also polls between a card's effects, and a command sent
    /// then would land mid-resolution.
    private const int SettleFrames = 6;
    /// A command that changed nothing visible (none should) would otherwise
    /// leave the client waiting forever; after this many settled frames the
    /// bridge says `ready` again.
    private const int StaleFrames = 300;

    private static readonly ConcurrentQueue<string> Inbox = new();
    private static readonly object WriteLock = new();
    private static TcpClient? _client;
    private static StreamWriter? _out;
    /// The current combat's records, replayed to a client that connects mid-fight.
    private static readonly List<string> Fight = new();

    private static readonly BridgeSelector Selector = new();
    private static IDisposable? _selectorScope;
    private static Selection? _selection;

    private static string? _settledSnap;
    private static ulong _settledFrame;
    private static int _stable;
    private static bool _announced;
    /// A command went in and its result has not shown up in a snapshot yet.
    private static bool _awaiting;

    public static bool Connected => _out != null;

    public static void Start()
    {
        int port = int.TryParse(System.Environment.GetEnvironmentVariable("STS2AI_PORT"), out int p) ? p : DefaultPort;
        var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        new Thread(() => Accept(listener)) { IsBackground = true, Name = "sts2ai-bridge" }.Start();
        GD.Print($"[sts2ai] bridge listening on 127.0.0.1:{port}");
    }

    private static void Accept(TcpListener listener)
    {
        while (true)
        {
            TcpClient client;
            try { client = listener.AcceptTcpClient(); }
            catch (SocketException) { return; }
            client.NoDelay = true;
            // Hand over on the main thread, which owns every other field.
            _pending = client;
            Inbox.Enqueue(JsonSerializer.Serialize(new { cmd = "connected" }));
            try
            {
                using var reader = new StreamReader(client.GetStream(), Encoding.UTF8);
                while (reader.ReadLine() is { } line)
                    Inbox.Enqueue(line);
            }
            catch (IOException) { }
            catch (ObjectDisposedException) { }
            Inbox.Enqueue(JsonSerializer.Serialize(new { cmd = "disconnected" }));
        }
    }

    private static volatile TcpClient? _pending;

    // ---- out --------------------------------------------------------------

    /// A recorder line: buffered for late joiners and sent on.
    internal static void Record(string line)
    {
        Fight.Add(line);
        Send(line);
    }

    /// The combat's file closed: forget its records, and any selection or
    /// settle state an abandoned combat left behind.
    internal static void CombatOver()
    {
        Fight.Clear();
        _selection = null;
        _settledSnap = null;
        _stable = 0;
        _announced = false;
        _awaiting = false;
    }

    private static void Send(string line)
    {
        lock (WriteLock)
        {
            if (_out == null) return;
            try
            {
                _out.WriteLine(line);
                _out.Flush();
            }
            catch (IOException) { Drop(); }
            catch (ObjectDisposedException) { Drop(); }
        }
    }

    private static void Send(object msg) => Send(JsonSerializer.Serialize(msg));

    private static void Error(string msg, bool wait = false)
    {
        GD.Print($"[sts2ai] bridge refused: {msg}");
        Send(new { t = "error", msg, wait });
    }

    private static void Drop()
    {
        lock (WriteLock)
        {
            _out?.Dispose();
            _out = null;
            _client?.Dispose();
            _client = null;
        }
    }

    // ---- per frame --------------------------------------------------------

    /// Run queued commands and keep the selector installed exactly while a
    /// client is connected and a combat is running.
    internal static void Poll()
    {
        while (Inbox.TryDequeue(out string? line))
        {
            try { Handle(line); }
            catch (Exception ex) { Error($"{ex.GetType().Name}: {ex.Message}"); }
        }

        var cm = CombatManager.Instance;
        bool want = Connected && cm.IsInProgress && !cm.IsEnding;
        // An emptied stack (`CardSelectCmd.Reset` at run cleanup) took ours with it.
        if (_selectorScope != null && CardSelectCmd.Selector == null) _selectorScope = null;
        if (want && _selectorScope == null)
        {
            _selectorScope = CardSelectCmd.PushSelector(Selector);
        }
        else if (!want && _selectorScope != null && _selection == null && CardSelectCmd.Selector == Selector)
        {
            // Only off the top of the stack: Whispering Earring pushes its
            // own selector over ours while it plays the opening hand.
            _selectorScope.Dispose();
            _selectorScope = null;
        }

        // Not at a decision point last frame: whatever settled has moved on.
        if (_settledFrame + 1 < Engine.GetProcessFrames()) _stable = 0;
    }

    /// Called by the recorder every frame the game sits at a decision point,
    /// with the snapshot it took. Says `ready` once it has held still.
    internal static void Settled(string snap)
    {
        _settledFrame = Engine.GetProcessFrames();
        if (snap != _settledSnap)
        {
            _settledSnap = snap;
            _stable = 0;
            _announced = false;
            _awaiting = false;
            return;
        }
        _stable++;
        if (_awaiting && _stable >= StaleFrames)
        {
            GD.Print("[sts2ai] bridge: command changed nothing; ready again");
            _awaiting = false;
            _announced = false;
        }
        if (!Connected || _announced || _awaiting || _stable < SettleFrames) return;
        if (RunManager.Instance.ActionExecutor.IsRunning) return;
        _announced = true;
        Send(new { t = "ready" });
    }

    // ---- in ---------------------------------------------------------------

    private static void Handle(string line)
    {
        var msg = JsonNode.Parse(line)?.AsObject() ?? throw new InvalidDataException("not an object");
        string cmd = msg["cmd"]?.GetValue<string>() ?? "";
        switch (cmd)
        {
            case "connected":
                Connect();
                return;
            case "disconnected":
                Drop();
                GD.Print("[sts2ai] bridge client left");
                // Nobody left to answer: the player picks.
                _selection?.Manual();
                return;
            case "pick":
            case "done":
            case "manual":
                if (_selection == null)
                {
                    Error($"{cmd} with no selection open");
                    return;
                }
                if (cmd == "pick") _selection.Pick(msg["card"]);
                else if (cmd == "done") _selection.Done();
                else _selection.Manual();
                return;
        }

        if (!AtDecision(out CombatState? state, out Player? me))
        {
            // Say `ready` again once it settles, even on the same state.
            _announced = false;
            Error($"{cmd} while the game is not waiting on the player", wait: true);
            return;
        }
        switch (cmd)
        {
            case "play":
                Play(state, me, msg);
                break;
            case "potion":
                Potion(state, me, msg);
                break;
            case "end":
                if (CombatManager.Instance.IsPlayerReadyToEndTurn(me))
                {
                    Error("turn already ended");
                    return;
                }
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, me.PlayerCombatState!.TurnNumber));
                _awaiting = true;
                break;
            default:
                Error($"unknown command {cmd}");
                break;
        }
    }

    private static void Connect()
    {
        var client = _pending;
        _pending = null;
        if (client == null) return;
        Drop();
        lock (WriteLock)
        {
            _client = client;
            _out = new StreamWriter(client.GetStream(), new UTF8Encoding(false)) { AutoFlush = false };
        }
        GD.Print("[sts2ai] bridge client connected");
        Send(new { t = "hello", version = 1 });
        foreach (string rec in Fight) Send(rec);
        _selection?.Announce();
        _announced = false;
    }

    private static bool AtDecision(out CombatState state, out Player me)
    {
        state = null!;
        me = null!;
        var cm = CombatManager.Instance;
        if (!cm.IsInProgress || cm.IsEnding || _selection != null) return false;
        var sync = RunManager.Instance.ActionQueueSynchronizer;
        if (sync == null || sync.CombatState != ActionSynchronizerCombatState.PlayPhase) return false;
        var s = cm.DebugOnlyGetState();
        var m = s == null ? null : LocalContext.GetMe(s);
        if (s == null || m?.PlayerCombatState == null || cm.IsExecutingCardOrPotionEffect(m)) return false;
        state = s;
        me = m;
        return true;
    }

    private static Creature? Target(CombatState state, JsonNode? t)
    {
        if (t == null) return null;
        int i = t.GetValue<int>();
        return i >= 0 && i < state.Enemies.Count ? state.Enemies[i] : throw new InvalidDataException($"no enemy {i}");
    }

    /// The card among `cards` that `want` names: id and upgrade must agree,
    /// then the one that also agrees on enchantment and cost wins. Copies
    /// differ only by those (a Sharp Strike, a Snecko-rolled cost).
    internal static CardModel? Find(IEnumerable<CardModel> cards, JsonNode? want)
    {
        if (want == null) return null;
        string? id = want["id"]?.GetValue<string>();
        bool up = want["up"]?.GetValue<bool>() ?? false;
        string ench = want["ench"]?.ToJsonString() ?? "null";
        int? cost = want["cost"]?.GetValue<int>();
        return cards
            .Where(c => c.Id.Entry == id && c.IsUpgraded == up)
            .OrderByDescending(c => (Ench(c) == ench ? 2 : 0) + (cost == null || Cost(c) == cost ? 1 : 0))
            .FirstOrDefault();
    }

    /// A card's enchantment as the recorder writes it, for comparing.
    private static string Ench(CardModel c) =>
        c.Enchantment is { } e
            ? JsonSerializer.Serialize(new object[] { e.Id.Entry, e.Amount, e.Status == EnchantmentStatus.Disabled })
            : "null";

    private static int Cost(CardModel c) =>
        c.EnergyCost.CostsX ? -1 : c.EnergyCost.GetWithModifiers(CostModifiers.All);

    private static void Play(CombatState state, Player me, JsonObject msg)
    {
        var want = msg["card"];
        var card = Find(me.PlayerCombatState!.Hand.Cards, want);
        if (card == null)
        {
            Error($"no {want?.ToJsonString()} in hand");
            return;
        }
        var target = Target(state, msg["target"]);
        if (!card.TryManualPlay(target))
        {
            Error($"{card.Id.Entry} cannot be played at {target?.Monster?.Id.Entry ?? "no target"}");
            return;
        }
        _awaiting = true;
    }

    private static void Potion(CombatState state, Player me, JsonObject msg)
    {
        string? id = msg["id"]?.GetValue<string>();
        int slot = msg["slot"]?.GetValue<int>() ?? -1;
        var slots = me.PotionSlots;
        // The sim's belt keeps the game's slot order; the id is the check.
        var potion = slot >= 0 && slot < slots.Count && slots[slot]?.Id.Entry == id
            ? slots[slot]
            : slots.FirstOrDefault(p => p?.Id.Entry == id);
        if (potion == null || potion.IsQueued || potion.Usage is not (PotionUsage.CombatOnly or PotionUsage.AnyTime))
        {
            Error($"potion {id} cannot be used");
            return;
        }
        var target = Target(state, msg["target"]);
        if (target != null && !potion.IsValidTarget(target))
        {
            Error($"{id} cannot target {target.Monster?.Id.Entry}");
            return;
        }
        potion.EnqueueManualUse(target);
        _awaiting = true;
    }

    // ---- selection --------------------------------------------------------

    private sealed class BridgeSelector : MegaCrit.Sts2.Core.TestSupport.ICardSelector
    {
        public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
        {
            var sel = new Selection(options.ToList(), minSelect, maxSelect);
            _selection = sel;
            Recorder.OnSelection(sel.Options, minSelect, maxSelect);
            if (Connected) sel.Announce();
            else sel.Manual();
            return sel.Task;
        }

        // Card rewards come after combat, when the bridge is not the
        // selector. Reaching this means it was left on the stack.
        public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        {
            GD.PrintErr("[sts2ai] bridge asked for a card reward; skipping it");
            return default;
        }
    }

    /// One open card selection. Picks arrive one at a time, as the sim
    /// makes them, and each is logged as a `picked` record so the replay
    /// follows the choice instead of inferring it.
    private sealed class Selection
    {
        public readonly List<CardModel> Options;
        private readonly int _min;
        private readonly int _max;
        private readonly List<CardModel> _picked = new();
        // Continuations run on a later frame, not inside the command handler.
        private readonly TaskCompletionSource<IEnumerable<CardModel>> _done = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private bool _manual;

        public Selection(List<CardModel> options, int min, int max)
        {
            Options = options;
            _min = min;
            _max = Math.Min(max, options.Count);
        }

        public Task<IEnumerable<CardModel>> Task => _done.Task;

        public void Announce() => Send(new { t = "select", min = _min, max = _max, picked = _picked.Count });

        public void Pick(JsonNode? want)
        {
            if (_manual) return;
            var card = Find(Options.Where(c => !_picked.Contains(c)), want);
            if (card == null)
            {
                Error($"{want?.ToJsonString()} is not on offer");
                return;
            }
            _picked.Add(card);
            Recorder.OnPicked(card);
            if (_picked.Count >= _max) Finish();
            else Announce();
        }

        public void Done()
        {
            if (_manual) return;
            if (_picked.Count < _min)
            {
                Error($"picked {_picked.Count} of at least {_min}");
                return;
            }
            Finish();
        }

        /// Put the selection on a card grid for the player: the client
        /// left, or cannot follow this fight.
        public async void Manual()
        {
            if (_manual) return;
            _manual = true;
            var rest = Options.Where(c => !_picked.Contains(c)).ToList();
            try
            {
                var prefs = new CardSelectorPrefs(new LocString("gameplay_ui", "CHOOSE_CARD_HEADER"), Math.Max(0, _min - _picked.Count), _max - _picked.Count);
                var screen = NSimpleCardSelectScreen.Create(rest, prefs);
                NOverlayStack.Instance!.Push(screen);
                foreach (var card in await screen.CardsSelected())
                {
                    _picked.Add(card);
                    Recorder.OnPicked(card);
                }
            }
            catch (Exception ex)
            {
                // No screen to show it on: take the minimum, in order.
                GD.PrintErr($"[sts2ai] manual selection failed: {ex.Message}");
                foreach (var card in rest.Take(Math.Max(0, _min - _picked.Count)))
                {
                    _picked.Add(card);
                    Recorder.OnPicked(card);
                }
            }
            Finish();
        }

        private void Finish()
        {
            if (_done.Task.IsCompleted) return;
            Recorder.OnPicked(null);
            _selection = null;
            _done.SetResult(_picked.ToList());
        }
    }
}
