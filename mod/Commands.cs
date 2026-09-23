// File-driven dev console. Each frame, if <user data>/sts2ai/commands.txt
// exists, its lines run through the game's DevConsole (debug commands
// allowed) and the file is removed. Results go to commands.log next to it.
// This lets scripts outside the game set up fights: `fight NIBBITS_NORMAL`,
// `card BODY_SLAM Deck`, `relic add VAJRA`, `potion FIRE_POTION`.
//
// Lines starting with `sts2ai` are this mod's own, for what the console
// cannot do from the main menu:
//   sts2ai continue                     load the saved run (the Continue button)
//   sts2ai new_run ASC [SEED] [ACT1]    a new Ironclad run; ACT1 overgrowth|underdocks
//   sts2ai menu                         back to the main menu (a dead run's way out)
//
// A console line other than these marks the run it lands in as scripted
// (`scripted_seeds.txt`), so tools that want played runs only
// (scripts/deckstats.py) can leave it out.

using Godot;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Acts;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Ai;

public static class Commands
{
    private static DevConsole? _console;

    private static string Dir => Path.Combine(OS.GetUserDataDir(), "sts2ai");
    private static string CommandFile => Path.Combine(Dir, "commands.txt");
    private static string LogFile => Path.Combine(Dir, "commands.log");
    private static string ScriptedFile => Path.Combine(Dir, "scripted_seeds.txt");
    private static HashSet<string>? _scripted;

    /// Whether a console line has changed the run with this seed. Kept in a
    /// file, so a run continued after a restart stays marked.
    public static bool IsScripted(string? seed)
    {
        _scripted ??= File.Exists(ScriptedFile) ? File.ReadAllLines(ScriptedFile).ToHashSet() : new HashSet<string>();
        return seed != null && _scripted.Contains(seed);
    }

    private static void MarkScripted()
    {
        string? seed = RunManager.Instance.DebugOnlyGetState()?.Rng.StringSeed;
        if (seed == null || IsScripted(seed)) return;
        _scripted!.Add(seed);
        try { File.AppendAllText(ScriptedFile, seed + "\n"); } catch (IOException) { }
    }

    public static void Poll()
    {
        if (!File.Exists(CommandFile)) return;
        string[] lines;
        try
        {
            lines = File.ReadAllLines(CommandFile);
            File.Delete(CommandFile);
        }
        catch (IOException)
        {
            return; // still being written; try next frame
        }
        _console ??= new DevConsole(shouldAllowDebugCommands: true);
        foreach (string raw in lines)
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            string outcome;
            try
            {
                if (line.StartsWith("sts2ai "))
                {
                    outcome = Own(line);
                }
                else
                {
                    MarkScripted();
                    CmdResult r = _console.ProcessCommand(line);
                    outcome = $"{(r.success ? "ok" : "FAIL")} {line}: {r.msg}";
                }
            }
            catch (Exception ex)
            {
                outcome = $"ERR {line}: {ex.Message}";
            }
            GD.Print($"[sts2ai] {outcome}");
            try { File.AppendAllText(LogFile, outcome + "\n"); } catch (IOException) { }
        }
    }

    /// The mod's own commands. They start async work and report that it
    /// started; `run.json` says when the run is up.
    private static string Own(string line)
    {
        string[] a = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        // Loading a run over the intro breaks the game's own startup
        // (`LaunchMainMenu` still owns the logo), so runs start from the menu.
        if (a.ElementAtOrDefault(1) is "continue" or "new_run" && NGame.Instance?.MainMenu == null)
            return $"FAIL {line}: not at the main menu yet";
        switch (a.ElementAtOrDefault(1))
        {
            case "continue":
            {
                if (RunManager.Instance.IsInProgress) return $"FAIL {line}: a run is already loaded";
                var save = SaveManager.Instance.LoadRunSave();
                if (!save.Success || save.SaveData == null) return $"FAIL {line}: no run save";
                TaskHelper.RunSafely(Continue(save.SaveData));
                return $"ok {line}: loading";
            }
            case "new_run":
            {
                if (RunManager.Instance.IsInProgress) return $"FAIL {line}: a run is already loaded";
                if (a.Length < 3 || !int.TryParse(a[2], out int asc)) return $"FAIL {line}: new_run ASC [SEED] [ACT1]";
                string seed = a.Length > 3 && a[3] != "-" ? SeedHelper.CanonicalizeSeed(a[3]) : SeedHelper.GetRandomSeed();
                TaskHelper.RunSafely(NewRun(asc, seed, a.ElementAtOrDefault(4)));
                return $"ok {line}: starting seed {seed}";
            }
            case "menu":
                TaskHelper.RunSafely(NGame.Instance!.ReturnToMainMenu());
                return $"ok {line}: leaving";
            default:
                return $"FAIL {line}: unknown";
        }
    }

    // As `FileDropHandler` loads a run, and the Continue button without its
    // transition: set up the saved state, the single-player net service,
    // then the run's scenes.
    private static async Task Continue(SerializableRun save)
    {
        RunState state = RunState.FromSerializable(save);
        await RunManager.Instance.SetUpSavedSingleplayer(state, save);
        NGame.Instance!.ReactionContainer.InitializeNetworking(new NetSingleplayerGameService());
        await NGame.Instance.LoadRun(state, save.PreFinishedRoom);
    }

    // As `StartRunLobby.BeginRunLocally` picks the acts from the seed and
    // `NCharacterSelectScreen` embarks.
    private static async Task NewRun(int asc, string seed, string? act1)
    {
        var rng = new MegaCrit.Sts2.Core.Random.Rng((uint)StringHelper.GetDeterministicHashCode(seed), "act_selection");
        var acts = ActModel.GetRandomList(rng, SaveManager.Instance.GenerateUnlockStateFromProgress(), false).ToList();
        if (act1 == "overgrowth") acts[0] = ModelDb.Act<Overgrowth>();
        else if (act1 == "underdocks") acts[0] = ModelDb.Act<Underdocks>();
        NGame.Instance!.ReactionContainer.InitializeNetworking(new NetSingleplayerGameService());
        await NGame.Instance.StartNewSingleplayerRun(ModelDb.Character<Ironclad>(), shouldSave: true, acts, Array.Empty<ModifierModel>(), seed, GameMode.Standard, asc);
    }
}
