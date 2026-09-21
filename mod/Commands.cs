// File-driven dev console. Each frame, if <user data>/sts2ai/commands.txt
// exists, its lines run through the game's DevConsole (debug commands
// allowed) and the file is removed. Results go to commands.log next to it.
// This lets scripts outside the game set up fights: `fight NIBBITS_NORMAL`,
// `card BODY_SLAM Deck`, `relic add VAJRA`, `potion FIRE_POTION`.

using Godot;
using MegaCrit.Sts2.Core.DevConsole;

namespace Sts2Ai;

public static class Commands
{
    private static DevConsole? _console;

    private static string Dir => Path.Combine(OS.GetUserDataDir(), "sts2ai");
    private static string CommandFile => Path.Combine(Dir, "commands.txt");
    private static string LogFile => Path.Combine(Dir, "commands.log");

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
                CmdResult r = _console.ProcessCommand(line);
                outcome = $"{(r.success ? "ok" : "FAIL")} {line}: {r.msg}";
            }
            catch (Exception ex)
            {
                outcome = $"ERR {line}: {ex.Message}";
            }
            GD.Print($"[sts2ai] {outcome}");
            try { File.AppendAllText(LogFile, outcome + "\n"); } catch (IOException) { }
        }
    }
}
