using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Acts;
using MegaCrit.Sts2.Core.Runs;

// Runs the game's own code outside the game and prints what it produced, one
// value per line, for the sim's tests to pin. `dotnet run -- COMMAND ARGS`.
switch (args[0])
{
    // map SEED ACT N ASCENSION: act N's map (1-based) as `StandardActMap.CreateFor`
    // builds it for a single player, one point per line: col row type
    // children, children as col,row.
    case "map":
    {
        var map = BuildMap(args[1], args[2], int.Parse(args[3]), int.Parse(args[4]));
        foreach (var p in map.GetAllMapPoints().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
        {
            Console.WriteLine(PointLine(p));
        }
        break;
    }
    // maps: one map per stdin line `SEED ACT ASCENSION`, each printed as a
    // `map SEED ACT ASCENSION` header and then its points like `map`, plus
    // the starting point and the boss points, which `map` leaves out. N is
    // the act's own index, as in a real run.
    case "maps":
    {
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            var n = parts[1] switch { "Hive" => 2, "Glory" => 3, _ => 1 };
            var map = BuildMap(parts[0], parts[1], n, int.Parse(parts[2]));
            Console.WriteLine($"map {parts[0]} {parts[1]} {parts[2]}");
            var points = map.GetAllMapPoints()
                .Append(map.StartingMapPoint)
                .Append(map.BossMapPoint)
                .Concat(map.SecondBossMapPoint is { } second ? new[] { second } : []);
            foreach (var p in points.OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
            {
                Console.WriteLine(PointLine(p));
            }
        }
        break;
    }
    // rng SEED: every Rng method the run layer uses, on the run's streams
    // (RunRngSet) and on a player stream seeded like `new PlayerRngSet(seed)`.
    case "rng":
    {
        var run = new RunRngSet(args[0 + 1]);
        Console.WriteLine($"hash {run.Seed}");
        Console.WriteLine($"snake {StringHelper.SnakeCase(nameof(RunRngType.CombatCardGeneration))}");
        foreach (var type in Enum.GetValues<RunRngType>())
        {
            var rng = new Rng(run.Seed, StringHelper.SnakeCase(type.ToString()));
            Console.WriteLine($"{type} {string.Join(" ", Enumerable.Range(0, 5).Select(_ => rng.NextInt(100)))}");
        }
        var r = new Rng(run.Seed, "up_front");
        Console.WriteLine($"int_range {string.Join(" ", Enumerable.Range(0, 5).Select(_ => r.NextInt(-3, 7)))}");
        Console.WriteLine($"bool {string.Join(" ", Enumerable.Range(0, 5).Select(_ => r.NextBool() ? 1 : 0))}");
        Console.WriteLine($"float {string.Join(" ", Enumerable.Range(0, 3).Select(_ => r.NextFloat().ToString("R")))}");
        Console.WriteLine($"double {string.Join(" ", Enumerable.Range(0, 3).Select(_ => r.NextDouble().ToString("R")))}");
        Console.WriteLine($"uint {string.Join(" ", Enumerable.Range(0, 3).Select(_ => r.NextUnsignedInt(5u, 50u)))}");
        Console.WriteLine($"gauss_int {string.Join(" ", Enumerable.Range(0, 3).Select(_ => r.NextGaussianInt(10, 3, 5, 15)))}");
        var list = Enumerable.Range(0, 10).ToList();
        r.Shuffle(list);
        Console.WriteLine($"shuffle {string.Join(" ", list)}");
        Console.WriteLine($"item {r.NextItem(new[] { 10, 20, 30, 40 })}");
        Console.WriteLine($"counter {r.Counter}");
        var ff = new Rng(run.Seed, "up_front");
        ff.FastForwardCounter(7);
        Console.WriteLine($"fast_forward {ff.NextInt(100)}");
        var player = new PlayerRngSet(run.Seed);
        Console.WriteLine($"rewards {string.Join(" ", Enumerable.Range(0, 5).Select(_ => player.Rewards.NextInt(100)))}");
        break;
    }
}

// Act N's map (1-based) as `StandardActMap.CreateFor` builds it for a single
// player.
static StandardActMap BuildMap(string seedText, string actName, int n, int ascension)
{
    ActModel act = actName switch
    {
        "Overgrowth" => new Overgrowth(),
        "Underdocks" => new Underdocks(),
        "Hive" => new Hive(),
        "Glory" => new Glory(),
        _ => throw new ArgumentException(actName),
    };
    var seed = (uint)StringHelper.GetDeterministicHashCode(seedText);
    // Ascension lives on the running game's RunManager, so the counts
    // are drawn here the way the act draws them, on the same stream,
    // with Swarming Elites (A1) applied by hand.
    var rng = new Rng(seed, $"act_{n}_map");
    var drawn = act.GetMapPointTypes(rng);
    var counts = new MapPointTypeCounts(drawn.NumOfUnknowns, drawn.NumOfRests)
    {
        NumOfElites = (int)Math.Round(5f * (ascension >= 1 ? 1.6f : 1f)),
        PointTypesThatIgnoreRules = drawn.PointTypesThatIgnoreRules,
    };
    return new StandardActMap(rng, act, false, false, n == 3 && ascension >= 10, counts);
}

// One point: col row type children, children as col,row.
static string PointLine(MapPoint p)
{
    var kids = string.Join(" ", p.Children.OrderBy(c => c.coord.col).Select(c => $"{c.coord.col},{c.coord.row}"));
    return $"{p.coord.col} {p.coord.row} {p.PointType} {kids}";
}
