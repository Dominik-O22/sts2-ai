using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

// Runs the game's own code outside the game and prints what it produced, one
// value per line, for the sim's tests to pin. `dotnet run -- COMMAND ARGS`.
switch (args[0])
{
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
