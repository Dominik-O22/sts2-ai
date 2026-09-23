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
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Models.RelicPools;
using MegaCrit.Sts2.Core.Timeline;
using MegaCrit.Sts2.Core.Unlocks;

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
    // rooms: one new singleplayer Ironclad run per stdin line
    // `SEED ASCENSION [LOCKED [UNSEEN]]`, planned the way the game plans it
    // at run start: the acts (`StartRunLobby.BeginRunLocally`), the relic
    // grab bags (`RunManager.InitializeNewRun`) and every act's rooms
    // (`RunManager.GenerateRooms`), all on the run's UpFront stream. The
    // profile has everything unlocked and seen, less the epoch ids in LOCKED
    // and the encounter entries in UNSEEN (comma-separated, `-` for none).
    // Prints the line as a `run` header, each bag per rarity, each act's
    // rooms and the UpFront counter after planning.
    //
    // `RunState` and `RunManager` need a running game (the player's
    // constructor reads the progress save), so the steps of those two
    // methods are spelled out here around the game's own pieces: the grab
    // bags, the shuffles, `ActModel.GenerateRooms` and
    // `ApplyDiscoveryOrderModifications`.
    case "rooms":
    {
        LoadModelDb();
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            var (seed, ascension) = (parts[0], int.Parse(parts[1]));
            string[] Listed(int i) => parts.Length > i && parts[i] != "-" ? parts[i].Split(',') : [];
            var (locked, unseen) = (Listed(2), Listed(3));
            var unlocks = new UnlockState(
                EpochModel.AllEpochIds.Except(locked),
                ModelDb.AllEncounters.Select(e => e.Id).Where(id => !unseen.Contains(id.Entry)),
                999999999);
            var seedHash = (uint)StringHelper.GetDeterministicHashCode(seed);
            // In singleplayer `GetRandomList` asks the progress save whether
            // an alt act was discovered, which needs the running game. The
            // multiplayer flag skips only that question, so every unlocked
            // act counts as discovered.
            var acts = ActModel.GetRandomList(new Rng(seedHash, "act_selection"), unlocks, isMultiplayer: true)
                .Select(a => a.ToMutable())
                .ToList();
            var upFront = new RunRngSet(seed).UpFront;

            // InitializeNewRun, then Player.PopulateRelicGrabBagIfNecessary
            // with RelicGrabBag.Populate(Player, Rng)'s list.
            var shared = new RelicGrabBag(refreshAllowed: true);
            shared.Populate(ModelDb.RelicPool<SharedRelicPool>().GetUnlockedRelics(unlocks), upFront);
            var playerRelics = ModelDb.RelicPool<SharedRelicPool>().GetUnlockedRelics(unlocks).ToList();
            playerRelics.AddRange(ModelDb.Character<Ironclad>().RelicPool.GetUnlockedRelics(unlocks));
            playerRelics.RemoveAll(r => r.Rarity is not (RelicRarity.Common or RelicRarity.Uncommon or RelicRarity.Rare or RelicRarity.Shop));
            var player = new RelicGrabBag();
            player.Populate(playerRelics, upFront);

            // GenerateRooms.
            var ancients = unlocks.SharedAncients.ToList().UnstableShuffle(upFront);
            foreach (var act in acts.Skip(1))
            {
                var count = upFront.NextInt(ancients.Count + 1);
                var subset = ancients.Take(count).ToList();
                ancients = ancients.Except(subset).ToList();
                act.SetSharedAncientSubset(subset);
            }
            for (var i = 0; i < acts.Count; i++)
            {
                var act = acts[i];
                act.GenerateRooms(upFront, unlocks);
                act.ApplyDiscoveryOrderModifications(unlocks);
                if (i == acts.Count - 1 && ascension >= (int)AscensionLevel.DoubleBoss)
                {
                    act.SetSecondBossEncounter(upFront.NextItem(act.AllBossEncounters.Where(e => e.Id != act.BossEncounter.Id)));
                }
            }

            Console.WriteLine($"run {string.Join(" ", parts)}");
            PrintBag("shared_bag", shared);
            PrintBag("player_bag", player);
            foreach (var act in acts)
            {
                var rooms = act.ToSave().SerializableRooms;
                Console.WriteLine($"act {act.Id.Entry}");
                Console.WriteLine($"events {string.Join(" ", rooms.EventIds.Select(e => e.Entry))}");
                Console.WriteLine($"normal {string.Join(" ", rooms.NormalEncounterIds.Select(e => e.Entry))}");
                Console.WriteLine($"elite {string.Join(" ", rooms.EliteEncounterIds.Select(e => e.Entry))}");
                Console.WriteLine($"boss {rooms.BossId?.Entry} {rooms.SecondBossId?.Entry}".TrimEnd());
                Console.WriteLine($"ancient {rooms.AncientId?.Entry}");
            }
            Console.WriteLine($"up_front {upFront.Counter}");
        }
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

// Every canonical model, as `ModelDb.Init` makes them, without the mod scan
// it needs a running game for. Model ids come from the type; the
// net-serialization sort ids `ModelDb.InitIds` adds are never read here.
static void LoadModelDb()
{
    foreach (var type in AbstractModelSubtypes.All)
    {
        ModelDb.Inject(type);
    }
}

// A relic grab bag, one line per rarity in the bag's own order.
static void PrintBag(string name, RelicGrabBag bag)
{
    foreach (var (rarity, ids) in bag.ToSerializable().RelicIdLists)
    {
        Console.WriteLine($"{name} {rarity} {string.Join(" ", ids.Select(id => id.Entry))}");
    }
}
