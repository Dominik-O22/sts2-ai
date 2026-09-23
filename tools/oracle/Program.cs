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
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Models.CardPools;
using MegaCrit.Sts2.Core.Models.PotionPools;
using MegaCrit.Sts2.Core.Commands;

// Runs the game's own code outside the game and prints what it produced, one
// value per line, for the sim's tests to pin. `dotnet run -- COMMAND ARGS`.
switch (args[0])
{
    // pools: the pools rewards and shops draw from, in the game's order, as
    // a fully unlocked profile has them: `card POOL ID RARITY TYPE
    // CONSTRAINT UPGRADABLE` and `potion POOL ID RARITY`.
    case "pools":
    {
        LoadModelDb();
        foreach (var pool in new CardPoolModel[] { ModelDb.CardPool<IroncladCardPool>(), ModelDb.CardPool<ColorlessCardPool>() })
        {
            foreach (var c in pool.AllCards)
            {
                Console.WriteLine($"card {pool.Id.Entry} {c.Id.Entry} {c.Rarity} {c.Type} {c.MultiplayerConstraint} {(c.IsUpgradable ? 1 : 0)}");
            }
        }
        foreach (var pool in new PotionPoolModel[] { ModelDb.PotionPool<IroncladPotionPool>(), ModelDb.PotionPool<SharedPotionPool>() })
        {
            foreach (var p in pool.AllPotions)
            {
                Console.WriteLine($"potion {pool.Id.Entry} {p.Id.Entry} {p.Rarity}");
            }
        }
        break;
    }
    // rewards: one new singleplayer Ironclad run per stdin line `SEED
    // ASCENSION STEP...`, fully unlocked, its grab bags filled the way
    // `RunManager.InitializeNewRun` fills them, then each step in order:
    // `M1`, `E1` or `B1` generates the rewards of a monster, elite or boss
    // room in act 1 (0-based) through `RewardsSet`, as combat ends; `S1`
    // stocks a merchant (`MerchantInventory.CreateForNormalMerchant`); `?`
    // rolls an unknown point, `?s` with shops blacklisted; `R` resets the
    // unknown odds as a new act does. Prints the line as a `run` header,
    // then one line per step: the rewards in the order the screen lists
    // them, or the shop's stock, with the player's Rewards counter (and
    // Shops counter), or the room type rolled. On the last shop stocked,
    // `$` prints every entry's price as rolled (`MerchantEntry._cost`) and
    // the card removal's; `C4` restocks entry 4 as The Courier does
    // (`RestockAfterPurchase`; the character's cards, the colorless ones,
    // the relics, the potions, counted from 0) and prints what it holds
    // now and its price; `X` counts a card removal bought.
    case "rewards":
    {
        LoadModelDb();
        RunOutsideTheGame();
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            var (state, player) = NewRun(parts[0], int.Parse(parts[1]));
            Console.WriteLine($"run {line}");
            MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory? lastShop = null;
            var costField = typeof(MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry).GetField("_cost", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            foreach (var step in parts.Skip(2))
            {
                if (step == "X")
                {
                    player.ExtraFields.CardShopRemovalsUsed++;
                    Console.WriteLine("X");
                    continue;
                }
                if (step == "$")
                {
                    var costs = lastShop!.CardEntries.Cast<MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry>().Concat(lastShop.RelicEntries).Concat(lastShop.PotionEntries);
                    Console.WriteLine($"$ {string.Join(" ", costs.Select(e => costField.GetValue(e)))} remove {costField.GetValue(lastShop.CardRemovalEntry)}");
                    continue;
                }
                if (step[0] == 'C')
                {
                    var entry = lastShop!.CardEntries.Cast<MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry>().Concat(lastShop.RelicEntries).Concat(lastShop.PotionEntries).ElementAt(int.Parse(step[1..]));
                    typeof(MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry).GetMethod("RestockAfterPurchase", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(entry, [lastShop]);
                    var item = entry switch
                    {
                        MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardEntry c => c.CreationResult!.Card.Id.Entry + (c.CreationResult.Card.IsUpgraded ? "+" : ""),
                        MegaCrit.Sts2.Core.Entities.Merchant.MerchantRelicEntry r => r.Model!.Id.Entry,
                        MegaCrit.Sts2.Core.Entities.Merchant.MerchantPotionEntry p => p.Model!.Id.Entry,
                        _ => throw new InvalidOperationException(),
                    };
                    Console.WriteLine($"{step} {item} {costField.GetValue(entry)} counter {player.PlayerRng.Rewards.Counter} shops {player.PlayerRng.Shops.Counter}");
                    continue;
                }
                if (step.StartsWith('?'))
                {
                    var blacklist = step == "?s" ? new[] { RoomType.Shop } : [];
                    Console.WriteLine($"{step} {state.Odds.UnknownMapPoint.Roll(blacklist, state)}");
                    continue;
                }
                if (step == "R")
                {
                    state.Odds.UnknownMapPoint.ResetToBase();
                    Console.WriteLine("R");
                    continue;
                }
                state.CurrentActIndex = step[1] - '0';
                if (step[0] == 'S')
                {
                    var shop = MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory.CreateForNormalMerchant(player);
                    lastShop = shop;
                    string Card(MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardEntry e) => e.CreationResult!.Card.Id.Entry + (e.CreationResult.Card.IsUpgraded ? "+" : "");
                    var sale = shop.CharacterCardEntries.Select((e, i) => (e, i)).First(x => x.e.IsOnSale).i;
                    Console.WriteLine($"{step} cards {string.Join(" ", shop.CharacterCardEntries.Select(Card))} sale {sale}"
                        + $" colorless {string.Join(" ", shop.ColorlessCardEntries.Select(Card))}"
                        + $" relics {string.Join(" ", shop.RelicEntries.Select(e => e.Model!.Id.Entry))}"
                        + $" potions {string.Join(" ", shop.PotionEntries.Select(e => e.Model!.Id.Entry))}"
                        + $" counter {player.PlayerRng.Rewards.Counter} shops {player.PlayerRng.Shops.Counter}");
                    continue;
                }
                EncounterModel encounter = step[0] switch
                {
                    'M' => ModelDb.Encounter<MegaCrit.Sts2.Core.Models.Encounters.NibbitsWeak>(),
                    'E' => ModelDb.Encounter<MegaCrit.Sts2.Core.Models.Encounters.BygoneEffigyElite>(),
                    _ => ModelDb.Encounter<MegaCrit.Sts2.Core.Models.Encounters.VantomBoss>(),
                };
                var set = new MegaCrit.Sts2.Core.Rewards.RewardsSet(player).WithRewardsFromRoom(new CombatRoom(encounter.ToMutable(), state));
                set.GenerateWithoutOffering().GetAwaiter().GetResult();
                var shown = set.Rewards.Select(r => r switch
                {
                    MegaCrit.Sts2.Core.Rewards.GoldReward g => $"gold {g.Amount}",
                    MegaCrit.Sts2.Core.Rewards.PotionReward p => $"potion {p.Potion!.Id.Entry}",
                    MegaCrit.Sts2.Core.Rewards.CardReward c => "cards " + string.Join(" ", c.Cards.Select(x => x.Id.Entry + (x.IsUpgraded ? "+" : ""))),
                    MegaCrit.Sts2.Core.Rewards.RelicReward rr => $"relic {rr.Relic!.Id.Entry}",
                    _ => throw new InvalidOperationException(r.ToString()),
                });
                Console.WriteLine($"{step} {string.Join(" ", shown)} counter {player.PlayerRng.Rewards.Counter}".Replace("  ", " "));
            }
        }
        break;
    }
    // ancients: one new singleplayer Ironclad run per stdin line `SEED
    // ASCENSION ACT ANCIENT [+CARD|-CARD]...`, fully unlocked, in act ACT
    // (0-based), its deck changed by the card ops (`+` adds a card, `-`
    // removes the first of its id), then the ancient's options as
    // `GenerateInitialOptions` lays them out on the event's own stream
    // (`EventModel.BeginEvent` seeds it). Prints the line as a `run` header,
    // then the options' relics (Sea Glass as `SEA_GLASS`, Dusty Tome with the
    // card it readies) and the player's Rewards counter.
    case "ancients":
    {
        LoadModelDb();
        RunOutsideTheGame();
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public;
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            var (state, player) = NewRun(parts[0], int.Parse(parts[1]));
            state.CurrentActIndex = int.Parse(parts[2]);
            foreach (var op in parts.Skip(4))
            {
                var id = op[1..];
                if (op[0] == '+')
                {
                    var canonical = ModelDb.AllCards.First(c => c.Id.Entry == id);
                    player.Deck.AddInternal(state.CreateCard(canonical, player));
                }
                else
                {
                    player.Deck.RemoveInternal(player.Deck.Cards.First(c => c.Id.Entry == id));
                }
            }
            var ancient = (AncientEventModel)ModelDb.AllAncients.First(a => a.Id.Entry == parts[3]).ToMutable();
            typeof(EventModel).GetProperty("Owner")!.SetValue(ancient, player);
            var seed = (uint)((uint)(int)state.Rng.Seed + (uint)StringHelper.GetDeterministicHashCode(ancient.Id.Entry));
            typeof(EventModel).GetProperty("Rng")!.SetValue(ancient, new Rng(seed));
            var options = (IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption>)typeof(AncientEventModel)
                .GetMethod("GenerateInitialOptions", flags)!.Invoke(ancient, null)!;
            string Option(MegaCrit.Sts2.Core.Events.EventOption o) => o.Relic switch
            {
                null => o.TextKey,
                MegaCrit.Sts2.Core.Models.Relics.DustyTome tome => $"DUSTY_TOME:{tome.AncientCard!.Entry}",
                var relic => relic.Id.Entry,
            };
            Console.WriteLine($"run {line}");
            Console.WriteLine($"{parts[3]} {string.Join(" ", options.Select(Option))} counter {player.PlayerRng.Rewards.Counter}");
        }
        break;
    }
    // obtain: one new singleplayer Ironclad run per stdin line `SEED
    // ASCENSION ACT RELIC...`, fully unlocked, in act ACT (0-based), each
    // relic obtained in turn through `RelicCmd.Obtain`, its pickup and all.
    // Every choice takes from the front: a deck pick the first cards it may
    // (as many as it may), a card screen its first card. Prints the line as
    // a `run` header, then per relic the player after it: HP, max HP, gold,
    // the deck (id, `+` if upgraded, `:ENCHANTMENT:AMOUNT`), the relics, the
    // potion slots (`-` for an empty one) and the Rewards, Niche,
    // Transformations and CombatPotionGeneration counters; or `error` and
    // why, where the game's code refuses the relic, and nothing after.
    case "obtain":
    {
        LoadModelDb();
        RunOutsideTheGame();
        CardSelectCmd.PushSelector(new FrontSelector());
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }
            var (state, player) = NewRun(parts[0], int.Parse(parts[1]));
            state.CurrentActIndex = int.Parse(parts[2]);
            Console.WriteLine($"run {line}");
            foreach (var id in parts.Skip(3))
            {
                var relic = ModelDb.AllRelics.First(r => r.Id.Entry == id).ToMutable();
                try
                {
                    RelicCmd.Obtain(relic, player).GetAwaiter().GetResult();
                }
                catch (InvalidOperationException e)
                {
                    // The game's own refusal (Leafy Poultice with every basic
                    // Strike Eternal): the run stops here.
                    Console.WriteLine($"{id} error {e.Message}");
                    break;
                }
                string Card(CardModel c) => c.Id.Entry + (c.IsUpgraded ? "+" : "") + (c.Enchantment is { } e ? $":{e.Id.Entry}:{e.Amount}" : "");
                var deck = string.Join(" ", player.Deck.Cards.Select(Card).OrderBy(c => c, StringComparer.Ordinal));
                var potions = string.Join(" ", player.PotionSlots.Select(p => p?.Id.Entry ?? "-"));
                var c = player.Creature;
                Console.WriteLine($"{id} hp {c.CurrentHp} {c.MaxHp} gold {player.Gold} deck {deck} relics {string.Join(" ", player.Relics.Select(r => r.Id.Entry))}"
                    + $" potions {potions} counters {player.PlayerRng.Rewards.Counter} {state.Rng.Niche.Counter} {player.PlayerRng.Transformations.Counter} {state.Rng.CombatPotionGeneration.Counter}");
            }
        }
        break;
    }
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

// What a player, a run and the reward code need that only a running game
// has, stood in for. None of it draws.
static void RunOutsideTheGame()
{
    // `SaveManager.Instance` builds itself on Godot's file system; players
    // read their progress from it.
    var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
    var progress = System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Saves.Managers.ProgressSaveManager));
    typeof(MegaCrit.Sts2.Core.Saves.Managers.ProgressSaveManager).GetProperty("Progress")!.SetValue(progress, MegaCrit.Sts2.Core.Saves.ProgressState.CreateDefault());
    var save = (MegaCrit.Sts2.Core.Saves.SaveManager)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Saves.SaveManager));
    typeof(MegaCrit.Sts2.Core.Saves.SaveManager).GetField("_progressSaveManager", flags)!.SetValue(save, progress);
    MegaCrit.Sts2.Core.Saves.SaveManager.MockInstanceForTesting(save);
    // No mods: the model lists that ask for mod types get none.
    typeof(MegaCrit.Sts2.Core.Modding.ModManager).GetProperty("State")!.SetValue(null, MegaCrit.Sts2.Core.Modding.ModManagerState.Skipped);
    typeof(ReflectionHelper).GetField("_modTypes", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!.SetValue(null, Array.Empty<Type>());
    // The rest is patched out with Harmony, which the game ships for mods.
    var harmony = new HarmonyLib.Harmony("oracle");
    void Stub(System.Reflection.MethodBase? method, string stub) =>
        harmony.Patch(method, prefix: new HarmonyLib.HarmonyMethod(typeof(GodotStubs).GetMethod(stub)));
    // The logger asks Godot for the command line and prints through it.
    Stub(typeof(Godot.OS).GetMethod("GetCmdlineArgs"), "NoArgs");
    Stub(typeof(Godot.OS).GetMethod("HasFeature"), "NoFeature");
    Stub(typeof(MegaCrit.Sts2.Core.Logging.ConsoleLogPrinter).GetMethod("Print"), "Skip");
    // A card reward lists its skip button, whose hotkey is a Godot input
    // name.
    Stub(typeof(MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative).GetMethod("Generate"), "NoAlternatives");
    // A few relics format localized text into their variables, and there is
    // no localization loaded.
    Stub(typeof(MegaCrit.Sts2.Core.Localization.LocString).GetMethod("GetFormattedText", Type.EmptyTypes), "NoText");
    // Healing and losing HP play sounds and effects through Godot.
    foreach (var type in new[] { typeof(MegaCrit.Sts2.Core.Commands.SfxCmd), typeof(MegaCrit.Sts2.Core.Commands.VfxCmd) })
    {
        foreach (var method in type.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public).Where(m => m.ReturnType == typeof(void) && !m.IsGenericMethod))
        {
            Stub(method, "Skip");
        }
    }
    // Waits read the player's speed setting.
    Stub(typeof(Cmd).GetMethod("CustomScaledWait"), "NoWait");
    // A relic's rewards (Lost Coffer, Neow's Bones) go to a screen; here
    // they are generated, and the first card, the potions and the relics
    // taken.
    Stub(typeof(MegaCrit.Sts2.Core.Rewards.RewardsSet).GetMethod("Offer"), "TakeRewards");
    // Scroll Boxes' bundle screen: the first bundle.
    Stub(typeof(CardSelectCmd).GetMethod("FromChooseABundleScreen"), "FirstBundle");
    // An event option looks its text up in the tables, which are not loaded:
    // a relic's option is its key and relic alone.
    Stub(typeof(MegaCrit.Sts2.Core.Events.EventOption).GetMethod("FromRelic"), "RelicOnly");
    // Archaic Tooth's setup names its cards in its text, which crashes with
    // none loaded; its answer is whether the deck holds a card it
    // transcends, which draws nothing.
    Stub(typeof(MegaCrit.Sts2.Core.Models.Relics.ArchaicTooth).GetMethod("SetupForPlayer"), "ToothFits");
}

// A new singleplayer Ironclad run for a fully unlocked profile, set on
// `RunManager.Instance` the way `SetUpNewSingleplayer` sets it, less the
// networking: its state and ascension, then `InitializeNewRun`, which fills
// the grab bags and applies the ascension.
static (RunState, Player) NewRun(string seed, int ascension)
{
    var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
    var player = Player.CreateForNewRun<Ironclad>(UnlockState.all, 1);
    var acts = new List<ActModel> { ModelDb.Act<Overgrowth>(), ModelDb.Act<Hive>(), ModelDb.Act<Glory>() }.Select(a => a.ToMutable()).ToList();
    var state = RunState.CreateForNewRun(new[] { player }, acts, [], GameMode.Standard, ascension, seed);
    typeof(RunManager).GetProperty("State", flags)!.SetValue(RunManager.Instance, state);
    typeof(RunManager).GetProperty("AscensionManager")!.SetValue(RunManager.Instance, new AscensionManager(ascension));
    typeof(RunManager).GetMethod("InitializeNewRun", flags)!.Invoke(RunManager.Instance, null);
    return (state, player);
}

// A relic grab bag, one line per rarity in the bag's own order.
static void PrintBag(string name, RelicGrabBag bag)
{
    foreach (var (rarity, ids) in bag.ToSerializable().RelicIdLists)
    {
        Console.WriteLine($"{name} {rarity} {string.Join(" ", ids.Select(id => id.Entry))}");
    }
}

// Takes from the front: as many cards as a deck pick allows, a card
// screen's first card.
class FrontSelector : MegaCrit.Sts2.Core.TestSupport.ICardSelector
{
    public System.Threading.Tasks.Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect) =>
        System.Threading.Tasks.Task.FromResult(options.Take(maxSelect).ToList().AsEnumerable());

    public MegaCrit.Sts2.Core.TestSupport.CardRewardSelection GetSelectedCardReward(IReadOnlyList<MegaCrit.Sts2.Core.Entities.Cards.CardCreationResult> options, IReadOnlyList<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative> alternatives) =>
        new() { card = options.First().Card };
}

static class GodotStubs
{
    public static bool Skip() => false;

    public static bool NoAlternatives(ref IReadOnlyList<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative> __result)
    {
        __result = [];
        return false;
    }

    public static bool NoArgs(ref string[] __result)
    {
        __result = [];
        return false;
    }

    public static bool NoText(ref string __result)
    {
        __result = "";
        return false;
    }

    public static bool NoWait(ref System.Threading.Tasks.Task __result)
    {
        __result = System.Threading.Tasks.Task.CompletedTask;
        return false;
    }

    public static bool FirstBundle(IReadOnlyList<IReadOnlyList<CardModel>> bundles, ref System.Threading.Tasks.Task<IEnumerable<CardModel>> __result)
    {
        __result = System.Threading.Tasks.Task.FromResult(bundles[0].AsEnumerable());
        return false;
    }

    public static bool TakeRewards(MegaCrit.Sts2.Core.Rewards.RewardsSet __instance, ref System.Threading.Tasks.Task __result)
    {
        __instance.GenerateWithoutOffering().GetAwaiter().GetResult();
        var player = __instance.Player;
        foreach (var reward in __instance.Rewards.ToList())
        {
            switch (reward)
            {
                case MegaCrit.Sts2.Core.Rewards.CardReward c:
                    CardPileCmd.Add(c.Cards.First(), MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck).GetAwaiter().GetResult();
                    break;
                case MegaCrit.Sts2.Core.Rewards.PotionReward p:
                    PotionCmd.TryToProcure(p.Potion!, player).GetAwaiter().GetResult();
                    break;
                case MegaCrit.Sts2.Core.Rewards.RelicReward r:
                    RelicCmd.Obtain(r.Relic!, player).GetAwaiter().GetResult();
                    break;
            }
        }
        __result = System.Threading.Tasks.Task.CompletedTask;
        return false;
    }

    public static bool ToothFits(Player player, ref bool __result)
    {
        var starters = new[] { "BASH", "NEUTRALIZE", "UNLEASH", "FALLING_STAR", "DUALCAST" };
        __result = player.Deck.Cards.Any(c => starters.Contains(c.Id.Entry));
        return false;
    }

    public static bool RelicOnly(RelicModel relic, string textKey, ref MegaCrit.Sts2.Core.Events.EventOption __result)
    {
        var option = (MegaCrit.Sts2.Core.Events.EventOption)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Events.EventOption));
        typeof(MegaCrit.Sts2.Core.Events.EventOption).GetProperty("TextKey")!.SetValue(option, textKey);
        __result = option.WithRelic(relic);
        return false;
    }

    public static bool NoFeature(ref bool __result)
    {
        __result = false;
        return false;
    }
}
