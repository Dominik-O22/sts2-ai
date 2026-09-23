# The bridge: the policy plays combats

The mod listens on `127.0.0.1:47474`. A Python player connects, follows the
fight through the recorder's records, and sends back what the policy picks.
You play the map, rewards, shops and events. The player takes every combat,
card choices included.

```
./scripts/build-mod.sh                      # with the game closed; it refuses otherwise
uv run python -m sts2ai.play runs/<run>/latest.pt --search 256
```

Start it before or during a run, in or out of combat. A client that
connects mid-fight gets the fight so far and picks up at the next decision
point. Ctrl-C hands the game back to you. A card choice open at that moment
goes to a card grid for you to pick.

The player prints the same advice lines as `sts2ai.advise`, then acts on
them. When the sim cannot follow a fight (a divergence, or an encounter it
does not model) it says so once and leaves the fight to you. Card choices
in that fight go to the grid.

## Card rewards

```
uv run python -m sts2ai.cards runs/<run>/latest.pt
```

A second process, alongside the player or without it, ranks each card
reward while its screen is open. The mod writes the offered cards into
`sts2ai/run.json` (`card_reward`, with the act, the bosses the map shows
and max energy), and `sts2ai.cards` plays the deck with each card added,
and without one, against every elite of the act and its boss: 512 greedy fights per encounter
per option, the same enemies for every option. It prints the options best
first, each against skipping, with the win rate and the worst encounter.
Values closer than about 0.03 are a tie. It judges the deck as it stands,
not the picks ahead, and the policy plays cards it was trained on: a card
the generator rarely hands out may be undervalued.

## Protocol

JSON lines both ways. `mod/Bridge.cs` has the full list at the top.

From the game: every recording line of the current combat, and

- `ready`: the play phase has held still for a few frames with nothing
  queued. The player may act.
- `select` (`min`, `max`, `picked`): a card selection is open and waits
  on a pick.
- `error` (`msg`, `wait`): the last command was refused. With `wait` the
  game was busy and the next `ready` follows. Without it, the action was
  illegal and the player masks it out and picks again.

From the player: `play` (card, target), `potion` (slot, id, target),
`end`, `pick` (card), `done`, `manual`. `sim::replay::command` writes
them from an action index. Cards are named by id, upgrade, enchantment and
cost, not hand slot, because the sim keeps its own hand order. The mod
takes the copy that agrees on the most of those. Targets index the living
enemies, as in the recording.

## Card choices

While a client is connected and a combat runs, the mod pushes itself as
the game's card selector (`CardSelectCmd.PushSelector`). Every in-combat
selection then calls the mod instead of opening a screen. The mod writes a
`choice` record and sends `select`.

The sim makes choices one card at a time, and the game asks for the whole
set at once. So the player sends one `pick` per sim choice. The mod logs
each as a `picked` record, which the replayer applies, and the sim's next
pending choice (Ashwater asks again) becomes the next decision. The mod
closes the selection at `max`, on `done` (the sim skipped), or when the
sim's choice has closed and the game already has its minimum.

The `picked` records make bot-played recordings exact about choices. The
replay suite applies them instead of trying every option.

After combat the mod steps aside, so rewards are yours to pick as usual.
