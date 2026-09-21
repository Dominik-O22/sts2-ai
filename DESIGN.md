# StS2 AI: design decisions

Recorded 2026-09-21 after a design interview. Each item is a decision, not a plan.
Change one here before changing the code that depends on it.

## Goal

- Win Ascension 10 with Ironclad at 50% or better over many runs.
- No LLM makes decisions in the running agent. Models we train ourselves are fine.
  Using an LLM at dev time (e.g. drafting card ports) is fine.
- Ironclad only for v1. Other characters should be data, not new code, but are not built.

## Game version

- Pinned to v0.107.1 (Major Update 2, 2026-06-18, Steam buildid 23811903, main branch).
- Install: `/home/doop/.local/share/Steam/steamapps/common/Slay the Spire 2`.
  Game logic is `data_sts2_linuxbsd_x86_64/sts2.dll`; `sts2.xml` holds doc comments;
  HarmonyLib and MonoMod ship with the game.
- Patches are pulled deliberately, when the replay suite (below) goes red, never automatically.

## Simulator (Rust)

- Rules are ported by hand from the decompiled assembly (ILSpy). Not from wikis, not from
  community simulators. Each ported piece cites its source class in a comment.
- Decompiled output is MegaCrit's code. It lives in a gitignored directory next to the repo
  and a script regenerates it from the installed game.
- Data model mirrors the game's own structure so the port stays a port, not a translation.
- Scope of the first slice: act 1 at A10, every encounter incl. elites and boss, the full
  Ironclad card pool plus colorless, curses, statuses, and every relic and potion reachable
  by then. Acts 2 and 3 are added later as data.
- Ascension is a parameter with explicit code paths. Training and evaluation use A10 only.
- Own RNG. We do not reproduce MegaCrit's generator.
- Hand and piles are multisets. Hand order does not matter. The draw pile tracks a known
  top-N when an effect revealed it, unknown order below.
- Run state and combat state are separate structs from day one so the full-run sim and the
  shared value function (below) can plug in later.
- Speed is the priority. The sim is game-interface independent.

## Fidelity and tests

- The bridge mod records every real-game state transition.
- A replay harness pushes those logs through the sim, injecting recorded draws and enemy
  rolls, and diffs state per action. Every real run played becomes a regression test.
- A red suite means a sim bug or a game patch.
- Unit tests only for mechanics fiddly enough that a replay diff would not say which card
  broke. No test per card. This is a fun project.

## Decision engine

- PPO over a masked action space: (card, target), potion, end turn.
- Cards, relics, and enemies are ID embeddings over a closed vocabulary plus a few numeric
  features (cost, upgraded, current modifiers).
- Terminal reward: win/loss plus HP and potions priced by a hand-authored table keyed on act
  and upcoming fights. This table is a stopgap; see Long term.
- Later: shallow lookahead (one or two ply) over the learned value at inference time.
  AlphaZero-style search is the upgrade if that plateaus.
- Budget: a few seconds per decision in the real game, aiming for sub-second.

## Training

- Rust sim exposed to Python via PyO3, vectorized across all 8 cores.
- PyTorch on the RTX 5070 Ti. uv manages Python. TensorBoard or W&B for curves.
- Fight setups come from a generator: plausible decks, relics, potions, HP per floor, with a
  curriculum from starter deck to act 1 boss. Wide, not clever.
- Setups harvested from real runs are the held-out test set.
- Milestone bar for the combat slice: beat the act 1 boss at A10 from real-run decks 80% of
  the time. Checked daily in the sim, weekly in the real game with a human on the overworld.

## Real-game interface

- A Harmony bridge mod (C#) forked from STS2MCP or auto-spire, whichever reads cleaner,
  plus a recording endpoint. Own bridge only if both are too tangled.
- Runs in-process, localhost. A Python player process drives the loop with the trained model.
- dotnet SDK is installed on this machine for the mod build.

## Long term (after the combat slice)

- Full-run sim: map generation, card rewards, shops, events, rest sites, boss relics.
- One value network estimates probability of winning the run from run state.
  Overworld decisions are argmax over the offered options by that value.
  Combat's terminal reward becomes that same value, replacing the price table.
  This is what makes combat and overworld agree on what a potion is worth.

## Repo

- One repo, three parts: Rust crate (sim + bindings), Python package (training + player),
  C# project (bridge mod). No TypeScript.

## Order of work

1. Install dotnet SDK and ILSpy. Decompile. Write a survey of how cards, enemies, statuses,
   and the combat loop are structured in the game code.
2. Rust sim core for act 1, driven by that survey.
3. Bridge mod fork with recording. Replay harness.
4. PyO3 bindings, setup generator, PPO baseline.
5. Real-game validation loop.

## Facts gathered on 2026-09-21

See `research-sts2-ai-agent.md`. Highlights: no official state/action API; community
Harmony bridges exist (STS2MCP, auto-spire); a community Python sim exists (zhiyue) with
unverified fidelity; best published bot result is an LLM agent at A6 to A8; seeds are
deterministic and the RNG was reworked in v0.107.1.
