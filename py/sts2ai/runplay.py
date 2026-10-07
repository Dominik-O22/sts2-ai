"""Play whole runs: a combat checkpoint plays the fights greedily; the run
decisions are made at random or by taking the first option, in the sim, or
by a run policy (`sts2ai.runtrain`), on the sim's threaded loop
(`sts2ai.runloop`, docs/run-env.md).

    uv run python -m sts2ai.runplay runs/<run>/latest.pt --envs 256 --runs-per-env 2
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --choices first
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --show 8
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --search 256
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --win-starts 1 --win-holdout

With `--search N` the fights of `--search-kinds` (elites and bosses by
default) are played as the live pilot plays them (`sts2ai.play --search
N`): a turn search of N copies at every decision with more than one legal
action, the searches of every env in such a fight in one batch
(`Loop`). It is much slower per run. `--search-mode hybrid` (or
`race`) plays those fights with the exact search's best `--top` lines
picked by `--playouts` playouts each instead (`exactsearch.Hybrid`).

The `--envs` envs play the `--envs * --runs-per-env` runs from `--seed`
on, each env taking the next unplayed seed as its run ends, so every run
counts and none is a filler. With a run policy it also prints what the policy
picks at each kind of decision, and `--show` prints that many decisions,
drawn at random, with the policy's odds for each option.

With `--win-starts SHARE` that share of the runs starts at the entrance
of act 2 or 3 with a winner's player there (`runtrain --win-starts`), the
train players' or with `--win-holdout` the held-out players', and the
report adds a line per start point.

A run policy trained with the forecast (`sts2ai.forecast`) keeps its
calibration, and its map steps get the forecast filled in (`--forecast
CAL` gives one to a policy without). With `--forecast-log FILE` every map
step into an elite writes what the forecast read for that elite and how
the fight went, for `forecast calibrate`. `--elite-gate P` passes up an
elite at a map step whose elites' forecast win chance is under P, when
another option is open: the forecast as a rule, whatever the policy.

`--event-table ROWS` makes every event choice from a table instead of the
policy: of the options offered, the one winners took most often when it
was offered to them (`EventTable`), from imitation rows.

`--runs-out FILE` writes each counted run as a JSON line as it ends, the
command line first: how it ended, and its fights, decisions and decks
(`sts2ai.runtrace`), for `sts2ai.runreport` and `sts2ai.paired` to
compare arms run on the same seeds.

`--afterstate K` makes every decision but a map step by its afterstates
(`sts2ai.afterstate`): each option applied in the sim, random outcomes
averaged over K samples, scored by the forecast of the player it leaves,
the policy's pick on a tie. It needs the forecast's calibration. The
report counts how often the pick differs from the policy's per decision
kind and what a decision costs; `--show-afterstates N` prints N decisions
with each option's text and score. `--afterstate-kinds` and
`--afterstate-acts` limit it to some decision kinds or acts, the policy
deciding the rest.
`--lookahead W` adds the next act to the score: its elites and bosses,
all of each since the map does not show the next boss, at weight `W`
beside this act's, so a pick that pays off later counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from sts2ai import _sim, deckvalue
from sts2ai.afterstate import Scorer
from sts2ai.env import START_POINTS, End, Envs, RunFight, RunLayout
from sts2ai.forecast import Calibration, Forecaster
from sts2ai.imitation import Rows
from sts2ai.model import Net, for_play, load_policy
from sts2ai.runloop import Loop
from sts2ai.runmodel import RunPolicy, load_run_policy
from sts2ai.runtrace import KINDS, offered, option_text
from sts2ai.setups import TRACKER, sim_floor, split_runs

NAMES = _sim.run_names()
# Seconds between progress lines, the first after as long.
NOTE_SECONDS = 300


def row_text(L: RunLayout, floats: np.ndarray, ids: np.ndarray) -> str:
    """The global token of a run row in words."""
    g = floats[: L.global_floats]
    decision = NAMES["decision"][ids[0]]
    room = NAMES["room"][ids[1]]
    event = NAMES["event"][ids[5]] if ids[5] else ""
    deck = int((floats[L.f_deck : L.f_relics : L.deck_floats] > 0).sum())
    return f"{decision} in {room} {event} floor {g[3] * 49:.0f} hp {g[0]:.0%} of {g[1] * 100:.0f} gold {g[2] * 500:.0f} deck {deck}"


class Picks:
    """What a run policy picks, per decision kind; a path by its room."""

    def __init__(self, layout: RunLayout, show: int):
        self.L, self.show = layout, show
        self.rng = np.random.default_rng(0)
        self.picks: dict[str, Counter[str]] = defaultdict(Counter)

    def add(self, floats: np.ndarray, ids: np.ndarray, probs: np.ndarray, options: np.ndarray) -> None:
        L = self.L
        for k, o in enumerate(options):
            decision = NAMES["decision"][ids[k, 0]]
            i = L.i_options + o * L.option_ids
            kind = NAMES["option"][ids[k, i]]
            self.picks[decision][f"{kind} {NAMES['room'][ids[k, i + 4 + L.option_cards]]}" if kind == "Path" else kind] += 1
            # One decision in fifty, so the ones shown are not all Neow's.
            if self.show > 0 and self.rng.random() < 0.02:
                self.show -= 1
                print(row_text(L, floats[k], ids[k]))
                for j in offered(L, floats[k]):
                    mark = "*" if j == o else " "
                    print(f"  {mark} {probs[k, j]:6.1%}  {option_text(L, floats[k], ids[k], j)}")

    def report(self) -> None:
        print("run policy picks, by decision:")
        for decision, c in sorted(self.picks.items(), key=lambda kv: -kv[1].total()):
            top = ", ".join(f"{k} {v / c.total():.0%}" for k, v in c.most_common(8))
            print(f"  {decision:8s} {c.total():6d}: {top}")
        paths, rest = self.picks["Path"], self.picks["Rest"]
        print(
            f"  map steps into an elite {paths['Path Elite'] / max(paths.total(), 1):.1%}, "
            f"rest sites healed at {rest['RestHeal'] / max(rest['RestHeal'] + rest['RestSmith'], 1):.1%} (of heal or smith)"
        )


class EventTable:
    """Winners' pick rate for each event option, keyed by (event id, option
    key), from imitation rows. `choose` overrides the event decisions of a
    batch with the offered option of the highest rate; options offered to
    winners fewer than `min_offered` times never win, and an event with
    none known keeps the policy's pick."""

    def __init__(self, rows: Rows, layout: RunLayout, min_offered: int = 5):
        self.L = layout
        self.event = NAMES["decision"].index("Event")
        seen: Counter[tuple[int, int]] = Counter()
        picked: Counter[tuple[int, int]] = Counter()
        for k in np.flatnonzero(rows.kind == self.event):
            for j in offered(layout, rows.floats[k]):
                key = self.key(rows.ids[k], j)
                seen[key] += 1
                picked[key] += j == rows.option[k]
        self.rate = {key: picked[key] / n for key, n in seen.items() if n >= min_offered}
        self.decisions = self.changed = 0

    def key(self, ids: np.ndarray, j: int) -> tuple[int, int]:
        L = self.L
        return int(ids[5]), int(ids[L.i_options + j * L.option_ids + 5 + L.option_cards])

    def choose(self, floats: np.ndarray, ids: np.ndarray, options: np.ndarray) -> np.ndarray:
        options = options.copy()
        for k in np.flatnonzero(ids[:, 0] == self.event):
            rated = [(self.rate[key], j) for j in offered(self.L, floats[k]) if (key := self.key(ids[k], j)) in self.rate]
            self.decisions += 1
            if rated:
                best = max(rated)[1]
                self.changed += best != options[k]
                options[k] = best
        return options


def as_setup(start: dict) -> dict:
    """A run fight's start record (`log_fights`) as a setup line, its
    floor in the generator's numbering and the last act's second boss
    marked, as `sts2ai.setups` writes played runs' fights."""
    boss = start["room"] == "Boss"
    return {
        "start": start,
        "hp": start["hp"],
        "max_hp": start["max_hp"],
        "encounter": start["encounter"],
        "floor": sim_floor(start["floor"], boss),
        "second": boss and start["floor"] == 49,
        "game_floor": start["floor"],
    }


def play(
    combat: Net,
    device: torch.device,
    envs: Envs,
    seed: int,
    per_env: int,
    minutes: float,
    choices: str,
    run_policy: RunPolicy | None,
    picks: Picks | None,
    fights_out: TextIO | None = None,
    search: int = 0,
    search_kinds: frozenset[str] = frozenset({"Elite", "Boss"}),
    groups: int = 4,
    late_policy: RunPolicy | None = None,
    late_from_act: int = 2,
    win_starts: float = 0.0,
    win_runs: Path = TRACKER,
    win_holdout: bool = False,
    hybrid: tuple[int, int, int] | None = None,
    forecast: Forecaster | None = None,
    forecast_log: TextIO | None = None,
    elite_gate: float = 0.0,
    event_table: EventTable | None = None,
    afterstate: Scorer | None = None,
    show_afterstates: int = 0,
    afterstate_acts: frozenset[int] = frozenset(),
    easy_fights: bool = False,
    fight_ends: TextIO | None = None,
    runs_out: TextIO | None = None,
) -> tuple[list[End], list[RunFight], Loop, float]:
    """Plays the `per_env * envs.n` runs from `seed` on, or until `minutes`
    pass, on the threaded loop (`sts2ai.runloop`). Returns every fight that
    ended, every run that ended, the loop (its counts of combat steps, run
    decisions and searched decisions), and the seconds spent. With
    `fights_out`, each elite and boss fight as it starts is written there
    as a setup (`sts2ai.setups`' format, `evaluate --source setups` plays
    them), with `easy_fights` the weak and normal fights too. With
    `fight_ends`, each run fight's outcome as it ends (won, HP lost,
    potions), keyed like the start records by run seed and floor. With
    `runs_out`, each run's trace goes there as it ends (`sts2ai.runtrace`,
    written by the sim). With `late_policy`, that policy makes the
    decisions from act `late_from_act` on (1-based) and `run_policy` the
    ones before. `win_starts` of the runs start with winners' players
    (`Envs.use_winner_starts`), the held-out players' with `win_holdout`.
    `hybrid` (top lines, playouts, race size or 0) searches with
    `exactsearch.Hybrid`. With `forecast`, map steps get the forecast
    (`Loop`); with `forecast_log` too, each map step into an elite is
    written there with the fight's outcome. With `afterstate`, it makes
    every decision but a map step (`Scorer.choose`), and
    `show_afterstates` decisions are printed with each option's score."""
    if win_starts > 0:
        held = envs.use_winner_starts(split_runs(win_runs, win_holdout), win_starts)
        print("winners' starts: " + ", ".join(f"{n} at {p}" for p, n in zip(START_POINTS, held) if n))
    # Before the runs start: with the sim deciding, each starts its first fight at once.
    envs.sim.log_fights(fights_out is not None, easy_fights)
    envs.sim.trace_runs(runs_out is not None)
    total = per_env * envs.n
    envs.use_runs(seed, choices="caller" if run_policy else choices, last=seed + total)
    fights: list[End] = []
    runs: list[RunFight] = []
    L = RunLayout.load()
    # What the forecast read at each env's step into an elite, by encounter.
    pending: dict[int, dict[str, tuple[float, float, float]]] = {}
    # One scored decision in twenty is shown, so the ones shown are not all Neow's.
    shown, show_rng = [0], np.random.default_rng(1)

    @torch.no_grad()
    def decide(waiting: list[int], floats: np.ndarray, ids: np.ndarray) -> np.ndarray:
        f, i = torch.from_numpy(floats).to(device), torch.from_numpy(ids).to(device)
        logits, _ = run_policy(f, i)
        if late_policy is not None:
            # The global token's act id: 1 and 2 are act 1's two acts, then
            # one per act (runobs.rs ACTS).
            late = i[:, 2] >= late_from_act + 1
            if late.any():
                logits[late] = late_policy(f[late], i[late])[0]
        if elite_gate > 0:
            room = i[:, L.i_options + 4 + L.option_cards : L.i_options + L.max_options * L.option_ids : L.option_ids]
            elite = (room == NAMES["room"].index("Elite")) & (i[:, :1] == NAMES["decision"].index("Path"))
            unsure = f[:, L.f_forecast] < elite_gate
            gate = elite & unsure[:, None] & (logits[:, : L.max_options] > -1e8).logical_and(~elite).any(1, keepdim=True)
            logits = logits.masked_fill(gate, -1e9)
        options = logits.argmax(1).cpu().numpy()
        if event_table is not None:
            options = event_table.choose(floats, ids, options)
        if afterstate is not None:
            policy = options
            # The global token's act id: 1 and 2 are act 1's two acts, then one
            # per act (runobs.rs ACTS); `afterstate_acts` counts from 1.
            counted = [k for k in range(len(waiting)) if not afterstate_acts or max(int(ids[k, 2]) - 1, 1) in afterstate_acts]
            options, score = afterstate.choose(waiting, ids, policy, counted)
            for k in np.flatnonzero(np.isfinite(score).any(1)):
                if shown[0] >= show_afterstates or show_rng.random() >= 0.05:
                    continue
                shown[0] += 1
                print(row_text(L, floats[k], ids[k]))
                for j in np.flatnonzero(np.isfinite(score[k])):
                    mark = ("*" if j == options[k] else " ") + ("p" if j == policy[k] else " ")
                    name, via = afterstate.option(k, j)
                    text = option_text(L, floats[k], ids[k], j)
                    if ids[k, 0] == NAMES["decision"].index("Event"):
                        text = f"Event {name}"
                    print(f"  {mark} {score[k, j]:5.2f}  {text}" + (f"  via {' > '.join(via)}" if via else ""))
        if forecast_log is not None and loop.read is not None:
            read = loop.read
            for k, env in enumerate(waiting):
                i = L.i_options + options[k] * L.option_ids
                if NAMES["option"][ids[k, i]] == "Path" and NAMES["room"][ids[k, i + 4 + L.option_cards]] == "Elite":
                    here = read.row == k
                    pending[env] = {e: (float(v), float(h), float(w)) for e, v, h, w, m in zip(read.encounter, read.value, read.hp, read.win, here) if m}
        if picks:
            picks.add(floats, ids, torch.softmax(logits, 1).cpu().numpy(), options)
        return options

    loop = Loop(combat, device, envs, search, search_kinds, groups, seed, hybrid, forecast)
    start = time.perf_counter()
    next_note = start + NOTE_SECONDS
    while time.perf_counter() - start < minutes * 60:
        taken = loop.step(decide)
        if fights_out is not None:
            for record in taken.starts:
                fight = json.loads(record)
                if easy_fights or KINDS[fight["encounter"]] in ("Elite", "Boss"):
                    fights_out.write(json.dumps(as_setup(fight)) + "\n")
        fights += taken.ends
        runs += [run for _, run in taken.ended]
        for e in taken.ends:
            if fight_ends is not None and e.run is not None:
                outcome = {"seed": e.run.seed, "floor": e.run.floor, "act": e.run.act, "encounter": e.encounter, "kind": e.kind}
                outcome |= {"won": e.won, "hp_lost": e.hp_lost, "hp_frac": e.hp_frac, "potions_used": e.potions_used, "steps": e.steps}
                fight_ends.write(json.dumps(outcome) + "\n")
            at = pending.pop(e.env, None)
            if forecast_log is not None and at and e.kind == "Elite" and e.encounter in at and e.run:
                value, hp, win = at[e.encounter]
                record = {
                    "act": e.run.act,
                    "encounter": e.encounter,
                    "value": value,
                    "hp": hp,
                    "win": win,
                    "pool_win": float(np.mean([v[2] for name, v in at.items() if name.endswith("Elite")])),
                    "won": e.won,
                    "kept": e.hp_frac,
                }
                forecast_log.write(json.dumps(record) + "\n")
        if runs_out is not None:
            for line in taken.traces:
                runs_out.write(line + "\n")
        if taken.active == 0:
            break
        if (now := time.perf_counter()) >= next_note:
            next_note = now + NOTE_SECONDS
            finished = len(runs)
            eta = (now - start) / max(finished, 1) * (total - finished)
            print(f"{(now - start) / 60:.0f} min: {finished} of {total} runs finished, about {eta / 60:.0f} min to go", flush=True)
    seconds = time.perf_counter() - start
    loop.stop()
    return fights, runs, loop, seconds


def report(fights: list[End], runs: list[RunFight], seed: int, last: int, loop: Loop, n: int, seconds: float) -> None:
    counted = [r for r in runs if r.seed < last]
    seeds = {r.seed for r in counted}
    in_counted = [e for e in fights if e.run and e.run.seed in seeds]
    print(f"{len(counted)} of {last - seed} runs finished, {len(in_counted)} fights in them")
    if not counted:
        return
    outcome = Counter(r.end if not r.end.startswith("stuck") else "stuck" for r in counted)
    print("  " + "  ".join(f"{k} {v}" for k, v in outcome.most_common()))
    for why, k in Counter(r.end for r in counted if r.end.startswith("stuck")).most_common(5):
        print(f"    {k:4d} {why}")

    floors = np.array([r.floor for r in counted])
    print(f"floor reached: mean {floors.mean():.1f}, median {np.median(floors):.0f}, max {floors.max()}")
    by_act = Counter("won" if r.end == "won" else f"act {r.act + 1}" for r in counted)
    print("  ended in: " + "  ".join(f"{k} {v} ({v / len(counted):.0%})" for k, v in sorted(by_act.items())))
    reached = lambda act: np.mean([r.act >= act for r in counted])
    print(f"  reached act 2 {reached(1):.1%}, act 3 {reached(2):.1%}; won {np.mean([r.end == 'won' for r in counted]):.1%}")
    print("  floor quantiles: " + "  ".join(f"p{q} {np.percentile(floors, q):.0f}" for q in (10, 25, 50, 75, 90)))

    for start in sorted({r.start for r in counted if r.start is not None}):
        for source in sorted({r.source for r in counted if r.start == start}):
            some = [r for r in counted if r.start == start and r.source == source]
            won = np.mean([r.end == "won" for r in some])
            print(
                f"  from {START_POINTS[start]} {source}: {len(some)} runs, won {won:.1%}, floor {np.mean([r.floor for r in some]):.1f}, reached act 3 {np.mean([r.act >= 2 for r in some]):.1%}"
            )
    decks = np.array([r.deck for r in counted])
    print(f"deck at the end: mean {decks.mean():.1f}, median {np.median(decks):.0f}, max {decks.max()}")

    print("combat win rate by act and kind (wins/fights):")
    table: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for e in in_counted:
        table[(e.run.act, e.kind)].append(e.won)
    for act in sorted({a for a, _ in table}):
        cells = [(k, table[(act, k)]) for k in ("Weak", "Normal", "Elite", "Boss") if (act, k) in table]
        print(f"  act {act + 1}: " + "   ".join(f"{k.lower():6s} {np.mean(w):6.1%} {sum(w):5d}/{len(w):<5d}" for k, w in cells))
    elites = [e.won for e in in_counted if e.kind == "Elite"]
    bosses = " ".join(f"act {a + 1} {sum(table[(a, 'Boss')])}" for a in range(3))
    print(f"  elites won {np.mean(elites or [0]):.1%} of {len(elites)}; bosses beaten: {bosses} (of {len(counted)} runs)")
    for kind in ("Elite", "Boss"):
        act1 = [e for e in in_counted if e.run.act == 0 and e.kind == kind and e.won]
        if act1:
            print(
                f"  act 1 {kind.lower()}s won: HP lost {np.mean([e.hp_lost for e in act1]):.1%} of max,"
                f" left {np.mean([e.hp_frac for e in act1]):.1%} (mean of {len(act1)})"
            )
    losses = Counter(e.encounter for e in in_counted if not e.won)
    print("  most runs lost to: " + ", ".join(f"{enc} {k}" for enc, k in losses.most_common(8)))

    rate = f"{loop.combat_steps / seconds:,.0f} combat steps/s, {loop.decisions / seconds:,.0f} run decisions/s, {len(runs) / seconds * 3600:,.0f} runs/hour"
    searched = f", {loop.searched} decisions searched" if loop.searching else ""
    print(f"throughput: {rate} ({seconds:.0f} s, {n} envs{searched})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="combat checkpoint")
    ap.add_argument("--old-vocab", type=Path, default=None, help="vocab.txt the checkpoint was trained with, if it predates the current sim")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--runs-per-env", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=10.0, help="stop early after this long")
    ap.add_argument("--seed", type=int, default=0, help="the first run's seed index")
    ap.add_argument("--choices", choices=("random", "first"), default="random", help="run decisions made in the sim, without --run-policy")
    ap.add_argument("--run-policy", type=Path, default=None, help="run policy checkpoint (sts2ai.runtrain), greedy")
    ap.add_argument("--show", type=int, default=0, help="print this many run decisions with the policy's odds")
    ap.add_argument("--fights-out", type=Path, default=None, help="write each elite and boss fight's start here as a setup")
    ap.add_argument("--easy-fights", action="store_true", help="--fights-out also writes the weak and normal fights")
    ap.add_argument("--fight-ends", type=Path, default=None, help="write each run fight's outcome here (won, HP lost), keyed by run seed and floor")
    ap.add_argument("--no-drain", action="store_true", help="accepted for old command lines; the loop answers each env's decisions as they come")
    ap.add_argument("--search", type=int, default=0, help="turn search with this many sim copies per decision, as the pilot (0: greedy)")
    ap.add_argument("--search-kinds", default="Elite,Boss", help="fight kinds searched (Weak,Normal,Elite,Boss); the rest greedy")
    ap.add_argument("--search-mode", choices=("copies", "hybrid", "race"), default="copies", help="copies: --search N copies; hybrid, race: exactsearch.Hybrid")
    ap.add_argument("--top", type=int, default=10, help="lines the hybrid plays out")
    ap.add_argument("--playouts", type=int, default=32, help="playouts per line in the hybrid")
    ap.add_argument("--race", type=int, default=8, help="playouts per round with --search-mode race")
    ap.add_argument("--late-policy", type=Path, default=None, help="run policy for the decisions from --late-from-act on")
    ap.add_argument("--late-from-act", type=int, default=2)
    ap.add_argument("--groups", type=int, default=4, help="draw-pile shuffles the search copies are split over")
    ap.add_argument("--win-starts", type=float, default=0.0, help="share of runs started from winners' act 2 and 3 entrances")
    ap.add_argument("--win-runs", type=Path, default=TRACKER, help="winners' history files; the train players' are used")
    ap.add_argument("--win-holdout", action="store_true", help="start with the held-out players' winners instead")
    ap.add_argument("--forecast-log", type=Path, default=None, help="write each map step into an elite with its forecast and outcome here")
    ap.add_argument("--forecast", type=Path, default=None, help="forecast calibration for a run policy that has none")
    ap.add_argument("--elite-gate", type=float, default=0.0, help="pass up elites whose forecast win chance is under this")
    ap.add_argument("--event-table", type=Path, default=None, help="imitation rows (.npz); event choices take winners' most picked option")
    ap.add_argument("--runs-out", type=Path, default=None, help="write each counted run's trace here as a JSON line (sts2ai.runtrace)")
    ap.add_argument("--deckvalue", type=Path, default=None, help="score afterstates with this deck value network (sts2ai.deckvalue) instead of the forecast")
    ap.add_argument("--afterstate", type=int, default=0, help="make every decision but a map step by its afterstates, K samples each (needs the forecast)")
    ap.add_argument("--afterstate-depth", type=int, default=3, help="sub-decisions an afterstate opens on the way, at most")
    ap.add_argument("--afterstate-nodes", type=int, default=256, help="branches an afterstate plays per option and sample, at most")
    ap.add_argument("--afterstate-acts", default="", help="acts (1-3, comma separated) whose decisions are made by afterstate; empty: every act")
    ap.add_argument("--afterstate-kinds", default="", help="decision kinds made by afterstate, comma separated (Event,Rest,...); empty: all but map steps")
    ap.add_argument(
        "--lookahead", type=float, default=0.0, help="afterstates also score the next act's elites and bosses, pooled, at this weight beside this act's"
    )
    ap.add_argument("--show-afterstates", type=int, default=0, help="print this many scored decisions with each option's score")
    args = ap.parse_args()
    # Ids the combat checkpoint never saw get fresh rows (`vocab.remap_state`),
    # drawn from here: unseeded, two plays of one seed differ.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combat = for_play(load_policy(args.checkpoint, device, args.old_vocab).eval())
    run_policy, run_ck = load_run_policy(args.run_policy, device) if args.run_policy else (None, {})
    if run_policy is not None:
        run_policy.eval()
    calibration = Calibration.load(args.forecast) if args.forecast else Calibration(**run_ck["forecast"]) if run_ck.get("forecast") else None
    if (args.elite_gate > 0 or args.afterstate > 0) and calibration is None:
        ap.error("--elite-gate and --afterstate need the forecast: a run policy trained with it, or --forecast")
    if args.afterstate > 0 and run_policy is None:
        ap.error("--afterstate needs --run-policy for the map steps")
    forecast = Forecaster(combat, device, calibration) if calibration or args.forecast_log else None
    forecast_log = args.forecast_log.open("w") if args.forecast_log else None
    envs = Envs(args.envs, seed=args.seed)
    envs.sim.set_lookahead(args.lookahead)
    picks = Picks(RunLayout.load(), args.show) if run_policy else None
    fights_out = args.fights_out.open("w") if args.fights_out else None
    fight_ends = args.fight_ends.open("w") if args.fight_ends else None
    last = args.seed + args.runs_per_env * args.envs
    runs_out = args.runs_out.open("w") if args.runs_out else None
    if runs_out is not None:
        runs_out.write(json.dumps({"argv": sys.argv}) + "\n")
    event_table = EventTable(Rows.load(args.event_table), RunLayout.load()) if args.event_table else None
    afterstate = (
        Scorer(
            envs,
            forecast,
            args.afterstate,
            args.afterstate_depth,
            args.afterstate_nodes,
            frozenset(filter(None, args.afterstate_kinds.split(","))),
            deckvalue.load(args.deckvalue, device) if args.deckvalue else None,
        )
        if args.afterstate > 0
        else None
    )
    fights, runs, loop, seconds = play(
        combat,
        device,
        envs,
        args.seed,
        args.runs_per_env,
        args.minutes,
        args.choices,
        run_policy,
        picks,
        fights_out,
        args.search,
        frozenset(args.search_kinds.split(",")),
        args.groups,
        load_run_policy(args.late_policy, device)[0].eval() if args.late_policy else None,
        args.late_from_act,
        args.win_starts,
        args.win_runs,
        args.win_holdout,
        None if args.search_mode == "copies" else (args.top, args.playouts, args.race if args.search_mode == "race" else 0),
        forecast,
        forecast_log,
        args.elite_gate,
        event_table,
        afterstate,
        args.show_afterstates,
        frozenset(int(a) for a in args.afterstate_acts.split(",") if a),
        easy_fights=args.easy_fights,
        fight_ends=fight_ends,
        runs_out=runs_out,
    )
    for f in (fights_out, fight_ends, forecast_log, runs_out):
        if f is not None:
            f.close()
    report(fights, runs, args.seed, last, loop, args.envs, seconds)
    if picks:
        picks.report()
    if event_table:
        print(f"event table: {event_table.changed} of {event_table.decisions} event decisions differ from the policy's pick")
    if afterstate:
        afterstate.report()


if __name__ == "__main__":
    main()
