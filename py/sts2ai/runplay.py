"""Play whole runs: a combat checkpoint plays the fights greedily; the run
decisions are made at random or by taking the first option, in the sim, or
by a run policy (`sts2ai.runtrain`) through `step_run` (docs/run-env.md).

    uv run python -m sts2ai.runplay runs/<run>/latest.pt --envs 256 --runs-per-env 2
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --choices first
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --show 8
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --search 256
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --win-starts 1 --win-holdout

With `--search N` the fights of `--search-kinds` (elites and bosses by
default) are played as the live pilot plays them (`sts2ai.play --search
N`): a turn search of N copies at every decision with more than one legal
action, the searches of every env in such a fight in one batch
(`RunLoop`). It is much slower per run.

Every env plays runs back to back. The numbers cover each env's first
`--runs-per-env` runs, so long runs count as often as short ones; the
envs that finish early keep playing, and those extra runs only count
toward throughput. With a run policy it also prints what the policy
picks at each kind of decision, and `--show` prints that many decisions,
drawn at random, with the policy's odds for each option.

With `--win-starts SHARE` that share of the runs starts at the entrance
of act 2 or 3 with a winner's player there (`runtrain --win-starts`), the
train players' or with `--win-holdout` the held-out players', and the
report adds a line per start point.

A run policy trained with the forecast (`sts2ai.forecast`) keeps its
calibration, and its map steps get the forecast filled in. With
`--forecast-log FILE` every map step into an elite writes what the
forecast read for that elite and how the fight went, for `forecast
calibrate`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import START_POINTS, End, Envs, RunFight, RunLayout
from sts2ai.forecast import Calibration, Forecaster
from sts2ai.model import Policy, load_policy
from sts2ai.runmodel import RunPolicy, load_run_policy
from sts2ai.runtrain import RunLoop
from sts2ai.setups import TRACKER, sim_floor, split_runs

NAMES = _sim.run_names()
CARDS = ["-"] + _sim.game_ids()["card"]
POTIONS = ["-"] + _sim.game_ids()["potion"]


def option_text(L: RunLayout, floats: np.ndarray, ids: np.ndarray, k: int) -> str:
    """Option token `k` of a run row in words: its kind and what it names."""
    i, f = L.i_options + k * L.option_ids, L.f_options + k * L.option_floats
    C = L.option_cards
    kind = NAMES["option"][ids[i]]
    parts = [kind]
    for c in range(C):
        if ids[i + 1 + c]:
            parts.append(CARDS[ids[i + 1 + c]] + ("+" if floats[f + 1 + c] else ""))
    if ids[i + 2 + C]:
        parts.append(NAMES["relic"][ids[i + 2 + C]])
    if ids[i + 3 + C]:
        parts.append(POTIONS[ids[i + 3 + C]])
    if ids[i + 4 + C]:
        parts.append(NAMES["room"][ids[i + 4 + C]])
    if kind == "Event":
        parts.append(f"key#{ids[i + 5 + C]}")
    if price := floats[f + 2 + C]:
        parts.append(f"{price * 100:.0f}g")
    if kind == "Path":
        m = floats[f + 3 + C : f + L.option_floats] * 8
        parts.append(f"elites {m[10]:.0f}-{m[11]:.0f} rests {m[6]:.0f}-{m[7]:.0f} ?{m[0]:.0f}-{m[1]:.0f}")
    return " ".join(parts)


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
                present = [j for j in range(L.max_options) if floats[k, L.f_options + j * L.option_floats]]
                print(row_text(L, floats[k], ids[k]))
                for j in present:
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
    combat: Policy,
    device: torch.device,
    envs: Envs,
    seed: int,
    per_env: int,
    minutes: float,
    choices: str,
    run_policy: RunPolicy | None,
    picks: Picks | None,
    drain: bool = True,
    fights_out: TextIO | None = None,
    search: int = 0,
    search_kinds: frozenset[str] = frozenset({"Elite", "Boss"}),
    groups: int = 4,
    late_policy: RunPolicy | None = None,
    late_from_act: int = 2,
    win_starts: float = 0.0,
    win_runs: Path = TRACKER,
    win_holdout: bool = False,
    forecast: Forecaster | None = None,
    forecast_log: TextIO | None = None,
) -> tuple[list[End], list[RunFight], RunLoop, float]:
    """Plays until each env has finished `per_env` runs or `minutes` pass.
    Returns every fight that ended, every run that ended, the loop (its
    counts of combat batch steps, run decisions and searched decisions),
    and the seconds spent. With `fights_out`,
    each elite and boss fight as it starts is written there as a setup
    (`sts2ai.setups`' format, `evaluate --source setups` plays them).
    With `late_policy`, that policy makes the decisions from act
    `late_from_act` on (1-based) and `run_policy` the ones before.
    `win_starts` of the runs start with winners' players (`Envs.use_winner_starts`),
    the held-out players' with `win_holdout`. With `forecast`, map steps
    get the forecast (`RunLoop`); with `forecast_log` too, each map step
    into an elite is written there with the fight's outcome."""
    if win_starts > 0:
        held = envs.use_winner_starts(split_runs(win_runs, win_holdout), win_starts)
        print("winners' starts: " + ", ".join(f"{n} at {p}" for p, n in zip(START_POINTS, held) if n))
    envs.use_runs(seed, choices="caller" if run_policy else choices)
    left = set(range(seed, seed + per_env * envs.n))
    fights: list[End] = []
    runs: list[RunFight] = []
    L = RunLayout.load()
    # What the forecast read at each env's step into an elite, by encounter.
    pending: dict[int, dict[str, tuple[float, float, float]]] = {}

    def ended(_: int, run: RunFight) -> None:
        runs.append(run)
        left.discard(run.seed)

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
        options = logits.argmax(1).cpu().numpy()
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

    loop = RunLoop(combat, device, envs, drain, search, search_kinds, groups, seed, forecast)
    start = time.perf_counter()
    envs.sim.log_fights(fights_out is not None)
    while left and time.perf_counter() - start < minutes * 60:
        ended_now = loop.step(decide, ended)
        fights += ended_now
        for e in ended_now:
            at = pending.pop(e.env, None)
            if forecast_log is not None and at and e.kind == "Elite" and e.encounter in at and e.run and e.run.seed < seed + per_env * envs.n:
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
        if fights_out is not None:
            for record in envs.sim.take_fights():
                fights_out.write(json.dumps(as_setup(json.loads(record))) + "\n")
    return fights, runs, loop, time.perf_counter() - start


def report(fights: list[End], runs: list[RunFight], seed: int, last: int, loop: RunLoop, n: int, seconds: float) -> None:
    counted = [r for r in runs if r.seed < last]
    seeds = {r.seed for r in counted}
    in_counted = [e for e in fights if e.run and e.run.seed in seeds]
    print(f"{len(counted)} of {last - seed} runs finished ({len(runs)} in all), {len(in_counted)} fights in them")
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
            print(f"  from {START_POINTS[start]} {source}: {len(some)} runs, won {won:.1%}, floor {np.mean([r.floor for r in some]):.1f}, reached act 3 {np.mean([r.act >= 2 for r in some]):.1%}")
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
    losses = Counter(e.encounter for e in in_counted if not e.won)
    print("  most runs lost to: " + ", ".join(f"{enc} {k}" for enc, k in losses.most_common(8)))

    rate = f"{loop.combat_steps * n / seconds:,.0f} combat steps/s, {loop.decisions / seconds:,.0f} run decisions/s, {len(runs) / seconds * 3600:,.0f} runs/hour"
    searched = f", {loop.searched} decisions searched" if loop.search else ""
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
    ap.add_argument("--no-drain", action="store_true", help="answer one round of run decisions per combat step, not all (RunLoop)")
    ap.add_argument("--search", type=int, default=0, help="turn search with this many sim copies per decision, as the pilot (0: greedy)")
    ap.add_argument("--search-kinds", default="Elite,Boss", help="fight kinds searched (Weak,Normal,Elite,Boss); the rest greedy")
    ap.add_argument("--late-policy", type=Path, default=None, help="run policy for the decisions from --late-from-act on")
    ap.add_argument("--late-from-act", type=int, default=2)
    ap.add_argument("--groups", type=int, default=4, help="draw-pile shuffles the search copies are split over")
    ap.add_argument("--win-starts", type=float, default=0.0, help="share of runs started from winners' act 2 and 3 entrances")
    ap.add_argument("--win-runs", type=Path, default=TRACKER, help="winners' history files; the train players' are used")
    ap.add_argument("--win-holdout", action="store_true", help="start with the held-out players' winners instead")
    ap.add_argument("--forecast-log", type=Path, default=None, help="write each map step into an elite with its forecast and outcome here")
    args = ap.parse_args()
    # Ids the combat checkpoint never saw get fresh rows (`vocab.remap_state`),
    # drawn from here: unseeded, two plays of one seed differ.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combat = load_policy(args.checkpoint, device, args.old_vocab).eval()
    run_policy, run_ck = load_run_policy(args.run_policy, device) if args.run_policy else (None, {})
    if run_policy is not None:
        run_policy.eval()
    calibration = Calibration(**run_ck["forecast"]) if run_ck.get("forecast") else None
    forecast = Forecaster(combat, device, calibration) if calibration or args.forecast_log else None
    forecast_log = args.forecast_log.open("w") if args.forecast_log else None
    envs = Envs(args.envs, seed=args.seed)
    picks = Picks(RunLayout.load(), args.show) if run_policy else None
    fights_out = args.fights_out.open("w") if args.fights_out else None
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
        not args.no_drain,
        fights_out,
        args.search,
        frozenset(args.search_kinds.split(",")),
        args.groups,
        load_run_policy(args.late_policy, device)[0].eval() if args.late_policy else None,
        args.late_from_act,
        args.win_starts,
        args.win_runs,
        args.win_holdout,
        forecast,
        forecast_log,
    )
    if fights_out is not None:
        fights_out.close()
    if forecast_log is not None:
        forecast_log.close()
    report(fights, runs, args.seed, args.seed + args.runs_per_env * args.envs, loop, args.envs, seconds)
    if picks:
        picks.report()


if __name__ == "__main__":
    main()
