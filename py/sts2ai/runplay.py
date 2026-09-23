"""Play whole runs: a combat checkpoint plays the fights greedily; the run
decisions are made at random or by taking the first option, in the sim, or
by a run policy (`sts2ai.runtrain`) through `step_run` (docs/run-env.md).

    uv run python -m sts2ai.runplay runs/<run>/latest.pt --envs 256 --runs-per-env 2
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --choices first
    uv run python -m sts2ai.runplay runs/<run>/latest.pt --run-policy runs/run-1/latest.pt --show 8

Every env plays runs back to back. The numbers cover each env's first
`--runs-per-env` runs, so long runs count as often as short ones; the
envs that finish early keep playing, and those extra runs only count
toward throughput. With a run policy it also prints what the policy
picks at each kind of decision, and `--show` prints that many decisions,
drawn at random, with the policy's odds for each option.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import End, Envs, RunFight, RunLayout
from sts2ai.model import Policy, load_policy
from sts2ai.runmodel import RunPolicy, load_run_policy
from sts2ai.runtrain import RunLoop

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
) -> tuple[list[End], list[RunFight], int, int, float]:
    """Plays until each env has finished `per_env` runs or `minutes` pass.
    Returns every fight that ended, every run that ended, the combat batch
    steps and run decisions taken, and the seconds spent."""
    envs.use_runs(seed, choices="caller" if run_policy else choices)
    left = set(range(seed, seed + per_env * envs.n))
    fights: list[End] = []
    runs: list[RunFight] = []

    def ended(_: int, run: RunFight) -> None:
        runs.append(run)
        left.discard(run.seed)

    @torch.no_grad()
    def decide(_: list[int], floats: np.ndarray, ids: np.ndarray) -> np.ndarray:
        logits, _ = run_policy(torch.from_numpy(floats).to(device), torch.from_numpy(ids).to(device))
        options = logits.argmax(1).cpu().numpy()
        if picks:
            picks.add(floats, ids, torch.softmax(logits, 1).cpu().numpy(), options)
        return options

    loop = RunLoop(combat, device, envs, drain)
    start = time.perf_counter()
    while left and time.perf_counter() - start < minutes * 60:
        fights += loop.step(decide, ended)
    return fights, runs, loop.combat_steps, loop.decisions, time.perf_counter() - start


def report(fights: list[End], runs: list[RunFight], seed: int, last: int, steps: int, decisions: int, n: int, seconds: float) -> None:
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
    print("  floor quantiles: " + "  ".join(f"p{q} {np.percentile(floors, q):.0f}" for q in (10, 25, 50, 75, 90)))

    decks = np.array([r.deck for r in counted])
    print(f"deck at the end: mean {decks.mean():.1f}, median {np.median(decks):.0f}, max {decks.max()}")

    print("combat win rate by act and kind (wins/fights):")
    table: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for e in in_counted:
        table[(e.run.act, e.kind)].append(e.won)
    for act in sorted({a for a, _ in table}):
        cells = [(k, table[(act, k)]) for k in ("Weak", "Normal", "Elite", "Boss") if (act, k) in table]
        print(f"  act {act + 1}: " + "   ".join(f"{k.lower():6s} {np.mean(w):6.1%} {sum(w):5d}/{len(w):<5d}" for k, w in cells))
    losses = Counter(e.encounter for e in in_counted if not e.won)
    print("  most runs lost to: " + ", ".join(f"{enc} {k}" for enc, k in losses.most_common(8)))

    rate = f"{steps * n / seconds:,.0f} combat steps/s, {decisions / seconds:,.0f} run decisions/s, {len(runs) / seconds * 3600:,.0f} runs/hour"
    print(f"throughput: {rate} ({seconds:.0f} s, {n} envs)")


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
    ap.add_argument("--no-drain", action="store_true", help="answer one round of run decisions per combat step, not all (RunLoop)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combat = load_policy(args.checkpoint, device, args.old_vocab).eval()
    run_policy = load_run_policy(args.run_policy, device)[0].eval() if args.run_policy else None
    envs = Envs(args.envs, seed=args.seed)
    picks = Picks(RunLayout.load(), args.show) if run_policy else None
    fights, runs, steps, decisions, seconds = play(combat, device, envs, args.seed, args.runs_per_env, args.minutes, args.choices, run_policy, picks, not args.no_drain)
    report(fights, runs, args.seed, args.seed + args.runs_per_env * args.envs, steps, decisions, args.envs, seconds)
    if picks:
        picks.report()


if __name__ == "__main__":
    main()
