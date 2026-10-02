"""Afterstate scoring of run decisions (docs/run-env.md, Afterstates):
instead of asking the run policy what a winner would pick, apply each
option in the sim and score the player it leaves with the forecast.

The sim expands each option of the decisions a batch waits at
(`Envs.afterstates`, every decision but a map step): the segment replayed
with the option, everything random reseeded from a sample index the
moment the option is chosen, settled at the first map step, fight or end,
with sub-decisions in the same room (which card to smith, an event's next
page, a shop after a purchase) opened under caps. It keeps the trees and
hands back the distinct settled states' forecast fights, which the combat
value head reads in as few batches as it takes (`Forecaster.values`); the
sim scores the trees from those values (`Envs.afterstate_scores`): a
settled state's score is the act's elites' mean calibrated win chance
plus the boss's (1 for a run won in the option, 0 for one that died or
is stuck), a sub-decision takes its best option, a random outcome is the
mean over its samples. The decision takes the best option, the policy's
pick on a tie.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

import numpy as np

from sts2ai import _sim
from sts2ai.env import Envs
from sts2ai.forecast import Forecaster

DECISIONS = _sim.run_names()["decision"]


class Scorer:
    """Scores and makes run decisions from their afterstates, and counts
    where its picks differ from the policy's and what the caps shut.
    With `kinds` only decisions of those kinds (`DECISIONS` names) are
    made by afterstate; the others keep the policy's pick."""

    def __init__(
        self,
        envs: Envs,
        forecaster: Forecaster,
        samples: int,
        depth: int = 3,
        nodes: int = 256,
        kinds: frozenset[str] = frozenset(),
        state_scorer: Callable[[list[dict]], np.ndarray] | None = None,
    ):
        assert forecaster.calibration is not None, "afterstates need a calibration"
        self.envs, self.forecaster, self.samples, self.depth, self.nodes = envs, forecaster, samples, depth, nodes
        # With a state scorer (`deckvalue.state_scores`,
        # `runvalue.state_scores`), settled states are scored from their run
        # records instead of the forecast's openings.
        self.state_scorer = state_scorer
        self.kinds = kinds
        self.L = forecaster.L
        self.decisions: Counter[str] = Counter()
        self.differs: Counter[str] = Counter()
        self.capped: Counter[str] = Counter()
        self.leaves = self.states = self.rows = 0
        self.sim_seconds = self.head_seconds = self.all_seconds = 0.0
        self.scored: dict[int, int] = {}

    def score(self, waiting: list[int]) -> np.ndarray:
        """Every option of the decisions `waiting` wait at: `score[k, j]` of
        row `k`'s option token `j`, NaN where there is none."""
        t = time.perf_counter()
        floats, ids, leaves, states = self.envs.afterstates(waiting, self.samples, self.depth, self.nodes, rows=self.state_scorer is None)
        self.sim_seconds += time.perf_counter() - t
        if self.state_scorer is not None:
            t = time.perf_counter()
            scores = self.state_scorer(self.envs.afterstate_runs())
            self.head_seconds += time.perf_counter() - t
            score, capped = self.envs.afterstate_scores_given(scores)
            self.capped.update(capped)
            self.leaves += leaves
            self.states += states
            return score
        t = time.perf_counter()
        values = self.forecaster.values(floats, ids)
        self.head_seconds += time.perf_counter() - t
        t = time.perf_counter()
        score, capped = self.envs.afterstate_scores(values, self.forecaster.calibration.win)
        self.sim_seconds += time.perf_counter() - t
        self.capped.update(capped)
        self.leaves += leaves
        self.states += states
        self.rows += len(floats)
        return score

    def choose(self, waiting: list[int], ids: np.ndarray, policy: np.ndarray, rows: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """`policy`'s picks with every scored decision's replaced by its
        best option, the policy's pick on an exact tie, and the scores by
        `waiting` row (NaN for a decision not scored). Only `rows` (all by
        default) of `kinds` are scored; the sim builds no afterstates for
        the others, whose picks stay the policy's."""
        t = time.perf_counter()
        path = DECISIONS.index("Path")
        rows = [k for k in (range(len(waiting)) if rows is None else rows) if ids[k, 0] != path and (not self.kinds or DECISIONS[ids[k, 0]] in self.kinds)]
        # Row `k` of `waiting` is row `scored[k]` of the batch the sim scored.
        self.scored = {k: i for i, k in enumerate(rows)}
        score = np.full((len(waiting), self.L.max_options), np.nan, dtype=np.float32)
        score[rows] = self.score([waiting[k] for k in rows])
        options = policy.copy()
        for k in np.flatnonzero(np.isfinite(score).any(1)):
            kind = DECISIONS[ids[k, 0]]
            best = np.nanmax(score[k])
            tied = np.flatnonzero(score[k] == best)
            options[k] = policy[k] if policy[k] in tied else tied[0]
            self.decisions[kind] += 1
            self.differs[kind] += options[k] != policy[k]
        self.all_seconds += time.perf_counter() - t
        return options, score

    def option(self, k: int, j: int) -> tuple[str, list[str]]:
        """Option `j` of row `k` of the batch `choose` scored last, in
        words, and the sub-decisions' options on its best path."""
        return self.envs.sim.afterstate_option(self.scored[k], j)

    def report(self) -> None:
        n = max(sum(self.decisions.values()), 1)
        print(f"afterstates: {n} decisions scored, {self.differs.total()} picked otherwise than the policy ({self.differs.total() / n:.0%})")
        for kind, d in sorted(self.decisions.items(), key=lambda kv: -kv[1]):
            print(f"  {kind:8s} {d:6d}: differs {self.differs[kind] / d:.0%}")
        print(
            f"  cost per decision: {self.leaves / n:.0f} leaves, {self.states / n:.0f} states, {self.rows / n:.0f} combat rows,"
            f" {self.sim_seconds / n * 1000:.1f} ms sim, {self.head_seconds / n * 1000:.1f} ms value head,"
            f" {(self.all_seconds - self.sim_seconds - self.head_seconds) / n * 1000:.1f} ms the rest"
        )
        if self.capped:
            print("  decisions with a sub-decision shut by a cap: " + ", ".join(f"{k} {v}" for k, v in self.capped.most_common()))
