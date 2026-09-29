"""Afterstate scoring of run decisions (docs/run-env.md, Afterstates):
instead of asking the run policy what a winner would pick, apply each
option in the sim and score the player it leaves with the forecast.

The sim expands each option of the decisions a batch waits at
(`Envs.afterstates`, every decision but a map step): the segment replayed
with the option, everything random reseeded from a sample index the
moment the option is chosen, settled at the first map step, fight or end,
with sub-decisions in the same room (which card to smith, an event's next
page, a shop after a purchase) opened under caps. Each leaf carries an
index into the settled states' forecast fights, which the combat value
head reads in one batch (`Forecaster.read`), and a calibration turns into
a win chance. A leaf's score is the act's elites' mean win chance plus
the boss's (1 for a run won in the option, 0 for one that died or is
stuck), a sub-decision takes its best option, a random outcome is the
mean over its samples, and the decision takes the best option, the
policy's pick on a tie.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import NamedTuple

import numpy as np

from sts2ai import _sim
from sts2ai.env import Envs
from sts2ai.forecast import Forecaster

DECISIONS = _sim.run_names()["decision"]


class Leaf(NamedTuple):
    """One leaf of an option's afterstate tree (`sim::env::Afterstate`)."""

    row: int
    option: int
    sample: int
    path: list[tuple[int, str]]
    hp: float
    end: str | None
    capped: bool
    fights: int


class Scored(NamedTuple):
    """A batch's decisions scored: `score[k, j]` of row `k`'s option token
    `j` (NaN where there is none), each row's options in words, and the
    sub-decisions' names on the best path under each option, as (row,
    option) to names."""

    score: np.ndarray
    names: list[list[str]]
    via: dict[tuple[int, int], list[str]]


def _best(items: list[tuple[tuple[int, ...], float, list[str]]]) -> tuple[float, list[str]]:
    """The value of a tree of leaves given as (path, score, names): a
    sub-decision on the path takes its best option."""
    if len(items[0][0]) == 0:
        return items[0][1], items[0][2]
    groups: dict[int, list[tuple[tuple[int, ...], float, list[str]]]] = defaultdict(list)
    for path, score, names in items:
        groups[path[0]].append((path[1:], score, names))
    return max((_best(g) for g in groups.values()), key=lambda v: v[0])


class Scorer:
    """Scores and makes run decisions from their afterstates, and counts
    where its picks differ from the policy's and what the caps shut.
    With `kinds` only decisions of those kinds (`DECISIONS` names) are
    made by afterstate; the others keep the policy's pick."""

    def __init__(
        self, envs: Envs, forecaster: Forecaster, samples: int, depth: int = 3, nodes: int = 256, kinds: frozenset[str] = frozenset()
    ):
        assert forecaster.calibration is not None, "afterstates need a calibration"
        self.envs, self.forecaster, self.samples, self.depth, self.nodes = envs, forecaster, samples, depth, nodes
        self.kinds = kinds
        self.L = forecaster.L
        self.decisions: Counter[str] = Counter()
        self.differs: Counter[str] = Counter()
        self.capped: Counter[str] = Counter()
        self.leaves = self.states = self.rows = 0
        self.sim_seconds = self.head_seconds = 0.0

    def score(self, waiting: list[int]) -> Scored:
        """Every option of the decisions `waiting` wait at."""
        t = time.perf_counter()
        leaves, fights, names, capped = self.envs.afterstates(waiting, self.samples, self.depth, self.nodes)
        self.sim_seconds += time.perf_counter() - t
        self.capped.update(capped)
        score = np.full((len(waiting), self.L.max_options), np.nan, dtype=np.float32)
        if not leaves:
            return Scored(score, names, {})
        states = max(l.fights for l in leaves) + 1
        hp = np.zeros((states, 1), dtype=np.float32)
        for l in leaves:
            hp[l.fights, 0] = l.hp
        t = time.perf_counter()
        read = self.forecaster.read(hp, fights)
        self.head_seconds += time.perf_counter() - t
        self.leaves += len(leaves)
        self.states += states
        self.rows += len(fights.row)
        # A state the forecast could not build (a fight the sim cannot
        # play) scores as a dead end.
        value = np.zeros(states, dtype=np.float32)
        boss = np.array([e.endswith("Boss") for e in read.encounter], dtype=bool)
        for part in (~boss, boss):
            count = np.bincount(read.row[part], minlength=states)
            value += np.bincount(read.row[part], read.win[part], states) / np.maximum(count, 1)
        trees: dict[tuple[int, int, int], list[tuple[tuple[int, ...], float, list[str]]]] = defaultdict(list)
        for l in leaves:
            s = 1.0 if l.end == "won" else 0.0 if l.end else float(value[l.fights])
            trees[(l.row, l.option, l.sample)].append((tuple(j for j, _ in l.path), s, [name for _, name in l.path]))
        by_option: dict[tuple[int, int], list[tuple[float, list[str]]]] = defaultdict(list)
        for (row, option, _), items in trees.items():
            by_option[(row, option)].append(_best(items))
        via = {}
        for (row, option), samples in by_option.items():
            score[row, option] = np.mean([s for s, _ in samples])
            via[(row, option)] = max(samples, key=lambda v: v[0])[1]
        return Scored(score, names, via)

    def choose(self, waiting: list[int], ids: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, Scored]:
        """`policy`'s picks with every scored decision's replaced by its
        best option, the policy's pick on an exact tie."""
        scored = self.score(waiting)
        options = policy.copy()
        for k in np.flatnonzero(np.isfinite(scored.score).any(1)):
            kind = DECISIONS[ids[k, 0]]
            if self.kinds and kind not in self.kinds:
                continue
            best = np.nanmax(scored.score[k])
            tied = np.flatnonzero(scored.score[k] == best)
            options[k] = policy[k] if policy[k] in tied else tied[0]
            self.decisions[kind] += 1
            self.differs[kind] += options[k] != policy[k]
        return options, scored

    def report(self) -> None:
        n = max(sum(self.decisions.values()), 1)
        print(f"afterstates: {n} decisions scored, {self.differs.total()} picked otherwise than the policy ({self.differs.total() / n:.0%})")
        for kind, d in sorted(self.decisions.items(), key=lambda kv: -kv[1]):
            print(f"  {kind:8s} {d:6d}: differs {self.differs[kind] / d:.0%}")
        print(
            f"  cost per decision: {self.leaves / n:.0f} leaves, {self.states / n:.0f} states, {self.rows / n:.0f} combat rows,"
            f" {self.sim_seconds / n * 1000:.1f} ms sim, {self.head_seconds / n * 1000:.1f} ms value head"
        )
        if self.capped:
            print("  decisions with a sub-decision shut by a cap: " + ", ".join(f"{k} {v}" for k, v in self.capped.most_common()))
