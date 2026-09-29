"""Checks the sim's afterstate scoring against the Python reduction it
replaces, on the same batches of a runplay run: the old leaves
(`afterstate_leaves`) reduced in Python as `Scorer.score` did, beside
`Scorer.score` now. Takes runplay's arguments.

    uv run python -m sts2ai.afterstate_check CHECKPOINT --run-policy ... --afterstate 4
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import NamedTuple

import numpy as np

from sts2ai import afterstate, runplay
from sts2ai.forecast import Fights


class Leaf(NamedTuple):
    row: int
    option: int
    sample: int
    path: list[tuple[int, str]]
    hp: float
    end: str | None
    capped: bool
    fights: int


def _best(items: list[tuple[tuple[int, ...], float, list[str]]]) -> tuple[float, list[str]]:
    if len(items[0][0]) == 0:
        return items[0][1], items[0][2]
    groups: dict[int, list[tuple[tuple[int, ...], float, list[str]]]] = defaultdict(list)
    for path, score, names in items:
        groups[path[0]].append((path[1:], score, names))
    return max((_best(g) for g in groups.values()), key=lambda v: v[0])


def python_scores(s: afterstate.Scorer, waiting: list[int]):
    """The old `Scorer.score`, on the old leaves."""
    leaves, fights, names, capped = s.envs.sim.afterstate_leaves(waiting, s.samples, s.depth, s.nodes)
    leaves = [Leaf(*l) for l in leaves]
    fights = Fights.of(fights, s.envs.layout.n_floats, s.envs.layout.n_ids)
    score = np.full((len(waiting), s.L.max_options), np.nan, dtype=np.float32)
    if not leaves:
        return score, {}, names, capped
    states = max(l.fights for l in leaves) + 1
    hp = np.zeros((states, 1), dtype=np.float32)
    for l in leaves:
        hp[l.fights, 0] = l.hp
    read = s.forecaster.read(hp, fights)
    value = np.zeros(states, dtype=np.float32)
    boss = np.array([e.endswith("Boss") for e in read.encounter], dtype=bool)
    for part in (~boss, boss):
        count = np.bincount(read.row[part], minlength=states)
        value += np.bincount(read.row[part], read.win[part], states) / np.maximum(count, 1)
    trees: dict[tuple[int, int, int], list[tuple[tuple[int, ...], float, list[str]]]] = defaultdict(list)
    for l in leaves:
        v = 1.0 if l.end == "won" else 0.0 if l.end else float(value[l.fights])
        trees[(l.row, l.option, l.sample)].append((tuple(j for j, _ in l.path), v, [name for _, name in l.path]))
    by_option: dict[tuple[int, int], list[tuple[float, list[str]]]] = defaultdict(list)
    for (row, option, _), items in trees.items():
        by_option[(row, option)].append(_best(items))
    via = {}
    for (row, option), samples in by_option.items():
        score[row, option] = np.mean([v for v, _ in samples])
        via[(row, option)] = max(samples, key=lambda v: v[0])[1]
    return score, via, names, capped


class Checked(afterstate.Scorer):
    """`Scorer` that also runs the old reduction and tallies the gaps."""

    batches = options = nan_mismatch = cap_mismatch = via_mismatch = via_differs = 0
    max_diff = 0.0

    def score(self, waiting: list[int]) -> np.ndarray:
        old, via, names, capped = python_scores(self, waiting)
        before = Counter(self.capped)
        new = super().score(waiting)
        Checked.batches += 1
        finite = np.isfinite(old)
        Checked.options += int(finite.sum())
        Checked.nan_mismatch += int((finite != np.isfinite(new)).sum())
        if finite.any():
            Checked.max_diff = max(Checked.max_diff, float(np.abs(old[finite] - new[finite]).max()))
        Checked.cap_mismatch += (self.capped - before) != Counter(capped)
        for (i, j), v in via.items():
            name, path = self.envs.sim.afterstate_option(i, j)
            # A tie between sub-decisions may break the other way: only
            # the value along the path has to match, which the scores show.
            Checked.via_mismatch += name != names[i][j] or len(path) != len(v)
            Checked.via_differs += path != v
        return new


def main() -> None:
    runplay.Scorer = Checked
    runplay.main()
    c = Checked
    print(
        f"check: {c.batches} batches, {c.options} options, max |python - sim| {c.max_diff:.3g},"
        f" {c.nan_mismatch} scored-or-not mismatches, {c.cap_mismatch} cap count mismatches, {c.via_mismatch} name or path length mismatches, {c.via_differs} paths that differ (ties)"
    )
    sys.exit(0 if c.nan_mismatch == c.cap_mismatch == 0 and c.max_diff < 1e-5 else 1)


if __name__ == "__main__":
    main()
