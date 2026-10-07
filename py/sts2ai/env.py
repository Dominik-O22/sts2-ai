"""Batch of sim combats with preallocated observation buffers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from sts2ai import _sim

DEFAULT_RECORDINGS = Path.home() / ".local/share/SlayTheSpire2/sts2ai/recordings"
# The ascension the policy trains at (`EnvConfig::default`). Played fights
# at another one are left out of what stands for its play.
ASCENSION = 10


def has_recordings(directory: Path = DEFAULT_RECORDINGS) -> bool:
    """Whether any run fight has been recorded (dev-console fights sit in dev/)."""
    return directory.is_dir() and any(directory.glob("*.jsonl"))


@dataclass(frozen=True)
class Layout:
    """Offsets into the observation and action vectors (`sim::encode`)."""

    n_floats: int
    n_ids: int
    n_actions: int
    card_vocab: int
    monster_vocab: int
    potion_vocab: int
    max_hand: int
    max_enemies: int
    max_potions: int
    max_choices: int
    targets: int
    a_play: int
    a_potion: int
    a_end_turn: int
    a_choose: int
    a_skip: int
    i_hand: int
    i_enemies: int
    i_potions: int
    i_choices: int
    i_moves: int
    i_enchants: int
    i_choice_enchants: int
    i_piles: int
    i_pile_enchants: int
    i_resumes: int
    move_vocab: int
    enchant_vocab: int
    global_len: int
    f_player_powers: int
    f_hand: int
    hand_feats: int
    f_piles: int
    max_pile_rows: int
    card_feats: int
    pile_feats: int
    f_enemies: int
    enemy_base: int
    intent_nums: int
    enemy_feats: int
    f_relics: int
    relic_feats: int
    f_potions: int
    f_choices: int
    choice_feats: int
    n_cards: int
    n_powers: int
    n_relics: int
    n_intents: int

    @classmethod
    def load(cls) -> Layout:
        return cls(**_sim.layout())


@dataclass(frozen=True)
class RunLayout:
    """Offsets and sizes of the run observation (`sim::runobs`): per row,
    a global token, then deck, relic, potion and option tokens, each
    segment a fixed number of tokens of fixed width in `floats` and `ids`,
    whose first float says the token is there; then, in `ids` alone, the
    map ahead at a map step (`map_rows` by `map_cols` nodes of type and
    links, type 0 where there is no point)."""

    run_floats: int
    run_ids: int
    max_deck: int
    max_relics: int
    max_potions: int
    max_options: int
    option_cards: int
    global_ids: int
    global_floats: int
    # The forecast's slots in the global token (`sts2ai.forecast`), and the
    # openings rolled per encounter.
    f_forecast: int
    forecast_floats: int
    forecast_rolls: int
    map_feats: int
    deck_ids: int
    deck_floats: int
    relic_ids: int
    relic_floats: int
    potion_ids: int
    potion_floats: int
    option_ids: int
    option_floats: int
    f_deck: int
    f_relics: int
    f_potions: int
    f_options: int
    i_deck: int
    i_relics: int
    i_potions: int
    i_options: int
    i_map: int
    map_rows: int
    map_cols: int
    map_node_ids: int
    card_vocab: int
    enchant_vocab: int
    potion_vocab: int
    relic_vocab: int
    decision_vocab: int
    option_vocab: int
    room_vocab: int
    act_vocab: int
    event_vocab: int
    boss_vocab: int
    event_key_vocab: int

    @classmethod
    def load(cls) -> RunLayout:
        return cls(**_sim.run_layout())


class RunFight(NamedTuple):
    """A run-mode fight's place in its run, or where a run ended between
    fights (`Envs.step_run`)."""

    seed: int
    act: int  # 0-based
    floor: int
    deck: int  # cards in the deck the fight was fought with
    # How the run ended with this fight: "won", "died", "stuck: <why>".
    end: str | None
    # Where the run started: its place in `START_POINTS`, None for floor 1;
    # and with what player there: "gen" generated, "own" the envs' own
    # state, "win" a winner's run (`Envs.use_winner_starts`), "" for floor 1.
    start: int | None
    source: str


# The points a run can start at besides floor 1, the latest first.
START_POINTS: list[str] = _sim.start_points()


class End(NamedTuple):
    """One finished fight. In run mode `floor` is the run's floor."""

    env: int
    won: bool
    hp_frac: float
    hp_lost: float
    potions_used: int
    steps: int
    floor: int
    encounter: str
    kind: str
    reward: float
    # Resumed from the restart pool (`set_restart_frac`), not played from
    # the fight's start.
    restart: bool
    run: RunFight | None


def pinned(shape: tuple[int, ...], dtype: torch.dtype) -> np.ndarray:
    """A zeroed numpy buffer in pinned memory when there is a GPU, so a
    non-blocking copy to it does not stage through pageable memory."""
    return torch.zeros(shape, dtype=dtype, pin_memory=torch.cuda.is_available()).numpy()


class Taken(NamedTuple):
    """One `Envs.take`: the envs at a combat decision and their rows
    (`floats[k]` is `combat[k]`'s), the envs at a run decision and theirs,
    the fights that ended since the last take and the runs that ended
    (by env), the fight starts logged (`log_fights`) and the trace lines
    written (`trace_runs`), and how many envs are still in the batch."""

    combat: list[int]
    decision: list[int]
    floats: np.ndarray
    ids: np.ndarray
    mask: np.ndarray
    run_floats: np.ndarray
    run_ids: np.ndarray
    ends: list[End]
    ended: list[tuple[int, RunFight]]
    starts: list[str]
    traces: list[str]
    active: int


class Envs:
    """`n` combats stepped together. `floats`, `ids`, and `mask` always hold
    the current observation; `step` overwrites them in place, so a
    non-blocking copy out of them must be done before the next `step`."""

    def __init__(self, n: int, seed: int = 0, **config: int):
        self.sim = _sim.VecEnv(n, seed, **config)
        self.layout = Layout.load()
        self.n = n
        self.floats = pinned((n, self.layout.n_floats), torch.float32)
        self.ids = pinned((n, self.layout.n_ids), torch.int64)
        self.mask = pinned((n, self.layout.n_actions), torch.bool)
        self.rewards = pinned((n,), torch.float32)
        self.dones = pinned((n,), torch.bool)
        self.sim.observe(self.floats, self.ids, self.mask)

    def step(self, actions: np.ndarray) -> list[End]:
        ends = self.sim.step(
            np.ascontiguousarray(actions, dtype=np.int64),
            self.floats,
            self.ids,
            self.mask,
            self.rewards,
            self.dones,
        )
        return [End(*e[:-1], RunFight(*e[-1]) if e[-1] else None) for e in ends]

    def set_floors(self, lo: int, hi: int) -> None:
        self.sim.set_floors(lo, hi)

    def set_hard_frac(self, frac: float) -> None:
        self.sim.set_hard_frac(frac)

    def set_restart_frac(self, frac: float) -> None:
        self.sim.set_restart_frac(frac)

    def set_hard_weights(self, weights: dict[str, float]) -> None:
        """Draw the forced elites and bosses by these weights, by encounter
        name; empty draws them evenly."""
        self.sim.set_hard_weights(list(weights.items()))

    def use_holdout(self, seed: int = 0, per_encounter: int = 10, acts: int = 3) -> int:
        """Cycle through a fixed generated set covering every encounter of
        the first `acts` acts."""
        n = self.sim.use_holdout(seed, per_encounter, acts)
        self.sim.observe(self.floats, self.ids, self.mask)
        return n

    def use_runs(self, seed: int = 0, asc: int = 10, choices: str = "random", last: int | None = None) -> None:
        """Play whole runs, fight after fight, on the seeds `seed..last`
        (`sim::env::Seeds`; no `last`: without end): env `i` starts on
        `seed + i`, and a run that ends takes the next unplayed seed; a
        seed plays the same run in any env. `choices` makes the run
        decisions: "random", "first", or "caller", where each run stops at
        its decisions until they are answered."""
        self.sim.use_runs(seed, asc, choices, last)
        self.run_layout = RunLayout.load()
        self.run_floats = pinned((self.n, self.run_layout.run_floats), torch.float32)
        self.run_ids = pinned((self.n, self.run_layout.run_ids), torch.int64)
        self.sim.observe(self.floats, self.ids, self.mask)

    def start_loop(self, workers: int = 0) -> None:
        """Start the threaded run loop (`sim::runloop`): `take` and `post`
        drive it (`sts2ai.runloop.Loop`); `workers` 0 is one per core."""
        self.sim.start_loop(workers)

    def take(self, min_rows: int, timeout_ms: int) -> Taken:
        """The envs ready for the network, their rows packed at the front
        of the observation buffers: waits for `min_rows` of them, or until
        no env is being stepped, or `timeout_ms`."""
        t = self.sim.take(min_rows, timeout_ms, self.floats, self.ids, self.mask, self.run_floats, self.run_ids)
        k, m = len(t.combat), len(t.decision)
        return Taken(
            t.combat,
            t.decision,
            self.floats[:k],
            self.ids[:k],
            self.mask[:k],
            self.run_floats[:m],
            self.run_ids[:m],
            [End(*e[:-1], RunFight(*e[-1]) if e[-1] else None) for e in t.ends],
            [(env, RunFight(*r)) for env, r in t.ended],
            t.starts,
            t.traces,
            t.active,
        )

    def post(self, combat: list[int], actions: np.ndarray, decision: list[int], options: np.ndarray) -> None:
        """Answer the envs taken: `actions[k]` for `combat[k]`, `options[k]`
        for `decision[k]`."""
        self.sim.post(combat, np.ascontiguousarray(actions, dtype=np.int64), decision, np.ascontiguousarray(options, dtype=np.int64))

    def set_starts(self, full: float, weights: list[float], own: list[float]) -> None:
        """Where the runs that start from now on start: floor 1 with chance
        `full`, else a start point by `weights`, from the envs' own state
        there with chance `own` when they have one; both lists in
        `START_POINTS` order."""
        self.sim.set_starts(full, weights, own)

    def use_winner_starts(self, runs: list[Path], share: float) -> list[int]:
        """Start a `share` of the runs that start from now on from the
        winners' history files `runs`, at the entrances of acts 2 and 3
        each reaches faithful to its record; returns how many runs each
        start point holds, in `START_POINTS` order."""
        return self.sim.use_winner_starts([p.read_text() for p in runs], share)

    def start_pools(self) -> list[int]:
        """States the runs have kept per start point."""
        return self.sim.start_pools()

    def run_waiting(self) -> list[int]:
        """The envs whose run waits at a decision. A combat `step` leaves
        them where they are."""
        return self.sim.run_waiting()

    def observe_run(self, envs: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """The run decisions `envs` wait at, one row each, in the first
        rows of `run_floats` and `run_ids`."""
        self.sim.observe_run(envs, self.run_floats, self.run_ids)
        return self.run_floats[: len(envs)], self.run_ids[: len(envs)]

    def forecast(self, envs: list[int]):
        """The forecast fights of the decisions `envs` wait at
        (`sts2ai.forecast.Fights`), their rows indexing `envs`."""
        from sts2ai.forecast import Fights

        return Fights.of(self.sim.forecast(envs), self.layout.n_floats, self.layout.n_ids)

    def afterstates(self, envs: list[int], samples: int, depth: int, nodes: int, rows: bool = True) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Builds the afterstates of every option of the decisions `envs`
        wait at (`sts2ai.afterstate`) and returns the combat rows the value
        head reads for them (floats, ids) and how many leaves and distinct
        settled states they hold. `afterstate_scores` scores them. With
        `rows` off the rows are empty, for `afterstate_scores_given`."""
        floats, ids, leaves, states = self.sim.afterstates(envs, samples, depth, nodes, rows)
        return floats.reshape(-1, self.layout.n_floats), ids.reshape(-1, self.layout.n_ids), leaves, states

    def afterstate_scores(self, values: np.ndarray, win: tuple[float, float, float]) -> tuple[np.ndarray, dict[str, int]]:
        """Scores the afterstates built last from the value head's read of
        their rows and the calibration's win coefficients: `score[k, j]` of
        row `k`'s option token `j` (NaN where there is none), and the
        decisions where a cap kept a sub-decision shut."""
        score, capped = self.sim.afterstate_scores(np.ascontiguousarray(values, dtype=np.float32), win)
        return score.reshape(-1, self.run_layout.max_options), capped

    def afterstate_runs(self) -> list[dict]:
        """The distinct settled states of the afterstates built last, as run
        records (deck, relics, potions, HP, act), in the order
        `afterstate_scores_given` takes their scores; {} for one without
        forecast fights."""
        return [json.loads(r) if r else {} for r in self.sim.afterstate_runs()]

    def afterstate_scores_given(self, scores: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """`afterstate_scores` from a score per distinct settled state."""
        score, capped = self.sim.afterstate_scores_given(np.asarray(scores, dtype=np.float64).tolist())
        return score.reshape(-1, self.run_layout.max_options), capped

    def step_run(self, envs: list[int], options: np.ndarray) -> list[tuple[int, RunFight]]:
        """Answer each env's decision with an option token and play on;
        the fights that start fill their combat rows. Returns the runs
        that ended, by env."""
        ended = self.sim.step_run(envs, np.ascontiguousarray(options, dtype=np.int64), self.floats, self.ids, self.mask)
        return [(env, RunFight(*r)) for env, r in ended]

    def use_setups(self, path: Path, repeats: int = 1, seed: int = 0) -> int:
        """Cycle through played runs' fights (`sts2ai.setups`), `repeats`
        of each."""
        n = self.sim.use_setups(path.read_text(), repeats, seed)
        self.sim.observe(self.floats, self.ids, self.mask)
        return n

    def load_recordings(self, directory: Path = DEFAULT_RECORDINGS, ascension: int | None = None) -> int:
        """Cycle through recorded fights instead of generated ones; with
        `ascension`, only those played at it."""
        n, errors = self.sim.load_recordings(str(directory), ascension)
        for e in errors:
            print(f"skipped {e}")
        self.sim.observe(self.floats, self.ids, self.mask)
        return n
