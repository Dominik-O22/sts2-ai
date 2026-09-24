"""Batch of sim combats with preallocated observation buffers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from sts2ai import _sim

DEFAULT_RECORDINGS = Path.home() / ".local/share/SlayTheSpire2/sts2ai/recordings"


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
    move_vocab: int
    enchant_vocab: int
    global_len: int
    f_player_powers: int
    f_hand: int
    hand_feats: int
    f_piles: int
    f_enemies: int
    enemy_base: int
    intent_nums: int
    enemy_feats: int
    f_relics: int
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
    whose first float says the token is there."""

    run_floats: int
    run_ids: int
    max_deck: int
    max_relics: int
    max_potions: int
    max_options: int
    option_cards: int
    global_ids: int
    global_floats: int
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
    # and whether from the envs' own state there, not a generated one.
    start: int | None
    own: bool


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
    run: RunFight | None


def pinned(shape: tuple[int, ...], dtype: torch.dtype) -> np.ndarray:
    """A zeroed numpy buffer in pinned memory when there is a GPU, so a
    non-blocking copy to it does not stage through pageable memory."""
    return torch.zeros(shape, dtype=dtype, pin_memory=torch.cuda.is_available()).numpy()


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

    def use_runs(self, seed: int = 0, asc: int = 10, choices: str = "random") -> None:
        """Play whole runs, fight after fight; a run that ends starts a
        fresh one from the next seed. `choices` makes the run decisions:
        "random", "first", or "caller", where each run stops at its
        decisions until `step_run` answers them."""
        self.sim.use_runs(seed, asc, choices)
        if choices == "caller":
            self.run_layout = RunLayout.load()
            self.run_floats = pinned((self.n, self.run_layout.run_floats), torch.float32)
            self.run_ids = pinned((self.n, self.run_layout.run_ids), torch.int64)
        self.sim.observe(self.floats, self.ids, self.mask)

    def set_starts(self, full: float, weights: list[float], own: list[float]) -> None:
        """Where the runs that start from now on start: floor 1 with chance
        `full`, else a start point by `weights`, from the envs' own state
        there with chance `own` when they have one; both lists in
        `START_POINTS` order."""
        self.sim.set_starts(full, weights, own)

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

    def step_run(self, envs: list[int], options: np.ndarray) -> list[tuple[int, RunFight]]:
        """Answer each env's decision with an option token and play on;
        the fights that start fill their combat rows. Returns the runs
        that ended, by env."""
        ended = self.sim.step_run(envs, np.ascontiguousarray(options, dtype=np.int64), self.floats, self.ids, self.mask)
        return [(env, RunFight(*r)) for env, r in ended]

    def load_recordings(self, directory: Path = DEFAULT_RECORDINGS) -> int:
        """Cycle through recorded fights instead of generated ones."""
        n, errors = self.sim.load_recordings(str(directory))
        for e in errors:
            print(f"skipped {e}")
        self.sim.observe(self.floats, self.ids, self.mask)
        return n
