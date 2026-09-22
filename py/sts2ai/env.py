"""Batch of sim combats with preallocated observation buffers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

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


class End(NamedTuple):
    """One finished fight."""

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


class Envs:
    """`n` combats stepped together. `floats`, `ids`, and `mask` always hold
    the current observation; `step` overwrites them in place."""

    def __init__(self, n: int, seed: int = 0, **config: int):
        self.sim = _sim.VecEnv(n, seed, **config)
        self.layout = Layout.load()
        self.n = n
        self.floats = np.zeros((n, self.layout.n_floats), np.float32)
        self.ids = np.zeros((n, self.layout.n_ids), np.int64)
        self.mask = np.zeros((n, self.layout.n_actions), np.bool_)
        self.rewards = np.zeros(n, np.float32)
        self.dones = np.zeros(n, np.bool_)
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
        return [End(*e) for e in ends]

    def set_floors(self, lo: int, hi: int) -> None:
        self.sim.set_floors(lo, hi)

    def set_hard_frac(self, frac: float) -> None:
        self.sim.set_hard_frac(frac)

    def use_holdout(self, seed: int = 0, per_encounter: int = 10, acts: int = 3) -> int:
        """Cycle through a fixed generated set covering every encounter of
        the first `acts` acts."""
        n = self.sim.use_holdout(seed, per_encounter, acts)
        self.sim.observe(self.floats, self.ids, self.mask)
        return n

    def load_recordings(self, directory: Path = DEFAULT_RECORDINGS) -> int:
        """Cycle through recorded fights instead of generated ones."""
        n, errors = self.sim.load_recordings(str(directory))
        for e in errors:
            print(f"skipped {e}")
        self.sim.observe(self.floats, self.ids, self.mask)
        return n
