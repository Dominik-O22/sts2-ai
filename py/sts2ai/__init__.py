"""Training and play loop for the StS2 Ironclad agent. The combat sim and
its vectorized environment live in the Rust extension `sts2ai._sim`."""

from sts2ai import _sim
from sts2ai.env import Envs, Layout

__all__ = ["_sim", "Envs", "Layout"]
