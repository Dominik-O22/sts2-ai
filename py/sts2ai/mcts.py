"""Tree search over whole fights (`sim::mcts`): PUCT over the policy's
priors, many turns deep, chance sampled per simulation, leaves scored by
the network in one batch across every fight searched.

`exactsearch.play`, and with it `sts2ai.bench` and the probes, runs it as
mode `mctsN`: N simulations per decision, the most visited action played.
Widening at chance is off by default: on the 350 held-out act 3 boss fights
256 simulations won 77.5% without it and 74.8% with `widen=1`, a tree that
goes deeper through a few sampled next hands judging worse than one that
samples a new hand each time and lets the value head take it from there.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import Envs
from sts2ai.model import Net, masked_logits


class TreeSearch:
    """`sims` simulations per decision for the fights of an `Envs`."""

    def __init__(self, envs: Envs, sims: int, c_puct: float = 1.25, widen: float = math.inf, seed: int = 0, clairvoyant: bool = False):
        self.envs, self.sims = envs, sims
        # Clairvoyant searches read the fight's real draw order and dice: a
        # measuring instrument for which fights are winnable at all, never
        # for play or training (no hidden information).
        self.inner = _sim.TreeSearch(envs.n, c_puct=c_puct, widen=widen, seed=seed, clairvoyant=clairvoyant)
        # Turns past the decision the trees reached, summed, and decisions.
        self.depth, self.searched = 0, 0
        L, pin = envs.layout, torch.cuda.is_available()
        self.floats = torch.zeros((envs.n, L.n_floats), pin_memory=pin)
        self.ids = torch.zeros((envs.n, L.n_ids), dtype=torch.int64, pin_memory=pin)
        self.mask = torch.zeros((envs.n, L.n_actions), dtype=torch.bool, pin_memory=pin)

    def visits(self, i: int) -> np.ndarray:
        """Env `i`'s root visit counts over the action space, after `choose`."""
        out = np.zeros(self.mask.shape[1], np.float32)
        for a, n, _ in self.inner.root_stats(i):
            out[a] = n
        return out

    @torch.no_grad()
    def choose(self, net: Net, device: torch.device, roots: list[int], actions: np.ndarray) -> None:
        """Search each of `roots` and play its most visited action."""
        self.inner.start(self.envs.sim, roots)
        for _ in range(self.sims):
            rows = self.inner.descend(roots, self.floats.numpy(), self.ids.numpy(), self.mask.numpy())
            if not rows:
                continue
            k = len(rows)
            logits, values = net(self.floats[:k].to(device, non_blocking=True), self.ids[:k].to(device, non_blocking=True))
            priors = masked_logits(logits.float(), self.mask[:k].to(device, non_blocking=True)).softmax(dim=1)
            self.inner.expand(priors.cpu().numpy(), values.float().cpu().numpy())
        for i in roots:
            if (a := self.inner.best_action(i)) is not None:
                actions[i] = a
            self.depth += self.inner.turns_deep(i)
        self.searched += len(roots)
