"""Policy and value network (DESIGN.md, Decision engine): id embeddings for
the cards in hand and on offer, the enemies, and the potions, concatenated
with the dense features and run through an MLP with a masked policy head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sts2ai.env import Layout


class Policy(nn.Module):
    def __init__(self, layout: Layout, hidden: int = 512, card_dim: int = 32, monster_dim: int = 16, potion_dim: int = 8):
        super().__init__()
        self.layout = layout
        self.card = nn.Embedding(layout.card_vocab, card_dim, padding_idx=0)
        self.monster = nn.Embedding(layout.monster_vocab, monster_dim, padding_idx=0)
        self.potion = nn.Embedding(layout.potion_vocab, potion_dim, padding_idx=0)
        in_dim = (
            layout.n_floats
            + (layout.max_hand + layout.max_choices) * card_dim
            + layout.max_enemies * monster_dim
            + layout.max_potions * potion_dim
        )
        self.torso = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, layout.n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns unmasked logits `[B, n_actions]` and values `[B]`."""
        L = self.layout
        hand = ids[:, L.i_hand : L.i_hand + L.max_hand]
        enemies = ids[:, L.i_enemies : L.i_enemies + L.max_enemies]
        potions = ids[:, L.i_potions : L.i_potions + L.max_potions]
        choices = ids[:, L.i_choices : L.i_choices + L.max_choices]
        x = torch.cat(
            [
                floats,
                self.card(hand).flatten(1),
                self.card(choices).flatten(1),
                self.monster(enemies).flatten(1),
                self.potion(potions).flatten(1),
            ],
            dim=1,
        )
        h = self.torso(x)
        return self.pi(h), self.v(h).squeeze(-1)


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Illegal actions get a logit small enough to vanish after softmax."""
    return logits.masked_fill(~mask, -1e9)
