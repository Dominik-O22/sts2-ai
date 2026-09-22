"""Policy and value network (DESIGN.md, Decision engine).

Id embeddings for the cards in hand and on offer, the enemies and their
next moves, and the potions are concatenated with the dense features and
run through an MLP. The value head reads the MLP state. The policy head
is keyed on the thing being acted on: each hand card, potion, or choice
option is scored from its own embedding plus the MLP state, so "what Bash
does" is learned once rather than once per hand slot.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from sts2ai.env import Layout


class KeyedHead(nn.Module):
    """Scores each of `[B, K, item_dim]` items against the `[B, state_dim]`
    state: one hidden layer over (state, item), then `out_dim` logits per
    item. The state is projected once and broadcast, which is the same
    function as a linear layer over the concatenation without building
    the `[B, K, state_dim]` tensor."""

    def __init__(self, state_dim: int, item_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.state = nn.Linear(state_dim, hidden)
        self.item = nn.Linear(item_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: Tensor, items: Tensor) -> Tensor:
        return self.out(torch.relu(self.state(state).unsqueeze(1) + self.item(items)))


def _head(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    nn.init.orthogonal_(head[-1].weight, gain=0.01)
    nn.init.zeros_(head[-1].bias)
    return head


class Policy(nn.Module):
    def __init__(
        self, layout: Layout, hidden: int = 512, card_dim: int = 32, monster_dim: int = 16, move_dim: int = 8, potion_dim: int = 8
    ):
        super().__init__()
        L = layout
        # The heads are concatenated in action-index order, so the layout
        # must be play, potion, end turn, choose, skip.
        assert L.a_play == 0 and L.a_potion == L.max_hand * L.targets
        assert L.a_end_turn == L.a_potion + L.max_potions * L.targets
        assert L.a_choose == L.a_end_turn + 1 and L.a_skip == L.a_choose + L.max_choices == L.n_actions - 1
        self.layout = L
        self.card = nn.Embedding(L.card_vocab, card_dim, padding_idx=0)
        self.monster = nn.Embedding(L.monster_vocab, monster_dim, padding_idx=0)
        self.move = nn.Embedding(L.move_vocab, move_dim, padding_idx=0)
        self.potion = nn.Embedding(L.potion_vocab, potion_dim, padding_idx=0)
        in_dim = (
            L.n_floats
            + (L.max_hand + L.max_choices) * card_dim
            + L.max_enemies * (monster_dim + move_dim)
            + L.max_potions * potion_dim
        )
        self.torso = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.play = KeyedHead(hidden, card_dim + L.hand_feats, L.targets)
        self.use_potion = KeyedHead(hidden, potion_dim + 1, L.targets)
        self.choose = KeyedHead(hidden, card_dim + L.choice_feats, 1)
        self.end_or_skip = _head(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns unmasked logits `[B, n_actions]` and values `[B]`."""
        L = self.layout
        B = floats.shape[0]
        hand = self.card(ids[:, L.i_hand : L.i_hand + L.max_hand])
        choices = self.card(ids[:, L.i_choices : L.i_choices + L.max_choices])
        enemies = self.monster(ids[:, L.i_enemies : L.i_enemies + L.max_enemies])
        moves = self.move(ids[:, L.i_moves : L.i_moves + L.max_enemies])
        potions = self.potion(ids[:, L.i_potions : L.i_potions + L.max_potions])
        x = torch.cat(
            [floats, hand.flatten(1), choices.flatten(1), enemies.flatten(1), moves.flatten(1), potions.flatten(1)], dim=1
        )
        h = self.torso(x)

        hand_feats = floats[:, L.f_hand : L.f_hand + L.max_hand * L.hand_feats].view(B, L.max_hand, L.hand_feats)
        potion_feats = floats[:, L.f_potions : L.f_potions + L.max_potions].unsqueeze(2)
        choice_feats = floats[:, L.f_choices : L.f_choices + L.max_choices * L.choice_feats].view(B, L.max_choices, L.choice_feats)
        play = self.play(h, torch.cat([hand, hand_feats], dim=2)).flatten(1)
        potion = self.use_potion(h, torch.cat([potions, potion_feats], dim=2)).flatten(1)
        choose = self.choose(h, torch.cat([choices, choice_feats], dim=2)).squeeze(2)
        end_skip = self.end_or_skip(h)
        logits = torch.cat([play, potion, end_skip[:, :1], choose, end_skip[:, 1:]], dim=1)
        return logits, self.v(h).squeeze(-1)


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Illegal actions get a logit small enough to vanish after softmax."""
    return logits.masked_fill(~mask, -1e9)


def load_policy(path: Path, policy: Policy, device: torch.device) -> None:
    """Load weights from a training checkpoint (or a bare state dict)."""
    ck = torch.load(path, map_location=device)
    policy.load_state_dict(ck["policy"] if "policy" in ck else ck)
