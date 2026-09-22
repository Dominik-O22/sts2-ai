"""Policy and value network (DESIGN.md, Decision engine).

Id embeddings for the cards in hand and on offer, their enchantments, and
the potions are concatenated with the dense features and run through an
MLP. Enemies are a set: one small encoder reads each enemy (its monster and
next-move embeddings plus its dense block), and the sum over enemies joins
the MLP input. The value head reads the MLP state.

The policy head is keyed on the thing being acted on. A card or potion is
scored once per target from its own embedding, the encoded enemy it would
hit, and the MLP state, so "what Bash does" and "which enemy to hit" are
learned once rather than once per slot. Choice options are scored from
their embedding and the MLP state.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from sts2ai.env import Layout

# Layout fields a checkpoint must share with the running sim. The rest are
# vocabulary sizes and the offsets they push around, which `load_state`
# remaps by name.
SHAPE_FIELDS = (
    "max_hand",
    "max_enemies",
    "max_potions",
    "max_choices",
    "targets",
    "hand_feats",
    "choice_feats",
    "global_len",
    "enemy_base",
    "intent_nums",
)


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


class PairHead(nn.Module):
    """Scores every (item, target) pair against the state: one hidden layer
    over (state, item, target), then one logit. Targets are the encoded
    enemies with a learned "no target" row in front, which is the order the
    action layout uses."""

    def __init__(self, state_dim: int, item_dim: int, target_dim: int, hidden: int = 128):
        super().__init__()
        self.state = nn.Linear(state_dim, hidden)
        self.item = nn.Linear(item_dim, hidden, bias=False)
        self.target = nn.Linear(target_dim, hidden, bias=False)
        self.none = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: Tensor, items: Tensor, targets: Tensor) -> Tensor:
        """`[B, K, item_dim]` items and `[B, E, target_dim]` enemies to
        `[B, K, E + 1]` logits, target 0 being "no target"."""
        B = state.shape[0]
        t = torch.cat([self.none.expand(B, 1, -1), self.target(targets)], dim=1)
        h = torch.relu(self.state(state)[:, None, None] + self.item(items)[:, :, None] + t[:, None])
        return self.out(h).squeeze(-1)


def _head(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    nn.init.orthogonal_(head[-1].weight, gain=0.01)
    nn.init.zeros_(head[-1].bias)
    return head


class Policy(nn.Module):
    def __init__(
        self,
        layout: Layout,
        hidden: int = 512,
        card_dim: int = 32,
        monster_dim: int = 16,
        move_dim: int = 8,
        potion_dim: int = 8,
        enchant_dim: int = 4,
        enemy_dim: int = 64,
    ):
        super().__init__()
        L = layout
        # The heads are concatenated in action-index order, so the layout
        # must be play, potion, end turn, choose, skip.
        assert L.a_play == 0 and L.a_potion == L.max_hand * L.targets
        assert L.a_end_turn == L.a_potion + L.max_potions * L.targets
        assert L.a_choose == L.a_end_turn + 1 and L.a_skip == L.a_choose + L.max_choices == L.n_actions - 1
        assert L.targets == L.max_enemies + 1
        self.layout = L
        self.card = nn.Embedding(L.card_vocab, card_dim, padding_idx=0)
        self.monster = nn.Embedding(L.monster_vocab, monster_dim, padding_idx=0)
        self.move = nn.Embedding(L.move_vocab, move_dim, padding_idx=0)
        self.potion = nn.Embedding(L.potion_vocab, potion_dim, padding_idx=0)
        self.enchant = nn.Embedding(L.enchant_vocab, enchant_dim, padding_idx=0)
        # One enemy: its two embeddings, then its dense block.
        self.enemy = nn.Sequential(
            nn.Linear(monster_dim + move_dim + L.enemy_feats, enemy_dim), nn.ReLU(), nn.Linear(enemy_dim, enemy_dim), nn.ReLU()
        )
        in_dim = (
            L.n_floats
            - L.max_enemies * L.enemy_feats
            + L.max_hand * (card_dim + enchant_dim)
            + L.max_choices * card_dim
            + L.max_potions * potion_dim
            + enemy_dim
        )
        self.torso = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.play = PairHead(hidden, card_dim + enchant_dim + L.hand_feats, enemy_dim)
        self.use_potion = PairHead(hidden, potion_dim + 1, enemy_dim)
        self.choose = KeyedHead(hidden, card_dim + L.choice_feats, 1)
        self.end_or_skip = _head(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns unmasked logits `[B, n_actions]` and values `[B]`."""
        L = self.layout
        B = floats.shape[0]
        hand = torch.cat(
            [self.card(ids[:, L.i_hand : L.i_hand + L.max_hand]), self.enchant(ids[:, L.i_enchants : L.i_enchants + L.max_hand])], dim=2
        )
        choices = self.card(ids[:, L.i_choices : L.i_choices + L.max_choices])
        potions = self.potion(ids[:, L.i_potions : L.i_potions + L.max_potions])

        enemy_floats = floats[:, L.f_enemies : L.f_relics].view(B, L.max_enemies, L.enemy_feats)
        enemies = self.enemy(
            torch.cat(
                [
                    self.monster(ids[:, L.i_enemies : L.i_enemies + L.max_enemies]),
                    self.move(ids[:, L.i_moves : L.i_moves + L.max_enemies]),
                    enemy_floats,
                ],
                dim=2,
            )
        )
        # Empty slots encode to the biases alone; the presence flag zeroes them.
        enemies = enemies * enemy_floats[:, :, :1]

        x = torch.cat(
            [
                floats[:, : L.f_enemies],
                floats[:, L.f_relics :],
                hand.flatten(1),
                choices.flatten(1),
                potions.flatten(1),
                enemies.sum(1),
            ],
            dim=1,
        )
        h = self.torso(x)

        hand_feats = floats[:, L.f_hand : L.f_hand + L.max_hand * L.hand_feats].view(B, L.max_hand, L.hand_feats)
        potion_feats = floats[:, L.f_potions : L.f_potions + L.max_potions].unsqueeze(2)
        choice_feats = floats[:, L.f_choices : L.f_choices + L.max_choices * L.choice_feats].view(B, L.max_choices, L.choice_feats)
        play = self.play(h, torch.cat([hand, hand_feats], dim=2), enemies).flatten(1)
        potion = self.use_potion(h, torch.cat([potions, potion_feats], dim=2), enemies).flatten(1)
        choose = self.choose(h, torch.cat([choices, choice_feats], dim=2)).squeeze(2)
        end_skip = self.end_or_skip(h)
        logits = torch.cat([play, potion, end_skip[:, :1], choose, end_skip[:, 1:]], dim=1)
        return logits, self.v(h).squeeze(-1)


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Illegal actions get a logit small enough to vanish after softmax."""
    return logits.masked_fill(~mask, -1e9)


def layout_mismatch(old: dict[str, int] | None, new: Layout) -> str | None:
    """Which shape field changed since the checkpoint, if any. A checkpoint
    from before layouts were recorded is taken on trust."""
    if old is None:
        return None
    cur = asdict(new)
    return next((f"{k} {old[k]} vs {cur[k]}" for k in SHAPE_FIELDS if k in old and old[k] != cur[k]), None)


def load_state(policy: Policy, state: dict[str, Tensor], old_vocab: str | None, old_layout: dict[str, int] | None = None) -> bool:
    """Load weights. When the sim's vocabularies grew since the checkpoint,
    `old_vocab` (the vocab.txt it was trained with) lets embedding rows and
    observation columns move by name. Returns whether a remap happened.
    A layout change (a capacity or a per-slot feature count) has no remap;
    that is a retrain."""
    own = policy.state_dict()
    if all(t.shape == own[k].shape for k, t in state.items()) and state.keys() == own.keys():
        policy.load_state_dict(state)
        return False
    if changed := layout_mismatch(old_layout, policy.layout):
        raise ValueError(f"checkpoint layout differs from the sim ({changed}); train from scratch")
    if old_vocab is None:
        raise ValueError("checkpoint does not fit the current sim and carries no vocabulary; pass --old-vocab <vocab.txt it was trained with>")
    from sts2ai.vocab import current_text, parse, remap_state

    policy.load_state_dict(remap_state(state, parse(old_vocab), parse(current_text()), own, policy.layout))
    return True


def checkpoint_vocab(ck: object, fallback: Path | None) -> str | None:
    """The vocabulary a checkpoint was trained with: stored in it, or read
    from `fallback` for checkpoints from before that was recorded."""
    if isinstance(ck, dict) and isinstance(ck.get("vocab"), str):
        return ck["vocab"]
    return fallback.read_text() if fallback else None


def checkpoint_layout(ck: object) -> dict[str, int] | None:
    return ck.get("layout") if isinstance(ck, dict) and isinstance(ck.get("layout"), dict) else None


def load_policy(path: Path, policy: Policy, device: torch.device, old_vocab: Path | None = None) -> None:
    """Load weights from a training checkpoint (or a bare state dict)."""
    ck = torch.load(path, map_location=device)
    load_state(policy, ck["policy"] if "policy" in ck else ck, checkpoint_vocab(ck, old_vocab), checkpoint_layout(ck))
