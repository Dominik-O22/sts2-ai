"""Policy and value networks (DESIGN.md, Decision engine). Two
architectures read the same observation and fill the same action layout,
so masks, search, decode and the bridge do not care which one runs; a
checkpoint records its `Arch` and `load_policy` builds the right one.

`SlotMLP`: id embeddings for the cards in hand and on offer, their enchantments, and
the potions are concatenated with the dense features and run through an
MLP. Enemies are a set: one small encoder reads each enemy (its monster and
next-move embeddings plus its dense block), and the sum over enemies joins
the MLP input. The value head reads the MLP state.

The policy head is keyed on the thing being acted on. A card or potion is
scored once per target from its own embedding, the encoded enemy it would
hit, and the MLP state, so "what Bash does" and "which enemy to hit" are
learned once rather than once per slot. Choice options are scored from
their embedding and the MLP state.

`SlotAttention`: the same slots as tokens (one global token for the rest
of the observation, then hand cards, enemies and potions) through a small
transformer, so a card's encoding has seen the enemies and the rest of the
hand before the heads score it. Empty slots are masked. Choice options stay
out of the sequence: there are 20 slots, almost always empty, and they
would double its length; their pooled encoding joins the global token and
each is scored against the encoded global token. Its `Arch` options
replace what it kept from `SlotMLP`: dot-product target heads, the piles
as tokens of card embeddings, choices that attend to the board.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


class PointerHead(nn.Module):
    """Scores every (item, target) pair by the dot product of the item's
    query and the target's key, plus the item's own bias; a learned key
    stands for "no target". The tokens have already attended to each other
    and to the global token, so the pair needs no hidden layer of its own:
    `PairHead` built one 128 wide per pair, as much work as the encoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.none = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Linear(dim, 1)
        # Near-uniform logits at the start, as `PairHead`'s small output layer gives.
        nn.init.orthogonal_(self.query.weight, gain=0.01)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)
        self.scale = dim**-0.5

    def forward(self, items: Tensor, targets: Tensor) -> Tensor:
        """`[B, K, dim]` items and `[B, E, dim]` targets to `[B, K, E + 1]`
        logits, target 0 being "no target"."""
        keys = torch.cat([self.none.expand(items.shape[0], 1, -1), self.key(targets)], dim=1)
        return torch.einsum("bkd,bed->bke", self.query(items), keys) * self.scale + self.bias(items)


@dataclass(frozen=True)
class Arch:
    """Which network and how big: `slots` (`SlotMLP`, `hidden` wide, `depth`
    torso layers) or `attn` (`SlotAttention`, `hidden` wide, `depth`
    transformer layers)."""

    kind: str = "slots"
    hidden: int = 512
    depth: int = 2
    # `attn` only. `pointer`: play and potion targets scored by a dot
    # product of the encoded tokens (`PointerHead`), not a hidden layer per
    # pair. `piles`: the draw, discard and exhaust piles as three more
    # tokens, built from the card embeddings. `choice_attn`: choice options
    # attend to the encoded tokens before they are scored.
    pointer: bool = False
    piles: bool = False
    choice_attn: bool = False
    # An auxiliary head predicting the damage the player takes this enemy
    # turn (`forward_incoming`), trained beside the policy.
    incoming: bool = False


class Policy(nn.Module):
    """What every architecture shares: the layout it was built for, its
    `Arch`, the card embedding (`sts2ai.deckvalue` seeds from it), and
    `forward(floats, ids) -> (logits [B, n_actions], values [B])`."""

    layout: Layout
    arch: Arch
    card: nn.Embedding


def _head(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    nn.init.orthogonal_(head[-1].weight, gain=0.01)
    nn.init.zeros_(head[-1].bias)
    return head


class SlotMLP(Policy):
    def __init__(
        self,
        layout: Layout,
        hidden: int = 512,
        depth: int = 2,
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
        self.arch = Arch("slots", hidden, depth)
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
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.torso = nn.Sequential(*layers)
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


class SlotAttention(Policy):
    def __init__(
        self,
        layout: Layout,
        hidden: int = 128,
        depth: int = 3,
        heads: int = 4,
        card_dim: int = 32,
        monster_dim: int = 16,
        move_dim: int = 8,
        potion_dim: int = 8,
        enchant_dim: int = 4,
        enemy_dim: int = 64,
        pointer: bool = False,
        piles: bool = False,
        choice_attn: bool = False,
        incoming: bool = False,
    ):
        super().__init__()
        L = layout
        assert L.targets == L.max_enemies + 1
        self.layout = L
        self.arch = Arch("attn", hidden, depth, pointer, piles, choice_attn, incoming)
        d = hidden
        self.card = nn.Embedding(L.card_vocab, card_dim, padding_idx=0)
        self.monster = nn.Embedding(L.monster_vocab, monster_dim, padding_idx=0)
        self.move = nn.Embedding(L.move_vocab, move_dim, padding_idx=0)
        self.potion = nn.Embedding(L.potion_vocab, potion_dim, padding_idx=0)
        self.enchant = nn.Embedding(L.enchant_vocab, enchant_dim, padding_idx=0)
        # Named like `SlotMLP`'s so `vocab.remap_state` moves their input
        # columns when a vocabulary grows: `enemy` as the enemy encoder,
        # `glob` as the torso with no embeddings after the floats.
        self.enemy = nn.Sequential(
            nn.Linear(monster_dim + move_dim + L.enemy_feats, enemy_dim), nn.ReLU(), nn.Linear(enemy_dim, enemy_dim), nn.ReLU()
        )
        self.glob = nn.Linear(L.n_floats - L.max_enemies * L.enemy_feats, d)
        # One projection per token kind; their biases tell the kinds apart.
        # Choices are not tokens (module docstring).
        self.hand_in = nn.Linear(card_dim + enchant_dim + L.hand_feats, d)
        self.enemy_in = nn.Linear(enemy_dim, d)
        self.potion_in = nn.Linear(potion_dim + 1, d)
        self.choice_in = nn.Linear(card_dim + L.choice_feats, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        if piles:
            # A pile's token: the mean embedding of its cards (an upgraded
            # copy shifted by `pile_up`) and its size, then which pile it is.
            # Zero at the start, so a warm start from a network without
            # them only gains three blank tokens.
            self.pile_up = nn.Parameter(torch.zeros(card_dim))
            self.pile_in = nn.Linear(card_dim + 1, d)
            self.pile_kind = nn.Parameter(torch.zeros(3, d))
            nn.init.zeros_(self.pile_in.weight)
            nn.init.zeros_(self.pile_in.bias)
        if choice_attn:
            self.choice_norm = nn.LayerNorm(d)
            self.choice_attn = nn.MultiheadAttention(d, heads, batch_first=True)
            # A residual that starts as nothing: choices score as before
            # until it learns something.
            nn.init.zeros_(self.choice_attn.out_proj.weight)
            nn.init.zeros_(self.choice_attn.out_proj.bias)
        self.play = PointerHead(d) if pointer else PairHead(d, d, d)
        self.use_potion = PointerHead(d) if pointer else PairHead(d, d, d)
        self.choose = KeyedHead(d, d, 1)
        self.end_or_skip = _head(d, 2)
        self.v = nn.Linear(d, 1)
        if incoming:
            # Auxiliary: the damage this enemy turn deals, given the play so far.
            self.incoming = nn.Linear(d, 1)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        logits, g = self.trunk(floats, ids)
        return logits, self.v(g).squeeze(-1)

    def forward_incoming(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`forward` and the damage the player is about to take this enemy
        turn (`incoming` head, a thirtieth of the HP), for its auxiliary loss."""
        logits, g = self.trunk(floats, ids)
        return logits, self.v(g).squeeze(-1), self.incoming(g).squeeze(-1)

    def trunk(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """The action logits and the encoded global token."""
        L = self.layout
        B = floats.shape[0]
        H, E, P, C = L.max_hand, L.max_enemies, L.max_potions, L.max_choices
        hand_feats = floats[:, L.f_hand : L.f_hand + H * L.hand_feats].view(B, H, L.hand_feats)
        enemy_floats = floats[:, L.f_enemies : L.f_relics].view(B, E, L.enemy_feats)
        potion_feats = floats[:, L.f_potions : L.f_potions + P].unsqueeze(2)
        choice_feats = floats[:, L.f_choices : L.f_choices + C * L.choice_feats].view(B, C, L.choice_feats)
        hand_ids = ids[:, L.i_hand : L.i_hand + H]
        potion_ids = ids[:, L.i_potions : L.i_potions + P]
        choice_ids = ids[:, L.i_choices : L.i_choices + C]

        enemies = self.enemy(
            torch.cat(
                [self.monster(ids[:, L.i_enemies : L.i_enemies + E]), self.move(ids[:, L.i_moves : L.i_moves + E]), enemy_floats],
                dim=2,
            )
        )
        choices = self.choice_in(torch.cat([self.card(choice_ids), choice_feats], dim=2))
        offered = (choice_ids != 0).unsqueeze(2)
        glob = self.glob(torch.cat([floats[:, : L.f_enemies], floats[:, L.f_relics :]], dim=1))
        glob = glob + (choices * offered).sum(1) / offered.sum(1).clamp(min=1)
        tokens = torch.cat(
            [
                glob.unsqueeze(1),
                self.hand_in(torch.cat([self.card(hand_ids), self.enchant(ids[:, L.i_enchants : L.i_enchants + H]), hand_feats], dim=2)),
                self.enemy_in(enemies),
                self.potion_in(torch.cat([self.potion(potion_ids), potion_feats], dim=2)),
            ]
            + ([self.pile_tokens(floats)] if self.arch.piles else []),
            dim=1,
        )
        # True where a slot is empty. The global token never is, so every
        # row attends to something; nor are the piles.
        empty = torch.cat(
            [torch.zeros_like(hand_ids[:, :1], dtype=torch.bool), hand_ids == 0, enemy_floats[:, :, 0] == 0, potion_ids == 0]
            + ([torch.zeros_like(hand_ids[:, :3], dtype=torch.bool)] if self.arch.piles else []),
            dim=1,
        )
        x = self.encoder(tokens, src_key_padding_mask=empty)
        g, hand, enemy, potion = x[:, : 1 + H + E + P].split([1, H, E, P], dim=1)
        g = g.squeeze(1)
        if self.arch.pointer:
            play, use = self.play(hand, enemy).flatten(1), self.use_potion(potion, enemy).flatten(1)
        else:
            play, use = self.play(g, hand, enemy).flatten(1), self.use_potion(g, potion, enemy).flatten(1)
        if self.arch.choice_attn:
            choices = choices + self.choice_attn(self.choice_norm(choices), x, x, key_padding_mask=empty, need_weights=False)[0]
        choose = self.choose(g, choices).squeeze(2)
        end_skip = self.end_or_skip(g)
        logits = torch.cat([play, use, end_skip[:, :1], choose, end_skip[:, 1:]], dim=1)
        return logits, g

    def pile_tokens(self, floats: Tensor) -> Tensor:
        """`[B, 3, d]`: the draw, discard and exhaust piles, from their
        counts per (card, upgraded) through the card embedding."""
        L = self.layout
        counts = floats[:, L.f_piles : L.f_piles + 3 * 2 * L.n_cards].view(-1, 3, L.n_cards, 2)
        cards = self.card.weight[1:].to(counts.dtype)
        summed = counts[..., 0] @ cards + counts[..., 1] @ (cards + self.pile_up.to(counts.dtype))
        size = counts.sum(dim=(2, 3)).unsqueeze(2)
        return self.pile_in(torch.cat([summed / size.clamp(min=1), size / 10], dim=2)) + self.pile_kind


def build_policy(layout: Layout, arch: Arch) -> Policy:
    if arch.kind == "slots":
        return SlotMLP(layout, hidden=arch.hidden, depth=arch.depth)
    if arch.kind == "attn":
        return SlotAttention(layout, arch.hidden, arch.depth, pointer=arch.pointer, piles=arch.piles, choice_attn=arch.choice_attn, incoming=arch.incoming)
    raise ValueError(f"unknown architecture {arch.kind!r}: slots or attn")


def checkpoint_arch(ck: object) -> Arch:
    """The architecture a checkpoint was trained with; `SlotMLP` at its old
    size for checkpoints from before that was recorded."""
    return Arch(**ck["arch"]) if isinstance(ck, dict) and isinstance(ck.get("arch"), dict) else Arch()


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


def warm_start(policy: Policy, state: dict[str, Tensor]) -> list[str]:
    """Loads every tensor of `state` whose name and shape `policy` shares,
    for a new architecture that starts from a trained one's torso. Returns
    the names left as initialised."""
    own = policy.state_dict()
    kept = {k: t for k, t in state.items() if k in own and own[k].shape == t.shape}
    policy.load_state_dict(kept, strict=False)
    return sorted(set(own) - set(kept))


def checkpoint_vocab(ck: object, fallback: Path | None) -> str | None:
    """The vocabulary a checkpoint was trained with: stored in it, or read
    from `fallback` for checkpoints from before that was recorded."""
    if isinstance(ck, dict) and isinstance(ck.get("vocab"), str):
        return ck["vocab"]
    return fallback.read_text() if fallback else None


def checkpoint_layout(ck: object) -> dict[str, int] | None:
    return ck.get("layout") if isinstance(ck, dict) and isinstance(ck.get("layout"), dict) else None


def load_policy(path: Path, device: torch.device, old_vocab: Path | None = None) -> Policy:
    """The network a training checkpoint (or a bare state dict) holds,
    built for the current sim, on `device`."""
    ck = torch.load(path, map_location=device)
    policy = build_policy(Layout.load(), checkpoint_arch(ck)).to(device)
    load_state(policy, ck["policy"] if "policy" in ck else ck, checkpoint_vocab(ck, old_vocab), checkpoint_layout(ck))
    return policy
