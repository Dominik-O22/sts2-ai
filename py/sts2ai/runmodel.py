"""The run policy (docs/run-env.md, Observation and scoring): a small
transformer over a run decision's tokens (`sim::runobs`), a pointer head
that scores each option token against the global token, and a value head
on the global token that estimates the run's reward from here.

Cards, enchantments, potions and relics index as the combat model and
`sts2ai.deckvalue` do (sim id + 1), so the card embedding can start from a
combat checkpoint's (`seed_cards`); relics the combat sim leaves out
follow its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from sts2ai.env import RunLayout
from sts2ai.model import Policy
from sts2ai.vocab import current_text


@dataclass(frozen=True)
class RunArch:
    hidden: int = 128
    depth: int = 2
    heads: int = 4


class RunPolicy(nn.Module):
    def __init__(self, layout: RunLayout, arch: RunArch | None = None, card_dim: int = 32):
        super().__init__()
        L = self.layout = layout
        self.arch = arch = arch or RunArch()
        d = arch.hidden
        self.card = nn.Embedding(L.card_vocab, card_dim, padding_idx=0)
        self.enchant = nn.Embedding(L.enchant_vocab, 8, padding_idx=0)
        self.relic = nn.Embedding(L.relic_vocab, 32, padding_idx=0)
        self.potion = nn.Embedding(L.potion_vocab, 16, padding_idx=0)
        self.decision = nn.Embedding(L.decision_vocab, 16, padding_idx=0)
        self.room = nn.Embedding(L.room_vocab, 8, padding_idx=0)
        self.act = nn.Embedding(L.act_vocab, 8, padding_idx=0)
        self.boss = nn.Embedding(L.boss_vocab, 16, padding_idx=0)
        self.event = nn.Embedding(L.event_vocab, 16, padding_idx=0)
        self.option = nn.Embedding(L.option_vocab, 16, padding_idx=0)
        self.event_key = nn.Embedding(L.event_key_vocab, 16, padding_idx=0)
        # One projection per token kind; their biases tell the kinds apart.
        self.global_in = nn.Linear(16 + 8 + 8 + 16 + 16 + 16 + L.global_floats, d)
        self.deck_in = nn.Linear(card_dim + 8 + L.deck_floats, d)
        self.relic_in = nn.Linear(32 + L.relic_floats, d)
        self.potion_in = nn.Linear(16 + L.potion_floats, d)
        self.option_in = nn.Linear(16 + L.option_cards * card_dim + 8 + 32 + 16 + 8 + 16 + L.option_floats, d)
        layer = nn.TransformerEncoderLayer(d, arch.heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, arch.depth, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.score_global = nn.Linear(d, d)
        self.score_option = nn.Linear(d, d, bias=False)
        self.score_out = nn.Linear(d, 1)
        nn.init.orthogonal_(self.score_out.weight, gain=0.01)
        nn.init.zeros_(self.score_out.bias)
        self.v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def tokens(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Token embeddings `[B, T, d]` and which are absent `[B, T]`."""
        L = self.layout
        B = floats.shape[0]

        def seg(x: Tensor, start: int, count: int, width: int) -> Tensor:
            return x[:, start : start + count * width].view(B, count, width)

        g_ids, g_f = ids[:, : L.global_ids], floats[:, : L.global_floats]
        glob = self.global_in(
            torch.cat(
                [
                    self.decision(g_ids[:, 0]),
                    self.room(g_ids[:, 1]),
                    self.act(g_ids[:, 2]),
                    self.boss(g_ids[:, 3]),
                    self.boss(g_ids[:, 4]),
                    self.event(g_ids[:, 5]),
                    g_f,
                ],
                dim=1,
            )
        ).unsqueeze(1)
        d_ids, d_f = seg(ids, L.i_deck, L.max_deck, L.deck_ids), seg(floats, L.f_deck, L.max_deck, L.deck_floats)
        deck = self.deck_in(torch.cat([self.card(d_ids[..., 0]), self.enchant(d_ids[..., 1]), d_f], dim=2))
        r_ids, r_f = seg(ids, L.i_relics, L.max_relics, L.relic_ids), seg(floats, L.f_relics, L.max_relics, L.relic_floats)
        relics = self.relic_in(torch.cat([self.relic(r_ids[..., 0]), r_f], dim=2))
        p_ids, p_f = seg(ids, L.i_potions, L.max_potions, L.potion_ids), seg(floats, L.f_potions, L.max_potions, L.potion_floats)
        potions = self.potion_in(torch.cat([self.potion(p_ids[..., 0]), p_f], dim=2))
        o_ids, o_f = seg(ids, L.i_options, L.max_options, L.option_ids), seg(floats, L.f_options, L.max_options, L.option_floats)
        C = L.option_cards
        options = self.option_in(
            torch.cat(
                [
                    self.option(o_ids[..., 0]),
                    self.card(o_ids[..., 1 : 1 + C]).flatten(2),
                    self.enchant(o_ids[..., 1 + C]),
                    self.relic(o_ids[..., 2 + C]),
                    self.potion(o_ids[..., 3 + C]),
                    self.room(o_ids[..., 4 + C]),
                    self.event_key(o_ids[..., 5 + C]),
                    o_f,
                ],
                dim=2,
            )
        )
        tokens = torch.cat([glob, deck, relics, potions, options], dim=1)
        absent = torch.cat([torch.zeros_like(g_f[:, :1], dtype=torch.bool), d_f[..., 0] == 0, r_f[..., 0] == 0, p_f[..., 0] == 0, o_f[..., 0] == 0], dim=1)
        return tokens, absent

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Masked option logits `[B, max_options]` and values `[B]`."""
        L = self.layout
        tokens, absent = self.tokens(floats, ids)
        # The slots are sized for the largest deck and option list; a batch
        # uses a few dozen of them. Only the token positions some row uses
        # go through the encoder, and option logits go back to their slots.
        used = ~absent.all(0)
        x = self.encoder(tokens[:, used], src_key_padding_mask=absent[:, used])
        options_used = used[-L.max_options :]
        n = int(options_used.sum())
        g, options = x[:, 0], x[:, x.shape[1] - n :]
        scores = self.score_out(torch.relu(self.score_global(g).unsqueeze(1) + self.score_option(options))).squeeze(2).float()
        logits = torch.full((floats.shape[0], L.max_options), -1e9, device=scores.device)
        logits[:, options_used] = scores
        logits = logits.masked_fill(absent[:, -L.max_options :], -1e9)
        return logits, self.v(g).squeeze(1).float()

    def seed_cards(self, combat: Policy) -> None:
        """Start the card embedding from a combat policy's: both index a
        card as its sim id + 1."""
        with torch.no_grad():
            n = min(self.card.num_embeddings, combat.card.num_embeddings)
            self.card.weight[:n].copy_(combat.card.weight[:n].to(self.card.weight.device))


def save_run_policy(path: Path, policy: RunPolicy, opt: torch.optim.Optimizer | None, **extra: object) -> None:
    """A run checkpoint records its arch, the run layout and the vocabulary
    it was trained with, as combat checkpoints do."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": opt.state_dict() if opt else None,
            "arch": asdict(policy.arch),
            "layout": asdict(policy.layout),
            "vocab": current_text(),
            **extra,
        },
        tmp,
    )
    tmp.replace(path)


def load_run_policy(path: Path, device: torch.device) -> tuple[RunPolicy, dict]:
    """The run policy a checkpoint holds, and the checkpoint. A layout or
    vocabulary that moved since is refused: run checkpoints have no remap
    yet."""
    ck = torch.load(path, map_location=device, weights_only=False)
    layout = RunLayout.load()
    if ck["layout"] != asdict(layout):
        raise ValueError(f"{path}: the run layout changed since this checkpoint; retrain")
    if ck["vocab"] != current_text():
        raise ValueError(f"{path}: the vocabulary changed since this checkpoint; retrain")
    policy = RunPolicy(layout, RunArch(**ck["arch"])).to(device)
    policy.load_state_dict(ck["policy"])
    return policy, ck
