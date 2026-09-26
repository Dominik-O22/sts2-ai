"""The run policy (docs/run-env.md, Observation and scoring): a small
transformer over a run decision's tokens (`sim::runobs`), a pointer head
that scores each option token against the global token, and a value head
on the global token that estimates the run's reward from here.

At a map step, a map encoder reads the map ahead backwards, from the row
below the boss to the options' row, and each path option's token gains
its node's reading (`RunPolicy.map_ahead`).

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
from sts2ai.vocab import current_text, parse


@dataclass(frozen=True)
class RunArch:
    hidden: int = 128
    depth: int = 2
    heads: int = 4
    map_dim: int = 64


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
        m = arch.map_dim
        self.map_row = nn.Embedding(L.map_rows, 8)
        # A node's reading: its own room type, row and the run's state, and
        # its children's readings, mixed.
        self.map_here = nn.Linear(8 + 8 + d, m)
        self.map_children = nn.Linear(2 * m, m, bias=False)
        self.map_mix = nn.Sequential(nn.ReLU(), nn.Linear(m, m), nn.LayerNorm(m))
        self.map_out = nn.Linear(m, d, bias=False)
        layer = nn.TransformerEncoderLayer(d, arch.heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, arch.depth, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.score_global = nn.Linear(d, d)
        self.score_option = nn.Linear(d, d, bias=False)
        self.score_out = nn.Linear(d, 1)
        nn.init.orthogonal_(self.score_out.weight, gain=0.01)
        nn.init.zeros_(self.score_out.bias)
        self.v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def tokens(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Token embeddings `[B, T, d]`, which are absent `[B, T]`, and
        which token positions some row uses `[T]`, on the CPU."""
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
        absent = torch.cat([torch.zeros_like(g_f[:, :1], dtype=torch.bool), d_f[..., 0] == 0, r_f[..., 0] == 0, p_f[..., 0] == 0, o_f[..., 0] == 0], dim=1)
        nodes = ids[:, L.i_map : L.i_map + L.map_rows * L.map_cols * L.map_node_ids].view(B, L.map_rows, L.map_cols, L.map_node_ids)
        # The one wait on the GPU: which token positions and map rows the
        # batch uses. Everything sized by them is sized on the CPU after.
        sizes = torch.cat([~absent.all(0), (nodes[..., 0] != 0).any(2).any(0)]).cpu()
        used, map_rows = sizes[: absent.shape[1]], sizes[absent.shape[1] :]
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
        if map_rows.any():
            def mean(tokens: Tensor, f: Tensor) -> Tensor:
                present = (f[..., :1] != 0).to(tokens.dtype)
                return (tokens * present).sum(1) / present.sum(1).clamp(min=1)

            depth = int(map_rows.nonzero().max()) + 1
            state = glob[:, 0] + mean(deck, d_f) + mean(relics, r_f)
            options = options + self.map_ahead(nodes[:, :depth], state, o_ids[..., 6 + C]).to(options.dtype)
        tokens = torch.cat([glob, deck, relics, potions, options], dim=1)
        return tokens, absent, used

    def map_ahead(self, nodes: Tensor, state: Tensor, column: Tensor) -> Tensor:
        """What each path option's node reads, as a token `[B, max_options,
        d]` to add to the option's (zero for other options and other
        decisions). `nodes` is the map segment's rows the batch uses, `state`
        the run's (the global token plus the mean deck and relic tokens),
        `column` each option's map column + 1.

        The map ahead is a DAG laid out by row and column; a node leads to
        at most the three nodes above it. One step per row, from the last
        row down: a node's reading mixes its room type, its row and the
        run's state with the max and mean of its children's readings (the
        max is the best the player can still choose there, the mean what
        the paths average), so a remerge is read once and an elite reads
        differently at low HP. It runs in float32 on every row of the
        batch, with no wait on the GPU and a few kernels per map row: this
        loop is most of a map step's forward."""
        B, depth, C, _ = nodes.shape
        m = self.arch.map_dim
        kind, links = nodes[..., 0], nodes[..., 1]
        with torch.autocast(nodes.device.type, enabled=False):
            here = torch.cat(
                [self.room(kind), self.map_row.weight[:depth, None].expand(B, depth, C, 8), state.float()[:, None, None].expand(B, depth, C, state.shape[1])],
                dim=3,
            )
            here = self.map_here(here)
            # Node c's children sit at columns c - 1, c, c + 1 of the row
            # above: link bits 0, 1, 2, and window 0, 1, 2 of `unfold`.
            linked = (links[..., None] >> torch.arange(3, device=nodes.device) & 1).float()
            count = linked.sum(3, keepdim=True)
            weights = (linked / count.clamp(min=1)).unsqueeze(4)
            unlinked = ((1 - linked) * -1e4).unsqueeze(3)
            has_children = (count > 0).float()
            present = (kind != 0).float().unsqueeze(3)
            h = here.new_zeros(B, C, m)
            for r in reversed(range(depth)):
                children = nn.functional.pad(h, (0, 0, 1, 1)).unfold(1, 3, 1)
                mean = (children @ weights[:, r]).squeeze(3)
                best = (children + unlinked[:, r]).amax(3) * has_children[:, r]
                h = self.map_mix(here[:, r] + self.map_children(torch.cat([mean, best], dim=2))) * present[:, r]
            # Each path option names its node's column in the options' row.
            picked = torch.gather(h, 1, (column - 1).clamp(min=0)[..., None].expand(-1, -1, m)) * (column > 0)[..., None]
            return self.map_out(picked)

    def forward(self, floats: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Masked option logits `[B, max_options]` and values `[B]`."""
        L = self.layout
        # The slots are sized for the largest deck and option list; a batch
        # uses a few dozen of them. Only the token positions some row uses
        # go through the encoder, and option logits go back to their slots.
        tokens, absent, used = self.tokens(floats, ids)
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


# Each embedding whose rows follow a vocabulary: its size in `RunLayout`,
# and the `vocab.txt` kinds whose names index its rows after the pad.
EMBEDDINGS = {
    "card": ("card_vocab", ("card",)),
    "enchant": ("enchant_vocab", ("enchant",)),
    "potion": ("potion_vocab", ("potion",)),
    "relic": ("relic_vocab", ("relic", "runrelic")),
    "decision": ("decision_vocab", ("decision",)),
    "option": ("option_vocab", ("option",)),
    "room": ("room_vocab", ("room",)),
    "act": ("act_vocab", ("act",)),
    "event": ("event_vocab", ("event",)),
    "boss": ("boss_vocab", ("boss",)),
}


def remap_run_state(state: dict[str, Tensor], old_text: str, new_text: str, fresh: dict[str, Tensor]) -> dict[str, Tensor]:
    """`state` for the vocabulary `new_text`: each embedding row moves to
    its name's new index, and names new since keep their rows in `fresh`."""
    old_v, new_v = parse(old_text), parse(new_text)
    out = dict(state)
    for table, (_, kinds) in EMBEDDINGS.items():
        key = f"{table}.weight"
        old_names = [n for k in kinds for n in old_v.get(k, [])]
        index = {n: i for i, n in enumerate(n for k in kinds for n in new_v.get(k, []))}
        rows = fresh[key].clone()
        rows[0] = state[key][0]
        for i, name in enumerate(old_names):
            if name in index:
                rows[index[name] + 1] = state[key][i + 1]
        out[key] = rows
    return out


# Float segments that have grown by columns appended at their end, and the
# input projection whose last columns read them; and the layout fields that
# only follow from them.
GROWN = {"global_floats": "global_in.weight", "option_floats": "option_in.weight"}
FOLLOWS = {"run_floats", "f_deck", "f_relics", "f_potions", "f_options", "f_forecast", "forecast_floats", "forecast_rolls", "map_feats"}


def grow_floats(state: dict[str, Tensor], old: dict[str, int], new: RunLayout) -> dict[str, Tensor]:
    """`state` for a layout whose float segments `GROWN` gained columns at
    their end: the new columns start at zero weight, so the policy reads
    as it did (a run policy from before the forecast ignores it)."""
    out = dict(state)
    for field, key in GROWN.items():
        extra = getattr(new, field) - old[field]
        if extra:
            w = state[key]
            out[key] = torch.cat([w, w.new_zeros(w.shape[0], extra)], dim=1)
    return out


def load_run_policy(path: Path, device: torch.device) -> tuple[RunPolicy, dict]:
    """The run policy a checkpoint holds, and the checkpoint. Ids the sim
    gained since move to their new rows (`remap_run_state`), and float
    columns appended since read at zero weight (`grow_floats`); a layout
    that moved otherwise is refused."""
    ck = torch.load(path, map_location=device, weights_only=False)
    layout = RunLayout.load()
    vocab_sizes = {size for size, _ in EMBEDDINGS.values()}
    moved = {k for k, v in asdict(layout).items() if ck["layout"].get(k) != v} - vocab_sizes
    grown = all(ck["layout"][f] <= getattr(layout, f) for f in GROWN)
    if moved - FOLLOWS - set(GROWN) or not grown:
        raise ValueError(f"{path}: the run layout changed since this checkpoint ({', '.join(sorted(moved))}); retrain")
    policy = RunPolicy(layout, RunArch(**ck["arch"])).to(device)
    state = ck["policy"]
    if moved:
        state = grow_floats(state, ck["layout"], layout)
    if ck["vocab"] != current_text():
        state = remap_run_state(state, ck["vocab"], current_text(), policy.state_dict())
    policy.load_state_dict(state)
    return policy, ck
