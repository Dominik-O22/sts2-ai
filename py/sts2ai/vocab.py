"""Vocabularies the policy embeds, and remapping a checkpoint across
vocabulary growth.

`sim/vocab.txt` lists every id in index order (`sim::encode::vocab_text`).
A checkpoint stores the text it was trained with. When the sim gains cards,
powers, monsters, relics, potions, moves, enchantments, or intent kinds,
the embedding rows and the input columns of the torso and the enemy
encoder move; this module maps them old name to new index so training
resumes. Layout constants (slot counts, per-slot feature counts) are not
remapped: those need a retrain (`model.SHAPE_FIELDS`).
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from sts2ai.env import Layout

Vocab = dict[str, list[str]]
# (segment name, length, vocabulary kind or None for a fixed block).
Segments = list[tuple[str, int, str | None]]

# Override with STS2AI_VOCAB to describe a sim other than the checked-in one.
VOCAB_FILE = Path(os.environ.get("STS2AI_VOCAB", Path(__file__).resolve().parents[2] / "sim" / "vocab.txt"))


def parse(text: str) -> Vocab:
    out: Vocab = {}
    for line in text.splitlines():
        if line.strip():
            kind, name = line.split(" ", 1)
            out.setdefault(kind, []).append(name)
    return out


@cache
def current_text() -> str:
    """The pinned vocabulary of the sim this package was built from."""
    return VOCAB_FILE.read_text()


def layout_for(v: Vocab, base: Layout) -> Layout:
    """The Layout `sim::encode` would produce for vocabulary `v`, taking the
    vocabulary-independent sizes from `base`. Used to describe checkpoints
    from a differently sized sim."""
    n_cards, n_powers, n_relics, n_intents = (len(v.get(k, [])) for k in ("card", "power", "relic", "intent"))
    f_hand = base.global_len + n_powers
    f_piles = f_hand + base.max_hand * base.hand_feats
    f_enemies = f_piles + 3 * 2 * n_cards
    enemy_feats = base.enemy_base + n_intents + base.intent_nums + n_powers
    f_relics = f_enemies + base.max_enemies * enemy_feats
    f_potions = f_relics + 2 * n_relics
    f_choices = f_potions + base.max_potions
    return replace(
        base,
        n_floats=f_choices + base.max_choices * base.choice_feats,
        f_hand=f_hand,
        f_piles=f_piles,
        f_enemies=f_enemies,
        enemy_feats=enemy_feats,
        f_relics=f_relics,
        f_potions=f_potions,
        f_choices=f_choices,
        n_cards=n_cards,
        n_powers=n_powers,
        n_relics=n_relics,
        n_intents=n_intents,
        card_vocab=n_cards + 1,
        monster_vocab=len(v["monster"]) + 1,
        potion_vocab=len(v["potion"]) + 1,
        move_vocab=len(v["move"]) + 1,
        enchant_vocab=len(v.get("enchant", [])) + 1,
    )


def float_segments(L: Layout) -> Segments:
    """The dense observation in layout order, mirroring `sim::encode`.
    Per-enemy blocks are listed too, so offsets can be read off, but the
    torso does not see them (`torso_segments`)."""
    segs: Segments = [
        ("global", L.global_len, None),
        ("player_powers", L.n_powers, "power"),
        ("hand", L.max_hand * L.hand_feats, None),
    ]
    for pile in ("draw", "discard", "exhaust"):
        segs.append((f"pile_{pile}", 2 * L.n_cards, "card2"))
    for i in range(L.max_enemies):
        segs.extend((f"enemy{i}_{name}", n, kind) for name, n, kind in enemy_segments(L))
    segs += [
        ("relics_hot", L.n_relics, "relic"),
        ("relics_counter", L.n_relics, "relic"),
        ("potions", L.max_potions, None),
        ("choices", L.max_choices * L.choice_feats, None),
    ]
    assert sum(n for _, n, _ in segs) == L.n_floats, (
        f"segments sum to {sum(n for _, n, _ in segs)} floats against a layout of {L.n_floats}; if this is the built sim, "
        "regenerate sim/vocab.txt with `cargo run --release --example vocab > vocab.txt` in sim/"
    )
    return segs


def enemy_segments(L: Layout) -> Segments:
    """One enemy's dense block."""
    return [("base", L.enemy_base, None), ("intent", L.n_intents, "intent"), ("intent_nums", L.intent_nums, None), ("powers", L.n_powers, "power")]


def torso_segments(L: Layout, dims: dict[str, int]) -> Segments:
    """Columns of the torso's first layer: the observation without the enemy
    blocks, then the embedding concat (`Policy.forward`)."""
    segs = [s for s in float_segments(L) if not s[0].startswith("enemy")]
    return segs + [("emb", dims["torso_emb"], None)]


def enemy_input_segments(L: Layout, dims: dict[str, int]) -> Segments:
    """Columns of the enemy encoder's first layer."""
    return [("emb", dims["enemy_emb"], None)] + enemy_segments(L)


def offsets(segs: Segments) -> dict[str, int]:
    out, off = {}, 0
    for name, n, _ in segs:
        out[name], off = off, off + n
    return out


def column_map(old_v: Vocab, new_v: Vocab, old_segs: Segments, new_segs: Segments) -> tuple[list[int], list[int]]:
    """(old columns, new columns) that hold the same named feature."""
    src: list[int] = []
    dst: list[int] = []
    new_off, old_off = offsets(new_segs), offsets(old_segs)
    new_len = {name: n for name, n, _ in new_segs}
    for name, n, kind in old_segs:
        a, b = old_off[name], new_off[name]
        if kind is None:
            assert n == new_len[name], f"fixed segment {name} changed size"
            src += range(a, a + n)
            dst += range(b, b + n)
        elif kind == "card2":
            index = {c: i for i, c in enumerate(new_v["card"])}
            for i, c in enumerate(old_v["card"]):
                src += [a + 2 * i, a + 2 * i + 1]
                dst += [b + 2 * index[c], b + 2 * index[c] + 1]
        else:
            index = {c: i for i, c in enumerate(new_v.get(kind, []))}
            src += [a + i for i in range(len(old_v.get(kind, [])))]
            dst += [b + index[c] for c in old_v.get(kind, [])]
    return src, dst


def remap_state(state: dict[str, Tensor], old_v: Vocab, new_v: Vocab, policy_state: dict[str, Tensor], L: Layout) -> dict[str, Tensor]:
    """A copy of `state` laid out for `new_v`. Embedding rows and the input
    columns of the torso (`SlotAttention`'s `glob`) and enemy encoder move
    by name; new entries keep the init in `policy_state`. Anything else
    must already match in shape."""
    out = dict(state)
    tables = {"card.weight": "card", "monster.weight": "monster", "move.weight": "move", "potion.weight": "potion", "enchant.weight": "enchant"}
    for key, kind in tables.items():
        new = policy_state[key].clone()
        index = {c: i for i, c in enumerate(new_v.get(kind, []))}
        for i, c in enumerate(old_v.get(kind, [])):
            new[index[c] + 1] = state[key][i + 1]  # row 0 is the pad
        new[0] = state[key][0]
        out[key] = new
    old_L = layout_for(old_v, L)
    inputs = (("torso.0.weight", torso_segments), ("glob.weight", torso_segments), ("enemy.0.weight", enemy_input_segments))
    for key, segs in inputs:
        if key not in policy_state:
            continue
        # Embedding widths do not depend on the vocabulary, so read them off
        # the new weights.
        width = policy_state[key].shape[1]
        dims = {"torso_emb": width - (L.n_floats - L.max_enemies * L.enemy_feats), "enemy_emb": width - L.enemy_feats}
        old_w, new_w = state[key], policy_state[key].clone()
        src, dst = column_map(old_v, new_v, segs(old_L, dims), segs(L, dims))
        new_w[:, dst] = old_w[:, src]
        out[key] = new_w
    for key, t in out.items():
        assert t.shape == policy_state[key].shape, f"{key}: {tuple(t.shape)} vs {tuple(policy_state[key].shape)}"
    return out


if __name__ == "__main__":
    # Self-check, independent of the built sim: take the pinned vocabulary,
    # grow every kind by appending, build policies for both layouts, remap
    # the old weights into the new one, and confirm the outputs agree on an
    # observation padded with zeros at the new columns.
    from sts2ai.model import Arch, build_policy

    L0 = Layout.load()
    v = parse(current_text())
    L = layout_for(v, L0)
    assert L == L0, "layout_for does not reproduce the built sim's layout"
    grown = {k: list(names) for k, names in v.items()}
    for kind, extra in [("card", 3), ("power", 2), ("relic", 4), ("monster", 1), ("potion", 2), ("move", 5), ("enchant", 2), ("intent", 3)]:
        grown[kind] += [f"NEW_{kind}_{i}" for i in range(extra)]
    GL = layout_for(grown, L0)

    floats = torch.rand(8, L.n_floats)
    ids = torch.zeros(8, L.n_ids, dtype=torch.long)
    ids[:, L.i_hand : L.i_hand + 3] = torch.tensor([1, 5, 9])
    ids[:, L.i_enchants] = 2
    ids[:, L.i_enemies : L.i_enemies + 2] = torch.tensor([2, 4])
    ids[:, L.i_moves : L.i_moves + 2] = torch.tensor([3, 7])
    src, dst = column_map(v, grown, float_segments(L), float_segments(GL))
    gf = torch.zeros(8, GL.n_floats)
    gf[:, dst] = floats[:, src]
    for arch in (Arch("slots"), Arch("attn", 128, 3)):
        torch.manual_seed(0)
        old, new = build_policy(L, arch).eval(), build_policy(GL, arch).eval()
        new.load_state_dict(remap_state(old.state_dict(), v, grown, new.state_dict(), GL))
        with torch.no_grad():
            lo, vo = old(floats, ids)
            ln, vn = new(gf, ids)
        assert torch.allclose(lo, ln, atol=1e-5) and torch.allclose(vo, vn, atol=1e-5), f"remapped {arch.kind} policy differs"
    print("remap self-check ok:", f"floats {L.n_floats} -> {GL.n_floats}")
