"""Vocabularies the policy embeds, and remapping a checkpoint across
vocabulary growth.

`sim/vocab.txt` lists every id in index order (`sim::encode::vocab_text`).
A checkpoint stores the text it was trained with. When the sim gains cards,
powers, monsters, relics, potions, or moves, both the embedding rows and
the columns of the first torso layer (the dense observation) move; this
module maps them old name to new index so training resumes.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from sts2ai.env import Layout

Vocab = dict[str, list[str]]

# Override with STS2AI_VOCAB to describe a sim other than the checked-in one.
VOCAB_FILE = Path(os.environ.get("STS2AI_VOCAB", Path(__file__).resolve().parents[2] / "sim" / "vocab.txt"))


def parse(text: str) -> Vocab:
    out: Vocab = {}
    for line in text.splitlines():
        if line.strip():
            kind, name = line.split(" ", 1)
            out.setdefault(kind, []).append(name)
    return out


def current_text() -> str:
    """The pinned vocabulary of the sim this package was built from."""
    return VOCAB_FILE.read_text()


@cache
def _fixed() -> tuple[int, int]:
    """Lengths of the global block and of one enemy's own features: the two
    parts of the dense observation no vocabulary sizes. Read off the sim
    this package was built from, since a smaller vocabulary does not make
    them smaller."""
    L, v = Layout.load(), parse(current_text())
    n_powers, n_relics = len(v["power"]), len(v["relic"])
    f_enemies = L.f_hand + L.max_hand * (L.hand_feats + len(v["enchant"])) + 3 * 2 * len(v["card"])
    enemy_feats = (L.f_potions - 2 * n_relics - f_enemies) // L.max_enemies
    return L.f_hand - n_powers, enemy_feats - n_powers


def float_segments(L: Layout, v: Vocab) -> list[tuple[str, int, str | None]]:
    """The dense observation as (segment name, length, vocabulary kind or
    None) in layout order, mirroring `sim::encode`. `L` must be the layout
    `v` produces, as `layout_for` builds it."""
    n_cards, n_powers, n_relics = len(v["card"]), len(v["power"]), len(v["relic"])
    # A vocabulary from before enchantments existed simply has none.
    n_ench = len(v.get("enchant", []))
    global_len, enemy_base = _fixed()
    segs: list[tuple[str, int, str | None]] = [
        ("global", global_len, None),
        ("player_powers", n_powers, "power"),
        ("hand", L.max_hand * L.hand_feats, None),
    ]
    for i in range(L.max_hand):
        segs.append((f"hand{i}_ench", n_ench, "enchant"))
    for pile in ("draw", "discard", "exhaust"):
        segs.append((f"pile_{pile}", 2 * n_cards, "card2"))
    for i in range(L.max_enemies):
        segs.append((f"enemy{i}_base", enemy_base, None))
        segs.append((f"enemy{i}_powers", n_powers, "power"))
    segs.append(("relics_hot", n_relics, "relic"))
    segs.append(("relics_counter", n_relics, "relic"))
    segs.append(("potions", L.max_potions, None))
    segs.append(("choices", L.max_choices * L.choice_feats, None))
    assert sum(n for _, n, _ in segs) == L.n_floats, (
        f"vocabulary of {sum(n for _, n, _ in segs)} floats against a layout of {L.n_floats}; if this is the built sim, "
        "regenerate sim/vocab.txt with `cargo run --release --example vocab > vocab.txt` in sim/"
    )
    return segs


def _column_map(old_v: Vocab, new_v: Vocab, old_segs, new_segs) -> list[tuple[int, int]]:
    """(old column, new column) pairs for the dense observation."""
    pairs: list[tuple[int, int]] = []
    old_off = new_off = 0
    new_by_name = {name: (off, n, kind) for (name, n, kind), off in zip(new_segs, _offsets(new_segs))}
    for (name, n, kind), old_start in zip(old_segs, _offsets(old_segs)):
        new_start, new_n, _ = new_by_name[name]
        if kind is None:
            assert n == new_n, f"fixed segment {name} changed size"
            pairs.extend((old_start + i, new_start + i) for i in range(n))
        elif kind == "card2":
            index = {c: i for i, c in enumerate(new_v["card"])}
            for i, c in enumerate(old_v["card"]):
                j = index[c]
                pairs.append((old_start + 2 * i, new_start + 2 * j))
                pairs.append((old_start + 2 * i + 1, new_start + 2 * j + 1))
        else:
            index = {c: i for i, c in enumerate(new_v.get(kind, []))}
            pairs.extend((old_start + i, new_start + index[c]) for i, c in enumerate(old_v.get(kind, [])))
    return pairs


def layout_for(v: Vocab, base: Layout) -> Layout:
    """The Layout `sim::encode` would produce for vocabulary `v`, taking the
    vocabulary-independent sizes from `base`. Used to describe checkpoints
    from a differently sized sim."""
    from dataclasses import replace

    cur = parse(current_text())
    d_cards, d_powers, d_relics = (len(v[k]) - len(cur[k]) for k in ("card", "power", "relic"))
    d_ench = len(v.get("enchant", [])) - len(cur.get("enchant", []))
    f_hand = base.f_hand + d_powers
    f_potions = (
        base.f_potions + d_powers + base.max_hand * d_ench + 3 * 2 * d_cards + base.max_enemies * d_powers + 2 * d_relics
    )
    return replace(
        base,
        n_floats=base.n_floats + (f_potions - base.f_potions),
        f_hand=f_hand,
        f_potions=f_potions,
        f_choices=f_potions + base.max_potions,
        card_vocab=len(v["card"]) + 1,
        monster_vocab=len(v["monster"]) + 1,
        potion_vocab=len(v["potion"]) + 1,
        move_vocab=len(v["move"]) + 1,
    )


def _offsets(segs) -> list[int]:
    out, off = [], 0
    for _, n, _ in segs:
        out.append(off)
        off += n
    return out


def remap_state(state: dict[str, Tensor], old_v: Vocab, new_v: Vocab, policy_state: dict[str, Tensor], L: Layout) -> dict[str, Tensor]:
    """A copy of `state` laid out for `new_v`. Embedding rows and the
    torso's input columns move by name; new entries keep the init in
    `policy_state`. Anything else must already match in shape."""
    out = dict(state)
    tables = {"card.weight": "card", "monster.weight": "monster", "move.weight": "move", "potion.weight": "potion"}
    for key, kind in tables.items():
        new = policy_state[key].clone()
        index = {c: i for i, c in enumerate(new_v[kind])}
        for i, c in enumerate(old_v[kind]):
            new[index[c] + 1] = state[key][i + 1]  # row 0 is the pad
        new[0] = state[key][0]
        out[key] = new
    # The checkpoint's own layout: its smaller vocabularies made it narrower.
    old_segs, new_segs = float_segments(layout_for(old_v, Layout.load()), old_v), float_segments(L, new_v)
    old_floats, new_floats = sum(n for _, n, _ in old_segs), L.n_floats
    old_w, new_w = state["torso.0.weight"], policy_state["torso.0.weight"].clone()
    pairs = _column_map(old_v, new_v, old_segs, new_segs)
    # The embedding concat after the floats keeps its shape; it only shifts.
    pairs.extend((old_floats + i, new_floats + i) for i in range(old_w.shape[1] - old_floats))
    src = torch.tensor([a for a, _ in pairs], device=old_w.device)
    dst = torch.tensor([b for _, b in pairs], device=old_w.device)
    new_w[:, dst] = old_w[:, src]
    out["torso.0.weight"] = new_w
    for key, t in out.items():
        assert t.shape == policy_state[key].shape, f"{key}: {tuple(t.shape)} vs {tuple(policy_state[key].shape)}"
    return out


if __name__ == "__main__":
    # Self-check, independent of the built sim: take the pinned vocabulary,
    # grow every kind by appending, build policies for both layouts, remap
    # the old weights into the new one, and confirm the outputs agree on an
    # observation padded with zeros at the new columns.
    L0 = Layout.load()
    v = parse(current_text())
    L = layout_for(v, L0)
    grown = {k: list(names) for k, names in v.items()}
    for kind, extra in [("card", 3), ("power", 2), ("relic", 4), ("monster", 1), ("potion", 2), ("move", 5), ("enchant", 2)]:
        grown[kind] += [f"NEW_{kind}_{i}" for i in range(extra)]
    GL = layout_for(grown, L0)
    from sts2ai.model import Policy

    torch.manual_seed(0)
    old, new = Policy(L), Policy(GL)
    new.load_state_dict(remap_state(old.state_dict(), v, grown, new.state_dict(), GL))
    floats = torch.rand(8, L.n_floats)
    ids = torch.zeros(8, L.n_ids, dtype=torch.long)
    ids[:, L.i_hand : L.i_hand + 3] = torch.tensor([1, 5, 9])
    ids[:, L.i_enemies] = 2
    ids[:, L.i_moves] = 3
    pairs = _column_map(v, grown, float_segments(L, v), float_segments(GL, grown))
    gf = torch.zeros(8, GL.n_floats)
    gf[:, [b for _, b in pairs]] = floats[:, [a for a, _ in pairs]]
    with torch.no_grad():
        lo, vo = old(floats, ids)
        ln, vn = new(gf, ids)
    assert torch.allclose(lo, ln, atol=1e-5) and torch.allclose(vo, vn, atol=1e-5), "remapped policy differs"
    print("remap self-check ok:", f"floats {L.n_floats} -> {GL.n_floats}")
