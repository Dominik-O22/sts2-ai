"""Print lost held-out fights turn by turn, decoded from the observation.

    uv run python -m sts2ai.fightlog runs/<run>/latest.pt --kind Boss --max 3

A debugging aid for reading what the policy does, not a tool for the run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import Envs, Layout
from sts2ai.evaluate import HOLDOUT_PER_ENCOUNTER
from sts2ai.model import Policy, load_policy, masked_logits

# Power indices from ids.rs, the ones worth printing.
POWER_NAMES = {0: "Str", 1: "Dex", 2: "Vuln", 3: "Weak", 4: "Frail", 33: "Slippery", 37: "Ringing", 38: "Plow", 43: "Artifact"}


@dataclass
class Offsets:
    """Float offsets `sim::encode` does not export, derived from the ones it does."""

    n_powers: int
    f_player_powers: int
    f_enemies: int
    enemy_feats: int

    @classmethod
    def of(cls, L: Layout) -> Offsets:
        f_player_powers = 30  # GLOBAL_LEN
        n_powers = L.f_hand - f_player_powers
        f_enemies = L.f_hand + L.max_hand * L.hand_feats + 3 * 2 * (L.card_vocab - 1)
        return cls(n_powers, f_player_powers, f_enemies, 7 + 15 + n_powers)


def powers(vec: np.ndarray) -> str:
    return " ".join(f"{POWER_NAMES.get(i, f'p{i}')}={int(round(v * 10))}" for i, v in enumerate(vec) if v != 0)


def describe_state(L: Layout, O: Offsets, f: np.ndarray, ids: np.ndarray, cards: list[str], monsters: list[str]) -> str:
    hp, block, energy, turn = f[0] * 100, f[3] * 50, f[4] * 5, f[6] * 10
    hand = [cards[i] for i in ids[L.i_hand : L.i_hand + L.max_hand] if i]
    enemies = []
    for s in range(L.max_enemies):
        e = f[O.f_enemies + s * O.enemy_feats :][: O.enemy_feats]
        if not e[0] or not e[1]:
            continue
        intent = f"hits {e[8] * 20:.0f}x{e[9] * 3:.0f}" if e[7] else " ".join(
            n for n, k in [("defend", 11), ("buff", 12), ("debuff", 13), ("status", 16), ("summon", 18), ("stun", 20)] if e[k]
        )
        enemies.append(f"{monsters[ids[L.i_enemies + s]]} {e[2] * 100:.0f}hp b{e[5] * 30:.0f} [{intent}] {powers(e[22:])}")
    out = f"T{turn:.0f} hp{hp:.0f} b{block:.0f} e{energy:.0f} {powers(f[O.f_player_powers : L.f_hand])}\n"
    out += "  hand: " + ", ".join(hand) + "\n"
    return out + "".join(f"  vs {e}\n" for e in enemies)


def describe_action(L: Layout, a: int, ids: np.ndarray, cards: list[str]) -> str:
    if a < L.a_potion:
        slot, t = divmod(a, L.targets)
        target = "" if t == L.max_enemies else f" -> enemy {t}"
        return f"play {cards[ids[L.i_hand + slot]]}{target}"
    if a < L.a_end_turn:
        slot, t = divmod(a - L.a_potion, L.targets)
        return f"potion slot {slot}" + ("" if t == L.max_enemies else f" -> enemy {t}")
    if a == L.a_end_turn:
        return "END TURN"
    if a < L.a_skip:
        return f"choose {cards[ids[L.i_choices + a - L.a_choose]]}"
    return "skip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--kind", default="Boss")
    ap.add_argument("--max", type=int, default=3, help="lost fights to print")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = Layout.load()
    O = Offsets.of(L)
    cards, monsters = _sim.card_names(), _sim.monster_names()
    policy = Policy(L).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()

    probe = Envs(1, seed=args.seed)
    n = probe.use_holdout(args.seed, HOLDOUT_PER_ENCOUNTER)
    envs = Envs(n, seed=args.seed)
    envs.use_holdout(args.seed, HOLDOUT_PER_ENCOUNTER)
    logs: list[list[str]] = [[] for _ in range(n)]
    done = [False] * n
    printed = 0
    with torch.no_grad():
        while not all(done) and printed < args.max:
            floats = torch.from_numpy(envs.floats).to(device)
            ids = torch.from_numpy(envs.ids).to(device)
            mask = torch.from_numpy(envs.mask).to(device)
            logits, value = policy(floats, ids)
            probs = torch.softmax(masked_logits(logits.float(), mask), dim=1)
            actions = probs.argmax(dim=1).cpu().numpy()
            for i in range(n):
                if done[i]:
                    continue
                f, idv = envs.floats[i], envs.ids[i]
                state = describe_state(L, O, f, idv, cards, monsters)
                act = describe_action(L, int(actions[i]), idv, cards)
                logs[i].append(f"{state}  => {act}  (p={probs[i, actions[i]]:.2f}, v={value[i]:.2f})\n")
            for e in envs.step(actions):
                if done[e.env]:
                    continue
                done[e.env] = True
                if e.kind == args.kind and not e.won and printed < args.max:
                    printed += 1
                    print(f"===== LOST {e.encounter} floor {e.floor} after {e.steps} actions =====")
                    print("".join(logs[e.env]))


if __name__ == "__main__":
    main()
