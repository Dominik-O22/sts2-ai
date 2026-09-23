"""Print lost held-out fights turn by turn, decoded from the observation.

    uv run python -m sts2ai.fightlog runs/<run>/latest.pt --kind Boss --max 3

A debugging aid for reading what the policy does, not a tool for the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import Envs, Layout
from sts2ai.evaluate import HOLDOUT_PER_ENCOUNTER
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.vocab import current_text, parse

# Power indices from ids.rs, the ones worth printing.
POWER_NAMES = {0: "Str", 1: "Dex", 2: "Vuln", 3: "Weak", 4: "Frail", 33: "Slippery", 37: "Ringing", 38: "Plow", 43: "Artifact"}


def powers(vec: np.ndarray) -> str:
    return " ".join(f"{POWER_NAMES.get(i, f'p{i}')}={int(round(v * 10))}" for i, v in enumerate(vec) if v != 0)


def describe_state(
    L: Layout, f: np.ndarray, ids: np.ndarray, cards: list[str], monsters: list[str], enchants: list[str], intents: list[str]
) -> str:
    hp, block, energy, turn = f[0] * 100, f[3] * 50, f[4] * 5, f[6] * 10
    hand = []
    for slot, i in enumerate(ids[L.i_hand : L.i_hand + L.max_hand]):
        if not i:
            continue
        name = cards[i]
        if ench := ids[L.i_enchants + slot]:
            feats = f[L.f_hand + slot * L.hand_feats :][: L.hand_feats]
            name += f"+{enchants[ench - 1]}" + ("(spent)" if feats[8] else f"{int(round(feats[7] * 3))}")
        hand.append(name)
    enemies = []
    for s in range(L.max_enemies):
        e = f[L.f_enemies + s * L.enemy_feats :][: L.enemy_feats]
        if not e[0] or not e[1]:
            continue
        kinds = e[L.enemy_base :][: L.n_intents]
        nums = e[L.enemy_base + L.n_intents :][: L.intent_nums]
        intent = " ".join(intents[k] for k, on in enumerate(kinds) if on)
        if nums[0]:
            intent += f" {nums[0] * 20:.0f}x{nums[1] * 3:.0f}"
        enemies.append(f"{monsters[ids[L.i_enemies + s]]} {e[2] * 100:.0f}hp b{e[5] * 30:.0f} [{intent}] {powers(e[L.enemy_feats - L.n_powers:])}")
    out = f"T{turn:.0f} hp{hp:.0f} b{block:.0f} e{energy:.0f} {powers(f[L.f_player_powers : L.f_hand])}\n"
    out += "  hand: " + ", ".join(hand) + "\n"
    return out + "".join(f"  vs {e}\n" for e in enemies)


def describe_action(L: Layout, a: int, ids: np.ndarray, cards: list[str]) -> str:
    if a < L.a_potion:
        slot, t = divmod(a, L.targets)
        target = "" if t == 0 else f" -> enemy {t - 1}"
        return f"play {cards[ids[L.i_hand + slot]]}{target}"
    if a < L.a_end_turn:
        slot, t = divmod(a - L.a_potion, L.targets)
        return f"potion slot {slot}" + ("" if t == 0 else f" -> enemy {t - 1}")
    if a == L.a_end_turn:
        return "END TURN"
    if a < L.a_skip:
        return f"choose {cards[ids[L.i_choices + a - L.a_choose]]}"
    return "skip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--old-vocab", type=Path, default=None, help="vocab.txt the checkpoint was trained with, if it predates the current sim")
    ap.add_argument("--kind", default="Boss")
    ap.add_argument("--encounter", default=None, help="only this one, as the evaluation names it (WaterfallGiantBoss)")
    ap.add_argument("--max", type=int, default=3, help="lost fights to print")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = Layout.load()
    cards, monsters = _sim.card_names(), _sim.monster_names()
    vocab = parse(current_text())
    enchants, intents = vocab["enchant"], vocab["intent"]
    policy = Policy(L).to(device)
    load_policy(args.checkpoint, policy, device, args.old_vocab)
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
                state = describe_state(L, f, idv, cards, monsters, enchants, intents)
                act = describe_action(L, int(actions[i]), idv, cards)
                logs[i].append(f"{state}  => {act}  (p={probs[i, actions[i]]:.2f}, v={value[i]:.2f})\n")
            for e in envs.step(actions):
                if done[e.env]:
                    continue
                done[e.env] = True
                wanted = e.encounter == args.encounter if args.encounter else e.kind == args.kind
                if wanted and not e.won and printed < args.max:
                    printed += 1
                    print(f"===== LOST {e.encounter} floor {e.floor} after {e.steps} actions =====")
                    print("".join(logs[e.env]))


if __name__ == "__main__":
    main()
