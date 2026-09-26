"""Exact turn search: every distinct end of the turn, scored by the value head.

    uv run python -m sts2ai.exactsearch runs/<run>/latest.pt --easy 400 --hard 300

`search.py` judges a first action by how the policy would finish the
turn. This one walks every line to the end of the turn over distinct
states (`sim::turnsearch`, `_sim.TurnPlanner`): shaped reward on the way
plus the value head where the next turn starts, the best action at each
state, the expectation over draws and dice. The policy's own action stays
unless another beats it by `PLAN_MARGIN`, as the advisor does.

`main` plays the same fights three ways, greedy, with the current search
(`searcheval`'s advise-style ranking over `--copies`) and with this one:
the weak and normal fights of held-out winners scored by HP lost against
what the winner lost (`evaluate --source easy`), and their elite and boss
fights by wins.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim, search
from sts2ai.advise import PLAN_MARGIN
from sts2ai.env import End, Envs, Layout
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.searcheval import pick
from sts2ai.setups import EASY_HOLDOUT, HOLDOUT

# Leaves per value-head call: a few hundred MB of observations, and small
# enough to share the GPU with a training run.
LEAF_CHUNK = 4096


class Planner:
    """The exact search for the fights of an `Envs`, with its own leaf
    buffers. `choose` overrides the policy's actions where the search
    finds a better one."""

    def __init__(self, envs: Envs, **config: int):
        self.envs = envs
        self.inner = _sim.TurnPlanner(envs.n, **config)
        L = Layout.load()
        pin = torch.cuda.is_available()
        self.floats = torch.zeros((LEAF_CHUNK, L.n_floats), pin_memory=pin).numpy()
        self.ids = torch.zeros((LEAF_CHUNK, L.n_ids), dtype=torch.int64, pin_memory=pin).numpy()
        self.mask = torch.zeros((LEAF_CHUNK, L.n_actions), dtype=torch.bool, pin_memory=pin).numpy()
        self.searched: list[dict] = []
        self.reused = 0
        self.overrides = 0

    @torch.no_grad()
    def score_leaves(self, policy: Policy, device: torch.device) -> None:
        n = self.inner.n_leaves()
        values = np.empty(n, np.float32)
        start = 0
        while start < n:
            k = self.inner.encode_leaves(start, self.floats, self.ids, self.mask)
            _, v = policy(torch.from_numpy(self.floats[:k]).to(device), torch.from_numpy(self.ids[:k]).to(device))
            values[start : start + k] = v.float().cpu().numpy()
            start += k
        self.inner.set_values(values)

    def choose(self, policy: Policy, device: torch.device, roots: list[int], probs: np.ndarray, actions: np.ndarray) -> None:
        """Search each of `roots` (reusing a search that already holds its
        state) and replace `actions[i]` by the search's best action when it
        beats the policy's by `PLAN_MARGIN`."""
        fresh = self.inner.prepare(self.envs.sim, roots, np.ascontiguousarray(probs[roots]))
        self.reused += len(roots) - len(fresh)
        self.score_leaves(policy, device)
        self.searched += [self.inner.stats(i) for i in fresh]
        for i in roots:
            values = dict(self.inner.action_values(self.envs.sim, i) or [])
            if not values:
                continue
            best = max(values, key=values.get)
            own = int(actions[i])
            if own not in values or values[best] - values[own] >= PLAN_MARGIN:
                self.overrides += actions[i] != best
                actions[i] = best


@dataclass
class Run:
    """One way of playing a set of fights: each fight's end, and cost."""

    ends: dict[int, End] = field(default_factory=dict)
    decisions: int = 0
    seconds: float = 0.0
    searched: list[dict] = field(default_factory=list)
    reused: int = 0
    overrides: int = 0


@torch.no_grad()
def play(policy: Policy, device: torch.device, lines: list[str], mode: str, seed: int, copies: int, config: dict[str, int]) -> Run:
    """Play each fight in `lines` (setups as `sts2ai.setups` writes them)
    once: `greedy`, `forks` (the current search) or `exact`."""
    envs = Envs(len(lines), seed=seed)
    envs.sim.use_setups("\n".join(lines), 1, seed)
    envs.sim.observe(envs.floats, envs.ids, envs.mask)
    planner = Planner(envs, seed=seed, **config) if mode == "exact" else None
    active = set(range(envs.n))
    run = Run()
    t0 = time.time()
    step = 0
    while active:
        floats = torch.from_numpy(envs.floats).to(device)
        ids = torch.from_numpy(envs.ids).to(device)
        mask = torch.from_numpy(envs.mask).to(device)
        logits, _ = policy(floats, ids)
        masked = masked_logits(logits.float(), mask)
        actions = masked.argmax(dim=1).cpu().numpy()
        roots = sorted(active)
        run.decisions += len(roots)
        if mode == "forks":
            forks = envs.sim.fork(roots, copies, seed=seed + step)
            first = np.concatenate([search.spread(np.flatnonzero(envs.mask[i]), copies) for i in roots])
            second = (np.full(len(first), -1), np.zeros(len(first), np.int64))
            score = search.rollout(policy, device, forks, first, second=second)
            for r, i in enumerate(roots):
                part = slice(r * copies, (r + 1) * copies)
                own = int(actions[i])
                actions[i] = pick(first[part], score[part], own, True, (second[0][part], second[1][part]))
                run.overrides += actions[i] != own
        elif mode == "exact":
            assert planner is not None
            probs = masked.softmax(dim=1).cpu().numpy()
            planner.choose(policy, device, roots, probs, actions)
        for e in envs.step(actions):
            if e.env in active:
                active.discard(e.env)
                run.ends[e.env] = e
        step += 1
    run.seconds = time.time() - t0
    if planner is not None:
        run.searched, run.reused, run.overrides = planner.searched, planner.reused, planner.overrides
    return run


def hp_lost(line: dict, e: End) -> float:
    """HP the fight cost, net of healing (`evaluate.easy`)."""
    max_end = line["hp"] / (e.hp_frac + e.hp_lost)
    return e.hp_lost * max_end


def sample(path: Path, n: int, seed: int) -> list[str]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return random.Random(seed).sample(lines, min(n, len(lines)))


def cost(run: Run) -> str:
    potions = np.mean([e.potions_used for e in run.ends.values()])
    return (
        f"potions {potions:.2f}  {run.seconds / max(run.decisions, 1) * 1000:6.1f} ms/decision over {run.decisions} decisions,"
        f" {run.overrides / max(run.decisions, 1):.1%} overruled"
    )


def search_report(run: Run) -> str:
    """States per search, leaves, cap hits and the best line's needs."""
    s = run.searched
    if not s:
        return ""
    nodes = np.array([x["nodes"] for x in s])
    leaves = np.array([x["end_leaves"] + x["cut_leaves"] for x in s])
    micros = np.array([x["micros"] for x in s]) / 1000
    capped = np.mean([x["capped"] for x in s])
    pv_cut = np.mean([x.get("pv_cut", False) for x in s])
    ranks = np.array([x.get("pv_rank", 0) for x in s])
    merge = np.sum([x["arrivals"] for x in s]) / max(np.sum(nodes + np.array([x["end_leaves"] for x in s])), 1)
    lines = np.array([x["lines"] for x in s])
    rules = {k: int(np.sum([x[k] for x in s])) for k in ("det", "drawn", "sampled", "end_turn")}
    q = lambda a: f"median {np.median(a):.0f} p90 {np.percentile(a, 90):.0f} max {a.max():.0f}"  # noqa: E731
    return "\n".join(
        [
            f"  searches {len(s)}, reused {run.reused} (decisions a search already held)",
            f"  states   {q(nodes)}",
            f"  leaves   {q(leaves)}",
            f"  lines    {q(lines)} (sequences the states stand for; arrivals per distinct state {merge:.1f})",
            f"  search   {q(micros)} ms (CPU, one thread per fight)",
            f"  capped   {capped:.1%} of searches; best line ran into a cut state in {pv_cut:.1%}",
            f"  best line needed more than 100/250/500 states in {np.mean(ranks >= 100):.1%}/{np.mean(ranks >= 250):.1%}/{np.mean(ranks >= 500):.1%}",
            f"  steps by rule {rules}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--easy", type=int, default=400, help="weak and normal fights of held-out winners")
    ap.add_argument("--hard", type=int, default=300, help="their elite and boss fights")
    ap.add_argument("--modes", default="greedy,forks,exact")
    ap.add_argument("--copies", type=int, default=256, help="copies per decision for the current search")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-states", type=int, default=500)
    ap.add_argument("--quiesce-states", type=int, default=1000)
    ap.add_argument("--end-samples", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--draw-cap", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4096, help="rows per network call in the current search")
    args = ap.parse_args()
    search.CHUNK = args.chunk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    policy.eval()
    config = dict(
        max_states=args.max_states, quiesce_states=args.quiesce_states, end_samples=args.end_samples, samples=args.samples, draw_cap=args.draw_cap
    )
    modes = args.modes.split(",")
    for name, path, n in (("easy", EASY_HOLDOUT, args.easy), ("hard", HOLDOUT, args.hard)):
        if n <= 0:
            continue
        lines = sample(path, n, args.seed)
        parsed = [json.loads(line) for line in lines]
        print(f"== {name}: {len(lines)} fights from {path.name}", flush=True)
        for mode in modes:
            run = play(policy, device, lines, mode, args.seed, args.copies, config)
            ends = [run.ends[i] for i in range(len(lines))]
            won = np.mean([e.won for e in ends])
            if name == "easy":
                ours = np.array([hp_lost(parsed[i], e) for i, e in enumerate(ends)])
                theirs = np.array([float(p["winner_hp_lost"]) for p in parsed])
                print(f"{mode:7s} won {won:6.1%}  HP lost {ours.mean():5.2f}  winner {theirs.mean():5.2f}  gap {np.mean(ours - theirs):+5.2f}  {cost(run)}", flush=True)
            else:
                print(f"{mode:7s} won {won:6.1%}  {cost(run)}", flush=True)
            if report := search_report(run):
                print(report, flush=True)


if __name__ == "__main__":
    main()
