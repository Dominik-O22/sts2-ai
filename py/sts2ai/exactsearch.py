"""Exact turn search: every distinct end of the turn, scored by the value head.

    uv run python -m sts2ai.exactsearch runs/<run>/latest.pt --easy 400 --hard 300

`search.py` judges a first action by how the policy would finish the
turn. This one walks every line to the end of the turn over distinct
states (`sim::turnsearch`, `_sim.TurnPlanner`): shaped reward on the way
plus the value head where the next turn starts, the best action at each
state, the expectation over draws and dice. The policy's own action stays
unless another beats it by `PLAN_MARGIN`, as the advisor does.

The hybrid (`Hybrid`) takes this search's best `--top` lines and plays
each out to the fight's end, greedy, `P` times on dice every line of the
fight shares, and picks by those: the value head's error between lines is
larger than the gaps between them (runs/scratch-keep/value_noise.py).

`main` plays the same fights several ways (`--modes`): greedy, with the
current search (`searcheval`'s advise-style ranking over `--copies`),
with this one (`exact`), and with the hybrid at P playouts (`hybridP`, or
`raceP` stopping lines early): the weak and normal fights of held-out
winners scored by HP lost against what the winner lost (`evaluate
--source easy`), and their elite and boss fights by wins.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim, search
from sts2ai.env import End, Envs, Layout
from sts2ai.mcts import TreeSearch
from sts2ai.model import Net, for_play, load_policy, masked_logits
from sts2ai.search import PLAN_MARGIN
from sts2ai.searcheval import pick
from sts2ai.setups import EASY_HOLDOUT, HOLDOUT

# Leaves per value-head call: a few hundred MB of observations, and small
# enough to share the GPU with a training run.
LEAF_CHUNK = 4096
# Copies one playout batch holds: its observation buffers take some 12 KB
# a row.
PLAYOUT_ROWS = 16384
# Steps a playout gets, the env's own cap per fight, and the turns it
# may take: to the fight's end.
PLAYOUT_STEPS = 500
PLAYOUT_DEPTH = 10_000


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
    def score_leaves(self, policy: Net, device: torch.device) -> None:
        n = self.inner.n_leaves()
        values = np.empty(n, np.float32)
        start = 0
        while start < n:
            k = self.inner.encode_leaves(start, self.floats, self.ids, self.mask)
            _, v = policy(torch.from_numpy(self.floats[:k]).to(device), torch.from_numpy(self.ids[:k]).to(device))
            values[start : start + k] = v.float().cpu().numpy()
            start += k
        self.inner.set_values(values)

    def choose(self, policy: Net, device: torch.device, roots: list[int], probs: np.ndarray, actions: np.ndarray) -> None:
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


class Hybrid:
    """The exact search's best `top` lines from a decision, and the
    policy's best line (the best that opens with its action), each played
    out `playouts` times to the fight's end on dice and shuffles every line
    of the fight shares: the search's best action where a copy is still in
    a state of the searched turn, greedy after. The policy's line stays
    unless the best by playouts beats it by two standard errors of their
    paired difference. With `race`, playouts come `race` at a time, and a
    line that far behind the leader stops. The line picked is followed
    until chance or the turn's end takes the fight off it."""

    def __init__(self, envs: Envs, top: int, playouts: int, race: int, seed: int, **config: int):
        self.envs = envs
        self.planner = Planner(envs, seed=seed, **config)
        self.top, self.playouts, self.race, self.seed = top, playouts, race, seed
        L = envs.layout
        pin = torch.cuda.is_available()
        self.floats = torch.zeros((PLAYOUT_ROWS, L.n_floats), pin_memory=pin)
        self.ids = torch.zeros((PLAYOUT_ROWS, L.n_ids), dtype=torch.int64, pin_memory=pin)
        self.mask = torch.zeros((PLAYOUT_ROWS, L.n_actions), dtype=torch.bool, pin_memory=pin)
        self.copy_steps = 0
        self.followed = 0
        self.overrides = 0
        self.calls = 0
        self.unreplayed = 0

    def choose(self, policy: Net, device: torch.device, roots: list[int], probs: np.ndarray, actions: np.ndarray) -> None:
        """Replace `actions[i]` for each of `roots` by its committed line's
        action, or pick a line afresh."""
        inner, sim = self.planner.inner, self.envs.sim
        need = []
        for i in roots:
            a = inner.planned_action(sim, i)
            if a is None:
                need.append(i)
            else:
                actions[i] = a
                self.followed += 1
        if not need:
            return
        fresh = inner.prepare(sim, need, np.ascontiguousarray(probs[need]))
        self.planner.reused += len(need) - len(fresh)
        self.planner.score_leaves(policy, device)
        self.planner.searched += [inner.stats(i) for i in fresh]
        # Per env: its lines, the ones played out, and the policy's.
        cands: dict[int, tuple[list, list[int], int]] = {}
        for i in need:
            lines = inner.lines(sim, i) or []
            mine = next((k for k, line in enumerate(lines) if line[0] == actions[i]), None)
            if mine is None:
                continue
            pool = list(range(min(self.top, len(lines))))
            cands[i] = (lines, pool if mine in pool else pool + [mine], mine)
        self.calls += 1
        exact, samples = self.play_lines(policy, device, cands)
        for i, (lines, pool, mine) in cands.items():
            if mine not in pool:
                continue
            best = max(pool, key=lambda k: self.mean(exact, samples, (i, k)))
            k = best if best != mine and self.beats(exact, samples, (i, best), (i, mine)) else mine
            inner.commit(i, k)
            self.overrides += lines[k][0] != actions[i]
            actions[i] = lines[k][0]

    @staticmethod
    def mean(exact: dict, samples: dict, x: tuple[int, int]) -> float:
        return exact[x] if x in exact else float(samples[x].mean())

    @staticmethod
    def beats(exact: dict, samples: dict, x: tuple[int, int], y: tuple[int, int]) -> bool:
        """Whether line x leads line y by two standard errors of their
        difference over the playouts both had (on the same dice); a line
        that ends the fight is exact."""
        m = min((len(samples[z]) for z in (x, y) if z not in exact), default=1)
        a, b = (np.full(m, exact[z]) if z in exact else samples[z][:m] for z in (x, y))
        d = a - b
        se = d.std(ddof=1) / np.sqrt(m) if m > 1 else 0.0
        return bool(d.mean() > 2 * se)

    def play_lines(self, policy: Net, device: torch.device, cands: dict) -> tuple[dict, dict]:
        """Each candidate line's value: exact for one that ends the fight,
        else its playouts (reward on the way plus what they collected)."""
        exact: dict[tuple[int, int], float] = {}
        samples: dict[tuple[int, int], np.ndarray] = {}
        alive = []
        for i, (lines, pool, _) in cands.items():
            for k in pool:
                if lines[k][3] == "Over":
                    exact[(i, k)] = lines[k][1]
                else:
                    samples[(i, k)] = np.zeros(0, np.float32)
                    alive.append((i, k))
        per = self.race or self.playouts
        for t in range(max(1, self.playouts // per)):
            batch = max(1, PLAYOUT_ROWS // per)
            for start in range(0, len(alive), batch):
                chunk = alive[start : start + batch]
                # One seed per fight and round: its lines meet the same luck.
                seeds = [hash((self.seed, self.calls, i, t)) & (2**63 - 1) for i, _ in chunk]
                forks, last, ok = self.planner.inner.fork_lines(self.envs.sim, chunk, per, seeds, PLAYOUT_DEPTH)
                self.unreplayed += len(ok) - sum(ok)
                chunk = [ik for ik, fine in zip(chunk, ok) if fine]
                sums = self.play_out(policy, device, forks, np.array(last, np.int64), np.array([i for i, _ in chunk], np.int64), per)
                for p, (i, k) in enumerate(chunk):
                    samples[(i, k)] = np.concatenate([samples[(i, k)], cands[i][0][k][2] + sums[p * per : (p + 1) * per]])
            # A line the fight could not follow is out.
            for ik in [ik for ik in alive if len(samples[ik]) == 0]:
                del samples[ik]
                i, k = ik
                cands[i][1].remove(k)
            alive = [ik for ik in alive if ik in samples]
            if self.race:
                alive = self.still_racing(exact, samples, cands, alive)
            if not alive:
                break
        return exact, samples

    def still_racing(self, exact: dict, samples: dict, cands: dict, alive: list) -> list:
        """The lines worth more playouts: a fight whose leader already beats
        the policy's line is decided, and so is one whose policy line leads
        with every other line beaten; a line the leader beats stops."""
        by_env = defaultdict(list)
        for i, k in alive:
            by_env[i].append(k)
        out = []
        for i, ks in by_env.items():
            _, pool, mine = cands[i]
            if mine not in pool:
                continue
            best = max(pool, key=lambda k: self.mean(exact, samples, (i, k)))
            if best != mine and self.beats(exact, samples, (i, best), (i, mine)):
                continue
            keep = [k for k in ks if k in (best, mine) or not self.beats(exact, samples, (i, best), (i, k))]
            if best == mine and set(keep) <= {mine}:
                continue
            out += [(i, k) for k in keep]
        return out

    @torch.no_grad()
    def play_out(self, policy: Net, device: torch.device, forks, last: np.ndarray, owners: np.ndarray, per: int) -> np.ndarray:
        """Play every copy to its fight's end, `per` copies per line: the
        line's last step first, then the search's best action while the
        copy is in a state of the searched turn, greedy after. Returns each
        copy's summed reward."""
        n = len(forks)
        first, owner = np.repeat(last, per), np.repeat(owners, per)
        sums = np.zeros(n, np.float32)
        rewards = np.zeros(n, np.float32)
        inverse = np.empty(n, np.int64)
        # Copies that may still be in the searched turn: the search is asked
        # about those only.
        in_turn = np.ones(n, bool)
        for step in range(PLAYOUT_STEPS):
            live = np.array(forks.live(), np.int64)
            if len(live) == 0:
                break
            u = forks.observe_unique(live.tolist(), self.floats.numpy(), self.ids.numpy(), self.mask.numpy(), inverse)
            logits, _ = search.forward(policy, device, self.floats[:u], self.ids[:u], chunk=search.CHOOSE_CHUNK)
            greedy = masked_logits(logits.float(), self.mask[:u].to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
            acts = greedy[inverse[: len(live)]]
            ask = live[in_turn[live]]
            if len(ask):
                tree = np.full(n, -1, np.int64)
                tree[ask] = self.planner.inner.tree_actions(forks, ask.tolist(), owner[ask].tolist())
                in_turn[ask[tree[ask] == -2]] = False
                acts = np.where(tree[live] >= 0, tree[live], acts)
            if step == 0:
                acts = np.where(first[live] >= 0, first[live], acts)
            full = np.zeros(n, np.int64)
            full[live] = acts
            forks.step(full, rewards)
            sums += rewards
            self.copy_steps += len(live)
        return sums


@dataclass
class Run:
    """One way of playing a set of fights: each fight's end, and cost."""

    ends: dict[int, End] = field(default_factory=dict)
    decisions: int = 0
    seconds: float = 0.0
    searched: list[dict] = field(default_factory=list)
    reused: int = 0
    overrides: int = 0
    # Copies stepped by playouts or rollouts, and decisions that followed
    # a line the hybrid had picked.
    copy_steps: int = 0
    followed: int = 0
    unreplayed: int = 0


@torch.no_grad()
def play(policy: Net, device: torch.device, lines: list[str], mode: str, seed: int, copies: int, config: dict[str, int], top: int = 10, race: int = 8) -> Run:
    """Play each fight in `lines` (setups as `sts2ai.setups` writes them)
    once: `greedy`, `forks` (the current search), `exact`, `hybridP` (the
    hybrid at P playouts per line), `raceP` (with races of `race`) or
    `mctsN` (the tree search, `sts2ai.mcts`, N simulations a decision) or
    `mctswN` (the same with progressive widening at chance) or `mctscN`
    (clairvoyant: it reads the fight's real dice, to measure which fights
    are winnable at all, never to play), or `pimcK[dD][bB]` (`sim::pimc`, K
    sampled futures solved D turns deep at beam B per turn plan)."""
    envs = Envs(len(lines), seed=seed)
    envs.sim.use_setups("\n".join(lines), 1, seed)
    envs.sim.observe(envs.floats, envs.ids, envs.mask)
    planner = Planner(envs, seed=seed, **config) if mode == "exact" else None
    hybrid = None
    tree = (
        TreeSearch(envs, int(m[2]), widen=1.0 if m[1] == "w" else math.inf, seed=seed, clairvoyant=m[1] == "c")
        if (m := re.fullmatch(r"mcts(w|c|)(\d+)", mode))
        else None
    )
    pimc = None
    if m := re.fullmatch(r"pimc(\d+)(?:d(\d+))?(?:b(\d+))?", mode):
        pimc = _sim.Pimc(envs.n, samples=int(m[1]), depth=int(m[2] or 2), beam=int(m[3] or 30), seed=seed)
    if m := re.fullmatch(r"(hybrid|race)(\d+)", mode):
        hybrid = Hybrid(envs, top, int(m[2]), race if m[1] == "race" else 0, seed, **config)
        planner = hybrid.planner
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
            score = search.rollout(policy, device, forks, first, second=second, on_step=lambda _, live: setattr(run, "copy_steps", run.copy_steps + len(live)))
            for r, i in enumerate(roots):
                part = slice(r * copies, (r + 1) * copies)
                own = int(actions[i])
                actions[i] = pick(first[part], score[part], own, True, (second[0][part], second[1][part]))
                run.overrides += actions[i] != own
        elif hybrid is not None:
            hybrid.choose(policy, device, roots, masked.softmax(dim=1).cpu().numpy(), actions)
        elif tree is not None:
            tree.choose(policy, device, roots, actions)
        elif pimc is not None:
            for i, a in zip(roots, pimc.choose(envs.sim, roots)):
                if a >= 0:
                    actions[i] = a
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
    if hybrid is not None:
        run.overrides, run.copy_steps, run.followed, run.unreplayed = hybrid.overrides, hybrid.copy_steps, hybrid.followed, hybrid.unreplayed
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
    d = max(run.decisions, 1)
    return (
        f"potions {potions:.2f}  {run.seconds / d * 1000:6.1f} ms/decision over {run.decisions} decisions,"
        f" {run.copy_steps / d:6.0f} copy-steps/decision, {run.overrides / d:.1%} overruled"
        + (f", {run.followed / d:.0%} followed a picked line, {run.unreplayed} lines did not replay" if run.followed else "")
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
    q = lambda a: f"median {np.median(a):.0f} p90 {np.percentile(a, 90):.0f} max {a.max():.0f}"
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
    ap.add_argument("--chunk", type=int, default=4096, help="rows per network call in the current search and playouts")
    ap.add_argument("--top", type=int, default=10, help="lines the hybrid plays out")
    ap.add_argument("--race", type=int, default=8, help="playouts per round in raceP")
    args = ap.parse_args()
    search.CHUNK = args.chunk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = for_play(load_policy(args.checkpoint, device).eval())
    config = {
        "max_states": args.max_states,
        "quiesce_states": args.quiesce_states,
        "end_samples": args.end_samples,
        "samples": args.samples,
        "draw_cap": args.draw_cap,
    }
    modes = args.modes.split(",")
    for name, path, n in (("easy", EASY_HOLDOUT, args.easy), ("hard", HOLDOUT, args.hard)):
        if n <= 0:
            continue
        lines = sample(path, n, args.seed)
        parsed = [json.loads(line) for line in lines]
        print(f"== {name}: {len(lines)} fights from {path.name}", flush=True)
        for mode in modes:
            run = play(policy, device, lines, mode, args.seed, args.copies, config, args.top, args.race)
            ends = [run.ends[i] for i in range(len(lines))]
            won = np.mean([e.won for e in ends])
            if name == "easy":
                ours = np.array([hp_lost(parsed[i], e) for i, e in enumerate(ends)])
                theirs = np.array([float(p["winner_hp_lost"]) for p in parsed])
                print(
                    f"{mode:7s} won {won:6.1%}  HP lost {ours.mean():5.2f}  winner {theirs.mean():5.2f}  gap {np.mean(ours - theirs):+5.2f}  {cost(run)}",
                    flush=True,
                )
            else:
                print(f"{mode:7s} won {won:6.1%}  {cost(run)}", flush=True)
            if report := search_report(run):
                print(report, flush=True)


if __name__ == "__main__":
    main()
