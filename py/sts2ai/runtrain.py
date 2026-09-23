"""PPO over run decisions (docs/run-env.md, build order step 3). A frozen
combat checkpoint plays the fights greedily; the run policy
(`sts2ai.runmodel`) makes every run decision: paths, rewards, shops, rest
sites, events, ancients, deck picks.

    uv run python -m sts2ai.runtrain runs/ab-attn/latest.pt --run-dir runs/run-1 --minutes 90

A run pays at its end: +1 for a win, else floors cleared / 49 - 1; a run
the sim cannot go on with (stuck) pays what the value head expected, so it
teaches nothing. Each env's decisions form one trajectory, cut into
batches: GAE with gamma 1 and `lam` over each env's decisions, bootstrapped
from the value of the env's next decision, which waits for the next batch.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sts2ai import _sim
from sts2ai.env import End, Envs, RunFight
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.runmodel import RunArch, RunPolicy, load_run_policy, save_run_policy

FLOORS = 49

# Picks options for the envs waiting at run decisions, given their rows.
Decide = Callable[[list[int], np.ndarray, np.ndarray], np.ndarray]
# Hears of a run that ended, by env, before that env's next decision.
OnRunEnd = Callable[[int, RunFight], None]


def run_reward(run: RunFight) -> float | None:
    """+1 for a win, else floors cleared / 49 - 1; None for a stuck run."""
    if run.end == "won":
        return 1.0
    if run.end == "died":
        return max(run.floor - 1, 0) / FLOORS - 1.0
    return None


class RunLoop:
    """Plays runs in `envs` (`use_runs(choices="caller")`): each `step`
    answers run decisions through `decide`, then takes one combat step for
    the whole batch with the combat policy, greedy. With `drain` it answers
    until no env waits (docs/run-env.md), so every combat row is live;
    without, it answers one round, and the envs still waiting sit the
    combat step out."""

    def __init__(self, combat: Policy, device: torch.device, envs: Envs, drain: bool = True):
        self.combat, self.device, self.envs, self.drain = combat, device, envs, drain
        self.autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        self.decisions = 0
        self.combat_steps = 0

    @torch.no_grad()
    def step(self, decide: Decide, on_run_end: OnRunEnd) -> list[End]:
        """Returns the fights that ended."""
        envs = self.envs
        while waiting := envs.run_waiting():
            floats, ids = envs.observe_run(waiting)
            for env, run in envs.step_run(waiting, decide(waiting, floats, ids)):
                on_run_end(env, run)
            self.decisions += len(waiting)
            if not self.drain:
                break
        floats = torch.from_numpy(envs.floats).to(self.device, non_blocking=True)
        ids = torch.from_numpy(envs.ids).to(self.device, non_blocking=True)
        mask = torch.from_numpy(envs.mask).to(self.device, non_blocking=True)
        with self.autocast:
            logits, _ = self.combat(floats, ids)
        actions = masked_logits(logits.float(), mask).argmax(dim=1).cpu().numpy()
        fights = envs.step(actions)
        self.combat_steps += 1
        for e in fights:
            if e.run and e.run.end:
                on_run_end(e.env, e.run)
        return fights


def policy_decide(policy: RunPolicy, device: torch.device, greedy: bool) -> Decide:
    """The run policy's picks: sampled, or its favourite."""

    @torch.no_grad()
    def decide(_: list[int], floats: np.ndarray, ids: np.ndarray) -> np.ndarray:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = policy(torch.from_numpy(floats).to(device), torch.from_numpy(ids).to(device))
        if greedy:
            return logits.argmax(1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits, validate_args=False).sample().cpu().numpy()

    return decide


@dataclass
class Config:
    envs: int = 512
    # Decisions per update.
    batch: int = 4096
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    depth: int = 2
    seed: int = 0
    minutes: float = 60.0
    seed_cards: bool = True
    # Answer run decisions until none wait before each combat step
    # (`RunLoop`); without, one round per step.
    drain: bool = True


@dataclass
class Trajectory:
    """One env's decisions since the last update: rows, the option taken,
    its log-prob and value when taken, and the reward and end after each."""

    floats: list[np.ndarray] = field(default_factory=list)
    ids: list[np.ndarray] = field(default_factory=list)
    option: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.option)

    def take(self, k: int) -> Trajectory:
        """The first `k` decisions, which leave this one."""
        head = Trajectory(*(getattr(self, f)[:k] for f in self.__dataclass_fields__))
        for f in self.__dataclass_fields__:
            del getattr(self, f)[:k]
        return head


def gae(traj: Trajectory, next_value: float, lam: float) -> np.ndarray:
    """Advantages over one env's decisions, gamma 1; `next_value` is the
    value of the decision after the last, if its run goes on."""
    adv = np.zeros(len(traj), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(traj))):
        nv = next_value if t + 1 == len(traj) else traj.value[t + 1]
        nonterminal = 0.0 if traj.done[t] else 1.0
        delta = traj.reward[t] + nonterminal * nv - traj.value[t]
        last = delta + lam * nonterminal * last
        adv[t] = last
    return adv


class Stats:
    """Rolling run outcomes for the log."""

    def __init__(self, window: int = 2000):
        self.runs: deque[RunFight] = deque(maxlen=window)
        self.fights: deque[End] = deque(maxlen=20000)
        self.picks: Counter[str] = Counter()

    def summary(self) -> dict[str, float]:
        if not self.runs:
            return {}
        floors = np.array([r.floor for r in self.runs])
        out = {
            "run/floor": float(floors.mean()),
            "run/won": float(np.mean([r.end == "won" for r in self.runs])),
            "run/act2": float(np.mean([r.act >= 1 for r in self.runs])),
            "run/act3": float(np.mean([r.act >= 2 for r in self.runs])),
            "run/deck": float(np.mean([r.deck for r in self.runs])),
            "run/stuck": float(np.mean([r.end.startswith("stuck") for r in self.runs])),
        }
        for kind in ("Elite", "Boss"):
            won = [e.won for e in self.fights if e.kind == kind]
            if won:
                out[f"fight/{kind.lower()}"] = float(np.mean(won))
        return out


def train(combat_path: Path, run_dir: Path, cfg: Config, resume: Path | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    combat = load_policy(combat_path, device).eval()
    envs = Envs(cfg.envs, seed=cfg.seed)
    envs.use_runs(cfg.seed * 1_000_000, choices="caller")
    if resume:
        policy, ck = load_run_policy(resume, device)
    else:
        policy, ck = RunPolicy(envs.run_layout, RunArch(cfg.hidden, cfg.depth)).to(device), None
        if cfg.seed_cards:
            policy.seed_cards(combat)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    if ck and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    loop = RunLoop(combat, device, envs, cfg.drain)
    trajs = [Trajectory() for _ in range(cfg.envs)]
    stats = Stats()
    names = _sim.run_names()
    L = envs.run_layout

    @torch.no_grad()
    def decide(waiting: list[int], floats: np.ndarray, ids: np.ndarray) -> np.ndarray:
        f, i = torch.from_numpy(floats).to(device), torch.from_numpy(ids).to(device)
        with loop.autocast:
            logits, values = policy(f, i)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        options = dist.sample()
        logp = dist.log_prob(options).cpu().numpy()
        options, values = options.cpu().numpy(), values.cpu().numpy()
        rows = np.arange(len(waiting))
        kinds = ids[rows, L.i_options + options * L.option_ids]
        rooms = ids[rows, L.i_options + options * L.option_ids + 4 + L.option_cards]
        for k, env in enumerate(waiting):
            t = trajs[env]
            t.floats.append(floats[k].astype(np.float16))
            t.ids.append(ids[k].astype(np.int16))
            t.option.append(int(options[k]))
            t.logp.append(float(logp[k]))
            t.value.append(float(values[k]))
            t.reward.append(0.0)
            t.done.append(False)
            kind = names["option"][kinds[k]]
            stats.picks[f"{kind} {names['room'][rooms[k]]}" if kind == "Path" else kind] += 1
        return options

    def run_ended(env: int, run: RunFight) -> None:
        stats.runs.append(run)
        t = trajs[env]
        if len(t) and not t.done[-1]:
            reward = run_reward(run)
            t.reward[-1] = t.value[-1] if reward is None else reward
            t.done[-1] = True

    it = int(ck.get("iter", 0)) if ck else 0
    start = time.perf_counter()
    last_log = start
    while time.perf_counter() - start < cfg.minutes * 60:
        # Collect until the batch fills with decisions whose successor is
        # known (or whose run ended).
        while sum(max(len(t) - (0 if t.done and t.done[-1] else 1), 0) for t in trajs) < cfg.batch:
            stats.fights.extend(e for e in loop.step(decide, run_ended) if e.run)
        it += 1
        update(policy, opt, cfg, trajs, device, writer, it)
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            secs = last_log - start
            summary = stats.summary()
            for k, v in summary.items():
                writer.add_scalar(k, v, it)
            writer.add_scalar("speed/decisions_per_s", loop.decisions / secs, it)
            writer.add_scalar("speed/combat_steps_per_s", loop.combat_steps * cfg.envs / secs, it)
            top = ", ".join(f"{k} {v / max(stats.picks.total(), 1):.0%}" for k, v in stats.picks.most_common(8))
            print(
                f"it {it} {secs / 60:.1f} min  floor {summary.get('run/floor', 0):.1f}  won {summary.get('run/won', 0):.1%}  "
                f"act2 {summary.get('run/act2', 0):.1%}  elite {summary.get('fight/elite', 0):.1%}  "
                f"{loop.decisions / secs:,.0f} dec/s  {loop.combat_steps * cfg.envs / secs:,.0f} steps/s  picks: {top}",
                flush=True,
            )
            stats.picks.clear()
            save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path))
    save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path))


def update(policy: RunPolicy, opt: torch.optim.Optimizer, cfg: Config, trajs: list[Trajectory], device: torch.device, writer: SummaryWriter, it: int) -> None:
    """One PPO update over every env's decisions whose successor is known;
    each env's last decision of a run still going waits for the next batch."""
    parts, advs = [], []
    for t in trajs:
        ready = len(t) if t.done and t.done[-1] else len(t) - 1
        if ready <= 0:
            continue
        next_value = t.value[ready] if ready < len(t) else 0.0
        head = t.take(ready)
        parts.append(head)
        advs.append(gae(head, next_value, cfg.lam))
    floats = torch.from_numpy(np.stack([f for p in parts for f in p.floats]).astype(np.float32)).to(device)
    ids = torch.from_numpy(np.stack([i for p in parts for i in p.ids]).astype(np.int64)).to(device)
    options = torch.tensor([o for p in parts for o in p.option], device=device)
    old_logp = torch.tensor([x for p in parts for x in p.logp], device=device)
    old_v = torch.tensor([x for p in parts for x in p.value], device=device)
    adv = torch.from_numpy(np.concatenate(advs)).to(device)
    ret = adv + old_v
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(options)
    for _ in range(cfg.epochs):
        for idx in torch.randperm(n, device=device).split(cfg.minibatch):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, v = policy(floats[idx], ids[idx])
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            logp = dist.log_prob(options[idx])
            a = adv[idx]
            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vf = 0.5 * (v - ret[idx]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg + cfg.value_coef * vf - cfg.entropy * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
    writer.add_scalar("loss/policy", pg.item(), it)
    writer.add_scalar("loss/value", vf.item(), it)
    writer.add_scalar("loss/entropy", ent.item(), it)
    writer.add_scalar("loss/ratio", ratio.mean().item(), it)
    writer.add_scalar("run/decisions", n, it)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("combat", type=Path, help="combat checkpoint that plays the fights, frozen")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    for f, default in asdict(Config()).items():
        kind = type(default)
        if kind is bool:
            ap.add_argument(f"--{f.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=kind, default=default)
    args = ap.parse_args()
    cfg = Config(**{f: getattr(args, f) for f in asdict(Config())})
    train(args.combat, args.run_dir, cfg, args.resume)


if __name__ == "__main__":
    main()
