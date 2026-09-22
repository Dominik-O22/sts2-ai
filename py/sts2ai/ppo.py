"""PPO over the masked action space (DESIGN.md, Decision engine).

Rollouts come from `Envs` on the CPU, the network runs on the GPU. Fights
end with a terminal reward and the env resets on its own, so a `done` at
step t cuts the value bootstrap for step t+1.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sts2ai.env import DEFAULT_RECORDINGS, End, Envs
from sts2ai.evaluate import evaluate
from sts2ai.model import Policy, masked_logits

BOSS_FLOOR = 16


@dataclass
class Config:
    envs: int = 1024
    steps: int = 32
    iters: int = 2000
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    # Curriculum: fights come from floors 1..max_floor, and max_floor grows
    # linearly from `floor_start` to the boss floor over `floor_ramp` iters.
    floor_start: int = 4
    floor_ramp: int = 500
    # Once the ramp is done, this fraction of fights is forced onto an
    # elite or boss; normal fights are won almost always by then.
    hard_frac: float = 0.4
    eval_every: int = 50
    eval_episodes: int = 200
    recordings: Path = DEFAULT_RECORDINGS
    run_dir: Path = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    # Checkpoint to continue from. Skips the floor ramp: the policy already
    # handles the early floors, so fights come from all of act 1 at once.
    resume: Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Speed: fuse the network with torch.compile, and run its matmuls in
    # bfloat16. Advantages, returns, and the loss stay in fp32.
    compile: bool = True
    bf16: bool = True


class Rollout:
    """Fixed-size rollout storage on the training device."""

    def __init__(self, cfg: Config, envs: Envs, device: torch.device):
        T, N, L = cfg.steps, cfg.envs, envs.layout
        self.floats = torch.zeros((T, N, L.n_floats), device=device)
        self.ids = torch.zeros((T, N, L.n_ids), dtype=torch.long, device=device)
        self.mask = torch.zeros((T, N, L.n_actions), dtype=torch.bool, device=device)
        self.actions = torch.zeros((T, N), dtype=torch.long, device=device)
        self.logp = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.dones = torch.zeros((T, N), device=device)

    def advantages(self, last_value: Tensor, gamma: float, lam: float) -> tuple[Tensor, Tensor]:
        """GAE. `dones[t]` means the transition at t was terminal, so nothing
        after it belongs to the same fight."""
        T = self.rewards.shape[0]
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(T)):
            live = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * live - self.values[t]
            gae = delta + gamma * lam * live * gae
            adv[t] = gae
            next_value = self.values[t]
        return adv, adv + self.values


class Stats:
    """Rolling episode statistics over the last `window` finished fights."""

    def __init__(self, window: int = 2000):
        self.ends: deque[End] = deque(maxlen=window)

    def add(self, ends: list[End]) -> None:
        self.ends.extend(ends)

    def summary(self) -> dict[str, float]:
        if not self.ends:
            return {}
        out = {
            "win_rate": float(np.mean([e.won for e in self.ends])),
            "hp_kept": float(np.mean([e.hp_frac for e in self.ends if e.won] or [0.0])),
            "hp_lost": float(np.mean([e.hp_lost for e in self.ends if e.won] or [0.0])),
            "ep_len": float(np.mean([e.steps for e in self.ends])),
            "reward": float(np.mean([e.reward for e in self.ends])),
        }
        for kind in ("Weak", "Normal", "Elite", "Boss"):
            won = [e.won for e in self.ends if e.kind == kind]
            if won:
                out[f"win_{kind.lower()}"] = float(np.mean(won))
        return out


def save_checkpoint(path: Path, policy: Policy, opt: torch.optim.Optimizer, it: int, global_step: int) -> None:
    torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(), "iter": it, "global_step": global_step}, path)


def train(cfg: Config) -> Policy:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    envs = Envs(cfg.envs, seed=cfg.seed, max_floor=cfg.floor_start)
    policy = Policy(envs.layout).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    # `net` is what runs; `policy` keeps the plain module for checkpoints.
    net = torch.compile(policy) if cfg.compile and device.type == "cuda" else policy
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda")
    roll = Rollout(cfg, envs, device)
    stats = Stats()
    start_iter, global_step = 1, 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        start_iter, global_step = ck["iter"] + 1, ck["global_step"]
        print(f"resumed {cfg.resume} at iteration {ck['iter']}")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(cfg.run_dir))
    print(f"training on {device}, {cfg.envs} envs x {cfg.steps} steps, logs in {cfg.run_dir}")

    step0 = global_step
    t0 = time.time()
    for it in range(start_iter, start_iter + cfg.iters):
        max_floor = BOSS_FLOOR if cfg.resume else min(BOSS_FLOOR, cfg.floor_start + (BOSS_FLOOR - cfg.floor_start) * it // max(1, cfg.floor_ramp))
        envs.set_floors(1, max_floor)
        envs.set_hard_frac(cfg.hard_frac if max_floor >= BOSS_FLOOR else 0.0)

        # Rollout.
        policy.eval()
        with torch.no_grad():
            for t in range(cfg.steps):
                floats = torch.from_numpy(envs.floats).to(device)
                ids = torch.from_numpy(envs.ids).to(device)
                mask = torch.from_numpy(envs.mask).to(device)
                with autocast:
                    logits, value = net(floats, ids)
                dist = torch.distributions.Categorical(logits=masked_logits(logits.float(), mask))
                action = dist.sample()
                roll.floats[t], roll.ids[t], roll.mask[t] = floats, ids, mask
                roll.actions[t], roll.logp[t], roll.values[t] = action, dist.log_prob(action), value.float()
                ends = envs.step(action.cpu().numpy())
                roll.rewards[t] = torch.from_numpy(envs.rewards).to(device)
                roll.dones[t] = torch.from_numpy(envs.dones).to(device)
                stats.add(ends)
            floats = torch.from_numpy(envs.floats).to(device)
            ids = torch.from_numpy(envs.ids).to(device)
            with autocast:
                _, last_value = net(floats, ids)
            adv, returns = roll.advantages(last_value.float(), cfg.gamma, cfg.lam)
        global_step += cfg.steps * cfg.envs

        # Update.
        policy.train()
        B = cfg.steps * cfg.envs
        flat = {
            "floats": roll.floats.reshape(B, -1),
            "ids": roll.ids.reshape(B, -1),
            "mask": roll.mask.reshape(B, -1),
            "actions": roll.actions.reshape(B),
            "logp": roll.logp.reshape(B),
            "values": roll.values.reshape(B),
            "adv": adv.reshape(B),
            "returns": returns.reshape(B),
        }
        mb = B // cfg.minibatches
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clipfrac": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                with autocast:
                    logits, value = net(flat["floats"][idx], flat["ids"][idx])
                logits, value = logits.float(), value.float()
                dist = torch.distributions.Categorical(logits=masked_logits(logits, flat["mask"][idx]))
                logp = dist.log_prob(flat["actions"][idx])
                ratio = torch.exp(logp - flat["logp"][idx])
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * (value - flat["returns"][idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    losses["policy"] += pg.item()
                    losses["value"] += vl.item()
                    losses["entropy"] += ent.item()
                    losses["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    losses["approx_kl"] += (flat["logp"][idx] - logp).mean().item()
                n_updates += 1

        # Logging.
        summary = stats.summary()
        sps = (global_step - step0) / (time.time() - t0)
        for k, v in summary.items():
            writer.add_scalar(f"episode/{k}", v, global_step)
        for k, v in losses.items():
            writer.add_scalar(f"loss/{k}", v / n_updates, global_step)
        writer.add_scalar("curriculum/max_floor", max_floor, global_step)
        writer.add_scalar("perf/sps", sps, global_step)
        if it % 10 == 0 or it == start_iter:
            win = summary.get("win_rate", float("nan"))
            print(
                f"it {it:5d} step {global_step:>10d} floor<={max_floor:2d} win {win:6.1%} "
                f"reward {summary.get('reward', float('nan')):6.3f} ent {losses['entropy'] / n_updates:5.3f} "
                f"kl {losses['approx_kl'] / n_updates:6.4f} {sps:8.0f} sps"
            )
        if it % cfg.eval_every == 0 or it == start_iter + cfg.iters - 1:
            save_checkpoint(cfg.run_dir / "latest.pt", policy, opt, it, global_step)
            if cfg.recordings.is_dir():
                policy.eval()
                win, by_enc = evaluate(policy, device, cfg.eval_episodes, cfg.recordings)
                writer.add_scalar("eval/recorded_win_rate", win, global_step)
                print(f"eval on recordings: {win:.1%} over {cfg.eval_episodes} fights")
    writer.close()
    return policy
