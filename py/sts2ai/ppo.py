"""PPO over the masked action space (DESIGN.md, Decision engine).

Rollouts come from `Envs` on the CPU, the network runs on the GPU. Fights
end with a terminal reward and the env resets on its own, so a `done` at
step t cuts the value bootstrap for step t+1.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sts2ai.env import DEFAULT_RECORDINGS, End, Envs, has_recordings
from sts2ai.evaluate import evaluate
from sts2ai.model import Policy, checkpoint_layout, checkpoint_vocab, load_state, masked_logits
from sts2ai.search import rollout, spread
from sts2ai.vocab import current_text

BOSS_FLOOR = 16


@dataclass
class Config:
    envs: int = 1024
    steps: int = 32
    iters: int = 2000
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    # Learning rate at the last iteration, reached linearly; None keeps
    # `lr` throughout.
    lr_final: float | None = None
    # Undiscounted: a fight always ends, and discounting would pay the policy
    # to spend HP and potions on finishing a turn sooner.
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    # Acts fights come from, each `BOSS_FLOOR` floors long.
    acts: int = 3
    # Curriculum: fights come from floors 1..max_floor, and max_floor grows
    # linearly from `floor_start` to the last boss floor over `floor_ramp` iters.
    floor_start: int = 4
    floor_ramp: int = 500
    # Once the ramp is done, this fraction of fights is forced onto an
    # elite or boss; normal fights are won almost always by then.
    hard_frac: float = 0.4
    # Draw those forced elites and bosses by how often the policy loses
    # them (over the last `Stats` window) instead of evenly, so the fights
    # it already wins stop taking the compute.
    focus: bool = False
    # Search distillation (expert iteration): each iteration, this many
    # envs get a turn search (`search.py`) over the policy's `search_top`
    # favourite first actions, `search_copies` copies shared between them,
    # and the policy is also trained toward the search's choice there. The
    # target is the policy's own distribution tilted by what the search
    # found (Gumbel AlphaZero's completed-Q improvement): each searched
    # action's logit moves by (its mean score - the policy's expected
    # score) / `search_temp`; unsearched actions keep theirs. A search that
    # cannot tell moves apart leaves the policy as it was. 0 turns it off.
    search_states: int = 0
    # Roots are drawn only where the policy's favourite first action leads
    # the runner-up by at most this much probability: where it leads by
    # more, the search's target is the policy's own and trains nothing
    # (docs/training.md). 1 keeps all.
    search_margin: float = 0.8
    # Fight kinds roots are drawn from, comma separated.
    search_kinds: str = "Normal,Elite,Boss"
    search_copies: int = 128
    search_top: int = 8
    search_temp: float = 0.05
    search_coef: float = 0.5
    # Recent search targets kept for the update, oldest dropped first; the
    # loss waits until `search_warmup` are in, so it does not fit a handful.
    search_buffer: int = 16384
    search_warmup: int = 4096
    # Wait for each iteration's search instead of letting a slow one run
    # on through the next iteration (which then starts none): one search
    # per iteration, at the search's speed.
    search_sync: bool = False
    eval_every: int = 50
    # Fights per held-out setup at each eval.
    eval_repeats: int = 2
    recordings: Path = DEFAULT_RECORDINGS
    run_dir: Path = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    # Checkpoint to continue from. Skips the floor ramp: the policy already
    # handles the early floors, so fights come from every act at once.
    resume: Path | None = None
    # vocab.txt the resumed checkpoint was trained with, for checkpoints
    # from before the vocabulary was stored in them.
    old_vocab: Path | None = None
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
            "potions_used": float(np.mean([e.potions_used for e in self.ends])),
            "ep_len": float(np.mean([e.steps for e in self.ends])),
            "reward": float(np.mean([e.reward for e in self.ends])),
        }
        for kind in ("Weak", "Normal", "Elite", "Boss"):
            won = [e.won for e in self.ends if e.kind == kind]
            if won:
                out[f"win_{kind.lower()}"] = float(np.mean(won))
        return out

    def loss_weights(self, floor: float = 0.05) -> dict[str, float]:
        """Per elite and boss seen: its loss rate with a win and a loss of
        prior (so a few fights do not zero it), plus `floor` so a fight it
        always wins still comes up now and then. Encounters not seen yet
        are left out and drawn by nobody until they are."""
        tally: dict[str, list[int]] = {}
        for e in self.ends:
            if e.kind in ("Elite", "Boss"):
                t = tally.setdefault(e.encounter, [0, 0])
                t[0] += int(e.won)
                t[1] += 1
        return {enc: (n - w + 1) / (n + 2) + floor for enc, (w, n) in tally.items()}


class SearchTargets:
    """Ring buffer of (observation, mask, search distribution) on the
    training device."""

    def __init__(self, size: int, layout, device: torch.device):
        self.floats = torch.zeros((size, layout.n_floats), device=device)
        self.ids = torch.zeros((size, layout.n_ids), dtype=torch.long, device=device)
        self.mask = torch.zeros((size, layout.n_actions), dtype=torch.bool, device=device)
        self.target = torch.zeros((size, layout.n_actions), device=device)
        self.size, self.next, self.full = size, 0, False

    def __len__(self) -> int:
        return self.size if self.full else self.next

    def add(self, floats: np.ndarray, ids: np.ndarray, mask: np.ndarray, target: np.ndarray) -> None:
        k = len(floats)
        idx = torch.arange(self.next, self.next + k, device=self.floats.device) % self.size
        dev = self.floats.device
        self.floats[idx] = torch.from_numpy(floats).to(dev)
        self.ids[idx] = torch.from_numpy(ids).to(dev)
        self.mask[idx] = torch.from_numpy(mask).to(dev)
        self.target[idx] = torch.from_numpy(target).to(dev)
        self.full |= self.next + k >= self.size
        self.next = (self.next + k) % self.size

    def sample(self, n: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`n` rows drawn with replacement. From an empty buffer, the
        all-zero first row, whose target scores nothing."""
        idx = torch.randint(max(len(self), 1), (n,), device=self.floats.device)
        return self.floats[idx], self.ids[idx], self.mask[idx], self.target[idx]


@torch.no_grad()
@torch.no_grad()
def start_search(
    policy: Policy,
    net: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    device: torch.device,
    envs: Envs,
    cfg: Config,
    seed: int,
    stream: torch.cuda.Stream | None,
    go: threading.Event,
):
    """Turn search on up to `cfg.search_states` random envs in a fight of
    `cfg.search_kinds` with a choice to make and a policy unsure between
    its top two (`cfg.search_margin`).
    Forks them now, while `envs` holds their states, and returns the rest
    of the work as a function: it plays the copies out and returns the
    roots' observations, masks, and the search's distribution over first
    actions, so it can run in a thread while `envs` moves on. The copies
    play on `net`, `policy` compiled, and their GPU work goes on `stream`,
    so it does not queue behind the update's. Each step of theirs waits
    for `go`."""
    kinds = set(cfg.search_kinds.split(","))
    choice = np.array([i for i in np.flatnonzero(envs.mask.sum(axis=1) > 1) if envs.sim.fight(int(i))[1] in kinds], np.int64)
    logits, _ = policy(torch.from_numpy(envs.floats[choice]).to(device), torch.from_numpy(envs.ids[choice]).to(device))
    logits = masked_logits(logits.float(), torch.from_numpy(envs.mask[choice]).to(device))
    top2 = torch.softmax(logits, dim=1).topk(2, dim=1).values
    unsure = np.flatnonzero((top2[:, 0] - top2[:, 1]).cpu().numpy() <= cfg.search_margin)
    pick = np.random.choice(unsure, min(cfg.search_states, len(unsure)), replace=False)
    roots, logits = choice[pick], logits[pick].cpu().numpy()
    n = cfg.search_copies
    tops = [np.argsort(-logits[r])[: min(cfg.search_top, int(envs.mask[i].sum()))] for r, i in enumerate(roots)]
    forks = envs.sim.fork([int(i) for i in roots], n, seed=seed)
    first = np.concatenate([spread(top, n) for top in tops] or [np.zeros(0, np.int64)])
    floats, ids, mask = envs.floats[roots], envs.ids[roots], envs.mask[roots]
    if stream is not None:
        # The weights were just copied in on this thread's stream.
        stream.wait_stream(torch.cuda.current_stream(device))

    def finish():
        # Autocast is per thread (and keeps state on the object), so the
        # searcher's thread enters its own.
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda")
        with torch.cuda.stream(stream), autocast:
            score = rollout(net, device, forks, first, on_step=lambda *_: go.wait())
        # Per (root, action): copies, and their summed score.
        R, A = logits.shape
        cell = np.repeat(np.arange(R) * A, n) + first
        count = np.bincount(cell, minlength=R * A).reshape(R, A)
        total = np.bincount(cell, weights=score, minlength=R * A).reshape(R, A)
        searched = count > 0
        q = total / np.maximum(count, 1)
        prior = np.where(searched, np.exp(logits - np.where(searched, logits, -np.inf).max(axis=1, keepdims=True)), 0.0)
        v = (prior * q).sum(axis=1, keepdims=True) / prior.sum(axis=1, keepdims=True)
        tilted = logits + np.where(searched, (q - v) / cfg.search_temp, 0.0)
        p = np.exp(tilted - tilted.max(axis=1, keepdims=True))
        return floats, ids, mask, (p / p.sum(axis=1, keepdims=True)).astype(np.float32)

    return finish


def save_checkpoint(path: Path, policy: Policy, opt: torch.optim.Optimizer, it: int, global_step: int) -> None:
    """Weights, optimizer, and what the sim looked like: the vocabulary
    (remappable) and the layout (checked, not remappable)."""
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": opt.state_dict(),
            "iter": it,
            "global_step": global_step,
            "vocab": current_text(),
            "layout": asdict(policy.layout),
        },
        path,
    )


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
    searched = SearchTargets(cfg.search_buffer, envs.layout, device) if cfg.search_states else None
    # The search runs in a thread on its own copy of the policy, synced
    # each iteration, while the update trains the original: both mostly
    # wait on the GPU or on Rust, which release the GIL. Its targets land
    # an iteration late.
    searcher = ThreadPoolExecutor(1) if searched is not None else None
    search_policy = copy.deepcopy(policy).eval() if searched is not None else None
    search_stream = torch.cuda.Stream(device) if searched is not None and device.type == "cuda" else None
    # Its batches shrink as copies end their turn, hence dynamic.
    search_net = torch.compile(search_policy, dynamic=True) if searched is not None and net is not policy else search_policy
    pending: Future | None = None
    searches = 0
    # Cleared while the training rollout runs, which pauses the search: the
    # rollout waits on the GPU and the sim every step and ran three to four
    # times slower beside it, while the update, mostly GPU work, barely
    # notices.
    search_go = threading.Event()
    search_go.set()
    stats = Stats()
    start_iter, global_step = 1, 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device)
        if load_state(policy, ck["policy"], checkpoint_vocab(ck, cfg.old_vocab), checkpoint_layout(ck)):
            print("vocabulary grew since the checkpoint: weights remapped by name, optimizer state reset")
        else:
            opt.load_state_dict(ck["optimizer"])
        start_iter, global_step = ck["iter"] + 1, ck["global_step"]
        print(f"resumed {cfg.resume} at iteration {ck['iter']}")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(cfg.run_dir))
    print(f"training on {device}, {cfg.envs} envs x {cfg.steps} steps, logs in {cfg.run_dir}")

    step0 = global_step
    t0 = time.time()
    for it in range(start_iter, start_iter + cfg.iters):
        last = BOSS_FLOOR * cfg.acts
        max_floor = last if cfg.resume else min(last, cfg.floor_start + (last - cfg.floor_start) * it // max(1, cfg.floor_ramp))
        envs.set_floors(1, max_floor)
        envs.set_hard_frac(cfg.hard_frac if max_floor >= BOSS_FLOOR else 0.0)
        if cfg.focus and it % 10 == 0:
            envs.set_hard_weights(stats.loss_weights())
        if cfg.lr_final is not None:
            frac = (it - start_iter) / max(1, cfg.iters - 1)
            for group in opt.param_groups:
                group["lr"] = cfg.lr + (cfg.lr_final - cfg.lr) * frac

        # Rollout.
        search_go.clear()
        policy.eval()
        with torch.no_grad():
            for t in range(cfg.steps):
                # Non-blocking out of pinned buffers: the sync on the
                # sampled actions below orders them before `envs.step`.
                floats = roll.floats[t].copy_(torch.from_numpy(envs.floats), non_blocking=True)
                ids = roll.ids[t].copy_(torch.from_numpy(envs.ids), non_blocking=True)
                mask = roll.mask[t].copy_(torch.from_numpy(envs.mask), non_blocking=True)
                with autocast:
                    logits, value = net(floats, ids)
                dist = torch.distributions.Categorical(logits=masked_logits(logits.float(), mask), validate_args=False)
                action = dist.sample()
                roll.actions[t], roll.logp[t], roll.values[t] = action, dist.log_prob(action), value.float()
                ends = envs.step(action.cpu().numpy())
                roll.rewards[t].copy_(torch.from_numpy(envs.rewards), non_blocking=True)
                roll.dones[t].copy_(torch.from_numpy(envs.dones), non_blocking=True)
                stats.add(ends)
            floats = torch.from_numpy(envs.floats).to(device)
            ids = torch.from_numpy(envs.ids).to(device)
            with autocast:
                _, last_value = net(floats, ids)
            adv, returns = roll.advantages(last_value.float(), cfg.gamma, cfg.lam)
        search_go.set()
        global_step += cfg.steps * cfg.envs
        # A search still running when the rollout ends keeps running, and
        # this iteration starts none, unless `search_sync`.
        if searched is not None and (pending is None or cfg.search_sync or pending.done()):
            if pending is not None:
                searched.add(*pending.result())
            searches += 1
            search_policy.load_state_dict(policy.state_dict())
            pending = searcher.submit(
                start_search(search_policy, search_net, device, envs, cfg, seed=cfg.seed * 1_000_003 + it, stream=search_stream, go=search_go)
            )
            if it == start_iter:
                # The first search compiles the searcher's graphs, and
                # nothing else may compile meanwhile (see below).
                wait([pending])

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
        # Summed on the GPU and read once after the update: a read per
        # minibatch (or `Categorical`'s argument check) waits for the GPU,
        # and the update stalls whenever the search holds the GPU or CPU.
        losses = {k: torch.zeros((), device=device) for k in ("policy", "value", "entropy", "clipfrac", "approx_kl", "search")}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                with autocast:
                    logits, value = net(flat["floats"][idx], flat["ids"][idx])
                logits, value = logits.float(), value.float()
                dist = torch.distributions.Categorical(logits=masked_logits(logits, flat["mask"][idx]), validate_args=False)
                logp = dist.log_prob(flat["actions"][idx])
                ratio = torch.exp(logp - flat["logp"][idx])
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * (value - flat["returns"][idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy * ent
                if searched is not None:
                    # Runs before the warmup too, weighing nothing, so its
                    # forward and backward compile in the first iteration
                    # with the rest.
                    s_floats, s_ids, s_mask, s_target = searched.sample(mb // 8)
                    with autocast:
                        s_logits, _ = net(s_floats, s_ids)
                    logp_all = torch.log_softmax(masked_logits(s_logits.float(), s_mask), dim=1)
                    ce = -(s_target * logp_all).sum(dim=1).mean()
                    warm = len(searched) >= cfg.search_warmup
                    loss = loss + (cfg.search_coef if warm else 0.0) * ce
                    if warm:
                        losses["search"] += ce.detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    losses["policy"] += pg
                    losses["value"] += vl
                    losses["entropy"] += ent
                    losses["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean()
                    losses["approx_kl"] += (flat["logp"][idx] - logp).mean()
                n_updates += 1

        if it == start_iter and net is not policy:
            # Every graph has compiled: rollout, update, search. Compiling
            # while the searcher runs is not safe (torch's "is compiling"
            # is process-wide, and eager code in the other thread takes the
            # compiled path), so from here a shape nothing compiled for runs
            # eager instead.
            torch.compiler.set_stance("eager_on_recompile")

        losses = {k: v.item() for k, v in losses.items()}

        # Logging.
        summary = stats.summary()
        sps = (global_step - step0) / (time.time() - t0)
        for k, v in summary.items():
            writer.add_scalar(f"episode/{k}", v, global_step)
        for k, v in losses.items():
            writer.add_scalar(f"loss/{k}", v / n_updates, global_step)
        writer.add_scalar("curriculum/max_floor", max_floor, global_step)
        writer.add_scalar("perf/sps", sps, global_step)
        writer.add_scalar("perf/searches_per_iter", searches / (it - start_iter + 1), global_step)
        if it % 10 == 0 or it == start_iter:
            win = summary.get("win_rate", float("nan"))
            print(
                f"it {it:5d} step {global_step:>10d} floor<={max_floor:2d} win {win:6.1%} "
                f"reward {summary.get('reward', float('nan')):6.3f} ent {losses['entropy'] / n_updates:5.3f} "
                f"kl {losses['approx_kl'] / n_updates:6.4f} {sps:8.0f} sps"
            )
        if it % cfg.eval_every == 0 or it == start_iter + cfg.iters - 1:
            save_checkpoint(cfg.run_dir / "latest.pt", policy, opt, it, global_step)
            policy.eval()
            win, _, by_kind = evaluate(policy, device, cfg.eval_repeats, acts=cfg.acts)
            writer.add_scalar("eval/holdout_win_rate", win, global_step)
            for k, v in by_kind.items():
                writer.add_scalar(f"eval/holdout_win_{k}", v, global_step)
            print(f"eval on holdout: {win:.1%}  " + "  ".join(f"{k} {v:.1%}" for k, v in by_kind.items()))
            if has_recordings(cfg.recordings):
                win, _, _ = evaluate(policy, device, 8, "recordings", cfg.recordings)
                writer.add_scalar("eval/recorded_win_rate", win, global_step)
    writer.close()
    return policy
