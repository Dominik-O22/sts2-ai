"""PPO over run decisions (docs/run-env.md, build order step 3). A frozen
combat checkpoint plays the fights greedily; the run policy
(`sts2ai.runmodel`) makes every run decision: paths, rewards, shops, rest
sites, events, ancients, deck picks.

    uv run python -m sts2ai.runtrain runs/ab-attn/latest.pt --run-dir runs/run-1 --minutes 90

A run pays at its end: +1 for a win, else `floor_weight` * floors cleared
/ 49 - 1; a run
the sim cannot go on with (stuck) pays what the value head expected, so it
teaches nothing. With `--potential` each decision also pays Phi(the next
decision's state) - Phi(its own), Phi 0 once the run is over, Phi the
deck-value net's view of the run (`deckvalue.RunPotential`). Each env's
decisions form one trajectory, cut into batches: GAE with gamma 1 and
`lam` over each env's decisions, bootstrapped from the value of the env's
next decision, which waits for the next batch. Each update runs on a
thread of its own while the next batch collects on the sim's threaded
loop (`sts2ai.runloop`), with the policy as it was before that update.

With `--start-full` below 1 the other runs start later in a run
(`Curriculum`); the log splits floors and wins by where runs started.
With `--win-starts SHARE` that share of the runs starts at the entrance
of act 2 or 3 (evenly) with a winner's player there, from the train
players' history files in `--win-runs`, in a fresh run (docs/run-env.md,
Winners' starts); the rest start as before.

With `--imitate TRAIN_ROWS` the policy first clones winners' decisions
(`sts2ai.imitation`), checked against the `holdout.npz` beside the rows,
and saves `imitated.pt`; `--minutes 0` stops there.

With `--forecast CALIBRATION` map steps carry the forecast
(`sts2ai.forecast`) from the frozen combat checkpoint's value head; the
checkpoints keep the calibration, so a resumed run and runplay fill it the
same way. Rows to imitate should be built with the same combat and
calibration (`imitation build --combat --forecast`).
"""

from __future__ import annotations

import argparse
import copy
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sts2ai import _sim
from sts2ai.deckvalue import RunPotential
from sts2ai.deckvalue import load as load_deckvalue
from sts2ai.env import START_POINTS, End, Envs, RunFight, RunLayout, Taken
from sts2ai.forecast import Calibration, Forecaster
from sts2ai.imitation import DECISIONS, Rows, batches, pretrain
from sts2ai.model import for_play, load_policy
from sts2ai.runloop import Loop
from sts2ai.runmodel import RunArch, RunPolicy, load_run_policy, save_run_policy
from sts2ai.setups import TRACKER, split_runs

FLOORS = 49


def run_reward(run: RunFight, floor_weight: float = 1.0) -> float | None:
    """+1 for a win, else `floor_weight` * floors cleared / 49 - 1; None for
    a stuck run."""
    if run.end == "won":
        return 1.0
    if run.end == "died":
        return floor_weight * max(run.floor - 1, 0) / FLOORS - 1.0
    return None


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
    # Accepted for old command lines; the threaded loop answers each
    # env's decisions as they come, so there is no round to drain.
    drain: bool = True
    # Paid for each relic gained, at the decision it arrived after: about a
    # floor's worth (1 / 49). A relic's worth shows many floors later, mixed
    # with everything else, and without this the policy settled on avoiding
    # elites. 0 turns it off; the potential is meant to replace it.
    relic_bonus: float = 0.0
    # Deck-value checkpoint for the shaping potential; empty turns it off.
    potential: str = ""
    # Phi is this times the mean predicted fight value (about -1 to 1.5):
    # at 0.1 a whole unit of fight value is five floors' reward.
    phi_scale: float = 0.1
    # Chance a run starts at floor 1; 1 turns the curriculum off.
    start_full: float = 1.0
    # Win rate from the frontier start point that moves the frontier
    # earlier, or minutes there that do.
    start_target: float = 0.25
    start_minutes: float = 8.0
    # Runs from the frontier that judge it.
    start_window: int = 200
    # Share of a start point's runs taken from the policy's own states
    # there once its pool holds `own_ramp` of them (less before).
    own_max: float = 0.75
    own_ramp: int = 512
    # Winners' decisions to clone before PPO (`sts2ai.imitation`); empty
    # skips it.
    imitate: str = ""
    imitate_epochs: int = 8
    imitate_lr: float = 1e-3
    imitate_batch: int = 256
    # During PPO, each minibatch also takes cross-entropy on this many
    # winners' decisions at this weight, so the policy can adapt what our
    # combat can afford without drifting from how winners build decks.
    imitate_coef: float = 0.0
    # Decision kinds the anchor holds, comma separated (`sts2ai.imitation`
    # names: Card, Shop, Deck, ...); empty holds all. Winners chose paths and
    # rests with a combat model stronger than ours, so those are left to PPO.
    imitate_kinds: str = ""
    # Updates at the start that train only the value head: a cloned policy
    # comes with no critic, and a random one's advantages undo the clone.
    value_warmup: int = 0
    # Fights of these kinds play the pilot's turn search with this many
    # copies (Loop), so the run policy learns the risks the combat it
    # will have can take: with greedy fights PPO gave up elites and
    # upgrades, and the decks that win act 3 with them. 0: greedy.
    search: int = 0
    search_kinds: str = "Elite,Boss"
    # What a floor cleared is worth in a lost run, as a share of 1 / 49. At
    # 1 dying in act 3 instead of act 1 was worth nearly half a win, and
    # PPO traded winners' deck building (elites, upgrades) for surviving
    # act 1: the clone wins 2.7% with searched fights, run-7 and run-9 0.4
    # to 1.3%. Lower puts the weight on winning.
    floor_weight: float = 1.0
    # Share of runs that start at the entrance of act 2 or 3 with a
    # winner's player there, from the train players' history files in
    # `win_runs`, in a fresh run; the others start as the curriculum says.
    # 0 turns it off.
    win_starts: float = 0.0
    win_runs: str = str(TRACKER)
    # Calibration of the forecast map steps carry (`sts2ai.forecast`);
    # empty takes the resumed checkpoint's, if it has one, else none.
    forecast: str = ""


class Batch(NamedTuple):
    """Decisions ready for an update: rows, the option taken, its log-prob
    and value when taken, and the advantage."""

    floats: np.ndarray
    ids: np.ndarray
    option: np.ndarray
    logp: np.ndarray
    value: np.ndarray
    adv: np.ndarray


class Decisions:
    """Every env's decisions since the last update, flat and in the order
    they were made: the row (float16 and int16), the option taken, its
    log-prob and value when taken, the reward after it, whether its run
    ended there, and its env. An env's decisions form one trajectory;
    `take` hands out those whose successor is known (the run ended, or
    the env decided again) with their advantages, and keeps the rest."""

    def __init__(self, n_envs: int, layout: RunLayout, capacity: int):
        self.L = layout
        self.floats = np.empty((capacity, layout.run_floats), np.float16)
        self.ids = np.empty((capacity, layout.run_ids), np.int16)
        self.option = np.empty(capacity, np.int64)
        self.logp = np.empty(capacity, np.float32)
        self.value = np.empty(capacity, np.float32)
        self.reward = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)
        self.env = np.empty(capacity, np.int64)
        self.n = 0
        # Each env's latest decision of a run still going, -1 for none.
        self.last = np.full(n_envs, -1, np.int64)
        # Relics the env held at it, and Phi there.
        self.held = np.zeros(n_envs, np.int64)
        self.phi = np.zeros(n_envs, np.float32)

    def __len__(self) -> int:
        return self.n

    @property
    def ready(self) -> int:
        """Decisions whose successor is known: all but each live env's last."""
        return self.n - int((self.last >= 0).sum())

    def grow(self, k: int) -> None:
        if self.n + k <= len(self.option):
            return
        cap = max(2 * len(self.option), self.n + k)
        for name in ("floats", "ids", "option", "logp", "value", "reward", "done", "env"):
            old = getattr(self, name)
            new = np.empty((cap, *old.shape[1:]), old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)

    def add(
        self,
        waiting: np.ndarray,
        floats: np.ndarray,
        ids: np.ndarray,
        option: np.ndarray,
        logp: np.ndarray,
        value: np.ndarray,
        relics: np.ndarray,
        phi: np.ndarray,
        relic_bonus: float,
    ) -> None:
        """The envs `waiting` decided. Each env's previous decision of the
        same run is paid the relics gained since and Phi here minus Phi
        there."""
        k = len(waiting)
        self.grow(k)
        prev = self.last[waiting]
        has = prev >= 0
        self.reward[prev[has]] += relic_bonus * np.maximum(relics[has] - self.held[waiting][has], 0) + phi[has] - self.phi[waiting][has]
        rows = np.arange(self.n, self.n + k)
        self.floats[rows] = floats
        self.ids[rows] = ids
        self.option[rows] = option
        self.logp[rows] = logp
        self.value[rows] = value
        self.reward[rows] = 0.0
        self.done[rows] = False
        self.env[rows] = waiting
        self.last[waiting] = rows
        self.held[waiting] = relics
        self.phi[waiting] = phi
        self.n += k

    def ended(self, env: int, run: RunFight, floor_weight: float) -> None:
        """`env`'s run ended: its last decision is paid the run's reward
        (Phi is 0 once the run is over) and closes the trajectory."""
        idx = self.last[env]
        if idx >= 0:
            reward = run_reward(run, floor_weight)
            self.reward[idx] = self.value[idx] if reward is None else reward - self.phi[env]
            self.done[idx] = True
        self.last[env] = -1

    def take(self, lam: float) -> Batch:
        """The ready decisions with their advantages (GAE, gamma 1, `lam`
        over each env's decisions, bootstrapped from the value of the
        env's next decision), which leave the buffer; each live env's
        last decision stays for the next batch."""
        n = self.n
        ready = np.ones(n, bool)
        tail = self.last[self.last >= 0]
        ready[tail] = False
        # Grouped by env, each group in time order.
        s = np.argsort(self.env[:n], kind="stable")
        env, value, reward, done = self.env[s], self.value[s], self.reward[s], self.done[s]
        same_next = np.zeros(n, bool)
        same_next[:-1] = env[:-1] == env[1:]
        next_value = np.zeros(n, np.float32)
        next_value[:-1][same_next[:-1]] = value[1:][same_next[:-1]]
        nonterminal = (~done).astype(np.float32)
        delta = reward + nonterminal * next_value - value
        # A trajectory ends at a run's end or at the env's last ready
        # decision (the one after it, if any, only lends its value);
        # advantages scan each from its end, all trajectories at once by
        # position from the end.
        last_ready = ~same_next
        last_ready[:-1] |= ~ready[s][1:]
        ends = np.flatnonzero(done | last_ready)
        pos = ends[np.searchsorted(ends, np.arange(n))] - np.arange(n)
        adv = delta.copy()
        for k in range(1, int(pos.max()) + 1 if n else 0):
            rows = np.flatnonzero(pos == k)
            adv[rows] += lam * nonterminal[rows] * adv[rows + 1]
        out = s[ready[s]]
        batch = Batch(self.floats[out], self.ids[out], self.option[out], self.logp[out], self.value[out], adv[ready[s]])
        # The unready rows move to the front.
        tail = np.sort(tail)
        for name in ("floats", "ids", "option", "logp", "value", "reward", "done", "env"):
            a = getattr(self, name)
            a[: len(tail)] = a[tail]
        self.n = len(tail)
        self.last[self.env[: self.n]] = np.arange(self.n)
        return batch


def start_name(run: RunFight) -> str:
    """Where a run started, for the log: "floor 1", "act 3 boss gen"."""
    return "floor 1" if run.start is None else f"{START_POINTS[run.start]} {run.source}"


class Stats:
    """Rolling run outcomes for the log: runs from floor 1 in full, the
    others by where they started. `picks` counts what the policy picked
    since the last log line, by kind (a path by its room); each run's map
    steps, the ones into an elite, and its rests healed and smithed are
    counted per env (`picked`) and kept with the run at its end."""

    def __init__(self, n_envs: int, window: int = 2000):
        self.full: deque[tuple[RunFight, np.ndarray]] = deque(maxlen=window)
        self.late: dict[str, deque[RunFight]] = {}
        self.fights: deque[End] = deque(maxlen=20000)
        self.picks: Counter[str] = Counter()
        # Per env: map steps, map steps into an elite, rests healed, smithed.
        self.per_env = np.zeros((n_envs, 4), np.int64)
        names = _sim.run_names()
        self.option_names = names["option"]
        self.room_names = names["room"]
        self.path = self.option_names.index("Path")
        self.elite = self.room_names.index("Elite")
        self.heal = self.option_names.index("RestHeal")
        self.smith = self.option_names.index("RestSmith")

    def picked(self, waiting: np.ndarray, kinds: np.ndarray, rooms: np.ndarray) -> None:
        """The envs `waiting` picked options of these kinds (a path into
        these rooms)."""
        keys = np.where(kinds == self.path, kinds * len(self.room_names) + rooms, kinds * len(self.room_names))
        for key, count in zip(*np.unique(keys, return_counts=True)):
            kind, room = divmod(int(key), len(self.room_names))
            name = self.option_names[kind]
            self.picks[f"{name} {self.room_names[room]}" if kind == self.path else name] += int(count)
        counts = np.stack([kinds == self.path, (kinds == self.path) & (rooms == self.elite), kinds == self.heal, kinds == self.smith], axis=1)
        np.add.at(self.per_env, waiting, counts)

    def ended(self, env: int, run: RunFight) -> None:
        """`env`'s run ended."""
        if run.start is None:
            self.full.append((run, self.per_env[env].copy()))
        else:
            self.late.setdefault(start_name(run), deque(maxlen=self.full.maxlen // 4)).append(run)
        self.per_env[env] = 0

    def summary(self) -> dict[str, float]:
        out = {}
        if self.full:
            runs = [r for r, _ in self.full]
            paths, elite, heal, smith = np.sum([p for _, p in self.full], axis=0)
            out = {
                "run/floor": float(np.mean([r.floor for r in runs])),
                "run/won": float(np.mean([r.end == "won" for r in runs])),
                "run/act2": float(np.mean([r.act >= 1 for r in runs])),
                "run/act3": float(np.mean([r.act >= 2 for r in runs])),
                "run/deck": float(np.mean([r.deck for r in runs])),
                "run/stuck": float(np.mean([r.end.startswith("stuck") for r in runs])),
                # Of the map steps taken, the share into an elite.
                "run/elite_paths": elite / max(paths, 1),
                "run/rest_heal": heal / max(heal + smith, 1),
            }
        for name, runs in self.late.items():
            out[f"start/{name}/won"] = float(np.mean([r.end == "won" for r in runs]))
            out[f"start/{name}/floor"] = float(np.mean([r.floor for r in runs]))
        for kind in ("Elite", "Boss"):
            won = [e.won for e in self.fights if e.kind == kind and e.run.start is None]
            if won:
                out[f"fight/{kind.lower()}"] = float(np.mean(won))
        return out


class Curriculum:
    """Where runs start (docs/training.md, The run policy): floor 1 with
    chance `start_full`, the others at a start point no earlier than the
    frontier. The frontier starts at the last boss door and moves one point
    earlier once `start_window` runs from it win `start_target` of the
    time, or after `start_minutes` there: the combat policy wins few runs
    from the last boss door, and waiting for it would keep the run policy
    out of the acts before. Half the late runs start at the frontier, the
    rest spread over the points after it. A point's runs start from the
    policy's own states there, as its pool fills, instead of generated
    ones."""

    def __init__(self, cfg: Config, frontier: int = 0):
        self.cfg, self.frontier = cfg, frontier
        self.results = [deque(maxlen=cfg.start_window) for _ in START_POINTS]
        self.since = time.perf_counter()

    def ended(self, run: RunFight) -> None:
        if run.start is not None and run.source != "win" and run.end in ("won", "died"):
            self.results[run.start].append(run.end == "won")

    def update(self, envs: Envs) -> None:
        """Moves the frontier if it is time, and tells the envs."""
        cfg, done = self.cfg, self.results[self.frontier]
        won = len(done) == done.maxlen and np.mean(done) >= cfg.start_target
        if self.frontier + 1 < len(START_POINTS) and (won or time.perf_counter() - self.since > cfg.start_minutes * 60):
            self.frontier += 1
            self.since = time.perf_counter()
            why = f"won {np.mean(done):.0%}" if won else f"{cfg.start_minutes:g} minutes"
            print(f"start frontier moves to {START_POINTS[self.frontier]} ({why})", flush=True)
        weights = [0.0] * len(START_POINTS)
        weights[self.frontier] = 0.5 if self.frontier else 1.0
        for j in range(self.frontier):
            weights[j] = 0.5 / self.frontier
        own = [cfg.own_max * min(1.0, n / cfg.own_ramp) for n in envs.start_pools()]
        envs.set_starts(cfg.start_full, weights, own)


def train(combat_path: Path, run_dir: Path, cfg: Config, resume: Path | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    combat = for_play(load_policy(combat_path, device).eval())
    if resume:
        policy, ck = load_run_policy(resume, device)
    else:
        policy, ck = RunPolicy(RunLayout.load(), RunArch(cfg.hidden, cfg.depth)).to(device), None
        if cfg.seed_cards:
            policy.seed_cards(combat)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    if ck and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
        # The checkpoint's Adam steps come back on the GPU. Adam keeps a step
        # as a tensor whenever torch thinks it is compiling, and that flag is
        # process-wide: while the main thread compiles the actor, the learner
        # thread's update would feed GPU scalars to the foreach kernels. A
        # fresh optimizer keeps its steps on the CPU, where they are accepted.
        for state in opt.state.values():
            state["step"] = state["step"].cpu()
    calibration = Calibration.load(Path(cfg.forecast)) if cfg.forecast else Calibration(**ck["forecast"]) if ck and ck.get("forecast") else None
    saved = {"forecast": asdict(calibration) if calibration else None}
    run_dir.mkdir(parents=True, exist_ok=True)
    anchor = Rows.load(Path(cfg.imitate)) if cfg.imitate and cfg.imitate_coef > 0 else None
    if anchor is not None and cfg.imitate_kinds:
        anchor = anchor.only([DECISIONS.index(k) for k in cfg.imitate_kinds.split(",")])
        print(f"anchoring {cfg.imitate_kinds}: {len(anchor)} winners' decisions")
    # A resumed policy (a clone's imitated.pt among them) keeps what it has;
    # the rows then only anchor PPO.
    if cfg.imitate and resume is None:
        holdout = Path(cfg.imitate).with_name("holdout.npz")
        pretrain(
            policy,
            Rows.load(Path(cfg.imitate)),
            device,
            cfg.imitate_epochs,
            cfg.imitate_lr,
            cfg.imitate_batch,
            Rows.load(holdout) if holdout.exists() else None,
        )
        save_run_policy(run_dir / "imitated.pt", policy, None, config=asdict(cfg), combat=str(combat_path), **saved)
        if cfg.minutes <= 0:
            return
    envs = Envs(cfg.envs, seed=cfg.seed)
    if cfg.win_starts > 0:
        winners = envs.use_winner_starts(split_runs(Path(cfg.win_runs)), cfg.win_starts)
        print("winners' starts: " + ", ".join(f"{n} at {p}" for p, n in zip(START_POINTS, winners) if n), flush=True)
    envs.use_runs(cfg.seed * 1_000_000, choices="caller")
    writer = SummaryWriter(str(run_dir))
    forecast = Forecaster(combat, device, calibration) if calibration else None
    L = envs.run_layout
    decisions = Decisions(cfg.envs, L, 2 * cfg.batch + 2 * cfg.envs)
    potential = RunPotential(load_deckvalue(Path(cfg.potential), device), L, cfg.phi_scale) if cfg.potential else None
    curriculum = Curriculum(cfg, int(ck.get("frontier", 0)) if ck else 0)
    stats = Stats(cfg.envs)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    # Updates run on a thread of their own, on a CUDA stream of their own,
    # while this thread collects the next batch with `actor`, a copy of the
    # policy one update behind: PPO's ratio against the log-prob stored at
    # collection corrects for that. Torch lets go of the GIL in its kernels
    # and the sim in its steps, so the two overlap. The actor is compiled,
    # as `for_play` compiles the combat net; only this thread calls it, so
    # only this thread compiles.
    actor = copy.deepcopy(policy)
    act = torch.compile(actor, dynamic=True) if device.type == "cuda" else actor
    learner = ThreadPoolExecutor(1)
    stream = torch.cuda.Stream(device) if device.type == "cuda" else None
    learning: Future[None] | None = None

    def learn(batch: Batch, it: int) -> None:
        with torch.cuda.stream(stream):
            update(policy, opt, cfg, batch, device, writer, it, anchor, warm=it <= cfg.value_warmup)
        if stream is not None:
            stream.synchronize()

    @torch.no_grad()
    def decide(waiting: list[int], floats: np.ndarray, ids: np.ndarray) -> np.ndarray:
        f = torch.from_numpy(floats).to(device, non_blocking=True)
        i = torch.from_numpy(ids).to(device, non_blocking=True)
        with autocast:
            logits, values = act(f, i)
        dist = torch.distributions.Categorical(logits=logits.float(), validate_args=False)
        options = dist.sample()
        phi = potential(f, i) if potential else torch.zeros_like(values)
        # One copy back for the four.
        out = torch.stack([options.float(), dist.log_prob(options), values.float(), phi.float()]).cpu().numpy()
        options = out[0].astype(np.int64)
        envs_ = np.asarray(waiting, dtype=np.int64)
        rows = np.arange(len(waiting))
        relics = (floats[:, L.f_relics : L.f_relics + L.max_relics * L.relic_floats : L.relic_floats] != 0).sum(1)
        decisions.add(envs_, floats, ids, options, out[1], out[2], relics, out[3], cfg.relic_bonus)
        kinds = ids[rows, L.i_options + options * L.option_ids]
        rooms = ids[rows, L.i_options + options * L.option_ids + 4 + L.option_cards]
        stats.picked(envs_, kinds, rooms)
        return options

    def events(taken: Taken) -> None:
        # Before the round's decisions: an ended run's env comes back at
        # the next run's first decision, which must not inherit the end.
        stats.fights.extend(e for e in taken.ends if e.run)
        for env, run in taken.ended:
            stats.ended(env, run)
            curriculum.ended(run)
            decisions.ended(env, run, cfg.floor_weight)

    loop = Loop(combat, device, envs, cfg.search, frozenset(cfg.search_kinds.split(",")), seed=cfg.seed, forecast=forecast)
    it = int(ck.get("iter", 0)) if ck else 0
    start = time.perf_counter()
    last_log = start
    while time.perf_counter() - start < cfg.minutes * 60:
        if cfg.start_full < 1.0:
            curriculum.update(envs)
        # Collect until the batch fills with decisions whose successor is
        # known (or whose run ended).
        while decisions.ready < cfg.batch:
            loop.step(decide, events)
        batch = decisions.take(cfg.lam)
        if learning is not None:
            learning.result()
            actor.load_state_dict(policy.state_dict())
        it += 1
        learning = learner.submit(learn, batch, it)
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            secs = last_log - start
            summary = stats.summary()
            for k, v in summary.items():
                writer.add_scalar(k, v, it)
            writer.add_scalar("speed/decisions_per_s", loop.decisions / secs, it)
            writer.add_scalar("speed/combat_steps_per_s", loop.combat_steps / secs, it)
            writer.add_scalar("start/frontier", curriculum.frontier, it)
            top = ", ".join(f"{k} {v / max(stats.picks.total(), 1):.0%}" for k, v in stats.picks.most_common(8))
            late = "  ".join(f"{name} {np.mean([r.end == 'won' for r in runs]):.0%} ({len(runs)})" for name, runs in sorted(stats.late.items()))
            print(
                f"it {it} {secs / 60:.1f} min  floor {summary.get('run/floor', 0):.1f}  won {summary.get('run/won', 0):.1%}  "
                f"act2 {summary.get('run/act2', 0):.1%}  act3 {summary.get('run/act3', 0):.1%}  "
                f"elite paths {summary.get('run/elite_paths', 0):.1%}  elites won {summary.get('fight/elite', 0):.1%}  "
                f"heal {summary.get('run/rest_heal', 0):.0%}  "
                f"{loop.decisions / secs:,.0f} dec/s  {loop.combat_steps / secs:,.0f} steps/s  picks: {top}",
                flush=True,
            )
            if cfg.start_full < 1.0 or cfg.win_starts > 0:
                print(f"    starts: frontier {START_POINTS[curriculum.frontier]}, pools {envs.start_pools()}; won {late}", flush=True)
            stats.picks.clear()
            # A save waits for the update to end: it reads the weights.
            learning.result()
            save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path), frontier=curriculum.frontier, **saved)
    if learning is not None:
        learning.result()
    loop.stop()
    save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path), frontier=curriculum.frontier, **saved)


def update(
    policy: RunPolicy,
    opt: torch.optim.Optimizer,
    cfg: Config,
    batch: Batch,
    device: torch.device,
    writer: SummaryWriter,
    it: int,
    anchor: Rows | None = None,
    warm: bool = False,
) -> None:
    """One PPO update over `batch`. `warm` trains the value head alone;
    `anchor` rows add the imitation loss at `cfg.imitate_coef`."""
    floats = torch.from_numpy(batch.floats.astype(np.float32)).to(device)
    ids = torch.from_numpy(batch.ids.astype(np.int64)).to(device)
    options = torch.from_numpy(batch.option).to(device)
    old_logp = torch.from_numpy(batch.logp).to(device)
    old_v = torch.from_numpy(batch.value).to(device)
    adv = torch.from_numpy(batch.adv).to(device)
    ret = adv + old_v
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(options)
    imitation = None
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
            loss = cfg.value_coef * vf if warm else pg + cfg.value_coef * vf - cfg.entropy * ent
            if anchor is not None and not warm and cfg.imitate_coef > 0:
                pick = np.random.randint(len(anchor), size=cfg.minibatch)
                a_floats, a_ids, a_option = next(batches(anchor, device, cfg.minibatch, pick))
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    a_logits, _ = policy(a_floats, a_ids)
                imitation = torch.nn.functional.cross_entropy(a_logits.float(), a_option)
                loss = loss + cfg.imitate_coef * imitation
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
    writer.add_scalar("loss/policy", pg.item(), it)
    writer.add_scalar("loss/value", vf.item(), it)
    writer.add_scalar("loss/entropy", ent.item(), it)
    writer.add_scalar("loss/ratio", ratio.mean().item(), it)
    # The last minibatch's, as above: a value read per minibatch waits on the GPU.
    if imitation is not None:
        writer.add_scalar("loss/imitation", imitation.item(), it)
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
