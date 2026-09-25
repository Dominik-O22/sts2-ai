"""PPO over run decisions (docs/run-env.md, build order step 3). A frozen
combat checkpoint plays the fights greedily; the run policy
(`sts2ai.runmodel`) makes every run decision: paths, rewards, shops, rest
sites, events, ancients, deck picks.

    uv run python -m sts2ai.runtrain runs/ab-attn/latest.pt --run-dir runs/run-1 --minutes 90

A run pays at its end: +1 for a win, else floors cleared / 49 - 1; a run
the sim cannot go on with (stuck) pays what the value head expected, so it
teaches nothing. With `--potential` each decision also pays Phi(the next
decision's state) - Phi(its own), Phi 0 once the run is over, Phi the
deck-value net's view of the run (`deckvalue.RunPotential`). Each env's
decisions form one trajectory, cut into batches: GAE with gamma 1 and
`lam` over each env's decisions, bootstrapped from the value of the env's
next decision, which waits for the next batch.

With `--start-full` below 1 the other runs start later in a run
(`Curriculum`); the log splits floors and wins by where runs started.

With `--imitate TRAIN_ROWS` the policy first clones winners' decisions
(`sts2ai.imitation`), checked against the `holdout.npz` beside the rows,
and saves `imitated.pt`; `--minutes 0` stops there.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sts2ai import _sim
from sts2ai.deckvalue import RunPotential
from sts2ai.deckvalue import load as load_deckvalue
from sts2ai.env import START_POINTS, End, Envs, RunFight, RunLayout
from sts2ai.imitation import DECISIONS, Rows, batches, pretrain
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


def start_name(run: RunFight) -> str:
    """Where a run started, for the log: "floor 1", "act 3 boss gen"."""
    return "floor 1" if run.start is None else f"{START_POINTS[run.start]} {'own' if run.own else 'gen'}"


class Stats:
    """Rolling run outcomes for the log: runs from floor 1 in full, the
    others by where they started."""

    def __init__(self, window: int = 2000):
        self.full: deque[tuple[RunFight, Counter[str]]] = deque(maxlen=window)
        self.late: dict[str, deque[RunFight]] = defaultdict(lambda: deque(maxlen=window // 4))
        self.fights: deque[End] = deque(maxlen=20000)
        self.picks: Counter[str] = Counter()

    def ended(self, run: RunFight, picks: Counter[str]) -> None:
        """A run that ended, with what it picked."""
        if run.start is None:
            self.full.append((run, picks))
        else:
            self.late[start_name(run)].append(run)

    def summary(self) -> dict[str, float]:
        out = {}
        if self.full:
            runs = [r for r, _ in self.full]
            paths = sum((p for _, p in self.full), Counter())
            out = {
                "run/floor": float(np.mean([r.floor for r in runs])),
                "run/won": float(np.mean([r.end == "won" for r in runs])),
                "run/act2": float(np.mean([r.act >= 1 for r in runs])),
                "run/act3": float(np.mean([r.act >= 2 for r in runs])),
                "run/deck": float(np.mean([r.deck for r in runs])),
                "run/stuck": float(np.mean([r.end.startswith("stuck") for r in runs])),
                # Of the map steps taken, the share into an elite.
                "run/elite_paths": paths["Path Elite"] / max(sum(v for k, v in paths.items() if k.startswith("Path")), 1),
                "run/rest_heal": paths["RestHeal"] / max(paths["RestHeal"] + paths["RestSmith"], 1),
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
        if run.start is not None and run.end in ("won", "died"):
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
    combat = load_policy(combat_path, device).eval()
    if resume:
        policy, ck = load_run_policy(resume, device)
    else:
        policy, ck = RunPolicy(RunLayout.load(), RunArch(cfg.hidden, cfg.depth)).to(device), None
        if cfg.seed_cards:
            policy.seed_cards(combat)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    if ck and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    run_dir.mkdir(parents=True, exist_ok=True)
    anchor = Rows.load(Path(cfg.imitate)) if cfg.imitate and cfg.imitate_coef > 0 else None
    if anchor is not None and cfg.imitate_kinds:
        anchor = anchor.only([DECISIONS.index(k) for k in cfg.imitate_kinds.split(",")])
        print(f"anchoring {cfg.imitate_kinds}: {len(anchor)} winners' decisions")
    # A resumed policy (a clone's imitated.pt among them) keeps what it has;
    # the rows then only anchor PPO.
    if cfg.imitate and resume is None:
        holdout = Path(cfg.imitate).with_name("holdout.npz")
        pretrain(policy, Rows.load(Path(cfg.imitate)), device, cfg.imitate_epochs, cfg.imitate_lr, cfg.imitate_batch, Rows.load(holdout) if holdout.exists() else None)
        save_run_policy(run_dir / "imitated.pt", policy, None, config=asdict(cfg), combat=str(combat_path))
        if cfg.minutes <= 0:
            return
    envs = Envs(cfg.envs, seed=cfg.seed)
    envs.use_runs(cfg.seed * 1_000_000, choices="caller")
    writer = SummaryWriter(str(run_dir))
    loop = RunLoop(combat, device, envs, cfg.drain)
    trajs = [Trajectory() for _ in range(cfg.envs)]
    # Relics each env held at its last decision, for `relic_bonus`.
    held: list[int | None] = [None] * cfg.envs
    potential = RunPotential(load_deckvalue(Path(cfg.potential), device), envs.run_layout, cfg.phi_scale) if cfg.potential else None
    # Phi at each env's last decision.
    phi = np.zeros(cfg.envs, dtype=np.float32)
    # What each env's run has picked so far.
    run_picks = [Counter() for _ in range(cfg.envs)]
    curriculum = Curriculum(cfg, int(ck.get("frontier", 0)) if ck else 0)
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
        now = potential(f, i).cpu().numpy() if potential else np.zeros(len(waiting), dtype=np.float32)
        rows = np.arange(len(waiting))
        kinds = ids[rows, L.i_options + options * L.option_ids]
        rooms = ids[rows, L.i_options + options * L.option_ids + 4 + L.option_cards]
        relics = (floats[:, L.f_relics : L.f_relics + L.max_relics * L.relic_floats : L.relic_floats] != 0).sum(1)
        for k, env in enumerate(waiting):
            t = trajs[env]
            if held[env] is not None and len(t) and not t.done[-1]:
                t.reward[-1] += cfg.relic_bonus * max(int(relics[k]) - held[env], 0) + now[k] - phi[env]
            held[env], phi[env] = int(relics[k]), now[k]
            t.floats.append(floats[k].astype(np.float16))
            t.ids.append(ids[k].astype(np.int16))
            t.option.append(int(options[k]))
            t.logp.append(float(logp[k]))
            t.value.append(float(values[k]))
            t.reward.append(0.0)
            t.done.append(False)
            kind = names["option"][kinds[k]]
            pick = f"{kind} {names['room'][rooms[k]]}" if kind == "Path" else kind
            stats.picks[pick] += 1
            run_picks[env][pick] += 1
        return options

    def run_ended(env: int, run: RunFight) -> None:
        stats.ended(run, run_picks[env])
        curriculum.ended(run)
        run_picks[env] = Counter()
        t = trajs[env]
        if len(t) and not t.done[-1]:
            reward = run_reward(run)
            # Phi is 0 once the run is over.
            t.reward[-1] = t.value[-1] if reward is None else reward - phi[env]
            t.done[-1] = True
        held[env] = None

    it = int(ck.get("iter", 0)) if ck else 0
    start = time.perf_counter()
    last_log = start
    while time.perf_counter() - start < cfg.minutes * 60:
        if cfg.start_full < 1.0:
            curriculum.update(envs)
        # Collect until the batch fills with decisions whose successor is
        # known (or whose run ended).
        while sum(max(len(t) - (0 if t.done and t.done[-1] else 1), 0) for t in trajs) < cfg.batch:
            stats.fights.extend(e for e in loop.step(decide, run_ended) if e.run)
        it += 1
        update(policy, opt, cfg, trajs, device, writer, it, anchor, warm=it <= cfg.value_warmup)
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            secs = last_log - start
            summary = stats.summary()
            for k, v in summary.items():
                writer.add_scalar(k, v, it)
            writer.add_scalar("speed/decisions_per_s", loop.decisions / secs, it)
            writer.add_scalar("speed/combat_steps_per_s", loop.combat_steps * cfg.envs / secs, it)
            writer.add_scalar("start/frontier", curriculum.frontier, it)
            top = ", ".join(f"{k} {v / max(stats.picks.total(), 1):.0%}" for k, v in stats.picks.most_common(8))
            late = "  ".join(
                f"{name} {np.mean([r.end == 'won' for r in runs]):.0%} ({len(runs)})" for name, runs in sorted(stats.late.items())
            )
            print(
                f"it {it} {secs / 60:.1f} min  floor {summary.get('run/floor', 0):.1f}  won {summary.get('run/won', 0):.1%}  "
                f"act2 {summary.get('run/act2', 0):.1%}  act3 {summary.get('run/act3', 0):.1%}  "
                f"elite paths {summary.get('run/elite_paths', 0):.1%}  elites won {summary.get('fight/elite', 0):.1%}  "
                f"heal {summary.get('run/rest_heal', 0):.0%}  "
                f"{loop.decisions / secs:,.0f} dec/s  {loop.combat_steps * cfg.envs / secs:,.0f} steps/s  picks: {top}",
                flush=True,
            )
            if cfg.start_full < 1.0:
                print(f"    starts: frontier {START_POINTS[curriculum.frontier]}, pools {envs.start_pools()}; won {late}", flush=True)
            stats.picks.clear()
            save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path), frontier=curriculum.frontier)
    save_run_policy(run_dir / "latest.pt", policy, opt, iter=it, config=asdict(cfg), combat=str(combat_path), frontier=curriculum.frontier)


def update(
    policy: RunPolicy,
    opt: torch.optim.Optimizer,
    cfg: Config,
    trajs: list[Trajectory],
    device: torch.device,
    writer: SummaryWriter,
    it: int,
    anchor: Rows | None = None,
    warm: bool = False,
) -> None:
    """One PPO update over every env's decisions whose successor is known;
    each env's last decision of a run still going waits for the next batch.
    `warm` trains the value head alone; `anchor` rows add the imitation
    loss at `cfg.imitate_coef`."""
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
            loss = cfg.value_coef * vf if warm else pg + cfg.value_coef * vf - cfg.entropy * ent
            if anchor is not None and not warm and cfg.imitate_coef > 0:
                pick = np.random.randint(len(anchor), size=cfg.minibatch)
                a_floats, a_ids, a_option = next(batches(anchor, device, cfg.minibatch, pick))
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    a_logits, _ = policy(a_floats, a_ids)
                imitation = torch.nn.functional.cross_entropy(a_logits.float(), a_option)
                loss = loss + cfg.imitate_coef * imitation
                writer.add_scalar("loss/imitation", imitation.item(), it)
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
