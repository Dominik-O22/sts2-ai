"""Deck value: a network that predicts what the deck advisor's fights find.

    uv run python -m sts2ai.deckvalue label runs/<run>/latest.pt --n 200 --out labels.jsonl [--starts FIGHTS.jsonl ...] [--next-act]
    uv run python -m sts2ai.deckvalue train labels.jsonl --out deckvalue.pt
    uv run python -m sts2ai.deckvalue check runs/<run>/latest.pt deckvalue.pt [RECORDING ...]

`sts2ai.cards` prices a deck choice by playing hundreds of fights per option.
This learns that price: a run state (deck, relics, potions, HP, act) in, the
mean fight reward against each elite and boss of its act out, so a choice
costs one forward pass instead of seconds of fights. The fights stay the
teacher: `label` plays them on run states (generated, or drawn from run
evals' fight logs with `--starts`), each with a variant one change away
(a card added, upgraded or removed) so the labels carry the differences a
choice makes, and `train` fits the network to them. `check` sets the
network's picks beside the fights' on decks from played runs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sts2ai import _sim
from sts2ai.cards import BOSS_FLOOR, NEXT_ACT, FightJob, fights_many, fights_stream, horizon
from sts2ai.env import ASCENSION, DEFAULT_RECORDINGS, End, Envs, RunLayout
from sts2ai.model import Policy, load_policy

ACTS = ["Overgrowth", "Underdocks", "Hive", "Glory"]
MAX_DECK, MAX_RELICS, MAX_POTIONS = 64, 32, 5
IDS = _sim.game_ids()
INDEX = {kind: {name: i + 1 for i, name in enumerate(names)} for kind, names in IDS.items()}
# The model's outputs: every elite and boss encounter, by game name.
ENCOUNTERS = [name for name, _, kind in _sim.encounters() if kind in ("Elite", "Boss")]
ACT_OF = {name: act for name, act, _ in _sim.encounters()}
KIND_OF = {name: kind for name, _, kind in _sim.encounters()}
# Cards a reward can offer: the Ironclad's Common, Uncommon and Rare (what
# Bash, a Basic, can transform into), less the ones the sim cannot play.
REWARD_POOL = sorted(set(_sim.transform_options("BASH")[1]) - set(_sim.unsupported_cards()))


def slug(debug_name: str) -> str:
    """`sim::replay::slug`: `KnightsElite` to `KNIGHTS_ELITE`, the game name
    an `End.encounter` (a Rust debug name) has in `_sim.encounters()`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", debug_name).upper()


def encode(runs: list[dict]) -> dict[str, Tensor]:
    """Run states to padded id tensors and a few numbers."""
    n = len(runs)
    cards = np.zeros((n, MAX_DECK), dtype=np.int64)
    upgraded = np.zeros((n, MAX_DECK), dtype=np.int64)
    enchants = np.zeros((n, MAX_DECK), dtype=np.int64)
    relics = np.zeros((n, MAX_RELICS), dtype=np.int64)
    potions = np.zeros((n, MAX_POTIONS), dtype=np.int64)
    numbers = np.zeros((n, 8), dtype=np.float32)
    card, enchant, relic, potion = INDEX["card"], INDEX["enchant"], INDEX["relic"], INDEX["potion"]
    for i, r in enumerate(runs):
        deck = r["deck"][:MAX_DECK]
        cards[i, : len(deck)] = [card.get(c["id"], 0) for c in deck]
        upgraded[i, : len(deck)] = [bool(c.get("up")) for c in deck]
        enchants[i, : len(deck)] = [enchant.get(c["ench"][0], 0) if c.get("ench") else 0 for c in deck]
        held = r["relics"][:MAX_RELICS]
        relics[i, : len(held)] = [relic.get(name, 0) for name in held]
        drinks = [p for p in r["potions"] if p][:MAX_POTIONS]
        potions[i, : len(drinks)] = [potion.get(name, 0) for name in drinks]
        numbers[i, :4] = (r["hp"] / max(1, r["max_hp"]), r["max_hp"] / 100, len(r["deck"]) / 40, r.get("max_energy", 3) / 3)
        numbers[i, 4 + min(ACTS.index(r["act"]), 3)] = 1.0
    arrays = {"cards": cards, "upgraded": upgraded, "enchants": enchants, "relics": relics, "potions": potions, "numbers": numbers}
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


class DeckValue(nn.Module):
    """Sets of cards, relics and potions, pooled, plus the numbers, to a
    value per elite and boss encounter (`ENCOUNTERS`), or `outputs` other
    values (`sts2ai.runvalue`)."""

    def __init__(self, card_dim: int = 32, hidden: int = 256, outputs: int = len(ENCOUNTERS)):
        super().__init__()
        self.card = nn.Embedding(len(IDS["card"]) + 1, card_dim, padding_idx=0)
        self.upgraded = nn.Embedding(2, card_dim)
        self.enchant = nn.Embedding(len(IDS["enchant"]) + 1, card_dim, padding_idx=0)
        self.card_mlp = nn.Sequential(nn.Linear(card_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.relic = nn.Embedding(len(IDS["relic"]) + 1, 32, padding_idx=0)
        self.potion = nn.Embedding(len(IDS["potion"]) + 1, 16, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(hidden + 32 + 16 + 8, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        present = (x["cards"] > 0).unsqueeze(-1).float()
        card = self.card(x["cards"]) + self.upgraded(x["upgraded"]) + self.enchant(x["enchants"])
        deck = (self.card_mlp(card) * present).sum(1) / 20
        relics = self.relic(x["relics"]).sum(1)
        potions = self.potion(x["potions"]).sum(1)
        return self.head(torch.cat([deck, relics, potions, x["numbers"]], dim=1))

    def seed_cards(self, policy: Policy) -> None:
        """Start the card embedding from the combat policy's: both index a
        card as its sim id + 1."""
        with torch.no_grad():
            self.card.weight.copy_(policy.card.weight[: self.card.num_embeddings].to(self.card.weight.device))


def load(path: Path, device: torch.device) -> DeckValue:
    """A saved net. Ids only ever append (CLAUDE.md), so ids added since it
    was trained get a zero embedding: they read as nothing held."""
    saved = torch.load(path, map_location=device)
    if saved["encounters"] != ENCOUNTERS:
        raise ValueError(f"{path}: the encounters changed since it was trained; retrain")
    model = DeckValue().to(device)
    model.load_state_dict(grown(saved, path))
    return model.eval()


def grown(saved: dict, path: Path) -> dict[str, Tensor]:
    """A saved net's weights with zero embeddings for the ids added since."""
    state = saved["model"]
    for kind in ("card", "enchant", "relic", "potion"):
        old = saved["vocab"][kind]
        if old != IDS[kind][: len(old)]:
            raise ValueError(f"{path}: the {kind} ids moved since it was trained; retrain")
        weight = state[f"{kind}.weight"]
        state[f"{kind}.weight"] = torch.cat([weight, weight.new_zeros((len(IDS[kind]) - len(old), weight.shape[1]))])
    return state


class RunPotential:
    """Phi for the run policy's shaping (docs/training.md, The run policy):
    the net's mean predicted fight value over the current act's elites and
    the boss or bosses its map shows, times `scale`, read off run decision
    rows (`sim::runobs`) alone, so it sees only what the policy sees. The
    rows index cards, enchantments, potions and the combat sim's relics as
    this net does; relics only the run knows read as none."""

    def __init__(self, model: DeckValue, layout: RunLayout, scale: float):
        self.model, self.L, self.scale = model, layout, scale
        device = next(model.parameters()).device
        names = _sim.run_names()
        # Per act id (the rows' `ACTS` order, which is ours), its elites;
        # per boss id, its column.
        acts = [None, *ACTS]
        self.elites = torch.tensor([[float(ACT_OF[e] == a and KIND_OF[e] == "Elite") for e in ENCOUNTERS] for a in acts], device=device)
        bosses = [slug(b) for b in names["boss"][1:]]
        self.bosses = torch.zeros((len(bosses) + 1, len(ENCOUNTERS)), device=device)
        for i, b in enumerate(bosses):
            self.bosses[i + 1, ENCOUNTERS.index(b)] = 1.0
        self.sim_relics = len(IDS["relic"])

    @torch.no_grad()
    def __call__(self, floats: Tensor, ids: Tensor) -> Tensor:
        """Phi per row `[B]`."""
        L, B = self.L, floats.shape[0]

        def seg(x: Tensor, start: int, count: int, width: int) -> Tensor:
            return x[:, start : start + count * width].view(B, count, width)

        deck_ids, deck_f = seg(ids, L.i_deck, L.max_deck, L.deck_ids), seg(floats, L.f_deck, L.max_deck, L.deck_floats)
        relics = seg(ids, L.i_relics, L.max_relics, L.relic_ids)[..., 0]
        g, act = floats[:, : L.global_floats], ids[:, 2]
        numbers = torch.zeros((B, 8), device=floats.device)
        # HP fraction, max HP / 100, deck size / 40, energy / 3, the act.
        numbers[:, 0], numbers[:, 1], numbers[:, 2], numbers[:, 3] = g[:, 0], g[:, 1], g[:, 11], 1.0
        numbers[torch.arange(B), 3 + act.clamp(min=1)] = 1.0
        x = {
            "cards": deck_ids[..., 0],
            "upgraded": deck_f[..., 1].long(),
            "enchants": deck_ids[..., 1],
            "relics": relics.where(relics <= self.sim_relics, 0),
            "potions": seg(ids, L.i_potions, L.max_potions, L.potion_ids)[..., 0],
            "numbers": numbers,
        }
        weights = self.elites[act] + self.bosses[ids[:, 3]] + self.bosses[ids[:, 4]]
        values = self.model(x).float()
        return self.scale * (values * weights).sum(1) / weights.sum(1).clamp(min=1.0)


def variant(run: dict, rng: np.random.Generator) -> dict:
    """The run one deck change away: a random reward card added, a random
    card upgraded, or one removed."""
    deck = [dict(c) for c in run["deck"]]
    kind = rng.integers(3)
    if kind == 0 or len(deck) < 6:
        deck.append({"id": str(rng.choice(REWARD_POOL)), "up": False})
    elif kind == 1 and (up := [i for i, c in enumerate(deck) if not c.get("up")]):
        deck[int(rng.choice(up))]["up"] = True
    else:
        deck.pop(int(rng.integers(len(deck))))
    return {**run, "deck": deck}


def state_scores(model: DeckValue, runs: list[dict]) -> np.ndarray:
    """A run state's score for afterstates (`afterstate.Scorer`): the
    network's mean fight reward over its act's and the next act's elites
    and bosses, mapped from reward (-1 a loss, up to 1.5 a clean win) to
    0..1, where afterstates put a run that died (0) and one won (1). 0 for
    a state with no run record."""
    scores = np.zeros(len(runs))
    have = [i for i, r in enumerate(runs) if r]
    if have:
        device = next(model.parameters()).device
        with torch.no_grad():
            x = {k: v.to(device, non_blocking=True) for k, v in encode([runs[i] for i in have]).items()}
            pred = ((model(x).float().cpu().numpy() + 1.0) / 2.5).clip(0.0, 1.0)
        for k, i in enumerate(have):
            scores[i] = pred[k, HORIZON[runs[i]["act"]]].mean()
    return scores


# Per act, the outputs a state's score averages: its elites and bosses and
# the next act's.
HORIZON = {act: [j for j, e in enumerate(ENCOUNTERS) if ACT_OF[e] in (act, NEXT_ACT.get(act))] for act in ACTS}


def label_many(policy: Policy, device: torch.device, runs: list[dict], repeats: int, seeds: list[int], next_act: bool = False) -> list[dict[str, float]]:
    """Per run, the mean fight reward per elite (at the run's HP) and boss
    (at full HP) of its act, and with `next_act` every elite and boss of the
    act after at full HP: the advisor's fights, all runs' at once."""
    return [mean_rewards(ends) for ends in fights_many(policy, device, [fight_job(r, s, next_act) for r, s in zip(runs, seeds)], repeats)]


def fight_job(run: dict, seed: int, next_act: bool) -> FightJob:
    """The fights a run state is labeled on (`label_many`)."""
    return FightJob(run, run["max_hp"], horizon(run["act"], None, run["hp"], run["max_hp"], next_act=next_act), seed)


def mean_rewards(ends: list[End]) -> dict[str, float]:
    """Mean fight reward by encounter."""
    by_enc: dict[str, list[float]] = {}
    for e in ends:
        by_enc.setdefault(slug(e.encounter), []).append(e.reward)
    return {k: float(np.mean(v)) for k, v in by_enc.items()}


def label(policy: Policy, device: torch.device, run: dict, repeats: int, seed: int, next_act: bool = False) -> dict[str, float]:
    """`label_many` for one run."""
    return label_many(policy, device, [run], repeats, [seed], next_act)[0]


def logged_starts(paths: list[Path], n: int, rng: np.random.Generator) -> list[dict]:
    """`n` run states drawn from fight logs (`runplay --fights-out`, setups
    files): each line's start, with its HP and act."""
    lines = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    starts = []
    for i in rng.choice(len(lines), size=min(n, len(lines)), replace=False):
        d = lines[i]
        starts.append({**d["start"], "hp": d["hp"], "max_hp": d["max_hp"], "act": ACT_OF[d["encounter"]]})
    return starts


def cmd_label(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device).eval()
    # The live-row count changes every step, hence dynamic; labeling runs
    # long enough to repay the compile.
    net = torch.compile(policy, dynamic=True) if device.type == "cuda" else policy
    rng = np.random.default_rng(args.seed)
    logged = logged_starts(args.starts, args.n, rng) if args.starts else None
    n = len(logged) if logged else args.n
    held: dict[int, tuple[dict, int]] = {}  # job index -> (run, pair seed), until its fights are in

    def jobs() -> Iterator[FightJob]:
        k = 0
        for i in range(n):
            run_seed = args.seed * 1_000_003 + i
            if logged:
                run = logged[i]
            else:
                floor = int(rng.integers(1, 3 * BOSS_FLOOR + 1))
                run = json.loads(_sim.generate_run(run_seed, floor))
                run["act"] = ACT_OF[run["encounter"]]
            for r in (run, variant(run, rng)):
                held[k] = (r, run_seed)
                k += 1
                yield fight_job(r, run_seed, args.next_act)

    written = 0
    with open(args.out, "a") as out:
        for j, ends in fights_stream(net, device, jobs(), args.repeats, pool=args.pool):
            run, pair = held.pop(j)
            out.write(json.dumps({"run": run, "values": mean_rewards(ends), "repeats": args.repeats, "pair": pair}) + "\n")
            written += 1
            if written % 100 == 0:
                out.flush()
                print(f"{written}/{2 * n} states", flush=True)


def cmd_train(args) -> None:
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    x = encode([r["run"] for r in rows])
    y = torch.zeros((len(rows), len(ENCOUNTERS)))
    mask = torch.zeros((len(rows), len(ENCOUNTERS)))
    for i, r in enumerate(rows):
        for enc, v in r["values"].items():
            j = ENCOUNTERS.index(enc)
            y[i, j], mask[i, j] = v, 1.0
    pairs = torch.tensor([r["pair"] for r in rows])
    # Split by pair, so a run and its variant land on the same side.
    rng = np.random.default_rng(0)
    val_pairs = set(rng.choice(pairs.unique().numpy(), size=max(1, len(pairs.unique()) // 5), replace=False).tolist())
    val = torch.tensor([p in val_pairs for p in pairs.tolist()])
    model = load(args.init, torch.device("cpu")).train() if args.init else DeckValue()
    if args.policy:
        policy = load_policy(args.policy, torch.device("cpu"))
        model.seed_cards(policy)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    def loss_on(idx: Tensor) -> Tensor:
        pred = model({k: v[idx] for k, v in x.items()})
        return ((pred - y[idx]) ** 2 * mask[idx]).sum() / mask[idx].sum()

    train_idx = torch.nonzero(~val).squeeze(1)
    val_idx = torch.nonzero(val).squeeze(1)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for b in train_idx[torch.randperm(len(train_idx))].split(256):
            opt.zero_grad()
            loss_on(b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_on(val_idx))
            if val_mse < best:
                best = val_mse
                torch.save({"model": model.state_dict(), "encounters": ENCOUNTERS, "vocab": IDS}, args.out)
            if (epoch + 1) % max(1, args.epochs // 10) == 0:
                print(f"epoch {epoch + 1}: train mse {loss_on(train_idx):.4f}  val mse {val_mse:.4f}  {pair_agreement(model, x, y, mask, pairs, val_idx)}")
    print(f"saved {args.out} at val mse {best:.4f}")


@torch.no_grad()
def pair_agreement(model: DeckValue, x, y: Tensor, mask: Tensor, pairs: Tensor, idx: Tensor) -> str:
    """How often the model gets the sign of a variant's change right, on
    the encounters both were labelled on: what ranking choices needs."""
    pred = model({k: v[idx] for k, v in x.items()})
    by_pair: dict[int, list[int]] = {}
    for n, i in enumerate(idx.tolist()):
        by_pair.setdefault(int(pairs[i]), []).append(n)
    right = total = 0
    for rows in by_pair.values():
        if len(rows) != 2:
            continue
        a, b = rows
        m = mask[idx[a]] * mask[idx[b]]
        if m.sum() == 0:
            continue
        true = ((y[idx[b]] - y[idx[a]]) * m).sum() / m.sum()
        guess = ((pred[b] - pred[a]) * m).sum() / m.sum()
        right += int(torch.sign(true) == torch.sign(guess))
        total += 1
    return f"pair sign agreement {right}/{total}" if total else "no pairs"


def played_starts(paths: list[Path]) -> list[dict]:
    """The start record of each recording, with the first snapshot's HP
    when the record predates carrying it. Without paths, every played run's
    fight at the trained ascension: a start with a seed, and no console
    line in its run."""
    starts = []
    for path in paths or sorted(DEFAULT_RECORDINGS.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        start = next((r for r in lines if r["t"] == "start"), None)
        snap = next((r for r in lines if r["t"] == "snapshot"), None)
        played = start is not None and start.get("seed") and not start.get("scripted") and start.get("ascension") == ASCENSION
        if start is None or snap is None or (not paths and not played):
            continue
        start.setdefault("hp", snap["hp"])
        start.setdefault("max_hp", snap["max_hp"])
        start["act"] = ACT_OF[start["encounter"]]
        starts.append(start)
    return starts


@torch.no_grad()
def opening_values(policy: Policy, device: torch.device, jobs: list[FightJob], rolls: int = 4) -> np.ndarray:
    """Each job's mean value-head read over its fights' openings, `rolls`
    per encounter: how afterstate's forecast reads a run state, before its
    calibration."""
    n = sum(len(job.encounters) for job in jobs) * rolls
    envs = Envs(n)
    starts = envs.sim.queue_fight_jobs([(json.dumps(job.start), job.max_hp, job.encounters, job.seed) for job in jobs], rolls)
    envs.sim.start_fights(list(range(n)), list(range(n)), envs.floats, envs.ids, envs.mask)  # an opening already over still reads
    values = np.concatenate(
        [
            policy(torch.from_numpy(envs.floats[i : i + 4096]).to(device), torch.from_numpy(envs.ids[i : i + 4096]).to(device))[1].float().cpu().numpy()
            for i in range(0, n, 4096)
        ]
    )
    return np.array([values[a:b].mean() for a, b in zip(starts, [*starts[1:], n])])


def options_of(start: dict, rng: np.random.Generator) -> list[dict]:
    """What a card reward and a rest site offer a deck: keeping it, three
    reward cards, an upgrade, a removal."""
    deck = start["deck"]
    options = [start] + [{**start, "deck": deck + [{"id": str(c), "up": False}]} for c in rng.choice(REWARD_POOL, 3, replace=False)]
    if up := [i for i, c in enumerate(deck) if not c.get("up")]:
        i = int(rng.choice(up))
        options.append({**start, "deck": deck[:i] + [{**deck[i], "up": True}] + deck[i + 1 :]})
    i = int(rng.integers(len(deck)))
    options.append({**start, "deck": deck[:i] + deck[i + 1 :]})
    return options


def cmd_check(args) -> None:
    """For each deck, the choices a card reward and a rest site offer,
    valued by the fights at `--repeats` (the truth), by the network, and by
    the value head on the fights' openings (afterstate's read). Prints how
    often each pick matches the fights' and what it gives up against the
    fights' best, beside a random pick's."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device).eval()
    model = load(args.model, torch.device("cpu"))
    rng = np.random.default_rng(args.seed)
    starts = logged_starts(args.starts, args.decks, rng) if args.starts else played_starts(args.recordings)
    if not starts:
        raise SystemExit("no decks: no played-run recordings (pass files, or --starts)")
    pickers = ("network", "opening", "random")
    same = dict.fromkeys(pickers, 0.0)
    regret = dict.fromkeys(pickers, 0.0)
    for first in range(0, len(starts), args.group):
        decks = [options_of(start, rng) for start in starts[first : first + args.group]]
        options = [o for d in decks for o in d]
        jobs = [fight_job(o, args.seed, args.next_act) for o in options]
        truth = [mean_rewards(ends) for ends in fights_many(policy, device, jobs, args.repeats)]
        opening = opening_values(policy, device, jobs)
        with torch.no_grad():
            pred = model(encode(options)).numpy()
        at = 0
        for start, opts in zip(starts[first:], decks):
            part = slice(at, at + len(opts))
            at += len(opts)
            cols = [ENCOUNTERS.index(e) for e in truth[part.start]]
            fought = np.array([np.mean(list(t.values())) for t in truth[part]])
            picks = {"network": int(pred[part][:, cols].mean(1).argmax()), "opening": int(opening[part].argmax())}
            for k, j in picks.items():
                same[k] += int(j == fought.argmax())
                regret[k] += float(fought.max() - fought[j])
            same["random"] += 1 / len(opts)
            regret["random"] += float(fought.max() - fought.mean())
            print(
                f"{start['encounter']:28s} {len(start['deck']):2d} cards  fights pick {int(fought.argmax())} ({fought.max():+.2f})  net {picks['network']} opening {picks['opening']}",
                flush=True,
            )
    n = len(starts)
    for k in pickers:
        print(f"{k:8s} same pick {same[k] / n:.0%}  gives up {regret[k] / n:.3f} a fight on average")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="play fights on run states (generated, or from fight logs) and write labels")
    lab.add_argument("checkpoint", type=Path)
    lab.add_argument("--n", type=int, default=200, help="run states, each with one variant")
    lab.add_argument("--repeats", type=int, default=64, help="fights per encounter per state")
    lab.add_argument("--seed", type=int, default=1)
    lab.add_argument("--out", type=Path, default=Path("deckvalue-labels.jsonl"))
    lab.add_argument("--starts", type=Path, nargs="+", help="fight logs (`runplay --fights-out`) to draw the run states from instead of generating them")
    lab.add_argument("--next-act", action="store_true", help="label the next act's elites and bosses too, at full HP")
    lab.add_argument("--pool", type=int, default=16384, help="envs playing the fights, each starting the next queued fight as its own ends")
    tr = sub.add_parser("train", help="fit the network to labels")
    tr.add_argument("labels", type=Path)
    tr.add_argument("--policy", type=Path, help="combat checkpoint to seed the card embedding from")
    tr.add_argument("--init", type=Path, help="a trained net to start from, for labels from a newer combat checkpoint")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--out", type=Path, default=Path("deckvalue.pt"))
    ch = sub.add_parser("check", help="set the network's picks beside the fights' on played decks")
    ch.add_argument("checkpoint", type=Path)
    ch.add_argument("model", type=Path)
    ch.add_argument("recordings", nargs="*", type=Path, help="recordings to take decks from (default: every played run's)")
    ch.add_argument("--repeats", type=int, default=256, help="fights per encounter per option")
    ch.add_argument("--starts", type=Path, nargs="+", help="fight logs to draw the decks from instead of played runs")
    ch.add_argument("--decks", type=int, default=200, help="decks drawn from --starts")
    ch.add_argument("--next-act", action="store_true", help="value the next act's fights too, as the labels did")
    ch.add_argument("--group", type=int, default=16, help="decks whose options are fought together")
    ch.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    {"label": cmd_label, "train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
