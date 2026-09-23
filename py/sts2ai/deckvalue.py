"""Deck value: a network that predicts what the deck advisor's fights find.

    uv run python -m sts2ai.deckvalue label runs/<run>/latest.pt --n 200 --out labels.jsonl
    uv run python -m sts2ai.deckvalue train labels.jsonl --out deckvalue.pt
    uv run python -m sts2ai.deckvalue check runs/<run>/latest.pt deckvalue.pt [RECORDING ...]

`sts2ai.cards` prices a deck choice by playing hundreds of fights per option.
This learns that price: a run state (deck, relics, potions, HP, act) in, the
mean fight reward against each elite and boss of its act out, so a choice
costs one forward pass instead of seconds of fights. The fights stay the
teacher: `label` plays them on generated run states, each with a variant
one change away (a card added, upgraded or removed) so the labels carry the
differences a choice makes, and `train` fits the network to them. `check`
sets the network's picks beside the fights' on decks from played runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sts2ai import _sim
from sts2ai.cards import BOSS_FLOOR, fights, horizon
from sts2ai.env import DEFAULT_RECORDINGS
from sts2ai.model import Policy, load_policy

ACTS = ["Overgrowth", "Underdocks", "Hive", "Glory"]
MAX_DECK, MAX_RELICS, MAX_POTIONS = 64, 32, 5
IDS = _sim.game_ids()
INDEX = {kind: {name: i + 1 for i, name in enumerate(names)} for kind, names in IDS.items()}
# The model's outputs: every elite and boss encounter, by game name.
ENCOUNTERS = [name for name, _, kind in _sim.encounters() if kind in ("Elite", "Boss")]
ACT_OF = {name: act for name, act, _ in _sim.encounters()}
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
    cards = torch.zeros((n, MAX_DECK), dtype=torch.long)
    upgraded = torch.zeros((n, MAX_DECK), dtype=torch.long)
    enchants = torch.zeros((n, MAX_DECK), dtype=torch.long)
    relics = torch.zeros((n, MAX_RELICS), dtype=torch.long)
    potions = torch.zeros((n, MAX_POTIONS), dtype=torch.long)
    numbers = torch.zeros((n, 8))
    for i, r in enumerate(runs):
        for j, c in enumerate(r["deck"][:MAX_DECK]):
            cards[i, j] = INDEX["card"].get(c["id"], 0)
            upgraded[i, j] = int(bool(c.get("up")))
            if c.get("ench"):
                enchants[i, j] = INDEX["enchant"].get(c["ench"][0], 0)
        for j, name in enumerate(r["relics"][:MAX_RELICS]):
            relics[i, j] = INDEX["relic"].get(name, 0)
        for j, name in enumerate(p for p in r["potions"] if p):
            if j < MAX_POTIONS:
                potions[i, j] = INDEX["potion"].get(name, 0)
        act = ACTS.index(r["act"])
        numbers[i, :4] = torch.tensor([r["hp"] / max(1, r["max_hp"]), r["max_hp"] / 100, len(r["deck"]) / 40, r.get("max_energy", 3) / 3])
        numbers[i, 4 + min(act, 3)] = 1.0
    return {"cards": cards, "upgraded": upgraded, "enchants": enchants, "relics": relics, "potions": potions, "numbers": numbers}


class DeckValue(nn.Module):
    """Sets of cards, relics and potions, pooled, plus the numbers, to a
    value per elite and boss encounter (`ENCOUNTERS`)."""

    def __init__(self, card_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.card = nn.Embedding(len(IDS["card"]) + 1, card_dim, padding_idx=0)
        self.upgraded = nn.Embedding(2, card_dim)
        self.enchant = nn.Embedding(len(IDS["enchant"]) + 1, card_dim, padding_idx=0)
        self.card_mlp = nn.Sequential(nn.Linear(card_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.relic = nn.Embedding(len(IDS["relic"]) + 1, 32, padding_idx=0)
        self.potion = nn.Embedding(len(IDS["potion"]) + 1, 16, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(hidden + 32 + 16 + 8, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(ENCOUNTERS)))

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


def label(policy: Policy, device: torch.device, run: dict, repeats: int, seed: int) -> dict[str, float]:
    """Mean fight reward per elite (at the run's HP) and boss (at full HP)
    of the run's act: the advisor's fights."""
    encounters = horizon(run["act"], None, run["hp"], run["max_hp"], next_act=False)
    by_enc: dict[str, list[float]] = {}
    for e in fights(policy, device, run, run["max_hp"], encounters, repeats, seed):
        by_enc.setdefault(slug(e.encounter), []).append(e.reward)
    return {k: float(np.mean(v)) for k, v in by_enc.items()}


def cmd_label(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device).eval()
    rng = np.random.default_rng(args.seed)
    with open(args.out, "a") as out, torch.no_grad():
        for i in range(args.n):
            floor = int(rng.integers(1, 3 * BOSS_FLOOR + 1))
            run_seed = args.seed * 1_000_003 + i
            run = json.loads(_sim.generate_run(run_seed, floor))
            run["act"] = ACT_OF[run["encounter"]]
            for r in (run, variant(run, rng)):
                values = label(policy, device, r, args.repeats, seed=run_seed)
                out.write(json.dumps({"run": r, "values": values, "repeats": args.repeats, "pair": run_seed}) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{args.n}", flush=True)


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
    model = DeckValue()
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
    fight: a start with a seed, and no console line in its run."""
    starts = []
    for path in paths or sorted(DEFAULT_RECORDINGS.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        start = next((r for r in lines if r["t"] == "start"), None)
        snap = next((r for r in lines if r["t"] == "snapshot"), None)
        if start is None or snap is None or (not paths and (not start.get("seed") or start.get("scripted"))):
            continue
        start.setdefault("hp", snap["hp"])
        start.setdefault("max_hp", snap["max_hp"])
        start["act"] = ACT_OF[start["encounter"]]
        starts.append(start)
    return starts


def cmd_check(args) -> None:
    """For each played deck, the choices a card reward and a rest site
    offer (three reward cards, an upgrade, a removal, or keeping it), valued
    by the fights at `--repeats` and by the network, over the same
    encounters. Prints how often the picks match and what the network's
    pick gives up against the fights' best, each beside a random pick's."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device).eval()
    saved = torch.load(args.model, map_location="cpu")
    model = DeckValue()
    model.load_state_dict(saved["model"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    starts = played_starts(args.recordings)
    if not starts:
        raise SystemExit("no played-run recordings (pass files to use others)")
    same = regret = random_regret = 0.0
    for n, start in enumerate(starts):
        deck = start["deck"]
        options = [start] + [{**start, "deck": deck + [{"id": str(c), "up": False}]} for c in rng.choice(REWARD_POOL, 3, replace=False)]
        if up := [i for i, c in enumerate(deck) if not c.get("up")]:
            i = int(rng.choice(up))
            options.append({**start, "deck": deck[:i] + [{**deck[i], "up": True}] + deck[i + 1 :]})
        i = int(rng.integers(len(deck)))
        options.append({**start, "deck": deck[:i] + deck[i + 1 :]})
        with torch.no_grad():
            truth = [label(policy, device, o, args.repeats, seed=args.seed) for o in options]
            pred = model(encode(options))
        cols = [ENCOUNTERS.index(e) for e in truth[0]]
        fought = np.array([np.mean(list(t.values())) for t in truth])
        guessed = pred[:, cols].mean(1).numpy()
        same += int(fought.argmax() == guessed.argmax())
        regret += float(fought.max() - fought[guessed.argmax()])
        random_regret += float(fought.max() - fought.mean())
        print(f"{start['encounter']:28s} {len(deck):2d} cards  fights pick {int(fought.argmax())} ({fought.max():+.2f})  net pick {int(guessed.argmax())} ({fought[guessed.argmax()]:+.2f})", flush=True)
    n = len(starts)
    print(f"{n} decks: same pick {same / n:.0%} (random {1 / len(options):.0%}), gives up {regret / n:.3f} a fight on average (random {random_regret / n:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="play fights on generated run states and write labels")
    lab.add_argument("checkpoint", type=Path)
    lab.add_argument("--n", type=int, default=200, help="run states, each with one variant")
    lab.add_argument("--repeats", type=int, default=64, help="fights per encounter per state")
    lab.add_argument("--seed", type=int, default=1)
    lab.add_argument("--out", type=Path, default=Path("deckvalue-labels.jsonl"))
    tr = sub.add_parser("train", help="fit the network to labels")
    tr.add_argument("labels", type=Path)
    tr.add_argument("--policy", type=Path, help="combat checkpoint to seed the card embedding from")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--out", type=Path, default=Path("deckvalue.pt"))
    ch = sub.add_parser("check", help="set the network's picks beside the fights' on played decks")
    ch.add_argument("checkpoint", type=Path)
    ch.add_argument("model", type=Path)
    ch.add_argument("recordings", nargs="*", type=Path, help="recordings to take decks from (default: every played run's)")
    ch.add_argument("--repeats", type=int, default=256, help="fights per encounter per option")
    ch.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    {"label": cmd_label, "train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
