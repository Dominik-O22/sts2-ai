"""Run value: the chance a run wins from where it stands, learned from how
our own runs end.

    uv run python -m sts2ai.runvalue train FIGHTS.jsonl RUNS.jsonl --out runvalue.pt [--policy COMBAT.pt]

The fight and boss scorers (the forecast, `sts2ai.deckvalue`) value a run
state by the next fights, so afterstates scored by them keep runs alive
early with decks that lose the act 3 bosses. This values a state by the
run's end instead. Its rows are the elite and boss fights a `runplay
--fights-out --runs-out` eval logged, each joined by the run's seed to
whether that run was won and the floor it ended on. The net is
`deckvalue.DeckValue`'s (deck, relics, potions, HP, act) with two outputs:
the logit of winning and the final floor over 49, which is denser and
helps the body learn. `state_scores` gives afterstates the win chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sts2ai.deckvalue import ACT_OF, IDS, DeckValue, encode, grown
from sts2ai.model import load_policy

LAST_FLOOR = 49


def rows(fights: Path, runs: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """The logged states whose run is in `runs`: run records, won, final
    floor over `LAST_FLOOR`, and run seed."""
    lines = [json.loads(line) for line in runs.read_text().splitlines() if line.strip()]
    ends = {r["seed"]: r for r in lines if "seed" in r}
    states, won, floor, seed = [], [], [], []
    for line in fights.read_text().splitlines():
        d = json.loads(line)
        s = d["start"].get("seed")
        if s in ends:
            states.append({**d["start"], "hp": d["hp"], "max_hp": d["max_hp"], "act": ACT_OF[d["encounter"]]})
            won.append(ends[s]["end"] == "won")
            floor.append(ends[s]["floor"] / LAST_FLOOR)
            seed.append(s)
    return states, np.array(won, np.float32), np.array(floor, np.float32), np.array(seed)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Chance a won run's state scores above a lost one's (ties half)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def cmd_train(args) -> None:
    states, won, floor, seed = rows(args.fights, args.runs)
    x = encode(states)
    y = torch.from_numpy(np.stack([won, floor], 1))
    # Split by run, so no run has states on both sides.
    rng = np.random.default_rng(0)
    seeds = np.unique(seed)
    held = set(rng.choice(seeds, size=len(seeds) // 5, replace=False).tolist())
    val = torch.from_numpy(np.array([s in held for s in seed]))
    print(f"{len(states)} states from {len(seeds)} runs, {won.mean():.1%} from won runs; {int(val.sum())} held out")
    model = DeckValue(outputs=2)
    if args.policy:
        model.seed_cards(load_policy(args.policy, torch.device("cpu")))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    def loss_on(idx: Tensor) -> Tensor:
        out = model({k: v[idx] for k, v in x.items()})
        return bce(out[:, 0], y[idx, 0]) + ((out[:, 1] - y[idx, 1]) ** 2).mean()

    train_idx, val_idx = torch.nonzero(~val).squeeze(1), torch.nonzero(val).squeeze(1)
    base = float(won[~val.numpy()].mean())
    base_bce = -(won[val.numpy()] * np.log(base) + (1 - won[val.numpy()]) * np.log(1 - base)).mean()
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for b in train_idx[torch.randperm(len(train_idx))].split(512):
            opt.zero_grad()
            loss_on(b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model({k: v[val_idx] for k, v in x.items()})
            win_bce = float(bce(out[:, 0], y[val_idx, 0]))
            if win_bce < best:
                best = win_bce
                torch.save({"model": model.state_dict(), "vocab": IDS}, args.out)
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(
                f"epoch {epoch + 1}: val win bce {win_bce:.4f} (base rate {base_bce:.4f}), auc {auc(out[:, 0].numpy(), y[val_idx, 0].numpy()):.3f}", flush=True
            )
    model.load_state_dict(torch.load(args.out)["model"])
    with torch.no_grad():
        p = torch.sigmoid(model({k: v[val_idx] for k, v in x.items()})[:, 0]).numpy()
    truth = y[val_idx, 0].numpy()
    print(f"saved {args.out} at val win bce {best:.4f}; held-out auc {auc(p, truth):.3f}")
    # Decisions compare states within an act, so the ranking that matters
    # is within one.
    acts = np.array([states[i]["act"] for i in val_idx.tolist()])
    for act in dict.fromkeys(acts):
        m = acts == act
        if 0 < truth[m].sum() < m.sum():
            print(f"  {act:10s} {m.sum():6d} states, {truth[m].mean():5.1%} from won runs, auc {auc(p[m], truth[m]):.3f}")
    print("predicted win chance -> won, held out:")
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)):
        b = (p >= lo) & (p < hi)
        if b.any():
            print(f"  {lo:4.0%}-{min(hi, 1):4.0%}  {b.sum():6d} states  predicted {p[b].mean():6.1%}  won {truth[b].mean():6.1%}")


def load(path: Path, device: torch.device) -> DeckValue:
    """A saved run value net."""
    model = DeckValue(outputs=2).to(device)
    model.load_state_dict(grown(torch.load(path, map_location=device), path))
    return model.eval()


def state_scores(model: DeckValue, runs: list[dict]) -> np.ndarray:
    """Each run record's win chance, 0 for a state with no record: the
    afterstate score (`afterstate.Scorer`), where a run won scores 1 and one
    lost 0."""
    scores = np.zeros(len(runs))
    have = [i for i, r in enumerate(runs) if r]
    if have:
        device = next(model.parameters()).device
        with torch.no_grad():
            x = {k: v.to(device, non_blocking=True) for k, v in encode([runs[i] for i in have]).items()}
            scores[have] = torch.sigmoid(model(x)[:, 0].float()).cpu().numpy()
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="fit the net to logged states and their runs' ends")
    tr.add_argument("fights", type=Path, help="runplay --fights-out of the runs")
    tr.add_argument("runs", type=Path, help="runplay --runs-out of the same runs")
    tr.add_argument("--policy", type=Path, help="combat checkpoint to seed the card embedding from")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--weight-decay", type=float, default=1.0)
    tr.add_argument("--out", type=Path, default=Path("runvalue.pt"))
    args = ap.parse_args()
    {"train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
