"""Winners' run decisions for the run policy to start from
(docs/run-env.md, Winners' decisions): each decision a recorded run made
where the walk is faithful to it (`sim::history::imitate`), as the policy
would see it, with the option the player took.

    uv run python -m sts2ai.imitation build          # tracker/*.run -> tracker/imitation/{train,holdout}.npz
    uv run python -m sts2ai.imitation agree CKPT...  # holdout top-1 agreement by decision kind
    uv run python -m sts2ai.runtrain COMBAT --run-dir runs/imitate-1 --imitate ~/.local/share/SlayTheSpire2/sts2ai/tracker/imitation/train.npz --minutes 0

Runs are split by player (`setups.holdout_player`), as the fights are.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import RunLayout
from sts2ai.runmodel import RunArch, RunPolicy, load_run_policy
from sts2ai.setups import TRACKER, holdout_player

OUT = TRACKER / "imitation"
DECISIONS = _sim.run_names()["decision"]


@dataclass
class Rows:
    """Run rows (float16 and int16, as `runtrain` keeps them), the option
    token each took, its floor, whether the Rewards stream was still
    followed, and the run it came from."""

    floats: np.ndarray
    ids: np.ndarray
    option: np.ndarray
    floor: np.ndarray
    streamed: np.ndarray
    run: np.ndarray
    runs: np.ndarray

    def __len__(self) -> int:
        return len(self.option)

    @property
    def kind(self) -> np.ndarray:
        """Each row's decision, a `DECISIONS` index."""
        return self.ids[:, 0].astype(np.int64)

    def save(self, path: Path) -> None:
        np.savez(path, **{f: getattr(self, f) for f in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> Rows:
        with np.load(path) as z:
            return cls(**{f: z[f] for f in cls.__dataclass_fields__})


def build(runs_dir: Path, out: Path, share: float) -> None:
    L = RunLayout.load()
    parts: dict[str, list[tuple]] = {"train": [], "holdout": []}
    names: dict[str, list[str]] = {"train": [], "holdout": []}
    players: dict[str, set[str]] = {"train": set(), "holdout": set()}
    left_out: Counter[str] = Counter()
    walked = 0
    for path in sorted(runs_dir.glob("*.run")):
        got = _sim.imitation(path.read_text())
        if got is None:
            continue
        walked += 1
        floats, ids, option, floor, streamed, dropped = got
        left_out.update(dropped)
        if not option:
            continue
        player = path.name.rsplit("-", 1)[0]
        split = "holdout" if holdout_player(player, share) else "train"
        n = len(option)
        run = np.full(n, len(names[split]), dtype=np.int32)
        parts[split].append((floats.reshape(n, L.run_floats).astype(np.float16), ids.reshape(n, L.run_ids).astype(np.int16), option, floor, streamed, run))
        names[split].append(path.stem)
        players[split].add(player)
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, Rows]] = {}
    for split, p in parts.items():
        cols = [np.concatenate(c) for c in zip(*p)]
        rows = Rows(cols[0], cols[1], cols[2].astype(np.int64), cols[3].astype(np.int16), cols[4].astype(bool), cols[5], np.array(names[split]))
        rows.save(out / f"{split}.npz")
        table[split] = rows
        print(f"{split}: {len(rows)} rows from {len(names[split])} runs of {len(players[split])} players")
    print(f"{walked} runs walked; rows per decision kind (train / holdout, then those before the first rewards mismatch):")
    for k in range(1, len(DECISIONS)):
        counts = [(r.kind == k).sum() for r in table.values()]
        streamed = [((r.kind == k) & r.streamed).sum() for r in table.values()]
        if sum(counts):
            print(f"  {DECISIONS[k]:8s} {counts[0]:6d} / {counts[1]:5d}   {streamed[0]:6d} / {streamed[1]:5d}")
    print("left out:")
    for why, n in left_out.most_common():
        print(f"  {n:6d} {why}")


def batches(rows: Rows, device: torch.device, size: int, order: np.ndarray):
    """Rows `order` in batches on `device`: floats, ids, options."""
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        yield (
            torch.from_numpy(rows.floats[idx].astype(np.float32)).to(device),
            torch.from_numpy(rows.ids[idx].astype(np.int64)).to(device),
            torch.from_numpy(rows.option[idx]).to(device),
        )


@torch.no_grad()
def agreement(policy: RunPolicy, rows: Rows, device: torch.device, size: int = 512) -> np.ndarray:
    """Whether the policy's favourite option is the one each row took."""
    policy.eval()
    agree = []
    for floats, ids, option in batches(rows, device, size, np.arange(len(rows))):
        logits, _ = policy(floats, ids)
        agree.append((logits.argmax(1) == option).cpu().numpy())
    return np.concatenate(agree)


def by_kind(rows: Rows, hits: np.ndarray) -> dict[str, float]:
    """Mean of `hits` per decision kind, and over all rows."""
    out = {DECISIONS[k]: float(hits[rows.kind == k].mean()) for k in np.unique(rows.kind)}
    out["all"] = float(hits.mean())
    return out


def pretrain(policy: RunPolicy, train: Rows, device: torch.device, epochs: int, lr: float, size: int, holdout: Rows | None = None) -> None:
    """Behaviour cloning: cross-entropy on the option each row took, over
    the policy's own masked logits (absent options at -1e9). The value
    head is left as it was."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        policy.train()
        start, losses = time.perf_counter(), []
        for floats, ids, option in batches(train, device, size, rng.permutation(len(train))):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = policy(floats, ids)
            loss = torch.nn.functional.cross_entropy(logits.float(), option)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        line = f"imitate epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.3f} ({time.perf_counter() - start:.0f} s)"
        if holdout is not None:
            line += f", holdout agreement {agreement(policy, holdout, device).mean():.1%}"
        print(line, flush=True)


def agree(checkpoints: list[Path], holdout: Rows, device: torch.device, untrained: bool) -> None:
    """Top-1 agreement with the winners on `holdout`, by decision kind: for
    chance (a uniform pick among the options shown), an untrained policy
    and each checkpoint."""
    L = RunLayout.load()
    shown = (holdout.floats[:, L.f_options : L.f_options + L.max_options * L.option_floats : L.option_floats] != 0).sum(1)
    columns = {"chance": by_kind(holdout, 1.0 / shown)}
    if untrained:
        torch.manual_seed(0)
        columns["untrained"] = by_kind(holdout, agreement(RunPolicy(L, RunArch()).to(device), holdout, device))
    for path in checkpoints:
        columns[str(path)] = by_kind(holdout, agreement(load_run_policy(path, device)[0], holdout, device))
    kinds = [DECISIONS[k] for k in np.unique(holdout.kind)] + ["all"]
    counts = {DECISIONS[k]: int((holdout.kind == k).sum()) for k in np.unique(holdout.kind)} | {"all": len(holdout)}
    for i, name in enumerate(columns):
        print(f"  [{i}] {name}")
    print(f"  {'kind':8s} {'rows':>6s} " + " ".join(f"{f'[{i}]':>7s}" for i in range(len(columns))))
    for k in kinds:
        print(f"  {k:8s} {counts[k]:6d} " + " ".join(f"{c[k]:7.1%}" for c in columns.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="walk the tracker runs into train and holdout rows")
    b.add_argument("--runs", type=Path, default=TRACKER)
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--holdout", type=float, default=0.15, help="share of players held out")
    a = sub.add_parser("agree", help="holdout top-1 agreement by decision kind")
    a.add_argument("checkpoints", type=Path, nargs="*")
    a.add_argument("--holdout", type=Path, default=OUT / "holdout.npz")
    a.add_argument("--no-untrained", dest="untrained", action="store_false")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.runs, args.out, args.holdout)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agree(args.checkpoints, Rows.load(args.holdout), device, args.untrained)


if __name__ == "__main__":
    main()
