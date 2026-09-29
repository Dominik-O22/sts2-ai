"""The forecast (docs/run-env.md, The forecast): at a map step, how the
combat model expects the act's elites and its boss to go for the player as
they stand, so the run policy can tell an elite it can afford from one
that ends the run.

The sim hands out each map step's forecast fights (`sim::runobs::
forecast_fights`: every elite the act can hold and its boss or bosses, a
few rolled openings each, encoded at their opening). The combat policy's
value head reads them in one batch. Its value is the fight's expected
reward (a win about 1 + half the HP fraction kept, a loss about -1), so a
fitted `Calibration` turns it, with the HP fraction now, into a win chance
and the HP fraction kept after a win. `Forecaster.fill` writes the act's
elites' mean and the boss's into the run row's forecast slots; RunLoop
(runtrain, runplay) and `sts2ai.imitation build` all go through it.

    uv run python -m sts2ai.forecast calibrate LOG.jsonl --out CALIBRATION.json

`calibrate` fits the calibration on the fights `runplay --forecast-log`
wrote (each map step into an elite: the value that elite read and how the
fight went) and prints win rate and HP kept by forecast bucket.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from sts2ai.env import RunLayout
from sts2ai.model import Policy


class Fights(NamedTuple):
    """Forecast fights at their openings (`VecEnv.forecast`, `_sim.imitation`):
    combat rows, the run row each belongs to, and its encounter."""

    floats: np.ndarray
    ids: np.ndarray
    row: np.ndarray
    encounter: list[str]

    @classmethod
    def of(cls, got: tuple, n_floats: int, n_ids: int) -> Fights:
        floats, ids, row, encounter = got
        return cls(floats.reshape(-1, n_floats), ids.reshape(-1, n_ids), np.asarray(row, dtype=np.int64), encounter)


class Read(NamedTuple):
    """What the value head read, per run row and encounter: the mean value
    over the openings, the HP fraction now, and, calibrated, the win chance
    and the HP fraction kept after a win (NaN without a calibration)."""

    row: np.ndarray
    encounter: list[str]
    value: np.ndarray
    hp: np.ndarray
    win: np.ndarray
    kept: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Calibration:
    """Win chance sigmoid(win . [V, hp, 1]) and HP fraction kept after a
    win clip(kept . [V, hp, 1], 0, 1), from the opening value V and the
    HP fraction now; fitted per combat checkpoint (`calibrate`)."""

    win: tuple[float, float, float]
    kept: tuple[float, float, float]
    combat: str = ""

    def apply(self, value: np.ndarray, hp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([value, hp, np.ones_like(value)], axis=1)
        return _sigmoid(x @ np.array(self.win)), np.clip(x @ np.array(self.kept), 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        d = json.loads(path.read_text())
        return cls(tuple(d["win"]), tuple(d["kept"]), d.get("combat", ""))


class Forecaster:
    """The combat policy's value head over forecast fights, filled into run
    rows. Without a calibration it only reads (`read`), for logging."""

    def __init__(self, combat: Policy, device: torch.device, calibration: Calibration | None, layout: RunLayout | None = None):
        self.combat, self.device, self.calibration = combat, device, calibration
        self.L = layout or RunLayout.load()
        self.autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    @torch.no_grad()
    def values(self, floats: np.ndarray, ids: np.ndarray, batch: int = 2048) -> np.ndarray:
        """The value head's read of combat rows, one value each."""
        values = [np.zeros(0, dtype=np.float32)]
        for s in range(0, len(floats), batch):
            f = torch.from_numpy(np.ascontiguousarray(floats[s : s + batch])).to(self.device)
            i = torch.from_numpy(np.ascontiguousarray(ids[s : s + batch])).to(self.device)
            with self.autocast:
                values.append(self.combat(f, i)[1].float().cpu().numpy())
        return np.concatenate(values)

    def read(self, floats: np.ndarray, fights: Fights) -> Read:
        """Values per run row and encounter, the openings averaged."""
        if not len(fights.row):
            e = np.zeros(0)
            return Read(np.zeros(0, dtype=np.int64), [], e, e, e, e)
        value = self.values(fights.floats, fights.ids)
        # Openings come in runs of `forecast_rolls` per (row, encounter).
        R = self.L.forecast_rolls
        row = fights.row[::R]
        encounter = fights.encounter[::R]
        value = value.reshape(-1, R).mean(1)
        hp = floats[row, 0].astype(np.float32)
        if self.calibration is None:
            win = kept = np.full_like(value, np.nan)
        else:
            win, kept = self.calibration.apply(value, hp)
        return Read(row, encounter, value, hp, win, kept)

    def fill(self, floats: np.ndarray, fights: Fights) -> Read:
        """Writes each row's forecast (the act's elites' mean win chance and
        HP kept, then the boss's) into its slots in `floats`, in place; rows
        with no fights keep zeros. Returns what it read."""
        read = self.read(floats, fights)
        if self.calibration is None or not len(read.row):
            return read
        f0 = self.L.f_forecast
        boss = np.array([e.endswith("Boss") for e in read.encounter])
        for k, part in enumerate((~boss, boss)):
            rows, win, kept = read.row[part], read.win[part], read.kept[part]
            count = np.bincount(rows, minlength=len(floats))
            has = count > 0
            floats[has, f0 + 2 * k] = (np.bincount(rows, win, len(floats)) / np.maximum(count, 1))[has]
            floats[has, f0 + 2 * k + 1] = (np.bincount(rows, kept, len(floats)) / np.maximum(count, 1))[has]
        return read


def fit(value: np.ndarray, hp: np.ndarray, won: np.ndarray, kept: np.ndarray, combat: str = "") -> Calibration:
    """Logistic regression of the win on [V, hp, 1] (Newton steps, a little
    ridge) and least squares of the HP kept on the wins."""
    x = np.stack([value, hp, np.ones_like(value)], axis=1).astype(np.float64)
    y = won.astype(np.float64)
    w = np.zeros(3)
    for _ in range(50):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + 1e-3 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + 1e-3 * np.eye(3)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    wins = won.astype(bool)
    k, *_ = np.linalg.lstsq(x[wins], kept[wins].astype(np.float64), rcond=None)
    return Calibration(tuple(float(v) for v in w), tuple(float(v) for v in k), combat)


def table(win: np.ndarray, won: np.ndarray, kept: np.ndarray, kept_pred: np.ndarray) -> None:
    """Fights by forecast win chance bucket: how many, forecast, won, and
    HP kept after a win forecast and seen."""
    edges = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    print(f"  {'forecast':>11s} {'fights':>7s} {'mean':>6s} {'won':>6s}   {'kept fc':>7s} {'kept':>6s}")
    for lo, hi in zip(edges, edges[1:]):
        b = (win >= lo) & (win < hi)
        if not b.any():
            continue
        w = b & won
        kept_fc = f"{kept_pred[w].mean():7.2f}" if w.any() else f"{'-':>7s}"
        kept_seen = f"{kept[w].mean():6.2f}" if w.any() else f"{'-':>6s}"
        print(f"  {lo:5.2f}-{min(hi, 1):4.2f} {b.sum():7d} {win[b].mean():6.1%} {won[b].mean():6.1%}   {kept_fc} {kept_seen}")
    print(f"  {'all':>11s} {len(win):7d} {win.mean():6.1%} {won.mean():6.1%}")


def load_log(path: Path) -> dict[str, np.ndarray]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {k: np.array([r[k] for r in records]) for k in records[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="fit the calibration on runplay --forecast-log fights and print the table")
    c.add_argument("logs", type=Path, nargs="+")
    c.add_argument("--out", type=Path, default=None, help="write the calibration here")
    c.add_argument("--check", type=Path, default=None, help="print the table of an existing calibration instead of fitting")
    c.add_argument("--acts", default="0", help="acts (0-based, comma separated) to fit on")
    c.add_argument("--combat", default="", help="the combat checkpoint the log's runs played with, to record")
    args = ap.parse_args()
    logs = [load_log(p) for p in args.logs]
    log = {k: np.concatenate([g[k] for g in logs]) for k in logs[0]}
    keep = np.isin(log["act"], [int(a) for a in args.acts.split(",")])
    log = {k: v[keep] for k, v in log.items()}
    cal = Calibration.load(args.check) if args.check else fit(log["value"], log["hp"], log["won"], log["kept"], args.combat)
    win, kept = cal.apply(log["value"], log["hp"])
    print(f"calibration: win {np.round(cal.win, 3).tolist()}, kept {np.round(cal.kept, 3).tolist()}")
    print(f"{len(win)} elite fights, value to win: by forecast bucket")
    table(win, log["won"].astype(bool), log["kept"], kept)
    for enc in sorted(set(log["encounter"])):
        e = log["encounter"] == enc
        print(f"  {enc:28s} {e.sum():6d} forecast {win[e].mean():6.1%} won {log['won'][e].mean():6.1%}")
    if args.out:
        cal.save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
