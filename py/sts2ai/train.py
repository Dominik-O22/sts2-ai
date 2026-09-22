"""Train the combat policy.

    uv run python -m sts2ai.train [--iters N] [--envs N] ...

Every `Config` field is a flag; `--resume runs/<time>/latest.pt` continues a run. Logs go to runs/<time>/ for TensorBoard,
and the latest checkpoint to runs/<time>/latest.pt.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from sts2ai.ppo import Config, train


def main() -> None:
    ap = argparse.ArgumentParser()
    defaults = Config()
    for f in fields(Config):
        default = getattr(defaults, f.name)
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
            continue
        kind = Path if default is None or isinstance(default, Path) else type(default)
        ap.add_argument(flag, type=kind, default=default)
    args = ap.parse_args()
    train(Config(**vars(args)))


if __name__ == "__main__":
    main()
