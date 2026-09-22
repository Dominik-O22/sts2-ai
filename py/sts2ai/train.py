"""Train the combat policy.

    uv run python -m sts2ai.train [--iters N] [--envs N] ...

Every `Config` field is a flag. Logs go to runs/<time>/ for TensorBoard,
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
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(default) if not isinstance(default, Path) else Path, default=default)
    args = ap.parse_args()
    train(Config(**vars(args)))


if __name__ == "__main__":
    main()
