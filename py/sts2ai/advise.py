"""Advice for the combat you are playing, from a trained policy.

    uv run python -m sts2ai.advise runs/<run>/latest.pt
    uv run python -m sts2ai.advise runs/<run>/latest.pt --replay FILE --delay 0.2

Follows the newest file the recorder mod writes, keeps a sim combat in
sync with the game the way the replay harness does (docs/replay.md), and
prints what the policy would do at every decision point. You play the
moves yourself; nothing is sent back to the game.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import DEFAULT_RECORDINGS, Layout
from sts2ai.model import Policy, load_policy, masked_logits

ALTERNATIVES = 2


class Session:
    """One recording followed line by line: the sim kept in sync, and the
    policy's pick printed once per decision point."""

    def __init__(self, policy: Policy, device: torch.device):
        self.policy = policy
        self.device = device
        layout = Layout.load()
        self.floats = np.zeros((1, layout.n_floats), np.float32)
        self.ids = np.zeros((1, layout.n_ids), np.int64)
        self.mask = np.zeros((1, layout.n_actions), np.bool_)
        self.sim = _sim.Advisor()
        # Decision points are identified by the sim's progress counters, so
        # the same one is not advised twice while the recorder is quiet.
        self.advised: tuple[int, int, int] | None = None

    def feed(self, line: str) -> None:
        self.handle(self.sim.feed_line(line))

    def flush(self) -> None:
        """Ask the sim to settle a snapshot it is holding back."""
        self.handle(self.sim.flush())

    def handle(self, status: str) -> None:
        if status.startswith("diverged"):
            print(f"  {status}")
            return
        if status == "ended":
            snapshots, actions, _ = self.sim.counts()
            print(f"  combat over: {snapshots} decision points, {actions} actions\n")
            self.advised = None
            return
        if status == "decision":
            key = self.sim.counts()
            if key != self.advised:
                self.advised = key
                self.advise()

    @torch.no_grad()
    def advise(self) -> None:
        self.sim.observe(self.floats, self.ids, self.mask)
        logits, _ = self.policy(
            torch.from_numpy(self.floats).to(self.device), torch.from_numpy(self.ids).to(self.device)
        )
        probs = masked_logits(logits, torch.from_numpy(self.mask).to(self.device)).softmax(dim=1)[0]
        best = torch.argsort(probs, descending=True)[: 1 + ALTERNATIVES]
        print(self.sim.summary())
        for rank, index in enumerate(int(i) for i in best):
            action = self.sim.describe(index)
            if action is None:
                continue
            arrow = "->" if rank == 0 else "  "
            print(f"  {arrow} {action:<38} {float(probs[index]):5.1%}")
        print()


def newest(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def live_lines(directory: Path, poll: float) -> Iterator[str | None]:
    """Lines of the newest recording as it grows, `None` whenever it is
    quiet. A new combat writes a new file, and this follows it there."""
    path: Path | None = None
    handle = None
    while True:
        latest = newest(directory)
        if latest is not None and latest != path:
            if handle is not None:
                handle.close()
            path, handle = latest, latest.open()
            print(f"following {latest.name}\n")
        while handle is not None:
            where = handle.tell()
            line = handle.readline()
            if not line.endswith("\n"):
                handle.seek(where)
                break
            yield line
        yield None
        time.sleep(poll)


def replay_lines(path: Path, delay: float) -> Iterator[str | None]:
    """An existing recording fed as if it were being written now."""
    print(f"replaying {path.name}\n")
    for line in path.read_text().splitlines():
        yield line
        yield None
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dir", type=Path, default=DEFAULT_RECORDINGS, help="recordings folder to follow")
    ap.add_argument("--replay", type=Path, help="feed this recording instead of following the game")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between records when replaying")
    ap.add_argument("--poll", type=float, default=0.1, help="seconds between checks of the recordings folder")
    args = ap.parse_args()
    # Advice is worth nothing late: keep it flowing when stdout is a pipe.
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()

    session = Session(policy, device)
    lines = replay_lines(args.replay, args.delay) if args.replay else live_lines(args.dir, args.poll)
    for line in lines:
        if line is None:
            session.flush()
        else:
            session.feed(line)


if __name__ == "__main__":
    main()
