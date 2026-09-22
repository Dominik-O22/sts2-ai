"""Advice for the combat you are playing, from a trained policy.

    uv run python -m sts2ai.advise runs/<run>/latest.pt
    uv run python -m sts2ai.advise runs/<run>/latest.pt --search 256
    uv run python -m sts2ai.advise runs/<run>/latest.pt --replay FILE --delay 0.2

Follows the newest file the recorder mod writes, keeps a sim combat in
sync with the game the way the replay harness does (docs/replay.md), and
prints what the policy would do at every decision point. You play the
moves yourself; nothing is sent back to the game.

With `--search N`, each decision also runs a turn search: N copies of the
sim play the rest of the turn under the policy, every legal first action
gets its share of copies, and the copies are scored by the shaped reward
they collect plus the value head at the end of the turn. The best plan is
printed as the whole line of plays. Draw piles are reshuffled per group of
copies, since the plan must not know what you will draw.
"""

from __future__ import annotations

import argparse
import json
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
# A turn is over well within this many actions; a runaway plan is cut here.
MAX_PLAN_STEPS = 40
# Plan scores closer than this are value-head noise: the policy's pick
# keeps the top line, and the search only overrules it by a clear margin.
# The unit is half an HP fraction, so 0.02 is about three HP.
PLAN_MARGIN = 0.02


class Session:
    """One recording followed line by line: the sim kept in sync, and the
    policy's pick printed once per decision point."""

    def __init__(self, policy: Policy, device: torch.device, search: int = 0, groups: int = 4):
        self.policy = policy
        self.device = device
        self.search = search
        self.groups = groups
        self.layout = Layout.load()
        self.floats = np.zeros((1, self.layout.n_floats), np.float32)
        self.ids = np.zeros((1, self.layout.n_ids), np.int64)
        self.mask = np.zeros((1, self.layout.n_actions), np.bool_)
        self.sim = _sim.Advisor()
        # Decision points are identified by the sim's progress counters, so
        # the same one is not advised twice while the recorder is quiet.
        self.advised: tuple[int, int, int] | None = None

    def feed(self, line: str) -> None:
        self.echo_player(line)
        self.handle(self.sim.feed_line(line))

    def echo_player(self, line: str) -> None:
        """What the player did after the last advice, for comparing."""
        if self.advised is None:
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return
        match rec.get("t"):
            case "play":
                name = rec["id"].replace("_", " ").title() + ("+" if rec.get("up") else "")
                target = "" if rec.get("target") is None else f" -> enemy {rec['target']}"
                print(f"  you: {name}{target}")
            case "potion":
                print(f"  you: use {rec['id'].replace('_', ' ').title()}")
            case "turn_start" if rec.get("turn", 1) > 1:
                print("  you: End turn")

    def flush(self) -> None:
        """Ask the sim to settle a snapshot it is holding back."""
        self.handle(self.sim.flush())

    def handle(self, status: str) -> None:
        if status.startswith("diverged"):
            print(f"  {status}")
            return
        # A fight outside what the sim models, an Underdocks run being the
        # common case. Say so once and keep following the recordings.
        if status.startswith("unsupported"):
            print(f"  cannot follow this fight: {status.removeprefix('unsupported: ')}\n")
            self.advised = None
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
        probs = masked_logits(logits, torch.from_numpy(self.mask).to(self.device)).softmax(dim=1)[0].cpu().numpy()
        # Two Strikes in hand are two actions to the policy and one to you.
        merged: dict[str, float] = {}
        for index in np.flatnonzero(self.mask[0]):
            if (action := self.sim.describe(int(index))) is not None:
                merged[action] = merged.get(action, 0.0) + float(probs[index])
        print(self.sim.summary())
        ranked = sorted(merged.items(), key=lambda kv: -kv[1])
        # With search on, the plan is the advice; the policy's own pick only
        # breaks ties inside it.
        if self.search and ranked:
            self.plan(ranked[0][0])
        else:
            for rank, (action, p) in enumerate(ranked[: 1 + ALTERNATIVES]):
                arrow = "->" if rank == 0 else "  "
                print(f"  {arrow} {action:<38} {p:5.1%}")
        print()

    @torch.no_grad()
    def plan(self, policy_pick: str) -> None:
        """Turn search: print the best line of plays for the rest of the turn.
        `policy_pick` is the policy's own first play; it keeps the top line
        unless a plan beats it by `PLAN_MARGIN`."""
        L, n = self.layout, self.search
        forks = self.sim.fork(n, self.groups, seed=int(time.time_ns() % (1 << 31)))
        floats = np.zeros((n, L.n_floats), np.float32)
        ids = np.zeros((n, L.n_ids), np.int64)
        mask = np.zeros((n, L.n_actions), np.bool_)
        rewards = np.zeros(n, np.float32)
        forks.observe(floats, ids, mask)
        legal = np.flatnonzero(mask[0])
        first = legal[np.arange(n) % len(legal)]
        score = np.zeros(n, np.float32)
        lines: list[list[str]] = [[] for _ in range(n)]
        for step in range(MAX_PLAN_STEPS):
            over = np.array(forks.turn_over())
            if over.all():
                break
            logits, value = self.policy(torch.from_numpy(floats).to(self.device), torch.from_numpy(ids).to(self.device))
            masked = masked_logits(logits, torch.from_numpy(mask).to(self.device))
            actions = first if step == 0 else torch.distributions.Categorical(logits=masked).sample().cpu().numpy()
            for i in np.flatnonzero(~over):
                if (what := forks.describe(i, int(actions[i]))) is not None:
                    lines[i].append(what)
            forks.step(np.ascontiguousarray(actions, dtype=np.int64), floats, ids, mask, rewards)
            score += rewards
        # Plans that ended the turn are scored on where the next turn starts;
        # a finished fight already paid its terminal reward.
        _, value = self.policy(torch.from_numpy(floats).to(self.device), torch.from_numpy(ids).to(self.device))
        score += np.where(forks.is_over(), 0.0, value.cpu().numpy())
        # Best fork per first play, grouped by what the play is: two Strikes
        # in hand are one opening to you.
        by_first: dict[str, int] = {}
        for i in range(n):
            if not lines[i]:
                continue
            j = by_first.get(lines[i][0])
            if j is None or score[i] > score[j]:
                by_first[lines[i][0]] = i
        ranked = sorted(by_first.values(), key=lambda i: -score[i])
        if (own := by_first.get(policy_pick)) is not None and score[ranked[0]] - score[own] < PLAN_MARGIN:
            ranked.remove(own)
            ranked.insert(0, own)
        for rank, i in enumerate(ranked[: 1 + ALTERNATIVES]):
            arrow = "=>" if rank == 0 else "  "
            print(f"  {arrow} plan {score[i]:+.2f}: " + " | ".join(lines[i]))


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
    ap.add_argument("--search", type=int, default=0, help="turn search with this many sim copies (0: policy only)")
    ap.add_argument("--groups", type=int, default=4, help="draw-pile shuffles the search copies are split over")
    args = ap.parse_args()
    # Advice is worth nothing late: keep it flowing when stdout is a pipe.
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()

    session = Session(policy, device, args.search, args.groups)
    lines = replay_lines(args.replay, args.delay) if args.replay else live_lines(args.dir, args.poll)
    for line in lines:
        if line is None:
            session.flush()
        else:
            session.feed(line)


if __name__ == "__main__":
    main()
