"""Advice for the combat you are playing, from a trained policy.

    uv run python -m sts2ai.advise runs/<run>/latest.pt
    uv run python -m sts2ai.advise runs/<run>/latest.pt --search 256
    uv run python -m sts2ai.advise runs/<run>/latest.pt --replay FILE --delay 0.2

Follows the newest file the recorder mod writes, keeps a sim combat in
sync with the game the way the replay harness does (docs/replay.md), and
prints what the policy would do at every decision point. You play the
moves yourself; `sts2ai.play` is the same session sending its pick back.

With `--search N`, each decision also runs a turn search: N copies of the
sim play the rest of the turn under the policy, every legal first action
gets its share of copies, and the copies are scored by the shaped reward
they collect plus the value head at the end of the turn. Copies that see
the same thing after their first play also try every second play, and an
opening is ranked by its best second play (`search.openings`), since the
next decision is searched again rather than played as the policy would.
That needs a few copies per second play: 2048 is a good N. The best plan
is printed as the whole line of plays. Draw piles are reshuffled per group of
copies, since the plan must not know what you will draw.

That is `--search-mode copies`. The default, `hybrid`, searches with
`exactsearch.Hybrid` instead, and N only turns it on: every distinct line
to the end of the turn, the best `--top` of them each played out
`--playouts` times to the fight's end on shared dice, and the line picked
is followed until chance or the turn's end takes the fight off it.
`race` plays the lines out `--race` at a time and drops the ones that
fall behind.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from sts2ai import _sim
from sts2ai.env import DEFAULT_RECORDINGS, Envs, Layout
from sts2ai.exactsearch import Hybrid
from sts2ai.model import Policy, load_policy, masked_logits
from sts2ai.search import PLAN_MARGIN, openings, rollout, spread

ALTERNATIVES = 2


class Session:
    """One recording followed line by line: the sim kept in sync, and the
    policy's pick printed once per decision point."""

    def __init__(self, policy: Policy, device: torch.device, search: int = 0, groups: int = 4, hybrid: Hybrid | None = None):
        self.policy = policy
        self.device = device
        self.search = search
        self.groups = groups
        self.hybrid = hybrid
        # Pick by sampling the policy instead of taking its favourite, so
        # repeated recordings of one fight take different branches.
        self.sample = False
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
            self.on_decision()

    def on_decision(self) -> None:
        """The sim reached a decision point: advise, once per point."""
        key = self.sim.counts()
        if key != self.advised:
            self.advised = key
            self.advise()

    @torch.no_grad()
    def advise(self, banned: set[int] = frozenset()) -> int | None:
        """Print the advice for the current decision point and return the
        action index it recommends, None if nothing is legal. `banned`
        actions are masked out: the game refused them."""
        self.sim.observe(self.floats, self.ids, self.mask)
        self.mask[0, list(banned)] = False
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
        if not ranked:
            return None
        # With search on, the plan is the advice; the policy's own pick only
        # breaks ties inside it.
        if (hybrid := self.hybrid) is not None:
            pick = self.hybrid_pick(hybrid, probs, banned)
        elif self.search:
            pick = self.plan(ranked[0][0], banned)
        else:
            for rank, (action, p) in enumerate(ranked[: 1 + ALTERNATIVES]):
                arrow = "->" if rank == 0 else "  "
                print(f"  {arrow} {action:<38} {p:5.1%}")
            best = ranked[0][0]
            pick = max(
                (int(i) for i in np.flatnonzero(self.mask[0]) if self.sim.describe(int(i)) == best),
                key=lambda i: probs[i],
            )
            if self.sample:
                legal = np.flatnonzero(self.mask[0])
                pick = int(np.random.choice(legal, p=probs[legal] / probs[legal].sum()))
        print()
        return pick

    def hybrid_pick(self, hybrid: Hybrid, probs: np.ndarray, banned: set[int]) -> int:
        """The hybrid search's pick for this decision, the fight searched as
        the hybrid's one env. `probs` is the policy's over the legal actions
        the game has not refused; the policy's favourite stands when it is
        the only one, or when the line picked opens with a refused action."""
        own = int(probs.argmax())
        if np.count_nonzero(self.mask[0]) == 1:
            print(f"  => {self.sim.describe(own)}")
            return own
        hybrid.envs.sim.load_combat(0, self.sim)
        actions = np.array([own])
        followed, t0 = hybrid.followed, time.perf_counter()
        hybrid.choose(self.policy, self.device, [0], probs[None], actions)
        # Search stats are for exactsearch's report; a session keeps none.
        hybrid.planner.searched.clear()
        ms = (time.perf_counter() - t0) * 1000
        pick = own if int(actions[0]) in banned else int(actions[0])
        how = "following the line" if hybrid.followed > followed else f"searched in {ms:.0f} ms"
        note = "" if pick == own else f", over the policy's {self.sim.describe(own)}"
        print(f"  => {self.sim.describe(pick)}  ({how}{note})")
        return pick

    @torch.no_grad()
    def plan(self, policy_pick: str, banned: set[int] = frozenset()) -> int:
        """Turn search: print the best line of plays for the rest of the turn
        and return its first action. `policy_pick` is the policy's own first
        play; it keeps the top line unless a plan beats it by `PLAN_MARGIN`."""
        n = self.search
        forks = self.sim.fork(n, self.groups, seed=int(time.time_ns() % (1 << 31)))
        # `self.mask` is this decision's, with the game's refusals out.
        first = spread(np.flatnonzero(self.mask[0]), n)
        lines: list[list[str]] = [[] for _ in range(n)]

        def record(actions: np.ndarray, live: np.ndarray) -> None:
            for i in live:
                if (what := forks.describe(i, int(actions[i]))) is not None:
                    lines[i].append(what)

        second = (np.full(n, -1), np.zeros(n, np.int64))
        score = rollout(self.policy, self.device, forks, first, record, second=second)
        # Openings ranked by their best line (`search.openings`: the best
        # second play for what each opening led to, averaged over chance),
        # grouped by what the play is: two Strikes in hand are one opening
        # to you. Each prints its best copy along its best second play.
        value = openings(first, score, second)
        copies: dict[str, list[int]] = {}
        for i in range(n):
            if lines[i]:
                copies.setdefault(lines[i][0], []).append(i)
        mean = {what: max(value[int(first[i])][0] for i in idx) for what, idx in copies.items()}
        ranked = sorted(copies, key=lambda what: -mean[what])
        if policy_pick in mean and mean[ranked[0]] - mean[policy_pick] < PLAN_MARGIN:
            ranked.remove(policy_pick)
            ranked.insert(0, policy_pick)
        for rank, what in enumerate(ranked[: 1 + ALTERNATIVES]):
            arrow = "=>" if rank == 0 else "  "
            follow = value[int(first[copies[what][0]])][1]
            along = [i for i in copies[what] if follow is None or second[1][i] == follow]
            best = max(along or copies[what], key=lambda i: score[i])
            print(f"  {arrow} plan {mean[what]:+.2f}: " + " | ".join(lines[best]))
        return int(first[copies[ranked[0]][0]])


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


def add_search_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--search", type=int, default=0, help="turn search at every decision (0: policy only); copies mode: this many sim copies")
    ap.add_argument("--search-mode", choices=("copies", "hybrid", "race"), default="hybrid", help="copies: --search N copies; hybrid, race: exactsearch.Hybrid")
    ap.add_argument("--groups", type=int, default=4, help="draw-pile shuffles the search copies are split over")
    ap.add_argument("--top", type=int, default=10, help="lines the hybrid plays out")
    ap.add_argument("--playouts", type=int, default=32, help="playouts per line in the hybrid")
    ap.add_argument("--race", type=int, default=8, help="playouts per round with --search-mode race")
    ap.add_argument("--seed", type=int, default=0, help="the hybrid's dice")


def hybrid_of(args: argparse.Namespace) -> Hybrid | None:
    """The hybrid `add_search_args` asked for, None for no search or copies."""
    if not args.search or args.search_mode == "copies":
        return None
    return Hybrid(Envs(1), args.top, args.playouts, args.race if args.search_mode == "race" else 0, args.seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dir", type=Path, default=DEFAULT_RECORDINGS, help="recordings folder to follow")
    ap.add_argument("--replay", type=Path, help="feed this recording instead of following the game")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between records when replaying")
    ap.add_argument("--poll", type=float, default=0.1, help="seconds between checks of the recordings folder")
    add_search_args(ap)
    args = ap.parse_args()
    # Advice is worth nothing late: keep it flowing when stdout is a pipe.
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    policy.eval()

    session = Session(policy, device, args.search, args.groups, hybrid_of(args))
    lines = replay_lines(args.replay, args.delay) if args.replay else live_lines(args.dir, args.poll)
    for line in lines:
        if line is None:
            session.flush()
        else:
            session.feed(line)


if __name__ == "__main__":
    main()
