"""Let a trained policy play combats in the running game.

    uv run python -m sts2ai.play runs/<run>/latest.pt --search 256

Connects to the mod's bridge (mod/Bridge.cs, 127.0.0.1:47474), follows the
fight the way the advisor does, and sends the advisor's pick back at every
decision point, card choices included. You play everything outside combat;
the next fight is picked up when it starts. Stop it with Ctrl-C to take a
fight over: with nobody connected the game is yours again, and a card
choice left open goes to a card grid for you.

When the sim loses track of a fight (a divergence, or an encounter it does
not model) the fight is handed to you: it says so once, and card choices
go to the grid.

`--record` is for unattended recording (scripts/record.py --pilot): it
samples its moves so repeated fights differ, and instead of handing a fight
over it ends it with the console's `win`. That happens on a divergence,
where the recording already holds what the replay can check, and in place
of ending a turn the sim says would kill you, Fairies in a Bottle
included, so a lost fight never ends the run but a fight runs as long as
it safely can.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Iterator

import torch

from sts2ai.advise import Session
from sts2ai.env import Layout
from sts2ai.model import Policy, load_policy

DEFAULT_PORT = 47474
# The mod runs dev console lines written here (scripts/game.sh does the same).
COMMANDS = Path.home() / ".local/share/SlayTheSpire2/sts2ai/commands.txt"


def console(*lines: str) -> None:
    tmp = COMMANDS.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(COMMANDS)


class Pilot(Session):
    """An advisor session that answers the bridge instead of waiting for you."""

    def __init__(
        self, conn: socket.socket, policy: Policy, device: torch.device, search: int = 0, groups: int = 4, record: bool = False
    ):
        super().__init__(policy, device, search, groups)
        self.conn = conn
        self.record = self.sample = record
        # This fight has been ended with `win`; nothing more to do in it.
        self.finished = False

        # What the game is waiting on: "play" after `ready`, "select" while
        # a card selection is open, None once a command is in flight.
        self.waiting: str | None = None
        self.sent_for: str | None = None
        # The open selection's `select` message: min, max, picked so far.
        self.selection: dict = {}
        # The sim cannot follow this fight; it is yours until the next one.
        self.blind = False
        self.told = False
        # Actions the game refused at this decision point.
        self.banned: set[int] = set()
        self.last: int | None = None

    def on_decision(self) -> None:
        pass  # decisions are taken when the bridge says the game waits

    def handle(self, status: str) -> None:
        if status.startswith(("diverged", "unsupported")):
            self.blind = True
            if self.record:
                self.finish("the sim lost track")
        super().handle(status)

    def finish(self, why: str) -> None:
        """Recording mode: end the fight with the console's `win`."""
        if not self.finished:
            self.finished = True
            print(f"  ending the fight: {why}\n")
            console("win")

    def message(self, line: str) -> None:
        msg = json.loads(line)
        match msg.get("t"):
            case "hello":
                print(f"connected to the game (bridge v{msg.get('version')})\n")
            case "ready":
                self.waiting, self.banned = "play", set()
                self.handle(self.sim.settle())
                self.act()
            case "select":
                if self.waiting != "select":
                    self.banned = set()
                self.waiting, self.selection = "select", msg
                self.act()
            case "error" if msg.get("wait"):
                print(f"  game busy: {msg.get('msg')}")
            case "error":
                print(f"  game refused: {msg.get('msg')}")
                if self.last is not None:
                    self.banned.add(self.last)
                self.waiting = self.sent_for
                self.act()
            case "start":
                self.blind = self.told = self.finished = False
                self.feed(line)
            case _:
                self.feed(line)

    def act(self) -> None:
        if self.finished:
            return
        enough = self.selection.get("picked", 0) >= self.selection.get("min", 1)
        match self.waiting:
            case None:
                return
            # The sim's choice closed but the game's allows more: enough.
            case "select" if not self.blind and not self.sim.choosing() and enough:
                self.send({"cmd": "done"})
                return
            case "select" if self.blind or not self.sim.choosing():
                print("  card choice is yours")
                self.send({"cmd": "manual"})
                return
            case "play" if self.blind or not self.sim.at_decision():
                if not self.told:
                    print("  your move: the sim cannot follow this fight\n")
                    self.told = True
                return
        index = self.advise(self.banned)
        command = None if index is None else self.sim.command(index)
        if command is None:
            if self.waiting == "select":
                self.send({"cmd": "manual"})
            else:
                print("  no legal action the game accepts; your move\n")
            return
        cmd = json.loads(command)
        if self.record and cmd.get("cmd") == "end" and not self.sim.end_turn_survives():
            self.finish("the enemy turn would kill you")
            return
        self.last = index
        self.send(cmd)

    def send(self, command: dict) -> None:
        self.sent_for, self.waiting = self.waiting, None
        self.conn.sendall((json.dumps(command) + "\n").encode())


def lines(conn: socket.socket) -> Iterator[str]:
    reader = conn.makefile("r", encoding="utf-8")
    yield from (line.rstrip("\n") for line in reader)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--search", type=int, default=0, help="turn search with this many sim copies (0: policy only)")
    ap.add_argument("--groups", type=int, default=4, help="draw-pile shuffles the search copies are split over")
    ap.add_argument("--record", action="store_true", help="unattended recording: sample moves, end lost or diverged fights with win")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(Layout.load()).to(device)
    load_policy(args.checkpoint, policy, device)
    policy.eval()

    try:
        conn = socket.create_connection(("127.0.0.1", args.port))
    except ConnectionRefusedError:
        sys.exit(f"nothing listening on port {args.port}: is the game running with the sts2ai mod? (scripts/build-mod.sh)")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    pilot = Pilot(conn, policy, device, args.search, args.groups, args.record)
    try:
        for line in lines(conn):
            if line:
                pilot.message(line)
    except KeyboardInterrupt:
        print("\nstopped; the game is yours")
    finally:
        conn.close()
    print("connection closed")


if __name__ == "__main__":
    main()
