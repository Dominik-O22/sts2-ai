"""Drive the game window directly, for what the dev console cannot do:
the main menu's Continue, pickup screens, anything with no command.

    uv run python scripts/game.py shot [FILE]      # screenshot of the game window only
    uv run python scripts/game.py click FX FY      # left click at a fraction of the window (0.5 0.5: centre)
    uv run python scripts/game.py key KEY [KEY..]  # keys to the game window (xdotool names: Return, Down, Escape)
    uv run python scripts/game.py launch           # start the game, wait for its window
    uv run python scripts/game.py continue         # launch if needed, continue the saved run, wait for it
    uv run python scripts/game.py new_run ASC [SEED] [ACT1]   # same, but a new Ironclad run
    uv run python scripts/game.py menu             # back to the main menu
    uv run python scripts/game.py where            # window position and size

Hyprland 0.56 only (its dispatchers are Lua): `hyprctl` finds the window,
moves the cursor and sends the button, `grim` crops the screenshot to the
window, so nothing else on screen ends up in the image. Clicks focus the
game window first: an unfocused window shows the hover but drops the
click. Positions are fractions of the window, so they survive a resize.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TITLE = "Slay the Spire 2"
APP_ID = 2868840
SHOT = Path("/tmp/sts2-window.png")
GAME_DIR = Path.home() / ".local/share/SlayTheSpire2/sts2ai"
RUN_STATE = GAME_DIR / "run.json"


@dataclass
class Window:
    address: str
    x: int
    y: int
    w: int
    h: int


def window() -> Window | None:
    """The game window, in layout coordinates, if it is open."""
    clients = json.loads(subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True).stdout)
    for c in clients:
        if c["title"] == TITLE:
            (x, y), (w, h) = c["at"], c["size"]
            return Window(c["address"], x, y, w, h)
    return None


def shot(path: Path = SHOT) -> Path:
    """Screenshot of the game window alone."""
    win = window()
    if win is None:
        raise SystemExit("the game window is not open")
    subprocess.run(["grim", "-g", f"{win.x},{win.y} {win.w}x{win.h}", str(path)], check=True)
    return path


def dispatch(lua: str) -> None:
    out = subprocess.run(["hyprctl", "dispatch", lua], capture_output=True, text=True, check=True).stdout.strip()
    if out != "ok":
        raise SystemExit(f"{lua}: {out}")


def click(fx: float, fy: float) -> None:
    """Left click at (fx, fy), fractions of the game window."""
    win = window()
    if win is None:
        raise SystemExit("the game window is not open")
    target = f'window = "address:{win.address}"'
    dispatch(f"hl.dsp.focus({{{target}}})")
    dispatch(f"hl.dsp.cursor.move({{x = {win.x + round(fx * win.w)}, y = {win.y + round(fy * win.h)}}})")
    time.sleep(0.3)
    for state in ("down", "up"):
        dispatch(f'hl.dsp.send_key_state({{mods = "", key = "mouse:272", state = "{state}", {target}}})')
        time.sleep(0.1)


def key(name: str, mods: str = "") -> None:
    """Press `name` in the game window without focusing it."""
    win = window()
    if win is None:
        raise SystemExit("the game window is not open")
    dispatch(f'hl.dsp.send_shortcut({{mods = "{mods}", key = "{name}", window = "address:{win.address}"}})')


def launch(timeout: float = 120) -> Window:
    """Start the game through Steam and wait for its window."""
    if (win := window()) is not None:
        return win
    subprocess.Popen(["setsid", "steam", f"steam://rungameid/{APP_ID}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + timeout
    while time.time() < end:
        if (win := window()) is not None:
            return win
        time.sleep(2)
    raise SystemExit("the game window did not appear")


def run_active() -> bool:
    try:
        return bool(json.loads(RUN_STATE.read_text()).get("active"))
    except (OSError, ValueError):
        return False


def console(line: str, timeout: float = 10) -> str | None:
    """Run one line through the mod's command file and return its outcome
    from commands.log, or None if the game did not pick it up in time."""
    log = GAME_DIR / "commands.log"
    seen = log.stat().st_size if log.exists() else 0
    tmp = GAME_DIR / "commands.tmp"
    tmp.write_text(line + "\n")
    tmp.rename(GAME_DIR / "commands.txt")
    end = time.time() + timeout
    while time.time() < end:
        if log.exists() and log.stat().st_size > seen:
            with log.open() as f:
                f.seek(seen)
                return f.read().strip()
        time.sleep(0.2)
    return None


def open_run(command: str, timeout: float = 120) -> None:
    """From a closed game or its main menu to a loaded run: the mod's
    `sts2ai continue` or `sts2ai new_run ...`, retried until the menu is up
    to take it."""
    # A closed game leaves its last run.json behind: after a fresh launch,
    # only a file the new process wrote counts.
    fresh, start = window() is None, time.time()
    launch()
    end = time.time() + timeout
    sent = False
    while not (run_active() and (not fresh or RUN_STATE.stat().st_mtime > start)):
        if time.time() > end:
            raise SystemExit(f"{command} did not load a run; look at `shot`")
        if not sent:
            out = console(f"sts2ai {command}")
            if out and "already loaded" in out:
                return
            # Before the main menu exists the command errors; try again.
            sent = out is not None and out.startswith("ok")
        time.sleep(2)


def main() -> None:
    match sys.argv[1:]:
        case ["shot", *rest]:
            print(shot(Path(rest[0]) if rest else SHOT))
        case ["key", *keys] if keys:
            for k in keys:
                key(k)
                time.sleep(0.3)
        case ["click", fx, fy]:
            click(float(fx), float(fy))
        case ["continue"]:
            open_run("continue")
            print("run loaded")
        case ["new_run", *rest] if 1 <= len(rest) <= 3:
            open_run("new_run " + " ".join(rest))
            print("run started")
        case ["menu"]:
            print(console("sts2ai menu"))
        case ["launch"]:
            print(launch())
        case ["where"]:
            print(window())
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
