"""Drive the game window directly, for what the dev console cannot do:
the main menu's Continue, pickup screens, anything with no command.

    uv run python scripts/game.py shot [FILE]      # screenshot of the game window only
    uv run python scripts/game.py click FX FY      # left click at a fraction of the window (0.5 0.5: centre)
    uv run python scripts/game.py key KEY [KEY..]  # keys to the game window (xdotool names: Return, Down, Escape)
    uv run python scripts/game.py launch           # start the game, wait for its window
    uv run python scripts/game.py continue         # launch if needed, click Continue, wait for the run
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
RUN_STATE = Path.home() / ".local/share/SlayTheSpire2/sts2ai/run.json"
# Main menu Continue, as a fraction of the window.
CONTINUE = (0.406, 0.633)


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


def continue_run(timeout: float = 90) -> None:
    """From a closed game or its main menu to the saved run, loaded."""
    # A closed game leaves its last run.json behind: after a fresh launch,
    # only a file the new process wrote counts.
    fresh, start = window() is None, time.time()
    launch()
    end = time.time() + timeout
    while not (run_active() and (not fresh or RUN_STATE.stat().st_mtime > start)):
        if time.time() > end:
            raise SystemExit("the run did not load; look at `shot`")
        # The menu takes a while to accept input after the window opens.
        click(*CONTINUE)
        time.sleep(5)


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
            continue_run()
            print("run loaded")
        case ["launch"]:
            print(launch())
        case ["where"]:
            print(window())
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
