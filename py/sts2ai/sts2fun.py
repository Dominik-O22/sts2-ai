"""Runs of strong players from sts2.fun, wins and losses alike, for fights
the winners-only ststracker data cannot hold: the ones experts lose HP in,
and the ones they die in.

    uv run python -m sts2ai.sts2fun crawl        # players and their A10 Ironclad runs -> DIR
    uv run python -m sts2ai.sts2fun setups       # saved runs -> DIR/setups/*.jsonl

sts2.fun keeps every run a player uploads. The crawl takes the players of
the site's A10 leaderboard, keeps those who win at least `--min-wr` of
their A10 Ironclad runs over at least `--min-runs` of them, and saves every
such run's page, one request per PACE seconds. Runs fetched before are
skipped, so stopping and running it again resumes.

A run page lists per floor the encounter, turns, damage, HP and gold, the
cards and relics taken and the rest-site choices, then the relics with the
floor each joined and the final deck. It leaves out the patch, the cards
offered but not taken, card removals and bought or transformed cards,
potion use, and enchantments. So the deck at a floor is rebuilt forward
from the starter deck and reconciled with the final one (`deck_at`), and
fights carry no potions. Each line says how far its run's deck is from
exact (`unexplained`).
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from sts2ai.setups import TRACKER, holdout_player, sim_floor

SITE = "https://sts2.fun"
DIR = TRACKER.parent / "sts2fun"
# Seconds between requests to the site, and to wait when it pushes back.
PACE = 2.0
BACKOFF = 60.0
_last_request = 0.0


def fetch(path: str) -> str:
    """A page of the site, at most one request per `PACE` seconds, waiting
    out a 429 or a server error."""
    global _last_request
    for attempt in range(4):
        time.sleep(max(0.0, _last_request + PACE - time.monotonic()))
        _last_request = time.monotonic()
        request = urllib.request.Request(SITE + path, headers={"User-Agent": "sts2ai-crawler/1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == 3:
                raise
            print(f"  {e.code} from the site, waiting {BACKOFF * (attempt + 1):.0f} s")
            time.sleep(BACKOFF * (attempt + 1))
    raise AssertionError("unreachable")


def text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def leaderboard(page: str) -> list[tuple[str, int, int]]:
    """(player, runs, wins) of the home page's A10+ leaderboard."""
    start = page.index('id="lb-a10"')
    table = page[start : page.index("</table>", start)]
    rows = re.findall(r'href="/player/([^"]+)".*?<td data-sort-value="(\d+)">\d+</td><td data-sort-value="(\d+)">', table, flags=re.S)
    return [(p, int(runs), int(wins)) for p, runs, wins in rows]


def history(page: str) -> list[dict]:
    """A player page's run history, newest first."""
    rows = re.findall(
        r'<tr class="[^"]*">\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*><a href="/run/([0-9a-f]+)"[^>]*>(\w+)</a></td>\s*<td>(\d+)</td>\s*<td class="(win|loss)"',
        page,
    )
    return [
        {"date": datetime.strptime(d.strip(), "%b %d, %Y").date().isoformat(), "run": run, "character": c.upper(), "ascension": int(a), "win": r == "win"}
        for d, run, c, a, r in rows
    ]


def ironclad_a10(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["character"] == "IRONCLAD" and r["ascension"] == 10]


def slug(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def encounter_id(name: str) -> str:
    """"Snapping Jaxfruit (Normal)" -> SNAPPING_JAXFRUIT_NORMAL, the game's
    encounter id."""
    return slug(name)


def parse_run(page: str) -> dict:
    """A run page as floors, relics and the final deck."""
    head = text(page[page.index("<title>") : page.index("</title>")])
    table = page[page.index('class="run-floors"') :]
    table = table[: table.index("</table>")]
    names: dict[str, str] = {}  # card name as shown -> id
    floors = []
    for row in re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        floor, act, kind, encounter, turns, damage, hp, gold = (text(c) for c in cells[:8])
        rewards = cells[8]
        cards = []
        for cid, up, name in re.findall(r'data-card-id="([^"]+)" data-upgraded="(\d)"[^>]*>(.*?)</span>', rewards):
            cards.append([cid, up == "1"])
            names[text(name).rstrip("+").strip()] = cid
        hp_now, max_hp = re.match(r"(-?\d+) ?/(\d+)", hp).groups()
        floors.append(
            {
                "floor": int(floor),
                "act": int(act),
                "type": kind,
                "encounter": encounter,
                "turns": int(turns) if turns else 0,
                "damage": int(damage) if damage else 0,
                "hp": int(hp_now),
                "max_hp": int(max_hp),
                "gold": int(gold.split()[0]),
                "cards": cards,
                "relics": re.findall(r'data-relic-id="([^"]+)"', rewards),
                "rest": [text(m) for m in re.findall(r'<span style="color: var\(--win\);">(.*?)</span>', rewards)],
            }
        )
    relic_part = page[page.index("<h2>Relics (") :]
    relics = [[r, int(f)] for r, f in re.findall(r'data-relic-id="([^"]+)">.*?>F(\d+)</span>', relic_part[: relic_part.index("Final Deck")], flags=re.S)]
    deck_part = page[page.index("Final Deck") :]
    deck = []
    for cid, up, name in re.findall(r'data-card-id="([^"]+)" data-upgraded="(\d)"[^>]*>\s*([^<]*)', deck_part):
        deck.append([cid, up == "1"])
        names[name.strip()] = cid
    return {"win": " WIN " in f" {text(page[: page.index('Floor by Floor')])} ", "title": head, "floors": floors, "relics": relics, "deck": deck, "names": names}


STARTER = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH", "ASCENDERS_BANE"]


def deck_at(run: dict, floor: int) -> tuple[list[dict], int]:
    """The deck as the fight on `floor` began, and how many of the run's
    card changes the page does not show.

    Forward from the starter deck (with Ascender's Bane, A10), adding the
    cards each floor lists and applying its smiths. The final deck then
    tells what the page leaves out: cards it holds beyond those joined
    (bought, transformed into, given by an event) and cards it lacks
    (removed, transformed away). Those changes all go at the run's first
    shop, where most of them happen, or before the first fight when it
    has none. Upgrades the page does not place are left off."""
    joined = [(c, False, 1) for c in STARTER]  # (id, upgraded, floor it joined)
    for f in run["floors"]:
        joined += [(c, up, f["floor"]) for c, up in f["cards"]]
    final = Counter(c for c, _ in run["deck"])
    listed = Counter(c for c, _, _ in joined)
    unseen, gone = final - listed, listed - final
    shop = next((f["floor"] for f in run["floors"] if f["type"] == "shop"), 1)
    held = []
    for c, up, at in joined:
        if at >= floor:
            continue
        if gone[c] and floor > shop:
            gone[c] -= 1
            continue
        held.append([c, up])
    if floor > shop:
        held += [[c, False] for c in unseen.elements()]
    for f in run["floors"]:
        if f["floor"] >= floor:
            break
        for choice in f["rest"]:
            if choice.startswith("Smith: "):
                name = choice[len("Smith: ") :]
                cid = run["names"].get(name, slug(name))
                card = next((h for h in held if h[0] == cid and not h[1]), None)
                if card:
                    card[1] = True
    unexplained = sum(unseen.values()) + sum((listed - final).values())
    return [{"id": c, "up": up} for c, up in held], unexplained


def fights(run: dict, kinds: tuple[str, ...]) -> list[dict]:
    """A line per fight of the run whose encounter ends in one of `kinds`
    (`_ELITE`, `_BOSS`, `_WEAK`, `_NORMAL`), in `sts2ai.setups`' form."""
    out = []
    floors = run["floors"]
    for before, f in zip(floors, floors[1:]):
        encounter = encounter_id(f["encounter"]) if f["encounter"].endswith(")") else ""
        if not encounter.endswith(kinds):
            continue
        relics = [r for r, at in run["relics"] if at < f["floor"]]
        slots = 4 if "POTION_BELT" in relics else 2
        deck, unexplained = deck_at(run, f["floor"])
        boss = encounter.endswith("_BOSS")
        out.append(
            {
                "start": {"ascension": 10, "deck": deck, "relics": relics, "potions": [None] * slots, "gold": before["gold"]},
                "hp": before["hp"],
                "max_hp": before["max_hp"],
                "encounter": encounter,
                "floor": sim_floor(f["floor"], boss),
                "second": boss and f["floor"] == 49,
                "game_floor": f["floor"],
                # HP in minus HP out, after Burning Blood, as in `sts2ai.setups`.
                "winner_hp_lost": before["hp"] - f["hp"],
                "winner_max_hp_lost": before["max_hp"] - f["max_hp"],
                "turns": f["turns"],
                "died": f is floors[-1] and not run["win"],
                "unexplained": unexplained,
            }
        )
    return out


def runs_path(out: Path, run: str) -> Path:
    return out / "runs" / f"{run}.html.gz"


def crawl(out: Path, min_wr: float, min_runs: int, prefilter: float) -> None:
    """Players of the A10+ leaderboard who win at least `prefilter` of
    their A10 runs over all characters are looked at; those who win
    `min_wr` of at least `min_runs` A10 Ironclad runs are kept, with every
    such run. `out/players.json` lists them and their runs."""
    (out / "players").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    board = [(p, n, w) for p, n, w in leaderboard(fetch("/")) if w >= prefilter * n]
    print(f"{len(board)} players win {prefilter:.0%} of their A10 runs or more")
    index_path = out / "players.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for player, _, _ in board:
        if player not in index:
            runs = ironclad_a10(history(fetch(f"/player/{player}")))
            index[player] = {"runs": runs, "wr": sum(r["win"] for r in runs) / max(1, len(runs))}
            index_path.write_text(json.dumps(index))
    kept = {p: v for p, v in index.items() if len(v["runs"]) >= min_runs and v["wr"] >= min_wr}
    todo = [r["run"] for v in kept.values() for r in v["runs"] if not runs_path(out, r["run"]).exists()]
    total = sum(len(v["runs"]) for v in kept.values())
    print(f"{len(kept)} players win {min_wr:.0%} of {min_runs}+ A10 Ironclad runs: {total} runs, {len(todo)} to fetch (~{len(todo) * PACE / 60:.0f} min)")
    try:
        for i, run in enumerate(todo, 1):
            try:
                page = fetch(f"/run/{run}")
            except (urllib.error.URLError, OSError) as e:
                print(f"  {run}: failed: {e}")
                continue
            runs_path(out, run).write_bytes(gzip.compress(page.encode()))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    except KeyboardInterrupt:
        print("stopped; running it again picks up where it left off")


def setups(out: Path, min_wr: float, min_runs: int, holdout: float) -> None:
    """The saved runs' fights, split by player like `sts2ai.setups`, into
    `out/setups/{train,holdout,easy-train,easy-holdout}.jsonl`. Each line
    also carries the `player`, their A10 Ironclad win rate (`player_wr`),
    the run's `date`, whether the run was won (`run_won`) and whether
    the fight ended it (`died`). A run that meets an encounter the sim
    does not have (Doormaker, since removed) is from another version of
    the game and is left out whole."""
    from sts2ai import _sim
    from sts2ai.env import Envs

    probe = Envs(1, seed=0)
    known = {name for name, _, _ in _sim.encounters()}
    index = json.loads((out / "players.json").read_text())
    groups = [("", ("_ELITE", "_BOSS")), ("easy-", ("_WEAK", "_NORMAL"))]
    kept: dict[str, list[str]] = {f"{prefix}{split}": [] for prefix, _ in groups for split in ("train", "holdout")}
    dropped: Counter[str] = Counter()
    runs = 0
    for player, v in index.items():
        if len(v["runs"]) < min_runs or v["wr"] < min_wr:
            continue
        split = "holdout" if holdout_player(player, holdout) else "train"
        for meta in v["runs"]:
            path = runs_path(out, meta["run"])
            if not path.exists():
                continue
            try:
                run = parse_run(gzip.decompress(path.read_bytes()).decode())
            except (ValueError, AttributeError) as e:
                dropped[f"unparsed page: {e}"] += 1
                continue
            other = {encounter_id(f["encounter"]) for f in run["floors"] if f["encounter"].endswith(")")} - known
            if other:
                dropped[f"run from another version ({', '.join(sorted(other))})"] += 1
                continue
            runs += 1
            extra = {"run": meta["run"], "player": player, "player_wr": round(v["wr"], 3), "date": meta["date"], "run_won": run["win"]}
            for prefix, kinds in groups:
                for fight in fights(run, kinds):
                    line = json.dumps({**fight, **extra})
                    try:
                        probe.sim.use_setups(line, 1)
                    except ValueError as e:
                        dropped[str(e).split(": ", 1)[-1].split(":")[0]] += 1
                        continue
                    kept[f"{prefix}{split}"].append(line)
    (out / "setups").mkdir(exist_ok=True)
    for name, lines in kept.items():
        (out / "setups" / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{runs} runs: " + ", ".join(f"{len(lines)} {name}" for name, lines in kept.items()) + f" fights in {out / 'setups'}")
    print(f"{sum(dropped.values())} left out: " + ", ".join(f"{k} {v}" for k, v in dropped.most_common(8)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Strong players' A10 Ironclad runs from sts2.fun, wins and losses.")
    ap.add_argument("step", choices=["crawl", "setups"])
    ap.add_argument("--out", type=Path, default=DIR)
    ap.add_argument("--min-wr", type=float, default=0.5, help="lowest A10 Ironclad win rate of a player kept")
    ap.add_argument("--min-runs", type=int, default=15, help="fewest A10 Ironclad runs of a player kept")
    ap.add_argument("--prefilter", type=float, default=0.4, help="lowest A10 win rate over all characters of a player looked at")
    ap.add_argument("--holdout", type=float, default=0.15, help="share of players held out")
    args = ap.parse_args()
    if args.step == "crawl":
        crawl(args.out, args.min_wr, args.min_runs, args.prefilter)
    else:
        setups(args.out, args.min_wr, args.min_runs, args.holdout)


if __name__ == "__main__":
    main()
