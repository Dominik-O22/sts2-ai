#!/usr/bin/env bash
# Run-policy A/B with the standard settings: both arms side by side on the
# same 2,048 seeds, fights played by $COMBAT (gen8) with the turn search in
# elites and bosses, then sts2ai.paired on the two runs files.
#
#   scripts/ab.sh NAME "RUNPLAY ARGS A" "RUNPLAY ARGS B"
#   scripts/ab.sh events "--run-policy runs/imitate-noforecast/imitated.pt" \
#       "--run-policy runs/imitate-noforecast/imitated.pt --event-table ROWS"
#
# Output lands in runs/ab/NAME/. It warns first about other processes
# holding a core, since a stray one halves an eval's speed unnoticed.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 a=$2 b=$3
root=/home/doop/Work/tries/2026-09-21-sts2-ai
combat=${COMBAT:-$root/runs/gen8/latest.pt}
out=$root/runs/ab/$name
mkdir -p "$out"
busy=$(ps -eo pcpu=,etime=,args= --sort=-pcpu | awk '$1 > 50' | cut -c1-150)
[ -n "$busy" ] && printf 'already busy (%%CPU, elapsed, command):\n%s\n' "$busy"
std=(--envs 256 --runs-per-env 8 --search 256 --search-kinds Elite,Boss --minutes 600)
# shellcheck disable=SC2086
uv run --no-sync python -u -m sts2ai.runplay "$combat" "${std[@]}" $a --runs-out "$out/a.jsonl" > "$out/a.log" 2>&1 &
# shellcheck disable=SC2086
uv run --no-sync python -u -m sts2ai.runplay "$combat" "${std[@]}" $b --runs-out "$out/b.jsonl" > "$out/b.log" 2>&1 &
wait
uv run --no-sync python -m sts2ai.paired "$out/a.jsonl" "$out/b.jsonl" | tee "$out/paired.txt"
