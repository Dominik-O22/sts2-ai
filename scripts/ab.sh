#!/usr/bin/env bash
# Run-policy A/B with the standard settings: both arms on the same 2,048
# seeds, fights played by $COMBAT (gen8) with the turn search in elites and
# bosses, then sts2ai.paired on the two runs files.
#
#   scripts/ab.sh NAME "RUNPLAY ARGS A" "RUNPLAY ARGS B"
#   scripts/ab.sh events "--run-policy runs/imitate-noforecast/imitated.pt" \
#       "--run-policy runs/imitate-noforecast/imitated.pt --event-table ROWS"
#
# Output lands in runs/ab/NAME/. Each arm runs in its own systemd scope
# capped at $ARM_GB (default 9) GB: an arm that outgrows it is killed
# alone, where before systemd-oomd killed the whole terminal scope, the
# session with it. A plain arm peaks near 4.3 GB, an --afterstate one near
# 7.8 GB. The arms run side by side when the memory available holds both,
# else one after the other. To outlive the session, start the script
# itself as a unit: systemd-run --user --unit=ab-NAME --same-dir scripts/ab.sh ...
# ENVS and RUNS_PER_ENV shrink it for a smoke test. It warns first about
# other processes holding a core, since a stray one halves an eval's speed
# unnoticed.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 a=$2 b=$3
root=/home/doop/Work/tries/2026-09-21-sts2-ai
combat=${COMBAT:-$root/runs/gen8/latest.pt}
arm_gb=${ARM_GB:-9}
out=$root/runs/ab/$name
mkdir -p "$out"
busy=$(ps -eo pcpu=,etime=,args= --sort=-pcpu | awk '$1 > 50' | cut -c1-150)
[ -n "$busy" ] && printf 'already busy (%%CPU, elapsed, command):\n%s\n' "$busy"
std=(--envs "${ENVS:-256}" --runs-per-env "${RUNS_PER_ENV:-8}" --search 256 --search-kinds Elite,Boss --minutes 600)

arm() { # arm LETTER "ARGS"
    # shellcheck disable=SC2086
    systemd-run --user --scope --quiet --unit="ab-$name-$1-$$" -p MemoryMax="${arm_gb}G" -p MemorySwapMax=0 \
        uv run --no-sync python -u -m sts2ai.runplay "$combat" "${std[@]}" $2 --runs-out "$out/$1.jsonl" > "$out/$1.log" 2>&1
}

available_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if [ "$available_gb" -ge $((2 * arm_gb)) ]; then
    arm a "$a" & arm b "$b" & wait
else
    echo "${available_gb} GB available, under 2 x ${arm_gb} GB: arms run one after the other"
    arm a "$a"
    arm b "$b"
fi
uv run --no-sync python -m sts2ai.paired "$out/a.jsonl" "$out/b.jsonl" | tee "$out/paired.txt"
