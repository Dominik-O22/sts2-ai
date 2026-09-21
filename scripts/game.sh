#!/usr/bin/env bash
# Send dev console commands to the running game through the sts2ai mod.
#   scripts/game.sh "fight NIBBITS_NORMAL" "relic add VAJRA"
# Results appear in ~/.local/share/SlayTheSpire2/sts2ai/commands.log.
set -euo pipefail
DIR="$HOME/.local/share/SlayTheSpire2/sts2ai"
mkdir -p "$DIR"
tmp="$DIR/commands.tmp"
printf '%s\n' "$@" > "$tmp"
mv "$tmp" "$DIR/commands.txt"
sleep 1
tail -n "$#" "$DIR/commands.log" 2>/dev/null || true
