#!/usr/bin/env bash
# Decompile the installed game's main assembly into ./decompiled (gitignored).
# Requires the user-local dotnet SDK and ilspycmd (see DESIGN.md, Game version).
set -euo pipefail
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$HOME/.dotnet/tools:$PATH"

GAME_DIR="${STS2_DIR:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}"
BIN="$GAME_DIR/data_sts2_linuxbsd_x86_64"
OUT="$(dirname "$0")/../decompiled"

echo "game version: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$GAME_DIR/release_info.json")"
rm -rf "$OUT" && mkdir -p "$OUT"
ilspycmd "$BIN/sts2.dll" --project --outputdir "$OUT" --referencepath "$BIN" --nested-directories
cp "$GAME_DIR/release_info.json" "$OUT/"
echo "written to $OUT"
