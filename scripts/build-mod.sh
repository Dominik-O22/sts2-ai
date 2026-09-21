#!/usr/bin/env bash
# Build the recorder mod against the game's assemblies and install it into
# the game's mods/ directory. Requires the user-local dotnet 9 SDK.
set -euo pipefail
cd "$(dirname "$0")/../mod"
GAME="${STS2GameDir:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}"
export PATH="$HOME/.dotnet:$PATH"
dotnet build -c Release -nologo -v q -p:STS2GameDir="$GAME"
DEST="$GAME/mods/sts2ai"
mkdir -p "$DEST"
cp bin/Release/net9.0/sts2ai.dll sts2ai.json "$DEST/"
echo "installed to $DEST"
