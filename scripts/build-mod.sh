#!/usr/bin/env bash
# Build the recorder mod against the game's assemblies and install it into
# the game's mods/ directory. Requires the user-local dotnet 9 SDK.
set -euo pipefail
cd "$(dirname "$0")/../mod"
GAME="${STS2GameDir:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}"
export PATH="$HOME/.dotnet:$PATH"
dotnet build -c Release -nologo -v q -p:STS2GameDir="$GAME"
# The running game keeps reading the DLL it loaded: overwriting it breaks
# every poll of the live mod until a restart.
if pgrep -f "$GAME/SlayTheSpire2" > /dev/null; then
    echo "built; not installed: the game is running (close it, then run this again)" >&2
    exit 1
fi
DEST="$GAME/mods/sts2ai"
mkdir -p "$DEST"
cp bin/Release/net9.0/sts2ai.dll sts2ai.json "$DEST/"
echo "installed to $DEST"
