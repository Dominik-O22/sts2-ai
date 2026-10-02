#!/usr/bin/env bash
# Build the Python extension with profile-guided optimization: an
# instrumented sim runs real fights (playouts, search nodes, state keys),
# and the extension is rebuilt with the branch profile it gathered. On
# 2026-10-03 that made a search node 8% and a card play about 20% faster
# than the plain release build (target-cpu=native gained nothing). Rerun it
# after sim changes; a plain `uv sync --reinstall-package sts2ai` goes back
# to the normal build. Needs llvm-profdata from the same LLVM as rustc.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
work=${PGO_DIR:-$root/target/pgo}
fights="$root/runs/act3boss/holdout.jsonl"
[ -f "$fights" ] || fights="$HOME/.local/share/SlayTheSpire2/sts2ai/tracker/setups/holdout.jsonl"
rm -rf "$work/data"
mkdir -p "$work/data"
(cd sim && RUSTFLAGS="-Cprofile-generate=$work/data" CARGO_TARGET_DIR="$work/gen" \
    cargo build --release --example bench_real --example solvecost --example handcheck)
for _ in 1 2; do
    "$work/gen/release/examples/bench_real" "$fights" > /dev/null
    "$work/gen/release/examples/solvecost" "$fights" > /dev/null
done
"$work/gen/release/examples/handcheck" 2 "$fights" > /dev/null
llvm-profdata merge -o "$work/merged.profdata" "$work/data"
RUSTFLAGS="-Cprofile-use=$work/merged.profdata" uv sync --reinstall-package sts2ai
echo "extension built with $work/merged.profdata"
