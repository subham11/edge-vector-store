#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Build WASM target with Emscripten
#  Prerequisites: emsdk installed and activated
#  Usage: ./scripts/build_wasm.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build/wasm"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake "$ROOT_DIR" \
    -DEVS_WASM=ON \
    -DCMAKE_BUILD_TYPE=Release

emmake make -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo ""
echo "✓ WASM build complete:"
ls -lh evs_wasm.{js,wasm} 2>/dev/null || true
