#!/usr/bin/env bash
# Rebuilds the game mods under mods/ and copies the WebAssembly into scenes/.
# The built files are committed, so tests and players do not need a wasm
# toolchain. Run this after changing a mod and commit the result.
#
# Needs the wasm target: rustup target add wasm32-unknown-unknown
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

build() {
  local name="$1" out="$2"
  (cd "$root/mods/$name" && cargo build --release --quiet)
  cp "$root/mods/$name/target/wasm32-unknown-unknown/release/$name.wasm" "$root/$out"
  echo "built $out ($(wc -c < "$root/$out") bytes)"
}

build siege scenes/games/siege/siege.wasm
