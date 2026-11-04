#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/target/qwen-image-metal"
SRC="$OUT/stable-diffusion.cpp"
PIN=68285df319457e2d79310b48513e4ba1bf8ebe9a
mkdir -p "$OUT"
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --no-checkout https://github.com/leejet/stable-diffusion.cpp.git "$SRC"
fi
if ! git -C "$SRC" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch --depth 1 origin "$PIN"
fi
git -C "$SRC" checkout --detach "$PIN"
git -C "$SRC" submodule update --init --depth 1 ggml
cmake -S "$ROOT/tools/qwen-image-metal" -B "$OUT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DSD_SOURCE_DIR="$SRC"
cmake --build "$OUT/build" --target gen2_qwen_probe -j "${GEN2_QWEN_BUILD_JOBS:-8}"

# Fail the build if a future dependency change leaks its GGML/SD symbols.
nm -gUj "$OUT/build/bin/libgen2_qwen_probe.dylib" | sort > "$OUT/exported-symbols.txt"
diff -u <(sed '/^$/d' "$ROOT/tools/qwen-image-metal/exports.txt" | sort) "$OUT/exported-symbols.txt"
git -C "$SRC" rev-parse HEAD > "$OUT/sd-commit.txt"
git -C "$SRC" submodule status --recursive > "$OUT/sd-submodules.txt"
printf 'Bridge: %s\n' "$OUT/build/bin/libgen2_qwen_probe.dylib"
