#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/target/stable-diffusion"
SRC="${GEN2_SD_SOURCE_DIR:-$ROOT/target/qwen-image-metal/stable-diffusion.cpp}"
PIN=68285df319457e2d79310b48513e4ba1bf8ebe9a
if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 --no-checkout https://github.com/leejet/stable-diffusion.cpp.git "$SRC"
fi
if ! git -C "$SRC" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch --depth 1 origin "$PIN"
fi
git -C "$SRC" checkout --detach "$PIN"
git -C "$SRC" submodule update --init --depth 1 ggml
cmake -S "$ROOT/native/stable-diffusion" -B "$OUT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DSD_SOURCE_DIR="$SRC"
cmake --build "$OUT/build" --target gen2_image -j "${GEN2_IMAGE_BUILD_JOBS:-8}"
nm -gUj "$OUT/build/bin/libgen2_image.dylib" | sort > "$OUT/exported-symbols.txt"
diff -u <(sed '/^$/d' "$ROOT/native/stable-diffusion/exports.txt" | sort) "$OUT/exported-symbols.txt"
printf 'Image bridge: %s\n' "$OUT/build/bin/libgen2_image.dylib"
