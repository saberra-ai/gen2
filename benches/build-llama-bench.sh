#!/usr/bin/env bash
# Build `llama-bench` from the llama.cpp commit that gen2 actually links.
#
# gen2 pins `llama-cpp-sys-2` from crates.io. That crate ships a *subset* of
# llama.cpp (no tools/, no .git), so the reference cannot be built from the
# registry copy directly. This script resolves the upstream commit the crate
# was cut from, proves the registry copy is byte-identical to it, and builds
# `llama-bench` from that commit with the cmake options the crate's build.rs
# uses — so both sides of a benchmark run the same ggml.
#
#   benches/build-llama-bench.sh            # macOS: Metal; Linux: CPU
#   GEN2_BENCH_CUDA=1 benches/build-llama-bench.sh
#
# Outputs (all under target/llama-bench/):
#   llama-bench            the binary
#   build-info.txt         commit, build number, flags, source proof
#   src/                   llama.cpp checkout (blobless partial clone)
#   llama-cpp-rs/          shallow clone of the binding repo at the pinned tag
# and benches/results/llama-cpp-pin.txt, which the harness and the table
# generator read to assert the reference matches Cargo.toml.
#
# Needs: git, cmake, a C++ toolchain (Xcode on macOS), network on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/target/llama-bench"
SRC="$OUT/src"
RS="$OUT/llama-cpp-rs"
PIN="$ROOT/benches/results/llama-cpp-pin.txt"

log() { printf '%s\n' "build-llama-bench: $*" >&2; }
die() { log "error: $*"; exit 1; }

# 1. The pinned crate version, from Cargo.toml (the manifest is the source of
#    truth; Cargo.lock agrees or `cargo` would refuse to build).
VER="$(sed -n 's/^llama-cpp-sys-2 *= *{ *version *= *"=\{0,1\}\([0-9][0-9.]*\)".*/\1/p' "$ROOT/Cargo.toml" | head -1)"
[ -n "$VER" ] || die "could not read the llama-cpp-sys-2 version from Cargo.toml"
log "llama-cpp-sys-2 = $VER"

# 2. The registry copy of that crate.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
REG="$(ls -d "$CARGO_HOME_DIR"/registry/src/*/llama-cpp-sys-2-"$VER" 2>/dev/null | head -1 || true)"
[ -n "$REG" ] && [ -d "$REG/llama.cpp" ] || die "registry copy not found under $CARGO_HOME_DIR/registry/src; run \`cargo fetch\` first"
log "registry copy: $REG"

# 3. The upstream llama.cpp commit: the submodule pointer in the binding repo
#    at the tag matching the crate version. No API call — git ls-tree on a
#    shallow clone of the tag.
mkdir -p "$OUT"
if [ ! -d "$RS/.git" ]; then
    log "cloning utilityai/llama-cpp-rs at tag $VER"
    git clone --quiet --depth 1 --branch "$VER" --no-recurse-submodules \
        https://github.com/utilityai/llama-cpp-rs "$RS"
else
    if ! git -C "$RS" describe --tags --exact-match 2>/dev/null | grep -qx "$VER"; then
        log "re-fetching llama-cpp-rs tag $VER"
        git -C "$RS" fetch --quiet --depth 1 origin "refs/tags/$VER:refs/tags/$VER"
        git -C "$RS" checkout --quiet --detach "refs/tags/$VER"
    fi
fi
SHA="$(git -C "$RS" ls-tree HEAD llama-cpp-sys-2/llama.cpp | awk '$2 == "commit" { print $3 }')"
[ -n "$SHA" ] || die "llama-cpp-sys-2/llama.cpp is not a submodule at tag $VER"
log "llama.cpp submodule at tag $VER: $SHA"

# 4. llama.cpp at that commit. Blobless partial clone: full commit history
#    (so `git rev-list --count` gives the real build number) without the
#    blobs for every historical revision.
if [ ! -d "$SRC/.git" ]; then
    log "cloning ggml-org/llama.cpp (blobless)"
    git clone --quiet --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$SRC"
fi
if ! git -C "$SRC" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch --quiet origin "$SHA"
fi
git -C "$SRC" checkout --quiet --detach "$SHA"
SHORT="$(git -C "$SRC" rev-parse --short HEAD)"
NUMBER="$(git -C "$SRC" rev-list --count HEAD)"
TAG="$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || true)"
log "checked out $SHA (short $SHORT, build number $NUMBER${TAG:+, tag $TAG})"

# 5. Prove the registry copy is the same source: every file the crate vendors
#    must be byte-identical to the file at that commit. The crate ships a
#    subset, so the check is one-directional by design.
checked=0
differs=0
while IFS= read -r -d '' f; do
    rel="${f#"$REG/llama.cpp/"}"
    checked=$((checked + 1))
    if ! cmp -s "$f" "$SRC/$rel"; then
        differs=$((differs + 1))
        log "differs: $rel"
    fi
done < <(find "$REG/llama.cpp" -type f -print0)
[ "$differs" -eq 0 ] || die "$differs of $checked vendored files differ from llama.cpp@$SHA — not the same source"
log "vendored tree: $checked files identical to llama.cpp@$SHA"

# 6. Configure with the options llama-cpp-sys-2's build.rs sets on this
#    platform (read from the registry copy: GGML_NATIVE=OFF, GGML_BLAS=OFF on
#    Apple, GGML_OPENMP=ON from llama-cpp-2's default `openmp` feature,
#    BUILD_SHARED_LIBS=OFF, LLAMA_CURL=OFF, LLAMA_BUILD_COMMON=ON), plus
#    LLAMA_BUILD_TOOLS=ON for llama-bench itself. Metal is ggml's default on
#    Apple; it is spelled out so the flag list is complete on its own.
FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_COMMON=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_APP=OFF
)
case "$(uname -s)" in
    Darwin) FLAGS+=(-DGGML_BLAS=OFF -DGGML_METAL=ON) ;;
esac
if [ "${GEN2_BENCH_CUDA:-0}" = "1" ]; then
    FLAGS+=(-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF)
fi

log "configuring: ${FLAGS[*]}"
cmake -S "$SRC" -B "$OUT/build" "${FLAGS[@]}" >"$OUT/configure.log" 2>&1 \
    || { tail -40 "$OUT/configure.log" >&2; die "cmake configure failed (see $OUT/configure.log)"; }
log "building llama-bench"
cmake --build "$OUT/build" --target llama-bench --config Release -j >"$OUT/build.log" 2>&1 \
    || { tail -40 "$OUT/build.log" >&2; die "cmake build failed (see $OUT/build.log)"; }

BIN="$(find "$OUT/build/bin" -maxdepth 1 -type f -name 'llama-bench*' | head -1)"
[ -n "$BIN" ] || die "llama-bench binary not found under $OUT/build/bin"
cp -f "$BIN" "$OUT/llama-bench"

# 7. What the binary will report as build_commit: read it from the generated
#    build-info.cpp rather than trusting the checkout.
GEN="$(find "$OUT/build" -name build-info.cpp -path '*common*' | head -1)"
BUILD_COMMIT="$(sed -n 's/^char const \* LLAMA_COMMIT = "\(.*\)";/\1/p' "$GEN")"
BUILD_NUMBER="$(sed -n 's/^int LLAMA_BUILD_NUMBER = \(.*\);/\1/p' "$GEN")"
case "$SHA" in
    "$BUILD_COMMIT"*) ;;
    *) die "compiled build_commit '$BUILD_COMMIT' is not a prefix of $SHA" ;;
esac

{
    echo "llama_cpp_sys_2 = $VER"
    echo "llama_cpp_commit = $SHA"
    echo "build_commit = $BUILD_COMMIT"
    echo "build_number = $BUILD_NUMBER"
    echo "tag = ${TAG:-}"
    echo "registry = $REG"
    echo "vendored_files_verified = $checked"
    echo "cmake_flags = ${FLAGS[*]}"
    echo "built_at = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$OUT/build-info.txt"

mkdir -p "$(dirname "$PIN")"
{
    echo "# Written by benches/build-llama-bench.sh. The harness refuses to run"
    echo "# and the table generator refuses to render when this disagrees with"
    echo "# the llama-cpp-sys-2 version in Cargo.toml."
    echo "llama_cpp_sys_2 = $VER"
    echo "llama_cpp_commit = $SHA"
    echo "build_number = $BUILD_NUMBER"
} >"$PIN"

log "done: $OUT/llama-bench"
echo "build_commit=$BUILD_COMMIT build_number=$BUILD_NUMBER llama_cpp_commit=$SHA"
