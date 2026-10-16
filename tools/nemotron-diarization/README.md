# Nemotron 3 — N0 qualification

This directory implements the first milestone of the
[roadmap](../../docs/plans/nemotron-3-diarization-roadmap.md): immutable assets,
native CPU build, independent original-weight reference, and coexistence with
Gen2's actual native chat backend. It does not expose a production diarization
API or change the default Gen2 dependency/features.

The initial target is macOS arm64. The standalone SDK is host-supplied and its
Homebrew dependency closure must remain installed. A self-contained application
bundle and other platforms require later packaging qualification.

## Acquire explicitly

Use Python 3.12, `uv`, Rust, Xcode command-line tools, CMake 3.26+, Ninja,
SentencePiece and Abseil. The checked-in `runtime.lock.json` pins runtime,
llama.cpp, source/header hashes, model, original weights, fixture and build flags.
`reference/uv.lock` records Python package artifacts and the exact Transformers
Git revision; `coexistence/Cargo.lock` freezes the diagnostic Rust driver.

```sh
task_cache="$HOME/.cache/gen2-nemotron3-n0"
mkdir -p "$task_cache"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/NeMo-Speech.cpp.git "$task_cache/NeMo-Speech.cpp"
git -C "$task_cache/NeMo-Speech.cpp" checkout --detach 083861c241061562b8ec29eb06d7f6cdc3af8775
git -C "$task_cache/NeMo-Speech.cpp" submodule update --init --depth 1 llama.cpp
brew install cmake ninja pkgconf sentencepiece abseil
python3 tools/nemotron-diarization/artifacts.py --cache "$task_cache/assets" --reference provision
UV_PROJECT_ENVIRONMENT="$task_cache/python-reference" uv sync --locked --project tools/nemotron-diarization/reference
```

Run from the Gen2 repository root. Acquisition is explicit; inference uses local
files and `local_files_only=True`. Models/audio and build outputs stay outside
Git. `check` validates the cache without network access. Corrupt/interrupted
downloads never replace a file until length and SHA-256 agree.

The real fixture is an unmodified 60-second AMI EN2002d meeting excerpt, licensed
CC-BY-4.0, with the upstream matching RTTM (Apache-2.0) and transcript. Preserve
`fixture-README.md` and attribution when redistributing it. Q8 and original FP32
weights use OpenMDW-1.1; the model card is acquired alongside them. The runtime
is Apache-2.0 plus its pinned `NOTICE` and `THIRD_PARTY_NOTICES.md`. Its llama.cpp
and ggml dependency is MIT; SentencePiece, Abseil and their dependencies retain
their respective notices in the host package installation.

## Build and run real native/reference inference

```sh
python3 tools/nemotron-diarization/build_sdk.py \
  --source "$task_cache/NeMo-Speech.cpp" --prefix "$task_cache/sdk-cpu" \
  --output "$task_cache/receipts/build"
python3 tools/nemotron-diarization/run_native.py \
  --shim "$task_cache/receipts/build/n0_probe.dylib" --assets "$task_cache/assets" \
  --output "$task_cache/receipts/native"
HF_HUB_OFFLINE=1 "$task_cache/python-reference/bin/python" tools/nemotron-diarization/reference.py \
  --assets "$task_cache/assets" --wav "$task_cache/assets/ami_en2002d_2132.wav" \
  --output "$task_cache/receipts/reference-ami"
"$task_cache/python-reference/bin/python" tools/nemotron-diarization/compare.py \
  --native "$task_cache/receipts/native/ami.json" --reference "$task_cache/receipts/reference-ami" \
  --rttm "$task_cache/assets/ami_en2002d_2132.rttm" --output "$task_cache/receipts/comparison.json"
```

`native_probe.cpp` is an N0 experiment compiled against the real installed
public header. It retains eight-speaker probabilities and final segments after
160 ms pushes, then tests real audio → silence → a 1.117-second tail → real audio
again. CPU is explicit (`gpu=-1`); left context `-1` preserves the V3 preset.
The independent Transformers reference uses official original FP32 weights on
CPU with matched low-latency geometry. It does not load the native SDK.

The pinned native frontend emits one extra terminal frame on the 60-second
fixture (6001 versus 6000). The comparator requires and records that investigated
difference; it retains the raw frame and scores only the actual audio span.
Native finish pads the centered STFT tail; production presentation must be
clipped to the input duration. The 1.117-second tail similarly emits 112 frames.

Both models are scored with the same external `probability > 0.5` postprocessing,
10 ms frames, zero forgiveness collar, overlap included and optimal global
speaker mapping. Native default hysteresis/padding is reported separately.
The native preset and Transformers `low_latency` are stateful streaming;
neither is the native stateless full-attention operation.

## Compare against the unmodified upstream C example

The upstream example zero-initializes `left_context_frames`, so match it
explicitly for this comparison instead of assuming zero preserves defaults:

```sh
"$task_cache/sdk-cpu/bin/diarize_file" "$task_cache/assets/Nemotron-3-Diarization.q8_0.gguf" \
  "$task_cache/assets/ami_en2002d_2132.wav" --gpu -1 --preset v3-streaming --rttm ami \
  > "$task_cache/receipts/upstream-ami.rttm" 2> "$task_cache/receipts/upstream-ami.log"
python3 tools/nemotron-diarization/run_native.py \
  --shim "$task_cache/receipts/build/n0_probe.dylib" --assets "$task_cache/assets" \
  --left-context 0 --output "$task_cache/receipts/native-example-settings"
```

## Prove coexistence in real Gen2

Use a compatible local chat GGUF, such as the previously qualified
Qwen3-0.6B-Q4_K_M. It is not downloaded by this tool. Initialize Gen2's pinned
llama.cpp submodule before compiling, as required by Gen2's normal chat build.

```sh
git submodule update --init --depth 1 vendor/llama-cpp-sys-2/llama.cpp
probe_target="$task_cache/probe-build"
CARGO_TARGET_DIR="$probe_target" cargo build --locked --release \
  --manifest-path tools/nemotron-diarization/coexistence/Cargo.toml
chat_model="/absolute/path/to/Qwen3-0.6B-Q4_K_M.gguf"
for order in chat-first diar-first; do
  DYLD_PRINT_LIBRARIES=1 "$probe_target/release/gen2-nemotron-n0-probe" "$order" \
    "$task_cache/receipts/build/n0_probe.dylib" "$task_cache/assets/Nemotron-3-Diarization.q8_0.gguf" \
    "$task_cache/assets/ami_en2002d_2132.wav" "$chat_model" "$task_cache/receipts/$order" \
    > "$task_cache/receipts/$order.json" 2> "$task_cache/receipts/$order.log"
done
python3 tools/nemotron-diarization/collect_receipt.py --assets "$task_cache/assets" \
  --evidence "$task_cache/receipts" --sdk "$task_cache/sdk-cpu" \
  --probe "$probe_target/release/gen2-nemotron-n0-probe" --chat-model "$chat_model" \
  --output "$task_cache/receipts/receipt.json"
```

The driver uses Gen2's actual vendored static llama.cpp/Metal chat path, not a
fake library with overlapping symbol names. It tests both load orders, chat
before/after native execution, repeat sessions, model destruction/reload and a
concurrent chat/diarization call from independent owning threads. The collector
checks outputs against standalone native results and inspects actual dyld
dependency paths and Mach-O two-level symbol binding. This supports the
in-process C ABI route on this specific Mac, not a universal dual-ggml guarantee.

## Verification

```sh
python3 -m unittest discover -s tools/nemotron-diarization -p 'test_*.py' -v
cargo test --locked
cargo clippy --locked --manifest-path tools/nemotron-diarization/coexistence/Cargo.toml -- -D warnings
```

Missing real assets fail the qualification commands. Unit-test success alone
does not establish native compatibility. N0's one fixture is a smoke test;
larger held-out quality, long-session behavior, admission, cancellation and the
public API remain subsequent roadmap gates.
