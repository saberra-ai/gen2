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
again. CPU is explicit (`gpu=-1`). Geometry is explicitly matched to the official
reference: chunk 9, right context 4, left context 0, FIFO 264, cache 264 and
update period 222 (all on the coarse 80 ms grid). The SDK's similarly named V3
preset instead uses chunk 13/right 1/FIFO 80/update 40, so names alone are not
evidence of matched settings.
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

Use a separate SDK-default geometry run to match the unmodified upstream
example; do not compare that run against the official Python low-latency mode:

```sh
"$task_cache/sdk-cpu/bin/diarize_file" "$task_cache/assets/Nemotron-3-Diarization.q8_0.gguf" \
  "$task_cache/assets/ami_en2002d_2132.wav" --gpu -1 --preset v3-streaming --rttm ami \
  > "$task_cache/receipts/upstream-ami.rttm" 2> "$task_cache/receipts/upstream-ami.log"
python3 tools/nemotron-diarization/run_native.py \
  --shim "$task_cache/receipts/build/n0_probe.dylib" --assets "$task_cache/assets" \
  --geometry upstream-example --output "$task_cache/receipts/native-example-settings"
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
"$task_cache/python-reference/bin/python" -m unittest discover -s tools/nemotron-diarization -p 'test_*.py' -v
cargo test --locked
cargo clippy --locked --manifest-path tools/nemotron-diarization/coexistence/Cargo.toml -- -D warnings
```

Missing real assets fail the qualification commands. Unit-test success alone
does not establish native compatibility. N0's one fixture is a smoke test;
larger held-out quality, long-session behavior, admission, cancellation and the
public API require their own subsequent roadmap evidence.

## Public Rust API and integration evidence

The optional `backend-diar-nemo` feature adds the CPU Rust wrapper and bounded
streaming worker. See [contracts and setup](../../docs/diarization.md). Real
tests are ignored by ordinary CI and fail if explicitly run without assets:

```sh
export GEN2_NEMOTRON_ASSETS="$task_cache/assets"
export GEN2_NEMO_LIBRARY="$task_cache/sdk-cpu/lib/libnemo_speech_asr_c.dylib"
export GEN2_NEMOTRON_REFERENCE="$task_cache/receipts/native"
cargo test --no-default-features --features backend-diar-nemo \
  --test diarization_api --test diarization_native -- --include-ignored --test-threads=1
cargo build --no-default-features --features backend-diar-nemo,hf \
  --example nemotron_diarize --example nemotron_stream
python3 tools/nemotron-diarization/qualify_provisioning.py \
  --executable target/debug/examples/nemotron_diarize \
  --cache /absolute/fresh/qualification-cache --output "$task_cache/n2-provision-receipt.json"
for minutes in 30 60; do
  /usr/bin/time -l target/debug/examples/nemotron_stream \
    "$task_cache/assets" "$GEN2_NEMO_LIBRARY" "$task_cache/assets/ami_en2002d_2132.wav" "$minutes" \
    > "$task_cache/n3-soak-${minutes}m.json" 2> "$task_cache/n3-soak-${minutes}m.time"
done
```

The repeated clip establishes continuous cache/transport/compaction behavior,
not quality on a long meeting. The development evidence collector
`capture_integration.py` checks actual passing real-test names, complete 30/60
minute streams, retained probability ranges, revision ordering and measured
RSS below the reservation. Its output explicitly leaves N4/N5 and clean-build
release gates open. Keep the executed binary and its hash unchanged while
capturing those experiments.

Mixed-model tests are in `diarization_mixed_native.rs` with
`backend-llamacpp,asr-dynamic,laya-dynamic,backend-diar-nemo` enabled. Supply
`GEN2_BARK_BUNDLE`, `GEN2_ORT_LIBRARY` (the qualified 1.24.4 runtime),
`GEN2_CHAT_GGUF` and the diarization variables above. Execute `mixed_chat_first`
and `mixed_diarization_first` in separate processes with `--ignored`. These
tests use real Bark/chat/diarization weights and the synthetic Laya smoke graph;
they qualify admission/coexistence, not Laya checkpoint quality. The corresponding
`mixed_real_laya_chat_first` / `mixed_real_laya_diarization_first` cases additionally
require `GEN2_LAYA_BUNDLE` and repeat the checks with the full checkpoint
and its manifest reservation; both cases have passed locally.

## Alignment investigation

AMI manual word annotations 1.6.2 are available under CC-BY-4.0 from
[the corpus maintainers](https://groups.inf.ed.ac.uk/ami/download/). Download
the manual annotation archive explicitly; `qualify_alignment.py` validates its
frozen SHA-256 before reading only the selected meeting's word XML files.

```sh
GEN2_ORT_LIBRARY=/absolute/path/to/libonnxruntime.1.24.4.dylib \
  cargo run --no-default-features --features asr-dynamic --example bark_transcribe -- \
  /absolute/path/to/bark/onnx "$task_cache/assets/ami_en2002d_2132.wav" \
  > "$task_cache/n4-bark-ami.json"
python3 tools/nemotron-diarization/qualify_alignment.py \
  --annotations "$task_cache/ami_public_manual_1.6.2.zip" \
  --transcription "$task_cache/n4-bark-ami.json" --output "$task_cache/n4-alignment-receipt.json"
```

The tool groups whole token intervals without proportional text timing. Its
chronological merged WER includes cleanup omissions and overlapping reference
speech; it is diagnostic, not speaker-attributed WER. Initial word-end errors
show that Bark's next-token intervals can span silence. This experiment does
not qualify precise word timing or the Pnola attribution product claim.


## Relocatable macOS CPU runtime package

`package_macos.py` consumes the pinned SDK and its actual `build.json` receipt.
It requires the qualified native entry, checks every SDK build-record hash and
identity before resolving aliases, discovers the complete dylib dependency closure,
requires the qualified Homebrew package versions, copies runtime and component
notices, and relocates every non-system dependency to a private
`@loader_path/libgen2_nemo_…` name. It removes source runpaths and signs each
library ad hoc. A stable `lib/libnemo_speech_asr_c.dylib` relative alias is the
public loader entry. Neither model weights nor a Python inference runtime are
included. Packaging and verification run offline.

```sh
python3 tools/nemotron-diarization/package_macos.py \
  --sdk-prefix /absolute/sdk-cpu \
  --build-receipt /absolute/receipts/build/build.json \
  --output /absolute/nemo-cpu-package
python3 tools/nemotron-diarization/package_macos.py \
  --verify /absolute/nemo-cpu-package
```

The output must not already exist. Source SDK/Homebrew files are never edited.
`package.json` records original/relocated hashes, component provenance, minimum
macOS version, dependency names and every distributed file. Verification checks
the complete file set, checksums, private identities, closed loader paths,
architecture, deployment metadata, signatures and entry alias. Preserve relative
symlinks when copying the package. A bundler may materialize the entry alias
as a regular file; verification accepts it only when its complete signed bytes
equal the qualified private entry. Other alias substitutions fail. Do not modify
runtime files while loaded.

The measured M4 Pro package has 92 libraries and is approximately 11 MiB. Its
actual deployment minimum is macOS 26.0; the tested host is macOS 26.3. This
metadata is not evidence for every Mac model or an older macOS version. Signatures
are ad hoc local signatures, not a Developer ID/notarization receipt. Runtime
and model terms remain separate; `notices/openmdw-1.1/` stores an immutable copy
of the [model agreement](https://openmdw.ai/license/1-1/) for model packaging.
SentencePiece's bundled Darts/protobuf license sources are pinned in
`notices/sentencepiece-0.2.2/sources.json`; installed native component licenses
and package metadata are also retained.

Local qualification copies the finished package to a different path containing
spaces, then runs the real public API tests with network and reads from
Homebrew, the source SDK and original package locations denied. Separate sandbox
controls prove those denials against a reachable local socket and existing source
files. Loader traces show the private libraries resolving inside the relocated
copy. This establishes relocation/offline behavior on the named host; a fresh
machine, app bundle, relocated mixed-model coexistence, Metal and other OS
qualification remain separate N5 gates.


## Metal development qualification

The same pinned source/preset digest also covers upstream `metal-diar`, which
uses unpatched ggml and embedded Metal shaders. Build it in separate directories:

```sh
python3 tools/nemotron-diarization/build_sdk.py --backend metal \
  --source "$task_cache/NeMo-Speech.cpp" --prefix "$task_cache/sdk-metal" \
  --output "$task_cache/receipts/build-metal"
python3 tools/nemotron-diarization/run_native.py \
  --shim "$task_cache/receipts/build-metal/n0_probe.dylib" --assets "$task_cache/assets" \
  --output "$task_cache/receipts/native-metal" --gpu 0
python3 tools/nemotron-diarization/package_macos.py \
  --sdk-prefix "$task_cache/sdk-metal" --build-receipt "$task_cache/receipts/build-metal/build.json" \
  --output "$task_cache/package-metal"
```

The native probe's GPU argument is strict: a missing device fails creation.
Gen2's explicit `LoadOptions::metal` additionally checks the selected registry is
Metal and records the actual device description. CPU/Metal native libraries get
different private loader names. Mixing their raw SDKs can reuse generic ggml
install names from the first load and hide the second SDK's device registry.
Package verification checks embedded shaders in the build receipt before
admitting a Metal package. Metal GPU and CPU outputs need their own numerical
and annotated-quality comparisons; CPU output is not an exact Metal oracle.
The ordinary ignored tests do not establish Metal support. To execute the
public Metal reference check, supply `GEN2_NEMO_LIBRARY` (relocated CPU package),
`GEN2_NEMO_METAL_LIBRARY` (relocated Metal package), `GEN2_NEMOTRON_ASSETS`,
`GEN2_NEMO_METAL_REFERENCE` (independent Metal probe `ami.json`) and
`GEN2_NEMO_METAL_DEVICE` (expected hardware description), then run the ignored
`explicit_metal_matches_native_reference_and_cpu_sdk_cannot_silently_fallback`
test in a fresh process. This is development qualification; full lifecycle,
long-session, mixed-backend, offline app and clean performance gates remain open.


The two `nemotron_diarize` / `nemotron_stream` Rust examples accept an optional
leading `--metal` flag; CPU remains explicit by default. The stream example's
receipt includes backend and selected GPU description. To reuse the existing
native lifecycle, streaming, cancellation, ABI, warm-cache and100-reload gates
against Metal, set `GEN2_NEMO_EXECUTION=metal`, point `GEN2_NEMO_LIBRARY` at the
private Metal package and `GEN2_NEMOTRON_REFERENCE` at its independently generated
native outputs. Run `cargo test --no-default-features --features backend-diar-nemo
--test diarization_native -- --ignored --skip explicit_metal --test-threads=1`.
The separate explicit-Metal test still requires both CPU and Metal packages and
must run in a fresh process with its own documented environment. Invalid backend
selection fails immediately. Compile-only or ignored results are not qualification.


`native_stream_probe.cpp` is an independent C-header long-session reference;
it links the private native package, retains raw probabilities only for the
bounded native tail, and reports actual native cold load/push median/p95/tail
finish timings. It accepts a20ms–600s licensed mono16k WAV, repeats totaling at
most one hour, and explicit CPU(-1)/GPU ordinal. A20MiB source-file cap, retained
probability/turn caps, finite-value checks and no-overwrite evidence paths bound
qualification input/output. Gen2's `nemotron_stream` example separately reports
public load time, queue backpressure, snapshots and native compute time.
`compare_stream.py` checks final native/wrapper speaker identities, boundaries,
geometry, frame retention/compaction and finite probability evidence. It clips
only presentation padding and reports timings separately from correctness.
The independent probe's successful compile/calibration is not a completed soak.
