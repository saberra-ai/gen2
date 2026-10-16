Current runtime migration: Gen2 requires exactly
`nemo-speech-asr 0.2.0+gen2-valid-mel-v1` and floor-valid final rows.
The original N0 SDK/lock and historical recipes below remain controls for their
recorded source versions. The tagged CPU SDK passes fixed F32/Q8 finish suites
and public Rust parity on short calibration/silence. Current macOS packaging
constructs v2 packages bound to that build receipt, patch and frame contract.
The CPU v2 package passes 60-second public/native parity, all seven lifecycle
gates and actual 92-library loader closure checks on the named M4 Pro.
The [30-minute and one-hour CPU qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-cpu-long/qualification.json)
passes retained-tail parity, compaction, bounded provisional revisions and peak
process RSS below 249 MB. It repeats the named smoke fixture; diverse long-audio
quality, coexistence, other platforms and desktop qualification remain open.
See the [migration and current evidence](../../docs/research/nemotron-v3-finish-valid-mel-patch.md).

The separate [Metal construction protocol](valid-mel-metal-build.protocol.json)
freezes a fresh tagged SDK build with pristine pinned GGML sources, 20 embedded
shader-text bundles and model-free state/version checks.
`build_valid_mel_metal_sdk.py` records the compiler inputs and verifies the
installed Mach-O payload bytes against independently flattened shader sources.
The [actual construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-construction/qualification.json)
passes all 11 construction phases, the state test and exact tagged version check;
7 real SDK libraries and 18 installed paths are bound. Construction does not
qualify inference, GPU placement or a desktop release.
The [Metal v2 package](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-macos-package/qualification.json)
also passes construction and relocation with 93 private libraries. Its
[fixed runtime protocol](valid-mel-metal-package.protocol.json) separately admits
60-second native/public Metal parity, seven lifecycle tests and explicit CPU-SDK
Metal rejection. The [actual Metal runtime receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-runtime/qualification.json)
passes all 13 commands, exact 6,000-row/26-turn parity, 100 reloads and actual
private loader closure on the named M4 Pro. The combined test checks sequential
CPU rejection followed by Metal parity in one process; simultaneous model
residency is a separate gate.
The [long-session attempt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long-failure/qualification.json)
retains a passing 30-minute Metal pair. Its one-hour native command failed to
persist output during a disk-space shortage; the child terminal status was not
recorded and the public one-hour command never started. That failed attempt
remains unchanged. The separate [guarded continuation](valid-mel-metal-long-v2.protocol.json)
and its [actual receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long/qualification.json)
pass the two fresh one-hour streams, with disk-space checks, a diagnostic reserve
and atomic command records. Together with the saved 30-minute pair, both
durations pass exact retained-tail parity, compaction, revisions and process RSS
below 261 MB. The passing 30-minute neural work was not repeated.

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

### Frozen quality corpus

`corpus.py` acquires only the fixed Vox ZIP members and AMI/ICSI PCM ranges in
`corpus.sources.json`. It requires exact HTTP206 ranges/ETags and annotation
SHA256s, verifies ZIP CRC/size and mono16kPCM16 headers, retains original labels,
and publishes a complete new directory atomically. Existing frozen directories
are never replaced. See `docs/research/nemotron-heldout-corpus.md` for selection,
training exposure, licenses and reference semantics.

```sh
python3 tools/nemotron-diarization/corpus.py \
  --output "$task_cache/quality-corpus-v1" \
  --ami-annotations "$task_cache/ami_public_manual_1.6.2.zip"
```

The committed acquisition receipt freezes eight slices (1896.992audio seconds;
paired AMI channels are one meeting). `corpus.protocol.json` pins that manifest,
scorer and reference scripts before inference. `qualify_corpus.py` checks the
frozen inputs, original CPUFP32 reference and exact public/native Q8 parity,
then reports every per-slice gate. Shared thresholding is probability>0.5,
zero collar, overlap included, full-audio UEM; native default segmentation is
also scored against the20% target. A quality failure is saved and does not
permit replacing a slice. Raw native terminal frames remain in diagnostics;
scoring clips their padding to the UEM. This set is local tuning-heldout,
not guaranteed checkpoint-unseen. No quality outcome is established by
acquisition alone.

Both Q8 runners include a SHA256 of the actual normalized float32 little-endian
PCM supplied to inference. The corpus gate checks it against the selected WAV,
including one-pass semantics, so equal-duration AMI channels cannot be swapped.
It also validates the FP32 interpreter's installed package versions and actual
Transformers source commit against the frozen environment.

The independent C++ stream probe links the public-domain SHA256 implementation
already present in the pinned runtime's llama.cpp tree. In addition to the
installed SDK includes/library and upstream `examples` include, compile with
`-I "$task_cache/NeMo-Speech.cpp/llama.cpp/vendor/hash"` and explicitly compile
`sha256/sha256.c` from that directory as C++ (`-x c++`). Preserve hashes of that
source, its header and `rotate-bits/rotate-bits.h` in the build receipt. This is
an evidence-tool dependency, not an additional production runtime dependency.

### Original smoke experiment

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
The optional trailing `PROBABILITIES.f32` path on `nemotron_stream` exports its
bounded retained tail as little-endian float32 and records its SHA256; an
existing destination is refused. `compare_stream.py --gen2-probabilities PATH`
requires that export and checks every retained native/wrapper probability with
1e-6 tolerance, in addition to speaker identities, boundaries, geometry and
retention/compaction. It rejects nonfinite/out-of-range values and excess native
padding before clipping, using the same terminal-frame bound as Gen2. Timings
remain separate from correctness.
The independent probe's successful compile/calibration is not a completed soak.

## Linux runtime construction and container qualification

Linux development SDK/native checks and runtime construction are separate stages.
On Ubuntu 24.04 x86_64, after `platform_native.py` succeeds, install packaging
prerequisites (`patchelf`, `binutils`, authenticated matching `deb-src` indexes),
then run:

```sh
python tools/nemotron-diarization/platform_linux_package.py \
  --sdk-prefix /absolute/path/nemo-sdk \
  --native-receipt /absolute/path/nemo-evidence/receipt.json \
  --output /absolute/path/new-package-evidence
python tools/nemotron-diarization/qualify_linux_package.py \
  --sdk-prefix /absolute/path/nemo-sdk \
  --native-evidence /absolute/path/nemo-evidence \
  --package-evidence /absolute/path/new-package-evidence \
  --output /absolute/path/new-container-evidence
```

The second command requires a Docker daemon, prebuilt Gen2 release helpers and
all locked assets already present in the native evidence directory. It pulls
Ubuntu 24.04 before testing, freezes the resolved image content identity, then
executes fresh containers with network disabled, read-only inputs and no SDK or
checkout mounted. It verifies the delivered runtime archive in a relocated
Unicode path, native/public outputs, seven executed lifecycle tests, actual
loaded library paths and a missing-library negative control. It creates no
release claim: chat/Bark/Laya coexistence, additional host baselines and quality
remain independent gates. Existing output directories are refused so failed
runs remain available. The workflow uploads JSON/logs/probability outputs and
the opaque runtime archive, without putting model/runtime/source binaries in Git.

The [construction receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-runtime-construction/qualification.json)
records the passing construction and downloaded filesystem checks. Actual
container execution is tracked in the roadmap; source and model-free tests do
not establish that result.

After that same-source container stage passes, the separate actual chat gate is:

```sh
python tools/nemotron-diarization/qualify_linux_chat.py \
  --container-evidence /absolute/path/new-container-evidence \
  --output /absolute/path/new-chat-evidence
```

It builds `nemotron_chat_coexistence` with the regular CPU/OpenMP chat backend,
acquires only the separately pinned public Qwen fixture/card/license during
setup, and freezes all inputs before chat execution. The OpenMP test library
must have the exact GCC source/copyright identity already present in the
runtime's full source bundle. It is mounted as a separate test dependency;
this stage does not turn the diarization archive into a chat release package.
Each isolated/load-order run gets a fresh offline container. All chat outputs
must match the isolated baseline, six diarization runs must match the independent
native result, shutdown/reload must release reservations, and exported intervals
must prove overlapping public inference calls (preparation/serialization do
not count). Actual loader paths and critical GGML symbol bindings must remain
within the declared payload. Bark/Laya, other baselines and accuracy stay open.

The actual `c242201` [offline container receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-offline-container/qualification.json)
passes relocated native/public parity, all seven lifecycle tests, eight-library
loader closure and the missing-private-library control on the selected Ubuntu
24.04 x86_64 AVX2/FMA/F16C baseline. The independent receiver rechecked all
delivered archive/source bytes; this is not a chat/Bark/Laya, other-baseline,
physical-machine, accuracy or complete release qualification.

Windows development qualification accepts `platform_native.py --windows-private-dlls`.
It uses an exact-source-guarded CMake hook to relink all five owned DLLs under
`gen2_nemo_windows_` names while retaining logical CMake targets and C ABI symbols.
The probe import library and public entry follow the new names. A bounded PE
scan rejects remaining generic native imports and bound/legacy delay forms;
private export-directory identities and the installed public C ABI import
library are required.
Source tests do not establish actual Windows private-DLL qualification, packaged
VC runtime provenance, offline operation or coexistence.

The actual `3364fd7` [Linux chat receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-chat-coexistence/qualification.json)
passes both load orders, repeated/reloaded/concurrent calls and recorded GGML
bindings in fresh offline containers. All six diarization outputs are numerically
identical to the independent native reference; all ten chat observations match
`Hello!`, with positive actual-call overlap and zero released reservations.
This qualifies only the selected Ubuntu/glibc/CPU binaries and calls, with
Bark/Laya, other machines/baselines, accuracy/performance and release gates open.

The actual `24ad927` [Windows private-SDK receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-private-native-development/qualification.json)
passes private PE identities/imports, native/public parity, seven lifecycle tests
and the default-loader control. Runtime provenance captures exact eight-DLL
source/install equality and Microsoft signature observations; absent accompanying
license candidates keep redistribution unqualified. Closed/offline runtime,
actual mixed-model calls, desktop machine and live performance gates remain open.
The Windows module snapshot diagnostic separately passes two Win32 tests,
including Unicode executable relocation, without a packaged-runtime claim.

Windows development runtime construction now accepts that exact qualified SDK:

```sh
python tools/nemotron-diarization/package_windows.py \
  --sdk-prefix /absolute/path/nemo-sdk \
  --native-evidence /absolute/path/nemo-evidence \
  --output /absolute/path/new-windows-package-evidence
```

This portable constructor rechecks all SDK file bytes, private PE tables,
CPU/CMake settings and retained Microsoft source/signature observations. It
copies five private native DLLs and eight unchanged VC DLLs, installed notices
and qualification metadata. The deterministic ZIP has a bounded, validated
central directory and exact byte inventory; extraction rejects Windows aliases,
links, case collisions and undeclared members, then verifies a Unicode-path
restoration. It deliberately makes no ACL or executable-permission claim.
The Windows probe workflow downloads the specific successful native run before
construction. Construction alone does not execute the package or establish
Microsoft redistribution rights, fresh-machine closure, coexistence or release
qualification; all those receipt flags stay false.

The actual `abcef06` [Windows construction receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-runtime-construction/qualification.json)
passes construction and restoration on Server 2022. All 43 archive entries and
13 unchanged DLLs were independently reverified on the macOS receiver.

After same-checkout prerequisites and construction pass on a Windows Docker
host, the separate execution gate is:

```sh
python tools/nemotron-diarization/qualify_windows_package.py \
  --sdk-prefix /absolute/path/nemo-sdk \
  --native-evidence /absolute/path/nemo-evidence \
  --package-evidence /absolute/path/new-windows-package-evidence \
  --prerequisites /absolute/path/windows-container-prerequisites/preflight.json \
  --output /absolute/path/new-windows-offline-evidence
```

It builds current Gen2 public/lifecycle helpers, acquires the locked assets on
the host and restores the delivered ZIP. Native inference, public inference,
seven lifecycle tests and a missing-private-GGML control run in separate fresh
Server Core process containers with network disabled and read-only input mounts.
Stable observations record actual child module paths and disk hashes, requiring
all five private DLLs at the relocated runtime and exact qualified CRT bytes.
Test executables receive identical app-local CRT copies for their startup
imports. Only the independent native probe uses the runtime directory on PATH
for its static C ABI import; Gen2 uses its explicit loader and system-only PATH.
The helper ZIP is a separate test artifact. The actual `855a83c`
[Windows offline receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-offline-container/qualification.json)
now passes all five fresh-container calls, exact 27-turn/6,001-frame native/public
parity, all seven lifecycle tests and the missing-private-library control.
Actual public/lifecycle modules include all five private DLLs in the relocated
package, VCRUNTIME140 in the helper directory and MSVCP140/VCRUNTIME140_1 in the
runtime directory, all with qualified bytes. The receiver reverified both
opaque archives, helpers/PE tables and all five exact frozen command specs.
Observed RTF is about 1.06; Windows live performance, desktop, mixed-model,
redistribution and held-out quality remain unqualified.

The separate `diarization-windows-dependencies.yml` job performs one-shot
historical metadata recovery on the cache-owning `codex/nemotron3-platforms`
branch. `windows_vcpkg_metadata.py` accepts `--archives`, `--native-evidence`,
`--sdk-prefix` and a new `--output` directory. It requires the exact retained
native/status hashes, recovers nine original ZIPs and selected metadata,
cross-checks raw ABI/dependency, CONTROL/status and SPDX identities, and matches
four target notices to the SDK. It executes no cached tools and never builds
replacement packages. Original installed-list bodies, source closure and actual
native linker membership remain separate. The actual `9be6a4b`
[recovery receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-cache-metadata-recovery/qualification.json)
now revalidates all nine opaque archives (304,920,284 bytes), 45 metadata files
and four target notices on the receiver. The receiver also reproduced all nine
installed-list bodies with exact historical size/SHA matches; the producer's
original no-reconstruction flag is preserved. The subsequent
[source-recipe receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-source-recipes/qualification.json)
verifies 34 pinned port files, 53 SPDX/53 ABI comparisons and three actual source
archive SHA512s. Complete downstream notice/source closure and actual linker
membership remain open, including the separate Darts-clone BSD notice and esaxx
scope. See the [recovery design](../../docs/research/nemotron-windows-static-provenance.md).
