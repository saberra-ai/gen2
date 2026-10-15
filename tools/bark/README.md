# Bark 2.0 ONNX CPU transcription

This is English automatic speech recognition, not Suno Bark text-to-speech and
not an LLM generation backend. `gen2::speech` loads one resident encoder/decoder
pair and resets recurrent decoder state for every completed recording. No
Python process, network call, second cleanup model, microphone capture or VAD
runs during inference. Streaming, diarization, word timestamps, mobile devices,
MLX and accelerator providers are not qualified by this integration.

## Use

Enable `asr-dynamic`; package an ONNX Runtime **1.24.4** CPU library for the host
architecture. Supply its full path as `LoadOptions.native_library`. Runtime
1.23.2 is the original Python helper version, not compatible with these Rust
API-24 bindings. A single process-global library policy is shared with Laya;
different library paths are rejected after initialization. Do not initialize a
different ORT library elsewhere in the host first. `backend-asr-onnx` instead
uses the app-linked runtime, following `ort`'s linking configuration.
Malformed libraries and older ABIs are rejected before initializing the Rust
bindings, and failed loads release admission before returning. A 25-clip smoke
comparison matched the original Python ORT 1.23.2 and qualified 1.24.4 helper
exactly; this is not a full-corpus compatibility claim for the old runtime.

```rust,no_run
use gen2::{Runtime, speech::{Audio, LoadOptions, TranscribeOptions}};
# fn main() -> gen2::Result<()> {
let runtime = Runtime::builder().resident_memory_budget_mb(8192).build()?;
let speech = runtime.load_transcriber("/models/bark/onnx", LoadOptions {
    native_library: Some("/runtime/libonnxruntime.dylib".into()),
    ..Default::default()
})?;
let result = speech.transcribe(Audio::mono_16khz(vec![0.; 16000])?, TranscribeOptions::default())?;
println!("{}", result.fulltext);
speech.shutdown();
# Ok(())
# }
```

`Audio::from_wav` accepts PCM16/float32 WAV and arithmetic downmixing of 1–8
channels. The core feature accepts 16 kHz. Opt into `asr-resample-soxr` for
8/44.1/48 kHz using libsoxr HQ, the helper's resampling algorithm. Resampling is
**experimental**: compiler/SIMD rounding can change greedy decisions. The
unmodified SoXR/PFFFT source is pinned to Python-SoXR 1.1.0's submodule and built
locally with CMake and a C compiler; there are no build-time downloads. It
replaced a scalar FFT4G build that caused measured punctuation/word differences.
The updated build matched normalized samples on all 54 non-16-kHz Mac fixtures
and all 112 supplementary transcripts exactly. Qualification still checks PCM
error and words separately and retains every raw difference; those measurements
are not a bit-exact guarantee for other compilers, CPUs or recordings.
This feature statically links **LGPL-2.1-or-later native code** plus PFFFT's
per-file license. Keep the accompanying [source and notices](../../assets/bark/soxr/UPSTREAM.md)
and review source/relinking obligations for distributed linked binaries. It is off by default, including
in `asr-dynamic`. Hosts may normalize audio themselves without this dependency.
Unsupported encodings/rates fail explicitly. The 128 MiB encoded-file bound and
20 ms–60 s original/post-normalization duration bounds are checked without
silently truncating recordings. Audio samples must be finite.

For model provisioning enable `hf` and call
`gen2::speech::download_bark(cache_directory)`. It fetches all nine `onnx/`
artifacts at revision `0d5f814105042fe801bb019dbf34b4e6f7d3eae6`, anchored to
manifest SHA `8cd94f24f677b74f9e92306039491f23e3e22863205e59a5bdda322026f98f8c`.
Staging is validated before atomic publication; interrupted staging is never
accepted as a bundle. Corrupt Hub data gets one forced-download retry. A corrupt
published entry is preserved as `.corrupt-<uuid>` when its replacement is ready.
A verified warm entry returns before creating a network client, even without
`hf`. Keep published bundles immutable while loaded.

```sh
export GEN2_ORT_LIBRARY=/full/path/to/libonnxruntime.1.24.4.dylib
cargo run --release --no-default-features --features asr-dynamic,asr-resample-soxr,hf \
  --example bark_transcribe -- --download /models/cache recording.wav another.wav
# Or use a complete, already-provisioned bundle without hf/network:
cargo run --release --no-default-features --features asr-dynamic \
  --example bark_transcribe -- /models/cache/bark-0d5f814105042fe801bb019dbf34b4e6f7d3eae6 recording-16khz.wav
```

## Ownership and limits

Default admission reserves **4096 MiB**, with the same minimum for overrides.
The 488,112,188-byte download is not an adequate RAM estimate: a measured 60 s
duration sweep peaked at about 3.1 GB process RSS on an Apple M4 Pro. The margin
covers native workspace and bounded queued audio. Queue capacity is 1–64
(default 2); threads 1–16 (default 4 intra-op, 1 inter-op). Speech and Laya
reservations and chat residency share runtime admission; speech workers are
explicitly shut down rather than automatically evicted like chat weights.

Clone a transcriber to submit competing recordings. Queue saturation returns
`speech_busy`, not an unbounded allocation. Cancellation/deadline checks occur
between frontend/graph/decoder calls. The caller can return before an active
native kernel finishes; its reservation remains charged until sessions drop.
`unload()` stops admission; `shutdown()` additionally waits for resource release.
Dropping the last handle stops the worker. A fresh load starts with clean state.
`Runtime::speech_models`, `speech_reserved_mb` and `stats` expose residency.

Other stable error codes distinguish invalid audio, corrupt bundle, unavailable
runtime, execution/output failure, resource limit, cancellation, deadline and
unloaded worker. Only queue saturation is marked retryable by `gen2::Error`.
Results include `fulltext`, `timestamps: Vec<Timestamp>`, token IDs, duration,
model/manifest pins, actual CPU provider, and queue/frontend/encoder/decoder/execution timings.
Each `Timestamp { startms, endms, text }` corresponds to the token ID at the same
index. Concatenating its text fragments reproduces `fulltext`. Starts measure
approximate emission positions on the pinned encoder's 80 ms grid; ends use the
next token's start, or the recording's duration truncated to milliseconds for the
final token. Intervals can have zero length or include silence, and text fragments
can be empty. These are approximate decoder intervals, not measured word or
phoneme boundaries. Bark's cleanup can omit spoken words.

```rust,ignore
for timestamp in &result.timestamps {
    println!("{}–{} ms: {}", timestamp.startms, timestamp.endms, timestamp.text);
}
```

## Verification and reproduction

Install Python 3.12 and the hash-locked test dependencies with
`python -m pip install --require-hashes -r tools/bark/reference.lock`.
These are verification tools only; they are not application dependencies.
The pure ASR feature does not build llama.cpp. After the native chat dependency
pin, default-feature builds and mixed GGUF-chat benchmarks also require
`git submodule update --init --recursive` for the pinned llama.cpp checkout.
CI skips the already-rejected candidate ONNX frontend audit and observes token
decisions during the helper's own `recognize` call. Fixed tensor probes are then
exported separately. This keeps the complete 250-clip oracle without loading a
second copy of the model or decoding the whole corpus twice for token capture.

```sh
python tools/bark/provision_reference.py target/bark
python tools/bark/qualify_reference.py --bundle target/bark/model/onnx \
  --parquet target/bark/test.parquet --output target/bark/oracle
python tools/bark/export_trace.py --bundle target/bark/model/onnx --reference target/bark/oracle
python tools/bark/qualify_audio.py --bundle target/bark/model/onnx --fluent target/bark/fluent.parquet \
  --disfluent target/bark/test.parquet --output target/bark/audio
export GEN2_BARK_BUNDLE="$PWD/target/bark/model/onnx"
export GEN2_BARK_REFERENCE="$PWD/target/bark/oracle"
export GEN2_BARK_AUDIO_REFERENCE="$PWD/target/bark/audio"
export GEN2_BARK_RESULTS="$PWD/target/bark/native.json"
for test in real_bark_matches_reference_and_reuses_resident_weights real_audio_workflow_matches_qualified_reference real_bundle_integrity_and_warm_offline_cache real_worker_bounds_queue_cancellation_shutdown_and_admission; do
  cargo test --release --no-default-features --features asr-dynamic,asr-resample-soxr,laya-dynamic \
    --test bark_native "$test" -- --ignored --exact
done
cargo test --release --no-default-features --features asr-dynamic --lib native_frontend -- --ignored
cargo test --release --no-default-features --features asr-dynamic --lib native_graphs -- --ignored
python tools/bark/compare_results.py --reference target/bark/oracle/reference.json \
  --native target/bark/native.json --output target/bark/qualification.json
python tools/bark/report_audio.py --reference target/bark/audio/audio-reference.json \
  --native target/bark/audio/native-audio.json --output target/bark/audio-quality.json
cargo run --release --no-default-features --features asr-dynamic --example bark_benchmark -- \
  "$GEN2_BARK_BUNDLE" target/bark/oracle/000.wav target/bark/performance.json
python tools/bark/benchmark_reference.py --bundle "$GEN2_BARK_BUNDLE" \
  --wav target/bark/oracle/000.wav --native target/bark/performance.json \
  --output target/bark/performance-comparison.json
```

The explicit real tests fail when their inputs are absent; ignored tests in an
ordinary Cargo invocation are **not** a compatibility pass. CI provisions the
real checkpoint and matched platform-local NumPy/ORT oracle on macOS, Linux and
Windows. Compare raw text/token decisions separately from human-reference WER.
The normalizer is lowercase, remove punctuation, collapse spaces, word split
(`v1`); it does not normalize names, numbers or abbreviations. Human references
use DisfluencySpeech `transcript_c`. Audio hashes and all 250 row IDs are frozen
in generated reference receipts; large weights/audio stay outside Git.

The native DSP frontend ports ONNX ASR 0.12.0's default NumPy implementation and
uses its original MIT-licensed filterbank coefficients. The packaged ONNX
frontend was rejected: under ORT 1.24.4 it agreed on only 239/250 transcripts.
The native frontend uses float64 FFT/Hann window and float32 mel/FMA/normalization;
the pinned Linux/Windows AVX2 OpenBLAS profile uses interleaved mel accumulators for its
edge/tail tiles. Platform-local stage traces verify this profile independently;
alternate BLAS providers and CPU configurations need separate qualification.
It passed features at `atol=rtol=1e-4`, independent encoder/decoder tensors, and
exact text/token decisions on all 250 clips on the reference Mac. WER was
3.8597%, equal to matched Python, not a reproduction of the model card's MLX
3.79% benchmark. [Receipts](../../benches/results/bark/) identify measured runs.
The merged implementation is qualified on the local M4 Pro and Linux CPU CI.
Merged macOS/Windows CI was still pending when the user requested publication;
those targets are not claimed as completed merged-code qualification.

Supplementary fixtures use 18 preselected LibriSpeech clips, quiet and with
deterministic 20 dB white noise, names/numerals/negation labels, 72 real-speech
WAV rate/channel/encoding variants, and silence/noise duration controls.
This is small domain coverage, not a universal dictation-quality claim.
The human-label scorer gates names/numerals/negation facts on both paths and
reports quiet/noisy WER separately. Frozen non-gating name probes expose shared
upstream weaknesses: Ann/Anne, Kaliko/Calico, Ruggedo and noisy Rocky Ithaca.
These are disclosed limitations, not spelling corrections hidden in scoring.
Cleanup can still omit or change words; review consequential transcripts. Silence
behavior is model behavior, with no hidden suppression policy.

Model weights are Conway Research Bark 2.0, CC-BY-4.0; its LICENSE stays in the
validated bundle. Filterbank notices are in `assets/bark/LICENSE` (ONNX ASR).
The x86 fused logarithm follows NumPy 2.5.4's rational approximation; its
BSD-3-Clause notice is in `assets/bark/NUMPY-LICENSE`.
DisfluencySpeech is Apache-2.0, pinned revision `b7da294fe3a70dd96df6640893f2a5dfc2c87638`.
The [LibriSpeech excerpt](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)
is from [LibriSpeech/OpenSLR](https://www.openslr.org/12), CC-BY-4.0, prepared by
Vassil Panayotov with Daniel Povey, from LibriVox recordings. All source revisions
and verification decisions are recorded in the [roadmap](../../docs/plans/bark-onnx-roadmap.md).
