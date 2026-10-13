# Whistle ONNX transcription

Whistle and Bark use one `gen2::speech` caller interface. The private adapters
own different frontends and decoders; callers select a model and submit an
audio request. Enable `asr-dynamic` and supply an ONNX Runtime 1.24 native
library. Native inference has no Node/Python dependency.

```rust,no_run
use gen2::{Runtime, speech::*};
let runtime = Runtime::builder().resident_memory_budget_mb(8192).build()?;
let speech = runtime.load_speech(
    SpeechSource::whistle("/models/whistle-14fc92cb"),
    LoadOptions {
        native_library: Some("/runtime/libonnxruntime.dylib".into()),
        reservation_mb: 2048,
        ..Default::default()
    },
)?;
let details = speech.transcribe_request(
    TranscriptionRequest::new(Audio::from_wav("recording.wav")?)
        .language(SpeechLanguage::Es),
)?;
println!("{}", details.transcription.text);
speech.shutdown();
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `SpeechSource::bark` with the same load/request calls for Bark. Existing
`load_transcriber` and `transcribe(audio, TranscribeOptions)` continue to work.
Inspect `profile()` for supported languages/controls and `capabilities()` for
compiled audio formats, duration bounds, cleanup and provider. Unsupported
controls produce `SpeechError::InvalidRequest` rather than being ignored.

## Package and input contract

Initial package: `mrfakename/whistle-ONNX` revision
`14fc92cb09ad5c271fe9a4ec4be70274790050d2`, Apache-2.0. Gen2 pins each file's
size/hash in [model-lock.json](model-lock.json). Layout:

```text
whistle.pack
onnx/encoder.onnx
onnx/encoder.onnx.data
onnx/decoder.onnx
onnx/decoder.onnx.data
```

Load that package root, not its `onnx` directory. The pack is required for
filterbank, tokenizer and engram data. The approximately 17 MB pack does not
replace the downloaded FP32 network weights or imply 17 MB memory usage.

`download_whistle(cache)` requires `hf` only for an uncached entry. It stages,
hash-checks and atomically publishes the full pinned package; complete warm
entries return before network-client creation. Invalid packages fail before
inference. Keep a loaded package immutable. Cache entries use
`whistle-<revision>` and corrupt entries are preserved during replacement.

```sh
cargo run --release --no-default-features --features asr-dynamic,hf \
  --example whistle_transcribe -- provision /models/cache
export GEN2_ORT_LIBRARY=/runtime/libonnxruntime.1.24.4.dylib
cargo run --release --no-default-features --features asr-dynamic \
  --example whistle_transcribe -- /models/cache/whistle-14fc92cb09ad5c271fe9a4ec4be70274790050d2 recording.wav es
```

Whistle accepts finite mono 16 kHz float32 PCM, 40 ms–60 s, through `Audio`.
Shared input handling decodes PCM16/float32 WAV and averages 1–8 channels.
Optional 8/44.1/48 kHz conversion uses `asr-resample-soxr`; retain the existing
experimental qualification limits described in the [Bark guide](../bark/README.md).
No microphone capture, VAD, live streaming or long-audio stitching is provided.

English, German, French, Spanish, Italian, Dutch and Polish are selectable.
Without a language override, the first decoder step selects one of those
languages. Detailed results keep that detected language separate from the
selected/forced language. The model can identify a language incorrectly.

The default greedy token limit follows the duration-based upstream rule and
never exceeds 320 text tokens. `max_tokens` can select 1–1024. Audio within the
duration envelope can still exhaust that limit. Ordinary calls return
`SpeechError::TokenLimit`; `.allow_partial(true)` on a detailed request returns
explicit `Completion::TokenLimit` metadata. The CLI prints those details and
exits unsuccessfully on partial output. Decoder histories/KV reset per request,
and immutable engram-row caching is bounded to 4096 rows.

Whistle does ordinary transcription; it does not advertise Bark's disfluency
cleanup. Word timestamps/probabilities and keyword biasing remain later work.
Native CPU qualification does not establish GPU, FP16, mobile or MLX support.

## Reproduce qualification

The reference uses the pinned upstream JavaScript on CPU with four intra-op
threads and one inter-op thread. Its Node ORT is 1.24.3; native Rust uses the
qualified ORT 1.24.4 library. Exact transcript/token decisions are compared under
matched settings. Reference tools and Node are development-only dependencies.
Source hashes and npm integrity lock are checked in. Reuse Bark's hash-locked
Python environment for fetching/normalizing FLEURS and locating native ORT.

```sh
python -m pip install --require-hashes -r tools/bark/reference.lock
npm ci --prefix tools/whistle/reference --ignore-scripts
python tools/whistle/provision_reference.py
python tools/whistle/fetch_corpus.py
node tools/whistle/reference/qualify.mjs target/whistle/model target/whistle/corpus target/whistle/oracle
export GEN2_WHISTLE_BUNDLE="$PWD/target/whistle/model"
export GEN2_WHISTLE_CORPUS="$PWD/target/whistle/corpus"
export GEN2_WHISTLE_REFERENCE="$PWD/target/whistle/oracle/reference.json"
export GEN2_WHISTLE_RESULTS="$PWD/target/whistle/native.json"
# Set GEN2_ORT_LIBRARY to the actual installed 1.24.4 native library.
cargo test --release --no-default-features --features asr-dynamic,laya-dynamic \
  --test whistle_native real_whistle_matches_reference_and_reports_language_and_completion -- --ignored --exact
cargo test --release --no-default-features --features asr-dynamic,laya-dynamic \
  --test whistle_native real_whistle_integrity_offline_cache_and_lifecycle -- --ignored --exact
python tools/whistle/report_quality.py --reference target/whistle/oracle/reference.json \
  --native target/whistle/native.json --output target/whistle/quality.json
```

Explicit real-model tests fail when assets are missing. The frozen FLEURS subset
uses 20 identified real recordings per language (first ten parity, next ten
held-out quality), with human transcripts and hashes. This is a compatibility
and small quality sample, not a broad accuracy benchmark. Raw text/token parity
and human-reference accuracy are separate evidence. The reference's own
transcripts can have substantial errors, particularly on Dutch/Polish in this
subset; matching the reference does not establish good dictation quality.

The three-platform workflow regenerates the oracle on the host, compares the
same pinned audio through public calls, and uploads results. Also run Bark's
existing full corpus and coexistence gates after shared speech changes. For
both models/Laya together, set `GEN2_BARK_BUNDLE` and run
`both_models_share_the_request_interface_and_admission` in `whistle_native`.

For measured warm performance, duration bounds and a 100-request residency soak:

```sh
cargo run --release --no-default-features --features asr-dynamic \
  --example speech_benchmark -- whistle "$GEN2_WHISTLE_BUNDLE" \
  /path/to/base.f32 target/whistle/performance.json
# Choose an actual PCM filename from corpus/corpus.json.
node tools/whistle/reference/benchmark.mjs "$GEN2_WHISTLE_BUNDLE" \
  /path/to/the/same.f32 target/whistle/performance.json target/whistle/performance-comparison.json
```

The proposed performance gates are RTF <1.0 and native median ≤1.25× the matched
Node reference on a named host. Sampler measurements include native allocations.
The memory floor is conservative; record peak residency for the longest input
and bounded token limit instead of inferring it from the weight file sizes.

Qualification status and machine receipts are tracked in
[the implementation roadmap](../../docs/plans/whistle-onnx-roadmap.md).
