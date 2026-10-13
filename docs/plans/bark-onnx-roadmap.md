# Bark 2.0 support: ONNX-first roadmap and verification plan

Status: native CPU implementation exists; desktop release qualification is in
progress. Main has not yet received this work. Updated: 2026-10-13 UTC.
Prepared: 2026-10-12. Original Gen2 baseline: `c98d99f`.

## Current qualification status

The isolated `codex/bark-onnx` branch implements the dedicated speech API,
native frontend/encoder/TDT decoder, bounded workers, shared speech/chat/Laya
admission, WAV handling, and pinned offline provisioning. Remote-main dependency
and ONNX text-generation changes have been merged without changing the ASR
operation into generation.

On the Apple M4 Pro, native text **and token IDs** match all 250 pinned
DisfluencySpeech clips. Human-reference WER is 3.8597%, equal to the independent
Python CPU reference. Eight frontend probes match exactly on this Mac;
independent fixed encoder/decoder tensors pass `atol=rtol=1e-4`. The candidate
ONNX preprocessing graph was rejected (239/250 raw transcript matches), so the
native implementation ports the helper's default NumPy frontend.

The matched 1/5/15/30/60-second CPU sweep passed the native/Python performance
gate, with a 60-second median of 2.21 seconds and sampled peak process RSS of
3.09 GB. A separate mixed run kept the real Qwen3-0.6B GGUF chat model, speech,
and a synthetic Laya smoke model simultaneously resident; 100 speech requests
were repeatable and final reservations were zero. This is not real Laya
checkpoint quality/performance evidence.

The original Python ORT 1.23.2 and candidate 1.24.4 environments now match on
25/25 pinned clips (PCM, text and token IDs). Native code uses API 24, not 1.23.2.
An incompatible-library test exposed a binding-initialization deadlock; ABI
preflight now rejects malformed/older libraries before binding initialization,
with zero leaked reservations and a subsequent successful correct-library load.

The user approved shipping optional resampling as **experimental with measured
limits**. The replacement pinned SoXR/PFFFT build now matches all 54 non-16-kHz
normalized sample fixtures and all 112 supplementary transcripts exactly on the
reference Mac, including the original three 8-kHz punctuation differences.
Human-quality gates pass. This is not a universal parity guarantee.

M4 remains open: Windows passed full 250-clip raw text/token parity but three
48 kHz float32 variants duplicated a word after native resampling. Linux has
two raw punctuation differences on the main corpus. Stage traces located the
first Linux drift in OpenBLAS Haswell's interleaved mel accumulation. The native
fix reproduces all eight captured mel tensors exactly in the diagnostic loop;
the core 16-kHz gate has not been relaxed. CI also exposed that macOS/Linux
admission used lifetime peak RSS as current memory. Live-RSS probes now pass a
red/green released-page regression without relaxing admission ceilings. The
updated three-OS real-model run must pass before release. Independent CI cases
use fresh processes to isolate unrelated model loads.
Accelerators, physical mobile, MLX, microphone capture and live caption stitching
remain follow-ups, not completion claims for M0–M4.

## Outcome and scope

Deliver native Rust transcription of English recordings using
`ConwayResearch/bark-0.6B-4bit-v2.0`, with ONNX Runtime CPU as the first qualified
backend. Load the weights once, submit independent recordings, and return the
model's cleaned transcript. Python is a verification reference, with no Python
dependency in the shipped inference path.

The first working slice accepts finite mono float32 samples at 16 kHz, up to
60 seconds. The desktop release then adds qualified WAV decoding, downmixing,
and resampling. The minimum supported recording length must be measured: the
reference frontend's normalization is problematic when it has fewer than two
valid feature frames. Reject unsupported inputs explicitly; do not silently
truncate or pad them to make a test pass.

This is a new scope after the original local-inference roadmap, which excluded
audio in wave 1. Keep that historical roadmap intact. This plan introduces
transcription as a separate operation alongside generation and typed decisions.

## Evidence and constraints

### Model package

Pin the first qualification run to Hugging Face commit
`0d5f814105042fe801bb019dbf34b4e6f7d3eae6`. The published ONNX release declares
488,112,188 download bytes, including its variant manifest. This is download
size, not a RAM reservation. [Pinned release inventory][release]

The package contains these inference artifacts:

| Artifact | Role |
| --- | --- |
| `onnx/encoder-model.onnx` and `.onnx.data` | Conformer encoder and external weights |
| `onnx/decoder_joint-model.onnx` and `.onnx.data` | Recurrent prediction/joint graph and external weights |
| `onnx/vocab.txt` | Token-to-text mapping, including the blank token |
| `onnx/config.json` | `nemo-conformer-tdt`, 128 features, subsampling factor 8, maximum 10 tokens per step |
| `onnx/greyhound.json` | English, 16 kHz, 60-second maximum, 4-bit variant |
| `onnx/manifest.json`, `bark-release.json`, licenses | File sizes, hashes, provenance and attribution |

The upstream helper uses ONNX ASR 0.12.0 and ONNX Runtime 1.23.2 on CPU, with
four intra-op threads and one inter-op thread. It reads float32 audio, averages
1–8 channels, and uses `soxr` HQ to resample to 16 kHz. [Pinned helper][helper]

Pin ONNX ASR source to `b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc`
(tag `v0.12.0`). Its default CPU frontend is **NumPy**, selected by `Manager`;
the Bark repository does not include a preprocessing graph. The package also
provides an ONNX frontend option. Extracting that graph from the pinned wheel
is a candidate shortcut, conditional on parity with the CPU reference.
[Frontend selection][loader] [NumPy frontend][numpy] [ONNX frontend][preprocessor]

The decoding algorithm is greedy TDT, with recurrent state and predicted time
advances. Loading the encoder alone, running CTC decoding, or feeding these
weights to an LLM loader cannot establish support.
[Graph invocation][nemo] [Decode loop][decode]

### Gen2 at the original planning baseline

| Existing module | Implication for this work |
| --- | --- |
| `src/decision/onnx.rs`, `Cargo.toml` | Gen2 already uses `ort = 2.0.0-rc.12`, `api-24`, and host-supplied ONNX Runtime 1.24 |
| `.github/workflows/ci.yml` | Laya native smoke tests run with Runtime 1.24.4 on Linux, Windows and macOS |
| `src/api/runtime.rs`, `src/decision/worker.rs` | Dedicated workers, bounded queues, admission reservations and shutdown are precedents for ASR |
| `src/decision/graph.rs`, `src/decision/bundle.rs` | Integrity and external-data validation exist, but their accepted signatures, domains and opset are Laya-specific |
| `src/advanced/plugin.rs`, `src/backend/traits.rs` | The current backend plugin interface assumes generation sessions and token pulling |
| `src/api/input.rs` | Message inputs have no audio part |
| `src/api/hf.rs` | The current Hub resolver selects GGUF files; it does not provision this multi-file ASR package |
| `src/backend/facade.rs` | General ONNX model loading is rejected |

The earlier answer's ONNX limitation applied to general generation loading at
that baseline. Remote main now has a separate ONNX Runtime GenAI text backend,
requiring its own API-31 runtime package. It is not this API-24 ASR adapter and
does not establish Bark compatibility. Existing Laya supplies the ASR runtime
precedent; mixed GenAI/API-24 native-library coexistence is not qualified here.

## Recommended interface and implementation

Add a deep `gen2::speech` module with a typed transcriber handle, analogous to
the separate `gen2::decision` operation. The small caller interface hides
bundle validation, frontend processing, both graph sessions, recurrent state,
text reconstruction and worker lifecycle. The names below are now callable;
see `tools/bark/README.md` for the runnable example and packaging instructions.

```rust,ignore
let runtime = gen2::Runtime::builder().build()?;
let speech = runtime.load_transcriber(
    "/models/bark/onnx",
    gen2::speech::LoadOptions {
        native_library: Some("/runtime/libonnxruntime.dylib".into()),
        ..Default::default()
    },
)?;

// The first slice's input contract is already-normalized mono 16 kHz PCM.
let result = speech.transcribe(
    gen2::speech::Audio::mono_16khz(samples)?,
    gen2::speech::TranscribeOptions::default(),
)?;
println!("{}", result.text);
speech.shutdown();
```

`Transcription` should report text, input duration, model revision, bundle
digest, actual provider, and queue/frontend/encoder/decoder timings. Its text
is the model's output with the helper's final trim; add no second cleanup LLM
or unqualified text rewriting. Keep confidence and word timestamps out of the
initial contract: cleanup can omit words, and token frame indices do not
establish reliable word alignment.

`SpeechCapabilities` should report English, the measured duration envelope,
input formats/rates, cleanup behavior, CPU execution and `streaming = false`.
The encoder is not a proven streaming encoder. Microphone capture and endpoint
detection belong to the host; the host submits a completed utterance.

Use one worker per loaded transcriber, with a bounded queue and independently
reset decoder state per request. Return typed failures for invalid audio,
incompatible/corrupt bundles, missing runtime, queue saturation, cancellation,
deadline, shutdown and resource limits. Check cancellation between graph calls
and decoder iterations. Native execution can retain resources until it returns;
do not promise immediate interruption of an in-flight encoder call.

Implemented features: `backend-asr-onnx` and `asr-dynamic`, optional and off by
default. Internally share dynamic runtime initialization with Laya. ORT is
process-global: a separate ASR-local initialization lock would fail to detect
a library conflict with Laya. Preserve the existing `laya-*` feature spellings;
if necessary, make them aliases of a common internal dynamic-runtime feature.
Do not enable Laya merely to enable ASR.

The existing generation plugin interface would require fake chat/token
semantics for this model. Start with a private ASR adapter. Introduce a public
speech adapter seam when a second real adapter, such as MLX ASR, demonstrates
what actually varies. Keep ordinary `Runtime::load` routing intact.

## Roadmap

Each milestone produces an observable result and blocks its dependent work
until its exit gate passes. M1 and M2 form the first useful vertical slice.

| Milestone | Result | Exit gate | Depends on |
| --- | --- | --- | --- |
| M0 — reference and artifact qualification | Immutable package, Python CPU reference, graph/wordpiece inventory, runtime compatibility receipt | Real speech transcribes with both 1.23.2 and candidate 1.24.4; all differences explained | None |
| M1 — native ONNX proof | Rust runs the frontend, encoder and TDT decoder on mono 16 kHz PCM | Matches reference features, token decisions and final text on the frozen smoke corpus | M0 |
| M2 — Gen2 transcription module | `Runtime::load_transcriber`, reusable worker, bounded resources, meaningful capabilities | Public-interface tests prove repeatability, cancellation, shutdown and chat/Laya coexistence | M1 |
| M3 — desktop audio and provisioning | WAV input, qualified rates/downmixing, complete revision-pinned Hub download and offline reuse | All advertised audio inputs and bundle recovery tests pass; inference requires no network | M2 |
| M4 — desktop release qualification | Real-model CI evidence, quality/performance receipts and runnable example | macOS, Linux and Windows CPU gates pass with the real checkpoint | M3 |
| M5 — optional acceleration and mobile | Independently qualified execution providers/device packages | Provider placement, transcript quality, memory and physical-device evidence pass per target | M4 |
| M6 — MLX ASR | Second speech adapter using the same caller interface | Backend-specific oracle parity and measured Mac performance | M4; follows ONNX work |

### M0: establish the reference before designing around assumptions

Provision only `onnx/*`, `bark-release.json`, `infer.py` and the root license,
at the pinned revision. Anchor the variant manifest to the published release
digest `8cd94f24f677b74f9e92306039491f23e3e22863205e59a5bdda322026f98f8c`,
then validate every declared size/hash. Record the actual graph input/output
names, ranks, dtypes, dynamic dimensions, opsets, operator domains and external
tensor locations. Validate vocabulary IDs and blank token. Confirm all graph
paths resolve within the immutable downloaded package.

Run the documented Python reference under 1.23.2, then run the same pinned
reference under 1.24.4 in a separate environment. Compare transcript/token
outputs and timing/memory. Rust's `api-24` bindings must use the qualified
1.24 runtime; pointing them at 1.23.2 is not a compatibility strategy.

Instrument the reference to export mono PCM, feature tensors and lengths,
encoder output/lengths, and selected decoder logits, token/duration decisions
and recurrent states. Freeze a lock file for Python, NumPy, ONNX ASR, ORT,
soundfile, soxr and evaluation dependencies, including wheel hashes. This
instrumentation is a test tool, not shipped application behavior.

Test the pinned wheel's `nemo128` ONNX frontend against the default NumPy
frontend. If it passes the numeric and transcript gates, bundle the graph with
its digest and MIT attribution; this gives a native Rust path without an
early DSP reimplementation. Otherwise port the pinned NumPy frontend and its
filterbank coefficients. Record the chosen route before M1 proceeds.

Artifacts to add: `tools/bark/qualify_reference.py`,
`tools/bark/reference.lock`, `tests/fixtures/bark/reference.json`, and graph
metadata/qualification results. Large weights and evaluation audio stay in a
cache/artifact store, outside Git.

### M1: prove complete native transcription

Use CPU only, batch size one. Start with the reference thread settings.
Implement the verified frontend and the two graph sessions, then the pinned
TDT loop and vocabulary reconstruction. Match the reference's handling of
blank tokens, zero-duration outputs, maximum tokens per step, positive time
advances and state updates. Reset recurrent state for every utterance and
follow the vocabulary text reconstruction exactly.

Bound decoder iterations using encoded length and the qualified per-step
limit. Reject malformed/non-finite outputs, impossible lengths, invalid
token IDs and graph-state shape changes. A non-progressing decoder must
terminate with a typed failure.

Suggested implementation locations: `src/speech/{bundle,frontend,tdt,onnx}.rs`,
`tests/bark_native.rs`, and an initial `examples/bark_transcribe.rs`. Keep M1
small enough to prove real speech before adding format breadth or routing.

### M2: make it a reusable runtime operation

Add `src/speech/{mod,worker}.rs`, exports, the runtime load method and an error
conversion. Extract only shared ORT initialization/native-library policy into
an internal module; avoid making Laya preprocessing or graph contracts generic.

Extend admission accounting so speech, Laya and chat share one reservation
budget. Charge ASR for weights, ORT allocations, frontend/encoder workspaces,
decoder state and bounded queued input. Estimate from measured peak use at
the longest qualified recording, not the 488 MB package size. Retain the
reservation until sessions and in-flight work are dropped. Include ASR in
load, restore, eviction and runtime-stat paths without double counting
existing decision reservations.

Test native-runtime reuse with Laya loaded first and with ASR loaded first.
Different runtime paths must fail consistently; missing/incompatible libraries
must produce an actionable failure. After cancellation or shutdown, a new
load must not inherit decoder state or lose reservation accounting.

### M3: complete the desktop input and download workflow

Add WAV decoding and a PCM constructor carrying rate/channel metadata. Qualify
PCM16 and float32 WAV, 1–8 channels, and 8/16/44.1/48 kHz. Bound both encoded
file bytes and decoded samples before large allocation; apply the duration
limit at the original sample rate and verify it after resampling.

Use arithmetic channel averaging and compare the chosen native resampler
against the helper's soxr HQ outputs. Its dependency/packaging choice is an
M3 decision backed by measurements. Unsupported codecs/rates must fail clearly;
only advertise the combinations that pass. Keep capture outside the library.

Add a dedicated revision-pinned multi-file Hub provisioning operation. Reuse
cache/progress infrastructure where useful, but do not select a GGUF from this
repository. Download into staging, validate the complete package, and publish
it atomically as an immutable cache entry. Include both `.onnx.data` files.
Handle interrupted downloads, cache corruption and stale manifests. After
provisioning, local loading/transcription performs no network requests.

### M4: qualify and document a release

Provide a runnable example and documentation covering native-runtime
provisioning, model provisioning, repeated transcription, limits, errors and
measured CPU behavior. Add a speech-specific conformance lane; a synthetic
graph test proves plumbing, while a real Bark run proves compatibility.
Follow existing Laya CI's explicit runtime installation and cached artifacts.

Documentation must distinguish verified CPU targets from untested providers.
Keep the model's attribution with cached/distributed weights and preserve
third-party frontend notices. Record download size and measured residency
separately. Keep changes to the public interface documented in `api_spec.md`.

### M5/M6: expand only against independent evidence

Profile the actual encoder and decoder operator placement before qualifying
CUDA, DirectML or CoreML. ONNX provider registration is insufficient proof of
acceleration, especially for 4-bit/contrib operators. Record any CPU partitions,
copy overhead and fallback policy. A requested unavailable provider must fail;
an allowed fallback must be visible in results. [ORT quantization guidance][ort]

Physical iOS/Android qualification requires packaged runtime/provider support,
the exact graphs, real audio, memory/thermal measurements and lifecycle tests.
Desktop success or an emulator compile is not a mobile-support receipt.

MLX follows as a dedicated ASR implementation, qualified against the pinned
MLX Audio reference. The existing Gen2 MLX LLM loader does not implement this
architecture. Introduce the second adapter behind the same speech interface.
Long recordings, utterance chunking, streaming, diarization and word timestamps
need separate designs and qualification after the initial desktop release.

## Verification plan

### Frozen inputs and distinct oracles

Maintain three kinds of evidence:

1. **Implementation parity:** the pinned Python ONNX CPU path is the oracle for
   native behavior. Python and Rust use the same weights, ORT version, provider
   and thread settings in the main comparison.
2. **Model quality:** human reference transcripts are the oracle for accuracy.
   Matching Python cannot prove that the model preserves meaning.
3. **Runtime behavior:** public-interface tests and measured traces establish
   queueing, memory, cancellation, shutdown and actual provider execution.

Freeze at least 25 licensed, identified English smoke clips covering fluent
speech, fillers, false starts, corrections, repetition, short/long utterances,
silence/background noise, names/numbers, negation and punctuation. Use real
speech in addition to generated arithmetic waveforms. Store clip IDs, audio
hashes, original rate/channels, manually expected content and raw oracle
transcripts. Include independently annotated meaning-sensitive dictation
examples; the model card's illustrative phrases are not measured predictions.

For the full cleanup benchmark, pin DisfluencySpeech to revision
`b7da294fe3a70dd96df6640893f2a5dfc2c87638`, `test`, all 250 examples, and use
`transcript_c`. Acquire the test parquet rather than the whole training corpus.
Export deterministic row IDs and audio hashes. Report corpus WER, per-clip
errors and substitutions/deletions/insertions. Define and version the text
normalizer before scoring; compare raw transcripts separately.
[Dataset][dataset]

Add an independently sourced fluent/dictation corpus with labels covering
names, numerals and negation, and separate quiet/noisy slices. DisfluencySpeech
alone does not establish behavior outside its domain. Freeze this corpus before
tuning. Keep implementation regressions distinct from weaknesses shared by
the upstream model.

### Gates

| Layer | Required checks | Pass condition |
| --- | --- | --- |
| Package integrity | Missing external weights, byte/hash mismatch, corrupt config, invalid vocabulary, unsafe/undeclared external paths, wrong signature | Typed error before inference; complete pinned bundle opens |
| Runtime compatibility | Python 1.23.2 vs Python 1.24.4; native 1.24.4; absent/wrong library; Laya/ASR initialization in both orders | Reference compatibility established; conflicts fail predictably |
| PCM/frontend | Constant/silent/impulse/tone/noise, real PCM, duration edges, frontend frame count/layout and normalization | Exact lengths/layout; proposed `atol=1e-4`, `rtol=1e-4` for finite features against NumPy |
| Encoder/decoder | Encoder length/output layout, fixed decoder calls, state updates, blank and duration choices, 10-token limit | Exact integer decisions; proposed float `atol=1e-4`, `rtol=1e-4`; decoder cannot loop indefinitely |
| End-to-end parity | Frozen smoke corpus and all 250 benchmark clips under matched CPU conditions | Identical raw text after helper trim; investigate every mismatch before claiming parity |
| Audio workflow | WAV types/rates/channels, original/post-resample limits, NaN/Inf, empty/truncated files, unsupported codecs, tiny input | Advertised inputs match the oracle; rejected inputs fail without truncation, runaway allocation or panic |
| Quality | Cleaned-reference WER, independent dictation labels, silence hallucination and meaning-sensitive cases | Native full-corpus WER no worse than matched ONNX reference; critical-content fixtures pass both paths |
| Lifecycle | A–B–A requests, competing requests, queue overflow, cancel queued/running work, expired deadlines, shutdown during inference, reload | No cross-request state; stable failures; reservation retained until native resources release |
| Coexistence | ASR plus Laya plus GGUF chat within/over budget; load/restore failure paths | Aggregate admission enforced; reservations charged/released exactly once; existing operations still work |
| Offline/cache | Interrupted provisioning, corruption detection, warm cache, network disabled after provisioning | Complete cache recovery and zero inference-time downloads |

Numeric tolerances are **proposed starting gates**, not measurements. M0 must
confirm them on the reference assets. Do not widen a tolerance solely to make
an unexplained discrepancy pass. Export tensors and the first divergent decoder
step so a failure identifies the faulty part of the pipeline.

Silence and minimum-duration behavior need an explicit supported policy. If
the upstream oracle hallucinates on the chosen silent/noisy fixtures, record
that limitation and qualify a suppression policy separately; do not invent a
successful empty result or introduce hidden VAD while reporting oracle parity.

### Performance and memory

On a named reference desktop, measure cold load separately from warm inference.
For each 1/5/15/30/60-second duration, use at least one warmup and five timed
repetitions. Report median and p95, frontend/encoder/decoder time, queue time,
total inference time, real-time factor (`processing seconds / audio seconds`),
peak process memory and provider/thread settings. Measure process memory from
a process-start baseline, including native allocations. Run an A–B–A repetition
and a 100-request soak to detect state contamination or retained native memory.

Proposed desktop usefulness gate: warm RTF below 1.0 on the named target, and
median native inference no more than 1.25 times the matched Python CPU reference.
Compare from normalized PCM with the same frontend route for the fair core
comparison; report file decode/resample overhead separately. These are target
criteria, not claimed current performance. Re-run with chat/Laya residency to
measure contention and validate the conservative reservation.

The model card's 3.79% WER and 261 ms per minute are MLX measurements on an M5
Max, with short clips and already-loaded weights. They are contextual references,
not ONNX acceptance thresholds or a continuous 60-second measurement.
[Benchmark methodology][card]

### CI and receipts

| Lane | Trigger | Purpose |
| --- | --- | --- |
| Pure contract/DSP tests | Every PR, without native library or weights | Validation, bounds, vocabulary, deterministic decoder fixtures |
| Native synthetic smoke | Every PR on Linux/Windows/macOS | Runtime loading, two-graph wiring, lifecycle and typed errors |
| Real Bark smoke | Speech/runtime/dependency changes and explicit release qualification | Actual checkpoint transcription through the public interface |
| Full benchmark/soak | Scheduled/on demand; required for release | All 250 clips, independent quality cases, memory and performance receipts |
| Accelerated/device qualification | Per provider/device expansion | Actual placement and physical-target behavior |

A requested real-model qualification must fail if its weights, runtime or
fixtures are absent; skipped/ignored tests cannot produce a green compatibility
claim. PR-only synthetic coverage is not the real-model release gate. Full
benchmark results must correspond to the release code/model/dependency pins.

Save machine-readable receipts under a proposed
`benches/results/bark/<machine>/<run-id>/` location: Gen2 SHA, model revision and
manifest SHA, ORT native build/library hash, frontend route/hash, oracle lock
digest, corpus revision/audio hashes, normalizer version, OS/architecture/CPU,
provider partitions, threads, raw transcripts, numeric parity, WER and timing/
memory samples. Keep speech content out of ordinary production logs.

Implemented verification commands (set the documented real-model environment
variables first; use a fresh process for each independent real-model case):

```bash
cargo test --no-default-features --lib speech
cargo clippy --no-default-features --features asr-dynamic --all-targets -- -D warnings
cargo test --release --no-default-features --features asr-dynamic,laya-dynamic --test bark_native coexistence_laya_first -- --ignored --exact
cargo test --release --no-default-features --features asr-dynamic,laya-dynamic --test bark_native coexistence_speech_first -- --ignored --exact
# Explicit opt-in real-model lane; missing assets are errors inside the test.
cargo test --no-default-features --features asr-dynamic --test bark_native real_bark -- --ignored
```

Also run the existing default, no-backend and Laya gates, documentation/example
checks, and publish-shape checks when their respective files change. These
commands are implemented and exercised locally. A skipped/ignored ordinary-test
invocation is not a native compatibility pass. Cross-platform release gates
and final provenance-rich checked-in receipts remain required.

## Remaining release work

Resolve the Linux frontend drift and Windows experimental-resampling quality
failure, complete all three real-model desktop CPU CI gates, capture receipts
against the final merged implementation, complete review and publish-shape
checks, then non-force push the verified commit to remote main. Keep unqualified
targets and live-caption behavior explicit in the final release summary.

## Sources

[release]: https://huggingface.co/ConwayResearch/bark-0.6B-4bit-v2.0/blob/0d5f814105042fe801bb019dbf34b4e6f7d3eae6/bark-release.json
[helper]: https://huggingface.co/ConwayResearch/bark-0.6B-4bit-v2.0/blob/0d5f814105042fe801bb019dbf34b4e6f7d3eae6/infer.py
[card]: https://huggingface.co/ConwayResearch/bark-0.6B-4bit-v2.0/blob/0d5f814105042fe801bb019dbf34b4e6f7d3eae6/README.md
[loader]: https://github.com/istupakov/onnx-asr/blob/b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc/src/onnx_asr/loader.py
[numpy]: https://github.com/istupakov/onnx-asr/blob/b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc/src/onnx_asr/preprocessors/numpy_preprocessor.py
[preprocessor]: https://github.com/istupakov/onnx-asr/blob/b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc/src/onnx_asr/preprocessors/preprocessor.py
[nemo]: https://github.com/istupakov/onnx-asr/blob/b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc/src/onnx_asr/models/nemo.py
[decode]: https://github.com/istupakov/onnx-asr/blob/b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc/src/onnx_asr/asr.py
[dataset]: https://huggingface.co/datasets/amaai-lab/DisfluencySpeech/tree/b7da294fe3a70dd96df6640893f2a5dfc2c87638
[ort]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
