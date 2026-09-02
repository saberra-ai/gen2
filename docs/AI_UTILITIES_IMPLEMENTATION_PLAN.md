# Gen2 AI Utilities Implementation Plan

Status: implementation-ready
Target branch: `main` (this repository has no `master` branch)
Scope: classification, structured extraction, embeddings, reranking, speech-to-text, OCR, and text-to-speech

## Goal

Turn gen2 from a chat/generation runtime into a small, embeddable Rust runtime for the AI primitives applications actually need, without turning the crate into a monolithic ML framework.

The end state should make these all feel native:

```rust
let engine = gen2::Engine::builder()
    .model("/models/chat.gguf")
    .embedder("/models/embedding.gguf")
    .reranker("/models/reranker.gguf")
    .build()?;

let label = engine
    .classify("this product is excellent")
    .labels(["positive", "negative", "neutral"])
    .label()?;

let invoice: Invoice = engine.extract(invoice_text).value()?;

let vectors = engine.embed(&documents)?;
let ranked = engine.rerank("rust embedded database", &documents)?;

let transcript = engine.transcribe_pcm(samples, 16_000)?;
let page = engine.ocr(image_bytes)?;
let speech = engine.speak("build finished").audio()?;
```

The public surface should stay small. Each utility should be a first-class operation, not a special prompt convention callers have to rediscover.

---

## Repository facts the implementation must preserve

The plan is based on the current architecture, not a clean-room redesign.

1. `src/api/engine.rs` is the ergonomic public facade. `Engine` talks to the controller over `ControllerCmd` and already exposes `load_embedder`, `embed`, and `embed_one`.
2. `src/controller/mod.rs` defines the wire-level command contract. It already separates model lifecycle, status, chat, system inference, and utility commands.
3. `src/controller/commands.rs` currently executes `GenerateEmbeddings` synchronously on the controller thread. That is acceptable for the existing small helper but is not acceptable for multi-second transcription/OCR/TTS workloads because it would stop chat token scheduling while the utility runs.
4. `src/backend/traits.rs` currently models embeddings as an optional capability of the *active generation backend* via `Backend::as_embeddings()`. That couples an auxiliary model to whichever backend happens to own the chat model.
5. `src/backend/facade.rs` follows that coupling: `load_embedder` and `generate_embeddings` probe the active backend. A chat model running under MLX therefore cannot independently use a llama.cpp embedding helper unless MLX itself implements the capability.
6. `src/backend/llama/embedder.rs` already contains the useful model-specific embedding work: EmbeddingGemma and Qwen3-Embedding family handling, pooling differences, Qwen EOS handling, MRL truncation, normalization, and llama.cpp context setup. Preserve this logic rather than reimplementing it from scratch.
7. `src/residency.rs` already thinks in terms of auxiliary runtimes. `RuntimeKind` has `Llm`, `Embedder`, `Stt`, and `Tts`, and the inventory already has separate helper slots plus helper idle/pressure eviction. Extend this model rather than creating a second memory policy.
8. `src/residency_policy.rs` already provides helper model memory estimation and a shared inference-resident budget.
9. `Capabilities` (`TEXT | IMAGES | AUDIO`) describes what the loaded *generative model can accept*. Do not overload `AUDIO` to mean “a speech-to-text helper happens to be installed.” Utility capabilities need their own status surface.
10. `.github/workflows/ci.yml` treats every backend/feature combination as an independently buildable product. New heavy utility dependencies must remain feature-gated and must get their own compile/test coverage.
11. The default crate feature is llama.cpp. Do not make Whisper, ONNX OCR, Kokoro, audio playback libraries, Python, or external services part of the default build.

---

# Architectural decision: auxiliary utilities get their own worker

Do **not** keep adding utility inference methods to `Backend`.

The generation backend owns one concern: generative sessions. Embedding, reranking, STT, OCR, and TTS are auxiliary runtimes with different model types, lifetimes, dependencies, and latency profiles.

Introduce a thread-confined auxiliary worker behind the existing controller:

```text
                         public Engine
                              |
                              v
                    +--------------------+
                    | primary controller |
                    | chat scheduling    |
                    | residency policy   |
                    +---------+----------+
                              |
               utility commands are forwarded
                              |
                              v
                    +--------------------+
                    | utility worker     |
                    |                    |
                    | embedder           |
                    | reranker           |
                    | stt                |
                    | ocr                |
                    | tts                |
                    +--------------------+
```

Why one worker first:

- It keeps non-`Send` native/FFI model state thread-confined, matching the design reason the primary backend is controller-thread-confined today.
- A long utility call no longer blocks token scheduling in `tick_active_chats`.
- Utility model ownership becomes independent of the active chat backend.
- It creates one stable seam for deterministic fake runtimes in tests.
- It is simpler than a thread per utility. If real workloads later show helper-to-helper contention, the worker can be sharded behind the same `UtilityHandle` without changing the public API.

### New internal module

Create:

```text
src/utilities/
  mod.rs
  worker.rs
  types.rs
  embedding.rs
  rerank.rs
  stt.rs
  ocr.rs
  tts.rs
```

The worker thread must create native runtimes **inside the worker thread**. Do not construct a potentially non-`Send` model on the primary controller thread and move it across.

A reasonable internal shape:

```rust
pub(crate) struct UtilityWorker {
    tx: std::sync::mpsc::Sender<UtilityCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

pub(crate) enum UtilityCmd {
    Embed { inputs: Vec<String>, resp: Sender<Result<Vec<Vec<f32>>, String>> },
    Rerank { query: String, documents: Vec<String>, resp: Sender<Result<Vec<RerankResult>, String>> },
    Transcribe { audio: AudioBuffer, options: TranscribeOptions, resp: Sender<Result<Transcription, String>> },
    Ocr { image: Vec<u8>, options: OcrOptions, resp: Sender<Result<OcrResult, String>> },
    Speak { text: String, options: SpeakOptions, resp: Sender<Result<AudioBuffer, String>> },
    // explicit load/unload commands per runtime
    Shutdown,
}
```

All payloads crossing into the worker must be owned. Do not put borrowed buffers or FFI pointers in `UtilityCmd`.

The worker can initially process helper jobs serially. The correctness contract is that chat inference remains responsive while a helper is working.

### Controller integration

Add the utility worker to `ControllerState` in `src/controller/state.rs`.

For utility *inference* commands, the primary controller should forward the caller's response sender directly to the utility worker and immediately return `ControlFlow::Continue`; it must not wait for the computation.

Utility model *loads* may remain synchronous in v1 because they are explicit lifecycle operations, just like `LoadModel`. Do not implement implicit first-use model loading in v1; otherwise a first transcription could unexpectedly freeze controller work during a large model load.

On controller shutdown, shut down and join the utility worker. Add `Drop` protection so a controller panic/failure does not leave a native worker thread alive.

---

# Public utility status: separate from generation capabilities

Add a small serializable status type, for example:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UtilityStatus {
    pub embedder: Option<LoadedUtility>,
    pub reranker: Option<LoadedUtility>,
    pub stt: Option<LoadedUtility>,
    pub ocr: Option<LoadedUtility>,
    pub tts: Option<LoadedUtility>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedUtility {
    pub name: String,
    pub estimated_resident_mb: u64,
}
```

The source of truth for “loaded” should be the controller's residency/runtime state, not `Capabilities`.

Add `ControllerCmd::GetUtilityStatus` and `Engine::utility_status()`.

Keep existing convenience methods such as `is_embedder_loaded()` for compatibility, but route them through the new state.

Do not remove the public `BackendCaps::embedding` field in this project. It is already public. Tighten its documentation/semantics if necessary, but use `UtilityStatus` for the actual independent helper state.

---

# Phase 1 — first-class classification and typed extraction

This phase needs no new model runtime and should land first.

## Files

Create:

```text
src/api/classify.rs
src/api/extract.rs
```

Update:

```text
src/api/mod.rs
src/lib.rs
README.md
```

## Classification

Public API target:

```rust
let label = engine
    .classify("The service was fantastic")
    .labels(["positive", "negative", "neutral"])
    .label()?;
```

Implementation rules:

- Build on `Engine::infer`; do not add a controller command.
- Require at least two non-empty labels and reject duplicates.
- Use grammar-constrained decoding, preferably a JSON-schema root string enum, so arbitrary model prose cannot escape the label set.
- Force deterministic decoding by default (`temperature = 0`) unless the builder explicitly overrides it.
- Keep output budget tiny.
- Return only the selected label. Do **not** invent a model “confidence” value.

## Typed extraction

Public API target:

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
}

let invoice: Invoice = engine.extract(text).value()?;
```

Implementation rules:

- `T: serde::de::DeserializeOwned + schemars::JsonSchema`.
- Generate the JSON schema from exactly the type the caller will deserialize.
- Feed it through the existing `GrammarSpec::JsonSchema` path.
- Use `Engine::infer`, not a new agent loop.
- Parse the final text with `serde_json`; distinguish generation failure from typed decode failure in the public error.
- Provide a `.prompt(...)`/`.instructions(...)` override, but make the zero-config prompt useful.

## Tests

Use the scripted engine/test seam. Tests must prove:

- classification can never return a label not supplied by the caller;
- duplicate/empty labels fail before inference;
- extraction sends a JSON grammar derived from `T`;
- valid JSON decodes to `T`;
- syntactically valid but type-invalid JSON returns a typed extraction error;
- README examples compile as doc tests.

Suggested commit:

```text
feat: add classify and typed extract utilities
```

---

# Phase 2 — utility worker + embedding migration

This is the architectural foundation for every non-generative helper.

## Files

Create `src/utilities/*` worker scaffolding and update:

```text
src/controller/state.rs
src/controller/mod.rs
src/controller/commands.rs
src/residency.rs
src/residency_policy.rs
src/api/engine.rs
src/backend/traits.rs
src/backend/facade.rs
src/backend/caps.rs
src/backend/llama/embedder.rs
src/lib.rs
```

## Migrate embedding ownership

Preserve the public API:

```rust
EngineBuilder::embedder(...)
EngineBuilder::embedder_kind(...)
Engine::load_embedder(...)
Engine::is_embedder_loaded()
Engine::embed(...)
Engine::embed_one(...)
```

Change only the internal owner.

Today `backend::Engine::load_embedder` asks the active chat backend for `as_embeddings()`. Replace the controller path with the independent utility worker.

The existing llama embedding implementation should be adapted into an independently constructible utility runtime. Preserve all existing family-specific behavior in `src/backend/llama/embedder.rs` or move it mechanically into `src/utilities/embedding/llama.rs`; do not rewrite the math during the ownership refactor.

A key acceptance test is:

```text
build with backend-mlx + backend-llamacpp
load an MLX chat model
load a GGUF embedding model
both remain usable simultaneously
```

That is the concrete proof that utility ownership is no longer coupled to the active chat backend.

## Avoid controller stalls

Change `ControllerCmd::GenerateEmbeddings` handling so it forwards to the utility worker rather than calling the embedder synchronously on the primary controller thread.

Add a deterministic concurrency test with a fake utility runtime that intentionally sleeps while a scripted chat is generating. Assert that chat tokens continue to be scheduled before the helper call completes.

## Residency changes

`Embedder` already exists in `RuntimeKind`/`ResidencyInventory`. Keep it.

When a helper inference request is successfully accepted for forwarding, touch its residency timestamp. On idle/pressure eviction, send an unload command to the utility worker instead of calling `state.engine.unload_embedder()`.

## Compatibility cleanup

`Backend::as_embeddings()` is an internal trait seam. Once all embedding tests run through the utility worker, either:

- remove the internal embedding upcast and leave `BackendCaps::embedding` as a compatibility/documentation field with accurate semantics, or
- retain it temporarily as a backend-native adapter but ensure the controller no longer depends on it.

Do not leave `external-api` claiming a working embedding implementation if its methods still return `Unimplemented`.

Suggested commits:

```text
refactor: add independent utility worker
refactor: move embeddings onto utility worker
```

---

# Phase 3 — reranking

Reranking is the first new model class and the best proof that gen2 is no longer only a chat runtime.

## Public API

Add:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

engine.load_reranker("/models/Qwen3-Reranker-0.6B.gguf")?;
let ranked = engine.rerank("query", &documents)?;
```

Requirements:

- `ranked` is descending by score.
- `index` always identifies the original document position.
- Empty document lists return an empty vector without invoking the runtime.
- Reject non-finite scores from backend implementations.
- Do not copy document text into every result; the caller already owns it.

Add builder support:

```rust
Engine::builder().reranker(path)
```

## Internal runtime

Add a dedicated `RerankerRuntime` trait inside `src/utilities` and a llama.cpp implementation.

Keep model-specific reranking details out of `backend::BackendSession`; reranking is not generation and must not manufacture a chat session.

Before coding the FFI path, inspect the exact `llama_cpp_2` revision pinned in `Cargo.toml` and verify the available rank/pooling APIs. If the binding is missing a required llama.cpp primitive, extend the pinned Rust binding minimally rather than emulating reranking with prompted generation.

## Controller and residency

Add commands:

```text
LoadReranker
IsRerankerLoaded (or derive from GetUtilityStatus)
Rerank
UnloadReranker (internal lifecycle is sufficient if public unload is not otherwise exposed)
```

Extend:

```rust
RuntimeKind::Reranker
ResidencyInventory::reranker
```

Include reranker in helper idle and pressure eviction loops.

## Tests

- fake reranker ordering/score contract;
- index preservation;
- empty inputs;
- load/admission/eviction lifecycle;
- chat continues ticking during a slow rerank;
- live llama reranker test behind an environment variable/model path, never a required CI download.

Suggested commit:

```text
feat: add local reranking utility
```

---

# Phase 4 — speech-to-text

Feature gate: `utility-stt`.

Do not add speech dependencies to the default build.

## Runtime choice

Use whisper.cpp through a Rust binding (or a minimal direct binding if the existing crate cannot satisfy the target matrix). The implementation must be native Rust/C/C++; no Python process, server, or shell command.

Create the model inside the utility worker thread.

## Public types

Keep gen2 out of the general-purpose media-decoder business. The core API should accept decoded PCM:

```rust
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub timestamps: bool,
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub language: Option<String>,
}
```

API:

```rust
engine.load_transcriber(path)?;
let out = engine.transcribe(AudioBuffer { samples, sample_rate }, options)?;
```

A small WAV convenience can be added if it stays lightweight, but MP3/M4A/FFmpeg-style decoding should not be required by the core runtime.

## Residency

`RuntimeKind::Stt` and the `stt` slot already exist. Wire them to real load/use/unload behavior.

## Tests

- PCM/sample-rate validation;
- deterministic fake transcript and segments;
- helper admission/idle eviction;
- controller remains responsive during slow transcription;
- feature-off API behavior produces a clear `FeatureUnsupported` error where applicable;
- live Whisper test is opt-in and uses a local fixture/model path.

Suggested commit:

```text
feat: add whisper speech-to-text utility
```

---

# Phase 5 — OCR

Feature gate: `utility-ocr`.

Prefer an ONNX implementation because the crate already has ONNX Runtime integration and image decoding. Do not require the generative `backend-onnx` feature merely to use OCR; both features can depend on the same optional `ort`/`ndarray` dependencies.

For example:

```toml
backend-onnx = ["dep:ort", "dep:ndarray"]
utility-ocr = ["dep:ort", "dep:ndarray"]
```

## Scope

Implement conventional OCR, not VLM prompting.

The loader should support the minimum model set needed by the chosen OCR family (typically detection + recognition, and optional orientation if required).

Public result:

```rust
#[derive(Debug, Clone)]
pub struct OcrBox {
    pub corners: [[f32; 2]; 4],
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub blocks: Vec<OcrBox>,
}
```

API:

```rust
engine.load_ocr(OcrLoadRequest { ... })?;
let result = engine.ocr(image_bytes)?;
```

Use the existing bounded image decode/preprocessing discipline. Never trust arbitrary image dimensions from untrusted input without limits.

## Residency

Add:

```rust
RuntimeKind::Ocr
ResidencyInventory::ocr
```

Include it in helper eviction.

For multi-file OCR model bundles, add a directory/multi-path resident-size estimator rather than recording a fabricated single-file number.

## Tests

- image size/format limits;
- deterministic fake box ordering;
- confidence must be finite and clamped/validated;
- multi-model load lifecycle;
- opt-in live fixture test with a tiny test image.

Suggested commit:

```text
feat: add local OCR utility
```

---

# Phase 6 — text-to-speech

Feature gate: `utility-tts`.

Target a small local model such as Kokoro, but perform a dependency spike before choosing the final Rust integration. Requirements for the accepted implementation:

- no Python runtime;
- no background service;
- no system audio/playback dependency;
- deterministic model/voice asset loading;
- returns audio samples to the caller instead of playing them.

## Public API

```rust
#[derive(Debug, Clone, Default)]
pub struct SpeakOptions {
    pub voice: Option<String>,
    pub speed: Option<f32>,
}

engine.load_tts(TtsLoadRequest { ... })?;
let audio: AudioBuffer = engine.speak("hello", SpeakOptions::default())?;
```

Reuse the same `AudioBuffer` type used by STT.

Validate text size and speed bounds before dispatch so a bad request does not reach native code.

## Residency

`RuntimeKind::Tts` and the `tts` slot already exist. Wire them to the actual worker runtime.

## Tests

- fake waveform contract;
- sample rate/samples non-empty for non-empty text;
- text/speed validation;
- idle/pressure eviction;
- opt-in live synthesis test.

Suggested commit:

```text
feat: add local text-to-speech utility
```

---

# Phase 7 — model tasks and the utility catalog

Do this after the runtime APIs work. Do not make model-catalog design block useful primitives.

## Common task vocabulary

Add to `src/types/model.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelTask {
    Generate,
    Embed,
    Rerank,
    Transcribe,
    Ocr,
    SynthesizeSpeech,
}
```

The existing generative zoo can be treated as implicitly `Generate` without rewriting every entry immediately.

## Separate utility catalog first

Do not force non-generative models into `ModelZooEntry` yet: the existing zoo assumes generation-family defaults (`family`, sampling, thinking, quant policy), and utility entries would make those assumptions false.

Add:

```text
resources/models/utilities.json
src/utilities/catalog.rs
```

A utility entry should contain:

```text
id
name
task
runtime/family
per-platform source/file(s)
minimum RAM
optional default options
```

Seed it with curated entries for:

- EmbeddingGemma-300M;
- Qwen3-Embedding-0.6B;
- Qwen3-Reranker-0.6B;
- Whisper tiny/base (and optionally small on larger machines);
- the selected OCR detection/recognition pair;
- the selected TTS model/voice assets.

Add `UtilityCatalog::recommended(task, &HardwareProfile)`.

The first catalog release should **select**, not silently download. gen2's current public model API is path-oriented; hidden network downloads would be a major new policy surface. Automatic fetching can be a later project built on an explicit cache/download API.

## Builder additions

Add explicit builder fields/methods:

```rust
.reranker(path)
.transcriber(path)
.ocr_model(...)
.tts_model(...)
```

Keep `.embedder(path)` working unchanged.

Suggested commit:

```text
feat: add utility model tasks and catalog
```

---

# Phase 8 — observability, docs, and CI hardening

## Observability

Extend controller runtime/observability snapshots so callers can inspect helper residency without exposing model payloads.

At minimum report:

```text
kind
name
estimated resident MB
last used timestamp
```

Add helper latency counters/spans with stable names, e.g.:

```text
gen2::utility::embed
gen2::utility::rerank
gen2::utility::stt
gen2::utility::ocr
gen2::utility::tts
```

Do not emit source text, image bytes, audio samples, API keys, or full model paths into telemetry by default.

## README

Add a top-level “AI utilities” section after the three generation entrypoints, showing:

- classify;
- typed extraction;
- embeddings;
- rerank;
- feature-gated STT/OCR/TTS.

Every Rust example should remain a doc test when practical.

## CI

Keep the default job unchanged in dependency scope.

Add feature jobs that at least compile + unit test each utility implementation without downloading models. Suggested matrix rows:

```text
ubuntu: utility-stt
ubuntu: utility-ocr
ubuntu: utility-tts
macos:  utility-stt (if the binding is platform-sensitive)
```

If a utility feature shares dependencies with an existing backend, also test the combination, e.g.:

```text
backend-onnx,utility-ocr
backend-llamacpp,utility-stt
```

Live model tests belong behind environment variables and/or the existing scheduled/manual pattern; PR CI must not pull gigabytes from Hugging Face.

Suggested commit:

```text
docs: document AI utility runtime
ci: cover utility feature matrix
```

---

# Required public error behavior

Do not expose raw native errors as the primary public contract.

Extend the existing API `Error` mapping with stable operation codes:

```text
classification_invalid_labels
extraction_decode_failed
utility_not_loaded
utility_feature_disabled
embedding_failed
rerank_failed
transcription_failed
ocr_failed
tts_failed
```

Native/backend details can remain in the message/source chain.

Bad inputs should fail before dispatch whenever possible.

---

# Required test architecture

Do not make unit tests depend on probabilistic real models.

Add deterministic fake utility factories/runtimes under `src/test_support` or `src/utilities/test_support`.

The production utility worker should accept an internal factory set when built in tests. A fake must be able to:

- return known embeddings;
- return known rerank scores;
- sleep for a specified duration;
- return a known transcript;
- return known OCR blocks;
- return a known waveform;
- fail load/inference on demand.

This is the utility equivalent of the existing scripted generation backend.

### Cross-cutting contract tests

The implementation is not complete until tests prove:

1. utility inference does not block chat scheduling;
2. changing/swapping the chat backend does not unload or change an independent helper;
3. helper idle eviction unloads the actual worker runtime;
4. memory-pressure eviction unloads helpers before the active foreground LLM;
5. helper load failure does not leave a residency slot marked loaded;
6. shutdown joins both primary controller and utility worker;
7. a failed `EngineBuilder::build` leaves no worker thread behind;
8. feature-disabled utilities fail clearly rather than panicking;
9. all public result scores/confidences/timestamps are validated before returning;
10. existing embedding API behavior remains source-compatible.

---

# File-by-file implementation map

| File/module | Change |
| --- | --- |
| `src/api/engine.rs` | Add utility lifecycle/call methods and builder fields; preserve existing embedding methods. |
| `src/api/classify.rs` | New constrained classification builder. |
| `src/api/extract.rs` | New typed JSON extraction builder. |
| `src/api/mod.rs`, `src/lib.rs` | Export the new public API/types. |
| `src/controller/mod.rs` | Add utility commands/status types. |
| `src/controller/commands.rs` | Forward heavy utility inference to worker; handle load/admission/status/eviction. |
| `src/controller/state.rs` | Own the utility worker plus existing residency inventory. |
| `src/utilities/worker.rs` | Thread, command loop, lifecycle, shutdown/join. |
| `src/utilities/types.rs` | Internal utility load/request types and shared validation. |
| `src/utilities/embedding.rs` | Independent embedding runtime adapter. |
| `src/utilities/rerank.rs` | Reranking runtime + llama implementation. |
| `src/utilities/stt.rs` | Feature-gated Whisper runtime. |
| `src/utilities/ocr.rs` | Feature-gated ONNX OCR pipeline. |
| `src/utilities/tts.rs` | Feature-gated TTS runtime. |
| `src/backend/llama/embedder.rs` | Reuse/move existing embedding implementation without changing model math. |
| `src/backend/traits.rs` | Stop making the controller depend on active-backend embeddings. |
| `src/backend/facade.rs` | Remove/delegate old embedder ownership after migration. |
| `src/backend/caps.rs` | Keep public compatibility; make embedding claim accurate. |
| `src/residency.rs` | Add `Reranker` and `Ocr`; wire real STT/TTS helper lifecycle. |
| `src/residency_policy.rs` | Add multi-file/directory helper size estimator where needed. |
| `src/types/model.rs` | Add `ModelTask`. |
| `src/utilities/catalog.rs` | Utility model catalog and recommendation. |
| `resources/models/utilities.json` | Curated helper model entries. |
| `Cargo.toml` | Add feature-gated native dependencies; do not enlarge default feature set. |
| `.github/workflows/ci.yml` | Utility feature matrix, no model downloads. |
| `README.md` | Public utility examples and feature documentation. |

---

# Explicit non-goals for this implementation

Do not expand scope into:

- Stable Diffusion / image generation;
- video generation;
- image editing;
- vector database/storage/indexing;
- RAG orchestration;
- document parsers for every file format;
- MP3/M4A/FFmpeg media transcoding;
- audio playback;
- remote utility routing protocols beyond what is necessary to preserve the current API;
- automatic internet model downloads.

Embeddings and reranking provide search primitives; the caller still chooses its index/store. OCR extracts text; the caller still chooses document ingestion. STT/TTS operate on audio buffers; the caller still owns capture/playback.

This boundary is what keeps gen2 an embeddable runtime instead of an application framework.

---

# Agent execution protocol

The implementing agent should follow this order exactly unless a compile/runtime discovery makes a step impossible.

1. Read the files named in the repository-facts and file-map sections before editing.
2. Run the current baseline tests before the first code change.
3. Implement Phase 1 and run the default checks.
4. Implement the utility worker and migrate embeddings before adding another model class.
5. Prove the non-blocking concurrency contract with a fake slow helper.
6. Add reranking.
7. Add STT, OCR, and TTS one feature at a time. A feature must compile/test independently before moving to the next.
8. Add the utility catalog only after the runtimes work by explicit path.
9. Update README/docs and CI last, after APIs have stabilized.
10. Run the full verification set below.
11. Re-read the diff specifically for accidental default-feature dependency growth, chat-controller blocking calls, public API drift, unbounded input allocation, leaked worker threads, and logging of user payloads.
12. Commit in coherent slices. Do not squash away useful architectural checkpoints unless required by repository policy.
13. Fetch/rebase onto the latest `origin/main` if remote main moved, rerun affected checks, and push normally. Never force-push main.

## Verification before push

At minimum:

```bash
find src -name '*.rs' -print0 | xargs -0 rustfmt --edition 2024 --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

Then run every new utility feature independently, using the same pattern as the CI backend matrix:

```bash
cargo clippy --all-targets --no-default-features --features <required-backend>,<utility-feature> -- -D warnings
cargo test --no-default-features --features <required-backend>,<utility-feature>
```

Also run the combinations introduced by the implementation (for example llama.cpp + MLX + independent embedder/reranker on Apple where supported, and ONNX backend + OCR feature).

No required test may download a production model.

## Push target

This repository's default branch is `main`; there is no `master` branch. The final successful implementation push should therefore be:

```bash
git push origin HEAD:main
```

Do not create a `master` branch merely to match colloquial wording.

---

# Definition of done

The project is done when a consumer can use one `gen2::Engine` to:

- run normal chat/agent inference exactly as before;
- classify and extract typed data using the already-loaded generative model;
- hold an embedding model independently of the active chat backend;
- rerank documents with a dedicated local ranker;
- transcribe PCM audio with a dedicated local STT model;
- OCR an image with a dedicated local OCR model;
- synthesize speech into an audio buffer with a dedicated local TTS model;
- inspect which helper runtimes are resident;
- survive helper eviction/reload cleanly under the existing memory governor;
- keep chat token scheduling responsive while a helper inference operation is running;
- compile the default crate without pulling any of the new heavy helper dependencies.

That is the boundary that turns gen2 into a general-purpose local AI runtime while preserving the architecture and ergonomics it already has.