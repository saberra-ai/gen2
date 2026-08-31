great brief — here’s the plan i’d ship if we were designing a robust execution wrapper atop your gguf/llama.cpp-based library. i’ll map every requirement to concrete components, describe the state model, and show the public API surface (traits + structs) i’d expose. no code yet; just the architecture and signatures so you can probe the design.

---

# high-level shape

We’ll add a thin, orthogonal “runtime shell” around your existing `llama_cpp_2` usage (I see `LlamaBackend`, `LlamaModel`, `LlamaContext`, `LlamaSampler`, `LlamaBatch`) and the MTMD example you included. The shell standardizes:

* **Model lifecycle** (load / reload / unload) with capabilities discovery
* **Settings ingest** (serde-driven, validated, hot-swappable)
* **Type-safe, pull-based token generation** (sync `Iterator` and async `Stream`)
* **Optional multimodal path** via MTMD/“mmproj” if present
* **KV cache save/load** with compatibility guards
* **Small, queryable state surface** for “is loaded?”, “does support X?”, etc.

The result is a single entrypoint object, `Engine`, that you construct once and then create lightweight, scoped `Session`s from. Sessions own prompts, media attachments, and KV segments and yield a **pull-based `TokenPuller`**. Hot reload swaps the underlying model bundle atomically without invalidating in-flight sessions.

```
┌───────── Engine ─────────┐
│  Arc<LlamaBackend>       │
│  AtomicPtr<ModelBundle>  │───┐  (hot-swap on reload)
│  SettingsRegistry        │   │
│  SessionRegistry         │◄──┘  (DashMap<SessionId, Weak<Session>>)
└──────────────────────────┘
            │
            ▼
      ┌────────────┐      ┌─────────────┐       ┌──────────────┐
      │  Session   │────► │ TokenPuller │ ◄──── │  KV Manager  │
      └────────────┘      └─────────────┘       └──────────────┘
           ▲                    │                         │
           │   (optional)       │ pull() -> TokenEvent    │ save() / load()
           │                    ▼                         ▼
           │              Sampler / Batch           Context snapshot
           │
           └── MediaEncoder (MTMD) if mmproj present
```

---

# core data model & traits

### ModelBundle

The currently-active, immutable model set; safe to share by `Arc`.

```rust
struct ModelBundle {
    model: LlamaModel,
    tokenizer: Arc<Tokenizer>, // or use model's tokenizer
    chat_template: ChatTemplate,
    ctx_params: LlamaContextParams,
    model_params: LlamaModelParams,
    capabilities: Capabilities,        // images/audio flags etc
    mmprog: Option<MmProgram>,         // parsed/validated if present
    meta: ModelMeta,                   // gguf metadata snapshot
}
```

* **Why immutable?** So sessions never see partial reloads, and we keep reloading simple: build a new `ModelBundle`, atomically swap a pointer.

### Capabilities

Derived from the GGUF metadata + optional mmproj.

```rust
bitflags::bitflags! {
    struct Capabilities: u32 {
        const TEXT   = 0b0001;
        const IMAGES = 0b0010;
        const AUDIO  = 0b0100;
    }
}
```

Discovery rules (deterministic, enforced at load):

* `TEXT` is always on (baseline).
* `IMAGES` iff a compatible vision/mmproj is found (via GGUF metadata or provided path) and MTMD init succeeds.
* `AUDIO` iff model/meta indicates audio front-end (e.g., Whisper-like or audio tokens) **or** mmproj declares audio pipeline.

### Engine

Long-lived orchestrator.

```rust
pub struct Engine {
    backend: Arc<LlamaBackend>,
    bundle: arc_swap::ArcSwap<ModelBundle>,       // atomic swap on reload
    sessions: DashMap<SessionId, Weak<Session>>,
    settings: SettingsRegistry,                   // validated runtime knobs
    hooks: HookBus,                               // tracing/telemetry
}
```

Key methods (map to your utilities):

```rust
impl Engine {
    pub fn load_model(&self, req: LoadRequest) -> Result<()>; // initial or switch
    pub fn reload_model(&self) -> Result<()>;
    pub fn is_model_loaded(&self) -> bool;
    pub fn does_model_support_images(&self) -> bool;
    pub fn does_model_support_audio(&self) -> bool;

    pub fn upload_settings(&self, settings: Settings) -> Result<()>;
    pub fn settings(&self) -> Settings; // snapshot (cheap Clone)

    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>>;

    // Optional: shutdown semantics, GC idle sessions, etc.
}
```

### LoadRequest & Settings

```rust
#[derive(Deserialize, Clone)]
pub struct LoadRequest {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub model_params: ModelParamsInput,    // subset of llama params you allow
    pub ctx_params: CtxParamsInput,        // n_ctx, seed, threads, etc
    pub template_override: Option<ChatTemplateSpec>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    pub sampling: SamplingSettings,        // top_p, top_k, temp, mirostat, ...
    pub stopping: StoppingSettings,        // stopwords, n_predict, max_tokens
    pub system:   SystemSettings,          // threads, batch sizes, gpu layers (if dynamic)
    pub mm:       MmSettings,              // vision frame size, audio fps, etc
}
```

* **Upload settings** means accept JSON/TOML/YAML, `serde`-deserialize, validate (range checks), then **store as a versioned snapshot**. Sessions read “the current settings snapshot” at start and can opt-in to live updates per field.

### Session

Per-conversation execution context; owns `LlamaContext`, KV segments, sampler, and optional media pre-encoders.

```rust
pub struct Session {
    id: SessionId,
    engine: Weak<Engine>,
    bundle: Arc<ModelBundle>,            // frozen at creation
    ctx: LlamaContext,                   // tied to bundle.ctx_params
    sampler: LlamaSampler,
    kv: KvManager,
    mm: Option<MediaEncoder>,            // if images/audio supported
    stats: ExecutionStats,               // prompt eval, decode, tokens/sec...
}
```

Create with:

```rust
pub struct SessionSpec {
    pub messages: Vec<Message>,                 // aligns with your types
    pub attachments: Vec<Attachment>,           // images/audio (optional)
    pub cache: Option<KvLoadSpec>,              // load policy
    pub overrides: Option<Settings>,            // per-session overrides
}

impl Session {
    pub fn pull(&self, gen: GenSpec) -> Result<TokenPuller>;
    pub fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot>;
    pub fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport>;

    pub fn stats(&self) -> ExecutionStats;
    pub fn pause(&self);    // cooperates with puller
    pub fn resume(&self);
    pub fn stop(&self);     // cooperative stop
}
```

---

# pull-based generation (type-safe)

We expose **two surfaces**:

1. A blocking iterator:

```rust
pub struct TokenPuller { /* holds a Session-scoped decode state */ }

impl Iterator for TokenPuller {
    type Item = Result<TokenEvent>;
    fn next(&mut self) -> Option<Self::Item>;
}
```

2. An async stream (when `tokio` is enabled):

```rust
impl TokenPuller {
    pub fn into_stream(self) -> impl futures_core::Stream<Item = Result<TokenEvent>>;
}
```

### why “pull”?

* **Deterministic control & backpressure**: the caller advances decoding when ready.
* **Cooperative pause/stop**: `pause()` and `stop()` flip a state machine; `next()` observes it and returns `Stop`.
* **Integrates with sync and async callers** without thread juggling.

### TokenEvent: type-safety for more than plain text

```rust
pub enum TokenEvent {
    Token(Token),                     // text token with id + utf8
    Special(SpecialToken),            // BOS/EOS/SEP or custom
    MediaBoundary(MediaBoundary),     // marks where media tokens start/end
    Tool(CallSpec),                   // if you add tool-calls later
    Eos(StopReason, FinalStats),
}

pub struct Token {
    pub id: u32,
    pub text: SmolStr,               // small-string optimized
    pub logprob: Option<f32>,
}

pub enum SpecialToken { Bos, Eos, Sep, Pad, Custom(u32) }

pub enum MediaBoundary { BeginImage { idx: usize }, EndImage { idx: usize },
                         BeginAudio { idx: usize }, EndAudio { idx: usize } }
```

* This matches your requirement to be “type safe” and future-proofs non-text yields. No stringly-typed sentinel tokens.

### Generation spec

```rust
pub struct GenSpec {
    pub max_tokens: usize,
    pub sampling: Option<SamplingSettings>,
    pub stopping: Option<StoppingSettings>,
    pub bias: Option<LogitBias>,          // your `logit_bias.rs` maps nicely
    pub seed: Option<u64>,                // per-run override
}
```

* If omitted, fields inherit from `Session` overrides, which in turn inherit from `Engine` settings snapshot at the moment `Session` was created.

---

# multimodal path (images/audio)

We encapsulate the MTMD / mmproj specifics behind a small trait so the rest of the engine doesn’t care if attachments are images, audio, or both.

```rust
trait MediaEncoder: Send + Sync {
    fn encode(&self, att: &Attachment, tok_cfg: &TokenizerCfg) -> Result<EncodedMedia>;
}

enum Attachment {
    Image(ImageBytes, ImageSpec),     // raw bytes + H/W/channels
    Audio(AudioBytes, AudioSpec),     // raw PCM + rate/channels
}

struct EncodedMedia {
    // representation compatible with llama.cpp’s MTMD pipeline
    mtmd_bitmap: Option<MtmdBitmap>,
    tokens: Vec<u32>,                 // media tokens/markers
    prefill_text: Option<String>,     // if the template needs textual anchors
}
```

* **Capability guard**: if you call `start_session` with `attachments` but the active `ModelBundle.capabilities` lacks the matching bit, you get a `CapabilityError` with a *fix hint* (e.g., “load an mmproj”).
* **Template integration**: your `ChatTemplate` already arranges messages; we extend it to embed media markers when `EncodedMedia` is present.

---

# KV cache save/load (robustness-first)

We provide explicit, versioned snapshots that carry compatibility metadata.

```rust
pub struct KvSnapshot {
    pub tokens_covered: usize,
    pub bytes: Bytes,                 // portable blob
    pub meta: KvMeta,
}

#[derive(Clone)]
pub struct KvMeta {
    pub model_uuid: Uuid,             // stable per GGUF file (hash)
    pub n_ctx: u32,
    pub n_layer: u32,
    pub rope_scale: f32,
    pub tokenizer_digest: [u8; 32],   // to detect tokenizer mismatch
    pub template_fingerprint: u64,
    pub created_at_us: i64,
}
```

**Save:**

* We serialize via the underlying `LlamaContext` snapshot function(s) (or chunk the KV region if supported), prepend a compact header with `KvMeta`, and write to disk or return in-memory `Bytes`.
* We store a SHA-256 of the KV payload for corruption checks.

**Load:**

* Validate **all** meta fields against the current `ModelBundle`; if anything mismatches (common case: different model, different n\_ctx, different tokenizer), return `KvIncompatible` with a clear reason.
* Support **prefix reuse**: if the cache covers only the system/priming tokens, we still load and continue eval with remaining prompt tokens.

**Policies** (`KvLoadSpec`):

* `Strict(Path)`: must match; otherwise error.
* `Lenient(Path)`: attempt; on incompatibility, continue without cache.
* `Auto(Path, Key)`: choose among a sharded cache directory by hashed prompt prefix.

---

# lifecycle, concurrency & reload

### State machine (per Engine)

```
Unloaded → Loading → Loaded → Reloading → Loaded
           ↘───────→ Error (recoverable) ↗
```

* `load_model()` builds a fresh `ModelBundle` (mmap GGUF, init tokenizer, inspect metadata, try to init mmproj to decide capabilities). If successful, `ArcSwap` the pointer.
* **Sessions pin the bundle they were created with**. Reload won’t invalidate them. New sessions see the new bundle.
* A tiny GC periodically prunes dead `Weak<Session>` entries in the registry.

### Threading

* One `LlamaContext` per `Session` (matches llama.cpp best practices).
* `TokenPuller` is not `Send` by default (it touches the context). We can offer an opt-in “background decode” variant that runs decoding on a dedicated thread and communicates via a bounded channel; the *API* remains pull-based from the caller point of view.

### Backpressure

* The iterator produces **exactly one token per `next()`**. If you opt into background decode, the channel is bounded (size N), so memory use is controlled.

---

# errors & observability

### Error taxonomy (using `thiserror`)

* `LoadError` (IO, mmap, GGUF parse, mmproj mismatch)
* `CapabilityError` (media attempted but unsupported)
* `SettingsError` (validation, unknown field)
* `GenerationError` (sampler failures, invalid state)
* `KvError` (`Incompatible { reason }`, IO, Corrupt, Deserialize)

All `Error`s carry `Context` (model path, session id) for actionable logs.

### Telemetry

* We already see `ExecutionStats` in your code; we’ll extend it:

  * `prompt_tokens`, `eval_us`, `decode_us`, `first_token_us`, `avg_tps`, `max_rss`, `gpu_ms`.
* Emit events via `tracing` with structured fields. Hook bus lets consumers add sinks (prometheus, logs, custom).

---

# documented public API

This is the surface I’d publish (names align with your repo’s vibe and types):

```rust
pub struct Engine { /* as above */ }

impl Engine {
    pub fn new(backend: Arc<LlamaBackend>) -> Self;

    // 1) dynamically load models / optional mmproj
    pub fn load_model(&self, req: LoadRequest) -> Result<()>;
    pub fn reload_model(&self) -> Result<()>;

    // 2) upload settings
    pub fn upload_settings(&self, settings: Settings) -> Result<()>;
    pub fn settings(&self) -> Settings;

    // 3) type-safe pull-based generation
    pub fn start_session(&self, spec: SessionSpec) -> Result<Arc<Session>>;

    // 6) utils
    pub fn is_model_loaded(&self) -> bool;
    pub fn does_model_support_images(&self) -> bool;
    pub fn does_model_support_audio(&self) -> bool;
}

impl Session {
    // Begin decoding with explicit generation spec (pull-based)
    pub fn pull(&self, gen: GenSpec) -> Result<TokenPuller>;

    // 5) KV cache
    pub fn save_cache(&self, dst: KvSaveSpec) -> Result<KvSnapshot>;
    pub fn load_cache(&self, src: KvLoadSpec) -> Result<KvLoadReport>;

    // convenience
    pub fn stats(&self) -> ExecutionStats;
    pub fn pause(&self);
    pub fn resume(&self);
    pub fn stop(&self);
}
```

---

# capability discovery (deterministic rules)

At model load:

1. Read GGUF metadata keys (architecture, tokenizer, rope config, context size, special tokens).
2. If `mmproj_path` provided, parse and validate dimension compatibility (vision/audio embedding size ↔ model mm head); try to init an `MtmdContext` with your default marker to hard-verify. If it fails, **do not** set `IMAGES`/`AUDIO`; return `LoadError::MmprojIncompatible`.
3. For **images** support: either `mmproj_path` is valid or GGUF announces a built-in vision adapter. For **audio** support: similar check for audio front-end in metadata.
4. Record a **`model_uuid`** = stable hash of the GGUF file (size + crc or BLAKE3) and tokenizer digest for KV compatibility.

This keeps `doesModelSupportImages()` and `doesModelSupportAudio()` fully predictable and testable.

---

# settings ingestion & validation

* `Settings` (serde) has `TryFrom<RawSettings>` with comprehensive validation:

  * `0.0 < temp ≤ 2.0`, `0 ≤ top_k ≤ 100_000`, `0.0 ≤ top_p ≤ 1.0`, etc.
  * Stopwords enforce UTF-8 and map to valid tokenizer token sets.
  * Mirostat parameters mutual exclusivity with nucleus sampling (if you enforce).
* Keep a `SettingsRegistry` internally as a single `Arc<RwLock<Settings>>` plus a monotonic version counter; sessions capture the version they started with (visible in stats).

---

# sample flows

### A. load model and generate text

1. `engine.load_model({ model_path, mmproj_path: None, ... })`
2. `engine.upload_settings(defaults)` (optional)
3. `let s = engine.start_session({ messages, attachments: vec![], cache: None, overrides: None })?;`
4. `let mut puller = s.pull(GenSpec { max_tokens: 256, ..Default::default() })?;`
5. `while let Some(ev) = puller.next() { ... }`

### B. image+text

* Same as A, but `attachments` includes one or more images; if not supported: `CapabilityError` with hint “Load mmproj or choose a VLM”.

### C. KV cache warming

* Start a session with `cache: Some(KvLoadSpec::Lenient(path))`.
* After prompt prefill, call `save_cache(KvSaveSpec::ToPath(dir))` to persist.

### D. reload without dropping active sessions

* `engine.reload_model()` builds a new `ModelBundle` and atomically swaps it in; in-flight sessions keep decoding with the pinned old bundle/context. New sessions use the fresh bundle.

---

# safety & pitfalls addressed

* **Resource leaks**: Every `Session` owns exactly one context; drop frees GPU/CPU memory deterministically.
* **Hot reload races**: Solved by `ArcSwap<ModelBundle>` and pinning per session.
* **KV corruption & mismatch**: Metadata-guarded; strict vs lenient modes are explicit.
* **Backpressure**: Pull API avoids unbounded queues; optional background decode uses bounded channels.
* **Determinism**: Seeds accepted at three scopes (engine default → session override → gen override). We reflect the final seed in `ExecutionStats`.
* **iOS mlock**: Respect your conditional toggles (I noticed mlock adjustments in your code); expose a setting to disable mlock when needed.

---

# tests i’d add

* **Load/Reload**: model swap keeps existing session alive and consistent.
* **KV Roundtrip**: save, drop session, new session load + continue; negative tests for incompatible KV.
* **Capabilities**: with/without mmproj; image attachment rejected/accepted.
* **Sampling Boundaries**: stopword hits, max\_tokens, EOS token, special token routing.
* **Throughput invariants**: first token latency and tokens/sec within expected bands (flaky-tolerant).
* **Pause/Resume/Stop**: ensure `TokenPuller::next()` respects state flips.

---

# why this is the right abstraction

* **Minimal surface, maximal composability**: `Engine` + `Session` + `TokenPuller` map cleanly onto llama.cpp’s ownership model (backend/model/context) without hiding too much or leaking low-level details.
* **Type safety where it matters**: `TokenEvent` prevents sentinel strings and leaves room for tool calls or structured multimodal yields.
* **Operationally friendly**: deterministic capability flags, explicit errors, strong KV compatibility checks, straightforward reload semantics.
* **Future-proof**: adding “tools”, “function calling”, or “speculative decoding”: just add new `TokenEvent` variants or a `GenSpec` toggle; the rest of the design stands.

If this direction looks sound, I can translate it into concrete Rust modules and wire it to your existing `runner.rs`/`mtmd.rs` pieces (e.g., reusing your `ChatTemplate`, `ExecutionStats`, `Message`, `LlamaSampler`), keeping the public API stable and well-documented.
