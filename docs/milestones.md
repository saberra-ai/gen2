awesome — here’s a no-surprises implementation plan that you can ship in small, independently verifiable slices. each slice ends with concrete demo scripts + acceptance tests so you can validate before moving on. no code here, just the execution map.

---

# guiding principles

* short feedback loops: each milestone compiles, runs, and exposes a minimal CLI + sample usage.
* additive diffs: new modules behind traits/feature flags; no breaking changes to earlier milestones.
* test-first on boundaries: capabilities detection, KV compatibility, token event typing.

---

# milestone 0 — scaffolding & API shells (1–2 PRs)

**Goal:** land the skeleton types and traits without behavior.

**Scope**

* Create crates/modules:

  * `engine/` (`Engine`, `LoadRequest`, `Settings`, `Capabilities`)
  * `bundle/` (`ModelBundle`, `ModelMeta`)
  * `session_rt/` (`Session`, `SessionSpec`)
  * `gen/` (`TokenPuller`, `TokenEvent`, `GenSpec`)
  * `kv/` (`KvSnapshot`, `KvMeta`, `KvSaveSpec`, `KvLoadSpec`)
  * `media/` (`MediaEncoder`, `Attachment`, `EncodedMedia`) behind `cfg(feature = "mm")`
* Wire `Engine::new`, `is_model_loaded()`, `doesModelSupport*()` to return defaults (false).
* Introduce error enum (`thiserror`) and `ExecutionStats` placeholder.
* Add `arc-swap`, `dashmap`, `tracing` dependencies.

**Validation**

* **Unit:** type compiles; `cargo check --features mm` succeeds.
* **Doc tests:** minimal examples compile.
* **CLI:** `llmctl version` prints component versions.

**Acceptance**

* CI green with feature matrix: `default`, `--features mm`.

---

# milestone 1 — model loading & capability discovery

**Goal:** dynamic load/unload and deterministic capabilities.

**Scope**

* Implement `Engine::load_model(LoadRequest)`:

  * mmap GGUF into `LlamaModel`
  * build `ModelBundle` with `ModelMeta` (hash, tokenizer digest, rope, n\_ctx)
  * detect capabilities (TEXT default; IMAGES/AUDIO if mmproj valid)
  * swap via `ArcSwap`
* Implement `reload_model()` (re-run last `LoadRequest`).
* Implement utils:

  * `is_model_loaded`, `doesModelSupportImages`, `doesModelSupportAudio`.

**Validation**

* **Unit:** parser for GGUF keys → `Capabilities`.
* **Integration:** load a plain text model; load with bogus mmproj → `LoadError::MmprojIncompatible`.
* **CLI demo:**

  * `llmctl load --model path.gguf` → prints meta & caps
  * `llmctl caps` → shows `{text:true, images:false, audio:false}`

**Acceptance**

* Loading never invalidates existing `Arc<ModelBundle>` in tests (pin old, reload, assert both co-exist).

---

# milestone 2 — settings ingestion & validation

**Goal:** upload settings that are range-checked and versioned.

**Scope**

* Define `Settings` (sampling, stopping, system, mm).
* `Engine::upload_settings(Settings)` stores to `Arc<RwLock<Settings>>` + version counter.
* `TryFrom<RawSettings>` with validations (temp, top\_p, top\_k, stopwords tokenizable).

**Validation**

* **Unit:** property tests for boundaries (quickcheck/arbitrary).
* **CLI demo:**

  * `llmctl settings apply settings.toml`
  * `llmctl settings show` prints snapshot + version.

**Acceptance**

* Invalid field produces `SettingsError` with helpful message.
* Settings roundtrip (serde) stable.

---

# milestone 3 — sessions & pull-based generation (text-only)

**Goal:** start a session and pull tokens via an iterator/stream.

**Scope**

* `Engine::start_session(SessionSpec)`:

  * pin current `ModelBundle`
  * create `LlamaContext` with `ctx_params`
  * build `LlamaSampler` from settings snapshot (+ overrides)
  * prefill messages using existing `ChatTemplate`
* Implement `Session::pull(GenSpec) -> TokenPuller`:

  * `Iterator<Item = Result<TokenEvent>>` yields `Token(Token{id,text,logprob?})`
  * `Eos(StopReason, FinalStats)` at end
* Add cooperative `pause/resume/stop` flags observed by `TokenPuller::next()`.

**Validation**

* **Unit:** state machine for pause/resume/stop.
* **Integration:** small prompt, assert first token latency < threshold on CI runners (loose).
* **CLI demo:**

  * `llmctl chat "Hello"` → streams tokens as lines
  * `llmctl chat --max-tokens 8` → stops at 8 with stats

**Acceptance**

* Backpressure test: `TokenPuller` produces exactly one token per `next()`.
* Async feature: `into_stream()` behind `tokio` feature compiles and streams.

---

# milestone 4 — KV cache save/load (strict + lenient)

**Goal:** persist/restore KV with compatibility guards.

**Scope**

* `Session::save_cache(KvSaveSpec) -> KvSnapshot`:

  * capture KV, prepend `KvMeta`, compute payload SHA-256
* `Session::load_cache(KvLoadSpec) -> KvLoadReport`:

  * validate `model_uuid`, `n_ctx`, tokenizer digest, template fingerprint
  * strict vs lenient behavior
* Prefix reuse: allow partial coverage (system/preamble cache).

**Validation**

* **Unit:** meta mismatch returns `KvError::Incompatible { reason }`.
* **Integration:**

  * Save after system+init prompt; new session loads & continues with shorter prefill time.
  * Corrupt payload → `KvError::Corrupt`.
* **CLI demo:**

  * `llmctl cache save --to my.kv`
  * `llmctl cache load --from my.kv --lenient`

**Acceptance**

* Perf delta: cached prefill path faster than cold start by measurable margin (loose %).

---

# milestone 5 — multimodal plumbing (feature-flagged)

**Goal:** attach images/audio if supported; remain a no-op otherwise.

**Scope**

* Implement `MediaEncoder` using your `mtmd.rs`:

  * `encode(Attachment)` → tokens + optional mtmd bitmaps
* Extend `ChatTemplate` path to insert media markers.
* Capability guard: `start_session` rejects attachments if unsupported (`CapabilityError` with hint).

**Validation**

* **Unit:** encoder dimension checks; graceful errors.
* **Integration:** when mmproj present:

  * image + text prompt produces `MediaBoundary` events then text tokens.
* **CLI demo:**

  * `llmctl chat --image cat.jpg "Describe"` prints a short caption.

**Acceptance**

* Feature flag off: attaching media returns explicit error stating missing feature.

---

# milestone 6 — observability, errors, and hooks

**Goal:** structured logs, metrics, and stable error taxonomy.

**Scope**

* Emit `tracing` spans: `engine.load`, `session.prefill`, `decode.step`.
* Populate `ExecutionStats`: `prompt_tokens`, `first_token_us`, `decode_tokens`, `avg_tps`, `max_rss`.
* Finalize error variants and messages.
* HookBus: opt-in listeners for events (metrics sink).

**Validation**

* **Unit:** serialize `ExecutionStats` as JSON; log snapshot includes IDs.
* **Integration:** capture a run and verify span nesting.
* **CLI demo:**

  * `llmctl chat --stats json` prints final stats blob.

**Acceptance**

* Error messages actionable (path, cause, hint) in tests.

---

# milestone 7 — hot reload invariants & session pinning

**Goal:** prove reload doesn’t break in-flight sessions.

**Scope**

* Keep `LoadRequest` cached; `reload_model()` builds new bundle & swaps.
* Sessions hold `Arc<ModelBundle>`; continue decoding on the old one.

**Validation**

* **Integration test:** start session, begin decoding; in parallel call `reload_model()` with a different model; assert the ongoing session completes; new session uses new bundle.
* **CLI demo:**

  * Start streaming; in another shell `llmctl reload`; verify stream unaffected.

**Acceptance**

* No panics or memory leaks detected by `valgrind`/`address sanitizer` on CI variant.

---

# milestone 8 — polish: bias, stopwords, seed, ABI/docs

**Goal:** round out GenSpec and docs.

**Scope**

* Implement `GenSpec.bias`, `stopping.stopwords`, `seed` precedence (engine < session < gen).
* Write rustdoc + README with examples and guarantees.
* Stabilize the public API: mark `#[non_exhaustive]` where wise.

**Validation**

* **Unit:** precedence tests, stopword hit stops generation.
* **CLI demo:**

  * `llmctl chat --seed 42` deterministic output snapshot.

**Acceptance**

* API review: surface area documented; semver plan noted.

---

## cross-cutting test matrix

* OS: Linux, macOS (arm64/x86\_64)
* Features: `default`, `--features mm`, `--no-default-features`
* Models: tiny GGUF (for CI), medium text model (developer), optional VLM + mmproj
* Thread counts: 1, N (cpu), gpu-layers (if applicable)
* KV policies: `Strict`, `Lenient`, corrupt, wrong model, tokenizer mismatch
* Concurrency: multiple sessions, simultaneous pullers, reload during decode

---

## CI & tooling

* **Jobs:** `cargo check`, `cargo clippy -D warnings`, `cargo test --all-features`, `cargo doc`.
* **Perf smoke:** run a tiny prompt; assert `first_token_us` < ceiling, `avg_tps` > floor (loose).
* **Sanitizers:** one nightly job with `-Zsanitizer=address` for integration tests.
* **Artifacts:** publish `llmctl` binary + example configs.

---

## risk ledger & mitigations

* **Tokenizer digest false-negatives:** hash both vocab and merges; include gguf tokenizer metadata bytes.
* **Template drift:** embed `template_fingerprint`; invalidate KV when template changes.
* **Backpressure bugs:** default to pull-only; background decode behind explicit feature + bounded channels.
* **mmproj mismatch subtleties:** do a live encode of a 1×1 dummy sample at load to hard-validate shape.

---

## demo scripts to validate each milestone

* m0: `llmctl version`
* m1: `llmctl load --model X && llmctl caps`
* m2: `llmctl settings apply examples/fast.toml && show`
* m3: `llmctl chat "Write a haiku" --max-tokens 24`
* m4: `llmctl cache save a.kv && llmctl cache load a.kv --lenient && chat`
* m5: `llmctl chat --image tests/cat.jpg "Describe"`
* m6: `llmctl chat "stats please" --stats json`
* m7: two shells: stream + reload; start a new session and see it bind to the new model
* m8: `llmctl chat --seed 123 --stop "###"` snapshot test

---

if you want, i can turn this into a tracked checklist (issues/PRs) with owners, or produce the initial module files with rustdoc stubs so your team can parallelize milestones 1–3.
