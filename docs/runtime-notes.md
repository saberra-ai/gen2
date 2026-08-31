gen2 runtime — Engine, Session, KV, Telemetry, and Pull-based Generation

Summary of Work Done
- Milestone 0: Scaffolding and API shells
  - Created Engine/Session/KV/Generation modules with compiling stubs.
  - Added basic types: LoadRequest, Settings, Capabilities, GenSpec, TokenEvent.
- Milestone 1: Model loading and capability discovery
  - Engine::load_model/reload_model using llama-cpp-2.
  - ModelBundle with meta (uuid, n_ctx, n_layer). Deterministic capability flags.
- Milestone 2: Settings ingestion and validation
  - Settings::validate with range checks; Engine keeps a version counter.
- Milestone 3: Sessions and pull-based generation (text)
  - Session builds context, pre-fills prompt via ChatTemplate, and returns an Iterator TokenPuller.
  - Cooperative controls: pause/resume/stop → TokenEvent::Paused/Stopped.
- Milestone 4: KV cache save/load
  - Versioned header with checksum and compatibility metadata.
  - Strict/Lenient load policies; actionable errors.
- Milestone 6 (observability):
  - Tracing spans and hook bus (HookBus + HookEvent) for load/prefill/decode/final stats.
- Milestone 5 groundwork (multimodal, Option B):
  - Structured media signaling: TokenEvent::MediaBoundary under feature "mm".
  - Capability guard for images; media detection in messages.

What’s Left To Be Done (M5 focus)
- MediaEncoder (feature "mm") using llama-cpp-2 MTMD:
  - Resolve image bytes (file://), encode with MTMD, and produce media tokens with dimension checks.
  - Validate mmproj at load with a tiny encode; set IMAGES only if verified.
- Template-aware media insertion (Option B):
  - Extend ChatTemplate inputs to carry media markers/slots.
  - Insert media token sequences at template-anchored positions during prefill and emit MediaBoundary precisely.
- Integration tests (ignored):
  - Gated by PIO_TEST_MODEL + PIO_TEST_MMPROJ to verify image+text flows end-to-end.

What Can Be Improved Next
- Tokenizer/template compatibility:
  - Replace heuristic tokenizer_digest/template_fingerprint with library-provided digests when available.
- Engine/session ergonomics:
  - Add seeds at engine/session/gen scopes; reflect in stats.
  - Tighten default ctx/batch sizing per model metadata.
- Testing:
  - Add trait abstractions to mock llama types in unit tests.
  - Add more property tests (e.g., Settings::validate).
- CLI:
  - Minimal llmctl demo commands for load/caps/chat/cache.

How To Use
- Load a model, start a session, pull tokens

  use pio_gen2::engine::{Engine, LoadRequest};
  use pio_gen2::session_rt::SessionSpec;
  use pio_gen2::generation::GenSpec;
  use pio_gen2::{Message, MessageBody, MessageContent};

  let engine = Engine::new();
  engine.load_model(LoadRequest { model_path: "/path/model.gguf".into(), ..Default::default() })?;
  let msgs = vec![
    Message { name: None, role: "user".into(), body: MessageBody::Content { content: MessageContent::SingleText("Hello".into()) }},
  ];
  let s = engine.start_session(SessionSpec { messages: msgs, ..Default::default() })?;
  let mut puller = s.pull(GenSpec { max_tokens: Some(32), ..Default::default() })?;
  while let Some(ev) = puller.next() {
    // match ev: Token(text), Paused, Stopped, Eos, and under feature "mm": MediaBoundary
  }

- Pause/stop controls
  - Call `s.pause()`, `s.resume()`, and `s.stop()`.
  - TokenPuller yields `Paused` and `Stopped` events.

- KV cache
  - Save after prefill:

    let snap = s.save_cache(pio_gen2::kv::KvSaveSpec::ToPath("my.kv".into()))?;

  - Load on new session (strict or lenient):

    let s2 = engine.start_session(SessionSpec {
      messages: msgs2,
      cache: Some(pio_gen2::kv::KvLoadSpec::Strict("my.kv".into())),
      ..Default::default()
    })?;

- Telemetry hooks
  - Register listeners to observe events:

    let hooks = engine.hooks();
    hooks.register(Arc::new(MyListener)); // implements HookListener

  - Events: EngineLoadStart/Ok, SessionPrefillStart/Ok, DecodeStep, FinalStats.

- Integration tests
  - Provide a small GGUF and run ignored tests:

    PIO_TEST_MODEL=/path/to/model.gguf cargo test -- --ignored --nocapture

- Multimodal (feature "mm")
  - Compilation: enable feature `mm` to include media types and guards.
  - Current behavior: Session emits MediaBoundary events for images found in messages; template keeps media text anchors.
  - MTMD-based token injection is marked TODO and will be added next.

Build Notes
- Tauri’s build script validates resources; when running tests or cargo check in CI/dev, use a minimal TAURI_CONFIG via env if needed.
- Features:
  - Default: text-only path.
  - mm: enables media signaling and guards; encoder/token injection will live behind this flag.
