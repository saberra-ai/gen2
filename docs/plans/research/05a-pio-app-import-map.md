# 05a — pio-app import map against `gen2::compat`

**Slice:** S5.1. **Date:** 2026-09-06. **Read-only** on pio-app
(`/Users/victor/workspace/pio-app` @ `64f220da`, the same tree receipt 05
inventoried). gen2 branch `s5.1-compat`.

Receipt 05 counted the host's imports statically and estimated ≈204 `use`-level
errors on a naive swap. This document is the exact list — every path pio-app
names, what it resolves to today, and where `gen2::compat` puts it — and the
residue that no re-export can cover.

## 1. Method

`pio-core/src`, `pio-core/tests`, `src-tauri`, `pio-daemon`, `pio-bridge`,
`pio-embed-seed`, and the four satellite Tauri crates, excluding
`pio-core/src/gen2/` itself and every `target/`. A parser expands each
`use …gen2::{…};` statement (brace groups, `as` aliases, multi-line) into
leaves; every other `gen2::…` token is an inline mention. `crate::gen2::` and
`pio_core::gen2::` are the same surface addressed from inside and outside
pio-core, so they are counted together and broken out below.

| | Count |
| --- | ---: |
| Files naming `gen2::` | **132** |
| `use` statements | 281 (155 `crate::gen2::`, 125 `pio_core::gen2::`, 1 other) |
| `use` leaves | **445** (77 distinct) |
| Inline mentions | **283** (208 `crate::gen2::`, 63 `pio_core::gen2::`, 12 bare; 83 distinct) |
| False positives removed | 1 leaf + 7 inline: pio-core's own `extract_gen2::{FactMemoryPipeline, LAST_K}` and `reflection_gen2::Gen2ReflectionModel` |

Receipt 05's "454 leaves" and "715 inline" included the in-tree copy's own
`crate::gen2::` self-references (`TokenPullerDyn` ×13, `bundle::ModelMeta`
×8, `messages_have_images` ×4 …); those are internal to the copy and vanish
with it. The receipt's `pub(crate)`-on-both-sides pair (`ChatTemplate`,
`messages_have_images`) is of that kind, and `compat` exports both anyway.

## 2. Every path, classified

"Today" is the crate at `519b29e` (S2.6): only `gen2::Message` survives at the
same path; a handful have a renamed public home from the S2.6 map; the rest
name modules that are `pub(crate)` since the API narrowed. The last column is
where `gen2::compat` puts it — always the old relative position, so
`use gen2::compat as gen2;` (or `pub mod gen2 { pub use gen2::compat::*; }`)
resolves the lot.

**Totals:** 87 distinct item paths — 1 same path (28 leaves),
38 renamed by S2.6 or since (327 leaves), 45 `pub(crate)` (71 leaves),
3 where `Engine` changed meaning (11 leaves). All 87 resolve
through `compat`: **437 of 445 `use` leaves and 268 of 283 inline
mentions**. The remainder is §4.

| Path | `use` | inline | files | today | S2.6 public path | `gen2::compat` |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `gen2::Message` | 28 | 96 | 32 | same path | `gen2::Message` (same) | `compat::Message` |
| `gen2::generation::GenSpec` | 35 | 35 | 29 | renamed (S2.6) | `advanced::generation::GenSpec` | `compat::generation::GenSpec` |
| `gen2::controller::ControllerCmd` | 50 | 13 | 50 | renamed (S2.6) | `advanced::controller::ControllerCmd` | `compat::controller::ControllerCmd` |
| `gen2::controller::InferenceHandle` | 22 | 32 | 28 | renamed (S2.6) | `advanced::controller::InferenceHandle` | `compat::controller::InferenceHandle` |
| `gen2::controller::ControllerHandle` | 19 | 13 | 24 | renamed (S2.6) | `advanced::controller::ControllerHandle` | `compat::controller::ControllerHandle` |
| `gen2::controller::ControllerEvent` | 30 | 1 | 29 | renamed (S2.6) | `advanced::controller::ControllerEvent` | `compat::controller::ControllerEvent` |
| `gen2::controller::SystemTask` | 26 | 2 | 24 | renamed (S2.6) | `advanced::controller::SystemTask` | `compat::controller::SystemTask` |
| `gen2::engine::Settings` | 21 | 5 | 20 | renamed (S2.6) | `advanced::generation::Settings` | `compat::engine::Settings` |
| `gen2::controller::start_controller` | 16 | 3 | 17 | renamed (S2.6) | `advanced::controller::start_controller` | `compat::controller::start_controller` |
| `gen2::backend::common::grammar::GrammarSpec` | 15 | 0 | 12 | renamed (S2.6) | `advanced::generation::GrammarSpec` | `compat::backend::common::grammar::GrammarSpec` |
| `gen2::engine::LoadRequest` | 15 | 0 | 8 | `pub(crate)` | — | `compat::engine::LoadRequest` |
| `gen2::engine::ExecError` | 1 | 13 | 2 | renamed (S2.6) | `advanced::controller::ExecError` | `compat::engine::ExecError` |
| `gen2::generation::TokenEvent` | 12 | 2 | 8 | `pub(crate)` | — | `compat::generation::TokenEvent` |
| `gen2::session_rt::SessionSpec` | 13 | 0 | 8 | renamed (S2.6) | `advanced::plugin::SessionSpec` | `compat::session_rt::SessionSpec` |
| `gen2::generation::ThinkingMode` | 3 | 9 | 9 | renamed (S2.6) | `gen2::model::ThinkingMode` | `compat::generation::ThinkingMode` |
| `gen2::MessageBody` | 9 | 0 | 5 | renamed (S2.6) | `advanced::wire::MessageBody` | `compat::MessageBody` |
| `gen2::backend::Engine` | 9 | 0 | 6 | semantic: `Engine` = backend facade | backend facade, no public path | `compat::backend::Engine` |
| `gen2::controller::ControllerRuntimeSnapshot` | 7 | 2 | 9 | renamed (S2.6) | `advanced::controller::ControllerRuntimeSnapshot` | `compat::controller::ControllerRuntimeSnapshot` |
| `gen2::controller::EVENT_CHANNEL_CAPACITY` | 5 | 2 | 6 | renamed (S2.6) | `advanced::controller::EVENT_CHANNEL_CAPACITY` | `compat::controller::EVENT_CHANNEL_CAPACITY` |
| `gen2::generation::ChannelMarkers` | 6 | 0 | 6 | renamed (post-S2.6) | `advanced::generation::ChannelMarkers` | `compat::generation::ChannelMarkers` |
| `gen2::generation::ReplyStateMachine` | 4 | 2 | 5 | renamed (post-S2.6) | `advanced::generation::ReplyStateMachine` | `compat::generation::ReplyStateMachine` |
| `gen2::ExecutionStats` | 3 | 2 | 5 | renamed (S2.6) | `advanced::controller::ExecutionStats` | `compat::ExecutionStats` |
| `gen2::backend::llama::embedder::EmbedderKind` | 2 | 3 | 4 | `pub(crate)` | — | `compat::backend::llama::embedder::EmbedderKind` |
| `gen2::backend::traits::Backend` | 5 | 0 | 1 | `pub(crate)` | — | `compat::backend::traits::Backend` |
| `gen2::Engine` | 1 | 3 | 2 | semantic: `Engine` = backend facade | `legacy::Engine` is the *API* facade — different type; the backend facade has no public path | `compat::Engine` |
| `gen2::MessageContent` | 4 | 0 | 3 | renamed (S2.6) | `advanced::wire::MessageContent` | `compat::MessageContent` |
| `gen2::Settings` | 2 | 2 | 4 | renamed (S2.6) | `advanced::generation::Settings` | `compat::Settings` |
| `gen2::controller::ControllerConfig` | 4 | 0 | 1 | renamed (S2.6) | `advanced::controller::ControllerConfig` | `compat::controller::ControllerConfig` |
| `gen2::controller::ControllerMetricsSnapshot` | 4 | 0 | 4 | renamed (S2.6) | `advanced::controller::ControllerMetricsSnapshot` | `compat::controller::ControllerMetricsSnapshot` |
| `gen2::controller::ControllerObservabilitySnapshot` | 4 | 0 | 4 | renamed (S2.6) | `advanced::controller::ControllerObservabilitySnapshot` | `compat::controller::ControllerObservabilitySnapshot` |
| `gen2::controller::start_controller_with_config` | 4 | 0 | 1 | renamed (S2.6) | `advanced::controller::start_controller_with_config` | `compat::controller::start_controller_with_config` |
| `gen2::backend::TokenPuller` | 0 | 3 | 2 | `pub(crate)` | — | `compat::backend::TokenPuller` |
| `gen2::backend::common::parse_hf_model_metadata` | 0 | 3 | 3 | `pub(crate)` | — | `compat::backend::common::parse_hf_model_metadata` |
| `gen2::backend::llama::embedder::LlamaEmbedder` | 1 | 2 | 2 | `pub(crate)` | — | `compat::backend::llama::embedder::LlamaEmbedder` |
| `gen2::generation::MediaBoundary` | 3 | 0 | 3 | renamed (S2.6) | `advanced::controller::MediaBoundary` | `compat::generation::MediaBoundary` |
| `gen2::generation::StreamEmission` | 3 | 0 | 3 | renamed (post-S2.6) | `advanced::generation::StreamEmission` | `compat::generation::StreamEmission` |
| `gen2::zoo::ModelZoo` | 3 | 0 | 3 | `pub(crate)` | — | `compat::zoo::ModelZoo` |
| `gen2::ResidencyInventory` | 2 | 0 | 1 | renamed (S2.6) | `advanced::runtime::ResidencyInventory` | `compat::ResidencyInventory` |
| `gen2::ResidentRuntime` | 2 | 0 | 1 | renamed (S2.6) | `advanced::runtime::ResidentRuntime` | `compat::ResidentRuntime` |
| `gen2::backend::llama::llama_config::ModelConfig` | 1 | 1 | 2 | `pub(crate)` | — | `compat::backend::llama::llama_config::ModelConfig` |
| `gen2::bundle::gguf::detect_format_from_path` | 1 | 1 | 2 | `pub(crate)` | — | `compat::bundle::gguf::detect_format_from_path` |
| `gen2::bundle::gguf::parse_gguf_metadata` | 1 | 1 | 2 | `pub(crate)` | — | `compat::bundle::gguf::parse_gguf_metadata` |
| `gen2::engine::ExecutionStats` | 2 | 0 | 2 | renamed (S2.6) | `advanced::controller::ExecutionStats` | `compat::engine::ExecutionStats` |
| `gen2::generation::CacheState` | 2 | 0 | 2 | `pub(crate)` | — | `compat::generation::CacheState` |
| `gen2::generation::ReplyShape` | 2 | 0 | 2 | `pub(crate)` | — | `compat::generation::ReplyShape` |
| `gen2::generation::Termination` | 2 | 0 | 2 | `pub(crate)` | — | `compat::generation::Termination` |
| `gen2::generation::ToolCall` | 1 | 1 | 2 | renamed (S2.6) | `advanced::controller::ToolCall` | `compat::generation::ToolCall` |
| `gen2::generation::TurnTelemetry` | 2 | 0 | 2 | `pub(crate)` | — | `compat::generation::TurnTelemetry` |
| `gen2::kv::KvLoadSpec` | 1 | 1 | 2 | `pub(crate)` | — | `compat::kv::KvLoadSpec` |
| `gen2::kv::KvSaveSpec` | 1 | 1 | 2 | `pub(crate)` | — | `compat::kv::KvSaveSpec` |
| `gen2::router` | 0 | 2 | 2 | `pub(crate)` | — | `compat::router` |
| `gen2::validate_model_file` | 0 | 2 | 1 | `pub(crate)` | — | `compat::validate_model_file` |
| `gen2::zoo::ModelFamily` | 2 | 0 | 2 | `pub(crate)` | — | `compat::zoo::ModelFamily` |
| `gen2::zoo::current_platform_id` | 1 | 1 | 2 | `pub(crate)` | — | `compat::zoo::current_platform_id` |
| `gen2::zoo::detect_ram_mb` | 2 | 0 | 2 | `pub(crate)` | — | `compat::zoo::detect_ram_mb` |
| `gen2::EmbedLoadRequest` | 1 | 0 | 1 | `pub(crate)` | — | `compat::EmbedLoadRequest` |
| `gen2::RuntimeKind` | 1 | 0 | 1 | renamed (S2.6) | `advanced::runtime::RuntimeKind` | `compat::RuntimeKind` |
| `gen2::backend::common::tokenizer::HfTokenizer` | 1 | 0 | 1 | `pub(crate)` | — | `compat::backend::common::tokenizer::HfTokenizer` |
| `gen2::backend::facade::SessionId` | 0 | 1 | 1 | `pub(crate)` | — | `compat::backend::facade::SessionId` |
| `gen2::backend::traits::SessionTokenizer` | 0 | 1 | 1 | `pub(crate)` | — | `compat::backend::traits::SessionTokenizer` |
The reply scanner (`ChannelMarkers`, `ReplyParts`, `ReplyStateMachine`,
`StreamEmission`) became public at `advanced::generation` on `main` after
S2.6 (`5d25291`), so those three inventory paths now have a supported home as
well as a `compat` one; `compat::generation` keeps the old spelling.

| `gen2::bundle::gguf` | 0 | 1 | 1 | `pub(crate)` | — | `compat::bundle::gguf` |
| `gen2::bundle::gguf::GgufMetadata` | 1 | 0 | 1 | `pub(crate)` | — | `compat::bundle::gguf::GgufMetadata` |
| `gen2::bundle::gguf::build_model_metadata` | 1 | 0 | 1 | `pub(crate)` | — | `compat::bundle::gguf::build_model_metadata` |
| `gen2::bundle::gguf::estimate_ram_bytes` | 0 | 1 | 1 | `pub(crate)` | — | `compat::bundle::gguf::estimate_ram_bytes` |
| `gen2::bundle::gguf::fit_context` | 1 | 0 | 1 | `pub(crate)` | — | `compat::bundle::gguf::fit_context` |
| `gen2::bundle::gguf::kv_bytes_per_token` | 1 | 0 | 1 | `pub(crate)` | — | `compat::bundle::gguf::kv_bytes_per_token` |
| `gen2::bundle::gguf::trim_optional` | 1 | 0 | 1 | `pub(crate)` | — | `compat::bundle::gguf::trim_optional` |
| `gen2::controller::ActiveChatSnapshot` | 1 | 0 | 1 | renamed (S2.6) | `advanced::controller::ActiveChatSnapshot` | `compat::controller::ActiveChatSnapshot` |
| `gen2::controller::CompletionReason` | 1 | 0 | 1 | renamed (S2.6) | `advanced::controller::CompletionReason` | `compat::controller::CompletionReason` |
| `gen2::controller::RuntimeLifecycleSnapshot` | 1 | 0 | 1 | renamed (S2.6) | `advanced::controller::RuntimeLifecycleSnapshot` | `compat::controller::RuntimeLifecycleSnapshot` |
| `gen2::controller::WorkloadKind` | 1 | 0 | 1 | renamed (S2.6) | `advanced::controller::WorkloadKind` | `compat::controller::WorkloadKind` |
| `gen2::controller::config` | 0 | 1 | 1 | renamed (S2.6) | `advanced::controller::config` | `compat::controller::config` |
| `gen2::controller::start_controller_with_limit` | 1 | 0 | 1 | renamed (S2.6) | `advanced::controller::start_controller_with_limit` | `compat::controller::start_controller_with_limit` |
| `gen2::engine::Engine` | 1 | 0 | 1 | semantic: `Engine` = backend facade | backend facade, no public path | `compat::engine::Engine` |
| `gen2::engine::SamplingSettings` | 1 | 0 | 1 | renamed (S2.6) | `advanced::generation::SamplingSettings` | `compat::engine::SamplingSettings` |
| `gen2::executor::BoxedOperation` | 1 | 0 | 1 | `pub(crate)` | — | `compat::executor::BoxedOperation` |
| `gen2::executor::ConcurrencyGuard` | 1 | 0 | 1 | `pub(crate)` | — | `compat::executor::ConcurrencyGuard` |
| `gen2::executor::StreamingToolExecutor` | 1 | 0 | 1 | `pub(crate)` | — | `compat::executor::StreamingToolExecutor` |
| `gen2::generation::TelemetrySnapshot` | 1 | 0 | 1 | `pub(crate)` | — | `compat::generation::TelemetrySnapshot` |
| `gen2::generation::global_aggregator` | 1 | 0 | 1 | `pub(crate)` | — | `compat::generation::global_aggregator` |
| `gen2::generation::ttft_bucket_upper_bounds_us` | 1 | 0 | 1 | `pub(crate)` | — | `compat::generation::ttft_bucket_upper_bounds_us` |
| `gen2::read_gguf_architecture` | 0 | 1 | 1 | `pub(crate)` | — | `compat::read_gguf_architecture` |
| `gen2::session_rt::prompt::generation_reserve` | 1 | 0 | 1 | `pub(crate)` | — | `compat::session_rt::prompt::generation_reserve` |
| `gen2::validate_model_architecture` | 0 | 1 | 1 | `pub(crate)` | — | `compat::validate_model_architecture` |
| `gen2::zoo::ModelZooEntry` | 0 | 1 | 1 | `pub(crate)` | — | `compat::zoo::ModelZooEntry` |
| `gen2::zoo::PlatformBundle` | 1 | 0 | 1 | `pub(crate)` | — | `compat::zoo::PlatformBundle` |
| `gen2::zoo::select_for_device` | 0 | 1 | 1 | `pub(crate)` | — | `compat::zoo::select_for_device` |

`gen2::bundle::gguf` (1, as a module path), `gen2::router` (2, doc links) and
`gen2::controller::config` (1, doc link) are module paths and resolve as
`compat::bundle::gguf`, `compat::router`, `compat::controller::config`.

### 2a. Variants and associated functions named inline

Each resolves on its (resolving) type; `tests/compat_pio_paths.rs` names every
one. `InferenceHandle::Flock` is the exception (§4).

| Path | inline |
| --- | ---: |
| `gen2::Message::user` | 32 |
| `gen2::generation::GenSpec::default` | 30 |
| `gen2::controller::ControllerHandle::new_for_test` | 9 |
| `gen2::Message::system` | 8 |
| `gen2::Message::assistant_structured` | 5 |
| `gen2::engine::Settings::default` | 3 |
| `gen2::generation::ThinkingMode::Off` | 3 |
| `gen2::generation::ThinkingMode::default` | 3 |
| `gen2::controller::ControllerCmd::GenerateEmbeddings` | 3 |
| `gen2::engine::ExecError::ModelNotLoaded` | 2 |
| `gen2::controller::ControllerCmd::IsMmprojLoaded` | 2 |
| `gen2::controller::ControllerCmd::IsModelLoaded` | 2 |
| `gen2::controller::ControllerCmd::IsEmbedderLoaded` | 2 |
| `gen2::Engine::detect_backend_for_path` | 2 |
| `gen2::controller::InferenceHandle::system_infer_with` | 2 |
| `gen2::controller::InferenceHandle::Local` | 2 |
| `gen2::Message::user_with_images` | 2 |
| `gen2::engine::ExecError::EmbedderNotLoaded` | 1 |
| `gen2::engine::ExecError::InvalidModelFile` | 1 |
| `gen2::engine::ExecError::SettingsError` | 1 |
| `gen2::engine::ExecError::Io` | 1 |
| `gen2::engine::ExecError::ContextOverflow` | 1 |
| `gen2::engine::ExecError::TemplateError` | 1 |
| `gen2::engine::ExecError::SessionPoisoned` | 1 |
| `gen2::controller::ControllerCmd::IsChatLoaded` | 1 |
| `gen2::Engine::available_backends` | 1 |
| `gen2::controller::ControllerCmd::LoadModel` | 1 |
| `gen2::controller::ControllerCmd::LoadEmbedder` | 1 |
| `gen2::controller::SystemTask::Title` | 1 |
| `gen2::controller::ControllerCmd::StartChat` | 1 |
| `gen2::backend::llama::embedder::LlamaEmbedder::load_from_path_with_kind` | 1 |
| `gen2::backend::llama::llama_config::ModelConfig::default` | 1 |
| `gen2::backend::llama::embedder::EmbedderKind::Qwen3` | 1 |
| `gen2::backend::llama::embedder::EmbedderKind::as_str` | 1 |
| `gen2::generation::ThinkingMode::On` | 1 |
| `gen2::generation::ThinkingMode::Auto` | 1 |
| `gen2::kv::KvSaveSpec::ToPath` | 1 |
| `gen2::kv::KvLoadSpec::Strict` | 1 |
| `gen2::generation::TokenEvent::Paused` | 1 |
| `gen2::generation::TokenEvent::Stopped` | 1 |
| `gen2::controller::SystemTask::Compact` | 1 |

## 3. `SystemTask`

The in-tree enum had thirteen `Copy` variants, each with a `GenSpec` in
`ControllerConfig::system_task_spec` and a `sabra-<suffix>-<uuid>` session
id. The crate keeps `Title`, `Suggestions`, `Compact`, `Summary` and folds the
rest into `Custom(Cow<'static, str>)` (one spec: 512 tokens at 0.3; id
`<label>-<uuid>`; not `Copy`). Host mentions of the removed variants: **29 in
11 files** (23 code sites in 9 files, 6 in doc comments):

| Variant | mentions | old spec (max_tokens / temperature) | `compat::system_task` |
| --- | ---: | --- | --- |
| `Answer` | 21 | 1024 / 0.3 | `answer()` |
| `TopicLabel` | 2 | 100 / 0.3 | `topic_label()` |
| `ContextualPrefix` | 2 | 120 / 0.1 | `contextual_prefix()` |
| `Triples` | 1 | 1024 / 0.1 | `triples()` |
| `QueryRewrite` | 1 | 80 / 0.1 | `query_rewrite()` |
| `EntityExtract` | 1 | 512 / 0.1 | `entity_extract()` |
| `Contradiction` | 1 | 512 / 0.2 | `contradiction()` |
| `Stance`, `QueryUnderstand` | 0 | 512 / 0.1, 256 / 0.1 | `stance()`, `query_understand()` |

`compat::system_task::gen_spec(&task)` reproduces the old table for every
label (and defers to the crate for the kept variants and unknown labels), so
the mapping is mechanical but not free of behaviour: `system_infer(task, ..)`
and `system_prompt(task, ..)` look the spec up *by variant*, so a `Custom`
task through them gets 512/0.3. Sites that relied on the per-task spec must
become `system_infer_with(task, chat_id, messages, gen_spec(&task))`. Of the
23 code sites, 12 already call `system_infer_with` with their own spec
(`chat_runtime.rs` ×2, `agent_eval.rs` ×4, `agent_loop.rs` ×2, `research.rs`
×2, `base_chat.rs`, `gen2_runner.rs`, `reflection_gen2.rs` — all `Answer`);
the 11 through `system_prompt`/`system_infer` (`sabra_inference.rs` ×9,
`capture/ask.rs`, `title_polish` n/a) are the ones that change shape. Nothing
in the host parses the `sabra-` session-id prefix (grep: only unrelated
`sabra-fact-extract`/`sabra-rerank-` ids), and no host `match` is over
`SystemTask`.

Kept variants keep working unchanged: `Title` 17, `Compact` 5, `Summary` 3,
`Suggestions` 2 mentions.

## 4. What `compat` cannot provide

| Path | mentions | Why, and the host change |
| --- | ---: | --- |
| `gen2::backend::mlxcel::MlxcelEngine` | 5 | mlxcel is the `gen2-mlxcel` companion crate (S1.3); the root manifest has no mlxcel dependency, so the crate cannot re-export it. Host: `gen2_mlxcel::MlxcelEngine` (4 `agent_eval` captests). |
| `gen2::controller::project_streaming_inference` | 3 | deleted with the `flock` feature; receipt 05 §3 moves it host-side verbatim (one caller, the `fit_route.rs:1281` seam test; two mentions are comments). |
| `gen2::backend::mlxcel::ProfileMode` | 2 | `MlxcelEngine::profile` and its `ProfileMode`/`ProfileRun` are not in `gen2-mlxcel`. Host: the two profiler captests (`agent_eval.rs:2732`, `:2886`) need a replacement or `#[ignore]`. |
| `gen2::backend::mlxcel::ProfileRun` | 1 | as above. |
| `gen2::backend::mlxcel::worker::METALLIB_ENV` | 1 | `pub(crate)` in `gen2-mlxcel` (`PIO_MLX_METALLIB`). Doc-comment mention only (`src-tauri/src/startup.rs:229`). |
| `gen2::zoo::tests::recommended_sampling_` | 1 | crate-private unit test named in a doc comment. |
| `gen2::zoo::tests::recommended_thinking_` | 1 | as above. |
| `gen2::controller::InferenceHandle::Flock` | 1 | variant removed; `InferenceHandle::Remote(Arc<dyn RemoteDispatch>)` (receipt 05 §3). |

**Total: 15 mentions on 8 paths** (9 are mlxcel, all in
`runners/agent_eval.rs` captests plus one doc comment).

## 5. Residual pio-app-side changes for S5.2 (semantic; counts)

These compile against the old paths and fail or misbehave on meaning. Counts
are `grep` over the same file set.

| Change | Sites | Files | What S5.2 does |
| --- | ---: | ---: | --- |
| `SystemTask` removed variants (§3) | 23 code (29 mentions) | 9 (11) | constructors from `compat::system_task`; 11 `system_prompt`/`system_infer` calls become `system_infer_with(.., gen_spec(&task))` |
| `ControllerCmd::LoadModel { .. }` constructors: `api_model: Option<String>` added; `resp` is `Sender<Result<LoadOutcome, String>>` | 27 | — | add the field; `Ok(())` arms become `Ok(_)` / read the outcome |
| `ControllerCmd::ContinueChat { .. }` constructors: `transcript`, `thinking`, `tools` added | 15 | — | supply the transcript (the rebuilt-runtime path), the thinking policy, the offered tools |
| `ControllerEvent::Accepted { .. }` emitted first | 27 files match `ControllerEvent::`; 1 exhaustive (receipt 05) | 27 | wildcard arms tolerate it; the exhaustive match gains an arm; SSE/first-event ordering test (receipt 05 §7.6) |
| `InferenceHandle::Remote` (13) / `Flock` (9) / `RegisteredFlockGateway` (12) arms; `.liveness()` (6); `compute_provenance()` (3, receipt) | 43 | ~12 | host `HostInference` enum + three `RemoteDispatch` impls (receipt 05 §3) |
| `controller::project_streaming_inference` | 1 call | 1 | move to `p2p/flock/handle.rs` |
| `backend::mlxcel::*` | 4 test fns + 2 profiler tests | 1 | `gen2_mlxcel` dependency under the `apple` bundle; profiler tests re-homed or ignored |
| `gen2::Engine` = backend facade | 3 (`models/service.rs`) | 1 | none with the shim: `compat::Engine` *is* the facade (`legacy::Engine` is the API one) |
| `system_infer*` → `ExecError` | 34 call sites (receipt) | — | `From<ExecError> for PioError` exists; only explicit `map_err` sites |
| `specta` derives | 45 → 71 types | — | `check-specta-compat.sh`; bindings diff |

Everything else in the inventory (437 leaves + 268 inline) is a path
change the shim absorbs.

## 6. Feature dependence of the compat paths

`cargo check --test compat_pio_paths` passes under default features
(`backend-llamacpp,hf,agent`), `--no-default-features --features
backend-external-api,agent`, and `--features specta,tokio`; the library
under `backend-litertlm`, `backend-mistralrs` and no features too.

| Paths | Feature |
| --- | --- |
| `backend::llama::{embedder::{EmbedderKind, LlamaEmbedder}, llama_config::ModelConfig}` (3 paths, 4 `use` + 6 inline) | `backend-llamacpp` |
| everything else (84 paths) | none — the controller, generation, zoo, bundle, kv, executor, session_rt, residency and types modules are unconditional |
| `executor` | none (`agent` is not needed by any compat path) |
| `specta` derives on `SystemTask`, `WorkloadKind`, `CompletionReason`, `ControllerEvent`, `GenSpec`, … | `--features specta` compiles (`cargo clippy --all-targets --features specta,tokio -D warnings` clean) |

pio-app builds with `backend-llamacpp` + `metal` (+ `backend-mlxcel` as the
companion) + `backend-external-api` + `specta` + `tokio`; every compat path is
present under that set. `backend-mlx` could not be verified: that lane does
not compile on `main` today (`src/backend/mlx/session.rs:421` predates
`Message.tool_call_id`), independent of this slice.

## 7. The shim (for S5.2)

```rust
// pio-core/src/lib.rs
#[cfg(not(feature = "gen2-crate"))]
pub mod gen2;
#[cfg(feature = "gen2-crate")]
pub mod gen2 {
    pub use gen2_crate::compat::*;
}
```

`compat::*` carries the old root re-export list (`Engine` = backend facade,
`EmbedLoadRequest`, `LoadRequest`, `Settings`, `ExecError`, `ExecutionStats`,
`Message*`, `Residency*`, `read_gguf_architecture`, `validate_model_*`,
`effective_context_budget`, …) and every module. Nothing else in the host
moves for the path change.
