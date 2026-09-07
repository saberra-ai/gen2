# Changelog

All notable changes to gen2. The format follows Keep a Changelog; versions
follow SemVer, and 0.x means the public surface may still move between minors.

## Unreleased

### Added
- `hf:owner/repo[:QUANT|:file.gguf]` model references, accepted wherever a
  model path is (`gen2::load`, `Runtime::load`, `Engine::load`,
  `EngineBuilder::model`): the file is chosen (Q4_K_M by default, then the
  smallest Q4, then the smallest GGUF), downloaded through `hf-hub` into a
  cache in the Hub's layout (`GEN2_MODELS_DIR` → `HF_HUB_CACHE` → platform
  cache dir), and loaded; a warm cache makes no network request. `HF_TOKEN`
  is honoured. The typed form is `gen2::hf::HfModel` (`on_progress`,
  `cache_dir`, `token`, `resolve`, `download`) via `EngineBuilder::hf` and
  `Runtime::load_hf`; errors are `gen2::hf::HfError`, each naming its fix,
  surfaced as `Error::Load`. A repo's `mmproj*.gguf` becomes the vision
  projector. Behind the default-on `hf` feature.
- `ModelSourceKind::HuggingFace { repo, file }`; the enum is no longer
  `Copy`.
- `Runtime::load_embedder` and `Runtime::load_reranker` return an
  `Embedder`/`Reranker` (api_spec.md §21): `embed`, `embed_one`, `rerank`
  on the runtime's settings and backends, with no session or turn.
- `Session` is the spec's model-agnostic conversational state (api_spec.md
  §7–§9): the system prompt and a data-only `ToolSet` are first-order state
  (`set_system`/`append_system`, `set_tools`/`add_tool`/`remove_tool`),
  every message has a `MessageId`, `messages()` is the active projection
  over an append-only `events()` log, and `replace_message`,
  `remove_message`, `replace_messages`, `restore_context`, `fork_at` change
  the projection without erasing any record (`all_messages()`). Every
  context-affecting mutation moves a `SessionRevision`. Types live in
  `gen2::session`; `gen2::tool_defs::{ToolDefinition, ToolSet}` are the
  definitions the model is told about, with nothing to run.
- `Session::push_tool_result(call_id, ..)` and `Message::user_parts(..)`.
- A turn that names no tools of its own offers the session's `tools()`.
- A public backend seam: `gen2::advanced::BackendPlugin` registers an
  out-of-tree backend through `Engine::builder().backend(..)`, and
  `gen2::advanced::plugin` exports the types an implementer needs.
- `crates/gen2-mlxcel`, a workspace companion crate (not published) carrying the
  MLX fast path that used to be the `backend-mlxcel` feature.
- `Capabilities` is re-exported at the crate root; `Engine::capabilities()`
  returns it.
- `penalty_last_n = -1` still means "the whole context": the llama backend
  translates it to the context size now that llama.cpp clamps negatives to 0.

### Changed
- The crate root is api_spec.md §25: `Runtime`, `Model`, `Session`, `Message`,
  `ToolDefinition`, `ToolSet`, `ToolChoice`, `GenerationOptions`, `Response`,
  `Event`, `Error`, `Result`, `gen2::load`, plus `Turn`, `Input`, `EventStream`
  (`AsyncEventStream` under `tokio`) and the `schemars` re-export. Supporting
  types live in `gen2::{model, session, input, output, event, tool_defs}`;
  `gen2::turn` is gone (`Turn`, `GenerationOptions`, `ToolChoice` are at the
  root). `gen2::api` is no longer public.
- The previous facade is `gen2::legacy` and deprecated: `Engine`,
  `EngineBuilder`, `Chat`, `OwnedChat`, `Inference`, `Classify`, `Extract`,
  `Completion`, `TokenStream`, `Tokens`, the token-level `Event`, `Finish`,
  `Budget`, `Struggle`, the spawned `Turn`/`Canceller`/`Update`,
  `DEFAULT_TOOL_DEPTH`, `AsyncTurn`, and the fit `ModelInfo`/`Fit`/`FitVerdict`.
  Each is a deprecated alias whose warning names the replacement.
- The agent layer is `gen2::agent` behind a default-on `agent` feature and
  off the root: `Agent`, `AgentConfig`, `AgentRun`, `AgentStep`,
  `ApprovalMode`, `Decision`, `Risk`, `Steering`, `ExecutionPolicy`,
  `ToolRegistry`, `FunctionTool`, `AgentTool`, `Skill`, `SkillLibrary`,
  `SEARCH_TOOL`, `OwnedAgent`, `AsyncAgentRun`, the executable `ToolSet`,
  `gen2::agent::mcp` (was `gen2::mcp`) and `gen2::agent::journal` (was
  `gen2::journal`). `Agent::on(&model, &mut session)` starts one from a
  `Model`. With the feature off the crate is inference only:
  `Error::Tools`, `Update::ToolResult` and the `ToolOutput` conversions
  do not exist.
- `gen2::advanced` gathers what was transitively at the root:
  `advanced::generation` (`GenSpec`, `ThinkingMode`, `Settings` and its
  parts, `GrammarSpec`, speculative decoding, `Capabilities`, `Degraded`,
  `LoadOutcome`, `BackendCaps`, `LatencyTier`), `advanced::runtime` (adds
  `RuntimeBuilder`, the memory governor and residency types),
  `advanced::fit`, `advanced::wire` (`MessageBody`, `MessageChunk`,
  `MessageContent`, `FunctionDefinition`, `ToolSpec`, `ToolCall`, `Url`,
  `ModelRecord`, `ModelConfig`, `ModelMetadata`), `advanced::controller`
  (everything `gen2::controller` exported, plus `ExecError`,
  `ExecutionStats`, `MediaBoundary`, the stream `ToolCall`), and
  `advanced::utilities`. `gen2::controller` is no longer public.
- `ThinkingMode` is reached as `gen2::model::ThinkingMode`; `RerankResult`
  as `gen2::model::RerankResult`.
- `Runtime::builder().backend(plugin)` registers an out-of-tree backend for
  the new surface; `crates/gen2-mlxcel` documents that path.
- The examples use the new surface (`minimal`, `basic`, `structured`,
  `chat_app`, `embeddings`, `fit`, `tools`, `async_chat`); `agent`,
  `coding_agent` and `continuity` use `gen2::agent` and need the feature.
- docs.rs builds with the `agent` feature so that layer is documented.
- `Session::with_system` and `Chat::system` set the session's system prompt
  rather than pushing a `system` message; `messages()` no longer contains
  one, and `len()` no longer counts it. The rendered prompt is unchanged.
- `Session::id()` returns a `SessionId` (`as_str()` for the text).
- `Session::edit` and `Session::clear` are lossless context replacements;
  `clear` keeps the system prompt and tools. `edit` trims to a tool-round
  boundary rather than leaving a call without its results.
- A serialized `Session` is its id and event log; the previous flat
  `messages` form is still read.
- `llama-cpp-2` comes from crates.io (`=0.1.156`, llama.cpp b10405) instead
  of a git pin.
- `mlx-rs` is a hybrid dependency: registry version 0.25.3, plus the
  saberra-ai fork by git for builds from this repository.
- Backends are tiered: llama.cpp and the OpenAI/Anthropic client are
  supported; LiteRT-LM is the mobile lane; mistral.rs and MLX are experimental.
- `metal`/`cuda` now also reach mistral.rs when `backend-mistralrs` is on.
- A build with no backend feature compiles; the first load fails with an
  error naming the features and the plugin seam instead of a compile error.

### Removed
- `backend-onnx` and `backend-candle`: neither had generated a token.
- `backend-mlxcel` as a root feature (see `crates/gen2-mlxcel`).

### Fixed
- Tool-call `arguments` gained a layer of string escaping on every
  serialize/deserialize cycle of a `Message`; the wire form is now a fixed
  point and reads back as the object that was written.
- `Chat::on_tool` results carry the id of the call they answer.
- Linux read free RAM instead of available RAM, so the residency governor
  refused helper loads after any large build filled the page cache.
- Rustdoc under `-D warnings` failed on two private links and one dead link.
- A helper-latency acceptance test asserted a wall-clock bound.
