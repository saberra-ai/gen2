# Changelog

All notable changes to gen2. The format follows Keep a Changelog; versions
follow SemVer, and 0.x means the public surface may still move between minors.

## Unreleased

### Added
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
