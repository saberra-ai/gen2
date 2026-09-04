# Changelog

All notable changes to gen2. The format follows Keep a Changelog; versions
follow SemVer, and 0.x means the public surface may still move between minors.

## Unreleased

### Added
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
- Linux read free RAM instead of available RAM, so the residency governor
  refused helper loads after any large build filled the page cache.
- Rustdoc under `-D warnings` failed on two private links and one dead link.
- A helper-latency acceptance test asserted a wall-clock bound.
