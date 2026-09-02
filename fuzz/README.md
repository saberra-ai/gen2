# gen2 fuzz targets

A separate workspace (`[workspace] members = ["."]` in `fuzz/Cargo.toml`), so
`cargo build` / `cargo test` / `cargo clippy` at the repo root never see it.
It only builds under nightly, which libFuzzer and the sanitizers require.

It also depends on gen2 with `default-features = false`, because no target here
fuzzes an inference backend — the `gguf` target is header parsing and
arithmetic. The default `backend-llamacpp` does not merely cost a C++ build:
llama.cpp's `common` bundles `download.cpp` and `hf-cache.cpp`, whose
cpp-httplib calls nothing in the archive defines, and the fuzz link pulls those
objects in and dies on undefined `httplib::` symbols. `backend-litertlm` stands
in because `src/backend/mod.rs` requires *some* backend and that one links
nothing at build time.

```sh
# Seed first — "GGUF" is four specific bytes the mutator will not find on its
# own, so without seeds almost every input dies at the magic check.
mkdir -p fuzz/corpus/gguf
cp tests/fixtures/gguf/corpus/*.gguf fuzz/corpus/gguf/
cp tests/fixtures/gguf/deeply_nested_arrays.gguf fuzz/corpus/gguf/

cargo +nightly fuzz run gguf -- -max_total_time=300
```

`fuzz/corpus/` and `fuzz/artifacts/` are gitignored; the checked-in seeds live
in `tests/fixtures/gguf/corpus/` so a run is reproducible from a clean tree.

Add `-s address` for AddressSanitizer (the default on most platforms; on
aarch64-apple-darwin it must be requested explicitly and slows execs down
considerably).

## Targets

| Target | Entry point | Invariant |
|---|---|---|
| `gguf` | `gen2::ModelInfo::read` | Arbitrary bytes parse or are refused — never panic, abort, hang, or allocate against a *declared* length. Covers the estimators downstream of the parse, which consume header-supplied dimensions. |
| `reply_parts` | `gen2::ReplyStateMachine` | **Written but not wired up** — see below. Any chunking under any marker set terminates; `push` and `push_emit` agree; the scanner never invents bytes. Structure-aware — chunks are drawn from real marker literals and their prefixes. |

## Targets that are written but not enabled

`fuzz_targets/reply_parts.rs` is complete and cannot be built, because
`ChannelMarkers`, `ReplyStateMachine` and `StreamEmission` are `pub(crate)`
(`src/lib.rs` declares `pub(crate) mod generation`). Widening a library's
public API so a fuzz harness can reach it is a design decision for the crate
owner, not something a test should force. To enable it:

1. add `pub use generation::{ChannelMarkers, ReplyParts, ReplyStateMachine, StreamEmission};`
   to `src/lib.rs`;
2. uncomment the `reply_parts` `[[bin]]` block in `fuzz/Cargo.toml`.

Until then the same invariants are enforced by `proptest` in
`src/generation/reply_parts.rs` — `push_and_push_emit_never_disagree`,
`no_chunking_of_any_output_loses_a_byte_that_is_not_a_marker`, and
`arbitrary_markers_over_arbitrary_text_always_terminate` — which generate and
shrink, and unlike a fuzz target they run in ordinary CI.

## What is NOT fuzzed here

`gen2::kv::codec` (`build_blob` / `parse_blob`). The `kv` module is
`pub(crate)`, so it has no external entry point to fuzz without widening the
crate's public API for the benefit of a test harness. It is covered instead by
`proptest` inside the crate (round-trip, single-byte-corruption, and
arbitrary-bytes properties in `src/kv/codec.rs`), which runs in ordinary CI.

## Fuzz targets do not run in CI

They are a discovery tool, not a gate. Every crash they have found is pinned by
a deterministic test next to the code it broke:

- `src/bundle/gguf.rs` — `an_enormous_declared_string_length_is_refused_not_allocated`,
  `array_nesting_past_the_depth_limit_is_refused_not_recursed`,
  `arbitrary_bytes_behind_a_valid_magic_never_panic`
- `src/generation/reply_parts.rs` — `an_empty_marker_does_not_spin_the_scanner_forever`,
  `push_and_push_emit_never_disagree`
