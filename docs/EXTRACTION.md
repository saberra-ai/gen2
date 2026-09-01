# Extraction record

`gen2` was `pio-core/src/gen2/` in [pio-app](https://github.com/saberra-ai/pio-app).
This records what moved, what was inverted to break the coupling, and what is
left before pio-app can consume this crate instead of its own copy.

History came across via `git subtree split -P pio-core/src/gen2`, so all 160
commits that touched gen2 are preserved. The split's contents were then moved
under `src/`.

Extracted 2026-08-31 from pio-app `5367cf0f`.

## Status

- `cargo test` — **521 unit tests + doc tests passing** on the default feature
  set; `cargo clippy --all-targets` and `cargo doc` both clean.
- **Live inference verified.** `tests/live_inference.rs` drives a real
  SmolLM2-360M-Instruct GGUF through the real llama.cpp backend under `metal`:
  it generates real text (`"Reply with exactly one word: hello"` → `"Hello"`,
  terminating on `Eos`), reproduces exactly across two turns under `.greedy()`,
  honours `max_tokens`, continues a named chat across turns (turn 2 recalls a
  fact stated only in turn 1), and shuts down cleanly on drop. It runs through
  the public API, so it also proves that surface is sufficient. This is what
  separates "the extraction compiles" from "the engine still works".
- `cargo check --no-default-features --features backend-llamacpp` — **clean**.
- `cargo clippy --all-targets` — **clean**.
- MLX / mlxcel / ONNX / Candle feature sets — **not built or run here.** They were
  building in pio-app immediately before the split and their dependency pins
  came across unchanged, but that is inference, not verification.
- **pio-app is not a consumer.** It still builds its own in-tree copy of gen2,
  and nothing in pio-app changed during or since this extraction.

## What moved in alongside gen2

These were `pio-core` modules that gen2 depended on. Each is a leaf (serde
structs or self-contained logic), and each is closer to inference than to the
host app, so they came across whole rather than being inverted:

| From `pio-core` | To | Why it belongs here |
| --- | --- | --- |
| `types/message.rs` | `types/message.rs` | The wire types every backend speaks. |
| `types/execution_stats.rs` | `types/execution_stats.rs` | What a generation reports back. |
| `types::{Model, ModelConfig, ModelMetadata}` | `types/model.rs` | The record a backend is asked to load. |
| `types::Persona` | `types/persona.rs` | Pinned into the system prompt at session start. |
| `hardware.rs` | `hardware.rs` | Also *depended on gen2* (`platform_defaults` returns `gen2::Settings`), so the split was already circular. |
| `diagnostics/memory_*.rs` | `memory/` | Sizes model residency; `pio-core`'s memory *reporting* reads it, not the reverse. |
| `compute::escalation::ComputeProvenance` | `provenance.rs` | The dispatching handle is the only thing that knows which brain ran the work. |
| `app/chat/compaction/tier1.rs` (+ its helpers) | `session_rt/compaction.rs` | Pure `Vec<Message>` manipulation; warm-start truncation calls it before dropping messages. |
| `tasks/spawn.rs` | `task_util.rs` | Panic-safe spawn used by the executor loop. |

## Couplings that were inverted

Five places reached out of gen2 into the host. Each was cut so the dependency
points inward:

1. **`session_rt/prompt.rs` read `store::AppStore`** to fetch the selected
   persona. Now `build_prompt_context(persona: Option<Persona>, …)` takes it as
   a parameter — the host resolves it. It also stopped being `async`, since the
   store read was the only await. It had no callers outside the module, so this
   changed nothing downstream.

2. **`bundle/gguf.rs` returned `PioError`.** Now returns gen2's own
   `ExecError`, with a new `ExecError::io(impl Display)` helper so the
   `map_err(ExecError::io)` call sites read as before. `pio-core` already has
   `From<ExecError> for PioError`, so host-side error routing is unchanged.

3. **`controller`'s `system_infer*` family returned `PioError`.** They now
   return `ExecError`. Two variants were added to carry what would otherwise be
   lost: `Generation(String)` for dispatch/join failures, and
   `Coded { code, message }` for a `ControllerEvent::Error` that already carries
   the host's snake_case error code. gen2 does not own that taxonomy — 27
   `ErrorCode` variants each route to a frontend action — so it passes the code
   through verbatim for the host to map back.

4. **`router.rs` took `flock::discovery::PeerAdvertisement`.** That struct is
   now defined here, field-for-field, as the router's input contract. It was
   already gen2-adjacent (the host's version calls `gen2::zoo::current_platform_id()`
   to fill its `platform` field).

5. **`hardware.rs`'s `captest_vram_detect` test** asserted against
   `pio-core::fit::model_fit`, whose three-tier verdict is host placement
   policy. The test stayed with the host; the detector it exercised
   (`parse_nvidia_smi_total_mib`) is still covered by unit tests here.

## The one seam still open: remote dispatch

`InferenceHandle` can dispatch a generation to another device instead of running
it locally. Those arms — `Remote`, `Flock`, `RegisteredFlockGateway` — are still
written against `pio-core`'s `p2p`/`flock` types:

- `controller/mod.rs`: the three enum arms and their dispatch/failover paths.
- `controller/mod.rs`: `project_streaming_inference`, which projects a
  `ControllerCmd` into a `RetryableInference`.
- `controller/mod.rs`: `InferenceHandle::liveness`, which returns
  `p2p::heartbeat::Liveness`. The only *public method* on the seam, so a host
  calling it today needs an equivalent on whatever trait replaces it.

All of it is behind `#[cfg(feature = "p2p-client")]` / `#[cfg(feature = "flock")]`,
so it compiles out here and the local path is complete without it. The features
are declared but **will not build standalone** — that is deliberate and marked
in `Cargo.toml`.

Closing this seam means defining a `RemoteDispatch`-style trait here that the
host implements with its `FlockHandle`. That inversion should be done against
pio-app's flock integration tests, which live on that side, so it was left for
the switchover rather than guessed at here.

## pio-app

Out of scope. pio-app keeps its own in-tree copy of gen2; this crate is not
trying to be drop-in for it, and its call sites are not a constraint on the API
here. The design decisions since extraction (controller-only surface, then the
`Engine` facade over it) were made for this crate on its own terms.
