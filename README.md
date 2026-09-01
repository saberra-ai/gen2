# pio-gen2

A local-first inference engine with pluggable backends. Load a model, run a
turn, stream tokens back — the same API whether the weights are running through
llama.cpp, MLX, ONNX Runtime, Candle, ExecuTorch, or a remote OpenAI/Anthropic-
compatible endpoint.

**The public API is the controller.** You send it commands and read events;
everything underneath — backend dispatch, session runtime, KV cache, the model
zoo, placement routing, residency policy — is internal, so it can change without
breaking you.

Extracted from [pio-app](https://github.com/saberra-ai/pio-app)'s `pio-core`
crate, with history. See [`docs/EXTRACTION.md`](docs/EXTRACTION.md) for what
moved and what a host app still supplies.

## Quick start

```rust
use pio_gen2::Engine;

let engine = Engine::load("/models/model.gguf")?;

let reply = engine.prompt("Explain entropy in one sentence.")
    .max_tokens(256)
    .text()?;
```

Streaming, when you want the tokens as they arrive:

```rust
let mut stream = engine.chat("chat-1")
    .user("Write a haiku about Rust.")
    .max_tokens(64)
    .stream()?;

for event in &mut stream {
    match event? {
        Event::Token(t) => print!("{t}"),
        _ => {}
    }
}
assert_eq!(stream.finish(), Some(Finish::Eos));
```

A second turn on the same chat id continues that conversation, reusing its warm
KV cache:

```rust
engine.chat("chat-1").user("Now make it about Go.").text()?;
```

Shutdown is automatic — dropping the `Engine` stops the controller and waits for
the backend to be released. Call `engine.shutdown()?` instead when you want
teardown failures to surface.

### Reproducibility

`.greedy()` pins temperature 0 and a fixed seed. Worth knowing that it is **not**
the default: an unconfigured turn samples with a random seed, so the same prompt
gives different text each run.

```rust
engine.prompt("Count to three").greedy().text()?;   // same output every time
```

### Constrained output

`.grammar(...)` shapes decoding with a JSON schema, regex, Lark grammar, or
GBNF. It is enforced *during* decoding — the model cannot emit anything that
violates it — and behaves identically across every backend.

### Remote endpoints

The same API, different weights:

```rust
let engine = Engine::builder().openai("gpt-4o-mini", api_key).build()?;
```

## Backends

Pick at least one — a build with none fails to compile, by design.

| Feature | Backend | Notes |
| --- | --- | --- |
| `backend-external-api` | OpenAI / Anthropic wire formats | **Default.** No C toolchain needed. |
| `backend-llamacpp` | llama.cpp (GGUF) | Add `metal`, `cuda`, or `vulkan` for GPU. |
| `backend-mlx` | MLX (Apple Silicon) | Mutually exclusive with `backend-mlxcel`. |
| `backend-mlxcel` | mlxcel (faster MLX decode) | Mac fast path. |
| `backend-onnx` | ONNX Runtime | |
| `backend-candle` | Candle (pure Rust) | |
| `backend-executorch` | ExecuTorch (mobile) | Scaffold; returns `Unimplemented`. |

```bash
cargo test                                                  # default, no C toolchain
cargo check --no-default-features --features backend-llamacpp
cargo check --no-default-features --features backend-mlxcel # Mac fast path
```

### Examples

```sh
cargo run --example basic      --no-default-features --features metal -- /path/model.gguf
cargo run --example structured --no-default-features --features metal -- /path/model.gguf
```

`basic` covers ask / stream / converse; `structured` covers grammar-constrained
output.

### Live inference

Unit tests never load a model. To prove the engine actually generates, point
`PIO_TEST_MODEL` at a small instruct GGUF:

```bash
PIO_TEST_MODEL=/path/SmolLM2-360M-Instruct-Q4_K_M.gguf \
  cargo test --test live_inference --no-default-features --features metal -- --nocapture
```

These skip without the env var, but never pass by skipping: once it's set, a
model that won't load or won't decode is a hard failure.

## What's in here

Public:

- **`Engine`**, **`Chat`**, **`TokenStream`**, **`Event`**, **`Error`** — the
  API, re-exported at the crate root.
- **`controller`** — the layer underneath: commands, events, handles, config,
  and the observability snapshots. Reach for it via `engine.controller()` when
  you need something the facade doesn't cover.
- The vocabulary those signatures are written in: `GenSpec`, `Settings`,
  `Message`, `ExecutionStats`, `ThinkingMode`, `ToolCall`, `GrammarSpec`, and
  the residency/memory types the snapshots expose.

Internal (`pub(crate)`), driven by the controller:

- **`engine/`** — load models, validate architectures, report stats.
- **`session_rt/`** — sessions, prompt assembly, context truncation + compaction.
- **`generation/`** — token events, reply/thinking parsing.
- **`backend/`** — the pluggable backends behind one trait set, plus shared
  chat-templating, tokenization, sampling, grammar-constrained decoding, and
  stop-sequence matching.
- **`kv/`** — versioned KV-cache save/load with checksums and strict/lenient
  compatibility policies.
- **`residency*`, `memory/`, `hardware.rs`** — how much of the machine a
  resident model may take, and what the machine actually is.
- **`zoo.rs`** — the canonical model zoo and per-platform bundle selector,
  editable at `resources/models/zoo.json`.
- **`router.rs`** — pure placement: given a request, local capability, and a
  peer list, pick which device runs it. Local-first.

The crate warns on `unnameable_types`, so a type that becomes reachable through
the public API without being nameable fails the build rather than silently
becoming a hole.

## Constrained decoding

All backends share one grammar stack (`llguidance` + `toktrie`), so a JSON
schema, Lark grammar, regex, or GBNF that shapes output under llama.cpp shapes
it identically under MLX.

## License

MIT
