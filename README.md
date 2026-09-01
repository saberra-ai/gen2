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
use pio_gen2::{Engine, Session};

let engine = Engine::load("/models/model.gguf")?;

// A conversation you own. The reply is appended to it.
let mut session = Session::new();
engine.chat(&mut session).user("Explain entropy in one sentence.").send()?;
println!("{}", session.latest_text().unwrap_or_default());

// A follow-up. The history is already in the session, so nothing is resent
// and the engine's warm KV cache is reused.
engine.chat(&mut session).user("Simpler?").send()?;

// The transcript is yours — render it, persist it, edit it.
for message in session.messages() { /* … */ }
```

Streaming, when you want tokens as they arrive:

```rust
engine.chat(&mut session)
    .user("Write a haiku about Rust.")
    .max_tokens(64)
    .send_streaming(|token| print!("{token}"))?;
```

One-offs with nothing to keep — a classification, a title:

```rust
let title = engine.infer("Title this in three words.").max_tokens(16).text()?;
```

Off the calling thread, for a UI. The session comes back on `Done`:

```rust
let turn = engine.chat_owned(session).user("Hello").spawn();

for update in turn {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::Done { session, .. } => save(session),
        Update::Failed { error, .. } => show(error),
        _ => {}
    }
}
```

Shutdown is automatic: dropping the `Engine` stops the controller and waits for
the backend to be released. `engine.shutdown()?` instead if you want teardown
failures to surface.

### Reproducibility

`.greedy()` pins temperature 0 and a fixed seed. It is **not** the default — an
unconfigured turn samples with a random seed. Set it per turn, or once for the
whole engine with `Engine::builder().greedy()`.

### Constrained output

`.grammar(...)` shapes decoding with a JSON schema, regex, Lark grammar, or
GBNF, enforced *during* decoding so the model cannot emit anything that
violates it. Per turn, or as an engine default:

```rust
let classifier = Engine::builder().model(path).grammar(schema).greedy().build()?;
let json = classifier.infer("Classify: '…'").text()?;      // always the right shape
```

A turn can still override it, or drop it with `.unconstrained()` — loading
weights is the expensive part, so one engine should serve several shapes.

### Embeddings

An embedder loads independently of the chat model — an engine can hold both, or
only one:

```rust
let engine = Engine::builder().embedder("/models/embedding-model.gguf").build()?;

let vectors = engine.embed(&corpus)?;          // one per input, in order
let query   = engine.embed_one("a question")?; // single
```

### Remote endpoints

Same API, different weights. The URL selects the backend:

```rust
let engine = Engine::builder().openai("https://api.openai.com/v1", key).build()?;
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
cargo run --example minimal    --no-default-features --features metal -- /path/model.gguf
cargo run --example basic      --no-default-features --features metal -- /path/model.gguf
cargo run --example structured --no-default-features --features metal -- /path/model.gguf
cargo run --example chat_app   --no-default-features --features metal -- /path/model.gguf
cargo run --example embeddings --no-default-features --features metal -- /path/embedding-model.gguf
```

- `minimal` — the smallest useful program: make a chat, stream the reply.
- `basic` — ask / stream / converse, and the four ways to consume a generation.
- `structured` — grammar-constrained output (JSON schema, bare JSON, regex).
- `embeddings` — embedder-only engine, batch and single, cosine similarity.
- `chat_app` — the shape of a real app: `Arc<Engine>` shared across threads,
  generation on a worker, cancellation, and concurrent conversations.

### Live inference

Unit tests never load a model. To prove the engine actually generates, point
`PIO_TEST_MODEL` at a small instruct GGUF:

```bash
PIO_TEST_MODEL=/path/SmolLM2-360M-Instruct-Q4_K_M.gguf \
  cargo test --test live_inference --no-default-features --features metal -- --nocapture
```

`PIO_TEST_EMBEDDER` additionally runs the embedding test. These skip without the
env vars, but never pass by skipping: once set, a model that won't load or won't
decode is a hard failure.

## What's in here

Public:

- **`Engine`**, **`Session`**, **`Chat`**, **`Completion`**, **`Event`**,
  **`Error`** — the API, re-exported at the crate root. `Session` holds the
  transcript; the engine keys its warm cache to it, so owning your history
  costs nothing in speed.
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
