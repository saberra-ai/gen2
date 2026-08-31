# pio-gen2

A local-first inference engine with pluggable backends. Load a model, start a
session, pull tokens — the same API whether the weights are running through
llama.cpp, MLX, ONNX Runtime, Candle, ExecuTorch, or a remote OpenAI/Anthropic-
compatible endpoint.

Extracted from [pio-app](https://github.com/saberra-ai/pio-app)'s `pio-core`
crate, with history. See [`docs/EXTRACTION.md`](docs/EXTRACTION.md) for what
moved and what a host app still supplies.

## Quick start

```rust
use pio_gen2::engine::{Engine, LoadRequest};
use pio_gen2::generation::GenSpec;
use pio_gen2::session_rt::SessionSpec;
use pio_gen2::{Message, MessageBody, MessageContent};

let engine = Engine::new();
engine.load_model(LoadRequest {
    model_path: "/path/model.gguf".into(),
    ..Default::default()
})?;

let messages = vec![Message {
    name: None,
    role: "user".into(),
    body: MessageBody::Content {
        content: MessageContent::SingleText("Hello".into()),
    },
}];

let session = engine.start_session(SessionSpec { messages, ..Default::default() })?;
let mut puller = session.pull(GenSpec { max_tokens: Some(32), ..Default::default() })?;
while let Some(event) = puller.next() {
    // Token(text) | Paused | Stopped | Eos
}
```

`session.pause()` / `resume()` / `stop()` drive generation cooperatively; the
puller yields `Paused` and `Stopped` in response.

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

## What's in here

- **`engine/`** — load models, validate architectures, report stats.
- **`session_rt/`** — sessions, prompt assembly, context truncation + compaction.
- **`generation/`** — `GenSpec`, token events, reply/thinking parsing.
- **`backend/`** — the pluggable backends behind one trait set, plus shared
  chat-templating, tokenization, sampling, grammar-constrained decoding, and
  stop-sequence matching.
- **`kv/`** — versioned KV-cache save/load with checksums and strict/lenient
  compatibility policies.
- **`controller/`** — lifecycle, scheduling, metrics, observability.
- **`residency*`, `memory/`, `hardware.rs`** — how much of the machine a
  resident model may take, and what the machine actually is.
- **`zoo.rs`** — the canonical model zoo and per-platform bundle selector,
  editable at `resources/models/zoo.json`.
- **`router.rs`** — pure placement: given a request, local capability, and a
  peer list, pick which device runs it. Local-first.

## Constrained decoding

All backends share one grammar stack (`llguidance` + `toktrie`), so a JSON
schema, Lark grammar, regex, or GBNF that shapes output under llama.cpp shapes
it identically under MLX.

## License

MIT
