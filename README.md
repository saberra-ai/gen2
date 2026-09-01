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
use std::sync::mpsc::{channel, sync_channel};
use pio_gen2::controller::start_controller;
use pio_gen2::{ControllerCmd, ControllerEvent, GenSpec, Message, Settings};

let handle = start_controller();

// Load a model and wait for it to be ready.
let (resp, resp_rx) = channel();
handle.send(ControllerCmd::LoadModel {
    model_path: "/path/model.gguf".into(),
    mmproj_path: None,
    settings: Settings::default(),
    api_key: None,
    api_format: None,
    resp,
})?;
resp_rx.recv()??;

// Run a turn; events stream back on `rx`.
let (tx, rx) = sync_channel(handle.config().event_channel_capacity);
handle.send(ControllerCmd::StartChat {
    chat_id: "chat-1".into(),
    messages: vec![Message::user("Hello")],
    gen_spec: GenSpec { max_tokens: Some(32), ..Default::default() },
    thinking: Default::default(),
    model_id: None,
    model_size_bytes: None,
    tools: None,
    tx,
})?;

for event in rx {
    match event {
        ControllerEvent::Token(t) => print!("{t}"),
        ControllerEvent::Error { code, message } => eprintln!("[{code}] {message}"),
        ControllerEvent::Eos | ControllerEvent::Stopped => break,
        _ => {}
    }
}

handle.send(ControllerCmd::Shutdown)?;
```

For prompt-in/text-out without driving the channel yourself, `InferenceHandle`
carries the `system_infer` family (`system_prompt`, `system_infer`,
`system_infer_streaming`, …).

Send `Shutdown` when you are done: the loop runs on its own thread holding the
backend, and exiting the process while it is live aborts inside llama.cpp's
static destructors.

`ControllerCmd::PauseChat` / `ResumeChat` / `StopChat` drive a generation
cooperatively by `chat_id`.

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

- **`controller`** — the API. Commands, events, handles, config, and the
  observability snapshots. `ControllerHandle` is the command channel;
  `InferenceHandle` adds the `system_infer` family and can dispatch to another
  device.
- The vocabulary those signatures are written in, re-exported at the crate root:
  `GenSpec`, `Settings`, `Message`, `ExecError`, `ExecutionStats`,
  `ThinkingMode`, `ToolCall`, `GrammarSpec`, and the residency/memory types the
  snapshots expose.

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
