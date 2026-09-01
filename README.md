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

A long conversation outgrows the context window. The engine sheds its oldest
messages to make room and carries on; the session keeps everything. So the
transcript you hold can be a superset of what the model still sees:

```rust
session.shed()               // messages no longer in the model's view
session.fully_in_context()   // false once anything has been shed
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

### Agents

An agent owns dispatch. You register tools; it resolves what the model named,
validates the arguments against that tool's schema, runs it, and routes the
failure by whether the model can fix it.

```rust
use pio_gen2::{Engine, FunctionTool, Session, ToolOutput, ToolSearch};
use pio_gen2::schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

let weather = FunctionTool::new("get_weather", "Current weather for a city",
    |_ctx, a: WeatherArgs| async move { Ok(ToolOutput::from(fetch(&a.city))) });

let done = engine.agent(&mut session)
    .add_tool(weather)
    .defer_tools(mcp_tools)          // absent from the prompt until searched for
    .tool_search(ToolSearch::Hybrid)
    .max_steps(12)
    .goal("What is the weather in Paris?")?;
```

The schema comes from `WeatherArgs`, so what the model sees and what the handler
reads cannot drift.

**Deferred tools** stay out of the prompt entirely. When the model calls
`search_tools`, matches are appended to the *conversation* — never the prefix —
so the warm KV cache survives. Search is hybrid by default: BM25 over names,
descriptions and argument names catches exact terminology, embeddings catch
intent, and RRF fuses them.

**Stopping** is a first-class answer, not a timeout. `Finish::OutOfBudget(Budget::Steps
| Tokens | Deadline)` says which limit; `Finish::GaveUp(Struggle::RepeatingCall
{ .. })` catches a model calling the same tool with the same arguments — the
failure a step count never sees.

**Approval** is off by default, because a gate nobody reads is worse than none.
Tools declare `Risk::Risky`; `ApprovalMode::AskOnRisky` routes those through
`on_approval`.

### Sub-agents, skills, MCP

A sub-agent is not a new concept — it's a `Tool` that happens to run a nested
loop, so the parent sees one call and one result instead of a context full of
intermediate reading:

```rust
let researcher = AgentTool::new("researcher", "Investigates a question", engine.clone())
    .tools(research_tools)
    .max_steps(5);
```

Skills are the same trade as deferred tools, applied to prose — descriptions
stay in the prompt, bodies load on demand:

```rust
let skills = SkillLibrary::new([
    Skill::new("migrations", "when writing a database migration", MIGRATION_GUIDE),
]);
engine.agent(&mut session).add_tool(skills).goal("…")?;
```

An MCP server's whole surface registers as an iterator:

```rust
let mcp = McpToolSet::connect("mcp-server-git", ["--repo", "."]).await?;
engine.agent(&mut session).defer_tools(mcp).tool_search(ToolSearch::Hybrid);
```

### Steering and forking

`agent.steering()` hands out a movable handle — the thread driving the agent is
busy, so mid-run input has to come from elsewhere. `follow_up` adds to the task;
`interrupt` also abandons the rest of the current round's tool calls. Both land
at a step boundary, never mid-call: a tool already running has side effects that
must be recorded.

`session.fork()` branches a conversation — same history, new engine identity, so
two directions run without overwriting each other's cached prefill. `Session` is
`Serialize`/`Deserialize`, with the engine-state fields deliberately skipped: a
restored session must never claim a prefill the engine doesn't hold.

### Tool calling

Without a handler, a tool call is an `Event` for you to act on. With one, the
turn becomes a loop — generate, dispatch, feed results back, generate again:

```rust
let done = engine.chat(&mut session)
    .user("What is the weather in Paris?")
    .tools(vec![weather_tool()], "Call a tool when you need data.")
    .on_tool(|call| match call.name.as_str() {
        "get_weather" => fetch_weather(&call.arguments),
        other => format!("no such tool: {other}"),
    })
    .send()?;

done.tool_rounds       // how many rounds ran
```

`.tool_depth(n)` caps the rounds, defaulting to **7**. Reaching it ends the turn
with `Finish::ToolDepthReached` rather than looping forever — which is what
stops a model stuck re-calling the same tool. Both halves land in the session:
the assistant turn that asked, and the results that came back.

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

### Async

Behind the `tokio` feature. The controller stays synchronous — decoding is a
blocking native call — so this runs it on `spawn_blocking` and hands you a
`Stream`, instead of every async caller writing that bridge themselves:

```rust
// Await a whole turn.
let (completion, session) = engine.chat_owned(session)
    .user("Name two colours.")
    .send_async()
    .await?;

// Or stream it.
let mut turn = engine.chat_owned(session).user("Now one more.").spawn_async();
while let Some(update) = turn.next().await {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::Done { session, .. } => keep(session),
        _ => {}
    }
}
```

`turn.canceller()` works from another task. Off by default, so sync consumers
need no runtime.

### Will it fit?

Reads the model's header and measures the machine, before loading anything:

```rust
let info = ModelInfo::read("/models/model.gguf")?;   // header only, no weights
let hw   = HardwareProfile::detect();

info.max_context(&hw);          // largest context this machine can give
info.fits(&hw, Some(8192));     // Fits / ContextTooLarge / TooLarge
```

Or let the builder size it, and fail with a verdict rather than a load error:

```rust
let engine = Engine::builder().model(path).auto_context().build()?;

match Engine::builder().model(path).context(1_000_000).build() {
    Err(e) => if let Some(fit) = e.fit() {
        println!("{fit}");                  // why, in bytes
        println!("{} would work", fit.max_context);
    },
    Ok(engine) => { /* … */ }
}
```

GGUF only — other formats keep the backend's default context.

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
cargo run --example fit        --no-default-features --features metal -- /path/model.gguf
cargo run --example agent      --no-default-features --features metal -- /path/model.gguf
cargo run --example tools      --no-default-features --features metal -- /path/model.gguf
cargo run --example async_chat --no-default-features --features metal,tokio -- /path/model.gguf
```

- `minimal` — the smallest useful program: make a chat, stream the reply.
- `basic` — ask / stream / converse, and the four ways to consume a generation.
- `structured` — grammar-constrained output (JSON schema, bare JSON, regex).
- `agent` — registered tools, owned dispatch, and a deferred tool found by search.
- `tools` — the lower-level `chat().on_tool()` loop.
- `async_chat` — the `tokio` feature: await a turn, stream one, cancel from a task.
- `fit` — inspect a model and ask whether this machine can run it.
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

`PIO_TEST_EMBEDDER` adds the embedding test and `PIO_TEST_TOOL_MODEL` the tool
loop (needs a model with native tool calling — Qwen3 works, SmolLM2 doesn't). These skip without the
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
