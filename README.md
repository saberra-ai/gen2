# gen2

A local-first inference engine with pluggable backends. Load a model, run a
turn, stream tokens back — the same API whether the weights are running through
llama.cpp, MLX, ONNX Runtime, Candle, ExecuTorch, or a remote OpenAI/Anthropic-
compatible endpoint.

**The public API is `Engine`, `Session`, and the three call modes below.**
Everything underneath — backend dispatch, session runtime, KV cache, the model
zoo, placement routing, residency policy — is internal, so it can change without
breaking you. `engine.controller()` is the escape hatch.

Extracted from [pio-app](https://github.com/saberra-ai/pio-app)'s `pio-core`
crate, with history. See [`docs/EXTRACTION.md`](docs/EXTRACTION.md) for what
moved and what a host app still supplies.

## Three ways to call a model

Pick by what you need to keep.

| | You get | Nothing kept | Conversation kept | Tools |
|---|---|---|---|---|
| `infer` | a string | ✓ | | |
| `chat` | a turn in a conversation | | ✓ | you dispatch |
| `agent` | a task carried out | | ✓ | it dispatches |

### `infer` — one prompt, nothing kept

For a classification, a title, an extraction. There's no session to read back.

```rust
use pio_gen2::Engine;

let engine = Engine::load("/models/model.gguf")?;

let title = engine.infer("Title this in three words: …").max_tokens(16).text()?;
```

Constrain the shape when you need to parse the answer:

```rust
use pio_gen2::GrammarSpec;

let json = engine.infer("Classify the sentiment of: '…'")
    .grammar(GrammarSpec::JsonSchema(schema))
    .greedy()
    .text()?;
let parsed: Sentiment = serde_json::from_str(&json)?;   // sound, not hopeful
```

### `chat` — a conversation you own

The reply is appended to a `Session` you hold, so the transcript is yours to
render, persist, or edit.

```rust
use pio_gen2::{Engine, Session};

let engine = Engine::load("/models/model.gguf")?;
let mut session = Session::new().with_system("Be terse.");

engine.chat(&mut session).user("Name two colours.").send()?;
println!("{}", session.latest_text().unwrap_or_default());

// A follow-up. The history is already in the session, so nothing is resent
// and the engine's warm KV cache is reused.
engine.chat(&mut session).user("Now one more.").send()?;

for message in session.messages() { /* render */ }
```

Streaming, for a UI on a blocking thread:

```rust
engine.chat(&mut session)
    .user("Write a haiku about Rust.")
    .max_tokens(64)
    .send_streaming(|token| print!("{token}"))?;
```

Off-thread, so the caller never blocks. The session comes back on `Done`:

```rust
use std::sync::Arc;
use pio_gen2::Update;

let engine = Arc::new(engine);
let turn = engine.chat_owned(session).user("Hello").spawn();

for update in turn {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::Done { session, .. } => keep(session),
        Update::Failed { error, .. } => show(error),
        _ => {}
    }
}
```

If you want the tool calls but want to run them yourself, `chat` gives you
`.tools(...)` plus `.on_tool(handler)` and a `.tool_depth(7)` bound.

### `agent` — a task carried out

You register tools; the agent resolves what the model named, validates the
arguments against that tool's schema, runs it, and decides whether a failure is
worth handing back.

```rust
use pio_gen2::{Engine, FunctionTool, Session, ToolOutput, ToolSearch};
use pio_gen2::schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

let weather = FunctionTool::new(
    "get_weather",
    "Current weather for a city",
    |_ctx, a: WeatherArgs| async move { Ok(ToolOutput::from(fetch(&a.city))) },
);

let done = engine.agent(&mut session)
    .add_tool(weather)
    .defer_tools(mcp_tools)              // 40 tools, none in the prompt
    .tool_search(ToolSearch::Hybrid)
    .max_steps(12)
    .goal("What is the weather in Paris?")?;

println!("{} after {} tool rounds", done.text, done.tool_rounds);
```

The schema comes from `WeatherArgs`, so what the model sees and what the handler
reads cannot drift.

Off-thread, with steering — this is the form a UI wants:

```rust
let run = engine.agent_owned(session)
    .add_tool(weather)
    .goal("Summarise the repository")
    .spawn();

let steering = run.steering();       // move this to wherever the user is
// steering.follow_up("also check the tests");
// steering.interrupt("stop, just summarise the README");

for update in run {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::ToolCall { tool, args, .. } => status(format!("{tool} {args}")),
        Update::Done { completion, session } => finish(completion, session),
        _ => {}
    }
}
```

`interrupt` on an owned run cuts the generation short — measured at 223 chars
against 3840 uninterrupted. On a borrowed `agent(&mut session)` there is no
engine handle to ask, so it lands at the next step boundary instead;
`steering.can_interrupt_generation()` tells you which you have.

Under the `tokio` feature, `spawn_async()` gives the same thing as a `Stream`,
and `send_async().await` awaits a whole turn.

### How an agent stops, and what it may run

**Deferred tools** stay out of the prompt entirely. When the model calls
`search_tools`, matches are appended to the *conversation* — never the prefix —
so the warm KV cache survives. Search is hybrid by default: BM25 over names,
descriptions and argument names catches exact terminology, embeddings catch
intent, and RRF fuses them on rank.

**Stopping** is a first-class answer, not a timeout. `Finish::OutOfBudget(Budget::Steps
| Tokens | Deadline)` says which limit; `Finish::GaveUp(Struggle::RepeatingCall
{ .. })` catches a model calling the same tool with the same arguments — the
failure a step count never sees.

**Approval** is off by default, because a gate nobody reads is worse than none.
Tools declare `Risk::Risky`; `ApprovalMode::AskOnRisky` routes those through
`on_approval`, synchronously, because you cannot approve something by observing
a stream.

**Scheduling** honours each tool's `ExecutionPolicy`. Independent calls in one
turn run concurrently; anything declaring itself unsafe to parallelise runs
alone. Results are appended in call order regardless of completion order.

```rust
FunctionTool::new(..).with_policy(ExecutionPolicy::exclusive())   // a shared write
FunctionTool::new(..).with_policy(ExecutionPolicy::gpu_bound())   // contends with the model
```

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
`Stream`, instead of every async caller writing that bridge themselves.

```rust
let (completion, session) = engine.chat_owned(session).user("…").send_async().await?;
let mut run = engine.agent_owned(session).goal("…").spawn_async();
while let Some(update) = run.next().await { /* … */ }
```

Off by default, so sync consumers need no runtime.

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
- `tools` — the lower-level `chat().on_tool()` loop, where you dispatch.
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

## License

MIT
