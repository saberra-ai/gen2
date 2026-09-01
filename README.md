# gen2

A local-first inference engine with pluggable backends. One API over
llama.cpp, MLX, ONNX Runtime, Candle, ExecuTorch, or an OpenAI-compatible
endpoint.

```toml
gen2 = { git = "https://github.com/saberra-ai/gen2" }
```

---

## Three ways to call a model

| | You get | Keeps |
| --- | --- | --- |
| `infer` | a string | nothing |
| `chat` | a turn | the conversation |
| `agent` | a task done | the conversation, and runs your tools |

### infer

```rust
use gen2::Engine;

let engine = Engine::load("/models/model.gguf")?;
let title = engine.infer("Title this in three words: …").max_tokens(16).text()?;
```

Shaped output, enforced during decoding:

```rust
use gen2::GrammarSpec;

let raw = engine.infer("Classify the sentiment of: '…'")
    .grammar(GrammarSpec::JsonSchema(schema))
    .greedy()
    .text()?;
let parsed: Sentiment = serde_json::from_str(&raw)?;
```

### chat

```rust
use gen2::{Engine, Session};

let mut session = Session::new().with_system("Be terse.");

engine.chat(&mut session).user("Name two colours.").send()?;
println!("{}", session.latest_text().unwrap_or_default());

engine.chat(&mut session).user("Now one more.").send()?;   // history is already there

for message in session.messages() { /* render */ }
```

Streaming:

```rust
engine.chat(&mut session)
    .user("Write a haiku about Rust.")
    .send_streaming(|token| print!("{token}"))?;
```

Off-thread. The session comes back on `Done`:

```rust
use std::sync::Arc;
use gen2::Update;

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

Images:

```rust
engine.chat(&mut session)
    .user_with_images("What is in this picture?", ["/tmp/photo.png"])
    .send()?;
```

### agent

```rust
use gen2::{FunctionTool, ToolOutput, ToolSearch};
use gen2::schemars::JsonSchema;

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
    .defer_tools(mcp_tools)              // absent from the prompt until searched for
    .tool_search(ToolSearch::Hybrid)
    .max_steps(12)
    .goal("What is the weather in Paris?")?;

println!("{} after {} tool rounds", done.text, done.tool_rounds);
```

`WeatherArgs` generates the schema, so renaming a field changes what the model
sees and what the handler reads together.

A typed final answer:

```rust
let done = engine.agent(&mut session)
    .add_tool(weather)
    .answer_as(GrammarSpec::JsonSchema(schema), "Answer as JSON with city and temperature_c.")
    .goal("What is the weather in Paris?")?;

let report: Report = serde_json::from_str(&done.text)?;
```

Off-thread, with steering:

```rust
let run = engine.agent_owned(session).add_tool(weather).goal("Summarise the repo").spawn();

let steering = run.steering();
// steering.follow_up("also check the tests");
// steering.interrupt("stop, just the README");

for update in run {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::ToolCall { tool, args, .. } => status(tool, args),
        Update::Done { completion, session } => finish(completion, session),
        _ => {}
    }
}
```

Reusable across runs:

```rust
use gen2::AgentConfig;

let researcher = AgentConfig::new().add_tool(weather).max_steps(8);

researcher.agent(&engine, &mut session).goal("Weather in Paris?")?;
researcher.agent(&engine, &mut session).goal("And which city was that?")?;
```

Approval, off by default:

```rust
use gen2::{ApprovalMode, Decision};

let delete_file = FunctionTool::new("delete_file", "Delete a file", handler).risky();

engine.agent(&mut session)
    .add_tool(delete_file)
    .approval(ApprovalMode::AskOnRisky)          // safe tools are not asked about
    .on_approval(|name, args, _spec| match confirm(name, args) {
        true => Decision::Allow,
        false => Decision::Deny("user declined".into()),
    })
    .goal("Clean up the temp directory")?;
```

---

## Tools

```rust
// Bundle and reuse.
let filesystem = ToolSet::new().add(read_file).add(write_file).add(list_dir);
engine.agent(&mut session).add_tools(filesystem);

// Independent calls in one turn run concurrently.
FunctionTool::new(..).with_policy(ExecutionPolicy::exclusive())   // a shared write
FunctionTool::new(..).with_policy(ExecutionPolicy::gpu_bound())   // contends with the model

// A whole agent as one tool.
let researcher = AgentTool::new("researcher", "Investigates a question", engine.clone())
    .tools(research_tools)
    .max_steps(5);

// Instructions loaded on demand. Descriptions sit in the prompt, bodies arrive
// when the model asks for them.
let skills = SkillLibrary::new([
    Skill::new("migrations", "when writing a database migration", MIGRATION_GUIDE),
]);

// Every tool an MCP server offers.
let mcp = McpToolSet::connect("mcp-server-git", ["--repo", "."]).await?;
engine.agent(&mut session).defer_tools(mcp).tool_search(ToolSearch::Hybrid);
```

## Sessions

```rust
session.messages();            // the transcript, yours to render or persist
session.latest_text();
session.shed();                // messages no longer in the model's context
session.fork();                // branch. Same history, independent from here
session.edit(|m| m.truncate(4));

let json = serde_json::to_string(&session)?;   // Serialize / Deserialize
```

Not `Clone`: two copies would share one cached prefill and overwrite each
other. `fork()` is the independent copy.

## Engine

```rust
Engine::builder()
    .model(path)                    // GGUF file, or an MLX / ONNX directory
    .mmproj(path)                   // vision projector
    .embedder(path)                 // alongside, or instead of, a chat model
    .openai("https://api.openai.com/v1", key)
    .auto_context()                 // size the window to the machine
    .greedy()                       // defaults every turn starts from
    .grammar(schema)
    .build()?;

engine.load_model(path)?;           // swap on a live engine
engine.reload_model()?;
engine.unload_model()?;

engine.capabilities();              // TEXT | IMAGES | AUDIO
engine.supports_images();

engine.embed(&corpus)?;             // one vector per input
engine.embed_one("a query")?;

engine.stop(id)?;  engine.pause(id)?;  engine.resume(id)?;
```

## Somewhere else

The controller can live in another process or on another machine. Implement
the transport, and everything above it is unchanged:

```rust
use gen2::{ControllerCmd, InferenceHandle, Placement, RemoteDispatch};

struct OverTheWire { /* your socket, your peer, your queue */ }

impl RemoteDispatch for OverTheWire {
    fn send(&self, cmd: ControllerCmd) -> Result<(), String> { self.dispatch(cmd) }
    fn label(&self) -> &str { "workshop-mac" }
}

let handle = InferenceHandle::remote(OverTheWire::connect()?);
assert_eq!(handle.placement(), Placement::Remote("workshop-mac"));
```

## Will it fit?

```rust
use gen2::{HardwareProfile, ModelInfo};

let info = ModelInfo::read("/models/model.gguf")?;   // header only, no weights
let hw   = HardwareProfile::detect();

info.max_context(&hw);
info.fits(&hw, Some(8192));         // Fits | ContextTooLarge | TooLarge

match Engine::builder().model(path).context(1_000_000).build() {
    Err(e) => if let Some(fit) = e.fit() { println!("{fit}") },
    Ok(engine) => { /* … */ }
}
```

## Async

Behind the `tokio` feature:

```rust
let (completion, session) = engine.chat_owned(session).user("…").send_async().await?;

let mut run = engine.agent_owned(session).goal("…").spawn_async();
while let Some(update) = run.next().await { /* … */ }
```

---

## Things that will otherwise cost you an afternoon

- **`.greedy()` is not the default.** An unconfigured turn leaves `temperature`
  and `seed` unset, which means backend-default sampling with a random seed. The
  same prompt gives different text each run.
- **Deferred tool specs never enter the prompt prefix.** Search puts them in the
  conversation instead, so the warm KV cache survives.
- **Changing the tool set between runs reopens the conversation.** Tool
  definitions live in the prefix, so a change costs one re-prefill. The
  alternative was ignoring the new tools without telling you.
- **Swapping a model invalidates every session's prefill.** Sessions notice and
  reopen on their own. `engine.model_generation()` is the same signal if you
  want to show it.
- **A load that fails part-way leaves no model.** `load_model` checks the path
  first, so a typo is refused before anything unloads. An out-of-memory mid-load
  cannot be undone.
- **Just drop the `Engine` when you're done.** The controller loop holds the
  backend on its own thread, and exiting while it runs aborts inside ggml's
  destructors. `Drop` stops and joins it for you.
- **A background task you define gets no tuning.** `SystemTask::Title` and its
  named siblings carry sampling defaults; `SystemTask::custom("triples")` gets
  a plain spec, because nothing here knows what your task is. Pass your own to
  `system_infer_with`.
- **A cancelled turn is `Done`, not `Failed`.** `completion.text` holds what was
  generated before the stop, and it is already in the session.

## Backends

Pick at least one. A build with none fails to compile.

| Feature | Backend |
| --- | --- |
| `backend-external-api` | OpenAI / Anthropic wire formats. Default. Needs no C toolchain. |
| `backend-llamacpp` | llama.cpp (GGUF). Add `metal`, `cuda`, or `vulkan`. |
| `backend-mlx` | MLX (Apple Silicon). Mutually exclusive with `backend-mlxcel`. |
| `backend-mlxcel` | mlxcel, the Mac fast path. |
| `backend-onnx` | ONNX Runtime |
| `backend-candle` | Candle (pure Rust) |
| `backend-executorch` | ExecuTorch (mobile). Stub, returns Unimplemented. |
| `tokio` | Async API. Off by default. |

```sh
cargo test
cargo check --no-default-features --features backend-llamacpp
```

## Examples

```sh
cargo run --example minimal --no-default-features --features metal -- /path/model.gguf
```

`minimal` · `basic` · `agent` · `tools` · `structured` · `chat_app` ·
`embeddings` · `fit` · `async_chat` (needs `metal,tokio`)

## Live tests

Unit tests never load a model. These do:

```sh
PIO_TEST_MODEL=/path/model.gguf \
PIO_TEST_TOOL_MODEL=/path/tool-capable.gguf \
PIO_TEST_EMBEDDER=/path/embedding-model.gguf \
  cargo test --test live_inference --no-default-features --features metal \
  -- --test-threads=1
```

Without the env vars they skip. With them set, a model that will not load or
will not decode fails the test. Serially, because otherwise the tests compete
for residency admission and each other's failures look like yours.

## License

MIT
