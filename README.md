# gen2

An embeddable AI runtime for Rust. Models, sessions, tools, agents and local
inference behind one stateful API — over llama.cpp, mistral.rs, MLX, LiteRT-LM,
ONNX Runtime, Candle, or an OpenAI-compatible endpoint.

```toml
[dependencies]
gen2 = { git = "https://github.com/saberra-ai/gen2" }

# Only if you build tools with typed arguments. The derives expand to paths in
# your own crate, so re-exporting them from gen2 is not enough.
schemars = "1"
serde = { version = "1", features = ["derive"] }
```

Defaults to llama.cpp, so a `.gguf` works out of the box and the first build
compiles a C++ toolchain. On Apple silicon add `metal`; on NVIDIA add `cuda`.
To skip all that and talk to a hosted endpoint instead:

```toml
gen2 = { git = "…", default-features = false, features = ["backend-external-api"] }
```

Every example below is compiled by `cargo test --doc`, so none of them can
drift from the API.

---

## Three ways to call a model

| | You get | Keeps |
| --- | --- | --- |
| `infer` | a string | nothing |
| `chat` | a turn | the conversation |
| `agent` | a task done | the conversation, and runs your tools |

### infer

```rust,no_run
use gen2::Engine;
# fn main() -> Result<(), gen2::Error> {
let engine = Engine::load("/models/model.gguf")?;
let title = engine.infer("Title this in three words: …").max_tokens(16).text()?;
# println!("{title}");
# Ok(())
# }
```

Shaped output, enforced during decoding:

```rust,no_run
use gen2::{Engine, GrammarSpec};
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
let schema = serde_json::json!({
    "type": "object",
    "properties": { "sentiment": { "type": "string", "enum": ["positive", "negative"] } },
    "required": ["sentiment"]
});

let raw = engine.infer("Classify the sentiment of: '…'")
    .grammar(GrammarSpec::JsonSchema(schema))
    .greedy()
    .text()?;
# let _ = raw;
# Ok(())
# }
```

For the two shapes people write that by hand for, there is a direct form.
Classification returns one of your labels — never model prose, and never a
label you did not supply:

```rust,no_run
use gen2::Engine;
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
let label = engine
    .classify("The service was fantastic")
    .labels(["positive", "negative", "neutral"])
    .label()?;
# println!("{label}");
# Ok(())
# }
```

Extraction gives you the type back. The JSON schema the model decodes under is
generated from the same declaration you deserialize into, so the constraint and
the parser cannot drift apart:

```rust,no_run
use gen2::Engine;
use gen2::schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
}

# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
let invoice: Invoice = engine.extract("Acme Ltd — total due $1,240.00").value()?;
# println!("{} {}", invoice.vendor, invoice.total);
# Ok(())
# }
```

A model that answers with something that is not the type you asked for gives
you `Error::Extraction`, carrying what it actually said — distinct from a
generation failure, because the two need different fixes.

### chat

```rust,no_run
use gen2::{Engine, Session};
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
let mut session = Session::new().with_system("Be terse.");

engine.chat(&mut session).user("Name two colours.").send()?;
println!("{}", session.latest_text().unwrap_or_default());

engine.chat(&mut session).user("Now one more.").send()?;   // history is already there

for message in session.messages() {
    println!("{}: {}", message.role, message.text());
}
# Ok(())
# }
```

Streaming:

```rust,no_run
# use gen2::{Engine, Session};
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
engine.chat(&mut session)
    .user("Write a haiku about Rust.")
    .send_streaming(|token| print!("{token}"))?;
# Ok(())
# }
```

Off-thread. The session comes back on `Done`:

```rust,no_run
use std::sync::Arc;
use gen2::{Engine, Session, Update};
# fn main() -> Result<(), gen2::Error> {
let engine = Arc::new(Engine::load("/models/model.gguf")?);
let turn = engine.chat_owned(Session::new()).user("Hello").spawn();

for update in turn {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::Done { session, .. } => drop(session),
        Update::Failed { error, .. } => eprintln!("{error}"),
        _ => {}
    }
}
# Ok(())
# }
```

Images:

```rust,no_run
# use gen2::{Engine, Session};
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
engine.chat(&mut session)
    .user_with_images("What is in this picture?", ["/tmp/photo.png"])
    .send()?;
# Ok(())
# }
```

### agent

```rust,no_run
use gen2::{Engine, FunctionTool, Session, ToolOutput};
use schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
}

# fn fetch(city: &str) -> String { format!("18C in {city}") }
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
let weather = FunctionTool::new(
    "get_weather",
    "Current weather for a city",
    |_ctx, a: WeatherArgs| async move { Ok(ToolOutput::from(fetch(&a.city))) },
);

let done = engine.agent(&mut session)
    .add_tool(weather)
    .max_steps(12)
    .goal("What is the weather in Paris?")?;

println!("{} after {} tool rounds", done.text, done.tool_rounds);
# Ok(())
# }
```

`WeatherArgs` generates the schema, so renaming a field changes what the model
sees and what the handler reads together.

Tools the model has to go looking for, so a large catalogue costs no prompt:

```rust,no_run
# use gen2::{Engine, FunctionTool, Session, ToolOutput, ToolSearch};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct NoArgs {}
# fn tool(n: &'static str) -> FunctionTool<NoArgs> {
#     FunctionTool::new(n, "does something", |_c, _a: NoArgs| async move { Ok(ToolOutput::from("ok")) })
# }
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
engine.agent(&mut session)
    .add_tool(tool("read_file"))
    .defer_tools([tool("kubectl_apply"), tool("resize_image")])
    .tool_search(ToolSearch::Hybrid)
    .goal("Apply the deployment manifest")?;
# Ok(())
# }
```

A typed final answer:

```rust,no_run
# use gen2::{Engine, GrammarSpec, Session};
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
# let schema = serde_json::json!({"type": "object"});
let done = engine.agent(&mut session)
    .answer_as(GrammarSpec::JsonSchema(schema), "Answer as JSON with city and temperature_c.")
    .goal("What is the weather in Paris?")?;

let report: serde_json::Value = serde_json::from_str(&done.text).unwrap();
# let _ = report;
# Ok(())
# }
```

Off-thread, with steering:

```rust,no_run
# use std::sync::Arc;
# use gen2::{Engine, Session, Update};
# fn main() -> Result<(), gen2::Error> {
# let engine = Arc::new(Engine::load("/models/model.gguf")?);
let run = engine.agent_owned(Session::new()).goal("Summarise the repo").spawn();

let steering = run.steering();
steering.follow_up("also check the tests");
// steering.interrupt("stop, just the README");

for update in run {
    match update {
        Update::Delta(t) => print!("{t}"),
        Update::ToolCall { tool, args, .. } => println!("calling {tool} with {args}"),
        Update::Done { completion, .. } => println!("{}", completion.text),
        _ => {}
    }
}
# Ok(())
# }
```

Reusable across runs:

```rust,no_run
use gen2::AgentConfig;
# use gen2::{Engine, FunctionTool, Session, ToolOutput};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct NoArgs {}
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
# let weather = FunctionTool::new("w", "weather", |_c, _a: NoArgs| async move { Ok(ToolOutput::from("ok")) });
let researcher = AgentConfig::new().add_tool(weather).max_steps(8);

researcher.agent(&engine, &mut session).goal("Weather in Paris?")?;
researcher.agent(&engine, &mut session).goal("And which city was that?")?;
# Ok(())
# }
```

Approval, off by default:

```rust,no_run
use gen2::{ApprovalMode, Decision};
# use gen2::{Engine, FunctionTool, Session, ToolOutput};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct Path { path: String }
# fn confirm(_n: &str, _a: &serde_json::Value) -> bool { false }
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
let delete_file = FunctionTool::new("delete_file", "Delete a file", |_c, a: Path| async move {
    Ok(ToolOutput::from(format!("deleted {}", a.path)))
})
.risky();

engine.agent(&mut session)
    .add_tool(delete_file)
    .approval(ApprovalMode::AskOnRisky)          // safe tools are not asked about
    .on_approval(|name, args, _spec| match confirm(name, args) {
        true => Decision::Allow,
        false => Decision::Deny("user declined".into()),
    })
    .goal("Clean up the temp directory")?;
# Ok(())
# }
```

---

## Tools

Bundle and reuse:

```rust,no_run
use gen2::ToolSet;
# use gen2::{Engine, FunctionTool, Session, ToolOutput};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct NoArgs {}
# fn tool(n: &'static str) -> FunctionTool<NoArgs> {
#     FunctionTool::new(n, "does something", |_c, _a: NoArgs| async move { Ok(ToolOutput::from("ok")) })
# }
# fn main() -> Result<(), gen2::Error> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
let filesystem = ToolSet::new()
    .add(tool("read_file"))
    .add(tool("write_file"))
    .add(tool("list_dir"));

engine.agent(&mut session).add_tools(filesystem).goal("Read the manifest")?;
# Ok(())
# }
```

Independent calls in one turn run concurrently, unless a tool says otherwise:

```rust
use gen2::{ExecutionPolicy, FunctionTool, ToolOutput};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct NoArgs {}
let shared_write = FunctionTool::new("commit", "Commit staged changes", |_c, _a: NoArgs| async move {
    Ok(ToolOutput::from("committed"))
})
.with_policy(ExecutionPolicy::exclusive());
```

A whole agent as one tool, with its own narrower set:

```rust,no_run
use gen2::AgentTool;
# use std::sync::Arc;
# use gen2::{Engine, FunctionTool, ToolOutput};
# use schemars::JsonSchema;
# #[derive(serde::Deserialize, JsonSchema)]
# struct NoArgs {}
# fn main() -> Result<(), gen2::Error> {
# let engine = Arc::new(Engine::load("/models/model.gguf")?);
# let search = FunctionTool::new("search", "searches", |_c, _a: NoArgs| async move { Ok(ToolOutput::from("ok")) });
let researcher = AgentTool::new("researcher", "Investigates a question", engine.clone())
    .tools([search])
    .max_steps(5);
# let _ = researcher;
# Ok(())
# }
```

Instructions loaded on demand — descriptions sit in the prompt, bodies arrive
when the model asks for them:

```rust
use gen2::{Skill, SkillLibrary};

let skills = SkillLibrary::new([
    Skill::new("migrations", "when writing a database migration", "Always add a down migration…"),
]);
# let _ = skills;
```

Every tool an MCP server offers:

```rust,no_run
# use gen2::{Engine, Session, ToolSearch};
use gen2::McpToolSet;
# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
# let engine = Engine::load("/models/model.gguf")?;
# let mut session = Session::new();
let mcp = McpToolSet::connect("mcp-server-git", ["--repo", "."]).await?;

engine.agent(&mut session)
    .defer_tools(mcp)
    .tool_search(ToolSearch::Hybrid)
    .goal("What changed in the last commit?")?;
# Ok(())
# }
```

## Sessions

```rust
use gen2::Session;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut session = Session::new();
session.push_user("hello");

session.messages();            // the transcript, yours to render or persist
session.latest_text();
session.shed();                // messages no longer in the model's context
session.edit(|m| m.truncate(1));

let branch = session.fork();   // same history, independent from here
let json = serde_json::to_string(&session)?;   // Serialize / Deserialize
# let _ = (branch, json);
# Ok(())
# }
```

Not `Clone`: two copies would share one cached prefill and overwrite each
other. `fork()` is the independent copy.

## Engine

```rust,no_run
use gen2::Engine;
# fn main() -> Result<(), gen2::Error> {
let engine = Engine::builder()
    .model("/models/model.gguf")    // GGUF file, or an MLX / ONNX directory
    .auto_context()                 // size the window to the machine
    .greedy()                       // defaults every turn starts from
    .build()?;

// Swapping on a live engine. A load that cannot run as asked is retried
// without the vision projector, then on the CPU, so check what you got.
let outcome = engine.load_model("/models/other.gguf")?;
if !outcome.as_requested() {
    println!("{}", outcome.summary().unwrap_or_default());
}
engine.reload_model()?;

engine.capabilities();              // TEXT | IMAGES | AUDIO
engine.supports_images();

engine.embed_one("a query")?;
# Ok(())
# }
```

An endpoint instead of a local model:

```rust,no_run
# use gen2::Engine;
# fn main() -> Result<(), gen2::Error> {
let engine = Engine::builder()
    .openai("https://api.openai.com/v1", std::env::var("OPENAI_API_KEY").unwrap_or_default())
    .build()?;
# let _ = engine;
# Ok(())
# }
```

## Somewhere else

The controller can live in another process or on another machine. Implement
the transport, and everything above it is unchanged:

```rust
use gen2::{ControllerCmd, InferenceHandle, Placement, RemoteDispatch};

struct OverTheWire; // your socket, your peer, your queue

impl RemoteDispatch for OverTheWire {
    fn send(&self, _cmd: ControllerCmd) -> Result<(), String> { Ok(()) }
    fn label(&self) -> &str { "workshop-mac" }
}

let handle = InferenceHandle::remote(OverTheWire);
assert_eq!(handle.placement(), Placement::Remote("workshop-mac"));
```

## Will it fit?

```rust,no_run
use gen2::{Engine, HardwareProfile, ModelInfo};
# fn main() -> Result<(), gen2::Error> {
let info = ModelInfo::read("/models/model.gguf")?;   // header only, no weights
let hw = HardwareProfile::detect();

info.max_context(&hw);
info.fits(&hw, Some(8192));         // Fits | ContextTooLarge | TooLarge

match Engine::builder().model("/models/model.gguf").context(1_000_000).build() {
    Err(e) => if let Some(fit) = e.fit() { println!("{fit}") },
    Ok(engine) => drop(engine),
}
# Ok(())
# }
```

## Async

Behind the `tokio` feature:

```rust,ignore
let (completion, session) = engine.chat_owned(session).user("…").send_async().await?;

let mut run = engine.agent_owned(session).goal("…").spawn_async();
while let Some(update) = run.next().await { /* … */ }
```

---

## Things that will otherwise cost you an afternoon

- **`.greedy()` is not the default.** An unconfigured turn leaves `temperature`
  and `seed` unset, which means backend-default sampling with a random seed. The
  same prompt gives different text each run.
- **A reasoning model's `<think>` block is yours to strip.** Qwen3, DeepSeek-R1
  and Gemma 4 with thinking on emit their working into the reply, and it lands
  in `latest_text()` and in the stored transcript. Filter it before you render,
  and before it costs you context on the next turn.
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
| `backend-llamacpp` | llama.cpp (GGUF). **Default.** Add `metal`, `cuda`, or `vulkan`. |
| `backend-external-api` | OpenAI / Anthropic wire formats. Needs no C toolchain. |
| `backend-mlx` | MLX (Apple Silicon). Needs the Metal Toolchain component. Mutually exclusive with `backend-mlxcel`. |
| `backend-mlxcel` | mlxcel, the Mac fast path. |
| `backend-onnx` | ONNX Runtime |
| `backend-mistralrs` | mistral.rs. GGUF, safetensors and UQFF in one backend. Claims only formats no other compiled backend takes. No per-request seed: `.seed()` under sampling is refused rather than ignored. |
| `backend-candle` | Candle (pure Rust) |
| `backend-litertlm` | LiteRT-LM (Google's on-device runtime), for `.litertlm` bundles. Loads Google's C ABI at run time — nothing is vendored, linked, or downloaded by a build. Point `GEN2_LITERTLM_LIBRARY` at the shared library, or install it where the platform loader finds it. |
| `tokio` | Async API. Off by default. |

llama.cpp, mistral.rs, MLX and LiteRT-LM have been shown to generate a token;
the rest compile and satisfy the parts of the backend contract that need no
weights. Adding a backend never moves an existing one's models:
`backend-mistralrs` takes GGUF only where llama.cpp is absent and safetensors
only where MLX is, and `backend-litertlm` takes only the `.litertlm` bundles
nothing else can read. The conformance suite says which on every run, and fails
if that list goes stale.

LiteRT-LM's shipped runtime cannot report a bundle's context window, so state
it — `Engine::builder().model(path).context(4096)`. gen2 refuses the load
rather than guessing a number the controller would then plan against.

It asks for the GPU by default, measured at roughly 1.7x the CPU's decode rate
on an Apple M-series machine, and falls back to the CPU through the same load
ladder every other backend uses — reported as `Degraded::GpuOffload`, not
hidden. It never asks for the NPU: that needs vendor libraries for a specific
chip, and without them the runtime accepts the request and runs slower than the
CPU.

`ios` and `android` carry it alongside llama.cpp. Nothing links: the runtime is
loaded through its C ABI at run time, so `cargo check --target
aarch64-apple-ios --features backend-litertlm` needs nothing but rustup, and CI
checks exactly that. Shipping the runtime itself is the host application's job
— Google publishes `liblitert-lm.so` for Android and a `CLiteRTLM` XCFramework
for iOS.

Not yet publishable to crates.io: `mlxcel` has no registry release, and the
registry does not accept git dependencies. CI checks this so a release cannot
be surprised by it.

```sh
cargo test
cargo check --no-default-features --features backend-external-api
```

## Examples

```sh
cargo run --example minimal --features metal -- /path/model.gguf
```

`minimal` · `basic` · `agent` · `tools` · `structured` · `chat_app` ·
`embeddings` · `fit` · `async_chat` (needs `tokio`)

## Live tests

Unit tests never load a model. These do:

```sh
PIO_TEST_MODEL=/path/model.gguf \
PIO_TEST_TOOL_MODEL=/path/tool-capable.gguf \
PIO_TEST_EMBEDDER=/path/embedding-model.gguf \
  cargo test --test live_inference --features metal -- --test-threads=1
```

Without the env vars they skip. With them set, a model that will not load or
will not decode fails the test. Serially, because otherwise the tests compete
for residency admission and each other's failures look like yours.

## License

MIT
