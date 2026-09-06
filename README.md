# gen2

Local model inference for Rust. One `Runtime` owns the machine, a `Model` is a
cheap handle you can clone, a `Session` is conversation state you own, and a
`Turn` is one complete model invocation: instructions, tools, messages and
generation options in; structured, streamable output out. Backends today are
llama.cpp, mistral.rs, MLX, LiteRT-LM, or an OpenAI-compatible endpoint.

```toml
[dependencies]
gen2 = { git = "https://github.com/saberra-ai/gen2" }

# Only if you declare tools with typed arguments or ask for typed output. The
# derives expand to paths in your own crate, so re-exporting them from gen2 is
# not enough.
schemars = "1"
serde = { version = "1", features = ["derive"] }
```

Defaults to llama.cpp, so a `.gguf` works out of the box and the first build
compiles a C++ toolchain (about a minute on an M-series Mac; needs `cmake`).
Metal is on automatically on Apple silicon; on NVIDIA add the `cuda` feature.
To skip the native build and talk to a hosted endpoint instead:

```toml
gen2 = { git = "…", default-features = false, features = ["backend-external-api"] }
```

Every example below is compiled by `cargo test --doc`, so none of them can
drift from the API. The design they follow is written down in
[`api_spec.md`](api_spec.md).

---

## The smallest useful thing

```rust,no_run
# fn main() -> gen2::Result<()> {
let model = gen2::load("/models/qwen3-0.6b-q4_k_m.gguf")?;
let answer = model.generate("Why is the sky blue?").text()?;
# println!("{answer}");
# Ok(())
# }
```

`gen2::load` inspects the file and the machine, picks a backend, sizes the
context window, loads the weights and hands back a `Model`. Nothing about
backends, controllers or KV caches reaches this line.

Configured:

```rust,no_run
# fn main() -> gen2::Result<()> {
# let model = gen2::load("/models/model.gguf")?;
let response = model
    .generate("Write a short story")
    .system("You write terse speculative fiction.")
    .temperature(0.8)
    .max_tokens(512)
    .run()?;

response.text();           // the prose
response.reasoning();      // a thinking model's working, kept apart from it
response.finish_reason();  // Stop | Length | ToolCall | Cancelled | …
response.usage();          // prompt and completion tokens
# Ok(())
# }
```

## A conversation

A `Runtime` can hold more than one model. A `Session` belongs to you and to
no model in particular.

```rust,no_run
use gen2::{Runtime, Session};
# fn main() -> gen2::Result<()> {
let runtime = Runtime::new()?;
let model = runtime.load("/models/model.gguf")?;

let mut session = Session::new().with_system("Be concise.");

model.turn(&mut session).user("My name is Bob").run()?;
let response = model.turn(&mut session).user("What is my name?").run()?;
assert!(response.text().contains("Bob"));
# Ok(())
# }
```

The system prompt is session state, not a message: `set_system` changes what
the model sees next turn, bumps the session's revision, and leaves the
transcript alone.

## Tools

gen2 renders tool definitions for the model and parses its calls. It never
executes one; your harness does, and the loop stays yours.

```rust,no_run
use gen2::{Session, tool_defs::{ToolDefinition, ToolSet}};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Weather { city: String }

# fn main() -> gen2::Result<()> {
# let model = gen2::load("/models/model.gguf")?;
let tools = ToolSet::new().with(
    ToolDefinition::new("get_weather")
        .description("Current weather for a city")
        .input_schema::<Weather>(),
);
let mut session = Session::new().with_tools(tools);

session.push_user("What is the weather in Paris?");
loop {
    let response = model.turn(&mut session).run()?;
    if response.tool_calls().is_empty() {
        println!("{}", response.text());
        break;
    }
    for call in response.tool_calls() {
        let args: Weather = call.parse_arguments().expect("schema-checked arguments");
        session.push_tool_result(call.id().as_str(), format!("{}: 18C, clear", args.city));
    }
}
# Ok(())
# }
```

A turn with no new user message is how a tool result reaches the model. Tools
are session state like the system prompt: `add_tool` and `remove_tool` change
the next turn and nothing that already happened.

## Streaming

Events are semantic, not tokenizer fragments. Reasoning arrives on its own
channel; a tool call arrives as a start, argument deltas and an end.

```rust,no_run
use gen2::{Session, event::Event};
# fn main() -> gen2::Result<()> {
# let model = gen2::load("/models/model.gguf")?;
let mut session = Session::new();
let mut stream = model.turn(&mut session).user("Inspect this repository").stream()?;
let cancel = stream.canceller();       // Clone + Send: hand it to a UI thread

while let Some(event) = stream.next() {
    match event? {
        Event::TextDelta(text) => print!("{text}"),
        Event::ReasoningDelta(_) => {}
        Event::ToolCallStart { name, .. } => println!("[{name}]"),
        _ => {}
    }
}
let response = stream.finish()?;       // the same Response run() returns
# let _ = (cancel, response);
# Ok(())
# }
```

`cancel.cancel()` from anywhere ends the stream with `FinishReason::Cancelled`
and whatever partial text was produced; the session records the partial
message under an id you can `remove_message` if you do not want it in context.

## Typed output

```rust,no_run
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Sentiment { label: Label, confidence: f32 }

#[derive(serde::Deserialize, schemars::JsonSchema)]
enum Label { Positive, Negative, Neutral }

# fn main() -> gen2::Result<()> {
# let model = gen2::load("/models/model.gguf")?;
let s: Sentiment = model
    .generate("Classify: 'the service was fantastic'")
    .greedy()
    .structured()?;
# let _ = s;
# Ok(())
# }
```

Where the backend can constrain decoding, the schema is enforced token by
token, so the result is your type and never prose around it. The same call
exists on a turn.

## Sessions are lossless

Every message has an id. Edits, removals and compaction change the active
projection and append to an event log; nothing rewrites history.

```rust
use gen2::{Message, Session};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut session = Session::new();
let original = session.push_user("helo");
let fixed = session.replace_message(original, Message::user("hello"))?;

assert_eq!(session.messages().len(), 1);        // the active conversation
assert_eq!(session.all_messages().len(), 2);    // every version ever pushed
session.events();                               // how it got here

// Compaction is your policy; gen2 only makes it reversible.
session.replace_messages([Message::user("Summary: we greeted each other.")])?;
session.restore_context([fixed])?;

let branch = session.fork();                    // new id, same past
let json = serde_json::to_string(&session)?;    // Serialize / Deserialize
# let _ = (branch, json);
# Ok(())
# }
```

A projection can never show the model a tool result without its call, or a
call without its result: an edit that would do so is refused.

## A hosted model on the same interface

```rust,no_run
use gen2::Runtime;
# fn main() -> gen2::Result<()> {
let runtime = Runtime::new()?;
let gpt = runtime
    .openai()
    .base_url("https://api.openai.com/v1")
    .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
    .model("gpt-5-mini")
    .connect()?;

let local = runtime.openai().base_url("http://localhost:11434/v1").model("qwen3:8b").connect()?;
# let _ = (gpt.capabilities(), local.capabilities());
# Ok(())
# }
```

A remote `Model` runs the same turns and sessions; `capabilities()` says what
it cannot do (grammar-constrained output, for one).

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

Behind the `tokio` feature, with the same builders and the same types.
Decoding is a blocking native call, so the async surface runs it on a blocking
task and bridges events through a bounded channel rather than pretending
otherwise:

```rust,ignore
let response = model.turn(&mut session).user("hello").run_async().await?;

let mut stream = model.turn(&mut session).user("hello").stream_async().await?;
while let Some(event) = stream.next().await { /* the same Event */ }
let response = stream.finish().await?;
```

Dropping the stream cancels the generation.

## The previous facade, and the layers below

`Engine`, `Engine::chat` and `Engine::agent` still compile: the agent loop with
its executable tools, approvals and budgets lives under `gen2::api` and is
moving out of the core surface, because deciding what invocation happens next
is a harness's job, not an inference runtime's. Embeddings and reranking are
`Engine::embed` and `Engine::rerank` until they get `Runtime` homes.

Below all of it, `gen2::advanced` is where local-only control lives: raw
grammars, residency, hardware, and a backend seam. A backend you write outside
this crate registers through `gen2::advanced::BackendPlugin`; `crates/gen2-mlxcel`
is one, and it is how the MLX fast path ships without a registry release.

The controller can also live in another process or on another machine.
Implement the transport, and everything above it is unchanged:

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

---

## Things that will otherwise cost you an afternoon

- **`.greedy()` is not the default.** An unconfigured turn leaves `temperature`
  and `seed` unset, which means backend-default sampling with a random seed. The
  same prompt gives different text each run.
- **A thinking model's working is `reasoning()`, not `text()`.** Qwen3 and
  Gemma 4 with thinking on stream it as `ReasoningDelta`; it never lands in the
  prose. `.reasoning(ThinkingMode::Off)` on a turn switches it off where the
  model allows.
- **Changing tools or the system prompt reopens the conversation.** Both live in
  the prompt prefix, so a change costs one re-prefill on the next turn. The
  alternative was ignoring the change without telling you.
- **A load that fails part-way leaves no model.** The path is checked before
  anything unloads, so a typo is refused up front. An out-of-memory mid-load
  cannot be undone.
- **Just drop the `Runtime` when you're done.** Each model's controller holds
  its backend on its own thread, and exiting while it runs aborts inside
  ggml's destructors. `Drop` stops and joins them for you.
- **A cancelled turn is a finish, not an error.** `finish_reason()` is
  `Cancelled`, `text()` holds what was generated before the stop, and the
  partial message is already in the session with an id.
- **Two clones of a `Model` are one model.** Handles are cheap and shareable;
  turns on them queue on one controller. Two `Session`s on one model are fine;
  one `Session` from two threads is not, and needs your own lock.

## Benchmarks

gen2 against `llama-bench` built from the same llama.cpp commit the crate links, on the same file; the method, the result schema and how to add a machine are in [benches/results/README.md](benches/results/README.md).

<!-- bench:begin -->
| Model | Machine | tg128 gen2 (tok/s) | tg128 llama-bench (tok/s) | tg128 ratio | pp512 gen2 (tok/s) † | pp512 llama-bench (tok/s) | pp512 ratio † | TTFT (ms) | llama.cpp | Date |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Llama-3.2-3B-Instruct Q4_K_M | Apple M4 Pro, 64 GB, macOS 26.3 (25D125) | 94.0 ± 0.8 | 101.1 ± 0.2 | 0.93 | 1057 ± 13 | 1110 ± 0 | 0.95 | 486.1 ± 6.1 | b10405 `e79e4bf6` | 2026-09-04 |
| Qwen3-0.6B Q4_K_M | Apple M4 Pro, 64 GB, macOS 26.3 (25D125) | 294.4 ± 4.1 | 310.2 ± 1.7 | 0.95 | 5656 ± 16 | 5923 ± 5 | 0.95 | 90.7 ± 0.3 | b10405 `e79e4bf6` | 2026-09-04 |

Median ± sample standard deviation over n=5 repetitions after one warmup, greedy, batch 1, both sides on the same GGUF and the same llama.cpp commit. Ratio is gen2 ÷ llama-bench. † pp512 through gen2 includes chat template + session setup, so it is reported, not targeted. TTFT is gen2's prefill-start-to-first-token; llama-bench has no equivalent. Raw samples, machine fingerprint and model hash: `benches/results/`.
<!-- bench:end -->

## Backends

Pick at least one — or none, and bring your own: `gen2::advanced::plugin`
registers a backend built outside the crate, ahead of every built-in rule. A
build with no backend feature compiles; a load then fails at run time, naming
both ways out, unless a registered plugin claims the path. Three tiers, by
what the crate can stand behind.

**Tier 1 — supported.** Built and tested in CI on every push; the README's
claims are about these.

| Feature | Backend |
| --- | --- |
| `backend-llamacpp` | llama.cpp (GGUF). **Default.** Add `metal`, `cuda`, or `vulkan`. |
| `backend-external-api` | OpenAI / Anthropic wire formats. Needs no C toolchain. |

**Mobile.** Supported the same way, on the platforms it exists for.

| Feature | Backend |
| --- | --- |
| `backend-litertlm` | LiteRT-LM (Google's on-device runtime), for `.litertlm` bundles. Loads Google's C ABI at run time — nothing is vendored, linked, or downloaded by a build. Point `GEN2_LITERTLM_LIBRARY` at the shared library, or install it where the platform loader finds it. |

**Experimental.** mistral.rs builds and runs its no-model tests in CI, MLX
gets a clippy pass. Real-model verification is by hand and noted in the
conformance suite. Interfaces and defaults may change.

| Feature | Backend |
| --- | --- |
| `backend-mistralrs` | mistral.rs. GGUF, safetensors and UQFF in one backend. Claims only formats no other compiled backend takes. No per-request seed: `.seed()` under sampling is refused rather than ignored. `metal` and `cuda` forward to it. |
| `backend-mlx` | MLX (Apple Silicon). Needs the Metal Toolchain component. Do not combine with the mlxcel companion below: both link MLX C++. |

mlxcel, the Mac fast path, is not a feature: it lives in `crates/gen2-mlxcel`,
a workspace companion that is never published (mlxcel has no registry release,
and crates.io refuses git dependencies) and joins through the plugin seam —
`Engine::builder().model(dir).backend(gen2_mlxcel::plugin())`. Build it with
`cargo build -p gen2-mlxcel`.

Not a backend, but chosen alongside them:

| Feature | What |
| --- | --- |
| `tokio` | Async API. Off by default. |

llama.cpp, mistral.rs, MLX and LiteRT-LM have been shown to generate a token.
The ONNX and Candle backends that used to sit beside them were
removed rather than kept unproven — a `.onnx` model now fails at load with an
error naming the format. Adding a backend never moves an existing one's models:
`backend-mistralrs` takes GGUF only where llama.cpp is absent and safetensors
only where MLX is, and `backend-litertlm` takes only the `.litertlm` bundles
nothing else can read. The conformance suite says which on every run, and fails
if that list goes stale.

LiteRT-LM's shipped runtime cannot report a bundle's context window, so state
it — `Engine::builder().model(path).context(4096)`. gen2 refuses the load
rather than guessing a number the controller would then plan against.

It asks for the GPU by default and falls back to the CPU through the same load
ladder every other backend uses — reported as `Degraded::GpuOffload`, not
hidden. What the GPU is worth on a given machine is a number for the benchmark
table above, not for this paragraph. It never asks for the NPU: that needs
vendor libraries for a specific chip, and without them the runtime accepts the
request and runs slower than the CPU.

`ios` and `android` carry it alongside llama.cpp. Nothing links: the runtime is
loaded through its C ABI at run time, so `cargo check --target
aarch64-apple-ios --features backend-litertlm` needs nothing but rustup, and CI
checks exactly that. Shipping the runtime itself is the host application's job
— Google publishes `liblitert-lm.so` for Android and a `CLiteRTLM` XCFramework
for iOS.

Publishable to crates.io: every dependency of the root crate resolves from
the registry, and the one backend whose dependencies cannot (mlxcel) lives in
the companion crate instead. CI runs `cargo publish --dry-run` so a release
cannot be surprised by a regression.

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
