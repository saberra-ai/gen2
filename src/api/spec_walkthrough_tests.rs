//! api_spec.md §28, compiled as written.
//!
//! Each block below is the spec's walkthrough snippet, changed only to add
//! the imports and the `?`-carrying `main` the spec leaves implicit (hidden
//! lines). If the spec's paths stop resolving at the root or in the §25
//! modules, `cargo test --doc` says so. Nothing here runs a model.
//!
//! # 28.1 Hello world
//!
//! ```no_run
//! fn main() -> gen2::Result<()> {
//!     let model = gen2::load("qwen3-8b.gguf")?;
//!
//!     println!(
//!         "{}",
//!         model.generate("Why is the sky blue?").text()?
//!     );
//!
//!     Ok(())
//! }
//! ```
//!
//! # 28.2 Configured generation
//!
//! ```no_run
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! let response = model
//!     .generate("Write a haiku about local inference")
//!     .temperature(0.8)
//!     .max_tokens(64)
//!     .run()?;
//!
//! println!("{}", response.text());
//! println!("{:?}", response.usage());
//! # Ok(())
//! # }
//! ```
//!
//! # 28.3 Persistent chat
//!
//! ```no_run
//! # use gen2::Session;
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! let mut session = Session::new()
//!     .with_system("Be concise.");
//!
//! let first = model
//!     .turn(&mut session)
//!     .user("Explain CRDTs")
//!     .run()?;
//!
//! let second = model
//!     .turn(&mut session)
//!     .user("Now compare them to Raft")
//!     .run()?;
//! # let _ = (first, second);
//! # Ok(())
//! # }
//! ```
//!
//! # 28.4 Switch models mid-chat
//!
//! ```no_run
//! # use gen2::{Runtime, Session};
//! # fn main() -> gen2::Result<()> {
//! let runtime = Runtime::new()?;
//! let qwen = runtime.load("qwen.gguf")?;
//! let gemma = runtime.load("gemma.gguf")?;
//!
//! let mut session = Session::new();
//!
//! qwen.turn(&mut session)
//!     .user("My name is Bob")
//!     .run()?;
//!
//! gemma.turn(&mut session)
//!     .user("What is my name?")
//!     .run()?;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.5 First-order tools
//!
//! ```no_run
//! # use gen2::{Session, ToolDefinition, ToolSet};
//! # #[derive(serde::Deserialize, schemars::JsonSchema)]
//! # struct BashArgs { command: String }
//! # #[derive(serde::Deserialize, schemars::JsonSchema)]
//! # struct ReadArgs { path: String }
//! # fn execute(call: &gen2::output::ToolCall) -> gen2::Result<String> { Ok(call.name().to_string()) }
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! let bash = ToolDefinition::new("bash")
//!     .description("Execute a shell command")
//!     .input_schema::<BashArgs>();
//!
//! let read = ToolDefinition::new("read")
//!     .description("Read a file")
//!     .input_schema::<ReadArgs>();
//!
//! let mut session = Session::new()
//!     .with_system("Inspect before editing.")
//!     .with_tools(ToolSet::new().with(bash).with(read));
//!
//! let response = model
//!     .turn(&mut session)
//!     .user("What files are in this repository?")
//!     .run()?;
//!
//! for call in response.tool_calls() {
//!     let result = execute(call)?;
//!     session.push_tool_result(call.id(), result);
//! }
//!
//! let response = model
//!     .turn(&mut session)
//!     .run()?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.6 Dynamic tool set
//!
//! ```no_run
//! # use gen2::{Session, ToolSet};
//! # fn read_only_tools() -> ToolSet { ToolSet::new() }
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! # let mut session = Session::new();
//! session.set_tools(read_only_tools());
//! model.turn(&mut session).run()?;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.7 Dynamic system prompt
//!
//! ```no_run
//! # use gen2::Session;
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! # let mut session = Session::new();
//! session.set_system("You are now in plan mode. Do not modify files.");
//! model.turn(&mut session).run()?;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.8 Edit a user message without losing history
//!
//! ```no_run
//! # use gen2::{Message, Session};
//! # fn main() -> gen2::Result<()> {
//! # let mut session = Session::new();
//! let id = session.push_user("helo");
//!
//! session.replace_message(
//!     id,
//!     Message::user("hello"),
//! )?;
//!
//! assert_eq!(session.messages().last().unwrap().text(), "hello");
//!
//! // Both revisions still exist.
//! for record in session.all_messages() {
//!     println!("{} active={}", record.message.text(), record.active);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # 28.9 Compact without erasing old history
//!
//! ```no_run
//! # use gen2::{Message, Session};
//! # fn main() -> gen2::Result<()> {
//! # let mut session = Session::new();
//! # let latest_message = Message::user("latest");
//! let compacted = vec![
//!     Message::user("Summary of prior context: ..."),
//!     latest_message,
//! ];
//!
//! session.replace_messages(compacted)?;
//!
//! // `messages()` is compacted.
//! // `all_messages()` and `events()` still retain pre-compaction history.
//! # Ok(())
//! # }
//! ```
//!
//! # 28.10 Streaming harness UI
//!
//! ```no_run
//! # use gen2::{Event, Session};
//! # fn render_text(_: String) {}
//! # fn render_reasoning(_: String) {}
//! # fn render_tool_start(_: String) {}
//! # fn render_tool_args(_: String) {}
//! # fn render_tool_call(_: gen2::output::ToolCall) {}
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! # let mut session = Session::new();
//! let mut stream = model
//!     .turn(&mut session)
//!     .user("Inspect the parser")
//!     .stream()?;
//!
//! while let Some(event) = stream.next() {
//!     match event? {
//!         Event::TextDelta(text) => render_text(text),
//!         Event::ReasoningDelta(text) => render_reasoning(text),
//!         Event::ToolCallStart { name, .. } => render_tool_start(name),
//!         Event::ToolCallArgumentsDelta { delta, .. } => render_tool_args(delta),
//!         Event::ToolCallEnd { call } => render_tool_call(call),
//!         _ => {}
//!     }
//! }
//!
//! let response = stream.finish()?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.11 Structured output
//!
//! ```no_run
//! # fn main() -> gen2::Result<()> {
//! # let model = gen2::load("qwen3-8b.gguf")?;
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct Invoice {
//!     vendor: String,
//!     total: f64,
//! }
//!
//! let invoice: Invoice = model
//!     .generate("Acme Ltd — total $1,240.00")
//!     .system("Extract the invoice fields")
//!     .structured()?;
//! # let _ = invoice;
//! # Ok(())
//! # }
//! ```
//!
//! # 28.12 OpenAI-compatible target
//!
//! ```no_run
//! # use gen2::{Runtime, Session};
//! # fn main() -> gen2::Result<()> {
//! let runtime = Runtime::new()?;
//!
//! let model = runtime
//!     .openai()
//!     .base_url("http://localhost:1234/v1")
//!     .model("qwen")
//!     .connect()?;
//!
//! let mut session = Session::new()
//!     .with_system("Be concise.");
//!
//! model.turn(&mut session)
//!     .user("hello")
//!     .run()?;
//! # Ok(())
//! # }
//! ```
