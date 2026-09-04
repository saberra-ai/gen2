# Gen2 Public Inference API Specification

**Status:** Proposed

Gen2 should own **local model inference superbly**.

Its public API should make the smallest useful thing trivial:

```rust
let model = gen2::load("qwen3-8b.gguf")?;
let answer = model.generate("Why is the sky blue?").text()?;
```

while remaining powerful enough to serve as the inference substrate for a serious coding-agent harness such as Pi.

The core boundary is:

> **Gen2 owns everything from “these are the instructions, tools, messages, and generation options for this model invocation” down to “the model produced these structured events efficiently.” The consumer owns what invocation happens next.**

That means system prompts, tools, sessions, model switching, structured output, streaming, cancellation, multimodal input, and local runtime state are first-order Gen2 concepts.

Agent loops, tool execution, approvals, planning, steering policy, compaction policy, memory, and workflow orchestration are not.

---

## 1. Goals

The public API should satisfy all of these at once.

### 1.1 Make local inference absurdly easy

The happy path should require no knowledge of backends, controllers, KV caches, residency policies, model formats, or hardware placement.

```rust
let model = gen2::load("model.gguf")?;
let text = model.generate("hello").text()?;
```

Gen2 should inspect the model and machine, select an appropriate backend, load the weights, configure sensible defaults, and make the model ready to use.

### 1.2 Preserve the power unique to local inference

Gen2 should be able to exploit facts remote-provider abstractions usually cannot:

- model residency and eviction;
- hardware-aware loading;
- backend selection;
- KV/prefix reuse;
- multiple warm sessions;
- model switching;
- context sizing;
- speculative decoding;
- structured decoding;
- local multimodal execution;
- accelerator contention and scheduling;
- embeddings and reranking runtimes.

These capabilities should improve behavior automatically before they become configuration knobs.

### 1.3 Be a complete model-turn primitive

A harness must be able to express one model turn completely:

- select a model;
- provide a system prompt;
- provide active tools;
- provide message history;
- add new user/tool messages;
- choose tool behavior;
- choose generation options;
- stream reasoning/text/tool calls;
- cancel generation;
- receive a structured assistant response;
- continue the same session afterward.

If Gen2 can express one turn perfectly, a harness can build whatever loop it wants above it.

### 1.4 Keep conversations independent from models

A conversation is not “Qwen’s chat.” A `Session` is application-owned state that any compatible `Model` can process.

```rust
let mut session = Session::new();

qwen.turn(&mut session)
    .user("My name is Bob")
    .run()?;

llama.turn(&mut session)
    .user("What is my name?")
    .run()?;
```

Gen2 handles whatever cache invalidation, reload, or re-prefill is required.

### 1.5 Preserve full history even when the active conversation is edited

Chat applications and agent harnesses edit history. They retry from earlier turns, branch, remove messages, replace messages, compact old context, change system prompts, and change tool sets.

The public API should expose both:

- the **active projection** currently sent to the model; and
- the **full append-only history** of what actually happened.

An edit must never require destroying the original record.

### 1.6 Make OpenAI-compatible inference plug-and-play

Gen2 is local-first, not local-only.

A remote OpenAI-compatible model should implement the same turn interface:

```rust
let model = runtime
    .openai()
    .base_url("http://localhost:11434/v1")
    .model("qwen3:8b")
    .connect()?;

let text = model.generate("hello").text()?;
```

Remote compatibility must not force Gen2 to pretend local and remote execution have identical runtime capabilities.

---

## 2. Non-goals

The core public inference API does **not** own:

- an autonomous agent loop;
- tool execution;
- tool retries;
- approval policy;
- permissions or risk classification;
- maximum agent steps;
- planning;
- follow-up queues;
- steering semantics;
- subagents;
- context-compaction policy;
- memory/RAG policy;
- workflow orchestration;
- UI state;
- MCP lifecycle;
- a universal cloud-provider framework.

Those layers may consume Gen2. They should not define Gen2.

There should be no requirement for a caller to understand `Agent`, `AgentRun`, `ApprovalMode`, `ExecutionPolicy`, `ControllerCmd`, residency internals, or backend internals in order to run inference.

---

## 3. The core object model

The normal conceptual vocabulary is:

```text
Runtime
  │
  ├── Model
  ├── Model
  ├── Embedder
  └── Reranker
        │
        │
Model ──┼──── runs against ──── Session
        │                         │
        │                         ├── current system prompt
        │                         ├── current tool set
        │                         ├── active messages
        │                         └── append-only history
        │
        └──── Turn ──── Response / EventStream
```

The critical separations are:

```text
Runtime != Model
Model   != Session
Session != Agent
Turn    != Agent loop
```

---

## 4. `Runtime`: machine and resource ownership

`Runtime` owns the resources required to make inference happen on this process/machine.

Conceptually it owns:

- backend instances;
- loaded/resident weights;
- hardware detection;
- model registry;
- session runtimes;
- KV/prefix state;
- local scheduling;
- memory/residency policy;
- auxiliary model runtimes;
- remote-provider clients.

The normal user should not need to configure most of this.

### 4.1 Construction

```rust
let runtime = gen2::Runtime::new()?;
```

Advanced construction may use a builder:

```rust
let runtime = gen2::Runtime::builder()
    .max_active_sessions(8)
    .build()?;
```

The exact low-level runtime knobs should remain deliberately small. Backend/controller implementation details should not leak into this builder.

### 4.2 Loading local models

```rust
let qwen = runtime.load("qwen3-8b.gguf")?;
let gemma = runtime.load("gemma-4b")?;
```

`Runtime::load` returns a logical `Model` handle.

A `Model` handle remains valid even if its weights are later evicted. Calling it again may cause Gen2 to restore the model to residency automatically.

The runtime is free to keep multiple models resident when hardware allows it and to evict/reload when it does not.

### 4.3 One-line shorthand

```rust
let model = gen2::load("qwen3-8b.gguf")?;
```

is shorthand for constructing a private runtime and loading one model into it.

The returned `Model` keeps that runtime alive.

### 4.4 Remote models

```rust
let gpt = runtime
    .openai()
    .base_url("https://api.openai.com/v1")
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model("gpt-5-mini")
    .connect()?;
```

OpenAI-compatible local servers use exactly the same path:

```rust
let ollama = runtime
    .openai()
    .base_url("http://localhost:11434/v1")
    .model("qwen3:8b")
    .connect()?;
```

A remote model participates in the common `Model` inference interface but naturally reports different runtime capabilities.

### 4.5 Advanced runtime controls

These are useful, but should live below the happy path:

```rust
runtime.hardware();
runtime.residency();
runtime.preload(&model)?;
runtime.evict(&model)?;
runtime.stats();
```

They may live under `gen2::runtime` or `gen2::advanced` rather than at crate root.

---

## 5. `Model`: a logical inference target

A `Model` is the thing application code asks to perform inference.

Conceptually:

```rust
pub struct Model {
    runtime: Arc<RuntimeInner>,
    id: ModelId,
}
```

The actual representation is private.

A `Model` should be cheap to clone and suitable for sharing:

```rust
let a = model.clone();
let b = model.clone();
```

The model does **not** own a conversation.

### 5.1 Core methods

```rust
impl Model {
    pub fn generate(&self, input: impl Into<Input>) -> Generation<'_>;

    pub fn turn<'a>(&'a self, session: &'a mut Session) -> Turn<'a>;

    pub fn info(&self) -> ModelInfo;
    pub fn capabilities(&self) -> ModelCapabilities;
    pub fn id(&self) -> ModelId;
}
```

### 5.2 Model information

`ModelInfo` should describe the logical model without exposing backend internals unnecessarily.

Possible shape:

```rust
pub struct ModelInfo {
    pub id: ModelId,
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub context_window: Option<u32>,
    pub source: ModelSourceKind,
    pub local: bool,
}
```

Local-only details such as format, quantization, weight size, and backend may be available through an advanced info surface.

### 5.3 Capabilities

```rust
pub struct ModelCapabilities {
    pub text: bool,
    pub images: bool,
    pub audio: bool,
    pub tools: bool,
    pub reasoning: bool,
    pub structured_output: bool,
}
```

Capabilities should describe what the caller can ask the model to do.

Backend/runtime machinery such as KV snapshotting or poison detection belongs to an advanced runtime capability surface, not the normal model capability type.

---

## 6. One-shot generation

The simplest useful API is:

```rust
let model = gen2::load("qwen.gguf")?;
let text = model.generate("Explain CRDTs simply").text()?;
```

A one-shot generation uses an ephemeral `Session` internally and discards it afterward.

Conceptually:

```text
Model::generate(input)
       │
       ▼
  Generation builder
       │
       ▼
 ephemeral Session
       │
       ▼
      Turn
```

### 6.1 Configuration

```rust
let response = model
    .generate("Write a short story")
    .system("You write terse speculative fiction.")
    .temperature(0.8)
    .max_tokens(512)
    .run()?;
```

### 6.2 Tools in one-shot inference

Tool declarations are valid even without a persistent session:

```rust
let response = model
    .generate("What is the weather in Paris?")
    .system("Use tools when appropriate.")
    .tools(tools)
    .run()?;
```

If a returned tool call needs a follow-up model invocation, the caller should use a persistent `Session` instead of one-shot generation.

### 6.3 Multimodal one-shot inference

```rust
let response = model
    .generate("What is unusual about this image?")
    .image("photo.png")
    .run()?;
```

or with explicit input parts:

```rust
let response = model
    .generate(
        Input::new()
            .text("Compare these images")
            .image("a.png")
            .image("b.png"),
    )
    .run()?;
```

---

## 7. `Session`: model-agnostic conversational state

A `Session` is application-owned conversation state.

It has four important pieces:

```rust
pub struct Session {
    id: SessionId,

    // Current model-visible instruction state.
    system: String,
    tools: ToolSet,

    // Current logical conversation after edits/compaction/rewinds.
    messages: Vec<Message>,

    // Append-only authoritative history from which the current projection can
    // be explained or reconstructed.
    history: Vec<SessionEvent>,
}
```

This is a conceptual shape. The implementation may derive/cache the active projection instead of storing both representations literally.

### 7.1 Three distinct notions of history

Gen2 should explicitly distinguish three things.

#### A. Full session history

Everything that happened, append-only:

```rust
session.events()
```

Edits do not delete old events.

#### B. All message records

Every immutable message version that has ever existed in the session:

```rust
session.all_messages()
```

If a user edits a message from `"helo"` to `"hello"`, both versions remain available here.

#### C. Active messages

The current logical conversation after edits, retries, removals, rewinds, or compaction:

```rust
session.messages()
```

These are the messages Gen2 considers when constructing the next model context.

The model may see an even smaller runtime view if its context window forces truncation. That is execution state, not durable session history.

This yields:

```text
append-only SessionEvent history
             │
             ▼
      all immutable messages
             │
             ▼
       active projection
             │
             ▼
 runtime/model context window
```

No layer destroys the layer above it.

### 7.2 Basic session API

```rust
let mut session = Session::new();

session.set_system("You are concise.");
session.set_tools(tools);

session.push_user("Hello");
```

Core accessors:

```rust
session.id();
session.revision();
session.system();
session.tools();
session.messages();
session.all_messages();
session.events();
```

### 7.3 System prompt is first-order state

System instructions are not ordinary historical chat messages.

```rust
session.set_system("You are a coding assistant.");
session.append_system("\nThe repository has uncommitted changes.");
```

Changing the system prompt:

- changes future model context;
- increments the session revision;
- is recorded in `SessionEvent` history;
- may invalidate/rebuild a model prefix/KV cache;
- does **not** rewrite message history.

Builder convenience:

```rust
let session = Session::new()
    .with_system("Be concise.");
```

### 7.4 Tools are first-order state

```rust
session.set_tools(tool_set);
session.add_tool(read_tool);
session.remove_tool("write");
```

Changing tools:

- changes future model context;
- increments the session revision;
- is recorded in session history;
- may invalidate/rebuild the prompt prefix;
- does not mutate chat messages.

Tool ordering must be deterministic because tool definitions may participate in prompt-prefix/KV reuse.

### 7.5 Message identity

Every message receives a stable `MessageId`.

```rust
let id = session.push_user("hello");
```

Messages are immutable records. Editing creates a replacement record rather than mutating the old record in place.

### 7.6 Editing while preserving history

```rust
let original = session.push_user("helo");

let replacement = session.replace_message(
    original,
    Message::user("hello"),
)?;
```

Afterward:

```text
session.messages()
    -> contains "hello"

session.all_messages()
    -> contains "helo" and "hello"

session.events()
    -> explains that replacement replaced original
```

This is important for:

- chat edit UI;
- retries;
- branch navigation;
- audit/debugging;
- agent traces;
- compaction provenance.

### 7.7 Removing messages

```rust
session.remove_message(id)?;
```

Removal means “not part of the active projection.” It does not erase the historical record.

### 7.8 Replacing active context / compaction

A harness chooses when and how to compact. Gen2 only provides a lossless state mutation primitive.

For example:

```rust
let summary = Message::user(
    "Context summary: we inspected the parser and found the bug in tokenize.rs...",
);

session.replace_messages([
    summary,
    latest_user_message,
])?;
```

The old messages disappear from `messages()` but remain available through `all_messages()` and `events()`.

The mutation is recorded as a context replacement rather than destructive deletion.

A lower-level API may allow restoring an earlier active projection by message IDs:

```rust
session.restore_context(previous_message_ids)?;
```

This makes undo, branch navigation, and retry-from-here possible without duplicating or losing historical records.

### 7.9 Forking

```rust
let branch = session.fork();
```

A fork:

- receives a new `SessionId`;
- begins with the same current system prompt, tools, active messages, and historical provenance;
- receives independent future events and runtime cache state.

Optionally:

```rust
let branch = session.fork_at(message_id)?;
```

may create a branch whose active projection ends at a selected historical point.

### 7.10 Session revision

Every context-affecting mutation increments a monotonic `SessionRevision`:

- system prompt changes;
- tool-set changes;
- message append;
- message replacement;
- message removal;
- active-context replacement;
- restore/rewind.

This gives the runtime a cheap correctness signal while full fingerprints can still be used for cache validation.

---

## 8. Append-only session events

The session event log is the authoritative history.

A possible initial public shape is:

```rust
#[non_exhaustive]
pub enum SessionEvent {
    SystemSet {
        system: String,
    },

    ToolsSet {
        tools: ToolSet,
    },

    MessageAdded {
        id: MessageId,
        message: Message,
    },

    MessageReplaced {
        old: MessageId,
        new: MessageId,
        message: Message,
    },

    MessageRemoved {
        id: MessageId,
    },

    ContextReplaced {
        active: Vec<MessageId>,
        reason: Option<String>,
    },
}
```

Exact variants may evolve, but the invariant is more important than the encoding:

> **A session mutation appends history; it does not rewrite history.**

`Session::messages()` is therefore a projection, not the authoritative record.

### 8.1 Message records

`Session::all_messages()` may expose records like:

```rust
pub struct MessageRecord<'a> {
    pub id: MessageId,
    pub message: &'a Message,
    pub active: bool,
    pub replaced_by: Option<MessageId>,
}
```

The event stream remains the lossless explanation of how those records reached their current state.

---

## 9. Message model

System prompts are intentionally separate from the chat message enum.

A proposed message shape is:

```rust
#[non_exhaustive]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolResultMessage),
}
```

### 9.1 User messages

```rust
pub struct UserMessage {
    pub content: Vec<InputPart>,
}
```

Convenience:

```rust
Message::user("hello")
```

Multimodal:

```rust
Message::user_parts([
    InputPart::text("Describe this image"),
    InputPart::image("photo.png"),
])
```

### 9.2 Input parts

```rust
#[non_exhaustive]
pub enum InputPart {
    Text(String),
    Image(Image),
    Audio(Audio),
}
```

Additional modalities may be added without introducing combinatorial methods such as `generate_with_image_and_audio`.

### 9.3 Assistant messages

```rust
pub struct AssistantMessage {
    pub content: Vec<OutputPart>,
}
```

```rust
#[non_exhaustive]
pub enum OutputPart {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
}
```

This preserves model output structure instead of flattening every response into a single string.

### 9.4 Tool results

```rust
pub struct ToolResultMessage {
    pub call_id: ToolCallId,
    pub content: ToolResult,
}
```

```rust
#[non_exhaustive]
pub enum ToolResult {
    Text(String),
    Json(serde_json::Value),
}
```

Convenience:

```rust
session.push_tool_result(call.id(), "18C, clear");
```

The exact wire formatting required by a model/backend remains Gen2’s responsibility.

---

## 10. Tools: model protocol, not execution

Tools are first-order inference context, but Gen2 core does not execute them.

### 10.1 Tool definition

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}
```

Builder conveniences may derive schemas from Rust types:

```rust
let read = ToolDefinition::new("read")
    .description("Read a file")
    .input_schema::<ReadArgs>();
```

Tool definitions contain data the model needs. They do not contain an execution closure.

### 10.2 Tool set

```rust
let tools = ToolSet::new()
    .with(read)
    .with(write)
    .with(bash);
```

`ToolSet` should preserve deterministic ordering and provide a stable fingerprint suitable for prompt-prefix caching.

### 10.3 Tool choice

```rust
#[non_exhaustive]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}
```

A turn can specify:

```rust
model.turn(&mut session)
    .tool_choice(ToolChoice::Auto)
    .run()?;
```

### 10.4 Tool calls

```rust
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: serde_json::Value,
}
```

A model response may contain zero or more tool calls.

Gen2 guarantees the call/result relationship is preserved by `ToolCallId`, including parallel tool calls.

### 10.5 Deliberately absent from inference core

Core Gen2 should not require:

```rust
trait Tool {
    async fn execute(...);
}
```

Nor should a turn expose `.on_tool(...)` or `.max_steps(...)`.

A higher-level convenience crate/module may eventually adapt executable Rust tools into `ToolDefinition`s, but the inference contract stops at declaring calls and accepting results.

---

## 11. `Turn`: the fundamental stateful invocation

A `Turn` is one invocation of one `Model` against one `Session`.

```rust
let response = model
    .turn(&mut session)
    .user("Fix the failing parser test")
    .temperature(0.2)
    .max_tokens(4096)
    .run()?;
```

Conceptually:

```rust
pub struct Turn<'a> {
    model: &'a Model,
    session: &'a mut Session,
    staged_messages: Vec<Message>,
    options: GenerationOptions,
    tool_choice: ToolChoice,
}
```

The exact fields are private.

### 11.1 Message staging

A turn may stage new messages:

```rust
turn.user("hello");
turn.message(message);
turn.messages(messages);
turn.image("photo.png");
```

Staged messages are committed when execution begins.

If validation fails before the model invocation starts, staged messages should not be left behind accidentally.

### 11.2 Running with no new user message

This must be valid:

```rust
let response = model
    .turn(&mut session)
    .run()?;
```

It is required for tool continuation:

```text
assistant requests tool
        ↓
harness executes tool
        ↓
harness pushes tool result
        ↓
model runs again with no new user message
```

### 11.3 Per-turn generation controls

Common controls live directly on the builder:

```rust
turn.max_tokens(4096)
turn.temperature(0.7)
turn.seed(42)
turn.top_p(0.9)
turn.top_k(40)
turn.greedy()
turn.tool_choice(ToolChoice::Auto)
```

The full object is available through:

```rust
turn.options(options)
```

### 11.4 Per-turn system/tools overrides

Persistent system/tool state belongs on `Session`.

If temporary overrides are needed, they should be named explicitly so persistence is never ambiguous:

```rust
turn.system_override("For this turn only, answer as JSON.")
turn.tools_override(temp_tools)
```

One-shot `Generation` may simply use `.system(...)` and `.tools(...)` because its session is ephemeral.

---

## 12. Generation options

The common options object should describe inference semantics that make sense across multiple backends.

```rust
#[non_exhaustive]
pub struct GenerationOptions {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub reasoning: ReasoningMode,
}
```

Gen2 already has valuable expert controls such as repetition penalties, DRY, XTC, speculative decoding, constrained grammars, and diffusion denoising controls.

They should remain reachable without making the ordinary surface noisy.

Possible organization:

```rust
let options = GenerationOptions::default()
    .temperature(0.7)
    .advanced(
        AdvancedGenerationOptions::default()
            .min_p(0.05)
            .speculative(Speculative::Ngram),
    );
```

or via `gen2::advanced`.

The exact grouping is less important than keeping the happy path small.

### 12.1 Backend-specific options

Backend-specific tuning should not become permanent generic fields merely because one backend exposes them.

A typed escape hatch may exist under an advanced surface.

---

## 13. Structured output

Structured output is an inference feature, not an agent feature.

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Person {
    name: String,
    age: u8,
}

let person: Person = model
    .generate("Bob is 31 years old")
    .structured()?;
```

The same should work on a session turn:

```rust
let decision: Decision = model
    .turn(&mut session)
    .user("Choose the best option")
    .structured()?;
```

Where a local backend supports grammar-constrained decoding, Gen2 should enforce the schema during decoding rather than relying only on prompt compliance.

### 13.1 Convenience operations

High-value conveniences may remain:

```rust
let invoice: Invoice = model.extract(text).value()?;

let sentiment = model
    .classify(text)
    .labels(["positive", "negative", "neutral"])
    .label()?;
```

They should be implemented conceptually on top of generation + constrained output rather than becoming separate engines.

---

## 14. Response model

The canonical model output is structured.

```rust
pub struct Response {
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub stats: GenerationStats,
}
```

Convenience accessors:

```rust
response.text();
response.reasoning();
response.tool_calls();
response.finish_reason();
response.usage();
response.stats();
```

One-shot convenience:

```rust
let text = model.generate("hello").text()?;
```

A harness should never be forced to parse model-protocol syntax out of that string in order to discover tool calls or reasoning.

### 14.1 Finish reasons

A stable public enum may include:

```rust
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    Cancelled,
    ContentFilter,
    Error,
    Other(String),
}
```

Backend-specific termination reasons are normalized where possible while raw details remain available through advanced stats/error data.

---

## 15. Semantic streaming

Streaming must expose semantic events, not only tokenizer fragments.

```rust
#[non_exhaustive]
pub enum Event {
    TextDelta(String),
    ReasoningDelta(String),

    ToolCallStart {
        id: ToolCallId,
        name: String,
    },

    ToolCallArgumentsDelta {
        id: ToolCallId,
        delta: String,
    },

    ToolCallEnd {
        call: ToolCall,
    },

    Usage(Usage),

    Finished(FinishReason),
}
```

A backend that only exposes complete tool calls can synthesize the same semantic event sequence.

### 15.1 Streaming usage

```rust
let mut stream = model
    .turn(&mut session)
    .user("Inspect this repository")
    .stream()?;

while let Some(event) = stream.next() {
    match event? {
        Event::TextDelta(text) => ui.append_text(text),
        Event::ReasoningDelta(text) => ui.append_reasoning(text),
        Event::ToolCallStart { name, .. } => ui.show_tool(name),
        _ => {}
    }
}

let response = stream.finish()?;
```

`finish()` returns the same `Response` shape as non-streaming execution.

A stream should be usable by terminals, Tauri applications, servers, websocket bridges, and agent harnesses without each consumer having to reverse-engineer backend token conventions.

---

## 16. Cancellation

Cancellation is a first-order inference primitive.

```rust
let mut stream = model
    .turn(&mut session)
    .user(prompt)
    .stream()?;

let cancel = stream.canceller();
```

From another task/thread:

```rust
cancel.cancel();
```

The invocation finishes as cancelled and exposes whatever partial structured output was already produced.

### 16.1 Partial output and history

If a cancelled invocation has produced a meaningful partial assistant message, the session may record it as a partial assistant response.

Because session edits are lossless, a harness that does not want that partial message in future active context can remove it from the active projection without destroying its history:

```rust
let response = stream.finish()?;

if response.finish_reason() == FinishReason::Cancelled {
    if let Some(id) = response.message_id() {
        session.remove_message(id)?;
    }
}
```

This makes higher-level steering policy possible without adding `steer`, `follow_up`, or `interrupt` semantics to Gen2 itself.

---

## 17. Model switching

Model switching is a normal chat operation, not a special migration.

```rust
let runtime = Runtime::new()?;

let qwen = runtime.load("qwen.gguf")?;
let llama = runtime.load("llama.gguf")?;

let mut session = Session::new()
    .with_system("Be concise.");

qwen.turn(&mut session)
    .user("My name is Bob")
    .run()?;

llama.turn(&mut session)
    .user("What is my name?")
    .run()?;
```

The session is unchanged merely because the model changed.

Gen2 decides whether the new invocation requires:

- loading weights;
- evicting another model;
- restoring an already-resident model;
- creating a new model-specific session runtime;
- re-prefilling the active transcript;
- reusing an existing prefix/KV state.

The application should not manually invalidate caches.

### 17.1 Cache identity

An implementation may cache model/session state using concepts such as:

```text
(ModelId, SessionId, ContextFingerprint)
```

where `ContextFingerprint` includes everything that changes the model-visible prefix/context, including:

- system prompt;
- tool definitions and ordering;
- active message identities/content;
- prompt/template configuration;
- relevant model-specific formatting.

`SessionRevision` is a cheap mutation signal; a stronger fingerprint proves cache validity.

### 17.2 Switching away and back

This sequence should be valid:

```rust
qwen.turn(&mut session).run()?;
llama.turn(&mut session).run()?;
qwen.turn(&mut session).run()?;
```

If Qwen remains resident and Gen2 can prove a previous Qwen prefix is still valid, it may reuse it and append only the missing context.

If it cannot prove validity, it re-prefills correctly.

Correctness is mandatory. Reuse is an optimization.

This is exactly the kind of local-inference complexity Gen2 should absorb for consumers.

---

## 18. Dynamic system prompts and tool sets

A harness can change its behavior without rebuilding the session.

### 18.1 Change system prompt

```rust
session.set_system(plan_mode_prompt());
model.turn(&mut session).run()?;
```

### 18.2 Change active tools

```rust
session.set_tools(read_only_tools());
model.turn(&mut session).run()?;
```

### 18.3 Runtime effect

Gen2 notices the context revision/fingerprint changed and automatically rebuilds whatever model-specific state is no longer valid.

The harness never needs to know which backend renders tool definitions in a prompt prefix, which backend sends them as structured request fields, or which backend needs a full re-prefill.

---

## 19. A Pi-class harness built on Gen2

Supporting this cleanly is an acceptance criterion for the inference API.

Harness state can remain tiny:

```rust
struct HarnessState {
    model: Model,
    session: Session,
}
```

Initialization:

```rust
let runtime = Runtime::new()?;

let model = runtime.load("qwen3-coder.gguf")?;

let session = Session::new()
    .with_system(build_system_prompt())
    .with_tools(tool_definitions());

let mut state = HarnessState { model, session };
```

User input:

```rust
state.session.push_user(user_prompt);
```

The agent loop belongs entirely to the harness:

```rust
loop {
    let response = state
        .model
        .turn(&mut state.session)
        .run()?;

    if response.tool_calls().is_empty() {
        break;
    }

    for call in response.tool_calls() {
        let result = execute_tool(call).await?;

        state
            .session
            .push_tool_result(call.id(), result);
    }
}
```

Change model:

```rust
state.model = runtime.load("another-model.gguf")?;
```

Change tools:

```rust
state.session.set_tools(read_only_tools());
```

Change instructions:

```rust
state.session.set_system(plan_mode_prompt());
```

Edit a prior message:

```rust
state.session.replace_message(message_id, replacement)?;
```

Branch:

```rust
let branch = state.session.fork();
```

Compact:

```rust
state.session.replace_messages(compacted_messages)?;
```

Cancel:

```rust
run.canceller().cancel();
```

Nothing here requires a Gen2 `Agent` abstraction.

That is the desired boundary.

---

## 20. Compaction belongs above Gen2

Gen2 should provide facts and mutation primitives:

```rust
model.context_window();
model.count_tokens(&session)?;
session.messages();
session.replace_messages(compacted)?;
```

It should not decide:

- when to compact;
- what should be summarized;
- which model performs summarization;
- how tool calls are preserved;
- whether old context is dropped or summarized;
- whether a branch is created.

A harness can implement any policy it wants while Gen2 preserves the pre-compaction history through `all_messages()` and `events()`.

---

## 21. Embeddings and reranking

Embeddings and reranking should share runtime resource management but should not be forced through `Turn` or pretend to be generative models.

### 21.1 Embedder

```rust
let embedder = runtime.load_embedder("embeddinggemma.gguf")?;

let vectors = embedder.embed([
    "first document",
    "second document",
])?;
```

### 21.2 Reranker

```rust
let reranker = runtime.load_reranker("bge-reranker.gguf")?;

let ranked = reranker.rerank(query, documents)?;
```

Possible conceptual types:

```rust
pub struct Embedder { /* runtime + id */ }
pub struct Reranker { /* runtime + id */ }
```

They may participate in the same residency/scheduling machinery as generative models.

---

## 22. Sync and async execution

The builder vocabulary should be shared between blocking and async execution.

Blocking:

```rust
let response = model
    .turn(&mut session)
    .user("hello")
    .run()?;
```

Optional Tokio surface:

```rust
let response = model
    .turn(&mut session)
    .user("hello")
    .run_async()
    .await?;
```

Likewise:

```rust
turn.stream()?;
turn.stream_async().await?;
```

Async support should not introduce a parallel object model.

---

## 23. Concurrency semantics

`Model` handles should be cloneable/shareable.

A mutable `Session` represents one ordered conversational timeline and should not be mutated concurrently without application synchronization.

Different sessions may invoke the same model concurrently:

```rust
model_a.turn(&mut session_a)...
model_b.turn(&mut session_b)...
```

The runtime decides whether the underlying backend truly runs them concurrently, interleaves them, or queues them.

Backend thread-safety limitations must not leak into the normal public API.

---

## 24. Error model

Public errors should describe user-observable inference failures rather than controller internals.

Possible top-level categories:

```rust
#[non_exhaustive]
pub enum Error {
    Load(LoadError),
    InvalidInput(InvalidInputError),
    Unsupported(UnsupportedError),
    Context(ContextError),
    Generation(GenerationError),
    Runtime(RuntimeError),
    Remote(RemoteError),
}
```

Cancellation is preferably a normal finish reason when a generation has begun rather than an opaque backend error.

Backend-specific diagnostics may be attached as source/details without forcing callers to pattern-match backend names.

---

## 25. Root namespace

The crate root should be intentionally boring.

A normal consumer should mostly see:

```rust
use gen2::{
    Runtime,
    Model,
    Session,
    Message,
    ToolDefinition,
    ToolSet,
    ToolChoice,
    GenerationOptions,
    Response,
    Event,
    Error,
    Result,
};
```

Supporting types may live in coherent modules:

```text
gen2::model
    ModelId
    ModelInfo
    ModelCapabilities

gen2::session
    SessionId
    SessionRevision
    SessionEvent
    MessageId
    MessageRecord

gen2::input
    Input
    InputPart
    Image
    Audio

gen2::output
    AssistantMessage
    OutputPart
    ToolCall
    ToolCallId
    Usage
    GenerationStats
    FinishReason

gen2::advanced
    local/backend-specific tuning
    residency controls
    hardware details
    raw grammar controls
```

Controller commands, backend traits, memory-governor details, and residency implementation structs should not sit beside `Model` and `Session` in the root namespace.

---

## 26. What should leave the primary public API

The current agent-oriented surface should not define the crate’s primary public identity.

The following concepts should be removed from the root inference API or moved to a higher-level/optional layer:

```text
Agent
AgentConfig
AgentRun
AgentStep
ApprovalMode
Decision
Risk
Steering
Struggle
ExecutionPolicy
ToolRegistry
FunctionTool execution
agent budgets / max steps
agent tool search / hydration policy
```

This is not because these concepts are bad.

They are simply decisions about **what model invocation should happen next**.

Gen2’s core responsibility ends one layer lower.

---

## 27. Migration from the current facade

A rough conceptual migration is:

| Current | Proposed |
| --- | --- |
| `Engine` as main user object | `Runtime` internally/resource-wise, `Model` as inference target |
| `Engine::load(path)` | `gen2::load(path)` or `Runtime::load(path)` |
| `Engine::infer(text)` | `Model::generate(text)` |
| `Inference` | `Generation` |
| `Engine::chat(&mut session)` | `Model::turn(&mut session)` |
| current `Chat` inference controls | `Turn` |
| `Chat::on_tool` loop | remove from core; harness owns loop |
| `Engine::agent` | remove from core inference surface |
| `GenSpec` | `GenerationOptions` |
| `Completion` | `Response` |
| token-centric stream | semantic `EventStream` |
| system as message | first-order `Session::system` state |
| tool set partly turn/agent state | first-order `Session::tools` state |
| mutable transcript only | active projection + append-only event history |
| model generation invalidates one engine session cache | model/session/context-fingerprint cache identity |
| OpenAI/Anthropic builder mode | remote `Model` constructors under `Runtime` |

This does not require discarding the current controller/backend implementation. Much of the existing machinery can remain below the new facade.

---

## 28. API walkthrough

### 28.1 Hello world

```rust
fn main() -> gen2::Result<()> {
    let model = gen2::load("qwen3-8b.gguf")?;

    println!(
        "{}",
        model.generate("Why is the sky blue?").text()?
    );

    Ok(())
}
```

### 28.2 Configured generation

```rust
let response = model
    .generate("Write a haiku about local inference")
    .temperature(0.8)
    .max_tokens(64)
    .run()?;

println!("{}", response.text());
println!("{:?}", response.usage());
```

### 28.3 Persistent chat

```rust
let mut session = Session::new()
    .with_system("Be concise.");

let first = model
    .turn(&mut session)
    .user("Explain CRDTs")
    .run()?;

let second = model
    .turn(&mut session)
    .user("Now compare them to Raft")
    .run()?;
```

### 28.4 Switch models mid-chat

```rust
let runtime = Runtime::new()?;
let qwen = runtime.load("qwen.gguf")?;
let gemma = runtime.load("gemma.gguf")?;

let mut session = Session::new();

qwen.turn(&mut session)
    .user("My name is Bob")
    .run()?;

gemma.turn(&mut session)
    .user("What is my name?")
    .run()?;
```

### 28.5 First-order tools

```rust
let bash = ToolDefinition::new("bash")
    .description("Execute a shell command")
    .input_schema::<BashArgs>();

let read = ToolDefinition::new("read")
    .description("Read a file")
    .input_schema::<ReadArgs>();

let mut session = Session::new()
    .with_system("Inspect before editing.")
    .with_tools(ToolSet::new().with(bash).with(read));

let response = model
    .turn(&mut session)
    .user("What files are in this repository?")
    .run()?;

for call in response.tool_calls() {
    let result = execute(call)?;
    session.push_tool_result(call.id(), result);
}

let response = model
    .turn(&mut session)
    .run()?;
```

### 28.6 Dynamic tool set

```rust
session.set_tools(read_only_tools());
model.turn(&mut session).run()?;
```

### 28.7 Dynamic system prompt

```rust
session.set_system("You are now in plan mode. Do not modify files.");
model.turn(&mut session).run()?;
```

### 28.8 Edit a user message without losing history

```rust
let id = session.push_user("helo");

session.replace_message(
    id,
    Message::user("hello"),
)?;

assert_eq!(session.messages().last().unwrap().text(), "hello");

// Both revisions still exist.
for record in session.all_messages() {
    println!("{} active={}", record.message.text(), record.active);
}
```

### 28.9 Compact without erasing old history

```rust
let compacted = vec![
    Message::user("Summary of prior context: ..."),
    latest_message,
];

session.replace_messages(compacted)?;

// `messages()` is compacted.
// `all_messages()` and `events()` still retain pre-compaction history.
```

### 28.10 Streaming harness UI

```rust
let mut stream = model
    .turn(&mut session)
    .user("Inspect the parser")
    .stream()?;

while let Some(event) = stream.next() {
    match event? {
        Event::TextDelta(text) => render_text(text),
        Event::ReasoningDelta(text) => render_reasoning(text),
        Event::ToolCallStart { name, .. } => render_tool_start(name),
        Event::ToolCallArgumentsDelta { delta, .. } => render_tool_args(delta),
        Event::ToolCallEnd { call } => render_tool_call(call),
        _ => {}
    }
}

let response = stream.finish()?;
```

### 28.11 Structured output

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
}

let invoice: Invoice = model
    .generate("Acme Ltd — total $1,240.00")
    .system("Extract the invoice fields")
    .structured()?;
```

### 28.12 OpenAI-compatible target

```rust
let runtime = Runtime::new()?;

let model = runtime
    .openai()
    .base_url("http://localhost:1234/v1")
    .model("qwen")
    .connect()?;

let mut session = Session::new()
    .with_system("Be concise.");

model.turn(&mut session)
    .user("hello")
    .run()?;
```

The rest of the harness does not care that this model is remote.

---

## 29. Design invariants

These should be treated as stronger than any particular method name.

### Invariant 1: local-first happy path

A local model must remain runnable in roughly two lines.

```rust
let model = gen2::load(path)?;
let text = model.generate(prompt).text()?;
```

### Invariant 2: sessions are model-agnostic

Switching models never requires converting, recreating, or migrating the application’s conversation object.

### Invariant 3: system prompts and tools are first-order inference context

They are not hidden inside an agent abstraction.

### Invariant 4: Gen2 never needs to execute a tool to support tool use

The model declares calls. The consumer executes them. Gen2 accepts results.

### Invariant 5: edits are lossless

`Session::messages()` may change. `Session::events()` never lies about what happened.

### Invariant 6: active context and historical record are separate

Compaction, deletion, retries, and branch navigation alter the active projection without destroying old message records.

### Invariant 7: cache behavior is invisible to correctness

A consumer can change models, system prompts, tools, or messages freely. Gen2 either proves cached state is valid or rebuilds it.

### Invariant 8: one turn is complete enough to build a harness

A caller must never need `Agent` merely to provide instructions, tools, history, tool results, structured output, streaming, or cancellation.

### Invariant 9: remote compatibility does not flatten local capabilities

Remote models share the turn protocol. Local runtime controls remain local-runtime controls.

### Invariant 10: implementation concepts do not become user concepts by accident

Controllers, backend traits, FFI constraints, memory governors, cache layouts, and scheduler commands remain below the facade unless a caller has a direct and stable reason to control them.

---

## 30. The litmus test for future API additions

Every proposed normal public feature should answer whether it is fundamentally one of:

1. a way to create/configure a `Runtime`;
2. a way to obtain/select a `Model`;
3. session/inference context (`system`, `tools`, `messages`);
4. an input modality;
5. a generation option;
6. an output/event shape;
7. a session/history mutation;
8. an auxiliary inference primitive such as embedding/reranking.

If a feature primarily decides **what invocation should happen next**, it probably belongs above Gen2.

If a feature primarily decides **how this invocation is represented or executed**, it probably belongs in Gen2.

That is the boundary that keeps Gen2 simple without making it weak.

---

## 31. Short form

The desired public experience is:

```rust
let runtime = Runtime::new()?;
let model = runtime.load("qwen.gguf")?;

let mut session = Session::new()
    .with_system(SYSTEM)
    .with_tools(TOOLS);

loop {
    let response = model
        .turn(&mut session)
        .run()?;

    if response.tool_calls().is_empty() {
        break;
    }

    for call in response.tool_calls() {
        session.push_tool_result(
            call.id(),
            execute(call).await?,
        );
    }
}
```

while the smallest useful experience remains:

```rust
let model = gen2::load("model.gguf")?;
let answer = model.generate("hello").text()?;
```

That is the API Gen2 should optimize around.