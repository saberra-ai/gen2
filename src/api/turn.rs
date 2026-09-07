//! [`Turn`] — one invocation of one [`Model`] against one [`Session`]
//! (api_spec.md §11), with the options (§12) and tool protocol (§10) it takes.
//!
//! A turn stages messages, sets per-turn controls, and runs — blocking to a
//! [`Response`], or streaming as [`EventStream`]. Whatever the model replies
//! is appended to the session: text with its reasoning kept apart, or the
//! tool calls it asked for, under ids a result can be pushed back against.
//! The inference core never runs a tool (invariant 4); the loop that does is
//! the caller's, and §28.5 is all it takes:
//!
//! ```no_run
//! use gen2::Session;
//! use gen2::tool_defs::{ToolDefinition, ToolSet};
//! # #[derive(gen2::schemars::JsonSchema)] struct ReadArgs { path: String }
//! # fn execute(_: &gen2::output::ToolCall) -> gen2::Result<String> { Ok("...".into()) }
//! # let model = gen2::load("m.gguf")?;
//!
//! let read = ToolDefinition::new("read")
//!     .description("Read a file")
//!     .input_schema::<ReadArgs>();
//! let mut session = Session::new()
//!     .with_system("Inspect before editing.")
//!     .with_tools(ToolSet::new().with(read));
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
//! let response = model.turn(&mut session).run()?;
//! # Ok::<(), gen2::Error>(())
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::common::grammar::GrammarSpec;
use crate::generation::{GenSpec, ThinkingMode};
use crate::types::message::Message;

use super::error::{Error, Result};
use super::event::{Canceller, EventStream, SessionSlot, Settled, StreamCore};
use super::generation::TOOL_PROMPT;
use super::input::Input;
use super::model::Model;
use super::response::Response;
use super::session::Session;
use super::tool_defs::ToolSet;

/// How the model may use the tools it is offered (api_spec.md §10.3).
///
/// Local backends render tools through the chat template and cannot force a
/// call at the decoder; `Required` and `Tool` narrow the set offered and
/// instruct the model, and the response's [`finish_reason`](Response::finish_reason)
/// says honestly whether a call came back.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides. The default.
    #[default]
    Auto,
    /// Offer no tools this turn, whatever the session holds.
    None,
    /// The model must call one of the tools offered.
    Required,
    /// The model must call this tool, and is offered only it.
    Tool(String),
}

impl ToolChoice {
    /// Force a call to a named tool.
    pub fn named(name: impl Into<String>) -> Self {
        Self::Tool(name.into())
    }
}

/// The controls that mean the same thing on every backend (api_spec.md §12).
///
/// Everything here maps onto the sampler every backend exposes. Expert
/// controls — repetition penalties, DRY, XTC, speculative decoding, grammars —
/// stay reachable through [`gen2::advanced`](crate::advanced) and the
/// [`GenSpec`] it exposes, so the ordinary surface stays small (§12.1).
///
/// Stop sequences are not a per-turn field: the engine applies them from
/// its [`Settings`](crate::advanced::generation::Settings), and a per-turn list has no sampler
/// slot to land in yet.
///
/// ```
/// use gen2::{GenerationOptions, model::ThinkingMode};
///
/// let options = GenerationOptions::default()
///     .temperature(0.7)
///     .max_tokens(512)
///     .reasoning(ThinkingMode::Off);
/// assert_eq!(options.max_tokens, Some(512));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct GenerationOptions {
    /// Cap on generated tokens. `None` is the engine's default.
    pub max_tokens: Option<usize>,
    /// Sampling temperature; `0.0` is greedy.
    pub temperature: Option<f32>,
    /// Sampler seed, for reproducible sampling.
    pub seed: Option<u64>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-k truncation.
    pub top_k: Option<i32>,
    /// Min-p threshold.
    pub min_p: Option<f32>,
    /// The reasoning channel, on models that expose one.
    pub reasoning: ThinkingMode,
}

impl GenerationOptions {
    /// Every field at its default: the engine decides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap how many tokens may be generated.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Sampling temperature. `0.0` is greedy — but prefer
    /// [`GenerationOptions::greedy`], which also pins the seed.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Seed the sampler, making a given temperature reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    pub fn greedy(mut self) -> Self {
        self.temperature = Some(0.0);
        self.seed = Some(self.seed.unwrap_or(0));
        self
    }

    /// Nucleus sampling threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Top-k truncation.
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Min-p threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.min_p = Some(p);
        self
    }

    /// The reasoning channel: on, off, or the model's default.
    pub fn reasoning(mut self, mode: ThinkingMode) -> Self {
        self.reasoning = mode;
        self
    }

    /// Lay these options over a backend spec. A field left `None` keeps
    /// whatever the spec had — the engine's configured default.
    pub(crate) fn apply_to(&self, spec: &mut GenSpec) {
        if let Some(n) = self.max_tokens {
            spec.max_tokens = Some(n);
        }
        if let Some(t) = self.temperature {
            spec.temperature = Some(t);
        }
        if let Some(s) = self.seed {
            spec.seed = Some(s);
        }
        if let Some(p) = self.top_p {
            spec.top_p = Some(p);
        }
        if let Some(k) = self.top_k {
            spec.top_k = Some(k);
        }
        if let Some(p) = self.min_p {
            spec.min_p = Some(p);
        }
    }
}

/// One invocation of a model against a session, being configured.
///
/// Built by [`Model::turn`]. Stage messages, set controls, then
/// [`run`](Turn::run) or [`stream`](Turn::stream). Nothing touches the
/// session until execution begins; a turn refused before that leaves it as
/// it was.
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// # let mut session = gen2::Session::new();
/// let response = model
///     .turn(&mut session)
///     .user("Fix the failing parser test")
///     .temperature(0.2)
///     .max_tokens(4096)
///     .run()?;
/// # Ok::<(), gen2::Error>(())
/// ```
#[must_use = "a Turn does nothing until .run(), .stream(), or .structured() is called"]
pub struct Turn<'a> {
    model: &'a Model,
    session: &'a mut Session,
    staged: Vec<Message>,
    options: GenerationOptions,
    tool_choice: ToolChoice,
    system_override: Option<String>,
    tools_override: Option<ToolSet>,
    grammar: Option<GrammarSpec>,
    cancel: Arc<AtomicBool>,
}

impl<'a> Turn<'a> {
    pub(crate) fn new(model: &'a Model, session: &'a mut Session) -> Self {
        Self {
            model,
            session,
            staged: Vec::new(),
            options: GenerationOptions::default(),
            tool_choice: ToolChoice::Auto,
            system_override: None,
            tools_override: None,
            grammar: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    // ── Staging (§11.1) ─────────────────────────────────────────────────────

    /// Stage a user message.
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::user(text))
    }

    /// Stage a user message built from [`Input`] parts — text and images.
    pub fn input(self, input: impl Into<Input>) -> Self {
        self.message(input.into().into_message())
    }

    /// Stage a user message with an image, by path or URL.
    ///
    /// The model must accept images, or the turn is refused before anything
    /// is staged or generated.
    pub fn image(self, source: impl AsRef<std::path::Path>) -> Self {
        self.input(Input::new().image(source))
    }

    /// Stage an already-built message.
    pub fn message(mut self, message: Message) -> Self {
        self.staged.push(message);
        self
    }

    /// Stage several messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.staged.extend(messages);
        self
    }

    // ── Per-turn controls (§11.3) ───────────────────────────────────────────

    /// Cap how many tokens this turn may generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.options.max_tokens = Some(n);
        self
    }

    /// Sampling temperature. `0.0` is greedy — but prefer [`Turn::greedy`],
    /// which also pins the seed.
    pub fn temperature(mut self, t: f32) -> Self {
        self.options.temperature = Some(t);
        self
    }

    /// Seed the sampler, making a given temperature reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.options.seed = Some(seed);
        self
    }

    /// Nucleus sampling threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.options.top_p = Some(p);
        self
    }

    /// Top-k truncation.
    pub fn top_k(mut self, k: i32) -> Self {
        self.options.top_k = Some(k);
        self
    }

    /// Min-p threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.options.min_p = Some(p);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    pub fn greedy(mut self) -> Self {
        self.options = std::mem::take(&mut self.options).greedy();
        self
    }

    /// The reasoning channel, on models that expose one.
    pub fn reasoning(mut self, mode: ThinkingMode) -> Self {
        self.options.reasoning = mode;
        self
    }

    /// How the model may use the tools it is offered.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Every common control at once, replacing what was set so far.
    pub fn options(mut self, options: GenerationOptions) -> Self {
        self.options = options;
        self
    }

    // ── Per-turn overrides (§11.4) ──────────────────────────────────────────

    /// A system prompt for this turn only.
    ///
    /// The session's own prompt is neither read nor changed: its
    /// [`revision`](Session::revision) is the same after the turn as before
    /// it, apart from the messages the turn appended.
    pub fn system_override(mut self, text: impl Into<String>) -> Self {
        self.system_override = Some(text.into());
        self
    }

    /// A tool set for this turn only. The session's own set is untouched.
    pub fn tools_override(mut self, tools: impl Into<ToolSet>) -> Self {
        self.tools_override = Some(tools.into());
        self
    }

    // ── Running ─────────────────────────────────────────────────────────────

    /// A handle that can stop this turn once it is running — or, used
    /// before [`run`](Turn::run), make it not run at all.
    ///
    /// The one from [`EventStream::canceller`] is the same handle; this exists
    /// for a caller that must hand the stop out before it can block on
    /// `run()`.
    pub fn canceller(&self) -> Canceller {
        Canceller::new(
            self.model.clone(),
            self.session.id().to_string(),
            Arc::clone(&self.cancel),
        )
    }

    /// Run the turn and return the [`Response`]. The assistant message is
    /// appended to the session.
    pub fn run(self) -> Result<Response> {
        self.stream()?.finish()
    }

    /// Run the turn as a stream of semantic [`Event`](super::event::Event)s.
    ///
    /// Iterate to the end, then [`EventStream::finish`] for the same
    /// [`Response`] [`run`](Turn::run) would have returned.
    pub fn stream(self) -> Result<EventStream<'a>> {
        let (core, session) = self.begin()?;
        Ok(EventStream::new(core, SessionSlot::Borrowed(session)))
    }

    /// Validate, commit, and dispatch: the stream's engine half, and the
    /// session it will settle into. What [`stream`](Turn::stream) is made of;
    /// the async surface bridges the two halves across a worker.
    pub(crate) fn begin(self) -> Result<(StreamCore, &'a mut Session)> {
        let model = self.model.clone();
        // Weights first: a handle whose model was evicted restores it here,
        // transparently (api_spec.md §4.2). Before validation, so an
        // evicted model's capabilities are read from a loaded one.
        self.model.ensure_resident()?;
        let engine = self.model.engine();
        let session = self.session;

        // ── Validate before anything is committed (§11.1) ──
        let staged_images = self
            .staged
            .iter()
            .any(|m| super::chat::has_images(std::slice::from_ref(m)));
        if (staged_images || super::chat::has_images(session.messages()))
            && !engine.supports_images()
        {
            return Err(Error::Unsupported("images".into()));
        }
        let effective: &ToolSet = self.tools_override.as_ref().unwrap_or(session.tools());
        // Offer nothing: asked for, or an override that is the empty set.
        let suppress = matches!(self.tool_choice, ToolChoice::None)
            || (matches!(self.tool_choice, ToolChoice::Auto)
                && self.tools_override.as_ref().is_some_and(ToolSet::is_empty));
        let tools: Option<(Vec<_>, String)> = match &self.tool_choice {
            ToolChoice::None => None,
            ToolChoice::Auto => self
                .tools_override
                .as_ref()
                .filter(|set| !set.is_empty())
                .map(|set| (set.to_wire(), TOOL_PROMPT.to_string())),
            ToolChoice::Required => {
                if effective.is_empty() {
                    return Err(Error::InvalidRequest(
                        "tool_choice is Required but no tools are offered".into(),
                    ));
                }
                Some((
                    effective.to_wire(),
                    format!("{TOOL_PROMPT} You must call one of the tools before answering."),
                ))
            }
            ToolChoice::Tool(name) => {
                let Some(only) = effective.get(name) else {
                    return Err(Error::InvalidRequest(format!(
                        "tool_choice names `{name}`, which is not among the tools offered"
                    )));
                };
                Some((
                    ToolSet::new().with(only.clone()).to_wire(),
                    format!("{TOOL_PROMPT} You must call the tool `{name}`."),
                ))
            }
        };
        // A per-turn prefix — prompt, or a tool set other than the session's
        // — is not the one the engine holds: rebuild for this turn, and
        // again after it.
        let per_turn_prefix = self.system_override.is_some()
            || self.tools_override.is_some()
            || !matches!(self.tool_choice, ToolChoice::Auto);

        if self.cancel.load(Ordering::SeqCst) {
            // Cancelled before it began: nothing runs, nothing is staged.
            let core =
                StreamCore::cancelled_before_start(session.next_message_id(), model, self.cancel);
            return Ok((core, session));
        }

        // ── Commit (§11.1) ──
        for message in self.staged {
            session.push(message);
        }
        if per_turn_prefix {
            session.opened = false;
        }

        let mut spec = engine.default_gen_spec();
        self.options.apply_to(&mut spec);
        if let Some(grammar) = self.grammar {
            spec.grammar = Some(grammar);
        }
        let max_tokens = spec.max_tokens;

        let mut chat = engine
            .chat(session)
            .gen_spec(spec)
            .thinking(self.options.reasoning);
        match tools {
            Some((specs, prompt)) => chat = chat.tools(specs, prompt),
            None if suppress => chat = chat.without_tools(),
            None => {}
        }
        if let Some(system) = self.system_override {
            chat = chat.system_override(Some(system));
        }
        let (stream, session) = chat.begin()?;

        // A cancel that raced the dispatch: the stop it sent found no chat
        // yet. Sent again now, it lands behind the start.
        if self.cancel.load(Ordering::SeqCst) {
            let _ = engine.stop(session.id().to_string());
        }
        let core = StreamCore::new(
            stream,
            session.next_message_id(),
            model,
            self.cancel,
            Settled {
                max_tokens,
                close_after: per_turn_prefix,
            },
        );
        Ok((core, session))
    }

    /// Run the turn and decode the reply as `T` (api_spec.md §13).
    ///
    /// The schema is derived from `T` and, where the backend constrains
    /// decoding by grammar, enforced token by token — see
    /// [`ModelCapabilities::structured_output`](super::model::ModelCapabilities::structured_output).
    /// A backend that cannot is asked for JSON and the reply is parsed; a
    /// reply that does not decode is [`Error::Extraction`], carrying what the
    /// model wrote. The reasoning channel is off unless set explicitly, since
    /// the schema has no place for it.
    ///
    /// ```no_run
    /// # let model = gen2::load("m.gguf")?;
    /// # let mut session = gen2::Session::new();
    /// #[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
    /// struct Decision { option: String, confidence: f32 }
    ///
    /// let decision: Decision = model
    ///     .turn(&mut session)
    ///     .user("Choose the best option")
    ///     .structured()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn structured<T>(mut self) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let schema = schema_of::<T>()?;
        if self.model.capabilities().structured_output {
            self.grammar = Some(GrammarSpec::JsonSchema(schema));
        } else if self.system_override.is_none() {
            // No grammar to lean on: say what shape is wanted.
            let mut prompt = self.session.system().unwrap_or_default().to_string();
            if !prompt.is_empty() {
                prompt.push('\n');
            }
            prompt.push_str(&format!(
                "Respond with a single JSON object matching this JSON Schema, and nothing else:\n{schema}"
            ));
            self.system_override = Some(prompt);
        }
        if self.options.reasoning == ThinkingMode::Auto {
            self.options.reasoning = ThinkingMode::Off;
        }
        if self.options.max_tokens.is_none() {
            self.options.max_tokens = Some(STRUCTURED_MAX_TOKENS);
        }
        let response = self.run()?;
        decode::<T>(&response.text())
    }
}

/// Enough for a substantial object without letting a model that has started
/// repeating itself run indefinitely. Overridable, because a document-sized
/// struct is a legitimate thing to ask for.
pub(crate) const STRUCTURED_MAX_TOKENS: usize = 1024;

/// The JSON schema for `T`, generated from the type itself: the constraint
/// the model decodes under and the type the reply is parsed into are the
/// same declaration, so they cannot drift.
pub(crate) fn schema_of<T: schemars::JsonSchema>() -> Result<serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T)).map_err(|e| {
        Error::InvalidRequest(format!(
            "could not build a JSON schema for {}: {e}",
            std::any::type_name::<T>()
        ))
    })
}

/// Read a reply as `T`. Generation succeeded; anything wrong from here is a
/// decode problem, and the error says so with what the model actually wrote.
pub(crate) fn decode<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T> {
    let text = strip_fence(raw.trim());
    serde_json::from_str::<T>(text).map_err(|e| Error::Extraction {
        type_name: std::any::type_name::<T>(),
        message: e.to_string(),
        raw: raw.to_string(),
    })
}

/// A model without a grammar often wraps JSON in a Markdown fence; the
/// fence is not the answer.
fn strip_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.trim().strip_suffix("```").unwrap_or(rest).trim()
}

impl std::fmt::Debug for Turn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Turn")
            .field("model", &self.model.id())
            .field("session", &self.session.id())
            .field("staged", &self.staged.len())
            .field("options", &self.options)
            .field("tool_choice", &self.tool_choice)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::event::Event;
    use crate::api::output::FinishReason;
    use crate::api::runtime::Runtime;
    use crate::api::tool_defs::ToolDefinition;
    use crate::test_support::{Gate, Script, Step};

    fn tools() -> ToolSet {
        ToolSet::new()
            .with(ToolDefinition::new("read").description("Read a file"))
            .with(ToolDefinition::new("write").description("Write a file"))
    }

    #[test]
    fn a_turn_appends_the_user_message_and_the_reply() {
        let model = Runtime::scripted(Script::new().say(["Hel", "lo"]));
        let mut session = Session::new().with_system("Be brief.");
        let before = session.revision();

        let response = model.turn(&mut session).user("Hi").run().unwrap();

        assert_eq!(response.text(), "Hello");
        assert_eq!(*response.finish_reason(), FinishReason::Stop);
        let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant"]);
        assert_eq!(session.latest_text().as_deref(), Some("Hello"));
        assert_eq!(
            response.message_id(),
            Some(*session.active_ids().last().unwrap()),
            "the response names the message it appended"
        );
        assert_eq!(
            session.revision().as_u64(),
            before.as_u64() + 2,
            "two messages, nothing else"
        );
    }

    #[test]
    fn staged_messages_commit_only_when_execution_begins() {
        // Images on a text-only model are refused before dispatch (§11.1):
        // the staged message must not be left behind.
        let model = Runtime::scripted(Script::new().say(["never"]));
        let mut session = Session::new();
        session.push_user("earlier");
        let revision = session.revision();

        let err = model
            .turn(&mut session)
            .user("look")
            .image("/tmp/photo.png")
            .run()
            .expect_err("a text-only model refuses images");
        assert!(matches!(err, Error::Unsupported(ref what) if what == "images"));
        assert_eq!(session.len(), 1, "nothing staged was committed");
        assert_eq!(session.revision(), revision);

        // A tool choice naming a tool nobody offered is refused the same way.
        let err = model
            .turn(&mut session)
            .user("go")
            .tool_choice(ToolChoice::named("nope"))
            .run()
            .expect_err("an unknown tool cannot be required");
        assert_eq!(err.code(), Some("invalid_request"));
        assert_eq!(session.len(), 1);
        assert_eq!(session.revision(), revision);
    }

    #[test]
    fn a_turn_with_no_new_message_runs_on_the_session_as_it_is() {
        let model = Runtime::scripted(Script::new().say(["continued"]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        session.push_user("first");
        let response = model.turn(&mut session).run().unwrap();
        assert_eq!(response.text(), "continued");
        assert_eq!(
            script.seen(),
            ["first"],
            "the existing message reached the model"
        );
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn the_tool_loop_continues_after_results_with_no_user_message() {
        // §28.5 verbatim, over a scripted model.
        let model = Runtime::scripted(Script::new().turns([
            vec![
                Step::tool_call("read", r#"{"path":"Cargo.toml"}"#),
                Step::eos(),
            ],
            vec![Step::token("It is a Rust crate."), Step::eos()],
        ]));
        let script = model.engine().script().clone();
        let mut session = Session::new()
            .with_system("Inspect before editing.")
            .with_tools(tools());

        let response = model
            .turn(&mut session)
            .user("What files are in this repository?")
            .run()
            .unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::ToolCall);
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "read");
        assert_eq!(calls[0].arguments()["path"], "Cargo.toml");

        for call in response.tool_calls() {
            session.push_tool_result(call.id(), "[package]\nname = \"gen2\"");
        }
        let response = model.turn(&mut session).run().unwrap();
        assert_eq!(response.text(), "It is a Rust crate.");
        assert_eq!(*response.finish_reason(), FinishReason::Stop);

        let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
        assert_eq!(
            session.messages()[2].tool_call_id.as_deref(),
            Some(calls[0].id().as_str()),
            "the result answers the call by id"
        );
        assert_eq!(
            script.tools_seen(),
            vec![vec!["read".to_string(), "write".to_string()]],
            "the session's tools were offered when the conversation opened"
        );
    }

    #[test]
    fn tool_call_ids_are_unique_across_a_session() {
        let model = Runtime::scripted(Script::new().turns([
            vec![Step::tool_call("read", "{}"), Step::eos()],
            vec![Step::tool_call("read", "{}"), Step::eos()],
        ]));
        let mut session = Session::new().with_tools(tools());
        let first = model.turn(&mut session).user("a").run().unwrap();
        let first_id = first.tool_calls()[0].id().clone();
        session.push_tool_result(&first_id, "x");
        let second = model.turn(&mut session).run().unwrap();
        let second_id = second.tool_calls()[0].id().clone();
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn overrides_are_for_the_turn_only_and_never_persist() {
        let model = Runtime::scripted(Script::new().say(["ok"]));
        let script = model.engine().script().clone();
        let mut session = Session::new()
            .with_system("Persistent prompt.")
            .with_tools(tools());
        let revision = session.revision();
        let context_events = |s: &Session| {
            s.events()
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        crate::api::session::SessionEvent::SystemSet { .. }
                            | crate::api::session::SessionEvent::ToolsSet { .. }
                    )
                })
                .count()
        };
        let logged = context_events(&session);
        let temp = ToolSet::new().with(ToolDefinition::new("bash").description("Run a command"));

        model
            .turn(&mut session)
            .user("One")
            .system_override("For this turn only, answer as JSON.")
            .tools_override(temp)
            .run()
            .unwrap();

        assert_eq!(session.system(), Some("Persistent prompt."));
        assert_eq!(session.tools().names(), ["read", "write"]);
        assert_eq!(
            session.revision().as_u64(),
            revision.as_u64() + 2,
            "only the two messages moved the revision"
        );
        assert_eq!(
            context_events(&session),
            logged,
            "no system or tools event was logged"
        );
        assert_eq!(
            script.seen()[0],
            "For this turn only, answer as JSON.",
            "the override reached the model as the system prompt"
        );
        assert_eq!(script.tools_seen(), vec![vec!["bash".to_string()]]);

        // The next turn rebuilds under the session's own prefix.
        model.turn(&mut session).user("Two").run().unwrap();
        let seen = script.seen();
        assert!(
            seen.contains(&"Persistent prompt.".to_string()),
            "the session's prompt is back: {seen:?}"
        );
        assert_eq!(
            script.tools_seen().last(),
            Some(&vec!["read".to_string(), "write".to_string()])
        );
    }

    #[test]
    fn tool_choice_none_offers_nothing_and_named_offers_only_that() {
        let model = Runtime::scripted(Script::new().say(["ok"]));
        let script = model.engine().script().clone();
        let mut session = Session::new().with_tools(tools());
        model
            .turn(&mut session)
            .user("a")
            .tool_choice(ToolChoice::None)
            .run()
            .unwrap();
        model
            .turn(&mut session)
            .user("b")
            .tool_choice(ToolChoice::named("write"))
            .run()
            .unwrap();
        model.turn(&mut session).user("c").run().unwrap();
        assert_eq!(
            script.tools_seen(),
            vec![
                vec![],
                vec!["write".to_string()],
                vec!["read".to_string(), "write".to_string()]
            ]
        );
        let err = model
            .turn(&mut Session::new())
            .user("d")
            .tool_choice(ToolChoice::Required)
            .run()
            .expect_err("nothing to require");
        assert_eq!(err.code(), Some("invalid_request"));
    }

    #[test]
    fn options_land_on_the_backend_spec() {
        let model = Runtime::scripted(Script::new().say(["ok"]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        model
            .turn(&mut session)
            .user("a")
            .options(
                GenerationOptions::new()
                    .top_p(0.5)
                    .top_k(7)
                    .min_p(0.1)
                    .seed(9),
            )
            .max_tokens(33)
            .temperature(0.25)
            .run()
            .unwrap();
        let spec = script.specs_seen().pop().unwrap();
        assert_eq!(spec.max_tokens, Some(33));
        assert_eq!(spec.temperature, Some(0.25));
        assert_eq!(spec.top_p, Some(0.5));
        assert_eq!(spec.top_k, Some(7));
        assert_eq!(spec.min_p, Some(0.1));
        assert_eq!(spec.seed, Some(9));
        model.turn(&mut session).user("b").greedy().run().unwrap();
        let spec = script.specs_seen().pop().unwrap();
        assert_eq!((spec.temperature, spec.seed), (Some(0.0), Some(0)));
    }

    #[test]
    fn a_stream_yields_semantic_events_and_finishes_with_the_same_response() {
        let model = Runtime::scripted(Script::new().program([
            Step::token("<think>\n"),
            Step::token("plan"),
            Step::token("\n</think>\n\n"),
            Step::token("Done"),
            Step::tool_call("write", r#"{"path":"a"}"#),
            Step::eos(),
        ]));
        let mut session = Session::new().with_tools(tools());
        // The user message takes the next id; the assistant's is the one after.
        let assistant_id = session.next_message_id().as_u64() + 1;
        let mut stream = model.turn(&mut session).user("go").stream().unwrap();
        let mut events = Vec::new();
        for event in stream.by_ref() {
            events.push(event.unwrap());
        }
        let id = crate::api::output::ToolCallId(format!("call_{assistant_id}_0"));
        assert_eq!(
            events,
            vec![
                Event::ReasoningDelta("plan".into()),
                Event::TextDelta("Done".into()),
                Event::ToolCallStart {
                    id: id.clone(),
                    name: "write".into()
                },
                Event::ToolCallArgumentsDelta {
                    id: id.clone(),
                    delta: r#"{"path":"a"}"#.into()
                },
                Event::ToolCallEnd {
                    call: crate::api::output::ToolCall {
                        id,
                        name: "write".into(),
                        arguments: serde_json::json!({"path": "a"}),
                    }
                },
                Event::Finished(FinishReason::ToolCall),
            ]
        );
        let response = stream.finish().unwrap();
        assert_eq!(response.text(), "Done");
        assert_eq!(response.reasoning().as_deref(), Some("plan"));
        assert_eq!(response.tool_calls().len(), 1);
        assert_eq!(
            session.len(),
            2,
            "the reply was appended when the stream ended"
        );
        assert_eq!(response.message_id(), session.active_ids().last().copied());
    }

    #[test]
    fn cancelling_mid_stream_ends_with_cancelled_and_records_the_partial_reply() {
        // §16.1 walkthrough.
        let gate = Gate::new();
        let model = Runtime::scripted(Script::new().program([
            Step::token("kept"),
            Step::Hold(Arc::clone(&gate)),
            Step::token(" never seen"),
            Step::eos(),
        ]));
        let mut session = Session::new();
        let mut stream = model
            .turn(&mut session)
            .user("Write a novel")
            .stream()
            .unwrap();
        let cancel = stream.canceller();
        assert!(!cancel.is_cancelled());

        let first = stream.next().unwrap().unwrap();
        assert_eq!(first, Event::TextDelta("kept".into()));
        assert!(gate.wait_until_reached(), "mid-flight");
        let stopper = std::thread::spawn({
            let gate = Arc::clone(&gate);
            move || {
                cancel.cancel();
                gate.open();
            }
        });
        let rest: Vec<Event> = stream.by_ref().map(|e| e.unwrap()).collect();
        stopper.join().unwrap();
        assert_eq!(
            rest.last(),
            Some(&Event::Finished(FinishReason::Cancelled)),
            "{rest:?}"
        );

        let response = stream.finish().unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
        // The scripted backend cannot interrupt a pull, so the stop lands
        // one token late; a real backend's stop flag ends decoding at once.
        assert!(response.text().starts_with("kept"), "{:?}", response.text());
        let id = response
            .message_id()
            .expect("the partial reply was recorded");
        assert_eq!(session.latest_text(), Some(response.text()));
        session.remove_message(id).unwrap();
        assert_eq!(
            session.len(),
            1,
            "and can be removed from the active context"
        );
        assert_eq!(session.all_messages().len(), 2, "without losing the record");
    }

    #[test]
    fn a_canceller_taken_before_run_stops_the_turn_from_running() {
        let model = Runtime::scripted(Script::new().say(["never"]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        let turn = model.turn(&mut session).user("go");
        let cancel = turn.canceller();
        cancel.cancel();
        let response = turn.run().unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::Cancelled);
        assert_eq!(response.message_id(), None);
        assert!(session.is_empty(), "nothing was staged or generated");
        assert_eq!(script.count("pull"), 0);
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    #[serde(rename_all = "lowercase")]
    enum Sentiment {
        Positive,
        Negative,
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Verdict {
        sentiment: Sentiment,
        score: f32,
    }

    #[test]
    fn structured_output_decodes_the_reply_into_the_type() {
        let model =
            Runtime::scripted(Script::new().say([r#"{"sentiment":"positive","score":0.9}"#]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        let verdict: Verdict = model
            .turn(&mut session)
            .user("I loved it")
            .structured()
            .unwrap();
        assert_eq!(
            verdict,
            Verdict {
                sentiment: Sentiment::Positive,
                score: 0.9
            }
        );
        assert_eq!(
            session.len(),
            2,
            "the JSON reply is part of the conversation"
        );
        if model.capabilities().structured_output {
            let grammar = script.grammars_seen().into_iter().next();
            assert!(
                matches!(grammar, Some(GrammarSpec::JsonSchema(ref s)) if s["properties"]["sentiment"].is_object()),
                "the grammar is the schema of the type: {grammar:?}"
            );
        } else {
            assert!(
                script.seen()[0].contains("JSON Schema"),
                "without a grammar the schema is asked for in the prompt"
            );
        }
    }

    #[test]
    fn a_reply_that_does_not_fit_the_type_is_an_extraction_error() {
        let model = Runtime::scripted(Script::new().say([r#"```json{"sentiment":"meh"}```"#]));
        let mut session = Session::new();
        let err = model
            .turn(&mut session)
            .user("x")
            .structured::<Verdict>()
            .expect_err("`meh` is not a sentiment");
        match err {
            Error::Extraction { raw, .. } => assert!(raw.contains("meh")),
            other => panic!("expected an extraction failure, got {other:?}"),
        }
        assert_eq!(
            decode::<Verdict>("```json\n{\"sentiment\":\"negative\",\"score\":0}\n```")
                .unwrap()
                .sentiment,
            Sentiment::Negative,
            "a fence is stripped"
        );
    }

    #[test]
    fn a_reasoning_change_between_turns_reopens_the_conversation() {
        let model = Runtime::scripted(Script::new().say(["ok"]));
        let script = model.engine().script().clone();
        let mut session = Session::new();
        model.turn(&mut session).user("a").run().unwrap();
        model
            .turn(&mut session)
            .user("b")
            .reasoning(ThinkingMode::Off)
            .run()
            .unwrap();
        assert_eq!(
            script.count("start_session"),
            2,
            "a different thinking policy is a different session"
        );
    }
}
