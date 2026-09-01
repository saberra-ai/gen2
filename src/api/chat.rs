//! [`Chat`] — one turn against a [`Session`].

use crate::backend::common::grammar::GrammarSpec;
use crate::controller::ControllerCmd;
use crate::generation::ToolCall;
use crate::generation::{GenSpec, ThinkingMode};
use crate::types::message::{FunctionDefinition, Message, ToolCall as MessageToolCall, ToolSpec};

use super::engine::{Engine, event_channel};
use super::error::{Error, Result};
use super::session::Session;
use super::stream::{Completion, Finish, TokenStream, Tokens};

/// A turn being built against a conversation.
///
/// Add messages, set sampling, then run it. Whatever the model replies is
/// appended to the session, so the conversation stays whole without you
/// managing it.
///
/// ```no_run
/// # use gen2::{Engine, Session};
/// # let engine = Engine::load("m.gguf")?;
/// let mut session = Session::new();
/// engine.chat(&mut session)
///     .user("Explain entropy in one sentence.")
///     .max_tokens(256)
///     .send()?;
/// println!("{}", session.latest_text().unwrap_or_default());
/// # Ok::<(), gen2::Error>(())
/// ```
#[must_use = "a Chat does nothing until .send(), .text(), .stream(), or .tokens() is called"]
pub struct Chat<'a> {
    engine: &'a Engine,
    session: &'a mut Session,
    /// How long the transcript was before this turn added anything, so a turn
    /// refused before it runs can leave the conversation as it found it.
    mark: usize,
    spec: GenSpec,
    thinking: ThinkingMode,
    tools: Option<(Vec<ToolSpec>, String)>,
    handler: Option<ToolHandler<'a>>,
    tool_depth: usize,
}

/// Dispatches one tool call and returns its output as text.
type ToolHandler<'a> = Box<dyn FnMut(&ToolCall) -> String + 'a>;

/// Rounds of tool calls a turn may run before giving up.
///
/// Deep enough for a genuine multi-step task, shallow enough that a model stuck
/// in a call/re-call loop stops costing tokens.
pub const DEFAULT_TOOL_DEPTH: usize = 7;

impl<'a> Chat<'a> {
    pub(crate) fn new(engine: &'a Engine, session: &'a mut Session) -> Self {
        Self {
            // Starts from the engine's configured defaults, so anything set at
            // build time applies unless this turn overrides it.
            spec: engine.default_gen_spec(),
            mark: session.len(),
            engine,
            session,
            thinking: ThinkingMode::default(),
            tools: None,
            handler: None,
            tool_depth: DEFAULT_TOOL_DEPTH,
        }
    }

    // ── Messages ────────────────────────────────────────────────────────────

    /// Append a user message to the conversation.
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::user(text))
    }

    /// Append a user message carrying images.
    ///
    /// Paths become `file://` URLs; existing URLs pass through. Needs a
    /// multimodal model loaded with a projector — see
    /// [`EngineBuilder::mmproj`](super::EngineBuilder::mmproj).
    ///
    /// ```no_run
    /// # use gen2::{Engine, Session};
    /// # let engine = Engine::load("m.gguf")?;
    /// # let mut session = Session::new();
    /// engine.chat(&mut session)
    ///     .user_with_images("What is in this picture?", ["/tmp/photo.png"])
    ///     .send()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn user_with_images<I, P>(self, text: impl Into<String>, images: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        self.session.push_user_with_images(text, images);
        self
    }

    /// Append a system message to the conversation.
    pub fn system(self, text: impl Into<String>) -> Self {
        self.message(Message::system(text))
    }

    /// Append an already-built message.
    pub fn message(self, message: Message) -> Self {
        self.session.push(message);
        self
    }

    /// Append several messages.
    pub fn messages(self, messages: impl IntoIterator<Item = Message>) -> Self {
        for m in messages {
            self.session.push(m);
        }
        self
    }

    // ── Sampling ────────────────────────────────────────────────────────────

    /// Cap how many tokens this turn may generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.spec.max_tokens = Some(n);
        self
    }

    /// Sampling temperature. `0.0` is greedy — but prefer [`Chat::greedy`],
    /// which also pins the seed.
    pub fn temperature(mut self, t: f32) -> Self {
        self.spec.temperature = Some(t);
        self
    }

    /// Seed the sampler, making a given temperature reproducible.
    ///
    /// Pairs with [`Chat::greedy`] in either order: `greedy()` only supplies a
    /// seed when you haven't set one.
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    ///
    /// Worth naming, because it is *not* the default — an unconfigured turn
    /// samples with a random seed. Set it once on
    /// [`EngineBuilder::greedy`](super::EngineBuilder::greedy) if every turn
    /// should be reproducible.
    pub fn greedy(mut self) -> Self {
        self.spec.temperature = Some(0.0);
        self.spec.seed = Some(self.spec.seed.unwrap_or(0));
        self
    }

    /// Nucleus sampling threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.spec.top_p = Some(p);
        self
    }

    /// Top-k truncation.
    pub fn top_k(mut self, k: i32) -> Self {
        self.spec.top_k = Some(k);
        self
    }

    /// Min-p threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.spec.min_p = Some(p);
        self
    }

    /// Constrain this turn's output to a grammar — JSON schema, regex, Lark,
    /// or GBNF.
    ///
    /// Enforced during decoding, so the model cannot produce output that
    /// violates it. Overrides
    /// [`EngineBuilder::grammar`](super::EngineBuilder::grammar) for this turn;
    /// [`Chat::unconstrained`] drops an engine-level default.
    pub fn grammar(mut self, grammar: GrammarSpec) -> Self {
        self.spec.grammar = Some(grammar);
        self
    }

    /// Drop any engine-level grammar for this turn.
    pub fn unconstrained(mut self) -> Self {
        self.spec.grammar = None;
        self
    }

    /// Use a fully-built [`GenSpec`], overriding everything above.
    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.spec = spec;
        self
    }

    // ── Tools and reasoning ─────────────────────────────────────────────────

    /// Offer tools to the model. `prompt` introduces them in the template.
    pub fn tools(mut self, tools: Vec<ToolSpec>, prompt: impl Into<String>) -> Self {
        self.tools = Some((tools, prompt.into()));
        self
    }

    /// Force the reasoning channel on or off for models that expose one.
    pub fn thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
        self
    }

    /// Run tools automatically, feeding results back until the model answers.
    ///
    /// Without this a tool call is just an [`Event`](super::Event) for you to
    /// handle; with it the turn becomes a loop — generate, dispatch each call
    /// through `f`, append the results, generate again — ending when the model
    /// stops asking or [`Chat::tool_depth`] is reached.
    ///
    /// Both halves land in the session: the assistant turn that asked, and the
    /// results that came back. `f` returns the tool's output as text.
    ///
    /// ```no_run
    /// # use gen2::{Engine, Session};
    /// # let engine = Engine::load("m.gguf")?;
    /// # let mut session = Session::new();
    /// # let (tools, prompt) = (vec![], String::new());
    /// let done = engine.chat(&mut session)
    ///     .user("What is the weather in Paris?")
    ///     .tools(tools, prompt)
    ///     .on_tool(|call| match call.name.as_str() {
    ///         "get_weather" => "18C, clear".to_string(),
    ///         other => format!("no such tool: {other}"),
    ///     })
    ///     .send()?;
    /// println!("answered after {} tool rounds", done.tool_rounds);
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn on_tool(mut self, f: impl FnMut(&ToolCall) -> String + 'a) -> Self {
        self.handler = Some(Box::new(f));
        self
    }

    /// Cap how many rounds of tool calls a turn may run.
    ///
    /// Defaults to [`DEFAULT_TOOL_DEPTH`]. Reaching it ends the turn with
    /// [`Finish::ToolDepthReached`], which is how a model looping on the same
    /// call stops rather than running forever.
    ///
    /// Only meaningful alongside [`Chat::on_tool`].
    pub fn tool_depth(mut self, depth: usize) -> Self {
        self.tool_depth = depth;
        self
    }

    // ── Running ─────────────────────────────────────────────────────────────

    /// Run the turn, appending the reply to the session.
    ///
    /// Read it with [`Session::latest`], or use the returned [`Completion`] for
    /// stats and the finish reason.
    pub fn send(self) -> Result<Completion> {
        self.send_streaming(|_| {})
    }

    /// [`Chat::send`], with `on_token` called per fragment as it arrives.
    ///
    /// This is the one a UI wants on a blocking thread: tokens land as they
    /// decode, and the session ends up holding the finished reply.
    pub fn send_streaming(mut self, mut on_token: impl FnMut(&str)) -> Result<Completion> {
        let Some(mut handler) = self.handler.take() else {
            // No handler: one turn, and tool calls are the caller's to act on.
            let engine = self.engine;
            let (mut stream, session) = self.begin()?;
            let done = stream.drain_streaming(on_token)?;
            settle(engine, session, &stream);
            session.note_shed(done.dropped + done.compacted);
            session.push(Message::assistant_structured(done.text.clone(), None));
            return Ok(done);
        };

        let engine = self.engine;
        let depth_limit = self.tool_depth;
        let spec = self.spec.clone();
        let thinking = self.thinking;
        let tools = self.tools.take();
        let session: &mut Session = self.session;
        let mut rounds = 0_usize;

        loop {
            let turn = Chat {
                mark: session.len(),
                engine,
                session,
                spec: spec.clone(),
                thinking,
                // The tool list is re-offered every round: the model needs to
                // see what it may call on the follow-up too, not just the first.
                tools: tools.clone(),
                handler: None,
                tool_depth: depth_limit,
            };

            let (mut stream, session_back) = turn.begin()?;
            let mut done = stream.drain_streaming(&mut on_token)?;
            settle(engine, session_back, &stream);
            done.tool_rounds = rounds;
            session_back.note_shed(done.dropped + done.compacted);

            if done.tool_calls.is_empty() {
                session_back.push(Message::assistant_structured(done.text.clone(), None));
                return Ok(done);
            }

            // Record what was asked before what came back, so the transcript
            // reads in the order it happened.
            session_back.push(Message::assistant_tool_calls(
                done.tool_calls.iter().map(as_message_call).collect(),
            ));

            if rounds >= depth_limit {
                // Still asking, but out of budget. Report it rather than
                // looping or pretending the reply is final.
                done.finish = Finish::ToolDepthReached;
                return Ok(done);
            }

            for call in &done.tool_calls {
                let result = handler(call);
                session_back.push(Message::tool_result(result));
            }
            rounds += 1;
        }
    }

    /// Run the turn and return just the reply text. Also appended to the
    /// session.
    pub fn text(self) -> Result<String> {
        Ok(self.send()?.text)
    }

    /// Run the turn and return the raw event stream.
    ///
    /// The reply is **not** appended to the session — you are draining the
    /// events, so only you know what the final text was. Push it yourself with
    /// [`Session::push`], or use [`Chat::send_streaming`], which does.
    pub fn stream(self) -> Result<TokenStream> {
        // The caller takes the stream, so nothing here can observe the
        // engine's acknowledgement. The conversation therefore stays closed
        // and the next turn rebuilds from the transcript: one extra prefill,
        // and no chance of sending a suffix of a conversation the backend does
        // not have.
        Ok(self.begin()?.0)
    }

    /// Run the turn and iterate the text fragments.
    ///
    /// As with [`Chat::stream`], the reply is not appended to the session.
    pub fn tokens(self) -> Result<Tokens> {
        Ok(self.stream()?.tokens())
    }

    /// Dispatch the turn, handing back the stream and the session to append to.
    fn begin(self) -> Result<(TokenStream, &'a mut Session)> {
        let engine = self.engine;
        let session = self.session;
        // A cached prefill belongs to the model that produced it. If the model
        // has been swapped since this conversation was opened, the engine's
        // cache is for weights that are gone, so the conversation reopens.
        session.note_model(engine.model_generation());

        // Checked before anything is sent, so a text-only model given an image
        // fails with the conversation untouched — the caller can drop the
        // attachment or swap the model and try again.
        if has_images(session.messages()) && !engine.supports_images() {
            // Leave the conversation as this turn found it. The builder appends
            // as it is configured, so without this the rejected message would
            // stay and every later turn would hit the same wall.
            session.rollback_to(self.mark);
            return Err(Error::Unsupported("images".into()));
        }

        let (tx, rx) = event_channel(engine.event_channel_capacity());

        // A conversation the engine already holds gets only what's new; one it
        // doesn't gets the whole history.
        let start = !session.opened;
        let messages = session.pending(engine.sent_through(session.id()));
        let sent = session.len();

        let cmd = if start {
            ControllerCmd::StartChat {
                chat_id: session.id().to_string(),
                messages,
                gen_spec: self.spec,
                thinking: self.thinking,
                model_id: None,
                model_size_bytes: None,
                tools: self.tools,
                tx,
            }
        } else {
            ControllerCmd::ContinueChat {
                chat_id: session.id().to_string(),
                new_messages: messages,
                // Carried so a conversation the controller has evicted can be
                // rebuilt instead of refused. `opened` is this layer's belief
                // about a cache it does not own, and eviction happens without
                // telling it.
                transcript: session.messages().to_vec(),
                gen_spec: self.spec,
                model_id: None,
                model_size_bytes: None,
                tx,
            }
        };

        engine.send(cmd)?;
        // Deliberately not recording delivery here. `send` only puts the
        // command on the controller's channel; starting the session,
        // appending, and opening the puller can all still fail. Marking the
        // conversation open and its messages delivered before any of that
        // succeeded meant a retry sent only what came *after* messages the
        // backend never took.
        //
        // The engine acknowledges instead, and [`Chat::settle`] records it.
        // A turn nobody reads far enough to see acknowledged simply stays
        // closed, and the next one rebuilds — slower, never wrong.
        let _ = sent;
        Ok((TokenStream::new(rx), session))
    }
}

/// Record what the engine acknowledged holding.
///
/// Called once a turn's stream has been read far enough to know. Until then
/// the conversation stays closed, so the next turn rebuilds from the
/// authoritative transcript rather than sending a suffix of a conversation the
/// backend may not have.
fn settle(engine: &Engine, session: &mut Session, stream: &TokenStream) {
    let Some(delivered) = stream.accepted() else {
        return;
    };
    let already = if session.opened {
        engine.sent_through(session.id())
    } else {
        0
    };
    engine.mark_sent(session.id(), already + delivered);
    session.opened = true;
}

/// Whether any message carries an image.
///
/// Scans the whole transcript, not just this turn: a conversation that picked
/// up an image earlier still needs a multimodal model to replay it.
pub(crate) fn has_images(messages: &[Message]) -> bool {
    use crate::types::message::{MessageBody, MessageChunk, MessageContent};
    messages.iter().any(|m| match &m.body {
        MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        } => chunks
            .iter()
            .any(|c| matches!(c, MessageChunk::ImageUrl { .. })),
        _ => false,
    })
}

/// Translate a streamed tool call into the form a message stores.
///
/// The two differ because they answer different questions: the event carries
/// raw argument text exactly as the model emitted it, while the stored message
/// has to round-trip through the chat template. Arguments that don't parse as
/// JSON are kept as a string rather than dropped — a malformed call is still
/// part of the transcript.
fn as_message_call(call: &ToolCall) -> MessageToolCall {
    MessageToolCall {
        // Models using native tool syntax emit no id; the template only needs
        // one to be present and distinct within the turn.
        id: call.id.clone().unwrap_or_else(|| call.name.clone()),
        r#type: "function".to_string(),
        function: FunctionDefinition {
            description: None,
            name: call.name.clone(),
            arguments: serde_json::from_str(&call.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone())),
        },
    }
}

impl std::fmt::Debug for Chat<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat")
            .field("session", &self.session.id())
            .field("messages", &self.session.len())
            .finish_non_exhaustive()
    }
}
