//! [`Chat`] — one turn against a [`Session`].

use crate::backend::common::grammar::GrammarSpec;
use crate::controller::ControllerCmd;
use crate::generation::{GenSpec, ThinkingMode};
use crate::types::message::{Message, Tool};

use super::engine::{Engine, event_channel};
use super::error::Result;
use super::session::Session;
use super::stream::{Completion, TokenStream, Tokens};

/// A turn being built against a conversation.
///
/// Add messages, set sampling, then run it. Whatever the model replies is
/// appended to the session, so the conversation stays whole without you
/// managing it.
///
/// ```no_run
/// # use pio_gen2::{Engine, Session};
/// # let engine = Engine::load("m.gguf")?;
/// let mut session = Session::new();
/// engine.chat(&mut session)
///     .user("Explain entropy in one sentence.")
///     .max_tokens(256)
///     .send()?;
/// println!("{}", session.latest_text().unwrap_or_default());
/// # Ok::<(), pio_gen2::Error>(())
/// ```
#[must_use = "a Chat does nothing until .send(), .text(), .stream(), or .tokens() is called"]
pub struct Chat<'a> {
    engine: &'a Engine,
    session: &'a mut Session,
    spec: GenSpec,
    thinking: ThinkingMode,
    tools: Option<(Vec<Tool>, String)>,
}

impl<'a> Chat<'a> {
    pub(crate) fn new(engine: &'a Engine, session: &'a mut Session) -> Self {
        Self {
            // Starts from the engine's configured defaults, so anything set at
            // build time applies unless this turn overrides it.
            spec: engine.default_gen_spec(),
            engine,
            session,
            thinking: ThinkingMode::default(),
            tools: None,
        }
    }

    // ── Messages ────────────────────────────────────────────────────────────

    /// Append a user message to the conversation.
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::user(text))
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
    pub fn tools(mut self, tools: Vec<Tool>, prompt: impl Into<String>) -> Self {
        self.tools = Some((tools, prompt.into()));
        self
    }

    /// Force the reasoning channel on or off for models that expose one.
    pub fn thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
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
    pub fn send_streaming(self, on_token: impl FnMut(&str)) -> Result<Completion> {
        let (stream, session) = self.begin()?;
        let done = stream.complete_streaming(on_token)?;
        session.push(Message::assistant_structured(done.text.clone(), None));
        Ok(done)
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
                gen_spec: self.spec,
                model_id: None,
                model_size_bytes: None,
                tx,
            }
        };

        engine.send(cmd)?;
        engine.mark_sent(session.id(), sent);
        session.opened = true;
        Ok((TokenStream::new(rx), session))
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
