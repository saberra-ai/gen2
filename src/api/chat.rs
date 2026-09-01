//! [`Chat`] — one turn, configured fluently.

use crate::backend::common::grammar::GrammarSpec;
use crate::controller::ControllerCmd;
use crate::generation::{GenSpec, ThinkingMode};
use crate::types::message::{Message, Tool};

use super::engine::{Engine, event_channel};
use super::error::Result;
use super::stream::TokenStream;

/// A turn being built.
///
/// Add messages, set sampling, then call [`Chat::stream`] for events or
/// [`Chat::text`] for the finished reply.
///
/// ```no_run
/// # use pio_gen2::Engine;
/// # let engine = Engine::load("model.gguf")?;
/// let reply = engine.chat("c1")
///     .user("Explain entropy in one sentence.")
///     .max_tokens(256)
///     .text()?;
/// # Ok::<(), pio_gen2::Error>(())
/// ```
#[must_use = "a Chat does nothing until .stream(), .text(), or .send() is called"]
pub struct Chat<'e> {
    engine: &'e Engine,
    chat_id: String,
    messages: Vec<Message>,
    spec: GenSpec,
    thinking: ThinkingMode,
    tools: Option<(Vec<Tool>, String)>,
    fresh: bool,
}

impl<'e> Chat<'e> {
    pub(crate) fn new(engine: &'e Engine, chat_id: String) -> Self {
        Self {
            engine,
            chat_id,
            messages: Vec::new(),
            spec: GenSpec::default(),
            thinking: ThinkingMode::default(),
            tools: None,
            fresh: false,
        }
    }

    /// Start this turn from scratch, ignoring any history under this chat id.
    pub fn fresh(mut self) -> Self {
        self.fresh = true;
        self
    }

    // ── Messages ────────────────────────────────────────────────────────────

    /// Append a user message.
    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::user(text));
        self
    }

    /// Append a system message.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::system(text));
        self
    }

    /// Append already-built messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
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
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    ///
    /// Worth naming, because it is *not* the default — an unconfigured turn
    /// samples with a random seed, so the same prompt gives different text
    /// each run.
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

    /// Constrain the output to a grammar — JSON schema, regex, Lark, or GBNF.
    ///
    /// Enforced during decoding, so the model cannot produce output that
    /// violates it. Works the same across every backend.
    pub fn grammar(mut self, grammar: GrammarSpec) -> Self {
        self.spec.grammar = Some(grammar);
        self
    }

    /// Use a fully-built [`GenSpec`], overriding anything set above.
    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.spec = spec;
        self
    }

    // ── Tools and reasoning ─────────────────────────────────────────────────

    /// Offer tools to the model. `prompt` introduces them in the template.
    ///
    /// The names also arm the output parser, so `name[...]`-shaped text
    /// outside a real call block stays text.
    pub fn tools(mut self, tools: Vec<Tool>, prompt: impl Into<String>) -> Self {
        self.tools = Some((tools, prompt.into()));
        self
    }

    /// Force the reasoning channel on or off for models that expose one.
    /// Defaults to the model's own template default.
    pub fn thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
        self
    }

    // ── Running ─────────────────────────────────────────────────────────────

    /// Start generating and return the event stream.
    ///
    /// The first turn under a given chat id opens the conversation; later
    /// turns continue it, reusing its warm KV cache rather than re-reading the
    /// history. [`Chat::fresh`] forces a restart.
    pub fn stream(self) -> Result<TokenStream> {
        let (tx, rx) = event_channel(self.engine.event_channel_capacity());
        let start = self.fresh || self.engine.claim_new_chat(&self.chat_id);

        let cmd = if start {
            ControllerCmd::StartChat {
                chat_id: self.chat_id,
                messages: self.messages,
                gen_spec: self.spec,
                thinking: self.thinking,
                model_id: None,
                model_size_bytes: None,
                tools: self.tools,
                tx,
            }
        } else {
            ControllerCmd::ContinueChat {
                chat_id: self.chat_id,
                new_messages: self.messages,
                gen_spec: self.spec,
                model_id: None,
                model_size_bytes: None,
                tx,
            }
        };

        self.engine.send(cmd)?;
        Ok(TokenStream::new(rx))
    }

    /// Generate and return the reply text, discarding other events.
    pub fn text(self) -> Result<String> {
        self.stream()?.text()
    }

    /// Generate, invoking `on_token` per fragment, and return the full text.
    pub fn text_streaming(self, on_token: impl FnMut(&str)) -> Result<String> {
        self.stream()?.text_streaming(on_token)
    }
}

impl std::fmt::Debug for Chat<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat")
            .field("chat_id", &self.chat_id)
            .field("messages", &self.messages.len())
            .field("fresh", &self.fresh)
            .finish_non_exhaustive()
    }
}
