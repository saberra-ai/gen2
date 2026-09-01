//! [`Inference`] — one prompt, no conversation kept.

use crate::backend::common::grammar::GrammarSpec;
use crate::generation::GenSpec;

use super::engine::Engine;
use super::error::Result;
use super::session::Session;
use super::stream::{Completion, Tokens};

/// A single prompt against a throwaway conversation.
///
/// Built by [`Engine::infer`]. Same sampling knobs as [`Chat`](super::Chat);
/// the difference is that nothing is retained afterwards, so there is no
/// session to read back.
#[must_use = "an Inference does nothing until .text(), .run(), or .tokens() is called"]
pub struct Inference<'e> {
    engine: &'e Engine,
    session: Session,
    text: String,
    spec: Option<GenSpec>,
}

impl<'e> Inference<'e> {
    pub(crate) fn new(engine: &'e Engine, text: String) -> Self {
        Self {
            engine,
            session: Session::new(),
            text,
            spec: None,
        }
    }

    /// Prepend a system prompt.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.session = Session::new().with_system(text);
        self
    }

    /// Cap how many tokens this may generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.spec_mut().max_tokens = Some(n);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.spec_mut().temperature = Some(t);
        self
    }

    /// Seed the sampler.
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec_mut().seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    pub fn greedy(mut self) -> Self {
        let spec = self.spec_mut();
        spec.temperature = Some(0.0);
        spec.seed = Some(spec.seed.unwrap_or(0));
        self
    }

    /// Constrain output to a grammar — JSON schema, regex, Lark, or GBNF.
    pub fn grammar(mut self, grammar: GrammarSpec) -> Self {
        self.spec_mut().grammar = Some(grammar);
        self
    }

    /// Drop any engine-level grammar for this call.
    pub fn unconstrained(mut self) -> Self {
        self.spec_mut().grammar = None;
        self
    }

    /// Use a fully-built [`GenSpec`], overriding everything above.
    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Run it and return the reply text.
    pub fn text(self) -> Result<String> {
        Ok(self.run()?.text)
    }

    /// Run it, streaming fragments to `on_token`, and return the text.
    pub fn text_streaming(self, on_token: impl FnMut(&str)) -> Result<String> {
        Ok(self.run_streaming(on_token)?.text)
    }

    /// Run it and return the full outcome — text, stats, finish reason.
    pub fn run(self) -> Result<Completion> {
        self.run_streaming(|_| {})
    }

    /// [`Self::run`], with `on_token` called per fragment.
    pub fn run_streaming(self, on_token: impl FnMut(&str)) -> Result<Completion> {
        let (engine, mut session, chat) = self.build();
        let done = chat_with(engine, &mut session, chat).send_streaming(on_token);
        engine.forget(&session);
        done
    }

    /// Run it and iterate the text fragments.
    ///
    /// The throwaway session is dropped as the tokens are consumed, so the
    /// engine forgets the conversation once the iterator is done with it.
    pub fn tokens(self) -> Result<Tokens> {
        let (engine, mut session, chat) = self.build();
        let tokens = chat_with(engine, &mut session, chat).tokens();
        engine.forget(&session);
        tokens
    }

    fn spec_mut(&mut self) -> &mut GenSpec {
        self.spec
            .get_or_insert_with(|| self.engine.default_gen_spec())
    }

    fn build(self) -> (&'e Engine, Session, (String, Option<GenSpec>)) {
        (self.engine, self.session, (self.text, self.spec))
    }
}

/// Apply the prompt and spec to a `Chat` over `session`.
fn chat_with<'a>(
    engine: &'a Engine,
    session: &'a mut Session,
    (text, spec): (String, Option<GenSpec>),
) -> super::chat::Chat<'a> {
    let chat = engine.chat(session).user(text);
    match spec {
        Some(spec) => chat.gen_spec(spec),
        None => chat,
    }
}
