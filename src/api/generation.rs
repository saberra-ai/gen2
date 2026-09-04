//! [`Generation`] — one prompt against a conversation that is thrown away.

use crate::generation::GenSpec;
use crate::types::message::ToolSpec;

use super::error::Result;
use super::input::Input;
use super::model::Model;
use super::response::Response;
use super::session::Session;

/// How tools are introduced to a model that is offered them one-shot. The
/// same wording the agent loop uses, so a model behaves the same either way.
const TOOL_PROMPT: &str =
    "Call a tool when you need information or an action. Answer directly when you don't.";

/// A one-shot generation being configured.
///
/// Built by [`Model::generate`]. Runs against an ephemeral conversation:
/// the system prompt, the input, and the reply exist only for this call.
/// When a later turn might refer back to this one, use a `Session` (S2.2)
/// instead.
///
/// ```no_run
/// # let model = gen2::load("m.gguf")?;
/// let response = model
///     .generate("Write a short story")
///     .system("You write terse speculative fiction.")
///     .temperature(0.8)
///     .max_tokens(512)
///     .run()?;
/// println!("{}", response.text());
/// # Ok::<(), gen2::Error>(())
/// ```
#[must_use = "a Generation does nothing until .run() or .text() is called"]
pub struct Generation<'m> {
    model: &'m Model,
    input: Input,
    system: Option<String>,
    spec: GenSpec,
    tools: Option<Vec<ToolSpec>>,
}

impl<'m> Generation<'m> {
    pub(crate) fn new(model: &'m Model, input: Input) -> Self {
        Self {
            // Starts from the engine's defaults, so anything set when the
            // model was loaded applies unless this call overrides it.
            spec: model.engine().default_gen_spec(),
            model,
            input,
            system: None,
            tools: None,
        }
    }

    // ── Context ─────────────────────────────────────────────────────────────

    /// A system prompt for this call.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(text.into());
        self
    }

    /// Attach an image to the input, by path or URL.
    ///
    /// Same as building the [`Input`] with [`Input::image`]. The model must
    /// accept images, or the call is refused before anything is generated.
    pub fn image(mut self, source: impl AsRef<std::path::Path>) -> Self {
        self.input = self.input.image(source);
        self
    }

    /// Offer tools. The model may answer with tool calls instead of text —
    /// see [`Response::tool_calls`] — and it is the caller's to run them.
    ///
    /// One-shot generation runs no tool loop: a call that needs a follow-up
    /// belongs in a `Session`. Takes anything that yields [`ToolSpec`]s,
    /// which includes [`ToolSet::specs`](crate::ToolSet::specs).
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolSpec>) -> Self {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    // ── Sampling ────────────────────────────────────────────────────────────

    /// Cap how many tokens may be generated.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.spec.max_tokens = Some(n);
        self
    }

    /// Sampling temperature. `0.0` is greedy — but prefer
    /// [`Generation::greedy`], which also pins the seed.
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

    // ── Running ─────────────────────────────────────────────────────────────

    /// Run it and return the reply text.
    pub fn text(self) -> Result<String> {
        Ok(self.run()?.text())
    }

    /// Run it and return the full [`Response`].
    pub fn run(self) -> Result<Response> {
        let engine = self.model.engine();
        let mut session = match self.system {
            Some(system) => Session::new().with_system(system),
            None => Session::new(),
        };
        let max_tokens = self.spec.max_tokens;

        let mut chat = engine
            .chat(&mut session)
            .message(self.input.into_message())
            .gen_spec(self.spec);
        if let Some(tools) = self.tools {
            chat = chat.tools(tools, TOOL_PROMPT);
        }
        let done = chat.send();
        // The conversation is over whether or not it succeeded; the engine
        // must not keep bookkeeping for it.
        engine.forget(&session);
        Ok(Response::from_completion(done?, max_tokens))
    }
}

impl std::fmt::Debug for Generation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("model", &self.model.id())
            .field("input", &self.input)
            .field("system", &self.system)
            .finish_non_exhaustive()
    }
}
