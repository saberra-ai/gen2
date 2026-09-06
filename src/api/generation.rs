//! [`Generation`] — one prompt against a conversation that is thrown away.

use crate::generation::ThinkingMode;

use super::error::Result;
use super::input::Input;
use super::model::Model;
use super::response::Response;
use super::session::Session;
use super::tool_defs::{ToolDefinition, ToolSet};
use super::turn::{GenerationOptions, ToolChoice, Turn};

/// How tools are introduced to a model that is offered them one-shot. The
/// same wording the agent loop uses, so a model behaves the same either way.
pub(crate) const TOOL_PROMPT: &str =
    "Call a tool when you need information or an action. Answer directly when you don't.";

/// A one-shot generation being configured.
///
/// Built by [`Model::generate`]. Runs as a [`Turn`] against an ephemeral
/// [`Session`] (api_spec.md §6): the system prompt, the input, and the reply
/// exist only for this call. When a later turn might refer back to this one —
/// a tool call that needs a follow-up, say — use a `Session` and
/// [`Model::turn`] instead.
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
#[must_use = "a Generation does nothing until .run(), .text(), or .structured() is called"]
pub struct Generation<'m> {
    model: &'m Model,
    input: Input,
    system: Option<String>,
    options: GenerationOptions,
    tools: Option<ToolSet>,
    tool_choice: ToolChoice,
}

impl<'m> Generation<'m> {
    pub(crate) fn new(model: &'m Model, input: Input) -> Self {
        Self {
            model,
            input,
            system: None,
            options: GenerationOptions::default(),
            tools: None,
            tool_choice: ToolChoice::Auto,
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
    /// belongs in a `Session`. Takes a [`ToolSet`], a list of
    /// [`ToolDefinition`]s, or wire [`ToolSpec`](crate::ToolSpec)s.
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ToolDefinition>,
    {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    /// How the model may use the tools it is offered.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    // ── Sampling ────────────────────────────────────────────────────────────

    /// Cap how many tokens may be generated.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.options.max_tokens = Some(n);
        self
    }

    /// Sampling temperature. `0.0` is greedy — but prefer
    /// [`Generation::greedy`], which also pins the seed.
    pub fn temperature(mut self, t: f32) -> Self {
        self.options.temperature = Some(t);
        self
    }

    /// Seed the sampler, making a given temperature reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.options.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    pub fn greedy(mut self) -> Self {
        self.options = std::mem::take(&mut self.options).greedy();
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

    /// The reasoning channel, on models that expose one.
    pub fn reasoning(mut self, mode: ThinkingMode) -> Self {
        self.options.reasoning = mode;
        self
    }

    /// Every common control at once, replacing what was set so far.
    pub fn options(mut self, options: GenerationOptions) -> Self {
        self.options = options;
        self
    }

    // ── Running ─────────────────────────────────────────────────────────────

    /// Run it and return the reply text.
    pub fn text(self) -> Result<String> {
        Ok(self.run()?.text())
    }

    /// Run it and return the full [`Response`].
    pub fn run(self) -> Result<Response> {
        let (model, mut session, staged) = self.prepare();
        let done = staged.turn(model, &mut session).run();
        // The conversation is over whether or not it succeeded; the engine
        // must not keep bookkeeping for it.
        model.engine().forget(&session);
        Ok(done?.detached())
    }

    /// Run it and decode the reply as `T` (api_spec.md §13, §28.11).
    ///
    /// See [`Turn::structured`] for how the schema is enforced.
    ///
    /// ```no_run
    /// # let model = gen2::load("m.gguf")?;
    /// #[derive(serde::Deserialize, gen2::schemars::JsonSchema)]
    /// struct Invoice { vendor: String, total: f64 }
    ///
    /// let invoice: Invoice = model
    ///     .generate("Acme Ltd — total $1,240.00")
    ///     .system("Extract the invoice fields")
    ///     .structured()?;
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn structured<T>(self) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let (model, mut session, staged) = self.prepare();
        let done = staged.turn(model, &mut session).structured::<T>();
        model.engine().forget(&session);
        done
    }

    /// The ephemeral session, and what the turn on it will carry.
    fn prepare(self) -> (&'m Model, Session, Staged) {
        let mut session = Session::new();
        if let Some(system) = self.system {
            session.set_system(system);
        }
        if let Some(tools) = self.tools {
            session.set_tools(tools);
        }
        let staged = Staged {
            message: self.input.into_message(),
            options: self.options,
            tool_choice: self.tool_choice,
        };
        (self.model, session, staged)
    }
}

/// What a one-shot call stages on its turn.
struct Staged {
    message: crate::types::message::Message,
    options: GenerationOptions,
    tool_choice: ToolChoice,
}

impl Staged {
    fn turn<'a>(self, model: &'a Model, session: &'a mut Session) -> Turn<'a> {
        model
            .turn(session)
            .message(self.message)
            .options(self.options)
            .tool_choice(self.tool_choice)
    }
}

impl std::fmt::Debug for Generation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("model", &self.model.id())
            .field("input", &self.input)
            .field("system", &self.system)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::output::FinishReason;
    use crate::api::runtime::Runtime;
    use crate::test_support::{Script, Step};

    #[test]
    fn one_shot_runs_over_a_throwaway_session_and_produces_a_response() {
        let model = Runtime::scripted(Script::new().say(["hi ", "there"]));
        let script = model.engine().script().clone();
        let response = model
            .generate("hello")
            .system("Be brief.")
            .max_tokens(8)
            .run()
            .unwrap();
        assert_eq!(response.text(), "hi there");
        assert_eq!(*response.finish_reason(), FinishReason::Stop);
        assert_eq!(response.message_id(), None, "there is no session to name");
        assert_eq!(script.seen(), ["Be brief.", "hello"]);
        assert_eq!(script.specs_seen()[0].max_tokens, Some(8));
    }

    #[test]
    fn one_shot_tools_come_back_as_calls() {
        let model = Runtime::scripted(Script::new().program([
            Step::tool_call("get_weather", r#"{"city":"Paris"}"#),
            Step::eos(),
        ]));
        let script = model.engine().script().clone();
        let response = model
            .generate("Weather in Paris?")
            .tools([ToolDefinition::new("get_weather").description("Weather")])
            .run()
            .unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::ToolCall);
        assert_eq!(response.tool_calls()[0].name(), "get_weather");
        assert_eq!(script.tools_seen(), vec![vec!["get_weather".to_string()]]);
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Invoice {
        vendor: String,
        total: f64,
    }

    #[test]
    fn one_shot_structured_output_decodes() {
        let model = Runtime::scripted(Script::new().say([r#"{"vendor":"Acme","total":1240.0}"#]));
        let invoice: Invoice = model
            .generate("Acme Ltd — total $1,240.00")
            .system("Extract the invoice fields")
            .structured()
            .unwrap();
        assert_eq!(
            invoice,
            Invoice {
                vendor: "Acme".into(),
                total: 1240.0
            }
        );
    }
}
