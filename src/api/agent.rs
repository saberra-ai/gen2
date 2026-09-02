//! [`Agent`] — a goal, a set of tools, and a loop that runs until it's done.
//!
//! The difference from [`Chat`](super::Chat) with a tool handler is that the
//! agent owns dispatch: it resolves the tool the model named, validates the
//! arguments against that tool's schema, schedules the call, routes failures by
//! whether the model can fix them, and stops when it's making no progress. A
//! caller supplies tools, not a `match`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::engine::Engine;
use super::error::{Error, Result};
use super::session::Session;
use super::stream::{Budget, Completion, Finish, Struggle};
use super::tools::{
    IntoTool, Tool, ToolContext, ToolError, ToolLoading, ToolOutput, ToolRegistry, ToolSearch,
    ToolSpec,
};
use crate::backend::common::grammar::GrammarSpec;
use crate::generation::GenSpec;
use crate::types::message::{FunctionDefinition, Message, ToolCall, ToolSpec as WireToolSpec};

/// How many rounds an agent may run by default.
pub const DEFAULT_MAX_STEPS: usize = 12;

/// How many identical calls in a row count as going in circles.
const REPEAT_LIMIT: usize = 3;

/// The name of the built-in tool that finds deferred tools.
pub const SEARCH_TOOL: &str = "search_tools";

/// Whether a tool needs a human's say-so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Risk {
    /// Runs without asking.
    #[default]
    Safe,
    /// Gated when the agent is in [`ApprovalMode::AskOnRisky`].
    Risky,
}

/// When the agent stops to ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ApprovalMode {
    /// Everything runs. The default, because it is what most agents actually
    /// want and a gate nobody reads is worse than none.
    #[default]
    Auto,
    /// Tools declaring [`Risk::Risky`] go through the approval callback.
    AskOnRisky,
}

/// What to do with a call awaiting approval.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Decision {
    /// Run it.
    Allow,
    /// Don't. The reason reaches the model as a denial, and the run ends —
    /// a denied action is not something the model should retry around.
    Deny(String),
}

type ApprovalFn = Box<dyn FnMut(&str, &Value, &ToolSpec) -> Decision + Send>;

/// Limits that stop a run.
#[derive(Debug, Clone)]
struct Budgets {
    max_steps: usize,
    max_tokens: Option<u32>,
    deadline: Option<Duration>,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_tokens: None,
            deadline: None,
        }
    }
}

/// Injects messages into a run that is already going.
///
/// Cheap to clone and safe to move to another thread — which is the point,
/// since the thread driving the agent is busy. Two shapes, because they answer
/// different questions: a follow-up adds to the task, an interruption changes
/// it.
#[derive(Clone, Default)]
pub struct Steering {
    queue: Arc<Mutex<VecDeque<Steer>>>,
    /// Set on a spawned run, where there is an owned engine to ask. Without it
    /// an interruption still lands, just at the next step boundary rather than
    /// mid-generation.
    stop: Option<(Arc<Engine>, String)>,
}

#[derive(Debug, Clone)]
pub(crate) struct Steer {
    pub(crate) message: String,
    pub(crate) interrupt: bool,
}

impl Steering {
    /// Add to the task. Delivered at the next step boundary, so the current
    /// generation and any tool already running finish first.
    pub fn follow_up(&self, message: impl Into<String>) {
        self.push(message.into(), false);
    }

    /// Change the task now.
    ///
    /// On a spawned run this cuts the generation short: the partial reply is
    /// kept in the transcript, the rest of the round's planned tool calls are
    /// abandoned, and the message is delivered before the model writes again.
    /// On a borrowed [`Agent`] there is no engine handle to ask, so it lands at
    /// the next step boundary instead.
    ///
    /// A tool already executing still completes either way — stopping mid-call
    /// would leave a side effect with nothing in the transcript to show it
    /// happened.
    pub fn interrupt(&self, message: impl Into<String>) {
        self.push(message.into(), true);
        // Cutting the generation is what makes this different from a follow-up.
        if let Some((engine, session_id)) = &self.stop {
            let _ = engine.stop(session_id.clone());
        }
    }

    /// Attach the engine this steers, so interruptions can stop a generation.
    pub(crate) fn with_engine(mut self, engine: Arc<Engine>, session_id: String) -> Self {
        self.stop = Some((engine, session_id));
        self
    }

    /// Whether anything is waiting.
    pub fn is_pending(&self) -> bool {
        self.queue.lock().map(|q| !q.is_empty()).unwrap_or(false)
    }

    fn push(&self, message: String, interrupt: bool) {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(Steer { message, interrupt });
        }
    }

    /// Whether this handle can stop a generation, or only queue.
    pub fn can_interrupt_generation(&self) -> bool {
        self.stop.is_some()
    }

    pub(crate) fn drain(&self) -> Vec<Steer> {
        self.queue
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default()
    }

    /// Whether an interruption is waiting, so the loop can stop the current
    /// generation rather than let it run to completion.
    pub(crate) fn wants_interrupt(&self) -> bool {
        self.queue
            .lock()
            .map(|q| q.iter().any(|s| s.interrupt))
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for Steering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Steering")
            .field("pending", &self.is_pending())
            .finish()
    }
}

/// Builds an agent run.
#[must_use = "an Agent does nothing until .run() is called"]
pub struct Agent<'a> {
    engine: &'a Engine,
    session: &'a mut Session,
    entries: Vec<(Arc<dyn Tool>, ToolLoading)>,
    search: Option<ToolSearch>,
    budgets: Budgets,
    approval: ApprovalMode,
    on_approval: Option<ApprovalFn>,
    tool_prompt: String,
    steering: Steering,
    spec: GenSpec,
    images: Vec<String>,
    answer_as: Option<(GrammarSpec, String)>,
}

impl<'a> Agent<'a> {
    pub(crate) fn new(engine: &'a Engine, session: &'a mut Session) -> Self {
        Self {
            engine,
            session,
            entries: Vec::new(),
            search: None,
            budgets: Budgets::default(),
            approval: ApprovalMode::Auto,
            on_approval: None,
            tool_prompt: "Call a tool when you need information or an action. \
                          Answer directly when you don't."
                .into(),
            steering: Steering::default(),
            spec: engine.default_gen_spec(),
            images: Vec::new(),
            answer_as: None,
        }
    }

    /// As [`Agent::new`], with a steering handle supplied by the caller — the
    /// spawned path builds one that can reach the engine.
    pub(crate) fn new_steered(
        engine: &'a Engine,
        session: &'a mut Session,
        steering: Steering,
    ) -> Self {
        let mut agent = Self::new(engine, session);
        agent.steering = steering;
        agent
    }

    /// Install an already-boxed approval callback.
    pub(crate) fn with_approval_fn(mut self, f: ApprovalFn) -> Self {
        self.on_approval = Some(f);
        self
    }

    /// A handle for injecting messages while this agent runs.
    ///
    /// Take it before starting the run and move it to whichever thread has the
    /// user's attention.
    pub fn steering(&self) -> Steering {
        self.steering.clone()
    }

    /// Install an already-built set of registrations.
    pub(crate) fn with_entries(mut self, entries: Vec<(Arc<dyn Tool>, ToolLoading)>) -> Self {
        self.entries = entries;
        self
    }

    /// Register a tool the model can see from the first turn.
    pub fn add_tool(mut self, tool: impl IntoTool) -> Self {
        self.entries.push((tool.into_tool(), ToolLoading::Resident));
        self
    }

    /// Register several resident tools.
    pub fn add_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.entries.extend(
            tools
                .into_iter()
                .map(|t| (t.into_tool(), ToolLoading::Resident)),
        );
        self
    }

    /// Register a tool the model finds only by searching.
    ///
    /// Its schema stays out of the prompt until search surfaces it, at which
    /// point the spec is appended to the conversation — leaving the prefix, and
    /// the warm cache built over it, untouched.
    pub fn defer_tool(mut self, tool: impl IntoTool) -> Self {
        self.entries.push((tool.into_tool(), ToolLoading::Deferred));
        self
    }

    /// Register several deferred tools — an MCP server's whole surface, say.
    pub fn defer_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.entries.extend(
            tools
                .into_iter()
                .map(|t| (t.into_tool(), ToolLoading::Deferred)),
        );
        self
    }

    /// How deferred tools are found. Required if anything is deferred.
    pub fn tool_search(mut self, search: ToolSearch) -> Self {
        self.search = Some(search);
        self
    }

    /// Attach images to the goal.
    ///
    /// Paths become `file://` URLs; existing URLs pass through. Needs a
    /// multimodal model loaded with a projector — a text-only one is rejected
    /// before anything is generated.
    pub fn images<I, P>(mut self, images: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        self.images
            .extend(images.into_iter().map(|p| p.as_ref().to_string()));
        self
    }

    /// Require the final answer to match a grammar.
    ///
    /// Applied to one extra turn *after* the agent has finished its work, not
    /// to the whole run: constraining every turn would forbid the tool-call
    /// syntax the model needs to do anything. So the agent reasons and calls
    /// tools freely, and is then asked once more for the answer in the
    /// required shape.
    ///
    /// `instruction` is what to ask for; it reaches the model as a final user
    /// message.
    pub fn answer_as(mut self, grammar: GrammarSpec, instruction: impl Into<String>) -> Self {
        self.answer_as = Some((grammar, instruction.into()));
        self
    }

    /// Sampling temperature for every turn of this run.
    pub fn temperature(mut self, t: f32) -> Self {
        self.spec.temperature = Some(t);
        self
    }

    /// Seed the sampler.
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec.seed = Some(seed);
        self
    }

    /// Decode deterministically: temperature 0 with a fixed seed.
    ///
    /// An agent's choices are as worth reproducing as its prose — more so,
    /// since which tool it reached for is the part you debug.
    pub fn greedy(mut self) -> Self {
        self.spec.temperature = Some(0.0);
        self.spec.seed = Some(self.spec.seed.unwrap_or(0));
        self
    }

    /// Use a fully-built [`GenSpec`] for every turn.
    pub fn gen_spec(mut self, spec: GenSpec) -> Self {
        self.spec = spec;
        self
    }

    /// Cap the rounds of generate-and-call. Defaults to [`DEFAULT_MAX_STEPS`].
    pub fn max_steps(mut self, steps: usize) -> Self {
        self.budgets.max_steps = steps;
        self
    }

    /// Cap total tokens generated across the run.
    pub fn max_tokens_total(mut self, tokens: u32) -> Self {
        self.budgets.max_tokens = Some(tokens);
        self
    }

    /// Stop after this much wall-clock.
    pub fn deadline(mut self, after: Duration) -> Self {
        self.budgets.deadline = Some(after);
        self
    }

    /// Ask before running tools that declare themselves risky.
    pub fn approval(mut self, mode: ApprovalMode) -> Self {
        self.approval = mode;
        self
    }

    /// Called for each risky tool call when in [`ApprovalMode::AskOnRisky`].
    ///
    /// Synchronous by necessity: the loop cannot proceed without an answer, so
    /// this is a callback rather than something observed on a stream.
    pub fn on_approval(
        mut self,
        f: impl FnMut(&str, &Value, &ToolSpec) -> Decision + Send + 'static,
    ) -> Self {
        self.on_approval = Some(Box::new(f));
        self
    }

    /// How the tool list is introduced to the model.
    pub fn tool_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.tool_prompt = prompt.into();
        self
    }

    /// Add the goal and run to completion.
    pub fn goal(self, text: impl Into<String>) -> Result<Completion> {
        self.run_with(Some(text.into()), |_| {}, |_| {})
    }

    /// Run against whatever is already in the session.
    pub fn run(self) -> Result<Completion> {
        self.run_with(None, |_| {}, |_| {})
    }

    /// Run, reporting each step to `observe`.
    pub fn run_streaming(
        self,
        goal: Option<String>,
        observe: impl FnMut(AgentStep<'_>),
    ) -> Result<Completion> {
        self.run_with(goal, |_| {}, observe)
    }

    /// Run, reporting generated text to `on_delta` and steps to `observe`.
    pub fn run_streaming_text(
        self,
        goal: Option<String>,
        on_delta: impl FnMut(&str),
        observe: impl FnMut(AgentStep<'_>),
    ) -> Result<Completion> {
        self.run_with(goal, on_delta, observe)
    }

    pub(crate) fn run_with(
        mut self,
        goal: Option<String>,
        mut on_delta: impl FnMut(&str),
        mut observe: impl FnMut(AgentStep<'_>),
    ) -> Result<Completion> {
        let mut registry = ToolRegistry::build(std::mem::take(&mut self.entries), self.search)?;

        // Index the deferred tools for semantic search. Skipped when no
        // embedder is loaded: hybrid then runs lexical-only, which costs recall
        // rather than the whole search.
        if registry
            .search_strategy()
            .is_some_and(|s| s.needs_embedder())
            && self.engine.is_embedder_loaded()
        {
            let texts = registry.deferred_search_text();
            let corpus: Vec<String> = texts.iter().map(|(_, t)| t.clone()).collect();
            if let Ok(vectors) = self.engine.embed(&corpus) {
                registry.set_embeddings(
                    texts
                        .into_iter()
                        .map(|(name, _)| name)
                        .zip(vectors)
                        .collect(),
                );
            }
        }

        // Tool definitions only reach the model when a conversation opens, so a
        // run registering a different set than the session was opened with must
        // reopen it — otherwise its tools are silently ignored and the model
        // keeps calling the old ones.
        self.session
            .note_tools(registry.prefix_fingerprint(&self.tool_prompt));

        let images = std::mem::take(&mut self.images);
        if let Some(text) = goal {
            if images.is_empty() {
                self.session.push_user(text);
            } else {
                self.session.push_user_with_images(text, images);
            }
        }

        let engine = self.engine;
        let budgets = self.budgets;
        let approval = self.approval;
        let mut on_approval = self.on_approval.take();
        let tool_prompt = self.tool_prompt.clone();
        let steering = self.steering.clone();
        let spec = self.spec.clone();
        let answer_as = self.answer_as.clone();
        let session: &mut Session = self.session;

        let started = Instant::now();
        let mut totals = Completion::default();
        let mut recent: VecDeque<(String, String)> = VecDeque::new();
        let mut spent_tokens: u32 = 0;
        let ctx = ToolContext::new(session.id());

        for step in 0..=budgets.max_steps {
            if step == budgets.max_steps {
                totals.finish = Finish::OutOfBudget(Budget::Steps);
                return Ok(totals);
            }
            if let Some(limit) = budgets.deadline
                && started.elapsed() >= limit
            {
                totals.finish = Finish::OutOfBudget(Budget::Deadline);
                return Ok(totals);
            }

            // Steer at the boundary, never mid-call: a tool already running has
            // side effects that must be recorded, and a half-applied change of
            // task is worse than a slightly late one.
            for steer in steering.drain() {
                session.push_user(steer.message);
            }

            // `stream` rather than `send`: the agent owns its transcript, and
            // `send` would append the assistant text itself — producing a
            // duplicate assistant turn on every round that also calls a tool.
            let mut chat = engine.chat(session).gen_spec(spec.clone());
            // An empty tool list is not the same as no tools: it renders an
            // empty `tools` array into the chat template, which some templates
            // handle badly enough to crash the backend.
            let specs = wire_specs(&registry);
            if !specs.is_empty() {
                chat = chat.tools(specs, tool_prompt.clone());
            }
            let turn = chat.stream()?.complete_streaming(&mut on_delta)?;

            totals.text = turn.text.clone();
            totals.dropped += turn.dropped;
            totals.compacted += turn.compacted;
            totals.tool_rounds = step;

            // Accumulate rather than replace. `ExecutionStats` describes one
            // turn, so a budget checked against the latest turn alone would let
            // the run spend its whole allowance on every step.
            spent_tokens += turn.stats.as_ref().map_or(0, |s| s.decode_tokens);
            if let Some(s) = &turn.stats {
                let mut merged = s.clone();
                merged.decode_tokens = spent_tokens;
                totals.stats = Some(merged);
            }

            if let Some(limit) = budgets.max_tokens
                && spent_tokens >= limit
            {
                totals.finish = Finish::OutOfBudget(Budget::Tokens);
                return Ok(totals);
            }

            if turn.tool_calls.is_empty() {
                session.push(Message::assistant_structured(turn.text.clone(), None));

                // A stop with steering waiting is an interruption, not an
                // answer: the caller cut this generation *in order to* say
                // something. Returning here would drop the message they
                // interrupted with, which is the opposite of what they asked
                // for. Loop instead, and the boundary delivers it.
                if turn.finish == Finish::Stopped && steering.is_pending() {
                    continue;
                }

                totals.finish = turn.finish.clone();

                // The work is done, so now shape the answer. One extra turn
                // rather than a grammar over the whole run, which would have
                // forbidden the tool-call syntax the model needed to get here.
                if let Some((grammar, instruction)) = &answer_as {
                    session.push_user(instruction.clone());
                    let shaped = engine
                        .chat(session)
                        .gen_spec(spec.clone())
                        .grammar(grammar.clone())
                        .send_streaming(&mut on_delta)?;
                    totals.text = shaped.text;
                    totals.finish = shaped.finish;
                }
                return Ok(totals);
            }

            // Record what was asked before what came back, so the transcript
            // reads in the order it happened.
            session.push(Message::assistant_tool_calls(
                turn.tool_calls.iter().map(as_wire_call).collect(),
            ));

            // Batch consecutive parallel-safe calls and run them together;
            // anything exclusive runs alone. Results are appended in call
            // order regardless of completion order, so the transcript reads
            // the way the model wrote it.
            let mut batch: Vec<(usize, &crate::generation::ToolCall, Value)> = Vec::new();
            let mut results: Vec<(usize, std::result::Result<ToolOutput, ToolError>)> = Vec::new();
            let mut interrupted = false;

            for (i, call) in turn.tool_calls.iter().enumerate() {
                // An interruption means the remaining calls were planned
                // against a task that no longer stands. The ones already run
                // this round stay in the transcript.
                if steering.wants_interrupt() {
                    interrupted = true;
                    break;
                }

                let args: Value = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| Value::String(call.arguments.clone()));

                // Going in circles: the same call with the same arguments is
                // the failure a step budget alone never catches.
                let key = (call.name.clone(), call.arguments.clone());
                recent.push_back(key.clone());
                if recent.len() > REPEAT_LIMIT {
                    recent.pop_front();
                }
                if recent.len() == REPEAT_LIMIT && recent.iter().all(|k| *k == key) {
                    totals.finish = Finish::GaveUp(Struggle::RepeatingCall {
                        tool: call.name.clone(),
                        times: REPEAT_LIMIT,
                    });
                    return Ok(totals);
                }

                // Search mutates the registry (hydration), so it can never
                // share a batch with calls that read it.
                let solo = call.name == SEARCH_TOOL
                    || registry
                        .get(&call.name)
                        .is_none_or(|t| !t.execution_policy().parallel_safe);

                if solo {
                    run_batch(
                        &registry,
                        &ctx,
                        std::mem::take(&mut batch),
                        approval,
                        on_approval.as_mut(),
                        &mut observe,
                        step,
                        &mut results,
                    );
                    let began = Instant::now();
                    observe(AgentStep::Calling {
                        step,
                        tool: &call.name,
                        args: &args,
                    });
                    let outcome = if call.name == SEARCH_TOOL {
                        run_search(engine, &mut registry, &args)
                    } else {
                        dispatch(&registry, &ctx, call, &args, approval, on_approval.as_mut())
                    };
                    observe(AgentStep::Called {
                        step,
                        tool: &call.name,
                        result: &outcome,
                        took: began.elapsed(),
                    });
                    results.push((i, outcome));
                } else {
                    batch.push((i, call, args));
                }
            }

            run_batch(
                &registry,
                &ctx,
                std::mem::take(&mut batch),
                approval,
                on_approval.as_mut(),
                &mut observe,
                step,
                &mut results,
            );

            results.sort_by_key(|(i, _)| *i);
            for (i, outcome) in results {
                let name = &turn.tool_calls[i].name;
                // The call this result answers. The id is right here in the
                // turn, and dropping it left the transcript unable to say
                // which of several parallel results belonged to which call —
                // every backend then had to guess, and guessed wrong.
                let call_id = turn.tool_calls[i].id.clone();
                match outcome {
                    Ok(out) => {
                        session.push(result_message(call_id, out.to_model_text()));
                    }
                    // A model-fixable failure goes back as a result so it can
                    // correct itself; anything else ends the run, because the
                    // model cannot repair a dead socket and telling it wastes a
                    // round of context.
                    Err(e) if e.is_model_actionable() => {
                        session.push(result_message(call_id, format!("error: {e}")));
                    }
                    Err(e) => {
                        totals.finish = Finish::Stopped;
                        return Err(Error::Generation {
                            code: "tool_failed".into(),
                            message: format!("{name}: {e}"),
                        });
                    }
                }
            }

            let _ = interrupted;
        }

        totals.finish = Finish::OutOfBudget(Budget::Steps);
        Ok(totals)
    }
}

/// What the agent is doing, reported as it happens.
///
/// Borrowed rather than owned: an observer renders or logs and moves on, and
/// the loop shouldn't allocate a copy of every argument for a callback that
/// usually ignores it.
#[derive(Debug)]
#[non_exhaustive]
pub enum AgentStep<'a> {
    /// About to run a tool.
    Calling {
        step: usize,
        tool: &'a str,
        args: &'a Value,
    },
    /// It finished.
    Called {
        step: usize,
        tool: &'a str,
        result: &'a std::result::Result<ToolOutput, ToolError>,
        took: Duration,
    },
}

/// Run a batch of parallel-safe calls concurrently, preserving call order in
/// the results.
///
/// Sequential dispatch is correct but wasteful: three independent reads in one
/// turn are three round trips of latency for no reason. Anything a tool
/// declared unsafe to parallelise never reaches here.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    registry: &ToolRegistry,
    ctx: &ToolContext,
    batch: Vec<(usize, &crate::generation::ToolCall, Value)>,
    approval: ApprovalMode,
    mut on_approval: Option<&mut ApprovalFn>,
    observe: &mut impl FnMut(AgentStep<'_>),
    step: usize,
    out: &mut Vec<(usize, std::result::Result<ToolOutput, ToolError>)>,
) {
    if batch.is_empty() {
        return;
    }

    // Approval is a synchronous callback, so it is resolved before anything is
    // dispatched — a human cannot answer three prompts concurrently anyway.
    let mut approved = Vec::new();
    for (i, call, args) in batch {
        observe(AgentStep::Calling {
            step,
            tool: &call.name,
            args: &args,
        });
        let denial = registry.get(&call.name).and_then(|tool| {
            if approval == ApprovalMode::AskOnRisky
                && tool.risk() == Risk::Risky
                && let Some(f) = on_approval.as_deref_mut()
                && let Decision::Deny(why) = f(&call.name, &args, tool.spec())
            {
                Some(why)
            } else {
                None
            }
        });
        match denial {
            Some(why) => out.push((i, Err(ToolError::Denied(why)))),
            None => approved.push((i, call, args)),
        }
    }

    if approved.is_empty() {
        return;
    }

    let started = Instant::now();
    let futures: Vec<_> = approved
        .iter()
        .map(|(i, call, args)| {
            let name = call.name.clone();
            let tool = registry.get(&call.name).cloned();
            let ctx = ctx.clone().with_call_id(call.id.clone());
            let args = args.clone();
            let idx = *i;
            async move {
                let result = match tool {
                    Some(t) => t.call(&ctx, args).await,
                    None => Err(ToolError::InvalidArguments(format!(
                        "no tool named '{name}'"
                    ))),
                };
                (idx, result)
            }
        })
        .collect();

    let finished =
        crate::task_util::block_on(async move { futures::future::join_all(futures).await });
    let took = started.elapsed();

    for (i, result) in finished {
        let name = approved
            .iter()
            .find(|(j, _, _)| *j == i)
            .map(|(_, c, _)| c.name.as_str())
            .unwrap_or("");
        observe(AgentStep::Called {
            step,
            tool: name,
            result: &result,
            took,
        });
        out.push((i, result));
    }
}

/// Resolve, gate, and run one call.
fn dispatch(
    registry: &ToolRegistry,
    ctx: &ToolContext,
    call: &crate::generation::ToolCall,
    args: &Value,
    approval: ApprovalMode,
    on_approval: Option<&mut ApprovalFn>,
) -> std::result::Result<ToolOutput, ToolError> {
    let Some(tool) = registry.get(&call.name) else {
        // A hallucinated name is the model's to fix, so say so rather than
        // failing the run.
        return Err(ToolError::InvalidArguments(format!(
            "no tool named '{}'",
            call.name
        )));
    };

    if approval == ApprovalMode::AskOnRisky
        && tool.risk() == Risk::Risky
        && let Some(f) = on_approval
        && let Decision::Deny(why) = f(&call.name, args, tool.spec())
    {
        return Err(ToolError::Denied(why));
    }

    let tool = Arc::clone(tool);
    let ctx = ctx.clone().with_call_id(call.id.clone());
    let args = args.clone();
    crate::task_util::block_on(async move { tool.call(&ctx, args).await })
}

/// The built-in tool that finds deferred tools.
fn run_search(
    engine: &Engine,
    registry: &mut ToolRegistry,
    args: &Value,
) -> std::result::Result<ToolOutput, ToolError> {
    let query = args
        .get("query")
        .and_then(|q| q.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("expected a 'query' string".into()))?;

    // The semantic half needs the query in the same space as the indexed tool
    // descriptions. Without this the strategy silently degraded to lexical, and
    // `Semantic` returned nothing at all.
    let embedding = registry
        .search_strategy()
        .filter(|s| s.needs_embedder())
        .and_then(|_| engine.embed_one(query).ok());

    let found = registry.search(query, 5, embedding.as_deref());
    if found.is_empty() {
        return Ok(ToolOutput::Text(format!("No tools match '{query}'.")));
    }

    // Hydration: the spec joins the *conversation*, never the prompt prefix.
    // That is what keeps the warm KV cache intact.
    let mut lines = Vec::new();
    for spec in &found {
        registry.mark_hydrated(&spec.name);
        lines.push(format!(
            "{}: {} — arguments: {}",
            spec.name, spec.description, spec.input_schema
        ));
    }
    Ok(ToolOutput::Text(format!(
        "Found {} tool(s), now available to call:\n{}",
        found.len(),
        lines.join("\n")
    )))
}

/// Specs the model should see this turn: everything visible, plus the search
/// tool when anything is deferred.
fn wire_specs(registry: &ToolRegistry) -> Vec<WireToolSpec> {
    let mut specs: Vec<WireToolSpec> = registry.visible_specs().iter().map(as_wire_spec).collect();
    if registry.search_strategy().is_some() && !registry.deferred_names().is_empty() {
        specs.push(search_tool_spec());
    }
    specs
}

fn search_tool_spec() -> WireToolSpec {
    WireToolSpec {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: SEARCH_TOOL.into(),
            description: Some(
                "Find tools that are not listed above. Describe what you need to do; \
                 matching tools become available to call."
                    .into(),
            ),
            arguments: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What you need to do" }
                },
                "required": ["query"]
            }),
        },
    }
}

fn as_wire_spec(spec: &ToolSpec) -> WireToolSpec {
    WireToolSpec {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: spec.name.clone(),
            description: Some(spec.description.clone()),
            arguments: spec.input_schema.clone(),
        },
    }
}

fn as_wire_call(call: &crate::generation::ToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone().unwrap_or_else(|| call.name.clone()),
        r#type: "function".into(),
        function: FunctionDefinition {
            description: None,
            name: call.name.clone(),
            arguments: serde_json::from_str(&call.arguments)
                .unwrap_or_else(|_| Value::String(call.arguments.clone())),
        },
    }
}

/// A tool result, tied to the call it answers when the model gave one an id.
///
/// Not every model does — some emit calls with no id at all — so this falls
/// back to an untied result rather than inventing one. A fabricated id is
/// worse than none: a backend would match it to a call that never existed.
fn result_message(call_id: Option<String>, content: String) -> Message {
    match call_id {
        Some(id) => Message::tool_result_for(id, content),
        None => Message::tool_result(content),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steering_queues_in_the_order_it_was_given() {
        let s = Steering::default();
        assert!(!s.is_pending());
        s.follow_up("also check the logs");
        s.interrupt("actually, use staging");
        assert!(s.is_pending());

        let drained = s.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].message, "also check the logs");
        assert!(!drained[0].interrupt);
        assert!(drained[1].interrupt);
        assert!(!s.is_pending(), "draining empties the queue");
    }

    #[test]
    fn an_interrupt_is_visible_before_it_is_drained() {
        // The loop checks this to decide whether to cut the current generation
        // short, which it must do without consuming the message.
        let s = Steering::default();
        s.follow_up("later");
        assert!(!s.wants_interrupt());
        s.interrupt("now");
        assert!(s.wants_interrupt());
        assert!(s.is_pending(), "checking must not consume");
    }

    #[test]
    fn a_steering_handle_shares_the_queue_with_its_clone() {
        // The whole point: the handle goes to another thread while the agent
        // runs here.
        let a = Steering::default();
        let b = a.clone();
        b.follow_up("from elsewhere");
        assert!(a.is_pending());
        assert_eq!(a.drain()[0].message, "from elsewhere");
    }

    #[test]
    fn a_follow_up_does_not_read_as_an_interruption() {
        // Only an interruption abandons the rest of a round's tool calls, so
        // the two must stay distinguishable in the queue.
        let s = Steering::default();
        s.follow_up("and also this");
        assert!(!s.wants_interrupt());
        assert!(!s.drain()[0].interrupt);
    }

    #[test]
    fn default_budgets_are_bounded() {
        // An agent with no configured limit must still stop.
        let b = Budgets::default();
        assert_eq!(b.max_steps, DEFAULT_MAX_STEPS);
        assert!(
            b.max_steps > 0,
            "an unbounded default would never terminate"
        );
    }

    #[test]
    fn approval_is_off_unless_asked_for() {
        assert_eq!(ApprovalMode::default(), ApprovalMode::Auto);
        assert_eq!(Risk::default(), Risk::Safe);
    }
}
