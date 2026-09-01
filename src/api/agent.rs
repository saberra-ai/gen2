//! [`Agent`] — a goal, a set of tools, and a loop that runs until it's done.
//!
//! The difference from [`Chat`](super::Chat) with a tool handler is that the
//! agent owns dispatch: it resolves the tool the model named, validates the
//! arguments against that tool's schema, schedules the call, routes failures by
//! whether the model can fix them, and stops when it's making no progress. A
//! caller supplies tools, not a `match`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use super::engine::Engine;
use super::error::{Error, Result};
use super::session::Session;
use super::stream::{Budget, Completion, Finish, Struggle};
use super::tools::{
    IntoTool, Tool, ToolConfigError, ToolContext, ToolError, ToolLoading, ToolOutput, ToolRegistry,
    ToolSearch, ToolSpec,
};
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
        }
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
        self.run_with(Some(text.into()), |_| {})
    }

    /// Run against whatever is already in the session.
    pub fn run(self) -> Result<Completion> {
        self.run_with(None, |_| {})
    }

    /// Run, reporting each step to `observe`.
    pub fn run_streaming(
        self,
        goal: Option<String>,
        observe: impl FnMut(AgentStep<'_>),
    ) -> Result<Completion> {
        self.run_with(goal, observe)
    }

    fn run_with(
        mut self,
        goal: Option<String>,
        mut observe: impl FnMut(AgentStep<'_>),
    ) -> Result<Completion> {
        let mut registry = ToolRegistry::build(std::mem::take(&mut self.entries), self.search)
            .map_err(Error::from)?;

        if let Some(text) = goal {
            self.session.push_user(text);
        }

        let engine = self.engine;
        let budgets = self.budgets;
        let approval = self.approval;
        let mut on_approval = self.on_approval.take();
        let tool_prompt = self.tool_prompt.clone();
        let session: &mut Session = self.session;

        let started = Instant::now();
        let mut totals = Completion::default();
        let mut recent: VecDeque<(String, String)> = VecDeque::new();
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

            let specs = wire_specs(&registry);
            // `stream` rather than `send`: the agent owns its transcript, and
            // `send` would append the assistant text itself — producing a
            // duplicate assistant turn on every round that also calls a tool.
            let turn = engine
                .chat(session)
                .tools(specs, tool_prompt.clone())
                .stream()?
                .complete()?;

            totals.text = turn.text.clone();
            totals.dropped += turn.dropped;
            totals.compacted += turn.compacted;
            totals.tool_rounds = step;
            if let Some(s) = &turn.stats {
                totals.stats = Some(s.clone());
            }

            if let Some(limit) = budgets.max_tokens
                && turn
                    .stats
                    .as_ref()
                    .is_some_and(|s| s.decode_tokens >= limit)
            {
                totals.finish = Finish::OutOfBudget(Budget::Tokens);
                return Ok(totals);
            }

            if turn.tool_calls.is_empty() {
                // The model answered; record it and stop.
                session.push(Message::assistant_structured(turn.text.clone(), None));
                totals.finish = turn.finish;
                return Ok(totals);
            }

            // Record what was asked before what came back, so the transcript
            // reads in the order it happened.
            session.push(Message::assistant_tool_calls(
                turn.tool_calls.iter().map(as_wire_call).collect(),
            ));

            for call in &turn.tool_calls {
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

                observe(AgentStep::Calling {
                    step,
                    tool: &call.name,
                    args: &args,
                });

                let began = Instant::now();
                // Search is handled here rather than in `dispatch` because it
                // mutates the registry (hydration) while dispatch only reads
                // it — one borrow each, taken at different times.
                let outcome = if call.name == SEARCH_TOOL {
                    run_search(&mut registry, &args)
                } else {
                    dispatch(&registry, &ctx, call, &args, approval, on_approval.as_mut())
                };
                let took = began.elapsed();

                observe(AgentStep::Called {
                    step,
                    tool: &call.name,
                    result: &outcome,
                    took,
                });

                match outcome {
                    Ok(out) => {
                        session.push(Message::tool_result(out.to_model_text()));
                    }
                    // A model-fixable failure goes back as a result so it can
                    // correct itself; anything else ends the run, because the
                    // model cannot repair a dead socket and telling it wastes a
                    // round of context.
                    Err(e) if e.is_model_actionable() => {
                        session.push(Message::tool_result(format!("error: {e}")));
                    }
                    Err(e) => {
                        totals.finish = Finish::Stopped;
                        return Err(Error::Generation {
                            code: "tool_failed".into(),
                            message: e.to_string(),
                        });
                    }
                }
            }
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
    registry: &mut ToolRegistry,
    args: &Value,
) -> std::result::Result<ToolOutput, ToolError> {
    let query = args
        .get("query")
        .and_then(|q| q.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("expected a 'query' string".into()))?;

    let found = registry.search(query, 5, None);
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

impl From<ToolConfigError> for Error {
    fn from(e: ToolConfigError) -> Self {
        Error::Load(e.to_string())
    }
}
