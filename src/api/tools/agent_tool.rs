//! [`AgentTool`] — a whole agent, callable as one tool.
//!
//! The cheapest useful form of delegation: a sub-agent is not a new concept,
//! it's a `Tool` whose implementation happens to run a nested loop. The parent
//! sees one call and one result, so a sub-task that would have filled the
//! parent's context with intermediate reading costs it a single message.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use super::{ExecutionPolicy, IntoTool, Tool, ToolContext, ToolError, ToolOutput, ToolSpec};
use crate::api::engine::Engine;
use crate::api::session::Session;
use crate::api::tools::ToolSearch;

/// The one argument a sub-agent takes: what to go and do.
#[derive(Deserialize, JsonSchema)]
struct Task {
    /// What you want the sub-agent to accomplish, in a sentence or two.
    task: String,
}

/// An agent exposed to a parent agent as a tool.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use pio_gen2::{AgentTool, Engine, Session};
/// # let engine = Arc::new(Engine::load("m.gguf")?);
/// # let mut session = Session::new();
/// # let research_tools: Vec<Arc<dyn pio_gen2::Tool>> = vec![];
/// let researcher = AgentTool::new(
///     "researcher",
///     "Investigates a question and reports back a short answer",
///     Arc::clone(&engine),
/// )
/// .tools(research_tools)
/// .max_steps(5);
///
/// engine.agent(&mut session).add_tool(researcher).goal("…")?;
/// # Ok::<(), pio_gen2::Error>(())
/// ```
pub struct AgentTool {
    spec: ToolSpec,
    engine: Arc<Engine>,
    tools: Vec<Arc<dyn Tool>>,
    deferred: Vec<Arc<dyn Tool>>,
    search: Option<ToolSearch>,
    max_steps: usize,
    system: Option<String>,
}

impl AgentTool {
    /// Name and describe the sub-agent. The description is what the parent
    /// model reads when deciding whether to delegate, so write it as a
    /// capability ("investigates a question"), not an implementation note.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        engine: Arc<Engine>,
    ) -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(Task))
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
        Self {
            spec: ToolSpec::new(name, description, schema),
            engine,
            tools: Vec::new(),
            deferred: Vec::new(),
            search: None,
            max_steps: 6,
            system: None,
        }
    }

    /// Tools the sub-agent may call. Deliberately its own set — narrowing what
    /// a delegate can reach is most of the value of delegating.
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.tools
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// Tools the sub-agent finds by searching.
    #[must_use]
    pub fn deferred_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.deferred
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// How the sub-agent finds its deferred tools.
    #[must_use]
    pub fn tool_search(mut self, search: ToolSearch) -> Self {
        self.search = Some(search);
        self
    }

    /// Cap the sub-agent's rounds. Kept low by default: a delegate that needs
    /// many steps probably wanted to be the main loop.
    #[must_use]
    pub fn max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// A system prompt scoping the sub-agent's behaviour.
    #[must_use]
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(
        &self,
        _ctx: &ToolContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let task: Task =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let engine = Arc::clone(&self.engine);
        let tools = self.tools.clone();
        let deferred = self.deferred.clone();
        let search = self.search;
        let max_steps = self.max_steps;
        let system = self.system.clone();

        // The nested run is synchronous, so it goes on a blocking thread rather
        // than holding an async worker for the length of a whole sub-agent.
        tokio::task::spawn_blocking(move || {
            let mut session = match system {
                Some(p) => Session::new().with_system(p),
                None => Session::new(),
            };
            let mut agent = engine
                .agent(&mut session)
                .add_tools(tools)
                .defer_tools(deferred)
                .max_steps(max_steps);
            if let Some(s) = search {
                agent = agent.tool_search(s);
            }
            agent
                .goal(task.task)
                .map(|done| ToolOutput::Text(done.text))
                .map_err(|e| ToolError::Failed(e.to_string()))
        })
        .await
        .map_err(|_| ToolError::Failed("the sub-agent panicked".into()))?
    }

    fn execution_policy(&self) -> ExecutionPolicy {
        // A sub-agent *is* inference. Running one beside the parent's
        // generation puts two decoders on the same accelerator.
        ExecutionPolicy::gpu_bound()
    }
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.spec.name)
            .field("tools", &self.tools.len())
            .field("max_steps", &self.max_steps)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subagent_takes_one_task_string() {
        // The parent model has to be able to call this without guessing a
        // schema, so it is deliberately the simplest possible one.
        let schema = serde_json::to_value(schemars::schema_for!(Task)).unwrap();
        assert!(schema["properties"].get("task").is_some(), "{schema}");
        assert_eq!(schema["required"][0], "task");
        assert!(
            schema["properties"]["task"]["description"]
                .as_str()
                .is_some_and(|d| d.contains("sub-agent")),
            "the doc comment reaches the model"
        );
    }
}
