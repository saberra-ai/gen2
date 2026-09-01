//! [`AgentConfig`] — a tool set and policy, reused across runs.
//!
//! An agent's tools are the stable part; the conversation is what changes.
//! Registering them on every run inverts that, and makes it easy to vary the
//! set by accident — which used to be silently ignored and now costs a
//! re-prefill.

use std::sync::Arc;
use std::time::Duration;

use super::agent::{Agent, ApprovalMode};
use super::engine::Engine;
use super::session::Session;
use super::tools::{IntoTool, Tool, ToolLoading, ToolSearch};

/// A reusable agent setup.
///
/// ```no_run
/// # use gen2::{AgentConfig, Engine, Session};
/// # let engine = Engine::load("m.gguf")?;
/// # let weather = gen2::FunctionTool::new("w", "d", |_c, _a: ()| async { unimplemented!() });
/// let researcher = AgentConfig::new().add_tool(weather).max_steps(8);
///
/// let mut session = Session::new();
/// researcher.agent(&engine, &mut session).goal("What is the weather in Paris?")?;
/// researcher.agent(&engine, &mut session).goal("And which city was that?")?;
/// # Ok::<(), gen2::Error>(())
/// ```
///
/// Cloning is cheap — tools are held behind `Arc`.
#[derive(Clone, Default)]
pub struct AgentConfig {
    entries: Vec<(Arc<dyn Tool>, ToolLoading)>,
    search: Option<ToolSearch>,
    max_steps: Option<usize>,
    max_tokens: Option<u32>,
    deadline: Option<Duration>,
    approval: ApprovalMode,
    tool_prompt: Option<String>,
    greedy: bool,
}

impl AgentConfig {
    /// An empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool the model sees from the first turn.
    #[must_use]
    pub fn add_tool(mut self, tool: impl IntoTool) -> Self {
        self.entries.push((tool.into_tool(), ToolLoading::Resident));
        self
    }

    /// Register several resident tools.
    #[must_use]
    pub fn add_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.entries.extend(
            tools
                .into_iter()
                .map(|t| (t.into_tool(), ToolLoading::Resident)),
        );
        self
    }

    /// Register a tool found only by searching.
    #[must_use]
    pub fn defer_tool(mut self, tool: impl IntoTool) -> Self {
        self.entries.push((tool.into_tool(), ToolLoading::Deferred));
        self
    }

    /// Register several deferred tools.
    #[must_use]
    pub fn defer_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.entries.extend(
            tools
                .into_iter()
                .map(|t| (t.into_tool(), ToolLoading::Deferred)),
        );
        self
    }

    /// How deferred tools are found.
    #[must_use]
    pub fn tool_search(mut self, search: ToolSearch) -> Self {
        self.search = Some(search);
        self
    }

    /// Cap the rounds *per run*. Runs do not share a budget — a run is a task,
    /// and "twelve steps for this task" is the useful unit.
    #[must_use]
    pub fn max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Cap tokens per run.
    #[must_use]
    pub fn max_tokens_total(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Wall-clock limit per run.
    #[must_use]
    pub fn deadline(mut self, after: Duration) -> Self {
        self.deadline = Some(after);
        self
    }

    /// Decode deterministically for every run built from this config.
    #[must_use]
    pub fn greedy(mut self) -> Self {
        self.greedy = true;
        self
    }

    /// Ask before running risky tools.
    #[must_use]
    pub fn approval(mut self, mode: ApprovalMode) -> Self {
        self.approval = mode;
        self
    }

    /// How the tool list is introduced to the model.
    #[must_use]
    pub fn tool_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.tool_prompt = prompt.into().into();
        self
    }

    /// How many tools are registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether anything is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build an agent for one run against `session`.
    ///
    /// The approval callback is not part of a configuration — it is a `FnMut`
    /// that usually closes over a UI, so set it per run with
    /// [`Agent::on_approval`].
    pub fn agent<'a>(&self, engine: &'a Engine, session: &'a mut Session) -> Agent<'a> {
        let mut agent = Agent::new(engine, session)
            .with_entries(self.entries.clone())
            .approval(self.approval);
        if let Some(s) = self.search {
            agent = agent.tool_search(s);
        }
        if let Some(n) = self.max_steps {
            agent = agent.max_steps(n);
        }
        if let Some(n) = self.max_tokens {
            agent = agent.max_tokens_total(n);
        }
        if let Some(d) = self.deadline {
            agent = agent.deadline(d);
        }
        if let Some(p) = &self.tool_prompt {
            agent = agent.tool_prompt(p.clone());
        }
        if self.greedy {
            agent = agent.greedy();
        }
        agent
    }

    /// Build an agent that runs off the calling thread.
    pub fn agent_owned(&self, engine: &Arc<Engine>, session: Session) -> super::OwnedAgent {
        let mut agent = engine.agent_owned(session).approval(self.approval);
        for (tool, loading) in &self.entries {
            agent = match loading {
                ToolLoading::Resident => agent.add_tool(Arc::clone(tool)),
                ToolLoading::Deferred => agent.defer_tool(Arc::clone(tool)),
            };
        }
        if let Some(s) = self.search {
            agent = agent.tool_search(s);
        }
        if let Some(n) = self.max_steps {
            agent = agent.max_steps(n);
        }
        if let Some(n) = self.max_tokens {
            agent = agent.max_tokens_total(n);
        }
        if let Some(d) = self.deadline {
            agent = agent.deadline(d);
        }
        if self.greedy {
            agent = agent.greedy();
        }
        agent
    }
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field(
                "tools",
                &self
                    .entries
                    .iter()
                    .map(|(t, l)| format!("{} ({l:?})", t.spec().name))
                    .collect::<Vec<_>>(),
            )
            .field("search", &self.search)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tools::{FunctionTool, ToolOutput};
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}

    fn tool(name: &str) -> FunctionTool<NoArgs> {
        FunctionTool::new(name, format!("does {name}"), |_c, _a: NoArgs| async move {
            Ok(ToolOutput::from("ok"))
        })
    }

    #[test]
    fn a_config_is_reusable_across_runs() {
        // The point: registering once, not per run.
        let cfg = AgentConfig::new().add_tool(tool("a")).add_tool(tool("b"));
        assert_eq!(cfg.len(), 2);
        let again = cfg.clone();
        assert_eq!(again.len(), 2, "cloning keeps the tools");
    }

    #[test]
    fn deferred_and_resident_tools_keep_their_loading() {
        let cfg = AgentConfig::new().add_tool(tool("a")).defer_tool(tool("b"));
        let described = format!("{cfg:?}");
        assert!(described.contains("a (Resident)"), "{described}");
        assert!(described.contains("b (Deferred)"), "{described}");
    }
}
