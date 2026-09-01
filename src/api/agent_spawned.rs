//! Running an agent off the calling thread.
//!
//! The shape a UI needs: the caller never blocks, tool activity and generated
//! text arrive as they happen, and steering works — including cutting a
//! generation short, which a borrowed [`Agent`](super::Agent) cannot do because
//! it has no owned engine to ask.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use super::agent::{Agent, AgentStep, ApprovalMode, Decision, Steering};
use super::engine::Engine;
use super::error::Result;
use super::session::Session;
use super::spawned::Update;
use super::tools::{IntoTool, Tool, ToolSearch};

type ApprovalFn =
    Box<dyn FnMut(&str, &serde_json::Value, &super::tools::ToolSpec) -> Decision + Send>;

/// An agent configured to run on a worker thread.
///
/// Built by [`Engine::agent_owned`](crate::Engine::agent_owned). Same surface as
/// [`Agent`](super::Agent); the difference is that it owns its engine and
/// session, which is what lets the run outlive the calling scope and lets
/// steering reach the engine.
#[must_use = "an OwnedAgent does nothing until .spawn() is called"]
pub struct OwnedAgent {
    engine: Arc<Engine>,
    session: Session,
    resident: Vec<Arc<dyn Tool>>,
    deferred: Vec<Arc<dyn Tool>>,
    search: Option<ToolSearch>,
    max_steps: Option<usize>,
    max_tokens: Option<u32>,
    deadline: Option<Duration>,
    approval: ApprovalMode,
    on_approval: Option<ApprovalFn>,
    goal: Option<String>,
}

impl OwnedAgent {
    pub(crate) fn new(engine: Arc<Engine>, session: Session) -> Self {
        Self {
            engine,
            session,
            resident: Vec::new(),
            deferred: Vec::new(),
            search: None,
            max_steps: None,
            max_tokens: None,
            deadline: None,
            approval: ApprovalMode::Auto,
            on_approval: None,
            goal: None,
        }
    }

    /// Register a resident tool.
    pub fn add_tool(mut self, tool: impl IntoTool) -> Self {
        self.resident.push(tool.into_tool());
        self
    }

    /// Register several resident tools.
    pub fn add_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.resident
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// Register a tool found only by searching.
    pub fn defer_tool(mut self, tool: impl IntoTool) -> Self {
        self.deferred.push(tool.into_tool());
        self
    }

    /// Register several deferred tools.
    pub fn defer_tools(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.deferred
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// How deferred tools are found.
    pub fn tool_search(mut self, search: ToolSearch) -> Self {
        self.search = Some(search);
        self
    }

    /// Cap the rounds.
    pub fn max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Cap total tokens generated.
    pub fn max_tokens_total(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Stop after this much wall-clock.
    pub fn deadline(mut self, after: Duration) -> Self {
        self.deadline = Some(after);
        self
    }

    /// Ask before running risky tools.
    pub fn approval(mut self, mode: ApprovalMode) -> Self {
        self.approval = mode;
        self
    }

    /// Decide whether a risky call runs.
    pub fn on_approval(
        mut self,
        f: impl FnMut(&str, &serde_json::Value, &super::tools::ToolSpec) -> Decision + Send + 'static,
    ) -> Self {
        self.on_approval = Some(Box::new(f));
        self
    }

    /// What to do.
    pub fn goal(mut self, text: impl Into<String>) -> Self {
        self.goal = Some(text.into());
        self
    }

    /// Run on a worker thread, streaming [`Update`]s back.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use gen2::{Engine, Session, Update};
    /// # let engine = Arc::new(Engine::load("m.gguf")?);
    /// let run = engine.agent_owned(Session::new())
    ///     .goal("Summarise the repository")
    ///     .spawn();
    ///
    /// let steering = run.steering();   // move this wherever the user is
    ///
    /// for update in run {
    ///     match update {
    ///         Update::Delta(t) => print!("{t}"),
    ///         Update::ToolCall { tool, .. } => println!("[{tool}]"),
    ///         Update::Done { session, .. } => drop(session),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn spawn(self) -> AgentRun {
        let (tx, rx) = channel();
        let engine = Arc::clone(&self.engine);
        let session_id = self.session.id().to_string();
        // The handle carries the engine, so `interrupt` can stop a generation
        // rather than only queue a message.
        let steering = Steering::default().with_engine(Arc::clone(&engine), session_id.clone());
        let steer_for_loop = steering.clone();

        let join = std::thread::spawn(move || {
            let Self {
                engine,
                mut session,
                resident,
                deferred,
                search,
                max_steps,
                max_tokens,
                deadline,
                approval,
                on_approval,
                goal,
            } = self;

            let mut agent = Agent::new_steered(&engine, &mut session, steer_for_loop)
                .add_tools(resident)
                .defer_tools(deferred)
                .approval(approval);
            if let Some(s) = search {
                agent = agent.tool_search(s);
            }
            if let Some(n) = max_steps {
                agent = agent.max_steps(n);
            }
            if let Some(n) = max_tokens {
                agent = agent.max_tokens_total(n);
            }
            if let Some(d) = deadline {
                agent = agent.deadline(d);
            }
            if let Some(f) = on_approval {
                agent = agent.with_approval_fn(f);
            }

            let deltas = tx.clone();
            let steps = tx.clone();
            let result = agent.run_with(
                goal,
                |fragment| {
                    let _ = deltas.send(Update::Delta(fragment.to_string()));
                },
                |step| match step {
                    AgentStep::Calling { tool, args, .. } => {
                        let _ = steps.send(Update::ToolCall {
                            id: tool.to_string(),
                            tool: tool.to_string(),
                            args: args.clone(),
                        });
                    }
                    AgentStep::Called {
                        tool, result, took, ..
                    } => {
                        let _ = steps.send(Update::ToolResult {
                            id: tool.to_string(),
                            tool: tool.to_string(),
                            result: match result {
                                Ok(o) => Ok(o.clone()),
                                // ToolError isn't Clone (it carries sources), so
                                // the streamed copy keeps the message and the
                                // actionable/unactionable distinction.
                                Err(e) => Err(super::tools::ToolError::Failed(e.to_string())),
                            },
                            took,
                        });
                    }
                },
            );

            let _ = match result {
                Ok(completion) => tx.send(Update::Done {
                    completion,
                    session,
                }),
                Err(error) => tx.send(Update::Failed { error, session }),
            };
        });

        AgentRun {
            rx,
            engine,
            session_id,
            steering,
            join: Some(join),
        }
    }
}

impl std::fmt::Debug for OwnedAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedAgent")
            .field("session", &self.session.id())
            .field("resident", &self.resident.len())
            .field("deferred", &self.deferred.len())
            .finish_non_exhaustive()
    }
}

/// An agent running on a worker thread.
///
/// Iterate for [`Update`]s. The channel closes when the run ends.
pub struct AgentRun {
    rx: Receiver<Update>,
    engine: Arc<Engine>,
    session_id: String,
    steering: Steering,
    join: Option<JoinHandle<()>>,
}

impl AgentRun {
    /// A handle for injecting messages while this runs.
    ///
    /// Its `interrupt` can cut a generation short, because the run owns an
    /// engine to ask.
    pub fn steering(&self) -> Steering {
        self.steering.clone()
    }

    /// Stop the run outright.
    pub fn cancel(&self) -> Result<()> {
        self.engine.stop(self.session_id.clone())
    }

    /// The conversation this run belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl Iterator for AgentRun {
    type Item = Update;

    fn next(&mut self) -> Option<Update> {
        self.rx.recv().ok()
    }
}

impl std::iter::FusedIterator for AgentRun {}

impl Drop for AgentRun {
    fn drop(&mut self) {
        // Join so the worker cannot outlive the handle and send into a dropped
        // receiver. It exits when the run ends; `cancel` first to get out sooner.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl std::fmt::Debug for AgentRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRun")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}
