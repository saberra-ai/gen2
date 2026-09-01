//! A backend that does exactly what a test tells it to, including misbehaving.
//!
//! Most of what is worth proving about this crate is orchestration: that a
//! terminal event arrives exactly once, that a stop between two tokens still
//! terminates cleanly, that a failed `start_session` leaves no runtime behind.
//! None of that is about a neural network, but all of it needed a real GGUF to
//! observe, which made it slow to run and impossible to run in CI.
//!
//! [`FakeBackend`] replaces the model with a script. Tests say what the token
//! stream does — including failing partway, blocking forever, or emitting a
//! token after `Eos` — and then assert on what the controller did about it.
//!
//! # Shape
//!
//! [`Script`] is the handle. It is `Send + Sync` and cheap to clone, because
//! [`Backend`] is deliberately neither: backends hold non-thread-safe FFI
//! state, so the engine is constructed on the controller's own thread. A test
//! builds the script first, hands a factory to
//! [`start_controller_with_engine`](crate::controller::start_controller_with_engine),
//! and keeps its clone to program and inspect the fake from outside.
//!
//! ```ignore
//! let script = Script::new().say(["hel", "lo"]);
//! let handle = start_controller_with_engine(config, script.clone().into_engine_factory());
//! // ... drive the controller ...
//! assert_eq!(script.calls(), ["load_model", "start_session", "pull"]);
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::backend::SessionId;
use crate::backend::{
    Backend, BackendSession, Embeddings, LocalBackend, Multimodal, TokenPullerDyn,
};
use crate::engine::telemetry::HookBus;
use crate::engine::{
    Capabilities, EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings,
};
use crate::generation::{GenSpec, TokenEvent};
use crate::session_rt::SessionSpec;
use crate::types::message::Message;

/// One step of a scripted token stream.
///
/// [`Step::Fail`] and [`Step::Block`] are the reason this is an enum rather
/// than a list of events: a stream that ends cleanly is the easy case, and the
/// interesting contracts are about the streams that do not.
#[derive(Debug, Clone)]
pub enum Step {
    /// Emit one event.
    Emit(TokenEvent),
    /// Fail the pull here. The puller yields the error and then ends.
    Fail(&'static str),
    /// Hold the stream here until the test opens the [`Gate`], so a test can
    /// act while a generation is provably mid-flight.
    Hold(Arc<Gate>),
    /// Mark the session poisoned from here on, as an FFI-level crash would.
    Poison,
}

impl Step {
    /// A tool call, as the cross-backend parser would have extracted it.
    pub fn tool_call(name: &str, arguments: &str) -> Self {
        Self::Emit(TokenEvent::ToolCall(crate::generation::ToolCall {
            id: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
        }))
    }

    /// End of generation.
    pub fn eos() -> Self {
        Self::Emit(TokenEvent::Eos)
    }

    /// A plain text token.
    pub fn token(text: &str) -> Self {
        Self::Emit(TokenEvent::Token(crate::generation::Token {
            id: 0,
            text: text.to_string(),
            logprob: None,
        }))
    }
}

/// A one-sided rendezvous for holding a generation open mid-stream.
///
/// Deliberately not a [`std::sync::Barrier`]. A barrier needs both parties to
/// arrive, so a test that stops the generation before the stream reaches the
/// hold would wait on a puller that has already been dropped, and hang. Here
/// each side waits only for the other's *signal*, both sides give up rather
/// than block forever, and [`Gate::open`] never blocks at all.
#[derive(Debug, Default)]
pub struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    reached: bool,
    open: bool,
}

/// How long either side of a [`Gate`] waits before giving up.
///
/// A generation that never arrives, or a test that never opens the gate, is a
/// bug — but it should fail as a failed assertion, not as a wedged suite.
const GATE_PATIENCE: Duration = Duration::from_secs(5);

impl Gate {
    /// A gate nothing has reached and nobody has opened.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Called from the stream: announce arrival, then wait to be let through.
    fn hold(&self) {
        let mut state = self.state.lock().unwrap();
        state.reached = true;
        self.changed.notify_all();
        let deadline = Instant::now() + GATE_PATIENCE;
        while !state.open {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                // The test went away without opening the gate. Carrying on is
                // the right call: the controller thread must not be held
                // hostage by a test that already failed.
                return;
            };
            let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
            state = next;
            if timeout.timed_out() {
                return;
            }
        }
    }

    /// Wait until the generation has reached the hold. Returns whether it did.
    pub fn wait_until_reached(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let deadline = Instant::now() + GATE_PATIENCE;
        while !state.reached {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
            state = next;
            if timeout.timed_out() {
                return state.reached;
            }
        }
        true
    }

    /// Let the generation continue. Never blocks, and is safe to call whether
    /// or not the stream ever reached the hold.
    pub fn open(&self) {
        let mut state = self.state.lock().unwrap();
        state.open = true;
        self.changed.notify_all();
    }
}

/// A scripted failure.
///
/// A constructor rather than a value, because [`ExecError`] carries an
/// `anyhow::Error` and so is not `Clone`, and a fake that could only fail once
/// would not catch a retry loop.
type ErrorFn = Box<dyn Fn() -> ExecError + Send + Sync>;

/// What the fake was asked to do, in order.
///
/// Recorded as strings rather than a typed enum: assertions read as a
/// transcript, which is how the failures are read too.
pub type CallLog = Vec<String>;

#[derive(Default)]
struct Inner {
    calls: CallLog,
    /// The program every pull runs, when no per-turn script was given.
    program: Vec<Step>,
    /// One program per turn, consumed in order. Takes precedence over
    /// `program`; see [`Script::turns`].
    turns: Vec<Vec<Step>>,
    /// How many pulls have happened, which is the index into `turns`.
    turn: usize,
    load_result: Option<ErrorFn>,
    start_session_result: Option<ErrorFn>,
    pull_result: Option<ErrorFn>,
    capabilities: Capabilities,
    embeddings: Option<Vec<Vec<f32>>>,
    n_ctx: usize,
}

/// A programmable, inspectable handle to a [`FakeBackend`].
///
/// Clone it freely: every clone shares one script and one call log.
#[derive(Clone)]
pub struct Script {
    inner: Arc<Mutex<Inner>>,
    loaded: Arc<AtomicBool>,
    embedder_loaded: Arc<AtomicBool>,
    next_session_id: Arc<AtomicU64>,
    live_sessions: Arc<AtomicU64>,
}

impl Default for Script {
    fn default() -> Self {
        Self::new()
    }
}

impl Script {
    /// A backend that loads, starts sessions, and immediately reaches `Eos`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                capabilities: Capabilities::TEXT,
                n_ctx: 4096,
                ..Default::default()
            })),
            loaded: Arc::new(AtomicBool::new(false)),
            embedder_loaded: Arc::new(AtomicBool::new(false)),
            next_session_id: Arc::new(AtomicU64::new(1)),
            live_sessions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Emit these tokens, then `Eos`. The common case.
    pub fn say<'a>(self, tokens: impl IntoIterator<Item = &'a str>) -> Self {
        let mut program: Vec<Step> = tokens.into_iter().map(Step::token).collect();
        program.push(Step::Emit(TokenEvent::Eos));
        self.program(program)
    }

    /// Run this exact program, with nothing appended. Use it to script a
    /// stream that ends badly, or does not end at all.
    pub fn program(self, steps: impl IntoIterator<Item = Step>) -> Self {
        self.inner.lock().unwrap().program = steps.into_iter().collect();
        self
    }

    /// Script each turn separately, in order.
    ///
    /// An agent loop pulls once per round, so a single fixed program would
    /// replay the same tool call forever. This is how a run that calls a tool
    /// and then answers is expressed. Once the turns run out, every further
    /// pull is a bare `Eos`, which ends the loop rather than hanging it.
    pub fn turns(self, turns: impl IntoIterator<Item = Vec<Step>>) -> Self {
        self.inner.lock().unwrap().turns = turns.into_iter().collect();
        self
    }

    /// Fail every `load_model`.
    pub fn failing_load(self, error: impl Fn() -> ExecError + Send + Sync + 'static) -> Self {
        self.inner.lock().unwrap().load_result = Some(Box::new(error));
        self
    }

    /// Fail every `start_session`.
    pub fn failing_start_session(
        self,
        error: impl Fn() -> ExecError + Send + Sync + 'static,
    ) -> Self {
        self.inner.lock().unwrap().start_session_result = Some(Box::new(error));
        self
    }

    /// Fail every `pull`, before any event is produced.
    pub fn failing_pull(self, error: impl Fn() -> ExecError + Send + Sync + 'static) -> Self {
        self.inner.lock().unwrap().pull_result = Some(Box::new(error));
        self
    }

    /// Report these capabilities.
    pub fn capable_of(self, capabilities: Capabilities) -> Self {
        self.inner.lock().unwrap().capabilities = capabilities;
        self
    }

    /// Report this context window, which is what truncation sizes against.
    pub fn context(self, n_ctx: usize) -> Self {
        self.inner.lock().unwrap().n_ctx = n_ctx;
        self
    }

    /// Answer `generate_embeddings` with these vectors, and advertise the
    /// embeddings capability.
    pub fn embedding(self, vectors: Vec<Vec<f32>>) -> Self {
        self.inner.lock().unwrap().embeddings = Some(vectors);
        self
    }

    /// Everything the backend was asked to do, in order.
    pub fn calls(&self) -> CallLog {
        self.inner.lock().unwrap().calls.clone()
    }

    /// How many times the backend was asked to do this.
    pub fn count(&self, call: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .calls
            .iter()
            .filter(|c| c.as_str() == call)
            .count()
    }

    /// Sessions started but not yet ended. A controller that leaks a runtime
    /// leaves this above zero.
    pub fn live_sessions(&self) -> u64 {
        self.live_sessions.load(Ordering::SeqCst)
    }

    /// Whether a model is loaded right now.
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    /// A factory that builds the engine on the controller's own thread.
    ///
    /// [`Backend`] is not `Send`, so the fake cannot be constructed here and
    /// moved there. The script can: it is the shared, `Send + Sync` half.
    pub fn into_engine_factory(self) -> Box<dyn FnOnce() -> crate::backend::Engine + Send> {
        Box::new(move || crate::backend::Engine::Fake(FakeBackend { script: self }))
    }

    fn record(&self, call: &str) {
        self.inner.lock().unwrap().calls.push(call.to_string());
    }
}

/// The [`Backend`] half of the fake. Build one through [`Script`].
pub struct FakeBackend {
    script: Script,
}

impl std::fmt::Debug for FakeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeBackend")
            .field("script", &self.script)
            .finish()
    }
}

impl std::fmt::Debug for Script {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Script")
            .field("calls", &self.calls().len())
            .field("loaded", &self.is_loaded())
            .finish_non_exhaustive()
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn load_model(&self, _req: LoadRequest) -> Result<(), ExecError> {
        self.script.record("load_model");
        if let Some(fail) = self.script.inner.lock().unwrap().load_result.as_ref() {
            return Err(fail());
        }
        self.script.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn reload_model(&self) -> Result<(), ExecError> {
        self.script.record("reload_model");
        if !self.script.loaded.load(Ordering::SeqCst) {
            return Err(ExecError::ModelNotLoaded);
        }
        Ok(())
    }

    fn unload_model(&self) {
        self.script.record("unload_model");
        self.script.loaded.store(false, Ordering::SeqCst);
    }

    fn is_model_loaded(&self) -> bool {
        self.script.loaded.load(Ordering::SeqCst)
    }

    fn upload_settings(&self, _settings: Settings) -> Result<(), ExecError> {
        self.script.record("upload_settings");
        Ok(())
    }

    fn settings(&self) -> Arc<Settings> {
        Arc::new(Settings::default())
    }

    fn settings_version(&self) -> u64 {
        0
    }

    fn hooks(&self) -> Arc<HookBus> {
        Arc::new(HookBus::default())
    }

    fn capabilities(&self) -> Capabilities {
        self.script.inner.lock().unwrap().capabilities
    }

    fn stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }

    fn first_token_tier(&self) -> crate::backend::caps::LatencyTier {
        crate::backend::caps::LatencyTier::Fast
    }

    fn start_session(&self, _spec: SessionSpec) -> Result<Arc<dyn BackendSession>, ExecError> {
        self.script.record("start_session");
        if let Some(fail) = self
            .script
            .inner
            .lock()
            .unwrap()
            .start_session_result
            .as_ref()
        {
            return Err(fail());
        }
        if !self.script.loaded.load(Ordering::SeqCst) {
            return Err(ExecError::ModelNotLoaded);
        }
        self.script.live_sessions.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(FakeSession {
            id: self.script.next_session_id.fetch_add(1, Ordering::SeqCst),
            script: self.script.clone(),
            poisoned: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn end_session(&self, _id: SessionId) -> Result<(), ExecError> {
        self.script.record("end_session");
        // Saturating: a controller that ends the same session twice is a bug
        // worth seeing as a count, not as a panic inside the fake.
        let _ = self
            .script
            .live_sessions
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(1))
            });
        Ok(())
    }

    fn as_embeddings(&self) -> Option<&dyn Embeddings> {
        self.script
            .inner
            .lock()
            .unwrap()
            .embeddings
            .is_some()
            .then_some(self as &dyn Embeddings)
    }

    fn as_multimodal(&self) -> Option<&dyn Multimodal> {
        Some(self)
    }
}

impl LocalBackend for FakeBackend {
    fn n_ctx(&self) -> usize {
        self.script.inner.lock().unwrap().n_ctx
    }
}

impl Embeddings for FakeBackend {
    fn load_embedder(&self, _req: EmbedLoadRequest) -> Result<(), ExecError> {
        self.script.record("load_embedder");
        self.script.embedder_loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_embedder_loaded(&self) -> bool {
        self.script.embedder_loaded.load(Ordering::SeqCst)
    }

    fn generate_embeddings(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ExecError> {
        self.script.record("generate_embeddings");
        if !self.script.embedder_loaded.load(Ordering::SeqCst) {
            return Err(ExecError::EmbedderNotLoaded);
        }
        let scripted = self.script.inner.lock().unwrap().embeddings.clone();
        let scripted = scripted.ok_or(ExecError::EmbedderNotLoaded)?;
        // One vector per input, as the contract requires, cycling the script
        // so a test does not have to supply as many vectors as it has inputs.
        Ok(inputs
            .iter()
            .enumerate()
            .map(|(i, _)| scripted[i % scripted.len()].clone())
            .collect())
    }

    fn unload_embedder(&self) {
        self.script.record("unload_embedder");
        self.script.embedder_loaded.store(false, Ordering::SeqCst);
    }
}

impl Multimodal for FakeBackend {
    fn supports_images(&self) -> bool {
        self.capabilities().contains(Capabilities::IMAGES)
    }

    fn supports_audio(&self) -> bool {
        self.capabilities().contains(Capabilities::AUDIO)
    }
}

#[derive(Debug)]
struct FakeSession {
    id: SessionId,
    script: Script,
    /// Shared with the puller: a scripted crash sets it there, and the
    /// controller reads it here, exactly as a real FFI poisoning propagates.
    poisoned: Arc<AtomicBool>,
}

impl BackendSession for FakeSession {
    fn id(&self) -> SessionId {
        self.id
    }

    fn pause(&self) {
        self.script.record("pause");
    }

    fn resume(&self) {
        self.script.record("resume");
    }

    fn stop(&self) {
        self.script.record("stop");
    }

    fn pull(&self, _spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError> {
        self.script.record("pull");
        if let Some(fail) = self.script.inner.lock().unwrap().pull_result.as_ref() {
            return Err(fail());
        }
        let program = {
            let mut inner = self.script.inner.lock().unwrap();
            if inner.turns.is_empty() {
                inner.program.clone()
            } else {
                let at = inner.turn;
                inner.turn += 1;
                // Past the end of the script, end the turn rather than
                // replaying it: an agent loop that keeps pulling would
                // otherwise never terminate.
                inner
                    .turns
                    .get(at)
                    .cloned()
                    .unwrap_or_else(|| vec![Step::eos()])
            }
        };
        Ok(Box::new(FakePuller {
            program,
            at: 0,
            done: false,
            poisoned: Arc::clone(&self.poisoned),
        }))
    }

    fn append_messages(&self, new_messages: Vec<Message>) -> Result<usize, ExecError> {
        self.script.record("append_messages");
        Ok(new_messages.len())
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }
}

struct FakePuller {
    program: Vec<Step>,
    at: usize,
    poisoned: Arc<AtomicBool>,
    /// Set once the puller has yielded `None`. A puller that resumes after
    /// ending would let a test pass on a stream no real backend can produce.
    done: bool,
}

impl TokenPullerDyn for FakePuller {
    fn next_event(&mut self) -> Option<Result<TokenEvent, ExecError>> {
        if self.done {
            return None;
        }
        loop {
            let Some(step) = self.program.get(self.at).cloned() else {
                self.done = true;
                return None;
            };
            self.at += 1;
            match step {
                Step::Emit(event) => return Some(Ok(event)),
                Step::Fail(message) => {
                    self.done = true;
                    return Some(Err(ExecError::Generation(message.to_string())));
                }
                Step::Hold(gate) => {
                    gate.hold();
                }
                Step::Poison => {
                    self.done = true;
                    self.poisoned.store(true, Ordering::SeqCst);
                    return Some(Err(ExecError::SessionPoisoned("scripted".into())));
                }
            }
        }
    }
}
