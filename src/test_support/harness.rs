//! Driving a real controller over a scripted backend, and checking what came
//! back against the contract every backend owes.

use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::time::Duration;

use super::Script;
use crate::controller::{
    ControllerCmd, ControllerConfig, ControllerEvent, ControllerHandle,
    start_controller_with_engine,
};
use crate::generation::GenSpec;
use crate::types::message::Message;

/// How long a test waits for an event before calling the controller wedged.
///
/// Generous, because CI machines are slow and a flaky timeout is worse than a
/// slow test. Nothing here should ever get close to it.
const PATIENCE: Duration = Duration::from_secs(5);

/// A controller running over a script, plus the script to inspect it with.
///
/// Stops and joins the controller on drop, so a test that panics mid-way does
/// not leave a thread holding the backend.
pub struct Harness {
    pub handle: ControllerHandle,
    pub script: Script,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    /// Start a controller over `script`, with a model already loaded.
    ///
    /// Loading here rather than in each test keeps the interesting part of a
    /// test to the part that is interesting.
    pub fn loaded(script: Script) -> Self {
        let harness = Self::empty(script);
        harness
            .load_model()
            .expect("the scripted backend should load");
        harness
    }

    /// Start a controller over `script` with no model loaded.
    pub fn empty(script: Script) -> Self {
        Self::with_config(script, ControllerConfig::default())
    }

    /// As [`Self::empty`], with controller policy the test chooses — the way
    /// to make eviction happen without opening dozens of chats.
    pub fn with_config(script: Script, config: ControllerConfig) -> Self {
        let (handle, join) =
            start_controller_with_engine(config, script.clone().into_engine_factory());
        Self {
            handle,
            script,
            join: Some(join),
        }
    }

    /// Load a model. The path is never opened: the script answers for it.
    pub fn load_model(&self) -> Result<(), String> {
        let (resp, rx) = std::sync::mpsc::channel();
        self.handle
            .send(ControllerCmd::LoadModel {
                model_path: "/scripted/model.gguf".into(),
                mmproj_path: None,
                settings: Default::default(),
                api_key: None,
                api_format: None,
                resp,
            })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(PATIENCE)
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }

    /// Start a chat and hand back its event stream.
    pub fn start_chat(&self, chat_id: &str) -> Events {
        self.start_chat_with(chat_id, vec![Message::user("hello")])
    }

    /// Start a chat with a transcript the test chooses.
    pub fn start_chat_with(&self, chat_id: &str, messages: Vec<Message>) -> Events {
        let (tx, rx) = sync_channel(self.handle.config().event_channel_capacity);
        self.handle
            .send(ControllerCmd::StartChat {
                chat_id: chat_id.to_string(),
                messages,
                gen_spec: GenSpec::default(),
                thinking: Default::default(),
                model_id: None,
                model_size_bytes: None,
                tools: None,
                tx,
            })
            .expect("controller should accept StartChat");
        Events { rx }
    }

    /// Send any command, for the cases the convenience methods do not cover.
    pub fn send(&self, cmd: ControllerCmd) {
        self.handle.send(cmd).expect("controller should be alive");
    }

    /// How many chat runtimes the controller is holding.
    pub fn resident_chats(&self) -> usize {
        self.handle
            .get_controller_runtime_snapshot()
            .expect("snapshot should be readable")
            .chats
            .len()
    }

    /// Stop the controller and wait for its thread, as `Engine::drop` does.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        let _ = self.handle.send(ControllerCmd::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

/// After a generation ends, how long to keep listening for anything that
/// should not have been sent.
///
/// A finished chat stays resident so it can be continued, so its sender does
/// not close and the stream cannot simply be read to the end. Draining briefly
/// past the terminal event is what turns "a token arrived after Eos" from an
/// invisible bug into a failed assertion.
const GRACE: Duration = Duration::from_millis(250);

/// A chat's event stream, collected with a deadline so a wedged controller
/// fails the test instead of hanging it.
pub struct Events {
    rx: Receiver<ControllerEvent>,
}

impl Events {
    /// Everything the controller sent for this generation.
    ///
    /// Waits for a terminal event, then keeps draining through [`GRACE`] so
    /// illegal trailing events are captured rather than missed.
    pub fn collect(self) -> Vec<ControllerEvent> {
        let mut events = Vec::new();
        let mut ended = false;
        loop {
            let patience = if ended { GRACE } else { PATIENCE };
            match self.rx.recv_timeout(patience) {
                Ok(event) => {
                    ended |= is_terminal(&event);
                    events.push(event);
                }
                Err(RecvTimeoutError::Disconnected) => return events,
                Err(RecvTimeoutError::Timeout) if ended => return events,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("the generation never ended; got {events:?}")
                }
            }
        }
    }

    /// Everything up to and including the terminal event, and no further.
    ///
    /// [`Self::collect`] pays [`GRACE`] per call to catch events that should
    /// not have been sent. That is the right trade for a test asserting the
    /// contract once, and the wrong one for a loop running a thousand
    /// generations — there it is the entire runtime. Use this where the
    /// per-iteration assertion is about residency rather than event ordering.
    pub fn collect_to_end(self) -> Vec<ControllerEvent> {
        let mut events = Vec::new();
        loop {
            match self.rx.recv_timeout(PATIENCE) {
                Ok(event) => {
                    let done = is_terminal(&event);
                    events.push(event);
                    if done {
                        return events;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return events,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("the generation never ended; got {events:?}")
                }
            }
        }
    }

    /// The next event, or a failed test.
    pub fn next(&self) -> ControllerEvent {
        self.rx
            .recv_timeout(PATIENCE)
            .expect("controller should have sent an event")
    }

    /// Wait until at least one token has been generated.
    ///
    /// The synchronisation point for anything that has to happen *during* a
    /// generation: a stop between two tokens, a pause, a model reload.
    pub fn wait_for_first_token(&self) -> String {
        match self.next() {
            ControllerEvent::Token(t) => t,
            other => panic!("expected a token first, got {other:?}"),
        }
    }
}

/// The text of every `Token` event, joined.
pub fn text_of(events: &[ControllerEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            ControllerEvent::Token(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

/// Whether an event ends a generation.
fn is_terminal(event: &ControllerEvent) -> bool {
    matches!(
        event,
        ControllerEvent::Eos | ControllerEvent::Stopped | ControllerEvent::Error { .. }
    )
}

/// Assert the contract every generation owes its caller, whatever produced it.
///
/// Three rules, and they are the ones a consumer writes code against:
///
/// 1. Exactly one terminal event. A caller that renders a spinner until a
///    terminal arrives hangs forever on zero, and double-frees on two.
/// 2. Nothing after it. A token arriving after `Eos` lands in a transcript the
///    caller has already closed.
/// 3. `FinalStats` only after the terminal, and at most once.
///
/// Reusable across backends on purpose: it is the shared contract, so the same
/// assertion should hold for llama.cpp, for an HTTP endpoint, and for a
/// script.
#[track_caller]
pub fn assert_valid_trace(events: &[ControllerEvent]) {
    let terminals: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| is_terminal(e))
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        terminals.len(),
        1,
        "a generation must end exactly once, got {} terminal events in {events:?}",
        terminals.len()
    );

    let terminal = terminals[0];
    for (i, event) in events.iter().enumerate().skip(terminal + 1) {
        assert!(
            matches!(event, ControllerEvent::FinalStats(_)),
            "nothing but FinalStats may follow the terminal event, \
             but {event:?} arrived at {i} after {:?} at {terminal}",
            events[terminal]
        );
    }

    let stats = events
        .iter()
        .filter(|e| matches!(e, ControllerEvent::FinalStats(_)))
        .count();
    assert!(
        stats <= 1,
        "FinalStats must be reported once at most, got {stats}"
    );
    if let Some(at) = events
        .iter()
        .position(|e| matches!(e, ControllerEvent::FinalStats(_)))
    {
        assert!(
            at > terminal,
            "FinalStats at {at} preceded the terminal event at {terminal}"
        );
    }
}
