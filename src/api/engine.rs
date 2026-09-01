//! [`Engine`] — load a model, run turns, shut down cleanly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::controller::{
    ControllerCmd, ControllerConfig, ControllerHandle, start_controller_joinable,
};
use crate::engine::Settings;

use super::chat::Chat;
use super::error::{Error, Result};

/// A running inference engine.
///
/// Owns the controller loop and the backend it holds. Build one with
/// [`Engine::builder`], then start turns with [`Engine::chat`] or
/// [`Engine::prompt`].
///
/// Shutting down is automatic: dropping an `Engine` stops the loop and waits
/// for it to release the backend. Use [`Engine::shutdown`] instead when you
/// want to observe teardown rather than let it happen silently.
pub struct Engine {
    handle: ControllerHandle,
    join: Option<JoinHandle<()>>,
    /// Chat ids already opened, so a second turn on one continues it instead
    /// of restarting and throwing away its warm KV cache.
    started: Mutex<HashSet<String>>,
}

impl Engine {
    /// Configure an engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Load a model from a path with default settings.
    ///
    /// Shorthand for `Engine::builder().model(path).build()`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::builder().model(path).build()
    }

    /// Start a turn in a named conversation.
    ///
    /// Reusing a `chat_id` continues that conversation, keeping its warm KV
    /// cache instead of re-reading the history from scratch.
    pub fn chat(&self, chat_id: impl Into<String>) -> Chat<&Engine> {
        Chat::new(self, chat_id.into())
    }

    /// Start a one-shot turn: a fresh conversation with a single user message.
    pub fn prompt(&self, text: impl Into<String>) -> Chat<&Engine> {
        Chat::new(self, format!("oneshot-{}", uuid::Uuid::new_v4()))
            .user(text)
            .fresh()
    }

    /// As [`Engine::chat`], but the turn owns its engine reference so it can be
    /// [`spawn`](Chat::spawn)ed onto a worker thread.
    pub fn chat_owned(self: &Arc<Self>, chat_id: impl Into<String>) -> Chat<Arc<Engine>> {
        Chat::new(Arc::clone(self), chat_id.into())
    }

    /// As [`Engine::prompt`], but spawnable. See [`Engine::chat_owned`].
    pub fn prompt_owned(self: &Arc<Self>, text: impl Into<String>) -> Chat<Arc<Engine>> {
        Chat::new(
            Arc::clone(self),
            format!("oneshot-{}", uuid::Uuid::new_v4()),
        )
        .user(text)
        .fresh()
    }

    /// Replace the sampling and prompt settings for subsequent turns.
    pub fn apply_settings(&self, settings: Settings) -> Result<()> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::ApplySettings { settings, resp })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)
    }

    /// Embed one or more strings with the loaded embedder.
    pub fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let (resp, rx) = channel();
        self.send(ControllerCmd::GenerateEmbeddings {
            inputs: inputs.to_vec(),
            resp,
        })?;
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(|e| Error::Generation {
                code: "embedding_failed".into(),
                message: e,
            })
    }

    /// Whether a model is currently loaded.
    ///
    /// False if the controller is gone — nothing is loaded in a dead engine.
    pub fn is_model_loaded(&self) -> bool {
        let (resp, rx) = channel();
        if self.send(ControllerCmd::IsModelLoaded { resp }).is_err() {
            return false;
        }
        rx.recv().unwrap_or(false)
    }

    /// Stop a running generation. The stream ends with [`Finish::Stopped`].
    ///
    /// [`Finish::Stopped`]: super::Finish::Stopped
    pub fn stop(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::StopChat {
            chat_id: chat_id.into(),
        })
    }

    /// Pause a running generation, keeping its session warm.
    pub fn pause(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::PauseChat {
            chat_id: chat_id.into(),
        })
    }

    /// Resume a paused generation.
    pub fn resume(&self, chat_id: impl Into<String>) -> Result<()> {
        self.send(ControllerCmd::ResumeChat {
            chat_id: chat_id.into(),
        })
    }

    /// Hint that a model directory is about to be used, so it can be paged in
    /// ahead of the first real request. Fire-and-forget.
    pub fn warm(&self, model_dir: impl Into<PathBuf>) {
        let _ = self.send(ControllerCmd::WarmModel {
            model_dir: model_dir.into(),
        });
    }

    /// Stop the engine and wait for the backend to be released.
    ///
    /// Equivalent to dropping it, except failures surface instead of being
    /// swallowed.
    pub fn shutdown(mut self) -> Result<()> {
        self.stop_and_join()
    }

    /// The lower-level controller handle, for what this facade doesn't cover.
    pub fn controller(&self) -> &ControllerHandle {
        &self.handle
    }

    pub(crate) fn send(&self, cmd: ControllerCmd) -> Result<()> {
        self.handle.send(cmd).map_err(|_| Error::ControllerGone)
    }

    pub(crate) fn event_channel_capacity(&self) -> usize {
        self.handle.config().event_channel_capacity
    }

    /// Record `chat_id` as opened, returning whether this call is the one that
    /// opened it (i.e. the turn should be a `StartChat`).
    pub(crate) fn claim_new_chat(&self, chat_id: &str) -> bool {
        match self.started.lock() {
            Ok(mut started) => started.insert(chat_id.to_string()),
            // A poisoned lock means a previous caller panicked mid-turn. Start
            // the chat rather than continue one whose state we can't vouch for.
            Err(_) => true,
        }
    }

    fn stop_and_join(&mut self) -> Result<()> {
        // Ignore the send error: if the loop is already gone there is nothing
        // to ask, and the join below is what actually matters.
        let _ = self.handle.send(ControllerCmd::Shutdown);
        match self.join.take() {
            Some(join) => join.join().map_err(|_| Error::Generation {
                code: "controller_panicked".into(),
                message: "the controller loop panicked during shutdown".into(),
            }),
            None => Ok(()),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Not just tidiness: the loop holds the backend's native context, and
        // a process that exits while it is still running tears down ggml's
        // statics underneath it and aborts. Joining here is what makes that
        // impossible to get wrong.
        let _ = self.stop_and_join();
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

/// Builds an [`Engine`].
#[derive(Default)]
pub struct EngineBuilder {
    model_path: Option<PathBuf>,
    mmproj_path: Option<PathBuf>,
    settings: Option<Settings>,
    config: Option<ControllerConfig>,
    api_key: Option<String>,
    api_format: Option<String>,
}

impl EngineBuilder {
    /// The model to load — a GGUF file, or an MLX/ONNX model directory. The
    /// backend is chosen from what's there; you never name one.
    pub fn model(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// A multimodal projector, for vision models that need one alongside the
    /// weights.
    pub fn mmproj(mut self, path: impl AsRef<Path>) -> Self {
        self.mmproj_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sampling, stopping, and prompt settings. Defaults are sensible; set
    /// this to override them wholesale.
    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Controller policy: how many chats may run at once, event buffering,
    /// generation timeout.
    pub fn config(mut self, config: ControllerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Serve from an OpenAI-compatible endpoint instead of local weights.
    ///
    /// `model` is the endpoint's model identifier rather than a file path.
    pub fn openai(mut self, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.model_path = Some(PathBuf::from(model.into()));
        self.api_key = Some(api_key.into());
        self.api_format = Some("openai".into());
        self
    }

    /// Serve from an Anthropic-compatible endpoint instead of local weights.
    pub fn anthropic(mut self, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.model_path = Some(PathBuf::from(model.into()));
        self.api_key = Some(api_key.into());
        self.api_format = Some("anthropic".into());
        self
    }

    /// Start the controller and load the model.
    ///
    /// Returns once the weights are resident and the engine is ready to
    /// generate — no separate "is it loaded yet" step.
    pub fn build(self) -> Result<Engine> {
        let model_path = self.model_path.ok_or_else(|| {
            Error::Load("no model given — call .model(path), .openai(..), or .anthropic(..)".into())
        })?;

        let (handle, join) = start_controller_joinable(self.config.unwrap_or_default());
        let engine = Engine {
            handle,
            join: Some(join),
            started: Mutex::new(HashSet::new()),
        };

        let (resp, rx) = channel();
        engine.send(ControllerCmd::LoadModel {
            model_path,
            mmproj_path: self.mmproj_path,
            settings: self.settings.unwrap_or_default(),
            api_key: self.api_key,
            api_format: self.api_format,
            resp,
        })?;

        // On failure `engine` drops here, which stops and joins the loop we
        // just started — no leaked thread behind a failed load.
        rx.recv()
            .map_err(|_| Error::ControllerGone)?
            .map_err(Error::Load)?;

        Ok(engine)
    }
}

impl std::fmt::Debug for EngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("model_path", &self.model_path)
            .field("mmproj_path", &self.mmproj_path)
            .finish_non_exhaustive()
    }
}

/// Build the event channel a turn streams over.
pub(crate) fn event_channel(
    capacity: usize,
) -> (
    std::sync::mpsc::SyncSender<crate::controller::ControllerEvent>,
    std::sync::mpsc::Receiver<crate::controller::ControllerEvent>,
) {
    sync_channel(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An app runs inference off its UI thread, so `Arc<Engine>` has to be
    /// shareable. If this stops compiling, every multi-threaded consumer breaks.
    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
        assert_send_sync::<std::sync::Arc<Engine>>();
    }

    #[test]
    fn building_without_a_model_is_an_error_not_a_panic() {
        let err = Engine::builder().build().unwrap_err();
        assert!(
            err.to_string().contains("no model given"),
            "error should say what is missing, got: {err}"
        );
    }

    #[test]
    fn a_failed_build_starts_no_lingering_controller() {
        // The builder starts the loop before loading. If the load fails, the
        // half-built Engine must drop and join it — otherwise every failed
        // load leaks a thread holding a backend.
        for _ in 0..8 {
            assert!(
                Engine::builder()
                    .model("/nonexistent/model.gguf")
                    .build()
                    .is_err()
            );
        }
    }
}
