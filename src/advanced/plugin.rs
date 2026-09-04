//! Bring your own backend.
//!
//! The crate routes a model path to a backend by what is compiled in. A
//! consumer with a backend the crate cannot carry — one that depends on a
//! crate with no registry release, say, or that links a library the build
//! cannot assume — registers it here instead, and it joins the same routing
//! ahead of every built-in rule.
//!
//! Three steps:
//!
//! 1. Implement [`LocalBackend`] (and so [`Backend`]) on your type. Sessions
//!    implement [`BackendSession`]; a generation is a [`TokenPullerDyn`]
//!    yielding [`TokenEvent`]s. Everything those signatures name is
//!    re-exported from this module, so `use gen2::advanced::plugin::*` is
//!    enough to write one.
//! 2. Build a [`BackendPlugin`]: a `name`, a `claims` predicate over the
//!    model path, and a `make` factory that constructs your backend.
//! 3. Register it: `Engine::builder().model(path).backend(plugin).build()`.
//!    A path your plugin claims lands on your backend; every other path is
//!    routed as before.
//!
//! ```no_run
//! use std::path::Path;
//! use gen2::advanced::BackendPlugin;
//! # fn my_backend() -> Box<dyn gen2::advanced::plugin::LocalBackend> { unimplemented!() }
//!
//! fn claims(path: &Path) -> bool {
//!     path.extension().is_some_and(|e| e == "mybundle")
//! }
//!
//! let plugin = BackendPlugin {
//!     name: "mine",
//!     claims,
//!     make: Box::new(my_backend),
//! };
//! let engine = gen2::Engine::builder()
//!     .model("/models/weights.mybundle")
//!     .backend(plugin)
//!     .build()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! # Why a factory
//!
//! [`Backend`] is deliberately not `Send`: backends hold native state that
//! belongs to one thread, and the controller loop is that thread. A plugin
//! therefore carries a `Send + Sync` factory rather than an instance; the
//! controller calls it, on its own thread, the first time a claimed path is
//! loaded. Loading the same plugin's paths again reuses that instance; a path
//! it declines switches the controller away from it.
//!
//! # What a backend receives
//!
//! The same things the in-tree backends do, and nothing routed around them:
//! a [`LoadRequest`] at load, [`Settings`] through `upload_settings`, a
//! [`SessionSpec`] per session, and a [`GenSpec`] per generation. Useful
//! pieces the in-tree backends share are here too — [`HfTokenizer`] and
//! [`load_chat_template`] to read a Hugging Face model directory,
//! [`ChatTemplate`] to render one, [`GrammarMatcher`] for constrained
//! decoding, [`HookBus`] to report load and decode events.
//!
//! # Building with no built-in backend
//!
//! A consumer whose only backend is a plugin compiles the crate with
//! `default-features = false` and no `backend-*` feature at all. That build
//! starts with no backend; a load whose path no plugin claims fails with an
//! error saying so, rather than at compile time.
//!
//! [`TokenEvent`]: crate::advanced::plugin::TokenEvent

use std::fmt;
use std::path::Path;

// ── The contract ────────────────────────────────────────────────────────────

/// A session's identity, as the controller refers to it.
pub use crate::backend::SessionId;
/// How fast a backend reaches first token — reported by
/// [`Backend::first_token_tier`].
pub use crate::backend::caps::LatencyTier;
/// The backend contract, its session, and its token stream.
pub use crate::backend::traits::{
    Backend, BackendSession, Embeddings, KvSnapshot, LocalBackend, Multimodal, TokenPullerDyn,
};

// ── What a backend is given ─────────────────────────────────────────────────

/// What a load, a settings update, and an embedder load carry.
pub use crate::engine::{
    Capabilities, ChatTemplateSpec, CtxParamsInput, EmbedLoadRequest, ExecError, LoadRequest,
    ModelParamsInput, SamplingSettings, Settings,
};
/// What one generation asks for.
pub use crate::generation::GenSpec;
/// Session-level inputs a [`SessionSpec`] can carry.
pub use crate::kv::{KvLoadReport, KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot as KvSnapshotBlob};
pub use crate::media::Attachment;
/// Everything a session starts from.
pub use crate::session_rt::SessionSpec;
pub use crate::types::Persona;
/// Conversation wire types a session receives and a template renders.
pub use crate::types::message::{
    Message, MessageBody, MessageChunk, MessageContent, TokenizerConfigToken, ToolSpec,
};

// ── What a backend reports ──────────────────────────────────────────────────

/// Tool-call parse outcomes, carried on [`HookEvent::ToolCallOutcomes`].
pub use crate::backend::common::tool_calls::ToolCallTally;
/// Facts about a loaded model, carried on
/// [`HookEvent::EngineLoadOk`].
pub use crate::bundle::ModelMeta;
/// The event bus a backend reports load and decode progress on.
pub use crate::engine::telemetry::{HookBus, HookEvent, HookListener};
/// What a generation yields, event by event.
pub use crate::generation::{MediaBoundary, Token, TokenEvent, ToolCall};
/// Throughput and timing for a generation.
pub use crate::types::ExecutionStats;

// ── Shared machinery ────────────────────────────────────────────────────────

/// A model's Jinja chat template, rendered the way the in-tree backends do.
pub use crate::backend::common::chat_template::ChatTemplate;
/// Grammar-constrained decoding: the spec a caller sets, and the matcher
/// that turns it into a per-step logit mask.
pub use crate::backend::common::grammar::{GrammarMatcher, GrammarSpec, GrammarVocab};
/// The raw template string from a model directory, if it ships one.
pub use crate::backend::common::load_chat_template;
/// A Hugging Face `tokenizer.json`, read from a model directory.
pub use crate::backend::common::tokenizer::HfTokenizer;
/// Whether any message carries an image — for a text-only backend to refuse
/// up front rather than silently flatten.
pub use crate::session_rt::media_util::messages_have_images;

// ── The plugin ──────────────────────────────────────────────────────────────

/// An out-of-tree backend, registered with
/// [`EngineBuilder::backend`](crate::EngineBuilder::backend).
///
/// Asked before every built-in routing rule, in registration order: the
/// first plugin whose [`claims`](Self::claims) accepts the model path wins.
pub struct BackendPlugin {
    /// What [`Backend::backend_name`] on the built backend returns. The
    /// controller uses it to tell one plugin's instance from another's, so
    /// two registered plugins must not share it.
    pub name: &'static str,
    /// Whether this backend takes the model at this path. Runs before
    /// anything reads the filesystem, so it may be as cheap as an extension
    /// check — and must not assume the path exists.
    pub claims: fn(&Path) -> bool,
    /// Constructs the backend. Called on the controller thread, at most once
    /// per stretch of loads the plugin claims; see the module docs.
    pub make: Box<dyn Fn() -> Box<dyn LocalBackend> + Send + Sync>,
}

impl fmt::Debug for BackendPlugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendPlugin")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::Script;
    use crate::{ControllerCmd, Engine};

    fn claims_fake(path: &Path) -> bool {
        path.extension().is_some_and(|e| e == "fake")
    }

    /// The whole seam, through the public API: a plugin over the scripted
    /// backend is registered with the builder, a path only it claims is
    /// loaded, and the controller both reports it as the active backend and
    /// streams what it scripted.
    ///
    /// The path does not exist, on purpose: nothing before the plugin may
    /// read the filesystem, or a consumer's format could be vetoed by a
    /// built-in rule before the plugin is asked.
    #[test]
    fn a_registered_plugin_is_routed_to_through_the_public_api() {
        let script = Script::new().say(["routed", " through", " the plugin"]);
        let kept = script.clone();
        let plugin = BackendPlugin {
            name: "fake",
            claims: claims_fake,
            make: Box::new(move || Box::new(script.backend())),
        };
        assert!(format!("{plugin:?}").contains("fake"));

        let engine = Engine::builder()
            .model("/nowhere/model.fake")
            .backend(plugin)
            .build()
            .expect("a claimed path loads through the plugin");

        let (resp, rx) = std::sync::mpsc::channel();
        engine
            .controller()
            .send(ControllerCmd::GetActiveBackendName { resp })
            .unwrap();
        assert_eq!(rx.recv().unwrap(), "fake");
        assert_eq!(engine.capabilities(), Capabilities::TEXT);

        let text = engine.infer("anything").text().unwrap();
        assert_eq!(text, "routed through the plugin");
        assert!(kept.calls().contains(&"load_model".to_string()));
        assert!(kept.calls().contains(&"pull".to_string()));
    }

    /// The factory runs on the controller's thread, not the caller's: the
    /// backend is not `Send`, and the seam must never ask it to be.
    #[test]
    fn the_factory_runs_on_the_controller_thread() {
        let script = Script::new();
        let caller = std::thread::current().id();
        let built_on = std::sync::Arc::new(std::sync::Mutex::new(None));
        let seen = built_on.clone();
        let plugin = BackendPlugin {
            name: "fake",
            claims: claims_fake,
            make: Box::new(move || {
                *seen.lock().unwrap() = Some(std::thread::current().id());
                Box::new(script.backend())
            }),
        };
        let _engine = Engine::builder()
            .model("/nowhere/model.fake")
            .backend(plugin)
            .build()
            .unwrap();
        let built_on = built_on.lock().unwrap().expect("the factory ran");
        assert_ne!(built_on, caller);
    }

    /// A plugin that declines the path changes nothing about how the path
    /// is routed, and says so when nothing else can take it.
    #[test]
    fn a_declining_plugin_leaves_routing_alone() {
        let script = Script::new();
        let plugin = BackendPlugin {
            name: "fake",
            claims: claims_fake,
            make: Box::new(move || Box::new(script.backend())),
        };
        let err = Engine::builder()
            .model("/nowhere/model.not-fake")
            .backend(plugin)
            .build()
            .expect_err("nothing loads a path that does not exist");
        let text = err.to_string();
        assert!(!text.contains("scripted"), "{text}");
    }
}
