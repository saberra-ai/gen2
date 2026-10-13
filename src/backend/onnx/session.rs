use parking_lot::{Mutex, RwLock};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use super::{config, engine::Bundle, puller::OnnxPuller};
use crate::backend::common::stop_matcher::StopMatcher;
use crate::backend::common::tool_calls::{Protocol, ToolCallParser};
use crate::backend::{BackendSession, SessionId, TokenPullerDyn};
use crate::engine::telemetry::{HookBus, HookEvent};
use crate::engine::{ExecError, ExecutionStats, Settings};
use crate::generation::GenSpec;
use crate::session_rt::{SessionSpec, prompt::merge_prompts};
use crate::types::message::{Message, ToolSpec};

pub(super) struct OnnxSession {
    id: SessionId,
    bundle: Arc<Bundle>,
    settings: Settings,
    messages: Mutex<Vec<Message>>,
    tools: Option<(Vec<ToolSpec>, String)>,
    thinking: Option<bool>,
    stopped: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    hooks: Arc<HookBus>,
    stats: Arc<RwLock<ExecutionStats>>,
}
impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
impl OnnxSession {
    pub(super) fn new(
        id: SessionId,
        bundle: Arc<Bundle>,
        settings: Settings,
        spec: SessionSpec,
        hooks: Arc<HookBus>,
        stats: Arc<RwLock<ExecutionStats>>,
    ) -> Result<Self, ExecError> {
        if !spec.attachments.is_empty()
            || crate::session_rt::media_util::messages_have_images(&spec.messages)
        {
            return Err(ExecError::FeatureUnsupported(
                "ONNX text generation does not accept media",
            ));
        }
        if spec.cache.is_some() {
            return Err(ExecError::FeatureUnsupported("ONNX KV snapshot import"));
        }
        let mut messages = spec.messages;
        let meta = if settings.prompt.include_meta.unwrap_or(true)
            && std::env::var("PIO_DISABLE_META_PROMPT").ok().as_deref() != Some("1")
        {
            crate::session_rt::prompt::build_meta_prompt()
        } else {
            String::new()
        };
        let system = merge_prompts(
            &meta,
            settings.prompt.system_prompt.as_deref(),
            spec.persona.as_ref(),
        );
        if !system.trim().is_empty() && !messages.iter().any(|m| m.role == "system") {
            if bundle.template.supports_system_role() {
                messages.insert(0, Message::system(system));
            } else if let Some(message) = messages.iter_mut().find(|m| m.role == "user") {
                *message = Message::user(format!("{system}\n\n{}", message.text()));
            }
        }
        let thinking = match spec.thinking {
            crate::generation::ThinkingMode::Auto => Some(
                crate::zoo::ModelFamily::detect(Some(&bundle.facts.architecture), None)
                    .default_enable_thinking(),
            ),
            thinking => thinking.as_enable_thinking(),
        };
        Ok(Self {
            id,
            bundle,
            settings,
            messages: Mutex::new(messages),
            tools: spec.tools,
            thinking,
            stopped: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            hooks,
            stats,
        })
    }
}
impl BackendSession for OnnxSession {
    fn id(&self) -> SessionId {
        self.id
    }
    fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }
    fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }
    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }
    fn ctx_size(&self) -> u32 {
        self.bundle.facts.context
    }
    fn append_messages(&self, messages: Vec<Message>) -> Result<usize, ExecError> {
        if crate::session_rt::media_util::messages_have_images(&messages) {
            return Err(ExecError::FeatureUnsupported("ONNX images"));
        }
        self.messages.lock().extend(messages);
        Ok(0)
    }
    fn pull(&self, spec: GenSpec) -> Result<Box<dyn TokenPullerDyn>, ExecError> {
        self.stopped.store(false, Ordering::Release);
        self.paused.store(false, Ordering::Release);
        let prompt = self
            .bundle
            .template
            .apply(
                self.messages.lock().clone(),
                self.tools.clone(),
                self.thinking,
            )
            .map_err(|e| ExecError::TemplateError(e.to_string()))?;
        let tokens = self
            .bundle
            .tokenizer
            .encode(&prompt, false)
            .map_err(ExecError::Other)?;
        let tokens: Vec<i32> = tokens
            .into_iter()
            .map(|id| {
                i32::try_from(id).map_err(|_| ExecError::InvalidArg("ONNX token ID exceeds i32"))
            })
            .collect::<Result<_, _>>()?;
        let (options, sample) = config::search_options(
            &self.settings,
            &spec,
            tokens.len(),
            self.bundle.facts.context,
        )?;
        let guidance = spec.grammar.as_ref().map(config::guidance);
        let started = Instant::now();
        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens: tokens.len(),
        });
        let mut generator = self.bundle.model.generate(
            &tokens,
            &options,
            sample,
            guidance.as_ref().map(|(kind, text)| (*kind, text.as_str())),
        )?;
        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens: tokens.len(),
        });
        let mut parser = ToolCallParser::new(Protocol::Auto);
        if let Some((tools, _)) = &self.tools {
            parser =
                parser.with_enabled_tools(tools.iter().map(|t| t.function.name.clone()).collect());
        }
        Ok(Box::new(OnnxPuller::new(
            self.id,
            Box::new(move || generator.next()),
            self.stopped.clone(),
            self.paused.clone(),
            parser,
            StopMatcher::from_strings(self.settings.stopping.stopwords.clone()),
            self.bundle.tokenizer.stop_ids().to_vec(),
            tokens.len(),
            started,
            self.hooks.clone(),
            self.stats.clone(),
        )))
    }
}
