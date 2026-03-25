use parking_lot::RwLock;
use std::fmt::Debug;
use std::sync::Arc;

use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::ExecutionStats;

#[derive(Debug, Clone)]
pub enum HookEvent {
    EngineLoadStart {
        path: String,
    },
    EngineLoadOk {
        caps_text: bool,
        caps_images: bool,
        caps_audio: bool,
        meta: ModelMeta,
    },
    SessionPrefillStart {
        session_id: u64,
        prompt_tokens: usize,
    },
    SessionPrefillOk {
        session_id: u64,
        prompt_tokens: usize,
    },
    DecodeStep {
        session_id: u64,
        token_id: u32,
        text_len: usize,
    },
    FinalStats {
        session_id: u64,
        stats: ExecutionStats,
    },
}

pub trait HookListener: Debug + Send + Sync {
    fn on_event(&self, ev: &HookEvent);
}

#[derive(Default, Debug)]
pub struct HookBus {
    listeners: RwLock<Vec<Arc<dyn HookListener>>>,
}

impl HookBus {
    pub fn new() -> Self {
        Self {
            listeners: RwLock::new(Vec::new()),
        }
    }
    pub fn register(&self, l: Arc<dyn HookListener>) {
        self.listeners.write().push(l);
    }
    pub fn emit(&self, ev: HookEvent) {
        for l in self.listeners.read().iter() {
            l.on_event(&ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Formatter;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct C(AtomicUsize);

    impl Debug for C {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            Ok(())
        }
    }

    impl HookListener for C {
        fn on_event(&self, _ev: &HookEvent) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn register_and_emit() {
        let bus = HookBus::new();
        let c = Arc::new(C(AtomicUsize::new(0)));
        bus.register(c.clone());
        bus.emit(HookEvent::EngineLoadStart { path: "x".into() });
        assert_eq!(c.0.load(Ordering::SeqCst), 1);
    }
}
