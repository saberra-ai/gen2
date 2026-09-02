//! The embedding helper, as a runtime the worker owns.
//!
//! The interesting part of embedding is model-family behaviour — pooling,
//! Qwen's EOS handling, MRL truncation, normalisation — and all of it already
//! exists in [`crate::backend::llama::embedder`]. None of it is reimplemented
//! here. This is an ownership change, not a rewrite: what moves is *who holds
//! the model*, and the arithmetic is left exactly where it was.

use crate::engine::EmbedLoadRequest;

/// A loaded embedding model.
///
/// Constructed inside the worker thread and never sent anywhere, so an
/// implementation is free to hold non-`Send` native state — which is the
/// reason the factory below returns one rather than taking one.
pub(crate) trait EmbeddingRuntime {
    /// What was loaded, for status reporting.
    fn name(&self) -> String;

    /// Embed each input, in order.
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

/// Builds an embedding runtime from a load request, on the worker's thread.
///
/// A `Box<dyn Fn>` rather than a plain function so tests can install a runtime
/// that behaves however the test needs — slowly, in particular, which is the
/// only way to prove chat scheduling is unaffected by a helper that is busy.
pub(crate) type EmbedderFactory =
    Box<dyn Fn(&EmbedLoadRequest) -> Result<Box<dyn EmbeddingRuntime>, String> + Send>;

/// The real factory: llama.cpp, via the existing loader.
#[cfg(feature = "backend-llamacpp")]
pub(crate) fn default_factory() -> EmbedderFactory {
    Box::new(|req| {
        let backend = crate::backend::llama::engine::get_backend().map_err(|e| format!("{e:?}"))?;
        let embedder = crate::backend::llama::loader::build_embedder(&backend, req)
            .map_err(|e| format!("{e:?}"))?;
        Ok(Box::new(LlamaEmbeddingRuntime {
            name: file_label(req),
            inner: embedder,
        }))
    })
}

/// Without llama.cpp there is no embedding implementation to offer.
///
/// Reporting that plainly is the point: the previous arrangement let a backend
/// advertise embedding support through `as_embeddings()` and then answer
/// `Unimplemented`, so a caller found out at the call site rather than at load.
#[cfg(not(feature = "backend-llamacpp"))]
pub(crate) fn default_factory() -> EmbedderFactory {
    Box::new(|_req| {
        Err("this build has no embedding runtime; enable `backend-llamacpp`".to_string())
    })
}

/// A readable name for what was loaded.
///
/// Only reachable from the llama factory, which is the only one that loads
/// anything today.
#[cfg(feature = "backend-llamacpp")]
fn file_label(req: &EmbedLoadRequest) -> String {
    req.model_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| req.model_path.display().to_string())
}

#[cfg(feature = "backend-llamacpp")]
struct LlamaEmbeddingRuntime {
    name: String,
    inner: crate::backend::llama::embedder::LlamaEmbedder,
}

#[cfg(feature = "backend-llamacpp")]
impl EmbeddingRuntime for LlamaEmbeddingRuntime {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let slices: Vec<&str> = inputs.iter().map(String::as_str).collect();
        // `normalize: false` matches what the backend path did. Changing it
        // here would silently alter every existing caller's vectors.
        self.inner
            .embed(&slices, false)
            .map_err(|e| format!("{e:?}"))
    }
}

/// An embedder that answers from a script, optionally slowly.
///
/// The seam the worker exists to provide. Its whole purpose is the test that
/// cannot otherwise be written: hold a helper call open for a measurable
/// stretch and prove chat tokens keep being scheduled while it is held.
#[cfg(test)]
pub(crate) struct ScriptedEmbedder {
    pub(crate) name: String,
    /// How long each `embed` takes.
    pub(crate) latency: std::time::Duration,
    /// Set while a call is in flight, so a test can observe the overlap
    /// rather than infer it from timing.
    pub(crate) busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// How many embed calls have completed.
    pub(crate) calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
impl EmbeddingRuntime for ScriptedEmbedder {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        use std::sync::atomic::Ordering;
        self.busy.store(true, Ordering::SeqCst);
        std::thread::sleep(self.latency);
        self.busy.store(false, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        // One dimension per input character, so a test can tell vectors apart.
        Ok(inputs
            .iter()
            .map(|s| vec![s.len() as f32, 1.0, 0.0])
            .collect())
    }
}
