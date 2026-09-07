//! Embeddings and reranking on the runtime (api_spec.md §21).
//!
//! An [`Embedder`] and a [`Reranker`] share the runtime's settings and
//! backend routing with generative models but never pretend to be one: no
//! `Turn`, no session, no options. Each is its own engine below, loaded by
//! [`Runtime::load_embedder`] / [`Runtime::load_reranker`].

use std::path::Path;
use std::sync::Arc;

use super::engine::Engine;
use super::error::Result;
use super::runtime::Runtime;
use crate::utilities::RerankResult;

/// An embedding model: strings in, one vector per string out.
///
/// Cheap to clone; every clone is the same loaded model, and the handle keeps
/// its runtime alive.
#[derive(Clone)]
pub struct Embedder {
    _runtime: Runtime,
    engine: Arc<Engine>,
}

impl Embedder {
    /// Embed several strings at once — the fast path for a corpus. One
    /// vector per input, in order.
    pub fn embed<I, S>(&self, inputs: I) -> Result<Vec<Vec<f32>>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs: Vec<String> = inputs.into_iter().map(Into::into).collect();
        self.engine.embed(&inputs)
    }

    /// Embed one string — a query, typically.
    pub fn embed_one(&self, input: impl Into<String>) -> Result<Vec<f32>> {
        self.engine.embed_one(input)
    }
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

/// A cross-encoder that scores documents against a query.
#[derive(Clone)]
pub struct Reranker {
    _runtime: Runtime,
    engine: Arc<Engine>,
}

impl Reranker {
    /// Score `documents` against `query`, best first. Each result carries
    /// the document's original index; the text is not copied back.
    pub fn rerank<I, S>(&self, query: impl Into<String>, documents: I) -> Result<Vec<RerankResult>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents: Vec<String> = documents.into_iter().map(Into::into).collect();
        self.engine.rerank(query, &documents)
    }
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker").finish_non_exhaustive()
    }
}

impl Runtime {
    /// Load an embedding model. The family (`qwen3`, …) is detected from the
    /// filename.
    ///
    /// ```no_run
    /// # let runtime = gen2::Runtime::new()?;
    /// let embedder = runtime.load_embedder("/models/embeddinggemma.gguf")?;
    /// let vectors = embedder.embed(["first document", "second document"])?;
    /// assert_eq!(vectors.len(), 2);
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn load_embedder(&self, path: impl AsRef<Path>) -> Result<Embedder> {
        let engine = self.engine_builder().embedder(path).build()?;
        Ok(Embedder {
            _runtime: self.clone(),
            engine: Arc::new(engine),
        })
    }

    /// Load a reranking model.
    ///
    /// ```no_run
    /// # let runtime = gen2::Runtime::new()?;
    /// let reranker = runtime.load_reranker("/models/bge-reranker.gguf")?;
    /// let ranked = reranker.rerank("how do I cancel?", [
    ///     "Our office hours are 9-5.",
    ///     "To cancel, open Settings and choose Close account.",
    /// ])?;
    /// assert_eq!(ranked[0].index, 1);
    /// # Ok::<(), gen2::Error>(())
    /// ```
    pub fn load_reranker(&self, path: impl AsRef<Path>) -> Result<Reranker> {
        let engine = self.engine_builder().reranker(path).build()?;
        Ok(Reranker {
            _runtime: self.clone(),
            engine: Arc::new(engine),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_and_reranker_are_clone_send_and_sync() {
        fn assert_all<T: Clone + Send + Sync + 'static>() {}
        assert_all::<Embedder>();
        assert_all::<Reranker>();
    }
}
