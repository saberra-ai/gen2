//! Reranking — scoring documents against a query with a cross-encoder.
//!
//! Not generation, and deliberately not built on it. A reranker reads the
//! query and one document *together* and emits a single relevance score; there
//! is no sampling, no session, and no text out. Prompting a chat model to
//! score documents is a different and much worse thing, so nothing here
//! manufactures a [`BackendSession`](crate::backend::BackendSession).
//!
//! llama.cpp does this natively: a context with `LlamaPoolingType::Rank`
//! returns one number from `embeddings_seq_ith` instead of a vector.

/// One document's relevance to the query.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// Where this document sat in the list the caller passed in.
    ///
    /// Results come back sorted by score, so this is the only way back to the
    /// caller's own data — and the reason the document text is not copied in
    /// here beside it. The caller already owns the text.
    pub index: usize,
    /// Relevance. Higher is better. The scale is the model's, not gen2's:
    /// comparable within one call, not across models.
    pub score: f32,
}

/// A loaded reranking model.
///
/// Built inside the worker thread, like every other helper runtime, so it may
/// hold non-`Send` native state.
pub(crate) trait RerankerRuntime {
    fn name(&self) -> String;

    /// Score each document against the query, in the order given.
    ///
    /// Returns one score per document. Ordering and index bookkeeping happen
    /// above this, so an implementation only has to answer the question.
    fn scores(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, String>;
}

pub(crate) type RerankerFactory =
    Box<dyn Fn(&std::path::Path) -> Result<Box<dyn RerankerRuntime>, String> + Send>;

/// Turn raw scores into the caller's answer.
///
/// Shared by every implementation so the contract — descending, original
/// indices, no fabricated numbers — is written once and cannot vary by
/// backend.
pub(crate) fn rank(scores: Vec<f32>, documents: usize) -> Result<Vec<RerankResult>, String> {
    if scores.len() != documents {
        return Err(format!(
            "the reranker returned {} scores for {documents} documents",
            scores.len()
        ));
    }
    // A NaN score would make the sort below meaningless and would silently
    // poison any threshold the caller applies. Refusing is the only honest
    // answer: there is no defensible value to substitute.
    if let Some(position) = scores.iter().position(|s| !s.is_finite()) {
        return Err(format!(
            "the reranker scored document {position} as {}, which cannot be ranked",
            scores[position]
        ));
    }

    let mut ranked: Vec<RerankResult> = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| RerankResult { index, score })
        .collect();
    // Descending by score, and stable — so documents the model scores
    // identically stay in the order the caller supplied rather than in
    // whatever order a sort happened to leave them.
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(ranked)
}

/// The real factory: llama.cpp with rank pooling.
#[cfg(feature = "backend-llamacpp")]
pub(crate) fn default_factory() -> RerankerFactory {
    Box::new(|path| {
        let backend = crate::backend::llama::engine::get_backend().map_err(|e| format!("{e:?}"))?;
        crate::engine::validate_model_file(path).map_err(|e| format!("{e:?}"))?;
        let params = llama_cpp_2::model::params::LlamaModelParams::default();
        let model = llama_cpp_2::model::LlamaModel::load_from_file(&backend, path, &params)
            .map_err(|e| format!("could not load the reranking model: {e}"))?;
        Ok(Box::new(LlamaReranker {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            backend,
            model,
        }))
    })
}

#[cfg(not(feature = "backend-llamacpp"))]
pub(crate) fn default_factory() -> RerankerFactory {
    Box::new(|_path| {
        Err("this build has no reranking runtime; enable `backend-llamacpp`".to_string())
    })
}

#[cfg(feature = "backend-llamacpp")]
struct LlamaReranker {
    name: String,
    backend: std::sync::Arc<llama_cpp_2::llama_backend::LlamaBackend>,
    model: llama_cpp_2::model::LlamaModel,
}

#[cfg(feature = "backend-llamacpp")]
impl RerankerRuntime for LlamaReranker {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn scores(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, String> {
        use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;

        // A cross-encoder reads query and document as one sequence, joined the
        // way llama.cpp's own reranking does it:
        //
        //     [BOS] query [EOS] [SEP] document [EOS]
        //
        // The separators are what tell the model where the query ends. Get
        // them wrong and it still returns a number, which is why this is
        // checked against a real model rather than by inspection.
        let query_tokens = self
            .model
            .str_to_token(query, AddBos::Never)
            .map_err(|e| format!("could not tokenize the query: {e}"))?;

        const MAX_RERANK_CTX: u32 = 2048;
        let ctx_size = self.model.n_ctx_train().min(MAX_RERANK_CTX);
        let n_ctx = std::num::NonZeroU32::new(ctx_size)
            .ok_or_else(|| "the reranking model reports a zero context".to_string())?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        // One context reused across documents. Building one per document would
        // pay Metal's kernel compilation every time, which dominates
        // everything else here.
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(ctx_size)
            .with_n_ubatch(ctx_size)
            .with_n_seq_max(1)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true)
            // The whole mechanism: rank pooling makes the model emit one
            // relevance score per sequence instead of an embedding.
            .with_pooling_type(LlamaPoolingType::Rank);

        let mut ctx = self
            .model
            .new_context(&self.backend, params)
            .map_err(|e| format!("could not create a reranking context: {e}"))?;
        let mut batch = LlamaBatch::new(ctx_size as usize, 1);

        let mut out = Vec::with_capacity(documents.len());
        for document in documents {
            let doc_tokens = self
                .model
                .str_to_token(document, AddBos::Never)
                .map_err(|e| format!("could not tokenize a document: {e}"))?;

            let mut tokens = Vec::with_capacity(query_tokens.len() + doc_tokens.len() + 4);
            tokens.push(self.model.token_bos());
            tokens.extend_from_slice(&query_tokens);
            tokens.push(self.model.token_eos());
            tokens.push(self.model.token_sep());
            tokens.extend_from_slice(&doc_tokens);
            tokens.push(self.model.token_eos());

            // Truncate the document rather than the query: a query cut in half
            // scores every document against the wrong question, while a
            // truncated document is still the document.
            if tokens.len() > ctx_size as usize {
                tokens.truncate(ctx_size as usize - 1);
                tokens.push(self.model.token_eos());
            }

            batch.clear();
            batch
                .add_sequence(&tokens, 0, false)
                .map_err(|e| format!("could not build the reranking batch: {e}"))?;
            ctx.clear_kv_cache_seq(Some(0), None, None)
                .map_err(|e| format!("could not clear the reranking cache: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| format!("reranking failed: {e}"))?;

            let scored = ctx
                .embeddings_seq_ith(0)
                .map_err(|e| format!("the reranker produced no score: {e}"))?;
            let score = scored
                .first()
                .copied()
                .ok_or_else(|| "the reranker returned an empty score".to_string())?;
            out.push(score);
        }
        Ok(out)
    }
}

/// A reranker that answers from a fixed table, optionally slowly.
#[cfg(test)]
pub(crate) struct ScriptedReranker {
    pub(crate) name: String,
    pub(crate) latency: std::time::Duration,
    /// Score for each document index, in the order given.
    pub(crate) scores: Vec<f32>,
    pub(crate) busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
impl RerankerRuntime for ScriptedReranker {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn scores(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>, String> {
        use std::sync::atomic::Ordering;
        self.busy.store(true, Ordering::SeqCst);
        std::thread::sleep(self.latency);
        self.busy.store(false, Ordering::SeqCst);
        Ok(self.scores.iter().copied().take(documents.len()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_best_first_carrying_their_original_positions() {
        let ranked = rank(vec![0.1, 0.9, 0.5], 3).expect("ranking");
        assert_eq!(
            ranked,
            vec![
                RerankResult {
                    index: 1,
                    score: 0.9
                },
                RerankResult {
                    index: 2,
                    score: 0.5
                },
                RerankResult {
                    index: 0,
                    score: 0.1
                },
            ],
            "the index has to survive the sort, or the caller cannot find the \
             document a score belongs to"
        );
    }

    /// Equal scores keep the caller's order.
    ///
    /// An unstable sort would shuffle documents the model considers equally
    /// relevant, so the same query over the same corpus could rank differently
    /// between runs for no reason the caller could see.
    #[test]
    fn documents_the_model_cannot_separate_stay_in_the_order_given() {
        let ranked = rank(vec![0.5, 0.5, 0.5], 3).expect("ranking");
        assert_eq!(
            ranked.iter().map(|r| r.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    /// A NaN cannot be ranked, and must not be pretended away.
    ///
    /// Sorting with one makes the whole order arbitrary, and any threshold the
    /// caller applies silently drops it. Substituting zero would be worse: it
    /// would look like a real, low score.
    #[test]
    fn a_score_that_is_not_a_number_is_refused_rather_than_sorted() {
        let outcome = rank(vec![0.5, f32::NAN], 2);
        let message = outcome.expect_err("a NaN cannot be ranked").to_string();
        assert!(
            message.contains("document 1"),
            "the refusal should name which document, got {message}"
        );
        assert!(rank(vec![f32::INFINITY], 1).is_err());
    }

    /// A runtime returning the wrong number of scores is a bug, not a partial
    /// answer.
    #[test]
    fn a_score_count_that_does_not_match_the_documents_is_refused() {
        assert!(rank(vec![0.1, 0.2], 3).is_err());
        assert!(rank(vec![0.1, 0.2, 0.3], 2).is_err());
    }

    #[test]
    fn no_documents_ranks_to_nothing() {
        assert_eq!(rank(Vec::new(), 0).expect("ranking"), Vec::new());
    }
}
