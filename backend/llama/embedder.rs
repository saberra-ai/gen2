use super::llama_config::ModelConfig;
use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    model::params::LlamaModelParams,
    model::{AddBos, LlamaModel},
};
use std::path::Path;
use std::sync::Arc;

/// Which embedding-model *family* an embedder belongs to.
///
/// This is the pluggable seam (best-in-class embedding option). Each kind
/// carries the family-specific quirks that the raw llama.cpp embedding path
/// must respect:
///
///   - **pooling**: how per-token hidden states collapse into one vector.
///     EmbeddingGemma / BGE use mean pooling; Qwen3-Embedding uses *last-token*
///     pooling (the `<|endoftext|>` position carries the sentence vector).
///   - **suffix**: a literal string appended to every input before tokenization.
///     Qwen3-Embedding requires the input to end with `<|endoftext|>`.
///   - **target_dim**: optional Matryoshka (MRL) truncation. Qwen3-Embedding is
///     trained with MRL, so its native 1024-d vector can be truncated (then
///     re-normalized) to any smaller dim with graceful quality loss. Truncating
///     to 768 makes Qwen3 a *drop-in* for Pio's existing 768-d libsql store —
///     no schema change, no separate-dim index.
///
/// `Gemma` is the default and reproduces the historical behavior byte-for-byte
/// (mean pooling, no suffix, no truncation). Selecting `Qwen3` is opt-in and
/// requires a one-time reindex (the vector space differs from Gemma's), but it
/// keeps the same 768-d store shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedderKind {
    /// EmbeddingGemma 300M (and BGE-style encoders): mean pooling, no suffix,
    /// no truncation. The existing/default Sabra embedder. Unchanged.
    #[default]
    Gemma,
    /// Qwen3-Embedding (0.6B default): last-token pooling, `<|endoftext|>`
    /// suffix, MRL-truncated to 768 so it drops into the existing 768-d store.
    Qwen3,
}

impl EmbedderKind {
    /// Best-effort detection from a model path / filename. Lets callers select
    /// Qwen3 without a config-schema change: any embedder file whose name
    /// contains "qwen3" + "embed" is treated as Qwen3. Everything else is
    /// Gemma (the safe default), so the default path is never affected.
    pub fn from_path(path: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let full = path.to_string_lossy().to_ascii_lowercase();
        let hay: &str = if name.is_empty() {
            full.as_str()
        } else {
            &name
        };
        if hay.contains("qwen3") && hay.contains("embed") {
            EmbedderKind::Qwen3
        } else {
            EmbedderKind::Gemma
        }
    }

    /// Parse the explicit config string (`EmbedderConfig.kind`). Unknown /
    /// empty values fall back to `Gemma` so a stray value never breaks the
    /// default path.
    pub fn from_config_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "qwen3" | "qwen3-embedding" | "qwen" => EmbedderKind::Qwen3,
            _ => EmbedderKind::Gemma,
        }
    }

    /// llama.cpp pooling strategy for this family.
    fn pooling_type(self) -> LlamaPoolingType {
        match self {
            EmbedderKind::Gemma => LlamaPoolingType::Mean,
            EmbedderKind::Qwen3 => LlamaPoolingType::Last,
        }
    }

    /// Literal suffix appended to every input before tokenization.
    fn input_suffix(self) -> &'static str {
        match self {
            EmbedderKind::Gemma => "",
            // Qwen3-Embedding GGUF expects the context to end with the EOS
            // token; last-token pooling then reads the vector at that position.
            EmbedderKind::Qwen3 => "<|endoftext|>",
        }
    }

    /// Optional Matryoshka truncation target. `None` = keep the model's native
    /// dimension. Qwen3 truncates to 768 to match the existing store.
    fn target_dim(self) -> Option<usize> {
        match self {
            EmbedderKind::Gemma => None,
            EmbedderKind::Qwen3 => Some(768),
        }
    }
}

pub struct LlamaEmbedder {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    kind: EmbedderKind,
}

impl LlamaEmbedder {
    pub fn load_from_path(
        backend: Arc<LlamaBackend>,
        path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<LlamaEmbedder> {
        let kind = EmbedderKind::from_path(path.as_ref());
        Self::load_from_path_with_kind(backend, path, config, kind)
    }

    /// Load an embedder with an explicitly chosen [`EmbedderKind`]. The
    /// path-sniffing constructor delegates here. Keeping the kind explicit lets
    /// the config layer override detection when needed.
    pub fn load_from_path_with_kind(
        backend: Arc<LlamaBackend>,
        path: impl AsRef<Path>,
        _config: ModelConfig,
        kind: EmbedderKind,
    ) -> Result<LlamaEmbedder> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)?;

        Ok(LlamaEmbedder {
            backend,
            model,
            kind,
        })
    }

    /// The family this embedder was loaded as.
    pub fn kind(&self) -> EmbedderKind {
        self.kind
    }

    pub fn embed(&self, prompts: &[&str], normalize: bool) -> Result<Vec<Vec<f32>>> {
        // Tokenize first so we know how many sequences we actually need and
        // can size the context to fit. Encoder-only embedding models
        // (EmbeddingGemma, BGE, etc.) impose two hard constraints that
        // decoder models don't:
        //
        //   1. `n_seq_max` must be ≥ the number of sequences packed into a
        //      single batch. Default is 1 → `invalid seq_id[..] >= 1`
        //      whenever we try to batch multiple prompts.
        //
        //   2. `n_ubatch` (micro-batch size) must be ≥ the number of tokens
        //      decoded in a single `ctx.decode()` call. Default is 512 →
        //      `encoder requires n_ubatch >= n_tokens` crashes whenever a
        //      batch exceeds 512 tokens total.
        //
        // We solve both by processing one prompt at a time with a context
        // sized to that specific prompt. This is slightly slower than
        // packed batching but robust: every prompt gets its own correctly-
        // sized context and there's no cross-prompt budget contention.
        //
        // Per-family input adaptation (pluggable embedder seam): some families
        // require a literal suffix on every input. Qwen3-Embedding wants the
        // context to end with `<|endoftext|>` so last-token pooling reads the
        // right position. Gemma's suffix is empty, so this is a no-op on the
        // default path.
        let suffix = self.kind.input_suffix();
        let prepared: Vec<std::borrow::Cow<'_, str>> = prompts
            .iter()
            .map(|line| {
                if suffix.is_empty() {
                    std::borrow::Cow::Borrowed(*line)
                } else {
                    std::borrow::Cow::Owned(format!("{line}{suffix}"))
                }
            })
            .collect();
        let tokens_lines_list = prepared
            .iter()
            .map(|line| self.model.str_to_token(line.as_ref(), AddBos::Always))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| "failed to tokenize prompts")?;

        // One slot per input; empty vec means no embedding (tokenless after tokenization).
        let mut output = vec![Vec::new(); prompts.len()];

        // Hard cap on context size — matches EmbeddingGemma 300M's native
        // 2048 training limit and prevents allocating huge contexts for a
        // single short query.
        const MAX_EMBED_CTX: u32 = 2048;
        let model_max_ctx = self.model.n_ctx_train().min(MAX_EMBED_CTX);
        let n_threads = std::thread::available_parallelism()?.get().try_into()?;

        // Create ONE context at the full supported size and reuse it for
        // every prompt in this call. Metal kernel JIT compilation is the
        // dominant cost (~5-10s), so creating a new context per prompt
        // would make backfill impossibly slow. Each `batch_decode` call
        // clears the KV cache, so sequential reuse is safe.
        let ctx_size = model_max_ctx;
        let n_ctx_nz =
            std::num::NonZeroU32::new(ctx_size).with_context(|| "context size must be non-zero")?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx_nz))
            .with_n_batch(ctx_size)
            .with_n_ubatch(ctx_size)
            .with_n_seq_max(1)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true)
            // Pooling is family-specific: Gemma/BGE = Mean (default path,
            // unchanged), Qwen3-Embedding = Last (last-token pooling).
            .with_pooling_type(self.kind.pooling_type());

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "unable to create the llama_context")?;

        let mut batch = LlamaBatch::new(ctx_size as usize, 1);

        for (orig_idx, tokens) in tokens_lines_list.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            // Truncate to ctx_size — longer inputs would exceed n_ctx.
            let truncated: &[_] = if tokens.len() > ctx_size as usize {
                &tokens[..ctx_size as usize]
            } else {
                tokens.as_slice()
            };

            batch.clear();
            batch.add_sequence(truncated, 0, false)?;

            let mut single_out: Vec<Vec<f32>> = Vec::with_capacity(1);
            batch_decode(&mut ctx, &mut batch, 1, &mut single_out, normalize)?;
            if let Some(emb) = single_out.into_iter().next() {
                // Matryoshka (MRL) truncation for families that support it.
                // Truncate to the target dim, then re-normalize so cosine
                // similarity stays well-defined. Gemma's target_dim is None,
                // so this is a pure pass-through on the default path.
                output[orig_idx] = match self.kind.target_dim() {
                    Some(dim) if emb.len() > dim => mrl_truncate(&emb, dim),
                    _ => emb,
                };
            }
        }

        Ok(output)
    }
}

/// Matryoshka-truncate an embedding to `dim` and L2-normalize the result.
///
/// MRL-trained models (Qwen3-Embedding) place the most informative dimensions
/// first, so a prefix slice is a valid lower-dimensional embedding — but only
/// after re-normalization, since the prefix is not unit-length on its own.
fn mrl_truncate(emb: &[f32], dim: usize) -> Vec<f32> {
    let dim = dim.min(emb.len());
    normalize_vec(&emb[..dim])
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch<'_>,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize_vec(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

fn normalize_vec(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    #[test]
    #[ignore]
    fn test_embedder_generation() {
        let Some(model_path) = std::env::var("LLAMA_EMBEDDER_MODEL")
            .ok()
            .map(PathBuf::from)
        else {
            eprintln!("skipping test: set LLAMA_EMBEDDER_MODEL to a local gguf file");
            return;
        };
        if !model_path.exists() {
            eprintln!(
                "skipping test: model fixture missing at {}",
                model_path.display()
            );
            return;
        }

        let backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skipping test: failed to init backend: {err:?}");
                return;
            }
        };

        let config: ModelConfig = ModelConfig::default();

        let embedder = match LlamaEmbedder::load_from_path(Arc::new(backend), model_path, config) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping test: failed to load model: {err:?}");
                return;
            }
        };

        let result = embedder
            .embed(&["merry christmas!", "happy new year", "potato pie"], true)
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result[0].is_empty());
    }

    // ── EmbedderKind selection (pure, no model) ──────────────────────

    #[test]
    fn kind_defaults_to_gemma() {
        // The pluggable seam must default to the historical embedder so the
        // default path is never silently changed.
        assert_eq!(EmbedderKind::default(), EmbedderKind::Gemma);
    }

    #[test]
    fn kind_detection_from_path() {
        // Qwen3 GGUF naming → Qwen3; everything else → Gemma.
        assert_eq!(
            EmbedderKind::from_path(&PathBuf::from("/models/Qwen3-Embedding-0.6B-Q8_0.gguf")),
            EmbedderKind::Qwen3
        );
        assert_eq!(
            EmbedderKind::from_path(&PathBuf::from("/models/qwen3-embedding-0.6b.gguf")),
            EmbedderKind::Qwen3
        );
        assert_eq!(
            EmbedderKind::from_path(&PathBuf::from("/models/embeddinggemma-300m-Q8_0.gguf")),
            EmbedderKind::Gemma
        );
        // A plain Qwen3 *chat* model (no "embed") must NOT be treated as an embedder family.
        assert_eq!(
            EmbedderKind::from_path(&PathBuf::from("/models/Qwen3-4B-Instruct.gguf")),
            EmbedderKind::Gemma
        );
    }

    #[test]
    fn kind_from_config_str() {
        assert_eq!(EmbedderKind::from_config_str("qwen3"), EmbedderKind::Qwen3);
        assert_eq!(
            EmbedderKind::from_config_str("Qwen3-Embedding"),
            EmbedderKind::Qwen3
        );
        // Unknown / empty → safe default, never breaks the default path.
        assert_eq!(EmbedderKind::from_config_str(""), EmbedderKind::Gemma);
        assert_eq!(EmbedderKind::from_config_str("gemma"), EmbedderKind::Gemma);
        assert_eq!(EmbedderKind::from_config_str("bogus"), EmbedderKind::Gemma);
    }

    #[test]
    fn gemma_path_is_unchanged() {
        // Mean pooling, no suffix, no truncation — byte-for-byte the old path.
        let k = EmbedderKind::Gemma;
        assert_eq!(k.pooling_type(), LlamaPoolingType::Mean);
        assert_eq!(k.input_suffix(), "");
        assert_eq!(k.target_dim(), None);
    }

    #[test]
    fn qwen3_uses_last_pooling_suffix_and_768_mrl() {
        let k = EmbedderKind::Qwen3;
        assert_eq!(k.pooling_type(), LlamaPoolingType::Last);
        assert_eq!(k.input_suffix(), "<|endoftext|>");
        // MRL-truncated to 768 → drop-in for the existing 768-d libsql store.
        assert_eq!(k.target_dim(), Some(768));
    }

    #[test]
    fn mrl_truncate_yields_unit_vector_of_target_dim() {
        // A non-trivial vector longer than the target; truncation must keep the
        // prefix and re-normalize to unit length.
        let raw: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.001).collect();
        let out = mrl_truncate(&raw, 768);
        assert_eq!(out.len(), 768);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
        // Prefix direction preserved: ratio of components matches the raw prefix.
        let ratio_raw = raw[1] / raw[0];
        let ratio_out = out[1] / out[0];
        assert!((ratio_raw - ratio_out).abs() < 1e-4);
    }

    #[test]
    fn mrl_truncate_noop_when_already_short() {
        let raw = vec![0.6f32, 0.8]; // already unit length, len < target
        let out = mrl_truncate(&raw, 768);
        assert_eq!(out.len(), 2);
    }

    /// Qwen3-Embedding integration test (gated on a local GGUF).
    ///
    /// Set `LLAMA_QWEN3_EMBEDDER_MODEL` to a `Qwen3-Embedding-0.6B-*.gguf`
    /// (e.g. from `Qwen/Qwen3-Embedding-0.6B-GGUF`). Validates:
    ///   1. the MRL-truncated output is 768-d (drop-in for the existing store);
    ///   2. semantically related sentences score higher than unrelated ones.
    #[test]
    #[ignore]
    fn qwen3_embedding_similarity_and_768_dim() {
        let Some(model_path) = std::env::var("LLAMA_QWEN3_EMBEDDER_MODEL")
            .ok()
            .map(PathBuf::from)
        else {
            eprintln!("skipping: set LLAMA_QWEN3_EMBEDDER_MODEL to a Qwen3-Embedding gguf");
            return;
        };
        if !model_path.exists() {
            eprintln!(
                "skipping: model fixture missing at {}",
                model_path.display()
            );
            return;
        }

        let backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skipping: failed to init backend: {err:?}");
                return;
            }
        };

        // Force Qwen3 explicitly (don't rely on filename) to exercise the
        // pooling + suffix + MRL path directly.
        let embedder = match LlamaEmbedder::load_from_path_with_kind(
            Arc::new(backend),
            &model_path,
            ModelConfig::default(),
            EmbedderKind::Qwen3,
        ) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping: failed to load model: {err:?}");
                return;
            }
        };

        let sentences = [
            "The cat sat on the warm windowsill in the sun.", // 0 — anchor
            "A kitten napped by the sunny window.",           // 1 — related to 0
            "Quarterly tax filings are due at the end of April.", // 2 — unrelated
        ];
        let embs = embedder.embed(&sentences, false).unwrap();
        assert_eq!(embs.len(), 3);

        // Drop-in invariant: MRL truncation produced the store's 768-d shape.
        for e in &embs {
            assert_eq!(e.len(), 768, "Qwen3 must MRL-truncate to 768 for the store");
        }

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };

        let related = cos(&embs[0], &embs[1]);
        let unrelated = cos(&embs[0], &embs[2]);
        println!("Qwen3 cos(related)={related:.4}  cos(unrelated)={unrelated:.4}");
        assert!(
            related > unrelated,
            "related sentences must score higher: related={related} unrelated={unrelated}"
        );
    }
}
