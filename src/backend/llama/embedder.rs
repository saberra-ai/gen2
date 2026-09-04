use super::llama_config::ModelConfig;
use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;
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

    /// Stable identity string for this embedder family — written to the
    /// embeddings store's meta so a change of family (even at the same stored
    /// dimension) is detected and the stale vectors are purged before any new
    /// one is written. MUST be stable across releases and DISTINCT per family
    /// (a clash would silently mix incompatible vector spaces). Add a new arm
    /// for every new family; never reuse a string.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedderKind::Gemma => "gemma",
            EmbedderKind::Qwen3 => "qwen3",
        }
    }

    /// Human-facing label for the opt-in "retrieval quality" surface.
    pub fn label(self) -> &'static str {
        match self {
            EmbedderKind::Gemma => "EmbeddingGemma-300M",
            EmbedderKind::Qwen3 => "Qwen3-Embedding-0.6B",
        }
    }

    /// The **best-quality** embedder this machine should use, for the opt-in
    /// "best for this Mac" upgrade (the default stays [`EmbedderKind::Gemma`]
    /// for everyone; switching is user-driven because it costs a re-embed).
    ///
    /// A capable machine (≥16 GB RAM) gets **Qwen3-Embedding-0.6B** — higher MTEB
    /// (≈70.7 vs EmbeddingGemma's ≈69.7, MTEB v2 leaderboard) + 32K context, and
    /// the ~600 MB model is trivial alongside the chat model there. Smaller /
    /// mobile machines stay on EmbeddingGemma (mobile-QAT, ~300 MB, best
    /// quality-per-byte). Selection only — applying it goes through the guarded
    /// `validate_and_load_embedder` + re-embed path.
    pub fn recommend(hardware: &crate::hardware::HardwareProfile) -> EmbedderKind {
        if hardware.total_ram_gb() >= 16 {
            EmbedderKind::Qwen3
        } else {
            EmbedderKind::Gemma
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
        // Both are satisfiable rather than fatal: size the context once to a
        // token budget, declare `n_seq_max` up front, and pack prompts into
        // each decode until either the token budget or the sequence limit is
        // reached. One prompt per decode left the GPU idle between forward
        // passes and capped ingest at roughly five documents a second, which
        // is the difference between indexing a corpus overnight and over a
        // week. A group that fails to decode falls back to one prompt at a
        // time, so the robustness the old path bought is kept.
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
        // Three limits have to line up, and they are not independent.
        //
        // `n_ctx` is the whole KV cache and llama.cpp divides it evenly across
        // `n_seq_max`, so each sequence really gets `n_ctx / n_seq_max` cells.
        // Ask for more sequences without growing `n_ctx` and every prompt
        // longer than that share fails with `NoKvCacheSlot`. So `n_ctx` is
        // sized as slots x per-sequence length, not as a token budget.
        //
        // `n_ubatch` must cover the tokens in one `decode`, and it drives the
        // activation buffer — 8192 asked Metal for 4.8 GB. It is therefore the
        // thing to keep small, and it, not `n_ctx`, is what bounds a group.
        const MAX_SEQS: usize = 8;
        // Tokens per decode. Also `n_ubatch`, hence the modest value: eight
        // ordinary documents fit comfortably, and one full-length document
        // still fits alone.
        const TOKEN_BUDGET: usize = 2048;

        // Individual inputs stay capped at what the model was trained on, so
        // packing never truncates a document that would have fit before.
        let per_seq_max = (model_max_ctx as usize).min(TOKEN_BUDGET);

        // Size the context to THIS call, not to the worst case.
        //
        // A context is built per `embed` call and its compute buffer is
        // allocated and freed with it — at a 16384 context that was 1.2 GB of
        // setup on every call, which cost more than the packing saved and grew
        // worse as memory fragmented. Tokenisation already happened above, so
        // the real shape of the work is known: use the longest sequence
        // actually present rather than the longest one permitted.
        let longest = tokens_lines_list
            .iter()
            .map(|tokens| tokens.len().min(per_seq_max))
            .max()
            .unwrap_or(1)
            .max(1);
        let seq_slots = MAX_SEQS.min(tokens_lines_list.len().max(1));
        let ctx_size = (seq_slots * longest) as u32;
        let n_ctx_nz =
            std::num::NonZeroU32::new(ctx_size).with_context(|| "context size must be non-zero")?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx_nz))
            // The micro-batch only has to cover one decode, and it drives the
            // activation buffer, so it follows the context rather than the
            // constant.
            .with_n_batch(ctx_size)
            .with_n_ubatch(ctx_size)
            .with_n_seq_max(seq_slots as u32)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true)
            // Pooling is family-specific: Gemma/BGE = Mean (default path,
            // unchanged), Qwen3-Embedding = Last (last-token pooling).
            .with_pooling_type(self.kind.pooling_type());

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "unable to create the llama_context")?;

        let group_budget = ctx_size as usize;
        let mut batch = LlamaBatch::new(group_budget, seq_slots as i32);

        // Greedily pack prompts into groups that fit both limits, then decode
        // each group in a single pass.
        let mut group: Vec<(usize, &[LlamaToken])> = Vec::with_capacity(MAX_SEQS);
        let mut group_tokens = 0usize;

        for (orig_idx, tokens) in tokens_lines_list.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let truncated: &[LlamaToken] = &tokens[..tokens.len().min(per_seq_max)];

            let would_overflow = group_tokens + truncated.len() > group_budget;
            if !group.is_empty() && (would_overflow || group.len() >= seq_slots) {
                self.decode_group(&mut ctx, &mut batch, &group, normalize, &mut output)?;
                group.clear();
                group_tokens = 0;
            }

            group_tokens += truncated.len();
            group.push((orig_idx, truncated));
        }
        if !group.is_empty() {
            self.decode_group(&mut ctx, &mut batch, &group, normalize, &mut output)?;
        }

        Ok(output)
    }
}

impl LlamaEmbedder {
    /// Decode one packed group, writing each embedding to its original slot.
    ///
    /// Falls back to one prompt per decode if the packed pass fails. Packing is
    /// the fast path, not a required one: a model or backend that dislikes
    /// multi-sequence batches should still produce correct embeddings, just
    /// slowly.
    fn decode_group(
        &self,
        ctx: &mut LlamaContext,
        batch: &mut LlamaBatch<'_>,
        group: &[(usize, &[LlamaToken])],
        normalize: bool,
        output: &mut [Vec<f32>],
    ) -> Result<()> {
        let packed = (|| -> Result<Vec<Vec<f32>>> {
            batch.clear();
            for (slot, (_, tokens)) in group.iter().enumerate() {
                batch.add_sequence(tokens, slot as i32, false)?;
            }
            let mut out = Vec::with_capacity(group.len());
            batch_decode(ctx, batch, group.len() as i32, &mut out, normalize)?;
            Ok(out)
        })();

        let embeddings = match packed {
            Ok(embeddings) if embeddings.len() == group.len() => embeddings,
            // Either the decode failed or it returned a different number of
            // embeddings than sequences, which would silently misalign every
            // document in the group with someone else's vector.
            _ => {
                let mut one_at_a_time = Vec::with_capacity(group.len());
                for (_, tokens) in group {
                    batch.clear();
                    batch.add_sequence(tokens, 0, false)?;
                    let mut single = Vec::with_capacity(1);
                    batch_decode(ctx, batch, 1, &mut single, normalize)?;
                    one_at_a_time.push(single.into_iter().next().unwrap_or_default());
                }
                one_at_a_time
            }
        };

        for ((orig_idx, _), embedding) in group.iter().zip(embeddings) {
            // Matryoshka (MRL) truncation for families that support it.
            // Truncate to the target dim, then re-normalize so cosine
            // similarity stays well-defined. Gemma's target_dim is None, so
            // this is a pure pass-through on the default path.
            output[*orig_idx] = match self.kind.target_dim() {
                Some(dim) if embedding.len() > dim => mrl_truncate(&embedding, dim),
                _ => embedding,
            };
        }
        Ok(())
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

    /// By-machine embedder pick: a capable machine (≥16 GB) gets the
    /// higher-MTEB Qwen3-Embedding; a smaller/mobile one stays on the default
    /// EmbeddingGemma. Selection only — the opt-in apply path is guarded.
    #[test]
    fn recommend_picks_qwen3_on_capable_machines_else_gemma() {
        use crate::hardware::{GpuBackend, HardwareProfile};
        let hw = |gb: u64| HardwareProfile {
            total_ram_bytes: gb * 1024 * 1024 * 1024,
            cpu_cores: 8,
            gpu_backend: GpuBackend::Metal,
            vram_bytes: 0,
        };
        assert_eq!(EmbedderKind::recommend(&hw(32)), EmbedderKind::Qwen3);
        assert_eq!(EmbedderKind::recommend(&hw(16)), EmbedderKind::Qwen3);
        assert_eq!(EmbedderKind::recommend(&hw(8)), EmbedderKind::Gemma);
        // The default stays Gemma (never auto-upgraded).
        assert_eq!(EmbedderKind::default(), EmbedderKind::Gemma);
        assert_ne!(EmbedderKind::Qwen3.label(), EmbedderKind::Gemma.label());
    }

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

    /// Does packing actually buy throughput, or is the GPU already token-bound?
    ///
    /// Batching only helps when per-call overhead dominates. If the forward
    /// pass is compute-bound on tokens, N prompts in one decode costs the same
    /// as N decodes and the packing is pure complexity. This prints both so the
    /// question is settled by measurement rather than assumption.
    #[test]
    #[ignore]
    fn packing_throughput_against_one_at_a_time() {
        let Some(model_path) = std::env::var("LLAMA_QWEN3_EMBEDDER_MODEL")
            .ok()
            .or_else(|| std::env::var("LLAMA_EMBEDDER_MODEL").ok())
        else {
            eprintln!("set LLAMA_QWEN3_EMBEDDER_MODEL to run this");
            return;
        };
        let backend = match LlamaBackend::init() {
            Ok(backend) => std::sync::Arc::new(backend),
            Err(err) => {
                eprintln!("skipping: {err:?}");
                return;
            }
        };
        let embedder = LlamaEmbedder::load_from_path(backend, &model_path, ModelConfig::default())
            .expect("load embedder");

        // Roughly the length of a real document chunk.
        // Sized to a real corpus: BEIR nfcorpus abstracts run ~400 tokens
        // median, which is what decides how many fit in one group.
        let words = std::env::var("PACK_TEST_WORDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(300);
        let doc = "the quarterly report describes revenue growth across every \
                   region with particular strength in subscription renewals "
            .repeat(words / 14);
        let docs: Vec<String> = (0..32).map(|i| format!("{i} {doc}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();

        // Warm the kernels so JIT does not land on whichever runs first.
        let _ = embedder.embed(&refs[..2], true).expect("warmup");

        let started = std::time::Instant::now();
        let _ = embedder.embed(&refs, true).expect("packed");
        let packed = started.elapsed().as_secs_f64();

        let started = std::time::Instant::now();
        for prompt in &refs {
            let _ = embedder.embed(&[prompt], true).expect("single");
        }
        let single = started.elapsed().as_secs_f64();

        println!(
            "\n  packed   {:.2}s  ({:.1} docs/sec)\n  one-by-one {:.2}s  ({:.1} docs/sec)\n  speedup  {:.2}x\n",
            packed,
            refs.len() as f64 / packed,
            single,
            refs.len() as f64 / single,
            single / packed,
        );
    }

    /// Packing must not change the answer.
    ///
    /// Prompts share one decode now, so a bug in sequence assignment would
    /// hand a document its neighbour's vector — a failure that produces
    /// perfectly plausible embeddings and silently ruins retrieval, rather
    /// than crashing. Compare a packed call against the same prompts embedded
    /// one at a time and require them to agree.
    #[test]
    #[ignore]
    fn packing_produces_the_same_vectors_as_one_at_a_time() {
        let Some(model_path) = std::env::var("LLAMA_QWEN3_EMBEDDER_MODEL")
            .ok()
            .or_else(|| std::env::var("LLAMA_EMBEDDER_MODEL").ok())
        else {
            eprintln!("set LLAMA_QWEN3_EMBEDDER_MODEL to run this");
            return;
        };

        // The backend is process-global; another test may already hold it.
        let backend = match LlamaBackend::init() {
            Ok(backend) => std::sync::Arc::new(backend),
            Err(err) => {
                eprintln!("skipping test: failed to init backend: {err:?}");
                return;
            }
        };
        let embedder = LlamaEmbedder::load_from_path(backend, &model_path, ModelConfig::default())
            .expect("load embedder");

        // Deliberately uneven lengths, so the packer actually has to split
        // rather than putting everything in one convenient group.
        let prompts: Vec<String> = vec![
            "the cat sat on the mat".into(),
            "mortgage refinancing options for a second home".into(),
            "a".repeat(400),
            "quarterly revenue grew twelve percent year over year".into(),
            "b ".repeat(900),
            "sourdough starter needs feeding twice daily".into(),
        ];
        let refs: Vec<&str> = prompts.iter().map(String::as_str).collect();

        let packed = embedder.embed(&refs, true).expect("packed embed");
        assert_eq!(packed.len(), prompts.len(), "one vector per prompt");

        for (i, prompt) in refs.iter().enumerate() {
            let alone = embedder.embed(&[prompt], true).expect("single embed");
            let single = &alone[0];
            assert_eq!(
                packed[i].len(),
                single.len(),
                "prompt {i} changed width when packed"
            );

            let dot: f32 = packed[i]
                .iter()
                .zip(single.iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                dot > 0.999,
                "prompt {i} embedded differently when packed: cosine {dot}"
            );
        }
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

    // ─── ADR-0036 capability smoke: embed-qwen3 ──────────────────────────────

    /// PERMANENT capability-smoke for the `embed-qwen3` capability (ADR-0036).
    ///
    /// Loads the real Qwen3-Embedding GGUF at the fixed runner path
    /// (`~/models/Qwen3-Embedding-0.6B-Q8_0.gguf`), embeds three fixed inputs,
    /// and asserts two objective metrics (SSS): (1) `dim == 768` (MRL-truncated
    /// — drop-in for the 768-d store); (2) `cos(cat, kitten) > cos(cat, market)`
    /// (relatedness ordering). Writes the metrics to `target/captest/` (SSS+)
    /// and prints exactly one marker line:
    ///
    /// - present: `CAPTEST embed-qwen3 RUN dim=768 cos_rel=… cos_unrel=…`
    /// - absent:  `CAPTEST embed-qwen3 SKIP <reason>` (then returns).
    ///
    /// `#[ignore]` keeps it out of the default `cargo test`; the runner invokes
    /// it with `--include-ignored` and the model present.
    #[test]
    #[ignore]
    fn captest_qwen3_embedding() {
        // Fixed model location (ADR-0036 runner probe).
        let model_path = dirs::home_dir()
            .map(|h| h.join("models/Qwen3-Embedding-0.6B-Q8_0.gguf"))
            .unwrap_or_default();
        if !model_path.exists() {
            println!(
                "CAPTEST embed-qwen3 SKIP model absent at {}",
                model_path.display()
            );
            return;
        }

        let backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(err) => {
                println!("\nCAPTEST embed-qwen3 SKIP backend init failed: {err:?}");
                return;
            }
        };

        // Force Qwen3 explicitly (last-token pooling + <|endoftext|> suffix +
        // 768-d MRL) — don't rely on filename sniffing for the captest.
        let embedder = LlamaEmbedder::load_from_path_with_kind(
            Arc::new(backend),
            &model_path,
            ModelConfig::default(),
            EmbedderKind::Qwen3,
        )
        .expect("embed-qwen3: load Qwen3-Embedding GGUF");

        // Fixed, reproducible inputs.
        let sentences = ["a cat sleeping", "a kitten napping", "stock market crash"];
        let embs = embedder
            .embed(&sentences, false)
            .expect("embed-qwen3: embed inputs");
        assert_eq!(embs.len(), 3);

        // SSS metric 1: MRL truncation produced the store's 768-d shape.
        let dim = embs[0].len();
        for e in &embs {
            assert_eq!(
                e.len(),
                768,
                "embed-qwen3: must MRL-truncate to 768 for the store"
            );
        }

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        let cos_rel = cos(&embs[0], &embs[1]); // cat vs kitten (related)
        let cos_unrel = cos(&embs[0], &embs[2]); // cat vs market (unrelated)

        // SSS+: inspectable artifact under target/captest/ (CWD = pio-core/).
        let arti_dir = PathBuf::from("../target/captest");
        let _ = std::fs::create_dir_all(&arti_dir);
        let _ = std::fs::write(
            arti_dir.join("embed-qwen3.metrics.txt"),
            format!("dim={dim}\ncos_rel={cos_rel:.4}\ncos_unrel={cos_unrel:.4}\n"),
        );

        // SSS metric 2: related sentences must score higher than unrelated.
        assert!(
            cos_rel > cos_unrel,
            "embed-qwen3: related must score higher: rel={cos_rel} unrel={cos_unrel}"
        );

        println!(
            "\nCAPTEST embed-qwen3 RUN dim={dim} cos_rel={cos_rel:.4} cos_unrel={cos_unrel:.4}"
        );
    }
}
