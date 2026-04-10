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

pub struct LlamaEmbedder {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
}

impl LlamaEmbedder {
    pub fn load_from_path(
        backend: Arc<LlamaBackend>,
        path: impl AsRef<Path>,
        _config: ModelConfig,
    ) -> Result<LlamaEmbedder> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)?;

        Ok(LlamaEmbedder { backend, model })
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
        let tokens_lines_list = prompts
            .iter()
            .map(|line| self.model.str_to_token(line, AddBos::Always))
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
            .with_pooling_type(LlamaPoolingType::Mean);

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
                output[orig_idx] = emb;
            }
        }

        Ok(output)
    }
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
}
