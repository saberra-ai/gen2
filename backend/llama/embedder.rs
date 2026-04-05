use super::llama_config::ModelConfig;
use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::{
    context::params::LlamaContextParams,
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
        let ctx_params = LlamaContextParams::default()
            .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
            .with_embeddings(true);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "unable to create the llama_context")?;

        let tokens_lines_list = prompts
            .iter()
            .map(|line| self.model.str_to_token(line, AddBos::Always))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| "failed to tokenize prompts")?;

        // One slot per input; empty vec means no embedding (tokenless after tokenization).
        let mut output = vec![Vec::new(); prompts.len()];
        let pending: Vec<(usize, &Vec<_>)> = tokens_lines_list
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_empty())
            .collect();

        if pending.is_empty() {
            return Ok(output);
        }

        let n_ctx = ctx.n_ctx() as usize;
        let n_seq = pending.len().max(1) as i32;
        let mut batch = LlamaBatch::new(n_ctx, n_seq);
        let mut max_seq_id_batch = 0;
        let mut decode_results = Vec::new();

        for (_, tokens) in &pending {
            // A single sequence must not exceed n_ctx; otherwise we would hit the flush branch
            // with an empty batch and call decode on zero tokens (llama: n_tokens == 0).
            let tokens_slice = if tokens.len() > n_ctx {
                &tokens[..n_ctx]
            } else {
                tokens.as_slice()
            };

            // Flush the batch if the next prompt would exceed our batch size.
            // Never decode when nothing was added yet (max_seq_id_batch == 0).
            if (batch.n_tokens() as usize + tokens_slice.len()) > n_ctx && max_seq_id_batch > 0 {
                batch_decode(
                    &mut ctx,
                    &mut batch,
                    max_seq_id_batch,
                    &mut decode_results,
                    normalize,
                )?;
                max_seq_id_batch = 0;
            }

            batch.add_sequence(tokens_slice, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        // Handle final batch (skip if all sequences were already flushed in the loop)
        if max_seq_id_batch > 0 {
            batch_decode(
                &mut ctx,
                &mut batch,
                max_seq_id_batch,
                &mut decode_results,
                normalize,
            )?;
        }

        debug_assert_eq!(decode_results.len(), pending.len());
        for ((orig_idx, _), emb) in pending.iter().zip(decode_results) {
            output[*orig_idx] = emb;
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
