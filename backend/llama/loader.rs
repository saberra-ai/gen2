use crate::gen2::engine::{Capabilities, EmbedLoadRequest, ExecError, LoadRequest};
use super::bundle::ModelBundle;
use crate::gen2::bundle::ModelMeta;
use crate::generation::model_runner::embedder::LlamaEmbedder;
use crate::generation::model_runner::llama_config::ModelConfig;
use anyhow::{Context, anyhow};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, params::LlamaModelParams};
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, mtmd_default_marker};
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

pub(crate) fn build_bundle(
    backend: &Arc<LlamaBackend>,
    req: &LoadRequest,
) -> Result<ModelBundle, ExecError> {
    // Validate model file before passing to llama-cpp (prevents C FFI hang on empty/corrupt files)
    crate::gen2::engine::validate_model_file(&req.model_path)?;

    // Load primary model
    let mut model_params = LlamaModelParams::default().with_use_mlock(true);
    if let Some(gpu_layers) = req.model_params.gpu_layers {
        model_params = model_params.with_n_gpu_layers(gpu_layers);
    }

    let model = LlamaModel::load_from_file(backend, &req.model_path, &model_params)
        .with_context(|| format!("failed to load model: {}", req.model_path.display()))
        .map_err(|e| ExecError::Other(e))?;

    // Fast model UUID from GGUF metadata + file size (microseconds, not seconds)
    let model_uuid = {
        let mut h = Sha256::new();
        h.update(model.n_ctx_train().to_le_bytes());
        h.update(model.n_layer().to_le_bytes());
        h.update(model.n_vocab().to_le_bytes());
        if let Ok(bos) = model.token_to_piece_bytes(model.token_bos(), 32, true, None) {
            h.update(&bos);
        }
        if let Ok(eos) = model.token_to_piece_bytes(model.token_eos(), 32, true, None) {
            h.update(&eos);
        }
        // Include file size to differentiate quants of the same architecture
        if let Ok(md) = fs::metadata(&req.model_path) {
            h.update(md.len().to_le_bytes());
        }
        format!("{:x}", h.finalize())
    };

    // Pre-compute tokenizer digest (SHA-256 of BOS || EOS || n_vocab)
    let tokenizer_digest = {
        let bos = model
            .token_to_piece_bytes(model.token_bos(), 32, true, None)
            .unwrap_or_default();
        let eos = model
            .token_to_piece_bytes(model.token_eos(), 32, true, None)
            .unwrap_or_default();
        let n_vocab = model.n_vocab().to_le_bytes();
        let mut h = Sha256::new();
        h.update(&bos);
        h.update(&eos);
        h.update(&n_vocab);
        h.finalize().into()
    };

    // Pre-compute template fingerprint (full SHA-256 of chat template string)
    let template_fingerprint: [u8; 32] = model
        .chat_template(None)
        .ok()
        .and_then(|t| t.to_string().ok())
        .map(|tpl| {
            let mut h = Sha256::new();
            h.update(tpl.as_bytes());
            h.finalize().into()
        })
        .unwrap_or([0u8; 32]);

    let meta = ModelMeta {
        model_uuid,
        n_ctx: model.n_ctx_train(),
        n_layer: model.n_layer(),
        tokenizer_digest,
        template_fingerprint,
    };

    // Capabilities discovery and optional MTMD context
    let mut caps = Capabilities::TEXT;
    let mut mtmd_ctx: Option<MtmdContext> = None;
    let mut mtmd_marker: Option<String> = None;

    if let Some(p) = &req.mmproj_path {
        if !Path::new(p).exists() {
            warn!(
                path = %p.display(),
                "mmproj path not found; continuing without projector"
            );
        } else {
            // MTMD GPU follows the same policy as the main model
            let use_gpu = req.model_params.gpu_layers.map_or(false, |n| n > 0);
            let marker = mtmd_default_marker().to_string();
            let params = MtmdContextParams {
                use_gpu,
                print_timings: false,
                n_threads: req.ctx_params.threads.unwrap_or(4) as i32,
                media_marker: CString::new(marker.clone())
                    .map_err(|e| ExecError::Other(e.into()))?,
            };
            let path = p
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF8 path: {}", p.display()))?;
            match MtmdContext::init_from_file(path, &model, &params)
                .with_context(|| format!("failed to init mtmd from {}", p.display()))
            {
                Ok(ctx) => {
                    mtmd_ctx = Some(ctx);
                    mtmd_marker = Some(marker);
                    caps |= Capabilities::IMAGES;
                }
                Err(err) => {
                    warn!(
                        error = %err,
                        path = %p.display(),
                        "mmproj init failed; continuing without projector"
                    );
                }
            }
        }
    }

    Ok(ModelBundle {
        model,
        capabilities: caps,
        meta,
        mtmd_ctx,
        mtmd_marker,
    })
}

fn _sha256_file(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

pub(crate) fn build_embedder(
    backend: &Arc<LlamaBackend>,
    req: &EmbedLoadRequest,
) -> Result<LlamaEmbedder, ExecError> {
    // Validate embedder file before passing to llama-cpp
    crate::gen2::engine::validate_model_file(&req.model_path)?;

    let config = ModelConfig::default();
    LlamaEmbedder::load_from_path(backend.clone(), &req.model_path, config)
        .with_context(|| format!("failed to load embedder: {}", req.model_path.display()))
        .map_err(|e| ExecError::Other(e.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn hashes_match() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "hello world").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"hello world"));
        let got = _sha256_file(f.path()).unwrap();
        assert_eq!(got, expected);
    }
}
