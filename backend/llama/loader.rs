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
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

pub(crate) fn build_bundle(
    backend: &Arc<LlamaBackend>,
    req: &LoadRequest,
) -> Result<ModelBundle, ExecError> {
    // Load primary model
    let mut model_params = LlamaModelParams::default().with_use_mlock(true);
    if let Some(gpu_layers) = req.model_params.gpu_layers {
        model_params = model_params.with_n_gpu_layers(gpu_layers);
    }

    let model = LlamaModel::load_from_file(backend, &req.model_path, &model_params)
        .with_context(|| format!("failed to load model: {}", req.model_path.display()))
        .map_err(|e| ExecError::Other(e))?;

    // Compute model UUID as SHA-256 of file contents
    let model_uuid = "".to_string();
    // sha256_file(&req.model_path)
    // .with_context(|| "failed to hash model file")
    // .map_err(|e| ExecError::Other(e))?;

    // Minimal meta; we can extend with tokenizer digest and rope settings later
    let meta = ModelMeta {
        model_uuid,
        n_ctx: model.n_ctx_train(),
        n_layer: model.n_layer(),
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
            // Initialize MTMD context properly (not as a model)
            let marker = mtmd_default_marker().to_string();
            let params = MtmdContextParams {
                use_gpu: false,
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

fn sha256_file(path: &Path) -> anyhow::Result<String> {
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
        let got = sha256_file(f.path()).unwrap();
        assert_eq!(got, expected);
    }
}
