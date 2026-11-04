use super::bundle::ModelBundle;
use super::embedder::LlamaEmbedder;
use super::llama_config::ModelConfig;
use crate::bundle::ModelMeta;
use crate::engine::{Capabilities, EmbedLoadRequest, ExecError, GpuOffload, LoadRequest};
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

/// Build a [`ModelBundle`] by loading the GGUF model into memory.
///
/// # Errors
///
/// Returns `ExecError::Other` if the model exceeds available memory (OOM
/// during `LlamaModel::load_from_file`). When this happens the error
/// message will contain the llama.cpp failure reason. Callers should
/// suggest the user try a smaller quantization variant.
pub(crate) fn build_bundle(
    backend: &Arc<LlamaBackend>,
    req: &LoadRequest,
    settings: &crate::engine::Settings,
    grammar_tracker: Arc<crate::backend::common::grammar::GrammarOwnershipTracker>,
) -> Result<ModelBundle, ExecError> {
    // Validate model file before passing to llama-cpp (prevents C FFI hang on empty/corrupt files)
    crate::engine::validate_model_file(&req.model_path)?;
    if let Some(draft) = &settings.system.dflash2 {
        crate::engine::validate_model_file(&draft.model_path)?;
        if req.mmproj_path.is_some() {
            return Err(ExecError::FeatureUnsupported(
                "DFlash2 multimodal generation",
            ));
        }
    }

    // iOS-only: budget the load against the process's jetsam memory limit and
    // enforce the device floor (A14 / iPhone 12) + model ceiling (~4B / Q4_K_M)
    // BEFORE committing to the C++ load. Returns a typed ExecError (never a
    // panic) so the shell can surface "device too small / model too large".
    // Desktop/flagship never compiles or runs this — behavior is unchanged.
    #[cfg(target_os = "ios")]
    super::ios_memory::preflight_ios(&req.model_path)?;

    // Load primary model.
    //
    // `use_mlock` pins the model's pages resident so the OS can't evict the
    // weights under memory pressure. On iOS this is what keeps the weights the
    // increased-memory-limit entitlement bought from being paged out (which
    // would otherwise cause thrash or a jetsam kill mid-generation); it is set
    // unconditionally here and was already the desktop default, so desktop
    // behavior is unchanged.
    let mut model_params = LlamaModelParams::default().with_use_mlock(true);
    // iOS simulator: GGML's Metal backend crashes there, so pin to CPU (0 GPU
    // layers) regardless of the configured/profile value. Real devices keep
    // their offload setting. See hardware::is_ios_simulator.
    let gpu_layers = if crate::hardware::is_ios_simulator() {
        Some(0)
    } else {
        req.model_params.gpu_layers
    };
    if let Some(gpu_layers) = gpu_layers {
        model_params = model_params.with_n_gpu_layers(gpu_layers);
    }

    let model = LlamaModel::load_from_file(backend, &req.model_path, &model_params)
        .with_context(|| format!("failed to load model: {}", req.model_path.display()))
        .map_err(ExecError::Other)?;
    let context_kv_bytes_per_token = native_context_kv_cost(model.n_layer(), |layer| {
        model.attention_kv_layer_geometry(layer)
    })?;
    let offload = gpu_offload(backend, &model, model_params.n_gpu_layers());
    #[cfg(feature = "llama-dflash2")]
    let draft = settings
        .system
        .dflash2
        .as_ref()
        .map(|config| super::dflash2::DraftModel::load(backend, &model, config, &model_params))
        .transpose()?;

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
        hex::encode(h.finalize())
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
        h.update(n_vocab);
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

    let architecture = model
        .meta_val_str("general.architecture")
        .ok()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());

    let meta = ModelMeta {
        model_uuid,
        n_ctx: model.n_ctx_train(),
        n_layer: model.n_layer(),
        tokenizer_digest,
        template_fingerprint,
        architecture,
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
            let use_gpu = req.model_params.gpu_layers.is_some_and(|n| n > 0);
            let marker = mtmd_default_marker().to_string();
            let params = MtmdContextParams {
                use_gpu,
                print_timings: false,
                n_threads: req.ctx_params.threads.unwrap_or(4) as i32,
                media_marker: CString::new(marker.clone())
                    .map_err(|e| ExecError::Other(e.into()))?,
                // New in the bumped llama-cpp-rs: per-image token clamps.
                // -1/-1 = no clamp (the upstream default), matching prior behavior.
                image_min_tokens: -1,
                image_max_tokens: -1,
            };
            let path = p
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF8 path: {}", p.display()))?;
            match MtmdContext::init_from_file(path, &model, &params)
                .with_context(|| format!("failed to init mtmd from {}", p.display()))
            {
                Ok(ctx) => {
                    if ctx.support_vision() {
                        caps |= Capabilities::IMAGES;
                    }
                    if ctx.get_audio_sample_rate() == Some(16_000) {
                        caps |= Capabilities::AUDIO;
                    }
                    if ctx.support_lfm3_omni_video() {
                        caps |= Capabilities::VIDEO;
                    }
                    if ctx.supports_lfm3_audio_generation() {
                        caps |= Capabilities::SPEECH;
                    }
                    mtmd_ctx = Some(ctx);
                    mtmd_marker = Some(marker);
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
        grammar: crate::backend::common::grammar::GrammarOwner::new(grammar_tracker),
        #[cfg(feature = "llama-dflash2")]
        draft,
        model,
        capabilities: caps,
        meta,
        context_kv_bytes_per_token,
        mtmd_ctx,
        mtmd_marker,
        offload,
    })
}

/// Where the weights went, computed as llama.cpp computes the line it logs
/// (`llama-model.cpp`, `load_tensors`: `offloaded min(n_gpu_layers,
/// n_layer + 1)/n_layer + 1 layers to GPU`, printed only when
/// `llama_supports_gpu_offload()`; a negative `n_gpu_layers` means all). The
/// public C API exposes no per-tensor placement, so this mirrors the same
/// inputs the loader used rather than parsing its log.
fn gpu_offload(backend: &LlamaBackend, model: &LlamaModel, n_gpu_layers: i32) -> GpuOffload {
    let total = model.n_layer().saturating_add(1);
    let requested = u32::try_from(n_gpu_layers).unwrap_or(total);
    let gpu = if backend.supports_gpu_offload() {
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .find(|d| {
                matches!(
                    d.device_type,
                    llama_cpp_2::LlamaBackendDeviceType::Gpu
                        | llama_cpp_2::LlamaBackendDeviceType::IntegratedGpu
                )
            })
    } else {
        None
    };
    let layers = if gpu.is_some() {
        requested.min(total)
    } else {
        0
    };
    let gpu = gpu.filter(|_| layers > 0);
    GpuOffload {
        layers,
        total,
        backend: gpu.as_ref().map(|d| d.backend.clone()),
        device: gpu
            .as_ref()
            .map(|d| d.description.trim().to_string())
            .filter(|d| !d.is_empty()),
    }
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
    Ok(hex::encode(result))
}

pub(crate) fn build_embedder(
    backend: &Arc<LlamaBackend>,
    req: &EmbedLoadRequest,
) -> Result<LlamaEmbedder, ExecError> {
    // Validate embedder file before passing to llama-cpp
    crate::engine::validate_model_file(&req.model_path)?;

    let config = ModelConfig::default();
    // Resolve the embedder family: an explicit `kind` override wins, otherwise
    // detect from the filename. Both default to EmbeddingGemma, so the default
    // path is byte-for-byte unchanged.
    let kind = match req.kind.as_deref() {
        Some(s) if !s.trim().is_empty() => super::embedder::EmbedderKind::from_config_str(s),
        _ => super::embedder::EmbedderKind::from_path(&req.model_path),
    };
    tracing::info!(
        embedder_kind = ?kind,
        path = %req.model_path.display(),
        "loading embedder"
    );
    LlamaEmbedder::load_from_path_with_kind(backend.clone(), &req.model_path, config, kind)
        .with_context(|| format!("failed to load embedder: {}", req.model_path.display()))
        .map_err(ExecError::Other)
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
        let expected = hex::encode(Sha256::digest(b"hello world"));
        let got = _sha256_file(f.path()).unwrap();
        assert_eq!(got, expected);
    }
}

/// Fold owned native layer geometry with the same saturating F16 formula as
/// header planning. No source pathname can substitute a different model here.
fn native_context_kv_cost(
    layers: u32,
    mut read: impl FnMut(u32) -> Option<(u32, u32, u32)>,
) -> Result<u64, ExecError> {
    if layers == 0 || layers > 512 {
        return Err(ExecError::Other(anyhow!(
            "Unsupported native KV layer geometry"
        )));
    }
    let mut bytes = 0_u64;
    for layer in 0..layers {
        let (heads, key, value) = read(layer)
            .ok_or_else(|| ExecError::Other(anyhow!("Native KV layer geometry is unavailable")))?;
        if heads != 0 && (key == 0 || value == 0) {
            return Err(ExecError::Other(anyhow!(
                "Invalid native attention KV width"
            )));
        }
        bytes = bytes.saturating_add(crate::bundle::gguf::kv_bytes_with_widths(
            1,
            u64::from(heads),
            u64::from(key),
            u64::from(value),
        ));
    }
    // No ordinary attention geometry does not establish an SSM/MLA state bound.
    // Force the existing minimum automatic context; explicit settings still use
    // ordinary residency admission and the generic working floor.
    Ok(if bytes == 0 { u64::MAX } else { bytes })
}

#[cfg(test)]
mod kv_geometry_tests {
    use super::*;

    #[test]
    fn native_geometry_shim_refuses_null_model_and_outputs() {
        let mut output = 0_u32;
        // SAFETY: the shim checks the deliberately null pointers before access.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_model_kv_layer_geometry(
                std::ptr::null(),
                0,
                &mut output,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn native_geometry_is_per_layer_independent_and_saturating() {
        assert_eq!(
            native_context_kv_cost(28, |_| Some((8, 128, 128))).unwrap(),
            114_688
        );
        let geometry = [(0, 0, 0), (8, 128, 64), (8, 256, 128)];
        assert_eq!(
            native_context_kv_cost(3, |i| geometry.get(i as usize).copied()).unwrap(),
            9_216
        );
        assert_eq!(
            native_context_kv_cost(512, |_| Some((u32::MAX, u32::MAX, u32::MAX))).unwrap(),
            u64::MAX
        );
        assert_eq!(
            native_context_kv_cost(2, |_| Some((0, 0, 0))).unwrap(),
            u64::MAX
        );
        assert!(native_context_kv_cost(0, |_| None).is_err());
        assert!(native_context_kv_cost(513, |_| Some((8, 128, 128))).is_err());
        assert!(native_context_kv_cost(2, |i| (i == 0).then_some((8, 128, 128))).is_err());
        assert!(native_context_kv_cost(1, |_| Some((8, 0, 128))).is_err());
    }

    #[test]
    #[ignore = "Requires pinned public Qwen3-0.6B Q4_K_M and ordinary1024MiB admission"]
    fn real_qwen_native_geometry_stays_bound_after_owned_path_replacement() {
        let source = std::path::PathBuf::from(
            std::env::var("GEN2_QWEN_KV_MODEL").expect("explicit model path"),
        );
        let runtime = crate::Runtime::new().unwrap();
        let admission = runtime
            .reserve_working_memory(1024)
            .expect("normal live admission must fit; no bypass");
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("owned-model.gguf");
        std::fs::copy(&source, &path).unwrap();
        let mut hash = Sha256::new();
        let mut input = File::open(&path).unwrap();
        let mut buffer = [0; 65536];
        loop {
            let n = input.read(&mut buffer).unwrap();
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        assert_eq!(
            hex::encode(hash.finalize()),
            "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"
        );
        let backend = Arc::new(LlamaBackend::init().unwrap());
        let request = LoadRequest {
            model_path: path.clone(),
            model_params: crate::engine::ModelParamsInput {
                gpu_layers: Some(0),
            },
            ..Default::default()
        };
        let bundle = build_bundle(
            &backend,
            &request,
            &crate::engine::Settings::default(),
            Arc::new(crate::backend::common::grammar::GrammarOwnershipTracker::default()),
        )
        .unwrap();
        assert_eq!(bundle.context_kv_bytes_per_token, 114_688);
        assert_eq!(
            bundle.model.attention_kv_layer_geometry(0),
            Some((8, 128, 128))
        );
        assert_eq!(
            bundle
                .model
                .attention_kv_layer_geometry(bundle.model.n_layer()),
            None
        );
        #[cfg(unix)]
        {
            std::fs::remove_file(&path).unwrap();
            std::fs::write(&path, b"different pathname contents").unwrap();
            assert_eq!(
                native_context_kv_cost(bundle.model.n_layer(), |i| bundle
                    .model
                    .attention_kv_layer_geometry(i))
                .unwrap(),
                114_688
            );
        }
        drop(bundle);
        drop(backend);
        drop(admission);
    }
}
