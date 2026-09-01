//! Turning a [`LoadRequest`] into a loaded mistral.rs model.
//!
//! Deliberately boring. mistral.rs decides its own quantization, paged
//! attention, device mapping and topology, and gen2 does not second-guess any
//! of it — a `LoadRequest` says which weights and how much context, and
//! everything else is the backend's business.

use std::path::Path;

use std::sync::Arc;

use mistralrs::blocking::BlockingModel;
use mistralrs::{GgufModelBuilder, ModelBuilder};

use crate::engine::{ExecError, LoadRequest};

/// Load whatever the path points at.
///
/// GGUF gets the GGUF builder; anything else — a safetensors directory, a UQFF
/// file, a Hugging Face repository id — goes to the auto builder, which does
/// its own model-category detection. Guessing less here means fewer ways to be
/// wrong about a format mistral.rs already understands.
pub(super) fn load(req: &LoadRequest) -> Result<BlockingModel, ExecError> {
    let path = req.model_path.as_path();
    let is_gguf = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));

    if is_gguf {
        load_gguf(path)
    } else {
        load_auto(path)
    }
}

/// A single GGUF file, split into the directory and filename the builder wants.
fn load_gguf(path: &Path) -> Result<BlockingModel, ExecError> {
    let dir = path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());
    let file = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .ok_or_else(|| ExecError::InvalidModelFile("GGUF path has no file name".into()))?;

    // `from_auto_builder` only takes the auto builder, so the GGUF one is
    // driven on a runtime built here and the result wrapped. The runtime has to
    // outlive the model, which is why `BlockingModel::new` takes it.
    let rt = runtime()?;
    let model = rt
        .block_on(GgufModelBuilder::new(dir, vec![file]).build())
        .map_err(|e| ExecError::Other(anyhow::anyhow!("mistral.rs GGUF load failed: {e}")))?;
    Ok(BlockingModel::new(model, rt))
}

/// A runtime for the blocking wrapper to own.
///
/// `BlockingModel` panics if built inside an existing tokio runtime, which the
/// controller loop is not — it is a plain thread, which is the reason gen2's
/// backend boundary is synchronous at all.
fn runtime() -> Result<Arc<tokio::runtime::Runtime>, ExecError> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map(Arc::new)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("mistral.rs runtime: {e}")))
}

/// A directory, repository id, or anything else the auto builder recognises.
fn load_auto(path: &Path) -> Result<BlockingModel, ExecError> {
    let target = path.to_string_lossy().into_owned();
    BlockingModel::from_auto_builder(ModelBuilder::new(target))
        .map_err(|e| ExecError::Other(anyhow::anyhow!("mistral.rs load failed: {e}")))
}
