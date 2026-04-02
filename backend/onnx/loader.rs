//! ONNX model loader via ort SessionBuilder.

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::ValueType;

use crate::gen2::engine::ExecError;

/// Build an ort Session from an ONNX model file.
pub fn build_session(model_path: &Path, threads: Option<u32>) -> Result<Session, ExecError> {
    let threads = threads.unwrap_or(4) as usize;

    let session = Session::builder()
        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort session builder: {}", e)))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort optimization: {}", e)))?
        .with_intra_threads(threads)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort threads: {}", e)))?
        .commit_from_file(model_path)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("ort model load: {}", e)))?;

    Ok(session)
}

/// KV cache shape metadata detected from the ONNX model inputs.
#[derive(Debug, Clone, Copy)]
pub struct KvShape {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Detect KV cache shape by scanning session input names for
/// `past_key_values.{N}.key` patterns and reading their static dimensions.
///
/// Expected shape pattern: `[batch, num_kv_heads, past_seq_len, head_dim]`
/// where `batch` and `past_seq_len` are dynamic (-1) and `num_kv_heads`/`head_dim` are static.
pub fn detect_kv_shape(session: &Session) -> KvShape {
    let mut max_layer: Option<usize> = None;
    let mut num_kv_heads: usize = 0;
    let mut head_dim: usize = 0;

    for input in session.inputs().iter() {
        let name = input.name();
        if let Some(rest) = name.strip_prefix("past_key_values.") {
            if let Some(num_str) = rest.strip_suffix(".key") {
                if let Ok(n) = num_str.parse::<usize>() {
                    max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));

                    // Extract static dims from shape: [batch, num_kv_heads, seq, head_dim]
                    if num_kv_heads == 0 {
                        if let ValueType::Tensor { shape, .. } = input.dtype() {
                            if shape.len() >= 4 {
                                let heads = shape[1];
                                let dim = shape[3];
                                if heads > 0 {
                                    num_kv_heads = heads as usize;
                                }
                                if dim > 0 {
                                    head_dim = dim as usize;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    KvShape {
        num_layers: max_layer.map_or(0, |m| m + 1),
        num_kv_heads,
        head_dim,
    }
}

/// Detect the number of transformer layers (backwards compat wrapper).
pub fn detect_num_layers(session: &Session) -> usize {
    detect_kv_shape(session).num_layers
}
