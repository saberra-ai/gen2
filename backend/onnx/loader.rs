//! ONNX model loader via ort SessionBuilder.

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

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

/// Detect the number of transformer layers by scanning session input names
/// for `past_key_values.{N}.key` patterns.
pub fn detect_num_layers(session: &Session) -> usize {
    let mut max_layer: Option<usize> = None;
    for input in session.inputs().iter() {
        let name = input.name();
        if let Some(rest) = name.strip_prefix("past_key_values.") {
            if let Some(num_str) = rest.strip_suffix(".key") {
                if let Ok(n) = num_str.parse::<usize>() {
                    max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));
                }
            }
        }
    }
    max_layer.map_or(0, |m| m + 1)
}
