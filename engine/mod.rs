//! Engine: long-lived orchestrator.

mod error;
mod stats;
pub(crate) mod telemetry;
mod types;

// Backend-specific Engine is re-exported from gen2::backend
pub use crate::gen2::backend::Engine;
pub use error::ExecError;
pub use stats::ExecutionStats;
pub use telemetry::{HookBus, HookEvent, HookListener};
pub use types::{Capabilities, ChatTemplateSpec, CtxParamsInput, EmbedLoadRequest, LoadRequest, ModelParamsInput, Settings};

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// GGUF magic bytes: `GGUF` (0x47 0x47 0x55 0x46).
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Validate that `path` points to a non-empty file with a valid GGUF header.
///
/// Call this **before** passing a path to `LlamaModel::load_from_file()` to
/// avoid hangs in the C FFI layer when the file is empty or corrupt.
pub fn validate_model_file(path: &Path) -> Result<(), ExecError> {
    let md = path.metadata().map_err(|e| {
        ExecError::InvalidModelFile(format!(
            "cannot read model file '{}': {e}",
            path.display()
        ))
    })?;

    if md.len() == 0 {
        return Err(ExecError::InvalidModelFile(format!(
            "model file is empty (0 bytes): {}",
            path.display()
        )));
    }

    // Read first 4 bytes and check GGUF magic.
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|e| {
            ExecError::InvalidModelFile(format!(
                "cannot read model header '{}': {e}",
                path.display()
            ))
        })?;

    if magic != GGUF_MAGIC {
        return Err(ExecError::InvalidModelFile(format!(
            "not a valid GGUF file (bad magic bytes): {}",
            path.display()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_nonexistent_file() {
        let r = validate_model_file(Path::new("/tmp/pio_test_nonexistent.gguf"));
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("cannot read model file"));
    }

    #[test]
    fn rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.gguf");
        File::create(&p).unwrap();
        let r = validate_model_file(&p);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("empty (0 bytes)"));
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.gguf");
        let mut f = File::create(&p).unwrap();
        f.write_all(b"NOT_GGUF_DATA_HERE").unwrap();
        let r = validate_model_file(&p);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("bad magic bytes"));
    }

    #[test]
    fn accepts_valid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("good.gguf");
        let mut f = File::create(&p).unwrap();
        f.write_all(&[0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00])
            .unwrap();
        assert!(validate_model_file(&p).is_ok());
    }
}
