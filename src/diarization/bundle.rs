use super::{DiarizationError, Result};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

pub const MODEL_FILENAME: &str = "Nemotron-3-Diarization.q8_0.gguf";
pub const MODEL_REVISION: &str = "f667ed73aee57d40cc39428eb768b4fd87a0a29e";
pub const MODEL_SHA256: &str = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1";
/// Immutable pinned Q8 model. Keep its verified files unchanged while loaded.
#[derive(Debug, Clone)]
pub struct DiarizationBundle {
    path: PathBuf,
}
impl DiarizationBundle {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let invalid = |e: String| DiarizationError::InvalidBundle(e);
        let root = directory
            .as_ref()
            .canonicalize()
            .map_err(|e| invalid(e.to_string()))?;
        let path = root
            .join(MODEL_FILENAME)
            .canonicalize()
            .map_err(|e| invalid(e.to_string()))?;
        if path.parent() != Some(root.as_path()) {
            return Err(invalid("model escapes bundle".into()));
        }
        let mut file = std::fs::File::open(&path).map_err(|e| invalid(e.to_string()))?;
        let metadata = file.metadata().map_err(|e| invalid(e.to_string()))?;
        if !metadata.is_file() || metadata.len() != 107_012_128 {
            return Err(invalid("wrong model size".into()));
        }
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer).map_err(|e| invalid(e.to_string()))?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        if hex::encode(hash.finalize()) != MODEL_SHA256 {
            return Err(invalid("model checksum mismatch".into()));
        }
        Ok(Self { path })
    }
    pub fn model_path(&self) -> &Path {
        &self.path
    }
    pub fn revision(&self) -> &'static str {
        MODEL_REVISION
    }
}
