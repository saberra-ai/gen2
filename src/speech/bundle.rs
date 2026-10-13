use super::{Result, SpeechError};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

pub const MODEL_REPOSITORY: &str = "ConwayResearch/bark-0.6B-4bit-v2.0";
pub const MODEL_REVISION: &str = "0d5f814105042fe801bb019dbf34b4e6f7d3eae6";
pub const MANIFEST_SHA256: &str =
    "8cd94f24f677b74f9e92306039491f23e3e22863205e59a5bdda322026f98f8c";

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    files: BTreeMap<String, FileDigest>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDigest {
    bytes: u64,
    sha256: String,
}

/// A complete, integrity-checked Bark ONNX release. Keep it immutable while loaded.
/// Only the pinned, qualified revision is accepted; arbitrary ONNX is not claimed.
#[derive(Clone, Debug)]
pub struct SpeechBundle {
    root: PathBuf,
}
impl SpeechBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let invalid = |e: String| SpeechError::InvalidBundle(e);
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| invalid(e.to_string()))?;
        let manifest_path = root.join("manifest.json");
        if std::fs::metadata(&manifest_path)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > 4096
        {
            return Err(invalid("oversized manifest".into()));
        }
        let bytes = std::fs::read(&manifest_path).map_err(|e| invalid(e.to_string()))?;
        if hex::encode(Sha256::digest(&bytes)) != MANIFEST_SHA256 {
            return Err(invalid(
                "manifest does not match the pinned Bark release".into(),
            ));
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        for (name, expected) in &manifest.files {
            let path = root
                .join(name)
                .canonicalize()
                .map_err(|e| invalid(format!("{name}: {e}")))?;
            if path.parent() != Some(root.as_path()) {
                return Err(invalid(format!(
                    "{name}: artifact escapes bundle directory"
                )));
            }
            let mut file = std::fs::File::open(&path).map_err(|e| invalid(e.to_string()))?;
            if file.metadata().map_err(|e| invalid(e.to_string()))?.len() != expected.bytes {
                return Err(invalid(format!("{name}: byte count mismatch")));
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file.read(&mut buffer).map_err(|e| invalid(e.to_string()))?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            if hex::encode(hash.finalize()) != expected.sha256 {
                return Err(invalid(format!("{name}: checksum mismatch")));
            }
        }
        Ok(Self { root })
    }
    pub fn directory(&self) -> &Path {
        &self.root
    }
    pub fn revision(&self) -> &'static str {
        MODEL_REVISION
    }
    pub fn manifest_sha256(&self) -> &'static str {
        MANIFEST_SHA256
    }
}
