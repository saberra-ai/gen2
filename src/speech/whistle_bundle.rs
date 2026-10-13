use super::{Result, SpeechBundle, SpeechError};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

pub const WHISTLE_REPOSITORY: &str = "mrfakename/whistle-ONNX";
pub const WHISTLE_REVISION: &str = "14fc92cb09ad5c271fe9a4ec4be70274790050d2";
pub(crate) const LOCK: &str = include_str!("../../tools/whistle/model-lock.json");
pub(crate) const LOCK_SHA256: &str =
    "06665e648931c3b843f641106bed0a001c1366d25bf86afb51b3d313515c7ee4";
#[derive(serde::Deserialize)]
pub(crate) struct Lock {
    pub files: BTreeMap<String, Artifact>,
}
#[derive(serde::Deserialize)]
pub(crate) struct Artifact {
    pub bytes: u64,
    pub sha256: String,
}

/// Integrity-checked, pinned Whistle package including host metadata/engram data.
#[derive(Debug, Clone)]
pub struct WhistleBundle(pub(crate) SpeechBundle);
impl WhistleBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = validate(root.as_ref())?;
        Ok(Self(SpeechBundle::whistle(root)))
    }
    pub fn directory(&self) -> &Path {
        self.0.directory()
    }
    pub fn revision(&self) -> &'static str {
        WHISTLE_REVISION
    }
    pub fn manifest_sha256(&self) -> &'static str {
        LOCK_SHA256
    }
}
fn validate(root: &Path) -> Result<PathBuf> {
    let invalid = |e: String| SpeechError::InvalidBundle(e);
    let root = root.canonicalize().map_err(|e| invalid(e.to_string()))?;
    let lock: Lock = serde_json::from_str(LOCK).map_err(|e| invalid(e.to_string()))?;
    for (name, expected) in lock.files {
        let p = root
            .join(&name)
            .canonicalize()
            .map_err(|e| invalid(format!("{name}: {e}")))?;
        // Reject links escaping the expected package directories, including links
        // from onnx to another cache entry. The exact graph bytes pin data paths.
        if p != root.join(&name) {
            return Err(invalid(format!("{name}: artifact path redirects")));
        }
        let mut file = std::fs::File::open(p).map_err(|e| invalid(e.to_string()))?;
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
    Ok(root)
}
