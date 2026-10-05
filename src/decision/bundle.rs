use super::{Calibration, DecisionError, Result, SpecialTokens};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub const CODE_REVISION: &str = "8a6e1328cce2460a0e5aa348ad465bb1b5821cd2";
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Checkpoint {
    English,
    Multilingual,
    TypedDecisions,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDigest {
    pub bytes: u64,
    pub sha256: String,
}
/// Export-tested limits. Hosts may select smaller execution budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEnvelope {
    pub max_len: usize,
    pub head_max_len: usize,
    pub max_batch: usize,
    pub max_options: usize,
    /// Conservative peak including weights, activations, tokenizer and workspace.
    pub reservation_mb: u64,
}
/// Versioned, offline encoder-plus-head bundle. File hashes detect corruption;
/// callers must obtain the manifest itself from a source they trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema: String,
    pub model_repository: String,
    pub model_revision: String,
    pub code_revision: String,
    pub checkpoint: Checkpoint,
    pub graph: String,
    pub tokenizer: String,
    pub config: String,
    pub calibration: Option<String>,
    pub licenses: Vec<String>,
    pub dtype: String,
    pub opset: u32,
    pub exporter_sha256: String,
    pub versions: BTreeMap<String, String>,
    pub envelope: SequenceEnvelope,
    pub special_tokens: SpecialTokens,
    pub files: BTreeMap<String, FileDigest>,
}
#[derive(Debug, Clone)]
pub struct LayaBundle {
    pub(crate) root: PathBuf,
    manifest: BundleManifest,
    calibration: Calibration,
    digest: String,
}
impl LayaBundle {
    pub fn directory(&self) -> &Path {
        &self.root
    }
    /// Validate all declared files and graph references without opening a native
    /// session. The bundle must remain immutable for the lifetime of the model.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize().map_err(io)?;
        let bytes = std::fs::read(root.join("manifest.json")).map_err(io)?;
        let manifest: BundleManifest =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if manifest.schema != "gen2-laya-bundle/v1" || manifest.code_revision != CODE_REVISION {
            return Err(invalid("unsupported bundle schema or source revision"));
        }
        // Repository names are provenance, not a model allowlist. Fine-tunes
        // use the same checked graph/tokenizer contract and an immutable source
        // commit (40 hex) or local checkpoint content digest (64 hex).
        if manifest.model_repository.trim().is_empty()
            || manifest.model_repository.chars().any(char::is_control)
            || !matches!(manifest.model_revision.len(), 40 | 64)
            || !manifest
                .model_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
            || manifest.exporter_sha256.len() != 64
            || !manifest
                .exporter_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid("missing or mutable model/exporter provenance"));
        }
        if manifest.dtype != "fp32" || manifest.opset != 18 {
            return Err(invalid("this revision requires FP32, opset 18"));
        }
        let e = &manifest.envelope;
        if e.max_len < 4
            || e.head_max_len < 16
            || e.head_max_len >= e.max_len
            || e.max_batch == 0
            || e.max_options == 0
            || e.reservation_mb == 0
        {
            return Err(invalid("invalid qualified sequence envelope"));
        }
        if manifest.licenses.is_empty() {
            return Err(invalid("bundle must include licenses"));
        }
        let declared_bytes = manifest
            .files
            .values()
            .try_fold(0u64, |sum, f| sum.checked_add(f.bytes))
            .ok_or_else(|| invalid("declared file sizes overflow"))?;
        if declared_bytes.div_ceil(1024 * 1024) > e.reservation_mb {
            return Err(invalid(
                "reservation is smaller than the bundle's declared files",
            ));
        }
        if manifest.graph.contains('/') {
            return Err(invalid("graph must be at the bundle root"));
        }
        let tokenizer_config = Path::new(&manifest.tokenizer)
            .with_file_name("tokenizer_config.json")
            .to_string_lossy()
            .replace('\\', "/");
        if root.join(&tokenizer_config).exists() && !manifest.files.contains_key(&tokenizer_config)
        {
            return Err(invalid(
                "tokenizer_config.json must be listed and hashed when present",
            ));
        }
        for name in [&manifest.graph, &manifest.tokenizer, &manifest.config]
            .into_iter()
            .chain(manifest.calibration.iter())
            .chain(manifest.licenses.iter())
        {
            if !manifest.files.contains_key(name) {
                return Err(invalid(format!("undeclared required file {name}")));
            }
        }
        for (name, expected) in &manifest.files {
            let path = resolve(&root, name)?;
            let mut file = File::open(path).map_err(io)?;
            if file.metadata().map_err(io)?.len() != expected.bytes {
                return Err(invalid(format!("size mismatch: {name}")));
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file.read(&mut buffer).map_err(io)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            if hex::encode(hash.finalize()) != expected.sha256 {
                return Err(invalid(format!("SHA-256 mismatch: {name}")));
            }
        }
        let graph = std::fs::read(resolve(&root, &manifest.graph)?).map_err(io)?;
        super::graph::validate(&graph, &manifest)?;
        let cfg: serde_json::Value =
            serde_json::from_reader(File::open(resolve(&root, &manifest.config)?).map_err(io)?)
                .map_err(|e| invalid(e.to_string()))?;
        let mut calibration = Calibration::from_json(&cfg)?;
        if let Some(name) = &manifest.calibration {
            let payload: serde_json::Value =
                serde_json::from_reader(File::open(resolve(&root, name)?).map_err(io)?)
                    .map_err(|e| invalid(e.to_string()))?;
            let mut identity = cfg.clone();
            if let Some(obj) = identity.as_object_mut() {
                obj.remove("temperature");
                obj.remove("temperature_by_options");
            }
            let subfolder = match manifest.checkpoint {
                Checkpoint::English => "",
                Checkpoint::Multilingual => "multilingual",
                Checkpoint::TypedDecisions => "typed-decisions",
            };
            if payload.get("version").and_then(|v| v.as_u64()) != Some(2)
                || payload.get("model_id_or_path").and_then(|v| v.as_str())
                    != Some(manifest.model_repository.as_str())
                || payload
                    .get("subfolder")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    != subfolder
                || payload.get("config") != Some(&identity)
            {
                return Err(invalid("calibration identity does not match checkpoint"));
            }
            calibration = Calibration::from_json(&payload)?;
        }
        Ok(Self {
            root,
            manifest,
            calibration,
            digest: hex::encode(Sha256::digest(bytes)),
        })
    }
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.digest
    }
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }
    #[cfg(feature = "backend-laya-onnx")]
    pub(crate) fn path(&self, name: &str) -> Result<PathBuf> {
        resolve(&self.root, name)
    }
}
pub(crate) fn invalid(s: impl Into<String>) -> DecisionError {
    DecisionError::InvalidBundle(s.into())
}
fn io(e: std::io::Error) -> DecisionError {
    invalid(e.to_string())
}
pub(crate) fn safe_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.contains(['\\', ':', '\0'])
        || name
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == "..")
    {
        return Err(invalid(format!("unsafe bundle path: {name:?}")));
    }
    Ok(())
}
fn resolve(root: &Path, name: &str) -> Result<PathBuf> {
    safe_name(name)?;
    let path = root.join(name).canonicalize().map_err(io)?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(invalid(format!(
            "file escapes bundle or is not regular: {name}"
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(crate) fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya/smoke")
    }
    fn copy() -> tempfile::TempDir {
        let t = tempfile::tempdir().unwrap();
        for file in std::fs::read_dir(fixture()).unwrap() {
            let file = file.unwrap();
            std::fs::copy(file.path(), t.path().join(file.file_name())).unwrap();
        }
        t
    }
    #[test]
    fn offline_bundle_checks_all_files_and_graph_signature() {
        let b = LayaBundle::open(fixture()).unwrap();
        assert_eq!(b.manifest().checkpoint, Checkpoint::English);
        let dir = copy();
        std::fs::write(dir.path().join("config.json"), b"modified").unwrap();
        assert!(
            matches!(LayaBundle::open(dir.path()),Err(DecisionError::InvalidBundle(s)) if s.contains("mismatch"))
        );
        let dir = copy();
        std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();
        assert!(LayaBundle::open(dir.path()).is_err());
    }
    #[test]
    fn paths_are_platform_independently_relative() {
        for path in [
            "../outside",
            "/absolute",
            "C:/weights",
            "a\\b",
            "a//b",
            "./a",
            "a/../b",
            "a\0b",
        ] {
            assert!(safe_name(path).is_err(), "{path:?}");
        }
        assert!(safe_name("tokenizer/tokenizer.json").is_ok());
    }
    #[test]
    fn fine_tunes_use_metadata_validation_not_a_repository_allowlist() {
        let dir = copy();
        let path = dir.path().join("manifest.json");
        let mut manifest = LayaBundle::open(dir.path()).unwrap().manifest().clone();
        manifest.model_repository = "local/customer-intent-finetune".into();
        manifest.model_revision = "ab".repeat(32);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(LayaBundle::open(dir.path()).is_ok());
        manifest.model_revision = "main".into();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(LayaBundle::open(dir.path()).is_err());
        manifest.model_revision = "ab".repeat(20);
        manifest.code_revision = "unknown".into();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(LayaBundle::open(dir.path()).is_err());
    }
}
