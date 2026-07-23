use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub tokens_covered: usize,
    pub bytes: Bytes,
    pub meta: KvMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct KvMeta {
    pub model_uuid: String,
    pub n_ctx: u32,
    pub n_layer: u32,
    pub tokenizer_digest: [u8; 32],
    pub template_fingerprint: [u8; 32],
    pub created_at_us: i64,
    /// Number of tokens the saved KV actually covers (the session's
    /// decode position at save time). `0` on pre-keepwarm blobs — those
    /// can't take the prefill-skipping restore path.
    #[serde(default)]
    pub kv_token_count: u64,
    /// Hash of the session transcript the KV covers (roles + bodies,
    /// after meta/persona injection). Restore requires an exact match on
    /// the new session's transcript prefix — the KV holds SAMPLED
    /// assistant tokens that a re-render can't reproduce (task #91
    /// drift), so state is resumed exactly or not at all.
    #[serde(default)]
    pub transcript_sha256: [u8; 32],
}

#[derive(Debug, Clone)]
pub enum KvSaveSpec {
    ToPath(std::path::PathBuf),
    InMemory,
}

#[derive(Debug, Clone)]
pub enum KvLoadSpec {
    Strict(std::path::PathBuf),
    Lenient(std::path::PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct KvHeader {
    pub version: u16,
    pub meta: KvMeta,
    pub payload_sha256: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct KvLoadReport {
    pub loaded: bool,
    pub reason: Option<String>,
    pub tokens_covered: usize,
}
