use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub tokens_covered: usize,
    pub bytes: Bytes,
    pub meta: KvMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvMeta {
    pub model_uuid: String,
    pub n_ctx: u32,
    pub n_layer: u32,
    pub tokenizer_digest: [u8; 32],
    pub template_fingerprint: [u8; 32],
    pub created_at_us: i64,
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
