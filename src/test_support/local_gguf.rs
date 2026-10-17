//! Sparse GGUF fixtures only; no native model tensors.
use std::io::Write;

fn string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

pub(crate) fn hybrid_header(directory: &std::path::Path) -> std::path::PathBuf {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&6u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, "lfm2");
    for (key, value) in [
        ("lfm2.context_length", 128000u32),
        ("lfm2.embedding_length", 2048),
        ("lfm2.block_count", 16),
        ("lfm2.attention.head_count", 32),
    ] {
        string(&mut bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    string(&mut bytes, "lfm2.attention.head_count_kv");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&16u64.to_le_bytes());
    for heads in [0u32, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0] {
        bytes.extend_from_slice(&heads.to_le_bytes());
    }
    let path = directory.join("synthetic-hybrid.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&bytes).unwrap();
    // Sparse body: declared size only; no acquired model or native compatibility.
    file.set_len(730895168).unwrap();
    path
}
