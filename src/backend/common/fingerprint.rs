//! Runtime model fingerprinting for HuggingFace-convention model directories.

use super::tokenizer::HfTokenizer;
use crate::bundle::ModelMeta;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Compute a [`ModelMeta`] for HuggingFace-convention model directories (MLX and the like).
///
/// Mirrors the fingerprinting logic in the Llama backend's `build_bundle`:
/// - `tokenizer_digest`: SHA-256(BOS bytes ‖ EOS bytes ‖ vocab_size LE)
/// - `template_fingerprint`: SHA-256(chat template string)
/// - `model_uuid`: SHA-256(n_ctx ‖ n_layer ‖ vocab_size ‖ BOS ‖ EOS ‖ total file size)
///
/// `chat_template` is the already-loaded template string. Pass `None` if the
/// model has no `tokenizer_config.json`. The caller is expected to load this
/// once and share it with the bundle — this function does not read from disk.
///
/// `architecture` is the lowercase model architecture (e.g. `"gemma3"`,
/// `"qwen2"`). Used by the sampler to apply per-architecture fix-ups; pass
/// `None` if unknown.
///
/// **Note on `tokenizer_digest`:** The Llama backend hashes raw token bytes
/// from `token_to_piece_bytes` (the GGUF model's internal representation),
/// while this path hashes the *decoded string* from `HfTokenizer::decode`.
/// The two schemes will produce different digests for the same tokenizer.
/// This is fine — the digest is only compared within a backend's own session
/// cache — but means cross-backend digest comparison is not meaningful.
pub fn compute_hf_model_meta(
    tokenizer: &HfTokenizer,
    model_dir: &Path,
    n_ctx: u32,
    n_layer: u32,
    chat_template: Option<&str>,
    architecture: Option<String>,
) -> ModelMeta {
    let bos_str = tokenizer
        .bos_id()
        .and_then(|id| tokenizer.decode(&[id]).ok())
        .unwrap_or_default();
    let eos_str = tokenizer
        .eos_id()
        .and_then(|id| tokenizer.decode(&[id]).ok())
        .unwrap_or_default();
    let vocab_size = tokenizer.vocab_size();

    let tokenizer_digest: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(bos_str.as_bytes());
        h.update(eos_str.as_bytes());
        h.update((vocab_size as u64).to_le_bytes());
        h.finalize().into()
    };

    let template_fingerprint: [u8; 32] = chat_template
        .map(|tpl| {
            let mut h = Sha256::new();
            h.update(tpl.as_bytes());
            h.finalize().into()
        })
        .unwrap_or([0u8; 32]);

    let model_uuid = {
        let mut h = Sha256::new();
        h.update(n_ctx.to_le_bytes());
        h.update(n_layer.to_le_bytes());
        h.update((vocab_size as u64).to_le_bytes());
        h.update(bos_str.as_bytes());
        h.update(eos_str.as_bytes());
        h.update(model_dir_weight_size(model_dir).to_le_bytes());
        hex::encode(h.finalize())
    };

    ModelMeta {
        model_uuid,
        n_ctx,
        n_layer,
        tokenizer_digest,
        template_fingerprint,
        architecture,
    }
}

/// Sum the size of all weight files (`.safetensors`, `.onnx`) in a model directory.
fn model_dir_weight_size(model_dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(model_dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") || name.ends_with(".onnx")
        })
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    const MINIMAL_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}"#;

    fn make_tokenizer(dir: &Path) -> HfTokenizer {
        fs::write(dir.join("tokenizer.json"), MINIMAL_TOKENIZER).unwrap();
        HfTokenizer::from_dir(dir).unwrap()
    }

    #[test]
    fn produces_nonzero_digests() {
        let dir = TempDir::new().unwrap();
        let tokenizer = make_tokenizer(dir.path());
        let meta = compute_hf_model_meta(&tokenizer, dir.path(), 4096, 32, Some("hello"), None);

        assert_eq!(meta.n_ctx, 4096);
        assert_eq!(meta.n_layer, 32);
        assert!(!meta.model_uuid.is_empty());
        assert_ne!(meta.tokenizer_digest, [0u8; 32]);
        assert_ne!(meta.template_fingerprint, [0u8; 32]);
    }

    #[test]
    fn zeroes_template_fingerprint_when_no_template() {
        let dir = TempDir::new().unwrap();
        let tokenizer = make_tokenizer(dir.path());
        let meta = compute_hf_model_meta(&tokenizer, dir.path(), 2048, 16, None, None);

        assert_eq!(meta.template_fingerprint, [0u8; 32]);
        assert_ne!(meta.tokenizer_digest, [0u8; 32]);
    }

    #[test]
    fn different_layers_different_uuids() {
        let dir = TempDir::new().unwrap();
        let tokenizer = make_tokenizer(dir.path());
        let meta_a = compute_hf_model_meta(&tokenizer, dir.path(), 4096, 32, None, None);
        let meta_b = compute_hf_model_meta(&tokenizer, dir.path(), 4096, 24, None, None);

        assert_ne!(meta_a.model_uuid, meta_b.model_uuid);
    }

    #[test]
    fn weight_size_sums_correct_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("model.safetensors"), vec![0u8; 100]).unwrap();
        fs::write(dir.path().join("model2.safetensors"), vec![0u8; 200]).unwrap();
        fs::write(dir.path().join("config.json"), "{}").unwrap();

        assert_eq!(model_dir_weight_size(dir.path()), 300);
    }
}
