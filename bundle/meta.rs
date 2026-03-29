#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub model_uuid: String,
    pub n_ctx: u32,
    pub n_layer: u32,
    /// SHA-256 of (BOS bytes || EOS bytes || n_vocab LE bytes). Computed once at load.
    pub tokenizer_digest: [u8; 32],
    /// Full SHA-256 of the chat template string. Computed once at load.
    pub template_fingerprint: [u8; 32],
}

impl Default for ModelMeta {
    fn default() -> Self {
        Self {
            model_uuid: String::new(),
            n_ctx: 0,
            n_layer: 0,
            tokenizer_digest: [0u8; 32],
            template_fingerprint: [0u8; 32],
        }
    }
}
