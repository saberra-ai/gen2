#[derive(Debug, Clone, Default)]
pub struct ModelMeta {
    pub model_uuid: String,
    pub n_ctx: u32,
    pub n_layer: u32,
    /// SHA-256 of (BOS bytes || EOS bytes || n_vocab LE bytes). Computed once at load.
    pub tokenizer_digest: [u8; 32],
    /// Full SHA-256 of the chat template string. Computed once at load.
    pub template_fingerprint: [u8; 32],
}
