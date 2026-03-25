#[derive(Debug, Clone, Default)]
pub struct ModelMeta {
    pub model_uuid: String,
    pub n_ctx: u32,
    pub n_layer: u32,
}
