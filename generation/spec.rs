#[derive(Debug, Clone, Default)]
pub struct GenSpec {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
}
