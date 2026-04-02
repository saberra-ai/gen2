use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ExecutionStats {
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub first_token_us: u64,
    pub avg_tps: f32,
}
