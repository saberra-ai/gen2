use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ExecutionStats {
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub first_token_us: u64,
    pub avg_tps: f32,
}
