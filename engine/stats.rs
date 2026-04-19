use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ExecutionStats {
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub first_token_us: u64,
    pub avg_tps: f32,
    /// Tokens currently held in the KV cache.
    #[serde(default)]
    pub cache_tokens: u32,
    /// Engine-level KV cache budget (max tokens before eviction).
    #[serde(default)]
    pub cache_budget: u32,
    /// Number of engine-level KV cache evictions in this session.
    #[serde(default)]
    pub evictions: u32,
    /// Total draft tokens submitted to speculative decode (MLX backend only).
    #[serde(default)]
    pub spec_drafted: u32,
    /// Total draft tokens accepted by the target model.
    /// Hit rate = spec_accepted / spec_drafted.
    #[serde(default)]
    pub spec_accepted: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_fields_default_to_zero() {
        let s = ExecutionStats::default();
        assert_eq!(s.spec_drafted, 0);
        assert_eq!(s.spec_accepted, 0);
    }

    #[test]
    fn serde_roundtrip_preserves_spec_fields() {
        let s = ExecutionStats {
            decode_tokens: 100,
            spec_drafted: 40,
            spec_accepted: 25,
            ..Default::default()
        };
        let js = serde_json::to_string(&s).expect("serialize");
        let back: ExecutionStats = serde_json::from_str(&js).expect("deserialize");
        assert_eq!(back.spec_drafted, 40);
        assert_eq!(back.spec_accepted, 25);
        assert_eq!(back.decode_tokens, 100);
    }

    #[test]
    fn legacy_json_without_spec_fields_deserializes() {
        // Messages persisted before this change must keep loading.
        let js = r#"{
            "prompt_tokens": 10,
            "decode_tokens": 5,
            "first_token_us": 123,
            "avg_tps": 12.5
        }"#;
        let s: ExecutionStats = serde_json::from_str(js).expect("legacy deserialize");
        assert_eq!(s.prompt_tokens, 10);
        assert_eq!(s.spec_drafted, 0);
        assert_eq!(s.spec_accepted, 0);
    }
}
