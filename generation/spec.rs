use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenSpec {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Step 3c: GenSpec tests ──────────────────────────────────────

    /// GenSpec::default() has all None fields.
    #[test]
    fn gen_spec_defaults() {
        let spec = GenSpec::default();
        assert!(spec.max_tokens.is_none(), "max_tokens should default to None");
        assert!(spec.temperature.is_none(), "temperature should default to None");
        assert!(spec.seed.is_none(), "seed should default to None");
    }

    /// Serde roundtrip for GenSpec with all fields set.
    #[test]
    fn gen_spec_serde_roundtrip() {
        let spec = GenSpec {
            max_tokens: Some(512),
            temperature: Some(0.7),
            seed: Some(42),
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        let back: GenSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.max_tokens, Some(512));
        assert!((back.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
        assert_eq!(back.seed, Some(42));
    }

    /// Serde roundtrip for GenSpec with all None fields.
    #[test]
    fn gen_spec_serde_roundtrip_defaults() {
        let spec = GenSpec::default();
        let json = serde_json::to_string(&spec).expect("serialize");
        let back: GenSpec = serde_json::from_str(&json).expect("deserialize");
        assert!(back.max_tokens.is_none());
        assert!(back.temperature.is_none());
        assert!(back.seed.is_none());
    }
}
