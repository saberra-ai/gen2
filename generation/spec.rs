use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct GenSpec {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,

    /// Nucleus sampling (top-p). Tokens whose cumulative probability exceeds
    /// `top_p` are discarded. Applied after temperature + before min-p/XTC.
    /// Default (if `None`): backend-specific, typically 0.9.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k truncation. Keeps only the K highest-probability tokens. Applied
    /// before top-p. Default (if `None`): disabled (infinite K).
    #[serde(default)]
    pub top_k: Option<i32>,

    /// Min-p threshold (Apfelmus 2023). Tokens with prob < `min_p * top_prob`
    /// are removed. A strong quality-preserving alternative to top-p;
    /// typical values 0.02–0.1. Applied after temperature and top-k, BEFORE
    /// top-p. Default (if `None`): disabled.
    #[serde(default)]
    pub min_p: Option<f32>,

    /// DRY (Don't Repeat Yourself) repetition penalty multiplier. When
    /// `Some(m)` with `m > 0`, penalise tokens that would extend an n-gram
    /// already present in recent output by `m * dry_base^(overlap -
    /// dry_allowed_length)`. Typical production value: 0.8. Default: off.
    #[serde(default)]
    pub dry_multiplier: Option<f32>,
    /// DRY penalty exponent base. Typical: 1.75. Ignored if `dry_multiplier`
    /// is None/0.
    #[serde(default)]
    pub dry_base: Option<f32>,
    /// DRY maximum tolerated repeat length before penalty scales up.
    /// Typical: 2. Ignored if `dry_multiplier` is None/0.
    #[serde(default)]
    pub dry_allowed_length: Option<usize>,

    /// XTC (Exclude Top Choices) probability — with this probability,
    /// remove all but the LOWEST-probability token above
    /// `xtc_threshold`. Encourages creative output at temp > 0.
    /// Typical: 0.5. Default: off.
    #[serde(default)]
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold — only tokens above this probability are
    /// eligible for exclusion. Typical: 0.1. Ignored if
    /// `xtc_probability` is None/0.
    #[serde(default)]
    pub xtc_threshold: Option<f32>,

    /// Additive logit bias on end-of-turn tokens (model's stop_ids). Works
    /// around quantized checkpoints where `\n` beats `<turn|>` at answer
    /// boundaries. Default (if `None`): backend-specific (2.0 on MLX).
    #[serde(default)]
    pub eot_bias: Option<f32>,

    /// Grammar-constrained decoding specification. When set, every
    /// sampling step is masked so only tokens compliant with the
    /// grammar can be chosen. Supports JSON-schema, regex, Lark, and a
    /// raw JSON-object shorthand — see `common/grammar.rs::GrammarSpec`.
    /// Note: `specta` can't serialise dynamic JSON bodies, so this field
    /// is conditionally serde-bounded and excluded from the TS bindings;
    /// callers construct it in Rust code or via a structured builder.
    #[serde(default, skip)]
    #[cfg_attr(feature = "specta", specta(skip))]
    pub grammar: Option<crate::gen2::backend::common::grammar::GrammarSpec>,

    /// Which speculative-decoding predictor to use for this session.
    /// Default (`None`): backend-specific default (`Ngram` on MLX).
    /// Set to `SpeculativeMode::Off` to disable speculative entirely,
    /// `Pld` / `Hybrid` to experiment with alternatives.
    #[serde(default)]
    pub speculative: Option<crate::gen2::backend::common::speculative::SpeculativeMode>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Step 3c: GenSpec tests ──────────────────────────────────────

    /// GenSpec::default() has all None fields.
    #[test]
    fn gen_spec_defaults() {
        let spec = GenSpec::default();
        assert!(
            spec.max_tokens.is_none(),
            "max_tokens should default to None"
        );
        assert!(
            spec.temperature.is_none(),
            "temperature should default to None"
        );
        assert!(spec.seed.is_none(), "seed should default to None");
    }

    /// Serde roundtrip for GenSpec with all fields set.
    #[test]
    fn gen_spec_serde_roundtrip() {
        let spec = GenSpec {
            max_tokens: Some(512),
            temperature: Some(0.7),
            seed: Some(42),
            ..Default::default()
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
