use crate::engine::ExecError;
use crate::generation::GenSpec;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct LoadRequest {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub model_params: ModelParamsInput,
    pub ctx_params: CtxParamsInput,
    pub template_override: Option<ChatTemplateSpec>,
    /// API key for external API backends. Set programmatically from the OS keychain
    /// at load time — never deserialized from JSON, never logged.
    #[serde(skip)]
    pub api_key: Option<String>,
    /// API format for external API backends: "openai" (default) or "anthropic".
    /// Set programmatically from config, not deserialized.
    #[serde(skip)]
    pub api_format: Option<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct EmbedLoadRequest {
    pub model_path: PathBuf,
    /// Optional explicit embedder family override (e.g. `"qwen3"`). When set,
    /// it wins over filename-based detection. `None`/empty → detect from the
    /// path, defaulting to EmbeddingGemma — so the default path is unaffected.
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Settings {
    pub sampling: SamplingSettings,
    pub stopping: StoppingSettings,
    pub system: SystemSettings,
    pub prompt: PromptSettings,
    pub mm: MmSettings,
}

impl Settings {
    pub fn validate(&self) -> Result<(), ExecError> {
        if let Some(t) = self.sampling.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(ExecError::SettingsError(format!(
                "temperature out of range: {}",
                t
            )));
        }
        if let Some(tp) = self.sampling.top_p
            && !(0.0..=1.0).contains(&tp)
        {
            return Err(ExecError::SettingsError(format!(
                "top_p out of range: {}",
                tp
            )));
        }
        if let Some(mp) = self.sampling.min_p
            && !(0.0..=1.0).contains(&mp)
        {
            return Err(ExecError::SettingsError(format!(
                "min_p out of range: {}",
                mp
            )));
        }
        if let Some(k) = self.sampling.top_k
            && k > 100_000
        {
            return Err(ExecError::SettingsError(format!("top_k too large: {}", k)));
        }
        if let Some(n) = self.sampling.penalty_last_n
            && n < -1
        {
            return Err(ExecError::SettingsError(format!(
                "penalty_last_n out of range: {}",
                n
            )));
        }
        if let Some(r) = self.sampling.penalty_repeat
            && r < 0.0
        {
            return Err(ExecError::SettingsError(format!(
                "penalty_repeat must be non-negative: {}",
                r
            )));
        }
        if let Some(f) = self.sampling.penalty_freq
            && f < 0.0
        {
            return Err(ExecError::SettingsError(format!(
                "penalty_freq must be non-negative: {}",
                f
            )));
        }
        if let Some(p) = self.sampling.penalty_present
            && p < 0.0
        {
            return Err(ExecError::SettingsError(format!(
                "penalty_present must be non-negative: {}",
                p
            )));
        }
        if let Some(n) = self.stopping.max_tokens
            && n == 0
        {
            return Err(ExecError::SettingsError("max_tokens must be > 0".into()));
        }
        if let Some(bs) = self.system.batch_size
            && (bs == 0 || bs > 4096)
        {
            return Err(ExecError::SettingsError(format!(
                "batch_size out of range: {}",
                bs
            )));
        }
        if let Some(ctx) = self.system.ctx_size
            && (!(64..=2_000_000).contains(&ctx))
        {
            return Err(ExecError::SettingsError(format!(
                "ctx_size out of range: {}",
                ctx
            )));
        }
        if let Some(t) = self.system.threads
            && (t == 0 || t > 1024)
        {
            return Err(ExecError::SettingsError(format!(
                "threads out of range: {}",
                t
            )));
        }
        if let Some(t) = self.system.threads_batch
            && (t == 0 || t > 1024)
        {
            return Err(ExecError::SettingsError(format!(
                "threads_batch out of range: {}",
                t
            )));
        }
        Ok(())
    }

    /// Return a clone with per-pull `GenSpec` sampling fields overlaid:
    /// each `Some` field on the spec wins; `None` keeps the existing
    /// Settings value. Mirrors the `gen_spec.field.or(settings.field)`
    /// pattern already used by the `external_api` backend, generalised
    /// so llama / MLX `pull()` can apply per-pull sampling overrides
    /// from `recommended_sampling(...)` (or any caller-supplied
    /// GenSpec) without first mutating engine-level Settings.
    ///
    /// Why this matters: before this merge existed, the llama and MLX
    /// samplers built their chains from `settings.sampling.*` and
    /// silently dropped GenSpec sampling fields (only `max_tokens` and
    /// `grammar` flowed through). The matrix harness's
    /// `recommended_sampling`-derived GenSpec was a no-op on those
    /// backends until this merge was wired in.
    pub fn with_gen_spec_overrides(&self, spec: &crate::generation::GenSpec) -> Settings {
        let mut out = self.clone();
        out.sampling.temperature = spec.temperature.or(out.sampling.temperature);
        out.sampling.top_p = spec.top_p.or(out.sampling.top_p);
        out.sampling.top_k = spec.top_k.or(out.sampling.top_k);
        out.sampling.min_p = spec.min_p.or(out.sampling.min_p);
        out.sampling.penalty_repeat = spec.penalty_repeat.or(out.sampling.penalty_repeat);
        out.sampling.penalty_freq = spec.penalty_freq.or(out.sampling.penalty_freq);
        out.sampling.penalty_present = spec.penalty_present.or(out.sampling.penalty_present);
        // Seed belongs here for the same reason as the rest: without it the
        // sampler falls back to a fresh random one per session, so `.seed(42)`
        // was accepted, documented, and had no effect — five runs at one seed
        // gave five different answers. `.greedy()` hid it, because temperature
        // zero is deterministic whatever the seed is.
        // Narrowed, because a backend's seed space is 32 bits while `GenSpec`
        // takes a `u64`. Deterministic either way — two seeds that differ only
        // above bit 32 land on the same stream, which is a far smaller
        // surprise than a seed that does nothing.
        out.sampling.seed = spec.seed.map(|s| s as u32).or(out.sampling.seed);
        out
    }

    /// Fill unset fields from a defaults snapshot while preserving explicit overrides.
    pub fn inherit_missing(&mut self, defaults: &Settings) {
        if self.sampling.temperature.is_none() {
            self.sampling.temperature = defaults.sampling.temperature;
        }
        if self.sampling.top_p.is_none() {
            self.sampling.top_p = defaults.sampling.top_p;
        }
        if self.sampling.min_p.is_none() {
            self.sampling.min_p = defaults.sampling.min_p;
        }
        if self.sampling.top_k.is_none() {
            self.sampling.top_k = defaults.sampling.top_k;
        }
        if self.sampling.penalty_last_n.is_none() {
            self.sampling.penalty_last_n = defaults.sampling.penalty_last_n;
        }
        if self.sampling.penalty_repeat.is_none() {
            self.sampling.penalty_repeat = defaults.sampling.penalty_repeat;
        }
        if self.sampling.penalty_freq.is_none() {
            self.sampling.penalty_freq = defaults.sampling.penalty_freq;
        }
        if self.sampling.penalty_present.is_none() {
            self.sampling.penalty_present = defaults.sampling.penalty_present;
        }

        if self.stopping.max_tokens.is_none() {
            self.stopping.max_tokens = defaults.stopping.max_tokens;
        }
        if self.stopping.stopwords.is_empty() && !defaults.stopping.stopwords.is_empty() {
            self.stopping.stopwords = defaults.stopping.stopwords.clone();
        }

        if self.system.threads.is_none() {
            self.system.threads = defaults.system.threads;
        }
        if self.system.threads_batch.is_none() {
            self.system.threads_batch = defaults.system.threads_batch;
        }
        if self.system.batch_size.is_none() {
            self.system.batch_size = defaults.system.batch_size;
        }
        if self.system.gpu_layers.is_none() {
            self.system.gpu_layers = defaults.system.gpu_layers;
        }
        if self.system.ctx_size.is_none() {
            self.system.ctx_size = defaults.system.ctx_size;
        }
        if self.system.flash_attn.is_none() {
            self.system.flash_attn = defaults.system.flash_attn;
        }

        if self.prompt.system_prompt.is_none() {
            self.prompt.system_prompt = defaults.prompt.system_prompt.clone();
        }
        if self.prompt.include_meta.is_none() {
            self.prompt.include_meta = defaults.prompt.include_meta;
        }

        if self.mm.image_size.is_none() {
            self.mm.image_size = defaults.mm.image_size;
        }
        if self.mm.audio_sample_rate.is_none() {
            self.mm.audio_sample_rate = defaults.mm.audio_sample_rate;
        }
    }

    /// Apply default generation settings onto a GenSpec, preserving explicit overrides.
    pub fn apply_to_gen_spec(&self, spec: &mut GenSpec) {
        if spec.max_tokens.is_none() {
            spec.max_tokens = self.stopping.max_tokens;
        }
        if spec.temperature.is_none() {
            spec.temperature = self.sampling.temperature;
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct SamplingSettings {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<u32>,
    pub penalty_last_n: Option<i32>,
    pub penalty_repeat: Option<f32>,
    pub penalty_freq: Option<f32>,
    pub penalty_present: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct StoppingSettings {
    pub stopwords: Vec<String>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct SystemSettings {
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub batch_size: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub flash_attn: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct PromptSettings {
    pub system_prompt: Option<String>,
    /// Include device/date meta prompt in system message. Defaults to true when None.
    pub include_meta: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MmSettings {
    pub image_size: Option<(u32, u32)>,
    pub audio_sample_rate: Option<u32>,
}

#[cfg(test)]
mod tests {
    /// Every sampling field a caller can set on a request has to survive the
    /// merge, because the backends build their sampler chain from `Settings`
    /// and never look at the `GenSpec`.
    ///
    /// Seed is the one that was missing. It was accepted by `.seed()` on four
    /// builders, documented, and dropped here — so the sampler fell back to a
    /// fresh random seed per session and five runs at one seed gave five
    /// different answers. `.greedy()` masked it, since temperature zero is
    /// deterministic whatever the seed.
    #[test]
    fn every_sampling_field_a_request_sets_survives_the_merge() {
        let spec = GenSpec {
            temperature: Some(0.3),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            seed: Some(42),
            penalty_repeat: Some(1.1),
            penalty_freq: Some(0.5),
            penalty_present: Some(0.4),
            ..Default::default()
        };

        let merged = Settings::default().with_gen_spec_overrides(&spec);

        assert_eq!(merged.sampling.temperature, Some(0.3));
        assert_eq!(merged.sampling.top_p, Some(0.9));
        assert_eq!(merged.sampling.top_k, Some(40));
        assert_eq!(merged.sampling.min_p, Some(0.05));
        assert_eq!(
            merged.sampling.seed,
            Some(42),
            "a seed the caller set must reach the sampler, or reproducibility \
             is a knob that does nothing"
        );
        assert_eq!(merged.sampling.penalty_repeat, Some(1.1));
        assert_eq!(merged.sampling.penalty_freq, Some(0.5));
        assert_eq!(merged.sampling.penalty_present, Some(0.4));
    }

    /// A request that says nothing leaves the engine's own settings alone.
    #[test]
    fn an_empty_request_overrides_nothing() {
        let mut base = Settings::default();
        base.sampling.temperature = Some(0.7);
        base.sampling.seed = Some(7);

        let merged = base.with_gen_spec_overrides(&GenSpec::default());

        assert_eq!(merged.sampling.temperature, Some(0.7));
        assert_eq!(merged.sampling.seed, Some(7));
    }

    use super::*;
    use crate::generation::GenSpec;

    /// Tracer for `Settings::with_gen_spec_overrides`: per-pull GenSpec
    /// fields must win over engine-level Settings.sampling. Without this,
    /// `recommended_sampling(model_id)` values plumbed through GenSpec
    /// at `session.pull(...)` get silently ignored by the llama/MLX
    /// samplers (they read settings.sampling.* exclusively). The matrix
    /// harness depends on this merge actually applying.
    #[test]
    fn with_gen_spec_overrides_lets_genspec_win_per_field() {
        let base = Settings {
            sampling: SamplingSettings {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(50),
                min_p: Some(0.0),
                penalty_repeat: Some(1.0),
                penalty_freq: None,
                penalty_present: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let spec = GenSpec {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(20),
            min_p: Some(0.1),
            penalty_repeat: Some(1.05),
            penalty_freq: Some(0.5),
            penalty_present: Some(1.5),
            ..Default::default()
        };
        let merged = base.with_gen_spec_overrides(&spec);
        assert_eq!(merged.sampling.temperature, Some(0.7), "temperature");
        assert_eq!(merged.sampling.top_p, Some(0.8), "top_p");
        assert_eq!(merged.sampling.top_k, Some(20), "top_k");
        assert_eq!(merged.sampling.min_p, Some(0.1), "min_p");
        assert_eq!(merged.sampling.penalty_repeat, Some(1.05), "penalty_repeat");
        assert_eq!(merged.sampling.penalty_freq, Some(0.5), "penalty_freq");
        assert_eq!(
            merged.sampling.penalty_present,
            Some(1.5),
            "penalty_present"
        );
    }

    #[test]
    fn settings_validate_ok() {
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(1.0),
                top_p: Some(0.9),
                top_k: Some(40),
                ..Default::default()
            },
            stopping: StoppingSettings {
                stopwords: vec![],
                max_tokens: Some(5),
            },
            system: SystemSettings {
                threads: Some(4),
                threads_batch: Some(4),
                batch_size: Some(64),
                gpu_layers: None,
                ctx_size: Some(2048),
                flash_attn: Some(true),
            },
            prompt: PromptSettings::default(),
            mm: MmSettings::default(),
        };
        s.validate().unwrap();
    }

    #[test]
    fn settings_validate_fail() {
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(9.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn settings_inherit_missing() {
        let defaults = Settings {
            sampling: SamplingSettings {
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_k: Some(40),
                ..Default::default()
            },
            stopping: StoppingSettings {
                stopwords: vec!["STOP".into()],
                max_tokens: Some(128),
            },
            system: SystemSettings {
                threads: Some(4),
                threads_batch: Some(2),
                batch_size: Some(64),
                gpu_layers: Some(1),
                ctx_size: Some(4096),
                flash_attn: Some(true),
            },
            prompt: PromptSettings {
                system_prompt: Some("Act like a helpful assistant".into()),
                include_meta: None,
            },
            mm: MmSettings {
                image_size: Some((256, 256)),
                audio_sample_rate: Some(44_100),
            },
        };

        let mut overrides = Settings {
            sampling: SamplingSettings {
                temperature: Some(0.75),
                ..Default::default()
            },
            stopping: StoppingSettings {
                stopwords: vec![],
                max_tokens: None,
            },
            system: SystemSettings {
                threads: Some(8),
                ..Default::default()
            },
            prompt: PromptSettings::default(),
            mm: MmSettings::default(),
        };

        overrides.inherit_missing(&defaults);

        assert_eq!(overrides.sampling.temperature, Some(0.75));
        assert_eq!(overrides.sampling.top_p, Some(0.9));
        assert_eq!(overrides.sampling.top_k, Some(40));
        assert_eq!(overrides.stopping.stopwords, vec!["STOP".to_string()]);
        assert_eq!(overrides.stopping.max_tokens, Some(128));
        assert_eq!(overrides.system.threads, Some(8));
        assert_eq!(overrides.system.batch_size, Some(64));
        assert_eq!(overrides.system.ctx_size, Some(4096));
        assert_eq!(
            overrides.prompt.system_prompt,
            Some("Act like a helpful assistant".into())
        );
        assert_eq!(overrides.mm.image_size, Some((256, 256)));
        assert_eq!(overrides.mm.audio_sample_rate, Some(44_100));
        assert_eq!(overrides.system.flash_attn, Some(true));
    }

    #[test]
    fn settings_apply_gen_spec_defaults() {
        let defaults = Settings {
            sampling: SamplingSettings {
                temperature: Some(0.5),
                ..Default::default()
            },
            stopping: StoppingSettings {
                max_tokens: Some(256),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut spec = GenSpec::default();
        defaults.apply_to_gen_spec(&mut spec);
        assert_eq!(spec.max_tokens, Some(256));
        assert_eq!(spec.temperature, Some(0.5));

        let mut spec_override = GenSpec {
            max_tokens: Some(32),
            temperature: Some(0.9),
            ..Default::default()
        };
        defaults.apply_to_gen_spec(&mut spec_override);
        assert_eq!(spec_override.max_tokens, Some(32));
        assert_eq!(spec_override.temperature, Some(0.9));
    }

    #[test]
    fn settings_validate_min_p_ok() {
        let s = Settings {
            sampling: SamplingSettings {
                min_p: Some(0.05),
                ..Default::default()
            },
            ..Default::default()
        };
        s.validate().unwrap();
    }

    #[test]
    fn settings_validate_min_p_out_of_range() {
        let s = Settings {
            sampling: SamplingSettings {
                min_p: Some(1.5),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn settings_inherit_min_p() {
        let defaults = Settings {
            sampling: SamplingSettings {
                min_p: Some(0.1),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut overrides = Settings::default();
        overrides.inherit_missing(&defaults);
        assert_eq!(overrides.sampling.min_p, Some(0.1));
    }

    #[test]
    fn settings_inherit_include_meta() {
        let defaults = Settings {
            prompt: PromptSettings {
                include_meta: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut overrides = Settings::default();
        overrides.inherit_missing(&defaults);
        assert_eq!(overrides.prompt.include_meta, Some(false));
    }

    // ── [Tomas] Boundary tests for every validation rule ─────────

    #[test]
    fn validate_temperature_exact_bounds() {
        // 0.0 is valid
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_ok());
        // 2.0 is valid
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_ok());
        // -0.01 is invalid
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(-0.01),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
        // 2.01 is invalid
        let s = Settings {
            sampling: SamplingSettings {
                temperature: Some(2.01),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_top_p_exact_bounds() {
        let ok_low = Settings {
            sampling: SamplingSettings {
                top_p: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_low.validate().is_ok());
        let ok_high = Settings {
            sampling: SamplingSettings {
                top_p: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_high.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                top_p: Some(1.01),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let neg = Settings {
            sampling: SamplingSettings {
                top_p: Some(-0.1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(neg.validate().is_err());
    }

    #[test]
    fn validate_top_k_boundary() {
        let ok = Settings {
            sampling: SamplingSettings {
                top_k: Some(100_000),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                top_k: Some(100_001),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_max_tokens_zero_rejected() {
        let s = Settings {
            stopping: StoppingSettings {
                max_tokens: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
        let ok = Settings {
            stopping: StoppingSettings {
                max_tokens: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn validate_batch_size_bounds() {
        let zero = Settings {
            system: SystemSettings {
                batch_size: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(zero.validate().is_err());
        let ok_low = Settings {
            system: SystemSettings {
                batch_size: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_low.validate().is_ok());
        let ok_high = Settings {
            system: SystemSettings {
                batch_size: Some(4096),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_high.validate().is_ok());
        let too_big = Settings {
            system: SystemSettings {
                batch_size: Some(4097),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(too_big.validate().is_err());
    }

    #[test]
    fn validate_ctx_size_bounds() {
        let too_small = Settings {
            system: SystemSettings {
                ctx_size: Some(63),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(too_small.validate().is_err());
        let ok_min = Settings {
            system: SystemSettings {
                ctx_size: Some(64),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_min.validate().is_ok());
        let ok_max = Settings {
            system: SystemSettings {
                ctx_size: Some(2_000_000),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok_max.validate().is_ok());
        let too_big = Settings {
            system: SystemSettings {
                ctx_size: Some(2_000_001),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(too_big.validate().is_err());
    }

    #[test]
    fn validate_threads_bounds() {
        let zero = Settings {
            system: SystemSettings {
                threads: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(zero.validate().is_err());
        let ok = Settings {
            system: SystemSettings {
                threads: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let max = Settings {
            system: SystemSettings {
                threads: Some(1024),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(max.validate().is_ok());
        let over = Settings {
            system: SystemSettings {
                threads: Some(1025),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(over.validate().is_err());
    }

    #[test]
    fn validate_threads_batch_bounds() {
        let zero = Settings {
            system: SystemSettings {
                threads_batch: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(zero.validate().is_err());
        let ok = Settings {
            system: SystemSettings {
                threads_batch: Some(512),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn validate_penalty_last_n_boundary() {
        // -1 is valid (disable), -2 is not
        let ok = Settings {
            sampling: SamplingSettings {
                penalty_last_n: Some(-1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                penalty_last_n: Some(-2),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_penalty_repeat_nonneg() {
        let ok = Settings {
            sampling: SamplingSettings {
                penalty_repeat: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                penalty_repeat: Some(-0.1),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_penalty_freq_nonneg() {
        let ok = Settings {
            sampling: SamplingSettings {
                penalty_freq: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                penalty_freq: Some(-0.01),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_penalty_present_nonneg() {
        let ok = Settings {
            sampling: SamplingSettings {
                penalty_present: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Settings {
            sampling: SamplingSettings {
                penalty_present: Some(-1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_none_fields_always_pass() {
        // All None = all valid (no constraints to violate)
        let s = Settings::default();
        assert!(s.validate().is_ok());
    }

    // ── [Tomas] Capabilities bitflags ────────────────────────────

    #[test]
    fn capabilities_compose() {
        let caps = Capabilities::TEXT | Capabilities::IMAGES;
        assert!(caps.contains(Capabilities::TEXT));
        assert!(caps.contains(Capabilities::IMAGES));
        assert!(!caps.contains(Capabilities::AUDIO));
    }

    #[test]
    fn capabilities_default_empty() {
        let caps = Capabilities::default();
        assert!(caps.is_empty());
    }
}

#[derive(Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelParamsInput {
    pub gpu_layers: Option<u32>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct CtxParamsInput {
    pub n_ctx: Option<u32>,
    pub seed: Option<u64>,
    pub threads: Option<u32>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum ChatTemplateSpec {
    #[default]
    Default,
}

bitflags! {
    // `Copy` and `PartialEq` because this is public API: `Engine::capabilities`
    // hands it to a caller, and a caller's first instinct is to compare it or
    // pass it on. Without them, reading a capability set forced a clone and
    // comparing one was impossible.
    #[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
    pub struct Capabilities: u32 {
        const TEXT   = 0b0001;
        const IMAGES = 0b0010;
        const AUDIO  = 0b0100;
    }
}
