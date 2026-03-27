use crate::gen2::engine::ExecError;
use crate::gen2::generation::GenSpec;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Deserialize, Clone, Debug, Default)]
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
pub struct EmbedLoadRequest {
    pub model_path: PathBuf,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Settings {
    pub sampling: SamplingSettings,
    pub stopping: StoppingSettings,
    pub system: SystemSettings,
    pub prompt: PromptSettings,
    pub mm: MmSettings,
}

impl Settings {
    pub fn validate(&self) -> Result<(), ExecError> {
        if let Some(t) = self.sampling.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(ExecError::SettingsError(format!(
                    "temperature out of range: {}",
                    t
                )));
            }
        }
        if let Some(tp) = self.sampling.top_p {
            if !(0.0..=1.0).contains(&tp) {
                return Err(ExecError::SettingsError(format!(
                    "top_p out of range: {}",
                    tp
                )));
            }
        }
        if let Some(k) = self.sampling.top_k {
            if k > 100_000 {
                return Err(ExecError::SettingsError(format!("top_k too large: {}", k)));
            }
        }
        if let Some(n) = self.sampling.penalty_last_n {
            if n < -1 {
                return Err(ExecError::SettingsError(format!(
                    "penalty_last_n out of range: {}",
                    n
                )));
            }
        }
        if let Some(r) = self.sampling.penalty_repeat {
            if r < 0.0 {
                return Err(ExecError::SettingsError(format!(
                    "penalty_repeat must be non-negative: {}",
                    r
                )));
            }
        }
        if let Some(f) = self.sampling.penalty_freq {
            if f < 0.0 {
                return Err(ExecError::SettingsError(format!(
                    "penalty_freq must be non-negative: {}",
                    f
                )));
            }
        }
        if let Some(p) = self.sampling.penalty_present {
            if p < 0.0 {
                return Err(ExecError::SettingsError(format!(
                    "penalty_present must be non-negative: {}",
                    p
                )));
            }
        }
        if let Some(n) = self.stopping.max_tokens {
            if n == 0 {
                return Err(ExecError::SettingsError("max_tokens must be > 0".into()));
            }
        }
        if let Some(bs) = self.system.batch_size {
            if bs == 0 || bs > 4096 {
                return Err(ExecError::SettingsError(format!(
                    "batch_size out of range: {}",
                    bs
                )));
            }
        }
        if let Some(ctx) = self.system.ctx_size {
            if ctx < 64 || ctx > 2_000_000 {
                return Err(ExecError::SettingsError(format!(
                    "ctx_size out of range: {}",
                    ctx
                )));
            }
        }
        if let Some(t) = self.system.threads {
            if t == 0 || t > 1024 {
                return Err(ExecError::SettingsError(format!(
                    "threads out of range: {}",
                    t
                )));
            }
        }
        if let Some(t) = self.system.threads_batch {
            if t == 0 || t > 1024 {
                return Err(ExecError::SettingsError(format!(
                    "threads_batch out of range: {}",
                    t
                )));
            }
        }
        Ok(())
    }

    /// Fill unset fields from a defaults snapshot while preserving explicit overrides.
    pub fn inherit_missing(&mut self, defaults: &Settings) {
        if self.sampling.temperature.is_none() {
            self.sampling.temperature = defaults.sampling.temperature;
        }
        if self.sampling.top_p.is_none() {
            self.sampling.top_p = defaults.sampling.top_p;
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
pub struct SamplingSettings {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<u32>,
    pub penalty_last_n: Option<i32>,
    pub penalty_repeat: Option<f32>,
    pub penalty_freq: Option<f32>,
    pub penalty_present: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct StoppingSettings {
    pub stopwords: Vec<String>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SystemSettings {
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub batch_size: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub flash_attn: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PromptSettings {
    pub system_prompt: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct MmSettings {
    pub image_size: Option<(u32, u32)>,
    pub audio_sample_rate: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::generation::GenSpec;

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
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct ModelParamsInput {
    pub gpu_layers: Option<u32>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct CtxParamsInput {
    pub n_ctx: Option<u32>,
    pub seed: Option<u64>,
    pub threads: Option<u32>,
}

#[derive(Deserialize, Clone, Debug)]
pub enum ChatTemplateSpec {
    Default,
}

impl Default for ChatTemplateSpec {
    fn default() -> Self {
        Self::Default
    }
}

bitflags! {
    #[derive(Default, Clone, Debug)]
    pub struct Capabilities: u32 {
        const TEXT   = 0b0001;
        const IMAGES = 0b0010;
        const AUDIO  = 0b0100;
    }
}
