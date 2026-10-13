//! Bundle facts and per-turn options, independent of the native runtime.

use crate::backend::common::grammar::GrammarSpec;
use crate::engine::{ExecError, Settings};
use crate::generation::GenSpec;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub(super) struct Facts {
    pub directory: PathBuf,
    pub architecture: String,
    pub context: u32,
}

impl Facts {
    pub fn read(path: &Path, requested_context: Option<u32>) -> Result<Self, ExecError> {
        let directory = if path.is_dir() {
            path
        } else {
            path.parent()
                .ok_or(ExecError::InvalidArg("ONNX model needs a bundle directory"))?
        };
        let config: Value = serde_json::from_slice(
            &std::fs::read(directory.join("genai_config.json")).map_err(|e| {
                ExecError::InvalidModelFile(format!(
                    "ONNX text generation needs genai_config.json beside the graph: {e}"
                ))
            })?,
        )
        .map_err(|e| ExecError::InvalidModelFile(format!("invalid genai_config.json: {e}")))?;
        let architecture = config
            .pointer("/model/type")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ExecError::InvalidModelFile("ONNX bundle is missing model.type".into()))?
            .to_owned();
        let available = config
            .pointer("/model/context_length")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                ExecError::InvalidModelFile(
                    "ONNX bundle needs a positive model.context_length".into(),
                )
            })?;
        let context = requested_context.unwrap_or(available);
        if context == 0 || context > available {
            return Err(ExecError::SettingsError(format!(
                "ONNX context {context} exceeds bundle context {available}"
            )));
        }
        let filename = config
            .pointer("/model/decoder/filename")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ExecError::InvalidModelFile("ONNX bundle is missing model.decoder.filename".into())
            })?;
        let graph = Path::new(filename);
        if graph.is_absolute()
            || graph
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err(ExecError::InvalidModelFile(
                "ONNX decoder filename must stay within its bundle".into(),
            ));
        }
        let graph = directory.join(graph);
        if !graph.is_file() || (!path.is_dir() && path != graph) {
            return Err(ExecError::InvalidModelFile(
                "ONNX graph is missing or differs from the configured decoder".into(),
            ));
        }
        Ok(Self {
            directory: directory.to_owned(),
            architecture,
            context,
        })
    }
}

pub(super) type SearchOptions = (Vec<(&'static str, f64)>, bool);

pub(super) fn search_options(
    settings: &Settings,
    spec: &GenSpec,
    prompt_tokens: usize,
    context: u32,
) -> Result<SearchOptions, ExecError> {
    let merged = settings.with_gen_spec_overrides(spec);
    merged.validate()?;
    let sampling = &merged.sampling;
    if sampling.min_p.is_some_and(|n| n != 0.0)
        || sampling.penalty_freq.is_some_and(|n| n != 0.0)
        || sampling.penalty_present.is_some_and(|n| n != 0.0)
        || sampling.penalty_last_n.is_some_and(|n| n != -1)
        || spec.dry_multiplier.is_some_and(|n| n != 0.0)
        || spec.xtc_probability.is_some_and(|n| n != 0.0)
        || spec.eot_bias.is_some_and(|n| n != 0.0)
    {
        return Err(ExecError::FeatureUnsupported(
            "ONNX GenAI does not implement min-p, frequency/presence/windowed penalties, DRY, XTC or EOT bias",
        ));
    }
    let max = spec
        .max_tokens
        .or(settings.stopping.max_tokens)
        .unwrap_or(512);
    let max_length = prompt_tokens.checked_add(max)
        .filter(|n| *n <= context as usize && max > 0)
        .ok_or_else(|| ExecError::ContextOverflow(format!("ONNX prompt ({prompt_tokens}) plus output budget ({max}) exceeds context ({context})")))?;
    let temperature = sampling.temperature.unwrap_or(1.0);
    let mut options = vec![("batch_size", 1.0), ("max_length", max_length as f64)];
    if temperature > 0.0 {
        options.push(("temperature", temperature as f64));
    }
    if let Some(top_p) = sampling.top_p {
        options.push(("top_p", top_p as f64));
    }
    if let Some(top_k) = sampling.top_k {
        options.push(("top_k", top_k.max(0) as f64));
    }
    if let Some(repeat) = sampling.penalty_repeat {
        options.push(("repetition_penalty", repeat as f64));
    }
    if let Some(seed) = spec.seed.or(sampling.seed.map(u64::from)) {
        if seed > u32::MAX as u64 {
            return Err(ExecError::InvalidArg("ONNX GenAI seeds must fit u32"));
        }
        options.push(("random_seed", seed as f64));
    }
    Ok((options, temperature > 0.0))
}

pub(super) fn guidance(grammar: &GrammarSpec) -> (&'static str, String) {
    match grammar {
        GrammarSpec::JsonObject => ("json_schema", r#"{"type":"object"}"#.into()),
        GrammarSpec::JsonSchema(schema) => ("json_schema", schema.to_string()),
        GrammarSpec::Regex(regex) => ("regex", regex.clone()),
        GrammarSpec::Lark(lark) => ("lark_grammar", lark.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundle_requires_configured_graph_and_bounded_context() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("genai_config.json");
        let write = |filename: &str| {
            std::fs::write(&config, serde_json::json!({
            "model": {"type":"qwen3_5_text", "context_length":128, "decoder":{"filename":filename}}
        }).to_string()).unwrap()
        };
        write("model.onnx");
        assert!(Facts::read(directory.path(), None).is_err());
        std::fs::write(directory.path().join("model.onnx"), b"graph").unwrap();
        assert_eq!(Facts::read(directory.path(), Some(64)).unwrap().context, 64);
        assert!(Facts::read(directory.path(), Some(129)).is_err());
        assert!(Facts::read(directory.path(), Some(0)).is_err());
        assert!(Facts::read(&directory.path().join("other.onnx"), None).is_err());
        write("../model.onnx");
        assert!(Facts::read(directory.path(), None).is_err());
    }
    #[test]
    fn output_budget_counts_prompt_and_refuses_overflow() {
        let spec = GenSpec {
            max_tokens: Some(20),
            temperature: Some(0.0),
            ..Default::default()
        };
        let (options, sample) = search_options(&Settings::default(), &spec, 30, 64).unwrap();
        assert_eq!(
            options
                .iter()
                .find(|(name, _)| *name == "max_length")
                .unwrap()
                .1,
            50.0
        );
        assert!(!sample);
        assert!(search_options(&Settings::default(), &spec, 50, 64).is_err());
    }
    #[test]
    fn sampler_refuses_options_instead_of_ignoring_them() {
        let spec = GenSpec {
            min_p: Some(0.1),
            ..Default::default()
        };
        assert!(matches!(
            search_options(&Settings::default(), &spec, 1, 1024),
            Err(ExecError::FeatureUnsupported(_))
        ));
        let spec = GenSpec {
            seed: Some(u32::MAX as u64 + 1),
            ..Default::default()
        };
        assert!(search_options(&Settings::default(), &spec, 1, 1024).is_err());
    }
}
