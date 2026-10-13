mod frontend;
mod pack;
use super::*;
use ort::memory::Allocator;
use ort::{session::Session, value::Tensor};
use std::time::Instant;

pub(crate) struct WhistleBackend {
    encoder: Session,
    decoder: Session,
    pack: pack::Pack,
    #[cfg(test)]
    trace_directory: Option<std::path::PathBuf>,
}
fn execution(e: impl std::fmt::Display) -> SpeechError {
    SpeechError::Execution(e.to_string())
}
fn invalid(message: &str) -> SpeechError {
    SpeechError::InvalidOutput(message.into())
}
fn extract(
    output: &ort::session::SessionOutputs<'_>,
    name: &str,
    expected: &[i64],
) -> Result<Vec<f32>> {
    let value = output
        .get(name)
        .ok_or_else(|| invalid("missing Whistle graph output"))?;
    let (shape, values) = value.try_extract_tensor::<f32>().map_err(execution)?;
    if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid Whistle graph output shape or values"));
    }
    Ok(values.to_vec())
}
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
impl WhistleBackend {
    pub(crate) fn load(bundle: &SpeechBundle, options: &LoadOptions) -> Result<Self> {
        crate::onnx_runtime::initialize(options.native_library.as_deref())
            .map_err(SpeechError::BackendUnavailable)?;
        let load = |name: &str| -> Result<Session> {
            Session::builder()
                .map_err(execution)?
                .with_intra_threads(options.intra_threads)
                .map_err(execution)?
                .with_inter_threads(1)
                .map_err(execution)?
                .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
                .map_err(execution)?
                .commit_from_file(bundle.directory().join("onnx").join(name))
                .map_err(execution)
        };
        Ok(Self {
            encoder: load("encoder.onnx")?,
            decoder: load("decoder.onnx")?,
            pack: pack::Pack::open(bundle.directory())?,
            #[cfg(test)]
            trace_directory: None,
        })
    }
    pub(crate) fn run(
        &mut self,
        audio: Audio,
        options: &TranscribeOptions,
        request: &TranscriptionSettings,
    ) -> Result<DetailedTranscription> {
        options.check()?;
        request.validate()?;
        let started = Instant::now();
        let (mel, frames) = frontend::features(&audio.samples, &self.pack.filters, options)?;
        #[cfg(test)]
        trace(self.trace_directory.as_deref(), "mel", &mel);
        let frontend_micros = super::worker::micros(started);
        options.check()?;
        let started = Instant::now();
        let encoded = self
            .encoder
            .run(ort::inputs!["mel"=>Tensor::from_array(([frames,80],mel)).map_err(execution)?])
            .map_err(execution)?;
        let value = encoded
            .get("cross_k")
            .ok_or_else(|| invalid("missing cross_k"))?;
        let (shape, _) = value.try_extract_tensor::<f32>().map_err(execution)?;
        if shape.len() != 4 || shape[2] < 1 || shape[2] > 750 {
            return Err(invalid("invalid encoder length"));
        }
        let steps = shape[2] as usize;
        let cross_k = extract(&encoded, "cross_k", &[8, 8, steps as i64, 48])?;
        let cross_v = extract(&encoded, "cross_v", &[8, 8, steps as i64, 64])?;
        #[cfg(test)]
        {
            trace(self.trace_directory.as_deref(), "cross_k", &cross_k);
            trace(self.trace_directory.as_deref(), "cross_v", &cross_v);
        }
        drop(encoded);
        let cross_k = Tensor::from_array(([8, 8, steps, 48], cross_k)).map_err(execution)?;
        let cross_v = Tensor::from_array(([8, 8, steps, 64], cross_v)).map_err(execution)?;
        let encoder_micros = super::worker::micros(started);
        options.check()?;
        let started = Instant::now();
        let mut previous = vec![0f32; 8 * 2 * 608];
        let mut past_k = Vec::new();
        let mut past_v = Vec::new();
        let mut history = vec![2u32];
        let mut tokens = Vec::new();
        let mut language = request.language;
        let mut detected_language = None;
        let max_tokens = request
            .max_tokens
            .unwrap_or((16 + frames.div_ceil(100) * 12).min(320));
        let mut completion = Completion::EndOfSpeech;
        // The final step observes EOS after the last allowed text token, matching
        // the oracle, but partial text is never returned as ordinary success.
        for position in 0..=max_tokens + 1 {
            options.check()?;
            let feature = self.pack.features(&history, position);
            #[cfg(test)]
            if position < 3 {
                trace(
                    self.trace_directory.as_deref(),
                    &format!("step{position}-efeat"),
                    &feature,
                );
            }
            // ORT's raw-vector constructor rejects zero dimensions. The native
            // allocator supports the empty initial KV tensors required by Whistle.
            let key = if position == 0 {
                Tensor::<f32>::new(&Allocator::default(), [8usize, 2, 0, 48]).map_err(execution)?
            } else {
                Tensor::from_array(([8, 2, position, 48], past_k)).map_err(execution)?
            };
            let value = if position == 0 {
                Tensor::<f32>::new(&Allocator::default(), [8usize, 2, 0, 64]).map_err(execution)?
            } else {
                Tensor::from_array(([8, 2, position, 64], past_v)).map_err(execution)?
            };
            let output=self.decoder.run(ort::inputs![
                "token"=>Tensor::from_array(([1],vec![history[position]as i64])).map_err(execution)?,
                "pos"=>Tensor::from_array(([1],vec![position as i64])).map_err(execution)?,
                "efeat"=>Tensor::from_array(([2,4,512],feature)).map_err(execution)?,
                "prev"=>Tensor::from_array(([8,2,608],previous)).map_err(execution)?,
                "past_k"=>key,
                "past_v"=>value,
                "cross_k"=>&cross_k,
                "cross_v"=>&cross_v,
            ]).map_err(execution)?;
            let logits = extract(&output, "logits", &[8199])?;
            previous = extract(&output, "prev_out", &[8, 2, 608])?;
            past_k = extract(&output, "present_k", &[8, 2, position as i64 + 1, 48])?;
            past_v = extract(&output, "present_v", &[8, 2, position as i64 + 1, 64])?;
            #[cfg(test)]
            if position < 3 {
                for (name, values) in [
                    ("logits", &logits),
                    ("prev", &previous),
                    ("k", &past_k),
                    ("v", &past_v),
                ] {
                    trace(
                        self.trace_directory.as_deref(),
                        &format!("step{position}-{name}"),
                        values,
                    );
                }
            }
            if position == 0 {
                let detected = SpeechLanguage::from_index(argmax(&logits[8192..]));
                detected_language = Some(detected);
                let lang = *language.get_or_insert(detected);
                history.push(8192 + lang.index() as u32);
                continue;
            }
            let token = argmax(&logits[..8192]);
            if token == 1 {
                break;
            }
            if tokens.len() >= max_tokens {
                completion = Completion::TokenLimit;
                break;
            }
            tokens.push(token as u32);
            history.push(token as u32);
        }
        options.check()?;
        if completion == Completion::TokenLimit && !request.allow_partial {
            return Err(SpeechError::TokenLimit);
        }
        Ok(DetailedTranscription {
            transcription: Transcription {
                text: self.pack.tokenizer.decode(&tokens)?,
                duration_seconds: audio.duration_seconds(),
                model_revision: WHISTLE_REVISION,
                manifest_sha256: super::whistle_bundle::LOCK_SHA256,
                provider: "CPUExecutionProvider",
                queue_micros: 0,
                frontend_micros,
                encoder_micros,
                decoder_micros: super::worker::micros(started),
                execution_micros: 0,
                token_ids: tokens,
            },
            language,
            detected_language,
            completion,
        })
    }
}

#[cfg(test)]
fn trace(directory: Option<&std::path::Path>, name: &str, values: &[f32]) {
    if let Some(directory) = directory {
        std::fs::write(
            directory.join(format!("{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Whistle bundle, pinned Node oracle and native runtime"]
    fn real_whistle_stage_probes_match_the_independent_oracle() {
        let bundle =
            WhistleBundle::open(std::env::var("GEN2_WHISTLE_BUNDLE").expect("bundle")).unwrap();
        let reference =
            std::path::PathBuf::from(std::env::var("GEN2_WHISTLE_REFERENCE").expect("reference"));
        let corpus =
            std::path::PathBuf::from(std::env::var("GEN2_WHISTLE_CORPUS").expect("corpus"));
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&reference).unwrap()).unwrap();
        let read = |p: std::path::PathBuf| {
            std::fs::read(p)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let audio = Audio::mono_16khz(read(
            corpus.join(receipt["records"][0]["pcm"].as_str().unwrap()),
        ))
        .unwrap();
        let mut backend = WhistleBackend::load(
            &bundle.0,
            &LoadOptions {
                native_library: Some(std::env::var("GEN2_ORT_LIBRARY").expect("runtime").into()),
                reservation_mb: 2048,
                ..Default::default()
            },
        )
        .unwrap();
        let output = tempfile::tempdir().unwrap();
        backend.trace_directory = Some(output.path().into());
        trace(
            Some(output.path()),
            "engram",
            &backend.pack.features(&[2, 8192, 42, 110, 77, 12, 400], 6),
        );
        backend
            .run(
                audio,
                &TranscribeOptions::default(),
                &TranscriptionSettings::default(),
            )
            .unwrap();
        let mut errors = Vec::new();
        for entry in std::fs::read_dir(output.path()).unwrap() {
            let name = entry.unwrap().file_name();
            let actual = read(output.path().join(&name));
            let expected = read(reference.parent().unwrap().join("probes").join(&name));
            assert_eq!(actual.len(), expected.len(), "shape: {name:?}");
            let mut maximum = 0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                let error = (a - b).abs();
                maximum = maximum.max(error);
                assert!(
                    a.is_finite() && b.is_finite() && error <= 1e-4 + 1e-4 * b.abs(),
                    "{name:?}: {a} vs {b}, error {error}"
                );
            }
            errors.push(serde_json::json!({"tensor":name.to_string_lossy(),"elements":actual.len(),"max_absolute_error":maximum,"atol":1e-4,"rtol":1e-4}));
        }
        if let Ok(path) = std::env::var("GEN2_WHISTLE_PROBE_RESULTS") {
            std::fs::write(path, serde_json::to_vec_pretty(&errors).unwrap()).unwrap();
        }
    }
}
