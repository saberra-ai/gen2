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
