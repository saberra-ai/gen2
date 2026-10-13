use super::*;
use ort::{session::Session, value::Tensor};
use std::time::Instant;

pub(crate) struct OnnxBackend {
    encoder: Session,
    decoder: Session,
    vocab: Vec<String>,
    blank: usize,
}
fn native(e: impl std::fmt::Display) -> SpeechError {
    SpeechError::Execution(e.to_string())
}
fn invalid(message: &str) -> SpeechError {
    SpeechError::InvalidOutput(message.into())
}

impl OnnxBackend {
    pub(crate) fn load(bundle: &SpeechBundle, options: &LoadOptions) -> Result<Self> {
        crate::onnx_runtime::initialize(options.native_library.as_deref())
            .map_err(SpeechError::BackendUnavailable)?;
        let load = |name: &str| -> Result<Session> {
            Session::builder()
                .map_err(native)?
                .with_intra_threads(options.intra_threads)
                .map_err(native)?
                .with_inter_threads(1)
                .map_err(native)?
                .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
                .map_err(native)?
                .commit_from_file(bundle.directory().join(name))
                .map_err(native)
        };
        let encoder = load("encoder-model.onnx")?;
        let decoder = load("decoder_joint-model.onnx")?;
        let text = std::fs::read_to_string(bundle.directory().join("vocab.txt")).map_err(native)?;
        let mut vocab = vec![String::new(); text.lines().count()];
        for line in text.lines() {
            let (token, id) = line
                .rsplit_once(' ')
                .ok_or_else(|| invalid("invalid vocabulary row"))?;
            let id: usize = id.parse().map_err(native)?;
            let slot = vocab
                .get_mut(id)
                .ok_or_else(|| invalid("invalid vocabulary ID"))?;
            if !slot.is_empty() {
                return Err(invalid("duplicate vocabulary ID"));
            }
            *slot = token.replace('\u{2581}', " ");
        }
        let blank = vocab
            .iter()
            .position(|s| s == "<blk>")
            .ok_or_else(|| invalid("missing blank token"))?;
        if vocab.len() != 8193 || vocab.iter().any(String::is_empty) {
            return Err(invalid("incompatible vocabulary"));
        }
        Ok(Self {
            encoder,
            decoder,
            vocab,
            blank,
        })
    }
    pub(crate) fn run(
        &mut self,
        audio: Audio,
        options: &TranscribeOptions,
    ) -> Result<Transcription> {
        options.check()?;
        let started = Instant::now();
        let (features, frames, valid) = super::frontend::features(&audio.samples)?;
        let frontend_micros = super::worker::micros(started);
        options.check()?;
        let started = Instant::now();
        let output = self.encoder.run(ort::inputs![
            "audio_signal" => Tensor::from_array(([1usize,128,frames], features)).map_err(native)?,
            "length" => Tensor::from_array(([1usize], vec![valid as i64])).map_err(native)?,
        ]).map_err(native)?;
        let (shape, encoded) = output["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(native)?;
        let (_, lengths) = output["encoded_lengths"]
            .try_extract_tensor::<i64>()
            .map_err(native)?;
        if shape.len() != 3
            || shape[0] != 1
            || shape[1] != 1024
            || shape[2] <= 0
            || shape[2] > 751
            || lengths.len() != 1
            || lengths[0] <= 0
            || lengths[0] > shape[2]
            || encoded.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("invalid encoder output"));
        }
        let encoded_frames = shape[2] as usize;
        let valid_frames = lengths[0] as usize;
        let encoded = encoded.to_vec();
        drop(output);
        let encoder_micros = super::worker::micros(started);
        options.check()?;
        let started = Instant::now();
        let mut state1 = vec![0f32; 1280];
        let mut state2 = vec![0f32; 1280];
        let mut tokens = Vec::<u32>::new();
        let mut t = 0usize;
        let mut emitted = 0usize;
        let mut iterations = 0usize;
        while t < valid_frames {
            options.check()?;
            iterations += 1;
            if iterations > valid_frames * 11 {
                return Err(invalid("TDT decoder made no progress"));
            }
            let frame: Vec<f32> = (0..1024)
                .map(|band| encoded[band * encoded_frames + t])
                .collect();
            let previous = tokens.last().copied().unwrap_or(self.blank as u32) as i32;
            let output = self.decoder.run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(([1usize,1024,1], frame)).map_err(native)?,
                "targets" => Tensor::from_array(([1usize,1], vec![previous])).map_err(native)?,
                "target_length" => Tensor::from_array(([1usize], vec![1i32])).map_err(native)?,
                "input_states_1" => Tensor::from_array(([2usize,1,640], state1.clone())).map_err(native)?,
                "input_states_2" => Tensor::from_array(([2usize,1,640], state2.clone())).map_err(native)?,
            ]).map_err(native)?;
            let (_, logits) = output["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(native)?;
            if logits.len() != self.vocab.len() + 5 || logits.iter().any(|x| !x.is_finite()) {
                return Err(invalid("invalid TDT logits"));
            }
            let token = argmax(&logits[..self.vocab.len()]);
            let duration = argmax(&logits[self.vocab.len()..]);
            if token != self.blank {
                let (shape1, next1) = output["output_states_1"]
                    .try_extract_tensor::<f32>()
                    .map_err(native)?;
                let (shape2, next2) = output["output_states_2"]
                    .try_extract_tensor::<f32>()
                    .map_err(native)?;
                if shape1.as_ref() != [2, 1, 640]
                    || shape2.as_ref() != [2, 1, 640]
                    || next1.iter().chain(next2).any(|x| !x.is_finite())
                {
                    return Err(invalid("invalid decoder recurrent state"));
                }
                state1.copy_from_slice(next1);
                state2.copy_from_slice(next2);
                tokens.push(token as u32);
                emitted += 1;
            }
            if duration > 0 {
                t += duration;
                emitted = 0;
            } else if token == self.blank || emitted == 10 {
                t += 1;
                emitted = 0;
            }
        }
        let raw = tokens
            .iter()
            .map(|&id| self.vocab[id as usize].as_str())
            .collect::<String>();
        // ONNX ASR's whitespace regex preserves a separator only before a word
        // character, and removes initial/trailing whitespace. No extra cleanup.
        let chars: Vec<char> = raw.chars().collect();
        let text: String = chars
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if !c.is_whitespace() {
                    Some(c)
                } else if i > 0
                    && chars
                        .get(i + 1)
                        .is_some_and(|c| c.is_alphanumeric() || *c == '_')
                {
                    Some(' ')
                } else {
                    None
                }
            })
            .collect();
        options.check()?;
        Ok(Transcription {
            text: text.trim().into(),
            duration_seconds: audio.duration_seconds(),
            model_revision: MODEL_REVISION,
            manifest_sha256: MANIFEST_SHA256,
            provider: "CPUExecutionProvider",
            queue_micros: 0,
            frontend_micros,
            encoder_micros,
            decoder_micros: super::worker::micros(started),
            execution_micros: 0,
            token_ids: tokens,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    fn floats(path: impl AsRef<std::path::Path>) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert!(bytes.as_chunks::<4>().1.is_empty());
        bytes.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect()
    }
    fn close(actual:&[f32],expected:&[f32]) {
        assert_eq!(actual.len(),expected.len());
        for (a,b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && (a-b).abs()<=1e-4+1e-4*b.abs(),"tensor {a} != {b}");
        }
    }
    #[test]
    #[ignore = "requires pinned GEN2_BARK_BUNDLE, GEN2_ORT_LIBRARY, GEN2_BARK_REFERENCE tensors"]
    fn native_graphs_match_independent_encoder_decoder_tensors() {
        let root=std::path::PathBuf::from(std::env::var("GEN2_BARK_REFERENCE").expect("reference"));
        let receipt:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
        let bundle=SpeechBundle::open(std::env::var("GEN2_BARK_BUNDLE").expect("bundle")).unwrap();
        let mut backend=OnnxBackend::load(&bundle,&LoadOptions { native_library:Some(std::env::var("GEN2_ORT_LIBRARY").unwrap().into()),..Default::default() }).unwrap();
        for index in 0..3 {
            let record=&receipt["records"][index];
            let features=floats(root.join(format!("{index:03}.features.f32")));
            let frames=features.len()/128;
            let valid=record["samples"].as_u64().unwrap() as i64/160;
            let output=backend.encoder.run(ort::inputs![
                "audio_signal"=>Tensor::from_array(([1usize,128,frames],features)).unwrap(),
                "length"=>Tensor::from_array(([1usize],vec![valid])).unwrap(),
            ]).unwrap();
            let (shape,encoded)=output["outputs"].try_extract_tensor::<f32>().unwrap();
            let encoded_frames=shape[2] as usize;
            let frozen=floats(root.join(format!("{index:03}.encoder.f32")));
            close(encoded,&frozen);
            assert_eq!(output["encoded_lengths"].try_extract_tensor::<i64>().unwrap().1[0],(valid+7)/8);
            drop(output);
            let frame:Vec<f32>=(0..1024).map(|band|frozen[band*encoded_frames]).collect();
            let output=backend.decoder.run(ort::inputs![
                "encoder_outputs"=>Tensor::from_array(([1usize,1024,1],frame)).unwrap(),
                "targets"=>Tensor::from_array(([1usize,1],vec![backend.blank as i32])).unwrap(),
                "target_length"=>Tensor::from_array(([1usize],vec![1i32])).unwrap(),
                "input_states_1"=>Tensor::from_array(([2usize,1,640],vec![0f32;1280])).unwrap(),
                "input_states_2"=>Tensor::from_array(([2usize,1,640],vec![0f32;1280])).unwrap(),
            ]).unwrap();
            let logits=output["outputs"].try_extract_tensor::<f32>().unwrap().1;
            close(&logits[..8193],&floats(root.join(format!("{index:03}.logits.f32"))));
            assert_eq!(argmax(&logits[8193..]),record["first_duration"].as_u64().unwrap() as usize);
            for (name,file) in [("output_states_1","state1"),("output_states_2","state2")] {
                close(output[name].try_extract_tensor::<f32>().unwrap().1,&floats(root.join(format!("{index:03}.{file}.f32"))));
            }
        }
    }
}
