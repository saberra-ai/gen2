use super::{DiarizationError, Result};

/// Validated mono 16 kHz PCM, independent of Bark's utterance limit.
#[derive(Debug, Clone)]
pub struct Audio {
    pub(crate) samples: Vec<f32>,
}
impl Audio {
    /// The initial native qualification envelope is 20 ms through 120 seconds.
    pub fn mono_16khz(samples: Vec<f32>) -> Result<Self> {
        if !(320..=120 * 16000).contains(&samples.len()) || samples.iter().any(|s| !s.is_finite()) {
            return Err(DiarizationError::InvalidAudio(
                "use finite mono 16 kHz PCM, 20 ms–120 seconds".into(),
            ));
        }
        Ok(Self { samples })
    }
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / 16000.
    }
    /// Decode only normalized mono 16 kHz PCM16/float32 WAV, with a bounded reader.
    #[cfg(feature = "backend-diar-nemo")]
    pub fn from_wav(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bad = |e: String| DiarizationError::InvalidAudio(e);
        let mut reader = hound::WavReader::open(path).map_err(|e| bad(e.to_string()))?;
        let spec = reader.spec();
        if spec.channels != 1 || spec.sample_rate != 16000 || reader.duration() > 120 * 16000 {
            return Err(bad("WAV must be mono 16 kHz, at most 120 seconds".into()));
        }
        let samples: std::result::Result<Vec<_>, _> =
            match (spec.sample_format, spec.bits_per_sample) {
                (hound::SampleFormat::Int, 16) => reader
                    .samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / 32768.))
                    .collect(),
                (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect(),
                _ => return Err(bad("WAV must use PCM16 or float32".into())),
            };
        Self::mono_16khz(samples.map_err(|e| bad(e.to_string()))?)
    }
}
