use super::{Result, SpeechError};

/// Validated finite, normalized mono float32 samples at 16 kHz.
#[derive(Debug, Clone)]
pub struct Audio {
    pub(crate) samples: Vec<f32>,
}
impl Audio {
    /// Bark's supported envelope is 20 ms through 60 seconds inclusive.
    /// Extremely short inputs cannot be normalized by the reference frontend.
    pub fn mono_16khz(samples: Vec<f32>) -> Result<Self> {
        if !(320..=960_000).contains(&samples.len()) || samples.iter().any(|x| !x.is_finite()) {
            return Err(SpeechError::InvalidAudio(
                "use finite mono 16 kHz PCM between 20 ms and 60 seconds".into(),
            ));
        }
        Ok(Self { samples })
    }
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / 16_000.
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Decode PCM16 or float32 WAV at 8/16/44.1/48 kHz and average 1–8 channels.
    /// Files and decoded samples are bounded before allocation.
    #[cfg(feature = "backend-asr-onnx")]
    pub fn from_wav(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let invalid = |e: String| SpeechError::InvalidAudio(e);
        let file = std::fs::File::open(path).map_err(|e| invalid(e.to_string()))?;
        if file.metadata().map_err(|e| invalid(e.to_string()))?.len() > 128 * 1024 * 1024 {
            return Err(invalid("WAV exceeds 128 MiB input limit".into()));
        }
        let mut reader = hound::WavReader::new(std::io::BufReader::new(file))
            .map_err(|e| invalid(e.to_string()))?;
        let spec = reader.spec();
        let frames = reader.duration() as usize;
        validate_pcm_shape(frames, spec.sample_rate, spec.channels as usize)?;
        if reader.len() as usize != frames * spec.channels as usize {
            return Err(invalid("incomplete interleaved frame".into()));
        }
        let samples = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|x| x.map(|x| x as f32 / 32768.))
                .collect::<std::result::Result<Vec<_>, _>>(),
            (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect(),
            _ => {
                return Err(invalid(
                    "supported WAV encodings are PCM16 and float32".into(),
                ));
            }
        }
        .map_err(|e| invalid(e.to_string()))?;
        Self::from_interleaved(samples, spec.sample_rate, spec.channels as usize)
    }

    /// Normalize interleaved PCM using arithmetic downmixing. Rates other than
    /// 16 kHz require the opt-in `asr-resample-soxr` feature (libsoxr HQ).
    #[cfg(feature = "backend-asr-onnx")]
    pub fn from_interleaved(samples: Vec<f32>, rate: u32, channels: usize) -> Result<Self> {
        if channels == 0
            || !samples.len().is_multiple_of(channels)
            || samples.iter().any(|s| !s.is_finite())
        {
            return Err(SpeechError::InvalidAudio(
                "invalid/non-finite interleaved PCM".into(),
            ));
        }
        let frames = samples.len() / channels;
        validate_pcm_shape(frames, rate, channels)?;
        let mono: Vec<f32> = samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        if rate == 16000 {
            return Self::mono_16khz(mono);
        }
        #[cfg(feature = "asr-resample-soxr")]
        {
            if mono.iter().any(|s| !s.is_finite()) {
                return Err(SpeechError::InvalidAudio("downmix overflow".into()));
            }
            Self::mono_16khz(super::resample::mono(&mono, rate)?)
        }
        #[cfg(not(feature = "asr-resample-soxr"))]
        {
            Err(SpeechError::InvalidAudio("enable asr-resample-soxr for 8/44.1/48 kHz PCM, or normalize to 16 kHz in the host".into()))
        }
    }
}

#[cfg(feature = "backend-asr-onnx")]
fn validate_pcm_shape(frames: usize, rate: u32, channels: usize) -> Result<()> {
    if ![8000, 16000, 44100, 48000].contains(&rate)
        || !(1..=8).contains(&channels)
        || frames < rate as usize / 50
        || frames > rate as usize * 60
    {
        return Err(SpeechError::InvalidAudio(
            "use 20 ms–60 s PCM at 8/16/44.1/48 kHz with 1–8 channels".into(),
        ));
    }
    Ok(())
}
