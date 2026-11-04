use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::{Error, Result};

const VOICE_SAMPLE_RATE: u32 = 16_000;
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 192_000;
const MAX_CHANNELS: u16 = 32;
const MAX_OUTPUT_SAMPLES: usize = VOICE_SAMPLE_RATE as usize;

pub(super) struct PcmNormalizer {
    channels: usize,
    frame_sum: f64,
    frame_fill: usize,
    mono: Vec<f32>,
    block_frames: usize,
    filter: Option<SincStream>,
}

impl PcmNormalizer {
    pub(super) fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
            return Err(Error::InvalidRequest(format!(
                "voice PCM sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz"
            )));
        }
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(Error::InvalidRequest(format!(
                "voice PCM must have between 1 and {MAX_CHANNELS} channels"
            )));
        }
        let filter = if sample_rate == VOICE_SAMPLE_RATE {
            None
        } else {
            Some(SincStream::new(sample_rate)?)
        };
        let block_frames = filter
            .as_ref()
            .map_or(256, |filter| filter.resampler.input_frames_next());

        Ok(Self {
            channels: usize::from(channels),
            frame_sum: 0.0,
            frame_fill: 0,
            mono: Vec::with_capacity(block_frames),
            block_frames,
            filter,
        })
    }

    pub(super) fn push(
        &mut self,
        interleaved: &[f32],
        mut emit: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        if interleaved
            .iter()
            .any(|sample| !(-1.0..=1.0).contains(sample))
        {
            return Err(Error::InvalidRequest(
                "voice PCM samples must be finite and between -1 and 1".into(),
            ));
        }
        if self.filter.is_none() && self.channels == 1 {
            return emit_chunks(interleaved, &mut emit);
        }
        if let Some(filter) = &mut self.filter {
            filter.input_frames +=
                (self.frame_fill as u128 + interleaved.len() as u128) / self.channels as u128;
        }

        for &sample in interleaved {
            self.frame_sum += f64::from(sample);
            self.frame_fill += 1;
            if self.frame_fill == self.channels {
                self.mono
                    .push((self.frame_sum / self.channels as f64) as f32);
                self.frame_sum = 0.0;
                self.frame_fill = 0;
                if self.mono.len() == self.block_frames {
                    self.emit_block(&mut emit)?;
                }
            }
        }
        if self.filter.is_none() {
            self.emit_block(&mut emit)?;
        }
        Ok(())
    }

    pub(super) fn finish(mut self, mut emit: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
        if self.frame_fill != 0 {
            return Err(Error::InvalidRequest(
                "voice PCM ended with an incomplete interleaved frame".into(),
            ));
        }
        if let Some(mut filter) = self.filter.take() {
            while filter.remaining() != 0 {
                // Zero-fill the filter window. The callback receives only the
                // output frames covered by the actual input duration.
                self.mono.resize(self.block_frames, 0.0);
                filter.process(&self.mono, &mut emit)?;
                self.mono.clear();
            }
        }
        Ok(())
    }

    fn emit_block(&mut self, emit: &mut impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
        if let Some(filter) = &mut self.filter {
            filter.process(&self.mono, emit)?;
        } else {
            emit_chunks(&self.mono, emit)?;
        }
        self.mono.clear();
        Ok(())
    }
}

struct SincStream {
    input_rate: u32,
    input_frames: u128,
    emitted_frames: u128,
    resampler: SincFixedIn<f32>,
    output: Vec<f32>,
}

impl SincStream {
    fn new(input_rate: u32) -> Result<Self> {
        const TAPS: usize = 96;
        let ratio = f64::from(VOICE_SAMPLE_RATE) / f64::from(input_rate);
        let sinc_len = (TAPS * input_rate.max(VOICE_SAMPLE_RATE) as usize)
            .div_ceil(VOICE_SAMPLE_RATE as usize);
        let window = WindowFunction::BlackmanHarris2;
        let parameters = SincInterpolationParameters {
            sinc_len,
            f_cutoff: rubato::calculate_cutoff(TAPS, window),
            oversampling_factor: 64,
            interpolation: SincInterpolationType::Linear,
            window,
        };
        let block_frames = (2 * sinc_len).max(256);
        let resampler =
            SincFixedIn::new(ratio, 1.0, parameters, block_frames, 1).map_err(resample_error)?;
        Ok(Self {
            input_rate,
            input_frames: 0,
            emitted_frames: 0,
            output: vec![0.0; resampler.output_frames_max()],
            resampler,
        })
    }

    fn remaining(&self) -> u128 {
        self.input_frames * u128::from(VOICE_SAMPLE_RATE) / u128::from(self.input_rate)
            - self.emitted_frames
    }

    fn process(&mut self, mono: &[f32], emit: &mut impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
        let (_, written) = self
            .resampler
            .process_into_buffer(&[mono], &mut [self.output.as_mut_slice()], None)
            .map_err(resample_error)?;
        if written == 0 {
            return Err(resample_error(
                "the filter produced no PCM for a full block",
            ));
        }

        // SincFixedIn begins at one output period (62.5 µs here). Its filter
        // lookahead holds back output until it has the next input frames.
        let count = self.remaining().min(written as u128) as usize;
        let output = &mut self.output[..count];
        for sample in output.iter_mut() {
            if !sample.is_finite() {
                return Err(resample_error(
                    "the filter produced a non-finite PCM sample",
                ));
            }
            *sample = sample.clamp(-1.0, 1.0);
        }
        emit_chunks(output, emit)?;
        self.emitted_frames += count as u128;
        Ok(())
    }
}

fn emit_chunks(samples: &[f32], emit: &mut impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
    for chunk in samples.chunks(MAX_OUTPUT_SAMPLES) {
        emit(chunk)?;
    }
    Ok(())
}

fn resample_error(error: impl std::fmt::Display) -> Error {
    Error::Generation {
        code: "audio_resample".into(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(
        rate: u32,
        channels: u16,
        chunks: impl IntoIterator<Item = impl AsRef<[f32]>>,
    ) -> (Vec<f32>, usize) {
        let mut normalizer = PcmNormalizer::new(rate, channels).unwrap();
        let mut output = Vec::new();
        for chunk in chunks {
            normalizer
                .push(chunk.as_ref(), |pcm| {
                    assert!(!pcm.is_empty() && pcm.len() <= MAX_OUTPUT_SAMPLES);
                    output.extend_from_slice(pcm);
                    Ok(())
                })
                .unwrap();
        }
        let before_finish = output.len();
        normalizer
            .finish(|pcm| {
                output.extend_from_slice(pcm);
                Ok(())
            })
            .unwrap();
        (output, before_finish)
    }

    #[test]
    fn resampling_is_independent_of_chunks_even_inside_stereo_frames() {
        let frames = 11_025 + 23;
        let stereo: Vec<_> = (0..frames)
            .flat_map(|i| [(i as f32 * 0.14).sin(), (i as f32 * 0.24).sin()])
            .collect();
        let (one_push, _) = normalize(44_100, 2, [&stereo]);
        let (fragmented, before_finish) = normalize(44_100, 2, stereo.chunks(37));
        assert_eq!(fragmented, one_push);
        assert_eq!(fragmented.len(), frames * 16_000 / 44_100);
        assert!(before_finish > 3_500 && before_finish < fragmented.len());
    }

    #[test]
    fn flushes_real_tail_at_the_correct_time() {
        let mut mono = vec![0.0; 3_003];
        mono[3] = 1.0;
        let (output, _) = normalize(48_000, 1, mono.chunks(107));
        assert!(output[0] > 0.2);
        assert!(output[1..].iter().all(|sample| sample.abs() < output[0]));

        let mut mono = vec![0.0; 3_003];
        mono[3_000] = 1.0;
        let (output, before_finish) = normalize(48_000, 1, mono.chunks(107));
        assert_eq!(output.len(), 1_001);
        let peak = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, 999);
        assert!(before_finish < peak);

        let (short, before_finish) = normalize(8_000, 1, [[0.5; 7]]);
        assert_eq!(before_finish, 0);
        assert_eq!(short.len(), 14);
        assert!(short[6] > 0.3);
    }

    #[test]
    fn downmixes_without_shifting_channels_at_chunk_boundaries() {
        let (mono, _) = normalize(16_000, 2, [&[1.0, 0.0, -1.0][..], &[1.0][..]]);
        assert_eq!(mono, [0.5, 0.0]);
        let (mono, _) = normalize(16_000, 1, [[0.25; MAX_OUTPUT_SAMPLES + 1]]);
        assert_eq!(mono, [0.25; MAX_OUTPUT_SAMPLES + 1]);
    }

    #[test]
    fn rejects_invalid_formats_samples_and_partial_final_frame() {
        assert!(matches!(
            PcmNormalizer::new(0, 1),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            PcmNormalizer::new(48_000, 0),
            Err(Error::InvalidRequest(_))
        ));

        let mut normalizer = PcmNormalizer::new(16_000, 2).unwrap();
        for sample in [f32::NAN, f32::INFINITY, 1.1, -1.1] {
            assert!(matches!(
                normalizer.push(&[0.0, sample], |_| panic!("invalid PCM was emitted")),
                Err(Error::InvalidRequest(_))
            ));
        }
        normalizer
            .push(&[1.0], |_| panic!("incomplete frame was emitted"))
            .unwrap();
        assert!(matches!(
            normalizer.finish(|_| panic!("incomplete frame was emitted")),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn downsampling_removes_tones_that_would_alias_into_speech() {
        let filtered_level = |hz: f32| {
            let mono: Vec<_> = (0..4_800)
                .map(|i| (std::f32::consts::TAU * hz * i as f32 / 48_000.0).sin())
                .collect();
            let (output, _) = normalize(48_000, 1, mono.chunks(107));
            let center = &output[200..1_400];
            (center.iter().map(|sample| sample * sample).sum::<f32>() / center.len() as f32).sqrt()
        };

        assert!(filtered_level(1_000.0) > 0.65);
        assert!(filtered_level(12_000.0) < 0.01);
    }

    #[test]
    fn full_scale_steps_stay_inside_the_pcm_range() {
        let mut mono = vec![-1.0; 1_000];
        mono.resize(2_000, 1.0);
        let (output, _) = normalize(48_000, 1, mono.chunks(107));
        assert!(output.contains(&-1.0) && output.contains(&1.0));
        assert!(output.iter().all(|sample| (-1.0..=1.0).contains(sample)));
    }
}
