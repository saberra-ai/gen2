//! The pinned ONNX ASR NumPy frontend, with its original filter coefficients.
use super::{Result, SpeechError};
use rustfft::{FftPlanner, num_complex::Complex};

pub(crate) fn features(wave: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    let valid = wave.len() / 160;
    let frames = valid + 1;
    let mut emphasized = vec![0f32; wave.len() + 512];
    for (i, &sample) in wave.iter().enumerate() {
        emphasized[i + 256] = sample - 0.97f32 * i.checked_sub(1).map_or(0., |j| wave[j]);
    }
    let window: Vec<f64> = (0..512)
        .map(|i| {
            if (56..456).contains(&i) {
                0.5 - 0.5 * (std::f64::consts::TAU * (i - 56) as f64 / 399.).cos()
            } else {
                0.
            }
        })
        .collect();
    let filters: Vec<f32> = include_bytes!("../../assets/bark/nemo128.f32")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(512);
    let mut input = vec![Complex::default(); 512];
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut mel = vec![0f32; frames * 128];
    for frame in 0..frames {
        for i in 0..512 {
            input[i] = Complex::new(emphasized[frame * 160 + i] as f64 * window[i], 0.);
        }
        fft.process_with_scratch(&mut input, &mut scratch);
        for (bin, value) in input.iter().take(257).enumerate() {
            let magnitude = value.norm() as f32;
            let power = magnitude * magnitude;
            for band in 0..128 {
                mel[frame * 128 + band] =
                    power.mul_add(filters[bin * 128 + band], mel[frame * 128 + band]);
            }
        }
        for band in 0..128 {
            mel[frame * 128 + band] = (mel[frame * 128 + band] + 2f32.powi(-24)).ln();
        }
    }
    let mut normalized = vec![0f32; frames * 128];
    for band in 0..128 {
        let mut mean = 0f32;
        for frame in 0..valid {
            mean += mel[frame * 128 + band];
        }
        mean /= valid as f32;
        let mut variance = 0f32;
        for frame in 0..valid {
            variance += (mel[frame * 128 + band] - mean).powi(2);
        }
        variance /= (valid - 1) as f32;
        for frame in 0..valid {
            normalized[band * frames + frame] =
                (mel[frame * 128 + band] - mean) / (variance.sqrt() + 1e-5);
        }
    }
    if normalized.iter().any(|x| !x.is_finite()) {
        return Err(SpeechError::InvalidOutput(
            "non-finite frontend features".into(),
        ));
    }
    Ok((normalized, frames, valid))
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires GEN2_BARK_REFERENCE containing independently exported NumPy tensors"]
    fn native_frontend_matches_pinned_numpy_reference() {
        let root = std::path::PathBuf::from(
            std::env::var("GEN2_BARK_REFERENCE").expect("reference corpus"),
        );
        for index in [0, 1, 2, 34] {
            let wave = super::super::Audio::from_wav(root.join(format!("{index:03}.wav"))).unwrap();
            let expected = std::fs::read(root.join(format!("{index:03}.features.f32"))).unwrap();
            let expected: Vec<_> = expected
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let (actual, _, _) = super::features(wave.samples()).unwrap();
            if std::env::var_os("GEN2_BARK_DUMP").is_some() {
                let bytes: Vec<u8> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(root.join(format!("{index:03}.native.features.f32")), bytes)
                    .unwrap();
            }
            assert_eq!(actual.len(), expected.len());
            let mut max_error = 0f32;
            for (a, b) in actual.iter().zip(&expected) {
                max_error = max_error.max((a - b).abs());
                assert!(
                    (a - b).abs() <= 1e-4 + 1e-4 * b.abs(),
                    "frontend error {}: {a} vs {b}",
                    (a - b).abs()
                );
            }
            eprintln!("frontend clip {index} max error {max_error}");
        }
    }
}
