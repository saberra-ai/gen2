//! The pinned ONNX ASR NumPy frontend, with its original filter coefficients.
use super::{Result, SpeechError};
use rustfft::{FftPlanner, num_complex::Complex};

pub(crate) fn features(wave: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    features_with_observer(wave, |_, _| {})
}

fn features_with_observer(
    wave: &[f32],
    mut observe: impl FnMut(&'static str, &[f32]),
) -> Result<(Vec<f32>, usize, usize)> {
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
        let mut power = [0f32; 257];
        for (bin, value) in input.iter().take(257).enumerate() {
            let magnitude = value.norm() as f32;
            power[bin] = magnitude * magnitude;
        }
        observe("power", &power);
        for (bin, power) in power.into_iter().enumerate() {
            for band in 0..128 {
                mel[frame * 128 + band] =
                    power.mul_add(filters[bin * 128 + band], mel[frame * 128 + band]);
            }
        }
        observe("mel", &mel[frame * 128..(frame + 1) * 128]);
        for band in 0..128 {
            mel[frame * 128 + band] = numpy_log(mel[frame * 128 + band] + 2f32.powi(-24));
        }
        observe("log-mel", &mel[frame * 128..(frame + 1) * 128]);
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

// NumPy 2.5.4's AVX2/AVX512 float32 log uses a fused rational approximation,
// not the platform's scalar logf. Preserve that rounding on x86 with FMA;
// ARM NumPy uses scalar logf. Source: loops_exponent_log.dispatch.c.src and
// npy_simd_data.h, v2.5.4 (BSD-3-Clause; assets/bark/NUMPY-LICENSE).
fn numpy_log(value: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return numpy_fma_log(value);
    }
    value.ln()
}

#[cfg(any(target_arch = "x86_64", test))]
fn numpy_fma_log(value: f32) -> f32 {
    // Frontend energies are finite, positive and at least 2^-24 (no denormals).
    let bits = value.to_bits();
    let mut exponent = ((bits >> 23) as i32 - 126) as f32;
    let mut x = f32::from_bits((bits & 0x7fffff) | (126 << 23));
    if x <= std::f32::consts::FRAC_1_SQRT_2 {
        x += x;
        exponent -= 1.;
    }
    x -= 1.;
    let numerator = [0.38088378f32, 1.4800006, 2.1126776, 1., 0.]
        .into_iter()
        .fold(0.025899792f32, |p, coefficient| p.mul_add(x, coefficient));
    let denominator = [0.15464763f32, 0.9864943, 2.453006, 2.6126776, 1.]
        .into_iter()
        .fold(0.0058750953f32, |p, coefficient| p.mul_add(x, coefficient));
    exponent.mul_add(std::f32::consts::LN_2, numerator / denominator)
}

#[cfg(test)]
mod tests {
    #[test]
    fn numpy_log_approximation_is_bounded_on_frontend_domain() {
        for exponent in -24..100 {
            for mantissa in 0..128 {
                let x = 2f32.powi(exponent) * (1. + mantissa as f32 / 128.);
                assert!((super::numpy_fma_log(x) - x.ln()).abs() <= 1e-5);
            }
        }
    }
    #[test]
    #[ignore = "requires GEN2_BARK_REFERENCE containing independently exported NumPy tensors"]
    fn native_frontend_matches_pinned_numpy_reference() {
        let root = std::path::PathBuf::from(
            std::env::var("GEN2_BARK_REFERENCE").expect("reference corpus"),
        );
        for index in [0, 1, 2, 13, 34, 41, 43, 249] {
            let wave = super::super::Audio::from_wav(root.join(format!("{index:03}.wav"))).unwrap();
            let expected = std::fs::read(root.join(format!("{index:03}.features.f32"))).unwrap();
            let expected: Vec<_> = expected
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let mut stages = std::collections::BTreeMap::<&str, Vec<f32>>::new();
            let dump = std::env::var_os("GEN2_BARK_DUMP").is_some();
            let (actual, _, _) = super::features_with_observer(wave.samples(), |name, values| {
                if dump {
                    stages.entry(name).or_default().extend_from_slice(values);
                }
            })
            .unwrap();
            if dump {
                let bytes: Vec<u8> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(root.join(format!("{index:03}.native.features.f32")), bytes)
                    .unwrap();
                for (name, values) in stages {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(root.join(format!("{index:03}.native.{name}.f32")), bytes)
                        .unwrap();
                }
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
