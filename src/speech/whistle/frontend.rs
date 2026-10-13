//! Arithmetic and window placement follow the pinned JavaScript CPU oracle.
use crate::speech::{Result, SpeechError, TranscribeOptions};
pub(super) fn features(
    pcm: &[f32],
    filters: &[f32],
    options: &TranscribeOptions,
) -> Result<(Vec<f32>, usize)> {
    if pcm.len() < 640 {
        return Err(SpeechError::InvalidAudio(
            "Whistle requires at least 40 ms of audio".into(),
        ));
    }
    let chunks = pcm.len() / 320;
    let mut rms: Vec<f64> = pcm
        .as_chunks::<320>()
        .0
        .iter()
        .map(|s| (s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / 320.).sqrt())
        .collect();
    rms.sort_by(f64::total_cmp);
    let percentile = rms[((chunks - 1) as f64 * 0.99).floor() as usize];
    let gain = if percentile > 1e-6 {
        0.1 / percentile
    } else {
        1.
    };
    let frames = pcm.len() / 160;
    let window: Vec<f64> = (0..400)
        .map(|n| 0.5 - 0.5 * (2. * std::f64::consts::PI * n as f64 / 399.).cos())
        .collect();
    let cos: Vec<f64> = (0..256)
        .map(|n| (-2. * std::f64::consts::PI * n as f64 / 512.).cos())
        .collect();
    let sin: Vec<f64> = (0..256)
        .map(|n| (-2. * std::f64::consts::PI * n as f64 / 512.).sin())
        .collect();
    let mut log = vec![0f64; frames * 80];
    let mut re = [0f64; 512];
    let mut im = [0f64; 512];
    let mut power = [0f64; 257];
    for t in 0..frames {
        if t % 32 == 0 {
            options.check()?;
        }
        re.fill(0.);
        im.fill(0.);
        for i in 0..400 {
            let k = t as isize * 160 - 200 + i as isize;
            if k >= 0 && (k as usize) < pcm.len() {
                re[i] = pcm[k as usize] as f64 * gain * window[i];
            }
        }
        for i in 0usize..512 {
            let j = i.reverse_bits() >> (usize::BITS - 9);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= 512 {
            let half = size / 2;
            let step = 512 / size;
            for start in (0..512).step_by(size) {
                for j in 0..half {
                    let k = j * step;
                    let a = start + j;
                    let b = a + half;
                    let tr = re[b] * cos[k] - im[b] * sin[k];
                    let ti = re[b] * sin[k] + im[b] * cos[k];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
        for k in 0..257 {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
        for m in 0..80 {
            let mut sum = 0.;
            for k in 0..257 {
                sum += power[k] * filters[k * 80 + m] as f64;
            }
            log[t * 80 + m] = (sum + 2f64.powi(-24)).ln();
        }
    }
    let mut mel = vec![0f32; frames * 80];
    for m in 0..80 {
        let mut mean = 0.;
        for t in 0..frames {
            mean += log[t * 80 + m];
        }
        mean /= frames as f64;
        let mut variance = 0.;
        for t in 0..frames {
            let d = log[t * 80 + m] - mean;
            variance += d * d;
        }
        let sd = (variance / (frames - 1) as f64).sqrt();
        for t in 0..frames {
            mel[t * 80 + m] = ((log[t * 80 + m] - mean) / (sd + 1e-5)) as f32;
        }
    }
    if mel.iter().any(|x| !x.is_finite()) {
        return Err(SpeechError::InvalidOutput(
            "non-finite Whistle frontend".into(),
        ));
    }
    Ok((mel, frames))
}
