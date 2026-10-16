use gen2::speech::Cancellation;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub const RATE: u32 = 16_000;
pub const MAX_SECONDS: u32 = 7200;
pub const MAX_BYTES: u64 = 256 * 1024 * 1024;
#[derive(Clone, Serialize)]
pub struct AudioFile {
    pub path: String,
    pub name: String,
    pub bytes: u64,
    pub duration: Option<f64>,
    pub sha256: String,
}

pub fn info(path: &Path) -> Result<AudioFile, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_BYTES {
        return Err("Choose an audio file under 256 MB.".into());
    }
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !["wav", "m4a", "mp3", "ogg", "webm", "flac", "aac", "mp4"].contains(&extension.as_str()) {
        return Err("Choose a supported audio file.".into());
    }
    let mut source = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut bytes = 0u64;
    loop {
        use std::io::Read;
        let count = source.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        if bytes > MAX_BYTES {
            return Err("Choose an audio file under 256 MB.".into());
        }
        hash.update(&buffer[..count]);
    }
    Ok(AudioFile {
        path: path.to_string_lossy().into_owned(),
        name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        bytes,
        duration: None,
        sha256: hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    })
}

#[cfg(target_os = "macos")]
fn source_channels(input: &Path, cancellation: &Cancellation) -> Result<Option<u16>, String> {
    if let Ok(reader) = hound::WavReader::open(input) {
        return Ok(Some(reader.spec().channels));
    }
    let mut child = Command::new("/usr/bin/afinfo")
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Audio inspection unavailable: {e}"))?;
    let started = Instant::now();
    loop {
        if cancellation.is_cancelled() || started.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(if cancellation.is_cancelled() {
                "Transcription cancelled."
            } else {
                "Audio inspection timed out."
            }
            .into());
        }
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let result = child.wait_with_output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Ok(None);
    }
    Ok(String::from_utf8_lossy(&result.stdout)
        .split("Data format:")
        .nth(1)
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok()))
}

pub fn convert(input: &Path, output: &Path, cancellation: &Cancellation) -> Result<(), String> {
    if cancellation.is_cancelled() {
        return Err("Transcription cancelled.".into());
    }
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("/usr/bin/afconvert");
        c.args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"]);
        // CoreAudio rejects its downmix property on mono inputs (-50).
        // Preserve true downmixing for multi-channel sources; do not silently
        // select one input channel as a fallback.
        #[cfg(target_os = "macos")]
        let channels = source_channels(input, cancellation)?;
        #[cfg(not(target_os = "macos"))]
        let channels: Option<u16> = None;
        if channels != Some(1) {
            c.arg("--mix");
        }
        c.arg(input).arg(output);
        c
    } else {
        let mut c = Command::new("ffmpeg");
        c.args(["-nostdin", "-v", "error", "-i"])
            .arg(input)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y"])
            .arg(output);
        c
    };
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Audio converter unavailable: {e}"))?;
    let start = Instant::now();
    loop {
        let too_large = fs::metadata(output)
            .is_ok_and(|m| m.len() > MAX_SECONDS as u64 * RATE as u64 * 2 + 1_048_576);
        if cancellation.is_cancelled() || start.elapsed() > Duration::from_secs(300) || too_large {
            let _ = child.kill();
            let _ = child.wait();
            return Err(if cancellation.is_cancelled() {
                "Transcription cancelled."
            } else if too_large {
                "Choose a recording up to 2 hours."
            } else {
                "Audio conversion timed out."
            }
            .into());
        }
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = child.wait_with_output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!(
            "Could not decode audio: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    Ok(())
}

pub fn cut(samples: &[i16]) -> usize {
    let maximum = RATE as usize * 60;
    if samples.len() <= maximum {
        return samples.len();
    }
    let mut end = maximum;
    let mut lowest = 0.0001;
    for window in ((maximum - RATE as usize * 8)..=(maximum - 1600)).step_by(800) {
        let energy = samples[window..window + 1600]
            .iter()
            .map(|x| (*x as f64 / 32768.).powi(2))
            .sum::<f64>()
            / 1600.;
        if energy < lowest {
            lowest = energy;
            end = window + 800;
        }
    }
    end.min(samples.len() - 320)
}

pub fn plan(path: &Path, cancellation: &Cancellation) -> Result<(u32, Vec<(u32, u32)>), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let total = reader.duration();
    if spec.channels != 1
        || spec.sample_rate != RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err("Expected mono 16 kHz PCM16 audio.".into());
    }
    if !(320..=MAX_SECONDS * RATE).contains(&total) {
        return Err("Choose audio between 20 milliseconds and 2 hours.".into());
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < total {
        if cancellation.is_cancelled() {
            return Err("Transcription cancelled.".into());
        }
        reader.seek(start).map_err(|e| e.to_string())?;
        let count = (total - start).min(RATE * 60 + 320);
        let samples = reader
            .samples::<i16>()
            .take(count as usize)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        if samples.len() != count as usize {
            return Err("Incomplete WAV audio.".into());
        }
        let length = cut(&samples) as u32;
        parts.push((start, length));
        start += length;
    }
    Ok((total, parts))
}

pub fn read_part(path: &Path, start: u32, count: u32) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    reader.seek(start).map_err(|e| e.to_string())?;
    let samples = reader
        .samples::<i16>()
        .take(count as usize)
        .map(|x| x.map(|v| v as f32 / 32768.))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if samples.len() != count as usize {
        return Err("Incomplete WAV audio.".into());
    }
    Ok(samples)
}

pub fn resource_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;
    #[cfg(debug_assertions)]
    {
        let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.desktop-assets");
        if dev.join("bark").is_dir() || dev.join("nemotron").is_dir() {
            return Ok(dev);
        }
    }
    let root = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Could not locate app resources: {e}"))?;
    if root.join("bark").is_dir() || root.join("nemotron").is_dir() {
        return Ok(root);
    }
    Err("Bundled Bark model is missing. Rebuild Pnola with desktop:prepare.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_cover_all_samples() {
        let mut input = vec![16384; RATE as usize * 125];
        input[RATE as usize * 56..RATE as usize * 57].fill(0);
        let mut start = 0;
        let mut lengths = vec![];
        while start < input.len() {
            let remaining = &input[start..input.len().min(start + RATE as usize * 60 + 320)];
            let n = cut(remaining);
            assert!((320..=RATE as usize * 60).contains(&n));
            start += n;
            lengths.push(n);
        }
        assert_eq!(start, input.len());
        assert_eq!(lengths.len(), 3);
        assert!(lengths[0] >= RATE as usize * 56 && lengths[0] <= RATE as usize * 57);
    }
    #[test]
    fn tiny_tail_is_kept() {
        let input = vec![16384; RATE as usize * 60 + 1];
        let n = cut(&input);
        assert_eq!(input.len() - n, 320);
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn native_conversion_accepts_mono_and_keeps_the_right_stereo_channel() {
        let directory = tempfile::tempdir().unwrap();
        for channels in [1, 2] {
            let input = directory.path().join(format!("input-{channels}.wav"));
            let output = directory.path().join(format!("output-{channels}.wav"));
            let mut writer = hound::WavWriter::create(
                &input,
                hound::WavSpec {
                    channels,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for index in 0..3200 {
                if channels == 2 {
                    writer.write_sample(0i16).unwrap();
                }
                writer
                    .write_sample(if index % 32 < 16 { 10000i16 } else { -10000i16 })
                    .unwrap();
            }
            writer.finalize().unwrap();
            convert(&input, &output, &Cancellation::default()).unwrap();
            let mut reader = hound::WavReader::open(output).unwrap();
            assert_eq!(reader.spec().channels, 1);
            assert_eq!(reader.duration(), 3200);
            assert!(
                reader
                    .samples::<i16>()
                    .any(|sample| sample.unwrap().abs() > 1000)
            );
        }
    }
}
