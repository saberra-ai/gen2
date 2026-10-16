//! Local multilingual Whisper CPU profiles. The host builds/ships the pinned
//! C shim; this module never downloads models or dispatches to a remote service.
//! Load and run on a host worker: contexts stay on their creating thread, take
//! one mutable call at a time, and hold Runtime's peak reservation until Drop.
mod ffi;
use crate::speech::{Cancellation, Result, SpeechError};
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    ptr::NonNull,
    time::Instant,
};
pub const MODEL_REPOSITORY: &str = "ggerganov/whisper.cpp";
pub const MODEL_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
pub const MODEL_SHA256: &str = "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898";
pub const MODEL_BYTES: u64 = 59_707_625;
/// Closed, exact artifact profiles; selecting tiny is explicit, never fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProfile {
    BaseQ5_1,
    TinyQ5_1,
}
impl ModelProfile {
    pub fn bytes(self) -> u64 {
        match self {
            Self::BaseQ5_1 => MODEL_BYTES,
            Self::TinyQ5_1 => 32_152_673,
        }
    }
    pub fn sha256(self) -> &'static str {
        match self {
            Self::BaseQ5_1 => MODEL_SHA256,
            Self::TinyQ5_1 => "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7",
        }
    }
    pub fn file_name(self) -> &'static str {
        match self {
            Self::BaseQ5_1 => "ggml-base-q5_1.bin",
            Self::TinyQ5_1 => "ggml-tiny-q5_1.bin",
        }
    }
}
pub const NATIVE_ABI: &str = "gen2-whisper-v1:d1be6fde11ac6e0407606b4e42fe72d34add8037";
/// Includes verified input weights during load plus native weights/workspace.
/// Base short physical-device peak was198MiB; tiny retains this conservative
/// floor. Whole-app/long-session qualification is still required for both.
pub const MINIMUM_RESERVATION_MIB: u64 = 384;
/// Pinned frontend/decoder needs more than100mel-ms;1800frames reaches110ms.
pub const MINIMUM_FRAMES: usize = 1800;
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub native_library: PathBuf,
    pub threads: usize,
    pub reservation_mb: u64,
}
impl LoadOptions {
    pub fn cpu(native_library: impl Into<PathBuf>) -> Self {
        Self {
            native_library: native_library.into(),
            threads: 4,
            reservation_mb: MINIMUM_RESERVATION_MIB,
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.threads) || self.reservation_mb < MINIMUM_RESERVATION_MIB {
            return Err(SpeechError::ResourceLimit(
                "Whisper needs1–8 threads and at least384MiB peak reservation".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Auto,
    English,
    Spanish,
}
impl Language {
    fn native(self) -> i32 {
        match self {
            Self::Auto => 0,
            Self::English => 1,
            Self::Spanish => 2,
        }
    }
}
#[derive(Clone)]
pub struct TranscribeOptions {
    pub language: Language,
    pub cancellation: Cancellation,
    pub deadline: Instant,
}
impl TranscribeOptions {
    fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(SpeechError::Cancelled)
        } else if Instant::now() >= self.deadline {
            Err(SpeechError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}
/// Playable intersection, suitable only for approximate source navigation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    pub start_ms: u64,
    pub end_ms: u64,
}
#[derive(Debug, Clone)]
pub struct Segment {
    /// Raw model estimates can extend into internal padding beyond real audio.
    pub model_start_ms: u64,
    pub model_end_ms: u64,
    pub source_span: SourceSpan,
    pub timestamp_clipped: bool,
    pub text: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    Inferred,
    DigitalSilence,
}
#[derive(Debug, Clone)]
pub struct Transcription {
    pub kind: ResultKind,
    pub text: String,
    pub segments: Vec<Segment>,
    pub detected_language: Option<i32>,
    pub frames: usize,
    pub execution_micros: u64,
}
pub(crate) struct Bundle {
    file: File,
    profile: ModelProfile,
}
impl Bundle {
    pub(crate) fn open(path: &Path, profile: ModelProfile) -> Result<Self> {
        let file = File::open(path)
            .map_err(|_| SpeechError::InvalidBundle("Local Whisper model is unavailable".into()))?;
        if !file
            .metadata()
            .is_ok_and(|m| m.is_file() && m.len() == profile.bytes())
        {
            return Err(SpeechError::InvalidBundle(format!(
                "Expected the pinned multilingual Whisper {} weights",
                profile.file_name()
            )));
        }
        Ok(Self { file, profile })
    }
    fn read_verified(mut self) -> Result<Vec<u8>> {
        // Keep the already-open regular file through admission; no path reopen
        // can switch it to a pipe or another inode. Allocation stays fixed even
        // if a concurrent writer grows the file.
        let mut bytes = vec![0; self.profile.bytes() as usize];
        self.file.read_exact(&mut bytes).map_err(|_| {
            SpeechError::InvalidBundle("Local Whisper model could not be read".into())
        })?;
        let mut extra = [0u8; 1];
        if self.file.read(&mut extra).map_err(|_| {
            SpeechError::InvalidBundle("Local Whisper model size could not be verified".into())
        })? != 0
        {
            return Err(SpeechError::InvalidBundle(
                "Local Whisper model size changed".into(),
            ));
        }
        if bytes.len() != self.profile.bytes() as usize
            || Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                != self.profile.sha256()
        {
            return Err(SpeechError::InvalidBundle(
                "Local Whisper model checksum changed; reinstall its pinned weights".into(),
            ));
        }
        Ok(bytes)
    }
}
pub struct Transcriber {
    context: NonNull<c_void>,
    api: ffi::Api,
    _reservation: Box<dyn Send>,
}
impl Transcriber {
    pub(crate) fn load(
        bundle: Bundle,
        options: LoadOptions,
        reservation: Box<dyn Send>,
    ) -> Result<Self> {
        let profile = bundle.profile;
        let bytes = bundle.read_verified()?;
        let api = ffi::Api::load(&options.native_library, profile)?;
        // SAFETY: exact verified weights stay alive for synchronous native loading;
        // upstream's buffer loader copies weights before returning its owned context.
        let context = NonNull::new(unsafe { (api.open)(bytes.as_ptr().cast(), bytes.len()) })
            .ok_or_else(|| {
                SpeechError::BackendUnavailable("Local Whisper context could not be created".into())
            })?;
        Ok(Self {
            context,
            api: api.with_threads(options.threads as i32),
            _reservation: reservation,
        })
    }
    ///112.5ms–30s finite normalized mono16kHz. Complete longer recordings belong to
    /// the host's bounded source journal, which must account for every input frame.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
    ) -> Result<Transcription> {
        validate_audio(samples)?;
        options.check()?;
        let started = Instant::now();
        // Exact digital zero contains no recoverable speech signal. This is not
        // acoustic VAD and does not qualify ordinary background/noise silence.
        if samples.iter().all(|sample| *sample == 0.) {
            options.check()?;
            return Ok(Transcription {
                kind: ResultKind::DigitalSilence,
                text: String::new(),
                segments: Vec::new(),
                detected_language: None,
                frames: samples.len(),
                execution_micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            });
        }
        // SAFETY: context is uniquely borrowed, callbacks cannot unwind, samples and
        // callback data remain live until the synchronous native call returns.
        let status = unsafe {
            (self.api.run)(
                self.context.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                options.language.native(),
                self.api.threads,
                Some(abort),
                (&options as *const TranscribeOptions).cast_mut().cast(),
            )
        };
        options.check()?;
        if status == 2 {
            return Err(SpeechError::Cancelled);
        }
        if status != 0 {
            return Err(SpeechError::Execution(
                "Local Whisper inference failed; saved audio is preserved".into(),
            ));
        }
        let mut segments = Vec::new();
        let mut text = String::new();
        // SAFETY: successful inference initialized the context's result table; every
        // segment is copied into a bounded owned UTF-8 buffer before the next call.
        let (count, detected_language) = unsafe {
            (
                (self.api.segments)(self.context.as_ptr()),
                (self.api.language)(self.context.as_ptr()),
            )
        };
        if !(0..=512).contains(&count) || !(0..100).contains(&detected_language) {
            return Err(SpeechError::InvalidOutput(
                "Invalid Whisper result metadata".into(),
            ));
        }
        let duration_ms = (samples.len() as u64).div_ceil(16);
        for index in 0..count {
            options.check()?;
            let (mut start, mut end, mut written) = (0i64, 0i64, 0usize);
            let mut buffer = vec![0u8; 65536];
            let code = unsafe {
                (self.api.segment)(
                    self.context.as_ptr(),
                    index,
                    &mut start,
                    &mut end,
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                    &mut written,
                )
            };
            if code != 0 || written > buffer.len() {
                return Err(SpeechError::InvalidOutput("Invalid Whisper segment".into()));
            }
            let (model_start_ms, model_end_ms, source_span) =
                project_span(start, end, duration_ms)?;
            let value = std::str::from_utf8(&buffer[..written])
                .map_err(|_| SpeechError::InvalidOutput("Whisper returned non-UTF8 text".into()))?;
            if text.len() + value.len() > 65536 {
                return Err(SpeechError::ResourceLimit(
                    "Whisper window text exceeds64KiB".into(),
                ));
            }
            text.push_str(value);
            segments.push(Segment {
                model_start_ms,
                model_end_ms,
                timestamp_clipped: source_span.start_ms != model_start_ms
                    || source_span.end_ms != model_end_ms,
                source_span,
                text: value.into(),
            });
        }
        options.check()?;
        Ok(Transcription {
            kind: ResultKind::Inferred,
            text: text.trim().into(),
            segments,
            detected_language: Some(detected_language),
            frames: samples.len(),
            execution_micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
        })
    }
}
impl Drop for Transcriber {
    fn drop(&mut self) {
        // SAFETY: owned context is closed on its creating thread before the API library
        // or Runtime peak reservation is dropped. No inference can borrow it now.
        unsafe {
            (self.api.close)(self.context.as_ptr());
        }
    }
}
unsafe extern "C" fn abort(data: *mut c_void) -> i32 {
    std::panic::catch_unwind(|| {
        // SAFETY: only our synchronous call installs this callback/data pair.
        let options = unsafe { &*data.cast::<TranscribeOptions>() };
        i32::from(options.check().is_err())
    })
    .unwrap_or(1)
}
/// Timestamp tokens are not clamped at source end upstream. Preserve their raw
/// estimate plus a playable intersection. The30s token-range envelope is only
/// a corruption check, not an accuracy tolerance. A segment entirely in padding
/// is rejected; its text is never silently dropped or accepted.
fn project_span(start: i64, end: i64, duration_ms: u64) -> Result<(u64, u64, SourceSpan)> {
    let invalid =
        || SpeechError::InvalidOutput("Whisper segment has no valid source interval".into());
    if start < 0 || end < start {
        return Err(invalid());
    }
    let start = (start as u64).checked_mul(10).ok_or_else(invalid)?;
    let end = (end as u64).checked_mul(10).ok_or_else(invalid)?;
    if end > duration_ms + 30_000 || start >= duration_ms || end <= start {
        return Err(invalid());
    }
    Ok((
        start,
        end,
        SourceSpan {
            start_ms: start,
            end_ms: end.min(duration_ms),
        },
    ))
}
fn validate_audio(samples: &[f32]) -> Result<()> {
    if !(MINIMUM_FRAMES..=480000).contains(&samples.len())
        || samples.iter().any(|x| !x.is_finite() || x.abs() > 1.)
    {
        return Err(SpeechError::InvalidAudio(
            "Use finite normalized mono16kHz PCM between1800frames (112.5ms) and30s".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn model_times_preserve_raw_estimates_and_expose_playable_projection() {
        let (start, end, span) = project_span(2960, 3010, 30000).unwrap();
        assert_eq!((start, end), (29600, 30100));
        assert_eq!(
            span,
            SourceSpan {
                start_ms: 29600,
                end_ms: 30000
            }
        );
        assert!(project_span(-1, 1, 8000).is_err());
        assert!(project_span(1, 0, 8000).is_err());
        assert!(project_span(800, 900, 8000).is_err());
        assert!(project_span(0, 0, 8000).is_err());
        assert!(project_span(0, 3801, 8000).is_err());
        assert!(project_span(i64::MAX, i64::MAX, 30000).is_err());
        assert_eq!(project_span(0, 780, 7844).unwrap().2.end_ms, 7800);
    }
    #[test]
    fn rejects_unbounded_nonfinite_and_out_of_range_pcm() {
        assert!(validate_audio(&[0.; MINIMUM_FRAMES]).is_ok());
        assert!(validate_audio(&vec![0.; 480000]).is_ok());
        assert!(validate_audio(&[0.; MINIMUM_FRAMES - 1]).is_err());
        assert!(validate_audio(&vec![0.; 480001]).is_err());
        assert!(validate_audio(&[f32::NAN; MINIMUM_FRAMES]).is_err());
        assert!(validate_audio(&[1.1; MINIMUM_FRAMES]).is_err());
    }
    #[test]
    fn cancellation_and_deadline_are_checked_independently_of_native_success() {
        let cancellation = Cancellation::default();
        let mut options = TranscribeOptions {
            language: Language::Spanish,
            cancellation: cancellation.clone(),
            deadline: Instant::now() + Duration::from_secs(1),
        };
        assert!(options.check().is_ok());
        options.deadline = Instant::now();
        assert!(matches!(
            options.check(),
            Err(SpeechError::DeadlineExceeded)
        ));
        cancellation.cancel();
        assert!(matches!(options.check(), Err(SpeechError::Cancelled)));
    }
    #[test]
    fn undersized_model_or_reservation_fail_before_native_load() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.bin");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        assert!(Bundle::open(&path, ModelProfile::BaseQ5_1).is_err());
        let mut options = LoadOptions::cpu("not-a-library");
        options.reservation_mb = 383;
        assert!(options.validate().is_err());
        options.reservation_mb = 384;
        options.threads = 0;
        assert!(options.validate().is_err());
    }
    #[test]
    fn runtime_budget_rejects_before_weight_allocation_or_native_loading() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.bin");
        let runtime = crate::Runtime::builder()
            .resident_memory_budget_mb(0)
            .build()
            .unwrap();
        for profile in [ModelProfile::BaseQ5_1, ModelProfile::TinyQ5_1] {
            File::create(&path)
                .unwrap()
                .set_len(profile.bytes())
                .unwrap();
            let error = runtime
                .load_whisper_profile(&path, profile, LoadOptions::cpu("no-library"))
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains("cannot reserve384") || error.contains("cannot reserve 384"),
                "{error}"
            );
            assert_eq!(runtime.speech_reserved_mb(), 0);
        }
    }
    #[test]
    fn wrong_profile_or_same_size_wrong_hash_is_rejected_before_native_load() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.bin");
        File::create(&path)
            .unwrap()
            .set_len(ModelProfile::TinyQ5_1.bytes())
            .unwrap();
        assert!(Bundle::open(&path, ModelProfile::BaseQ5_1).is_err());
        assert!(matches!(
            Bundle::open(&path, ModelProfile::TinyQ5_1)
                .unwrap()
                .read_verified(),
            Err(SpeechError::InvalidBundle(_))
        ));
        File::create(&path)
            .unwrap()
            .set_len(ModelProfile::BaseQ5_1.bytes())
            .unwrap();
        assert!(Bundle::open(&path, ModelProfile::TinyQ5_1).is_err());
    }
    #[test]
    #[ignore = "requires pinned tiny weights and original base-only native shim"]
    fn older_native_host_rejects_tiny_and_releases_shared_reservation() {
        let model = std::env::var_os("GEN2_WHISPER_TINY_MODEL").expect("pinned tiny model fixture");
        let native =
            std::env::var_os("GEN2_WHISPER_LEGACY_LIBRARY").expect("base-only native fixture");
        let runtime = crate::Runtime::new().unwrap();
        let error = runtime
            .load_whisper_profile(
                PathBuf::from(model),
                ModelProfile::TinyQ5_1,
                LoadOptions::cpu(native),
            )
            .err()
            .expect("old shim must reject tiny");
        assert!(error.to_string().contains("TinyQ5_1 support"), "{error}");
        assert_eq!(runtime.speech_reserved_mb(), 0);
    }
}
