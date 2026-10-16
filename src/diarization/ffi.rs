//! Layouts mirror the pinned public diar.h, not the SDK's internal C++ types.
use super::{DiarizationError, Result};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
};

#[repr(C)]
pub(super) struct ModelConfig {
    pub size: usize,
    pub model_path: *const c_char,
    pub gpu: i32,
    pub preset: *const c_char,
    pub chunk_frames: i32,
    pub right_context_frames: i32,
    pub left_context_frames: i32,
    pub fifo_frames: i32,
    pub spkcache_frames: i32,
    pub update_period_frames: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

// Checked against the pinned public C header on the qualified 64-bit desktops.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<ModelConfig>() == 56);
    assert!(std::mem::offset_of!(ModelConfig, preset) == 24);
    assert!(std::mem::offset_of!(ModelConfig, update_period_frames) == 52);
    assert!(std::mem::size_of::<Segment>() == 24);
    assert!(std::mem::offset_of!(Segment, speaker) == 16);
};

pub(super) struct Api {
    pub create: unsafe extern "C" fn(*const ModelConfig, *mut *mut c_void) -> i32,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub speakers: unsafe extern "C" fn(*const c_void) -> i32,
    pub cadence: unsafe extern "C" fn(*const c_void) -> f64,
    pub open: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub push: unsafe extern "C" fn(*mut c_void, *const f32, usize, i32) -> i32,
    pub finish: unsafe extern "C" fn(*mut c_void) -> i32,
    pub close: unsafe extern "C" fn(*mut c_void),
    pub frame_count: unsafe extern "C" fn(*const c_void) -> i64,
    pub probs_start: unsafe extern "C" fn(*const c_void) -> i64,
    pub probs: unsafe extern "C" fn(*const c_void, *mut f32, usize) -> i32,
    pub segments:
        unsafe extern "C" fn(*const c_void, *const c_void, *mut Segment, usize, *mut usize) -> i32,
    error: unsafe extern "C" fn() -> *const c_char,
    pub version: String,
    _library: Library,
}
impl Api {
    pub fn load(path: &Path) -> Result<Self> {
        let bad = |e: String| DiarizationError::BackendUnavailable(e);
        // SAFETY: the host supplies the qualified C ABI. All required symbols
        // are resolved before native model creation. The library outlives calls.
        unsafe {
            let library = Library::new(path).map_err(|e| bad(e.to_string()))?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| bad(format!("missing native symbol {}: {e}", $name)))?
                };
            }
            let version_fn: unsafe extern "C" fn() -> *const c_char =
                symbol!("nemo_speech_asr_version");
            let pointer = version_fn();
            if pointer.is_null() {
                return Err(bad("null native version".into()));
            }
            let version = CStr::from_ptr(pointer)
                .to_str()
                .map_err(|e| bad(e.to_string()))?
                .to_owned();
            if version
                .strip_prefix("nemo-speech-asr ")
                .and_then(|v| v.split('+').next())
                != Some("0.2.0")
            {
                return Err(bad(format!("unqualified NeMo ABI version: {version}")));
            }
            Ok(Self {
                create: symbol!("nemo_speech_diar_create"),
                destroy: symbol!("nemo_speech_diar_destroy"),
                speakers: symbol!("nemo_speech_diar_num_speakers"),
                cadence: symbol!("nemo_speech_diar_seconds_per_frame"),
                open: symbol!("nemo_speech_diar_stream_open"),
                push: symbol!("nemo_speech_diar_stream_push_f32"),
                finish: symbol!("nemo_speech_diar_stream_finish"),
                close: symbol!("nemo_speech_diar_stream_close"),
                frame_count: symbol!("nemo_speech_diar_frame_count"),
                probs_start: symbol!("nemo_speech_diar_frame_probs_start"),
                probs: symbol!("nemo_speech_diar_frame_probs"),
                segments: symbol!("nemo_speech_diar_segments"),
                error: symbol!("nemo_speech_asr_last_error"),
                version,
                _library: library,
            })
        }
    }
    pub fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: this thread just made the failing call; SDK owns its error string.
        let message = unsafe {
            let pointer = (self.error)();
            if pointer.is_null() {
                format!("native status {status}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(DiarizationError::Execution(message))
    }
}
