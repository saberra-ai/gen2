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
            #[cfg(target_os = "windows")]
            let library: Library = {
                use libloading::os::windows::{
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
                };
                // LoadLibrary's default search does not include an absolute
                // entry DLL's directory for its imported sibling DLLs. Scope
                // that search to this load, without changing process PATH.
                let absolute = std::fs::canonicalize(path).map_err(|e| bad(e.to_string()))?;
                libloading::os::windows::Library::load_with_flags(
                    absolute,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                )
                .map_err(|e| bad(loader_message(e)))?
                .into()
            };
            #[cfg(not(target_os = "windows"))]
            let library = Library::new(path).map_err(|e| bad(loader_message(e)))?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library.get(concat!($name, "\0").as_bytes()).map_err(|e| {
                        bad(format!(
                            "missing native symbol {}: {}",
                            $name,
                            loader_message(e)
                        ))
                    })?
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
    pub fn metal_device(&self) -> Result<(i32, String)> {
        let bad = |message: String| DiarizationError::BackendUnavailable(message);
        // SAFETY: these signatures come from the pinned ggml-backend.h. Symbol
        // and returned registry/device lifetimes are covered by this library.
        // NeMo chooses gpu by ordinal among GPU/IGPU devices, not all devices.
        unsafe {
            macro_rules! get {
                ($name:literal, $type:ty) => {
                    *self
                        ._library
                        .get::<$type>(concat!($name, "\0").as_bytes())
                        .map_err(|e| {
                            bad(format!(
                                "Metal placement requires {}: {}",
                                $name,
                                loader_message(e)
                            ))
                        })?
                };
            }
            let count = get!("ggml_backend_dev_count", unsafe extern "C" fn() -> usize);
            let device = get!(
                "ggml_backend_dev_get",
                unsafe extern "C" fn(usize) -> *mut c_void
            );
            let kind = get!(
                "ggml_backend_dev_type",
                unsafe extern "C" fn(*mut c_void) -> i32
            );
            let registry = get!(
                "ggml_backend_dev_backend_reg",
                unsafe extern "C" fn(*mut c_void) -> *mut c_void
            );
            let name = get!(
                "ggml_backend_reg_name",
                unsafe extern "C" fn(*mut c_void) -> *const c_char
            );
            let description = get!(
                "ggml_backend_dev_description",
                unsafe extern "C" fn(*mut c_void) -> *const c_char
            );
            let n = count();
            if n > 1024 {
                return Err(bad("excessive native device count".into()));
            }
            let mut gpu_index = 0;
            for i in 0..n {
                let dev = device(i);
                if dev.is_null() {
                    return Err(bad("null native device".into()));
                }
                // Pinned enum: CPU=0, GPU=1, IGPU=2, ACCEL=3, META=4.
                if !matches!(kind(dev), 1 | 2) {
                    continue;
                }
                let reg = registry(dev);
                if reg.is_null() {
                    return Err(bad("null GPU registry".into()));
                }
                let reg_name = name(reg);
                if reg_name.is_null() {
                    return Err(bad("null GPU backend name".into()));
                }
                // The pinned ggml-metal registry names itself "MTL".
                if CStr::from_ptr(reg_name).to_bytes() == b"MTL" {
                    let text = description(dev);
                    if text.is_null() {
                        return Err(bad("null Metal device description".into()));
                    }
                    let text = CStr::from_ptr(text)
                        .to_str()
                        .map_err(|e| bad(e.to_string()))?;
                    if text.is_empty() {
                        return Err(bad("empty Metal device description".into()));
                    }
                    // Pinned NeMo BackendManager throws if this exact GPU cannot
                    // initialize. It does not fall back to CPU after gpu>=0.
                    return Ok((gpu_index, text.to_owned()));
                }
                gpu_index += 1;
            }
            Err(bad(
                "requested Metal device is unavailable in the native SDK".into(),
            ))
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

fn loader_message(error: libloading::Error) -> String {
    use std::error::Error;
    match error.source() {
        Some(cause) => format!("{error}: {cause}"),
        None => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_error_without_a_system_cause_keeps_its_context() {
        assert_eq!(
            loader_message(libloading::Error::InteriorZeroElements),
            "interior zero element in parameter"
        );
    }

    #[cfg(unix)]
    #[test]
    fn missing_library_retains_the_operating_system_loader_cause() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory
            .path()
            .join("gen2-intentionally-missing-native-library.so");
        match Api::load(&path) {
            Err(DiarizationError::BackendUnavailable(message)) => {
                assert!(
                    message.contains("gen2-intentionally-missing-native-library.so"),
                    "{message}"
                );
                assert!(message.contains("dlopen"), "{message}");
            }
            _ => panic!("missing library must fail before resolving native symbols"),
        }
    }
}
