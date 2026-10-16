use super::{ModelProfile, NATIVE_ABI, Result, SpeechError};
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
};
type Abort = Option<unsafe extern "C" fn(*mut c_void) -> i32>;
pub(super) struct Api {
    pub open: unsafe extern "C" fn(*const c_void, usize) -> *mut c_void,
    pub close: unsafe extern "C" fn(*mut c_void),
    pub run:
        unsafe extern "C" fn(*mut c_void, *const f32, usize, i32, i32, Abort, *mut c_void) -> i32,
    pub segments: unsafe extern "C" fn(*mut c_void) -> i32,
    pub language: unsafe extern "C" fn(*mut c_void) -> i32,
    pub segment: unsafe extern "C" fn(
        *mut c_void,
        i32,
        *mut i64,
        *mut i64,
        *mut c_char,
        usize,
        *mut usize,
    ) -> i32,
    pub threads: i32,
    _library: libloading::Library,
}
impl Api {
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }
    pub fn load(path: &Path, profile: ModelProfile) -> Result<Self> {
        let bad = || {
            SpeechError::BackendUnavailable(format!(
                "Expected the pinned gen2-whisper-v1 native library with {profile:?} support"
            ))
        };
        // SAFETY: host ships this reviewed C ABI. All symbols/version are checked
        // before loading weights. The library outlives context and every function call.
        unsafe {
            let library = libloading::Library::new(path).map_err(|_| bad())?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|_| bad())?
                };
            }
            let version: unsafe extern "C" fn() -> *const c_char = symbol!("gen2_whisper_abi");
            let pointer = version();
            if pointer.is_null() || CStr::from_ptr(pointer).to_bytes() != NATIVE_ABI.as_bytes() {
                return Err(bad());
            }
            Ok(Self {
                // Base keeps its original ABI and rejects tiny bytes. A distinct
                // versioned tiny constructor fails closed on older native hosts.
                open: match profile {
                    ModelProfile::BaseQ5_1 => symbol!("gen2_whisper_open"),
                    ModelProfile::TinyQ5_1 => symbol!("gen2_whisper_open_tiny_v1"),
                },
                close: symbol!("gen2_whisper_close"),
                run: symbol!("gen2_whisper_run"),
                segments: symbol!("gen2_whisper_segments"),
                language: symbol!("gen2_whisper_language"),
                segment: symbol!("gen2_whisper_segment"),
                threads: 4,
                _library: library,
            })
        }
    }
}
