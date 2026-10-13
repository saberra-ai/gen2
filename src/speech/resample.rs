//! Small, bounded libsoxr HQ adapter. This optional dependency includes LGPL
//! native code; see tools/bark/README.md before distributing linked binaries.
use super::{Result, SpeechError};
use std::ffi::{c_ulong, c_void};

// Minimal private ABI from the pinned soxr.h. No native pointers escape.
#[repr(C)]
struct IoSpec {
    input_type: i32,
    output_type: i32,
    scale: f64,
    error: *mut c_void,
    flags: c_ulong,
}
#[repr(C)]
struct QualitySpec {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    error: *mut c_void,
    flags: c_ulong,
}
unsafe extern "C" {
    fn soxr_io_spec(input_type: i32, output_type: i32) -> IoSpec;
    fn soxr_quality_spec(recipe: c_ulong, flags: c_ulong) -> QualitySpec;
    fn soxr_oneshot(
        input_rate: f64,
        output_rate: f64,
        channels: u32,
        input: *const c_void,
        input_length: usize,
        consumed: *mut usize,
        output: *mut c_void,
        output_length: usize,
        produced: *mut usize,
        io: *const IoSpec,
        quality: *const QualitySpec,
        runtime: *const c_void,
    ) -> *const std::ffi::c_char;
}
pub(super) fn mono(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    let expected = (samples.len() * 16000 + rate as usize / 2) / rate as usize;
    // Extra capacity ensures oneshot consumes the entire final source frame,
    // even when the rounded target length is just below its exact ratio.
    let mut output = vec![0f32; expected + 1024];
    let mut consumed = 0usize;
    let mut produced = 0usize;
    // SAFETY: source/output buffers are disjoint, initialized f32 arrays with
    // exactly the supplied lengths, mono interleaved I/O is explicitly selected,
    // specs live through the call, and all output counts/errors are checked.
    let error = unsafe {
        // SOXR_FLOAT32_I == 0. Null runtime == soxr_runtime_spec(1).
        let io = soxr_io_spec(0, 0);
        // SOXR_HQ == SOXR_20_BITQ == 4, from the pinned libsoxr header.
        let quality = soxr_quality_spec(4, 0);
        soxr_oneshot(
            rate as f64,
            16000.,
            1,
            samples.as_ptr().cast(),
            samples.len(),
            &mut consumed,
            output.as_mut_ptr().cast(),
            output.len(),
            &mut produced,
            &io,
            &quality,
            std::ptr::null(),
        )
    };
    if !error.is_null() {
        // SAFETY: libsoxr returns a static NUL-terminated error string.
        let message = unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        return Err(SpeechError::InvalidAudio(message));
    }
    if consumed != samples.len() || produced != expected {
        return Err(SpeechError::InvalidAudio(format!(
            "libsoxr returned an incomplete recording: consumed={consumed}/{}, produced={produced}/{expected}, rate={rate}",
            samples.len()
        )));
    }
    output.truncate(produced);
    Ok(output)
}
