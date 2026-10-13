//! Small, bounded libsoxr HQ adapter. This optional dependency includes LGPL
//! native code; see tools/bark/README.md before distributing linked binaries.
use super::{Result, SpeechError};
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
        let io = soxr_sys::soxr_io_spec(
            soxr_sys::soxr_datatype_t_SOXR_FLOAT32_I,
            soxr_sys::soxr_datatype_t_SOXR_FLOAT32_I,
        );
        // SOXR_HQ == SOXR_20_BITQ == 4, from the pinned libsoxr header.
        let quality = soxr_sys::soxr_quality_spec(4, 0);
        let runtime = soxr_sys::soxr_runtime_spec(1);
        soxr_sys::soxr_oneshot(
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
            &runtime,
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
