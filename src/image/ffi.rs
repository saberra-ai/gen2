//! ABI-2 bridge. All unsafe native access and owned-pointer lifetimes stay here.
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    time::Duration,
};

use super::{
    Driver, GeneratedImage, ImageBundle, ImageCanceller, ImageError, ImageMemoryPlan, NativeResult,
    job::Request,
};
use libloading::Library;

// The upstream progress callback is global, even for distinct model contexts.
static NATIVE_GATE: Mutex<()> = Mutex::new(());
fn gate() -> MutexGuard<'static, ()> {
    NATIVE_GATE.lock().unwrap_or_else(|e| e.into_inner())
}
fn job_gate(cancel: &ImageCanceller) -> NativeResult<MutexGuard<'static, ()>> {
    loop {
        if cancel.is_cancelled() {
            return Err(ImageError::Cancelled);
        }
        match NATIVE_GATE.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(error)) => return Ok(error.into_inner()),
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

type ProgressFn = unsafe extern "C" fn(c_int, c_int, f32, *mut c_void) -> bool;
type LoadFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    u64,
    *mut c_char,
    usize,
) -> *mut c_void;
type GenerateFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    u32,
    u32,
    i64,
    ProgressFn,
    *mut c_void,
    *mut NativeImage,
    *mut c_char,
    usize,
) -> c_int;

#[repr(C)]
#[derive(Default)]
struct NativeImage {
    width: u32,
    height: u32,
    channels: u32,
    pixels: *const u8,
    owner: *mut c_void,
    count: c_int,
}

pub(super) struct Api {
    _library: Library,
    load: LoadFn,
    generate: GenerateFn,
    free_image: unsafe extern "C" fn(*mut NativeImage),
    unload: unsafe extern "C" fn(*mut c_void),
}

impl Api {
    pub fn open(path: &Path) -> NativeResult<Arc<Self>> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err(ImageError::BackendUnavailable(
                "this bridge is qualified for Apple silicon macOS only".into(),
            ));
        }
        let _gate = gate();
        let unavailable = |e: libloading::Error| ImageError::BackendUnavailable(e.to_string());
        // SAFETY: the host supplies the native library; its functions are
        // resolved only after checking our versioned ABI. Library owns their
        // lifetime and every context keeps an Arc<Api> until native teardown.
        unsafe {
            let library = Library::new(path).map_err(unavailable)?;
            let abi = library
                .get::<unsafe extern "C" fn() -> u32>(b"gen2_image_abi\0")
                .map_err(unavailable)?;
            if abi() != 2 {
                return Err(ImageError::BackendUnavailable(
                    "image bridge ABI must be 2".into(),
                ));
            }
            let devices = library
                .get::<unsafe extern "C" fn(*mut c_char, usize) -> usize>(b"gen2_image_devices\0")
                .map_err(unavailable)?;
            let size = devices(std::ptr::null_mut(), 0);
            if size > 1024 * 1024 {
                return Err(ImageError::BackendUnavailable(
                    "invalid native device list".into(),
                ));
            }
            let mut buffer = vec![0u8; size + 1];
            devices(buffer.as_mut_ptr().cast(), buffer.len());
            let list = CStr::from_ptr(buffer.as_ptr().cast())
                .to_string_lossy()
                .to_lowercase();
            if !list.contains("mtl") && !list.contains("metal") {
                return Err(ImageError::BackendUnavailable(
                    "Metal device is unavailable".into(),
                ));
            }
            Ok(Arc::new(Self {
                load: *library.get(b"gen2_image_load\0").map_err(unavailable)?,
                generate: *library.get(b"gen2_image_generate\0").map_err(unavailable)?,
                free_image: *library
                    .get(b"gen2_image_free_image\0")
                    .map_err(unavailable)?,
                unload: *library.get(b"gen2_image_unload\0").map_err(unavailable)?,
                _library: library,
            }))
        }
    }
}

pub(super) struct Context {
    api: Arc<Api>,
    ctx: NonNull<c_void>,
}
// SAFETY: SD's synchronous context may be moved between host threads. Every
// load/generate/free/unload call is serialized under NATIVE_GATE; no C pointer
// escapes, and a context cannot be accessed concurrently or freed in flight.
unsafe impl Send for Context {}

impl Context {
    pub fn load(api: Arc<Api>, bundle: &ImageBundle, plan: &ImageMemoryPlan) -> NativeResult<Self> {
        let _gate = gate();
        let paths = [&bundle.denoiser, &bundle.text_encoder, &bundle.vae]
            .map(|path| CString::new(path.as_os_str().as_encoded_bytes()));
        let [denoiser, encoder, vae] = paths;
        let invalid = |_| ImageError::InvalidBundle("component path contains NUL".into());
        let (denoiser, encoder, vae) = (
            denoiser.map_err(invalid)?,
            encoder.map_err(invalid)?,
            vae.map_err(invalid)?,
        );
        let mut error = [0 as c_char; 2048];
        // SAFETY: strings and error buffer outlive the synchronous call. The
        // non-null context is exclusively owned and paired with this same API.
        let ctx = unsafe {
            (api.load)(
                denoiser.as_ptr(),
                encoder.as_ptr(),
                vae.as_ptr(),
                plan.total_mb,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        Ok(Self {
            api,
            ctx: NonNull::new(ctx).ok_or_else(|| ImageError::Execution(error_text(&error)))?,
        })
    }
}

impl Driver for Context {
    fn generate(
        &mut self,
        request: &Request,
        cancel: &ImageCanceller,
        callback: &mut dyn FnMut(u32, u32) -> bool,
    ) -> NativeResult<GeneratedImage> {
        let _gate = job_gate(cancel)?;
        let prompt = CString::new(request.prompt.as_str())
            .map_err(|_| ImageError::InvalidRequest("prompt contains NUL".into()))?;
        let mut state = Progress {
            callback,
            panicked: false,
        };
        let mut image = OwnedImage {
            api: &self.api,
            raw: NativeImage::default(),
        };
        let mut error = [0 as c_char; 2048];
        // SAFETY: all inputs, callback state and buffers live until this
        // synchronous native call clears its callback and returns. The image
        // guard frees every allocation via the allocating library on all paths.
        let status = unsafe {
            (self.api.generate)(
                self.ctx.as_ptr(),
                prompt.as_ptr(),
                request.width,
                request.height,
                request.seed,
                on_progress,
                (&mut state as *mut Progress<'_>).cast(),
                &mut image.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if state.panicked {
            return Err(ImageError::Execution("progress callback panicked".into()));
        }
        if cancel.is_cancelled() || status == 2 {
            return Err(ImageError::Cancelled);
        }
        if status != 0 {
            return Err(ImageError::Execution(error_text(&error)));
        }
        let raw = &image.raw;
        if raw.width != request.width
            || raw.height != request.height
            || !matches!(raw.channels, 3 | 4)
            || raw.pixels.is_null()
            || raw.owner.is_null()
            || raw.count != 1
        {
            return Err(ImageError::Execution("invalid native image output".into()));
        }
        let len = raw.width as usize * raw.height as usize * raw.channels as usize;
        // SAFETY: ABI 2 supplies tightly packed bytes until free_image. The
        // validated request bounds dimensions; channels and pointers checked above.
        let pixels = unsafe { std::slice::from_raw_parts(raw.pixels, len) }.to_vec();
        GeneratedImage::from_pixels(raw.width, raw.height, raw.channels, pixels)
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        let _gate = gate();
        // SAFETY: this exclusive context is idle and its API is still alive.
        unsafe { (self.api.unload)(self.ctx.as_ptr()) };
    }
}

struct OwnedImage<'a> {
    api: &'a Api,
    raw: NativeImage,
}
impl Drop for OwnedImage<'_> {
    fn drop(&mut self) {
        // SAFETY: called while the gate is held; empty or owned ABI-2 output.
        unsafe { (self.api.free_image)(&mut self.raw) };
    }
}
struct Progress<'a> {
    callback: &'a mut dyn FnMut(u32, u32) -> bool,
    panicked: bool,
}
unsafe extern "C" fn on_progress(step: c_int, total: c_int, _: f32, user: *mut c_void) -> bool {
    // SAFETY: this state is exclusively borrowed until native generation ends.
    let state = unsafe { &mut *user.cast::<Progress<'_>>() };
    if step < 0 || total != 8 {
        return false;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (state.callback)(step as u32, total as u32)
    })) {
        Ok(keep_going) => keep_going,
        Err(_) => {
            state.panicked = true;
            false
        }
    }
}
fn error_text(buffer: &[c_char]) -> String {
    // SAFETY: buffers are zero-initialized; bridge errors always NUL-terminate.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_native_work_can_cancel_while_another_context_holds_the_gate() {
        let guard = gate();
        let cancel = ImageCanceller::default();
        let worker_cancel = cancel.clone();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let result = job_gate(&worker_cancel).map(|_| ());
            tx.send(result).unwrap();
        });
        cancel.cancel();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            Err(ImageError::Cancelled)
        );
        drop(guard);
        worker.join().unwrap();
    }
}
