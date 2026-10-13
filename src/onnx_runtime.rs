//! One process-global native library policy shared by speech and Laya.

pub(crate) fn initialize(path: Option<&std::path::Path>) -> Result<(), String> {
    #[cfg(feature = "onnx-dynamic")]
    {
        static LIBRARY: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);
        let path = path
            .ok_or("set native_library to the packaged ONNX Runtime 1.24 library")?
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let mut loaded = LIBRARY
            .lock()
            .map_err(|_| "ONNX initialization lock poisoned")?;
        if loaded.as_ref().is_some_and(|p| p != &path) {
            return Err("another ONNX library is already initialized in this process".into());
        }
        if loaded.is_none() {
            // `ort` constructs a native error while its library OnceLock is
            // initializing on a load/version failure, which can recurse into
            // that same lock. Preflight through the stable C API before any
            // binding calls. Keep this handle alive until ort owns its handle.
            let native = preflight(&path)?;
            ort::init_from(&path).map_err(|e| e.to_string())?.commit();
            *loaded = Some(path);
            drop(native);
        }
    }
    #[cfg(not(feature = "onnx-dynamic"))]
    let _ = path;
    Ok(())
}

#[cfg(feature = "onnx-dynamic")]
fn preflight(path: &std::path::Path) -> Result<libloading::Library, String> {
    // SAFETY: the host-selected library is trusted native code. Only ORT's
    // stable two-function API base is read; its pointer and requested ABI are
    // checked before the versioned bindings are allowed to initialize.
    let native = unsafe { libloading::Library::new(path) }.map_err(|e| e.to_string())?;
    let base = unsafe {
        native.get::<unsafe extern "system" fn() -> *const ort::sys::OrtApiBase>(b"OrtGetApiBase\0")
    }
    .map_err(|e| format!("not an ONNX Runtime library: {e}"))?;
    let base = unsafe { base().as_ref() }.ok_or("ONNX Runtime returned a null API base")?;
    let version = unsafe { (base.GetVersionString)() };
    if version.is_null() {
        return Err("ONNX Runtime returned a null version".into());
    }
    let version = unsafe { std::ffi::CStr::from_ptr(version) }.to_string_lossy();
    let minor = version
        .split('.')
        .nth(1)
        .and_then(|v| v.parse::<u32>().ok());
    if minor.is_none_or(|v| v < ort::MINOR_VERSION)
        || unsafe { (base.GetApi)(ort::MINOR_VERSION) }.is_null()
    {
        return Err(format!(
            "ONNX Runtime API {} (1.{}.x or newer) is required; got {version}",
            ort::MINOR_VERSION,
            ort::MINOR_VERSION
        ));
    }
    Ok(native)
}
