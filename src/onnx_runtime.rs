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
            ort::init_from(&path).map_err(|e| e.to_string())?.commit();
            *loaded = Some(path);
        }
    }
    #[cfg(not(feature = "onnx-dynamic"))]
    let _ = path;
    Ok(())
}
