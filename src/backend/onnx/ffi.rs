//! ONNX Runtime GenAI 0.15.2 C ABI. All native pointers stay in this module.
//! The host supplies the library; Cargo neither links nor downloads it.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use libloading::Library;

use crate::engine::ExecError;

macro_rules! opaque {
    ($($name:ident),* $(,)?) => { $(#[repr(C)] struct $name { _private: [u8; 0] })* };
}
opaque!(
    ResultHandle,
    Config,
    ModelHandle,
    Params,
    GeneratorHandle,
    Tokenizer,
    Stream
);

macro_rules! api {
    ($($name:ident: $ty:ty),* $(,)?) => {
        #[allow(non_snake_case)]
        struct Api { $($name: $ty),* }
        impl Api {
            unsafe fn load(library: &Library) -> Result<Self, ExecError> {
                Ok(Self { $( $name: *unsafe { library.get::<$ty>(concat!(stringify!($name), "\0").as_bytes()) }
                    .map_err(|e| ExecError::Other(anyhow::anyhow!("ONNX Runtime GenAI missing {}: {e}", stringify!($name))))?, )* })
            }
        }
    };
}
api! {
    OgaShutdown: unsafe extern "system" fn(),
    OgaResultGetError: unsafe extern "system" fn(*const ResultHandle) -> *const c_char,
    OgaDestroyResult: unsafe extern "system" fn(*mut ResultHandle),
    OgaCreateConfig: unsafe extern "system" fn(*const c_char, *mut *mut Config) -> *mut ResultHandle,
    OgaDestroyConfig: unsafe extern "system" fn(*mut Config),
    OgaConfigOverlay: unsafe extern "system" fn(*mut Config, *const c_char) -> *mut ResultHandle,
    OgaConfigClearProviders: unsafe extern "system" fn(*mut Config) -> *mut ResultHandle,
    OgaCreateModelFromConfig: unsafe extern "system" fn(*const Config, *mut *mut ModelHandle) -> *mut ResultHandle,
    OgaDestroyModel: unsafe extern "system" fn(*mut ModelHandle),
    OgaCreateGeneratorParams: unsafe extern "system" fn(*const ModelHandle, *mut *mut Params) -> *mut ResultHandle,
    OgaDestroyGeneratorParams: unsafe extern "system" fn(*mut Params),
    OgaGeneratorParamsSetSearchNumber: unsafe extern "system" fn(*mut Params, *const c_char, f64) -> *mut ResultHandle,
    OgaGeneratorParamsSetSearchBool: unsafe extern "system" fn(*mut Params, *const c_char, bool) -> *mut ResultHandle,
    OgaGeneratorParamsSetGuidance: unsafe extern "system" fn(*mut Params, *const c_char, *const c_char, bool) -> *mut ResultHandle,
    OgaCreateGenerator: unsafe extern "system" fn(*const ModelHandle, *const Params, *mut *mut GeneratorHandle) -> *mut ResultHandle,
    OgaDestroyGenerator: unsafe extern "system" fn(*mut GeneratorHandle),
    OgaGenerator_AppendTokens: unsafe extern "system" fn(*mut GeneratorHandle, *const i32, usize) -> *mut ResultHandle,
    OgaGenerator_IsDone: unsafe extern "system" fn(*mut GeneratorHandle) -> bool,
    OgaGenerator_GenerateNextToken: unsafe extern "system" fn(*mut GeneratorHandle) -> *mut ResultHandle,
    OgaGenerator_GetNextTokens: unsafe extern "system" fn(*const GeneratorHandle, *mut *const i32, *mut usize) -> *mut ResultHandle,
    OgaCreateTokenizer: unsafe extern "system" fn(*const ModelHandle, *mut *mut Tokenizer) -> *mut ResultHandle,
    OgaDestroyTokenizer: unsafe extern "system" fn(*mut Tokenizer),
    OgaCreateTokenizerStream: unsafe extern "system" fn(*const Tokenizer, *mut *mut Stream) -> *mut ResultHandle,
    OgaDestroyTokenizerStream: unsafe extern "system" fn(*mut Stream),
    OgaTokenizerStreamDecode: unsafe extern "system" fn(*mut Stream, i32, *mut *const c_char) -> *mut ResultHandle,
}

/// Library must outlive all function pointers, models and generators.
pub(super) struct Runtime {
    api: Api,
    _library: Library,
    _ort_library: Library,
}

// GenAI has process-global state. Multiple engines may share the same library,
// but shutdown must wait for ALL models/generators, including live sessions.
// Serialize acquisition and final teardown; never dlclose initialized globals.
#[derive(Default)]
struct Lifecycle {
    active: usize,
    path: Option<(std::ffi::OsString, std::ffi::OsString)>,
}
static LIFECYCLE: std::sync::Mutex<Lifecycle> = std::sync::Mutex::new(Lifecycle {
    active: 0,
    path: None,
});

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "system" fn(u32) -> *const std::ffi::c_void,
    get_version_string: unsafe extern "system" fn() -> *const c_char,
}

impl Runtime {
    pub(super) fn load() -> Result<Arc<Self>, ExecError> {
        let path = std::env::var_os("GEN2_ONNX_LIBRARY").unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                "onnxruntime-genai.dll"
            } else if cfg!(target_os = "macos") {
                "libonnxruntime-genai.dylib"
            } else {
                "libonnxruntime-genai.so"
            }
            .into()
        });
        let ort_path = std::env::var_os("ORT_LIB_PATH").ok_or(ExecError::FeatureUnsupported(
            "ONNX GenAI requires ORT_LIB_PATH pointing to an ONNX Runtime 1.31 shared library file",
        ))?;
        let mut lifecycle = LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        if lifecycle
            .path
            .as_ref()
            .is_some_and(|active| active != &(path.clone(), ort_path.clone()))
        {
            return Err(ExecError::InvalidArg(
                "cannot change ONNX runtime library while another ONNX engine is alive",
            ));
        }
        // GenAI initializes ORT during dlopen and throws an uncaught C++
        // exception if it cannot find a compatible library. Preload and check
        // its stable API-base ABI FIRST, so configuration mistakes are errors.
        let ort_library = unsafe { Library::new(&ort_path) }.map_err(|e| {
            ExecError::Other(anyhow::anyhow!(
                "cannot load ORT_LIB_PATH {}: {e}",
                Path::new(&ort_path).display()
            ))
        })?;
        let base = unsafe {
            ort_library.get::<unsafe extern "system" fn() -> *const OrtApiBase>(b"OrtGetApiBase\0")
        }
        .map_err(|e| {
            ExecError::Other(anyhow::anyhow!(
                "ORT_LIB_PATH is not an ONNX Runtime library: {e}"
            ))
        })?;
        // SAFETY: this is ONNX Runtime's documented stable, two-function base.
        let base = unsafe { base().as_ref() }.ok_or(ExecError::InvalidArg(
            "ONNX Runtime returned a null API base",
        ))?;
        if unsafe { (base.get_api)(31) }.is_null() {
            return Err(ExecError::FeatureUnsupported(
                "ONNX GenAI backend requires ONNX Runtime API 31 (1.31 or newer)",
            ));
        }
        // SAFETY: the selected library must implement the documented GenAI C ABI.
        // Function signatures are pinned to its shipped 0.15.2 header. No pointer
        // leaves this module and every handle retains its library through Arc.
        let library = unsafe { Library::new(&path) }.map_err(|e| {
            ExecError::Other(anyhow::anyhow!("cannot load ONNX Runtime GenAI {}: {e}; set GEN2_ONNX_LIBRARY to the host's 0.15.2 shared library", Path::new(&path).display()))
        })?;
        let api = unsafe { Api::load(&library) }?;
        lifecycle.active += 1;
        lifecycle.path = Some((path, ort_path));
        Ok(Arc::new(Self {
            api,
            _library: library,
            _ort_library: ort_library,
        }))
    }

    fn check(&self, result: *mut ResultHandle) -> Result<(), ExecError> {
        let Some(result) = NonNull::new(result) else {
            return Ok(());
        };
        // SAFETY: a non-null result is owned by this call. Copy its borrowed
        // message before releasing it, exactly once, through the same runtime.
        let message = unsafe {
            let ptr = (self.api.OgaResultGetError)(result.as_ptr());
            let text = if ptr.is_null() {
                "unknown native error".into()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            };
            (self.api.OgaDestroyResult)(result.as_ptr());
            text
        };
        Err(ExecError::Generation(format!(
            "ONNX Runtime GenAI: {message}"
        )))
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        let mut lifecycle = LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        lifecycle.active -= 1;
        if lifecycle.active == 0 {
            // SAFETY: every native handle retains an Arc<Runtime>. There are no
            // handles from any engine left, and acquisition holds the same lock.
            unsafe { (self.api.OgaShutdown)() };
            lifecycle.path = None;
        }
    }
}

/// Owned native handle, with the destructor and runtime that created it.
struct Owned<T> {
    ptr: NonNull<T>,
    destroy: unsafe extern "system" fn(*mut T),
    runtime: Arc<Runtime>,
}

impl<T> Owned<T> {
    fn create(
        runtime: Arc<Runtime>,
        destroy: unsafe extern "system" fn(*mut T),
        call: impl FnOnce(*mut *mut T) -> *mut ResultHandle,
    ) -> Result<Self, ExecError> {
        let mut ptr = std::ptr::null_mut();
        runtime.check(call(&mut ptr))?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            ExecError::Generation("ONNX Runtime GenAI returned a null handle".into())
        })?;
        Ok(Self {
            ptr,
            destroy,
            runtime,
        })
    }
    fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}
impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner; the matching library is still alive.
        unsafe { (self.destroy)(self.ptr()) };
    }
}

pub(super) struct Model {
    tokenizer: Owned<Tokenizer>,
    model: Owned<ModelHandle>,
}

fn cstring(value: &str) -> Result<CString, ExecError> {
    CString::new(value).map_err(|_| ExecError::InvalidArg("ONNX strings cannot contain NUL"))
}

impl Model {
    pub(super) fn load(
        runtime: Arc<Runtime>,
        directory: &Path,
        threads: Option<u32>,
        cpu: bool,
    ) -> Result<Self, ExecError> {
        let path = directory
            .to_str()
            .ok_or(ExecError::InvalidArg("ONNX bundle path must be UTF-8"))?;
        let path = cstring(path)?;
        let api = &runtime.api;
        let config = Owned::create(runtime.clone(), api.OgaDestroyConfig, |out| unsafe {
            (api.OgaCreateConfig)(path.as_ptr(), out)
        })?;
        if cpu {
            runtime.check(unsafe { (api.OgaConfigClearProviders)(config.ptr()) })?;
        }
        if let Some(threads) = threads {
            let overlay = cstring(&serde_json::json!({"model":{"decoder":{"session_options":{"intra_op_num_threads":threads}}}}).to_string())?;
            runtime.check(unsafe { (api.OgaConfigOverlay)(config.ptr(), overlay.as_ptr()) })?;
        }
        let model = Owned::create(runtime.clone(), api.OgaDestroyModel, |out| unsafe {
            (api.OgaCreateModelFromConfig)(config.ptr(), out)
        })?;
        let tokenizer = Owned::create(runtime.clone(), api.OgaDestroyTokenizer, |out| unsafe {
            (api.OgaCreateTokenizer)(model.ptr(), out)
        })?;
        Ok(Self { tokenizer, model })
    }

    pub(super) fn generate(
        self: &Arc<Self>,
        tokens: &[i32],
        numbers: &[(&str, f64)],
        sample: bool,
        guidance: Option<(&str, &str)>,
    ) -> Result<Generator, ExecError> {
        let runtime = self.model.runtime.clone();
        let api = &runtime.api;
        let params = Owned::create(
            runtime.clone(),
            api.OgaDestroyGeneratorParams,
            |out| unsafe { (api.OgaCreateGeneratorParams)(self.model.ptr(), out) },
        )?;
        for &(key, value) in numbers {
            let key = cstring(key)?;
            runtime.check(unsafe {
                (api.OgaGeneratorParamsSetSearchNumber)(params.ptr(), key.as_ptr(), value)
            })?;
        }
        runtime.check(unsafe {
            (api.OgaGeneratorParamsSetSearchBool)(params.ptr(), c"do_sample".as_ptr(), sample)
        })?;
        if let Some((kind, value)) = guidance {
            let kind = cstring(kind)?;
            let value = cstring(value)?;
            // One token per step keeps budgets, cancellation and statistics exact.
            runtime.check(unsafe {
                (api.OgaGeneratorParamsSetGuidance)(
                    params.ptr(),
                    kind.as_ptr(),
                    value.as_ptr(),
                    false,
                )
            })?;
        }
        let generator = Owned::create(runtime.clone(), api.OgaDestroyGenerator, |out| unsafe {
            (api.OgaCreateGenerator)(self.model.ptr(), params.ptr(), out)
        })?;
        runtime.check(unsafe {
            (api.OgaGenerator_AppendTokens)(generator.ptr(), tokens.as_ptr(), tokens.len())
        })?;
        let stream = Owned::create(
            runtime.clone(),
            api.OgaDestroyTokenizerStream,
            |out| unsafe { (api.OgaCreateTokenizerStream)(self.tokenizer.ptr(), out) },
        )?;
        Ok(Generator {
            stream,
            generator,
            _model: self.clone(),
        })
    }
}

/// Generator, then model, then library stay alive through destruction. A fresh
/// generator per turn sees exactly the current transcript, including edits.
pub(super) struct Generator {
    stream: Owned<Stream>,
    generator: Owned<GeneratorHandle>,
    _model: Arc<Model>,
}
impl Generator {
    pub(super) fn next(&mut self) -> Result<Option<(u32, String)>, ExecError> {
        let runtime = &self.generator.runtime;
        let api = &runtime.api;
        if unsafe { (api.OgaGenerator_IsDone)(self.generator.ptr()) } {
            return Ok(None);
        }
        runtime.check(unsafe { (api.OgaGenerator_GenerateNextToken)(self.generator.ptr()) })?;
        let mut data = std::ptr::null();
        let mut count = 0;
        runtime.check(unsafe {
            (api.OgaGenerator_GetNextTokens)(self.generator.ptr(), &mut data, &mut count)
        })?;
        if data.is_null() || count != 1 {
            return Err(ExecError::Generation(format!(
                "ONNX expected one generated token, received {count}"
            )));
        }
        // SAFETY: GenAI owns this one-element sequence until its next operation.
        let id = unsafe { *data };
        let mut text = std::ptr::null();
        runtime
            .check(unsafe { (api.OgaTokenizerStreamDecode)(self.stream.ptr(), id, &mut text) })?;
        if id < 0 || text.is_null() {
            return Err(ExecError::Generation(
                "ONNX returned an invalid token or decoded string".into(),
            ));
        }
        // Streaming decoder retains incomplete UTF-8 bytes internally. Copy its
        // borrowed complete fragment before requesting the next one.
        let text = unsafe { CStr::from_ptr(text) }
            .to_str()
            .map_err(|e| ExecError::Generation(format!("ONNX invalid decoded UTF-8: {e}")))?
            .to_owned();
        Ok(Some((id as u32, text)))
    }
}
