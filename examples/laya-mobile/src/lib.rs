//! Small host boundary for Swift/C/JNI integration tests. Calls that load,
//! infer, resume or wait for shutdown belong on a host background queue.
use gen2::{
    Runtime,
    decision::{
        DecisionModel, DecisionOptions, DecisionRequest, EncodeOptions, ExecutionOptions,
        LoadOptions, LongStateOptions,
    },
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Handle {
    runtime: Runtime,
    model: DecisionModel,
    load: LoadOptions,
    epoch: u64,
}
static HANDLES: OnceLock<Mutex<HashMap<u64, Handle>>> = OnceLock::new();
static NEXT: AtomicU64 = AtomicU64::new(1);
fn handles() -> &'static Mutex<HashMap<u64, Handle>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}
fn get(id: u64) -> Result<Handle, String> {
    handles()
        .lock()
        .map_err(|_| "registry poisoned")?
        .get(&id)
        .cloned()
        .ok_or_else(|| "unknown handle".into())
}
fn reply(f: impl FnOnce() -> Result<serde_json::Value, String>) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let value = match result {
        Ok(Ok(v)) => serde_json::json!({"ok":true,"value":v}),
        Ok(Err(e)) => serde_json::json!({"ok":false,"error":e}),
        Err(_) => serde_json::json!({"ok":false,"error":"native boundary panicked"}),
    };
    CString::new(value.to_string())
        .expect("JSON escapes NUL")
        .into_raw()
}
// Safety: host owns a valid, NUL-terminated UTF-8 C string until this call returns.
unsafe fn text<'a>(input: *const c_char) -> Result<&'a str, String> {
    if input.is_null() {
        return Err("null input".into());
    }
    unsafe { CStr::from_ptr(input) }
        .to_str()
        .map_err(|e| e.to_string())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    bundle: String,
    native_library: Option<String>,
    resident_budget_mb: u64,
    #[serde(default = "queue")]
    queue_capacity: usize,
    #[serde(default = "threads")]
    intra_threads: usize,
    #[serde(default = "input_bytes")]
    max_input_bytes: usize,
    #[serde(default)]
    execution: ExecutionOptions,
}
fn queue() -> usize {
    LoadOptions::default().queue_capacity
}
fn threads() -> usize {
    LoadOptions::default().intra_threads
}
fn input_bytes() -> usize {
    LoadOptions::default().max_input_bytes
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CallOptions {
    encode: Option<EncodeOptions>,
    language: Option<String>,
    /// Relative monotonic deadline, including admission, queueing and inference.
    timeout_ms: Option<u64>,
}
impl CallOptions {
    fn native(self) -> Result<DecisionOptions, String> {
        let deadline = self
            .timeout_ms
            .map(|ms| {
                Instant::now()
                    .checked_add(Duration::from_millis(ms))
                    .ok_or_else(|| "timeout_ms exceeds the monotonic clock range".to_string())
            })
            .transpose()?;
        Ok(DecisionOptions {
            encode: self.encode,
            language: self.language,
            deadline,
            ..Default::default()
        })
    }
}
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScanOptions {
    window_tokens: Option<usize>,
    stride_tokens: Option<usize>,
    max_windows: usize,
}
impl Default for ScanOptions {
    fn default() -> Self {
        let defaults = LongStateOptions::default();
        Self {
            window_tokens: defaults.window_tokens,
            stride_tokens: defaults.stride_tokens,
            max_windows: defaults.max_windows,
        }
    }
}
#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Invocation {
    Describe,
    Decide {
        request: DecisionRequest,
        #[serde(default)]
        options: CallOptions,
    },
    Batch {
        requests: Vec<DecisionRequest>,
        #[serde(default)]
        options: CallOptions,
    },
    Long {
        request: DecisionRequest,
        #[serde(default)]
        scan: ScanOptions,
        #[serde(default)]
        options: CallOptions,
    },
}

/// # Safety
/// `config` must point to a valid NUL-terminated UTF-8 string for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen2_laya_open(config: *const c_char) -> *mut c_char {
    reply(|| {
        let cfg: Config =
            serde_json::from_str(unsafe { text(config) }?).map_err(|e| e.to_string())?;
        let runtime = Runtime::builder()
            .resident_memory_budget_mb(cfg.resident_budget_mb)
            .build()
            .map_err(|e| e.to_string())?;
        let load = LoadOptions {
            native_library: cfg.native_library.map(Into::into),
            queue_capacity: cfg.queue_capacity,
            intra_threads: cfg.intra_threads,
            max_input_bytes: cfg.max_input_bytes,
            execution: cfg.execution,
        };
        let model = runtime
            .load_decider(cfg.bundle, load.clone())
            .map_err(|e| e.to_string())?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        handles().lock().map_err(|_| "registry poisoned")?.insert(
            id,
            Handle {
                runtime,
                model,
                load,
                epoch: 0,
            },
        );
        Ok(serde_json::json!({"handle":id}))
    })
}
/// # Safety
/// `request` must point to a valid NUL-terminated UTF-8 string for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen2_laya_decide(id: u64, request: *const c_char) -> *mut c_char {
    reply(|| {
        let handle = get(id)?;
        let request: DecisionRequest =
            serde_json::from_str(unsafe { text(request) }?).map_err(|e| e.to_string())?;
        let result = handle
            .model
            .decide(request, DecisionOptions::default())
            .map_err(|e| e.to_string())?;
        serde_json::to_value(result).map_err(|e| e.to_string())
    })
}
/// JSON dispatch for describe, decide with options, multi-state batch and long scans.
/// Run inference operations on a background executor. A timeout never frees
/// resources still used by a native kernel; suspension cancels admitted work.
/// # Safety
/// `invocation` must point to a valid NUL-terminated UTF-8 string for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen2_laya_invoke(id: u64, invocation: *const c_char) -> *mut c_char {
    reply(|| {
        let call: Invocation =
            serde_json::from_str(unsafe { text(invocation) }?).map_err(|e| e.to_string())?;
        let handle = get(id)?;
        match call {
            Invocation::Describe => Ok(serde_json::json!({
                "info": handle.model.info(),
                "capabilities": handle.model.capabilities(),
                "status": format!("{:?}", handle.model.status()),
            })),
            Invocation::Decide { request, options } => {
                let result = handle
                    .model
                    .decide(request, options.native()?)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(result).map_err(|e| e.to_string())
            }
            Invocation::Batch { requests, options } => {
                let result = handle
                    .model
                    .decide_batch(requests, options.native()?)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(result).map_err(|e| e.to_string())
            }
            Invocation::Long {
                request,
                scan,
                options,
            } => {
                let result = handle
                    .model
                    .decide_long(
                        request,
                        LongStateOptions {
                            window_tokens: scan.window_tokens,
                            stride_tokens: scan.stride_tokens,
                            max_windows: scan.max_windows,
                        },
                        options.native()?,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(result).map_err(|e| e.to_string())
            }
        }
    })
}
/// Nonblocking suspension / memory-pressure hook. In-flight kernels keep buffers.
#[unsafe(no_mangle)]
pub extern "C" fn gen2_laya_suspend(id: u64) -> *mut c_char {
    reply(|| {
        let model = {
            let mut registry = handles().lock().map_err(|_| "registry poisoned")?;
            let entry = registry.get_mut(&id).ok_or("unknown handle")?;
            entry.epoch = entry.epoch.wrapping_add(1);
            entry.model.clone()
        };
        model.unload();
        Ok(serde_json::json!(null))
    })
}
/// Background-queue only: wait for suspension then reload the identical bundle.
#[unsafe(no_mangle)]
pub extern "C" fn gen2_laya_resume(id: u64) -> *mut c_char {
    reply(|| {
        let old = {
            let mut registry = handles().lock().map_err(|_| "registry poisoned")?;
            let entry = registry.get_mut(&id).ok_or("unknown handle")?;
            entry.epoch = entry.epoch.wrapping_add(1);
            entry.clone()
        };
        old.model.shutdown();
        let model = old
            .runtime
            .reload_decider(&old.model, old.load.clone())
            .map_err(|e| e.to_string())?;
        let mut registry = handles().lock().map_err(|_| "registry poisoned")?;
        let Some(entry) = registry.get_mut(&id) else {
            model.unload();
            return Err("handle closed while resuming".into());
        };
        if entry.epoch != old.epoch {
            model.unload();
            return Err("resume superseded by a lifecycle event".into());
        }
        entry.model = model;
        Ok(serde_json::json!(null))
    })
}
/// Remove a handle and wait for native resource release. Background-queue only.
#[unsafe(no_mangle)]
pub extern "C" fn gen2_laya_close(id: u64) -> *mut c_char {
    reply(|| {
        let old = handles()
            .lock()
            .map_err(|_| "registry poisoned")?
            .remove(&id)
            .ok_or("unknown handle")?;
        old.model.shutdown();
        Ok(serde_json::json!(null))
    })
}
/// # Safety
/// Pass each non-null pointer returned by this library exactly once. No other
/// thread may access the string during or after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen2_laya_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}
