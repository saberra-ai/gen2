//! The LiteRT-LM C ABI, loaded at runtime.
//!
//! gen2 does not vendor Google's runtime, link it, or download it during a
//! build: the feature compiles on a machine that has never heard of LiteRT-LM,
//! and the library is found when a model is loaded. That keeps
//! `backend-litertlm` a standalone feature like every other backend, and keeps
//! Bazel out of Cargo.
//!
//! # Required and optional
//!
//! Two groups, because the shipped runtime and the upstream source are not the
//! same thing. Every symbol in [`RequiredApi`] is present in the v0.16.0
//! prebuilt and the backend refuses to load without all of them.
//! [`CapabilityApi`] is loaded only if it happens to be there — as of that
//! release it is not, since `capabilities.h` exists upstream but ships neither
//! its header nor its symbols. Assuming a symbol exists because `main` has it
//! is how a dynamically loaded backend turns a missing feature into a crash.
//!
//! # Safety
//!
//! Every `unsafe` in this backend is in this file. Callers above it see owned
//! handles with `Drop`, and no raw pointer escapes.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Arc;

use libloading::{Library, Symbol};

use crate::engine::ExecError;

// ── Opaque handles ──────────────────────────────────────────────────────────
// Each is a C struct gen2 only ever holds a pointer to.

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {
        $(
            #[repr(C)]
            pub(super) struct $name {
                _private: [u8; 0],
            }
        )*
    };
}

opaque!(
    LiteRtLmEngine,
    LiteRtLmEngineSettings,
    LiteRtLmSessionConfig,
    LiteRtLmSamplerParams,
    LiteRtLmRepetitionPenaltyConfig,
    LiteRtLmConversation,
    LiteRtLmConversationConfig,
    LiteRtLmConversationOptionalArgs,
    LiteRtLmThinkingConfig,
    LiteRtLmStreamChunk,
    LiteRtLmLoadedFile,
);

/// Sampler selection, from `engine.h`.
#[allow(dead_code)]
pub(super) mod sampler_type {
    pub(in super::super) const TOP_K: i32 = 1;
    pub(in super::super) const TOP_P: i32 = 2;
    pub(in super::super) const GREEDY: i32 = 3;
}

/// Constraint selection, from `conversation.h`.
#[allow(dead_code)]
pub(super) mod constraint_type {
    pub(in super::super) const NONE: i32 = 0;
    pub(in super::super) const REGEX: i32 = 1;
    pub(in super::super) const JSON_SCHEMA: i32 = 2;
}

/// The streaming callback. Invoked from LiteRT-LM's own thread; `chunk` is
/// valid only for the duration of the call.
pub(super) type StreamCallback =
    unsafe extern "C" fn(callback_data: *mut c_void, chunk: *const LiteRtLmStreamChunk);

// ── The loaded library ──────────────────────────────────────────────────────

/// Function pointers the backend cannot work without.
///
/// Held alongside the `Library` they came from, and never outliving it — the
/// `Arc<Runtime>` every handle carries is what enforces that.
pub(super) struct RequiredApi {
    pub engine_settings_create: unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
    ) -> *mut LiteRtLmEngineSettings,
    pub engine_settings_delete: unsafe extern "C" fn(*mut LiteRtLmEngineSettings),
    pub engine_settings_set_max_num_tokens:
        unsafe extern "C" fn(*mut LiteRtLmEngineSettings, c_int),
    pub engine_settings_set_num_threads: unsafe extern "C" fn(*mut LiteRtLmEngineSettings, c_int),
    pub engine_settings_set_cache_dir:
        unsafe extern "C" fn(*mut LiteRtLmEngineSettings, *const c_char),
    pub engine_create: unsafe extern "C" fn(*const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine,
    pub engine_delete: unsafe extern "C" fn(*mut LiteRtLmEngine),

    pub sampler_params_create: unsafe extern "C" fn(c_int) -> *mut LiteRtLmSamplerParams,
    pub sampler_params_delete: unsafe extern "C" fn(*mut LiteRtLmSamplerParams),
    pub sampler_params_set_temperature: unsafe extern "C" fn(*mut LiteRtLmSamplerParams, f32),
    pub sampler_params_set_top_k: unsafe extern "C" fn(*mut LiteRtLmSamplerParams, i32),
    pub sampler_params_set_top_p: unsafe extern "C" fn(*mut LiteRtLmSamplerParams, f32),
    pub sampler_params_set_seed: unsafe extern "C" fn(*mut LiteRtLmSamplerParams, i32),

    pub session_config_create: unsafe extern "C" fn() -> *mut LiteRtLmSessionConfig,
    pub session_config_delete: unsafe extern "C" fn(*mut LiteRtLmSessionConfig),
    pub session_config_set_sampler_params:
        unsafe extern "C" fn(*mut LiteRtLmSessionConfig, *const LiteRtLmSamplerParams),
    pub session_config_set_max_output_tokens:
        unsafe extern "C" fn(*mut LiteRtLmSessionConfig, c_int),

    pub repetition_penalty_config_create:
        unsafe extern "C" fn() -> *mut LiteRtLmRepetitionPenaltyConfig,
    pub repetition_penalty_config_delete:
        unsafe extern "C" fn(*mut LiteRtLmRepetitionPenaltyConfig),
    pub repetition_penalty_config_set_repetition_penalty:
        unsafe extern "C" fn(*mut LiteRtLmRepetitionPenaltyConfig, f32),
    pub repetition_penalty_config_set_frequency_penalty:
        unsafe extern "C" fn(*mut LiteRtLmRepetitionPenaltyConfig, f32),
    pub repetition_penalty_config_set_presence_penalty:
        unsafe extern "C" fn(*mut LiteRtLmRepetitionPenaltyConfig, f32),
    pub repetition_penalty_config_set_window_size:
        unsafe extern "C" fn(*mut LiteRtLmRepetitionPenaltyConfig, c_int),

    pub conversation_config_create: unsafe extern "C" fn() -> *mut LiteRtLmConversationConfig,
    pub conversation_config_delete: unsafe extern "C" fn(*mut LiteRtLmConversationConfig),
    pub conversation_config_set_session_config:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, *const LiteRtLmSessionConfig),
    pub conversation_config_set_system_message:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, *const c_char),
    pub conversation_config_set_tools:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, *const c_char),
    pub conversation_config_set_messages:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, *const c_char),
    pub conversation_config_set_thinking_config:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, *const LiteRtLmThinkingConfig),
    pub conversation_config_set_enable_constrained_decoding:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, bool),
    pub conversation_config_set_stream_tool_calls:
        unsafe extern "C" fn(*mut LiteRtLmConversationConfig, bool, *const c_char),

    pub conversation_create: unsafe extern "C" fn(
        *mut LiteRtLmEngine,
        *mut LiteRtLmConversationConfig,
    ) -> *mut LiteRtLmConversation,
    pub conversation_delete: unsafe extern "C" fn(*mut LiteRtLmConversation),
    pub conversation_send_message_stream: unsafe extern "C" fn(
        *mut LiteRtLmConversation,
        *const c_char,
        *const c_char,
        *const LiteRtLmConversationOptionalArgs,
        StreamCallback,
        *mut c_void,
    ) -> c_int,
    pub conversation_cancel_process: unsafe extern "C" fn(*mut LiteRtLmConversation),
    pub conversation_get_token_count: unsafe extern "C" fn(*mut LiteRtLmConversation) -> c_int,

    pub optional_args_create: unsafe extern "C" fn() -> *mut LiteRtLmConversationOptionalArgs,
    pub optional_args_delete: unsafe extern "C" fn(*mut LiteRtLmConversationOptionalArgs),
    pub optional_args_set_max_output_tokens:
        unsafe extern "C" fn(*mut LiteRtLmConversationOptionalArgs, c_int),
    pub optional_args_set_constraint:
        unsafe extern "C" fn(*mut LiteRtLmConversationOptionalArgs, c_int, *const c_char),
    pub optional_args_set_repetition_penalty_config: unsafe extern "C" fn(
        *mut LiteRtLmConversationOptionalArgs,
        *const LiteRtLmRepetitionPenaltyConfig,
    ),

    pub thinking_config_create: unsafe extern "C" fn() -> *mut LiteRtLmThinkingConfig,
    pub thinking_config_delete: unsafe extern "C" fn(*mut LiteRtLmThinkingConfig),
    pub thinking_config_set_enable_thinking:
        unsafe extern "C" fn(*mut LiteRtLmThinkingConfig, bool),

    pub stream_chunk_get_text: unsafe extern "C" fn(*const LiteRtLmStreamChunk) -> *const c_char,
    pub stream_chunk_get_error: unsafe extern "C" fn(*const LiteRtLmStreamChunk) -> *const c_char,
    pub stream_chunk_is_final: unsafe extern "C" fn(*const LiteRtLmStreamChunk) -> bool,
}

/// Model introspection, present only on runtimes built from a source tree that
/// has it.
///
/// The v0.16.0 prebuilt does not export these, so everything reading them must
/// cope with `None` rather than assuming a newer runtime.
pub(super) struct CapabilityApi {
    pub loaded_file_create: unsafe extern "C" fn(*const c_char) -> *mut LiteRtLmLoadedFile,
    pub loaded_file_delete: unsafe extern "C" fn(*mut LiteRtLmLoadedFile),
    pub loaded_file_max_context_tokens: unsafe extern "C" fn(*mut LiteRtLmLoadedFile) -> u32,
    pub loaded_file_supports_function_calling:
        unsafe extern "C" fn(*mut LiteRtLmLoadedFile) -> bool,
    pub loaded_file_supports_thinking: unsafe extern "C" fn(*mut LiteRtLmLoadedFile) -> bool,
    pub loaded_file_supports_input_modality:
        unsafe extern "C" fn(*mut LiteRtLmLoadedFile, c_int) -> bool,
}

/// The library, plus everything resolved out of it.
///
/// Shared by `Arc` so that a conversation cannot outlive the code that
/// implements it — unloading the library while a handle is alive would leave
/// function pointers into freed memory.
pub(super) struct Runtime {
    pub required: RequiredApi,
    pub capabilities: Option<CapabilityApi>,
    /// Last, so it drops last. Every pointer above points into it.
    _library: Library,
}

/// The environment variable naming the runtime explicitly.
///
/// An operational escape hatch for development and embedding, deliberately an
/// environment variable rather than a gen2 setting: which file the runtime
/// lives in is a deployment fact, not something an application's code should
/// be expressing.
const LIBRARY_PATH_VAR: &str = "GEN2_LITERTLM_LIBRARY";

/// Where to look for the runtime.
///
/// Takes the override rather than reading it, so the tests below never touch
/// process-global state. They did once, and clearing the variable in one test
/// made every live test that ran after it fail to find a runtime that was
/// installed the whole time.
fn candidate_paths(explicit: Option<&str>) -> Vec<PathBuf> {
    if let Some(explicit) = explicit.filter(|p| !p.is_empty()) {
        return vec![PathBuf::from(explicit)];
    }
    // The names Google's own prebuilts use, left for the platform loader to
    // resolve against its normal search path so an application that bundles
    // the runtime needs no configuration.
    let bare = if cfg!(target_os = "macos") {
        ["liblitert-lm.dylib", "litert-lm.dylib"].as_slice()
    } else if cfg!(target_os = "windows") {
        ["litert-lm.dll"].as_slice()
    } else {
        ["liblitert-lm.so", "litert-lm.so"].as_slice()
    };
    bare.iter().map(PathBuf::from).collect()
}

macro_rules! resolve {
    ($lib:expr, $name:literal) => {{
        let symbol: Symbol<_> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|_| missing_symbol($name))?;
        *symbol
    }};
}

/// Optional symbol: absent is an answer, not a failure.
macro_rules! resolve_optional {
    ($lib:expr, $name:literal) => {{
        let symbol: Option<Symbol<_>> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.ok();
        match symbol {
            Some(s) => *s,
            None => return None,
        }
    }};
}

fn missing_symbol(name: &str) -> ExecError {
    ExecError::Other(anyhow::anyhow!(
        "the LiteRT-LM runtime was found but does not export `{name}`. It is \
         probably older than the C API this backend targets (v0.1.0, shipped \
         with LiteRT-LM v0.16.0)."
    ))
}

impl Runtime {
    /// Find and load the runtime.
    ///
    /// The error names every path tried, because "library not found" without
    /// saying where it looked is the least actionable message a dynamically
    /// loaded backend can produce.
    pub(super) fn load() -> Result<Arc<Self>, ExecError> {
        let mut tried = Vec::new();
        let explicit = std::env::var(LIBRARY_PATH_VAR).ok();
        for path in candidate_paths(explicit.as_deref()) {
            match unsafe { Library::new(&path) } {
                Ok(library) => return Self::bind(library).map(Arc::new),
                Err(e) => tried.push(format!("{}: {e}", path.display())),
            }
        }
        Err(ExecError::Other(anyhow::anyhow!(
            "could not load the LiteRT-LM runtime. gen2 does not bundle it — \
             install or embed Google's LiteRT-LM C runtime, or set \
             GEN2_LITERTLM_LIBRARY to its path. Tried: [{}]",
            tried.join(" | ")
        )))
    }

    fn bind(library: Library) -> Result<Self, ExecError> {
        let required = RequiredApi {
            engine_settings_create: resolve!(library, "litert_lm_engine_settings_create"),
            engine_settings_delete: resolve!(library, "litert_lm_engine_settings_delete"),
            engine_settings_set_max_num_tokens: resolve!(
                library,
                "litert_lm_engine_settings_set_max_num_tokens"
            ),
            engine_settings_set_num_threads: resolve!(
                library,
                "litert_lm_engine_settings_set_num_threads"
            ),
            engine_settings_set_cache_dir: resolve!(
                library,
                "litert_lm_engine_settings_set_cache_dir"
            ),
            engine_create: resolve!(library, "litert_lm_engine_create"),
            engine_delete: resolve!(library, "litert_lm_engine_delete"),

            sampler_params_create: resolve!(library, "litert_lm_sampler_params_create"),
            sampler_params_delete: resolve!(library, "litert_lm_sampler_params_delete"),
            sampler_params_set_temperature: resolve!(
                library,
                "litert_lm_sampler_params_set_temperature"
            ),
            sampler_params_set_top_k: resolve!(library, "litert_lm_sampler_params_set_top_k"),
            sampler_params_set_top_p: resolve!(library, "litert_lm_sampler_params_set_top_p"),
            sampler_params_set_seed: resolve!(library, "litert_lm_sampler_params_set_seed"),

            session_config_create: resolve!(library, "litert_lm_session_config_create"),
            session_config_delete: resolve!(library, "litert_lm_session_config_delete"),
            session_config_set_sampler_params: resolve!(
                library,
                "litert_lm_session_config_set_sampler_params"
            ),
            session_config_set_max_output_tokens: resolve!(
                library,
                "litert_lm_session_config_set_max_output_tokens"
            ),

            repetition_penalty_config_create: resolve!(
                library,
                "litert_lm_repetition_penalty_config_create"
            ),
            repetition_penalty_config_delete: resolve!(
                library,
                "litert_lm_repetition_penalty_config_delete"
            ),
            repetition_penalty_config_set_repetition_penalty: resolve!(
                library,
                "litert_lm_repetition_penalty_config_set_repetition_penalty"
            ),
            repetition_penalty_config_set_frequency_penalty: resolve!(
                library,
                "litert_lm_repetition_penalty_config_set_frequency_penalty"
            ),
            repetition_penalty_config_set_presence_penalty: resolve!(
                library,
                "litert_lm_repetition_penalty_config_set_presence_penalty"
            ),
            repetition_penalty_config_set_window_size: resolve!(
                library,
                "litert_lm_repetition_penalty_config_set_window_size"
            ),

            conversation_config_create: resolve!(library, "litert_lm_conversation_config_create"),
            conversation_config_delete: resolve!(library, "litert_lm_conversation_config_delete"),
            conversation_config_set_session_config: resolve!(
                library,
                "litert_lm_conversation_config_set_session_config"
            ),
            conversation_config_set_system_message: resolve!(
                library,
                "litert_lm_conversation_config_set_system_message"
            ),
            conversation_config_set_tools: resolve!(
                library,
                "litert_lm_conversation_config_set_tools"
            ),
            conversation_config_set_messages: resolve!(
                library,
                "litert_lm_conversation_config_set_messages"
            ),
            conversation_config_set_thinking_config: resolve!(
                library,
                "litert_lm_conversation_config_set_thinking_config"
            ),
            conversation_config_set_enable_constrained_decoding: resolve!(
                library,
                "litert_lm_conversation_config_set_enable_constrained_decoding"
            ),
            conversation_config_set_stream_tool_calls: resolve!(
                library,
                "litert_lm_conversation_config_set_stream_tool_calls"
            ),

            conversation_create: resolve!(library, "litert_lm_conversation_create"),
            conversation_delete: resolve!(library, "litert_lm_conversation_delete"),
            conversation_send_message_stream: resolve!(
                library,
                "litert_lm_conversation_send_message_stream"
            ),
            conversation_cancel_process: resolve!(library, "litert_lm_conversation_cancel_process"),
            conversation_get_token_count: resolve!(
                library,
                "litert_lm_conversation_get_token_count"
            ),

            optional_args_create: resolve!(library, "litert_lm_conversation_optional_args_create"),
            optional_args_delete: resolve!(library, "litert_lm_conversation_optional_args_delete"),
            optional_args_set_max_output_tokens: resolve!(
                library,
                "litert_lm_conversation_optional_args_set_max_output_tokens"
            ),
            optional_args_set_constraint: resolve!(
                library,
                "litert_lm_conversation_optional_args_set_constraint"
            ),
            optional_args_set_repetition_penalty_config: resolve!(
                library,
                "litert_lm_conversation_optional_args_set_repetition_penalty_config"
            ),

            thinking_config_create: resolve!(library, "litert_lm_thinking_config_create"),
            thinking_config_delete: resolve!(library, "litert_lm_thinking_config_delete"),
            thinking_config_set_enable_thinking: resolve!(
                library,
                "litert_lm_thinking_config_set_enable_thinking"
            ),

            stream_chunk_get_text: resolve!(library, "litert_lm_stream_chunk_get_text"),
            stream_chunk_get_error: resolve!(library, "litert_lm_stream_chunk_get_error"),
            stream_chunk_is_final: resolve!(library, "litert_lm_stream_chunk_is_final"),
        };
        let capabilities = bind_capabilities(&library);
        Ok(Self {
            required,
            capabilities,
            _library: library,
        })
    }
}

fn bind_capabilities(library: &Library) -> Option<CapabilityApi> {
    Some(CapabilityApi {
        loaded_file_create: resolve_optional!(library, "litert_lm_loaded_file_create"),
        loaded_file_delete: resolve_optional!(library, "litert_lm_loaded_file_delete"),
        loaded_file_max_context_tokens: resolve_optional!(
            library,
            "litert_lm_loaded_file_max_context_tokens"
        ),
        loaded_file_supports_function_calling: resolve_optional!(
            library,
            "litert_lm_loaded_file_supports_function_calling"
        ),
        loaded_file_supports_thinking: resolve_optional!(
            library,
            "litert_lm_loaded_file_supports_thinking"
        ),
        loaded_file_supports_input_modality: resolve_optional!(
            library,
            "litert_lm_loaded_file_supports_input_modality"
        ),
    })
}

/// Borrow a Rust string as a C string for the duration of a call.
pub(super) fn c_string(text: &str) -> Result<CString, ExecError> {
    CString::new(text)
        .map_err(|_| ExecError::InvalidArg("a string crossing into LiteRT-LM contained a NUL byte"))
}

/// Read a C string the library owns. `None` for null.
///
/// # Safety
/// `ptr` must be null or point at a NUL-terminated string valid for this call.
pub(super) unsafe fn read_c_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

// ── Owned handles ───────────────────────────────────────────────────────────
//
// Every raw pointer the C API hands back is wrapped before it leaves this
// file. Each wrapper owns exactly one allocation, frees it once in `Drop`, and
// keeps an `Arc<Runtime>` so the library that implements `delete` cannot be
// unloaded first.

/// A configured engine: one loaded model, shared by its conversations.
pub(super) struct OwnedEngine {
    ptr: *mut LiteRtLmEngine,
    rt: Arc<Runtime>,
}

impl OwnedEngine {
    /// Build settings, apply them, and create the engine.
    ///
    /// `max_num_tokens` is passed only when the caller actually supplied a
    /// context size. Inventing one here would make [`super::engine::n_ctx`]
    /// report a number nothing configured — a backend that lies to the
    /// controller about context capacity produces truncation decisions made
    /// against a fiction.
    pub(super) fn create(
        rt: Arc<Runtime>,
        model_path: &str,
        backend: &str,
        max_num_tokens: Option<i32>,
        num_threads: Option<i32>,
        cache_dir: Option<&str>,
    ) -> Result<Self, ExecError> {
        let api = &rt.required;
        let path = c_string(model_path)?;
        let backend_str = backend;
        let backend = c_string(backend)?;

        // SAFETY: both strings outlive the call; the settings pointer is
        // checked before use and freed on every path out.
        let settings = unsafe {
            (api.engine_settings_create)(
                path.as_ptr(),
                backend.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if settings.is_null() {
            return Err(ExecError::InvalidModelFile(format!(
                "LiteRT-LM rejected the model path `{model_path}`"
            )));
        }
        // From here on `settings` must be freed before returning.
        let result = (|| -> Result<*mut LiteRtLmEngine, ExecError> {
            // SAFETY: `settings` is non-null and owned by this scope.
            unsafe {
                if let Some(n) = max_num_tokens {
                    (api.engine_settings_set_max_num_tokens)(settings, n);
                }
                if let Some(n) = num_threads {
                    (api.engine_settings_set_num_threads)(settings, n);
                }
                if let Some(dir) = cache_dir {
                    let dir = c_string(dir)?;
                    (api.engine_settings_set_cache_dir)(settings, dir.as_ptr());
                }
                let engine = (api.engine_create)(settings);
                if engine.is_null() {
                    // Deliberately not `InvalidModelFile`: the C API gives one
                    // null for two very different causes — a bundle this
                    // runtime cannot read, and an accelerator this device
                    // cannot set up — and `InvalidModelFile` is classified
                    // fatal, so it would stop the controller's load ladder from
                    // ever retrying on the CPU. A retryable error lets the
                    // ladder do its job; if the file really is the problem, the
                    // CPU rung fails too and the caller still hears about it.
                    return Err(ExecError::Other(anyhow::anyhow!(
                        "LiteRT-LM could not open `{model_path}` on the `{backend_str}` \
                         accelerator. Either the file is not a `.litertlm` bundle this \
                         runtime can read, or this device has no working `{backend_str}` \
                         path."
                    )));
                }
                Ok(engine)
            }
        })();
        // SAFETY: `settings` came from `engine_settings_create` and is freed
        // exactly once, whether the engine was created or not.
        unsafe { (api.engine_settings_delete)(settings) };

        Ok(Self {
            ptr: result?,
            rt: Arc::clone(&rt),
        })
    }

    pub(super) fn runtime(&self) -> &Arc<Runtime> {
        &self.rt
    }
}

impl Drop for OwnedEngine {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `engine_create`, is non-null, and this is
        // the only owner.
        unsafe { (self.rt.required.engine_delete)(self.ptr) };
    }
}

/// What a conversation should be created with.
pub(super) struct ConversationSetup<'a> {
    pub system_message: Option<&'a str>,
    /// A JSON array of messages the conversation opens with.
    pub messages_json: &'a str,
    /// A JSON array of tool declarations, or `None` for a conversation with
    /// no tools.
    pub tools_json: Option<&'a str>,
    pub sampler: Option<super::convert::Sampler>,
    pub max_output_tokens: Option<i32>,
    pub enable_thinking: Option<bool>,
    pub constrained_decoding: bool,
}

/// One conversation, holding LiteRT-LM's own prefilled state.
pub(super) struct OwnedConversation {
    ptr: *mut LiteRtLmConversation,
    rt: Arc<Runtime>,
}

impl OwnedConversation {
    pub(super) fn create(
        engine: &OwnedEngine,
        setup: ConversationSetup<'_>,
    ) -> Result<Self, ExecError> {
        let rt = Arc::clone(&engine.rt);
        let api = &rt.required;

        // Every child object below is deep-copied by the setter that consumes
        // it, so each is freed here rather than being kept alive alongside the
        // conversation.
        // SAFETY: each pointer is null-checked before use and freed exactly
        // once by the guards below.
        let sampler = match &setup.sampler {
            Some(s) => {
                let ptr = unsafe { (api.sampler_params_create)(s.kind) };
                if ptr.is_null() {
                    return Err(ExecError::Other(anyhow::anyhow!(
                        "LiteRT-LM rejected the sampler configuration"
                    )));
                }
                ptr
            }
            // Nobody asked for any sampling, so the runtime's own default
            // stands. Building one here would replace a configuration the
            // model shipped with by values gen2 invented.
            None => std::ptr::null_mut(),
        };
        let session_config = unsafe { (api.session_config_create)() };
        let conversation_config = unsafe { (api.conversation_config_create)() };
        let thinking = setup
            .enable_thinking
            .map(|_| unsafe { (api.thinking_config_create)() });

        let built = (|| -> Result<*mut LiteRtLmConversation, ExecError> {
            if session_config.is_null() || conversation_config.is_null() {
                return Err(ExecError::Other(anyhow::anyhow!(
                    "LiteRT-LM could not allocate a conversation configuration"
                )));
            }
            // SAFETY: every pointer here is non-null and owned by this scope;
            // every string outlives the call that reads it.
            unsafe {
                if let Some(wanted) = &setup.sampler {
                    if let Some(t) = wanted.temperature {
                        (api.sampler_params_set_temperature)(sampler, t);
                    }
                    if let Some(k) = wanted.top_k {
                        (api.sampler_params_set_top_k)(sampler, k);
                    }
                    if let Some(p) = wanted.top_p {
                        (api.sampler_params_set_top_p)(sampler, p);
                    }
                    if let Some(s) = wanted.seed {
                        (api.sampler_params_set_seed)(sampler, s);
                    }
                    (api.session_config_set_sampler_params)(session_config, sampler);
                }
                if let Some(n) = setup.max_output_tokens {
                    (api.session_config_set_max_output_tokens)(session_config, n);
                }

                (api.conversation_config_set_session_config)(conversation_config, session_config);
                if let Some(system) = setup.system_message {
                    let system = c_string(system)?;
                    (api.conversation_config_set_system_message)(
                        conversation_config,
                        system.as_ptr(),
                    );
                }
                if let Some(tools) = setup.tools_json {
                    let tools = c_string(tools)?;
                    (api.conversation_config_set_tools)(conversation_config, tools.as_ptr());
                }
                let messages = c_string(setup.messages_json)?;
                (api.conversation_config_set_messages)(conversation_config, messages.as_ptr());
                if let (Some(cfg), Some(on)) = (thinking, setup.enable_thinking) {
                    (api.thinking_config_set_enable_thinking)(cfg, on);
                    (api.conversation_config_set_thinking_config)(conversation_config, cfg);
                }
                (api.conversation_config_set_enable_constrained_decoding)(
                    conversation_config,
                    setup.constrained_decoding,
                );

                let conversation = (api.conversation_create)(engine.ptr, conversation_config);
                if conversation.is_null() {
                    return Err(ExecError::Other(anyhow::anyhow!(
                        "LiteRT-LM could not open a conversation on the loaded model"
                    )));
                }
                Ok(conversation)
            }
        })();

        // SAFETY: each was created above and is freed exactly once, on both
        // the success and failure paths.
        unsafe {
            if let Some(cfg) = thinking.filter(|c| !c.is_null()) {
                (api.thinking_config_delete)(cfg);
            }
            if !conversation_config.is_null() {
                (api.conversation_config_delete)(conversation_config);
            }
            if !session_config.is_null() {
                (api.session_config_delete)(session_config);
            }
            if !sampler.is_null() {
                (api.sampler_params_delete)(sampler);
            }
        }

        Ok(Self { ptr: built?, rt })
    }

    /// How many tokens the conversation currently holds, as LiteRT-LM counts
    /// them.
    pub(super) fn token_count(&self) -> usize {
        // SAFETY: `ptr` is a live conversation owned by `self`.
        let n = unsafe { (self.rt.required.conversation_get_token_count)(self.ptr) };
        n.max(0) as usize
    }

    /// Start streaming a reply. Returns as soon as the stream is running; the
    /// chunks arrive on `sink` from LiteRT-LM's own thread.
    ///
    /// The sink is handed over as a raw `Arc` so it stays alive for exactly as
    /// long as the callback might touch it — see [`StreamSink`].
    pub(super) fn send_stream(
        &self,
        message_json: &str,
        options: &OwnedOptionalArgs,
        sink: Arc<StreamSink>,
    ) -> Result<(), ExecError> {
        let message = c_string(message_json)?;
        // The callback holds this raw pointer for the life of the stream. It
        // is reclaimed by whoever observes the final chunk.
        let raw = Arc::into_raw(sink) as *mut c_void;

        // SAFETY: `message` outlives the call (the stream copies what it
        // needs before returning), `options` is a live args object, and `raw`
        // is a valid `Arc<StreamSink>` pointer that outlives every callback.
        let rc = unsafe {
            (self.rt.required.conversation_send_message_stream)(
                self.ptr,
                message.as_ptr(),
                std::ptr::null(),
                options.ptr,
                trampoline,
                raw,
            )
        };
        if rc != 0 {
            // The stream never started, so no callback will ever run and the
            // reference handed over above is ours to take back.
            // SAFETY: `raw` came from `Arc::into_raw` and nothing else holds
            // it — the callback was never invoked.
            drop(unsafe { Arc::from_raw(raw as *const StreamSink) });
            return Err(ExecError::Generation(format!(
                "LiteRT-LM refused to start generation (code {rc})"
            )));
        }
        Ok(())
    }
}

impl Drop for OwnedConversation {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `conversation_create` and this is the only
        // owner. A `Canceller` may still exist, which is why the session drops
        // its cancellers before its conversations.
        unsafe { (self.rt.required.conversation_delete)(self.ptr) };
    }
}

/// Stops a running generation.
///
/// Held by the puller so a caller's `stop()` reaches the runtime. It needs no
/// `Send`: `send_message_stream` returns as soon as the stream is running, so
/// the generation is started and cancelled from the same thread — the one the
/// controller confines every backend to.
///
/// It holds the conversation by `Arc` rather than by raw pointer, so the thing
/// it cancels cannot be destroyed while the handle exists. That is not a
/// stylistic choice: a live run had LiteRT-LM report "destructed with 1 living
/// sessions", which is what a lifetime kept by convention rather than by the
/// type system eventually produces.
pub(super) struct Canceller {
    conversation: Arc<OwnedConversation>,
}

impl Canceller {
    pub(super) fn of(conversation: &Arc<OwnedConversation>) -> Self {
        Self {
            conversation: Arc::clone(conversation),
        }
    }

    pub(super) fn cancel(&self) {
        let c = &self.conversation;
        // SAFETY: the `Arc` keeps the conversation alive for this call.
        unsafe { (c.rt.required.conversation_cancel_process)(c.ptr) };
    }
}

/// Per-request arguments: output cap, constraint, penalties.
pub(super) struct OwnedOptionalArgs {
    ptr: *mut LiteRtLmConversationOptionalArgs,
    rt: Arc<Runtime>,
}

impl OwnedOptionalArgs {
    pub(super) fn new(rt: Arc<Runtime>) -> Result<Self, ExecError> {
        // SAFETY: no arguments; the result is null-checked.
        let ptr = unsafe { (rt.required.optional_args_create)() };
        if ptr.is_null() {
            return Err(ExecError::Other(anyhow::anyhow!(
                "LiteRT-LM could not allocate request arguments"
            )));
        }
        Ok(Self { ptr, rt })
    }

    pub(super) fn set_max_output_tokens(&mut self, n: i32) {
        // SAFETY: `ptr` is a live args object owned by `self`.
        unsafe { (self.rt.required.optional_args_set_max_output_tokens)(self.ptr, n) };
    }

    pub(super) fn set_constraint(&mut self, kind: i32, body: &str) -> Result<(), ExecError> {
        let body = c_string(body)?;
        // SAFETY: `body` outlives the call, which copies what it needs.
        unsafe { (self.rt.required.optional_args_set_constraint)(self.ptr, kind, body.as_ptr()) };
        Ok(())
    }

    /// Repetition, frequency and presence penalties, as one config.
    ///
    /// `window` is how many recent tokens they look back over — gen2's
    /// `penalty_last_n`. `None` leaves the runtime's own window alone.
    pub(super) fn set_penalties(
        &mut self,
        repeat: f32,
        freq: f32,
        present: f32,
        window: Option<i32>,
    ) -> Result<(), ExecError> {
        let api = &self.rt.required;
        // SAFETY: null-checked, configured, attached (the setter deep-copies)
        // and freed on every path.
        unsafe {
            let cfg = (api.repetition_penalty_config_create)();
            if cfg.is_null() {
                return Err(ExecError::Other(anyhow::anyhow!(
                    "LiteRT-LM could not allocate a repetition penalty config"
                )));
            }
            (api.repetition_penalty_config_set_repetition_penalty)(cfg, repeat);
            (api.repetition_penalty_config_set_frequency_penalty)(cfg, freq);
            (api.repetition_penalty_config_set_presence_penalty)(cfg, present);
            if let Some(n) = window {
                (api.repetition_penalty_config_set_window_size)(cfg, n);
            }
            (api.optional_args_set_repetition_penalty_config)(self.ptr, cfg);
            (api.repetition_penalty_config_delete)(cfg);
        }
        Ok(())
    }
}

impl Drop for OwnedOptionalArgs {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `optional_args_create`; sole owner.
        unsafe { (self.rt.required.optional_args_delete)(self.ptr) };
    }
}

// ── The streaming callback ──────────────────────────────────────────────────

/// One piece of a reply, as it crossed out of LiteRT-LM.
#[derive(Debug)]
pub(super) enum Chunk {
    Text(String),
    Failed(String),
    Done,
}

/// Where callback chunks land.
///
/// Shared between the puller, which reads, and LiteRT-LM's background thread,
/// which writes. `finished` is what lets the puller know whether the stream is
/// over — and therefore whether the reference the callback holds has already
/// been released.
///
/// The runtime rides along because the callback needs it. LiteRT-LM creates
/// the thread that calls back, so nothing of gen2's is reachable from there
/// except `callback_data` — and the chunk accessors live on a dynamically
/// loaded library, so without the runtime in hand the callback could not read
/// its own argument.
pub(super) struct StreamSink {
    /// Behind a lock because the callback reaches it through a shared
    /// reference from LiteRT-LM's own thread, and `Sender` is `Send` but not
    /// `Sync`. Nothing documents that the runtime uses one callback thread for
    /// the life of a stream, and an uncontended lock per token is a small
    /// price for not depending on that.
    tx: parking_lot::Mutex<std::sync::mpsc::Sender<Chunk>>,
    finished: Arc<std::sync::atomic::AtomicBool>,
    rt: Arc<Runtime>,
}

/// The sink is handed to a foreign thread as a raw pointer, which switches off
/// every check the compiler would otherwise make. This turns one back on: a
/// field that is not safe to share across threads fails the build here rather
/// than becoming a data race in a callback.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StreamSink>();
};

impl StreamSink {
    /// The sink, the channel it writes to, and the flag it sets when the
    /// stream ends.
    pub(super) fn new(
        rt: Arc<Runtime>,
    ) -> (
        Arc<Self>,
        std::sync::mpsc::Receiver<Chunk>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel();
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        (
            Arc::new(Self {
                tx: parking_lot::Mutex::new(tx),
                finished: Arc::clone(&finished),
                rt,
            }),
            rx,
            finished,
        )
    }
}

/// The `extern "C"` entry point LiteRT-LM calls, once per chunk, on its own
/// thread.
///
/// Two rules keep it sound. It must not unwind into C, so everything it does
/// is wrapped in `catch_unwind`. And it must not free the sink while another
/// chunk could still be in flight, so the reference it was given is only
/// released after a chunk that says the stream is over.
///
/// # Safety
/// `callback_data` must be a pointer produced by `Arc::into_raw` on a
/// `StreamSink` that no other code has reclaimed. `chunk` must be valid for
/// the duration of this call.
unsafe extern "C" fn trampoline(callback_data: *mut c_void, chunk: *const LiteRtLmStreamChunk) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if callback_data.is_null() {
            return;
        }
        // Borrowed, not owned: the strong count handed to
        // `send_message_stream` is released below and only on the final chunk.
        // SAFETY: the caller guarantees `callback_data` came from
        // `Arc::into_raw` on a live `StreamSink`.
        let sink = unsafe { &*(callback_data as *const StreamSink) };

        let (text, error, final_chunk) = if chunk.is_null() {
            // A null chunk is not a shape the header documents. Treating it as
            // the end is the only safe reading: continuing would mean waiting
            // forever for a stream that has stopped speaking.
            (None, None, true)
        } else {
            let api = &sink.rt.required;
            // SAFETY: `chunk` is valid for the duration of this call, and each
            // accessor returns a string owned by the chunk.
            unsafe {
                (
                    read_c_string((api.stream_chunk_get_text)(chunk)),
                    read_c_string((api.stream_chunk_get_error)(chunk)),
                    (api.stream_chunk_is_final)(chunk),
                )
            }
        };

        if let Some(message) = error.filter(|e| !e.is_empty()) {
            let _ = sink.tx.lock().send(Chunk::Failed(message));
        } else if let Some(text) = text.filter(|t| !t.is_empty()) {
            let _ = sink.tx.lock().send(Chunk::Text(text));
        }

        if final_chunk {
            sink.finished
                .store(true, std::sync::atomic::Ordering::Release);
            let _ = sink.tx.lock().send(Chunk::Done);
            // The stream is over, so nothing will touch the sink again and the
            // reference handed to `send_message_stream` is released here.
            // SAFETY: this runs exactly once — `is_final` is true for one
            // chunk, and a null chunk ends the stream the same way.
            drop(unsafe { Arc::from_raw(callback_data as *const StreamSink) });
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing symbol has to say which one and why, because the alternative
    /// is a caller staring at "load failed" with a runtime that did load.
    #[test]
    fn a_missing_symbol_names_itself_and_says_what_is_wrong() {
        let err = missing_symbol("litert_lm_engine_create");
        let text = err.to_string();
        assert!(
            text.contains("litert_lm_engine_create"),
            "the error should name the symbol, got: {text}"
        );
        assert!(
            text.contains("older"),
            "the error should say what to do about it, got: {text}"
        );
    }

    /// The search path is the first thing anyone debugging a dynamic backend
    /// needs, and an environment override is the only way to point gen2 at a
    /// runtime that is not on the loader's path.
    #[test]
    fn an_explicit_library_path_wins_over_the_platform_default() {
        let paths = candidate_paths(Some("/nowhere/liblitert-lm.dylib"));
        assert_eq!(
            paths.len(),
            1,
            "an explicit path should not be second-guessed"
        );
        assert_eq!(paths[0], PathBuf::from("/nowhere/liblitert-lm.dylib"));
    }

    /// An empty variable means "unset", not "look in a file called nothing".
    #[test]
    fn an_empty_override_falls_back_to_the_platform_default() {
        assert!(!candidate_paths(Some("")).is_empty());
    }

    /// Google's prebuilts are named `liblitert-lm.*`, not `libLiteRtLm.*`. A
    /// wrong default here fails with "not found" on a machine that has the
    /// runtime installed.
    #[test]
    fn the_default_library_name_matches_what_google_ships() {
        let names: Vec<String> = candidate_paths(None)
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert!(
            names.iter().any(|n| n.contains("litert-lm")),
            "expected the shipped library name, got {names:?}"
        );
    }

    /// A NUL in a caller's string would truncate the conversation silently at
    /// the C boundary.
    #[test]
    fn a_string_with_an_embedded_nul_is_refused() {
        let err = c_string("hello\0world").expect_err("a NUL cannot cross into C");
        assert!(matches!(err, ExecError::InvalidArg(_)));
    }

    #[test]
    fn a_null_c_string_reads_back_as_nothing_rather_than_panicking() {
        // SAFETY: null is the documented "no value" case.
        assert_eq!(unsafe { read_c_string(std::ptr::null()) }, None);
    }
}
