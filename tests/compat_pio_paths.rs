//! Compile-only proof that every path pio-app imports from its in-tree `gen2`
//! resolves through `gen2::compat` at the same relative position.
//!
//! One `use` per inventory path (`docs/plans/research/05a-pio-app-import-map.md`;
//! the comment is how many `use` leaves and inline mentions the host has), so
//! a removal upstream fails here before it reaches the host. The `inline`
//! module names the variants and associated functions the host reaches
//! inline, under the host's own `use gen2::compat as gen2;` shape.
//!
//! What is deliberately absent — the semantic changes the host carries — is
//! the table in `gen2::compat`'s module doc.

// A proof of resolution, not a program: every variant is matched with `{ .. }`
// so the line is right for any variant shape, and the checks take one
// parameter per type.
#![allow(
    unused_imports,
    dead_code,
    deprecated,
    clippy::too_many_arguments,
    clippy::unneeded_struct_pattern
)]

use gen2::compat::EmbedLoadRequest; // 1 use, 1 file
use gen2::compat::Engine; // 1 use, 3 inline, 2 files
use gen2::compat::ExecutionStats; // 3 use, 2 inline, 5 files
use gen2::compat::Message; // 28 use, 96 inline, 32 files
use gen2::compat::MessageBody; // 9 use, 5 files
use gen2::compat::MessageContent; // 4 use, 3 files
use gen2::compat::ResidencyInventory; // 2 use, 1 file
use gen2::compat::ResidentRuntime; // 2 use, 1 file
use gen2::compat::RuntimeKind; // 1 use, 1 file
use gen2::compat::Settings; // 2 use, 2 inline, 4 files
use gen2::compat::backend::Engine as _; // 9 use, 6 files
use gen2::compat::backend::TokenPuller; // 3 inline, 2 files
use gen2::compat::backend::common::grammar::GrammarSpec; // 15 use, 12 files
use gen2::compat::backend::common::parse_hf_model_metadata; // 3 inline, 3 files
use gen2::compat::backend::common::tokenizer::HfTokenizer; // 1 use, 1 file
use gen2::compat::backend::facade::SessionId; // 1 inline, 1 file
#[cfg(feature = "backend-llamacpp")]
use gen2::compat::backend::llama::embedder::EmbedderKind; // 2 use, 3 inline, 4 files
#[cfg(feature = "backend-llamacpp")]
use gen2::compat::backend::llama::embedder::LlamaEmbedder; // 1 use, 2 inline, 2 files
#[cfg(feature = "backend-llamacpp")]
use gen2::compat::backend::llama::llama_config::ModelConfig; // 1 use, 1 inline, 2 files
use gen2::compat::backend::traits::Backend; // 5 use, 1 file
use gen2::compat::backend::traits::SessionTokenizer; // 1 inline, 1 file
use gen2::compat::bundle::gguf::GgufMetadata; // 1 use, 1 file
use gen2::compat::bundle::gguf::build_model_metadata; // 1 use, 1 file
use gen2::compat::bundle::gguf::detect_format_from_path; // 1 use, 1 inline, 2 files
use gen2::compat::bundle::gguf::estimate_ram_bytes; // 1 inline, 1 file
use gen2::compat::bundle::gguf::fit_context; // 1 use, 1 file
use gen2::compat::bundle::gguf::kv_bytes_per_token; // 1 use, 1 file
use gen2::compat::bundle::gguf::parse_gguf_metadata; // 1 use, 1 inline, 2 files
use gen2::compat::bundle::gguf::trim_optional; // 1 use, 1 file
use gen2::compat::controller::ActiveChatSnapshot; // 1 use, 1 file
use gen2::compat::controller::CompletionReason; // 1 use, 1 file
use gen2::compat::controller::ControllerCmd; // 50 use, 13 inline, 50 files
use gen2::compat::controller::ControllerConfig; // 4 use, 1 file
use gen2::compat::controller::ControllerEvent; // 30 use, 1 inline, 29 files
use gen2::compat::controller::ControllerHandle; // 19 use, 13 inline, 24 files
use gen2::compat::controller::ControllerMetricsSnapshot; // 4 use, 4 files
use gen2::compat::controller::ControllerObservabilitySnapshot; // 4 use, 4 files
use gen2::compat::controller::ControllerRuntimeSnapshot; // 7 use, 2 inline, 9 files
use gen2::compat::controller::EVENT_CHANNEL_CAPACITY; // 5 use, 2 inline, 6 files
use gen2::compat::controller::InferenceHandle; // 22 use, 32 inline, 28 files
use gen2::compat::controller::RuntimeLifecycleSnapshot; // 1 use, 1 file
use gen2::compat::controller::SystemTask; // 26 use, 2 inline, 24 files
use gen2::compat::controller::WorkloadKind; // 1 use, 1 file
use gen2::compat::controller::start_controller; // 16 use, 3 inline, 17 files
use gen2::compat::controller::start_controller_with_config; // 4 use, 1 file
use gen2::compat::controller::start_controller_with_limit; // 1 use, 1 file
use gen2::compat::engine::Engine as _; // 1 use, 1 file
use gen2::compat::engine::ExecError; // 1 use, 13 inline, 2 files
use gen2::compat::engine::ExecutionStats as _; // 2 use, 2 files
use gen2::compat::engine::LoadRequest; // 15 use, 8 files
use gen2::compat::engine::SamplingSettings; // 1 use, 1 file
use gen2::compat::engine::Settings as _; // 21 use, 5 inline, 20 files
use gen2::compat::executor::BoxedOperation; // 1 use, 1 file
use gen2::compat::executor::ConcurrencyGuard; // 1 use, 1 file
use gen2::compat::executor::StreamingToolExecutor; // 1 use, 1 file
use gen2::compat::generation::CacheState; // 2 use, 2 files
use gen2::compat::generation::ChannelMarkers; // 6 use, 6 files
use gen2::compat::generation::GenSpec; // 35 use, 35 inline, 29 files
use gen2::compat::generation::MediaBoundary; // 3 use, 3 files
use gen2::compat::generation::ReplyShape; // 2 use, 2 files
use gen2::compat::generation::ReplyStateMachine; // 4 use, 2 inline, 5 files
use gen2::compat::generation::StreamEmission; // 3 use, 3 files
use gen2::compat::generation::TelemetrySnapshot; // 1 use, 1 file
use gen2::compat::generation::Termination; // 2 use, 2 files
use gen2::compat::generation::ThinkingMode; // 3 use, 9 inline, 9 files
use gen2::compat::generation::TokenEvent; // 12 use, 2 inline, 8 files
use gen2::compat::generation::ToolCall; // 1 use, 1 inline, 2 files
use gen2::compat::generation::TurnTelemetry; // 2 use, 2 files
use gen2::compat::generation::global_aggregator; // 1 use, 1 file
use gen2::compat::generation::ttft_bucket_upper_bounds_us; // 1 use, 1 file
use gen2::compat::kv::KvLoadSpec; // 1 use, 1 inline, 2 files
use gen2::compat::kv::KvSaveSpec; // 1 use, 1 inline, 2 files
use gen2::compat::read_gguf_architecture; // 1 inline, 1 file
use gen2::compat::session_rt::SessionSpec; // 13 use, 8 files
use gen2::compat::session_rt::prompt::generation_reserve; // 1 use, 1 file
use gen2::compat::validate_model_architecture; // 1 inline, 1 file
use gen2::compat::validate_model_file; // 2 inline, 1 file
use gen2::compat::zoo::ModelFamily; // 2 use, 2 files
use gen2::compat::zoo::ModelZoo; // 3 use, 3 files
use gen2::compat::zoo::ModelZooEntry; // 1 inline, 1 file
use gen2::compat::zoo::PlatformBundle; // 1 use, 1 file
use gen2::compat::zoo::current_platform_id; // 1 use, 1 inline, 2 files
use gen2::compat::zoo::detect_ram_mb; // 2 use, 2 files
use gen2::compat::zoo::select_for_device; // 1 inline, 1 file

// Modules the host names as paths (`gen2::bundle::gguf::…`, doc links to
// `gen2::router`).
use gen2::compat::bundle::gguf;
use gen2::compat::controller::config;
use gen2::compat::router;

/// The host's shim shape: `pub mod gen2 { pub use gen2::compat::*; }` or,
/// per file, `use gen2::compat as gen2;`. Variants and associated functions
/// the host names inline resolve under it.
mod inline {
    use gen2::compat as gen2;

    fn inline_paths(
        execerror: &gen2::engine::ExecError,
        controllercmd: &gen2::controller::ControllerCmd,
        thinkingmode: &gen2::generation::ThinkingMode,
        systemtask: &gen2::controller::SystemTask,
        inferencehandle: &gen2::controller::InferenceHandle,
        kvsavespec: &gen2::kv::KvSaveSpec,
        kvloadspec: &gen2::kv::KvLoadSpec,
        tokenevent: &gen2::generation::TokenEvent,
    ) {
        let _ = matches!(execerror, gen2::engine::ExecError::ModelNotLoaded { .. }); // 2
        let _ = matches!(execerror, gen2::engine::ExecError::EmbedderNotLoaded { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::InvalidModelFile { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::SettingsError { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::Io { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::ContextOverflow { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::TemplateError { .. }); // 1
        let _ = matches!(execerror, gen2::engine::ExecError::SessionPoisoned { .. }); // 1
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::IsMmprojLoaded { .. }
        ); // 2
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::IsChatLoaded { .. }
        ); // 1
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::IsModelLoaded { .. }
        ); // 2
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::IsEmbedderLoaded { .. }
        ); // 2
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::LoadModel { .. }
        ); // 1
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::LoadEmbedder { .. }
        ); // 1
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::StartChat { .. }
        ); // 1
        let _ = matches!(
            controllercmd,
            gen2::controller::ControllerCmd::GenerateEmbeddings { .. }
        ); // 3
        let _ = matches!(thinkingmode, gen2::generation::ThinkingMode::Off { .. }); // 3
        let _ = matches!(thinkingmode, gen2::generation::ThinkingMode::On { .. }); // 1
        let _ = matches!(thinkingmode, gen2::generation::ThinkingMode::Auto { .. }); // 1
        let _ = matches!(systemtask, gen2::controller::SystemTask::Title { .. }); // 1
        let _ = matches!(systemtask, gen2::controller::SystemTask::Compact { .. }); // 1
        let _ = matches!(
            inferencehandle,
            gen2::controller::InferenceHandle::Local { .. }
        ); // 2
        let _ = matches!(kvsavespec, gen2::kv::KvSaveSpec::ToPath { .. }); // 1
        let _ = matches!(kvloadspec, gen2::kv::KvLoadSpec::Strict { .. }); // 1
        let _ = matches!(tokenevent, gen2::generation::TokenEvent::Paused { .. }); // 1
        let _ = matches!(tokenevent, gen2::generation::TokenEvent::Stopped { .. }); // 1
        let _ = gen2::engine::Settings::default; // 3
        // gen2::Message::user — 32; impl-Trait arguments, called in `associated_fns` below
        // gen2::Message::system — 8; impl-Trait arguments, called in `associated_fns` below
        // gen2::Message::user_with_images — 2; impl-Trait arguments, called in `associated_fns` below
        // gen2::Message::assistant_structured — 5; impl-Trait arguments, called in `associated_fns` below
        let _ = gen2::generation::ThinkingMode::default; // 3
        let _ = gen2::Engine::available_backends; // 1
        let _ = gen2::Engine::detect_backend_for_path; // 2
        let _ = gen2::generation::GenSpec::default; // 30
        let _ = gen2::controller::ControllerHandle::new_for_test; // 9
        // gen2::controller::InferenceHandle::system_infer_with — 2; impl-Trait arguments, called in `associated_fns` below
    }
    #[cfg(feature = "backend-llamacpp")]
    fn inline_paths_llama(embedderkind: &gen2::backend::llama::embedder::EmbedderKind) {
        let _ = matches!(
            embedderkind,
            gen2::backend::llama::embedder::EmbedderKind::Qwen3 { .. }
        ); // 1
        // gen2::backend::llama::embedder::LlamaEmbedder::load_from_path_with_kind — 1; impl-Trait arguments, called in `associated_fns` below
        let _ = gen2::backend::llama::llama_config::ModelConfig::default; // 1
        let _ = gen2::backend::llama::embedder::EmbedderKind::as_str; // 1
    }

    /// Associated functions whose arguments are `impl Trait`, so the
    /// function item cannot be named without a call.
    fn associated_fns(handle: &gen2::controller::InferenceHandle) {
        let _ = gen2::Message::user("");
        let _ = gen2::Message::system("");
        let _ = gen2::Message::user_with_images("", Vec::new());
        let _ = gen2::Message::assistant_structured("", None);
        let _future = handle.system_infer_with(
            gen2::controller::SystemTask::Title,
            "",
            Vec::new(),
            gen2::generation::GenSpec::default(),
        );
        // `LlamaEmbedder::load_from_path_with_kind` takes an `Arc<LlamaBackend>`
        // from llama-cpp-2, which this test does not depend on; the host's own
        // compile covers it.
    }
}

#[test]
fn every_inventory_path_resolves() {
    // The proof is that this file compiled.
}
