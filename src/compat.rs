//! The module tree pio-app imports against, re-exported under its old names.
//!
//! pio-app grew up on the in-tree copy of this crate and addresses it as
//! `crate::gen2::controller::ControllerCmd`, `crate::gen2::engine::Settings`,
//! `crate::gen2::backend::common::grammar::GrammarSpec`, and so on — 445
//! `use` leaves and 283 inline paths across 132 files (the inventory is
//! `docs/plans/research/05a-pio-app-import-map.md`). The public API chose a
//! different shape (api_spec.md §25: the root, [`advanced`](crate::advanced),
//! [`legacy`](crate::legacy), [`agent`](crate::agent)), and the modules those
//! paths name are crate-private now.
//!
//! This module is the bridge: every one of those paths resolves here, at the
//! same relative position, so the host's shim is one line —
//!
//! ```ignore
//! pub mod gen2 { pub use gen2::compat::*; }
//! ```
//!
//! or, in a single file, `use gen2::compat as gen2;`. It is `#[doc(hidden)]`
//! because none of this is the supported surface: it is the list of what
//! pio-app still reaches for, and it shrinks as the host moves to the root API.
//! Nothing here is new — each line names an item that already exists, and a
//! removal upstream fails `tests/compat_pio_paths.rs` before it reaches the
//! host.
//!
//! # What cannot be provided
//!
//! These changed meaning or left the crate; the host carries the change.
//!
//! | Old path | Now | Host change |
//! | --- | --- | --- |
//! | `SystemTask::{Answer, Triples, Stance, EntityExtract, TopicLabel, QueryUnderstand, Contradiction, QueryRewrite, ContextualPrefix}` | [`SystemTask::Custom`](crate::advanced::controller::SystemTask::Custom) with a label | [`system_task`] has a constructor per old variant and [`system_task::gen_spec`] with the old per-task tuning; call `system_infer_with(task, .., gen_spec(&task))` where `system_infer`/`system_prompt` used to pick the spec. Session ids lose the `sabra-` prefix. `SystemTask` is no longer `Copy`. |
//! | `ControllerEvent` | gained `Accepted { .. }`, emitted before the first token of a `StartChat`/`ContinueChat` | every event loop tolerates a first non-token event; one exhaustive `match` gains an arm |
//! | `ControllerCmd::LoadModel` | gained `api_model: Option<String>`; `resp` carries `Result<LoadOutcome, String>` | constructors add the field; `Ok(())` matches become `Ok(_)` |
//! | `ControllerCmd::ContinueChat` | gained `transcript`, `thinking`, `tools` | constructors add the fields (the transcript is what a rebuilt runtime is fed) |
//! | `InferenceHandle::{Remote, Flock, RegisteredFlockGateway}`, `liveness()`, `compute_provenance()` | one `Remote(Arc<dyn RemoteDispatch>)` arm; see [`RemoteDispatch`](crate::advanced::controller::RemoteDispatch) | host-typed enum over its three handles, each `impl RemoteDispatch` (receipt 05 §3) |
//! | `controller::project_streaming_inference` | deleted with the flock feature | moves host-side verbatim (its one caller is a host test) |
//! | `backend::mlxcel::{MlxcelEngine, ProfileMode, ProfileRun, worker::METALLIB_ENV}` | `gen2-mlxcel` companion crate: `MlxcelEngine`, `plugin()`; no profiler; `METALLIB_ENV` is private | depend on `gen2-mlxcel` directly; the two profiler captests need a replacement |
//! | `backend::onnx`, `backend::candle` | removed (S1.2) | none in the inventory |
//! | `system_infer*` return type | `ExecError`, not `PioError` | `From<ExecError> for PioError` exists; explicit `map_err` sites change |
//! | `zoo::tests::*` | crate-private unit tests | doc links only |
//!
//! # Feature dependence
//!
//! `backend::llama` needs `backend-llamacpp` and `backend::external_api`
//! needs `backend-external-api`; everything else is unconditional. The
//! per-backend modules exist so that what the facade's `Engine`, `Session`,
//! `TokenPuller` and `ModelBundle` wrap can be named (`unnameable_types`);
//! outside the in-tree copy itself pio-app names only `backend::llama`.

#![allow(deprecated)]
// A compat surface re-exports what exists; some of it is reachable only
// through here, so an alias the lint cannot see through is expected.
#![allow(unnameable_types)]

// ── The root of the old tree: pio-app's `gen2/mod.rs` re-export list ───────
// `Engine` here is the backend facade, which is what `crate::gen2::Engine`
// meant in the host (`detect_backend_for_path`, `available_backends`). The
// root's `Engine` was the deprecated API facade, now `legacy::Engine`.
pub use crate::backend::Engine;
pub use crate::engine::{
    EmbedLoadRequest, ExecError, ExecutionStats, LoadRequest, Settings, read_gguf_architecture,
    validate_model_architecture, validate_model_file,
};
pub use crate::residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};
pub use crate::residency_policy::{
    ContextBudget, ResidencyPolicy, default_context_budget_for_tier, effective_context_budget,
    estimate_resident_mb_for_path,
};
pub use crate::residency_stats::ResidencyStats;
pub use crate::types::message::{Message, MessageBody, MessageChunk, MessageContent};

/// `gen2::controller` as pio-app addresses it.
pub mod controller {
    pub use crate::controller::{
        ActiveChatSnapshot, CompletionReason, ControllerCmd, ControllerConfig, ControllerEvent,
        ControllerHandle, ControllerMetricsSnapshot, ControllerObservabilitySnapshot,
        ControllerPolicySnapshot, ControllerRuntimeSnapshot, ControllerState,
        EVENT_CHANNEL_CAPACITY, FailureReason, InferenceHandle, Placement, RemoteDispatch,
        RuntimeLifecycleSnapshot, SystemTask, WorkloadKind, start_controller,
        start_controller_with_config, start_controller_with_limit,
    };

    /// The controller's configuration, where the per-task generation
    /// defaults live (`ControllerConfig::system_task_spec`).
    pub mod config {
        pub use crate::controller::ControllerConfig;
    }
}

/// `gen2::engine` as pio-app addresses it.
pub mod engine {
    pub use crate::backend::Engine;
    pub use crate::engine::{
        Capabilities, ChatTemplateSpec, CtxParamsInput, Degraded, EmbedLoadRequest, ExecError,
        ExecutionStats, LoadOutcome, LoadRequest, MmSettings, ModelParamsInput, PromptSettings,
        SamplingSettings, Settings, StoppingSettings, SystemSettings, read_gguf_architecture,
        read_gguf_file_type, validate_model_architecture, validate_model_file,
    };

    /// Engine hooks: the bus and the events it carries.
    pub mod telemetry {
        pub use crate::engine::{HookBus, HookEvent, HookListener};
    }
}

/// `gen2::generation` as pio-app addresses it.
pub mod generation {
    pub use crate::backend::TokenPuller;
    pub use crate::generation::{
        CacheState, ChannelMarkers, GenSpec, MediaBoundary, ReplyParts, ReplyShape,
        ReplyStateMachine, StreamEmission, TelemetryAggregator, TelemetrySnapshot, Termination,
        ThinkingMode, Token, TokenEvent, ToolCall, TurnTelemetry, global_aggregator,
        ttft_bucket_upper_bounds_us,
    };

    /// The token event vocabulary.
    pub mod events {
        pub use crate::generation::{MediaBoundary, Token, TokenEvent, ToolCall};
    }
    /// Channel markers and the reply state machine that splits thinking
    /// from the answer.
    pub mod reply_parts {
        pub use crate::generation::{
            ChannelMarkers, ReplyParts, ReplyStateMachine, StreamEmission,
        };
    }
    /// The generation spec.
    pub mod spec {
        pub use crate::generation::GenSpec;
    }
    /// Per-turn telemetry and the process-wide aggregator.
    pub mod telemetry {
        pub use crate::generation::{
            CacheState, ReplyShape, TelemetryAggregator, TelemetrySnapshot, Termination,
            TurnTelemetry, global_aggregator, ttft_bucket_upper_bounds_us,
        };
    }
    /// The reasoning policy.
    pub mod thinking {
        pub use crate::generation::ThinkingMode;
    }
}

/// `gen2::backend` as pio-app addresses it: the facade, the traits, the
/// shared helpers, and each backend under its feature.
pub mod backend {
    pub use crate::backend::{
        Backend, BackendCaps, BackendSession, Embeddings, Engine, KvSnapshotTrait, LatencyTier,
        LocalBackend, ModelBundle, Multimodal, RemoteBackend, Session, SessionHealth, SessionId,
        SessionTokenizer, TokenPuller, TokenPullerDyn,
    };

    /// Backend capability flags and latency tiers.
    pub mod caps {
        pub use crate::backend::caps::{BackendCaps, LatencyTier};
    }
    /// The routing facade over every compiled backend.
    pub mod facade {
        pub use crate::backend::{Engine, ModelBundle, Session, SessionId, TokenPuller};
    }
    /// Session health as the controller sees it.
    pub mod health {
        pub use crate::backend::SessionHealth;
    }
    /// The backend traits.
    pub mod traits {
        pub use crate::backend::traits::{
            Backend, BackendSession, Embeddings, KvSnapshot, LocalBackend, Multimodal,
            RemoteBackend, SessionTokenizer, TokenPullerDyn,
        };
    }

    /// Helpers shared by every backend: grammar, tokenizer, chat templates,
    /// sampling, stop matching, tool-call detection.
    pub mod common {
        pub use crate::backend::common::{
            compute_hf_model_meta, load_chat_template, parse_hf_model_metadata,
        };

        /// Jinja chat templates (crate-private in the in-tree copy too; pio-app
        /// reached it only because the copy lived inside pio-core).
        pub mod chat_template {
            pub use crate::backend::common::chat_template::ChatTemplate;
        }
        /// Grammar-constrained decoding.
        pub mod grammar {
            pub use crate::backend::common::grammar::{GrammarMatcher, GrammarSpec};
        }
        /// Output filtering of channel scaffolds.
        pub mod output_filter {
            pub use crate::backend::common::output_filter::OutputFilter;
        }
        /// The shared sampler and its DRY / XTC parameters.
        pub mod sampler {
            pub use crate::backend::common::sampler::{DryParams, Sampler, XtcParams};
        }
        /// Speculative decoding modes.
        pub mod speculative {
            #[cfg(feature = "backend-mlx")]
            pub use crate::backend::common::speculative::DraftContext;
            pub use crate::backend::common::speculative::{SpeculativeMode, SpeculativePredictor};
        }
        /// Stop-sequence matching.
        pub mod stop_matcher {
            pub use crate::backend::common::stop_matcher::{StopMatcher, StopPattern, StopState};
        }
        /// The Hugging Face tokenizer wrapper.
        pub mod tokenizer {
            pub use crate::backend::common::tokenizer::HfTokenizer;
        }
        /// Tool-call tallying across a stream.
        pub mod tool_calls {
            pub use crate::backend::common::tool_calls::{Protocol, ToolCallTally};
        }
    }

    /// The llama.cpp backend (`backend-llamacpp`).
    #[cfg(feature = "backend-llamacpp")]
    pub mod llama {
        pub use crate::backend::llama::{Engine, ModelBundle, Session, TokenPuller};

        /// GGUF embedding models.
        pub mod embedder {
            pub use crate::backend::llama::embedder::{EmbedderKind, LlamaEmbedder};
        }
        /// llama.cpp model/context parameters.
        pub mod llama_config {
            pub use crate::backend::llama::llama_config::ModelConfig;
        }
    }

    // The remaining backends are here so that what the facade's `Engine`,
    // `Session`, `TokenPuller` and `ModelBundle` wrap can be named
    // (`unnameable_types`), not because pio-app imports them: outside the
    // in-tree copy itself it names only `backend::llama`.

    /// The OpenAI-compatible HTTP backend (`backend-external-api`).
    #[cfg(feature = "backend-external-api")]
    pub mod external_api {
        pub use crate::backend::external_api::{Engine, RemotePuller, Session};

        /// Anthropic Messages API streaming.
        pub mod anthropic_puller {
            pub use crate::backend::external_api::anthropic_puller::AnthropicPuller;
        }
        /// OpenAI-compatible SSE streaming.
        pub mod puller {
            pub use crate::backend::external_api::puller::TokenPuller;
        }
    }

    /// The MLX backend (`backend-mlx`). Unverified here: that lane does not
    /// compile on `main` today (`mlx/session.rs` predates `Message.tool_call_id`).
    #[cfg(feature = "backend-mlx")]
    pub mod mlx {
        pub use crate::backend::mlx::{Engine, ModelBundle, Session, TokenPuller};
    }

    /// The LiteRT-LM backend (`backend-litertlm`).
    #[cfg(feature = "backend-litertlm")]
    pub mod litertlm {
        pub use crate::backend::litertlm::LiteRtLmEngine;
    }

    /// The mistral.rs backend (`backend-mistralrs`).
    #[cfg(feature = "backend-mistralrs")]
    pub mod mistralrs {
        pub use crate::backend::mistralrs::MistralRsEngine;
    }
}

/// `gen2::session_rt` as pio-app addresses it.
pub mod session_rt {
    pub use crate::session_rt::{
        ColdStart, CompactResult, CompactionStrategy, PromptContext, Session, SessionId,
        SessionSpec, TruncationOutcome, WarmStart, build_prompt_context, compact_algorithmic,
        generation_reserve, merge_prompts,
    };

    /// Algorithmic compaction of a transcript.
    pub mod compaction {
        pub use crate::session_rt::compaction::{
            CompactResult, CompactionStrategy, compact_algorithmic,
        };
    }
    /// Image validation and detection on messages (`messages_have_images`
    /// was crate-private in the in-tree copy too).
    pub mod media_util {
        pub use crate::session_rt::media_util::{
            MAX_DECODE_ALLOC_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, check_dimensions,
            decode_limits, messages_have_images, validate_image_path,
        };
    }
    /// Prompt assembly and the generation reserve.
    pub mod prompt {
        pub use crate::session_rt::prompt::{
            PromptContext, build_meta_prompt, build_prompt_context, generation_reserve,
            merge_prompts,
        };
    }
    /// The session spec.
    pub mod spec {
        pub use crate::session_rt::SessionSpec;
    }
    /// Warm- and cold-start truncation.
    pub mod truncate {
        pub use crate::session_rt::truncate::{ColdStart, TruncationOutcome, WarmStart};
    }
}

/// `gen2::zoo` as pio-app addresses it.
pub mod zoo {
    pub use crate::zoo::{
        FamilyDefaults, ModelFamily, ModelZoo, ModelZooEntry, PlatformBundle, QuantTier,
        TemplateKind, ThinkingDefault, ZooError, auto_quant_for_ram, current_platform_id,
        detect_ram_mb, select_for_device,
    };
}

/// `gen2::bundle` as pio-app addresses it.
pub mod bundle {
    pub use crate::bundle::{ModelBundle, ModelMeta};

    /// GGUF header parsing and the fit/RAM estimates built on it.
    pub mod gguf {
        pub use crate::bundle::gguf::{
            GgufMetadata, backfill_metadata, build_model_metadata, detect_format_from_path,
            estimate_parameter_count, estimate_ram_bytes, file_type_bits_per_weight,
            file_type_to_quantization_label, fit_context, kv_bytes_per_token, parse_gguf_metadata,
            trim_optional,
        };
    }
    /// The model metadata record.
    pub mod meta {
        pub use crate::bundle::ModelMeta;
    }
}

/// `gen2::kv` as pio-app addresses it.
pub mod kv {
    pub use crate::kv::{KvHeader, KvLoadReport, KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot};

    /// The on-disk keep-warm store.
    pub mod store {
        pub use crate::kv::store::{
            candidate_for_chat, enforce_budget, keepwarm_enabled, kv_dir, path_for_chat,
            remove_stale,
        };
    }
    /// The KV snapshot types.
    pub mod types {
        pub use crate::kv::{KvHeader, KvLoadReport, KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot};
    }
}

/// `gen2::executor` as pio-app addresses it.
pub mod executor {
    pub use crate::executor::{
        BoxedOperation, ConcurrencyGuard, FnOperation, Operation, StreamingToolExecutor,
    };
}

/// `gen2::residency` as pio-app addresses it.
pub mod residency {
    pub use crate::residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};
}
/// `gen2::residency_policy` as pio-app addresses it.
pub mod residency_policy {
    pub use crate::residency_policy::{
        ContextBudget, ResidencyPolicy, default_context_budget_for_tier, effective_context_budget,
        estimate_resident_mb_for_path, estimate_resident_mb_for_path_offloaded,
    };
}
/// `gen2::residency_stats` as pio-app addresses it.
pub mod residency_stats {
    pub use crate::residency_stats::ResidencyStats;
}

/// `gen2::router` as pio-app addresses it — inference placement across
/// advertised peers.
pub mod router {
    pub use crate::router::{
        LocalCapability, PeerAdvertisement, RouteDecision, RoutePriority, RouteRequest, route,
    };
}

/// `gen2::media` as pio-app addresses it.
pub mod media {
    pub use crate::media::{Attachment, EncodedMedia};
}

/// `gen2::hardware` as pio-app addresses it.
pub mod hardware {
    pub use crate::hardware::{
        GpuBackend, HardwareProfile, ModelRecommendation, auto_tune_ctx, auto_tune_ctx_with_dims,
        free_disk_bytes, is_ios_simulator, parse_nvidia_smi_total_mib, platform_defaults,
        recommend_model,
    };
}

/// `gen2::memory` as pio-app addresses it — the memory governor.
pub mod memory {
    pub use crate::memory::{
        MachineMemoryTier, MemoryBudgets, MemoryGovernor, MemoryPolicyInput, MemoryPressureLevel,
        MemorySnapshot,
    };
}

/// `gen2::utilities` as pio-app addresses it.
pub mod utilities {
    pub use crate::utilities::{LoadedUtility, RerankResult, UtilityStatus};
}

/// `gen2::types` as pio-app addresses it — the wire types that came across
/// with the extraction.
pub mod types {
    pub use crate::types::{ExecutionStats, Model, ModelConfig, ModelMetadata, Persona};

    /// Execution statistics a generation reports.
    pub mod execution_stats {
        pub use crate::types::ExecutionStats;
    }
    /// The role-based wire message and its parts.
    pub mod message {
        pub use crate::types::message::{
            FunctionDefinition, Message, MessageBody, MessageChunk, MessageContent, ToolCall,
            ToolSpec, Url,
        };
    }
    /// The model record a session is started from.
    pub mod model {
        pub use crate::types::model::{Model, ModelConfig, ModelMetadata};
    }
    /// The persona pinned into a system prompt.
    pub mod persona {
        pub use crate::types::Persona;
    }
}

/// The `SystemTask` variants pio-app used and this crate no longer has,
/// as `SystemTask::Custom` labels with their old generation tuning.
///
/// The in-tree copy had thirteen variants, each with a `GenSpec` in
/// `ControllerConfig::system_task_spec`; the crate keeps `Title`,
/// `Suggestions`, `Compact`, `Summary` and folds the rest into
/// [`SystemTask::Custom`](crate::advanced::controller::SystemTask::Custom), whose spec is one plain default (512 tokens at
/// 0.3). `InferenceHandle::system_infer` and `system_prompt` look the spec
/// up by variant, so a `Custom` task through them gets the plain default,
/// not the old tuning. To keep the old behaviour, pass the spec explicitly:
///
/// ```
/// use gen2::compat::system_task;
///
/// let task = system_task::answer();
/// let spec = system_task::gen_spec(&task);
/// assert_eq!(spec.max_tokens, Some(1024));
/// assert_eq!(spec.temperature, Some(0.3));
/// // handle.system_infer_with(task, chat_id, messages, spec).await
/// ```
///
/// Two things this cannot restore: the old session ids were
/// `sabra-<suffix>-<uuid>` and are now `<label>-<uuid>`, and `SystemTask`
/// is no longer `Copy` (a `Custom` label is a `Cow<'static, str>`), so a
/// value used twice is cloned.
pub mod system_task {
    use crate::controller::SystemTask;
    use crate::generation::GenSpec;

    /// Label of the old `Answer` variant.
    pub const ANSWER: &str = "answer";
    /// Label of the old `Triples` variant.
    pub const TRIPLES: &str = "triples";
    /// Label of the old `Stance` variant.
    pub const STANCE: &str = "stance";
    /// Label of the old `EntityExtract` variant.
    pub const ENTITY_EXTRACT: &str = "entities";
    /// Label of the old `TopicLabel` variant.
    pub const TOPIC_LABEL: &str = "topic";
    /// Label of the old `QueryUnderstand` variant.
    pub const QUERY_UNDERSTAND: &str = "query";
    /// Label of the old `Contradiction` variant.
    pub const CONTRADICTION: &str = "contradiction";
    /// Label of the old `QueryRewrite` variant.
    pub const QUERY_REWRITE: &str = "rewrite";
    /// Label of the old `ContextualPrefix` variant.
    pub const CONTEXTUAL_PREFIX: &str = "ctxprefix";

    /// Grounded answer synthesis from evidence chunks (was `SystemTask::Answer`).
    pub fn answer() -> SystemTask {
        SystemTask::custom(ANSWER)
    }
    /// Triple (subject, predicate, object) extraction (was `SystemTask::Triples`).
    pub fn triples() -> SystemTask {
        SystemTask::custom(TRIPLES)
    }
    /// Entity stance extraction (was `SystemTask::Stance`).
    pub fn stance() -> SystemTask {
        SystemTask::custom(STANCE)
    }
    /// Named entity recognition (was `SystemTask::EntityExtract`).
    pub fn entity_extract() -> SystemTask {
        SystemTask::custom(ENTITY_EXTRACT)
    }
    /// Topic cluster labelling (was `SystemTask::TopicLabel`).
    pub fn topic_label() -> SystemTask {
        SystemTask::custom(TOPIC_LABEL)
    }
    /// Query understanding (was `SystemTask::QueryUnderstand`).
    pub fn query_understand() -> SystemTask {
        SystemTask::custom(QUERY_UNDERSTAND)
    }
    /// Contradiction detection (was `SystemTask::Contradiction`).
    pub fn contradiction() -> SystemTask {
        SystemTask::custom(CONTRADICTION)
    }
    /// Conversational query rewriting (was `SystemTask::QueryRewrite`).
    pub fn query_rewrite() -> SystemTask {
        SystemTask::custom(QUERY_REWRITE)
    }
    /// Contextual-retrieval prefix generation (was `SystemTask::ContextualPrefix`).
    pub fn contextual_prefix() -> SystemTask {
        SystemTask::custom(CONTEXTUAL_PREFIX)
    }

    fn spec(max_tokens: usize, temperature: f32) -> GenSpec {
        GenSpec {
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            ..Default::default()
        }
    }

    /// The generation tuning the in-tree copy gave each task, by variant or
    /// by label for the ones that are now `Custom`.
    ///
    /// Identical to `ControllerConfig::system_task_spec` for the four kept
    /// variants; for an unknown label it returns what that function returns
    /// for any `Custom` task, so this is a superset of the crate's own table.
    pub fn gen_spec(task: &SystemTask) -> GenSpec {
        match task {
            SystemTask::Custom(label) => match label.as_ref() {
                ANSWER => spec(1024, 0.3),
                TRIPLES => spec(1024, 0.1),
                STANCE | ENTITY_EXTRACT => spec(512, 0.1),
                TOPIC_LABEL => spec(100, 0.3),
                QUERY_UNDERSTAND => spec(256, 0.1),
                CONTRADICTION => spec(512, 0.2),
                QUERY_REWRITE => spec(80, 0.1),
                CONTEXTUAL_PREFIX => spec(120, 0.1),
                _ => task.default_gen_spec(),
            },
            other => other.default_gen_spec(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn every_old_variant_keeps_its_tuning() {
            let table: [(SystemTask, usize, f32); 9] = [
                (answer(), 1024, 0.3),
                (triples(), 1024, 0.1),
                (stance(), 512, 0.1),
                (entity_extract(), 512, 0.1),
                (topic_label(), 100, 0.3),
                (query_understand(), 256, 0.1),
                (contradiction(), 512, 0.2),
                (query_rewrite(), 80, 0.1),
                (contextual_prefix(), 120, 0.1),
            ];
            for (task, max_tokens, temperature) in table {
                let spec = gen_spec(&task);
                assert_eq!(spec.max_tokens, Some(max_tokens), "{task:?}");
                assert_eq!(spec.temperature, Some(temperature), "{task:?}");
            }
        }

        #[test]
        fn kept_variants_and_unknown_labels_defer_to_the_crate() {
            for task in [
                SystemTask::Title,
                SystemTask::Suggestions,
                SystemTask::Compact,
                SystemTask::Summary,
                SystemTask::custom("something-else"),
            ] {
                let (ours, theirs) = (gen_spec(&task), task.default_gen_spec());
                assert_eq!(ours.max_tokens, theirs.max_tokens, "{task:?}");
                assert_eq!(ours.temperature, theirs.temperature, "{task:?}");
            }
        }

        #[test]
        fn labels_namespace_session_ids() {
            assert!(answer().session_id().starts_with("answer-"));
            assert!(contextual_prefix().session_id().starts_with("ctxprefix-"));
            assert_eq!(entity_extract().label(), "entities");
        }
    }
}
