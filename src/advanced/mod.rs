//! Below the happy path.
//!
//! Everything a normal consumer needs is at the crate root (api_spec.md
//! §25). This module is for the consumer who needs to reach past it — local
//! tuning a remote model would not understand, what the machine is and what
//! is resident on it, the wire types the backends speak, the controller the
//! facade is built on, and a seam to bring a backend of their own.
//!
//! - [`generation`] — the full [`GenSpec`](generation::GenSpec) and the
//!   engine-wide [`Settings`](generation::Settings) behind
//!   [`GenerationOptions`](crate::GenerationOptions); raw
//!   [`GrammarSpec`](generation::GrammarSpec); speculative decoding.
//! - [`runtime`] — the residency controls of api_spec.md §4.5: what a
//!   [`Runtime`](crate::Runtime) has resident, the machine's
//!   [`HardwareProfile`](runtime::HardwareProfile), the memory governor, and
//!   [`RuntimeBuilder`](runtime::RuntimeBuilder).
//! - [`fit`] — a GGUF header and whether it runs here, before loading.
//! - [`wire`] — the parts of a [`Message`](crate::Message) and the model
//!   record the backends render.
//! - [`controller`] — commands, events, and handles, including
//!   [`RemoteDispatch`](controller::RemoteDispatch) for a controller in
//!   another process.
//! - [`utilities`] — which auxiliary runtimes are loaded.
//! - [`plugin`] — implement [`LocalBackend`](plugin::LocalBackend) outside
//!   the crate, wrap it in a [`BackendPlugin`], and register it with
//!   [`RuntimeBuilder::backend`](runtime::RuntimeBuilder::backend).

pub mod plugin;

/// Runtime-level residency and hardware, below the happy path (api_spec.md
/// §4.5). The methods are on [`Runtime`](crate::Runtime); the types they
/// return live here so the root stays boring.
pub mod runtime {
    pub use crate::api::{ModelResidency, ResidencySnapshot, RuntimeBuilder, RuntimeStats};
    pub use crate::hardware::{GpuBackend, HardwareProfile};
    /// Memory governance the residency decisions are made against.
    pub use crate::memory::{
        MachineMemoryTier, MemoryBudgets, MemoryGovernor, MemoryPolicyInput, MemoryPressureLevel,
        MemorySnapshot,
    };
    /// What is resident and the policy deciding what may join it, as the
    /// controller's observability snapshots report them.
    pub use crate::residency::{ResidencyInventory, ResidentRuntime, RuntimeKind};
    pub use crate::residency_policy::ResidencyPolicy;
    pub use crate::residency_stats::ResidencyStats;
}

/// Will a model run here, and at what context — from the file header alone.
///
/// [`ModelInfo::read`](fit::ModelInfo::read) parses a GGUF header without
/// loading weights; [`ModelInfo::fits`](fit::ModelInfo::fits) sizes it
/// against a [`HardwareProfile`](runtime::HardwareProfile). A load that
/// cannot fit fails with [`Error::fit`](crate::Error::fit) set.
pub mod fit {
    pub use crate::api::{FileModelInfo as ModelInfo, Fit, FitVerdict};
    pub use crate::types::model::ModelMetadata;
}

/// Local generation tuning: what a turn is asked to do, in full.
///
/// [`GenerationOptions`](crate::GenerationOptions) covers the common knobs;
/// these are the rest — the complete [`GenSpec`](generation::GenSpec), the
/// engine-wide [`Settings`](generation::Settings) a
/// [`RuntimeBuilder`](runtime::RuntimeBuilder) takes, raw
/// [`GrammarSpec`](generation::GrammarSpec)s, and speculative decoding.
pub mod generation {
    pub use crate::backend::caps::{BackendCaps, LatencyTier};
    pub use crate::backend::common::grammar::GrammarSpec;
    #[cfg(feature = "backend-mlx")]
    pub use crate::backend::common::speculative::DraftContext;
    pub use crate::backend::common::speculative::{SpeculativeMode, SpeculativePredictor};
    pub use crate::engine::{
        Capabilities, Degraded, LoadOutcome, MmSettings, PromptSettings, SamplingSettings,
        Settings, StoppingSettings, SystemSettings,
    };
    /// The reply scanner: how a model's raw token text is split into prose,
    /// reasoning and tool calls. The semantic [`Event`](crate::Event) stream
    /// is built on this; a consumer that renders its own channels, or fuzzes
    /// the scanner, needs it directly.
    pub use crate::generation::{ChannelMarkers, ReplyParts, ReplyStateMachine, StreamEmission};
    pub use crate::generation::{GenSpec, ThinkingMode};
}

/// The wire types: what a [`Message`](crate::Message) is made of, and the
/// model record a session is started from.
///
/// `Message` is the role-based shape every backend renders; api_spec.md §9's
/// `enum Message` is not this type yet, so its parts are reachable here.
pub mod wire {
    pub use crate::types::message::{
        FunctionDefinition, Message, MessageBody, MessageChunk, MessageContent, ToolCall, ToolSpec,
        Url,
    };
    pub use crate::types::model::{Model as ModelRecord, ModelConfig, ModelMetadata};
}

/// The controller the facade is built on: commands, events, handles, and
/// the seam for running it in another process
/// ([`RemoteDispatch`](controller::RemoteDispatch)).
///
/// Reached through [`legacy::Engine::controller`](crate::legacy::Engine)
/// today; nothing on the new surface hands one out.
pub mod controller {
    pub use crate::controller::{
        ActiveChatSnapshot, CompletionReason, ControllerCmd, ControllerConfig, ControllerEvent,
        ControllerHandle, ControllerMetricsSnapshot, ControllerObservabilitySnapshot,
        ControllerPolicySnapshot, ControllerRuntimeSnapshot, ControllerState,
        EVENT_CHANNEL_CAPACITY, FailureReason, InferenceHandle, Placement, RemoteDispatch,
        RuntimeLifecycleSnapshot, SystemTask, WorkloadKind, start_controller,
        start_controller_with_config, start_controller_with_limit,
    };
    /// The error every engine-internal failure is reported as.
    pub use crate::engine::ExecError;
    /// Structured payloads carried by [`ControllerEvent`].
    pub use crate::generation::{MediaBoundary, ToolCall};
    pub use crate::types::ExecutionStats;
}

/// Which auxiliary runtimes — embedder, reranker — are loaded.
pub mod utilities {
    pub use crate::utilities::{LoadedUtility, UtilityStatus};
}

pub use plugin::BackendPlugin;
