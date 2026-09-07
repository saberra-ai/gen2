//! The previous facade, kept for one release and deprecated.
//!
//! `Engine`, `Chat`, `Inference` and the types around them still work
//! exactly as before; they are reached from here instead of the root, and
//! every use warns. The new surface does the same work with fewer concepts
//! (api_spec.md §27):
//!
//! | Here | Now |
//! | --- | --- |
//! | `Engine::load(path)` | [`gen2::load`](crate::load), [`Runtime::load`](crate::Runtime::load) |
//! | `Engine::infer(text)` | [`Model::generate`](crate::Model::generate) |
//! | `Inference` | [`Generation`](crate::model::Generation) |
//! | `Engine::chat(&mut session)` | [`Model::turn`](crate::Model::turn) |
//! | `Chat` controls | [`Turn`](crate::Turn), [`GenerationOptions`](crate::GenerationOptions) |
//! | `Chat::on_tool` loop | your loop over [`Response::tool_calls`](crate::Response::tool_calls) and [`Session::push_tool_result`](crate::Session::push_tool_result) |
//! | `Engine::agent` | [`gen2::agent`](crate::agent) (feature `agent`) |
//! | `Completion`, `Finish` | [`Response`](crate::Response), [`FinishReason`](crate::output::FinishReason) |
//! | `TokenStream` / `Event::Token` | [`EventStream`](crate::EventStream) of semantic [`Event`](crate::Event)s |
//! | `OwnedChat::spawn` / `Update` | [`Turn::stream`](crate::Turn::stream) with a [`Canceller`](crate::event::Canceller); `run_async` under `tokio` |
//! | `Engine::embed` / `rerank` | [`Runtime::load_embedder`](crate::Runtime::load_embedder), [`Runtime::load_reranker`](crate::Runtime::load_reranker) |
//! | `ModelInfo::read` (fit) | [`advanced::fit::ModelInfo`](crate::advanced::fit::ModelInfo) |
//! | `Engine::builder().backend(plugin)` | [`Runtime::builder().backend(plugin)`](crate::advanced::runtime::RuntimeBuilder::backend) |
//!
//! Each name is a deprecated alias of the real type, so `#[allow(deprecated)]`
//! on a `use` is the whole migration cost for code that is not ready to move.
//! The new facade is built over these types internally, which is why they
//! cannot simply be deleted yet.

#![allow(deprecated)]

/// A running inference engine. Now a [`Runtime`](crate::Runtime) owns the
/// backends and a [`Model`](crate::Model) is what you ask.
#[deprecated(
    since = "0.1.0",
    note = "use gen2::load / Runtime::load and Model; see api_spec.md §27"
)]
pub type Engine = crate::api::Engine;
/// Builds an [`Engine`]. Now [`Runtime::builder`](crate::Runtime::builder).
#[deprecated(since = "0.1.0", note = "use Runtime::builder(); see api_spec.md §27")]
pub type EngineBuilder = crate::api::EngineBuilder;
/// One turn against a session. Now [`Turn`](crate::Turn) from
/// [`Model::turn`](crate::Model::turn).
#[deprecated(
    since = "0.1.0",
    note = "use Model::turn(&mut session); see api_spec.md §27"
)]
pub type Chat<'a> = crate::api::Chat<'a>;
/// A turn on an owned session, run on a worker thread. Now
/// [`Turn::stream`](crate::Turn::stream) or `run_async` under `tokio`.
#[deprecated(
    since = "0.1.0",
    note = "use Turn::stream / Turn::run_async; see api_spec.md §27"
)]
pub type OwnedChat = crate::api::OwnedChat;
/// A one-shot prompt. Now [`Model::generate`](crate::Model::generate).
#[deprecated(since = "0.1.0", note = "use Model::generate; see api_spec.md §27")]
pub type Inference<'a> = crate::api::Inference<'a>;
/// Pick one label. Now [`Generation::structured`](crate::model::Generation::structured)
/// over an enum.
#[deprecated(
    since = "0.1.0",
    note = "use Model::generate(..).structured::<Enum>(); see api_spec.md §27"
)]
pub type Classify<'a> = crate::api::Classify<'a>;
/// Extract a typed value. Now [`Generation::structured`](crate::model::Generation::structured).
#[deprecated(
    since = "0.1.0",
    note = "use Model::generate(..).structured::<T>(); see api_spec.md §27"
)]
pub type Extract<'a, T> = crate::api::Extract<'a, T>;
/// What a finished generation produced. Now [`Response`](crate::Response).
#[deprecated(since = "0.1.0", note = "use Response; see api_spec.md §27")]
pub type Completion = crate::api::Completion;
/// Token-level events. Now [`EventStream`](crate::EventStream).
#[deprecated(since = "0.1.0", note = "use EventStream; see api_spec.md §27")]
pub type TokenStream = crate::api::TokenStream;
/// Text fragments as an iterator. Now [`EventStream`](crate::EventStream)
/// filtered to `Event::TextDelta`.
#[deprecated(since = "0.1.0", note = "use EventStream; see api_spec.md §27")]
pub type Tokens = crate::api::Tokens;
/// A token-centric stream event. Now the semantic [`Event`](crate::Event).
#[deprecated(since = "0.1.0", note = "use gen2::Event; see api_spec.md §27")]
pub type Event = crate::api::TokenEvent;
/// Why a generation ended. Now [`FinishReason`](crate::output::FinishReason).
#[deprecated(
    since = "0.1.0",
    note = "use output::FinishReason; see api_spec.md §27"
)]
pub type Finish = crate::api::Finish;
/// Which agent budget ran out. Lives on in [`gen2::agent`](crate::agent).
#[deprecated(since = "0.1.0", note = "use gen2::agent::Budget")]
pub type Budget = crate::api::Budget;
/// Why an agent gave up. Lives on in [`gen2::agent`](crate::agent).
#[deprecated(since = "0.1.0", note = "use gen2::agent::Struggle")]
pub type Struggle = crate::api::Struggle;
/// A spawned turn's handle. Now [`EventStream`](crate::EventStream) and its
/// [`Canceller`](crate::event::Canceller).
#[deprecated(
    since = "0.1.0",
    note = "use Turn::stream and EventStream::canceller; see api_spec.md §27"
)]
pub type Turn = crate::api::ChatTurn;
/// Stops a spawned turn. Now [`event::Canceller`](crate::event::Canceller).
#[deprecated(since = "0.1.0", note = "use event::Canceller; see api_spec.md §27")]
pub type Canceller = crate::api::ChatCanceller;
/// Updates from a spawned turn. Now [`Event`](crate::Event); an agent run's
/// updates are [`gen2::agent::Update`](crate::agent).
#[deprecated(
    since = "0.1.0",
    note = "use gen2::Event, or gen2::agent::Update for an agent run"
)]
pub type Update = crate::api::Update;
/// Default `Chat::tool_depth`. The new surface has no built-in tool loop.
#[deprecated(
    since = "0.1.0",
    note = "the new surface runs no tool loop; see api_spec.md §10"
)]
pub const DEFAULT_TOOL_DEPTH: usize = crate::api::DEFAULT_TOOL_DEPTH;
/// A spawned turn as a `Stream`. Now
/// [`AsyncEventStream`](crate::AsyncEventStream) from `Turn::stream_async`.
#[cfg(feature = "tokio")]
#[deprecated(since = "0.1.0", note = "use Turn::stream_async; see api_spec.md §22")]
pub type AsyncTurn = crate::api::AsyncTurn;
/// A GGUF header and a fit check. Now
/// [`advanced::fit::ModelInfo`](crate::advanced::fit::ModelInfo).
#[deprecated(since = "0.1.0", note = "use gen2::advanced::fit::ModelInfo")]
pub type ModelInfo = crate::api::FileModelInfo;
/// A fit verdict. Now [`advanced::fit::Fit`](crate::advanced::fit::Fit).
#[deprecated(since = "0.1.0", note = "use gen2::advanced::fit::Fit")]
pub type Fit = crate::api::Fit;
/// A fit verdict's outcome. Now [`advanced::fit::FitVerdict`](crate::advanced::fit::FitVerdict).
#[deprecated(since = "0.1.0", note = "use gen2::advanced::fit::FitVerdict")]
pub type FitVerdict = crate::api::FitVerdict;
