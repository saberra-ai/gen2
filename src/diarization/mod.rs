//! Local Nemotron 3 speaker timelines, independent of transcription and chat.
//!
//! Load through [`crate::Runtime::load_diarizer`]. Native calls stay on the
//! owning worker; the host supplies the qualified NeMo-Speech.cpp library.

mod audio;
mod bundle;
mod error;
#[cfg(feature = "backend-diar-nemo")]
mod ffi;
#[cfg(feature = "backend-diar-nemo")]
mod nemo;
mod provision;
mod stream;
mod types;
mod worker;

/// Cooperative cancellation shared with other local speech operations.
pub use crate::speech::Cancellation;
pub use audio::Audio;
pub use bundle::{DiarizationBundle, MODEL_FILENAME, MODEL_REVISION, MODEL_SHA256};
pub use error::{DiarizationError, Result};
pub use provision::download_nemotron;
pub use stream::{
    DiarizationStream, MAX_PUSH_SAMPLES, MAX_SESSION_SAMPLES, PushError, Snapshot, StreamStatus,
};
pub use types::*;
pub use worker::{Diarizer, Status};
pub(crate) use worker::{Monitor, spawn};
