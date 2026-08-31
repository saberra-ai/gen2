mod codec;
pub mod store;
mod types;

// Consumed by the llama backend (gen2/backend/llama/session.rs); that consumer
// is compiled out of the default feature surface, so this re-export reads as
// unused there while being required under backend-llamacpp.
#[allow(unused_imports)]
pub(crate) use codec::{build_blob, parse_blob, read_from_path, write_to_path};
pub use types::{KvHeader, KvLoadReport};
pub use types::{KvLoadSpec, KvMeta, KvSaveSpec, KvSnapshot};
