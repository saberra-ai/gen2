//! ONNX model bundle — holds the ort Session, tokenizer, and metadata.

use crate::gen2::backend::common::tokenizer::HfTokenizer;
use crate::gen2::bundle::ModelMeta;
use crate::gen2::engine::Capabilities;
use parking_lot::Mutex;

pub struct ModelBundle {
    pub session: Mutex<ort::session::Session>,
    pub tokenizer: HfTokenizer,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
    pub num_layers: usize,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBundle(ONNX)")
            .field("capabilities", &self.capabilities)
            .field("meta", &self.meta)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}
