use crate::bundle::ModelMeta;
use crate::engine::{Capabilities, GpuOffload};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::MtmdContext;

pub struct ModelBundle {
    pub model: LlamaModel,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
    pub mtmd_ctx: Option<MtmdContext>,
    pub mtmd_marker: Option<String>,
    /// Where the weights went, as the load placed them.
    pub offload: GpuOffload,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("ModelBundle");
        ds.field("capabilities", &self.capabilities)
            .field("meta", &self.meta)
            .field("offload", &self.offload);
        {
            ds.field("has_mtmd", &self.mtmd_ctx.is_some());
        }
        ds.finish()
    }
}

impl ModelBundle {}
