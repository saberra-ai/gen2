use crate::bundle::ModelMeta;
use crate::engine::Capabilities;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::MtmdContext;

pub struct ModelBundle {
    pub model: LlamaModel,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
    pub mtmd_ctx: Option<MtmdContext>,
    pub mtmd_marker: Option<String>,
}

impl std::fmt::Debug for ModelBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("ModelBundle");
        ds.field("capabilities", &self.capabilities)
            .field("meta", &self.meta);
        {
            ds.field("has_mtmd", &self.mtmd_ctx.is_some());
        }
        ds.finish()
    }
}

impl ModelBundle {}
