use crate::bundle::ModelMeta;
use crate::engine::{Capabilities, GpuOffload};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::MtmdContext;

pub struct ModelBundle {
    #[cfg(feature = "llama-dflash2")]
    pub(crate) draft: Option<super::dflash2::DraftModel>,
    pub model: LlamaModel,
    pub capabilities: Capabilities,
    pub meta: ModelMeta,
    /// Shared F16 geometry cost read from this owned native model after load.
    /// This covers attention arithmetic, not native workspace measurement.
    pub(crate) context_kv_bytes_per_token: u64,
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
