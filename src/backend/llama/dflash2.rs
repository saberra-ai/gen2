use crate::engine::{DFlash2Config, ExecError};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, params::LlamaModelParams};

pub(crate) struct DraftModel {
    pub model: LlamaModel,
    pub config: DFlash2Config,
    pub block_size: usize,
}

impl DraftModel {
    pub fn load(
        backend: &LlamaBackend,
        target: &LlamaModel,
        config: &DFlash2Config,
        params: &LlamaModelParams,
    ) -> Result<Self, ExecError> {
        let model = LlamaModel::load_from_file(backend, &config.model_path, params)
            .map_err(|e| ExecError::Other(e.into()))?;
        let selector = model
            .meta_val_str("dflash.selector_top_k")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let block = model
            .meta_val_str("dflash.block_size")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if selector == 0 || block <= config.max_tokens {
            return Err(ExecError::InvalidModelFile(
                "not a DFlash2 draft, or proposal size exceeds its trained block".into(),
            ));
        }
        if !model.is_dflash2_compatible_with(target) {
            return Err(ExecError::InvalidModelFile(
                "DFlash2 target and draft vocabulary, dimensions or extraction layers do not match"
                    .into(),
            ));
        }
        Ok(Self {
            model,
            config: config.clone(),
            block_size: block,
        })
    }
}
