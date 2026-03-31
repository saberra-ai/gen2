use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use rand::{RngCore, rng};
use std::num::NonZeroU32;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Max context size (total tokens in prompt + generation)
    pub ctx_size: u32,

    /// Optional random seed for reproducibility
    pub seed: u32,

    pub batch_size: usize,

    key_value_overrides: Vec<(String, ParamOverrideValue)>,

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    disable_gpu: bool,

    /// Optional # of layers to offload to GPU
    pub gpu_layers: Option<u32>,

    /// Threads used during generation
    pub threads: Option<i32>,

    /// Threads used during prompt/batch processing
    pub threads_batch: Option<i32>,

    pub temperature: f32,
    pub top_p: f32,
    pub top_p_keep: usize,
    pub top_k: i32,
    pub repeat_penalty: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            ctx_size: 8000,
            seed: 123,
            batch_size: 4000,
            key_value_overrides: vec![],
            #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
            disable_gpu: false,
            gpu_layers: None,
            threads: None,
            threads_batch: None,
            temperature: 0.7,
            top_p: 0.95,
            top_p_keep: 1,
            top_k: 20,
            repeat_penalty: 65,
        }
    }
}

impl ModelConfig {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx_size: u32,
        seed: Option<u32>,
        gpu_layers: Option<u32>,
        threads: Option<i32>,
        threads_batch: Option<i32>,
        temperature: f32,
        top_p: f32,
        top_p_keep: usize,
        top_k: i32,
        repeat_penalty: i32,
    ) -> ModelConfig {
        let seed = seed.unwrap_or(rng().next_u32());
        ModelConfig {
            ctx_size,
            seed,
            batch_size: 1024,
            key_value_overrides: vec![],
            #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
            disable_gpu: false,
            gpu_layers,
            threads,
            threads_batch,
            temperature,
            top_p,
            top_p_keep,
            top_k,
            repeat_penalty,
        }
    }

    pub fn get_ctx_size(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.ctx_size)
    }
}
