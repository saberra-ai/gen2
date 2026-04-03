use std::num::NonZeroU32;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Max context size (total tokens in prompt + generation)
    pub ctx_size: u32,

    /// Optional random seed for reproducibility
    pub seed: u32,

    pub batch_size: usize,

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
    pub fn get_ctx_size(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.ctx_size)
    }
}
