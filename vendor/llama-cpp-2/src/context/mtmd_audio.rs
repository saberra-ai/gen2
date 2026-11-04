use std::ptr::{self, NonNull};
use std::slice;

use llama_cpp_sys_2 as sys;

use super::LlamaContext;
use crate::mtmd::MtmdContext;
use crate::token::LlamaToken;

/// Output mode for a response that continues an LFM3 sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MtmdAudioOutputMode {
    /// Let LFM3 choose whether to emit text or audio.
    #[default]
    Auto,
    /// Force text output.
    Text,
    /// Open the reply with speech.
    Speech,
    /// Open the reply with interleaved text and speech.
    Mixed,
}

impl MtmdAudioOutputMode {
    fn native(self) -> sys::mtmd_helper_gen_audio_out_mode {
        match self {
            Self::Auto => sys::MTMD_HELPER_GEN_AUDIO_OUT_AUTO,
            Self::Text => sys::MTMD_HELPER_GEN_AUDIO_OUT_TEXT,
            Self::Speech => sys::MTMD_HELPER_GEN_AUDIO_OUT_AUDIO,
            Self::Mixed => sys::MTMD_HELPER_GEN_AUDIO_OUT_MIXED,
        }
    }
}

/// Mono floating-point PCM produced by the LFM3 codec.
#[derive(Debug)]
pub struct MtmdPcmChunk {
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples in playback order.
    pub samples: Vec<f32>,
}

/// Output of one decoder step.
#[derive(Debug)]
pub struct MtmdAudioStep {
    /// Position at which the next decoded token or frame would be placed.
    pub position: i32,
    /// Whether the model emitted the end of the reply.
    pub stopped: bool,
    /// PCM produced by this step, if any.
    pub audio: Option<MtmdPcmChunk>,
}

/// Error from an LFM3 speech response.
#[derive(Debug, thiserror::Error)]
pub enum MtmdAudioGenerationError {
    /// The supplied projector or language context cannot run LFM3 speech.
    #[error("the loaded model and projector do not support LFM3 audio generation")]
    Unsupported,
    /// A response is already active on this language context.
    #[error("an LFM3 audio response is already active")]
    AlreadyActive,
    /// An operation requires an active audio response.
    #[error("no LFM3 audio response is active")]
    NotActive,
    /// There must be a prefilled sequence and a nonempty prompt suffix.
    #[error("LFM3 audio generation requires a prefilled sequence and a nonempty prompt suffix")]
    InvalidPrompt,
    /// Text slots require a sampled token and audio slots require none.
    #[error("the sampled token does not match the next LFM3 audio generation slot")]
    InvalidStep,
    /// The prompt did not expose the backbone hidden state.
    #[error("the LFM3 backbone did not produce a hidden state")]
    MissingHiddenState,
    /// A native operation failed.
    #[error("LFM3 audio {operation} failed with code {code}")]
    Native {
        /// Operation the caller was performing.
        operation: &'static str,
        /// Status from llama.cpp.
        code: i32,
    },
    /// The native PCM buffer is malformed.
    #[error("the LFM3 codec returned an invalid PCM buffer")]
    InvalidPcm,
}

#[derive(Debug)]
pub(super) struct AudioGenerator<'model> {
    native: NonNull<sys::mtmd_helper_gen_audio>,
    mtmd: &'model MtmdContext,
    hidden: Vec<f32>,
    restore_embeddings: bool,
}

impl AudioGenerator<'_> {
    fn next_is_text(&self) -> bool {
        unsafe { sys::mtmd_helper_gen_audio_next_is_text(self.native.as_ptr()) }
    }

    fn read_pcm(&mut self) -> Result<Option<MtmdPcmChunk>, MtmdAudioGenerationError> {
        let mut sample_rate = 0;
        let mut data = ptr::null();
        let mut data_len = 0;
        let mut n_samples = 0;
        status("PCM read", unsafe {
            sys::mtmd_helper_gen_audio_get_output(
                self.native.as_ptr(),
                &mut sample_rate,
                &mut data,
                &mut data_len,
                &mut n_samples,
            )
        })?;
        if data_len == 0 && n_samples == 0 {
            return Ok(None);
        }

        let rate = u32::try_from(sample_rate).map_err(|_| MtmdAudioGenerationError::InvalidPcm)?;
        let count = usize::try_from(n_samples).map_err(|_| MtmdAudioGenerationError::InvalidPcm)?;
        let samples = data.cast::<f32>();
        if rate == 0
            || samples.is_null()
            || !samples.is_aligned()
            || count.checked_mul(size_of::<f32>()) != Some(data_len)
            || data_len > isize::MAX as usize
        {
            return Err(MtmdAudioGenerationError::InvalidPcm);
        }
        // In PCM mode the upstream buffer is a contiguous array of native floats.
        let samples = unsafe { slice::from_raw_parts(samples, count) }.to_vec();
        Ok(Some(MtmdPcmChunk {
            sample_rate: rate,
            samples,
        }))
    }
}

impl Drop for AudioGenerator<'_> {
    fn drop(&mut self) {
        let _guard = self.mtmd.lock_operations();
        unsafe { sys::mtmd_helper_gen_audio_free(self.native.as_ptr()) };
    }
}

impl<'model> LlamaContext<'model> {
    /// Continue sequence zero and set up the LFM3 speech generator. The
    /// native helper evaluates the unprocessed assistant prompt suffix and
    /// may append a control token for the selected output mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the projector lacks the codec or setup fails.
    pub fn start_mtmd_audio(
        &mut self,
        mtmd: &'model MtmdContext,
        position: i32,
        prompt_suffix: &str,
        mode: MtmdAudioOutputMode,
        seed: Option<u32>,
    ) -> Result<i32, MtmdAudioGenerationError> {
        if self.mtmd_audio.is_some() {
            return Err(MtmdAudioGenerationError::AlreadyActive);
        }
        if position <= 0
            || position as u32 >= self.n_ctx()
            || self.kv_cache_seq_pos_max(0) != position - 1
            || prompt_suffix.is_empty()
            || prompt_suffix.len() > i32::MAX as usize - 16
        {
            return Err(MtmdAudioGenerationError::InvalidPrompt);
        }
        if mtmd.model_identity != self.model.model
            || !mtmd.supports_lfm3_audio_generation()
            || self.embeddings_out_len() != usize::try_from(self.model.n_embd()).unwrap_or(0)
            || unsafe { sys::llama_pooling_type(self.context.as_ptr()) }
                != sys::LLAMA_POOLING_TYPE_NONE
        {
            return Err(MtmdAudioGenerationError::Unsupported);
        }

        let native = {
            let _guard = mtmd.lock_operations();
            unsafe { sys::mtmd_helper_gen_audio_init(self.context.as_ptr(), mtmd.context.as_ptr()) }
        };
        let native = NonNull::new(native).ok_or(MtmdAudioGenerationError::Unsupported)?;
        let generator = AudioGenerator {
            native,
            mtmd,
            hidden: Vec::new(),
            restore_embeddings: self.embeddings_enabled,
        };

        self.mtmd_audio = Some(generator);
        self.set_mtmd_embeddings(true);
        if let Err(error) = self.prefill_mtmd_audio(prompt_suffix, position, mode, seed) {
            self.initialized_logits.clear();
            self.discard_mtmd_audio();
            return Err(error);
        }
        Ok(self.kv_cache_seq_pos_max(0) + 1)
    }

    fn set_mtmd_embeddings(&mut self, enabled: bool) {
        if self.embeddings_enabled != enabled {
            unsafe { sys::llama_set_embeddings(self.context.as_ptr(), enabled) };
            self.embeddings_enabled = enabled;
        }
    }

    fn prefill_mtmd_audio(
        &mut self,
        prompt_suffix: &str,
        position: i32,
        mode: MtmdAudioOutputMode,
        seed: Option<u32>,
    ) -> Result<(), MtmdAudioGenerationError> {
        let generator = self
            .mtmd_audio
            .as_ref()
            .ok_or(MtmdAudioGenerationError::NotActive)?;
        {
            let _guard = generator.mtmd.lock_operations();
            let mut input = unsafe { sys::mtmd_helper_gen_audio_inp_default() };
            input.prompt = prompt_suffix.as_ptr().cast();
            input.prompt_len = prompt_suffix.len();
            input.n_past = position;
            input.seed = seed.unwrap_or(u32::MAX);
            input.stream = true;
            input.out_mode = mode.native();
            input.out_type = sys::MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
            status("setup", unsafe {
                sys::mtmd_helper_gen_audio_set_input(generator.native.as_ptr(), &input)
            })?;
            let n_batch = i32::try_from(self.n_batch()).unwrap_or(i32::MAX);
            loop {
                let remaining = unsafe {
                    sys::mtmd_helper_gen_audio_step_prompt(generator.native.as_ptr(), n_batch)
                };
                if remaining <= 0 {
                    status("prompt decode", remaining)?;
                    break;
                }
            }
        }

        let hidden = self
            .embeddings_ith(-1)
            .map_err(|_| MtmdAudioGenerationError::MissingHiddenState)?
            .to_vec();
        if let Some(generator) = self.mtmd_audio.as_mut() {
            generator.hidden = hidden;
        }
        self.record_mtmd_logits();
        Ok(())
    }

    fn record_mtmd_logits(&mut self) {
        self.initialized_logits.clear();
        self.initialized_logits.push(-1);
    }

    /// Whether the next audio generator step expects a sampled text token.
    #[must_use]
    pub fn mtmd_audio_next_is_text(&self) -> bool {
        self.mtmd_audio
            .as_ref()
            .is_some_and(AudioGenerator::next_is_text)
    }

    /// Text logit adjustments for the configured output mode.
    #[must_use]
    pub fn mtmd_audio_logit_bias(&self) -> Vec<(LlamaToken, f32)> {
        let Some(generator) = self.mtmd_audio.as_ref() else {
            return Vec::new();
        };
        let mut count = 0;
        let bias = unsafe {
            sys::mtmd_helper_gen_audio_get_logit_bias(generator.native.as_ptr(), &mut count)
        };
        if count == 0 || bias.is_null() {
            return Vec::new();
        }
        unsafe { slice::from_raw_parts(bias, count) }
            .iter()
            .map(|entry| (LlamaToken(entry.token), entry.bias))
            .collect()
    }

    /// Feed a text token or generate an audio frame from the backbone hidden
    /// state. The helper updates the language context directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the generator is inactive or a native step fails.
    pub fn step_mtmd_audio(
        &mut self,
        sampled: Option<LlamaToken>,
    ) -> Result<MtmdAudioStep, MtmdAudioGenerationError> {
        let generator = self
            .mtmd_audio
            .as_mut()
            .ok_or(MtmdAudioGenerationError::NotActive)?;
        let token_is_invalid =
            sampled.is_some_and(|token| !(0..self.model.n_vocab()).contains(&token.0));
        if generator.next_is_text() != sampled.is_some() || token_is_invalid {
            return Err(MtmdAudioGenerationError::InvalidStep);
        }

        let mut hidden = ptr::null();
        let mut stopped = false;
        let audio = {
            let _guard = generator.mtmd.lock_operations();
            status("decode", unsafe {
                sys::mtmd_helper_gen_audio_step_gen(
                    generator.native.as_ptr(),
                    sampled.map_or(-1, |token| token.0),
                    generator.hidden.as_ptr(),
                    &mut hidden,
                    &mut stopped,
                )
            })?;
            if !hidden.is_null() {
                // The helper owns this buffer until the next step.
                let next_hidden = unsafe { slice::from_raw_parts(hidden, generator.hidden.len()) };
                generator.hidden.copy_from_slice(next_hidden);
            } else if !stopped {
                return Err(MtmdAudioGenerationError::MissingHiddenState);
            }
            generator.read_pcm()?
        };

        if !hidden.is_null() {
            self.record_mtmd_logits();
        }
        Ok(MtmdAudioStep {
            position: self.kv_cache_seq_pos_max(0) + 1,
            stopped,
            audio,
        })
    }

    /// Flush remaining PCM and release the native speech generator.
    ///
    /// # Errors
    ///
    /// Returns an error if the codec cannot flush its final PCM.
    pub fn finish_mtmd_audio(&mut self) -> Result<Option<MtmdPcmChunk>, MtmdAudioGenerationError> {
        let Some(mut generator) = self.mtmd_audio.take() else {
            return Ok(None);
        };
        let output = {
            let _guard = generator.mtmd.lock_operations();
            status("flush", unsafe {
                sys::mtmd_helper_gen_audio_flush(generator.native.as_ptr())
            })
            .and_then(|()| generator.read_pcm())
        };
        self.set_mtmd_embeddings(generator.restore_embeddings);
        output
    }

    /// Release an active speech generator without flushing its buffered PCM.
    pub fn discard_mtmd_audio(&mut self) {
        if let Some(generator) = self.mtmd_audio.take() {
            self.set_mtmd_embeddings(generator.restore_embeddings);
        }
    }
}

fn status(operation: &'static str, code: i32) -> Result<(), MtmdAudioGenerationError> {
    if code == 0 {
        Ok(())
    } else {
        Err(MtmdAudioGenerationError::Native { operation, code })
    }
}
