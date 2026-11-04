use std::ptr::{self, NonNull};
use std::slice;

use llama_cpp_sys_2 as sys;

use super::LlamaContext;
use crate::mtmd::MtmdContext;

const SAMPLES_PER_STEP: usize = 5_120;

/// Error from an incremental LFM3 audio input.
#[derive(Debug, thiserror::Error)]
pub enum MtmdAudioInputError {
    /// The projector cannot stream audio into this model.
    #[error("the loaded model and projector do not support incremental LFM3 audio input")]
    Unsupported,
    /// A microphone input or spoken reply is already in progress.
    #[error("another LFM3 audio operation is already active")]
    AlreadyActive,
    /// There is no active audio input.
    #[error("no LFM3 audio input is active")]
    NotActive,
    /// The caller must decode the user turn header before opening audio input.
    #[error("incremental LFM3 audio input requires a prefilled user turn header")]
    MissingPrefix,
    /// The PCM contains an invalid floating-point value.
    #[error("incremental LFM3 audio input contains a non-finite PCM sample")]
    InvalidPcm,
    /// No samples were pushed before commit.
    #[error("incremental LFM3 audio input is empty")]
    EmptyInput,
    /// The audio does not fit in the language context.
    #[error("incremental LFM3 audio input exceeds the remaining context")]
    ContextFull,
    /// The encoder did not return a valid embedding matrix.
    #[error("the LFM3 audio encoder returned invalid embeddings")]
    InvalidEmbeddings,
    /// The encoder or language model rejected a native operation.
    #[error("incremental LFM3 audio {operation} failed with code {code}")]
    Native {
        /// Name of the operation that failed.
        operation: &'static str,
        /// Status from llama.cpp.
        code: i32,
    },
}

#[derive(Debug)]
pub(super) struct AudioInput<'model> {
    stream: NonNull<sys::mtmd_audio_stream>,
    projector: &'model MtmdContext,
    embedding_width: usize,
    received_pcm: bool,
}

impl Drop for AudioInput<'_> {
    fn drop(&mut self) {
        let _guard = self.projector.lock_operations();
        unsafe { sys::mtmd_audio_stream_free(self.stream.as_ptr()) };
    }
}

impl<'model> LlamaContext<'model> {
    /// Open mono 16 kHz PCM input after prefilling the text before the audio.
    /// The encoder emits its own audio boundary embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is unavailable or the sequence is empty.
    pub fn start_mtmd_audio_input(
        &mut self,
        projector: &'model MtmdContext,
    ) -> Result<(), MtmdAudioInputError> {
        if self.mtmd_audio_input.is_some() || self.mtmd_audio.is_some() {
            return Err(MtmdAudioInputError::AlreadyActive);
        }
        if projector.model_identity != self.model.model
            || !projector.support_audio_stream()
            || projector.get_audio_sample_rate() != Some(16_000)
        {
            return Err(MtmdAudioInputError::Unsupported);
        }
        if self.kv_cache_seq_pos_max(0) < 0 {
            return Err(MtmdAudioInputError::MissingPrefix);
        }
        let embedding_width = unsafe { sys::llama_model_n_embd_inp(self.model.model.as_ptr()) };
        let embedding_width = usize::try_from(embedding_width)
            .ok()
            .filter(|&width| width != 0)
            .ok_or(MtmdAudioInputError::Unsupported)?;
        let stream = {
            let _guard = projector.lock_operations();
            unsafe {
                sys::mtmd_audio_stream_init(
                    projector.context.as_ptr(),
                    sys::mtmd_audio_stream_params_default(),
                )
            }
        };
        self.mtmd_audio_input = Some(AudioInput {
            stream: NonNull::new(stream).ok_or(MtmdAudioInputError::Unsupported)?,
            projector,
            embedding_width,
            received_pcm: false,
        });
        Ok(())
    }

    /// Push microphone samples and return the number of newly decoded positions.
    /// A small push may produce zero rows while the encoder buffers PCM.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is inactive, invalid, or cannot be decoded.
    /// After a native failure the input is closed. The sequence can contain
    /// positions from earlier successful decode batches.
    pub fn push_mtmd_audio_input(&mut self, samples: &[f32]) -> Result<usize, MtmdAudioInputError> {
        if self.mtmd_audio_input.is_none() {
            return Err(MtmdAudioInputError::NotActive);
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(MtmdAudioInputError::InvalidPcm);
        }
        let mut input = self
            .mtmd_audio_input
            .take()
            .ok_or(MtmdAudioInputError::NotActive)?;
        let mut decoded = 0;
        {
            let _guard = input.projector.lock_operations();
            for pcm in samples.chunks(SAMPLES_PER_STEP) {
                status("push", unsafe {
                    sys::mtmd_audio_stream_push(input.stream.as_ptr(), pcm.as_ptr(), pcm.len())
                })?;
                input.received_pcm = true;
                decoded += self.drain_mtmd_audio_input(&input)?;
            }
        }
        self.mtmd_audio_input = Some(input);
        Ok(decoded)
    }

    /// Commit the input, flushing its remaining samples and audio boundary
    /// into sequence zero. Returns the number of positions decoded on commit.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is inactive, empty, or cannot be decoded.
    /// After a native failure the input is closed.
    pub fn finish_mtmd_audio_input(&mut self) -> Result<usize, MtmdAudioInputError> {
        let input = self
            .mtmd_audio_input
            .take()
            .ok_or(MtmdAudioInputError::NotActive)?;
        if !input.received_pcm {
            self.mtmd_audio_input = Some(input);
            return Err(MtmdAudioInputError::EmptyInput);
        }
        let _guard = input.projector.lock_operations();
        status("commit", unsafe {
            sys::mtmd_audio_stream_flush(input.stream.as_ptr(), true)
        })?;
        self.drain_mtmd_audio_input(&input)
    }

    /// Release the encoder and discard PCM it has yet to decode. Already
    /// decoded positions remain in the language context.
    pub fn discard_mtmd_audio_input(&mut self) {
        drop(self.mtmd_audio_input.take());
    }

    fn drain_mtmd_audio_input(
        &mut self,
        input: &AudioInput<'_>,
    ) -> Result<usize, MtmdAudioInputError> {
        let mut decoded = 0;
        loop {
            let rows = unsafe { sys::mtmd_audio_stream_step(input.stream.as_ptr(), -1) };
            match rows {
                -1 | 0 => return Ok(decoded),
                error if error < 0 => return Err(native_error("encode", error)),
                _ => {}
            }

            let position = self
                .kv_cache_seq_pos_max(0)
                .checked_add(1)
                .ok_or(MtmdAudioInputError::ContextFull)?;
            let end = position
                .checked_add(rows)
                .ok_or(MtmdAudioInputError::ContextFull)?;
            let context_size = unsafe { sys::llama_n_ctx_seq(self.context.as_ptr()) };
            if position <= 0 || end.unsigned_abs() >= context_size {
                return Err(MtmdAudioInputError::ContextFull);
            }

            let sample_count = (rows as usize)
                .checked_mul(input.embedding_width)
                .filter(|&count| count <= isize::MAX as usize / size_of::<f32>())
                .ok_or(MtmdAudioInputError::InvalidEmbeddings)?;
            let data = unsafe { sys::mtmd_audio_stream_get_output_embd(input.stream.as_ptr()) };
            if data.is_null() || !data.is_aligned() {
                return Err(MtmdAudioInputError::InvalidEmbeddings);
            }
            let embeddings = unsafe { slice::from_raw_parts(data, sample_count) };
            self.decode_mtmd_audio_rows(position, embeddings, input.embedding_width)?;
            decoded += rows as usize;
        }
    }

    fn decode_mtmd_audio_rows(
        &mut self,
        mut position: i32,
        embeddings: &[f32],
        width: usize,
    ) -> Result<(), MtmdAudioInputError> {
        let max_batch_rows = self.n_batch() as usize;
        let batch_width = max_batch_rows
            .checked_mul(width)
            .filter(|&count| count > 0)
            .ok_or(MtmdAudioInputError::InvalidEmbeddings)?;
        self.initialized_logits.clear();
        for chunk in embeddings.chunks(batch_width) {
            let count = chunk.len() / width;
            let count_i32 =
                i32::try_from(count).map_err(|_| MtmdAudioInputError::InvalidEmbeddings)?;
            let mut rows = chunk.to_vec();
            let mut positions: Vec<i32> = (position..position + count_i32).collect();
            let mut n_seq_ids = vec![1; count];
            let mut seq_ids = vec![0; count];
            let mut seq_ptrs: Vec<_> = seq_ids.iter_mut().map(ptr::from_mut).collect();
            let mut logits = vec![0; count];
            let batch = sys::llama_batch {
                n_tokens: count_i32,
                token: ptr::null_mut(),
                embd: rows.as_mut_ptr(),
                pos: positions.as_mut_ptr(),
                n_seq_id: n_seq_ids.as_mut_ptr(),
                seq_id: seq_ptrs.as_mut_ptr(),
                logits: logits.as_mut_ptr(),
            };
            status("decode", unsafe {
                sys::llama_decode(self.context.as_ptr(), batch)
            })?;
            position += count_i32;
        }
        Ok(())
    }
}

fn status(operation: &'static str, code: i32) -> Result<(), MtmdAudioInputError> {
    if code == 0 {
        Ok(())
    } else {
        Err(native_error(operation, code))
    }
}

fn native_error(operation: &'static str, code: i32) -> MtmdAudioInputError {
    MtmdAudioInputError::Native { operation, code }
}
