use super::{DecodeState, Session};
use crate::engine::{ExecError, HookEvent};
use crate::types::message::{MessageBody, MessageChunk, MessageContent};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::mtmd::MtmdAudioInputError;
use llama_cpp_2::token::LlamaToken;
use std::sync::atomic::Ordering;

use super::super::media::MAX_AUDIO_SAMPLES;

#[derive(Debug)]
pub(super) struct MicrophoneCapture {
    pub(super) suffix: Vec<LlamaToken>,
    pub(super) pcm: Vec<f32>,
}

pub(super) fn native_error(err: MtmdAudioInputError) -> ExecError {
    match err {
        MtmdAudioInputError::ContextFull => ExecError::ContextOverflow(err.to_string()),
        _ => ExecError::Other(err.into()),
    }
}

impl DecodeState {
    fn check_microphone_capacity(&self, suffix_len: usize) -> Result<(), ExecError> {
        let available = self.ctx_cell.with_dependent(|_, ctx| ctx.n_ctx()) as usize;
        if (self.cur_pos as usize).saturating_add(suffix_len) >= available {
            return Err(ExecError::ContextOverflow(
                "microphone turn exceeds the remaining context".into(),
            ));
        }
        Ok(())
    }

    fn sync_microphone_position(&mut self) {
        self.cur_pos = self
            .ctx_cell
            .with_dependent(|_, ctx| ctx.kv_cache_seq_pos_max(0) + 1);
    }

    fn finish_microphone_prefill(&mut self, suffix: &[LlamaToken]) -> Result<(), ExecError> {
        self.prefill_start_us = llama_cpp_2::ggml_time_us() as u64;
        self.ctx_cell
            .with_dependent_mut(|_, ctx| ctx.finish_mtmd_audio_input())
            .map_err(native_error)?;
        self.sync_microphone_position();
        self.check_microphone_capacity(suffix.len())?;

        self.ctx_cell.with_dependent_mut(|_, ctx| {
            let batch_size = ctx.n_batch() as usize;
            let mut batch = LlamaBatch::new(batch_size.min(suffix.len()), 1);
            for chunk in suffix.chunks(batch_size) {
                batch.clear();
                for (i, &token) in chunk.iter().enumerate() {
                    batch
                        .add(token, self.cur_pos + i as i32, &[0], i + 1 == chunk.len())
                        .map_err(|err| ExecError::Other(err.into()))?;
                }
                ctx.decode(&mut batch)?;
                self.cur_pos += chunk.len() as i32;
                self.logits_i = chunk.len() as i32 - 1;
            }
            Ok(())
        })
    }
}

impl Session {
    pub(super) fn is_capturing_microphone(&self) -> bool {
        self.state
            .lock()
            .as_ref()
            .is_some_and(|state| state.microphone.is_some())
    }

    pub(super) fn push_microphone(&self, samples: &[f32]) -> Result<usize, ExecError> {
        let mut guard = self.state.lock();
        let state = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let capture = state
            .microphone
            .as_mut()
            .ok_or(ExecError::InvalidArg("microphone is not capturing"))?;
        if !(1..=16_000).contains(&samples.len()) {
            return Err(ExecError::InvalidArg(
                "microphone chunks must contain 1 to 16000 samples",
            ));
        }
        if capture.pcm.len().saturating_add(samples.len()) > MAX_AUDIO_SAMPLES {
            return Err(ExecError::InvalidArg("microphone turn exceeds 192 seconds"));
        }
        if samples.iter().any(|sample| !(-1.0..=1.0).contains(sample)) {
            return Err(ExecError::InvalidArg(
                "microphone samples must be finite and normalized",
            ));
        }
        let suffix_len = capture.suffix.len();
        let result = match state
            .ctx_cell
            .with_dependent_mut(|_, ctx| ctx.push_mtmd_audio_input(samples))
        {
            Ok(rows) => {
                capture.pcm.extend_from_slice(samples);
                state.sync_microphone_position();
                state.check_microphone_capacity(suffix_len).map(|()| rows)
            }
            Err(err) => Err(native_error(err)),
        };
        if result.is_err() {
            *guard = None;
            self.stopped.store(true, Ordering::Release);
        }
        result
    }

    pub(super) fn commit_microphone(&self) -> Result<Vec<f32>, ExecError> {
        let mut guard = self.state.lock();
        let state = guard
            .as_mut()
            .ok_or(ExecError::InvalidArg("session already consumed"))?;
        let capture = state
            .microphone
            .as_ref()
            .ok_or(ExecError::InvalidArg("microphone is not capturing"))?;
        if capture.pcm.len() < 320 {
            return Err(ExecError::InvalidArg(
                "microphone turn requires at least 20 milliseconds of audio",
            ));
        }

        let mut messages = self.messages.write();
        let Some(MessageBody::Content {
            content: MessageContent::MultipleChunks(chunks),
        }) = messages.last_mut().map(|message| &mut message.body)
        else {
            return Err(ExecError::InvalidArg(
                "microphone turn is missing from the transcript",
            ));
        };
        let Some(part @ MessageChunk::Microphone) = chunks.last_mut() else {
            return Err(ExecError::InvalidArg(
                "microphone turn is missing from the transcript",
            ));
        };

        let capture = state.microphone.take().expect("active capture");
        if let Err(err) = state.finish_microphone_prefill(&capture.suffix) {
            *guard = None;
            self.stopped.store(true, Ordering::Release);
            return Err(err);
        }
        *part = MessageChunk::AudioPcm {
            samples: capture.pcm.clone(),
        };
        let prompt_tokens = state.cur_pos as usize;
        self.hooks.emit(HookEvent::SessionPrefillStart {
            session_id: self.id,
            prompt_tokens,
        });
        self.hooks.emit(HookEvent::SessionPrefillOk {
            session_id: self.id,
            prompt_tokens,
        });
        Ok(capture.pcm)
    }
}
