mod pcm;

use crate::generation::{OutputMode, SpeechChunk};

use super::error::Result;
use super::event::Event;
use super::response::Response;
use super::turn::{Turn, microphone_cancelled};
use pcm::PcmNormalizer;

/// Interleaved PCM chunks for one voice turn. All chunks use the same sample
/// rate and channel count. The source can yield chunks as they are recorded.
pub struct VoiceInput<I> {
    chunks: I,
    sample_rate: u32,
    channels: u16,
}

impl<I> VoiceInput<I> {
    /// Declare the source format once. Samples are normalized, finite `f32`
    /// values in `[-1, 1]`, with channels interleaved in frame order. Supported
    /// formats have 1 to 32 channels and a sample rate from 8 to 192 kHz.
    pub fn new(chunks: I, sample_rate: u32, channels: u16) -> Self {
        Self {
            chunks,
            sample_rate,
            channels,
        }
    }
}

impl Turn<'_> {
    /// Read one spoken turn, then stream speech to `on_speech` and return
    /// the complete reply. This convenience method requests both text and
    /// speech.
    ///
    /// Gen2 downmixes and resamples incoming audio to mono 16 kHz PCM, and
    /// feeds the model while the source produces samples. The source ending
    /// commits the turn. A source error discards the uncommitted capture.
    /// A playback error stops the reply and preserves the committed input.
    ///
    /// Call this from a worker thread. The source and sink can block, so a
    /// host that supports cancellation should also unblock its own I/O.
    pub fn voice<I, B>(
        self,
        input: VoiceInput<I>,
        mut on_speech: impl FnMut(&SpeechChunk) -> Result<()>,
    ) -> Result<Response>
    where
        I: IntoIterator<Item = Result<B>>,
        B: AsRef<[f32]>,
    {
        let mut pcm = PcmNormalizer::new(input.sample_rate, input.channels)?;
        let cancel = self.canceller();
        let mut capture = self
            .output_mode(OutputMode::TextAndSpeech)
            .start_microphone()?;

        let mut feed = |samples: &[f32]| capture.push(samples).map(|_| ());
        let mut chunks = input.chunks.into_iter();
        loop {
            if cancel.is_cancelled() {
                return Err(microphone_cancelled());
            }
            let Some(chunk) = chunks.next() else { break };
            pcm.push(chunk?.as_ref(), &mut feed)?;
        }
        drop(chunks);
        pcm.finish(feed)?;

        let mut reply = capture.commit()?;
        for event in &mut reply {
            if let Event::SpeechDelta(chunk) = event?
                && let Err(err) = on_speech(&chunk)
            {
                reply.canceller().cancel();
                return Err(err);
            }
        }
        reply.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Capabilities;
    use crate::test_support::Script;
    use crate::{Error, Runtime, Session};

    #[test]
    fn voice_feeds_before_eof_and_discards_a_failed_source() {
        let script = Script::new()
            .capable_of(Capabilities::TEXT | Capabilities::MICROPHONE | Capabilities::SPEECH);
        let model = Runtime::scripted(script.clone());
        let mut session = Session::new();
        let chunks = std::iter::once(Ok(vec![0.25f32; 320])).chain(std::iter::once_with(|| {
            assert!(script.calls().iter().any(|call| call == "push_microphone"));
            Err(Error::Generation {
                code: "capture_failed".into(),
                message: "input device disconnected".into(),
            })
        }));

        let error = model
            .turn(&mut session)
            .voice(VoiceInput::new(chunks, 16_000, 1), |_| Ok(()))
            .unwrap_err();
        assert_eq!(error.code(), Some("capture_failed"));
        assert!(
            !script
                .calls()
                .iter()
                .any(|call| call == "commit_microphone")
        );
        assert!(session.all_messages().is_empty());
    }
}
