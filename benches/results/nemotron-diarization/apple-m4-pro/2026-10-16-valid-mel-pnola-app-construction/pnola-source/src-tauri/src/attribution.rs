use crate::engine::Job;
use gen2::diarization::{
    AttributedWord, Diarization, DiarizationError, DiarizeOptions, MAX_PUSH_SAMPLES,
    MAX_SESSION_SAMPLES,
};
use serde::Serialize;
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SpeakerAttribution {
    Unavailable {
        reason: String,
    },
    Complete {
        session_id: String,
        revision: u64,
        complete: bool,
        timeline: Box<Diarization>,
        words: Vec<AttributedWord>,
        words_complete: bool,
        timing_method: &'static str,
        alignment_error: Option<String>,
    },
}
impl SpeakerAttribution {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }
    pub fn add_words(&mut self, result: &gen2::speech::Transcription, offset: f64) {
        if let Self::Complete {
            timeline,
            words,
            alignment_error,
            ..
        } = self
        {
            match gen2::diarization::attribute_bark(result, timeline, offset) {
                Ok(result) => words.extend(result.words),
                Err(error) => *alignment_error = Some(error.to_string()),
            }
        }
    }
    pub fn finish_words(&mut self, asr_succeeded: bool) {
        if let Self::Complete {
            words_complete,
            alignment_error,
            ..
        } = self
        {
            *words_complete = asr_succeeded && alignment_error.is_none();
        }
    }
}

/// One native session spans the entire normalized file, independently of ASR cuts.
pub fn run(
    job: &Job,
    path: &Path,
    frames: u32,
    notify: impl Fn(f64, f64) -> Result<(), String>,
) -> SpeakerAttribution {
    let Some(model) = &job.diarizer else {
        return SpeakerAttribution::unavailable(
            job.diarization_error
                .as_deref()
                .unwrap_or("Speaker labels are unavailable."),
        );
    };
    if frames as usize > MAX_SESSION_SAMPLES {
        return SpeakerAttribution::unavailable(
            "Speaker labels support recordings up to one hour. Transcription can continue.",
        );
    }
    let result = (|| -> Result<SpeakerAttribution, String> {
        let seconds = frames as f64 / 16000.;
        let mut stream = model
            .start_stream(DiarizeOptions {
                cancellation: job.cancellation.child_token(),
                deadline: Some(Instant::now() + Duration::from_secs_f64((seconds * 2.).max(120.))),
                ..Default::default()
            })
            .map_err(|e| e.to_string())?;
        let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
        let mut source = reader.samples::<i16>();
        let mut last_report = 0;
        loop {
            let mut block = Vec::with_capacity(MAX_PUSH_SAMPLES);
            for sample in source.by_ref().take(MAX_PUSH_SAMPLES) {
                block.push(sample.map_err(|e| e.to_string())? as f32 / 32768.);
            }
            if block.is_empty() {
                break;
            }
            loop {
                match stream.try_push(block, 16000) {
                    Ok(()) => break,
                    Err(rejected) if matches!(rejected.error, DiarizationError::Busy) => {
                        block = rejected.samples;
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(rejected) => return Err(rejected.error.to_string()),
                }
            }
            if stream.accepted_samples() - last_report >= 5 * 16000 {
                notify(stream.accepted_samples() as f64 / 16000., seconds)?;
                last_report = stream.accepted_samples();
            }
        }
        if stream.accepted_samples() != frames as usize {
            return Err("Incomplete normalized audio for speaker labels.".into());
        }
        let timeline = stream.finish().map_err(|e| e.to_string())?;
        let snapshot = stream.snapshot().ok_or("Missing final speaker timeline.")?;
        Ok(SpeakerAttribution::Complete {
            session_id: stream.session_id().to_string(),
            revision: snapshot.revision,
            complete: true,
            timeline: Box::new((*timeline).clone()),
            words: Vec::new(),
            words_complete: false,
            timing_method: "bark_token_intervals_v1",
            alignment_error: None,
        })
    })();
    result.unwrap_or_else(SpeakerAttribution::unavailable)
}
