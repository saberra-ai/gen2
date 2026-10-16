//! Bounded, optional speaker processing beside durable live ASR capture.
use crate::engine::Job;
use gen2::{
    diarization::{
        Diarization, DiarizationError, DiarizationStream, DiarizeOptions, MAX_PUSH_SAMPLES,
        MAX_SESSION_SAMPLES, SpeakerTurn,
    },
    speech::Cancellation,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Unavailable,
    Active,
    Complete,
    Interrupted,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub turns: Vec<SpeakerTurn>,
    pub audio_duration_seconds: f64,
    pub model_revision: String,
    pub runtime_version: String,
    pub backend: String,
    /// chunk, right, left, FIFO, cache, update, in coarse encoder frames.
    pub geometry: [i32; 6],
    pub native_compute_micros: u64,
}
impl From<&Diarization> for Timeline {
    fn from(value: &Diarization) -> Self {
        let g = value.geometry;
        Self {
            turns: value.turns.clone(),
            audio_duration_seconds: value.audio_duration_seconds,
            model_revision: value.model_revision.into(),
            runtime_version: value.runtime_version.clone(),
            backend: value.backend.into(),
            geometry: [g.chunk, g.right, g.left, g.fifo, g.cache, g.update],
            native_compute_micros: value.native_compute_micros,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub status: Status,
    pub session_id: Option<String>,
    pub revision: u64,
    /// Input accepted by this bounded adapter, not a finality watermark.
    pub accepted_audio_seconds: f64,
    pub processed_audio_seconds: f64,
    pub timeline: Option<Timeline>,
    pub reason: Option<String>,
}
impl Snapshot {
    fn active(session_id: String) -> Self {
        Self {
            status: Status::Active,
            session_id: Some(session_id),
            revision: 0,
            accepted_audio_seconds: 0.,
            processed_audio_seconds: 0.,
            timeline: None,
            reason: None,
        }
    }
    pub fn mark_recovered(&mut self) {
        if self.status == Status::Active {
            self.status = Status::Interrupted;
            self.reason = Some("The live speaker cache was interrupted and cannot be resumed. Saved labels remain provisional; transcription can resume.".into());
        }
    }
}
struct Input {
    tx: Option<mpsc::SyncSender<Vec<f32>>>,
    buffer: Vec<f32>,
    received: usize,
}
pub struct LiveSpeakers {
    input: Mutex<Input>,
    state: Arc<Mutex<Snapshot>>,
    cancellation: Cancellation,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}
impl LiveSpeakers {
    fn from_channel(
        tx: mpsc::SyncSender<Vec<f32>>,
        cancellation: Cancellation,
        snapshot: Snapshot,
    ) -> Self {
        Self {
            input: Mutex::new(Input {
                tx: Some(tx),
                buffer: Vec::with_capacity(MAX_PUSH_SAMPLES),
                received: 0,
            }),
            state: Arc::new(Mutex::new(snapshot)),
            cancellation,
            worker: Mutex::new(None),
        }
    }
    pub fn start(job: &Job) -> Arc<Self> {
        let cancellation = job.cancellation.child_token();
        let result = job
            .diarizer
            .as_ref()
            .ok_or_else(|| {
                job.diarization_error
                    .clone()
                    .unwrap_or("Speaker labels are unavailable.".into())
            })
            .and_then(|model| {
                model
                    .start_stream(DiarizeOptions {
                        cancellation: cancellation.clone(),
                        ..Default::default()
                    })
                    .map_err(|e| e.to_string())
            });
        let mut stream = match result {
            Ok(stream) => stream,
            Err(reason) => {
                return Arc::new(Self {
                    input: Mutex::new(Input {
                        tx: None,
                        buffer: Vec::new(),
                        received: 0,
                    }),
                    state: Arc::new(Mutex::new(Snapshot {
                        status: Status::Unavailable,
                        session_id: None,
                        reason: Some(reason),
                        ..Snapshot::active(String::new())
                    })),
                    cancellation,
                    worker: Mutex::new(None),
                });
            }
        };
        // At most four 1.04-second blocks here, plus the Gen2 two-block queue.
        let (tx, rx) = mpsc::sync_channel(4);
        let speakers = Arc::new(Self::from_channel(
            tx,
            cancellation,
            Snapshot::active(stream.session_id().to_string()),
        ));
        let state = speakers.state.clone();
        let handle = thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                loop {
                    match rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(mut samples) => loop {
                            match stream.try_push(samples, 16000) {
                                Ok(()) => break,
                                Err(rejected)
                                    if matches!(rejected.error, DiarizationError::Busy) =>
                                {
                                    samples = rejected.samples;
                                    publish(&stream, &state);
                                    thread::sleep(Duration::from_millis(5));
                                }
                                Err(rejected) => return Err(rejected.error.to_string()),
                            }
                        },
                        Err(mpsc::RecvTimeoutError::Timeout) => (),
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    publish(&stream, &state);
                }
                stream.finish().map_err(|e| e.to_string())?;
                publish(&stream, &state);
                Ok(())
            })();
            if let Err(reason) = result {
                let mut snapshot = state.lock().unwrap();
                if snapshot.status == Status::Active {
                    snapshot.status = Status::Interrupted;
                    snapshot.reason = Some(reason);
                }
            }
        });
        *speakers.worker.lock().unwrap() = Some(handle);
        speakers
    }
    /// Never wait for inference on the microphone writer. Saturation invalidates
    /// labels rather than silently dropping PCM or stopping durable ASR capture.
    pub fn push(&self, samples: &[i16]) {
        let mut input = self.input.lock().unwrap();
        if input.tx.is_none() {
            return;
        }
        if samples.len() > MAX_SESSION_SAMPLES.saturating_sub(input.received) {
            self.fail(
                &mut input,
                "Live speaker labels reached the one-hour limit. Transcription continues.",
            );
            return;
        }
        input.received += samples.len();
        let mut remaining = samples;
        while !remaining.is_empty() {
            let count = remaining.len().min(MAX_PUSH_SAMPLES - input.buffer.len());
            input.buffer.extend(
                remaining[..count]
                    .iter()
                    .map(|sample| *sample as f32 / 32768.),
            );
            remaining = &remaining[count..];
            if input.buffer.len() == MAX_PUSH_SAMPLES && !self.send(&mut input) {
                break;
            }
        }
    }
    fn send(&self, input: &mut Input) -> bool {
        if input.buffer.is_empty() {
            return true;
        }
        let block = std::mem::replace(&mut input.buffer, Vec::with_capacity(MAX_PUSH_SAMPLES));
        let count = block.len();
        if input
            .tx
            .as_ref()
            .is_some_and(|tx| tx.try_send(block).is_ok())
        {
            self.state.lock().unwrap().accepted_audio_seconds += count as f64 / 16000.;
            true
        } else {
            self.fail(input, "Speaker processing could not keep up. Saved labels are incomplete; transcription continues.");
            false
        }
    }
    fn fail(&self, input: &mut Input, reason: &str) {
        input.tx = None;
        input.buffer.clear();
        self.cancellation.cancel();
        let mut snapshot = self.state.lock().unwrap();
        snapshot.status = Status::Interrupted;
        snapshot.reason = Some(reason.into());
    }
    pub fn close_input(&self) {
        let mut input = self.input.lock().unwrap();
        if input.tx.is_some() {
            self.send(&mut input);
        }
        input.tx = None;
    }
    pub fn finish(&self) {
        self.close_input();
        if let Some(worker) = self.worker.lock().unwrap().take()
            && worker.join().is_err()
        {
            let mut input = self.input.lock().unwrap();
            self.fail(
                &mut input,
                "Speaker processing stopped unexpectedly. Saved labels remain incomplete.",
            );
        }
    }
    pub fn snapshot(&self) -> Snapshot {
        self.state.lock().unwrap().clone()
    }
}
impl Drop for LiveSpeakers {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.finish();
    }
}
fn publish(stream: &DiarizationStream, state: &Mutex<Snapshot>) {
    let Some(output) = stream.snapshot() else {
        return;
    };
    let mut snapshot = state.lock().unwrap();
    if snapshot.status != Status::Active || output.revision <= snapshot.revision {
        return;
    }
    snapshot.revision = output.revision;
    snapshot.processed_audio_seconds = output.processed_audio_seconds;
    snapshot.timeline = Some(output.result.as_ref().into());
    if output.complete {
        snapshot.status = Status::Complete;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_full_speaker_queue_preserves_asr_and_marks_labels_incomplete() {
        let parent = Cancellation::default();
        let token = parent.child_token();
        let (tx, _rx) = mpsc::sync_channel(1);
        let speakers = LiveSpeakers::from_channel(tx, token, Snapshot::active("test".into()));
        speakers.push(&vec![0; MAX_PUSH_SAMPLES]);
        speakers.push(&vec![0; MAX_PUSH_SAMPLES]);
        let snapshot = speakers.snapshot();
        assert_eq!(snapshot.status, Status::Interrupted);
        assert!(!parent.is_cancelled());
        assert!(snapshot.reason.unwrap().contains("keep up"));
        assert_eq!(
            snapshot.accepted_audio_seconds,
            MAX_PUSH_SAMPLES as f64 / 16000.
        );
    }

    #[test]
    fn closing_the_feed_preserves_the_short_tail_once() {
        let (tx, rx) = mpsc::sync_channel(1);
        let parent = Cancellation::default();
        let speakers =
            LiveSpeakers::from_channel(tx, parent.child_token(), Snapshot::active("tail".into()));
        speakers.push(&[123; 17]);
        speakers.close_input();
        assert_eq!(rx.recv().unwrap(), vec![123. / 32768.; 17]);
        assert!(rx.recv().is_err());
        speakers.close_input();
        assert_eq!(speakers.snapshot().accepted_audio_seconds, 17. / 16000.);
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn reaching_the_speaker_duration_cap_does_not_cancel_transcription() {
        let (tx, _rx) = mpsc::sync_channel(1);
        let parent = Cancellation::default();
        let speakers =
            LiveSpeakers::from_channel(tx, parent.child_token(), Snapshot::active("cap".into()));
        speakers.input.lock().unwrap().received = MAX_SESSION_SAMPLES;
        speakers.push(&[0]);
        assert_eq!(speakers.snapshot().status, Status::Interrupted);
        assert!(speakers.snapshot().reason.unwrap().contains("one-hour"));
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn an_interrupted_process_cannot_restore_native_speaker_state() {
        let mut snapshot = Snapshot::active("session-one".into());
        snapshot.revision = 7;
        snapshot.mark_recovered();
        assert_eq!(snapshot.status, Status::Interrupted);
        assert_eq!(snapshot.session_id.as_deref(), Some("session-one"));
        assert_eq!(snapshot.revision, 7);
        let mut complete = snapshot.clone();
        complete.status = Status::Complete;
        complete.mark_recovered();
        assert_eq!(complete.status, Status::Complete);
    }
}
