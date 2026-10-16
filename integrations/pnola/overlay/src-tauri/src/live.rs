//! Persistent, sectioned live transcription. PCM and transcript sections are
//! bounded; elapsed session time and total transcript size have no fixed cap.
use crate::{
    capture::Capture,
    engine::{Engine, Job},
};
use gen2::speech::{Audio, TranscribeOptions};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub id: String,
    pub name: String,
    pub date: String,
    pub phase: String,
    pub text: String,
    pub duration: f64,
    pub execution: u64,
    pub words: u64,
    pub completed: u64,
    pub queued: u64,
    pub seconds: f64,
    pub level: f32,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speakers: Option<crate::live_speakers::Snapshot>,
}
#[derive(Serialize, Deserialize)]
struct Section {
    text: String,
    duration: f64,
    execution: u64,
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary).map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, value).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    fs::rename(temporary, path).map_err(|e| e.to_string())
}
fn location(root: &Path, id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("Invalid live session ID.".into());
    }
    Ok(root.join(id))
}
fn rebuild(directory: &Path) -> Result<Snapshot, String> {
    let mut snapshot: Snapshot = serde_json::from_reader(
        fs::File::open(directory.join("session.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if let Some(speakers) = &mut snapshot.speakers {
        speakers.mark_recovered();
    }
    snapshot.completed = 0;
    snapshot.duration = 0.;
    snapshot.execution = 0;
    snapshot.words = 0;
    let mut recent = VecDeque::new();
    loop {
        let path = directory.join(format!("{:020}.json", snapshot.completed));
        if !path.exists() {
            break;
        }
        let section: Section =
            serde_json::from_reader(fs::File::open(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        snapshot.duration += section.duration;
        snapshot.execution += section.execution;
        snapshot.words += section.text.split_whitespace().count() as u64;
        recent.push_back(section.text);
        if recent.len() > 30 {
            recent.pop_front();
        }
        snapshot.completed += 1;
    }
    snapshot.text = recent
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    snapshot.queued = fs::read_dir(directory)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().is_some_and(|e| e == "wav")
                && entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .is_some_and(|n| n >= snapshot.completed)
        })
        .count() as u64;
    snapshot.seconds = snapshot.duration;
    snapshot.phase = if snapshot.queued > 0 {
        "interrupted"
    } else {
        "complete"
    }
    .into();
    snapshot.level = 0.;
    Ok(snapshot)
}

pub struct Session {
    directory: PathBuf,
    state: Mutex<Snapshot>,
    capture: Arc<Capture>,
    capture_done: AtomicBool,
    published: AtomicU64,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    speakers: Option<Arc<crate::live_speakers::LiveSpeakers>>,
}
impl Session {
    fn snapshot(&self) -> Snapshot {
        let mut snapshot = self.state.lock().unwrap().clone();
        if let Some(speakers) = &self.speakers {
            snapshot.speakers = Some(speakers.snapshot());
        }
        let capture = self.capture.state();
        let published = self.published.load(Ordering::Acquire).max(
            if self.capture_done.load(Ordering::Acquire) {
                0
            } else {
                capture.sections
            },
        );
        snapshot.queued = published.saturating_sub(snapshot.completed);
        if snapshot.phase == "recording" {
            snapshot.seconds = capture.seconds;
            snapshot.level = capture.level;
            if capture.phase == "error" || capture.phase == "stopped" {
                snapshot.phase = "stopping".into();
                snapshot.error = capture.error;
            }
        } else {
            snapshot.seconds = snapshot.duration;
        }
        snapshot
    }
    fn spawn(self: &Arc<Self>, job: Job) {
        let session = self.clone();
        let handle = thread::spawn(move || {
            let result = session.run(&job);
            // Stop and flush capture before finalizing its speaker feed, also
            // when ASR fails. A valid standalone timeline survives that failure.
            session.capture.request_stop();
            if !session.capture_done.load(Ordering::Acquire) {
                if let Ok(final_state) = session.capture.stop_live() {
                    session
                        .published
                        .store(final_state.sections, Ordering::Release);
                }
                session.capture_done.store(true, Ordering::Release);
            }
            if let Some(speakers) = &session.speakers {
                speakers.finish();
            }
            if let Err(error) = result {
                session.capture.request_stop();
                let mut state = session.state.lock().unwrap();
                state.phase = "error".into();
                state.error = Some(error);
            } else {
                let mut state = session.state.lock().unwrap();
                state.phase = if state.error.is_some() {
                    "error"
                } else {
                    "complete"
                }
                .into();
            }
            let state = session.snapshot();
            if let Err(error) = atomic_json(&session.directory.join("session.json"), &state) {
                let mut state = session.state.lock().unwrap();
                state.phase = "error".into();
                state.error = Some(format!(
                    "Could not save session metadata: {error}. Section files remain on disk."
                ));
            }
        });
        *self.worker.lock().unwrap() = Some(handle);
    }
    fn run(&self, job: &Job) -> Result<(), String> {
        loop {
            if job.cancellation.is_cancelled() {
                return Err("Live transcription interrupted. Saved audio can be resumed.".into());
            }
            let sequence = self.state.lock().unwrap().completed;
            let path = self.directory.join(format!("{sequence:020}.wav"));
            if !path.is_file() {
                if self.capture_done.load(Ordering::Acquire) {
                    return Ok(());
                }
                let capture = self.capture.state();
                if (capture.phase == "error" || capture.phase == "stopped")
                    && let Ok(final_state) = self.capture.stop_live()
                {
                    self.published
                        .store(final_state.sections, Ordering::Release);
                    if final_state.error.is_some() {
                        self.state.lock().unwrap().error = final_state.error;
                    }
                    self.capture_done.store(true, Ordering::Release);
                }
                thread::sleep(Duration::from_millis(30));
                continue;
            }
            let mut reader = hound::WavReader::open(&path).map_err(|e| e.to_string())?;
            let frames = reader.duration();
            let mut samples = reader
                .samples::<i16>()
                .map(|v| v.map(|s| s as f32 / 32768.))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            if samples.len() > crate::live_audio::SECTION_FRAMES {
                return Err("Live audio section exceeds its buffer limit.".into());
            }
            samples.resize(samples.len().max(320), 0.);
            let result = job
                .require_speech()?
                .transcribe(
                    Audio::mono_16khz(samples).map_err(|e| e.to_string())?,
                    TranscribeOptions {
                        cancellation: job.cancellation.clone(),
                        deadline: Some(Instant::now() + Duration::from_secs(120)),
                    },
                )
                .map_err(|e| e.to_string())?;
            let section = Section {
                text: serde_json::to_value(&result)
                    .map_err(|e| e.to_string())?
                    .get("fulltext")
                    .and_then(|value| value.as_str())
                    .ok_or("Bark returned no transcript text.")?
                    .trim()
                    .to_owned(),
                duration: frames as f64 / 16000.,
                execution: result.execution_micros,
            };
            // Save the result before deleting its source. A crash between these
            // operations is recovered from the immutable result, without repeats.
            atomic_json(
                &self.directory.join(format!("{sequence:020}.json")),
                &section,
            )?;
            {
                let mut state = self.state.lock().unwrap();
                let mut recent: VecDeque<String> = if state.text.is_empty() {
                    VecDeque::new()
                } else {
                    state.text.split("\n\n").map(str::to_owned).collect()
                };
                recent.push_back(section.text.clone());
                while recent.len() > 30 {
                    recent.pop_front();
                }
                state.text = recent
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                state.completed += 1;
                state.duration += section.duration;
                state.execution += section.execution;
                state.words += section.text.split_whitespace().count() as u64;
                if let Some(speakers) = &self.speakers {
                    state.speakers = Some(speakers.snapshot());
                }
                atomic_json(&self.directory.join("session.json"), &*state)?;
            }
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
    }
    fn finish(&self) -> Result<Snapshot, String> {
        {
            let mut state = self.state.lock().unwrap();
            if state.phase == "recording" {
                state.phase = "stopping".into();
            }
        }
        if !self.capture_done.load(Ordering::Acquire) {
            if let Ok(final_state) = self.capture.stop_live() {
                self.published
                    .store(final_state.sections, Ordering::Release);
                if let Some(error) = final_state.error {
                    self.state.lock().unwrap().error = Some(error);
                }
            }
            self.capture_done.store(true, Ordering::Release);
        }
        if let Some(handle) = self.worker.lock().unwrap().take() {
            handle
                .join()
                .map_err(|_| "Live worker stopped unexpectedly.".to_string())?;
        }
        Ok(self.snapshot())
    }
}

#[derive(Default)]
pub struct Live {
    current: Mutex<Option<Arc<Session>>>,
}
impl Live {
    pub fn start(
        &self,
        root: &Path,
        engine: &Arc<Engine>,
        capture: Arc<Capture>,
        id: String,
        name: String,
        date: String,
    ) -> Result<Snapshot, String> {
        let mut current = self.current.lock().unwrap();
        if current
            .as_ref()
            .is_some_and(|s| matches!(s.snapshot().phase.as_str(), "recording" | "stopping"))
        {
            return Err("A live session is already active.".into());
        }
        capture.shutdown();
        let job = engine.claim(id.clone())?;
        job.require_speech()?;
        let directory = location(root, &id)?;
        fs::create_dir_all(root).map_err(|e| e.to_string())?;
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let speakers = crate::live_speakers::LiveSpeakers::start(&job);
        let snapshot = Snapshot {
            id,
            name,
            date,
            phase: "recording".into(),
            text: String::new(),
            duration: 0.,
            execution: 0,
            words: 0,
            completed: 0,
            queued: 0,
            seconds: 0.,
            level: 0.,
            error: None,
            speakers: Some(speakers.snapshot()),
        };
        atomic_json(&directory.join("session.json"), &snapshot)?;
        capture.start_live(directory.clone(), speakers.clone())?;
        let session = Arc::new(Session {
            directory,
            state: Mutex::new(snapshot),
            capture,
            capture_done: AtomicBool::new(false),
            published: AtomicU64::new(0),
            worker: Mutex::new(None),
            speakers: Some(speakers),
        });
        session.spawn(job);
        let snapshot = session.snapshot();
        *current = Some(session);
        Ok(snapshot)
    }
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.current.lock().unwrap().as_ref().map(|s| s.snapshot())
    }
    pub fn stop(&self) -> Result<Snapshot, String> {
        let session = self
            .current
            .lock()
            .unwrap()
            .clone()
            .ok_or("No live session is active.")?;
        session.finish()
    }
    pub fn resume(
        &self,
        root: &Path,
        engine: &Arc<Engine>,
        capture: Arc<Capture>,
        id: &str,
    ) -> Result<Snapshot, String> {
        let mut current = self.current.lock().unwrap();
        if current
            .as_ref()
            .is_some_and(|s| matches!(s.snapshot().phase.as_str(), "recording" | "stopping"))
        {
            return Err("Stop the current live session first.".into());
        }
        capture.shutdown();
        let job = engine.claim(id.into())?;
        job.require_speech()?;
        let directory = location(root, id)?;
        let mut snapshot = rebuild(&directory)?;
        for sequence in 0..snapshot.completed {
            let path = directory.join(format!("{sequence:020}.wav"));
            if path.exists() {
                fs::remove_file(path).map_err(|e| e.to_string())?;
            }
        }
        let published = snapshot.completed + snapshot.queued;
        snapshot.phase = "stopping".into();
        snapshot.error = None;
        let session = Arc::new(Session {
            directory,
            state: Mutex::new(snapshot),
            capture,
            capture_done: AtomicBool::new(true),
            published: AtomicU64::new(published),
            worker: Mutex::new(None),
            speakers: None,
        });
        session.spawn(job);
        let snapshot = session.snapshot();
        *current = Some(session);
        Ok(snapshot)
    }
    pub fn list(root: &Path) -> Result<Vec<Snapshot>, String> {
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = fs::read_dir(root)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .filter(|e| e.path().join("session.json").is_file())
            .map(|e| rebuild(&e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        sessions.sort_by(|a, b| b.date.cmp(&a.date));
        sessions.truncate(20);
        Ok(sessions)
    }
    pub fn remove(&self, root: &Path, id: &str) -> Result<(), String> {
        if self
            .current
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|session| {
                let value = session.snapshot();
                value.id == id && matches!(value.phase.as_str(), "recording" | "stopping")
            })
        {
            return Err("Stop the live session before deleting it.".into());
        }
        fs::remove_dir_all(location(root, id)?).map_err(|e| e.to_string())
    }
    pub fn export(root: &Path, id: &str, output: &Path) -> Result<(), String> {
        let directory = location(root, id)?;
        let snapshot = rebuild(&directory)?;
        let mut output =
            std::io::BufWriter::new(fs::File::create(output).map_err(|e| e.to_string())?);
        for sequence in 0..snapshot.completed {
            let section: Section = serde_json::from_reader(
                fs::File::open(directory.join(format!("{sequence:020}.json")))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if sequence > 0 {
                output.write_all(b"\n\n").map_err(|e| e.to_string())?;
            }
            output
                .write_all(section.text.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        output.flush().map_err(|e| e.to_string())
    }
    pub fn text(root: &Path, id: &str) -> Result<String, String> {
        let output = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
        Self::export(root, id, output.path())?;
        fs::read_to_string(output.path()).map_err(|e| e.to_string())
    }
}

/// Exercise the native section writer, resident Bark worker, durable results,
/// final flush and complete export using real speech without opening a mic.
pub fn verify(with_speakers: bool) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let packaged = executable
        .parent()
        .and_then(Path::parent)
        .ok_or("Missing resources")?
        .join("Resources");
    let resources = if packaged.join("bark").is_dir() {
        packaged
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../.desktop-assets")
    };
    let input = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| resources.join("sample.wav"));
    let mut reader = hound::WavReader::open(input).map_err(|e| e.to_string())?;
    if reader.spec().sample_rate != 16000 || reader.spec().channels != 1 {
        return Err("Verification fixture must be mono 16 kHz.".into());
    }
    let sample = if reader.spec().sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .map(|value| value.map(|v| (v.clamp(-1., 1.) * 32767.) as i16))
            .collect::<Result<Vec<_>, _>>()
    } else {
        reader.samples::<i16>().collect::<Result<Vec<_>, _>>()
    }
    .map_err(|e| e.to_string())?;
    if sample.is_empty() {
        return Err("Empty verification fixture".into());
    }
    let temporary = tempfile::tempdir().map_err(|e| e.to_string())?;
    let directory = temporary.path().join("native-live-verification");
    let mut writer = crate::live_audio::Sections::new(&directory)?;
    let engine = Engine::new();
    engine.load(&resources);
    let job = engine
        .claim("native-live-verification".into())
        .map_err(|_| engine.status(false).detail)?;
    let speakers = with_speakers.then(|| crate::live_speakers::LiveSpeakers::start(&job));
    if speakers
        .as_ref()
        .is_some_and(|s| s.snapshot().status != crate::live_speakers::Status::Active)
    {
        return Err("Live speaker qualification requires a real loaded Nemotron model.".into());
    }
    let reference_model = if with_speakers {
        job.diarizer.clone()
    } else {
        None
    };
    let snapshot = Snapshot {
        id: "native-live-verification".into(),
        name: "Native live verification".into(),
        date: "verification".into(),
        phase: "stopping".into(),
        text: String::new(),
        duration: 0.,
        execution: 0,
        words: 0,
        completed: 0,
        queued: 11,
        seconds: 0.,
        level: 0.,
        error: None,
        speakers: speakers.as_ref().map(|s| s.snapshot()),
    };
    atomic_json(&directory.join("session.json"), &snapshot)?;
    let session = Session {
        directory,
        state: Mutex::new(snapshot),
        capture: Arc::new(Capture::default()),
        capture_done: AtomicBool::new(false),
        published: AtomicU64::new(0),
        worker: Mutex::new(None),
        speakers: speakers.clone(),
    };
    let session = Arc::new(session);
    session.spawn(job);
    let total = 80 * 16000 + 640;
    let mut written = 0;
    let started = Instant::now();
    let mut revisions = Vec::new();
    let mut last_revision = 0;
    while written < total {
        let offset = written % sample.len();
        let count = (sample.len() - offset)
            .min(total - written)
            .min(if with_speakers { 320 } else { sample.len() });
        let block = &sample[offset..offset + count];
        writer.push(block)?;
        if let Some(speakers) = &speakers {
            speakers.push(block);
            let snapshot = speakers.snapshot();
            if snapshot.status != crate::live_speakers::Status::Active {
                return Err(format!(
                    "Live speaker processing failed: {:?}",
                    snapshot.reason
                ));
            }
            if snapshot.revision > last_revision {
                last_revision = snapshot.revision;
                revisions.push(serde_json::json!({"revision":snapshot.revision,"elapsed_seconds":started.elapsed().as_secs_f64(),"processed_audio_seconds":snapshot.processed_audio_seconds}));
            }
        }
        written += count;
        session.published.store(
            writer.frames / crate::live_audio::SECTION_FRAMES as u64,
            Ordering::Release,
        );
        if with_speakers
            && let Some(wait) =
                Duration::from_secs_f64(written as f64 / 16000.).checked_sub(started.elapsed())
        {
            thread::sleep(wait);
        }
    }
    writer.flush()?;
    if let Some(speakers) = &speakers {
        speakers.close_input();
    }
    session.published.store(11, Ordering::Release);
    session.capture_done.store(true, Ordering::Release);
    let finished = session.finish()?;
    if let Some(error) = finished.error {
        return Err(error);
    }
    let recovered = rebuild(&session.directory)?;
    let export = temporary.path().join("complete.txt");
    Live::export(temporary.path(), &recovered.id, &export)?;
    let text = fs::read_to_string(export).map_err(|e| e.to_string())?;
    if recovered.completed != 11
        || (recovered.duration - 80.04).abs() > 0.001
        || text.is_empty()
        || recovered.queued != 0
    {
        return Err("Native live verification did not cover the complete session.".into());
    }
    if let Some(model) = reference_model {
        let result = model
            .diarize(
                gen2::diarization::Audio::mono_16khz(
                    (0..total)
                        .map(|index| sample[index % sample.len()] as f32 / 32768.)
                        .collect(),
                )
                .map_err(|e| e.to_string())?,
                Default::default(),
            )
            .map_err(|e| e.to_string())?;
        let actual = recovered
            .speakers
            .as_ref()
            .ok_or("Missing saved speaker state")?;
        if actual.status != crate::live_speakers::Status::Complete
            || revisions.is_empty()
            || actual
                .timeline
                .as_ref()
                .is_none_or(|timeline| timeline.turns != result.turns)
            || (actual.processed_audio_seconds - 80.04).abs() > 1e-6
        {
            let failure = serde_json::json!({"passed":false,"scope":"live speaker parity diagnostics","actual":actual,"reference":result,"provisional_revisions":revisions});
            if let Some(path) = std::env::args().nth(3) {
                fs::write(path, serde_json::to_vec_pretty(&failure).unwrap())
                    .map_err(|e| e.to_string())?;
            }
            return Err(format!(
                "Live speakers failed final parity, revisions, or tail coverage: status={:?}, revisions={}, processed={}.",
                actual.status,
                revisions.len(),
                actual.processed_audio_seconds
            ));
        }
    }
    engine.shutdown();
    let receipt = serde_json::json!({"passed":true,"duration_seconds":recovered.duration,"sections":recovered.completed,"words":recovered.words,"queued":recovered.queued,"execution_micros":recovered.execution,"export_bytes":text.len(),"microphone_used":false,"speaker_qualification":with_speakers,"speakers":recovered.speakers,"provisional_revisions":revisions});
    if let Some(path) = std::env::args().nth(3) {
        fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
    }
    println!("{receipt}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovery_and_export_cross_two_hours_without_unbounded_preview() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("test-session");
        fs::create_dir(&directory).unwrap();
        let snapshot = Snapshot {
            id: "test-session".into(),
            name: "Test".into(),
            date: "2026-10-15".into(),
            phase: "recording".into(),
            text: String::new(),
            duration: 0.,
            execution: 0,
            words: 0,
            completed: 0,
            queued: 0,
            seconds: 0.,
            level: 0.,
            error: None,
            speakers: None,
        };
        atomic_json(&directory.join("session.json"), &snapshot).unwrap();
        for index in 0..901 {
            atomic_json(
                &directory.join(format!("{index:020}.json")),
                &Section {
                    text: format!("section-{index}"),
                    duration: 8.,
                    execution: 1,
                },
            )
            .unwrap();
        }
        let recovered = rebuild(&directory).unwrap();
        assert_eq!(recovered.duration, 7208.);
        assert_eq!(recovered.completed, 901);
        assert_eq!(recovered.text.split("\n\n").count(), 30);
        let output = root.path().join("export.txt");
        Live::export(root.path(), "test-session", &output).unwrap();
        let text = fs::read_to_string(output).unwrap();
        assert_eq!(text.split("\n\n").count(), 901);
        assert!(text.ends_with("section-900"));
        assert!(location(root.path(), "../escape").is_err());
    }
}
