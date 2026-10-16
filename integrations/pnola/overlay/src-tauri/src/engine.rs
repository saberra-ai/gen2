use gen2::{
    Runtime,
    diarization::Diarizer,
    speech::{Cancellation, LoadOptions, Transcriber},
};
use serde::Serialize;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

#[derive(Serialize)]
pub struct Status {
    pub status: String,
    pub detail: String,
    pub busy: bool,
    pub sample: bool,
    pub transcription_available: bool,
    pub speakers_available: bool,
}
struct Models {
    _runtime: Runtime,
    speech: Option<Transcriber>,
    diarizer: Option<Diarizer>,
    speech_error: Option<String>,
    diarization_error: Option<String>,
}
pub struct Engine {
    model: Mutex<Option<Models>>,
    message: Mutex<String>,
    active: Mutex<Option<(String, Cancellation)>>,
}
impl Engine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            model: Mutex::new(None),
            message: Mutex::new("Loading local speech models…".into()),
            active: Mutex::new(None),
        })
    }
    pub fn load(&self, resources: &Path) {
        let result = (|| {
            let runtime = Runtime::builder().build().map_err(|e| e.to_string())?;
            let name = if cfg!(target_os = "macos") {
                "libonnxruntime.dylib"
            } else if cfg!(target_os = "windows") {
                "onnxruntime.dll"
            } else {
                "libonnxruntime.so"
            };
            let framework = resources
                .parent()
                .unwrap_or(resources)
                .join("Frameworks")
                .join(name);
            let native_library = std::env::var_os("GEN2_ORT_LIBRARY")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    if cfg!(target_os = "macos") && framework.is_file() {
                        framework
                    } else {
                        resources.join("runtime").join(name)
                    }
                });
            let bark = std::env::var_os("GEN2_BARK_BUNDLE")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| resources.join("bark"));
            let speech = runtime.load_transcriber(
                bark,
                LoadOptions {
                    native_library: Some(native_library),
                    queue_capacity: 1,
                    ..Default::default()
                },
            );
            let (speech, speech_error) = match speech {
                Ok(model) => (Some(model), None),
                Err(e) => (None, Some(e.to_string())),
            };
            let nemo_name = if cfg!(target_os = "macos") {
                "libnemo_speech_asr_c.dylib"
            } else if cfg!(target_os = "windows") {
                "nemo_speech_asr_c.dll"
            } else {
                "libnemo_speech_asr_c.so"
            };
            let nemo_library = std::env::var_os("GEN2_NEMO_LIBRARY")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    let framework = resources
                        .parent()
                        .unwrap_or(resources)
                        .join("Frameworks")
                        .join(nemo_name);
                    if cfg!(target_os = "macos") && framework.is_file() {
                        framework
                    } else {
                        resources.join("nemo/lib").join(nemo_name)
                    }
                });
            let nemo_model = std::env::var_os("GEN2_NEMOTRON_ASSETS")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| resources.join("nemotron"));
            let diarizer = runtime.load_diarizer(
                nemo_model,
                gen2::diarization::LoadOptions::cpu(nemo_library),
            );
            let (diarizer, diarization_error) = match diarizer {
                Ok(model) => (Some(model), None),
                Err(e) => (None, Some(e.to_string())),
            };
            Ok::<_, String>(Models {
                _runtime: runtime,
                speech,
                diarizer,
                speech_error,
                diarization_error,
            })
        })();
        match result {
            Ok(model) => {
                *self.message.lock().unwrap() =
                    match (model.speech.is_some(), model.diarizer.is_some()) {
                        (true, true) => "Bark and speaker labels ready · running locally".into(),
                        (true, false) => "Bark ready · speaker labels unavailable".into(),
                        (false, true) => {
                            "Speaker timeline ready · transcription unavailable".into()
                        }
                        (false, false) => format!(
                            "Local models unavailable: {}",
                            model
                                .speech_error
                                .as_deref()
                                .unwrap_or("could not load speech model")
                        ),
                    };
                *self.model.lock().unwrap() = Some(model);
            }
            Err(error) => *self.message.lock().unwrap() = error,
        }
    }
    pub fn status(&self, sample: bool) -> Status {
        let model = self.model.lock().unwrap();
        let transcription_available = model.as_ref().is_some_and(|m| m.speech.is_some());
        let speakers_available = model.as_ref().is_some_and(|m| m.diarizer.is_some());
        let detail = self.message.lock().unwrap().clone();
        Status {
            status: if transcription_available || speakers_available {
                "ready"
            } else if detail.starts_with("Loading") {
                "loading"
            } else {
                "unavailable"
            }
            .into(),
            detail,
            busy: self.active.lock().unwrap().is_some(),
            sample,
            transcription_available,
            speakers_available,
        }
    }
    pub fn claim(self: &Arc<Self>, id: String) -> Result<Job, String> {
        let model = self.model.lock().unwrap();
        let model = model
            .as_ref()
            .ok_or("Local speech models are not ready yet.")?;
        if model.speech.is_none() && model.diarizer.is_none() {
            return Err("Local speech models are unavailable.".into());
        }
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err("Another transcription is already running.".into());
        }
        let cancellation = Cancellation::default();
        *active = Some((id.clone(), cancellation.clone()));
        Ok(Job {
            engine: self.clone(),
            id,
            cancellation,
            speech: model.speech.clone(),
            diarizer: model.diarizer.clone(),
            speech_error: model.speech_error.clone(),
            diarization_error: model.diarization_error.clone(),
        })
    }
    pub fn cancel(&self, id: &str) {
        if let Some((active_id, token)) = self.active.lock().unwrap().as_ref()
            && id == active_id
        {
            token.cancel();
        }
    }
    pub fn shutdown(&self) {
        if let Some((_, token)) = self.active.lock().unwrap().as_ref() {
            token.cancel();
        }
        if let Some(model) = self.model.lock().unwrap().as_ref() {
            if let Some(speech) = &model.speech {
                speech.unload();
            }
            if let Some(diarizer) = &model.diarizer {
                diarizer.unload();
            }
        }
    }
}
pub struct Job {
    engine: Arc<Engine>,
    id: String,
    pub cancellation: Cancellation,
    pub speech: Option<Transcriber>,
    pub diarizer: Option<Diarizer>,
    pub speech_error: Option<String>,
    pub diarization_error: Option<String>,
}
impl Job {
    pub fn require_speech(&self) -> Result<&Transcriber, String> {
        self.speech.as_ref().ok_or_else(|| {
            self.speech_error
                .clone()
                .unwrap_or("Bark is unavailable.".into())
        })
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        let mut active = self.engine.active.lock().unwrap();
        if active.as_ref().is_some_and(|(id, _)| id == &self.id) {
            *active = None;
        }
    }
}
