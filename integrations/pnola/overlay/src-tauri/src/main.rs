#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod attribution;
mod audio;
mod capture;
mod engine;
mod live;
mod live_audio;
mod live_speakers;
mod store;
mod transcription;
mod verification_images;
use std::{path::PathBuf, sync::Arc};
use tauri::{Manager, State, ipc::Channel};
use transcription::Transcript;

struct AppState {
    engine: Arc<engine::Engine>,
    capture: Arc<capture::Capture>,
    store: Arc<store::Store>,
    resources: PathBuf,
    live: Arc<live::Live>,
    live_root: PathBuf,
    history: PathBuf,
    session: tempfile::TempDir,
}

#[tauri::command]
fn engine_status(state: State<AppState>) -> engine::Status {
    state
        .engine
        .status(state.resources.join("sample.wav").is_file())
}
#[tauri::command]
async fn file_info(app: tauri::AppHandle, path: String) -> Result<audio::AudioFile, String> {
    let input = path.clone();
    let info =
        tauri::async_runtime::spawn_blocking(move || audio::info(std::path::Path::new(&input)))
            .await
            .map_err(|e| e.to_string())??;
    app.asset_protocol_scope()
        .allow_file(&path)
        .map_err(|e| e.to_string())?;
    Ok(info)
}
#[tauri::command]
async fn sample_file(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<audio::AudioFile, String> {
    file_info(
        app,
        state
            .resources
            .join("sample.wav")
            .to_string_lossy()
            .into_owned(),
    )
    .await
}
#[tauri::command]
fn cancel_transcription(state: State<AppState>, job_id: String) {
    state.engine.cancel(&job_id);
}
#[tauri::command]
async fn transcribe_file(
    state: State<'_, AppState>,
    path: String,
    job_id: String,
    on_event: Channel<serde_json::Value>,
) -> Result<Transcript, String> {
    let job = state.engine.claim(job_id)?;
    tauri::async_runtime::spawn_blocking(move || {
        transcription::run(job, std::path::Path::new(&path), |value| {
            on_event
                .send(value)
                .map_err(|_| "The studio disconnected. Transcription stopped.".to_string())
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn load_history(state: State<AppState>) -> Result<serde_json::Value, String> {
    state.store.read(&state.history)
}
#[tauri::command]
fn save_history(state: State<AppState>, entries: serde_json::Value) -> Result<(), String> {
    state.store.save(&state.history, entries)
}
#[tauri::command]
fn save_transcript(path: String, text: String) -> Result<(), String> {
    if text.len() > 2 * 1024 * 1024 {
        return Err("Transcript is too large.".into());
    }
    std::fs::write(path, text).map_err(|e| e.to_string())
}
#[tauri::command]
async fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    if state.engine.status(false).busy {
        return Err("Wait for the current transcription before recording.".into());
    }
    let path = tempfile::Builder::new()
        .prefix("recording-")
        .suffix(".wav")
        .tempfile_in(state.session.path())
        .map_err(|e| e.to_string())?
        .into_temp_path()
        .keep()
        .map_err(|e| e.to_string())?;
    let capture = state.capture.clone();
    tauri::async_runtime::spawn_blocking(move || capture.start(path))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
fn recording_state(state: State<AppState>) -> capture::CaptureState {
    state.capture.state()
}
#[tauri::command]
async fn stop_recording(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<audio::AudioFile, String> {
    let capture = state.capture.clone();
    let path = tauri::async_runtime::spawn_blocking(move || capture.stop())
        .await
        .map_err(|e| e.to_string())??;
    let mut file = file_info(app, path.to_string_lossy().into_owned()).await?;
    file.duration = hound::WavReader::open(path)
        .ok()
        .map(|r| r.duration() as f64 / r.spec().sample_rate as f64);
    Ok(file)
}

#[tauri::command]
async fn start_live(
    state: State<'_, AppState>,
    id: String,
    name: String,
    date: String,
) -> Result<live::Snapshot, String> {
    let live = state.live.clone();
    let root = state.live_root.clone();
    let engine = state.engine.clone();
    let capture = state.capture.clone();
    tauri::async_runtime::spawn_blocking(move || {
        live.start(&root, &engine, capture, id, name, date)
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn live_state(state: State<AppState>) -> Option<live::Snapshot> {
    state.live.snapshot()
}
#[tauri::command]
async fn stop_live(state: State<'_, AppState>) -> Result<live::Snapshot, String> {
    let live = state.live.clone();
    tauri::async_runtime::spawn_blocking(move || live.stop())
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn resume_live(state: State<'_, AppState>, id: String) -> Result<live::Snapshot, String> {
    let live = state.live.clone();
    let root = state.live_root.clone();
    let engine = state.engine.clone();
    let capture = state.capture.clone();
    tauri::async_runtime::spawn_blocking(move || live.resume(&root, &engine, capture, &id))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn list_live(state: State<'_, AppState>) -> Result<Vec<live::Snapshot>, String> {
    let root = state.live_root.clone();
    let live = state.live.clone();
    tauri::async_runtime::spawn_blocking(move || live.list(&root))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn live_text(state: State<'_, AppState>, id: String) -> Result<String, String> {
    let root = state.live_root.clone();
    tauri::async_runtime::spawn_blocking(move || live::Live::text(&root, &id))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn export_live(state: State<'_, AppState>, id: String, path: String) -> Result<(), String> {
    let root = state.live_root.clone();
    tauri::async_runtime::spawn_blocking(move || {
        live::Live::export(&root, &id, std::path::Path::new(&path))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_live(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let root = state.live_root.clone();
    let live = state.live.clone();
    tauri::async_runtime::spawn_blocking(move || live.remove(&root, &id))
        .await
        .map_err(|e| e.to_string())?
}

fn main() {
    if matches!(
        std::env::args().nth(1).as_deref(),
        Some("--verify-bundle" | "--verify-live" | "--verify-live-speakers")
    ) {
        verification_images::start();
    }
    if matches!(
        std::env::args().nth(1).as_deref(),
        Some("--verify-live" | "--verify-live-speakers")
    ) {
        let speakers = std::env::args().nth(1).as_deref() == Some("--verify-live-speakers");
        if let Err(error) = live::verify(speakers) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--verify-bundle") {
        if let Err(error) = transcription::verify_bundle() {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init());
    #[cfg(debug_assertions)]
    let builder = builder.plugin(tauri_plugin_mcp_bridge::init_with_config(
        tauri_plugin_mcp_bridge::Config {
            bind_address: "127.0.0.1".into(),
            base_port: 9337,
        },
    ));
    let app = builder
        .invoke_handler(tauri::generate_handler![
            engine_status,
            file_info,
            sample_file,
            transcribe_file,
            cancel_transcription,
            load_history,
            save_history,
            save_transcript,
            start_recording,
            stop_recording,
            recording_state,
            start_live, live_state, stop_live, resume_live, list_live, live_text, export_live, delete_live
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            app.add_capability(r#"{"identifier":"pnola-local-debug","windows":["main"],"permissions":["mcp-bridge:default"]}"#)?;
            let resources = audio::resource_root(app.handle()).map_err(std::io::Error::other)?;
            let data = app.path().app_data_dir()?;
            #[cfg(debug_assertions)]
            let data=std::env::var_os("PNOLA_QA_DATA_DIR").map(PathBuf::from).unwrap_or(data);
            std::fs::create_dir_all(&data)?;
            let cache = app.path().app_cache_dir()?;
            std::fs::create_dir_all(&cache)?;
            let engine = engine::Engine::new();
            app.manage(AppState {
                engine: engine.clone(),
                capture: Arc::new(capture::Capture::default()),
                store: Arc::new(store::Store::default()),
                resources: resources.clone(),
                live: Arc::new(live::Live::default()),
                live_root: data.join("live"),
                history: data.join("transcripts.json"),
                session: tempfile::tempdir_in(cache)?,
            });
            std::thread::spawn(move || engine.load(&resources));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Could not start Pnola");
    app.run(|app, event| {
        if matches!(
            event,
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
        ) {
            let state = app.state::<AppState>();
            state.engine.shutdown();
            state.capture.shutdown();
        }
    });
}
