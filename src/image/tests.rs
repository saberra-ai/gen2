use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use super::*;
use crate::{Runtime, test_support::Script};

fn plan(total_mb: u64) -> ImageMemoryPlan {
    ImageMemoryPlan {
        weights_mb: total_mb / 2,
        working_mb: total_mb - total_mb / 2,
        total_mb,
        max_pixels: 1024 * 1024,
        max_prompt_bytes: 4096,
    }
}

#[derive(Default)]
struct State {
    loads: AtomicUsize,
    drops: AtomicUsize,
    requests: Mutex<Vec<(String, u32, u32, i64)>>,
    started: Mutex<Option<mpsc::SyncSender<()>>>,
    released: Mutex<bool>,
    wake: Condvar,
    block: bool,
    fail: bool,
}

struct FakeDriver(Arc<State>);
impl Driver for FakeDriver {
    fn generate(
        &mut self,
        request: &job::Request,
        cancel: &ImageCanceller,
        progress: &mut dyn FnMut(u32, u32) -> bool,
    ) -> NativeResult<GeneratedImage> {
        self.0.requests.lock().unwrap().push((
            request.prompt.clone(),
            request.width,
            request.height,
            request.seed,
        ));
        if let Some(sender) = self.0.started.lock().unwrap().take() {
            sender.send(()).unwrap();
        }
        if self.0.block {
            let mut released = self.0.released.lock().unwrap();
            while !*released {
                released = self.0.wake.wait(released).unwrap();
            }
        }
        if cancel.is_cancelled() || !progress(1, 8) {
            return Err(ImageError::Cancelled);
        }
        if self.0.fail {
            return Err(ImageError::Execution("injected native failure".into()));
        }
        progress(8, 8);
        GeneratedImage::from_pixels(
            request.width,
            request.height,
            4,
            vec![123; request.width as usize * request.height as usize * 4],
        )
    }
}
impl Drop for FakeDriver {
    fn drop(&mut self) {
        self.0.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn factory(state: &Arc<State>) -> Factory {
    let state = state.clone();
    Arc::new(move || {
        state.loads.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeDriver(state.clone())))
    })
}
fn release(state: &State) {
    *state.released.lock().unwrap() = true;
    state.wake.notify_all();
}
fn blocking() -> (Arc<State>, mpsc::Receiver<()>) {
    let (tx, rx) = mpsc::sync_channel(1);
    (
        Arc::new(State {
            started: Mutex::new(Some(tx)),
            block: true,
            ..State::default()
        }),
        rx,
    )
}
fn wait_idle(model: &ImageModel) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while model.is_busy() {
        assert!(
            Instant::now() < deadline,
            "worker did not release busy state"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn chat_and_image_models_share_lru_admission_and_restore_the_same_handles() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1000)
        .build()
        .unwrap();
    let state = Arc::new(State::default());
    let image = runtime
        .scripted_image_in(plan(600), factory(&state))
        .unwrap();
    let chat = runtime.scripted_in(Script::new().say(["ok"]), 500);
    assert!(
        !image.is_resident(),
        "loading chat evicts the older image context"
    );
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.models(), [image.id(), chat.id()]);
    let result = image
        .generate("a teapot")
        .size(32, 32)
        .seed(7)
        .run()
        .unwrap();
    assert_eq!(result.image.width(), 32);
    assert_eq!(result.seed, 7);
    assert!(image.is_resident());
    assert!(!runtime.residency().is_resident(chat.id()));
    assert_eq!(state.loads.load(Ordering::SeqCst), 2);
    assert_eq!(runtime.stats().image_reserved_mb, 600);
    assert_eq!(runtime.stats().evictions, 2);
    assert_eq!(chat.generate("answer").text().unwrap(), "ok");
    assert!(!image.is_resident());
    assert_eq!(runtime.stats().image_reserved_mb, 0);
    assert_eq!(runtime.stats().evictions, 3);
}

#[test]
fn refusal_happens_before_allocating_or_registering_an_image_context() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(100)
        .build()
        .unwrap();
    let state = Arc::new(State::default());
    let error = runtime
        .scripted_image_in(plan(101), factory(&state))
        .unwrap_err();
    assert_eq!(error.code(), Some("image_resource_limit"));
    assert_eq!(state.loads.load(Ordering::SeqCst), 0);
    assert!(runtime.models().is_empty());
    assert_eq!(runtime.stats().image_reserved_mb, 0);
}

#[test]
fn busy_requests_stay_reserved_and_cannot_be_evicted_until_native_returns() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1000)
        .build()
        .unwrap();
    let (state, started) = blocking();
    let model = runtime
        .scripted_image_in(plan(600), factory(&state))
        .unwrap();
    let stream = model.generate("a teapot").size(32, 32).stream().unwrap();
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    stream.canceller().cancel();
    assert_eq!(
        runtime.reserve_working_memory(401).err().unwrap().code(),
        Some("resource_limit")
    );
    assert_eq!(runtime.stats().active_image_jobs, 1);
    assert_eq!(runtime.stats().image_reserved_mb, 600);
    assert_eq!(
        runtime.evict_image_model(&model).unwrap_err().code(),
        Some("image_busy")
    );
    assert_eq!(
        model
            .generate("second")
            .size(32, 32)
            .stream()
            .err()
            .unwrap()
            .code(),
        Some("image_busy")
    );
    let other = Arc::new(State::default());
    assert_eq!(
        runtime
            .scripted_image_in(plan(500), factory(&other))
            .unwrap_err()
            .code(),
        Some("image_resource_limit")
    );
    assert_eq!(other.loads.load(Ordering::SeqCst), 0);
    release(&state);
    assert_eq!(stream.finish().unwrap_err().code(), Some("image_cancelled"));
    assert!(!model.is_busy());
    assert_eq!(runtime.stats().active_image_jobs, 0);
    assert_eq!(
        runtime.stats().image_reserved_mb,
        600,
        "retained caches stay charged while resident"
    );
    runtime.evict_image_model(&model).unwrap();
    assert_eq!(runtime.stats().image_reserved_mb, 0);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert!(
        model
            .generate("retry after cancellation")
            .size(32, 32)
            .run()
            .is_ok()
    );
}

#[test]
fn dropping_a_stream_cancels_without_releasing_inflight_reservations() {
    let runtime = Runtime::builder()
        .resident_memory_budget_mb(1000)
        .build()
        .unwrap();
    let (state, started) = blocking();
    let model = runtime
        .scripted_image_in(plan(600), factory(&state))
        .unwrap();
    let stream = model.generate("a teapot").size(32, 32).stream().unwrap();
    let cancel = stream.canceller();
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    drop(stream);
    assert!(cancel.is_cancelled());
    assert_eq!(runtime.stats().image_reserved_mb, 600);
    assert!(model.is_busy());
    release(&state);
    wait_idle(&model);
    runtime.evict_image_model(&model).unwrap();
    assert_eq!(runtime.stats().image_reserved_mb, 0);
}

#[test]
fn validation_and_precancellation_do_not_restore_evicted_weights_or_start_native_work() {
    let runtime = Runtime::new().unwrap();
    let state = Arc::new(State::default());
    let model = runtime
        .scripted_image_in(plan(100), factory(&state))
        .unwrap();
    runtime.evict_image_model(&model).unwrap();
    for request in [
        model.generate(""),
        model.generate("has\0nul"),
        model.generate("x").size(31, 32),
        model.generate("x").size(u32::MAX, u32::MAX),
        model.generate("x").seed(-1),
        model.generate("x".repeat(4097)),
    ] {
        assert_eq!(
            request.run().unwrap_err().code(),
            Some("image_invalid_request")
        );
    }
    let request = model.generate("cancel before start");
    request.canceller().cancel();
    assert_eq!(request.run().unwrap_err().code(), Some("image_cancelled"));
    assert_eq!(state.loads.load(Ordering::SeqCst), 1);
    assert!(state.requests.lock().unwrap().is_empty());
    assert!(!model.is_resident());
}

#[test]
fn progress_completion_and_saved_pixels_are_usable_after_model_eviction() {
    let runtime = Runtime::new().unwrap();
    let state = Arc::new(State::default());
    let model = runtime
        .scripted_image_in(plan(100), factory(&state))
        .unwrap();
    let mut stream = model
        .generate("custom prompt")
        .size(64, 32)
        .seed(91)
        .stream()
        .unwrap();
    let mut steps = Vec::new();
    for event in stream.by_ref() {
        if let ImageEvent::Progress(progress) = event.unwrap() {
            steps.push(progress.step);
        }
    }
    assert_eq!(steps, [1, 8]);
    let response = stream.finish().unwrap();
    runtime.evict_image_model(&model).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("result.png");
    response.image.save(&path).unwrap();
    let saved = ::image::open(path).unwrap().to_rgba8();
    assert_eq!(saved.dimensions(), (64, 32));
    assert_eq!(saved.as_raw(), response.image.bytes());
    assert_eq!(
        state.requests.lock().unwrap()[0],
        ("custom prompt".into(), 64, 32, 91)
    );
}

#[test]
fn native_failure_releases_busy_but_keeps_the_context_charged() {
    let runtime = Runtime::new().unwrap();
    let state = Arc::new(State {
        fail: true,
        ..State::default()
    });
    let model = runtime
        .scripted_image_in(plan(100), factory(&state))
        .unwrap();
    for _ in 0..2 {
        assert_eq!(
            model
                .generate("fail")
                .size(32, 32)
                .run()
                .unwrap_err()
                .code(),
            Some("image_execution_failed")
        );
        assert!(!model.is_busy());
        assert_eq!(runtime.stats().image_reserved_mb, 100);
    }
    assert_eq!(state.loads.load(Ordering::SeqCst), 1);
}

#[test]
fn foreign_runtime_cannot_evict_or_preload_an_image_handle() {
    let runtime = Runtime::new().unwrap();
    let other = Runtime::new().unwrap();
    let model = runtime
        .scripted_image_in(plan(100), factory(&Arc::new(State::default())))
        .unwrap();
    assert_eq!(
        other.evict_image_model(&model).unwrap_err().code(),
        Some("invalid_request")
    );
    assert_eq!(
        other.preload_image_model(&model).unwrap_err().code(),
        Some("invalid_request")
    );
    assert!(model.is_resident());
}

#[test]
fn changed_components_are_rejected_on_reload_and_memory_plan_is_bounded() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("component");
    std::fs::write(&file, b"weights").unwrap();
    let (_, files) = ImageBundle::qwen_image_2_1_turbo(&file, &file, &file)
        .verify()
        .unwrap();
    let options = ImageLoadOptions::new("library");
    let plan = options.plan(&files).unwrap();
    assert_eq!(plan.total_mb, plan.weights_mb + plan.working_mb);
    std::fs::write(&file, b"changed component").unwrap();
    assert!(matches!(
        files[0].check(),
        Err(ImageError::InvalidBundle(_))
    ));
    assert!(
        ImageLoadOptions {
            max_pixels: u64::MAX,
            ..options
        }
        .plan(&files)
        .is_err()
    );
}

#[test]
#[cfg(not(feature = "backend-stable-diffusion"))]
fn a_chat_only_build_reports_the_missing_image_backend_before_reading_files() {
    let runtime = Runtime::new().unwrap();
    let error = runtime
        .load_image_model(
            ImageBundle::qwen_image_2_1_turbo("missing", "missing", "missing"),
            ImageLoadOptions::new("missing"),
        )
        .unwrap_err();
    assert_eq!(error.code(), Some("image_backend_unavailable"));
    assert!(runtime.models().is_empty());
}
