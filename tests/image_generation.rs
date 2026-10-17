//! Real Metal qualification of the public API. See docs/image-generation.md.
#![cfg(all(
    feature = "backend-stable-diffusion",
    feature = "backend-llamacpp",
    target_os = "macos",
    target_arch = "aarch64"
))]

use gen2::{
    Runtime, Session,
    image::{ImageBundle, ImageEvent, ImageLoadOptions},
    model::ThinkingMode,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

fn recall(model: &gen2::Model, session: &mut Session, prompt: &str) -> serde_json::Value {
    let started = Instant::now();
    let result = model
        .turn(session)
        .user(prompt)
        .reasoning(ThinkingMode::Off)
        .temperature(0.0)
        .max_tokens(32)
        .run()
        .unwrap();
    assert_eq!(result.text().trim(), "MAPLE");
    assert!(model.runtime().residency().is_resident(model.id()));
    json!({"text": result.text(), "seconds": started.elapsed().as_secs_f64()})
}

#[test]
#[ignore = "requires the ABI-2 Metal bridge, Qwen Image components and a local llama.cpp model"]
fn metal_public_image_api_coexistence_cancellation_and_reload() {
    let side = std::env::var("GEN2_IMAGE_TEST_SIDE")
        .map(|s| s.parse::<u32>().unwrap())
        .unwrap_or(1024);
    let library =
        PathBuf::from(std::env::var_os("GEN2_IMAGE_LIBRARY").expect("GEN2_IMAGE_LIBRARY"));
    let weights = PathBuf::from(std::env::var_os("GEN2_IMAGE_MODELS").expect("GEN2_IMAGE_MODELS"));
    let llama = PathBuf::from(std::env::var_os("GEN2_IMAGE_LLAMA").expect("GEN2_IMAGE_LLAMA"));
    let output = PathBuf::from(
        std::env::var_os("GEN2_IMAGE_EVIDENCE_DIR").expect("GEN2_IMAGE_EVIDENCE_DIR"),
    );
    std::fs::create_dir_all(&output).unwrap();
    assert!(
        !output.join("report.json").exists(),
        "use a fresh evidence directory"
    );
    let bundle = ImageBundle::qwen_image_2_1_turbo(
        weights.join("qwen-image-2.1-turbo-Q4_K_M.gguf"),
        weights.join("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
        weights.join("vae/qwen_image_2.1_vae_bf16.safetensors"),
    );
    let options = ImageLoadOptions::new(library);
    let denied = Runtime::builder()
        .resident_memory_budget_mb(1)
        .build()
        .unwrap();
    assert_eq!(
        denied
            .load_image_model(bundle.clone(), options.clone())
            .unwrap_err()
            .code(),
        Some("image_resource_limit")
    );
    assert!(denied.models().is_empty());

    let runtime = Runtime::builder()
        .resident_memory_budget_mb(32768)
        .build()
        .unwrap();
    let chat = runtime.load(llama).unwrap();
    let offload = chat.info().offload.unwrap();
    assert!(offload.on_gpu());
    assert_eq!(offload.backend.as_deref(), Some("MTL"));
    let mut session = Session::new();
    let before = recall(
        &chat,
        &mut session,
        "The secret word is MAPLE. Remember it. Reply with only the secret word.",
    );
    let started = Instant::now();
    let model = runtime.load_image_model(bundle, options).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let plan = model.memory_plan().clone();
    assert_eq!(runtime.stats().image_reserved_mb, plan.total_mb);
    assert_eq!(runtime.models().len(), 2);
    assert!(runtime.residency().is_resident(model.id()));
    let prompt = "A red ceramic teapot on a wooden table beside a small green plant, warm morning sunlight through a window, beautiful editorial product photograph, detailed, natural colors.";
    let mut stream = model
        .generate(prompt)
        .size(side, side)
        .seed(42)
        .stream()
        .unwrap();
    let mut steps = Vec::new();
    let mut during = None;
    for event in stream.by_ref() {
        if let ImageEvent::Progress(progress) = event.unwrap() {
            eprintln!(
                "image API step {}/{} at {:.2}s",
                progress.step,
                progress.total_steps,
                progress.elapsed.as_secs_f64()
            );
            steps.push(progress.step);
            if progress.step == 1 {
                assert!(model.is_busy());
                assert_eq!(runtime.stats().active_image_jobs, 1);
                assert_eq!(
                    runtime.evict_image_model(&model).unwrap_err().code(),
                    Some("image_busy")
                );
                assert_eq!(
                    model
                        .generate("another request")
                        .stream()
                        .err()
                        .unwrap()
                        .code(),
                    Some("image_busy")
                );
                during = Some(recall(
                    &chat,
                    &mut session,
                    "What is the secret word? Reply with only that word.",
                ));
                eprintln!(
                    "llama during public image generation: {}",
                    during.as_ref().unwrap()
                );
            }
        }
    }
    let response = stream.finish().unwrap();
    assert_eq!(steps, (0..=8).collect::<Vec<_>>());
    assert_eq!(
        (response.image.width(), response.image.height()),
        (side, side)
    );
    assert!(
        response
            .image
            .bytes()
            .chunks_exact(response.image.channels() as usize)
            .any(|p| p[..3] != response.image.bytes()[..3])
    );
    response.image.save(output.join("image.png")).unwrap();
    let after = recall(
        &chat,
        &mut session,
        "Repeat the secret word, and nothing else.",
    );
    assert_eq!(runtime.stats().active_image_jobs, 0);
    assert_eq!(runtime.stats().image_reserved_mb, plan.total_mb);

    let mut cancelled = model
        .generate("a green tree on a hill")
        .size(256, 256)
        .stream()
        .unwrap();
    let cancel = cancelled.canceller();
    loop {
        if let ImageEvent::Progress(progress) = cancelled
            .next()
            .expect("progress before cancellation")
            .unwrap()
            && progress.step == 1
        {
            cancel.cancel();
            break;
        }
    }
    assert_eq!(
        cancelled.finish().unwrap_err().code(),
        Some("image_cancelled")
    );
    assert!(!model.is_busy());
    assert!(model.is_resident());
    let retry = model
        .generate("a tiny red teapot")
        .size(256, 256)
        .run()
        .unwrap();
    retry
        .image
        .save(output.join("after-cancellation.png"))
        .unwrap();
    eprintln!("native cancellation and retry passed");

    let id = model.id();
    runtime.evict_image_model(&model).unwrap();
    assert!(!runtime.residency().is_resident(id));
    assert_eq!(runtime.stats().image_reserved_mb, 0);
    let after_eviction = recall(
        &chat,
        &mut session,
        "What secret word did I give you? Reply with that word only.",
    );
    let reload_start = Instant::now();
    let restored = model
        .generate("a small lighthouse on a green hill")
        .size(256, 256)
        .run()
        .unwrap();
    let restore_and_generate_seconds = reload_start.elapsed().as_secs_f64();
    restored
        .image
        .save(output.join("after-reload.png"))
        .unwrap();
    assert_eq!(model.id(), id);
    assert!(model.is_resident());
    assert_eq!(runtime.stats().image_reserved_mb, plan.total_mb);
    assert_eq!(runtime.stats().evictions, 0);
    runtime.evict_image_model(&model).unwrap();
    let png = std::fs::read(output.join("image.png")).unwrap();
    let hash: String = Sha256::digest(&png)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let report = json!({
        "passed": true, "pid": std::process::id(), "llama_placement": offload.to_string(),
        "hardware": runtime.hardware(), "width": side, "height": side, "steps": steps,
        "image_sha256": hash, "load_seconds": load_seconds, "generation_seconds": response.elapsed.as_secs_f64(),
        "peak_reservation_mb": plan.total_mb, "weights_reservation_mb": plan.weights_mb,
        "working_reservation_mb": plan.working_mb, "llama_before": before, "llama_during": during,
        "llama_after": after, "llama_after_image_eviction": after_eviction,
        "native_cancel_and_retry": true, "lazy_restore": true,
        "restore_and_generate_seconds": restore_and_generate_seconds,
        "final_image_reserved_mb": runtime.stats().image_reserved_mb,
        "automatic_evictions": runtime.stats().evictions,
    });
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}
