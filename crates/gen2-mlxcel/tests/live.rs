//! The seam, end to end: a real MLX model, loaded through the plugin, decodes
//! a token through `gen2`'s public API.
//!
//! Needs `PIO_TEST_MLX_MODEL` pointing at an MLX safetensors directory (a
//! 4-bit Qwen3-0.6B is enough). Skips loudly without it — a silently skipped
//! live test reads as a pass.

use std::path::Path;

/// What the plugin says about itself, with no weights involved.
#[test]
fn the_plugin_names_itself_and_claims_safetensors_directories() {
    let plugin = gen2_mlxcel::plugin();
    assert_eq!(plugin.name, "mlxcel");

    let dir = tempfile::tempdir().unwrap();
    assert!(
        !(plugin.claims)(dir.path()),
        "an empty directory is not a model"
    );
    std::fs::write(dir.path().join("x.safetensors"), b"").unwrap();
    assert!(
        (plugin.claims)(dir.path()),
        "a directory with a safetensors file is"
    );

    let gguf = dir.path().join("model.gguf");
    std::fs::write(&gguf, b"not a model either").unwrap();
    assert!(!(plugin.claims)(&gguf), "a GGUF file is llama.cpp's");
    assert!(!(plugin.claims)(Path::new("/nowhere/model.gguf")));
}

/// A token, decoded by mlxcel, observed through `Engine::infer`.
#[test]
fn a_registered_mlxcel_plugin_generates_through_the_public_api() {
    let Ok(model) = std::env::var("PIO_TEST_MLX_MODEL") else {
        eprintln!("live: SKIPPED — set PIO_TEST_MLX_MODEL to an MLX safetensors directory");
        return;
    };
    let engine = gen2::Engine::builder()
        .model(&model)
        .backend(gen2_mlxcel::plugin())
        .build()
        .expect("the plugin loads the model");

    let text = engine
        .infer("Reply with exactly one word: hello")
        .max_tokens(8)
        .text()
        .expect("a generation completes");
    eprintln!("live: mlxcel said {text:?}");
    assert!(!text.trim().is_empty(), "the model produced no text");
}
