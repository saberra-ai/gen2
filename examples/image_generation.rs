//! cargo run --no-default-features --features backend-stable-diffusion \
//!   --example image_generation -- BRIDGE_DYLIB WEIGHTS_DIR 'a red teapot' image.png
use gen2::{
    Runtime,
    image::{ImageBundle, ImageEvent, ImageLoadOptions},
};
use std::path::PathBuf;

fn main() -> gen2::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(gen2::Error::InvalidRequest(
            "usage: image_generation BRIDGE_DYLIB WEIGHTS_DIR PROMPT OUTPUT.png".into(),
        ));
    }
    let weights = PathBuf::from(&args[1]);
    let bundle = ImageBundle::qwen_image_2_1_turbo(
        weights.join("qwen-image-2.1-turbo-Q4_K_M.gguf"),
        weights.join("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
        weights.join("vae/qwen_image_2.1_vae_bf16.safetensors"),
    );
    let runtime = Runtime::new()?;
    let model = runtime.load_image_model(bundle, ImageLoadOptions::new(&args[0]))?;
    println!("peak reservation: {:?}", model.memory_plan());
    let mut events = model.generate(&args[2]).seed(42).stream()?;
    for event in events.by_ref() {
        if let ImageEvent::Progress(progress) = event? {
            println!(
                "{}/{} steps, {:.1}s",
                progress.step,
                progress.total_steps,
                progress.elapsed.as_secs_f64()
            );
        }
    }
    let response = events.finish()?;
    response.image.save(&args[3])?;
    runtime.evict_image_model(&model)?;
    println!(
        "saved {} in {:.1}s",
        args[3],
        response.elapsed.as_secs_f64()
    );
    Ok(())
}
