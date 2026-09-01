//! Will this model run here, and at what context?
//!
//! ```sh
//! cargo run --example fit --no-default-features --features metal -- /path/model.gguf
//! ```

use pio_gen2::{Engine, HardwareProfile, ModelInfo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: fit <model.gguf>")?;

    // Reads the header only — no weights loaded, cheap enough to run over a
    // directory of candidates.
    let info = ModelInfo::read(&path)?;
    println!(
        "{} · {} · {:.2} GB · trained for {} context",
        info.architecture.as_deref().unwrap_or("unknown"),
        info.quantization.as_deref().unwrap_or("unknown"),
        info.file_bytes as f64 / 1e9,
        info.train_context.unwrap_or(0),
    );

    let hw = HardwareProfile::detect();
    println!(
        "machine: {:.0} GB RAM, {} cores, {:?}",
        hw.total_ram_bytes as f64 / 1e9,
        hw.cpu_cores,
        hw.gpu_backend
    );

    println!("\nlargest context here: {}", info.max_context(&hw));
    println!("at 8192:   {}", info.fits(&hw, Some(8192)));
    println!("at 10M:    {}", info.fits(&hw, Some(10_000_000)));

    // Sized to the machine rather than guessed.
    let engine = Engine::builder().model(&path).auto_context().build()?;
    println!(
        "\nloaded with auto_context, embedder: {}",
        engine.is_embedder_loaded()
    );

    // An impossible context fails with the verdict, not a load error.
    match Engine::builder().model(&path).context(10_000_000).build() {
        Ok(_) => println!("unexpectedly loaded at 10M context"),
        Err(e) => {
            println!("refused: {e}");
            if let Some(fit) = e.fit() {
                println!("  → {} would have worked", fit.max_context);
            }
        }
    }

    Ok(())
}
