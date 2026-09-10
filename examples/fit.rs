//! Will this model run here, and at what context?
//!
//! ```sh
//! cargo run --example fit -- /path/model.gguf
//! ```

use gen2::advanced::fit::ModelInfo;
use gen2::advanced::runtime::HardwareProfile;

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

    // `gen2::load` runs the same check and sizes the context to the machine.
    // A model that cannot fit fails there with the verdict on the error
    // (`Error::fit()`), not a load error.
    match gen2::load(&path) {
        Ok(model) => {
            let loaded = model.info();
            println!(
                "\nloaded {} with context {:?}",
                loaded.name.as_deref().unwrap_or("?"),
                loaded.context_window
            );
        }
        Err(e) => {
            println!("refused: {e}");
            if let Some(fit) = e.fit() {
                println!("  → {} would have worked", fit.max_context);
            }
        }
    }
    Ok(())
}
