//! Embedding text — turning strings into vectors you can compare.
//!
//! ```sh
//! cargo run --example embeddings --no-default-features --features metal -- /path/embedding-model.gguf
//! ```

use pio_gen2::Engine;

/// Cosine similarity — the usual way to compare two embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: embeddings <embedding-model.gguf>")?;

    // An embedder-only engine. No chat model needed — generation would return
    // ModelNotLoaded, embedding works.
    let engine = Engine::builder().embedder(&model).build()?;
    println!("embedder loaded: {}", engine.is_embedder_loaded());

    // Batch — one call, one vector per input, in order. This is the fast path
    // for embedding a corpus.
    let corpus = [
        "The cat sat on the mat.".to_string(),
        "A feline rested on the rug.".to_string(),
        "Rust has a borrow checker.".to_string(),
    ];
    let vectors = engine.embed(&corpus)?;

    println!(
        "{} vectors of {} dimensions\n",
        vectors.len(),
        vectors[0].len()
    );

    // The first two say the same thing; the third doesn't. Similarity should
    // reflect that.
    println!("cat/feline    {:.3}", cosine(&vectors[0], &vectors[1]));
    println!("cat/rust      {:.3}", cosine(&vectors[0], &vectors[2]));

    // Single input, when you're embedding a query rather than a corpus.
    let query = engine.embed_one("Where did the cat sit?")?;
    println!("\nquery vs each:");
    for (text, v) in corpus.iter().zip(&vectors) {
        println!("  {:.3}  {text}", cosine(&query, v));
    }

    Ok(())
}
