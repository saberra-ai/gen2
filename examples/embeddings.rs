//! Embedding text — turning strings into vectors you can compare.
//!
//! ```sh
//! cargo run --example embeddings -- /path/embedding-model.gguf
//! ```

use gen2::Runtime;

/// Cosine similarity — the usual way to compare two embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: embeddings <embedding-model.gguf>")?;

    // An embedder is not a chat model and never pretends to be one: no
    // session, no turn. It shares the runtime's settings and backends.
    let runtime = Runtime::new()?;
    let embedder = runtime.load_embedder(&path)?;

    // Batch — one call, one vector per input, in order. This is the fast path
    // for embedding a corpus.
    let corpus = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Rust has a borrow checker.",
    ];
    let vectors = embedder.embed(corpus)?;
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
    let query = embedder.embed_one("Where did the cat sit?")?;
    println!("\nquery vs each:");
    for (text, v) in corpus.iter().zip(&vectors) {
        println!("  {:.3}  {text}", cosine(&query, v));
    }
    Ok(())
}
