//! Isolated microbenchmark: does mlx-rs `try_index_mut` DONATE the buffer
//! (O(1) in-place slice-write, like MLX-Python `slice_update`) or COPY it
//! (O(buffer))? This decides whether the fast-path KV cache can do a true
//! in-place per-token write instead of the current O(context) concat rebuild.
//!
//! Run: cargo test -p pio-core --release --no-default-features \
//!   --features backend-mlx,backend-llamacpp --lib kv_donation_probe \
//!   -- --ignored --nocapture --test-threads=1
//!
//! No model load — pure tensor ops. Three variants timed over many iters at a
//! realistic KV shape [1, n_kv, CAP, head_dim] bf16:
//!   A. concat-rebuild  (current `inplace_assign`): concat(head, new, tail)
//!   B. try_index_mut into a UNIQUELY-OWNED buffer (no view retained)
//!   C. try_index_mut into a buffer whose prefix VIEW is retained each step
//!      (mirrors returning the attention view — the suspected donation blocker)
//! If B is ~flat and >>faster than A, donation works and Fix #1 is viable.
//! If B tracks A, mlx-rs does not donate here and Fix #1 needs another route.

#[cfg(test)]
mod tests {
    use mlx_rs::Array;
    use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
    use mlx_rs::transforms::eval;

    const N_KV: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const CAP: i32 = 2048;
    const ITERS: usize = 300;

    fn bf16_zeros(rows: i32) -> Array {
        Array::zeros::<f32>(&[1, N_KV, rows, HEAD_DIM])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    }

    fn new_row() -> Array {
        Array::ones::<f32>(&[1, N_KV, 1, HEAD_DIM])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    }

    fn ms(f: impl FnOnce()) -> f64 {
        let t = std::time::Instant::now();
        f();
        t.elapsed().as_secs_f64() * 1000.0
    }

    #[test]
    #[ignore]
    fn kv_donation_probe() {
        // Warm up Metal kernels.
        let _w = bf16_zeros(CAP);
        eval([&_w]).unwrap();

        // ── A: concat-rebuild (the current approach) ──────────────────────────
        let a_ms = ms(|| {
            let mut buf = bf16_zeros(CAP);
            for i in 0..ITERS as i32 {
                let row = new_row();
                let head = buf.index((.., .., 0..i, ..));
                let tail = buf.index((.., .., (i + 1)..CAP, ..));
                buf = mlx_rs::ops::concatenate_axis(&[&head, &row, &tail], 2).unwrap();
                eval([&buf]).unwrap();
            }
        });

        // ── B: try_index_mut into a uniquely-owned buffer, no view retained ──
        let b_ms = ms(|| {
            let mut buf = bf16_zeros(CAP);
            for i in 0..ITERS as i32 {
                let row = new_row();
                buf.try_index_mut((.., .., i..(i + 1), ..), &row).unwrap();
                eval([&buf]).unwrap();
            }
        });

        // ── C: try_index_mut, but retain the prefix VIEW each step (mirrors
        //      returning the attention view — suspected donation blocker) ──────
        let c_ms = ms(|| {
            let mut buf = bf16_zeros(CAP);
            let mut _retained: Option<Array> = None;
            for i in 0..ITERS as i32 {
                let row = new_row();
                buf.try_index_mut((.., .., i..(i + 1), ..), &row).unwrap();
                let view = buf.index((.., .., 0..(i + 1), ..));
                eval([&view]).unwrap();
                _retained = Some(view); // hold a ref into the buffer across steps
            }
        });

        let per = |t: f64| t / ITERS as f64;
        println!("\n=== KV donation probe ({ITERS} iters, CAP={CAP}) ===");
        println!("A concat-rebuild      : {:.3} ms/iter", per(a_ms));
        println!("B try_index_mut (own) : {:.3} ms/iter", per(b_ms));
        println!("C try_index_mut (view): {:.3} ms/iter", per(c_ms));
        println!(
            "\nVERDICT: B/A = {:.2}x  C/A = {:.2}x  (B << A ⇒ donation works; B≈A ⇒ it copies)",
            per(b_ms) / per(a_ms),
            per(c_ms) / per(a_ms),
        );
    }
}
