//! Gemma 4 fast autoregressive decode path (Stage A).
//!
//! Opt-in behind the `PIO_MLX_FAST` runtime flag (read once at model
//! construction — see `Gemma4Model::fast`). When OFF, none of this code runs
//! and the default f32 / manual-attention path in `gemma4.rs` is byte-identical
//! to before. The whole point of this module is to behave like **mlx-lm**, not
//! mlx-vlm: bf16 activations + the fused SDPA Metal kernel + a step-buffer KV
//! cache, mirroring `mlx_lm/models/gemma4_text.py` and `mlx_lm/models/cache.py`.
//!
//! Citations below reference those two files (and `mlx_lm/models/base.py`).
//!
//! Dtype flow (mirrors mlx-lm, which loads weights bf16 and runs the trunk in
//! bf16, upcasting only where the fused kernels do):
//!   - embedding output is cast to **bf16** (`gemma4_text.py:518` `h = embeddings`,
//!     then `* embed_scale` at :519 — bf16 weights ⇒ bf16 here),
//!   - all projection matmuls run in bf16 (quantized_matmul/plain matmul keep
//!     the activation dtype),
//!   - RMSNorm upcasts the variance reduction to f32 then returns bf16
//!     (`mx.fast.rms_norm` semantics — see `norm.rs::forward_fast`),
//!   - RoPE runs at activation dtype (bf16 × f32 table ⇒ bf16, MLX promotion),
//!   - SDPA's softmax is f32 internally regardless of input dtype (documented in
//!     mlx-rs `fast::scaled_dot_product_attention`), matching
//!     `mx.fast.scaled_dot_product_attention`,
//!   - final RMSNorm + lm_head run in bf16; logit softcapping
//!     (`gemma4_text.py:85` `logit_softcap`) is applied on the bf16 logits.
//!
//! Attention scale: Gemma 4 sets `self.scale = 1.0` (`gemma4_text.py:202`) and
//! relies on the learnable q_norm / k_norm for scaling — so the fast SDPA is
//! called with `scale = 1.0`, NOT `1/sqrt(head_dim)`.

use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::IndexOp;

/// bf16 step buffer for the fast KV cache.
const KV_STEP: usize = 256;

/// Cast an array to bf16 (no-op semantics if already bf16).
#[inline]
pub fn to_bf16(x: &Array) -> Array {
    x.as_dtype(Dtype::Bfloat16).expect("mlx op: bf16 cast")
}

/// Build a Gemma-4 attention additive mask of shape `[1, 1, query_len, kv_len]`
/// for the **fast prefill** path (seq > 1), mirroring `create_causal_mask`
/// (`mlx_lm/models/base.py`): `linds >= rinds` (causal), optionally
/// `& (linds < rinds + window_size)` for sliding-window layers.
///
/// `offset` is the number of already-cached positions before this chunk, so the
/// query absolute positions are `offset .. offset+query_len` and key positions
/// are `0 .. kv_len`. The mask dtype is bf16 with 0 / -inf entries (additive),
/// matching the activation dtype fed to the fused kernel.
fn build_fast_mask(query_len: usize, kv_len: usize, window: Option<usize>) -> Array {
    let q = query_len as i32;
    let k = kv_len as i32;
    // The returned K/V buffer is contiguous: the last `query_len` positions are
    // this chunk's queries; positions `0..(kv_len-query_len)` are the cached
    // prefix. So query qi maps to key column `(kv_len - query_len) + qi`,
    // regardless of the absolute RoPE offset — derive the alignment from shapes,
    // which stays correct even after sliding-window truncation.
    let off = k - q;
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0.0f32; (q * k) as usize];
    for qi in 0..q {
        let lpos = off + qi; // position of this query within the kv buffer
        for ki in 0..k {
            let causal_ok = lpos >= ki;
            let window_ok = window.map_or(true, |w| lpos < ki + w as i32);
            if !causal_ok || !window_ok {
                data[(qi * k + ki) as usize] = neg_inf;
            }
        }
    }
    let raw = Array::from_slice(&data, &[q, k]);
    let raw = raw.reshape(&[1, 1, q, k]).expect("mlx op");
    // Additive mask in bf16 to match the bf16 scores the kernel produces.
    raw.as_dtype(Dtype::Bfloat16).expect("mlx op")
}

/// Fused scaled-dot-product attention mirroring `gemma4_text.py:262` →
/// `base.py::scaled_dot_product_attention` → `mx.fast.scaled_dot_product_attention`.
///
/// Handles GQA **natively** (queries: `[B, n_heads, L, D]`, keys/values:
/// `[B, n_kv_heads, S, D]` — NOT pre-tiled), so the manual `repeat_kv` of the
/// default path is gone. `scale = 1.0` per Gemma 4.
///
/// - decode (`seq == 1`): no mask — the single query attends every cached
///   position; for sliding layers the cache is already truncated to the window
///   by the caller, so a window mask is unnecessary (mirrors
///   `create_attention_mask` returning `None` when `N == 1`).
/// - prefill (`seq > 1`): an explicit causal (+ optional sliding-window) mask.
pub fn fast_sdpa(q: &Array, k: &Array, v: &Array, seq: usize, window: Option<usize>) -> Array {
    if seq == 1 {
        mlx_rs::fast::scaled_dot_product_attention(q, k, v, 1.0, None, None::<&Array>)
            .expect("mlx op: fast sdpa decode")
    } else {
        let kv_len = k.shape()[2] as usize;
        let mask = build_fast_mask(seq, kv_len, window);
        let m = ScaledDotProductAttentionMask::Array(&mask);
        mlx_rs::fast::scaled_dot_product_attention(q, k, v, 1.0, Some(m), None::<&Array>)
            .expect("mlx op: fast sdpa prefill")
    }
}

/// Step-buffer KV cache update, mirroring `KVCache.update_and_fetch`
/// (`mlx_lm/models/cache.py:333`): keep an over-allocated buffer grown in
/// `KV_STEP` (256) chunks and write the new `keys`/`values` in place at
/// `[.., prev:offset, :]`, returning a **view of the filled prefix**
/// `[.., :offset, :]`.
///
/// State convention for the fast path: the `(Array, Array)` slot holds the
/// **over-allocated buffer**; `prev` (= the true fill before this call, tracked
/// by the session as `cur_pos`) tells us where to write. The returned view is
/// what attention consumes; the buffer is what we store back into the slot.
///
/// Sliding-window layers cap the *returned* view (and the stored buffer) to the
/// last `window` positions, mirroring the default path's truncation so the
/// decode-time `seq == 1` no-mask shortcut stays valid.
pub fn step_buffer_update(
    slot: &mut Option<(Array, Array)>,
    new_k: &Array,
    new_v: &Array,
    abs_offset: usize,
    window: Option<usize>,
) -> (Array, Array) {
    let s = new_k.shape();
    let (b, n_kv, seq, kd) = (s[0], s[1], s[2], s[3]);
    let vd = new_v.shape()[3];

    // `abs_offset` is the true absolute position (RoPE offset). The PHYSICAL
    // fill of THIS slot's buffer is NOT abs_offset for sliding-window layers —
    // after the buffer saturates we keep only the last `window` positions, so
    // the contiguous fill caps at `window`. Using abs_offset directly here was
    // the post-turn-2 collapse bug (writing past the truncated buffer). The
    // default path sidesteps this because concat derives length from the array;
    // the step buffer must derive it from min(offset, window). Mirrors the
    // "do NOT infer offset from cache shape" warning in gemma4.rs::forward.
    let prev = match window {
        Some(w) => abs_offset.min(w),
        None => abs_offset,
    };

    // Grow / allocate the buffer when the current buffer can't hold prev+seq.
    let need = prev + seq as usize;
    let (mut buf_k, mut buf_v) = match slot.take() {
        Some((bk, bv)) if (bk.shape()[2] as usize) >= need => (bk, bv),
        Some((bk, bv)) => {
            // Trim any unused tail (when prev isn't a step multiple) then extend
            // by whole steps — mirrors cache.py:343-348.
            let bk = bk.index((.., .., 0..(prev as i32), ..));
            let bv = bv.index((.., .., 0..(prev as i32), ..));
            let n_steps = (KV_STEP + seq as usize - 1) / KV_STEP;
            let grow = (n_steps * KV_STEP) as i32;
            let zk = Array::zeros::<f32>(&[b, n_kv, grow, kd])
                .expect("mlx op")
                .as_dtype(new_k.dtype())
                .expect("mlx op");
            let zv = Array::zeros::<f32>(&[b, n_kv, grow, vd])
                .expect("mlx op")
                .as_dtype(new_v.dtype())
                .expect("mlx op");
            let bk = mlx_rs::ops::concatenate_axis(&[&bk, &zk], 2).expect("mlx op");
            let bv = mlx_rs::ops::concatenate_axis(&[&bv, &zv], 2).expect("mlx op");
            (bk, bv)
        }
        None => {
            let n_steps = (KV_STEP + seq as usize - 1) / KV_STEP;
            let cap = (n_steps * KV_STEP) as i32;
            let bk = Array::zeros::<f32>(&[b, n_kv, cap, kd])
                .expect("mlx op")
                .as_dtype(new_k.dtype())
                .expect("mlx op");
            let bv = Array::zeros::<f32>(&[b, n_kv, cap, vd])
                .expect("mlx op")
                .as_dtype(new_v.dtype())
                .expect("mlx op");
            (bk, bv)
        }
    };

    // Write the new chunk at [.., prev:offset, :] (cache.py:353). We use an
    // explicit concat rebuild (head[..prev] ++ new ++ tail[offset..cap]) rather
    // than `try_index_mut`: the latter silently produced an un-attended cache
    // for delta prefills into a non-empty buffer (the post-turn-2 collapse),
    // whereas the concat form is unambiguous. The over-allocation still buys us
    // amortised growth (realloc only every KV_STEP), so this stays a step
    // buffer, just with a safe in-place-equivalent write.
    let p = prev as i32;
    let o = need as i32;
    let _ = (n_kv, kd, vd, b);
    buf_k = inplace_assign(&buf_k, new_k, p, o);
    buf_v = inplace_assign(&buf_v, new_v, p, o);

    // Filled prefix view (cache.py:355).
    let view_k = buf_k.index((.., .., 0..o, ..));
    let view_v = buf_v.index((.., .., 0..o, ..));

    // Sliding-window truncation of BOTH the stored buffer and the returned view,
    // so the next decode step's no-mask `seq == 1` shortcut remains correct.
    if let Some(w) = window {
        if need > w {
            let start = (need - w) as i32;
            let tk = view_k.index((.., .., start.., ..));
            let tv = view_v.index((.., .., start.., ..));
            // Store the truncated prefix as the new buffer (re-materialized; the
            // sliding cache stays bounded at `window`, so this is cheap).
            *slot = Some((tk.clone(), tv.clone()));
            return (tk, tv);
        }
    }

    *slot = Some((buf_k, buf_v));
    (view_k, view_v)
}

/// Write `new` into the over-allocated buffer at `[.., prev:offset, :]` by
/// rebuilding `concat(buf[..prev], new, buf[offset..cap])`. This is the safe
/// in-place-equivalent of `self.keys[..., prev:offset, :] = keys` (cache.py:353)
/// — the filled prefix `[..offset]` is exactly the cached K/V; the `[offset..]`
/// tail stays zero-padded capacity, sliced off by the caller before attention.
fn inplace_assign(buf: &Array, new: &Array, prev: i32, offset: i32) -> Array {
    let cap = buf.shape()[2];
    let head = buf.index((.., .., 0..prev, ..));
    let tail = buf.index((.., .., offset..cap, ..));
    mlx_rs::ops::concatenate_axis(&[&head, new, &tail], 2).expect("mlx op")
}
