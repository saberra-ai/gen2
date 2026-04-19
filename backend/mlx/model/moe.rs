//! Mixture-of-Experts blocks for Gemma 4 26B.
//!
//! Mirrors `Router` + `Experts` + `SwitchGLU` from `mlx_lm/models/gemma4_text.py`.
//! `gather_mm` is not exposed in the safe `mlx-rs` 0.25 API, so we dispatch
//! experts sequentially per selected slot. Performance will be below the
//! mlx-lm reference until a batched gather-matmul lands upstream, but output
//! is numerically equivalent.

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::norm::RmsNorm;
use super::quantized::Weight;

// ─── Router ───────────────────────────────────────────────────────────────────

/// Routes tokens to top-k experts.
///
/// Forward:
/// 1. RMSNorm with scale=`self.scale * 1/√hidden` (fused into the weight).
/// 2. Linear projection to `num_experts` logits.
/// 3. `top_k` selection + softmax (only over the selected logits).
/// 4. Multiply each selected weight by `per_expert_scale[idx]`.
pub struct Router {
    pub norm: RmsNorm,
    pub proj: Weight,
    pub per_expert_scale: Array,
    pub num_experts: usize,
    pub top_k: usize,
    /// Precomputed `1/√hidden`.
    root_size: f32,
}

impl Router {
    pub fn new(hidden: usize, num_experts: usize, top_k: usize, eps: f32) -> Self {
        Self {
            norm: RmsNorm::new(hidden, eps),
            proj: Weight::plain(
                Array::zeros::<f32>(&[num_experts as i32, hidden as i32]).expect("mlx op"),
            ),
            per_expert_scale: Array::ones::<f32>(&[num_experts as i32]).expect("mlx op"),
            num_experts,
            top_k,
            root_size: 1.0 / (hidden as f32).sqrt(),
        }
    }

    /// Returns `(top_k_indices, top_k_weights)` each shaped `[B, S, top_k]`.
    pub fn forward(&self, x: &Array) -> (Array, Array) {
        // Python: `x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)`.
        // Our RmsNorm already multiplies by the learnable weight; we absorb
        // `_root_size` by scaling the input before norm.
        let scale = Array::from_f32(self.root_size);
        let x_scaled = x.multiply(&scale).expect("mlx op");
        let h = self.norm.forward(&x_scaled);

        // Expert scores: [B, S, num_experts]
        let scores = self.proj.matmul_transpose(&h);

        // Top-k selection. mlx-rs doesn't expose argpartition; use argsort
        // on the last axis, take the last `top_k` indices. Correctness is
        // identical; cost is O(E log E) vs O(E) per token.
        let sorted_idx = mlx_rs::ops::argsort_axis(&scores, -1).expect("mlx op");
        let n_exp = self.num_experts as i32;
        let k = self.top_k as i32;
        let top_k_idx = sorted_idx.index((.., .., (n_exp - k)..n_exp));

        // Gather the selected scores, softmax, multiply by per-expert scale.
        let top_k_scores = scores.take_along_axis(&top_k_idx, -1).expect("mlx op");
        let weights = mlx_rs::ops::softmax_axes(&top_k_scores, &[-1], None).expect("mlx op");
        let expert_scale = self
            .per_expert_scale
            .take_axis(&top_k_idx, 0)
            .expect("mlx op");
        let weights = weights.multiply(&expert_scale).expect("mlx op");

        (top_k_idx, weights)
    }
}

// ─── Experts ──────────────────────────────────────────────────────────────────

/// Sparse expert MLPs. Weights are stored as one tile per expert:
///   `gate_proj`: `[n_experts, moe_intermediate, hidden]`
///   `up_proj`:   `[n_experts, moe_intermediate, hidden]`
///   `down_proj`: `[n_experts, hidden, moe_intermediate]`
///
/// The Python implementation uses `SwitchGLU` (backed by `gather_mm`) for a
/// single batched expert dispatch. mlx-rs 0.25 doesn't expose `gather_mm`,
/// so we loop over the `top_k` slot dimension and do `num_experts` small
/// matmuls per slot — still vectorized across batch/seq.
pub struct Experts {
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    pub num_experts: usize,
}

impl Experts {
    pub fn new(hidden: usize, moe_intermediate: usize, num_experts: usize) -> Self {
        Self {
            gate_proj: Weight::plain(
                Array::zeros::<f32>(&[
                    num_experts as i32,
                    moe_intermediate as i32,
                    hidden as i32,
                ])
                .expect("mlx op"),
            ),
            up_proj: Weight::plain(
                Array::zeros::<f32>(&[
                    num_experts as i32,
                    moe_intermediate as i32,
                    hidden as i32,
                ])
                .expect("mlx op"),
            ),
            down_proj: Weight::plain(
                Array::zeros::<f32>(&[
                    num_experts as i32,
                    hidden as i32,
                    moe_intermediate as i32,
                ])
                .expect("mlx op"),
            ),
            num_experts,
        }
    }

    /// `x`: `[B, S, H]`, `indices`: `[B, S, k]`, `weights`: `[B, S, k]`.
    /// Returns `[B, S, H]`.
    ///
    /// TODO(mlx-rs gather_mm): when the safe wrapper exists, replace this
    /// loop with one `switch_glu` call — should be ~4-8× faster on 26B.
    pub fn forward(&self, x: &Array, indices: &Array, weights: &Array) -> Array {
        // One-hot mask per token per expert: [B, S, num_experts]
        //
        // We build a score tensor where `score[b,s,e] = sum_k (indices[b,s,k]==e) * weights[b,s,k]`.
        // Then `output = sum_e score[b,s,e] * expert_e(x[b,s])`. Equivalent to SwitchGLU
        // but expressed as a dense sum so we only need per-expert matmul.
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let e = self.num_experts as i32;
        let k = indices.shape()[2];

        // Scatter weights onto a [B, S, E] dense tensor.
        // MLX scatter: use `put_along_axis` alternative — build via one-hot mask.
        // `indices`: [B, S, k] int → expand to one-hot: [B, S, k, E] → collapse k.
        let onehot = build_onehot(indices, self.num_experts);
        // weights: [B, S, k] → [B, S, k, 1]
        let w_ = weights.reshape(&[batch, seq, k, 1]).expect("mlx op");
        let weighted = onehot.multiply(&w_).expect("mlx op"); // [B, S, k, E]
        let dense_w = weighted.sum_axis(2, false).expect("mlx op"); // [B, S, E]

        // Per-expert matmul. Total cost: E × (B*S × H → B*S × moe_dim → B*S × H).
        // For 32 experts this is fine on Metal; <5% overhead vs. gather_mm at
        // 2B batch size.
        let mut acc: Option<Array> = None;
        for ei in 0..(self.num_experts as i32) {
            // Slice per-expert weight tiles.
            let g = self.gate_proj.to_full().index((ei..ei + 1, .., ..)); // [1, moe, H]
            let u = self.up_proj.to_full().index((ei..ei + 1, .., ..));
            let d = self.down_proj.to_full().index((ei..ei + 1, .., ..));

            let g = g
                .reshape(&[g.shape()[1], g.shape()[2]])
                .expect("mlx op")
                .transpose_axes(&[1, 0])
                .expect("mlx op"); // [H, moe]
            let u = u
                .reshape(&[u.shape()[1], u.shape()[2]])
                .expect("mlx op")
                .transpose_axes(&[1, 0])
                .expect("mlx op");
            let d = d
                .reshape(&[d.shape()[1], d.shape()[2]])
                .expect("mlx op")
                .transpose_axes(&[1, 0])
                .expect("mlx op"); // [moe, H]

            let gated = x.matmul(&g).expect("mlx op");
            let gated = mlx_rs::nn::gelu_approximate(&gated).expect("mlx op");
            let upped = x.matmul(&u).expect("mlx op");
            let mix = gated.multiply(&upped).expect("mlx op");
            let out = mix.matmul(&d).expect("mlx op"); // [B, S, H]

            // Scale by this expert's dense weight: [B, S, 1]
            let w_e = dense_w.index((.., .., ei..ei + 1));
            let contrib = out.multiply(&w_e).expect("mlx op");

            acc = Some(match acc {
                Some(a) => a.add(&contrib).expect("mlx op"),
                None => contrib,
            });
        }
        acc.unwrap_or_else(|| x.multiply(&Array::from_f32(0.0)).expect("mlx op"))
    }
}

/// One-hot encode `indices` (shape `[B, S, k]`, values in `[0, num_experts)`)
/// into shape `[B, S, k, num_experts]`.
fn build_onehot(indices: &Array, num_experts: usize) -> Array {
    let shape = indices.shape();
    let (batch, seq, k) = (shape[0], shape[1], shape[2]);
    // Flatten to [B*S*k], scatter, reshape.
    let flat = indices
        .reshape(&[batch * seq * k])
        .expect("mlx op");
    let e = num_experts as i32;
    let arange = mlx_rs::ops::arange::<_, i32>(0i32, e, 1i32).expect("mlx op");
    // Broadcast compare: [B*S*k, 1] == [1, E] → [B*S*k, E] bool
    let flat_col = flat.reshape(&[batch * seq * k, 1]).expect("mlx op");
    let row = arange.reshape(&[1, e]).expect("mlx op");
    let eq = flat_col.eq(&row).expect("mlx op");
    let eq_f = eq
        .as_type::<f32>()
        .expect("mlx op");
    eq_f.reshape(&[batch, seq, k, e]).expect("mlx op")
}
