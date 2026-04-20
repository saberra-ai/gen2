//! Mixture-of-Experts blocks for Gemma 4 26B.
//!
//! Mirrors `Router` + `Experts` + `SwitchGLU` from `mlx_lm/models/gemma4_text.py`.
//! Uses `gather_qmm` (quantized batched matmul with gather) for the fused path
//! when all three expert projections are quantized — matches mlx-lm's
//! `SwitchGLU` performance profile. Falls back to a dense-sum loop for the
//! float-weights case (rare, mostly diagnostic via `PIO_FORCE_DEQUANT=1`).

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
        // That applies weight `(scale * 1/√hidden)` during the norm. RMS norm
        // normalizes by magnitude, so scaling the input by `1/√hidden` first
        // would cancel out (constant factor on numerator and denominator) and
        // leave us off by a `1/√hidden` multiplier — sharpening softmax ~53×
        // on hidden=2816 and collapsing routing onto a single expert.
        // Instead, apply `root_size` AFTER the norm so the output matches the
        // reference: `(x/rms(x)) * scale * root_size`.
        let h = self.norm.forward(x);
        let h = h
            .multiply(&Array::from_f32(self.root_size))
            .expect("mlx op");

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
///
/// **TODO(grouped GEMM)**: replace the per-expert loop with a single
/// `mlx_sys::mlx_gather_mm` call once we trust the unsafe FFI surface.
/// The C signature is:
/// ```c
/// int mlx_gather_mm(
///     mlx_array* res, const mlx_array a, const mlx_array b,
///     const mlx_array lhs_indices /* may be null */,
///     const mlx_array rhs_indices,                 // [B*S*top_k] expert ids
///     bool sorted_indices, const mlx_stream s);
/// ```
/// Pattern: flatten tokens to `[B*S, hidden]`, expand per-token to
/// `[B*S*top_k, hidden]` via gather, call `gather_mm` with `b =
/// [n_experts, hidden, moe_dim]` and rhs_indices = expert ids, GELU,
/// multiply by up branch, second `gather_mm` with down weights, scatter-add
/// back into `[B*S, hidden]` weighted by `top_k_weights`. Expected speedup
/// on 26B: ~4-8× over the current dense-sum loop.
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
                Array::zeros::<f32>(&[num_experts as i32, moe_intermediate as i32, hidden as i32])
                    .expect("mlx op"),
            ),
            up_proj: Weight::plain(
                Array::zeros::<f32>(&[num_experts as i32, moe_intermediate as i32, hidden as i32])
                    .expect("mlx op"),
            ),
            down_proj: Weight::plain(
                Array::zeros::<f32>(&[num_experts as i32, hidden as i32, moe_intermediate as i32])
                    .expect("mlx op"),
            ),
            num_experts,
        }
    }

    /// `x`: `[B, S, H]`, `indices`: `[B, S, k]`, `weights`: `[B, S, k]`.
    /// Returns `[B, S, H]`.
    pub fn forward(&self, x: &Array, indices: &Array, weights: &Array) -> Array {
        // Fast path: all three projections are quantized. Fuse gate/up/down
        // into two `gather_qmm` calls — one forward pass handles all
        // (token, expert) pairs in parallel. Mirrors mlx-lm's SwitchGLU.
        if let Some(out) = self.forward_gather_qmm(x, indices, weights) {
            return out;
        }

        // Fallback: dense-sum loop over all experts. Only reachable when
        // weights are plain float (PIO_FORCE_DEQUANT=1 or non-quantized
        // checkpoint). Kept for diagnostic parity with the reference impl.
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

    /// Fused batched forward using `gather_qmm`. Returns `None` when any of
    /// the three projections isn't quantized (fall through to dense-sum).
    ///
    /// Strategy: flatten tokens to `[M, 1, H]` one-row matrices, build a
    /// `[M*top_k]` index array mapping each slot to its source token
    /// (`lhs_indices`) and selected expert (`rhs_indices`). Two `gather_qmm`
    /// calls fuse per-slot gate/up matmuls; one more fuses down. Slot-level
    /// router weights are applied element-wise, then summed across `top_k`
    /// to get the per-token contribution.
    ///
    /// Complexity: O(M * top_k * (H * moe + moe * H)) matmul work, matching
    /// the Python reference. No per-expert dequantize, no dense-sum over
    /// unselected experts.
    fn forward_gather_qmm(
        &self,
        x: &Array,
        indices: &Array,
        weights: &Array,
    ) -> Option<Array> {
        use mlx_rs::ops::gather_qmm;

        let (gate, up, down) = match (&self.gate_proj, &self.up_proj, &self.down_proj) {
            (
                Weight::Quantized {
                    weight: gw,
                    scales: gs,
                    biases: gb,
                    group_size: gg,
                    bits: gbi,
                },
                Weight::Quantized {
                    weight: uw,
                    scales: us,
                    biases: ub,
                    ..
                },
                Weight::Quantized {
                    weight: dw,
                    scales: ds,
                    biases: db,
                    group_size: dg,
                    bits: dbi,
                },
            ) => ((gw, gs, gb, *gg, *gbi), (uw, us, ub), (dw, ds, db, *dg, *dbi)),
            _ => return None,
        };

        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];
        let m = batch * seq;
        let k_top = indices.shape()[2];
        let total = m * k_top;

        // x_flat: [M, 1, H] — one row per token, so gather_qmm's last-two
        // axes form a (1×H) × (H×moe) matmul per selected slot.
        let x_flat = x.reshape(&[m, 1, hidden]).expect("mlx op");

        // lhs_indices = [0*k, 1*k, ..., (M-1)*k] flattened: each token id
        // repeated top_k times. Build as `arange(M).reshape(M,1)` broadcast
        // across a `[1, k_top]` ones tensor → reshape to [M*k_top].
        let arange_m = mlx_rs::ops::arange::<_, u32>(0u32, m as u32, 1u32).expect("mlx op");
        let col = arange_m.reshape(&[m, 1]).expect("mlx op");
        let ones_row = Array::ones::<u32>(&[1, k_top]).expect("mlx op");
        let lhs = col
            .multiply(&ones_row)
            .expect("mlx op")
            .reshape(&[total])
            .expect("mlx op");

        // rhs_indices = flat expert ids for each slot.
        let rhs = indices
            .as_type::<u32>()
            .expect("mlx op")
            .reshape(&[total])
            .expect("mlx op");

        // Gate branch: gather_qmm(x, Wg, sg, bg, lhs, rhs, transpose=true)
        // Shape trace: x=[M,1,H], W=[E,moe,H_packed], result=[total,1,moe].
        let gate_out = gather_qmm(
            &x_flat,
            gate.0,
            gate.1,
            Some(gate.2),
            Some(&lhs),
            Some(&rhs),
            Some(true),
            Some(gate.3),
            Some(gate.4),
            None,
        )
        .expect("mlx op");
        let up_out = gather_qmm(
            &x_flat,
            up.0,
            up.1,
            Some(up.2),
            Some(&lhs),
            Some(&rhs),
            Some(true),
            Some(gate.3),
            Some(gate.4),
            None,
        )
        .expect("mlx op");

        let gated = mlx_rs::nn::gelu_approximate(&gate_out).expect("mlx op");
        let mix = gated.multiply(&up_out).expect("mlx op"); // [total, 1, moe]

        // Down branch: `mix` already has batch=total (one row per slot), so
        // we pass lhs_indices=None (no gather on x side), rhs_indices=rhs
        // (pick the matching expert's down weight for each slot).
        let down_out = gather_qmm(
            &mix,
            down.0,
            down.1,
            Some(down.2),
            None::<&Array>,
            Some(&rhs),
            Some(true),
            Some(down.3),
            Some(down.4),
            None,
        )
        .expect("mlx op"); // [total, 1, H]

        // Multiply by per-slot router weight, then reduce across top_k slots.
        let w_flat = weights.reshape(&[total, 1, 1]).expect("mlx op");
        let scaled = down_out.multiply(&w_flat).expect("mlx op"); // [total, 1, H]
        let grouped = scaled.reshape(&[m, k_top, 1, hidden]).expect("mlx op");
        let summed = grouped.sum_axis(1, false).expect("mlx op"); // [M, 1, H]

        Some(summed.reshape(&[batch, seq, hidden]).expect("mlx op"))
    }

    /// Sparse single-position forward (B=1, S=1). Iterates the `top_k`
    /// selected experts for the single token instead of summing over every
    /// expert. Host-sync on `indices` + `weights` is required (we need scalar
    /// expert ids on the CPU to index into the weight tensor), but that sync
    /// is trivially cheap for a `[1,1,k]` tensor.
    #[allow(dead_code)] // superseded by forward_gather_qmm when weights are quantized
    fn forward_sparse_single(&self, x: &Array, indices: &Array, weights: &Array) -> Array {
        let k = indices.shape()[2] as usize;

        // Hoist the dequantize ops outside the loop: MLX will dedupe, but
        // expressing it once keeps the compute graph small.
        let full_gate = self.gate_proj.to_full(); // [E, moe, H]
        let full_up = self.up_proj.to_full();
        let full_down = self.down_proj.to_full(); // [E, H, moe]

        // Evaluate indices + weights to host. argsort returns i32 on mlx-rs 0.25.
        let idx_i32 = indices.as_type::<i32>().expect("mlx op");
        let idx_flat = idx_i32.reshape(&[k as i32]).expect("mlx op");
        let w_flat = weights.reshape(&[k as i32]).expect("mlx op");
        let expert_ids: Vec<i32> = idx_flat.as_slice::<i32>().to_vec();
        let w_host: Vec<f32> = w_flat.as_slice::<f32>().to_vec();

        let mut acc: Option<Array> = None;
        for i in 0..k {
            let ei = expert_ids[i];
            let w_i = w_host[i];

            // Slice this expert's tiles. Index into the leading (expert) axis,
            // then reshape away the size-1 axis and transpose to matmul shape.
            let g = full_gate.index((ei..ei + 1, .., ..));
            let u = full_up.index((ei..ei + 1, .., ..));
            let d = full_down.index((ei..ei + 1, .., ..));

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
            let out = mix.matmul(&d).expect("mlx op"); // [1, 1, H]

            let scale = Array::from_f32(w_i);
            let contrib = out.multiply(&scale).expect("mlx op");

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
    let flat = indices.reshape(&[batch * seq * k]).expect("mlx op");
    let e = num_experts as i32;
    let arange = mlx_rs::ops::arange::<_, i32>(0i32, e, 1i32).expect("mlx op");
    // Broadcast compare: [B*S*k, 1] == [1, E] → [B*S*k, E] bool
    let flat_col = flat.reshape(&[batch * seq * k, 1]).expect("mlx op");
    let row = arange.reshape(&[1, e]).expect("mlx op");
    let eq = flat_col.eq(&row).expect("mlx op");
    let eq_f = eq.as_type::<f32>().expect("mlx op");
    eq_f.reshape(&[batch, seq, k, e]).expect("mlx op")
}
