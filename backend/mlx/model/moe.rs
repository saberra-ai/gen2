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

/// Normalize x by RMS without applying a learnable weight. Matches
/// `RMSNormNoScale` in mlx-vlm's gemma4/language.py. Used by the MoE
/// router — it applies `* scale * root_size` as separate ops AFTER this,
/// which gives a numerically different graph than a fused
/// `rms_norm(x, scale * root_size)` even though the math is equivalent.
fn rms_norm_no_scale(x: &Array, eps: f32) -> Array {
    let x_sq = x.multiply(x).expect("mlx op");
    let var = x_sq.mean_axis(-1, true).expect("mlx op");
    let var_eps = var.add(&Array::from_f32(eps)).expect("mlx op");
    let norm_factor = var_eps.rsqrt().expect("mlx op");
    x.multiply(&norm_factor).expect("mlx op")
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Routes tokens to top-k experts.
///
/// Forward:
/// 1. RMSNorm with scale=`self.scale * 1/√hidden` (fused into the weight).
/// 2. Linear projection to `num_experts` logits.
/// 3. `top_k` selection + softmax (only over the selected logits).
/// 4. Multiply each selected weight by `per_expert_scale[idx]`.
pub struct Router {
    /// Learnable per-hidden-feature scale vector (shape `[hidden]`). Loaded
    /// from the `router.scale` checkpoint tensor. NOT baked into a weighted
    /// RMSNorm — kept as a separate multiply so the op graph matches
    /// mlx-vlm's Router exactly.
    pub scale: Array,
    pub proj: Weight,
    pub per_expert_scale: Array,
    pub num_experts: usize,
    pub top_k: usize,
    /// Precomputed `1/√hidden`.
    root_size: f32,
    /// RMS norm epsilon (fed into `rms_norm_no_scale`).
    eps: f32,
    /// Legacy `norm: RmsNorm` field kept for loader backward-compat
    /// (takes `router.scale` into `.weight`) but unused by forward.
    /// Slated for removal once the loader consistently writes `scale`.
    #[allow(dead_code)]
    pub norm: RmsNorm,
    /// Fast-path cache of `(scale * root_size)` pre-cast to the activation
    /// dtype (bf16), built once on the first `forward_fast` call — the fused
    /// `mx.fast.rms_norm` weight (mirrors the golden's `scale * self._root_size`
    /// at `gemma4_text.py:130`). Immutable post-load, so caching is numerically
    /// identical and removes the per-token f32 promotion + glue ops.
    scale_root_fast: parking_lot::Mutex<Option<Array>>,
}

impl Router {
    pub fn new(hidden: usize, num_experts: usize, top_k: usize, eps: f32) -> Self {
        Self {
            scale: Array::ones::<f32>(&[hidden as i32]).expect("mlx op"),
            norm: RmsNorm::new(hidden, eps),
            proj: Weight::plain(
                Array::zeros::<f32>(&[num_experts as i32, hidden as i32]).expect("mlx op"),
            ),
            per_expert_scale: Array::ones::<f32>(&[num_experts as i32]).expect("mlx op"),
            num_experts,
            top_k,
            root_size: 1.0 / (hidden as f32).sqrt(),
            eps,
            scale_root_fast: parking_lot::Mutex::new(None),
        }
    }

    /// Returns `(top_k_indices, top_k_weights)` each shaped `[B, S, top_k]`.
    ///
    /// Matches mlx-vlm's `gemma4/language.py::Router.__call__` op-for-op:
    ///   1. `RMSNormNoScale(x)` — normalize without weight
    ///   2. `x * root_size` — 1/√hidden
    ///   3. `x * scale` — learnable per-feature gate
    ///   4. `proj` → `[B, S, num_experts]` logits
    ///   5. `softmax(scores, axis=-1)` over ALL experts
    ///   6. `argpartition` → top-k indices (we use argsort; equivalent)
    ///   7. gather probs at top-k indices
    ///   8. renormalize (divide by sum) so weights over top-k sum to 1
    ///   9. multiply by `per_expert_scale[top_k_indices]`
    ///
    /// Previous version fused steps 1-3 into a weighted `RmsNorm::forward(x)`
    /// then `* root_size`. That is mathematically equivalent but produces
    /// a different fp op graph — observed to cause a 1-token divergence
    /// on 26B MoE greedy vs mlx-vlm at position 23 where top-1 was a
    /// near-tie. This matches mlx-vlm's graph exactly.
    pub fn forward(&self, x: &Array) -> (Array, Array) {
        // 1-3: norm → * root_size → * scale.
        let h = rms_norm_no_scale(x, self.eps);
        let h = h
            .multiply(&Array::from_f32(self.root_size))
            .expect("mlx op");
        let h = h.multiply(&self.scale).expect("mlx op");
        self.select(x, &h)
    }

    /// `PIO_MLX_FAST` router — mirrors the golden's norm exactly
    /// (`gemma4_text.py:130`): `mx.fast.rms_norm(x, scale * root_size, eps)`,
    /// ONE fused bf16 kernel, instead of the default path's hand-rolled
    /// f32 chain (`rms_norm_no_scale` → `* root_size` (f32 array) → `* scale`),
    /// which promotes the whole `[B,S,hidden]` router input to **f32** (a bf16
    /// array × an `Array::from_f32` scalar promotes), then runs a 5-op glue
    /// reduction every layer every token. The fused kernel keeps the trunk in
    /// bf16 (the golden's weight `scale * root_size` is bf16) and collapses the
    /// reduction into the single `mx.fast.rms_norm` Metal kernel. The expert
    /// SELECTION math below (`select`) is intentionally left as the frozen
    /// softmax-over-all + renormalize variant — swapping it for the golden's
    /// softmax-over-top-k broke turn-5 context recall (see `select`), and it is
    /// the *selection* that must stay frozen, not the norm dtype/fusion.
    ///
    /// Cast `scale * root_size` to the activation dtype so the fused kernel
    /// returns bf16 (mlx-rs `fast::rms_norm` returns the WEIGHT's dtype). The
    /// weight is cached on first call (immutable post-load).
    pub fn forward_fast(&self, x: &Array) -> (Array, Array) {
        let target = x.dtype();
        let w = {
            let mut slot = self.scale_root_fast.lock();
            let stale = slot.as_ref().map(|w| w.dtype() != target).unwrap_or(true);
            if stale {
                let fused = self
                    .scale
                    .multiply(&Array::from_f32(self.root_size))
                    .expect("mlx op")
                    .as_dtype(target)
                    .expect("mlx op");
                *slot = Some(fused);
            }
            slot.as_ref().expect("scale_root_fast set above").clone()
        };
        let h = mlx_rs::fast::rms_norm(x, &w, self.eps).expect("mlx op: fast rms_norm router");
        self.select(x, &h)
    }

    /// Shared expert-selection tail (steps 4-9). `h` is the normed+scaled
    /// router input; `x` is unused beyond shape but kept for signature parity.
    fn select(&self, _x: &Array, h: &Array) -> (Array, Array) {
        // 4: project. Shape [B, S, num_experts].
        let scores = self.proj.matmul_transpose(h);

        // 5: softmax over ALL experts.
        let router_probs = mlx_rs::ops::softmax_axes(&scores, &[-1], None).expect("mlx op");

        // 6: top-k indices. mlx-rs has no argpartition; argsort is
        // equivalent for top-k selection (last k of ascending sort).
        let sorted_idx = mlx_rs::ops::argsort_axis(&scores, -1).expect("mlx op");
        let n_exp = self.num_experts as i32;
        let k = self.top_k as i32;
        let top_k_idx = sorted_idx.index((.., .., (n_exp - k)..n_exp));

        // 7-8: gather top-k probs, renormalize so they sum to 1 over top-k.
        let top_k_probs = router_probs
            .take_along_axis(&top_k_idx, -1)
            .expect("mlx op");
        let sum = top_k_probs.sum_axis(-1, true).expect("mlx op");
        let weights = top_k_probs.divide(&sum).expect("mlx op");

        // 9: per-expert scale multiply.
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
        self.forward_dense_sum(x, indices, weights)
    }

    /// `PIO_MLX_FAST` expert dispatch. Same math as `forward`, but mirrors
    /// mlx-lm's `SwitchGLU.__call__` (`switch_layers.py:176-199`) exactly so
    /// `gather_qmm` runs the (1×H)×(H×moe) matmuls as a single broadcast over
    /// the `top_k` expert dimension instead of materializing one (1×H) row per
    /// (token, expert) slot (the prior `[M,1,H]` + explicit `lhs_indices`
    /// gather, which had poor GPU occupancy at M=1 decode).
    ///
    /// Kept separate from `forward` so the default (non-fast) path's
    /// `forward_gather_qmm` op graph stays byte-identical.
    pub fn forward_fast(&self, x: &Array, indices: &Array, weights: &Array) -> Array {
        if let Some(out) = self.forward_switch_glu(x, indices, weights) {
            return out;
        }
        self.forward_dense_sum(x, indices, weights)
    }

    /// Dense-sum fallback over all experts. Only reachable when weights are
    /// plain float (PIO_FORCE_DEQUANT=1 or non-quantized checkpoint). Kept for
    /// diagnostic parity with the reference impl. Shared by both paths since it
    /// is never hit with a quantized checkpoint.
    fn forward_dense_sum(&self, x: &Array, indices: &Array, weights: &Array) -> Array {
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
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
    fn forward_gather_qmm(&self, x: &Array, indices: &Array, weights: &Array) -> Option<Array> {
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
            ) => (
                (gw, gs, gb, *gg, *gbi),
                (uw, us, ub),
                (dw, ds, db, *dg, *dbi),
            ),
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

    /// `PIO_MLX_FAST` fused expert dispatch — a 1:1 port of mlx-lm's
    /// `SwitchGLU.__call__` (`mlx_lm/models/switch_layers.py:176-199`) and
    /// `QuantizedSwitchLinear.__call__` (`switch_layers.py:75-90`). Returns
    /// `None` when any projection isn't quantized (fall through to dense-sum).
    ///
    /// Key difference vs `forward_gather_qmm`: we do NOT build/pass an
    /// `lhs_indices` gather on the x side. Instead, following mlx-lm, `x` is
    /// shaped `[..., 1, 1, H]` via `expand_dims(x, (-2, -3))` and passed with
    /// `rhs_indices` only; `gather_qmm` broadcasts the single x row across the
    /// `top_k` expert axis selected by `rhs_indices`. That avoids the prior
    /// per-(token,expert) `[1,1,H]` one-row matmul layout (poor M=1 occupancy).
    ///
    /// Output reduction mirrors mlx-lm: `(weights * y).sum(-2)` over the
    /// `top_k` axis (mlx-lm's `swiglu` activation is `up * gelu(gate)`; our
    /// `gelu_approximate` matches `nn.GELU(approx="tanh")`, the same approx
    /// gemma4_text.py uses).
    fn forward_switch_glu(&self, x: &Array, indices: &Array, weights: &Array) -> Option<Array> {
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
            ) => (
                (gw, gs, gb, *gg, *gbi),
                (uw, us, ub),
                (dw, ds, db, *dg, *dbi),
            ),
            _ => return None,
        };

        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];
        let k_top = indices.shape()[2];

        // mlx-lm: `x = mx.expand_dims(x, (-2, -3))` → x: [B, S, 1, 1, H].
        // The last-two axes form a (1×H) matmul; `gather_qmm` broadcasts the
        // size-1 `top_k` axis against `rhs_indices`'s `top_k` entries.
        let x_exp = x.reshape(&[batch, seq, 1, 1, hidden]).expect("mlx op");

        // rhs_indices = expert ids, shape [B, S, k] (mlx-lm passes `indices`
        // straight through). gather_qmm gathers w's leading expert axis.
        let rhs = indices.as_type::<u32>().expect("mlx op");

        // mlx-lm sorts only when `indices.size >= 64`; at M=1 decode (k≈4)
        // it stays unsorted, so we pass sorted_indices=false (gather_qmm's
        // default) and no _gather_sort, matching the decode hot path exactly.

        // Gate / up branch: gather_qmm(x, W, scales, biases, rhs_indices=idx,
        // transpose=true). Result: [B, S, k, 1, moe].
        let gate_out = gather_qmm(
            &x_exp,
            gate.0,
            gate.1,
            Some(gate.2),
            None::<&Array>,
            Some(&rhs),
            Some(true),
            Some(gate.3),
            Some(gate.4),
            None,
        )
        .expect("mlx op");
        let up_out = gather_qmm(
            &x_exp,
            up.0,
            up.1,
            Some(up.2),
            None::<&Array>,
            Some(&rhs),
            Some(true),
            Some(gate.3),
            Some(gate.4),
            None,
        )
        .expect("mlx op");

        // swiglu(gate, x) = up * gelu(gate). [B, S, k, 1, moe].
        // Dtype-preserving GELU (see `gemma4_fast::gelu_approx_fast`): the
        // mlx-rs `gelu_approximate` promotes bf16 → f32, which would leak f32
        // into `h2` and force the per-layer recast. Keep it bf16.
        let gated = super::gemma4_fast::gelu_approx_fast(&gate_out);
        let mix = gated.multiply(&up_out).expect("mlx op");

        // Down branch: gather_qmm(mix, Wd, ..., rhs_indices=idx). [B, S, k, 1, H].
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
        .expect("mlx op");

        // mlx-lm returns `x.squeeze(-2)` → [B, S, k, H]; the caller weights
        // and sums over `top_k`. We fold that reduction in here to keep the
        // `Experts::forward` contract ([B,S,H]): `(weights * y).sum(-2)`.
        let y = down_out
            .reshape(&[batch, seq, k_top, hidden])
            .expect("mlx op");
        let w = weights.reshape(&[batch, seq, k_top, 1]).expect("mlx op");
        let weighted = y.multiply(&w).expect("mlx op"); // [B, S, k, H]
        let summed = weighted.sum_axis(2, false).expect("mlx op"); // [B, S, H]

        Some(summed)
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
