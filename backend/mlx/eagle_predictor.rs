//! MLX-specific EAGLE-3 speculative predictor.
//!
//! Wraps a loaded [`EagleDraftModel`] behind the cross-backend
//! [`SpeculativePredictor`] trait. Each `draft_with_context()` call
//! runs one greedy forward step of the draft model with the target's
//! auxiliary hidden states and the last accepted token, producing a
//! single next-token prediction.
//!
//! Current scope: K=1 (single-token draft per call). K>1 autoregressive
//! drafting would require running the draft forward K times with the
//! prenorm hidden state feeding back as aux — straightforward to add
//! on top of this primitive but deferred until we've validated K=1
//! produces useful speedup numbers end-to-end.

use mlx_rs::Array;

use super::model::eagle3::{Eagle3Config, EagleDraftModel};
use super::model::rope::RotaryEmbedding;
use crate::gen2::backend::common::speculative::{DraftContext, SpeculativePredictor};

/// Speculative predictor backed by a loaded EAGLE-3 draft model.
///
/// Holds:
///  - the draft-model weights (owned directly — MLX `Array` isn't
///    `Sync`, so we can't share across sessions via `Arc`; each puller
///    owns its own copy. The arrays themselves are MLX-refcounted so
///    the clone-to-move cost is negligible)
///  - a RoPE table sized to the draft's config (theta=10000, head_dim
///    from `transformer_layer_config`)
///  - the target-layer ids we expect in `DraftContext::aux_hidden_states`
///    (surfaced via [`Self::aux_layer_ids`] so the puller knows which
///    layers to stash from the target)
pub struct EagleDraftPredictor {
    model: EagleDraftModel,
    rope: RotaryEmbedding,
    aux_layer_ids: Vec<usize>,
}

impl EagleDraftPredictor {
    pub fn new(model: EagleDraftModel) -> Self {
        let cfg: &Eagle3Config = &model.cfg;
        let layer_cfg = &cfg.transformer_layer_config;
        let rope = RotaryEmbedding::new(
            layer_cfg.head_dim,
            layer_cfg.max_position_embeddings,
            layer_cfg.rope_theta,
        );
        let aux_layer_ids = cfg
            .eagle_aux_hidden_state_layer_ids
            .clone()
            .unwrap_or_default();
        Self {
            model,
            rope,
            aux_layer_ids,
        }
    }
}

impl SpeculativePredictor for EagleDraftPredictor {
    fn draft(&mut self, _max: usize) -> Vec<u32> {
        // Without aux states we can't draft. Return empty — caller
        // (puller) should route through `draft_with_context` when it
        // has the aux states in hand. Empty draft = speculative path
        // falls back to single-token decode for that step.
        Vec::new()
    }

    fn draft_with_context(&mut self, ctx: &DraftContext<'_>, _max: usize) -> Vec<u32> {
        if ctx.aux_hidden_states.len() != self.aux_layer_ids.len() {
            tracing::warn!(
                got = ctx.aux_hidden_states.len(),
                want = self.aux_layer_ids.len(),
                "EAGLE-3 aux state count mismatch; skipping draft this step"
            );
            return Vec::new();
        }
        // Concatenate the per-layer aux states along the last axis.
        // Expected: each is [1, 1, H_target]; output [1, 1, num_aux * H].
        let refs: Vec<&Array> = ctx.aux_hidden_states.iter().collect();
        let aux = match mlx_rs::ops::concatenate_axis(&refs, -1) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(?e, "EAGLE-3 aux concat failed; no draft this step");
                return Vec::new();
            }
        };
        match self
            .model
            .forward_step_argmax(ctx.last_token, &aux, ctx.pos, &self.rope)
        {
            Ok((target_id, _prenorm)) => vec![target_id],
            Err(e) => {
                tracing::warn!(?e, "EAGLE-3 forward failed; no draft this step");
                Vec::new()
            }
        }
    }

    fn observe(&mut self, _token: u32) {
        // Token history isn't needed for EAGLE-3 — the draft model
        // re-reads the target's aux hidden states each step, which
        // already encode the full history via the KV cache.
    }

    fn name(&self) -> &'static str {
        "eagle3"
    }

    fn needs_context(&self) -> bool {
        true
    }

    fn aux_layer_ids(&self) -> &[usize] {
        &self.aux_layer_ids
    }
}
