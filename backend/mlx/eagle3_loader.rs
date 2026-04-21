//! Loader for EAGLE-3 draft-model checkpoints.
//!
//! Reads a HuggingFace-style EAGLE-3 bundle (single `model.safetensors`
//! + `config.json`) from a directory and produces an `EagleDraftModel`
//! with all weights verified against the config shapes. Used by the
//! MLX backend when `GenSpec::speculative = SpeculativeMode::Eagle3 { ... }`.
//!
//! Weight names match the RedHatAI upstream layout:
//! - `embed_tokens.weight`                  → target-vocab embeddings
//! - `fc.weight`                            → aux-state projection
//! - `layers.0.{self_attn,mlp,*norm*}.weight` → single decoder layer
//! - `norm.weight`                          → final rmsnorm
//! - `lm_head.weight`                       → draft-vocab head
//! - `d2t`                                  → draft→target token id map (i64)
//! - `t2d`                                  → target→draft availability mask (bool)

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use mlx_rs::Array;

use super::model::eagle3::{Eagle3Config, EagleDraftModel};

/// Load an EAGLE-3 checkpoint from `dir`, parsing `config.json` and
/// `model.safetensors` into a verified `EagleDraftModel`.
pub fn load_from_dir(dir: &Path) -> Result<EagleDraftModel> {
    let cfg_path = dir.join("config.json");
    let cfg = Eagle3Config::from_path(&cfg_path)?;
    let st_path = dir.join("model.safetensors");
    load(cfg, &st_path)
}

/// Load weights from `safetensors_path` into an `EagleDraftModel` with
/// the given config.
pub fn load(cfg: Eagle3Config, safetensors_path: &Path) -> Result<EagleDraftModel> {
    if !safetensors_path.exists() {
        return Err(anyhow!(
            "eagle3 safetensors not found: {}",
            safetensors_path.display()
        ));
    }
    let tensors: HashMap<String, Array> = Array::load_safetensors(safetensors_path)
        .map_err(|e| anyhow!("load eagle3 safetensors: {}", e))?;

    let take =
        |key: &str| -> Result<Array> {
            tensors
                .get(key)
                .cloned()
                .ok_or_else(|| anyhow!("eagle3 checkpoint missing tensor: {key}"))
        };

    let model = EagleDraftModel {
        embed_tokens: take("embed_tokens.weight")?,
        fc: take("fc.weight")?,
        input_layernorm: take("layers.0.input_layernorm.weight")?,
        hidden_norm: take("layers.0.hidden_norm.weight")?,
        post_attention_layernorm: take("layers.0.post_attention_layernorm.weight")?,
        q_proj: take("layers.0.self_attn.q_proj.weight")?,
        k_proj: take("layers.0.self_attn.k_proj.weight")?,
        v_proj: take("layers.0.self_attn.v_proj.weight")?,
        o_proj: take("layers.0.self_attn.o_proj.weight")?,
        mlp_gate_proj: take("layers.0.mlp.gate_proj.weight")?,
        mlp_up_proj: take("layers.0.mlp.up_proj.weight")?,
        mlp_down_proj: take("layers.0.mlp.down_proj.weight")?,
        norm: take("norm.weight")?,
        lm_head: take("lm_head.weight")?,
        d2t: take("d2t")?,
        t2d: take("t2d")?,
        cfg,
    };

    model
        .verify_shapes()
        .context("eagle3 shape verification")?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end loader test — loads the RedHatAI Gemma 4 26B EAGLE-3
    /// checkpoint from `$TEST_EAGLE3_DIR` and asserts every tensor has
    /// the expected shape. Gated on the env var so CI doesn't need the
    /// 1.85 GB file; point at a local download to run locally:
    ///
    /// ```sh
    /// TEST_EAGLE3_DIR=~/models/eagle3-26b cargo test -p pio-core \
    ///   --features backend-mlx --release eagle3_loader \
    ///   -- --ignored --nocapture
    /// ```
    /// End-to-end forward-pass smoke test. Loads the real EAGLE-3
    /// checkpoint, runs one draft step with dummy aux hidden states,
    /// verifies the returned token id is in target vocab range. Doesn't
    /// assert numerical correctness (that requires a real target's aux
    /// states — handled by integration test once task #19 lands) but
    /// proves the whole forward pipeline compiles, dispatches, and
    /// returns a valid id without shape / dtype errors.
    #[test]
    #[ignore = "requires TEST_EAGLE3_DIR — runs ~1GB forward pass"]
    fn forward_step_produces_valid_target_token() {
        use super::super::model::rope::RotaryEmbedding;
        use mlx_rs::Array;

        let Ok(dir) = std::env::var("TEST_EAGLE3_DIR") else {
            eprintln!("TEST_EAGLE3_DIR not set — skipping");
            return;
        };
        let model = load_from_dir(&std::path::PathBuf::from(dir)).expect("load");
        let cfg = &model.cfg.transformer_layer_config;
        let h = cfg.hidden_size;
        let aux_dim = model.cfg.aux_concat_dim();

        // Dummy aux: [1, 1, 3*H] of small random values.
        let data: Vec<f32> = (0..aux_dim).map(|i| (i as f32) * 1e-4).collect();
        let aux = Array::from_slice(&data, &[1, 1, aux_dim as i32]);

        let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta);
        let last_tok: u32 = 42; // arbitrary target-vocab token
        let (next_tok, prenorm) = model
            .forward_step_argmax(last_tok, &aux, 0, &rope)
            .expect("forward");
        assert!(
            (next_tok as usize) < cfg.vocab_size,
            "predicted target id {next_tok} out of target vocab ({})",
            cfg.vocab_size
        );
        assert_eq!(
            prenorm.shape(),
            &[1, 1, h as i32],
            "prenorm hidden shape mismatch"
        );
        println!("eagle3 forward ok: draft predicted target token {next_tok}");
    }

    #[test]
    #[ignore = "requires TEST_EAGLE3_DIR pointing at an EAGLE-3 checkpoint"]
    fn loads_and_verifies_shapes() {
        let Ok(dir) = std::env::var("TEST_EAGLE3_DIR") else {
            eprintln!("TEST_EAGLE3_DIR not set — skipping");
            return;
        };
        let path = std::path::PathBuf::from(dir);
        let model = load_from_dir(&path).expect("load eagle3");
        assert!(model.verify_shapes().is_ok());
        // Spot-check: for the Gemma 4 26B draft, these are the published
        // shapes — catching a checkpoint regression early.
        assert_eq!(model.cfg.draft_vocab_size, 32000);
        assert_eq!(model.cfg.transformer_layer_config.hidden_size, 2816);
        assert_eq!(model.cfg.transformer_layer_config.head_dim, 256);
        assert_eq!(model.embed_tokens.shape(), &[262144, 2816]);
        assert_eq!(model.fc.shape(), &[2816, 8448]);
        assert_eq!(model.lm_head.shape(), &[32000, 2816]);
        assert_eq!(model.d2t.shape(), &[32000]);
        assert_eq!(model.t2d.shape(), &[262144]);
    }
}
