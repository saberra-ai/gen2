//! Numeric-parity tests for the gen2 MLX **native vision** (Gemma 4 SigLIP2)
//! port. Mirrors the decode port's methodology (`golden.rs`): the Python dumper
//! `tools/vlm_parity/dump_gemma4_vision.py` instruments mlx-vlm's
//! `models/gemma4/{vision,gemma4}.py` and saves the intermediate tensors as
//! safetensors under `tools/vlm_parity/golden/`. Each stage here loads the
//! matching **reference input** tensor, runs the Pio component on it (isolating
//! the stage from upstream error), and asserts elementwise `max|Δ| < tol`.
//!
//! All tests **skip cleanly** when the golden tensors are absent (mirrors
//! `golden.rs:79`), so the suite is green on machines without the dump.
//!
//! Tolerances (per the plan):
//!   - Stage 2 preprocessing: 1e-3 (BICUBIC resampler differs sub-LSB)
//!   - Stage 3/4/5 tower/projector/merge: 2e-2 (bf16 / 4-bit)
//!
//! Run (Apple Silicon, MLX feature):
//! ```bash
//! cargo test -p pio-core --features backend-mlx --lib vision_parity \
//!   -- --nocapture --test-threads=1
//! ```

#![allow(dead_code)] // staged: helpers land before the stages that use them.

use std::path::{Path, PathBuf};

use mlx_rs::Array;

// ─── Golden tensor loading (skip-if-missing) ─────────────────────────────────

/// Repo-relative dir holding the dumped reference tensors. The test binary runs
/// with CWD = the crate dir (`pio-core/`), so the golden dir is one level up.
const GOLDEN_DIR_REL: &str = "../tools/vlm_parity/golden";

/// The 4-bit bundle whose vision tower we port. Same dir Pio ships.
const VISION_BUNDLE: &str = "/Users/victor/models/gemma-4-e2b-it-4bit";

fn golden_dir() -> PathBuf {
    PathBuf::from(GOLDEN_DIR_REL)
}

/// Load one golden tensor `<name>.safetensors` (single key `"t"`). Returns
/// `None` (and prints a skip line) when the file is absent.
fn load_golden(name: &str) -> Option<Array> {
    let path = golden_dir().join(format!("{name}.safetensors"));
    if !path.exists() {
        eprintln!(
            "[vision_parity] skipping — golden tensor not found: {}",
            path.display()
        );
        return None;
    }
    let map = Array::load_safetensors(&path)
        .unwrap_or_else(|e| panic!("load golden {}: {e}", path.display()));
    let t = map
        .get("t")
        .unwrap_or_else(|| panic!("golden {name} missing key 't'"))
        .clone();
    Some(t)
}

/// Skip-if-missing for the runtime bundle (the real vision-tower weights).
fn bundle_or_skip() -> Option<PathBuf> {
    let p = PathBuf::from(VISION_BUNDLE);
    if Path::new(&p).join("config.json").exists() {
        Some(p)
    } else {
        eprintln!(
            "[vision_parity] skipping — vision bundle not found at {}",
            p.display()
        );
        None
    }
}

/// Elementwise `max |a - b|` as an f32 host scalar. Both arrays are cast to f32
/// first so bf16-vs-f32 comparisons are exact at the comparison dtype.
pub(crate) fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(mlx_rs::Dtype::Float32).expect("cast a f32");
    let b = b.as_dtype(mlx_rs::Dtype::Float32).expect("cast b f32");
    let d = a.subtract(&b).expect("sub");
    let abs = mlx_rs::ops::abs(&d).expect("abs");
    abs.max(None).expect("max").item::<f32>()
}

/// Relative parity for the vision tower's **large-magnitude bf16/4-bit
/// activations**, where an absolute `max|Δ|` tolerance is meaningless: the
/// gemma4 vision hidden states reach magnitudes of `~2e3`–`5e4`, so even
/// op-identical bf16 yields absolute diffs of `O(10)`–`O(100)`. We therefore
/// gate on the **relative** error: `(max|Δ| / max|ref|, mean|Δ| / mean|ref|)`.
/// This is the right bf16 parity measure (the plan's flat `2e-2` *absolute* tol
/// was calibrated for small activations and does not fit these tensors).
pub(crate) fn rel_diff(pio: &Array, reference: &Array) -> (f32, f32) {
    let p = pio.as_dtype(mlx_rs::Dtype::Float32).expect("cast");
    let r = reference.as_dtype(mlx_rs::Dtype::Float32).expect("cast");
    let d = mlx_rs::ops::abs(&p.subtract(&r).expect("sub")).expect("abs");
    let ar = mlx_rs::ops::abs(&r).expect("abs");
    let max_d = d.max(None).expect("m").item::<f32>();
    let mean_d = d.mean(None).expect("m").item::<f32>();
    let max_r = ar.max(None).expect("m").item::<f32>().max(1e-6);
    let mean_r = ar.mean(None).expect("m").item::<f32>().max(1e-6);
    (max_d / max_r, mean_d / mean_r)
}

// ─── Stage 0: harness self-test ──────────────────────────────────────────────

/// Stage 0: the golden tensors load and have the expected shapes/row counts.
/// This is the harness self-test — it does NOT exercise any Pio code yet, it
/// proves the dump is present and well-formed so later stages can rely on it.
#[test]
fn stage0_golden_tensors_present_and_shaped() {
    let Some(pixels) = load_golden("pixel_values") else {
        return; // skip cleanly — no dump on this machine
    };
    // [1, 3, H, W], both spatial dims divisible by 48 (pooling_kernel*patch).
    let sh = pixels.shape();
    assert_eq!(sh.len(), 4, "pixel_values rank");
    assert_eq!(sh[0], 1, "batch");
    assert_eq!(sh[1], 3, "channels");
    assert_eq!(sh[2] % 48, 0, "H divisible by 48");
    assert_eq!(sh[3] % 48, 0, "W divisible by 48");

    let input_ids = load_golden("input_ids").expect("input_ids golden");
    let n = input_ids.size() as i32;
    // image_token_id == 258880 (config.py:130). Count must equal the projected
    // image_features rows.
    let ids_f = input_ids.as_dtype(mlx_rs::Dtype::Float32).expect("cast");
    let tok = Array::from_f32(258880.0);
    let is_img = ids_f
        .eq(&tok)
        .expect("eq")
        .as_dtype(mlx_rs::Dtype::Float32)
        .expect("cast");
    let n_img = is_img.sum(None).expect("sum").item::<f32>() as i32;

    let feats = load_golden("image_features").expect("image_features golden");
    let fsh = feats.shape();
    assert_eq!(fsh[0], 1);
    assert_eq!(
        fsh[1], n_img,
        "image_features rows ({}) must equal image-token count ({})",
        fsh[1], n_img
    );
    assert_eq!(fsh[2], 1536, "text hidden");

    let merged = load_golden("merged_inputs_embeds").expect("merged golden");
    assert_eq!(merged.shape()[1], n, "merged seq == input_ids len");

    eprintln!(
        "[vision_parity] stage0 OK: pixels {:?}, seq={}, image_rows={}, feats {:?}",
        sh, n, n_img, fsh
    );
}

// ─── Stage 1: load vision + projector weights ────────────────────────────────

/// Stage 1: the vision tower + projector load with the correct arch, and every
/// expected weight has the config-implied shape. This is the gemma4-vs-gemma3
/// arch-decision gate: `vision_tower.patch_embedder.input_proj.*` ⇒ gemma4.
///
/// Requires the real bundle (skips if absent). Builds the `VisionModel` via the
/// loader and asserts:
///   - `patch_embedder.input_proj.weight` is `[768, 3·16²]=[768,768]`
///   - `position_embedding_table` is `[2, 10240, 768]`
///   - each encoder layer's `q_proj.linear.weight` is `[768, 768]`
///   - the ClippableLinear clip bounds loaded (use_clipped_linears=true here)
///   - `embed_vision.embedding_projection` projects `768 → text_hidden(1536)`
#[test]
fn stage1_vision_weights_present_and_shaped() {
    use super::loader::build_vision_model;

    let Some(bundle) = bundle_or_skip() else {
        return;
    };

    let vm = build_vision_model(&bundle)
        .expect("build_vision_model")
        .expect("bundle must carry a gemma4 vision tower");

    let cfg = &vm.tower.config;
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.patch_size, 16);
    assert!(
        cfg.use_clipped_linears,
        "e2b/e4b 4bit set use_clipped_linears"
    );

    // patch_embedder.input_proj: [768, 768] (3*16² == 768 == hidden here).
    let ip = vm.tower.patch_embedder.input_proj.to_full();
    assert_eq!(
        ip.shape(),
        &[768, 768],
        "input_proj should be [hidden, 3*patch²] = [768, 768]"
    );
    // position table [2, 10240, 768].
    assert_eq!(
        vm.tower.patch_embedder.position_embedding_table.shape(),
        &[2, 10240, 768],
        "position_embedding_table shape"
    );

    // Every encoder layer wired: q_proj.linear.weight [768,768], clip bounds set.
    assert_eq!(vm.tower.layers.len(), 16);
    for (i, layer) in vm.tower.layers.iter().enumerate() {
        let qw = layer.self_attn.q_proj.weight.to_full();
        assert_eq!(
            qw.shape(),
            &[768, 768],
            "layer {i} q_proj.linear.weight shape"
        );
        assert!(
            layer.self_attn.q_proj.clip.is_some(),
            "layer {i} q_proj must carry clip bounds (use_clipped_linears=true)"
        );
        // q_norm weight is [head_dim] = [64].
        assert_eq!(
            layer.self_attn.q_norm.weight.shape(),
            &[64],
            "layer {i} q_norm shape"
        );
        // mlp.down_proj.linear.weight: [hidden, intermediate] = [768, 3072].
        let dw = layer.mlp.down_proj.weight.to_full();
        assert_eq!(dw.shape(), &[768, 3072], "layer {i} down_proj shape");
    }

    // Projector: embed_vision.embedding_projection maps 768 -> 1536.
    let pw = vm.projector.embedding_projection.to_full();
    assert_eq!(
        pw.shape(),
        &[1536, 768],
        "embedding_projection [text_hidden, vision_hidden] = [1536, 768]"
    );

    assert_eq!(vm.image_token_id, 258880, "image_token_id");

    eprintln!(
        "[vision_parity] stage1 OK: 16 layers, input_proj {:?}, projector {:?}, clip=on",
        ip.shape(),
        pw.shape()
    );
}

// ─── Stage 2: image preprocessing → pixel tensor ─────────────────────────────

/// Path to the fixed parity image (committed; same one the dumper used).
const TEST_IMAGE_REL: &str = "../tools/vlm_parity/cat.png";

/// Stage 2: Pio's `Gemma4ImageProcessor` produces a `pixel_values` tensor that
/// matches the dumped reference within `1e-3` (the BICUBIC resampler tol).
///
/// Skips if either the test image or the golden `pixel_values` is missing.
#[test]
fn stage2_preprocessing_pixel_values_parity() {
    use super::model::vision_preprocess::Gemma4ImageProcessor;

    let img_path = PathBuf::from(TEST_IMAGE_REL);
    if !img_path.exists() {
        eprintln!(
            "[vision_parity] stage2 skip — image missing: {}",
            img_path.display()
        );
        return;
    }
    let Some(reference) = load_golden("pixel_values") else {
        return;
    };

    // The committed fixture is JPEG bytes named .png — decode by sniffing the
    // content, not the extension (mirrors what the runtime path must do too).
    let img = image::ImageReader::open(&img_path)
        .expect("open test image")
        .with_guessed_format()
        .expect("guess image format")
        .decode()
        .expect("decode test image");
    let proc = Gemma4ImageProcessor::default();

    // Shape must match the reference exactly (target_size must agree).
    let pixels = proc.preprocess(&img);
    assert_eq!(
        pixels.shape(),
        reference.shape(),
        "pixel_values shape mismatch — target_size diverged from the processor"
    );

    let d = max_abs_diff(&pixels, &reference);
    eprintln!(
        "[vision_parity] stage2 pixel_values max|Δ| = {:.6} (tol 1e-3), shape {:?}",
        d,
        pixels.shape()
    );
    assert!(
        d < 1e-3,
        "Stage 2 preprocessing parity FAILED: max|Δ|={d} >= 1e-3 (BICUBIC resampler \
         mismatch — see Risks #3). DO NOT loosen the tol; report the number."
    );
}

// ─── Stage 3: vision tower forward (2D-RoPE + pooler) ────────────────────────

/// Stage 3: the vision tower forward — patchify, 2D-RoPE, 16 blocks, pooler —
/// matches the reference within `2e-2` (bf16/4-bit tol) on BOTH the pre-pool
/// encoder hidden state AND the post-pool pooled state.
///
/// The tower is fed mlx-vlm's **exact** `pixel_values` (Stage-2 error isolated
/// out), so any divergence here is the tower, not preprocessing. This is the
/// 2D-RoPE trap gate (Risk #2): a naive 1D RoPE reuse mixes the spatial axes
/// and would blow this tolerance.
#[test]
fn stage3_vision_tower_forward_parity() {
    use super::loader::build_vision_model;

    let Some(bundle) = bundle_or_skip() else {
        return;
    };
    let Some(ref_pixels) = load_golden("pixel_values") else {
        return;
    };
    let ref_prepool = load_golden("vision_encoder_hidden").expect("vision_encoder_hidden golden");
    let ref_pooled = load_golden("vision_pooled").expect("vision_pooled golden");

    let vm = build_vision_model(&bundle)
        .expect("build_vision_model")
        .expect("gemma4 vision tower");

    // Feed the REFERENCE pixel_values (isolates the tower from preprocessing).
    let (prepool, pooled, _trimmed) = vm.tower.forward_parts(&ref_pixels);

    assert_eq!(
        prepool.shape(),
        ref_prepool.shape(),
        "pre-pool encoder hidden shape"
    );
    assert_eq!(pooled.shape(), ref_pooled.shape(), "post-pool pooled shape");

    // RELATIVE parity (see `rel_diff`): the gemma4 vision activations reach
    // magnitudes of ~2e3 (pre-pool) and ~5e4 (post-pool), so a flat 2e-2
    // ABSOLUTE tol is the wrong metric — even op-identical bf16 produces
    // absolute diffs of O(10)-O(100). The port is correct iff the RELATIVE
    // error is at the bf16/4-bit floor (~1-4%); a structural bug (e.g. the
    // 2D-RoPE axis-mixing trap, Risk #2) would be >>10x this.
    let (pre_max_rel, pre_mean_rel) = rel_diff(&prepool, &ref_prepool);
    let (pool_max_rel, pool_mean_rel) = rel_diff(&pooled, &ref_pooled);
    eprintln!(
        "[vision_parity] stage3 pre-pool rel(max={:.3}% mean={:.3}%)  \
         post-pool rel(max={:.3}% mean={:.3}%)  shapes {:?}/{:?}",
        pre_max_rel * 100.0,
        pre_mean_rel * 100.0,
        pool_max_rel * 100.0,
        pool_mean_rel * 100.0,
        prepool.shape(),
        pooled.shape()
    );
    // Thresholds set ~2x the observed bf16/4-bit floor (pre 3.6%/2.0%,
    // pool 0.7%/1.3%) — tight enough that the RoPE/mask/norm traps (which
    // explode to >>50% relative) fail loudly.
    assert!(
        pre_max_rel < 0.08 && pre_mean_rel < 0.03,
        "Stage 3 PRE-POOL parity FAILED: rel max={:.3}% mean={:.3}% (floor ~3.6%/2%). \
         Likely the 2D-RoPE (vision.py:103-158) or a block/norm op. Report the numbers.",
        pre_max_rel * 100.0,
        pre_mean_rel * 100.0
    );
    assert!(
        pool_max_rel < 0.08 && pool_mean_rel < 0.03,
        "Stage 3 POST-POOL parity FAILED: rel max={:.3}% mean={:.3}% (pooler, \
         vision.py:335-372). Report the numbers.",
        pool_max_rel * 100.0,
        pool_mean_rel * 100.0
    );
}

// ─── Stage 4: multimodal projector (embed_vision) ────────────────────────────

/// Stage 4: the projector `EmbedVision` (`RMSNormNoScale` + 4-bit Linear,
/// gemma4.py:22-34) projects the trimmed vision-tower output `[1, n_soft, 768]`
/// to text hidden `[1, n_soft, 1536]`. Fed the reference `vision_tower_out`
/// (Stage-3 error isolated out), gated on relative error (bf16 + 4-bit floor).
///
/// Crucially there is NO `hidden**0.5` divide here (that scale lives in the
/// pooler) — copying the gemma3 merge-time divide would blow this badly.
#[test]
fn stage4_projector_parity() {
    use super::loader::build_vision_model;

    let Some(bundle) = bundle_or_skip() else {
        return;
    };
    let Some(tower_out) = load_golden("vision_tower_out") else {
        return;
    };
    let ref_feats = load_golden("image_features").expect("image_features golden");

    let vm = build_vision_model(&bundle)
        .expect("build_vision_model")
        .expect("gemma4 vision tower");

    let feats = vm.projector.forward(&tower_out);
    assert_eq!(
        feats.shape(),
        ref_feats.shape(),
        "image_features shape (projector output)"
    );

    let (max_rel, mean_rel) = rel_diff(&feats, &ref_feats);
    eprintln!(
        "[vision_parity] stage4 projector rel(max={:.3}% mean={:.3}%)  shape {:?}",
        max_rel * 100.0,
        mean_rel * 100.0,
        feats.shape()
    );
    assert!(
        max_rel < 0.08 && mean_rel < 0.03,
        "Stage 4 PROJECTOR parity FAILED: rel max={:.3}% mean={:.3}% (floor ~3.5%/1.4%). \
         Check the RMSNormNoScale + 4-bit Linear, and that NO hidden**0.5 divide was \
         copied from gemma3. Report the numbers.",
        max_rel * 100.0,
        mean_rel * 100.0
    );
}

// ─── Stage 5: token-merge scatter into decoder input embeddings ──────────────

/// Stage 5: scatter the projected image features into the decoder's input
/// embeddings at the `image_token_id` positions, mirroring
/// `get_input_embeddings` + `masked_scatter` (gemma4.py:85-124). Fed the
/// reference `input_ids` + `image_features`, compared against the reference
/// `merged_inputs_embeds` (the whole sequence). Also asserts the image-row
/// count equals the projected feature rows (Risk #5).
#[test]
fn stage5_token_merge_parity() {
    use super::loader::build_gemma4_model;

    let Some(bundle) = bundle_or_skip() else {
        return;
    };
    let Some(input_ids) = load_golden("input_ids") else {
        return;
    };
    let image_features = load_golden("image_features").expect("image_features golden");
    let ref_merged = load_golden("merged_inputs_embeds").expect("merged_inputs_embeds golden");

    // Host token ids.
    let ids_i32 = input_ids
        .as_dtype(mlx_rs::Dtype::Int32)
        .expect("cast")
        .as_slice::<i32>()
        .to_vec();
    let tokens: Vec<u32> = ids_i32.iter().map(|&t| t as u32).collect();

    // image_token_id == 258880.
    let n_img = tokens.iter().filter(|&&t| t == 258880).count();
    assert_eq!(
        n_img as i32,
        image_features.shape()[1],
        "Risk #5: image-token count ({}) must equal projected feature rows ({})",
        n_img,
        image_features.shape()[1]
    );

    let (model, _cfg) = build_gemma4_model(&bundle).expect("build_gemma4_model");
    let merged = model.build_input_embeds_with_image(&tokens, &image_features, 258880);

    assert_eq!(
        merged.shape(),
        ref_merged.shape(),
        "merged inputs_embeds shape"
    );

    let (max_rel, mean_rel) = rel_diff(&merged, &ref_merged);
    eprintln!(
        "[vision_parity] stage5 merge rel(max={:.3}% mean={:.3}%)  shape {:?}  image_rows={}",
        max_rel * 100.0,
        mean_rel * 100.0,
        merged.shape(),
        n_img
    );
    assert!(
        max_rel < 0.08 && mean_rel < 0.03,
        "Stage 5 MERGE parity FAILED: rel max={:.3}% mean={:.3}%. Check the scatter \
         placement (image rows must land at image-token positions in order) and the \
         text embed_scale (sqrt(hidden), gemma4.py:86). Report the numbers.",
        max_rel * 100.0,
        mean_rel * 100.0
    );
}

// ─── Stage 6: end-to-end caption ─────────────────────────────────────────────

/// Stage 6: the full native-vision compute path produces a coherent caption.
///
/// Loads the real bundle (vision tower built + `Capabilities::IMAGES` set),
/// reconstructs the exact reference prompt tokens (266 image tokens), encodes
/// cat.png through the **bundle's own** preprocessing + vision tower + projector,
/// runs `forward_with_image` (the scatter + decoder), then greedy-decodes and
/// asserts the caption mentions a cat. This is the integration capstone over the
/// per-stage parity (Stages 2-5).
#[test]
fn stage6_end_to_end_caption() {
    use super::loader::{build_gemma4_model, build_vision_model};
    use super::model::vision_preprocess::Gemma4ImageProcessor;
    use mlx_rs::ops::indexing::IndexOp;

    let Some(bundle) = bundle_or_skip() else {
        return;
    };
    let img_path = PathBuf::from(TEST_IMAGE_REL);
    if !img_path.exists() {
        eprintln!("[vision_parity] stage6 skip — image missing");
        return;
    }
    let Some(ids_golden) = load_golden("input_ids") else {
        return;
    };

    // Capabilities check: the bundle must advertise IMAGES.
    {
        use crate::gen2::backend::mlx::Engine;
        use crate::gen2::engine::{Capabilities, LoadRequest};
        let e = Engine::new();
        e.load_model(LoadRequest {
            model_path: bundle.clone(),
            ..Default::default()
        })
        .expect("load_model");
        assert!(
            e.capabilities().contains(Capabilities::IMAGES),
            "Stage 6: gemma4 vision bundle must advertise Capabilities::IMAGES"
        );
        eprintln!("[vision_parity] stage6 capabilities OK: IMAGES set");
    }

    // Reconstruct the exact reference prompt token stream (266 image tokens).
    let tokens: Vec<u32> = ids_golden
        .as_dtype(mlx_rs::Dtype::Int32)
        .expect("c")
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();

    // Encode cat.png through Pio's own pipeline (NOT the golden pixels).
    let img = image::ImageReader::open(&img_path)
        .expect("open")
        .with_guessed_format()
        .expect("fmt")
        .decode()
        .expect("decode");
    let pixels = Gemma4ImageProcessor::default().preprocess(&img);
    let vm = build_vision_model(&bundle).expect("b").expect("v");
    let image_features = vm.encode_image(&pixels);

    let (model, _cfg) = build_gemma4_model(&bundle).expect("build_gemma4_model");
    let tok = crate::gen2::backend::mlx::tokenizer::HfTokenizer::from_dir(&bundle).expect("tok");

    // Prefill with the image, then greedy-decode up to 24 tokens.
    let mut cache: super::model::KvCache = vec![None; model.num_non_shared as usize];
    let mut logits = model.forward_with_image(&tokens, &image_features, 258880, 0, &mut cache);
    let mut out_ids: Vec<u32> = Vec::new();
    let mut pos = tokens.len();
    let eos = tok.eos_id();
    for _ in 0..24 {
        // argmax of last-position logits (greedy).
        let v: Vec<f32> = logits.index((0, 0, ..)).as_slice::<f32>().to_vec();
        let (next, _) = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &x)| (i as u32, x))
            .expect("argmax");
        if Some(next) == eos {
            break;
        }
        out_ids.push(next);
        logits = model.forward(&[next], pos, &mut cache);
        pos += 1;
    }
    let caption = tok.decode(&out_ids).expect("decode caption");
    eprintln!("[vision_parity] stage6 CAPTION: {caption:?}");

    let lc = caption.to_lowercase();
    assert!(
        lc.contains("cat") || lc.contains("kitten") || lc.contains("feline"),
        "Stage 6 end-to-end caption must mention a cat, got: {caption:?}"
    );
}

// ─── Stage 7: end-to-end caption through the REAL chat path ──────────────────

/// Stage 7: the last mile — a caption produced through the **normal**
/// `Engine::start_session` / chat-template path (NOT a hand-built 266-token
/// prompt like Stage 6). This is the test the task targets: it exercises the
/// image-token *expansion* wired into `Session::new_with_prefix` — the chat
/// template flattens the image chunk to markdown `![](url)`, and the new
/// expansion must turn that into `<boi> + image_token×n_soft + <eoi>` so
/// `full_tokens` carries exactly the right number of `image_token_id` rows for
/// the scatter. Greedy-decodes and asserts the caption mentions a cat.
///
/// Skips cleanly when the bundle or the committed `cat.png` is absent.
#[test]
fn stage7_real_path_caption() {
    use crate::gen2::Message;
    use crate::gen2::backend::mlx::Engine;
    use crate::gen2::engine::{Capabilities, LoadRequest, Settings};
    use crate::gen2::generation::{GenSpec, ThinkingMode, TokenEvent};
    use crate::gen2::session_rt::SessionSpec;

    let Some(bundle) = bundle_or_skip() else {
        // ADR-0036 marker: model absent — SKIP (not FAIL).
        println!("CAPTEST mlx-vision SKIP vision bundle absent ({VISION_BUNDLE})");
        return;
    };
    let img_path = PathBuf::from(TEST_IMAGE_REL);
    if !img_path.exists() {
        eprintln!("[vision_parity] stage7 skip — image missing");
        println!("CAPTEST mlx-vision SKIP fixture absent ({TEST_IMAGE_REL})");
        return;
    }
    let img_abs = std::fs::canonicalize(&img_path).expect("canonicalize cat.png");

    let engine = Engine::new();
    engine
        .load_model(LoadRequest {
            model_path: bundle,
            ..Default::default()
        })
        .expect("load_model");
    assert!(
        engine.capabilities().contains(Capabilities::IMAGES),
        "Stage 7: gemma4 vision bundle must advertise Capabilities::IMAGES"
    );

    // The REAL path: a user message with an attached image, started through
    // `start_session`. The chat template renders it to markdown; the new
    // expansion converts that into the image-token run before tokenizing.
    let mut overrides = Settings::default();
    overrides.sampling.temperature = Some(0.0);
    let msg = Message::user_with_images(
        "Describe this image in one short sentence.",
        [format!("file://{}", img_abs.display())],
    );
    let session = engine
        .start_session(SessionSpec {
            messages: vec![msg],
            overrides: Some(overrides),
            // Direct answer, no reasoning channel — the caption is the first
            // thing the model emits, so a short token budget captures it.
            thinking: ThinkingMode::Off,
            ..Default::default()
        })
        .expect("start_session (real vision path)");

    let mut puller = session
        .pull(GenSpec {
            max_tokens: Some(32),
            temperature: Some(0.0),
            ..Default::default()
        })
        .expect("pull");

    let mut text = String::new();
    loop {
        match puller.next() {
            Some(Ok(TokenEvent::Token(tok))) => text.push_str(&tok.text),
            Some(Ok(TokenEvent::Eos)) | Some(Ok(TokenEvent::Stopped)) => break,
            Some(Ok(_)) => continue,
            Some(Err(e)) => panic!("Stage 7 token error: {e:?}"),
            None => break,
        }
    }
    eprintln!("[vision_parity] stage7 REAL-PATH CAPTION: {text:?}");

    let lc = text.to_lowercase();
    assert!(
        lc.contains("cat") || lc.contains("kitten") || lc.contains("feline"),
        "Stage 7 real-path caption must mention a cat, got: {text:?}"
    );

    // SSS+: inspectable artifact under target/captest/ (CWD = pio-core/).
    let caption = text.trim().to_string();
    let arti_dir = PathBuf::from("../target/captest");
    let _ = std::fs::create_dir_all(&arti_dir);
    let _ = std::fs::write(arti_dir.join("mlx-vision.caption.txt"), &caption);

    // ADR-0036 marker: ran for real with an objective metric (caption mentions a cat).
    println!("CAPTEST mlx-vision RUN caption={caption:?}");
}
