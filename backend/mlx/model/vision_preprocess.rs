//! Gemma 4 image preprocessing → `pixel_values` tensor.
//!
//! Port of `Gemma4ImageProcessor` (mlx-vlm `models/gemma4/processing_gemma4.py`
//! :52-210). Steps, in order:
//!   1. convert to RGB (`do_convert_rgb=True`)
//!   2. aspect-ratio-preserving BICUBIC resize so the patch count fits the
//!      budget AND both sides are divisible by `pooling_kernel_size*patch_size`
//!      (= 48) — `aspect_ratio_preserving_resize` :91-147. PIL `Image.BICUBIC`.
//!   3. rescale `/255` (`do_rescale=True`), NO normalize (`do_normalize=False` —
//!      the `2*(x-0.5)` centering happens later in `_patchify`, vision.py:319)
//!   4. emit channels-first `[1, 3, H, W]` f32 (`model_input_names=["pixel_values"]`)
//!
//! ## BICUBIC parity (the load-bearing detail)
//! The processor calls `pil_img.resize(..., Image.BICUBIC)` on a **uint8** image.
//! PIL resamples uint8 with a **fixed-point integer** kernel (`Resample.c`):
//! the `a=-0.5` cubic coefficients are normalized, quantized to
//! `PRECISION_BITS = 32-8-2 = 22`-bit integers, then the horizontal and vertical
//! passes accumulate in `i64` and round/clip via `clip8`. A *float* cubic (or
//! the `image` crate's `resize`) matches the coefficients but not the per-pixel
//! fixed-point rounding — leaving a worst-pixel `max|Δ| ≈ 0.031` (8/255) vs PIL.
//! Reproducing PIL's exact fixed-point path here makes Pio's `pixel_values`
//! **bit-identical** to PIL (`max|Δ| = 0`), so we hand-roll the resampler.
//!
//! Note: the parity fixture must be a **lossless** image (PNG) — a JPEG would
//! decode to subtly different pixels under the `image` crate's decoder vs PIL's
//! libjpeg (different chroma reconstruction), which is the input, not the
//! resampler, diverging.

use image::DynamicImage;
use mlx_rs::Array;

/// Pillow `PRECISION_BITS` for the 8-bpc resample path (`Resample.c`).
const PRECISION_BITS: i64 = 32 - 8 - 2; // = 22
/// Bicubic filter support radius (Pillow `BICUBIC.support`).
const SUPPORT: f64 = 2.0;

/// Processor params (from `preprocessor_config.json` / dataclass defaults
/// :63-89). `patch_size=16`, `max_soft_tokens=280`, `pooling_kernel_size=3`.
#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessor {
    pub patch_size: u32,
    pub max_soft_tokens: u32,
    pub pooling_kernel_size: u32,
    pub rescale_factor: f32,
}

impl Default for Gemma4ImageProcessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

/// Pillow `a=-0.5` cubic convolution kernel (`bicubic_filter` in `Resample.c`).
#[inline]
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel resample coefficients (Pillow `precompute_coeffs`): the
/// quantized integer weights plus `(xmin, count)` source bounds.
struct Coeffs {
    kk: Vec<Vec<i64>>,
    bounds: Vec<(usize, usize)>,
}

/// Mirror Pillow `precompute_coeffs` (`Resample.c`) for one axis: `in_size →
/// out_size`. Coefficients are the float cubic, normalized to sum 1, then
/// quantized to `PRECISION_BITS` fixed-point.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let ss = 1.0 / filterscale;

    let mut kk = Vec::with_capacity(out_size);
    let mut bounds = Vec::with_capacity(out_size);

    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax_raw = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let count = xmax_raw.saturating_sub(xmin);

        let mut weights = Vec::with_capacity(count);
        let mut ww = 0.0f64;
        for x in 0..count {
            let w = cubic(((x + xmin) as f64 - center + 0.5) * ss);
            weights.push(w);
            ww += w;
        }
        if ww != 0.0 {
            for w in weights.iter_mut() {
                *w /= ww;
            }
        }
        // Quantize to fixed-point (Pillow normalize_coeffs_8bpc).
        let one = (1i64) << PRECISION_BITS;
        let ik: Vec<i64> = weights
            .iter()
            .map(|&w| (w * one as f64).round() as i64)
            .collect();

        kk.push(ik);
        bounds.push((xmin, count));
    }
    Coeffs { kk, bounds }
}

/// Pillow `clip8`: round the fixed-point accumulator and clamp to `[0, 255]`.
#[inline]
fn clip8(acc: i64) -> u8 {
    let v = (acc + (1i64 << (PRECISION_BITS - 1))) >> PRECISION_BITS;
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Resample one channel plane `[h, w]` (row-major u8): horizontal `w → out_w`,
/// then vertical `h → out_h`. Mirrors Pillow's two-pass
/// `ImagingResampleHorizontal_8bpc` + `…Vertical_8bpc`.
fn resample_plane(
    src: &[u8],
    w: usize,
    h: usize,
    out_w: usize,
    out_h: usize,
    hc: &Coeffs,
    vc: &Coeffs,
) -> Vec<u8> {
    // Horizontal pass: [h, w] -> [h, out_w].
    let mut horiz = vec![0u8; h * out_w];
    for (y, row) in horiz.chunks_exact_mut(out_w).enumerate() {
        let src_row = &src[y * w..(y + 1) * w];
        for (xx, dst) in row.iter_mut().enumerate() {
            let (xmin, count) = hc.bounds[xx];
            let weights = &hc.kk[xx];
            let mut acc: i64 = 0;
            for x in 0..count {
                acc += src_row[xmin + x] as i64 * weights[x];
            }
            *dst = clip8(acc);
        }
    }

    // Vertical pass: [h, out_w] -> [out_h, out_w].
    let mut out = vec![0u8; out_h * out_w];
    for yy in 0..out_h {
        let (ymin, count) = vc.bounds[yy];
        let weights = &vc.kk[yy];
        let dst_row = &mut out[yy * out_w..(yy + 1) * out_w];
        for (x, dst) in dst_row.iter_mut().enumerate() {
            let mut acc: i64 = 0;
            for k in 0..count {
                acc += horiz[(ymin + k) * out_w + x] as i64 * weights[k];
            }
            *dst = clip8(acc);
        }
    }
    out
}

impl Gemma4ImageProcessor {
    /// `max_patches = max_soft_tokens * pooling_kernel_size²` (:155). For the
    /// defaults: `280 * 9 = 2520`.
    pub fn max_patches(&self) -> u32 {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Compute the aspect-preserving target `(width, height)` — mirrors
    /// `aspect_ratio_preserving_resize` :104-125.
    pub fn target_size(&self, width: u32, height: u32) -> (u32, u32) {
        let max_patches = self.max_patches() as f64;
        let patch = self.patch_size as f64;
        let side_mult = (self.pooling_kernel_size * self.patch_size) as f64; // 48
        let (w, h) = (width as f64, height as f64);

        let target_px = max_patches * patch * patch;
        let factor = (target_px / (h * w)).sqrt();

        let mut target_h = ((factor * h / side_mult).floor() * side_mult) as i64;
        let mut target_w = ((factor * w / side_mult).floor() * side_mult) as i64;

        let sm = side_mult as i64;
        let max_side_length = (max_patches as i64
            / (self.pooling_kernel_size * self.pooling_kernel_size) as i64)
            * sm;
        if target_h == 0 {
            target_h = sm;
            target_w = ((w / h).floor() as i64 * sm).min(max_side_length);
        } else if target_w == 0 {
            target_w = sm;
            target_h = ((h / w).floor() as i64 * sm).min(max_side_length);
        }
        (target_w.max(sm) as u32, target_h.max(sm) as u32)
    }

    /// Full preprocess of ONE image → `[1, 3, H, W]` f32 `pixel_values`.
    pub fn preprocess(&self, img: &DynamicImage) -> Array {
        // 1. RGB convert.
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);

        // 2. aspect-preserving BICUBIC resize (PIL-exact fixed-point path).
        let (tw, th) = self.target_size(w as u32, h as u32);
        let (tw, th) = (tw as usize, th as usize);

        // Split into 3 row-major u8 channel planes.
        let mut planes = [vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]];
        for (i, px) in rgb.pixels().enumerate() {
            planes[0][i] = px[0];
            planes[1][i] = px[1];
            planes[2][i] = px[2];
        }

        let (rw, rh, resized_planes) = if (tw, th) == (w, h) {
            (w, h, planes)
        } else {
            let hc = precompute_coeffs(w, tw);
            let vc = precompute_coeffs(h, th);
            let r = [
                resample_plane(&planes[0], w, h, tw, th, &hc, &vc),
                resample_plane(&planes[1], w, h, tw, th, &hc, &vc),
                resample_plane(&planes[2], w, h, tw, th, &hc, &vc),
            ];
            (tw, th, r)
        };

        // 3. rescale /255, channels-first [1, 3, H, W] f32.
        let plane = rh * rw;
        let mut chw = vec![0f32; 3 * plane];
        for c in 0..3 {
            for i in 0..plane {
                chw[c * plane + i] = resized_planes[c][i] as f32 * self.rescale_factor;
            }
        }
        Array::from_slice(&chw, &[1, 3, rh as i32, rw as i32])
    }
}
