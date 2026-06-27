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

use crate::gen2::engine::ExecError;
use crate::gen2::session_rt::media_util::{check_dimensions, decode_limits, validate_image_path};

/// Decode an attached image from a `file://`/bare path into a `DynamicImage`,
/// hardened against malformed and adversarial inputs.
///
/// This is the single trust boundary for user-supplied image bytes on the MLX
/// vision path. It NEVER panics on bad input — every failure (missing/unreadable
/// path, a directory, undetectable format, truncated/garbage bytes, or an
/// over-cap decompression bomb) maps to a graceful [`ExecError`] the UI can
/// surface.
///
/// Caps are enforced in two layers: the decoder [`image::Limits`] reject the
/// allocation at the header before pixels are touched, and an explicit
/// post-decode dimension/pixel check catches anything the limits let through.
/// Valid in-cap images decode byte-identically to a plain
/// `ImageReader::open(..).with_guessed_format().decode()` — limits only ever
/// *reject*, never alter pixels — so vision parity is unaffected.
pub fn load_attached_image(url_or_path: &str) -> Result<DynamicImage, ExecError> {
    // Layer 1 — shared path validation + header-only dimension cap. Rejects a
    // missing/unreadable path, a directory, a `file://../../etc/passwd`
    // traversal that resolves to a non-image, and an over-cap declared-dimension
    // bomb (incl. a within-per-side pixel bomb like 60000×60000) BEFORE any
    // decode allocation.
    let path = validate_image_path(url_or_path)?;
    let path = path.as_str();

    // Layer 2 — decode under the same caps. The decoder `Limits` are the hard
    // wall that stops a tiny-on-disk, huge-declared-dimensions image from
    // allocating gigabytes: it returns Err instead of OOMing.
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| ExecError::Io(format!("open image {path}: {e}")))?
        .with_guessed_format()
        .map_err(|e| ExecError::Io(format!("guess image format {path}: {e}")))?;
    reader.limits(decode_limits());

    let img = reader
        .decode()
        .map_err(|e| ExecError::Other(anyhow::anyhow!("decode image {path}: {e}")))?;

    // Layer 3 (belt-and-suspenders) — re-check the actual decoded dimensions.
    check_dimensions(path, img.width(), img.height())?;
    Ok(img)
}

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

    /// Number of vision **soft tokens** this image pools down to — the
    /// per-image count that must equal BOTH the pooled vision-feature rows AND
    /// the number of `image_token_id` placeholders expanded into the prompt
    /// (the scatter-count invariant `forward_with_image` asserts). Mirrors the
    /// processor's `num_soft_tokens_per_image` (processing_gemma4.py:196-198):
    /// `num_patches = (H/patch)*(W/patch); num_soft = num_patches / kernel²`,
    /// computed from the post-resize target size so it agrees exactly with what
    /// the tower's pooler trims to (`VisionTower::forward_parts` n_valid) without
    /// running the tower.
    pub fn num_soft_tokens(&self, width: u32, height: u32) -> usize {
        let (tw, th) = self.target_size(width, height);
        let patch = self.patch_size as usize;
        let kernel = self.pooling_kernel_size as usize;
        let num_patches = (th as usize / patch) * (tw as usize / patch);
        num_patches / (kernel * kernel)
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

#[cfg(test)]
mod adversarial_image_tests {
    //! Hardening tests for the untrusted image-attach input path. Each test
    //! feeds an adversarial input through [`load_attached_image`] and asserts it
    //! returns a graceful `Err` (no panic, no OOM, no hang) — or, for valid
    //! inputs, that behaviour is unchanged.
    use super::*;
    use std::io::Write;

    /// Write `bytes` to a uniquely-named temp file with `ext`, return its path.
    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "pio_advimg_{}_{}_{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&p).expect("create temp file");
        f.write_all(bytes).expect("write temp file");
        p
    }

    /// CRC-32 (PNG polynomial) for hand-crafting raw PNG chunks.
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Build a minimal valid-header PNG whose IHDR *declares* `w`×`h` but carries
    /// essentially no pixel data — the classic "decompression bomb": a few
    /// hundred bytes on disk that decode to billions of pixels.
    fn bomb_png(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(b"IHDR");
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, RGB, no interlace
        out.extend_from_slice(&(13u32).to_be_bytes());
        out.extend_from_slice(&ihdr);
        out.extend_from_slice(&crc32(&ihdr).to_be_bytes());
        // A tiny IDAT (deflate empty / minimal) — enough to look structured.
        let mut idat = Vec::new();
        idat.extend_from_slice(b"IDAT");
        idat.extend_from_slice(&[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(&((idat.len() - 4) as u32).to_be_bytes());
        out.extend_from_slice(&idat);
        out.extend_from_slice(&crc32(&idat).to_be_bytes());
        out
    }

    /// Encode a valid PNG of the given size via the `image` crate.
    fn valid_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode png");
        buf.into_inner()
    }

    #[test]
    fn zero_byte_file_errors_gracefully() {
        let p = temp_file("empty.png", &[]);
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "0-byte file must error, not panic");
    }

    #[test]
    fn truncated_png_errors_gracefully() {
        let full = valid_png(64, 64);
        // Keep the signature + IHDR but cut mid-stream.
        let truncated = &full[..(full.len() / 2).max(40)];
        let p = temp_file("truncated.png", truncated);
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "truncated PNG must error, not panic");
    }

    #[test]
    fn truncated_jpeg_errors_gracefully() {
        // Minimal JPEG SOI + a partial header, then cut.
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F'];
        let p = temp_file("truncated.jpg", &bytes);
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "truncated JPEG must error, not panic");
    }

    #[test]
    fn non_image_bytes_with_image_extension_errors() {
        let p = temp_file("fake.png", b"this is definitely not an image, just text\n");
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_err(), "non-image bytes must error, not panic");
    }

    #[test]
    fn decompression_bomb_dimensions_rejected_without_oom() {
        // 100000×100000 = 10^10 px declared; a few hundred bytes on disk.
        // Must be rejected by the cap BEFORE allocating ~30 GB. The header probe
        // reads the IHDR dims and `check_dimensions` rejects them by message —
        // we assert the error names the pixel cap, proving the cap fired (not an
        // incidental truncated-IDAT decode error).
        let p = temp_file("bomb.png", &bomb_png(100_000, 100_000));
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        let err = r.expect_err("decompression-bomb dimensions must be rejected, not OOM");
        assert!(
            err.to_string().contains("decompression bomb") || err.to_string().contains("exceed"),
            "bomb must be rejected by the dimension cap, got: {err}"
        );
    }

    #[test]
    fn within_per_side_pixel_bomb_rejected() {
        // 60000×60000 = 3.6 GP: each side is under the 65535 per-side cap, but
        // the pixel product blows the 89 MP cap. Must still be rejected up front.
        let p = temp_file("bomb2.png", &bomb_png(60_000, 60_000));
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(
            r.is_err(),
            "within-per-side pixel bomb must be rejected, not OOM"
        );
    }

    #[test]
    fn one_by_one_image_decodes() {
        let p = temp_file("tiny.png", &valid_png(1, 1));
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok(), "1x1 valid image must decode: {r:?}");
        let img = r.unwrap();
        assert_eq!((img.width(), img.height()), (1, 1));
    }

    #[test]
    fn large_real_image_under_cap_decodes() {
        // 2000×2000 = 4 MP — a real, in-cap image must still decode unaffected.
        // (Kept modest so the test stays fast; the 89 MP cap is exercised by
        // the dimension checks below without allocating an 8000² buffer.)
        let p = temp_file("big.png", &valid_png(2000, 2000));
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok(), "in-cap large image must decode: {r:?}");
    }

    #[test]
    fn check_dimensions_enforces_pixel_cap() {
        // Unit-level proof of the cap boundary without huge allocations.
        assert!(check_dimensions("x", 8000, 8000).is_ok(), "64 MP under cap");
        assert!(
            check_dimensions("x", 10_000, 10_000).is_err(),
            "100 MP over cap"
        );
        assert!(check_dimensions("x", 0, 100).is_err(), "zero width");
        assert!(check_dimensions("x", 100, 0).is_err(), "zero height");
        assert!(
            check_dimensions("x", 70_000, 1).is_err(),
            "over per-side cap"
        );
    }

    #[test]
    fn wrong_format_webp_decodes_or_errors_gracefully() {
        // A WEBP file where PNG/JPEG might be expected. The `image` crate
        // supports WEBP, so this should DECODE (graceful, not panic). Either
        // way: never a panic.
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([10, 20, 30]));
        let mut buf = std::io::Cursor::new(Vec::new());
        let res = image::DynamicImage::ImageRgb8(img).write_to(&mut buf, image::ImageFormat::WebP);
        if res.is_err() {
            return; // WEBP encode not built in; nothing to assert.
        }
        let p = temp_file("img.webp", &buf.into_inner());
        let r = load_attached_image(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        // Sniffing means a valid WEBP decodes fine; the point is no panic.
        assert!(r.is_ok() || r.is_err());
    }

    #[test]
    fn nonexistent_path_errors_gracefully() {
        let r = load_attached_image("/no/such/path/does-not-exist-12345.png");
        assert!(r.is_err(), "missing path must error, not panic");
        assert!(
            matches!(r, Err(ExecError::Io(_))),
            "missing path should map to ExecError::Io"
        );
    }

    #[test]
    fn directory_path_errors_gracefully() {
        let r = load_attached_image(std::env::temp_dir().to_str().unwrap());
        assert!(r.is_err(), "a directory must error, not panic");
    }

    #[test]
    fn file_url_prefix_is_stripped() {
        let p = temp_file("url.png", &valid_png(4, 4));
        let url = format!("file://{}", p.display());
        let r = load_attached_image(&url);
        std::fs::remove_file(&p).ok();
        assert!(r.is_ok(), "file:// URL must decode: {r:?}");
    }

    #[test]
    fn traversal_path_to_non_image_errors() {
        // A traversal-style path that resolves to a real but non-image file
        // (/etc/hosts exists on macOS/Linux) must error gracefully, not panic.
        let candidates = ["file:///etc/hosts", "/etc/hosts"];
        for c in candidates {
            let stripped = c.strip_prefix("file://").unwrap_or(c);
            if std::path::Path::new(stripped).exists() {
                let r = load_attached_image(c);
                assert!(
                    r.is_err(),
                    "non-image system file {c} must error, not panic"
                );
                return;
            }
        }
        // No suitable system file present; nothing to assert.
    }
}
