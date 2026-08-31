use crate::engine::ExecError;
use crate::{Message, MessageBody, MessageChunk, MessageContent};

/// Hard caps on attached-image inputs, shared by every vision backend so the
/// untrusted decode path can't panic or OOM on adversarial bytes (truncated
/// streams, dimension "decompression bombs", etc).
///
/// `MAX_IMAGE_PIXELS` mirrors Pillow's `MAX_IMAGE_PIXELS` default
/// (`89_478_485` ≈ a 9462² image, ~256 MiB as RGB888) — the same reference
/// library mlx-vlm decodes with, so any image the upstream processor accepts,
/// Pio accepts too. An 8000×8000 (64 MP) real photo stays under the cap; a
/// 100000×100000 declared-dimension bomb (10^10 px) is rejected before a single
/// pixel is allocated. See
/// <https://pillow.readthedocs.io/en/stable/reference/Image.html>.
pub const MAX_IMAGE_PIXELS: u64 = 89_478_485;
/// Per-side cap. `65_535` is the largest dimension a 16-bit PNG/JPEG field can
/// even encode; combined with the pixel-product cap this bounds the decoder's
/// up-front allocation when one side is enormous.
pub const MAX_IMAGE_SIDE: u32 = 65_535;
/// Allocation ceiling for the `image` crate decoder `Limits`. Sized to
/// `MAX_IMAGE_PIXELS` as RGBA (4 bytes) plus headroom — above any in-cap image,
/// but a hard wall a bomb's decoder hits as a graceful `Err`, not an OOM.
pub const MAX_DECODE_ALLOC_BYTES: u64 = MAX_IMAGE_PIXELS * 4 + (64 * 1024 * 1024);

/// Decoder limits (per-side dimensions + total allocation) for hardened decode.
pub fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(MAX_DECODE_ALLOC_BYTES);
    limits
}

/// Reject zero-size and over-cap dimensions with a graceful error.
pub fn check_dimensions(path: &str, w: u32, h: u32) -> Result<(), ExecError> {
    if w == 0 || h == 0 {
        return Err(ExecError::Other(anyhow::anyhow!(
            "image {path} has zero width or height"
        )));
    }
    if w > MAX_IMAGE_SIDE || h > MAX_IMAGE_SIDE || (w as u64 * h as u64) > MAX_IMAGE_PIXELS {
        return Err(ExecError::Other(anyhow::anyhow!(
            "image {path} dimensions {w}x{h} exceed the {MAX_IMAGE_PIXELS}-pixel cap \
             (possible decompression bomb)"
        )));
    }
    Ok(())
}

/// Validate an attached-image path BEFORE handing it to a native decoder
/// (e.g. llama.cpp's `MtmdBitmap::from_file`, whose C++ stb_image path Pio
/// can't cap directly). Strips a `file://` prefix, rejects a missing/unreadable
/// path or a directory, and rejects an over-cap image by reading only the
/// header dimensions — so a decompression bomb is refused before the native
/// decoder allocates. Returns the resolved filesystem path on success.
///
/// Never panics: every failure maps to a graceful [`ExecError`].
pub fn validate_image_path(url_or_path: &str) -> Result<String, ExecError> {
    let path = url_or_path.strip_prefix("file://").unwrap_or(url_or_path);

    let meta =
        std::fs::metadata(path).map_err(|e| ExecError::Io(format!("open image {path}: {e}")))?;
    if !meta.is_file() {
        return Err(ExecError::Io(format!(
            "open image {path}: not a regular file"
        )));
    }

    // Header-only dimension probe — rejects over-cap declared dims up front.
    let mut probe = image::ImageReader::open(path)
        .map_err(|e| ExecError::Io(format!("open image {path}: {e}")))?
        .with_guessed_format()
        .map_err(|e| ExecError::Io(format!("guess image format {path}: {e}")))?;
    probe.limits(decode_limits());
    let (w, h) = probe
        .into_dimensions()
        .map_err(|e| ExecError::Other(anyhow::anyhow!("read image dimensions {path}: {e}")))?;
    check_dimensions(path, w, h)?;
    Ok(path.to_string())
}

pub(crate) fn messages_have_images(messages: &Vec<Message>) -> bool {
    for msg in messages {
        if let MessageBody::Content { content } = &msg.body {
            match content {
                MessageContent::SingleText(_) => {}
                MessageContent::StructuredAssistant { .. } => {
                    // Structured assistant replies carry no media —
                    // models that produce them don't emit images.
                }
                MessageContent::MultipleChunks(chunks) => {
                    if chunks
                        .iter()
                        .any(|c| matches!(c, MessageChunk::ImageUrl { .. }))
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{MessageBody, Url};

    #[test]
    fn detect_images() {
        let msgs = vec![Message {
            name: None,
            role: "user".into(),
            body: MessageBody::Content {
                content: MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: "hi".into() },
                    MessageChunk::ImageUrl {
                        image_url: Url {
                            url: "file:///tmp/x.png".into(),
                        },
                    },
                ]),
            },
        }];
        assert!(messages_have_images(&msgs));
    }

    use std::io::Write;

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

    /// Minimal valid-header PNG declaring `w`×`h` with no real pixel data — a
    /// decompression bomb (tiny on disk, billions of pixels declared).
    fn bomb_png(w: u32, h: u32) -> Vec<u8> {
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        out.extend_from_slice(&13u32.to_be_bytes());
        out.extend_from_slice(&ihdr);
        out.extend_from_slice(&crc32(&ihdr).to_be_bytes());
        out
    }

    fn valid_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([1, 2, 3]));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "pio_mu_{}_{}_{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    #[test]
    fn validate_rejects_missing_path() {
        let r = validate_image_path("/no/such/file/xyz-12345.png");
        assert!(matches!(r, Err(ExecError::Io(_))));
    }

    #[test]
    fn validate_rejects_directory() {
        let r = validate_image_path(std::env::temp_dir().to_str().unwrap());
        assert!(r.is_err());
    }

    #[test]
    fn validate_rejects_dimension_bomb() {
        let p = temp_file("bomb.png", &bomb_png(100_000, 100_000));
        let r = validate_image_path(p.to_str().unwrap());
        std::fs::remove_file(&p).ok();
        let e = r.expect_err("bomb must be rejected before native decode");
        assert!(
            e.to_string().contains("decompression bomb") || e.to_string().contains("exceed"),
            "got: {e}"
        );
    }

    #[test]
    fn validate_accepts_in_cap_image_and_strips_file_url() {
        let p = temp_file("ok.png", &valid_png(8, 8));
        let url = format!("file://{}", p.display());
        let r = validate_image_path(&url);
        let resolved = r.expect("in-cap image must validate");
        assert_eq!(resolved, p.to_str().unwrap(), "file:// prefix stripped");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn check_dimensions_caps() {
        assert!(check_dimensions("x", 8000, 8000).is_ok());
        assert!(check_dimensions("x", 10_000, 10_000).is_err());
        assert!(check_dimensions("x", 0, 5).is_err());
        assert!(check_dimensions("x", 70_000, 1).is_err());
    }
}
