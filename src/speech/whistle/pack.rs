//! Host-side metadata and bounded engram dequantization from the pinned WPK1 pack.
use crate::speech::{Result, SpeechError};
use std::{collections::HashMap, path::Path};

#[derive(serde::Deserialize)]
struct Manifest {
    version: u32,
    codebooks: HashMap<String, Vec<f64>>,
    engram: Vec<Segment>,
    tok: Tokenizer,
    melfb: MelFilter,
}
#[derive(serde::Deserialize)]
struct MelFilter {
    src: usize,
    shape: Vec<usize>,
}
#[derive(serde::Deserialize)]
struct Segment {
    k: String,
    src: usize,
    n: usize,
    out: usize,
    #[serde(rename = "in")]
    input: usize,
    bits: usize,
    group: usize,
    rows: Vec<usize>,
}
#[derive(serde::Deserialize)]
pub(super) struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f64>,
    types: Vec<u8>,
}
impl Tokenizer {
    pub(super) fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let id = id as usize;
            let piece = self
                .pieces
                .get(id)
                .ok_or_else(|| invalid("token ID out of range"))?;
            match self.types[id] {
                4 => {
                    let b = piece
                        .strip_prefix("<0x")
                        .and_then(|p| p.strip_suffix('>'))
                        .and_then(|p| u8::from_str_radix(p, 16).ok())
                        .ok_or_else(|| invalid("invalid byte token"))?;
                    bytes.push(b);
                }
                0 | 3 => bytes.extend(piece.replace('\u{2581}', " ").bytes()),
                _ => {}
            }
        }
        // Matches the reference TextDecoder's replacement behavior on byte fallback.
        Ok(String::from_utf8_lossy(&bytes)
            .trim_matches(js_whitespace)
            .into())
    }
}
pub(super) struct Pack {
    bytes: Vec<u8>,
    base: usize,
    segments: Vec<Segment>,
    codebook: Vec<f64>,
    pub(super) tokenizer: Tokenizer,
    pub(super) filters: Vec<f32>,
    rows: HashMap<(usize, usize), [f32; 128]>,
}
fn invalid(message: &str) -> SpeechError {
    SpeechError::InvalidBundle(message.into())
}
impl Pack {
    pub(super) fn open(root: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(root.join("whistle.pack")).map_err(|e| invalid(&e.to_string()))?;
        if bytes.len() < 8 || &bytes[..4] != b"WPK1" {
            return Err(invalid("invalid WPK1 header"));
        }
        let length = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if length > 1024 * 1024 || 8 + length > bytes.len() {
            return Err(invalid("invalid pack header length"));
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes[8..8 + length]).map_err(|e| invalid(&e.to_string()))?;
        let base = (8 + length).next_multiple_of(16);
        if manifest.version != 2
            || base > bytes.len()
            || manifest.engram.len() != 2
            || manifest.melfb.shape != [257, 80]
            || manifest.tok.pieces.len() != 8199
            || manifest.tok.scores.len() != 8199
            || manifest.tok.types.len() != 8199
            || manifest.tok.scores.iter().any(|s| !s.is_finite())
        {
            return Err(invalid("incompatible Whistle metadata"));
        }
        let codebook = manifest
            .codebooks
            .get("2")
            .filter(|c| c.len() == 4 && c.iter().all(|v| v.is_finite()))
            .ok_or_else(|| invalid("invalid engram codebook"))?
            .clone();
        for seg in &manifest.engram {
            if seg.k != "cq"
                || seg.bits != 2
                || seg.group != 128
                || seg.input != 128
                || seg.out != 73728
                || seg.rows != [0, 73728]
                || seg.n != 73728 * 34
            {
                return Err(invalid("incompatible engram segment"));
            }
            checked_slice(&bytes, base, seg.src, seg.n)?;
        }
        let filter_bytes = checked_slice(&bytes, base, manifest.melfb.src, 257 * 80 * 4)?;
        let filters: Vec<f32> = filter_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if filters.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite filter coefficients"));
        }
        Ok(Self {
            bytes,
            base,
            segments: manifest.engram,
            codebook,
            tokenizer: manifest.tok,
            filters,
            rows: HashMap::new(),
        })
    }
    fn row(&mut self, e: usize, index: usize) -> [f32; 128] {
        if let Some(row) = self.rows.get(&(e, index)) {
            return *row;
        }
        let seg = &self.segments[e];
        let src = self.base + seg.src;
        let norm_offset = src + seg.out * 32 + index * 2;
        let norm = half::f16::from_bits(u16::from_le_bytes(
            self.bytes[norm_offset..norm_offset + 2].try_into().unwrap(),
        ))
        .to_f64()
            / 128f64.sqrt();
        let mut tmp = [0f64; 128];
        for (j, v) in tmp.iter_mut().enumerate() {
            let b = self.bytes[src + index * 32 + j / 4];
            *v = self.codebook[((b >> (2 * (j % 4))) & 3) as usize] * norm;
        }
        let mut h = 1;
        while h < 128 {
            for start in (0..128).step_by(2 * h) {
                for j in start..start + h {
                    let x = tmp[j];
                    let y = tmp[j + h];
                    tmp[j] = x + y;
                    tmp[j + h] = x - y;
                }
            }
            h *= 2;
        }
        let row = tmp.map(|x| x as f32);
        // Immutable rows are independent of request history. Clearing only the
        // lookup cache keeps residency bounded and never changes their values.
        if self.rows.len() >= 4096 {
            self.rows.clear();
        }
        self.rows.insert((e, index), row);
        row
    }
    pub(super) fn features(&mut self, tokens: &[u32], t: usize) -> Vec<f32> {
        let mut result = vec![0f32; 2 * 4 * 512];
        for tap in 0..4 {
            let Some(p) = t.checked_sub(3 * tap) else {
                continue;
            };
            for table in 0..4 {
                let order = if table < 2 { 2 } else { 3 };
                if p + 1 < order {
                    continue;
                }
                let mut hash = 0x9e3779b9u32.wrapping_mul(table as u32 + 1);
                for j in 0..order {
                    hash = (hash ^ tokens[p - j]).wrapping_mul(0x01000193);
                }
                hash ^= hash >> 15;
                let index = table * 18432 + (hash as usize % 18432);
                for e in 0..2 {
                    let offset = (e * 4 + tap) * 512 + table * 128;
                    result[offset..offset + 128].copy_from_slice(&self.row(e, index));
                }
            }
        }
        result
    }
}
fn checked_slice(bytes: &[u8], base: usize, offset: usize, length: usize) -> Result<&[u8]> {
    let start = base
        .checked_add(offset)
        .ok_or_else(|| invalid("pack offset overflow"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| invalid("pack extent overflow"))?;
    bytes
        .get(start..end)
        .ok_or_else(|| invalid("pack section out of bounds"))
}

// ECMAScript String.trim differs from Rust's Unicode White_Space: BOM is
// whitespace in JavaScript, while NEL (U+0085) is preserved. TextDecoder drops
// an initial BOM, which this edge trim also removes.
fn js_whitespace(c: char) -> bool {
    matches!(c, '\u{0009}'..='\u{000d}' | '\u{0020}' | '\u{00a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}'
        | '\u{205f}' | '\u{3000}' | '\u{feff}')
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_fallback_matches_textdecoder_and_ecmascript_trim() {
        let tokenizer = Tokenizer {
            pieces: [
                "<0xEF>", "<0xBB>", "<0xBF>", "<0xC2>", "<0x85>", "<0x41>", "<0xFF>",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            scores: vec![0.; 7],
            types: vec![4; 7],
        };
        assert_eq!(tokenizer.decode(&[0, 1, 2, 5]).unwrap(), "A");
        assert_eq!(
            tokenizer.decode(&[3, 4, 5, 3, 4]).unwrap(),
            "\u{0085}A\u{0085}"
        );
        assert_eq!(tokenizer.decode(&[5, 6]).unwrap(), "A\u{fffd}");
        assert_eq!(tokenizer.decode(&[5, 0, 1, 2, 5]).unwrap(), "A\u{feff}A");
    }
}
