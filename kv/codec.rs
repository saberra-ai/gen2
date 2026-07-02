// The KV-cache blob codec is exercised by persistence paths compiled out of the
// default feature surface the pre-commit clippy runs (7–32 refs elsewhere), so
// its items read as dead there while being live under those features.
#![allow(dead_code)]

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::Context;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use super::types::{KvHeader, KvMeta};

const MAGIC: &[u8; 7] = b"PIOKV1\0";

pub fn build_blob(meta: KvMeta, payload: &[u8]) -> anyhow::Result<Bytes> {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    let digest: [u8; 32] = hasher.finalize().into();
    let header = KvHeader {
        version: 1,
        meta,
        payload_sha256: digest,
    };
    let header_json = serde_json::to_vec(&header)?;
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + header_json.len() + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(payload);
    Ok(Bytes::from(out))
}

pub fn parse_blob(bytes: &[u8]) -> anyhow::Result<(KvHeader, &[u8])> {
    if bytes.len() < MAGIC.len() + 4 {
        anyhow::bail!("blob too small");
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        anyhow::bail!("bad magic");
    }
    let len_bytes: [u8; 4] = bytes[MAGIC.len()..MAGIC.len() + 4]
        .try_into()
        .context("invalid header length bytes")?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;
    let header_start = MAGIC.len() + 4;
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        anyhow::bail!("blob truncated");
    }
    let header: KvHeader = serde_json::from_slice(&bytes[header_start..header_end])?;
    let payload = &bytes[header_end..];
    let mut hasher = Sha256::new();
    hasher.update(payload);
    let digest: [u8; 32] = hasher.finalize().into();
    if digest != header.payload_sha256 {
        anyhow::bail!("payload checksum mismatch");
    }
    Ok((header, payload))
}

pub fn write_to_path(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(bytes)?;
    Ok(())
}

pub fn read_from_path(path: &Path) -> anyhow::Result<Bytes> {
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(Bytes::from(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrip_blob() {
        let meta = KvMeta {
            model_uuid: "abc".into(),
            n_ctx: 1024,
            n_layer: 12,
            tokenizer_digest: [1; 32],
            template_fingerprint: [42; 32],
            created_at_us: 123,
        };
        let payload = b"hello world";
        let blob = build_blob(meta.clone(), payload).unwrap();
        let (hdr, pl) = parse_blob(&blob).unwrap();
        assert_eq!(hdr.meta, meta);
        assert_eq!(pl, payload);
    }

    #[test]
    fn file_io() {
        let meta = KvMeta {
            model_uuid: "x".into(),
            n_ctx: 1,
            n_layer: 1,
            tokenizer_digest: [0; 32],
            template_fingerprint: [0; 32],
            created_at_us: 0,
        };
        let payload = b"abc";
        let blob = build_blob(meta, payload).unwrap();
        let tmp = NamedTempFile::new().unwrap();
        write_to_path(tmp.path(), &blob).unwrap();
        let back = read_from_path(tmp.path()).unwrap();
        assert_eq!(&back[..], &blob[..]);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        // Build a valid blob then corrupt the magic
        let meta = KvMeta {
            model_uuid: "x".into(),
            n_ctx: 1,
            n_layer: 1,
            tokenizer_digest: [0; 32],
            template_fingerprint: [0; 32],
            created_at_us: 0,
        };
        let blob = build_blob(meta, b"payload").unwrap();
        let mut bad = blob.to_vec();
        bad[0] ^= 0xFF; // flip a bit in the magic
        assert!(parse_blob(&bad).is_err());
    }

    #[test]
    fn parse_rejects_truncated_header() {
        let meta = KvMeta {
            model_uuid: "y".into(),
            n_ctx: 2,
            n_layer: 2,
            tokenizer_digest: [1; 32],
            template_fingerprint: [1; 32],
            created_at_us: 1,
        };
        let blob = build_blob(meta, b"xyz").unwrap();
        let mut truncated = blob.to_vec();
        // Cut into the header JSON region to simulate truncation
        let cut = MAGIC.len() + 4 + 2; // inside header json
        truncated.truncate(cut);
        assert!(parse_blob(&truncated).is_err());
    }

    #[test]
    fn parse_rejects_checksum_mismatch() {
        let meta = KvMeta {
            model_uuid: "z".into(),
            n_ctx: 3,
            n_layer: 3,
            tokenizer_digest: [2; 32],
            template_fingerprint: [2; 32],
            created_at_us: 2,
        };
        let blob = build_blob(meta, b"ok").unwrap();
        let mut tampered = blob.to_vec();
        // Overwrite last payload byte to break checksum (payload is at the end)
        let len = tampered.len();
        tampered[len - 1] ^= 0xAA;
        assert!(parse_blob(&tampered).is_err());
    }
}
