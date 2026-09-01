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

/// The only header layout this build understands. A blob written by a
/// different codec carries a payload this one cannot interpret, and every
/// other check would still pass it: the SHA-256 is over bytes, not over
/// their meaning. The version is the only thing standing between a
/// re-layout and a wrong-format KV blob handed to the backend as raw state.
///
/// Version 2 added `header_sha256` to the framing (see [`build_blob`]).
/// Version-1 blobs are refused, not upgraded — they are cache entries, and
/// the store's budget sweep reclaims them.
const FORMAT_VERSION: u16 = 2;

/// Width of the framing's header digest.
const HEADER_DIGEST_LEN: usize = 32;

/// Offset of the header JSON: magic, then the 4-byte length, then the digest.
const HEADER_START: usize = MAGIC.len() + 4 + HEADER_DIGEST_LEN;

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Frame a KV snapshot:
///
/// ```text
/// MAGIC(7) | header_len: u32 | header_sha256: [u8; 32] | header_json | payload
/// ```
///
/// Two digests, because the header is as load-bearing as the payload:
/// `payload_sha256` (inside the header) covers the state bytes, and
/// `header_sha256` (in the framing, where it can cover the header itself)
/// covers the metadata that decides whether those state bytes may be
/// restored at all. A header-only digest cannot live inside the header.
pub fn build_blob(meta: KvMeta, payload: &[u8]) -> anyhow::Result<Bytes> {
    let header = KvHeader {
        version: FORMAT_VERSION,
        meta,
        payload_sha256: sha256(payload),
    };
    let header_json = serde_json::to_vec(&header)?;
    let header_digest = sha256(&header_json);
    let mut out = Vec::with_capacity(HEADER_START + header_json.len() + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_digest);
    out.extend_from_slice(&header_json);
    out.extend_from_slice(payload);
    Ok(Bytes::from(out))
}

pub fn parse_blob(bytes: &[u8]) -> anyhow::Result<(KvHeader, &[u8])> {
    if bytes.len() < HEADER_START {
        anyhow::bail!("blob too small");
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        anyhow::bail!("bad magic");
    }
    let len_bytes: [u8; 4] = bytes[MAGIC.len()..MAGIC.len() + 4]
        .try_into()
        .context("invalid header length bytes")?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;
    let header_digest = &bytes[MAGIC.len() + 4..HEADER_START];
    // `header_len` is whatever the file says; on a 32-bit target the sum
    // overflows usize before the truncation check can reject it.
    let header_end = HEADER_START
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("header length out of range"))?;
    if bytes.len() < header_end {
        anyhow::bail!("blob truncated");
    }
    let header_json = &bytes[HEADER_START..header_end];
    // Before trusting anything the header says. Serde would otherwise
    // accept a corrupted header: a damaged field *name* becomes an unknown
    // field and the field it displaced falls back to its `serde(default)`,
    // which is how a saved token count silently became zero.
    if sha256(header_json) != header_digest {
        anyhow::bail!("header checksum mismatch");
    }
    let header: KvHeader = serde_json::from_slice(header_json)?;
    if header.version != FORMAT_VERSION {
        anyhow::bail!(
            "unsupported KV blob version {} (this build writes {FORMAT_VERSION})",
            header.version
        );
    }
    let payload = &bytes[header_end..];
    if sha256(payload) != header.payload_sha256 {
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
    use proptest::prelude::prop;
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
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
            kv_token_count: 0,
            transcript_sha256: [0u8; 32],
        };
        let blob = build_blob(meta, b"ok").unwrap();
        let mut tampered = blob.to_vec();
        // Overwrite last payload byte to break checksum (payload is at the end)
        let len = tampered.len();
        tampered[len - 1] ^= 0xAA;
        assert!(parse_blob(&tampered).is_err());
    }

    // ── Adversarial blob fixtures ──────────────────────────────────────
    //
    // `parse_blob` reads a file the process wrote earlier but does not
    // control: the KV directory is on disk, is not integrity-protected as a
    // whole, and its contents are handed to the backend as raw session
    // state. Arbitrary bytes must produce `Ok` or `Err` and never a panic.

    fn meta_fixture() -> KvMeta {
        KvMeta {
            model_uuid: "model-uuid".into(),
            n_ctx: 4096,
            n_layer: 32,
            tokenizer_digest: [7; 32],
            template_fingerprint: [9; 32],
            created_at_us: 1_700_000_000_000_000,
            kv_token_count: 12_500,
            transcript_sha256: [3; 32],
        }
    }

    /// Frame arbitrary header bytes into an otherwise well-formed blob,
    /// recomputing the length prefix and the header digest so that only the
    /// header *content* differs from a blob this codec would have written.
    fn blob_with_header_json(header_json: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&sha256(header_json));
        out.extend_from_slice(header_json);
        out.extend_from_slice(payload);
        out
    }

    fn blob_with_header(header: &KvHeader, payload: &[u8]) -> Vec<u8> {
        blob_with_header_json(&serde_json::to_vec(header).unwrap(), payload)
    }

    // ── round trip ─────────────────────────────────────────────────────

    #[test]
    fn an_empty_payload_round_trips() {
        // The degenerate case: a header with nothing behind it. The payload
        // slice must come back empty rather than borrowing into the header.
        let blob = build_blob(meta_fixture(), b"").unwrap();
        let (hdr, payload) = parse_blob(&blob).unwrap();
        assert_eq!(hdr.meta, meta_fixture());
        assert!(payload.is_empty());
    }

    #[test]
    fn every_identity_field_survives_the_round_trip_byte_for_byte() {
        // The restore path compares these field by field, so a codec that
        // dropped or defaulted one would turn a mismatch into a false
        // match — the KV blob equivalent of loading someone else's state.
        let meta = meta_fixture();
        let blob = build_blob(meta.clone(), b"payload").unwrap();
        let (hdr, _) = parse_blob(&blob).unwrap();
        assert_eq!(hdr.version, FORMAT_VERSION);
        assert_eq!(hdr.meta.model_uuid, meta.model_uuid);
        assert_eq!(hdr.meta.n_ctx, meta.n_ctx);
        assert_eq!(hdr.meta.n_layer, meta.n_layer);
        assert_eq!(hdr.meta.tokenizer_digest, meta.tokenizer_digest);
        assert_eq!(hdr.meta.template_fingerprint, meta.template_fingerprint);
        assert_eq!(hdr.meta.created_at_us, meta.created_at_us);
        assert_eq!(hdr.meta.kv_token_count, meta.kv_token_count);
        assert_eq!(hdr.meta.transcript_sha256, meta.transcript_sha256);
    }

    // ── rejection ──────────────────────────────────────────────────────

    #[test]
    fn a_blob_shorter_than_the_fixed_prefix_is_refused() {
        // The `< HEADER_START` guard, at each of its boundaries.
        for len in 0..=HEADER_START {
            let short = vec![0u8; len];
            let result = parse_blob(&short);
            assert!(result.is_err(), "{len} bytes parsed as a blob");
        }
    }

    #[test]
    fn a_blob_truncated_at_every_prefix_length_is_refused() {
        let blob = build_blob(meta_fixture(), b"the payload").unwrap();
        for cut in 0..blob.len() {
            let result = parse_blob(&blob[..cut]);
            assert!(
                result.is_err(),
                "a {cut}-byte prefix of a {}-byte blob parsed as valid",
                blob.len()
            );
        }
        assert!(parse_blob(&blob).is_ok());
    }

    #[test]
    fn every_single_byte_of_the_magic_is_checked() {
        let blob = build_blob(meta_fixture(), b"x").unwrap();
        for i in 0..MAGIC.len() {
            let mut bad = blob.to_vec();
            bad[i] ^= 0x01;
            assert!(
                parse_blob(&bad).is_err(),
                "magic byte {i} is not being checked"
            );
        }
    }

    #[test]
    fn a_header_length_running_past_the_blob_is_refused_not_sliced() {
        // The length prefix is the one attacker-controlled offset in the
        // format; an unchecked slice here would panic on an out-of-range
        // index rather than return an error.
        let blob = build_blob(meta_fixture(), b"payload").unwrap();
        for declared in [u32::MAX, u32::MAX - 1, blob.len() as u32, 1 << 30] {
            let mut bad = blob.to_vec();
            bad[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&declared.to_le_bytes());
            // (the header digest no longer matches either, but the length
            // check must fire first — an out-of-range slice would panic)
            let result = parse_blob(&bad);
            assert!(
                result.is_err(),
                "declared header length {declared} accepted"
            );
        }
    }

    #[test]
    fn a_header_length_of_zero_is_refused_as_unparseable_json() {
        let blob = build_blob(meta_fixture(), b"payload").unwrap();
        let mut bad = blob.to_vec();
        bad[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse_blob(&bad).is_err());
    }

    #[test]
    fn a_blob_from_a_future_format_version_is_refused() {
        // Nothing else in the format would catch this: the checksum covers
        // the payload's bytes, not their layout, and the identity fields
        // would compare equal. Version is the only gate.
        let payload = b"state bytes";
        for version in [0u16, 1, 3, u16::MAX] {
            let header = KvHeader {
                version,
                meta: meta_fixture(),
                payload_sha256: sha256(payload),
            };
            let blob = blob_with_header(&header, payload);
            let err = parse_blob(&blob)
                .expect_err("version {version} must be refused")
                .to_string();
            assert!(err.contains("unsupported KV blob version"), "{err}");
        }
    }

    #[test]
    fn a_header_that_is_not_valid_json_is_refused() {
        let payload = b"payload";
        let junk = br#"{"version":2,"meta":"#;
        assert!(parse_blob(&blob_with_header_json(junk, payload)).is_err());
    }

    #[test]
    fn a_header_missing_a_required_field_is_refused() {
        // `kv_token_count` and `transcript_sha256` carry `#[serde(default)]`
        // for pre-keepwarm blobs, but the identity fields do not — a header
        // without `model_uuid` must not deserialize to an empty one.
        let payload = b"payload";
        let digest = sha256(payload);
        let zeroes = vec![0u8; 32];
        let header_json = serde_json::json!({
            "version": 2,
            "meta": { "n_ctx": 1, "n_layer": 1,
                      "tokenizer_digest": zeroes, "template_fingerprint": zeroes,
                      "created_at_us": 0 },
            "payload_sha256": digest,
        });
        let header_bytes = serde_json::to_vec(&header_json).unwrap();
        assert!(parse_blob(&blob_with_header_json(&header_bytes, payload)).is_err());
    }

    #[test]
    fn a_single_flipped_bit_anywhere_in_the_payload_fails_the_checksum() {
        let payload: Vec<u8> = (0u8..64).collect();
        let blob = build_blob(meta_fixture(), &payload).unwrap();
        let payload_start = blob.len() - payload.len();
        for i in payload_start..blob.len() {
            for bit in [0x01u8, 0x80] {
                let mut tampered = blob.to_vec();
                tampered[i] ^= bit;
                assert!(
                    parse_blob(&tampered).is_err(),
                    "flipping bit {bit:#x} of payload byte {i} passed the checksum"
                );
            }
        }
    }

    #[test]
    fn a_payload_extended_with_extra_bytes_fails_the_checksum() {
        // Truncation and extension both change the digest; neither may be
        // read back as a shorter or longer valid payload.
        let blob = build_blob(meta_fixture(), b"exactly this").unwrap();
        let mut extended = blob.to_vec();
        extended.push(0);
        assert!(parse_blob(&extended).is_err());

        let mut shortened = blob.to_vec();
        shortened.pop();
        assert!(parse_blob(&shortened).is_err());
    }

    #[test]
    fn a_declared_checksum_that_matches_nothing_is_refused() {
        let payload = b"real payload";
        let header = KvHeader {
            version: FORMAT_VERSION,
            meta: meta_fixture(),
            payload_sha256: [0xAB; 32],
        };
        assert!(parse_blob(&blob_with_header(&header, payload)).is_err());
    }

    #[test]
    fn a_fingerprint_mismatch_survives_the_codec_so_the_caller_can_see_it() {
        // The codec does not judge identity — the restore path does, by
        // comparing these fields. What the codec owes is that a differing
        // fingerprint comes back differing, so the comparison can fail.
        let mut other = meta_fixture();
        other.template_fingerprint = [0xEE; 32];
        other.tokenizer_digest = [0xDD; 32];
        let blob = build_blob(other.clone(), b"payload").unwrap();
        let (hdr, _) = parse_blob(&blob).unwrap();
        assert_ne!(
            hdr.meta.template_fingerprint,
            meta_fixture().template_fingerprint
        );
        assert_ne!(hdr.meta.tokenizer_digest, meta_fixture().tokenizer_digest);
        assert_eq!(hdr.meta, other);
    }

    #[test]
    fn arbitrary_bytes_never_panic_the_parser() {
        // Deterministic stand-in for the fuzz target, so this invariant is
        // enforced in ordinary CI too.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let valid = build_blob(meta_fixture(), b"seed payload").unwrap();
        for _ in 0..4000 {
            let mut bytes = valid.to_vec();
            // Mutate between one and four bytes of a valid blob, plus a
            // truncation, so the generator lands near the format's edges
            // rather than in undifferentiated noise.
            let mutations = 1 + (next() % 4);
            for _ in 0..mutations {
                let idx = (next() as usize) % bytes.len();
                bytes[idx] = (next() & 0xFF) as u8;
            }
            let keep = (next() as usize) % (bytes.len() + 1);
            bytes.truncate(keep);
            let _ = parse_blob(&bytes);
        }
    }

    // ── property: decode(encode(x)) == x ───────────────────────────────

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(512))]

        /// The round-trip law, over arbitrary well-formed snapshots.
        #[test]
        fn decoding_an_encoded_snapshot_returns_exactly_what_was_encoded(
            model_uuid in ".{0,64}",
            n_ctx: u32,
            n_layer: u32,
            tokenizer_digest: [u8; 32],
            template_fingerprint: [u8; 32],
            created_at_us: i64,
            kv_token_count: u64,
            transcript_sha256: [u8; 32],
            payload in proptest::collection::vec(proptest::num::u8::ANY, 0..4096),
        ) {
            let meta = KvMeta {
                model_uuid, n_ctx, n_layer, tokenizer_digest,
                template_fingerprint, created_at_us, kv_token_count,
                transcript_sha256,
            };
            let blob = build_blob(meta.clone(), &payload).unwrap();
            let (hdr, decoded) = parse_blob(&blob).unwrap();
            proptest::prop_assert_eq!(hdr.meta, meta);
            proptest::prop_assert_eq!(decoded, &payload[..]);
            proptest::prop_assert_eq!(hdr.version, FORMAT_VERSION);
        }

        /// Any single-byte corruption of a valid blob is either detected or
        /// is a no-op rewrite of the same byte — it never decodes to a
        /// *different* payload while reporting success.
        #[test]
        fn no_single_byte_corruption_ever_decodes_to_a_different_payload(
            payload in proptest::collection::vec(proptest::num::u8::ANY, 1..256),
            index: prop::sample::Index,
            replacement: u8,
        ) {
            let meta = meta_fixture();
            let blob = build_blob(meta.clone(), &payload).unwrap();
            let i = index.index(blob.len());
            let mut corrupted = blob.to_vec();
            corrupted[i] = replacement;
            if let Ok((hdr, decoded)) = parse_blob(&corrupted) {
                proptest::prop_assert_eq!(decoded, &payload[..]);
                proptest::prop_assert_eq!(hdr.meta, meta);
            }
        }

        /// Arbitrary bytes: any result is fine, a panic is not.
        #[test]
        fn parsing_arbitrary_bytes_never_panics(
            bytes in proptest::collection::vec(proptest::num::u8::ANY, 0..512),
        ) {
            let _ = parse_blob(&bytes);
        }

        /// Same, but seeded with the real magic so the generator gets past
        /// the first gate and exercises the length/JSON/checksum path.
        #[test]
        fn parsing_arbitrary_bytes_behind_a_valid_magic_never_panics(
            tail in proptest::collection::vec(proptest::num::u8::ANY, 0..512),
        ) {
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&tail);
            let _ = parse_blob(&bytes);
        }
    }
}
