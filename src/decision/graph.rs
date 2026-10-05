//! Inspect ONNX protobuf metadata before native code can follow external paths.
//! Unknown fields are skipped without decoding tensor weights. This supports
//! dense Laya exports; sparse tensors and local functions are rejected.
use super::{
    BundleManifest, Result,
    bundle::{invalid, safe_name},
};
use std::collections::BTreeMap;
#[derive(Clone, Copy)]
enum Value<'a> {
    Int(u64),
    Bytes(&'a [u8]),
    Other,
}
fn varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut n = 0u64;
    for shift in (0..70).step_by(7) {
        let b = *data
            .get(*pos)
            .ok_or_else(|| invalid("truncated protobuf"))?;
        *pos += 1;
        if shift == 63 && b > 1 {
            return Err(invalid("protobuf integer overflow"));
        }
        n |= ((b & 127) as u64) << shift;
        if b & 128 == 0 {
            return Ok(n);
        }
    }
    Err(invalid("invalid protobuf integer"))
}
fn fields(data: &[u8]) -> Result<Vec<(u64, Value<'_>)>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let key = varint(data, &mut pos)?;
        if key >> 3 == 0 {
            return Err(invalid("invalid protobuf field"));
        }
        let v = match key & 7 {
            0 => Value::Int(varint(data, &mut pos)?),
            2 => {
                let n = usize::try_from(varint(data, &mut pos)?)
                    .map_err(|_| invalid("oversized protobuf field"))?;
                let end = pos
                    .checked_add(n)
                    .filter(|e| *e <= data.len())
                    .ok_or_else(|| invalid("truncated protobuf field"))?;
                let b = &data[pos..end];
                pos = end;
                Value::Bytes(b)
            }
            1 | 5 => {
                pos = pos
                    .checked_add(if key & 7 == 1 { 8 } else { 4 })
                    .filter(|p| *p <= data.len())
                    .ok_or_else(|| invalid("truncated protobuf fixed value"))?;
                Value::Other
            }
            _ => return Err(invalid("unsupported protobuf wire type")),
        };
        out.push((key >> 3, v));
    }
    Ok(out)
}
fn bytes(v: Value<'_>) -> Result<&[u8]> {
    if let Value::Bytes(b) = v {
        Ok(b)
    } else {
        Err(invalid("expected protobuf message"))
    }
}
fn string(v: Value<'_>) -> Result<String> {
    std::str::from_utf8(bytes(v)?)
        .map(str::to_owned)
        .map_err(|_| invalid("invalid protobuf text"))
}
fn tensor(data: &[u8], m: &BundleManifest) -> Result<()> {
    let mut location = None;
    let mut external = false;
    for (n, v) in fields(data)? {
        match n {
            13 => {
                let mut key = String::new();
                let mut val = String::new();
                for (n, v) in fields(bytes(v)?)? {
                    if n == 1 {
                        key = string(v)?;
                    } else if n == 2 {
                        val = string(v)?;
                    }
                }
                if key == "location" {
                    if location.is_some() {
                        return Err(invalid("duplicate external location"));
                    }
                    location = Some(val);
                }
            }
            14 => {
                external = matches!(v, Value::Int(1));
            }
            _ => {}
        }
    }
    if let Some(path) = location {
        safe_name(&path)?;
        if !m.files.contains_key(&path) {
            return Err(invalid(format!("undeclared external tensor data: {path}")));
        }
    } else if external {
        return Err(invalid("external tensor has no location"));
    }
    Ok(())
}
fn graph(data: &[u8], m: &BundleManifest, depth: usize) -> Result<()> {
    if depth > 32 {
        return Err(invalid("ONNX subgraph nesting exceeds parser depth"));
    }
    for (n, v) in fields(data)? {
        match n {
            5 => tensor(bytes(v)?, m)?,
            15 => return Err(invalid("sparse ONNX graphs are unsupported")),
            1 => {
                for (n, v) in fields(bytes(v)?)? {
                    if n == 5 {
                        for (n, v) in fields(bytes(v)?)? {
                            match n {
                                5 | 10 => tensor(bytes(v)?, m)?,
                                6 | 11 => graph(bytes(v)?, m, depth + 1)?,
                                22 | 23 => {
                                    return Err(invalid("sparse ONNX attributes are unsupported"));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
fn signature(data: &[u8]) -> Result<(String, (u64, usize))> {
    let mut name = String::new();
    let mut element = 0;
    let mut rank = None;
    for (n, v) in fields(data)? {
        if n == 1 {
            name = string(v)?;
        } else if n == 2 {
            for (n, v) in fields(bytes(v)?)? {
                if n == 1 {
                    for (n, v) in fields(bytes(v)?)? {
                        if n == 1 {
                            if let Value::Int(i) = v {
                                element = i;
                            }
                        } else if n == 2 {
                            rank = Some(fields(bytes(v)?)?.iter().filter(|(n, _)| *n == 1).count());
                        }
                    }
                }
            }
        }
    }
    Ok((
        name,
        (
            element,
            rank.ok_or_else(|| invalid("missing tensor shape"))?,
        ),
    ))
}
pub(super) fn validate(data: &[u8], m: &BundleManifest) -> Result<()> {
    let mut graphs = Vec::new();
    let mut opset = None;
    for (n, v) in fields(data)? {
        match n {
            7 => graphs.push(bytes(v)?),
            25 => return Err(invalid("local ONNX functions are unsupported")),
            8 => {
                let mut domain = String::new();
                let mut version = 0;
                for (n, v) in fields(bytes(v)?)? {
                    if n == 1 {
                        domain = string(v)?;
                    } else if n == 2
                        && let Value::Int(i) = v
                    {
                        version = i;
                    }
                }
                if domain.is_empty() {
                    opset = Some(version);
                } else {
                    return Err(invalid("custom ONNX domains are unsupported"));
                }
            }
            _ => {}
        }
    }
    if graphs.len() != 1 || opset != Some(u64::from(m.opset)) {
        return Err(invalid("ONNX graph count or opset mismatch"));
    }
    graph(graphs[0], m, 0)?;
    let mut inputs = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    for (n, v) in fields(graphs[0])? {
        if n == 11 || n == 12 {
            let (name, sig) = signature(bytes(v)?)?;
            if (if n == 11 { &mut inputs } else { &mut outputs })
                .insert(name, sig)
                .is_some()
            {
                return Err(invalid("duplicate tensor name"));
            }
        }
    }
    let expected_inputs = BTreeMap::from([
        ("input_ids".into(), (7, 2)),
        ("attention_mask".into(), (7, 2)),
        ("marker_pos".into(), (7, 2)),
        ("marker_mask".into(), (9, 2)),
        ("qtype".into(), (7, 1)),
    ]);
    let expected_outputs =
        BTreeMap::from([("logits".into(), (1, 2)), ("act_logits".into(), (1, 2))]);
    if inputs != expected_inputs || outputs != expected_outputs {
        return Err(invalid("ONNX tensor signature mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> BundleManifest {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/laya/smoke/manifest.json"
        ))
        .unwrap()
    }
    #[test]
    fn external_data_must_be_declared_and_cannot_escape() {
        for path in ["../escape.data", "unlisted.data", "C:/data", "/absolute"] {
            let mut pair = vec![10, 8];
            pair.extend_from_slice(b"location");
            pair.extend([18, path.len() as u8]);
            pair.extend_from_slice(path.as_bytes());
            let mut t = vec![106, pair.len() as u8];
            t.extend(pair);
            t.extend([112, 1]);
            assert!(tensor(&t, &manifest()).is_err());
        }
        assert!(tensor(&[112, 1], &manifest()).is_err());
        assert!(
            validate(
                include_bytes!("../../tests/fixtures/laya/smoke/model.onnx"),
                &manifest()
            )
            .is_ok()
        );
    }
    proptest::proptest! {
        #[test]
        fn arbitrary_protobuf_fails_without_panicking(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(),0..2048)) {
            let _ = validate(&bytes,&manifest());
        }
    }
}
