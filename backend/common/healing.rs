//! Repair helpers for malformed tool-call bodies from local models.
//!
//! Local models emit tool calls in several near-JSON dialects that
//! `serde_json` rejects outright: Gemma 4's native `call:name{key:value}`
//! form with `<|"|>` quote tokens and bare keys/values, Mistral's
//! comma-less multi-call arrays (`[{...}{...}]`), and friends. Dropping
//! those calls (or echoing them as literal text) loses the action the
//! model clearly attempted; these helpers normalise each dialect into
//! strict JSON so `tool_calls.rs` can emit a structured call instead.
//!
//! Ported from Unsloth Studio `studio/backend/core/tool_healing.py`
//! (reference clone at `~/workspace/unsloth`); the streaming state
//! machine that decides *where* these run lives in `tool_calls.rs`.

/// Gemma 4's native string-quote token: `<|"|>value<|"|>`.
pub(crate) const GEMMA_QUOTE: &str = "<|\"|>";

/// Byte index of the `}` matching the `{` at `brace_start`, or `None`
/// while unbalanced (still streaming). Honours `\`-escapes and `"`
/// strings; with `gemma_quotes`, also treats `<|"|>...<|"|>` spans as
/// string data so braces inside them don't count.
///
/// Structural chars are all ASCII, so byte scanning is UTF-8 safe and
/// the returned index is always a char boundary.
pub(crate) fn balanced_brace_end(
    content: &str,
    brace_start: usize,
    gemma_quotes: bool,
) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut depth: i32 = 0;
    let mut i = brace_start;
    let mut in_string = false;
    let mut in_gemma_string = false;
    while i < bytes.len() {
        if gemma_quotes && !in_string && content[i..].starts_with(GEMMA_QUOTE) {
            in_gemma_string = !in_gemma_string;
            i += GEMMA_QUOTE.len();
            continue;
        }
        let b = bytes[i];
        if in_gemma_string {
            i += 1;
            continue;
        }
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Byte index of the `]` matching the `[` at `start`, or `None` while
/// unbalanced. Tracks nested `[]`/`{}` and double-quoted strings.
pub(crate) fn balanced_bracket_end(src: &str, start: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth: i32 = 0;
    let mut i = start;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == b'[' || b == b'{' {
            depth += 1;
        } else if b == b']' || b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Decode each top-level element of a JSON array body, tolerating the
/// comma-less object separators Mistral/Ollama multi-call templates emit
/// (`[{...}{...}]`). A single `serde_json::from_str` of the whole body
/// rejects that form and would drop every call.
///
/// `body` is the text *between* the array's brackets.
pub(crate) fn decode_array_items(body: &str) -> Vec<serde_json::Value> {
    let mut items = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n' | b',') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b']' {
            break;
        }
        let mut stream =
            serde_json::Deserializer::from_str(&body[i..]).into_iter::<serde_json::Value>();
        match stream.next() {
            Some(Ok(v)) => {
                let consumed = stream.byte_offset();
                if consumed == 0 {
                    break;
                }
                i += consumed;
                items.push(v);
            }
            _ => break,
        }
    }
    items
}

/// Replace Gemma `<|"|>value<|"|>` spans with strict JSON strings.
fn normalise_gemma_quoted_strings(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        if !src[i..].starts_with(GEMMA_QUOTE) {
            let ch = src[i..].chars().next().expect("in-bounds char");
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        match src[i + GEMMA_QUOTE.len()..].find(GEMMA_QUOTE) {
            None => {
                out.push_str(&src[i..]);
                break;
            }
            Some(rel_end) => {
                let raw = &src[i + GEMMA_QUOTE.len()..i + GEMMA_QUOTE.len() + rel_end];
                out.push_str(&serde_json::Value::String(raw.to_string()).to_string());
                i += GEMMA_QUOTE.len() + rel_end + GEMMA_QUOTE.len();
            }
        }
    }
    out
}

/// Split on commas that are not inside nested `[]`/`{}` or a string.
fn split_top_level_commas(src: &str) -> Vec<&str> {
    let bytes = src.as_bytes();
    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == b'[' || b == b'{' {
            depth += 1;
        } else if b == b']' || b == b'}' {
            depth -= 1;
        } else if b == b',' && depth == 0 {
            parts.push(&src[start..i]);
            start = i + 1;
        }
        i += 1;
    }
    parts.push(&src[start..]);
    parts
}

/// Quote the bare elements of a Gemma array value so JSON parsing
/// succeeds: `labels:[bug,ui]` → `labels:["bug","ui"]`. Object and
/// nested-array elements are normalised recursively; quoted strings,
/// numbers, and JSON literals pass through.
fn quote_gemma_array_elements(body: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for element in split_top_level_commas(body) {
        let stripped = element.trim();
        if stripped.is_empty() || stripped.starts_with('"') {
            out.push(element.to_string());
            continue;
        }
        if stripped.starts_with('{') {
            out.push(quote_gemma_object_keys(stripped));
            continue;
        }
        if stripped.starts_with('[') {
            match balanced_bracket_end(stripped, 0) {
                Some(end) if end == stripped.len() - 1 => {
                    out.push(format!(
                        "[{}]",
                        quote_gemma_array_elements(&stripped[1..end])
                    ));
                }
                _ => out.push(element.to_string()),
            }
            continue;
        }
        if serde_json::from_str::<serde_json::Value>(stripped).is_ok() {
            out.push(element.to_string());
        } else {
            out.push(serde_json::Value::String(stripped.to_string()).to_string());
        }
    }
    out.join(",")
}

/// True when the text at `pos` starts an identifier-shaped `key:` token —
/// the signal that a comma ends a bare value rather than living inside it
/// (`location:New York, NY` keeps its comma; `a:1, b:2` splits). A comma
/// followed by digits-then-colon is value text such as a timestamp
/// (`meet at 10:00, 11:00`), not a new key.
fn next_key_follows(src: &str, pos: usize) -> bool {
    let bytes = src.as_bytes();
    let mut i = pos;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
        return false;
    }
    while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'.' | b'-'))
    {
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i < bytes.len() && bytes[i] == b':'
}

fn json_quote(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

/// Add JSON quotes to Gemma's bare `{key:value}` object form, including
/// bare string values (`{unit:celsius}` → `{"unit":"celsius"}`) and bare
/// array elements. Already-quoted keys/values and JSON scalars pass
/// through untouched.
fn quote_gemma_object_keys(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 16);
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                out.push_str(&src[i..i + 2]);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            let ch = src[i..].chars().next().expect("in-bounds char");
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            continue;
        }
        if b != b'{' && b != b',' {
            let ch = src[i..].chars().next().expect("in-bounds char");
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }

        out.push(b as char);
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_name_start = i;
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-' | b'.'))
        {
            i += 1;
        }
        let key_name = &src[key_name_start..i];
        let mut colon_pos = i;
        while colon_pos < bytes.len() && bytes[colon_pos].is_ascii_whitespace() {
            colon_pos += 1;
        }
        if !key_name.is_empty() && colon_pos < bytes.len() && bytes[colon_pos] == b':' {
            out.push_str(&src[key_start..key_name_start]);
            out.push_str(&json_quote(key_name));
            out.push_str(&src[i..colon_pos]);
            out.push(':');
            i = colon_pos + 1;
            let ws = i;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            out.push_str(&src[ws..i]);
            if i < bytes.len() && bytes[i] == b'[' {
                match balanced_bracket_end(src, i) {
                    None => {
                        out.push_str(&src[i..]);
                        i = bytes.len();
                    }
                    Some(arr_end) => {
                        out.push('[');
                        out.push_str(&quote_gemma_array_elements(&src[i + 1..arr_end]));
                        out.push(']');
                        i = arr_end + 1;
                    }
                }
            } else if i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'{' {
                let v_start = i;
                // Consume the bare value up to `}` or a comma that starts
                // the next key:value pair; a comma inside the value (e.g.
                // `New York, NY`) does not terminate it.
                while i < bytes.len() {
                    if bytes[i] == b'}' {
                        break;
                    }
                    if bytes[i] == b',' && next_key_follows(src, i + 1) {
                        break;
                    }
                    i += 1;
                }
                let raw = &src[v_start..i];
                if serde_json::from_str::<serde_json::Value>(raw.trim()).is_ok() {
                    out.push_str(raw);
                } else {
                    // Quote bare value; empty (`{k:}`) becomes "" so the
                    // parse sees {"k":""} rather than invalid {"k":}.
                    out.push_str(&json_quote(raw.trim()));
                }
            }
        } else {
            out.push_str(&src[key_start..i]);
        }
    }
    out
}

/// Parse Gemma 4's native `call:name{key:value}` argument object into a
/// strict JSON object value. `args_src` is the text between the braces.
pub(crate) fn gemma_arguments_to_json(args_src: &str) -> Option<serde_json::Value> {
    let args_src = args_src.trim();
    if args_src.is_empty() {
        return Some(serde_json::json!({}));
    }
    let src = normalise_gemma_quoted_strings(args_src);
    let src = format!("{{{src}}}");
    let src = quote_gemma_object_keys(&src);
    let v: serde_json::Value = serde_json::from_str(&src).ok()?;
    v.is_object().then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_bare_keys_and_values() {
        let v = gemma_arguments_to_json("unit:celsius,city:Lisbon").unwrap();
        assert_eq!(v["unit"], "celsius");
        assert_eq!(v["city"], "Lisbon");
    }

    #[test]
    fn gemma_quote_tokens_become_strings() {
        let v = gemma_arguments_to_json(r#"query:<|"|>rust {braces} tutorial<|"|>"#).unwrap();
        assert_eq!(v["query"], "rust {braces} tutorial");
    }

    #[test]
    fn gemma_value_comma_not_a_key_boundary() {
        let v = gemma_arguments_to_json("location:New York, NY,unit:f").unwrap();
        assert_eq!(v["location"], "New York, NY");
        assert_eq!(v["unit"], "f");
    }

    #[test]
    fn gemma_timestamp_comma_stays_in_value() {
        let v = gemma_arguments_to_json("when:meet at 10:00, 11:00 tomorrow").unwrap();
        assert_eq!(v["when"], "meet at 10:00, 11:00 tomorrow");
    }

    #[test]
    fn gemma_bare_array_elements_quoted() {
        let v = gemma_arguments_to_json("labels:[bug,ui],count:2").unwrap();
        assert_eq!(v["labels"], serde_json::json!(["bug", "ui"]));
        assert_eq!(v["count"], 2);
    }

    #[test]
    fn gemma_nested_object_array() {
        let v = gemma_arguments_to_json("items:[{path:a},{path:b}]").unwrap();
        assert_eq!(v["items"][0]["path"], "a");
        assert_eq!(v["items"][1]["path"], "b");
    }

    #[test]
    fn gemma_empty_value_becomes_empty_string() {
        let v = gemma_arguments_to_json("k:").unwrap();
        assert_eq!(v["k"], "");
    }

    #[test]
    fn gemma_numbers_and_literals_pass_through() {
        let v = gemma_arguments_to_json("n:3,flag:true,x:null").unwrap();
        assert_eq!(v["n"], 3);
        assert_eq!(v["flag"], true);
        assert!(v["x"].is_null());
    }

    #[test]
    fn balanced_brace_gemma_quotes_hide_braces() {
        let s = r#"{q:<|"|>a } b<|"|>}"#;
        let end = balanced_brace_end(s, 0, true).unwrap();
        assert_eq!(end, s.len() - 1);
    }

    #[test]
    fn comma_less_array_items_decode() {
        let items =
            decode_array_items(r#"{"name":"a","arguments":{}}{"name":"b","arguments":{"x":1}}"#);
        assert_eq!(items.len(), 2);
        assert_eq!(items[1]["name"], "b");
    }

    #[test]
    fn comma_separated_array_items_decode() {
        let items = decode_array_items(r#"{"name":"a"}, {"name":"b"}"#);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn multibyte_text_survives_key_quoting() {
        let v = gemma_arguments_to_json("city:São Paulo,emoji:🐣").unwrap();
        assert_eq!(v["city"], "São Paulo");
        assert_eq!(v["emoji"], "🐣");
    }
}
