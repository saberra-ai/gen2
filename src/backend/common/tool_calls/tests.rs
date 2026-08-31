//! Parser + healing corpus. The `corpus` test is the ADR-0036 gate for
//! this capability: every entry is a (possibly malformed) model output
//! with its expected structured calls, and the suite FAILS LOUD if the
//! corpus shrinks below the floor — a capability must not go green by
//! deleting its fixtures.

use std::collections::HashSet;

use super::*;

fn texts(events: &[ParserOutput]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            ParserOutput::Text(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

fn calls(events: &[ParserOutput]) -> Vec<&ToolCall> {
    events
        .iter()
        .filter_map(|e| match e {
            ParserOutput::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect()
}

fn args_of(tc: &ToolCall) -> serde_json::Value {
    // Bare-string arguments stay raw on the ToolCall (parser contract);
    // normalise to a JSON string for comparison.
    serde_json::from_str(&tc.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(tc.arguments.clone()))
}

/// Run one input through the parser (single push + flush).
fn run(input: &str) -> Vec<ParserOutput> {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = p.push(input);
    out.extend(p.flush());
    out
}

/// Run one input pushed byte-by-byte — every fixture must parse
/// identically under worst-case chunking.
fn run_bytewise(input: &str) -> Vec<ParserOutput> {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in input.chars() {
        buf.push(ch);
        out.extend(p.push(&buf));
        buf.clear();
    }
    out.extend(p.flush());
    out
}

// ---------------------------------------------------------------
// Regression pins from the pre-healing parser (behavior unchanged)
// ---------------------------------------------------------------

#[test]
fn plain_text_passes_through() {
    let out = run("hello world");
    assert_eq!(texts(&out), "hello world");
    assert!(calls(&out).is_empty());
}

#[test]
fn tag_tool_call_single_chunk() {
    let out = run(
        r#"preface <tool_call>{"name":"get_weather","arguments":{"city":"Lisbon"}}</tool_call> done"#,
    );
    let cs = calls(&out);
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].name, "get_weather");
    assert!(cs[0].arguments.contains("Lisbon"));
    assert_eq!(texts(&out), "preface  done");
}

#[test]
fn tag_tool_call_split_across_chunks() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut all = Vec::new();
    all.extend(p.push("hello <tool"));
    all.extend(p.push("_call>{\"name"));
    all.extend(p.push(r#"":"foo","arguments":{}}</too"#));
    all.extend(p.push("l_call> bye"));
    all.extend(p.flush());
    assert!(matches!(&all[0], ParserOutput::Text(s) if s == "hello "));
    let cs = calls(&all);
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].name, "foo");
}

#[test]
fn bracket_format_parses() {
    let out =
        run(r#"go [TOOL_CALLS][{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}] end"#);
    let cs = calls(&out);
    assert_eq!(cs.len(), 2);
    assert_eq!(cs[0].name, "a");
    assert_eq!(cs[1].name, "b");
}

#[test]
fn malformed_tool_call_emits_as_text() {
    let out = run(r#"<tool_call>not json</tool_call>"#);
    assert!(calls(&out).is_empty());
    let joined = texts(&out);
    assert!(joined.contains("<tool_call>"));
    assert!(joined.contains("not json"));
}

#[test]
fn partial_marker_at_tail_is_held() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let out = p.push("hello <tool");
    assert_eq!(texts(&out), "hello ");
    let out2 = p.push(r#"_call>{"name":"x","arguments":{}}</tool_call>"#);
    assert!(calls(&out2).iter().any(|tc| tc.name == "x"));
}

#[test]
fn flush_releases_partial_as_text() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let _ = p.push("hello <tool");
    let flushed = p.flush();
    assert!(texts(&flushed).contains("<tool"));
}

// ---------------------------------------------------------------
// Healing corpus — every entry: (label, raw model output,
// expected calls as (name, args-json)). Text fidelity is asserted
// separately where it matters.
// ---------------------------------------------------------------

struct Fixture {
    label: &'static str,
    input: &'static str,
    expect: &'static [(&'static str, &'static str)],
}

const CORPUS: &[Fixture] = &[
    // -- TagJson variants --
    Fixture {
        label: "tag-json canonical",
        input: r#"<tool_call>{"name":"search","arguments":{"q":"rust"}}</tool_call>"#,
        expect: &[("search", r#"{"q":"rust"}"#)],
    },
    Fixture {
        label: "tag-json parameters alias (llama3.2 hermes drift)",
        input: r#"<tool_call>{"name":"search","parameters":{"q":"rust"}}</tool_call>"#,
        expect: &[("search", r#"{"q":"rust"}"#)],
    },
    Fixture {
        label: "tag-json args alias",
        input: r#"<tool_call>{"name":"search","args":{"q":"rust"}}</tool_call>"#,
        expect: &[("search", r#"{"q":"rust"}"#)],
    },
    Fixture {
        label: "tag-json string arguments stay raw",
        input: r#"<tool_call>{"name":"echo","arguments":"weather"}</tool_call>"#,
        expect: &[("echo", "\"weather\"")],
    },
    // -- Gemma 4 native --
    Fixture {
        label: "gemma bare keys and values",
        input: "<|tool_call>call:get_weather{city:Lisbon,unit:celsius}<tool_call|>",
        expect: &[("get_weather", r#"{"city":"Lisbon","unit":"celsius"}"#)],
    },
    Fixture {
        label: "gemma quote tokens with braces inside",
        input: r#"<|tool_call>call:search{query:<|"|>rust {book}<|"|>}<tool_call|>"#,
        expect: &[("search", r#"{"query":"rust {book}"}"#)],
    },
    Fixture {
        label: "gemma bare array elements",
        input: "<|tool_call>call:label{labels:[bug,ui],n:2}<tool_call|>",
        expect: &[("label", r#"{"labels":["bug","ui"],"n":2}"#)],
    },
    Fixture {
        label: "gemma value comma is not a key boundary",
        input: "<|tool_call>call:geo{location:New York, NY}<tool_call|>",
        expect: &[("geo", r#"{"location":"New York, NY"}"#)],
    },
    Fixture {
        label: "gemma missing close tag at eos",
        input: "<|tool_call>call:ping{host:localhost}",
        expect: &[("ping", r#"{"host":"localhost"}"#)],
    },
    Fixture {
        label: "gemma dotted tool name",
        input: "<|tool_call>call:fs.read{path:a.txt}<tool_call|>",
        expect: &[("fs.read", r#"{"path":"a.txt"}"#)],
    },
    // -- Function XML (qwen3-coder style) --
    Fixture {
        label: "func-xml two parameters",
        input: "<function=create_issue><parameter=title>Bug in parser</parameter><parameter=body>It broke.</parameter></function>",
        expect: &[(
            "create_issue",
            r#"{"title":"Bug in parser","body":"It broke."}"#,
        )],
    },
    Fixture {
        label: "func-xml wrapped in tool_call tags",
        input: "<tool_call><function=grep><parameter=pattern>fn main</parameter></function></tool_call>",
        expect: &[("grep", r#"{"pattern":"fn main"}"#)],
    },
    Fixture {
        label: "func-xml literal close tag inside parameter value",
        input: "<function=write><parameter=content>end with </function> marker</parameter></function>",
        expect: &[("write", r#"{"content":"end with </function> marker"}"#)],
    },
    // -- Mistral bracket forms --
    Fixture {
        label: "mistral array canonical",
        input: r#"[TOOL_CALLS][{"name":"a","arguments":{"x":1}}]"#,
        expect: &[("a", r#"{"x":1}"#)],
    },
    Fixture {
        label: "mistral comma-less multi-call array",
        input: r#"[TOOL_CALLS][{"name":"a","arguments":{}}{"name":"b","arguments":{"x":1}}]"#,
        expect: &[("a", "{}"), ("b", r#"{"x":1}"#)],
    },
    Fixture {
        label: "mistral array null arguments become empty object",
        input: r#"[TOOL_CALLS][{"name":"a","arguments":null}]"#,
        expect: &[("a", "{}")],
    },
    Fixture {
        label: "mistral name form",
        input: r#"[TOOL_CALLS]web_search{"query":"pio"}"#,
        expect: &[("web_search", r#"{"query":"pio"}"#)],
    },
    Fixture {
        label: "mistral v11 call-id and args metadata",
        input: r#"[TOOL_CALLS]web_search[CALL_ID]abc123[ARGS]{"query":"pio"}[/TOOL_CALLS]"#,
        expect: &[("web_search", r#"{"query":"pio"}"#)],
    },
    // -- Think-block rehearsals --
    Fixture {
        label: "rehearsal inside think block is not executed",
        input: "<think>I could call <tool_call>{\"name\":\"rm\",\"arguments\":{}}</tool_call> here</think>ok",
        expect: &[],
    },
    Fixture {
        label: "real call after think block executes",
        input: "<think>plan</think><tool_call>{\"name\":\"go\",\"arguments\":{}}</tool_call>",
        expect: &[("go", "{}")],
    },
    Fixture {
        label: "bracket-think rehearsal skipped",
        input: "[THINK]try [TOOL_CALLS][{\"name\":\"x\",\"arguments\":{}}][/THINK]done",
        expect: &[],
    },
    Fixture {
        label: "gemma thought channel rehearsal skipped",
        input: "<|channel>thought\nmaybe <|tool_call>call:rm{path:/}<tool_call|>\n<channel|>safe",
        expect: &[],
    },
    // -- Adversarial / pathological --
    Fixture {
        label: "unclosed tag at eos stays text",
        input: r#"<tool_call>{"name":"x","argu"#,
        expect: &[],
    },
    Fixture {
        label: "empty gemma args",
        input: "<|tool_call>call:noop{}<tool_call|>",
        expect: &[("noop", "{}")],
    },
    Fixture {
        label: "nested json depth in tag body",
        input: r#"<tool_call>{"name":"deep","arguments":{"a":{"b":{"c":[1,{"d":"}"}]}}}}</tool_call>"#,
        expect: &[("deep", r#"{"a":{"b":{"c":[1,{"d":"}"}]}}}"#)],
    },
    Fixture {
        label: "two calls in one message",
        input: "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> and <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        expect: &[("a", "{}"), ("b", "{}")],
    },
    Fixture {
        label: "prose mentioning TOOL_CALLS without call shape stays text",
        input: "the [TOOL_CALLS] marker is used by Mistral models",
        expect: &[],
    },
];

/// Minimum corpus size — shrinking the corpus below this fails the
/// suite. Raise it when adding fixtures; never lower it to go green.
const CORPUS_FLOOR: usize = 25;

#[test]
fn corpus_floor_holds() {
    assert!(
        CORPUS.len() >= CORPUS_FLOOR,
        "healing corpus shrank to {} fixtures (floor {CORPUS_FLOOR}) — capability must not go green by deleting fixtures",
        CORPUS.len()
    );
}

#[test]
fn corpus_parses_single_push() {
    for f in CORPUS {
        let out = run(f.input);
        let cs = calls(&out);
        assert_eq!(cs.len(), f.expect.len(), "[{}] call count", f.label);
        for (i, (name, args)) in f.expect.iter().enumerate() {
            assert_eq!(cs[i].name, *name, "[{}] call {i} name", f.label);
            let expected: serde_json::Value = serde_json::from_str(args).unwrap();
            assert_eq!(args_of(cs[i]), expected, "[{}] call {i} args", f.label);
        }
    }
}

#[test]
fn corpus_parses_bytewise_chunking() {
    for f in CORPUS {
        let out = run_bytewise(f.input);
        let cs = calls(&out);
        assert_eq!(
            cs.len(),
            f.expect.len(),
            "[{}] bytewise call count",
            f.label
        );
        for (i, (name, args)) in f.expect.iter().enumerate() {
            assert_eq!(cs[i].name, *name, "[{}] bytewise call {i} name", f.label);
            let expected: serde_json::Value = serde_json::from_str(args).unwrap();
            assert_eq!(
                args_of(cs[i]),
                expected,
                "[{}] bytewise call {i} args",
                f.label
            );
        }
    }
}

#[test]
fn corpus_never_loses_bytes_when_unparsed() {
    // For fixtures expecting zero calls, every input byte must come back
    // out as text — malformed attempts are shown, not swallowed.
    for f in CORPUS.iter().filter(|f| f.expect.is_empty()) {
        let out = run(f.input);
        assert_eq!(texts(&out), f.input, "[{}] text fidelity", f.label);
    }
}

// ---------------------------------------------------------------
// Rehearsal gating (needs enabled_tools)
// ---------------------------------------------------------------

fn enabled(names: &[&str]) -> HashSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn rehearsal_parses_for_enabled_tool() {
    let mut p = ToolCallParser::new(Protocol::Auto).with_enabled_tools(enabled(&["web_search"]));
    let mut out = p.push(r#"web_search[ARGS]{"query":"pio"}"#);
    out.extend(p.flush());
    let cs = calls(&out);
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].name, "web_search");
}

#[test]
fn rehearsal_prose_for_disabled_tool_stays_text() {
    let mut p = ToolCallParser::new(Protocol::Auto).with_enabled_tools(enabled(&["web_search"]));
    let input = r#"use foo[ARGS]{"x":1} like this"#;
    let mut out = p.push(input);
    out.extend(p.flush());
    assert!(calls(&out).is_empty());
    assert_eq!(texts(&out), input);
}

#[test]
fn rehearsal_disabled_without_enabled_tools() {
    let input = r#"web_search[ARGS]{"query":"pio"}"#;
    let out = run(input);
    assert!(calls(&out).is_empty());
    assert_eq!(texts(&out), input);
}

#[test]
fn rehearsal_split_name_across_chunks() {
    let mut p = ToolCallParser::new(Protocol::Auto).with_enabled_tools(enabled(&["web_search"]));
    let mut out = Vec::new();
    out.extend(p.push("web_se"));
    out.extend(p.push(r#"arch[ARGS]{"query":"pio"}"#));
    out.extend(p.flush());
    let cs = calls(&out);
    assert_eq!(cs.len(), 1, "name held across chunk boundary");
    assert_eq!(cs[0].name, "web_search");
}

// ---------------------------------------------------------------
// Streaming-specific behaviors
// ---------------------------------------------------------------

#[test]
fn think_text_streams_through_incrementally() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = Vec::new();
    out.extend(p.push("<think>long reasoning "));
    // Reasoning must not be held until the close marker — the UI streams it.
    assert!(
        texts(&out).contains("long reasoning"),
        "reasoning text must stream, not buffer"
    );
    out.extend(p.push("more</think>after"));
    out.extend(p.flush());
    assert_eq!(texts(&out), "<think>long reasoning more</think>after");
}

#[test]
fn forced_tag_protocol_ignores_other_formats() {
    let mut p = ToolCallParser::new(Protocol::TagJson);
    let input = r#"[TOOL_CALLS][{"name":"a","arguments":{}}]"#;
    let mut out = p.push(input);
    out.extend(p.flush());
    assert!(calls(&out).is_empty());
    assert_eq!(texts(&out), input);
}

#[test]
fn gemma_close_waits_before_deciding() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = Vec::new();
    out.extend(p.push("<|tool_call>call:ping{host:a}"));
    out.extend(p.push("<tool_"));
    out.extend(p.push("call|>tail"));
    out.extend(p.flush());
    let cs = calls(&out);
    assert_eq!(cs.len(), 1);
    assert_eq!(texts(&out), "tail");
}

#[test]
fn multibyte_utf8_across_chunks_survives() {
    let mut p = ToolCallParser::new(Protocol::Auto);
    let s = "🐣 chick <tool_call>{\"name\":\"x\",\"arguments\":{\"emoji\":\"🪺\"}}</tool_call>";
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(3) {
        // Simulate byte-level chunking by accumulating only valid UTF-8.
        if let Ok(valid) = std::str::from_utf8(chunk) {
            out.extend(p.push(valid));
        } else {
            // Feed char-by-char fallback for split codepoints.
            continue;
        }
    }
    // Simpler deterministic variant: char-wise.
    let mut p2 = ToolCallParser::new(Protocol::Auto);
    let mut out2 = Vec::new();
    for ch in s.chars() {
        out2.extend(p2.push(&ch.to_string()));
    }
    out2.extend(p2.flush());
    let cs = calls(&out2);
    assert_eq!(cs.len(), 1);
    assert!(cs[0].arguments.contains("🪺"));
}

// ── Healing telemetry tally (unsloth-adoption 12) ─────────────────────

#[test]
fn tally_labels_clean_gemma_and_commaless() {
    // Clean JSON protocol call.
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = p.push("<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1}}</tool_call>");
    out.extend(p.flush());
    assert_eq!(calls(&out).len(), 1);
    let t = p.tally();
    assert_eq!((t.clean, t.gemma_dialect, t.fell_through), (1, 0, 0));

    // Gemma dialect decode counts separately.
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = p.push("<|tool_call>call:get_weather{city:Lisbon}<tool_call|>");
    out.extend(p.flush());
    assert_eq!(calls(&out).len(), 1);
    let t = p.tally();
    assert_eq!((t.clean, t.gemma_dialect), (0, 1));

    // Comma-less multi-call array = repaired; strict array = clean.
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out =
        p.push("[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}}{\"name\":\"b\",\"arguments\":{}}]");
    out.extend(p.flush());
    assert_eq!(calls(&out).len(), 2);
    assert_eq!(p.tally().commaless_array, 2);
    assert_eq!(p.tally().clean, 0);

    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out =
        p.push("[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"arguments\":{}}]");
    out.extend(p.flush());
    assert_eq!(calls(&out).len(), 2);
    assert_eq!(p.tally().clean, 2);
    assert_eq!(p.tally().commaless_array, 0);
}

#[test]
fn tally_counts_malformed_fallthrough_not_prose() {
    // Malformed body after a real call trigger -> fell_through.
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = p.push("<tool_call>{utterly broken###}</tool_call>");
    out.extend(p.flush());
    assert!(calls(&out).is_empty());
    assert_eq!(p.tally().fell_through, 1);

    // Prose that merely resembles a trigger is NOT a fall-through.
    let mut p = ToolCallParser::new(Protocol::Auto);
    let mut out = p.push("I could call:maybe later, just prose.");
    out.extend(p.flush());
    assert!(calls(&out).is_empty());
    assert_eq!(p.tally().fell_through, 0, "prose must not count as failure");
}
