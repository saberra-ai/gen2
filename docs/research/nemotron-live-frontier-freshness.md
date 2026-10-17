# Saved live CLI frontier observations

N5 requires actual capture/UI label delay as well as compute throughput. Existing
complete-app live receipts retain a smaller observation: elapsed time on the
CLI feed clock and the processed-audio frontier at each provisional revision.
This report describes that trace without treating the frontier as a word/speaker
boundary, visible label or finality guarantee. End-to-end label delay remains
unmeasured.

## Fixed descriptive analysis

[`live-frontier-analysis.protocol.json`](../../tools/nemotron-diarization/live-frontier-analysis.protocol.json)
freezes all three completed80.04s app cases: valid-Mel CPU, valid-Mel Metal and
pooled CPU. Every saved result, receiver receipt and construction protocol is
hash-bound. Both construction protocols record the exact same `live.rs` source
as the versioned overlay: after model loading and session startup, it starts its
clock before feeding320samples every
20ms and polls the speaker snapshot after each push, before scheduled sleep.
A block can therefore lead its nominal endpoint by20ms; signed ages are retained.
The first-observation time excludes model cold load and application startup.

For every consecutive retained revision, report:

- Observation age: `elapsed[i] - processed[i]`.
- Observation gap: `elapsed[i] - elapsed[i-1]`.
- Age of the previous observed frontier at the next observation:
  `elapsed[i] - processed[i-1]`.

Retain every observation and summarize minimum, median, nearest-rank sample p95
and maximum. With roughly15observations, these are descriptive samples rather
than population latency estimates. The third statistic describes only this
polling trace; it cannot establish a GUI staleness bound. A missing revision,
nonmonotonic clock/frontier or impossible lead fails admission.

No model, app, helper, native runtime or audio acquisition is executed. The three
profiles were measured at different times/builds on the same M4 Pro under
ordinary background activity, so differences do not establish a pooling effect.
This is a post-hoc analysis of already-seen repeated audio, not a fresh test or a
selected cadence/configuration. All original quality and release gates remain.

## Remaining measurement

Intermediate turn contents and actual IPC/GUI delivery/render timestamps were
not retained. Thus this trace cannot measure acoustic speaker/word onset to
first correct displayed label, exact stop-to-final-result delay, or separate
input/file writing, queues, native lookahead/compute, snapshot policy and poll
cost. Provisional labels can still change. That fuller measurement needs new
instrumentation and actual GUI/capture execution, with timing-reference
uncertainty reported separately.

All242 local tools tests pass, including trace semantics and rejection of missing
or impossible observations. The protocol was committed before execution at
`87c542b9937b71632c1ce5debf21456c03f53aa7`. The
[completed seven-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-live-frontier-analysis/collection-qualification.json)
verifies all11 current/historical input/source pins and replays all45 retained
observations without app, model or native-runtime execution. Every trace has
15consecutive provisional observations; the completed source result has a
separate final revision whose delivery timestamp was not recorded.

| Saved case | First observed revision, s | Median frontier age, s | Maximum frontier age, s | Median observation gap, s | Maximum previous-frontier age at next observation, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid-Mel CPU | 5.306 | 0.966 | 1.385 | 5.200 | 6.486 |
| Valid-Mel Metal | 5.245 | 0.841 | 1.106 | 5.201 | 6.347 |
| Pooled CPU | 5.246 | 0.925 | 1.206 | 5.200 | 6.487 |

These traces show a roughly5.2second observed update cadence alongside much
smaller age at each new observation. Both matter to a consumer, but neither
establishes when an acoustic turn first received the correct visible label.
In particular, the6.487second pooled statistic is the age of a previously
observed frontier at a later CLI poll, not a bound on GUI display or label
accuracy. No update cadence or production configuration was changed.
