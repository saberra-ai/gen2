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
as the versioned overlay: it starts its clock before feeding320samples every
20ms and polls the speaker snapshot after each push, before scheduled sleep.
A block can therefore lead its nominal endpoint by20ms; signed ages are retained.

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
or impossible observations. At this preregistration, the saved-data analysis
has not run.
