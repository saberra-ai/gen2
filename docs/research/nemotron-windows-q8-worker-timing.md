# Windows Q8 worker timing within one process

Status: implemented; actual Windows execution pending. This follows
the [failed captured-geometry prediction](nemotron-windows-q8-mx4-results.md)
and [saved loop inspection](nemotron-windows-q8-loop-inspection.md). Neither
existing kernel, production SDK, model configuration nor release gate changes.

## Hypotheses declared before execution

1. Fresh-process/worker scheduling variability contributes substantially to the
   earlier first-candidate slowdown. Predict that comparisons within one process
   and persistent pool will be more repeatable, and/or slow dispatches will show
   larger worker start spreads or completion tails. A small dispatch excess does
   not exclude preemption inside a worker's measured interval.
2. Kernel work or memory traversal dominates the difference. Predict that the
   same candidate direction repeats in both candidate positions within processes,
   with the larger interval inside worker kernel calls. Wall intervals alone
   cannot distinguish arithmetic, memory stalls, frequency or preemption.
3. The observer distorts timing. Predict that enabling worker timestamps exceeds
   the separately fixed observer bound. Such a result withholds bounded observer
   interpretation, irrespective of any attractive kernel timing.

These are diagnostic predictions, not automatic causal classifications. Preserve
all measurements and failed predicates; no selection of favorable processes,
shapes, workers or repetitions is permitted.

## Fixed implementation and observations

Use the exact preceding canonical SGEMM and existing 2×4 candidate, compiler,
linker, global/Release options, contraction off, separate aliases, AVX2/FMA/F16C
and candidate platform guard. Reconstruct and admit the same nineteen patches
and 3,425 canonical source descriptors. No arithmetic/helper/dispatch delta is
introduced beyond the existing candidate already tested in the prior experiment.

Retain all 22 ordered shapes, identical padded strides/seeds, immutable inputs,
output guards, complete half-table validation and three support rejections.
Run twelve fresh processes: workers 1/2/4, each with telemetry off/on/on/off.
Every process uses one persistent worker pool. For each shape run the existing
control/2×4/2×4/control kernels in that order, each with one warmup and eight
timed calls. Reuse the same input bytes for all four phases. Retain a complete
independent output vector for every phase: 1,056 vectors in total, byte-exact
per shape across all phases/processes/workers. All twelve half tables and all
36 exact support rejection exits must independently pass. All 95 actual owned
command groups must preserve expected exit and cleanup evidence.

Whole-dispatch wall time surrounds the original pool call in both observer
modes. With telemetry enabled, each worker captures a steady-clock timestamp
immediately before and immediately after its kernel call, relative to that
dispatch's coordinator epoch. Store every worker ID/start/end for all eight
timed calls. Disabled mode performs no worker clock reads and emits an empty
interval list. Require integer `0 <= start <= end <= whole_dispatch`, complete
ordered worker coverage and the exact enabled/disabled schema. Zero reported
worker duration is allowed for calls below clock resolution, including workers
with no assigned tile; no synthetic work or duration is inserted. Warmup intervals
are not used in timing predicates.

Derive maximum worker duration, start spread, completion tail and whole-dispatch
minus maximum worker duration from retained intervals. These are wall measures;
they are not thread CPU time, instruction counts or proof of a scheduling cause.
The observer also includes its timestamp and result-storage overhead.

## Fixed decisions

For every process and shape retain the original same-process integer predicates
on eight-call totals: `10*abs(last-first) <= first` and
`20*max(candidates) <= 19*min(controls)`. Primary shapes remain indices 4–15.
All twelve primaries must pass in all four four-worker processes for the kernel
speedup prediction. Preserve secondary and one/two-worker results as well.
Separately record the two candidate totals' 10% repeat-drift predicate.

For every primary shape and each of its four kernel positions, compare whole
dispatch totals across telemetry off/on/on/off processes on four workers.
Require `10*abs(off_last-off_first) <= off_first` and
`20*max(on) <= 21*min(off)`: both on observations at most 5% above both off
controls. All 48 position/shape pairs must pass for bounded observer
interpretation. Report both predicates and all totals even on failure. Overall
interpretable kernel-speedup support additionally requires this observer bound.
Do not replace a failed former experiment with this new diagnostic.

A useful result still requires a separately versioned full SDK/model comparison,
original native/public numerical and below-one RTF gates, lifecycle, package,
offline/coexistence and desktop qualification. Quality, deferred human word-time
review, fresh-machine deployment and signing remain open. Main is unchanged.
