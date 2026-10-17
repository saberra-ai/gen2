# Explicit-LLAMAFILE matrix observer control passes locally

The separate macOS arm64 control passes all twelve fresh graph processes and
all 144 byte-identical per-shape outputs. Its enabled traces observe the four
predicted paths: direct F32 SGEMM, converted Q8 SGEMM, and one-column vector-dot
fallback with and without input conversion. This qualifies the local observer
control. It supplies no Windows/model dispatch, model RTF or release result.

The [216-file receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-sgemm-control/collection-qualification.json)
retains the frozen [pre-execution specification](nemotron-matrix-path-sgemm-control.md),
complete tool import closure, harness and CMake project, actual configure/build/
dependency/graph command logs, compiler database/cache, source/object/compiler/
runtime identities and all 144 raw output arrays. Native binaries and the full
source tree remain in the local development cache. The specification's pending
status describes its preserved pre-execution state; this document records the
subsequent result. Both [earlier failures](nemotron-matrix-path-control-failure.md)
remain unchanged and are not counted as passing controls.

## Construction and observations

One fresh build explicitly enabled LLAMAFILE. Before the first graph process,
the controller verified the actual cache, ARM feature checks and both actual
CPU/SGEMM compile definitions. Native tuning, Metal, Accelerate, BLAS, OpenMP and
backend loading remained off. All 28 compiler sources/objects and their compiler
binaries were bound. Command-relative `-o` paths agree with CMake's build-relative
object fields. Dylib contents and symlink aliases were checked between graph
processes and afterward. All 24 configure/build/dependency/graph commands exited
zero, and their owned process groups were reaped.

The complete 3,425-file canonical source map changes only the declared CPU
observer file, still 142,477 bytes with SHA-256
`93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2`.
The observer, graph harness and numerical algorithms are unchanged from the
previous attempt. The separate control project explicitly enables the feature
that the earlier project left disabled.

Workers1/2/4 each ran off/on/on/off in fresh processes. Every process evaluated
four graphs three times with one persistent pool. All six enabled processes
have complete twelve-graph coverage, totaling 168 separate worker matrix
records. Every graph has one matrix root and the expected RMS normalization/
multiply fusion. The 32-column F32 graph succeeds at the first SGEMM stage;
the Q8 graph rejects that stage, converts F32 input to Q8, then succeeds at the
second stage. Both one-column graphs reject SGEMM and use vector-dot fallback,
with conversion only for Q8. Attempt counts and conversion flags match the
preregistered predictions for every worker and repetition.

All 144 outputs are finite, nonzero and byte-identical per shape across worker
counts, observer modes and repetitions. The 32-column outputs contain 8,192
bytes each; the one-column outputs contain 256 bytes each. Saved-only collection
independently replays trace admission, construction checks and all comparisons,
then rechecks original inputs and copied payload hashes. No native retry or
earlier failed-receipt alteration was needed.

## Next gate

Build a separately pinned Windows diagnostic SDK and run actual Windows graph
controls before acquiring the model. Preserve the original compiler/linker,
contraction-off numerical policy, portable ISA, four-worker model policy,
97-command audit, exact twenty-patch source composition, public ABI and full
probability/turn comparisons. The known model profile supplies falsifiable
coverage of 84 graphs and 129 ordinary matrix roots per graph per worker.

Retain the original off/on/on/off model sequence and observer drift/overhead
bounds. Keep raw complete traces and worker-local matrix shapes, types, routes,
attempts, conversion and timing data. Avoid duplicating full matrix records in
both CPU and matrix JSON views when constructing that producer's saved report;
all original coverage checks and raw evidence must remain independently
replayable. Worker walls overlap and include scheduling, waits and observer
effects. These local small graphs do not identify model bottlenecks or qualify
an optimization.

Quality, Windows native/public realtime, deployment, physical desktop and signing
gates remain open. Human annotation and independent review remain documented
and deferred at the user's request. N4/N5 and main promotion remain open.
