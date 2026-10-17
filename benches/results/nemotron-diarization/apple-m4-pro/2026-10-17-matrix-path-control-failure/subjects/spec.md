# Matrix dispatch observer and model-free controls

Status: implemented, native control unexecuted. The preceding Windows
[operation profile](nemotron-windows-cpu-ops-results.md) places MUL_MAT first in
per-worker kernel wall, but the [actual 2×2 kernel comparison](nemotron-windows-q8-tile-v2-results.md)
fails its speedup prediction. Selecting another candidate requires actual model
root shapes/types and dispatch evidence. Existing SDKs, profiles and receipts
remain unchanged; this observer establishes no production optimization.

Ranked hypotheses for the eventual model measurement: converted Q8 SGEMM
dominates; direct F32 SGEMM dominates; fallback/extra kernels dominate; or input
conversion/internal synchronization accounts for much of matrix wall time.
The saved model profile supplies 84 graphs and 129 ordinary MUL_MAT roots per
graph; these will be falsifiable coverage requirements for a future Windows
model protocol, not counts inferred from an incomplete trace.

## Observation contract

The hash-guarded transform requires the original canonical nineteen-patch CPU
file (132,918 bytes, SHA-256
`d4dd2c161a3f5c0ed45216683de99dd9b13b93330d81c992f4c4163fea904a04`).
It derives the existing operation observer, switches to
`GEN2_DIAR_MATRIX_PROFILE=1`, and adds worker-local matrix records. The resulting
file is 142,477 bytes, SHA-256
`93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2`.
The complete 3,425-file source tree must otherwise match the admitted original.
No SGEMM source, arithmetic algorithm, chunk assignment, fusion or barrier changes.

Each graph worker has a bounded 256-record stack array and a thread-local pointer
to its current root record. Capture root index, operand/output types and names,
all four dimensions and byte strides before execution. Wrap the two existing
SGEMM call sites without changing arguments or return decisions, recording each
attempt, success/failure and wall interval. Record original conversion presence,
conversion wall, the existing inner barrier wall and fallback chunk counts/wall.
Classify Hadamard, direct SGEMM, IQ-panel, converted SGEMM, vector-dot fallback,
extra-buffer or fused bypass. Extra/fused internal kernels outside the ordinary
routine are classified as bypasses; their private internals are not measured.
Suspend and restore the thread-local record around the existing extra-buffer
and fusion calls, retaining their original single invocation and return decision.

The pointer is cleared after each kernel and at graph entry. Reporting uses
captured metadata after the existing final barrier and never rereads graph
objects there. Each row is emitted with one locked `fprintf` call. Overflow or
unclassified execution rejects admission. Disabled controls clear inherited
case variants of both old and new observer environment variables.

The decoder requires complete graph/worker/operation coverage plus every matrix
root. Matrix counts and summed enclosing intervals must equal the corresponding
MUL_MAT operation records. Dimensions/broadcasting, counters, dispatch outcomes
and interval bounds are checked; SGEMM attempts cannot exceed the root's batch
count. Common F32/F16/Q8/BF16 ID/name pairs are bound.
Other bounded actual type IDs/names remain recorded without interpretation.
Workers must agree on root metadata. Worker-local paths, counts and durations
remain separate observations, including legitimate differences. The full trace
has an explicit 64 MiB bound; the original operation decoder still receives
only its own rows under its original 16 MiB bound.

## Planned native development control

First validate the observer on macOS arm64 CPU without any NeMo SDK or model.
Build a fresh standalone GGML CPU tree from the admitted canonical map, with
only the declared CPU file changed. Disable native tuning, Metal, Accelerate,
BLAS, OpenMP and backend loading. Freeze complete tool import closure, control,
CMake project, fixture and observer bytes before invocation. Retain actual
compiler/database/object/source and final binary identities, build logs and
commands. Actual source/subject/build inputs must remain unchanged between
processes and afterward; preserve failed attempts without native retries.

Each fresh process uses one persistent pool and four small graphs: F32/F32 with
32 columns, Q8/F32 with 32 columns, F32/F32 with one column, Q8/F32 with one column.
Every graph has a matrix root followed by RMS normalization/multiply fusion;
evaluate each three times. Require respectively direct SGEMM, converted SGEMM,
vector-dot fallback without conversion and vector-dot fallback with conversion.
Require exactly twelve graphs, every worker, one matrix root per graph and the
known fusion. Confirm actual attempts/rejections/conversion flags for each path.

Run workers 1/2/4, each off/on/on/off, in twelve fresh processes. All 144 finite,
nonzero outputs must match per shape across repeats, observer modes and worker
counts. Output files are created exclusively. This establishes observer controls
on this host, without a model/performance/release claim. Seven new decoder/source
controls cover completeness, environment isolation, malformed/missing rows,
metadata/timing/count bounds, converted/fallback paths and worker-local variation.

A separately pinned Windows SDK and model experiment remains necessary. It must
preserve original compiler/source/ABI/numerical gates, actual Windows controls
before model acquisition, byte-exact full histories and observer drift/overhead
bounds. Worker wall regions overlap and include scheduling/internal waits and
observer perturbation; no sum of workers or pure arithmetic claim is valid.
Quality, native/public performance, packaging/offline/coexistence, desktop,
signing and the user-deferred human word-reference review remain open. Main
remains unqualified.
