# CPU operation timing diagnostic

Status: instrumentation and model-free controls implemented. Actual Windows
model profiling, its frozen producer/protocol and interpretation remain pending.
The preceding [contraction result](nemotron-windows-compiler-fp-results.md)
restores numerical parity but leaves every observed native RTF above one.
The next investigation measures where graph execution spends time before
selecting another optimization.

Ranked, falsifiable hypotheses:

1. A small number of operation classes dominate: their timed regions should
   consistently account for most of each worker's graph interval.
2. Outer synchronization dominates: existing outer-barrier intervals should
   account for much of the interval. Internal kernel waits remain a limitation.
3. Compiler optimization is limiting throughput: a later matched compiler-option
   experiment should improve timing while preserving complete numerical output.
4. Runner variation explains the margin: repeated surrounding controls should
   drift materially. The last experiment's 0.58% drift weakens that explanation
   for that host and sequence, without ruling it out on another run.

## Instrumentation contract

[The source transform](../../tools/nemotron-diarization/ggml_cpu_profile_source.py)
requires the exact CPU file from the independently reconstructed nineteen-patch
GGML tree, SHA-256
`d4dd2c161a3f5c0ed45216683de99dd9b13b93330d81c992f4c4163fea904a04`.
It inserts timers around the existing fused-kernel call and outer/final barriers.
It adds no barrier, evaluation callback, scheduler split or changed fusion call.
Each worker accumulates local operation counts and wall durations. Graph topology
and executed-root/fusion fingerprints must agree across workers. Metadata needed
for reporting is captured before the final barrier; reporting does not dereference
graph tensors or the graph plan after that barrier.

`GEN2_DIAR_CPU_PROFILE=1` enables measurement. The environment helper removes
inherited case variants for controls. [The parser](../../tools/nemotron-diarization/ggml_cpu_profile.py)
requires every expected graph, worker and operation row; rejects duplicates,
count/fusion disagreement, malformed or excessive data and intervals exceeding
their enclosing worker time. All 103 operation ID/name pairs are bound to the
canonical GGML table, rather than inferred from internal consistency.

Worker intervals overlap and must not be summed into graph wall time. Kernel
regions include internal synchronization and OS scheduling. Topology hashing and
post-barrier reporting can themselves induce waiting or delay the next graph;
reported costs are observer-perturbed. Disabled/enabled complete-output and
overhead controls remain mandatory for the eventual Windows experiment. These
measurements alone cannot prove arithmetic cost, identify a causal optimization
or qualify uninstrumented production performance.

## Actual model-free control

The [92-file development-test receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-cpu-profile-control/collection.json)
preserves twelve successful processes: one/two/four workers, each using disabled,
enabled, enabled, disabled profiling. Each process evaluates the same graph three
times with a persistent pool. The graph contains matrix multiplication and a
fused RMS-normalization/multiply pair. Every timed worker reports that fusion;
all 36 finite, nonzero 8192-byte output arrays are byte-identical, including across
worker counts. The actual 27 compiler inputs and complete 3425-file canonical
source map are retained, with only the declared CPU file instrumented.

All 407 Python tests pass; seven Windows-only tests are skipped on this Mac.
Seven new tests cover environment isolation, complete overlapping-worker coverage,
missing/duplicate/malformed rows, topology/fusion disagreement, enclosing bounds,
unknown or consistently mislabeled operations, and disabled controls. The native
control and its CMake project, compiler database, logs, source snapshots and saved
runner are in the receipt. The CMake project expects the admitted/instrumented
tree at `source/ggml` and the copied control at `control.cpp`.

This is a working-tree development test on macOS arm64. Compilation inputs and
the corrected execution decoder are recorded separately. An initial compiled
draft was retained unexecuted after review found a late graph-metadata read;
the tested second build uses the pre-barrier snapshot. A later parser correction
bound operation IDs to their canonical names before the first native control.
Publication review then found an inventory-helper import outside the collector's
admitted closure. The complete original collection is preserved locally, with its
collector, manifest, qualification and review record retained in the new receipt.
The replacement collector uses a standard-library inventory and replays saved
bytes; it does not repeat native tests or change their observations.
No SDK, model or Windows graph ran. No production runtime, package, real-time,
quality, human annotation or release gate is closed by these controls.
