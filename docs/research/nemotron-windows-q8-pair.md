# Separate Q8 two-row consumption experiment

Status: frozen experiment implementation; actual Windows construction and
execution pending. Production, model, public/lifecycle, package and release
qualification remain unchanged.

Local preparation passes all545 tool tests with seven Windows-only skips,
including six new source/control/vector/timing checks. The protocol binds131
tools,3,745 repository references and39 local import modules. These model-free
checks do not compile or execute the Windows candidate.

The [production Mx4 candidate](nemotron-windows-q8-candidate-results.md) passes
full native/public parity and lifecycle checks but all four RTFs exceed one on
the Xeon8573C. The earlier [bounded matrix observer](nemotron-windows-matrix-batch-results.md)
places98.6% of each worker's matrix kernel wall in converted Q8 SGEMM. Those
enclosing wall times include synchronization/scheduling; they do not prove an
arithmetic or register-spill bottleneck. Prior failed speed predictions and the
[worker observer result](nemotron-windows-q8-worker-timing-results.md) remain failed.

## Hypothesis and isolated change

The current 2x4 helper loads four raw B vectors before visiting its two A rows.
The candidate loads two raw A vectors and consumes one B vector across both A
rows before visiting the next B column. The hypothesis is that shorter raw-B
lifetimes improve generated register allocation and kernel wall time. This is
a testable proposal, not a measured cause or an assumed speedup.

Control source is the exact `q8_mx4_source.transform` output, equivalent to the
current production SGEMM with its embedded compilation define supplied by CMake.
The candidate preserves that 2x4 dispatch, tile ownership, worker count and tail
recursion. Only its full-tile call selects a new private helper. Existing helpers
are unchanged. The candidate keeps A's original `unhalf` lookup and B's original
four-half F16C conversion. Packing the two A scales and shuffling each B scale
computes each original `A_scale * B_scale` with the same operand order. Every
output retains increasing block order, original signed `updot`, `madd` and `hsum`.
No quantized layout, reduction grouping, ISA, FMA or contraction policy changes.

The source transformer rejects any changed canonical162,829-byte fixture.
Tests reconstruct the exact production control by removing the new helper and
restoring only the selected call and compilation-guard name. Non-Q8/non-Windows
dispatch and all tails remain the original implementation. Actual compiled
numerical equivalence must still be established by the Windows experiment.

## Predeclared execution and decisions

The producer, benchmark, CMake project and metrics are exact checked derivations
of the original worker experiment. Changes select the new control/candidate
files, role label `pair`, private compilation guard and separate protocol.
The original persistent pool, guards, input generation, half table, support
rejections, clock placement and integer timing decisions remain unchanged.
The original exact Clang/linker binaries, AVX2/FMA/F16C and explicit
`/clang:-ffp-contract=off` flags remain mandatory.

Run twelve fresh processes: workers1/2/4, each off/on/on/off. Every process keeps
all22 captured/boundary shapes; each shape runs control/pair/pair/control with
one warmup and eight timed calls per phase. Retain all1,056 complete output
vectors,12 complete half tables,36 exact support rejections, every observed
timing and all95 owned/reaped command records. Compare all vectors byte for
byte within and across processes; do not filter unstable observations.

For all12 primary shapes in each of the four four-worker processes, both
candidates must be at least5% faster than both surrounding controls and control
drift must be at most10%. Independently, all48 primary/position observer pairs
must meet the original at-most5% observer overhead and10% control-drift bounds.
These inclusive integer predicates are inherited unchanged. A failed numerical
or observer gate withholds the corresponding interpretation. One/two-worker
and non-primary observations remain reported but cannot replace failed primaries.

Retain both actual kernel objects/disassemblies and all declared source,
compiler, executable and dependency evidence. Generated-loop stack operand
counts may test whether generated code changed; they are static instruction
widths and cannot establish traffic, CPU time, a spill cause or production speed.

A model-free pass supports the next separately frozen full-model comparison.
It does not establish live RTF or qualify an SDK. Full native/public/lifecycle,
package/offline/coexistence/desktop and original quality, deferred human word-time,
fresh-machine app and signing/release gates remain mandatory. No main promotion
is authorized by this experiment's result alone.
