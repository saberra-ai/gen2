# Windows full-model matrix path experiment

Status: implemented, actual Windows SDK/model execution pending. The
[explicit-LLAMAFILE Mac control](nemotron-matrix-path-sgemm-results.md) passes
all twelve processes and 144 per-shape byte-identical outputs. The earlier
[Windows operation profile](nemotron-windows-cpu-ops-results.md) has complete
84-graph/four-worker coverage and puts MUL_MAT first in each worker's kernel
wall observations. The [Q8 tile comparison](nemotron-windows-q8-tile-v2-results.md)
fails its speedup prediction. Measure actual model matrix roots before choosing
another optimization. This diagnostic supplies no production/release pass.

Ranked hypotheses, declared before execution: converted Q8 SGEMM dominates matrix
wall; direct F32 SGEMM dominates; fallback or extra-buffer kernels dominate; or
input conversion/internal synchronization accounts for much of matrix wall.
Record evidence for every path, including bypasses. No inference about model
paths follows from the standalone shapes, and a worker's kernel wall is not pure
arithmetic time.

## Frozen construction

The separate producer uses runtime identity
`nemo-speech-asr 0.2.0+gen2-diagnostic-matrix-paths-v1`. Mechanically derive its
worker, verifier and compiler adapter from the unchanged CPU-operation producer
by exact import substitutions only. Preserve the pinned native/llama revisions,
bounded inference patches, private DLL naming/PE/ABI/source checks, original
compiler/MSVC probe/linker identities and portable AVX2/FMA policy. Keep explicit
Clang contraction off and require all 97 compiler commands to otherwise match
the original baseline, including Release flags and exact version-token deltas.

Append exactly one new GGML patch after the canonical nineteen patches. It
replaces the earlier timing patch rather than stacking it. The 15,356-byte
`matrix-path-profile.patch` has SHA-256
`8e541d97b33801970ab17be9933373045caea71bf5ecfd752e5fee9cc88666cc`.
The exact resulting series SHA-256 is
`34f75ee463bc0a86b27a7b50c3c98f8ddfcf4265faa7bc0f0177b83afca19186`.
Only the CPU C file changes in the 3,425-file canonical GGML map, to 142,477 bytes,
SHA-256 `93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2`,
identical to the locally compiled/verified observer. No arithmetic, SGEMM source,
chunk assignment, fusion decision or barrier changes are permitted.

The freezer checks parent protocol/receipt identities, exact derived adapters,
all local import closure and referenced payloads. Freeze its workflow, this
specification, patch, fixture, control harness, tests and complete preceding
receipts before producer commit and execution. Preserve successful or failed
Windows evidence; never change original coverage or decisions after observing it.

## Execution and numerical checks

Use the existing scoped Windows Job supervision, including normal-background
cleanup, timeout/failure evidence and actual Windows negative controls. Admit
source/compiler/SDK state before compute. Run twelve fresh actual Windows graph
controls before model acquisition: workers1/2/4, each off/on/on/off, four original
shapes evaluated three times per process. Require direct F32 SGEMM, converted Q8
SGEMM and both one-column fallback paths, exact attempts/conversion/fusion,
complete graph/worker coverage and all 144 finite, nonzero outputs byte-exact per
shape. Clear inherited variants of both observer environment flags. Recheck the
actual source/compiler/SDK, harness/import libraries and earlier outputs/logs.

Only then acquire the original pinned model/audio. Retain the original 60-second
mono input/PCM identity, CPU/-1 device, four-worker pool, 58 pushes, 6,000×8 full
probability history and all turns. Run four fresh processes off/on/on/off.
Every history must meet the original numerical limits and additionally match
the original full raw reference byte-for-byte. Every enabled trace must contain
84 graphs ×4 workers ×129 ordinary MUL_MAT roots, with complete enclosing
operation coverage. Require one persistent pool and matching graph/fusion and
captured matrix-metadata signatures across both enabled runs; retain worker-local
routes, scheduling counters and durations separately.

The observer captures types/names, shapes/byte strides, dot type, actual dispatch,
both SGEMM stages, conversion, existing inner-barrier and chunk walls plus each
root's enclosing wall. Extra/fused private internals are classified bypasses.
No late graph dereference, new barrier or scheduling change is introduced.
Reject missing, duplicate, overflowing, unclassified or inconsistent evidence.
The original operation decoder keeps its 16 MiB limit; complete raw matrix traces
have a separate 64 MiB bound.

Retain each full decoded model profile once in a compact, exclusively created,
64 MiB-bounded JSON file. The qualification holds its filename/hash/size/counts;
all graph/operation/matrix records remain in that file and the original raw log.
Saved admission must reproduce exact compact bytes from a complete raw-trace
replay. Never duplicate full matrix records through both CPU and matrix views
in the qualification. Bind model outputs, logs and saved profiles between cases
and afterward.

## Interpretation and remaining gates

Keep the original integer timing decisions: control drift at most10%; both
instrumented observations at most10% above both surrounding controls. Preserve
all observations even if either bound fails. Diagnostic completion does not
imply these bounds pass or that an optimization qualifies. Do not average away
controls, drop slow cases, sum overlapping workers or infer a spill/arithmetic
cause from wall intervals alone.

Windows native/public below-one RTF, actual bounded packaging/receiver cleanup,
original DER/attribution, desktop/fresh-machine and public signing remain open.
Human annotation and independent review remain documented and deferred at the
user's request. N4/N5 and main promotion remain open.
