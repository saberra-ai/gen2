# Windows CPU graph profile experiment

Status: preregistered, not executed. The [measurement implementation and Mac
model-free controls](nemotron-windows-cpu-ops-profile.md) are complete; this new
experiment qualifies the actual Windows measurement before interpreting a model
profile. The [preceding full compiler result](nemotron-windows-compiler-fp-results.md)
restores numerical parity but fails real-time performance. Original quality,
performance and release requirements remain unchanged.

Predictions, in ranked order: a small number of kernel classes dominate graph
time; outer synchronization dominates instead; additional compiler optimization
is needed; or runner drift explains the margin. Only the first two are directly
measured here. Kernel calls include internal synchronization and OS scheduling,
and observer overhead can affect both regions. No hot operator is presumed.

## Frozen construction and admission

Construct one fresh Windows x64 CPU SDK using the exact previously admitted
Clang compiler/linker binaries and explicit contraction-off flags. Preserve all
platform, Release and portable AVX2/FMA/F16C definitions. Compare all 97 compiler
commands to the original corresponding baseline, normalizing only selected
compiler path aliases, isolated source root, the single contraction flag and the
exact diagnostic version string. The new version is
`nemo-speech-asr 0.2.0+gen2-diagnostic-cpu-operations-v1`; it never enters production
runtime or package admission.

Derive the worker build class, source verifier and compiler adapter from their
frozen predecessors at explicit single-occurrence sites. Retain every original
inference/source/probe/ABI/ISA/PE/state/version and owned-Job check. Append one
hash-guarded patch after the original nineteen GGML patches. Reconstruct all
twenty independently and require the complete 3425-file tree to match the prior
canonical tree except the declared CPU file. Its expected instrumented SHA-256
is `dbd58cbf43f80728ea4bdc75337aee4adb8e9651c0327af7316d70d3e949049a`.
All original inference files and arithmetic algorithms remain unchanged.

Before model acquisition, require actual host AVX2/FMA, then compile the frozen
model-free control with the same MSVC probe compiler against this SDK's admitted
private GGML import libraries. For one/two/four workers, run off/on/on/off in
fresh processes with three graph evaluations each. Retain every output and log;
require complete worker/operation coverage, one observed fused normalization/mul
pair per timed worker, finite nonzero vectors and all 36 outputs byte-identical.
Bind control source, executable and import-library bytes before and after use.
No model assets may be acquired until these controls and SDK admission pass.

## Full model measurement

Use the original Q8 model and 60-second licensed meeting, exact PCM identity,
geometry, four-worker budget and 58 pushes. Run off/on/on/off in four fresh native
processes against the same SDK. Clear inherited profiling flags for controls.
Preserve complete output before comparison; require all 6000x8 probabilities
byte-exact with the frozen reference plus all original turn/geometry/1e-6 checks.
SDK/compiler/source/assets and full input bindings are rechecked between cases
and afterward. Preserve failures without automatic retries.

Expect 84 CPU graph executions in each enabled case, matching the prior 84
single-CPU session observations. This count is a falsifiable prerequisite,
not inferred from how many rows happen to arrive. Require all four workers on
every graph, exactly one persistent pool, canonical operation identities,
complete counts and enclosing intervals. Both enabled cases must have identical
ordered topology and actual root/fusion signatures. A mismatch stops admission
and retains the trace; it cannot be repaired by silently changing the count.

Use integer microseconds for timing. Report every time and RTF. Descriptive
observer interpretation is supported only when control drift is at most 10%
and both instrumented cases are at most 10% above both controls. This adds no
release gate and does not waive the original strict below-one native/public RTF
requirement. Failed drift/overhead decisions retain complete data without a
claim of representative uninstrumented operation costs.

Keep timings separate by worker; their intervals overlap. Kernel calls include
internal waits and OS scheduling. Topology hashing and reporting can induce
additional waiting or delay the next graph, so no sample is pure arithmetic cost.
Do not average away drift, combine different hosts or attribute operator causality
from these wall regions alone. Successful measurement still requires separate
production construction, full lifecycle/performance, packaging/offline/coexistence,
physical desktop, signing, quality and human-reference qualification before main.

The workflow runs the original 15 compiler controls, 15 CPU-profile controls and
10 actual Windows Job controls before construction. All producer inputs freeze
at first invocation. The existing successful/failed producers and receipts stay
unchanged.
