# Windows thread CPU and matrix-cycle diagnostic

The failed A-packing comparison preserves exact outputs but supplies no speedup.
Earlier full-model wall profiles place about70% of worker operation time inside
matrix kernels, including scheduling, stalls and internal waits. Static stack
counts and repeated microkernel changes have not established a cause. This
separate diagnostic adds CPU-resource counters before another optimization is
selected. It does not alter inference arithmetic or qualify a release.

Three hypotheses motivate the measurements: matrix dispatch may dominate
scheduled CPU resources; enclosing wall intervals may contain substantial waits
or preemption; or conversion/setup may dominate the portion outside SGEMM.
Counter shares and stage coverage can distinguish observed resource use, but
cannot establish pure arithmetic cost, cache traffic, exact unscheduled time or
a causal scheduling explanation. A further intervention needs its own evidence
and frozen comparison.

The original pinned native source, nineteen GGML patches, bounded valid-Mel
inference, four-worker policy, Clang/MSVC ABI, contraction-off options, compiler/
linker/probe fingerprints and97 compiler inputs remain fixed. One distinct
diagnostic patch changes only the existing batched observer in `ggml-cpu.c`.
The existing production kernel stays unchanged. Runtime metadata becomes
`gen2-diagnostic-thread-cpu-v1`; it is never an admissible production package.
All historical producers, protocols, failed predictions and quality gates stay
unchanged.

The observer uses only the executing worker's `GetCurrentThread` pseudo-handle.
It reads graph-level kernel/user CPU durations through
[GetThreadTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadtimes)
in their documented100ns units. It records
[QueryThreadCycleTime](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)
counts around graph, operation, outer/final barriers, first/second SGEMM,
conversion, inner wait and fallback chunks. Microsoft explicitly disallows
converting those cycles into elapsed time; no CPU-frequency conversion is used.
API failure, error bits or counter regression rejects interpretation. Genuine
zero-duration/resolution samples remain recorded, rather than synthesized or
discarded. API durations and wall clocks have different capture boundaries and
granularity; their difference is not declared exact wait time.

All existing wall rows remain, with separate CPU graph/operation/matrix rows
joined by pool, graph generation, worker and node/operation identity. Complete
coverage is mandatory:84 model graphs,129 matrix roots per graph and all four
workers. Actual thread IDs must be distinct within a graph. Every nested cycle
region fits its enclosing kernel/graph counter; matrix kernel cycles exactly
cover the corresponding MUL_MAT operation. Counters without dispatched work,
duplicates, orphans, missing records and overflowing values reject the trace.
Worker walls overlap and are never summed into model wall time. Cycles retain
their own unit and are never summed as time across workers.

Collection and emission preserve the original graph/fusion/planning/task and
barrier calls. Records stay worker-local. After the existing final barrier,
matrix wall/cycle rows share one bounded heap buffer and one write per worker.
The declared bound is256 records ×4096bytes, at most1MiB. Allocation, formatting,
truncation and short-write failures produce the existing rejected marker. Clock
errors are explicit graph fields. Disabled mode makes no CPU API calls and
emits no profile rows. Profiling is controlled by a separate environment key;
both historical observer keys are excluded from child mappings.

Before model acquisition, the actual Windows graph controller runs twelve fresh
processes: one/two/four workers, each off/on/on/off. All144 complete finite,
nonzero shape/repeat vectors must agree byte for byte. Original SGEMM attempt,
conversion, fallback, fusion and graph coverage checks remain mandatory, now
including successful CPU-counter coverage. Build-only graph import libraries
are first-bound, copied and retained explicitly, and rechecked throughout the
controls and later model calls.

The full-model sequence remains four fresh processes on the same runner:
off/on/on/off, identical60s PCM and Q8 model. All58 pushes,6000×8 probabilities
and27 turns must match the original pinned reference exactly. Save every raw
log, full array and complete compact profile; repeat topology/fusion/matrix
signatures must agree. The original integer predicates remain:

- Control drift: `10*abs(final_control-first_control) <= first_control`.
- Observer overhead: `10*max(enabled1,enabled2) <= 11*min(control0,control3)`.

Both must pass before interpreting enabled CPU-resource observations. Native
RTFs remain recorded individually; no averaging, rerun-to-pass, speedup or
below-one claim follows from successful profiling. This experiment is not a
new kernel speed comparison and does not erase any prior5% prediction failure.
Every child retains scoped Windows Job ownership, observed/reaped exit status,
background cleanup and closed handles. Fresh source/compiler/state/private PE
checks and frozen tool/import/reference identities surround each stage.

At preregistration, local C adapter/error and parser controls pass; no actual
Windows CPU-clock graph, compiler, model or new artifact has run. Local fake API
controls establish adapter behavior only. Actual Windows execution, complete
delivery admission, saved-data replay and interpretation remain pending. No
new native SDK, public runtime, packaging, accuracy, human-word, physical app,
fresh-machine, platform or signing/release gate closes here. Main stays unchanged.
