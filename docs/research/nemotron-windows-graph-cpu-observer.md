# Graph-only CPU observer prototype

The [completed Windows thread-counter experiment](nemotron-windows-thread-cpu-results.md)
preserves numerical parity but fails both original timing predicates. Its
22.2853% control drift does not establish a cause, and its per-region counters
cannot select an optimization. This prototype implements a distinct observer
design before another model experiment is considered. No existing failed trial
is repeated or relabeled.

The previous observer already reads kernel/user CPU durations at graph
boundaries. It additionally queries cycle counters around operations, barriers,
SGEMM attempts, conversion, inner waits and fallback chunks, retaining and
emitting their separate ledgers. The new design keeps the exact batched wall
observer, adds only two graph-boundary `GetThreadTimes` reads per enabled worker
graph, and contains no `QueryThreadCycleTime` calls or per-region CPU ledger.
This removes detailed cycle instrumentation and its storage/emission together;
any later timing contrast would not isolate API-call cost alone.

[Microsoft documents](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadtimes)
kernel/user CPU durations as 64-bit values in 100 ns units. The adapter uses the
executing worker's
[current-thread pseudo-handle](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentthread).
It preserves both halves of each value, genuine zero deltas, API errors and
counter regression. Disabled mode calls no CPU-clock or thread-ID API.
CPU and wall capture boundaries and granularity differ. Their subtraction is
not exact descheduled time; these counters supply no per-operation CPU share,
pure arithmetic cost or production speed result.

The [source transform](../../tools/nemotron-diarization/ggml_graph_cpu_profile_source.py)
requires the existing canonical nineteen-patch CPU file, then applies the
unchanged batched observer and four explicit additions plus a distinct mode key.
The inverse-source test removes only those declared additions and recovers every
baseline byte. Planning, fusion, evaluation, dispatch, wall timers and all
barriers remain unchanged. The new key is `GEN2_DIAR_GRAPH_CPU_PROFILE`.
All historical observer keys are removed from prepared child environments.

The [parser](../../tools/nemotron-diarization/ggml_graph_cpu_profile.py) first
requires the complete original wall/graph/fusion/matrix trace, then joins exactly
one CPU row to each expected pool/generation/worker. It rejects missing,
duplicate, malformed, orphan, mixed historical, error and out-of-bound rows, and
requires distinct executing thread IDs within each graph. Kernel/user durations
remain separate from wall microseconds; overlapping workers and nested regions
are not added.

Seven focused model-free tests compile the actual new C insertion blocks against
fake Windows APIs. They exercise 64-bit values, successful zero deltas, both API
failure positions, counter regression and disabled call counts. Trace mutation
controls exercise worker identity and strict admission. The first positive
fixture used the wrong pool/generation; strict joining rejected it, and only the
fixture was corrected to match the unchanged wall fixture. These tests establish
adapter/parser/source behavior on macOS, not actual Windows API operation,
native graph parity, timing overhead or model performance.

A separate [Windows-only control](../../tools/nemotron-diarization/test_ggml_graph_cpu_windows.py)
compiles the exact new insertion blocks against the installed Windows headers
and Kernel32 using the installed clang-cl and a child-only MSVC environment.
It runs a short activity loop with a 40 ms deadline, verifies the actual current-thread ID and
successful monotonic kernel/user counters, and preserves possible zero deltas.
It supplies no fake API definition. The featureless Windows CI job runs this
control once. The actual invocation now passes at source
`8db9c0e5100de91a8e18b431e72359d586ed16bd` in
[run 39091355966](https://github.com/saberra-ai/gen2/actions/runs/39091355966).
The [retained raw job log](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-api-ci/windows-api-job.log)
records one executed test, thread ID8648, kernel312500/user0 in 100 ns units,
and zero errors/error bits/first error. The genuine zero user delta is retained;
it supplies no resource-share or timer-resolution interpretation. Actual GGML
graph parity and observer-overhead controls remain separate and unexecuted.

The complete model-free suite passes locally: 648 tests in 47.815 seconds,
with eight Windows-only skips. This includes seven passing focused prototype
tests; the additional real-Windows API control is one of the skips. No native
GGML or model execution is represented by this result.

All thirteen feature CI jobs pass in that same run. The Linux evidence-tools
job records648 tests in76.710 seconds with18 platform-dependent skips; the
separate Windows job above executes the API control rather than skipping it.
The [six-payload receipt and manifest](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-api-ci/collection.json)
retain terminal run metadata, both complete job logs, the local test log and
the stdlib-only collector. Collection verifies the exact thirteen job names,
statuses, source commit, selected job IDs and single successful API step. It
checks five source files against the CI commit before and after reading logs,
and parses the raw row without executing project imports. Review caught and
fixed an initially unbound evidence-tools job ID and transitive parser imports
before collection. An early log request was refused while CI was active;
the subsequent terminal log requests were sequential, with no CI rerun.

Before any model execution, the next stage must freeze a separate diagnostic
runtime identity, complete source/compiler/import bindings and actual Windows
graph controls. The original one/two/four-worker off/on/on/off graph vector
checks, complete finite nonzero parity, fusion/dispatch coverage, owned-process
cleanup and private runtime checks remain required. The subsequent four fresh
60-second model processes must retain the original off/on/on/off order and
complete 6000×8 histories, 58 pushes and 27 turns. Both original integer
predicates remain required before interpretation:

- `10 * abs(final_control - first_control) <= first_control`.
- `10 * max(enabled1, enabled2) <= 11 * min(control0, control3)`.

No local native SDK/model build, acquisition or inference is admitted by this
prototype. Current local space is also below the existing model-run admission
floors; no floor is lowered or retained data deleted. A future Windows run needs
its own frozen protocol and artifact-return/storage admission, including the
newly qualified long-path checkout configuration. Only the existing model-free
feature CI is used for the adapter control; no SDK/model diagnostic workflow is
dispatched at this prototype checkpoint. The existing production runtime, failed Windows
RTFs, AMI accuracy, deferred human review and every app/deployment/release gate
remain unchanged. Main remains unpromoted.
