# Windows CPU worker diagnostic

Status: preregistered diagnostic; no actual matrix result yet. It addresses the
bounded Windows native/public performance failure retained at `dfdede8e`:
RTF1.03335105/1.0005604167 on a reported2-core/4-logical hosted CPU. That receipt
does not establish the effective worker count or identify oversubscription.

Ranked falsifiable hypotheses:

1. Worker contention dominates: two workers beat four while full outputs match.
2. Compute parallelism dominates: reducing four to two/one makes it slower.
3. Scheduling drift dominates: repeated four-worker controls vary comparably to
   differences between budgets.
4. Another inference component dominates: all three budgets have similar timings.

The fixed order is **4,2,1,4**, one fresh C ABI probe process per case, on the same
Windows2022 job. Three separate SDKs are built from fresh pinned source clones,
using the same three bounded patches, private DLL names, MSVC compiler selection,
portable AVX2/FMA/F16C baseline, OpenMP/GPU/dynamic backend loading off, and exact
Windows LF materializer overlay. All19 GGML patches are replayed independently
and every materialized/actual compiler source is audited before model admission.
The strict `diar_state` CTest case must execute once per SDK.

The only inference-setting variable is the requested CPU worker budget1/2/4.
All variants add identical diagnostic logging at parameter selection and after
successful persistent-pool attachment. Logs retain reported hardware concurrency,
requested/effective workers and attached pool size. This reports configured pool
size, not an independently measured OS thread census. Instrumentation is tagged
`[DEBUG-worker-v1]`, kept only in this clearly marked diagnostic derivation, and
must not be carried into a production profile.

Each variant has a distinct `gen2-diagnostic-cpu-threads-N` version metadata tag.
The original probe is derived by replacing exactly one quoted historical version
literal, preserving all remaining bytes. Production locks, runtime admission,
historical producers and package gates are unchanged. A diagnostic SDK cannot
be admitted by the bounded SDK/package verifier.

Every case consumes the same pinned Q8 model and full60-second WAV, exactly
960000 float PCM samples, streaming geometry9/4/0/264/264/222, and one repeat.
All6000x8 float32 probabilities and final turns are retained and compared to the
first four-worker result with the unchanged1e-6 probability/turn tolerances.
Nonfinite/out-of-range values, partial histories, changed geometry/PCM/version,
or missing/contradictory worker telemetry fail the diagnostic. Positive results
retain original load/push median/p95/max/finish/elapsed metrics. Timing is not
averaged into a passing live-performance result; both four-worker controls and
all intermediate cases remain individually visible.

The protocol binds all historical bounded-producer tool/repository inputs plus
these new sources, tests, workflow and this plan to committed Git bytes before
execution. Derived native source hashes for every cap are preregistered. Before
and after each inference, the driver checks the three SDK/source/compiler trees
and assets for mutation. Successful or failed evidence and SDKs are uploaded;
model assets and full source/build trees are excluded from that upload.

This experiment supplies native performance evidence only. It cannot close
public API, lifecycle, cleanup, offline packaging, coexistence, held-out quality,
signing or fresh-machine gates. Any useful budget needs an explicitly identified
runtime candidate followed by the full actual qualification pipeline.
