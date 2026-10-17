# Windows native session-time decomposition

Status: preregistered diagnostic, not yet executed. The previous
[worker-budget experiment](nemotron-windows-worker-results.md) rejects fewer
workers as a remedy on its host. Four-worker native RTF1.04734/1.05379 remains
above1. This next probe locates broad costs before selecting an optimization.

Ranked falsifiable predictions:

1. Synchronized graph computation dominates: compute plus output/synchronization
   time is at least85% of native elapsed time.
2. Work outside the timed graph/frontend regions is material: their residual is
   at least15%, directing the next investigation toward graph preparation,
   cache construction, state updates or queueing rather than arithmetic alone.
3. Transfers/allocation dominate: summed input/allocation phases are large
   relative to graph compute. CPU frontend is measured separately.
4. Instrumentation distorts the workload: both profiled observations differ
   substantially (a10% effect is the investigation threshold) from surrounding
   controls, limiting interpretation of the phase proportions.

These thresholds select investigation directions only. They never redefine the
original strictly-below-one performance gate or numerical/quality requirements.

Use the exact four-worker SDK/probe from successful producer run39025306014,
artifact12098214420, source `1745243b221368855381ea36879ccdc1bb731528`.
The45,429,953-byte ZIP SHA256 is
`81433805f066deb99d449ceeab93a2ada0d1c50052b3d698eb117cc5f411c786`.
The reviewed368-member collection, all516 ZIP members and66 four-worker SDK
files are bound to the new committed protocol. All three original SDKs/evidence
are downloaded for byte admission; only an exact copy of SDK4 is executed.
No SDK, helper or runtime is rebuilt. The private PE/import structure is replayed.
The fresh Windows receiver records CPU/OS and AVX2/FMA support before model use.
Host CRT and dynamic module closure are not qualified by this diagnostic.

On one receiver, run four fresh processes in fixed **off,on,on,off** order.
Only `NEMO_SPEECH_TIMING` differs between native child mappings: absent for
controls, present with value1 for profiling. GitHub credentials are omitted in
both modes. Reuse the existing timers in `src/runtime/ggml/session.cpp` and
`src/asr/features/fe.cpp` without new C++ instrumentation, graph alterations or
compiler options. Existing
diagnostic configured/attached-pool telemetry must still report four workers.
All cases consume the pinned Q8 model and60-second WAV,960000 actual PCM samples,
one repeat, geometry9/4/0/264/264/222,58 native pushes,6000x8 final probabilities
and complete speaker turns. Every vector/turn is compared at unchanged1e-6
tolerances to the retained numerical reference. That reference came from a
different receiver: its RTF is never a matched timing control for this run.

The fixed geometry requires84 scheduled model sessions and59 CPU frontend calls
(58 pushes plus the final padded frontend). Both profiled cases must retain
every matching timing row; controls must emit none. Unexpected, direct/GPU,
missing, nonfinite or additively inconsistent rows fail the diagnostic.
Upstream times are rounded to0.01ms; per-row arithmetic admits only that
representation error, and summed regions must fit the enclosing elapsed time
within the derived143x0.005ms rounding bound. This is not a timing pass tolerance.

CPU frontend regions and scheduled-session regions are sequential here.
Session totals include input, allocation, graph compute, output and sync; they
start after graph-cache lookup/construction and lock acquisition. Report the
remaining elapsed time as an unattributed residual, including state work,
queueing and logging. Compute plus output/sync is the safe synchronized region;
this does not supply per-operator timing or directly identify a hot kernel.
The four observations and their original load/push/finish metrics remain
individual. No timing averaging or successful-release promotion occurs.

Required preflight gates are committed source/protocol bytes, exact successful
GitHub run/artifact identity/digest, bounded safe Windows archive extraction,
complete member/SDK copies, private PE identities, receiver CPU prerequisites
and pinned assets. Recheck all protected SDK/delivery/assets before every case
and after the sequence. Retain complete arrays, stderr timers, commands and
success/failure evidence. Downloaded ZIP/delivery/SDK/model assets are excluded
from the evidence upload; their descriptors remain in the receipt.

No public Gen2 API, lifecycle, cleanup, package, coexistence, held-out quality,
signing, microphone, GUI or fresh-machine supported-release gate is closed.
Every release/package/public/lifecycle flag remains false even if a control
observation happens to be below1 on this receiver.
