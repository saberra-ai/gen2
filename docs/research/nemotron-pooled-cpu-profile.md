# Current owned-pool CPU sampling

The previous M1 sampling result predates owned CPU pools. Pool reuse produced a
measured improvement, but the subsequent public receiver still observed RTF
2.10983 on a shared M1 VM. That does not identify the remaining bottleneck.
The pinned backend `Params` default is four CPU threads; the recorded VM exposes
three CPUs. This is a testable oversubscription hypothesis, not a proven cause.

Ranked predictions before another model call:

1. If worker synchronization/oversubscription dominates, the owned-pool sample
   will show substantial graph/barrier waiting rather than repeated pool creation.
2. If arithmetic dominates, matrix/convolution/Accelerate paths will account for
   much of the sampled worker activity.
3. If shared-host scheduling pressure dominates, host load and process CPU/wall
   accounting may show pressure; a sample alone cannot establish scheduler causality.

The [protocol](../../tools/nemotron-diarization/macos-pooled-cpu-profile.protocol.json)
registers exactly one sampled 12-second C ABI process, using the immutable
M1-built owned-pool package and already qualified public C ABI executable.
No private C++ helper or old-layout executable is used. The model, input, native
geometry, postprocessing and four-thread default remain unchanged. Both complete
delivery archives and every member are admitted before restoration. Every local
imported module is admitted before import, and committed source/protocol bytes
are verified before execution.

A newly compiled, model-denied sleeping C control must pass actual PID, live
executable and dyld/sample UUID binding before asset acquisition. Its trace must
contain zero private libraries. The same existing sampler observes the native
process for a requested five seconds; a shorter surviving report cannot imply
five seconds of observed activity. Model admission remains 5 GiB free space,
1 GiB running minimum, a 256 MiB physical reserve, 600-second command deadlines
and process-group termination. One failed command stops the trial, without retry.

The native child cannot read the source, delivery/reference data, previous
result payloads or Homebrew files, write protected inputs, or use the network.
Protected package, helper, source-delivery and acquired-input inventories remain
bound before each command and after execution. Actual normalized PCM, all 1,200
rows/eight channels and final turns must match the saved independent pooled
C ABI reference under the unchanged 1e-6 bounds before timing interpretation.
The exact pooled version is required on both sides before mathematical reuse
of the existing comparator. Byte equality is reported separately.

Reports retain sampled stacks, exact helper and control executable payloads,
time/RSS logs, CPU counts/load, full probabilities and provenance. No neural
execution has occurred at this preregistration. Shared-VM sampled performance
is descriptive, with no speedup, pure scheduling attribution, uncontended live,
new thread policy, quality, GUI/microphone, fresh-machine app or release claim.
The original failed gates remain open.

## Actual owned-pool sample

The [35-member receipt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-17-pooled-cpu-profile/collection-qualification.json)
records successful run [39001644761](https://github.com/saberra-ai/gen2/actions/runs/39001644761)
at `761c9ca1`. The 200,631-byte artifact contains 25 delivered files, including
both actual sampled executable payloads. Four commands pass, with exactly one
neural process. Independent collection admits every delivered byte, source pin,
both prior archives and package/helper payloads before replaying zero/92 private
image audits and all 1,200 rows/eight channels plus five turns. Probabilities and
turns match the saved pooled reference exactly. Acquired model/audio remain
controller-recorded descriptors rather than delivered receiver payloads.

The first collection attempt selected a different old local construction cache;
the exact ZIP guard rejected it before package/helper replay. Its script and log
remain in the receipt. V2 selects the already preregistered producer whose source
and ZIP hash match; neither inference nor protocol/tolerance changed.

The actual M1 virtual receiver exposes three logical/physical CPUs on macOS
26.6.2. Native RTF is 2.37116, still above the below-1 live target. Whole-process
usage is 28.78 real, 69.82 user and 1.31 system seconds, or 2.47151 CPU-seconds
per wall second; peak RSS is 151,617,536 bytes. Load averages change from
5.93/5.95/5.60 to 21.18/9.55/6.91. These are shared-VM observations, without a
matched speedup or uncontended performance claim.

The report records 410 main-thread stack samples. Its largest coalesced
`ggml_graph_compute_thread` node contains 355 samples and lists offsets
`+320,+332,...`; three persistent worker threads are also visible. No
`ggml_threadpool_new_impl` frame appears in this sample. This does not prove that
pool creation never occurs outside the sampling window.

Read-only decoding of the exact admitted M1 library locates the reported +320
and +332 addresses in an atomic-counter load/compare/yield loop. The corresponding
GGML source bytes match the M1 compiler audit and show barrier waiting. The
instruction report binds the actual library and decoder identities. The sample
node coalesces additional undisplayed offsets, so its entire 355 count cannot be
assigned to that loop or interpreted as whole-process CPU utilization.

This strengthens the case for a fixed four-versus-three-thread contrast before
another runtime change. It does not prove that oversubscription causes all
remaining latency: worker imbalance, arithmetic and shared-host scheduling still
interact. A future contrast must retain the same package/source provenance,
model, PCM, geometry and output-parity checks, use fresh processes in a fixed
order, and precede any production thread policy or broader requalification.
No thread setting or supported-release status changes here. Original quality,
human timing, desktop GUI/microphone, fresh-machine app and release gates remain.
