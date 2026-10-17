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
