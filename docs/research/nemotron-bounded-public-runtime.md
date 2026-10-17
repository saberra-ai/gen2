# Bounded CPU public Rust qualification

The [bounded SDK C ABI qualification](nemotron-bounded-cpu-runtime.md) passes
complete 12/60-second output parity and warm RTF below one on the three-CPU
M1 virtual receiver. Qualify the actual public Rust consumer next, using that
same exact construction and its independently verified native reference.

Rust admits only the additional exact version
`nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2`. A synthetic version-only
library regression first fails on that tag and then passes; suffixes and future
tags remain rejected. This enables explicit qualification of an SDK supplied by
the caller. It does not choose a default package or promote desktop support.

The [public protocol](../../tools/nemotron-diarization/macos-bounded-public.protocol.json)
binds 54 tool/data inputs, 312 repository/workspace inputs, construction artifact
`12089814731` (200 members), and native artifact `12090213937` (44 members).
It includes complete external artifact identity fields and requires successful
model-free construction, native parity/realtime gates, exact runtime identity,
and the same original SDK. Bind current local modules before importing them and
verify committed source/protocol before any execution. After committing this
preregistration, run the actual local `--admit-only` check on both full archives
and API records. That mode must return before SDK restoration, asset copying,
helper construction or model execution. Retain its result before the M1 push.

On the receiver, acquire the pinned Q8/card/60-second WAV before offline
compilation. Deny those assets and all model formats while building the generic
CPU Rust stream example, unchanged native lifecycle tests, and public C ABI
version probe. Freeze installed SDK/package/helper payloads. Require compile and
runtime sandbox controls, exact version, missing-library rejection, all seven
compiled lifecycle test names, and exact loaded-image audits before admitting
models. The standalone Rust process cannot read source, SDK headers, compiler
outputs, previous results, or native references. Only the lifecycle tests may
read their immutable assertion copies; preserve explicit read/write controls.

Run one complete 60-second Rust stream with default geometry and compare all
6,000 rows/eight channels and every final turn against the bounded C ABI result.
Retain the unchanged 1e-6 comparison bounds and report byte equality. Require
warm compute/audio below one on exactly Apple M1 (Virtual), three logical and
physical CPUs. Then run all seven existing real-model lifecycle cases once,
including 100 reloads, cancellation, deadlines, admission and offline cache
reuse. Model-loading targets must load exactly the 92-library private closure;
listing/offline-cache targets must load none. Record full-process timing/RSS
without treating shared-VM timing as physical-desktop latency.

Keep the 8 GiB start/1 GiB running free-space floors, 256 MiB physical reserve,
1,200-second child deadlines, process-group cleanup and durable failure logs.
Stop on failure without automatic neural retries. This is separate from the
historical pooled public run, whose realtime failure remains recorded.

At preregistration no public bounded-SDK helper has been built or executed,
and no controller/model execution has occurred. Model-free Rust tests pass
1,041 cases with 13 appropriately ignored cases. New comparator tests retain
full-history/final-output rejection and exact profile/placement admission.
Actual public qualification, long/mixed-model and app/platform checks under
this runtime identity remain pending. The original 20% per-slice DER, human word
gold, GUI/microphone, fresh-machine, public signing and release gates remain
open; main remains unchanged.
