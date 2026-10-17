# Windows compiler diagnostic: owned-descendant failure

The first [compiler experiment](nemotron-windows-compiler-diagnostic.md) stops
before SDK installation/admission or speech inference. Actual
[run39029556195](https://github.com/saberra-ai/gen2/actions/runs/39029556195),
source `8f9ad71cfe9045e17135c4be1bd3633fc1a1a37c`, fails in the first MSVC
`sdk-build` supervision step. It establishes no compiler speed comparison.

The MSVC build script prints its final `==> done. Binaries in ...` marker. After
the top-level gated worker exits, its Windows Job still has active descendants
after the fixed five-second grace period. The supervisor reports
`RuntimeError('owned command left live descendant processes')`, terminates the
owned Job, reaps the worker and verifies zero active Job processes before
propagating that failure. All13 supervised commands record verified reaping.
The first12 record returncode0. The failed SDK command's wrapped exit code was
not retained; a script completion marker does not establish that missing code.

The receipt contains no remaining-process image paths or IDs. Do not name a
compiler service, telemetry process or other tool as the cause from this data.
The evidenced failure is the supervisor's live-descendant condition after the
worker exited. No child is permitted to keep modifying the evidence after
verified cleanup, and no successful SDK qualification is inferred.

Actual Windows preflight first passes all eight compiler/provenance controls and
all five Job controls. The latter execute real normal/nonzero child exits,
assignment-failure gating, and a timeout after a real grandchild has spawned;
its delayed write never occurs. These observations verify those Windows API
paths. They do not cover successful cleanup of background descendants after a
normal top-level build exit, the condition this actual SDK build exposes.

The first SDK's source-patch operations complete and its build output is
retained, but SDK installation, independent compiler-input replay, state test,
native probe, actual version admission, Clang construction, model acquisition
and all four numerical/timing cases are never reached. There is no installed
SDK payload in the artifact, and no raw probability or inference result file.
Every production/public/lifecycle/package qualification flag remains false.

The [114-member failure collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-job-failure/collection-qualification.json)
binds all40 members of artifact12100021994,63248bytes, SHA256
`caf4c9142794d28540ee600bf922ca714274691b84bf9ba80b2a34053f0945c2`.
Its saved-data collector admits all66 tool/675 repository inputs against the
exact producer Git commit and every ZIP payload before importing local evaluators.
It replays the13-command failure schema, patch/diff copies, original protocol,
actual API-step conclusions and log evidence. Every copied artifact, source,
metadata and collector file is rebound before success publication. Original
producer inputs, protocol, workflow and prerequisites remain unchanged. The
collection runs no SDK executable, compiler or speech model.

The next correction must be separately versioned. Retain the observed wrapped
exit code and a membership-scoped snapshot of remaining owned processes. Clean
up owned background descendants after normal command exit as well as after
timeout/error, then require the complete Job to be empty before returning the
actual command status. Add an actual Windows regression with a successful
top-level process and live background child, alongside the existing timeout and
assignment controls. A cleanup failure must remain unverified and unsuccessful.
This is a process-ownership correction, not a change to model/source/geometry,
portable ISA, worker count, full-output tolerances or performance acceptance.

Only after the versioned correction and actual admission can the original fixed
MSVC/Clang/Clang/MSVC experiment run. Preserve its integer decision rule, both
controls and all original production requirements. Windows real-time performance,
bounded package/receiver cleanup, quality, human annotation and desktop release
gates remain open. No main promotion is justified by this failed diagnostic.
