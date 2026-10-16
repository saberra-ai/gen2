# Fresh Mac receiver: architecture prerequisites

2026-10-16. The initial inspection below performed no SDK construction or native
execution. The later construction attempt is recorded separately below. Current
same-host CPU/Metal app receipts do not qualify a second Mac.

## Current artifact and available runner are different CPU targets

The actual tagged CPU compiler database includes
`-mcpu=native+dotprod+i8mm+nosve+sme` for `ggml-cpu.cpp`. It was built on the
Apple M4 Pro. The retained compiler database is157,109bytes, SHA256
`879f6f44e9dc6cddc864c4ec64b783ed25ed44067db3a79aee980a29d03e9e2e`, at
`/Users/victor/.cache/gen2-nemotron3-n0/valid-mel-tagged-metadata-v1/compile_commands.json`.
The actual package declares arm64/macOS26.0 and the valid-Mel runtime profile;
that declaration alone does not establish instruction-set compatibility with
all Apple Silicon CPUs.

The already completed feature CI reports macOS26.6.2/build25G83 and image
`macos-26-arm64`, version20260907.0351.1. Its retained log is10,090,063bytes,
SHA256 `b92e4d0b41ce67d0098646d2a05cc379471038c45dfd3fe71737896aaad42011`.
Those feature tests do not execute the real model/SDK. [CI receipt](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-182/qualification.json).

GitHub currently documents the ordinary arm64 Mac runner as a3-core M1 with7GB
RAM/14GB storage, and each normal hosted job as a fresh VM. Treat that as the
published runner specification; record actual `uname`, `sysctl`, OS and compiler
facts in a future qualification. [Runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[pinned image inventory](https://github.com/actions/runner-images/blob/macos-26-arm64/20260907.0351/images/macos/macos-26-arm64-Readme.md).

The current M4-specific artifact must not be relabeled M1-qualified from those
facts. Native CPU feature detection in the build configuration establishes what
was compiled on the build host, not successful execution on the receiver. This
note does not assert that the current binary necessarily crashes on an M1;
actual receiver arithmetic/inference has not been attempted.

## Two reviewable qualification routes

1. An independent M4-class receiver can test the existing exact signed app,
   model/framework inventories, offline automatic resolution and runtime closure.
   Record the actual receiver CPU and avoid expanding support to untested CPUs.
2. A separately named portable CPU candidate can use the pinned source's
   `GGML_NATIVE=OFF` and explicit `GGML_CPU_ARM_ARCH` path. An initial candidate
   is `armv8.2-a+dotprod+fp16`, with arm64/macOS26.0 fixed separately. This is a
   source-supported build option, not a built or qualified package. Verify actual
   compiler commands/macros and emitted payloads, all compatibility patches,
   model-free state/version, private package closure, then real-model parity and
   lifecycle on a fresh M1 receiver before claiming that baseline. Keep the
   original M4 artifacts and receipts unchanged. Qualify performance anew.

The pinned GGML CMake source explicitly branches on `GGML_NATIVE`; when disabled,
a nonempty `GGML_CPU_ARM_ARCH` becomes an explicit `-march=` flag, followed by
compile-time feature checks. It does not prove output accuracy or hardware
compatibility by itself. [Pinned CPU CMake source](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/src/ggml-cpu/CMakeLists.txt).

A fresh VM CPU runtime check would still leave native GUI, physical microphone,
public signing/notarization, uncontended performance and the failed original
quality criteria separate. The Mac is currently locked for native UI automation;
a read-only codesigning-identity query returned0valid identities. No existing
user app/process or Keychain state was changed.

## Frozen construction candidate

The separate [portable construction protocol](../../tools/nemotron-diarization/macos-portable-construction.protocol.json)
now freezes the explicit ARM target for all SDK compiler units, independent
GGML patch/source verification, model-free state/version checks and private
library packaging. The [workflow](../../.github/workflows/diarization-macos-portable.yml)
will retain the exact SDK/runtime archives and original dependency payloads.
Bootstrap dependency versions are measured candidate inputs, not a production
dependency lock. Runtime qualification requires a separate protocol binding the
actual resulting artifacts.

All 191 local model-free tool tests pass. A new regression checks each dangling
destination symlink before canonical path resolution. The construction sandbox
also actually denies reads of six 11-byte dummy weight formats and loopback.
That test exposed an over-escaped wildcard in the earlier original-NeMo import
policy; the [correction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-sandbox-regex-control/qualification.json)
preserves the old policy, demonstrates its wildcard failure and verifies the
corrected pattern. The old policy's explicit known-model-directory and network
denials still pass. Later neural diagnostics used separate explicit path rules;
none of their outputs or qualification statuses changed.

The [first actual construction attempt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-construction-failure/qualification.json)
ran on `Apple M1 (Virtual)`, macOS 26.6.2, Xcode 26.6. All 24 native build commands
returned0, including compilation, installation and `diar_state` 1/1 Passed.
The validator rejected CTest's new `100% tests passed out of 1` summary because
it expected the older zero-failure clause. The attempt remains failed: terminal
architecture/source audits, version query, packaging and artifact snapshots
never ran. Its82-member receipt binds the actual 11,884,596-byte artifact ZIP and
all 78 extracted members. No model ran. The 121 storage checks observed at least
101,361,618,944 free bytes; the 256 MiB reserve was released.

The separate [v2 protocol](../../tools/nemotron-diarization/macos-portable-construction-v2.protocol.json)
keeps the original target, storage and construction criteria and binds that
failure. Its parser requires exactly one passed test and accepts both observed
CTest summary formats. Missing, skipped, failed, duplicate and multiple test
outputs are rejected. All 193 local model-free tests pass. This retry admits no
model execution or quality-gate change.

The [actual v2 construction receipt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-construction-v2/qualification.json)
now passes on an Apple M1 virtual machine. All four outer phases and 26 native
build commands pass. Every one of 98 compiler units has the explicit portable
target; all 3,425 materialized source files and every compiled source hash match
the independently guarded pinned source. State/version checks and 92-library
private packaging pass. Measured dependencies are SentencePiece 0.2.2, Abseil
20260817.0 and Protobuf 36.2; these remain candidate observations.

Byte-only collection verifies the 17,579,487-byte GitHub artifact against its
API digest and all 186 extracted members. Both SDK/runtime archives restore
with exact 84/123-entry inventories, including modes and aliases. Original and
relocated Mach-O section metadata/payloads match for all 92 libraries. The
115-member receipt retains metadata and raw command evidence; opaque archives,
original library bytes and pristine GGML tar remain in cache/CI. No native code
or model ran during collection. The 148 storage checks observed at least
101,429,686,272 free bytes, and the 256 MiB reserve was released.

This qualifies construction and static payload delivery. Real-model execution
on a fresh receiver, lifecycle, application packaging, GUI/microphone,
performance, public signing/notarization and the original quality gates remain
open. The first attempt remains failed, and the M4 packages are unchanged.

## Frozen M1 receiver qualification

The separate [receiver protocol](../../tools/nemotron-diarization/macos-portable-runtime.protocol.json)
now binds artifact 12074849757 from run 38967939615 and all 186 delivered files,
17 tools and 265 repository files. The [receiver workflow](../../.github/workflows/diarization-macos-portable-runtime.yml)
uses a fresh `macos-26` job and requires the observed `Apple M1 (Virtual)` CPU.
The [actual first receiver attempt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-offline-runtime-failure/qualification.json)
failed before model admission. All four helper builds, the compile/runtime
sandbox controls and the version-only command returned zero, but the strict
image auditor rejected `/usr/bin/sandbox-exec` in the trace. The harness had
enabled tracing in the launcher's environment. All 26 delivered files and the
compiled helper archive were verified; no inference or lifecycle case ran.

Preparation verifies the opaque ZIP against the frozen/API digests and every
member before atomic publication, then restores the exact SDK and private
runtime. It acquires only the locked Q8 model, card, previously observed 60-second
WAV and four small source files at the pinned native/LLAMA revisions. Locked
Cargo dependencies are fetched before offline admission. C++ helpers use the
explicit ARM target; Rust uses `target-cpu=generic`. Offline compiler children
cannot read the staged model/audio directory or modify the SDK/package/checkout.
The compiled helper archive and actual build/source metadata are retained.

Runtime children cannot read the checkout, original SDK, producer archive,
Cargo target, helper sources, Cargo/Rustup homes or Homebrew. Actual positive
read controls precede sandbox denial tests; loopback and protected writes must
also fail. A version-only query and a missing-private-library control precede
model admission. Public CPU-package Metal rejection, one native/public 60-second
pair and all seven existing lifecycle tests follow. Every valid probability row
and turn must match; no cached HF/original-NeMo reference or quality labels are
staged. Each lifecycle test must actually execute and pass. Actual dyld traces
must contain exactly the private dependency closure, or no native runtime for
the offline cache-only case.

The separate [v2 receiver protocol](../../tools/nemotron-diarization/macos-portable-runtime-v2.protocol.json)
binds the unchanged construction artifact, 18 tools and 269 repository files,
including the failed attempt and the [actual model-free loader regression](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-child-loader-environment/qualification.json).
It places `DYLD_PRINT_LIBRARIES=1` inside the sandbox using `/usr/bin/env`,
removes it from the launcher's environment, and records both the exact argv
and parent environment. The strict auditor still permits only the target,
system images and the exact private closure. The standalone C regression
fails with the old placement and passes with the correction; all 198 local
model-free tools tests pass. No SDK or pretrained model runs in that regression.

The [actual v2 receiver](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-offline-runtime-v2/qualification.json)
now passes run 38972811283 at source `7db170d0c8f17b3aea670769577d2d3cd2deaf5b`
on a fresh Apple M1 virtual machine, macOS 26.6.2, Xcode 26.6. All 19 named commands
return their expected statuses, including missing-library and CPU-package Metal
rejection. Native/public outputs match all 6,000 valid probability rows and
27 turns exactly. All seven lifecycle tests execute and pass. The 12 strict image
histories include 10 complete 92-library closures and two runtime-free cases;
no launcher or foreign library is admitted.

Byte-only collection independently verifies artifact 12080325210, its exact
2,128,651 bytes and API SHA256
`76edbc485fd8223b7419188e189eac5d62642455555c72761aebeddf10fd3296`,
all 56 delivered files, 287 current/historical source pins, seven recorded
acquisition descriptors and all four archived helpers. The 60-file receipt includes the raw saved outputs,
traces, command records, protocol and collector. Opaque archives/helper binaries
remain in cache/CI. Collection executes no native code or model. All 3,158 storage
checks pass, observing at least 98,122,481,664 free bytes; the 256 MiB reserve is
released. The 5 GiB neural admission and 1 GiB running floors are unchanged.

Observed native/public RTF is 6.8817/7.1871 for this 60-second smoke recording.
The VM therefore does not meet the below-1.0 live performance gate. This is a
single functional receiver result, with no causal performance diagnosis or
uncontended/live qualification. The first receiver failure remains failed.
No reference policy, quality criterion or held-out claim changes.
GUI/microphone, complete-app delivery/execution, mixed-model coexistence,
uncontended performance, signing/notarization and the original accuracy failures
remain separate.

## Frozen CPU sampling diagnostic

The separate [CPU profile protocol](../../tools/nemotron-diarization/macos-portable-cpu-profile.protocol.json)
registers one 12-second native command using the exact qualified package and
archived native executable. Model, SDK, preset and thread count stay unchanged.
It samples five seconds of stacks and retains `/usr/bin/time -l` process usage,
actual CPU counts and host load. An actual model-free C sampling control must
pass before model acquisition/admission. Offline input/source/write controls,
the original storage floors and the strict private-library auditor still apply.

The ranked hypotheses are fixed-thread contention, matrix-kernel/Accelerate
work, and shared-VM scheduling delay. Their predictions are in the protocol;
none is established by the existing RTF alone. The instrumented shorter crop
cannot establish a speedup against the earlier 60-second run.

Local model-free testing exposed macOS redaction of sample-report paths. The
original report and failed exact-path check remain in cache. The corrected
observer requires the live PID, exact `ps` executable path and matching target
UUID in dyld and the sample's Binary Images section. Redacted report paths stay
redacted; the separate dyld audit still checks exact actual paths and hashes.
Identity queries, sampling and observer waits are bounded. This prepares a
diagnostic, with no completed performance fix or release qualification.

The [actual diagnostic receipt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-profile/collection-qualification.json)
now completes run 38979777083 at `0770aaaa`. Its 24-file receipt binds the
161,241-byte artifact and all 19 delivered files, 21 source pins, the exact
native helper/package bytes and 92 private-library images. The model-free sampling
control and the single native command pass; all 1,200 rows are finite and
bounded. The control executable was not delivered, so its payload hash and
live `ps` identity remain CI observations. Acquired inputs are recorded
descriptors, rather than receiver-rehashed delivered payloads.

The observed receiver has three logical/physical CPUs. The sampled 12-second
call reports RTF 5.83664; `/usr/bin/time -l` reports 70.32 real, 193.45 user and
1.86 system seconds, or 2.777 CPU-seconds per wall second. Initial load averages
are 25.74/40.20/20.54. This is neither an uncontended measurement nor a speedup
comparison with the earlier 60-second run.

Of 430 main-thread stack samples, 83 include `ggml_threadpool_new_impl` and
151 include the mutex-drop kernel call under graph computation. These are
sampled main-thread paths, not whole-process CPU percentages. The
[pinned CPU backend](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/src/ggml-cpu/ggml-cpu.cpp)
initializes its thread-pool pointer to null;
[graph computation](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/src/ggml-cpu/ggml-cpu.c)
then creates and frees a disposable pool when none was attached. The current
[NeMo runtime](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/runtime/ggml/session.cpp)
sets thread counts but does not attach a persistent pool. These source
facts and stacks support testing pool reuse as the next controlled candidate;
they do not establish the cause or a fix. The qualified artifacts stay unchanged.
All 203 local model-free tests pass, including five sampler safety/control
cases. The [source feature matrix](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-203/qualification.json)
passes all13jobs at0770aaaa. Its Linux evidence step reports203tests,
199passed and four platform skips; this is separate from all203passing locally.

The [private pool helper control](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pool-helper-model-free/collection-qualification.json)
links against the unchanged portable SDK/package. Its model-free sandbox run
passes 20 backend lifetimes, including 10 persistent-pool lifetimes at four threads;
no model or frontend executes. The receipt verifies all 84 SDK and 123 runtime
inventory entries against manifests bound to the prior committed receiver
protocol. It retains the initial inherited-warning compile failure, initial
missing-registry-library link failure, successful object reuse/link continuation,
117 compiler inputs and strict 92-image audit.

The helper uses private C++ headers and two unchanged upstream helper-local
backend/logging implementation files because the SDK hides those constructors.
Its successful control establishes lifecycle preparation only. At that checkpoint,
neural private-ABI compatibility and whole-output parity remained unproven. The same-input
four-process disposable/persistent/persistent/disposable comparison was then
prepared and executed as recorded below.
The model-constructor clock excludes backend/pool setup; external process timing
must include it. Neither this control nor the source profile establishes a
performance fix, live qualification or a production runtime change.

The [pool comparison protocol](../../tools/nemotron-diarization/macos-cpu-pools.protocol.json)
now freezes the diagnostic source, 111 native source/patch inputs, 15 transitive
tool inputs, the original construction artifact and the qualified 12-second
probability artifact. The M1 workflow compiles the helper, requires actual
format/network/read-only controls and all 20 model-free lifetimes, then runs four
fresh processes in fixed disposable/persistent/persistent/disposable order.
Each complete 1,200-by-8 probability history must match the qualified C ABI output
within the established 1e-6 wrapper tolerance before timing interpretation.
Native children cannot read source, SDK, reference artifacts or prior results.
The original package is checked unchanged between runs. All 207 local model-free
tests pass; actual controller header replay matches all 22 qualified headers.
The [actual M1 comparison receipt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-cpu-pool-comparison/collection-qualification.json)
now completes run 38982552508 at `643377e`. All 31 commands pass, including 19 header
patch replays, compilation/linking, sandbox controls and 20 backend/10 pool lifetimes
before the four neural processes. Every 38,400-byte probability history is
bit-for-bit identical to the previously qualified C ABI result. All five loaded
image traces contain exactly the same 92 private libraries.

| Process order | Pool | Warm RTF | Whole-process real seconds |
| --- | --- | ---: | ---: |
| 1 | Disposable | 5.94547 | 71.55 |
| 2 | Persistent | 1.99964 | 24.17 |
| 3 | Persistent | 1.78461 | 21.50 |
| 4 | Disposable | 5.96922 | 71.71 |

The ratio of disposable/persistent mean warm RTF is 3.14850; mean whole-process
time improves by 3.13685. This is a measured improvement for this fixed input and
shared three-CPU M1 VM. The load averages change from 7.09/15.02/11.28 to
32.67/23.99/15.88; this is not uncontended hardware qualification. Persistent
RTF 1.78–2.00 still fails the below-1 live gate. No original SDK/package/profile or
production path changed. The next implementation is a separately identified
runtime candidate that owns its CPU pool, followed by production API/lifecycle,
quality, mixed-model and packaging requalification.

The 78 delivered files preserve all results and logs. The helper executable was
not delivered; its payload hash remains a CI observation. Collection rechecks
all delivered bytes, source bindings, exact trace paths and all 92 private-library
payload hashes per trace, rather than claiming to rehash the absent executable.
Acquired model/audio payloads are recorded descriptors rather than delivered
inputs. Storage stays above 100,641,947,648 bytes over 216 checks, and the 256 MiB
reserve is released; the 5 GiB/1 GiB admission/running floors are unchanged.

The [native ownership candidate](../../tools/nemotron-diarization/patches/persistent-cpu-pool.patch)
now makes each backend manager own one CPU pool at its existing thread count.
The manager detaches and pauses the pool before destroying backend handles;
member order then frees and joins workers. Constructor failures preserve the
same handle-before-pool cleanup order. Sessions already serialize computation
through the manager's mutex. The candidate changes only `backend.cpp` and
`runtime.h` and retains the existing valid-Mel patch.

The separate `gen2-valid-mel-pool-v1` [profile lock](../../tools/nemotron-diarization/runtime-valid-mel-pool.lock.json)
keeps the old model, frame contract, preset and source bases. Selection is
explicit, and the original Rust and build defaults remain unchanged. Candidate
validation lives in a separate module so the historical Linux/Windows receipt
consumers and their frozen tool bytes stay intact. An initial full-suite failure
caught that compatibility boundary; after separation, all 211 model-free tests
pass. The previous 207-test feature matrix also passes all 13 jobs at `643377e`.

The [construction protocol](../../tools/nemotron-diarization/macos-pooled-construction.protocol.json)
freezes 38 tool inputs plus its workflow. It requires a complete portable SDK
rebuild, native state CTest, exact version query, relocated private package and
101 public C ABI missing-model construction attempts under weight-file and
network denial. These negative attempts exercise backend cleanup before any
valid weights, frontend or inference. The control permits at most two incidental
system threads above the first warmed failure and retains every count. The
probe first passed against the original qualified runtime, with one thread
after every attempt; that is a probe control, not a pooled-candidate result.
The candidate SDK and cleanup controls have not yet executed at this checkpoint.
Production API, numerical, long-session, mixed-model, quality and release
qualification remain pending.

## Portable CPU complete-app construction

The [local construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-construction/qualification.json)
records a separate APFS clone of the previously qualified complete CPU app.
Only its Nemo resource package and corresponding Frameworks were replaced with
the exact M1-constructed portable package. All 100 clone/static/signing commands
passed, retaining the original hardened ad-hoc entitlements and unchanged model
assets, ORT and application payload. The existing qualified apps are unchanged.

The 211-file receipt includes independent byte comparisons of the main and all
94 Frameworks. Only signature payloads and their declared extent fields are
excluded; every remaining byte must match the original executable or the exact
portable runtime/ORT source. Construction used a 3 GiB admission floor, 1 GiB
running floor and 256 MiB reserve; the separate 5 GiB neural floor is unchanged.
No model, native version function or app was executed. This is local M4
construction only; fresh-machine app execution, GUI/microphone, performance,
accuracy and public signing/notarization remain unqualified.

The [receiver-source feature CI](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-198/qualification.json)
passes all 13 jobs at `7db170d`. Its Linux tools suite reports 198 tests with
195 passing and three platform skips; the local macOS suite passes all 198.
These feature checks execute no diarization model or desktop application.

The [local delivery archive receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-archive/collection-qualification.json)
now binds the complete 573,473,763-byte app snapshot. Independent tar iteration
matches all 246 entries (225 regular files, 20 directories, one alias), including
every payload hash and mode, without extraction or execution. The 13-file
metadata receipt preserves the executed four-tool protocol and an explicit
post-execution binding of all nine Python imports to the earlier constructor
and historical Git sources. It also records the verifier's initial `sdk/` root
correction; the archive was unchanged. No fresh-machine delivery or app runtime
qualification has occurred.
