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

The v2 receiver has not run yet. It preserves the 5 GiB neural
admission floor, 1 GiB running floor and 256 MiB reserve. It changes no reference
policy, quality criterion or held-out claim. GUI/microphone, app distribution,
mixed-model coexistence, uncontended performance, signing/notarization and
the original accuracy failures remain separate.
