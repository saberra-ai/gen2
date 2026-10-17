# Windows math errno result

The [frozen experiment](nemotron-windows-math-errno.md) completed at source
`e76a3199a468f9207c8bc70ab861f2fe918c39c1` in
[run 39039873138](https://github.com/saberra-ai/gen2/actions/runs/39039873138),
job `117416985932`. Removing math errno requirements preserves complete output
bytes for this configuration, but the preregistered speedup prediction fails.
Both candidate RTFs remain above one. This supplies no production optimization
or release qualification.

The [326-file receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno/collection.json)
retains full arrays, commands, source/compiler evidence and frozen tools. Its
[saved-data replay](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno/collection-qualification.json)
executes no compiler, SDK or model.

## Full output and same-host timing

Both SDKs use the exact original Clang compiler/linker and the same original MSVC
probe compiler. Both retain explicit contraction off and the original diagnostic
four-worker identity. The only candidate compiler-policy addition is
`/clang:-fno-math-errno`. SDK labels `control` and `no-errno` are variant names;
every SDK compiler is Clang.

All four fresh processes have four effective/attached workers, the original
960000-sample PCM and geometry, and 58 pushes. Each 6000x8 probability array is
192000 bytes with SHA-256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
All 27 turns match the reference; probability and turn errors are zero. The
original 1e-6 comparisons and the added byte-identity requirement both pass.

| Ordered variant | Native time (seconds) | Native RTF |
| --- | ---: | ---: |
| Control | 64.983824 | 1.083063733 |
| No math errno | 61.428663 | 1.023811050 |
| No math errno | 61.343578 | 1.022392967 |
| Control | 61.321003 | 1.022016717 |

Control drift is 5.6365119%, within the declared 10% bound. Both candidates are
slower than the final control, so they are not both at least 5% faster than both
controls. The integer speedup decision and both-candidates-strictly-real-time
decision are false. Comparing candidates only with the first control would hide
the surrounding-control result. This does not establish that errno handling is
a material bottleneck or a useful production change.

The runner reports AMD EPYC 7763, two cores/four logical processors, Windows
Server 2022 build 20348, AVX2 and FMA. Earlier matching CPU names do not establish
the same runner instance or equivalent conditions. Keep timing comparisons
within this sequence; do not combine prior profile/worker/compiler times.

## Admission and scope

All 15 existing compiler controls, eight math-errno controls and ten actual
Windows Job controls pass. Saved supervision replay verifies all 87 command
statuses are integer zero, with preserved observed/reaped status, assigned
ownership, reaped trees, closed handles and no cleanup errors. There are no
nonempty remaining-process snapshots. The control/candidate command counts are
44/43, including the control's host observation.

Each 66-file SDK passes state/version/private-PE checks and complete source
admission. Independent replay reconstructs the unchanged nineteen-patch,
3425-file GGML tree and checks all 97 actual compiler inputs for both SDKs.
Every compiler command matches its saved baseline after normalizing only the
selected compiler alias and isolated source root, removing one contraction-off
flag and, for the candidate, one math-errno flag. C/CXX initializers have exactly
the declared flags; platform and Release flags remain unchanged. Original
compiler/linker and shared MSVC probe fingerprints match the frozen baseline.

Actual saved `-###` expansions confirm effective O3, Haswell target, loop/SLP
vectorization and contraction off in both builds. The control requires math
errno; the candidate does not. Both logs remain bound with compiler evidence
through the final checks. The driver parser requires one unambiguous target CPU
and rejects conflicting/truncated targets, extra optimization levels and the listed unsafe
math flags. These observations confirm the intended policy was tested; they
do not identify generated instructions for a particular model kernel.

Delivery admission binds all 86 tools and 1411 repository files to the producer
commit before local imports and verifies all 366 delivered members. Artifact
`12103753072` is 31530786 bytes, SHA-256
`2d2064547651367d6f56efc31a6175dcb867f74f16c524e2974f3a75679de19c`.
The collector retains 228 artifact text/raw members and all 86 frozen tools;
Git omits native binaries and pristine archives while retaining their full
descriptors. Physical Windows compiler-alias identity remains a recorded CI
observation; saved replay checks hashes/paths without making a new same-file
claim on macOS.

Matrix execution, conversion/work partitioning and detailed unary paths remain
the next investigation targets from the earlier profile. This result does not
justify adopting the errno flag as a performance fix. A production candidate
still needs a fresh build without diagnostic telemetry and full native/public/
lifecycle performance, packaging/offline/coexistence/receiver qualification.
Original per-slice quality, deferred human-word review, physical desktop and
signing gates remain open. Main remains unpromoted.
