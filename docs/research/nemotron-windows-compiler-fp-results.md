# Full SDK contraction-off result

The [preregistered experiment](nemotron-windows-compiler-fp.md) completed at
source `65d6afcdcc979318932e96d01b721b2116a58f36` in
[run 39033952288](https://github.com/saberra-ai/gen2/actions/runs/39033952288),
job `117399906774`. Disabling implicit Clang contraction restores byte-exact
output for the tested clip and compiler binaries. It does not meet the original
real-time requirement, and the preregistered 10% speedup prediction is unsupported.

The [313-file receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp/collection.json)
retains the full arrays, commands, compiler/source evidence and frozen tools.
Its [collection qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp/collection-qualification.json)
replays the actual result without executing a compiler, SDK or model.

## Numerical and timing observations

All four fresh processes use the unchanged four-worker policy, 58 pushes,
960000 input samples and original geometry. Every 6000x8 probability array is
192000 bytes with SHA-256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
All 27 turns match the frozen reference; maximum probability and turn errors
are zero. The original 1e-6 limits remain unchanged.

| Ordered case | Native time (seconds) | Native RTF |
| --- | ---: | ---: |
| MSVC default | 71.005684 | 1.183428067 |
| Clang contraction off | 64.067791 | 1.067796517 |
| Clang contraction off | 65.194315 | 1.086571917 |
| MSVC default | 71.417239 | 1.190287317 |

Control drift is 0.5796085%, within the preregistered 10% bound. Both candidates
are faster than both controls, but they are not both at least 10% faster than
both controls. The integer-microsecond decision is therefore false. All four
RTFs exceed one; no native or public real-time pass follows.

The host is an Intel Xeon Platinum 8370C runner reporting two cores and four
logical processors, Windows Server 2022 build 20348, AVX2 and FMA. Prior full and
scalar experiments used different CPU models; their times cannot be combined
with these observations to estimate a contraction cost or speedup.

## Admission and scope

All 15 compiler controls and 10 actual Windows Job controls pass. Saved replay
verifies all 84 commands have integer zero status, preserved observed/reaped
worker status, assigned Job ownership, empty cleanup, closed handles and no
cleanup errors. Scoped background-process observations remain in the receipt.

Both 66-file SDKs pass state/version/private-PE checks, independent reconstruction
of the 19-patch, 3425-file GGML tree and all 97 actual compiler inputs. Compiler
and linker fingerprints match the preceding failed full experiment. Each
compiler command matches its corresponding saved baseline after normalizing
only the admitted compiler path alias and isolated source root, and removing
exactly one candidate `/clang:-ffp-contract=off` token. Platform and Release
flags otherwise remain unchanged; explicit FMA intrinsics remain enabled.

Delivery replay binds all 79 tools and 804 repository inputs to the producer
commit before local imports, then verifies the complete 360-member artifact.
Artifact `12101294542` is 31038980 bytes, SHA-256
`58dc50651e1cfea1b6c6a86cba8e76f2eb8771f9cd0e5e19fd0b9b9a8abb8e59`.
Git omits native binaries and pristine archives while retaining their descriptors.
Physical Windows compiler-alias identity was measured in CI; saved replay checks
recorded paths and hashes without claiming a second Windows same-file observation.

This result supports the implicit-contraction explanation for the observed
full-model discrepancy in this configuration. It does not identify a particular
operator, establish broader quality, or qualify a production SDK. Both SDKs keep
their diagnostic identity. A production candidate would require fresh construction
without diagnostic telemetry and complete native/public/lifecycle, performance,
package, offline, coexistence and receiver qualification. Human word annotation,
physical desktop checks, signing and the original per-slice quality gates remain
open. Main remains unpromoted.
