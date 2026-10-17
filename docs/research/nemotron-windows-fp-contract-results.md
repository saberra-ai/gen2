# Scalar contraction result: compiler default difference confirmed

The [preregistered scalar probe](nemotron-windows-fp-contract-probe.md) completes
on actual Windows2022 in [run39032867461](https://github.com/saberra-ai/gen2/actions/runs/39032867461),
job117396700802, source `aa70983a0ad741d92c9b29c3326ab2846da6e6af`.
It confirms a difference between the exact compiler binaries used by the failed
SDK experiment. It does not establish the cause of that model's divergence.

All three binaries use runtime inputs `(1+2^-23),(1-2^-23),-1`, common scalar
optimization/AVX2 options and the same source. Ordinary `a*b+c` produces:

| Variant | Result bits | Arithmetic | Ordinary function disassembly |
| --- | --- | --- | --- |
| MSVC default | `0x00000000` | Zero, with separate product rounding | `vmulss`, `vaddss` |
| Clang default | `0xa8800000` | Exactly-2^-46, fused | `vfmadd213ss` |
| Clang, contraction disabled | `0x00000000` | Zero, with separate product rounding | `vmulss`, `vaddss` |

Every binary also returns zero for the volatile separate-rounding control and
`0xa8800000` for explicit scalar FMA. Each repeats identically in three fresh
processes. MXCSR is0x1f80 in all nine observations. Clang driver logs record
`-ffp-contract=on` by default and `-ffp-contract=off` when the single explicit
`/clang:-ffp-contract=off` option is added. Explicit FMA remains available; this
flag does not remove the portable AVX2/FMA requirement or the source intrinsics.

The observed MSVC/Clang compiler binary hashes exactly match the preceding
failed SDK experiment. This host reports EPYC7763, two cores/four logical
processors, with AVX2/FMA support, rather than that experiment's Intel host.
No timing comparison across these runs is made. The scalar probe constructs
only three tiny programs: no SDK is built/executed and no model is acquired or
loaded. Four interpretation controls and all ten owned-process controls pass.
All19 probe commands return zero and record verified worker/Job cleanup.

The [saved-data collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-fp-contract-probe/collection-qualification.json)
binds all44 members of artifact12100149564,41444bytes, SHA256
`db9a4ffe9f57105d8437cba39bb548831bca65835b73e0d26b8e6aadf48f67bc`.
Before imports/extraction, its collector admits the17 tool/three repository
inputs against the exact producer commit and all bounded ZIP payloads. It replays
compiler identities, recorded build/run commands, all nine arithmetic results,
both Clang driver modes, ordinary-function disassembly, six binary/object hashes
and19 status/cleanup records without running any delivered executable. Private
compiler-environment output is absent; local executables/objects are omitted
from Git while their descriptors remain retained.

This supports one scalar prediction of the implicit-contraction hypothesis.
The failing model may also differ through other arithmetic or kernel choices;
neither scalar agreement nor matching compiler binaries prove model agreement.
The next useful test is a separately frozen full SDK comparison that changes
only Clang's implicit-contraction flag and verifies that flag in every actual
C/CXX compiler command. Preserve the same inference source, portable ISA,
four-worker policy, complete6000x8/turn checks at1e-6 and fixed surrounding MSVC
controls. A scalar result supplies no model parity, real-time performance,
public/lifecycle/package, quality, desktop or release qualification. Main remains
unpromoted.
