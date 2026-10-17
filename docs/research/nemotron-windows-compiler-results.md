# Windows compiler comparison: Clang parity failure

The [versioned ownership correction](nemotron-windows-compiler-diagnostic-v2.md)
passes its actual Windows controls and allows both SDKs to finish construction.
The compiler experiment then stops at its first Clang numerical comparison.
[Run39031011256](https://github.com/saberra-ai/gen2/actions/runs/39031011256),
job117391281377, source `dd4a107e13b1f63a0cd796fd45511621fdba08cd`, is a failed
diagnostic. It establishes no qualifying compiler improvement or release pass.

Both ten-test preflight suites pass. All82 supervised commands return zero,
retain matching observed/reaped exit statuses, verify an empty owned Job and
close its handle. The MSVC SDK-build snapshot records owned `vctip.exe` and
`mspdbsrv.exe` processes; the Clang SDK build and both MSVC-built probes record
owned `vctip.exe` processes. The helper drains the owned Job and verifies it empty
before returning the successful command status. These membership-confirmed images
describe this run only: the earlier failure did not record descendant identities.

Both SDKs pass source, state, version, private PE/import and compiler admission
before model use. Each installed inventory has66 files; both use the same
four-worker diagnostic inference source and MSVC probe source/compiler. CMake
records MSVC19.44.35229.0 and Clang19.1.5 with the MSVC ABI. Both compiler databases
contain97 inputs and16 GGML CPU entries with the original AVX2/FMA/F16C portable
configuration. Neither diagnostic SDK is production-admitted. The host reports
Intel Xeon Platinum8573C, two cores/four logical processors, AVX2 and FMA.

Only the first two processes of the fixed MSVC/Clang/Clang/MSVC sequence execute:

| Index | SDK compiler | Native elapsed | Native RTF | Retained output |
| --- | --- | ---: | ---: | --- |
| 0 | MSVC | 62.204758s | 1.0367459666666667 | All6000x8 values byte-exact to the historical reference;27 turns match |
| 1 | Clang | 59.423602s | 0.9903933666666667 | Probability comparison fails;26 turns versus27 |

Each process accepts the same960000-sample PCM, performs58 pushes, retains all
6000x8 probabilities, uses four effective/attached workers and the unchanged
9/4/0/264/264/222 geometry. Timing instrumentation remains off. The first MSVC
history has SHA256 `6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`;
Clang's is `60f7d9542bf60e5d24b793bc56056e02d4f1668c1ae9a51f7da36267d6abc0be`.

Full saved-array replay reproduces the original
`ValueError('native/Gen2 retained probability values differ')` at the unchanged
1e-6 limit. Of48000 values,13975 exceed that limit and47748 differ numerically.
Maximum absolute error is0.2715801000595093 at zero-based frame3754/slot3:
MSVC0.7293691039085388 versus Clang0.45778900384902954. All values remain finite
and within[0,1]. The outputs are materially different; changing their tolerance
or using Clang's single below-one timing cannot qualify this candidate.

The second Clang process and closing MSVC control never start. There is no full
timing comparison, matched control drift measurement, supported10% improvement
prediction, public/lifecycle run or package qualification. Preserve both actual
observations; do not average, select the faster observation or attribute a
particular operator/compiler mechanism from these data.

The [failure collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure/collection-qualification.json)
binds all352 members of artifact12100258999,30675366bytes, SHA256
`ae52aa11901185175fbe755e66f20fa7d7b28a54d0810fbbe032216edc6fce1f`.
Its delivery collector admits73 tool/792 repository inputs against the exact
producer commit before local imports. Subsequent saved-data replay rebinds all
command observations, both66-file SDKs, private PE results, state/version/probe
evidence, generated compiler identities, all97 compiler inputs and an independent
canonical19-patch GGML reconstruction. It retains both full192000-byte arrays and
reproduces the original pass/failure without executing a compiler, SDK or model.
Windows physical same-file alias verification remains a recorded CI observation;
the saved replay verifies its hashes/path evidence without claiming to repeat
that Windows API operation on this Mac.

The first saved-data collector reached receipt export, then stopped on an
assertion rejecting four probe compiler-generated `.obj` files. Its original
script, failure record and87 byte-exact partial copies are preserved. A separately
versioned collector excludes build objects alongside archives/native binaries;
all352 payload descriptors remain retained. No native result, gate or producer
input changes during this collection correction. SDK binaries/pristine archives
remain local; the repository retains scoped logs, source/compiler evidence and
complete arrays.

The next investigation must explain the Clang numerical divergence or identify
a different parity-preserving execution improvement. Preserve MSVC as the
admitted numerical control, all original quality limits, strict below-one native
and public performance, cleanup/receiver/offline/coexistence controls and desktop
release requirements. Human annotation remains deferred as requested. Main
promotion remains unjustified.
