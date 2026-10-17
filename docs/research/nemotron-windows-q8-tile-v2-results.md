# Windows Q8 tile control: exact outputs, slower candidate

The corrected standalone harness builds and completes every control. For each
of the seven synthetic shapes, outputs match exactly across all twelve
variant/worker processes. The 2×2 candidate is slower than both surrounding
controls on every primary shape, so its original speedup prediction fails.
This result supplies no production optimization or model RTF qualification.

Producer `4390e16a73bd4e12f3c703c2b75d8c52c16d2e26` completed
[run 39045070426](https://github.com/saberra-ai/gen2/actions/runs/39045070426),
job `117432055187`. The [frozen corrected experiment](nemotron-windows-q8-tile-v2.md)
preserves the original candidate, inputs, worker order and timing criteria. The
[first failed build](nemotron-windows-q8-tile-build-failure.md) remains unchanged.

## Admitted construction and controls

The [saved receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-tile-v2/collection-qualification.json)
binds artifact `12104769570`, 37,776,813 bytes, SHA-256
`9abdc2a7f198bd4e3210cca92d9c81e45fc1013f408ac638b56d727d8253723b`.
Saved-data replay verifies all 3,724 delivered members, 102 frozen tools and
1,955 producer repository references. All 3,425 canonical source files, the
pristine archive, nineteen patch reads and the exact candidate transform match.
The fourteen compiler-input records map to admitted canonical/candidate/harness
source. Original Clang/linker fingerprints, global/Release flags and explicit
contraction off match the preceding admissions. Producer object descriptors,
the two delivered kernel objects/disassemblies and final executable are bound.

All 95 command records preserve the expected actual exit and successful owned
cleanup: 59 zero exits and twelve each of 73, 74 and 75 for the rejection modes.
There are no cleanup errors; every tree is reaped and Job closed. Before cleanup,
94 snapshots are empty and configuration records one membership-confirmed
`vctip.exe` descendant handled by owned cleanup.

Each kernel case is preceded by three separately supervised support rejections
using the same bound executable. All 36 exact codes/diagnostics pass. Every case
also retains the complete 262,144-byte half table; all twelve tables match
SHA-256 `b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf`.
Independent replay checks all 63,488 finite values, signed zero, infinity signs
and NaN classes. Actual cache/build/compiler/dependency evidence confirms OpenMP
exclusion and absence of the prior exception-contract warnings. The harness
corrections resolve the observed construction/control defects.

All 84 compact output vectors are complete, finite and nonzero. Per-shape bytes
match across control/candidate and one/two/four workers. The bound harness checks
input immutability, padded strides, output guards and repeat equality after every
call. The original 15 compiler controls, seven original tile controls, seven
corrected controls and ten actual Job controls pass before construction. The local
suite passed 437 tests with seven Windows-only skips before the producer push.

## Timing result

The following are totals of all eight timed calls, in milliseconds, on four
workers in the fixed control/candidate/candidate/control order. Every individual
integer nanosecond measurement remains in the receipt; none is discarded.

| Primary `(m,n,k)` | Control 1 | Candidate 1 | Candidate 2 | Control 2 | Control drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(512,33,512)` | 1.7567 | 1.9985 | 1.8577 | 1.7231 | 1.9127% |
| `(2048,33,512)` | 5.0888 | 5.5833 | 5.5338 | 5.0758 | 0.2555% |
| `(512,264,512)` | 9.7793 | 10.3756 | 10.5909 | 9.7078 | 0.7311% |

Every primary shape passes the unchanged 10% control-drift predicate. Every one
fails the unchanged requirement that both candidates be at least 5% faster than
both controls. Both candidates are slower than both controls on all three shapes.
The overall kernel-speedup prediction is false. Timings include harness worker
wake/dispatch and scheduling effects; the observations do not establish a
register-spill or input-reuse cause.

The host reports Windows Server 2022 build 20348, AMD EPYC 7763, two cores/four
logical processors, AVX2 and FMA. These are same-run surrounding comparisons;
the CPU name does not establish the same machine as earlier runs. The shapes are
synthetic and have not been established as actual model tensor paths/shapes.

Retain the existing production kernel and investigate actual model MUL_MAT
shapes, types and dispatch coverage before selecting another candidate. No SDK
or model is built/loaded in this experiment. Full-model/native/public numerical,
quality/performance, lifecycle, package/offline/coexistence, desktop and signing
qualification remain open. Human word-time review stays documented and deferred
at the user's request. Main remains unqualified.

## Retained evidence

The receipt preserves textual/raw artifact copies, all individual timings and
vectors/tables, the full member map, producer metadata, frozen tool sources and
executed receiver/publisher provenance. The admitted ZIP, complete canonical
source, pristine archive, native objects and executable stay in the local cache.
Collection verifies saved bytes without repeating compilation or benchmark work.
