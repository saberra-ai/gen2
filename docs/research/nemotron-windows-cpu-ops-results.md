# Windows CPU operation profile result

The [frozen experiment](nemotron-windows-cpu-ops-experiment.md) completed at
source `a705db243ab134c8f0ed9aa375731a17057d145b` in
[run 39037988751](https://github.com/saberra-ai/gen2/actions/runs/39037988751),
job `117411575626`. Actual measurement admission passes, including complete
graph coverage, exact output parity and the preregistered observer bounds.
Real-time performance still fails; this diagnostic does not qualify production.

The [292-file receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-cpu-profile/collection.json)
retains all source/compiler text, full raw outputs and actual commands. Its
[saved-data replay](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-cpu-profile/collection-qualification.json)
executes no compiler, SDK or model.

## Output and timing

All four fresh processes use four effective and attached workers, the original
60-second PCM and geometry, and 58 pushes. All 6000x8 probability arrays are
192000 bytes with SHA-256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
All 27 turns match the frozen reference; probability and turn errors are zero.
The original 1e-6 limits are unchanged.

| Ordered case | Native time (seconds) | Native RTF |
| --- | ---: | ---: |
| Timing off | 60.811208 | 1.013520133 |
| Timing on | 61.247873 | 1.020797883 |
| Timing on | 60.919708 | 1.015328467 |
| Timing off | 60.987695 | 1.016461583 |

Control drift is 0.2902212%. Both instrumented cases are within 10% of both
controls, so the integer-microsecond observer predicates pass. Every RTF exceeds
one. These are native diagnostic measurements, not public-runtime qualification.
The runner reports an AMD EPYC 7763, two cores/four logical processors, Windows
Server 2022 build 20348, AVX2 and FMA. Compare timings within this sequence only;
other experiments used different runner CPUs or instances.

Each enabled case contains exactly 84 graphs with all four workers and one
persistent pool. Both cases have identical ordered topology and actual root/
fusion signatures. The parser checks canonical operation IDs/names, complete
counts and worker intervals, including outer barriers and final waits.

## Where measured time goes

The following reports worker 0 separately for each enabled process. Percentages
divide an operation's cumulative kernel wall time by that worker's total kernel
wall time across the 84 sequential graphs. They are not shares of arithmetic
work or sums across workers.

| Operation | First enabled: seconds / kernel share | Second enabled: seconds / kernel share |
| --- | ---: | ---: |
| MUL_MAT | 41.188064 / 69.57% | 41.438961 / 69.79% |
| UNARY | 9.611931 / 16.23% | 9.590282 / 16.15% |
| FLASH_ATTN_EXT | 6.567378 / 11.09% | 6.507396 / 10.96% |

These three classes account for about 97% of observed kernel wall time in worker 0.
All eight worker/case observations have the same leading classes: MUL_MAT takes
69.54–69.89%, UNARY 15.74–16.23%, and FLASH_ATTN_EXT 10.91–11.29%. Full per-worker
totals remain in the receipt. Worker0's cumulative outer-barrier time is
1.776104/1.280821 seconds in the two enabled cases; outer barriers do not dominate
this measurement.

Kernel calls include internal synchronization, scheduling and observer effects.
The workers overlap; adding their times would overstate elapsed work. The
topology hash and reporting can also delay subsequent graphs. This supports
investigating the matrix/unary/attention paths, but does not identify a slow
instruction, unary subtype, tensor shape or a validated optimization. Preserve
this producer and introduce separately versioned evidence for any next change.

## Admission and remaining gates

All 15 original compiler controls, 15 profiling controls and 10 actual Windows
Job controls pass. Saved replay checks all 59 commands have zero integer status,
preserved observed/reaped status, assigned ownership, empty final membership,
closed handles and no cleanup errors. No nonempty background snapshots occur.

The 66-file SDK passes state/version/private-PE checks. Fresh saved-data replay
reconstructs all twenty patches and 3425 GGML files, admitting only the exact
declared CPU timing-file change from the canonical nineteen-patch tree. All 97
actual compiler inputs match. Compiler/linker hashes, original platform/Release
flags and explicit contraction-off initialization match the frozen baseline;
only the exact diagnostic version token changes in all 97 commands.

Before model acquisition, twelve fresh Windows processes exercise the admitted
SDK with one/two/four workers and off/on/on/off timing. All 36 finite nonzero
8192-byte synthetic vectors are byte-identical, and every enabled worker reports
the expected fused normalization/multiply pair. CI binds the control executable,
source and three private import libraries before and after use. Saved replay
independently checks retained executable/source/output bytes; absent build-tree
import-library bytes and physical compiler aliases remain recorded CI observations.

Delivery admission binds all 93 tools and 1212 repository files to the producer
commit before imports and verifies all 258 artifact members. Artifact
`12102284080` is 16476579 bytes, SHA-256
`0562b69efe5cc49703b108e9b55d088ade5136b0706f32dc9514617045f972e6`.
Git omits binaries and pristine archives while retaining all payload descriptors.
The collector explicitly requires four effective workers in disabled controls
as well as enabled cases; this check was added during prospective review before
collection. No producer input changed after execution.

The distinct `gen2-diagnostic-cpu-operations-v1` SDK cannot enter production
admission. A production optimization requires fresh construction without
diagnostic telemetry and complete native/public/lifecycle performance, package,
offline, coexistence and receiver qualification. Original quality, deferred
human-word review, physical desktop and signing gates remain open. Main remains
unpromoted.
