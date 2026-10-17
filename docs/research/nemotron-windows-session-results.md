# Windows native session-time result

The [fixed off/on/on/off diagnostic](nemotron-windows-session-profile.md) completes
in [run39027810779](https://github.com/saberra-ai/gen2/actions/runs/39027810779),
source `c183c6cf3501c8b45992a53638ddbbd4b943e449`. Both profiled cases spend about
**99.6% of native elapsed time inside graph compute plus output/synchronization**.
This directs the next investigation inside graph execution. It does not identify
a hot operator, prove arithmetic rather than worker scheduling dominates, or
establish a compiler cause. Every observed RTF remains above1.

| Fixed order | Timing | Native elapsed, 60s input | Native RTF |
| --- | --- | --- | --- |
| 1 | off | 63.644086s | 1.0607347667 |
| 2 | on | 63.465587s | 1.0577597833 |
| 3 | on | 65.941087s | 1.0990181167 |
| 4 | off | 85.380946s | 1.4230157667 |

All four fresh processes use the same admitted four-worker SDK without a rebuild,
one Windows2022 receiver and fixed original model/PCM/streaming geometry. CIM
reports AMD EPYC7763, two cores/four logical processors, with AVX2/FMA available.
Each process logs configured/effective/attached pool size4. All complete6000x8
float32 histories are byte-exact to the prior numerical reference, SHA256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`;
all27 final turns agree with zero boundary error. No numerical tolerance changes.

| Measured region | First profiled case | Second profiled case |
| --- | --- | --- |
| Graph compute | 63214.22ms | 65690.70ms |
| Output and synchronization | 1.40ms | 1.37ms |
| Input and allocation | 63.88ms | 64.92ms |
| CPU frontend | 57.44ms | 57.13ms |
| Outside timed regions | 128.667ms | 127.037ms |
| Compute plus output / native elapsed | 99.6061% | 99.6224% |

Each profiled stderr contains all84 scheduled-session and59 CPU frontend rows;
both controls contain none. The parser replays every row and rounding-bound
check, rather than accepting only the aggregate. The first preregistered
prediction, at least85% in synchronized graph computation, is supported for these
observations. The at-least15% outside-region prediction is not supported: the
residual is about0.20%/0.19%. Input/allocation is about0.10%, and frontend about
0.09%. In either profiled case, removing those surrounding costs alone would
still leave more than60seconds inside graph compute/output.

The final control is **34.15% slower** than the first. Its native push p95 rises
from2.349342s to5.422185s and maximum from2.424760s to9.036286s. Both profiled
observations are within3.61% of the first control, but the control drift prevents
a clean causal estimate of instrumentation overhead. Retain the four individual
measurements; do not average them, compare their times as matched controls to
the previous producer, or infer a logging effect from the slow final control.
Graph-region wall time can include worker scheduling and host contention; these
timers alone do not distinguish those from kernel computation. A next diagnostic
may examine operator costs or a controlled compiler comparison while preserving
the portable CPU baseline, four-worker policy, full-output parity and original
performance gate. No optimization is selected or qualified by this result.

The [34-member collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-session-profile/collection-qualification.json)
binds all28 members of artifact12099018948,963031bytes, SHA256
`2db162d7389b7f58ecbcdd416935ac8c0834cce2b648938b0b59f91ffe9d9e34`.
Before importing local evaluators, its collector admits all59 tool/638 repository
inputs against the exact producer Git commit and protocol. It replays all nine
successful child commands, source-delivery identities, complete vectors/turns,
worker telemetry and143 timing rows in each profiled case. Destination evidence,
metadata and collector copies are checked before success publication.

The actual CI driver rehashes all516 downloaded source-artifact members and the
66-file SDK before copying, before every case and after the sequence. The new
profiling artifact excludes SDK/delivery/model payloads. The local collector
compares its SDK manifest and PE report to the prior frozen receipt and also
rehashes the retained prior ZIP and66-file SDK; it does not have a second copy of
the new receiver's SDK payload. No model or delivered binary is executed during
collection. The preregistered producer/protocol remain unchanged.

The full366 model-free tests passed before dispatch; the reviewed parser fix then
passed all seven focused tests and the actual Windows focused test step. The
actual checkout preserves LF through process-scoped Git configuration, leaving
saved host settings unchanged. Release, public/lifecycle and package qualification
remain false. Windows bounded package construction/receiver cleanup, quality,
human annotation, physical desktop and other release gates remain open.
