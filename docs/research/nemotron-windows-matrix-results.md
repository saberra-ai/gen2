# Windows model matrix paths: complete coverage, observer bound fails

The first full Windows matrix-path experiment completes all source/compiler,
graph-control and numerical checks. Both enabled model traces have complete
84-graph/four-worker/129-root coverage. Converted Q8 SGEMM is the selected path
for 10,584 of 10,836 ordinary matrix roots per worker in each enabled run.
However, the observer exceeds its original overhead bound. Preserve the timing
failure; these observations do not qualify an optimization or production release.

The [421-file receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-profile/collection-qualification.json)
replays [run 39049459015](https://github.com/saberra-ai/gen2/actions/runs/39049459015),
producer `59d1ff1a34dcc3d6bd1e29dda33ee47e3bd8a832`, job 117444894388.
Artifact 12106739798 is 19,405,651bytes, SHA-256
`081908949bca84c319d195aac426abe22f84893540e4e53c44086d5e48ddb9ce`.
All 370 archive members are admitted; 299 delivered evidence payloads and all 110
frozen tools are copied, alongside terminal metadata, receiver code and replay
results. SDK binaries/pristine archives remain in the admitted local delivery.
The [frozen experiment specification](nemotron-windows-matrix-profile.md) retains
its original pre-execution pending status; this document records the actual run.

## Construction, supervision and parity

The source-only local composition first passes eleven commands and all 3,425
GGML files. Actual Windows construction independently admits the same complete
tree and exact twenty-patch series, with only the declared CPU observer file
changed. All 97 compiler inputs retain the original Clang/MSVC probe/linker
identities, contraction-off policy, portable flags and exact diagnostic-version
delta. SDK version/state, private PE/dependency/ABI and installed 66-file checks
pass. No production kernel or runtime lock changes.

All 59 commands exit zero with their raw exit status preserved and their owned
process trees reaped/Jobs closed. Three pre-cleanup membership snapshots contain
one scoped `vctip.exe` each: SDK build PID 3348, native-probe build PID 760 and
matrix-control build PID 7396. The other 56 snapshots are empty. Cleanup is
admitted for every command; these observations concern only the owned Jobs.

Before model acquisition, all twelve actual Windows graph-control processes
pass the required routes, stage attempts/rejections, conversion flags, complete
coverage and fusion. All 144 finite, nonzero raw outputs are byte-identical per
shape across modes, repeats and 1/2/4 workers. Control-source/executable/import-
library identities and earlier evidence are rechecked through final admission.
Build-tree GGML import libraries are actual CI observations; their bytes are not
installed in the delivered SDK, so the saved receiver does not claim to stat
those unavailable Windows inputs.

All four full 60-second model histories contain 6,000×8 probabilities, 58 pushes
and 27 turns. Every 192,000-byte history matches the original reference exactly,
SHA-256 `6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
The complete 84-graph topology/fusion and matrix-metadata signatures match between
enabled runs. Each contains 43,344 admitted worker matrix records; original
raw logs and compact complete profiles are retained and independently replayed.

## Actual paths and geometry

Every worker in each enabled case reports the same route counts:

| Route | Roots per worker across 84 graphs |
| --- | ---: |
| Converted Q8 SGEMM |10,584 |
| Direct SGEMM |190 |
| Vector-dot fallback |62 |
| Hadamard/IQ-panel/extra/fused bypass |0 |

The four repeated Q8 weight geometries are `[2048,512,1,1]`,
`[512,2048,1,1]`, `[512,1536,1,1]` and `[512,512,1,1]` in captured GGML dimension
order. Each accounts for 2,604 roots per worker; corresponding F32 input column
counts span 13–535 with 63 distinct values. The remaining converted roots use
`[1024,512,1,1]` and `[512,192,1,1]` weights, 84 roots each. Complete strides,
output shapes, dot types and per-worker groups remain in the receipt.

In the instrumented observations, converted routes account for 97.29–98.68% of
each worker's matrix kernel wall. Worker0 matrix wall is 40.024252/38.349330s;
second-stage SGEMM wall is 37.024220/36.076162s, conversion 0.378593/0.380116s and
the existing inner barrier 2.081971/0.989088s. These are separate worker-local
observations, with internal waits/scheduling included. The failed observer bound
precludes treating these ratios as validated uninstrumented costs or causal
speedup predictions. Never add overlapping workers into model wall time.

## Timing and next action

The receiver is Windows Server 2022 build 20348, AMD EPYC 9V74 with two reported
cores/four logical processors and AVX2/FMA. Earlier experiments used EPYC 7763;
this is not a same-host performance comparison or evidence of a code speedup.

| Fresh process | Elapsed seconds | Native RTF |
| --- | ---: | ---: |
| Off0 |58.323447 |0.972057450 |
| On1 |73.079115 |1.217985250 |
| On2 |66.115085 |1.101918083 |
| Off3 |55.115330 |0.918588833 |

Control drift 5.5005614% passes the unchanged 10% bound. Both instrumented cases
exceed 110% of the faster surrounding control, so the observer-overhead decision
and bounded timing interpretation fail. Keep both controls and enabled cases;
do not average away this result or raise the bound. The off controls are below
one on this receiver, but this diagnostic SDK is not production/public/lifecycle
or package qualification.

Next, investigate observer reporting cost in a separately versioned control.
Batched emission is a source-supported candidate for testing, not an established
explanation: retain every captured record, exact coverage and raw output parity
while measuring the original observer bounds again. The selected Q8 paths and
actual geometries can guide later kernel controls; no synthetic tile result
establishes a model speedup. Original quality/attribution, Windows production/
public/package, desktop/fresh-machine and signing gates remain open. Human word
annotation and independent review remain documented and deferred. Main remains
unqualified.
