# Q8 one-row dispatch: exact outputs, failed speedup and observer gates

The [frozen comparison](nemotron-windows-q8-row.md) completes construction,
numerical and cleanup checks. All1,056 full phase vectors match the production
Mx4 control byte-for-byte. None of48 required four-worker primary comparisons
meets the original5% speedup predicate. The observer gate also fails. This
candidate supplies no basis for promotion or a qualifying full-model comparison.

Producer `2dbf3a008bde6ad405dccd33bb7f015da1901b46` completes
[run39070438530](https://github.com/saberra-ai/gen2/actions/runs/39070438530),
job117506743687. Workflow success establishes a completed diagnostic, not a
passing timing prediction. The same producer's
[feature CI](https://github.com/saberra-ai/gen2/actions/runs/39070438487)
also completes all thirteen jobs successfully. Original failures and the frozen
preregistration remain unchanged.

## Independent admission and replay

The [1,436-member receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row/collection.json)
and [qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row/collection-qualification.json)
bind artifact12114758099,750,522,299bytes, SHA256
`e2e86f74522a50099a1bc5c3e7dbe279656eebaf9d46f56e56abc5be4fec732d`.
All4,697 delivered members,140 tool identities and3,758 producer repository
references pass independent admission. Every raw vector, disassembly, text and
source proof is retained. Canonical source, pristine archive, actual kernel
objects and executable remain in the admitted local/GitHub delivery, with the
complete member map preserved.

Replay verifies the3,425-file canonical source and nineteen original patch
reads, exact production control and one-row source, original compiler/linker
identities and contraction-off flags, five project inputs, both kernel
source/command/object associations and complete dependency evidence. Fixed
control/candidate aliases and guards must agree with their roles, and both
translation units occur exactly once. Guard and benchmark executable, complete
argv, output prefix and working directory remain bound.

All95 commands retain their expected integer exits and complete owned-Job
cleanup:59 zero exits and twelve each of73/74/75. The configure snapshot
contains one membership-confirmed `vctip.exe` PID7928, reaped with its closed
Job. Other completion snapshots are empty. All36 support rejections, twelve
complete65,536-entry half tables and1,056 finite, nonzero full vectors pass,
including padded guards, tails, input immutability and repeated calls.

All8,448 whole-dispatch timings and9,856 enabled worker intervals remain.
Sixty-five zero-duration intervals satisfy the original below-resolution rule;
disabled processes emit none. Every worker interval fits its whole dispatch.
Wall intervals include preemption, stalls and observer costs.

Actual Windows controls pass15 compiler,7 original tile,7 corrected support,
6 geometry,6 worker timing,6 one-row and10 owned-Job tests. Local preparation
passes563 tool tests with seven Windows-only skips. Saved-data collection
invokes no compiler, benchmark, SDK or model.

## Original timing decisions

The host reports AMD EPYC7763, two cores/four logical processors, AVX2/FMA,
Windows Server2022 build20348. It differs from the prior pair's EPYC9V74 and
production candidate's Xeon8573C. Names shared with any other diagnostic do
not establish a matched host. Only this run's paired observations are compared.

Each fresh process uses one persistent pool and all22 fixed shapes. Every
shape executes control/row1/row1/control with one warmup and eight timed calls
per phase. The following counts cover all twelve original primary shapes.
One/two-worker and non-primary observations cannot replace failed primaries.

| Workers | Process/mode | Speed passes | Control-drift passes | Both pass | Candidate-repeat-drift passes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 1/on | 0/12 | 11/12 | 0/12 | 12/12 |
| 1 | 2/on | 0/12 | 11/12 | 0/12 | 12/12 |
| 1 | 3/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 2 | 0/off | 0/12 | 7/12 | 0/12 | 10/12 |
| 2 | 1/on | 0/12 | 6/12 | 0/12 | 6/12 |
| 2 | 2/on | 0/12 | 9/12 | 0/12 | 7/12 |
| 2 | 3/off | 0/12 | 7/12 | 0/12 | 10/12 |
| 4 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 1/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 2/on | 0/12 | 12/12 | 0/12 | 11/12 |
| 4 | 3/off | 0/12 | 12/12 | 0/12 | 12/12 |

Four-worker ratios below are100 times the larger candidate eight-call total
divided by the smaller surrounding control total. The original speed predicate
requires each ratio at most95%, together with its separate drift requirement.
These rounded ratios do not replace the exact inclusive integer predicates.

| Shape `(m,n,k)` | Off0 % | On1 % | On2 % | Off3 % |
| --- | ---: | ---: | ---: | ---: |
| `(512,13,2048)` | 109.378 | 106.546 | 107.856 | 109.179 |
| `(512,292,2048)` | 116.656 | 113.044 | 132.752 | 115.946 |
| `(512,535,2048)` | 114.378 | 115.857 | 114.574 | 114.568 |
| `(2048,13,512)` | 106.151 | 104.658 | 106.696 | 105.317 |
| `(2048,292,512)` | 115.167 | 112.744 | 113.248 | 112.186 |
| `(2048,535,512)` | 114.153 | 116.177 | 116.360 | 116.244 |
| `(1536,13,512)` | 106.057 | 105.580 | 102.489 | 107.469 |
| `(1536,292,512)` | 114.548 | 113.572 | 112.491 | 110.782 |
| `(1536,535,512)` | 120.595 | 116.711 | 115.204 | 114.577 |
| `(512,13,512)` | 104.963 | 104.998 | 100.660 | 110.287 |
| `(512,292,512)` | 119.673 | 116.977 | 112.576 | 111.816 |
| `(512,535,512)` | 114.476 | 118.587 | 114.365 | 112.298 |

The separate primary observer comparisons pass drift47/48, overhead42/48 and
both bounds41/48. Every failing row remains, with exact off/on/on/off totals:

| Shape index / phase | Off0 ns | On1 ns | On2 ns | Off3 ns | Failed bound |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 / 3 | 2496500 | 2262600 | 2241500 | 2236700 | Drift |
| 5 / 2 | 44056200 | 43156200 | 50551600 | 43906000 | Overhead |
| 7 / 1 | 2414800 | 2469300 | 2540100 | 2486000 | Overhead |
| 13 / 0 | 1059600 | 1016400 | 1060200 | 964300 | Overhead |
| 14 / 0 | 10063000 | 10421000 | 10845200 | 10177400 | Overhead |
| 14 / 2 | 11493600 | 11529800 | 11788600 | 10904400 | Overhead |
| 14 / 3 | 9891900 | 9856500 | 10471700 | 9992300 | Overhead |

All three flags remain false: `same_process_kernel_prediction_supported`,
`bounded_observer_interpretation_supported` and `kernel_speedup_prediction_supported`.

## Qualification limits

The candidate has four source accumulator vectors rather than eight. This
run does not establish lower generated register pressure, measured traffic,
a spill cause or any bottleneck explanation. B reloads, repeated scale
conversion and changed tile ownership remain documented tradeoffs, not measured
causes of these observations. No new static-loop claim is made.

These are synthesized matrix values at captured model geometries. No full-model
RTF, SDK, public/lifecycle, package, offline/coexistence, desktop or release
qualification follows. Production remains unchanged. The original AMI quality
failures, deferred human word annotation/review, GUI/microphone/playback,
fresh-machine app operation and platform/signing requirements remain open.
