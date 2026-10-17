# Q8 worker timing: exact outputs, failed speedup and observer bounds

The same-process experiment completes every construction, numerical and cleanup
check. All 1,056 complete phase vectors match byte for byte across both kernels,
all twelve fresh processes and one/two/four workers. The original all-primary
speedup prediction still fails. The separately declared observer bound also
fails, so bounded observer interpretation is withheld. No production kernel or
release qualification changes.

Producer `7fadd1497f58f2d287dc44e90c1337283074ad5e` completed
[run 39056905799](https://github.com/saberra-ai/gen2/actions/runs/39056905799),
job `117466626836`. The [frozen experiment](nemotron-windows-q8-worker-timing.md)
keeps its pre-execution pending status and original decisions. The
[earlier fresh-process result](nemotron-windows-q8-mx4-results.md) and
[static loop inspection](nemotron-windows-q8-loop-inspection.md) remain unchanged.

## Independently admitted evidence

The [saved receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-worker-timing/collection-qualification.json)
binds artifact `12110037894`, 750,483,648 bytes, SHA-256
`a919b26dfa075a15c195c0e4df7a6715109dbc4df2d65dc479bcfa37016d14fa`.
Saved-data replay admits all 4,696 delivered members, 120 frozen tools, 3,420
producer repository references, the exact pristine archive, nineteen patch reads
and complete 3,425-file canonical source map. Both existing kernel sources,
candidate platform guard, project/compiler inputs, original compiler/linker
fingerprints, global/Release options, contraction off, objects, disassemblies
and executable/dependency evidence match. Collection invokes no compiler,
native benchmark, SDK or model.

All 95 command records retain expected exits and admitted owned cleanup:
59 zero exits and twelve each of 73/74/75 for the support rejection controls.
All 36 rejection diagnostics and twelve complete half tables independently pass,
including all 63,488 finite half values and special-value classes. Every phase
retains its own full output, with byte-exact same-process and cross-process
comparisons, input immutability, padded strides, guards and repeated-call checks.

All 8,448 whole-dispatch timings are retained. The six enabled processes retain
4,224 complete timed dispatch samples containing 9,856 ordered worker intervals.
All satisfy `0 <= start <= end <= whole_dispatch`. There are 57 reported
zero-duration worker intervals, allowed by the frozen below-clock-resolution
rule; no duration is synthesized. Disabled processes emit no worker intervals.
Replay independently reconstructs the worker wall summaries and every timing
predicate from the saved raw observations.

Actual Windows prerequisites pass 15 compiler, seven original tile, seven
corrected support, six captured-geometry, six new worker-timing and ten owned-Job
controls. The local suite passed 485 tests with seven Windows-only skips. The
separate [feature-check run](https://github.com/saberra-ai/gen2/actions/runs/39056905757)
also completes all thirteen jobs successfully at this producer, including the
macOS/Linux/Windows feature combinations. Those checks do not establish model
RTF or desktop release readiness.

## Same-process kernel decisions

Every process uses one persistent pool. Each shape executes control/candidate/
candidate/control with one warmup and eight timed calls per phase. The outer
fresh-process observer order is off/on/on/off for each worker count.

Counts below refer to the twelve fixed primary shapes. Every shape, process,
worker count and individual integer observation remains in the receipt.

| Workers | Process/mode | Speed predicate passes | Control-drift passes | Both pass | Candidate-repeat-drift passes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0/off | 2 | 12 | 2 | 12 |
| 1 | 1/on | 0 | 12 | 0 | 12 |
| 1 | 2/on | 3 | 12 | 3 | 12 |
| 1 | 3/off | 2 | 12 | 2 | 12 |
| 2 | 0/off | 2 | 8 | 1 | 9 |
| 2 | 1/on | 0 | 7 | 0 | 8 |
| 2 | 2/on | 4 | 8 | 2 | 8 |
| 2 | 3/off | 1 | 6 | 1 | 8 |
| 4 | 0/off | 2 | 12 | 2 | 12 |
| 4 | 1/on | 2 | 12 | 2 | 12 |
| 4 | 2/on | 3 | 12 | 3 | 12 |
| 4 | 3/off | 1 | 11 | 1 | 12 |

Only eight of 48 four-worker primary comparisons pass the original speedup and
control-drift predicates; all 48 would be required. Candidate-repeat drift passes
all 48, but this does not replace either original predicate. Forty-seven of 48
control-drift comparisons pass. The exception is shape 5 in final off process 3:
first/final control totals 39,167,300/45,581,300ns. No observation is removed.

For transparency, each entry below is the larger candidate eight-call total
divided by the smaller surrounding control total. Values are rounded for display;
the actual predicate uses integer nanoseconds and requires at most 95%.

| Primary `(m,n,k)` | 0/off | 1/on | 2/on | 3/off |
| --- | ---: | ---: | ---: | ---: |
| `(512,13,2048)` | 96.685% | 96.430% | 97.352% | 97.754% |
| `(512,292,2048)` | 97.258% | 96.329% | 97.333% | 96.850% |
| `(512,535,2048)` | 96.960% | 96.703% | 97.134% | 97.871% |
| `(2048,13,512)` | 94.882% | 100.197% | 96.452% | 93.967% |
| `(2048,292,512)` | 95.918% | 94.310% | 93.615% | 96.264% |
| `(2048,535,512)` | 97.024% | 95.570% | 95.649% | 96.006% |
| `(1536,13,512)` | 97.020% | 99.964% | 95.745% | 97.964% |
| `(1536,292,512)` | 95.563% | 95.323% | 94.488% | 96.068% |
| `(1536,535,512)` | 94.868% | 95.434% | 97.951% | 96.056% |
| `(512,13,512)` | 97.556% | 102.742% | 96.429% | 100.423% |
| `(512,292,512)` | 98.081% | 93.017% | 92.402% | 95.564% |
| `(512,535,512)` | 95.879% | 95.701% | 97.017% | 98.682% |

## Observer decision and interpretation limits

Among all 48 primary/position comparisons across off/on/on/off processes,
47 pass control drift, 43 pass the at most 5% observer-overhead predicate, and
42 pass both. The six failed combined comparisons retain these eight-call totals
in nanoseconds, in off/on/on/off order:

| Shape index | Kernel position | Totals | Failed predicate |
| ---: | ---: | --- | --- |
| 4 | 3 | 2361700,2463300,2337600,2337000 | Observer bound |
| 5 | 3 | 39569800,39142900,39751900,45581300 | Control drift |
| 7 | 2 | 2277100,2391000,2316300,2300400 | Observer bound |
| 13 | 1 | 1009700,1067700,1028800,1045700 | Observer bound |
| 14 | 0 | 10662900,10609600,11277100,10822000 | Observer bound |
| 14 | 3 | 10461900,10474200,11021100,11299300 | Observer bound |

The observer interpretation and combined speedup flags are false. Retained worker
wall summaries include maximum worker duration, start spread, completion tail
and dispatch-minus-maximum-worker duration. They include preemption, stalls and
observer costs; the failed bound prevents using them as a qualified attribution
of the earlier slowdown. CPU time and a scheduling/register-spill cause are not
established. Stable candidate repeats in this experiment do not prove why a
different earlier run varied.

The host reports AMD EPYC 7763, two cores/four logical processors, AVX2/FMA and
Windows Server 2022 build 20348. It differs from the earlier Xeon 8370C kernel host.
The same CPU name as a prior model-profile run does not prove a matched host.
Values remain synthetic Q8 data at captured model dimensions, not captured
model tensor payloads. No model RTF is measured here.

Keep the production kernel. Any next optimization needs a separately frozen
experiment and actual full-model/native/public output and RTF evidence; neither
the failed 5% microkernel target nor the failed observer bound is reclassified.
Lifecycle, package/offline/coexistence, quality, deferred human word-time review,
physical desktop/fresh-machine and signing gates remain open. Main is unchanged.

## Retention

The 1,407-file receipt has 1,406 manifested payloads, including 1,267 artifact copies,
120 frozen tools, original metadata/logs and executed admission/supervision/
collection/publication sources. Canonical source, pristine archive, objects and
executable remain in the admitted local delivery and are bound by the complete
member map. All four actual saved-receiver stages pass on their first invocation.
