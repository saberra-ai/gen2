# Q8 A-scale conversion: exact outputs, failed performance prediction

The [frozen comparison](nemotron-windows-q8-scale.md) completes construction,
numerical and cleanup checks. All1,056 full phase vectors match the production
Mx4 control. The actual candidate broadcast matches the independent binary16
reference for every finite value and infinity, with preserved NaN classification
and four identical lanes. None of48 required four-worker primary comparisons
meets the original5% speedup predicate; the observer gate also fails.

Producer `4c72846674d373915fec72fbb0fccbc5a8a1e068` completes
[run39072824849](https://github.com/saberra-ai/gen2/actions/runs/39072824849),
job117513788511. The producer's
[feature CI](https://github.com/saberra-ai/gen2/actions/runs/39072824905)
also completes all thirteen jobs successfully. Workflow success does not turn
the failed timing predictions into passing qualification.

## Independent admission and numerical replay

The [1,460-member receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-scale/collection.json)
and [qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-scale/collection-qualification.json)
bind artifact12115742532,751,443,751bytes, SHA256
`f68babf1649d0b8df7ec1c8e84355e9b17346abfd730da78cb67086ffa7eb077`.
All4,709 delivered members,150 tools and3,763 repository references pass
independent admission. Complete raw vectors, half/broadcast tables, source,
disassembly and text evidence are retained. Canonical/pristine source, remaining
objects and executable remain in the admitted delivery with its full member map.

Replay verifies the3,425-file canonical source, nineteen original patch reads,
exact production control and private scale-conversion clone, compiler/linker
identities and original flags. Five project inputs, both uniquely present
kernel translation units, fixed aliases/guards, complete source/command/object
associations, dependencies and executable/case argv/cwd remain bound.

All95 commands have the expected integer exits and complete owned-Job cleanup:
59 zero exits and twelve each of73/74/75. The configure completion snapshot has
one membership-confirmed `vctip.exe`, PID5284, reaped with its closed Job.
Other completion snapshots are empty. All36 support rejections, twelve complete
original65,536-entry half tables and1,056 complete phase vectors pass, including
tails, padded guards, input immutability and repeated execution.

Each of twelve processes also retains1,048,576bytes from the actual candidate
broadcast helper:65,536 inputs times four binary32 lanes. All twelve files
match SHA256`023ca9cc096f7eb23275938e4f4c8fd70bb2fe9bfbc9a7b0bc1e09551d45017f`.
Independent replay checks63,488 finite values per lane bit-for-bit, including
signed zeros and subnormals, both infinities, NaN classification and identical
lane bytes. It does not claim lookup/hardware NaN payload equivalence.

All8,448 whole-dispatch timings and9,856 enabled worker intervals remain.
The156 zero-duration intervals meet the original below-resolution rule;
disabled processes emit none. Intervals fit their whole dispatches and include
preemption, stalls and observer costs.

Actual Windows controls pass15 compiler,7 original tile,7 corrected support,
6 geometry,6 worker timing,7 A-scale and10 owned-Job tests. Local preparation
passes570 tool tests with seven Windows-only skips. Eight pure receiver controls
pass. A stale receiver metadata count was corrected before artifact admission;
the invoked V3 receivers record seven A-scale tests. The incomplete V2 preparer
stopped at an exact whitespace assertion before any artifact action. Saved-data
admission, replay, publication and inspection invoke no compiler, SDK or model.

## Original timing decisions

The host reports AMD EPYC9V74, two cores/four logical processors, AVX2/FMA,
Windows Server2022 build20348. This differs from the preceding row's EPYC7763
and production candidate's Xeon8573C. A CPU name shared with another experiment
does not establish a matched physical host. Only this run's paired observations
are compared.

Each process uses one persistent pool and all22 shapes. Each shape executes
control/scale/scale/control, one warmup and eight timed calls per phase.
The counts below cover all twelve original primary shapes.

| Workers | Process/mode | Speed passes | Control-drift passes | Both pass | Candidate-repeat-drift passes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 1/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 2/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 3/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 2 | 0/off | 0/12 | 5/12 | 0/12 | 9/12 |
| 2 | 1/on | 0/12 | 6/12 | 0/12 | 10/12 |
| 2 | 2/on | 0/12 | 8/12 | 0/12 | 9/12 |
| 2 | 3/off | 1/12 | 5/12 | 1/12 | 10/12 |
| 4 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 1/on | 0/12 | 12/12 | 0/12 | 11/12 |
| 4 | 2/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 3/off | 0/12 | 12/12 | 0/12 | 11/12 |

Four-worker ratios are100 times the larger candidate eight-call total divided
by the smaller surrounding control total. The original speed predicate requires
each ratio at most95%, with the separate drift requirement. Rounded ratios
do not replace the exact integer decisions. One/two-worker or non-primary
results cannot replace a failed required comparison.

| Shape `(m,n,k)` | Off0 % | On1 % | On2 % | Off3 % |
| --- | ---: | ---: | ---: | ---: |
| `(512,13,2048)` | 105.132 | 103.326 | 101.755 | 104.101 |
| `(512,292,2048)` | 103.508 | 104.794 | 104.640 | 103.449 |
| `(512,535,2048)` | 104.288 | 103.451 | 104.449 | 104.264 |
| `(2048,13,512)` | 104.016 | 105.953 | 104.436 | 106.386 |
| `(2048,292,512)` | 105.734 | 106.238 | 105.031 | 105.399 |
| `(2048,535,512)` | 110.634 | 106.819 | 105.514 | 105.247 |
| `(1536,13,512)` | 108.077 | 124.500 | 102.735 | 124.216 |
| `(1536,292,512)` | 106.899 | 106.587 | 105.937 | 106.317 |
| `(1536,535,512)` | 105.157 | 104.897 | 106.824 | 105.309 |
| `(512,13,512)` | 100.720 | 101.161 | 105.477 | 103.502 |
| `(512,292,512)` | 106.793 | 104.930 | 105.552 | 106.102 |
| `(512,535,512)` | 105.435 | 104.923 | 105.445 | 105.222 |

The separate observer comparisons pass drift46/48, overhead44/48 and both44/48.
All failing rows remain, with exact off/on/on/off totals:

| Shape index / phase | Off0 ns | On1 ns | On2 ns | Off3 ns | Failed bound |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 / 0 | 1718500 | 1719800 | 1815700 | 1714600 | Overhead |
| 10 / 0 | 1397100 | 1552700 | 1448000 | 1546700 | Drift, overhead |
| 10 / 1 | 1477300 | 1782100 | 1447900 | 1762500 | Drift, overhead |
| 12 / 0 | 42442000 | 44040100 | 41958700 | 41832900 | Overhead |

All three flags remain false: `same_process_kernel_prediction_supported`,
`bounded_observer_interpretation_supported` and `kernel_speedup_prediction_supported`.

## Saved generated-loop evidence and limits

The separate [17-member static/CI receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-scale-loop-inspection/collection.json)
retains both kernel objects/raw disassemblies, the unchanged parent loop-parser
algorithm, admitted input bindings and actual terminal feature-CI metadata.
Select the unique Q8/Q8/F32 helper signature and its unique innermost backward
conditional loop containing eight FMAs. Count explicit XMM/YMM memory operands
relative to RSP using their16/32byte widths, excluding prologue/epilogue/reduction.

| Helper | Loop bounds | Loop instructions | FMAs | Vector stack loads/stores | Sum of operand widths |
| --- | --- | ---: | ---: | --- | ---: |
| Production `gemmMx4<2>` | `0x3d0`–`0x5a7` | 95 | 8 | 3 loads / 3 stores, all32bytes | 192bytes |
| Candidate `gemmScaleMx4<2>` | `0x3c0`–`0x5b4` | 100 | 8 | 4 loads / 4 stores, all32bytes | 256bytes |

This contradicts a static stack-access reduction prediction; it establishes no
measured traffic, CPU time, spill bottleneck or cause of the failed timing.
The corrected static publisher binds the exact parsed/admitted bytes before
copying and rechecks those identities afterward. Three pure publication-seam
controls pass; the original uninvoked publisher reproduces two mutation failures.

These are synthesized matrices at captured model geometries. No full-model RTF,
SDK/public/lifecycle, package/offline/coexistence, GUI/microphone, fresh-machine,
platform or signing/release qualification follows. Production remains unchanged.
The original AMI accuracy failures and user-deferred human word review remain.
