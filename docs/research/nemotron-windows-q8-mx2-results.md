# Packed-B-scale 2×2: exact outputs, failed performance prediction

The [frozen comparison](nemotron-windows-q8-mx2.md) completes construction,
numerical and cleanup checks. All1,056 full phase vectors match the production
Mx4 control. None of48 required four-worker primary comparisons meets the
original5% speedup predicate; the separate observer gate also fails. Production
remains unchanged.

Producer `ebdf051a1be03df766a28e03e901e7dc3513d500` completes
[run39075328381](https://github.com/saberra-ai/gen2/actions/runs/39075328381),
job117521191434. Its [feature CI](https://github.com/saberra-ai/gen2/actions/runs/39075328456)
also completes all thirteen jobs successfully. Workflow success does not turn
failed performance predictions into passing qualification.

## Independent admission and numerical replay

The [1,456-member receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2/collection.json)
and [qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2/collection-qualification.json)
bind artifact12116809011,750,527,618bytes, SHA256
`d7c36f13374ad03368ba4876cd5eebb9741e678c7e1b56e45feaa731b39489f4`.
All4,697 delivered members,149 tools and3,802 repository references pass
independent admission. Complete raw vectors, half tables, source, disassembly
and text evidence are retained. Canonical/pristine source, other objects and
the executable remain in the admitted local delivery with its full member map.

Replay verifies all3,425 canonical source files, nineteen original patch reads,
the exact production control and private packed-B-scale clone, compiler/linker
identities and unchanged flags. Five project inputs, both uniquely present
kernel translation units, fixed aliases/guards, fourteen compiler inputs,
source/command/object associations, dependencies and executable/case argv/cwd
remain bound.

All95 commands have the expected integer exits and complete owned-Job cleanup:
59 zero exits and twelve each of73/74/75. The configure completion snapshot
contains one membership-confirmed `vctip.exe`, PID2216, reaped with its closed
Job. Other completion snapshots are empty. All36 support rejections, twelve
complete65,536-entry half tables and1,056 complete phase vectors pass, including
tails, padded guards, input immutability and repeated execution.

All8,448 whole-dispatch timings and9,856 enabled worker intervals remain.
The54 zero-duration intervals meet the original below-resolution rule; disabled
processes emit none. Intervals fit their whole dispatches and include preemption,
stalls and observer costs.

Actual Windows controls pass15 compiler,7 original tile,7 corrected support,
6 geometry,6 worker timing,6 packed-B-scale and10 owned-Job tests. Local
preparation passes576 tool tests with seven Windows-only skips. Six pure
receiver controls pass. A publisher identity-binding gap was corrected before
invocation: four actual publication-seam tests reproduce three failures in the
uninvoked V1 and pass in invoked V2. V2 preserves first parsed/admitted identities
through copying and final rehash; all nineteen pre-execution file pins are
retained. Saved-data admission, replay, publication and inspection invoke no
compiler, SDK or model.

## Original timing decisions

The host reports AMD EPYC7763, two cores/four logical processors, AVX2/FMA and
Windows Server2022 build20348. This differs from the preceding A-scale EPYC9V74
and production candidate's Xeon8573C. A CPU name shared with another experiment
does not establish a matched physical host. Only this run's paired observations
are compared.

Each process uses one persistent pool and all22 shapes. Each shape executes
control/mx2/mx2/control, one warmup and eight timed calls per phase. All twelve
original primary shapes remain required; one/two-worker or non-primary passes
cannot replace a failed required comparison.

| Workers | Process/mode | Speed passes | Control-drift passes | Both pass | Candidate-repeat-drift passes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 1/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 2/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 1 | 3/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 2 | 0/off | 0/12 | 5/12 | 0/12 | 8/12 |
| 2 | 1/on | 1/12 | 7/12 | 1/12 | 9/12 |
| 2 | 2/on | 0/12 | 7/12 | 0/12 | 7/12 |
| 2 | 3/off | 0/12 | 7/12 | 0/12 | 9/12 |
| 4 | 0/off | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 1/on | 0/12 | 12/12 | 0/12 | 12/12 |
| 4 | 2/on | 0/12 | 12/12 | 0/12 | 11/12 |
| 4 | 3/off | 0/12 | 12/12 | 0/12 | 12/12 |

Four-worker ratios are100 times the larger candidate eight-call total divided
by the smaller surrounding control total. Each original speed predicate requires
a ratio at most95%, with the separate control-drift bound. Rounded ratios do
not replace exact integer decisions.

| Shape `(m,n,k)` | Off0 % | On1 % | On2 % | Off3 % |
| --- | ---: | ---: | ---: | ---: |
| `(512,13,2048)` | 110.077 | 109.204 | 108.274 | 109.027 |
| `(512,292,2048)` | 113.139 | 112.076 | 111.176 | 111.080 |
| `(512,535,2048)` | 110.751 | 110.552 | 110.691 | 109.657 |
| `(2048,13,512)` | 110.303 | 105.842 | 298.134 | 104.815 |
| `(2048,292,512)` | 118.603 | 115.748 | 107.719 | 108.012 |
| `(2048,535,512)` | 107.883 | 108.697 | 106.939 | 109.532 |
| `(1536,13,512)` | 106.967 | 105.165 | 104.832 | 107.680 |
| `(1536,292,512)` | 109.584 | 106.586 | 109.104 | 109.363 |
| `(1536,535,512)` | 107.725 | 106.259 | 106.567 | 106.313 |
| `(512,13,512)` | 103.306 | 102.311 | 101.014 | 104.954 |
| `(512,292,512)` | 112.140 | 111.773 | 112.185 | 109.386 |
| `(512,535,512)` | 108.028 | 108.707 | 107.505 | 108.112 |

The separate observer passes drift44/48, overhead44/48 and both41/48.
All failed rows remain, with exact off/on/on/off totals:

| Shape index / phase | Off0 ns | On1 ns | On2 ns | Off3 ns | Failed bound |
| --- | ---: | ---: | ---: | ---: | --- |
| 7 / 0 | 2661400 | 2373800 | 2368600 | 2349900 | Drift |
| 7 / 1 | 2935600 | 2434800 | 2501700 | 2451100 | Drift |
| 7 / 2 | 2752900 | 2462000 | 6948900 | 2436300 | Drift, Overhead |
| 7 / 3 | 2794000 | 2326100 | 2330800 | 2338500 | Drift |
| 8 / 2 | 41487200 | 44657400 | 42352800 | 41757800 | Overhead |
| 13 / 0 | 1071500 | 1076000 | 1085700 | 1021400 | Overhead |
| 14 / 1 | 10926200 | 11472400 | 11153800 | 10897300 | Overhead |

All three flags remain false: `same_process_kernel_prediction_supported`,
`bounded_observer_interpretation_supported` and `kernel_speedup_prediction_supported`.

## Saved generated-loop evidence and limits

The separate [19-member static/CI receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2-loop-inspection/collection.json)
retains both actual kernel objects/raw disassemblies, the admitted parent parser,
input bindings and actual terminal feature-CI metadata. The parser is
AST-identical to the row inspector. Four wrong helper/FMA selections reject;
three pure static-publication seam controls pass.

Select the unique Q8/Q8/F32 helper and its unique innermost backward conditional
loop containing eight control or four candidate FMAs. Count explicit XMM/YMM
memory operands relative to RSP using their16/32byte widths, excluding
prologue/epilogue/reduction. The control has95 instructions/eight FMAs and
192bytes of explicit vector stack operand widths per eight outputs. Candidate
`gemmMx2<2>` has45 instructions/four FMAs and zero such operands per four
outputs. Two candidate iterations normalize to90 instructions/eight FMAs/zero
bytes per eight outputs. The separately frozen static zero-stack prediction
passes; the original speed prediction still fails.

These static counts exclude tile setup, reduction, scheduling and other function
regions. They do not establish a stack-free function, measured memory traffic,
CPU time, a spill bottleneck or the cause of the failed timing. This distinct
packed-B-scale helper does not repeat or relabel the earlier generic scalar-scale
`gemm<2,2>` experiment.

Synthesized matrices at captured model geometries supply no full-model RTF,
SDK/public/lifecycle, package/offline/coexistence, GUI/microphone, fresh-machine,
platform or signing/release qualification. Preserve this failed candidate;
an unchanged retry is not an implementation fix. The original AMI accuracy
failures and user-deferred human word annotation/review remain open. No main
promotion is established.
