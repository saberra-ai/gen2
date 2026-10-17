# Q8 two-row consumption: exact outputs, failed speedup and observer bounds

The [frozen experiment](nemotron-windows-q8-pair.md) completes construction,
numerical and cleanup checks. All1,056 full phase vectors are byte-exact against
the production Mx4 control across twelve fresh processes. None of the48
four-worker primary comparisons meets the original5% speedup predicate. The
separate observer bound also fails. The candidate is not qualified for promotion.

Producer `259966ceca113f50d7d172fa63269b0bcb17b37a` completed
[run39066832336](https://github.com/saberra-ai/gen2/actions/runs/39066832336),
job117496146656. Workflow success establishes a completed diagnostic, not a
passing timing prediction or SDK. The preregistration and all previous failed
predictions remain unchanged.

The same producer's [feature CI](https://github.com/saberra-ai/gen2/actions/runs/39066832281)
completes all thirteen jobs successfully. This confirms the feature checks,
without changing the diagnostic's failed timing decisions or release status.

## Independent replay

The [1,428-member receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair/collection.json)
and [qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair/collection-qualification.json)
bind artifact12113369532,750,512,476bytes, SHA256
`718c0425723839fd22d9d095c6213f1235181af5fca69fff933de5f9cadd3b65`.
All4,697 archive members,131 tools and3,745 producer repository references pass
independent admission. Canonical source, pristine archive, two actual kernel
objects and executable remain available in the admitted local/GitHub delivery;
the Git receipt retains every raw phase vector, disassembly, text and source proof.

Replay checks the complete3,425-file canonical map and nineteen original patch
reads, exact existing production Mx4 control and new pair source, original
compiler/linker identities and contraction-off flags, all five project inputs,
both actual kernel source/command/object associations and dependency evidence.
Control and candidate are fixed to their respective benchmark entry aliases;
both translation units must occur exactly once. Full guard and benchmark argv,
including executable, output prefix and working directory, are bound.
Collection invokes no compiler, native benchmark, SDK or model.

All95 commands retain their expected integer exits and complete owned-Job
cleanup:59 zero exits and twelve each of73/74/75 for support rejections. The
configure snapshot contains one membership-confirmed `vctip.exe` PID4560,
which is reaped with its Job; other completion snapshots are empty. All36
support rejections, twelve complete65,536-entry half tables and all1,056 finite,
nonzero full vectors independently pass, including padded strides, tails,
input immutability and repeated calls.

All8,448 whole-dispatch timings and9,856 enabled worker intervals remain.
There are180 zero-duration intervals, allowed by the original below-resolution
rule. Disabled processes emit none. Every interval fits its whole dispatch;
worker durations include preemption, stalls and observer costs.

Actual Windows controls pass15 compiler,7 original tile,7 corrected support,
6 geometry,6 worker timing,6 new pair and10 owned-Job tests. The producer's
local suite passed545 tests with seven Windows-only skips. These checks do not
establish model quality, live RTF or desktop release readiness.

## Original timing decisions

The host reports AMD EPYC9V74, two cores/four logical processors, AVX2/FMA,
Windows Server2022 build20348. It differs from the prior production Xeon8573C
and earlier worker experiment's EPYC7763; their observations are not a matched
host comparison.

Each process uses one persistent pool; every shape runs control/pair/pair/control
with one warmup and eight timed calls per phase. Counts below cover all twelve
fixed primary shapes. Every non-primary and one/two-worker observation remains.

| Workers | Process/mode | Speed passes | Control-drift passes | Both pass | Candidate-repeat-drift passes |
| ---: | --- | ---: | ---: | ---: | ---: |
|1|0/off|0|10|0|12|
|1|1/on|0|11|0|10|
|1|2/on|0|11|0|12|
|1|3/off|0|12|0|12|
|2|0/off|1|7|0|11|
|2|1/on|0|6|0|10|
|2|2/on|1|9|1|8|
|2|3/off|0|11|0|9|
|4|0/off|0|12|0|12|
|4|1/on|0|12|0|12|
|4|2/on|0|12|0|12|
|4|3/off|0|11|0|12|

The four-worker speed predicate fails all48 comparisons. Control drift passes
47/48, and candidate-repeat drift passes48/48; neither substitutes for speed.
The table below reports the larger candidate eight-call total divided by the
smaller surrounding control total. The original integer gate requires at most95%.

| Primary `(m,n,k)` |0/off|1/on|2/on|3/off|
| --- | ---: | ---: | ---: | ---: |
|(512,13,2048)|101.693%|104.651%|103.599%|102.949%|
|(512,292,2048)|104.952%|104.925%|105.848%|104.983%|
|(512,535,2048)|104.961%|105.428%|104.560%|105.859%|
|(2048,13,512)|102.906%|105.320%|102.592%|107.738%|
|(2048,292,512)|104.856%|104.578%|104.797%|104.448%|
|(2048,535,512)|104.986%|103.632%|104.897%|104.646%|
|(1536,13,512)|107.712%|101.492%|104.809%|102.826%|
|(1536,292,512)|104.222%|108.676%|104.682%|105.265%|
|(1536,535,512)|105.198%|104.457%|105.146%|104.512%|
|(512,13,512)|106.056%|99.358%|105.945%|104.484%|
|(512,292,512)|104.652%|105.886%|104.501%|105.582%|
|(512,535,512)|104.540%|103.562%|104.477%|104.394%|

The independent observer check passes control drift47/48, overhead46/48 and
both45/48. Three combined failures retain these off/on/on/off totals in ns:

| Shape/phase | Totals | Failure |
| --- | --- | --- |
|5/3|30041100,29758900,29952400,36777100|Control drift|
|10/3|1327800,1459100,1392800,1408100|Observer bound|
|13/3|688600,724800,711100,711900|Observer bound|

Both speedup and bounded-observer flags remain false. No CPU-time or causal
scheduling/register-spill explanation follows from these wall observations.

## Generated loop check

The separate [eight-member static receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair-loop-inspection/collection-qualification.json)
binds the admitted objects/disassemblies, inspected function/loop instructions,
reviewed inspector/publisher and five parser controls. It selects the unique
Q8/Q8/F32 function and innermost backwards conditional loop with eight FMA
instructions. Ambiguous or differently unrolled output requires manual inspection.

| Helper | Loop target/backedge | Loop instructions | FMA | Explicit vector stack operand bytes |
| --- | --- | ---: | ---: | ---: |
|Production `gemmMx4<2>`|0x3d0/0x5a7|95|8|192|
|Candidate `gemmPair2x4`|0x3d0/0x5b2|96|8|256|

Changing source operand lifetimes did not yield fewer static stack operands in
this generated loop. These operand widths are neither measured memory traffic
nor a proven bottleneck or explanation of the failed timing prediction.

The production kernel remains unchanged. This candidate supplies no basis for
a full-model qualification or promotion. Any next change needs a separately
justified experiment, with every failed result retained and all original gates
preserved. Native/public/live-RTF, lifecycle/package/offline/coexistence, quality,
deferred human word-time, desktop/fresh-machine and signing/release requirements
remain open where previously recorded. Main remains unchanged.
