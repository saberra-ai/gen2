# Windows production Q8 candidate: functional pass, realtime failure

The [frozen production producer](nemotron-windows-q8-candidate-producer.md) at
`fee2177adcd7b75b488c67bea5a593ea17df65de` completed SDK construction, all four
native/public histories and all seven original lifecycle tests in
[run39063644421](https://github.com/saberra-ai/gen2/actions/runs/39063644421),
job117486629205. The job correctly failed the original strict realtime gate.
No SDK or package promotion is established.

The [321-member collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate/collection.json)
retains raw graph/model arrays, text, frozen producer sources and receiver
scripts. Its [independent replay](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate/collection-qualification.json)
binds artifact12113415778: 21,683,920 bytes, SHA256
`30e4a56c0e56996a9558a1b1fdb198c42435490bdeefd7fae8c0648a11a06e18`.
All368 delivered files were independently hashed before extraction. All105
tool and818 repository references agree with the recorded Git producer and
current frozen inputs. The receivers execute no Windows compiler, SDK or model.

## Actual observations

The host is a Xeon Platinum8573C, two cores/four logical processors, AVX2/FMA,
Windows Server2022 build20348. It differs from the earlier diagnostic EPYC7763;
the earlier diagnostic candidate's below-one native timings do not establish
production realtime behavior on this host.

| Fresh process | Elapsed microseconds | RTF on60 seconds | Original RTF<1 |
| --- | ---: | ---: | --- |
| Native0 | 63,782,456 | 1.0630409333 | fail |
| Public1 | 62,791,514 | 1.0465252333 | fail |
| Public2 | 63,739,538 | 1.0623256333 | fail |
| Native3 | 63,348,272 | 1.0558045333 | fail |

Both complete native/public comparisons pass against the same frozen reference.
Each history retains all6,000 rows/eight channels and27 turns; every192,000-byte
probability array has SHA256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
The two native processes each perform58 pushes. All seven actual lifecycle tests
pass, including offline reuse, cancellation/admission failures, streaming tails,
reload, dropped streams and parent/child lifetime behavior.

All61 controlled commands exit integer zero, preserve their observed exit and
reap their owned Windows Job trees with closed handles. Four completion snapshots
contain an owned `vctip.exe`; their confirmed membership and cleanup are retained.
No empty-background claim is made. The producer's final failure is the explicit
qualification decision after these successful commands.

The receiver independently rehashes all97 compiler objects, three retained graph
import libraries,36 complete graph vectors,66 SDK files and the public example/
test executable copies. Compiler/source associations, actual contraction-off
arguments, private PE identities, native probe derivation, graph links and all
completed output bindings replay. Independent source-only preparation verifies
the bounded native inference and3,425-file GGML map, with only the declared SGEMM
dispatch change. Compiler/linker identities remain CI observations; those tools
are not delivered or executed by the receiver.

## Consequence

Functional/lifecycle and complete retention pass; realtime, candidate SDK,
package and release qualification remain false. The separate original5% speedup
prediction remains failed. This run supplies no causal host, scheduling or kernel
explanation. Retain these failures; do not select another identical timing run
or weaken the below-one requirement to obtain a pass. A subsequent performance
change needs an independently justified, separately frozen experiment and full
native/public requalification.

The [package payload receiver](nemotron-windows-q8-package-admission.md) now also
replays this actual failed delivery and rejects it at the strict package gate.
Package construction/offline/coexistence/desktop qualification and original
quality, deferred human word-time, Pnola visual/microphone/fresh-machine and
signing/release requirements remain open. Main remains unchanged.
