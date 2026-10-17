# Windows batched matrix observer: timing bounds pass, realtime still fails

The [separate batched observer](nemotron-windows-matrix-batch-profile.md) now
passes complete source/compiler/numerical/coverage admission and the original
drift and observer-overhead bounds. All four 60-second full-model histories are
byte-exact. Every native RTF still exceeds one; this is diagnostic evidence, not
a production or public runtime qualification.

Producer `96e5e6bc95eea8e096603323d89ef6fff1b8daa7` completed
[run 39051809615](https://github.com/saberra-ai/gen2/actions/runs/39051809615),
job `117451767146`. The pre-execution specification retains its historical
pending status. The [433-file saved receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile/collection.json)
and [independent collection qualification](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile/collection-qualification.json)
record the actual result; collection did not run a compiler, SDK or model.

## Construction, cleanup and output parity

Delivery admission binds all 122 tools and 2,445 repository references to the
exact producer before local imports. Protocol 564,099 bytes has SHA-256
`04e3b9a4dc3376ed577b7e3bc5df9b314b7d8c7e8350b5115394d51fe70aa225`.
Artifact `12108461445` is 19,675,034 bytes, SHA-256
`79de2f52b6f0d35c500828538de5b12858900a13229250a3a3015e8d75cb1cbd`;
all 370 members and 299 retained evidence payloads match. Native binaries and
pristine archives remain in the cache with their exact retained descriptors.

All 59 commands exit zero with the original observed/reaped statuses preserved,
owned process trees reaped, Jobs closed and no cleanup errors. Before cleanup,
three scoped snapshots each contain one membership-confirmed `vctip.exe`:
SDK build PID 7764, probe build PID 2016, control build PID 5672. The other
56 snapshots are empty. These observations do not describe unrelated processes.
Actual Windows controls pass: 15 compiler, 21 original matrix source/report,
10 new candidate source/compiler/evidence and 10 owned-Job tests.

Independent replay reconstructs all twenty patches from the pristine pinned
source and matches all 3,425 GGML files. Only the CPU C file changes, to the
143,331-byte batching candidate, SHA-256
`2d0ea19dcc51f53ffc376f51ebd53aa5c575adeead0b29cf4bf1993a9ac58000`.
The original compiler/MSVC-probe/linker fingerprints, contraction-off policy,
97 otherwise unchanged compiler commands, source/version/state checks, private
PE/ABI and 66 installed SDK payloads all pass. The runtime identifies itself as
`nemo-speech-asr 0.2.0+gen2-diagnostic-matrix-batch-v1`.

All twelve fresh Windows model-free processes complete: workers 1/2/4, each
off/on/on/off, four graphs repeated three times. All 144 finite/nonzero outputs
are byte-exact per shape; complete dispatch [2,4,5,5], attempts/rejections,
conversion and fusion checks pass. Build-tree GGML import-library bytes were
bound and rechecked on CI; they are not installed/delivered SDK payloads, and
saved collection does not claim to stat those absent Windows files.

The four full-model processes each retain 58 pushes, 6,000×8 probabilities and
27 final turns. Every 192,000-byte history matches the original reference SHA-256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`.
Both enabled traces contain all 84 graphs ×4 workers ×129 matrix roots, one
persistent pool and matching complete graph/fusion/matrix-metadata signatures.
Exact compact-file replay from raw traces passes for all four cases; each
enabled file retains 43,344 worker/matrix records without a duplicate CPU view.

## Timing and actual matrix paths

The host is Windows Server 2022 build 20348, AMD EPYC 7763, two reported cores
and four logical processors, with AVX2/FMA. The earlier unbatched matrix run
used EPYC 9V74. These are different receivers: their times do not establish a
matched batching speedup or prove that per-row writes caused the earlier miss.

| Fresh process | Elapsed seconds | Native RTF |
| --- | ---: | ---: |
| Off0 | 61.420208 | 1.023670133 |
| On1 | 61.011271 | 1.016854517 |
| On2 | 61.022612 | 1.017043533 |
| Off3 | 61.564487 | 1.026074783 |

Control drift 0.2349048% passes the unchanged 10% bound. Both enabled cases pass
the original integer requirement of at most 110% of both surrounding controls.
`bounded_observer_interpretation_supported` is true for this run. Slightly
shorter enabled times do not imply negative overhead or a production speedup.
All four RTFs exceed one; no below-one native/public performance pass follows.

In each enabled case and each worker, 10,836 matrix roots comprise 10,584
converted Q8 SGEMM, 190 direct SGEMM and 62 vector-dot fallback roots. No root
takes the Hadamard/IQ/extra/fused bypass routes. Matrix kernel wall accounts for
69.74–70.10% of each worker's summed operation kernel wall; UNARY accounts for
15.45–16.04%, FLASH_ATTN_EXT for 11.05–11.40%. Converted Q8 roots account for
98.61–98.63% of that worker's matrix kernel wall. These are observed enclosing
walls, including internal synchronization and scheduling, not pure arithmetic
times. Workers overlap; never add their wall intervals.

The four repeated Q8 A shapes, in GGML dimension order, are [2048,512,1,1],
[512,2048,1,1], [512,1536,1,1] and [512,512,1,1], each 2,604 roots per worker.
Their F32 B inputs span 13–535 columns with 63 distinct values. Converted
A [1024,512,1,1] and [512,192,1,1] each add 84 roots per worker. The complete
type/shape/stride/route groups and stage durations remain in the qualification.
For worker 0, cases 1/2 respectively record matrix kernel wall 41.554898 /
41.425215 seconds, converted second-SGEMM wall 40.385692 / 40.178338 seconds,
conversion 0.528720 / 0.603388 seconds and converted inner wait 0.061541 /
0.059318 seconds. Keep these nested intervals separate.

The bounded observations now support selecting a separately frozen Q8 kernel
control using captured model geometries. Preserve the earlier slower 2×2
candidate, test one new change with exact output/guard checks and surrounding
controls, then require full-model and production/native/public/lifecycle
qualification before packaging. No synthetic kernel result can replace those
gates. Original DER/attribution, human-word, desktop/fresh-machine and signing
requirements remain open. Human annotation/review remains documented and
deferred; main remains unchanged and unqualified.
