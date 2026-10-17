# Windows batched matrix observer experiment

Status: implemented; actual Windows SDK/model execution pending. The
[local batched-emission control](nemotron-matrix-batch-results.md) passes all
twelve processes, 144 original-baseline-identical outputs and 168 enabled worker
records. The [original full-model matrix observer](nemotron-windows-matrix-results.md)
has exact outputs and complete paths but fails the original overhead bound.

The ranked hypotheses remain per-row stderr writes, row formatting, then timers
and contention within matrix operations. This experiment changes only emission
after the final barrier. It tests whether batching is sufficient to pass the
original observer bound; it does not presume a cause or a speedup. Preserve
every observation and the original failed result.

## Exact change and inherited requirements

Use distinct runtime identity `nemo-speech-asr 0.2.0+gen2-diagnostic-matrix-batch-v1`.
Derive worker/verifier/compiler adapters by exact import substitutions from the
unchanged CPU-operation producer. Retain the original matrix decoder and compact
profile writer. All requirements of the
[original Windows matrix experiment](nemotron-windows-matrix-profile.md) remain
mandatory: pinned source/lock/compiler/MSVC probe/linker/PE/ABI identities,
portable AVX2/FMA, contraction off, all 97 otherwise unchanged compiler commands,
source and installed SDK checks, scoped Job cleanup and fail-closed evidence.

After the canonical nineteen patches, append one batched observer patch; replace
the earlier observer patch rather than stacking it. The resulting CPU C file is
143,331 bytes, SHA-256
`2d0ea19dcc51f53ffc376f51ebd53aa5c575adeead0b29cf4bf1993a9ac58000`.
Its patch SHA-256 is
`15984f440c563ebb3c83d8f0dda99412141d321ffab41e92545aa5586a959a62`;
the resulting series SHA-256 is
`20bfe382c65fe1c222b7f43b1554f885fbc347260429dbfe92150e3e3b143f89`.
Only that CPU file changes in the canonical 3,425-file map. Require exact
agreement with the locally compiled candidate. Captures, arithmetic, timers,
task assignment, fusion and barriers stay unchanged. Batched storage is bounded
to 512 KiB of temporary heap per worker. Missing/truncated/error records reject.

Freeze this specification, workflow, patch, full local import/test closure,
original compiler/CPU/matrix protocols and their complete referenced inputs,
the actual original Windows matrix receipt and the local batch receipt. Perform
an independent source-only twenty-patch materialization before Windows launch;
compare all source descriptors. Retain earlier failed preparation attempts.

## Execution and decisions

Require actual Windows compiler controls, original matrix source/report controls,
new candidate compiler/source/evidence controls and scoped Job negative controls.
After SDK admission, run the original twelve fresh Windows graph processes
(workers 1/2/4, off/on/on/off), four shapes three times each, before model
acquisition. Require all 144 finite/nonzero vectors byte-exact per shape and
complete route [2,4,5,5], attempts/rejections, conversion and fusion evidence.
Bind their binary, sources/import libraries, outputs and logs through final
completion. The local executable C-emitter fault tests remain Mac evidence;
actual Windows behavior must be established by the compiled SDK controls.

Then use the same pinned 60-second PCM/model and four fresh full-model processes
off/on/on/off, CPU device -1, four-worker pool, 58 pushes and 6,000×8 full raw
probabilities. All four histories must be byte-exact to the original independent
raw reference and meet the original turn/state/numerical checks. Both enabled
traces must contain all 84 graphs ×4 workers ×129 matrix roots, one persistent
pool, matching complete graph/fusion/matrix-metadata signatures, all operation
records and original raw logs. Retain each full compact profile once and replay
its exact bytes from raw logs; recheck prior evidence between cases and finally.

Keep each worker's timings separate. The original integer decisions remain:
control drift at most 10%, and both enabled times at most 110% of both surrounding
controls. No averaging, dropped observation, changed bound or narrowed coverage
is permitted. Diagnostic completion is separate from those timing decisions.
New runner hardware is not a matched comparison with older CI hardware. Passing
observer bounds still supplies no production/public/lifecycle/package pass.
Quality, attribution, desktop/fresh-machine and signing gates remain open;
human annotation/review remains deferred and main unqualified.
