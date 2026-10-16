# N1–N3 development qualification

These receipts establish working public Rust CPU execution, native-reference
parity, bounded streaming, admission/lifetime behavior, real mixed-model
execution and provisioning recovery. They do not establish a supported desktop
release or complete N4/N5. See `receipt.json` for hashes and remaining gates.

The 30-minute and one-hour inputs repeat NVIDIA's licensed 60-second AMI
EN2002d fixture through one native session. They establish continuity and
probability compaction, not accuracy on held-out long meetings. The development
executable was preserved by SHA-256; it predates review changes to allocation
capacity rejection and missing-symbol diagnostics. A clean committed release
build rerun remains an explicit N5 requirement.

| Run | Audio | Wall | RTF | Peak RSS | Final frames | Retained probability start / rows |
| --- | --- | --- | --- | --- | --- | --- |
| 30 minutes | 1800 s | 350.53 s | 0.1947 | 236.70 MiB | 180001 | 117176 / 62825 |
| One hour | 3600 s | 777.77 s | 0.2160 | 237.36 MiB | 360001 | 296627 / 63374 |

Both streams finished and released their reservations. The 1024 MiB runtime
reservation covers observed peaks with substantial margin. Segment output is
bounded to 100000 turns and session input to one hour; queue capacity and
transferred allocation capacity are independently bounded. Development builds
and tests ran alongside the experiments, so these are not isolated performance
comparisons.

Seven explicitly executed real-model tests passed, including parity for every
turn/probability, A–B–A sessions, 100 actual reloads, full-queue cancellation,
dropped streams, silence, fractional tail, idle deadline, repeated finish,
uneven/one-sample splits and ABI rejection. Four separate mixed-model processes
passed: both load orders with the Laya smoke graph, then both orders with full
English Laya weights alongside real Bark, chat and diarization. Exact auxiliary
budgets forced chat eviction/rejected restore; shutdown permitted restoration.

The real downloader was killed after observing an incomplete GGUF with
88958883 bytes. Three concurrent subsequent processes recovered the same pinned
entry. Corrupt notices were quarantined/repaired; abandoned staging was ignored;
a warm verified cache worked with network access disabled through offline/proxy
settings. The separate no-`hf` warm-cache test also passed.

The ordinary combined default + ASR/Laya/diarization/Metal run passed 1348
tests with 43 environment-dependent tests ignored. Those skipped tests are not
counted as real-model evidence. Minimal-feature tests and clippy passed; vendor
documentation/deprecation warnings are recorded in logs. CI platform jobs have
been added but are not represented as executed remote-platform receipts.

## Standards review

Review scope: `9e125b1...76938fb`. No hard coding-standard violations or
substantive baseline smells were found.

**[P2] Initialize llama.cpp for the mixed-feature CI lane.** Checkout omitted
submodules while enabling `backend-llamacpp`; its build requires the pinned
llama.cpp gitlink. Fresh runners would fail before checking compatibility.
Resolution: recursive submodule checkout was added in `a66c88e`.

## Spec review

**[P2] Bound transferred allocation capacity.** `try_push` bounded `Vec::len()`
but retained the caller's allocation; a truncated large recording could bypass
the PCM byte bound. `Audio::mono_16khz` had the same retained-capacity issue.
This violated the roadmap's requirement to bound queued PCM bytes. Resolution:
oversized stream allocations are rejected with their PCM returned; recording
input sheds spare capacity. The regression failed before the fix and passed
afterward, along with the final seven-test real-model run.

**[P2] Initialize native submodule for feature CI.** The mixed feature matrix
required a checkout of the native gitlink. Resolution: recursive checkout as
above. No additional lifecycle, result-validation, provisioning or scope-creep
defect was confirmed.

Review totals: Standards 1 finding, worst P2; Spec 2 findings, worst P2. All
confirmed findings were resolved. Final release review remains part of N5.

## Initial alignment investigation

`n4-alignment-receipt.json` uses AMI manual annotations 1.6.2, CC-BY-4.0, from
[the AMI corpus maintainers](https://groups.inf.ed.ac.uk/ami/download/). The
archive hash and selected XML filenames are included. The audio source is the
same AMI EN2002d window, 2132.5–2192.5 seconds; NVIDIA's fixture provenance is
locked by the N0 artifact manifest. `n4-bark-ami.json` is the cleaned Bark output.

Grouping whole measured token intervals yielded 108 matched words from 147
reference / 117 predicted words. Chronological merged WER was 27.21%, including
31 deletions. Matched start error was median 80 ms / p95 210 ms; matched end error
was median 100 ms / p95 1150 ms. Next-token ends can span silence. These are
diagnostic timing/cleanup measurements, not validated word attribution or
speaker-attributed WER. No proportional text timing or N4 completion is claimed.
