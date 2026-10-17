# Windows CPU worker-budget result

The [corrected diagnostic run39025306014](https://github.com/saberra-ai/gen2/actions/runs/39025306014)
completes all four preregistered cases at source
`1745243b221368855381ea36879ccdc1bb731528`. Its
[368-member saved-data collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/collection-qualification.json)
verifies the actual artifact, source/compiler inputs and all complete outputs.
This is native diagnostic evidence only; release, package, public API and
lifecycle qualification remain false.

| Fixed order | Configured/attached workers | Native elapsed, 60s input | Native RTF |
| --- | --- | --- | --- |
| 1 | 4 / 4 | 62.840321s | 1.0473386833 |
| 2 | 2 / 2 | 77.770717s | 1.2961786167 |
| 3 | 1 / 1 | 135.615090s | 2.2602515000 |
| 4 | 4 / 4 | 63.227234s | 1.0537872333 |

All cases report hardware concurrency4 and run in fresh processes on one
Windows2022 job. CIM reports **AMD EPYC7763**, two cores/four logical processors,
for all three builds. The earlier production failure used an **EPYC9V45** host;
these are not matched cross-run speed measurements. Interpret the worker
comparison within this fixed same-host sequence.

All four6000x8 float32 histories are byte-exact, SHA256
`6b08c607371bab2b7965faebe88d8ee31ae39c41c377043d137cbbf424f3af7d`;
all27 final turns match with zero boundary error. The original1e-6 limits,
960000-sample PCM digest, Q8 model and streaming geometry are unchanged.
Configured/effective cap and successful attached-pool size are independently
logged at their respective native source sites. This is not an OS thread census.

Relative to the first four-worker control, two workers are about24% slower and
one is more than twice as slow. The two four-worker controls differ by about0.6%.
This rejects the proposed smaller-worker remedy on this host and supports the
prediction that this workload benefits from four workers. It does not identify
which kernels or compiler decisions dominate, or rule out scheduling effects on
other hosts. Every observed RTF remains above1; no averaging or tolerance turns
this diagnostic into a passing real-time result. Keep the production worker
policy unchanged. The next performance investigation needs to examine computation
or compiler/kernel behavior while preserving the portable CPU baseline and
the original streaming/quality requirements.

Collection checks include all516 artifact members,53 tool and268 repository
source pins against the exact producer Git commit, three66-file SDK inventories,
three strict `diar_state` CTest passes, private PE/import replay, exact derived
probe/version identities and all121 successful child commands. The collector
independently constructs the pinned57,466,880-byte GGML archive, verifies the
canonical19-patch order and bytes, and replays all3425 materialized files plus
all97 actual compiler sources for each SDK. Every copied artifact-derived
receipt file is rebound to its admitted ZIP descriptor before publication.

MSVC records `GGML_FMA` and `GGML_F16C` CLI cache entries as `UNINITIALIZED`,
rather than Boolean options. The first local collector stopped on its incorrect
Boolean-type assumption before publishing a receipt. The corrected collector
preserves that observation and requires the actual `/arch:AVX2`, `GGML_AVX2`,
`GGML_FMA` and `GGML_F16C` definitions on every GGML CPU compiler entry. This
changes a saved-data schema check, not native source, inference or acceptance.
No model or delivered binary is executed during collection. The45,429,953-byte
ZIP, SDK binaries and large tar files stay local; scoped metadata, sources,
logs and all four complete raw arrays are retained in Git.

The initial unrelated Git-LFS checkout failure remains preserved separately in
the [v2 correction record](nemotron-windows-worker-diagnostic-v2.md). No Windows
bounded package/receiver or desktop release is qualified by either diagnostic.
