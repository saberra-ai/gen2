# Endpoint-paced public stream timing

N5 requires actual label delivery and finalization measurements. The
[saved app-frontier analysis](nemotron-live-frontier-freshness.md) lacks
intermediate turns and stop/final timestamps. Its roughly5.2s update cadence is
consistent with the explicit five-second input coalescing in
`src/diarization/nemo.rs` and Pnola's1.04s adapter blocks. No policy change follows
from those observations.

The new `nemotron_paced_stream` qualification example consumes an exact80.04s
mono16k PCM16 WAV through public Gen2 APIs. Each320-sample block becomes available
at its20ms endpoint on a clock started after model load/session opening. It does
not intentionally feed future PCM. It records all4002 input endpoints, readiness,
successful admission times and Busy retries. The latest coalesced observations
retain actual turns, UUID/revision, accepted samples, native output frontier and
cumulative native compute time. Only final output requests raw probabilities.

The helper records finish request/return and final snapshot observation times,
verifies repeated finish returns the same Arc, and checks reservation release.
It retains at most17 observed snapshots. The final public revision is17; revision16
may be replaced by final output before polling observes it. That missing
observation is reported, never given an invented timestamp. Intermediate
observations must otherwise be consecutive. The public coalescing policy,
native geometry, thresholds and bounded queues remain unchanged.
Returned errors preserve a partial JSON with `complete:false`, the error,
available input/turn observations and load/open/finish checkpoints. A model-free
Rust regression checks that serialization; the constructor runs it under the
same model-denying sandbox before declaring construction complete. The final
completeness check reuses the recorded observation, so every snapshot lookup is
included in the counter.

The [model-free construction protocol](../../tools/nemotron-diarization/paced-stream-construction.protocol.json)
binds261 Gen2 source/coefficient/manifest files,10 tools and11 compiler/config
files. Cargo builds offline/locked to a distinct helper artifact. Global model
format reads and network access are denied, with only the exact pinned built-in
Mel coefficient table admitted as source. Cargo's dependency cache is reused;
this is not a hermetic third-party/toolchain reconstruction or a native SDK build.
The3GiB/1GiB/256MiB/600s construction protections do not relax the separate
5GiB/1GiB/256MiB/600s neural-execution protections.

The report separates the observations that can actually be measured:

- Ready minus nominal endpoint: scheduling overrun, which also includes earlier
  caller work; it is not a pure OS scheduling metric.
- Accepted minus ready: normalization/hashing/admission wall time, including Busy
  waits; it is not isolated native queue residence.
- Native compute: cumulative synchronous native-call wall time, which overlaps
  the paced producer clock and must not be added to it.
- Finish return minus request: public drain/flush/result-delivery delay.
- Final observation minus finish return: caller observation delay, including the
  repeated-finish check and snapshot lookup.
- Intermediate observed clock minus frontier: provisional output freshness,
  distinct from an acoustic word/speaker onset or a stable-prefix guarantee.

No pure queue/lookahead/publication decomposition, actual GUI/IPC render delay,
physical microphone behavior, or first correct acoustic-label latency is claimed.
The trial will reuse the existing qualified pooled CPU package and saved independent
public-C-ABI80.04s reference, bind exact actual PCM and all8004 output rows, and
retain every clock/turn observation. A separate execution protocol is required
after construction. No construction or inference has run at this preregistration.
The original quality, word-gold, fresh-machine, signing and release gates remain.

## Actual construction and runtime admission

The [21-member construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-paced-public-construction/collection-qualification.json)
records all five commands passing at `e658b0cd`, including the exact Rust
error-evidence regression (one passed, zero filtered out). Cargo's dependency
record identifies142 used Gen2 files, all covered by the261 source pins.
The1,026,560-byte helper has SHA256
`4405446b59a1a1c780144ed3b2532ead7af59a795270fe54131b6ffa3ed3303c`.
No model or native helper inference occurs during construction; all247 Python
tests also pass. Payloads stay outside Git.

The separately frozen [runtime protocol](../../tools/nemotron-diarization/paced-public-runtime.protocol.json)
binds12 tools,135 staged entries, seven preparation/reference records and the
controller interpreter. It uses the same previously qualified pooled CPU package,
exact80.04s WAV and saved independent native output; no new native-reference
process is implied. The helper cannot read that reference. Actual input F32LE
SHA256 must remain
`dc94f3f490576f50d11e23728f8fa8bb129ff78b4af41eff3d4961a776b45858`.
All8004 rows/eight channels and turns retain the established1e-6 comparison
bounds; byte equality is reported separately.

The five commands are runtime/libc denial controls, premodel denial controls,
native version, a compiled helper failure before model load, and one fresh paced
neural process. Version and neural calls require strict92-image audits; the
premodel failure must return1, persist an incomplete error JSON and load zero
private runtime images. Protected data/writes/network are denied, and original
5GiB/1GiB/256MiB/600s protections remain. No runtime command has executed at this
preregistration. Deliberate pacing means public elapsed/audio ratio is not a
compute-speed benchmark, even though the inherited comparator retains that field.

## Actual paced replay

The [20-member runtime receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-paced-public-runtime/collection-qualification.json)
records the frozen trial at `1bc54c85`. All five command statuses match
`0, 0, 0, 1, 0`; the expected failure persisted its incomplete error JSON.
Loader audits independently replay as 92, zero and 92 private libraries for
version, premodel failure and paced inference. Exactly one fresh neural process
ran. The reserve was released and the controller completed successfully.

All 8,004 rows, eight probability channels and 35 final turns match the prior
independent native reference exactly, including probability bytes. The collector
independently recomputes normalized PCM from the WAV and binds every result JSON
and raw probability file before and after comparison. Its saved-output replay
performs no inference. Audio, model, binary and raw arrays remain outside Git.

| Observed quantity | Result |
| --- | ---: |
| Input blocks / Busy retries | 4,002 / 2,166 |
| Provisional observations / final revision | 16 / 17 |
| First provisional observation after clock start | 5.020736 s |
| Provisional observation gap, median / maximum | 4.999620 / 5.060948 s |
| Provisional frontier age, median / maximum | 0.683499 / 1.065133 s |
| Previous frontier age at next provisional observation, maximum | 6.063870 s |
| Endpoint scheduling overrun, median / p95 / maximum | 5.944 / 21.156 / 127.470 ms |
| Normalization, hashing and admission, median / p95 / maximum | 0.007 / 0.215 / 139.224 ms |
| Public finish drain, flush and delivery | 229.687 ms |
| Final observation after finish return | 0.002 ms |
| Cumulative native compute / paced public wall | 12.956117 / 80.274308 s |
| Cold load / stream open | 70.916 / 0.024 ms |
| Process peak RSS | 250,544,128 bytes |

Every scheduled endpoint was supplied no earlier than its nominal time and all
samples were accepted. No provisional revision was missed. The public queue
returned Busy without dropping the caller's PCM; retry time is included in
admission, and pacing overrun also includes earlier caller work. Snapshot lookups
total 6,169 and consume 1.749 ms cumulatively. These components overlap.

Native compute divided by audio duration is approximately 0.162 for this paced
run. The inherited comparator's 1.002927 public elapsed/audio ratio includes
intentional waits. Neither is an uncontended throughput comparison with the
earlier native reference. Five-second provisional coalescing remains unchanged;
frontier age does not measure when the correct speaker label became available.
This closes the missing endpoint-paced API timing observation only. Actual GUI,
physical microphone, acoustic label delay, human word gold, original quality,
fresh-machine delivery and public release gates remain open.
