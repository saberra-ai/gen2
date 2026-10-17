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
