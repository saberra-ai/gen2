# Local Nemotron 3 speaker timelines

Implementation qualification is in progress. macOS arm64 CPU has real-model
wrapper parity; quality, attribution and desktop release gates remain open.
See [the roadmap](plans/nemotron-3-diarization-roadmap.md).

Build with `backend-diar-nemo`. This adds no build-time model downloads or C++
SDK requirement. Supply the pinned NeMo-Speech.cpp C ABI library and its
dependencies explicitly; the build instructions and locked native revisions
are in [the qualification tools](../tools/nemotron-diarization/README.md).
Gen2's chat `metal`/`cuda` flags do not change diarization placement. This
implementation requests CPU and reports `backend: "cpu"`.

```rust,no_run
use gen2::{Runtime, diarization::{Audio, DiarizeOptions, LoadOptions}};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let runtime = Runtime::builder().resident_memory_budget_mb(2048).build()?;
let model = runtime.load_diarizer("nemotron-model", LoadOptions::cpu("/sdk/lib/libnemo_speech_asr_c.dylib"))?;
let result = model.diarize(Audio::from_wav("meeting.wav")?, DiarizeOptions::default())?;
for turn in result.turns { println!("{}–{}: speaker {}", turn.start_seconds, turn.end_seconds, turn.speaker); }
model.shutdown();
# Ok(()) }
```

Provision explicitly with `download_nemotron(cache)` and the optional `hf`
feature. The model revision, filename, size and SHA-256 are fixed. Downloads
include the original model card and its terms, use a file lock across
processes, stage before verification and publish by rename. A verified warm
cache needs neither `hf` nor network access. Inference never fetches missing
files. Runtime binaries are packaged independently and must remain unchanged
while loaded.

The recording API accepts finite mono float32 PCM at 16 kHz, from 20 ms to
120 seconds. Its WAV helper accepts mono 16 kHz PCM16 or float32. Bark's own
input limit is independent. Hosts handle microphone capture, conversion and
resampling. Speaker IDs are 1–8, anonymous and scoped to each recording;
overlapping turns remain overlapping. Silence can return an empty timeline.
Output cadence is 10 ms, which does not imply 10 ms boundary accuracy. The
native centered STFT produces one padded terminal frame; diagnostics retain it
and presentation turns are clipped to real audio duration.

For continuous input, `start_stream(options)` holds one native cache across all
transport blocks and ASR cuts. `try_push(Vec<f32>, 16000)` accepts 1–16640
samples per block. The two-block queue returns `PushError { samples, error }`
when saturated; retry the returned samples on `Busy` without advancing your
audio cursor. No audio is silently dropped. Rate changes and nonfinite samples
are rejected. A safety cap limits a session to one hour and results to 100000
turns. These are hard bounds; long-session release qualification is recorded
separately in the roadmap.

`snapshot()` returns the latest coalesced snapshot at most once per five
seconds of input, plus the final snapshot. Its session UUID and monotonic
revision identify a replacement timeline. Consumers replace earlier revisions
and bound their own retained history. All provisional turns can change; the
processed-audio position is not a finality watermark. On reconnect, fetch the
current snapshot rather than replaying/appending previous revisions. Raw
probabilities are copied only on finalization when explicitly requested;
native history compaction can discard older diagnostic frames while retaining
the full segment timeline. Only native default segmentation is exposed.

`finish()` stops admission, drains accepted PCM, flushes once, and returns a
cached `Arc<Diarization>`. Repeated finish returns the same result or failure.
Pushes after finish fail in Rust. Cancellation and deadlines cooperate between
bounded native calls. Dropping a stream requests cancellation. A failed or
cancelled stream never publishes a successful complete result. A live stream
owns the model's one active slot; competing recordings or streams return
`Busy`. Idle loaded weights still hold their reservation.

`unload()` stops new work without waiting for a synchronous native kernel.
`shutdown()` waits for actual stream/model/library destruction and releases
runtime admission afterward. The initial conservative reservation is 1024 MiB;
it shares the runtime budget with chat, Bark and Laya. Cancellation cannot
promise a hard interrupt deadline.

Examples: `nemotron_diarize MODEL_DIRECTORY LIBRARY WAV`,
`nemotron_diarize --download CACHE`, and
`nemotron_stream MODEL_DIRECTORY LIBRARY WAV REPEATS`. The stream example
reopens the input between repetitions without resetting the native session;
its JSON includes revisions, retained frame range, compute/wall timings and
backpressure counts. It is a transport/soak harness; a repeated clip is not a
held-out long meeting.

Model terms: OpenMDW-1.1. Native runtime and transitive dependency notices are
separate. Model-free CI does not establish native packaging, acceleration or
real-model support for an OS. Speaker-attributed transcription requires the
roadmap's separate alignment and Pnola work.
