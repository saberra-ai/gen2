# Speaker diarization options for Gen2

Reviewed 2026-10-14. Scope: local speaker diarization alongside Gen2 Bark ASR and Pnola. Research and source inspection only; no models downloaded, builds attempted, or inference benchmarks run.

## Recommendation

Evaluate **speakrs** first for recorded meetings on the current Mac; evaluate **Nemotron 3 Diarization through NeMo-Speech.cpp** if live speaker tracking matters. Keep **official sherpa-onnx Rust bindings** as the incremental option closest to the existing Pio implementation. These are integration recommendations, not measured accuracy rankings. Sources and specific constraints follow.

## Current local state

- The active Bark checkout is `/Users/victor/workspace/gen2-bark-onnx`, inspected at commit `6a659ff`. Its [Transcription type](/Users/victor/workspace/gen2-bark-onnx/src/speech/worker.rs:91) returns text, duration, token IDs and execution metadata, with no word timestamps or speakers. Its [decoder](/Users/victor/workspace/gen2-bark-onnx/src/speech/onnx.rs:109) tracks acoustic frame advancement but does not retain per-token timing. Deriving useful word timing from those decisions is a hypothesis requiring validation, especially because Bark cleans up speech.
- [Pnola's README](/Users/victor/workspace/pnola/README.md) explicitly lists no streaming, diarization or word timestamps. This project is being developed concurrently, so this is a snapshot.
- Pio already contains a [Diarizer boundary, ONNX adapter and alignment logic](/Users/victor/workspace/pio-app/pio-core/src/app/capture/diarization.rs), using pyannote segmentation-3.0 plus TitaNet-small through `sherpa-rs`. Its auto-count path explicitly sets zero clusters; a supplied count overrides threshold-based clustering. Reuse the contract and lessons, without assuming its tuned threshold transfers to different recordings.
- The [existing Pio note](/Users/victor/workspace/pio-app/docs/plans/issues/nola-diarization.md) reports 3.8% DER on a synthetic two-speaker fixture. This is historical repository evidence, not a new measurement or proof of real-meeting performance. The note leaves real recordings and iOS packaging as open gates.

## Candidates

| Candidate | Runtime and mode | Why evaluate it | Integration limits |
| --- | --- | --- | --- |
| **speakrs** | Rust; current 0.6 sources expose native CPU, macOS CoreML and Linux CUDA; ONNX Runtime for MIGraphX | Full community-1-style pipeline, including speaker-conditioned embeddings, PLDA/VBx clustering and overlap reconstruction | Qualify as a recording-wide batch pipeline; background queuing does not establish live diarization. Pin library and model revisions. [Repository](https://github.com/avencera/speakrs), [manifest](https://raw.githubusercontent.com/avencera/speakrs/master/Cargo.toml) |
| **Nemotron 3 Diarization** | Streaming or batch; native C++/ggml through NeMo-Speech.cpp, or Python reference implementations | Up to eight speakers; native SDK has a dedicated diarization C ABI suitable for a Rust wrapper | New Gen2 backend and packaging work; its GGUF is a speech architecture and cannot simply be loaded through the existing language-model API. [Model](https://huggingface.co/nvidia/Nemotron-3-Diarization), [SDK](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/sdk.md) |
| **sherpa-onnx** | C++/ONNX Runtime with official Rust API; offline diarization | Closest to the existing Pio code; segmentation, embeddings and clustering are already assembled upstream | Native-library distribution and ONNX Runtime coexistence need verification. [Official Rust example](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_speaker_diarization.rs) |
| **pyannote community-1** | Python/PyTorch; recording-wide pipeline | Reference implementation for correctness comparisons; supports speaker-count bounds and an exclusive timeline for transcript reconciliation | Python packaging is additional deployment work; initial model access requires accepting conditions. Can run offline afterward. [Model card](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| **FluidAudio** | Swift SDK with Apple-specific inference paths | Community-1 offline; LS-EEND streaming variants cover four, seven or ten speakers | Rust needs a Swift/C bridge or separate worker. LS-EEND uses 8 kHz input and recommends CPU-only execution; weight terms need verification separately from the SDK. [SDK](https://github.com/FluidInference/FluidAudio), [LS-EEND guide](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Diarization/LS-EEND.md) |

### speakrs qualification details

The current repository advertises 0.6, Rust 1.89+, Apache-2.0 code, and explicit backend features. Native CPU/CoreML configurations avoid its optional ORT dependency. Gen2 Bark currently pins `ort = 2.0.0-rc.12`; speakrs' optional ORT path uses rc.13. Prefer native CoreML for the initial Mac experiment to avoid coupling those runtimes; compatibility still needs a real build. [Manifest](https://raw.githubusercontent.com/avencera/speakrs/master/Cargo.toml)

Upstream reports 7.1% DER and 529× real-time throughput on VoxConverse dev on an M4 Pro. These are author-reported results, not estimates for this Mac or a meeting recording. [Published methodology and results](https://github.com/avencera/speakrs)

Model artifacts derive from pyannote/WeSpeaker and PLDA/VBx components; the export repository directs users to upstream licenses and access terms. Its card still shows 0.5 instructions while current source shows 0.6. Use the selected release's code and model revision, rather than combining examples across versions. The model manager pins its own revision. [Artifact provenance](https://huggingface.co/avencera/speakrs-models), [model manager](https://raw.githubusercontent.com/avencera/speakrs/master/src/models.rs)

### Nemotron qualification details

Released September 23, 2026, the model supports streaming with recommended input buffers down to 0.32 seconds; that excludes computation and downstream transcript delay. It has 100M parameters and uses OpenMDW-1.1 model terms. [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)

NVIDIA's native runtime has CPU, Metal, CUDA and Vulkan build paths and Apache-2.0 NVIDIA-authored code. Its dedicated C interface exposes offline and streaming diarization with opaque model handles. This makes a wrapper a plausible Gen2 path without requiring Python in the deployed app. Backend performance and packaging remain unverified here. [Runtime](https://github.com/NVIDIA/NeMo-Speech.cpp), [C ABI and ownership](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/sdk.md)

For long recordings, keep streaming state across chunks. The runtime distinguishes large-buffer streaming presets from full-attention offline inference, whose positional table limits recordings to about 6.6 minutes. [CLI guide](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md)

The official Q8 GGUF is 107,012,128 bytes, approximately 107 MB; this is download size, not total process memory. [Artifact metadata](https://huggingface.co/api/models/nvidia/Nemotron-3-Diarization/tree/main)

### Model and runtime licenses

Community-1 model/pipeline terms are CC-BY-4.0, while pyannote.audio code is MIT. Sherpa-onnx code is Apache-2.0 and the older segmentation-3.0 model is MIT; the chosen embedding has its own terms. FluidAudio code is Apache-2.0, with model-specific terms. This distinction matters when packaging a distributable bundle. [Community-1 card](https://huggingface.co/pyannote/speaker-diarization-community-1), [pyannote code license](https://github.com/pyannote/pyannote-audio/blob/main/LICENSE), [sherpa license](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE), [segmentation card](https://huggingface.co/pyannote/segmentation-3.0), [FluidAudio](https://github.com/FluidInference/FluidAudio)

For the existing sherpa combination, published release metadata lists a 6,958,444-byte segmentation archive and a 40,257,283-byte TitaNet-small ONNX. The archive includes variants; those two download sizes are not resident memory measurements. [Segmentation release metadata](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-segmentation-models), [embedding release metadata](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-recongition-models)

### Other paths

- **Legacy Sortformer:** v2.1 is limited to four speakers and its card now points to Nemotron 3. New work should consider the latter first unless a qualified existing export justifies the older model. [v2.1 card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- **Offline Sortformer v1:** its weights use CC-BY-NC-4.0; do not confuse that checkpoint with later differently licensed NVIDIA models. [v1 card](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
- **pyannote-rs:** lightweight Rust building blocks, but its segmentation code reduces non-silence classes to speech and uses non-overlapping ten-second windows. Do not treat that code path as equivalent to the full community-1 pipeline for rapid turns or overlap. [Source](https://raw.githubusercontent.com/thewh1teagle/pyannote-rs/master/src/segment.rs)
- **sherpa-rs:** the wrapper used in the existing Pio code is now deprecated; upstream directs new work to the official sherpa-onnx Rust bindings. [Maintainer notice](https://github.com/thewh1teagle/sherpa-rs)

## Transcript alignment is a separate requirement

Recommended Gen2 boundary: PCM audio in; speaker turns with start/end, anonymous speaker ID, model identity and supported capabilities out. Preserve overlapping intervals and keep IDs stable within one recording. Names require user mapping or a separately designed speaker-identification feature.

Two implementation choices need evaluation:

1. **Diarize first, then transcribe turns.** Works without word timestamps and gives an early prototype. Short turns, clipped phonemes, lost context and overlapping speakers can hurt ASR; use bounded context padding and reconcile duplicate text.
2. **Transcribe normally, then align words to turns.** Better preserves ASR context, but requires validated word timing or a compatible forced aligner. Keep Bark's cleaned presentation text separate from timing evidence when cleanup changes the spoken wording.

These are proposed designs. Do not label an entire 60-second chunk as one speaker when it contains multiple turns. Diarization can mark overlapping voices; it does not by itself recover both speakers' words from mixed audio.

## Evaluation before choosing a default

Use the same hand-annotated recordings for every candidate: a clean two-person conversation, rapid interruptions, a far-field meeting, overlapping speech, noise and silence. Include recordings longer than the ASR chunk boundary and test speakers returning after long gaps.

Measure DER with identical collar and overlap rules, speaker-count errors, speaker-attributed transcript accuracy, missing/duplicated words at cuts, wall time, cold-load time and peak memory. For streaming, measure actual end-to-end delay and label revisions. Compare CPU and CoreML on the target Mac; validate mobile separately if it becomes a requirement. A clean synthetic test is useful for regression coverage but cannot select the meeting-quality default.
