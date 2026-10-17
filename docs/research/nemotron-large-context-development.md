# Publisher large-context development contrast

Initial source audit, 2026-10-16 local / metadata retrieved 2026-10-17 UTC. The audit executed no model, native helper, build or inference and acquired no complete audio. Subsequent input preparation, model-free helper construction and the incomplete first inference attempt are recorded below. This separately frozen development experiment does not replace the failed ES2004 test, its references, or the roadmap’s proposed ≤20% DER gate.

## Exact geometry and native applicability

At publisher revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`, the model’s own “Setting up Streaming Configuration” table and evaluation command prescribe these **80 ms encoder-frame** values:

| Configuration | Chunk | Right | Explicit left | FIFO | Speaker cache | Update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low latency | 9 | 4 | 0 | 264 | 264 | 222 |
| Very high latency / chunked offline | 340 | 40 | 0 | 40 | 264 | 300 |

The large input-buffer latency is 30.4 s, excluding compute. The separate cache188 table belongs to the predecessor baseline, not this model. These are publisher-prescribed configurations, not settings selected from the failed cases. The card’s training inventory includes AMI train **and development**, including individual-headset mixture inputs; this experiment is therefore training-exposed development, not unseen-data evaluation. [Pinned publisher card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md).

**Left0 is source-resolved, not a missing table value guessed from a preset.** The acquired original `.nemo` configuration has `chunk_left_context: 0`; the pinned official NeMo eight-speaker recipe also sets0. HF’s whole-recording path slices each current chunk plus future right context, then prepends cached embeddings; it adds no previous-Mel overlap. Left0 therefore retains historical information through FIFO/cache. HF’s nonstreaming forward uses top-level `chunk_length=340`, `chunk_right_context=40`, `fifo_length=40`, `speaker_cache_update_period=300`; its streaming cache configuration supplies cache264. Do not use the HF processor’s low-latency streaming mode without overriding its different geometry. [Official NeMo recipe at218d0b4](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_streaming_8spk.yaml#L121-L132), [HF pinned forward](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L730-L800), [publisher config](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/config.json).

The pinned native **public C ABI already exposes all six overrides**. Set `size=sizeof(config)`, CPU `gpu=-1`, `preset="v3-streaming"`, then explicitly supply every row value, including left0; open/push/finish a stream. Native `v3-offline` resolves to different values, and `nemo_speech_diar_offline_f32` is a separate stateless full-attention path: neither represents this contrast. The ABI supports high-resolution V3’s eight channels/10 ms output; independent streams serialize compute internally. [C header](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/include/nemo_speech/diar.h#L31-L118), [override application](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/c_api.cpp#L485-L520).

Source validation requires positive chunk/update, nonnegative FIFO/contexts, cache≥`(1+silence_frames_per_speaker)*speakers`, and `cache+FIFO+left+chunk+right≤position_limit`. For source/config silence1, speakers8, limit5000, both rows satisfy it: cache264≥16; maximum encoder sequences are541 versus684. The native scheduler’s maximum current Mel windows are104 versus3040 frames (`8*(chunk+right)`); V3 pads feature stacking to multiples of8. An update300 with FIFO40 is supported: AOSC appends the current chunk before clamping/populating its FIFO/cache update. These checks establish **source-level admissibility**, not measured memory, speed, backend allocation success, or parity at this geometry. Bind actual GGUF metadata and the selected `gen2-valid-mel-v1` SDK/package hash before execution; do not relabel an older artifact. [Geometry/presets](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/aosc_state.h#L25-L49), [validation and update](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/aosc_state.cpp#L200-L310), [scheduler](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.cpp#L173-L222), [V3 graph](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.cpp#L170-L200).

## Deterministic recording and availability

Use **AMI Full-corpus-ASR SB**, matching the publisher-linked NTT reference repository, not a generic “AMI dev” label. The official list is ES2011, IS1008, TS3004, IB4001–4004, IB4010–4011; expand available submeetings, exclude the entire EN2002 and ES2004 groups, sort recording IDs lexicographically, and take **ES2011a**. Scenario-only/full-corpus SB instead also includes ES2003 and TS3006: ES2003a would win there but is **training** under Full-corpus-ASR and is ineligible for this rule. Its earlier header-only check is retained as unselected metadata. [Official split definitions](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml), [pinned NTT split declaration](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/README.md#annotations).

The existing annotation ZIP was rehashed, not downloaded:22,887,865B, SHA256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`. Its `meetings.xml` marks ES2011a development; IDs A/B/C/D map to FEE041/FEE042/FEE043/FEE044. This confirms identity, not acoustic similarity to the failed recordings. Official signals/manual1.6.2 are CC BY4; retain corpus attribution. [Official download/license](https://groups.inf.ed.ac.uk/ami/download/), [signal/channel documentation](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml).

Bounded `Range: bytes=0-63` GETs returned206,64B and `Content-Range: bytes 0-63/35643096` for both exact official URLs:

| Audio | ETag | SHA256 of64-byte prefix |
| --- | --- | --- |
| [ES2011a.Mix-Headset.wav](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2011a/audio/ES2011a.Mix-Headset.wav) | `"21fded8-3fc2cda65d7c0"` | `c424e5b2e6c074eda8b42a7c7f989b616fc97d873230404f4c1daff2670e8333` |
| [ES2011a.Array1-01.wav](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2011a/audio/ES2011a.Array1-01.wav) | `"21fded8-3fa87bc5ef280"` | `c3b4b55d947150273061328adf6b065f6187aca290f2ad90914fb18a7599f375` |

Headers describe mono16kHz16-bit PCM, data35,643,052B =1113.845375s; the pinned BUT UEM independently ends at1113.845375. The older meeting metadata says1128s; use the actual WAV header for the bounded crop, without inventing an explanation for the difference. ETags/prefix hashes are **not full-file SHA256s**. [Pinned BUT UEM](https://github.com/BUTSpeechFIT/AMI-diarization-setup/blob/2509d8933721023fab4def2618aabd5c28eb82e9/uems/dev/ES2011a.uem).

## Reference identities and fixed experiment

Keep **[0,300)s** on both channels, regardless of subsequent predictions. Annotation-only extraction using the established lexical-child/transcriber-boundary policy yields37 intervals: A126.673s, B1.792s, C54.511s, D23.963s after same-speaker union. This has weak B coverage; do not move the window to improve it. These human-reviewed transcriber segments are distinct from automatically aligned word intervals. [AMI transcription procedure](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml).

A concrete optional **published-reference diagnostic** is available: [NTT `AMI/dev/ES2011a.rttm`](https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AMI/dev/ES2011a.rttm),33,395B SHA256 `1ea587676a2082947f08370c5a49fe7890badcc62a7fb1377cd37f383b198d3b`. It is MFA-produced using their documented models and is CC BY4 under their component-specific license; it is **not** the local AMI word-union or a human-timed word gold. The separate [BUT only_words RTTM](https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/2509d8933721023fab4def2618aabd5c28eb82e9/only_words/rttms/dev/ES2011a.rttm) is12,250B SHA256 `2372b3775e1093a128cc30852f2119affacb2f44c7e815413bd3a84df2bb414b`; do not substitute it under the NTT name. [NTT procedure](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/README.md), [NTT AMI license](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/LICENSE).

Preregister exactly two geometries × two channels using the same selected locked weights/runtime, crop PCM, push cadence, reset policy, segment postprocessor, full300s UEM, zero collar and included overlap. Query final results after finish. Change no thresholds, speaker limits, reference rules or gate. Name primary transcriber, optional NTT-aligned and local word-union results separately; none replaces frozen ES2004 references. Freeze references and input hashes before any output, report DER components, speaker count, latency/RSS and failures for every cell. Large-context improvement is a hypothesis; a development gain does not establish the unchanged test gate or broad generalization.

## Retained metadata

Cache: `/Users/victor/.cache/gen2-nemotron3-n0/research/large-context-development/`; manifest6,486B SHA256 `440ffea2ef8b3ef4b3af1225305416124c0b966aa09145515fadf9de17c582c1` inventories headers, exact reference/notice bytes, archive-member hashes and annotation-only intervals. Official unversioned split HTML14,560B SHA256 `30d038f540a91ba6e68133e8fdfaa1d2b4c1b7291f871b0d4821ea04f4c776ed`; publisher README43,239B SHA256 `5fe88ca4020d2ee134f806a0a02c52c19855b73047bc2e6f25b9171e1abeebfd`; config1,458B SHA256 `dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873`; original `.nemo` config3,461B SHA256 `0a738c13104761be11da4a315af0a94b3c14a1cd6509a3e788a78555751e1ccb`. Future acquisition still needs complete selected audio/crop hashes and actual runtime resource qualification.

## Input and probe construction preregistration

The [input protocol](../../tools/nemotron-diarization/context-development-inputs.protocol.json)
freezes nine tool sources, eight external metadata inputs, both reference JSON
hashes and the original audio ETags before acquisition. The preparer downloads
exactly9,600,000 PCM bytes per channel, writes mono16kHz PCM16 WAV without
resampling, and records actual normalized F32LE input hashes. Partial acquisition
failures remain in fresh external output; there is no automatic retry or crop
substitution. Generalized reference extraction reproduces the historical
ES2004a transcriber/word/member outputs exactly; historical modules are unchanged.

The separate [construction protocol](../../tools/nemotron-diarization/context-probe-construction.protocol.json)
freezes seven source/header inputs, seven tools, the compiler identity and the
123-entry filesystem inventory of the previously qualified non-pooled CPU
package. Compilation uses only staged public C headers and separate probe/SHA256
dependency files. Model reads, network access, original source/SDK/package and
Homebrew reads are denied to construction children. A rejected profile performs
no model call; its dynamic initialization must still match all92 private images.
Construction has a3GiB start floor,1GiB running floor,256MiB reserve and120s
per-command process-group deadline. This model-free floor does not change the
5GiB/1GiB/256MiB/600s protections required for later neural execution.

Neither protocol performs inference or establishes large-context public Rust
parity. Before inference, a separate execution protocol must bind the acquired
audio, constructed helper, exact Q8 artifact, reused public low-latency binary,
scorers, command sequence and failure/resource policy. The four native
geometry/channel cells require two additional public low-latency controls.

## Actual preparation and construction

The [30-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-development-preparation/collection-qualification.json)
records successful acquisition and construction at `3d40e166`. Both channels
contain exactly4,800,000 samples /300s, with9,600,044-byte WAVs. Actual normalized
F32LE SHA256 is `ceaa332efb5324dc6c47bad6c8cdcd2bcfce4d6544a21e0a1780b3379ab0cfdb`
for Mix-Headset and `5c7f3ff83edbd30b39abf42add41ade1b7b44976bbcb1acf60eb4b1ec87a7459`
for Array1-01. Independent collection replays all samples and both reference
policies. Primary37 intervals remain distinct from104 published MFA intervals.

All six construction commands return their expected status, including invalid
profile1. The actual compiler dependency files cover exactly seven staged inputs;
their final identities and all123 package filesystem entries remain unchanged.
The constructed62,304-byte helper has SHA256
`3207eb7a1a5ba3f61361c7246f5a56550840495a7631679ffd93d981aed3e840`.
Its invalid-profile process matches all92 private runtime images and makes no
version/model call. Sandbox denial controls pass and the256MiB reserve is released.
All244 local tools tests pass. Corpus audio, annotations and executable payloads
remain outside Git; the receipt contains metadata and logs only.

Review strengthened collection to verify14 current replay-tool identities and
the historical verifier bytes before importing them. The initial unadmitted
collection is retained unchanged in external cache; V2 repeats metadata/sample
replay only, without acquisition, compilation or native calls. No large-context
inference, performance, FP32 comparison, production selection or release gate has
been completed by this preparation.

## Frozen execution protocol

The [execution protocol](../../tools/nemotron-diarization/context-development-diagnostic.protocol.json)
now binds14 transitive tool sources plus the bounds regression,140 staged filesystem entries and seven
preparation records. It reuses the exact earlier public low-context binary and
verifies its258 historical Rust source blobs. After offline runtime/premodel
denial controls, version-only and invalid-profile checks, it admits native low,
native large and public low in that order for each channel. Every neural case
uses a fresh process; eight strict92-image audits include the two non-neural
native controls. All30,000 rows must be present, actual normalized PCM must match
the acquisition hash, and low-context public/native probabilities must match
byte-for-byte with the established1e-6 turn bound. Both native profiles reject
turn endpoints exceeding300s by more than the established100µs rounding bound
before scoring; a regression rejects excessive or wholly out-of-recording turns.

Both primary transcriber and published MFA references are scored separately with
unchanged shared and native production postprocessors,10ms frame and independent
continuous scorers, a full300s UEM, included overlap and zero collar. Four scorer
calibrations precede execution. Actual process RSS, load/push/finish timings and
RTF are retained;30.4s large-context nominal buffering is not measured acoustic
label delay. The5GiB start floor,1GiB running floor,256MiB reserve,600s deadlines
and process-group cleanup remain unchanged. Inputs are read-only, child reads of
references/scorer/provenance/prior results are denied, and any failure stops the
trial without automatic retry. V1 stopped as recorded below; the complete contrast
remains unqualified.

## V1 canonicalization failure and separate V2 admission

The [28-member failure receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-canonicalization-failure/collection-qualification.json)
retains V1 at `aaa97b587`: runtime/premodel/version/invalid-profile controls pass,
then both Mix-Headset native profiles complete30,000 rows each. The public
low-context process exits1 with `InvalidBundle(Operation not permitted)` before
loading any private runtime image. Four completed native controls/runs have
strict92-image audits. The controller stops; no distant-microphone process runs,
and the reserve is released. Partial scores/arrays remain diagnostics, without a
completed public/native comparison or gate promotion.

A saved model-free ctypes reproduction calls libc `realpath`, matching Rust
canonicalization, rather than Python's different path resolver. V1 denies cache
ancestor metadata, returning errno1 before bundle verification. Changing only
the cache-wide outside-work denial from `file-read*` to `file-read-data` permits
the required directory/model path canonicalization while reads of the actual
outside model and staged reference remain denied. No neural call is needed for
this diagnosis.

The separately frozen [V2 protocol](../../tools/nemotron-diarization/context-development-diagnostic-v2.protocol.json)
binds16 tools,11 prerequisite records and fresh byte-identical140-entry staging.
Its versioned controller preserves V1 unchanged, narrows only that cache data
denial and adds an actual libc canonicalization control before native version or
neural admission. Model bytes, geometry, channels/crop, references, postprocessors,
scorers, six neural cases, eight image audits and all resource/gate criteria remain
unchanged. V2 has not executed at preregistration.
