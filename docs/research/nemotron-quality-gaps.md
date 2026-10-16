# Nemotron AMI quality gaps: evidence and next investigations

Read-only investigation on2026-10-16. No inference, rescoring, threshold/profile search, reference changes or qualifier edits. Existing receipts remain authoritative under their frozen protocol. **The AMI quality gate still fails; wrapper parity and the Q8 degradation bound pass.** The clearest discrepancy with published evaluation is the AMI reference policy, not an established Gen2 inference defect.

## What the existing evidence establishes

The CPU receipts under `/Users/victor/.cache/gen2-nemotron3-n0/quality qualification 125f907/` report:

| Fixed120s slice | FP32/shared DER | Q8/shared DER | Q8/native-default DER | Q8 minus FP32 |
|---|---:|---:|---:|---:|
|ES2004a Mix-Headset|41.4447%|41.4715%|33.2146%|+0.0267pp|
|ES2004a Array1-01|50.0223%|49.9599%|43.8407%|−0.0623pp|

Close-mic FP32 errors are45.67missed speaker-seconds,0.02false-alarm seconds and0.84confusion seconds, over112.27reference speaker-seconds. Distant-mic FP32 has51.49miss,0.13false alarm and4.54confusion. Both predict four speakers. Thus missed activity dominates; a lost-speaker-count explanation is insufficient. The Gen2/native probability and turn errors are zero in these receipts, and common-frame FP32/Q8 mean probability differences are small. This points away from a wrapper or quantization regression as the principal cause; it does not prove perfect checkpoint-level model equivalence. [Close receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frozen-corpus/ami-ES2004a-Mix-Headset-cpu-score.json), [distant receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frozen-corpus/ami-ES2004a-Array1-01-cpu-score.json).

The five frozen Vox cases pass their current gates, according to the existing qualification results. Those recordings are explicitly inside the checkpoint's reported Vox development/test training inventory; their success is useful integration evidence, not proof of unseen-meeting generalization. [Frozen corpus research](nemotron-heldout-corpus.md), [checkpoint training inventory](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#training-datasets).

## Current valid-Mel profile requalification

The separate [current-profile receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-frozen-corpus/receiver-qualification.json)
retains 32 fresh native/public processes at the fixed 9/4 geometry, comparing
all valid rows with the exact eight original hash-locked FP32 oracles. All 16
wrapper comparisons are byte-exact and all Q8 degradation bounds pass, while
four AMI backend pairs still fail the unchanged quality gates. CPU production
DER is 33.214572% close / 43.8407411% distant; Metal is 33.2234791% / 43.8229269%.
The five Vox slices and ICSI pass on both backends. These results confirm that
removing the extra terminal frame does not resolve the frozen AMI quality gap.
They neither change reference policy nor establish a new untouched test set.

## Concrete protocol discrepancy: transcriber segments versus published AMI labels

The immutable model card's **Performance Evaluation** warning explicitly identifies broad AMI transcription segments as a source of inflated missed speech: internal silence can be labeled as speaker activity. Its published AMI results use separately linked forced-alignment RTTMs and warn that different references are not directly comparable. This matches the *kind* of error above; it does not establish that every missed second is annotation error. [Pinned evaluation warning and label inventory](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#performance-evaluation).

Our frozen `corpus.py::ami_reference` deliberately uses lexical-containing segments' **transcriber_start/end** from manual1.6.2, not the published benchmark RTTM. Earlier research recommended those human-reviewed boundaries for primary DER but overlooked the model card's explicit task-specific warning. That recommendation should not be treated as evidence that transcription segments are the expected benchmark reference. Human-reviewed transcript content/segments and frame-level speech-boundary suitability are different properties. **Preserve the frozen failure and document this correction; do not retroactively replace references to manufacture a pass.** [Frozen extractor](../../tools/nemotron-diarization/corpus.py), [reference-policy research](nemotron-heldout-corpus.md), [AMI's transcription/forced-alignment description](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml).

The benchmark-linked first-party repository is pinned for a possible *separate* reproduction protocol:

- Repository commit `9527b7c64846fb38316a610f32e9d3466bd6d8b7`.
- `AMI/test/ES2004a.rttm`:34,230bytes; SHA256 `3ed314bb088e11a2a53f6eb1a8ed3876f6ae6f3f92d3340640baabac90167c98`, verified from a small source download without scoring it.
- Its README says AMI uses the full-corpus-ASR split and lexical `only_words`, with Montreal Forced Aligner acoustic/dictionary3.1.0 and G2P3.0.0. It acknowledges alignment errors, ambiguity and OOV limitations. Its AMI labels are CC BY4.0. This is **not necessarily the same alignment as the old word times in AMI manual1.6.2**, nor the vendor smoke RTTM.

Sources: [pinned reference RTTM](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AMI/test/ES2004a.rttm), [method](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/README.md), [license](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/LICENSE).

## Defaults audit: no identified geometry mistake in the frozen run

The frozen geometry chunk9/right4/left0/FIFO264/cache264/update222 matches the model processor's `low_latency` mode and streaming cache configuration. Native probe code explicitly overrides every geometry field; FP32 metadata records the same values. Input normalization/hash, finite probabilities and terminal-frame differences are checked. No evidence here establishes a sample-rate, speaker-index, missing-finish or geometry mismatch. [Frozen protocol](../../tools/nemotron-diarization/corpus.protocol.json), [probe](../../tools/nemotron-diarization/native_stream_probe.cpp), [official processor JSON](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/processor_config.json), [official model JSON](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/config.json).

**Named native presets are not interchangeable with the card's identically themed settings.** In SDK commit083861..., `v3_streaming()` is cache264/FIFO80/chunk13/update40/left0/right1; `v3_offline()` is cache264/FIFO0/chunk264/update188/left1/right1. The former has a1.04s center chunk plus80ms right context, rather than the processor's9+4geometry. The frozen explicit overrides avoid this trap. Future preset changes must record actual resolved numbers; do not claim selecting `v3-offline` reproduces the card's30.4s geometry. [Pinned preset definitions](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/aosc_state.h).

Shared `probability>0.5` segmentation matches the pinned Transformers processor's simple threshold behavior. NeMo's inspected official evaluation script also defaults to bypassing smoothing, with default0.5onset/offset and zero padding/min durations. The native SDK instead uses callhome-tuned hysteresis0.641/0.561, padding0.229/0.079s, minimum-on0.511s and gap-fill0.296s. These are distinct, already separately scored production semantics; native defaults are not proof that shared thresholding is an incorrect oracle. [Transformers processor](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py), [NeMo evaluation defaults](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py), [NeMo postprocessing defaults](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/parts/utils/vad_utils.py), [SDK segmentation defaults](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.h).

The local scorer is10ms midpoint DER over explicit full-WAV UEM with a global optimal speaker permutation. Published evaluation uses NeMo's scorer. Zero collar/overlap inclusion agree for AMI; exact scorer and UEM equivalence still need an independent known-input cross-check. A120s cold-start crop also differs from a full-meeting stateful evaluation. Neither fact alone demonstrates a bug or explains tens of missed seconds. [Frozen scorer](../../tools/nemotron-diarization/score.py), [official evaluation/reporting convention](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/diarization_evaluation.md).

## What published numbers do and do not promise

The card reports corpus-level AMI DER at1.04s as9.48%(headset mix) and12.80%(single distant microphone), across16recordings using its identified references. At30.4s the reported values are9.25%and11.14%. These are not per120s-crop upper bounds and do not promise the roadmap's20%per-slice gate. Its speed table is BF16 on NVIDIA Blackwell; those conditions are not AppleCPU FP32/Q8 performance claims. [Pinned AMI and speed tables](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#performance-evaluation-results).

## Legitimate next routes, without using held-out results to tune

1. **Protocol diagnosis before ML work.** Keep all existing receipts. Preregister a separately named published-reference reproduction using the linked immutable RTTM, exact channel/split/UEM/scorer and explicit model-card geometry; optionally score already-saved outputs to isolate reference effects without changing inference. Report it beside the original transcriber-segment gate, never as a replacement pass. A diagnostic rescore cannot restore the data's untouched-held-out status.
2. **Validate scorer and model implementations independently.** Use small synthetic reference/hypothesis cases for overlap, mapping, silence, crop endpoints and collars against official scoring. If published-reference discrepancies remain, compare original `.nemo` through pinned NeMo with the existing official safetensors/Transformers path at fixed geometry. No threshold search is needed to locate conversion/frontend/state discrepancies. Passing Gen2/native parity does not establish original-NeMo parity.
3. **Engineering changes need separate development data.** Document latency/context tradeoffs, preserving a separate cache/worker per session, explicit geometry, bounded PCM ingestion and correct tail handling. A larger-context configuration, postprocessing calibration or different distant-microphone pipeline may be evaluated on disjoint development recordings and locked before a fresh test. No profile should be chosen by ranking these failed AMI outputs. The published context gain does not establish that changing context will cure this failure.
4. **Further ML work is conditional.** If a correctly reproduced reference protocol and original checkpoint still miss the actual meeting product requirement, options include training/fine-tuning on separately licensed in-domain data, improving speech/overlap detection, or comparing another independently qualified diarizer. The card provides offline and streaming training recipes, but no evidence here proves any such change will meet20%. Define lexical speech versus broad utterance occupancy first: training toward transcription silence and altering a valid acoustic-speech model are different objectives. [Official training recipes](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#training).

No result in this note changes release status. The frozen reference-policy miss is real under its stated protocol; the first justified action is to explain and independently reproduce that protocol distinction, not declare that the model needs retraining or that the failed slices can be discarded.
