# Nemotron held-out corpus: model-blind selection and obtainable sources

Status: **candidate identities and annotation-derived selection frozen in this note; audio hashes and qualification receipts still required**. Researched 2026-10-16. This investigation read annotations, ZIP directories and small HTTP ranges. It ran no audio inference and downloaded no complete audio corpus. The existing AMI EN2002d 2132.5–2192.5 s sample is a tuning/smoke case and is excluded; exclude its entire EN2002 recording group from the new gate.

The proposed set has **eight target slices from seven recordings**, covering exact reference counts of 2, 4 and 8, a paired distant-microphone case, substantial overlap, annotated brief responses, and documented real incidental noise. Selection depends only on source annotations and fixed rules below. These are local tuning-held-out cases, **not a claim of pretraining-independent evaluation**. NVIDIA's training inventory explicitly lists ICSI Full and VoxConverse v0.3 Development and test; its AMI entry is narrower: Train and development. ES2004 belongs to AMI's published evaluation partition, so it must not be described as training-exposed merely because the corpus name appears. The exact exposure evidence and limits are detailed below. [Pinned model training inventory](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md).

## Frozen selection and acceptance protocol

1. Freeze the source versions, selection rules, recording IDs, time spans, input channels, reference policy and scorer before running either FP32 or Q8 on these recordings. Write immutable SHA-256 manifests when the audio is acquired. Reading labels to enforce target speaker counts is permitted; inspecting model predictions to change the set is not.
2. VoxConverse: use only `test/*.rttm` from annotation commit `24bf60be297701cd7e4ef18550c6d390c1b87365`. Eligible full recordings have exactly 2, 4 or 8 unique reference speaker IDs; every speaker has at least 2.0 reference seconds; latest reference endpoint is at most 600 s; at least 1.0 s of simultaneous speech; and at least two reference intervals of at most 1.0 s. Sort eligible recording IDs lexicographically within each count and take the first two, or all if fewer exist. This yields 2/2/1 recordings. Do not replace the lone eight-speaker result with one that happens to score better. These counts were recomputed from pinned RTTM contents; they are not model speaker counts. [Pinned annotation tree](https://github.com/joonson/voxconverse/tree/24bf60be297701cd7e4ef18550c6d390c1b87365/test).
3. AMI: fix ES2004a, outside the smoke recording group. On a 120 s window / 60 s stride grid, take the first window with all four speakers and at least 5.0 seconds of positive-duration lexical word annotation per speaker (the transcript text is human; these word timings are automatically forced-aligned). This is [300,420) s. Evaluate the **same crop and reference intervals** on both Mix-Headset and Array1-01; the recommended primary DER reference uses the transcriber segments described below. Source channel identity supplies the distant-microphone condition; it does not establish a measured SNR. [Manual 1.6.2 download](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip), [signal documentation](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml).
4. ICSI: iterate meeting IDs lexicographically; on a 300 s / 60 s grid take the first window with exactly eight participating speaker IDs and at least 5.0 seconds per speaker, using positive-duration MRT `Segment` elements with a `Participant` and alphabetic transcribed text. This yields Bmr005 [60,360) s. Exclude noise-only/vocalization-only segments from speech references; preserve the noise annotations as condition metadata. [Original manual transcript archive](https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip).
5. Freeze segmentation and scoring policies too: a shared postprocessor for FP32 and Q8; one reset per target slice; overlap included; explicit 0.0 s collar and full slice UEM; optimize reference-to-hypothesis speaker mapping. Report DER components and reference speaker counts. Do not pool the paired AMI channels as independent meeting evidence. The roadmap’s proposed gate is **DER <=20% on every target slice and Q8 DER degradation <=2 percentage points versus FP32 on every slice**. A failing slice stays in the receipt. No threshold selection, deletion or substitution using held-out results. Model-card evaluation guidance requires source/version, overlap, collar, reference, precision and geometry reporting. [Evaluation subcard](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/diarization_evaluation.md).

For intervals, use half-open [start,end). Compute per-speaker speech seconds after unioning any same-speaker overlaps. Compute overlap wall time with at least two distinct active speakers; its denominator below is the union of all reference speech. Counts of intervals <=1 s are **short annotation spans**, not proof of conversational function. Never count punctuation or zero-duration words as speech.

## Concrete target slices

These values were computed from pinned reference annotations before any output on this set. AMI has human-reviewed transcriber segments and automatically forced-aligned word times; the table uses the recommended transcriber-segment DER policy clarified below, while the original crop-selection rule remains unchanged. Vox rows cover each complete WAV; the endpoint shown is the final reference speech endpoint, **not the WAV duration or a crop boundary**. Use the complete decoded sample length as UEM and retain leading/trailing silence. The other three rows are exact fixed crops.

| ID / source | Input / crop | Exact speakers | Minimum reference seconds per speaker | Overlap wall seconds | Short-span evidence |
|---|---|---:|---:|---:|---|
| Vox test bjruf | Complete WAV; reference ends351.36s | 2 |185.50|43.86 (13.35% of speech)|31 RTTM intervals <=1s|
| Vox test bxcfq | Complete WAV; reference ends196.44s |2|54.06|5.05 (2.57%)|6 intervals <=1s|
| Vox test aepyx | Complete WAV; reference ends168.27s |4|18.18|3.39 (2.34%)|2 intervals <=1s|
| Vox test cqfmj | Complete WAV; reference ends113.95s |4|2.34|5.11 (4.94%)|2 intervals <=1s|
| Vox test auzru | Complete WAV; reference ends515.93s |8|9.48|3.03 (0.60%)|4 intervals <=1s|
| AMI ES2004a close | Mix-Headset [300,420)s |4|11.577 transcriber-segment seconds|24.223 (28.73%)|6 lexical transcriber segments <=1s; human transcript contains brief “Yeah” replies|
| AMI ES2004a distant | Array1-01 [300,420)s |4|Same manual reference|Same reference|Same replies, different channel|
| ICSI Bmr005 | interaction mix [60,360)s |8|14.493|3.284 (1.32%)|55 lexical MRT segments <=1s|

Pinned Vox references: [bjruf](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/bjruf.rttm), [bxcfq](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/bxcfq.rttm), [aepyx](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/aepyx.rttm), [cqfmj](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/cqfmj.rttm), [auzru](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/auzru.rttm).

The eight Vox auzru reference speaker totals are spk00:61.85, spk01:82.49, spk02:119.13, spk03:94.96, spk04:9.48, spk05:23.60, spk06:54.92, spk07:62.34 seconds. This proves eight annotated identities, **not eight simultaneous speakers**. Bmr005 totals are me011:14.493, me013:36.361, me001:36.556, mn017:31.509, me018:46.071, mn005:35.545, fe008:26.983, fe016:24.039 seconds under the lexical-segment filter. [Vox pinned reference](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/test/auzru.rttm), [ICSI original archive](https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip).

Acquisition clarification before inference: complete decoded auzru audio is
515.904 seconds; its provided final reference ends at515.930 seconds. Applying
the already frozen full-WAV UEM clips26ms from spk07, yielding62.314seconds.
The original pinned RTTM is retained unchanged. No audio or reference-based
selection changed. The ICSI crop contains substantial annotated digit reading
(161 selected segments have a Digits comment), alongside the two incidental
noise events. It supplies real eight-participant audio, not an exclusively
spontaneous-conversation or severe-noise claim.

AMI transcript content is manual, but its word timings are automatically forced-aligned. AMI also supplies transcriber segment boundaries; these are the recommended primary DER reference below. ICSI has manual utterance boundaries; do not invent word-level timing inside them. Document each reference policy in the receipt rather than implying the corpora have identical silence conventions. Existing `tools/nemotron-diarization/qualify_alignment.py` verifies the AMI archive and reads lexical words, but its whole-word-containment behavior is designed for alignment diagnostics; the DER extractor should clip valid positive-duration intervals at crop boundaries, preserve speaker identity and emit UEM explicitly.

## Downloadability verified against actual first-party endpoints

### VoxConverse: old official archive URLs are live despite landing-page warning

The current [Oxford landing page](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/) says audio is unavailable. The pinned [author repository README](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/README.md) still links the original ZIP endpoints. Small GET ranges on **both exact endpoints** returned HTTP206, `Accept-Ranges: bytes`, ZIP signatures and the following total lengths on2026-10-16. This is an observed endpoint discrepancy, not an inference that missing website links guarantee availability:

| Archive | Total bytes | ETag | Last-Modified |
|---|---:|---|---|
| [dev ZIP](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip)|1,988,647,478|`"76885a36-5c7a2104ed3f9"`|2021-07-21 13:22:35 GMT|
| [test ZIP](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip)|4,272,764,397|`"fead35ed-5c7a1fc24ca4c"`|2021-07-21 13:16:57 GMT|

Only ZIP metadata was fetched: 65,557-byte tails and their central directories (dev18,436bytes at offset1,988,629,020; test52,872bytes at offset4,272,711,503). The five selected test WAV members are present. Their compressed payloads total **36,299,315 bytes**, expanded members43,424,134bytes, so a bounded range extractor can retrieve them without a4.27GB transfer. No full selected member was fetched or decoded in this research. Archive ETags and CRC32 are operational identity checks, **not cryptographic file hashes**. Compute SHA256 after extraction and freeze it before inference.

| WAV under `voxconverse_test_wav/` | Compressed bytes | Expanded bytes | ZIP local-header offset | CRC32 |
|---|---:|---:|---:|---|
|bjruf.wav|10,289,398|11,536,462|2,889,360,004|c0b1b68b|
|bxcfq.wav|5,303,849|6,287,438|2,042,970,517|a19107dd|
|aepyx.wav|4,458,980|5,412,942|2,623,721,036|d77c6b11|
|cqfmj.wav|3,037,871|3,678,286|1,534,141,565|86526190|
|auzru.wav|13,209,217|16,509,006|2,807,818,362|6fed260d|

All five use ZIP method8(deflate). Range extraction must read and validate the local header, add its filename/extra-field lengths to find payload start, fetch exactly the compressed extent, inflate with raw deflate, and verify expected member length/CRC32. Require206 and correct `Content-Range`; abort if the server falls back to a whole-file200 response. Pin `If-Match` to the observed archive ETag; on change, recheck metadata and abort the frozen run rather than silently using new bytes. Reject unexpected paths. These are implementation recommendations derived from observed ZIP metadata, not an existing tested downloader.

Annotation ZIP: [commit archive](https://codeload.github.com/joonson/voxconverse/zip/24bf60be297701cd7e4ef18550c6d390c1b87365),355,266bytes, downloaded SHA256 `c340d2cd9da2f19eff8a3adc55b0818ec45cd0770ef72c6926c4de6efe9e9a1d`;448RTTM files. Eligible counts under the frozen test-only rule were15two-speaker,7four-speaker,1eight-speaker. Git commit and raw RTTM content hashes should be canonical, since GitHub generated ZIP container bytes can change without source changes.

### AMI: direct PCM endpoints, no archive extraction needed

Both [ES2004a.Mix-Headset.wav](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav) and [ES2004a.Array1-01.wav](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Array1-01.wav) returned206 for a64-byte range, length**33,579,394bytes each**. Headers identify RIFF PCM16, mono16kHz,44-byte data start; raw clip[300,420) therefore needs3,840,000PCM bytes beginning at offset9,600,044, plus a generated WAV header. Prefer a validated RIFF parser rather than hardcoding44for arbitrary files. This observed header permits a small exact crop download; no full signal downloaded here.

The cached [AMI manual1.6.2 archive](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip) is pinned by SHA256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`. References are `words/ES2004a.{A,B,C,D}.words.xml`; use `corpusResources/meetings.xml` for the identities/channel mapping. Archive counts above were derived from these files. Preserve their individual hashes in the acquisition manifest. [Official downloads and license notice](https://groups.inf.ed.ac.uk/ami/download/).

### ICSI: exact interaction-mix endpoint confirmed; plausible mirror paths were wrong

The official [download chooser](https://groups.inf.ed.ac.uk/ami/icsi/download/) returns a wget script for sessionBmr005 / Mix-Headset. That script points to **[Bmr005.interaction.wav](https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/Bmr005.interaction.wav)**. This exact URL returned206 for128bytes, total**136,402,136bytes**, ETag`"82154d8-53837057d2e80"`, Last-Modified2016-07-22 10:37:30GMT. Its RIFF header identifies16kHz monoPCM16 and44-byte data start. Fixed[60,360)crop needs9,600,000PCM bytes beginning at offset1,920,044. The guessed `ICSICorpusMirror/...Mix-Headset.wav` URLs returned404 and must not be used as availability evidence.

The [original MRT transcript archive](https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip) was downloaded:3,720,243bytes, SHA256 `9db98ebdd61a085b4f45f24f46e9c9f6c6f26c3c9cd41643a3b9e74362c85de7`. Reference is `transcripts/Bmr005.mrt`, with named participant segments. In the chosen300s window it contains explicit non-vocal-sound annotations in segments140.033–140.913s(squeak) and161.510–162.430s(mike noise), relative to the **original** recording. These verify real incident-noise events, not a measured noise floor or a generally severe noise condition. No synthetic noise is proposed. [ICSI download documentation](https://groups.inf.ed.ac.uk/ami/icsi/download/).

## Licenses, attribution and label quality

AMI and ICSI official distribution pages state signals and transcription are released under **CC BY4.0**. Include corpus name, recording/channel, original-source links, license link and changes(crop/sample conversion), retain accompanying credits, and cite their corpus papers in the evaluation report. Do not assume every third-party annotation package has the same terms; this proposal uses core/manual transcripts. [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml), [ICSI distribution license](https://groups.inf.ed.ac.uk/ami/icsi/download/).

The VoxConverse author README describes research-purpose distribution under **CC BY4.0**, with original video owners retaining copyright. Preserve that wording and the Chung/Huh/Nagrani/Afouras/Zisserman Interspeech2020 citation in provenance; do not re-label its recordings public-domain or describe ownership as transferred. This research proposal uses the first-party research download, not an unverified mirror. [Pinned license/citation text](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/README.md).

VoxConverse references are **semi-automatic labels manually checked/corrected by the authors**, not completely automatic pseudo-labels and not fresh local hand annotation. The primary paper describes verification, exclusion of laughter from speaker speech, inclusion of short transcribable replies, and boundary care within0.1s. This supports using its curated RTTM; “exact” means fixed provided boundaries/identities, not error-free ground truth. Use v0.3 corrections pinned above. [Primary paper, manual verification section](https://robots.ox.ac.uk/~vgg/publications/2020/Chung20/chung20.pdf), [version0.3 notice](https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/README.md).

## Remaining qualification gates

The source inventory solves the2/4/8label-count and first-party availability problem without a multiGB fetch. Next acquire only the frozen ranges/members, check hashes/CRC and PCM headers, freeze exact WAV/UEM lengths, validate speaker-reference extraction and human provenance, and then run the already-frozen gate. Decode failures or source mismatches stop qualification; they do not authorize swapping a low-scoring recording later.

Do a model-blind listening/provenance check if a **severe/continuous noise** product requirement exists: only incidental Bmr005 noise and the known distant array channel are established here. Vox's corpus-wide challenging-background description does not prove the noise level of these five specific recordings. Eight speakers are represented across a recording, with limited8-speaker-slice overlap; this set does not establish eight-way simultaneous-speech quality. Neither caveat should be hidden by aggregate DER.


## AMI boundary-policy clarification before inference

**Yes: manual1.6.2 includes per-speaker utterance/transcriber speech segments, so a union of forced-aligned word intervals is not the only available reference.** The pinned archive contains `segments/ES2004a.{A,B,C,D}.segments.xml`. Its `<segment>` elements have explicit `transcriber_start`, `transcriber_end` and channel attributes, and NXT child links into the corresponding word/nonword stream. The archive's `corpusdoc/annot_segment.html` documents this layer. For example, ES2004a.A's first segment has transcriber bounds[0.0,1.792), whereas its contained lexical words begin0.37s and end1.76s. Those are different reference policies, not interchangeable copies of an identical boundary. [Pinned archive containing the segment layer](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip).

The primary corpus description explains that initial speech segmentation was energy-based; first-pass transcribers corrected segment boundaries when needed and second-pass transcribers reviewed all segments, including missed speech. It separately states word/phoneme timing was produced by automatic forced alignment. Therefore describe these as **human-reviewed transcriber segments initialized automatically**, not boundaries all drawn from scratch, and describe word times as **forced-aligned human transcript**, not manually timed words. The archive README also notes timing propagation failures caused by forced alignment. [Official transcription process](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml).

Available policies and recommendation:

- **Primary DER: lexical transcriber segments.** Resolve each segment's NXT child ID/range against its speaker's words XML; retain segments containing at least one real lexical `<w>` child, using the explicit `transcriber_start/end` bounds. Reject invalid/nonfinite intervals and noise-only or vocalization-only segments. Union overlapping same-speaker spans, clip to the unchanged[300,420)crop, preserve inter-speaker overlap and score the full120s UEM. This is a recommendation based on the human-reviewed speech segmentation and its closer fit to utterance-level diarization. It still includes within-segment pauses and may encompass vocalizations mixed with words; record that limitation. No model-based VAD or forced realignment should modify this reference.
- **Secondary diagnostic: union of forced-aligned lexical words.** Retain the previously derived word reference for Bark alignment and as a separately named sensitivity diagnostic. It removes inter-word gaps and can penalize a diarizer for marking a fluent utterance continuously. It is not the primary human-boundary DER. Do not adjust the gap-merge threshold using held-out outputs; if one is ever used, freeze it as a separate protocol before inference.
- **All transcriber segments / nonlexical-inclusive reference.** Technically obtainable, but it can count pure laughter/noises as speech, conflicting with the lexical-only interpretation and VoxConverse's laughter policy. Do not silently mix this into the primary set. Dialogue-act segments are intention annotations, not an automatic substitute for acoustic speech boundaries. [AMI annotation definitions](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml), [VoxConverse annotation guidelines in the primary paper](https://robots.ox.ac.uk/~vgg/publications/2020/Chung20/chung20.pdf).

For the fixed ES2004a[300,420)slice, lexical transcriber segments yield **27 intervals**, including6of at most1s; per-speaker annotated durations A46.595,B11.577,C39.891,D14.215s; speech union84.321s; two-or-more-speaker overlap24.223s(28.73%of speech union). The old word-union diagnostic is A40.130,B9.690,C32.380,D10.930s; union76.270s and overlap16.570s. The table above now labels the recommended transcriber reference explicitly. This is a reference-policy correction before inference; **no recording ID, input channel or crop span changed**, and original model-blind word-support selection remains documented. Independently validate the extractor and preserve its hash before scoring.

## Precisely what the checkpoint says about training exposure

Use the immutable model-card revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`, under **Training and Evaluation Datasets → Training Datasets → Real conversations**, not the later **Evaluation Dataset** table. Its explicit training rows are:

| Corpus | Training-table split/description | Consequence for this frozen set |
|---|---|---|
|AMI Meeting Corpus|Train and development; force-aligned|Corpus overlap alone does not prove ES2004a exposure.|
|ICSI|Full|Bmr005 is within the declared full-corpus training inventory.|
|VoxConverse v0.3|Development and test|All five selected test recordings belong to the declared training inventory.|

The separate training subsection **Data used to simulate multi-talker audio mixtures** also lists AMI individual headsets, Train and development. These are actual training statements; the later evaluation table is a distinct section. [Pinned raw model card, searchable section headings and exact rows](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#training-datasets).

The AMI authority lists ES2004 among **SC—unseen data for evaluation** in its scenario-only, full-corpus and full-corpus-ASR partitions. Thus the declared AMI Train/development scope is consistent with ES2004a being a checkpoint-training holdout **if the model used that same partition**. The model card does not identify a complete per-recording training manifest/partition mapping, and other licensed or pseudolabeled training collections are not exhaustively itemized by recording. The justified statement is “outside the declared AMI train/dev split under the official partition,” not a guarantee of zero exposure through every training source. [AMI authority's exact partition definitions](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).

This frozen set can test Gen2/native integration, FP32-versus-Q8 degradation, and generalization away from the locally tuned EN2002d crop. It cannot support a broad claim that the model generalizes to genuinely unseen recordings: six of its seven source recordings are explicitly inside the reported ICSI/Vox training inventory, and the remaining AMI meeting has only conditional split-level evidence. For a stronger unseen-data gate, add a **further separately frozen corpus with documented recording-level training exclusion**, or newly consented/licensed recordings created after the immutable checkpoint was published, with independently prepared human labels. A new dataset name alone is insufficient; verify source/identity duplication and all relevant audio sources. Keep those results separate from this local tuning-held-out benchmark. This is a requirement for the strength of that future claim, not a reason to discard or reselect the current frozen cases.
