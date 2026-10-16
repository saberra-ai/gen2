# Proposed V3 finish-time valid-mel crop and regression gates

2026-10-16; initial source-only design. At that research stage no native checkout, production code, frozen receipt, model lock or qualifier was edited and no build, frontend or neural execution occurred. The subsequent isolated source candidate is recorded below. The implementer reports the separately retained calibration graph control at sourcef8f7d71: native39valid/HF full maximum error0.000422716; native40explicit-zero is bitwise identical to39; replacing only the zero slot with the computed terminal mel row produces0.24104017maximum error against HF and0.24105334versus zero. Those results motivate a candidate framing fix; they do not prove universal HF/NeMo equivalence or pass the original AMI quality gates.

## Smallest candidate change

Patch only the V3 branch of [DiarStream::finish](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.cpp#L142-L164), retaining the existing geometric raw tail and feature extraction. Capture the **original** global audio endpoint before appending padding. After `ensure_mel()` and before `run_ready_chunks(true)`, crop the retained mel buffer to the original valid feature endpoint. Initial proposed shape (subsequently applied only to the isolated candidate recorded below):

```cpp
// After the existing finished_ guard, before appending decay samples:
const size_t original_audio_end = audio_base_ + audio_buf_.size();
// Existing finished_ assignment, geometric-tail append and ensure_mel unchanged.
// Immediately before final scheduling:
if (m_.cfg().is_v3()) {
    const int64_t valid_mel_end = static_cast<int64_t>(
        original_audio_end / static_cast<size_t>(m_.fe().hop_length()));
    if (valid_mel_end < mel_base_ || valid_mel_end < mel_consumed_ ||
        valid_mel_end > mel_produced()) {
        throw std::runtime_error("DiarStream: invalid v3 final mel bounds");
    }
    mel_buf_.resize(
        static_cast<size_t>(valid_mel_end - mel_base_) *
        static_cast<size_t>(n_mels_));
}
run_ready_chunks(/*end_of_stream=*/true); // existing code
```

The crop changes graph input/context before inference. Cropping only emitted probabilities would retain the terminal input's attention/convolution influence and fail to address the demonstrated calibration mechanism. Do not change `produce_new_mel_frames`, shared FE masking, feature stacking, graph heads, thresholds, AOSC update or the V2 birth gate. Do not delete the geometric raw padding: valid final FFT windows still need right-edge padding after preemphasis. Leave `diarize_offline` outside this patch's scope; this proposal corrects V3 streaming finish semantics only. [Original NeMo valid-length evidence](/Users/victor/workspace/gen2-nemotron3-n0/docs/research/nemotron-original-nemo-tail-reference.md), [HF/native tail source audit](/Users/victor/workspace/gen2-nemotron3-n0/docs/research/nemotron-partial-tail-control.md).

## Buffer and lifecycle safety argument

- [Buffer invariants](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.h#L208-L222): `audio_base_` is the global sample index of the first retained sample; the vector is a contiguous suffix. Its base+size is therefore the cumulative real endpoint even when lazy front erasure has occurred. No new persistent counter or reset field is needed. Compute it before adding256decay samples, not from final vector size and not from retained size alone.
- [Mel invariants](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.h#L211-L222): `mel_buf_` holds `[mel_base_,mel_produced())`. Resize with **global endpoint minus mel_base_**, multiplied by128; resizing to `valid_mel_end*128` would be wrong after front compaction. Do not move `mel_base_` or `mel_consumed_` during the crop.
- Before padding, [feature readiness](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/features/fe.cpp#L784-L790) is `floor((N-256)/160)+1` for enough audio, and never exceeds `floor(N/160)`. The committed scheduler consumes no more than produced mel. Thus it cannot already have committed the one terminal row being removed. After256padding samples, readiness is `floor(N/160)+1`; the proposed crop removes that final computed row while retaining all original valid rows. The explicit bounds check prevents silent underflow or expanding a buffer with synthesized zeros if assumptions are violated.
- [Window retention](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.cpp#L267-L280) keeps mel from at most `consumed-left_context` onward and raw audio from `keep_mel*hop-256`. Clipping only the back preserves left context and the real last sample required for padding, including with nonzero left context. The final scheduler's `w0>=mel_base_` check remains authoritative.
- [Provisional flush](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.cpp#L179-L260) operates on a copied AOSC state and does not advance `mel_consumed_`. It preserves right context and rounds provisional center to the encoder grid. It cannot preview beyond original valid mel. The existing first nonempty final chunk removes/replaces the preview; leave that logic intact. If final valid endpoint equals committed endpoint, no preview can require replacement: a nonempty preview would require additional produced mel beyond it. Do not manually shorten `probs_`, adjust `probs_base_` or change `provisional_frames_` at finish.
- Empty audio yields valid0/produced0 and no model call. Tiny nonempty audio below160samples yields valid0 after clipping, also no model call. Reset clears base/buffers/consumption/provisional/finished state; double finish remains idempotent and later feed remains a no-op. Probability timeline compaction is independent of PCM/mel front trimming and must retain global `n_frames()` accounting. [Existing lifecycle implementation](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/diar_pipeline.cpp#L106-L164).

These are source-level arguments, not exercised invariants. Keep the patch narrow and require candidate SDK qualification before admitting its output convention.

## Meaningful regression design

**Primary real-model regression, calibration only.** With the patched candidate SDK and unchanged17872sample fixture, require111finite eight-channel output rows. Also verify common-prefix probabilities against a direct native-graph reference that bypasses `DiarStream`: construct valid native mel111rows from the same fixed frontend/config/bank; run first104mel with empty state; use its first9preencoder embeddings as the final FIFO (no cache compression on this fixture); run final39mel with that FIFO; concatenate first outputs `[0:72]` and final outputs `[72:111]`. This independent assembly uses the same model and raw frontend but explicitly supplies the correct graph boundary. A false fix that runs40computed rows and merely clips112outputs to111will pass the count assertion but fail the probability comparison around frame107. Use a predeclared same-runtime numerical policy, retaining maxima and all input/output hashes; do not invent a threshold after seeing patched outputs. Separately retain the prior HF/FIFO teacher-forced evidence; it is a different state-control boundary.

**Chunk/lifecycle regression using the same model instance with fresh streams.** Compare whole-file push, deterministic uneven pushes and a sequence calling `flush_available` between pushes. Require the same final row count, common-prefix outputs under a fixed policy, zero provisional remainder after finish (via behavior/stable frontier), `stable_frames()==n_frames()`, unchanged second finish, ignored later feed and reset/replay equivalence. A non-final preview may differ temporarily; compare finalized output and verify previously reported stable rows never change. Include a fixture schedule that actually produces a preview; with9/4context this requires at least40ready mel rows beyond the committed cursor. Reuse the calibration's two-call tail to keep neural work bounded.

**Boundary count/geometry cases.** For native streaming tests, cover `N={0,1,159,160,161,319,320,321}`, center boundary `72*160±1`, and the104mel region `N={16639,16640,16679,16680,16735,16736,16799,16800}`. Also cover the calibration17872 and a stack-divisible end such as `N=112*160`. Expected V3 final global count is `floor(N/160)`; assert no empty model invocation. The16680HF first-call sample size and16736native104FFT-readiness point are distinct. At valid104mel, native final scheduling may run104mel(first72+32right) then32mel; do not assert it is one call. HF's single first-and-last call can select a different top-level mode when cacheNone/lookahead absent, so this boundary suite proves native invariants, not universal HF session equivalence.

**Front compaction and left context.** Use enough synthetic audio and small repeated pushes to force actual PCM/mel front erasure, then finish after a non-hop-aligned tail; compare final count and retained tail with a whole-file-push reference. Exercise left0 and one valid nonzero-left geometry to catch `valid_mel_end-mel_base_` and `w0` mistakes. Existing [compact_front tests](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/tests/cpp/asr/test_diar_state.cpp#L217-L264) already verify lazy suffix invariants, but do not exercise this new finish crop; duplicating only the integer formula is insufficient. If forcing probability compaction, use a long silent gap and existing `set_compaction`; compare global count, frame base and frozen+retained segment behavior, rather than expecting the retained vector alone to contain the whole stream.

**V2 preservation.** Keep all crop behavior inside `cfg().is_v3()`. Run existing V2 streaming/lifecycle fixture tests when the pinned predecessor model is available; retain exact old frame/probability behavior. If that model is absent, report the missing execution coverage rather than acquiring it or claiming V2 parity from branch inspection. The shared FE and V2 scheduler remain byte-unchanged by design.

The smallest indispensable executed regression is the calibration count **plus probability-level** comparison and provisional/idempotence replay. Boundary/compaction checks can reuse the model without repeated loads. Test code should exercise the patched finish implementation; avoid creating a production API or duplicating an entire scheduler solely for tests.

## Qualification and provenance boundary

Retain pristine upstream083861source identities and the new small patch payload/hash separately; an out-of-tree compatible build can preserve the original checkout. Bind candidate SDK/compiler/options/library hashes and new tool/protocol identities. The current frozen native count contracts intentionally expect112/12001 and must remain unchanged as historical evidence. A candidate protocol may expect111/12000, but cannot rewrite old results or reuse an old lock to imply the runtime is identical. Do not run targets or broaden deployment until the patch's review and calibration admission gates are complete; later target/quality/platform qualification is distinct. No claim that this alone fixes the AMI20% proposed gate or every numerical difference is justified.

## Explicit contract migration, after candidate qualification

This is an externally visible frame-contract change even though the C ABI signatures stay the same. Current [Rust results validation](/Users/victor/workspace/gen2-nemotron3-n0/src/diarization/nemo.rs:202) derives `samples/160+1` and requires that count on finish. The frozen [native probe](/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/native_stream_probe.cpp:93) independently requires the same count. [compare_stream](/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/compare_stream.py:27), [capture_integration](/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/capture_integration.py:21), corpus protocol metadata and native tests also encode it. A patched SDK cannot be silently dropped into the current runtime and called compatible; current strict validation should reject it.

Use this sequence rather than changing historical comparators to make a new result pass:

1. Preserve the original SDK, source qualification, probes, protocol/receipt files and their old112/12001count expectations. Freeze a separate candidate patch/source identity, SDK/package identity, explicit V3 `floor(samples/hop)` contract, candidate probe and candidate comparison suite. Include both exact upstream source input hash and compatibility-patch hash in build provenance. Do not allow either old/new count as an unbound alternative; the expected contract must follow the selected verified runtime artifact.
2. Admit the candidate through reviewed build-only checks and calibration graph/finish regressions first. The candidate native probe must assert111calibration rows, partial-hop cases and strict native10ms dimensions; it must not rewrite or execute the old count-locked probe as though its failure were numerical evidence. Preserve old and new failures/successes separately. Keep reference annotations, UEM, scorer and quality thresholds unchanged.
3. Only after candidate admission, migrate production Rust's finished-count expectation and related current tests to the newly pinned runtime contract in an explicit implementation change. Preserve pre-finish upper-bound and cancellation/lifecycle checks; update only the frame rule they intentionally bind. Test minimum public20ms input (320samples), non-hop-aligned inputs, both cold and provisional finish, double finish, late push, reset, front compaction and retained/global frame accounting. Core-native zero/tiny tests and Gen2's stricter20ms validation are separate contracts.
4. Rebuild/repackage and requalify every supported platform/backend profile with candidate source/patch/model/package bindings, exact wrapper/native probability agreement, lifecycle, coexistence and offline deployment controls. Historical platform qualification of pristine083861does not qualify the candidate. Rerun the unchanged frozen quality corpus only under a new candidate receipt/protocol after prerequisite admission; compare Q8 against its appropriately matched FP32 control and retain original failures beside it. All-platform and quality gates remain open until actual results close them.

The production native test expectation sites identified by the implementer (including test lines83/143/299) need an implementation-time inventory tied to the then-current commit. This note records the inspected validation/probe files above; it does not edit them or assert the inventory is exhaustive.

## Isolated source candidate, before build admission

The reviewed1194-byte [compatibility patch](../../tools/nemotron-diarization/patches/v3-finish-valid-mel.patch), SHA256 `8c1434c14f63ebeb2515c719a90594124fd4f75af85b80451fd145b5caef3c36`, is now applied only to `/Users/victor/.cache/gen2-nemotron3-n0/NeMo-Speech-valid-mel-v1`, an independent local clone at083861with pristine llama bd4f. The patched pipeline SHA256 is `fe00ae415f4741d7e07579acdc77c60f5d7327be024cf7f8aff8f52142ae9af2`; its complete root diff equals that patch, with no other tracked root-source change. The original checkout, SDK, locks and frozen evidence remain unchanged. Source preparation required no network acquisition, build or model execution. A separately frozen build-only protocol will construct a fresh CPU SDK and run the existing model-free diar_state test; it does not admit the candidate for inference or production use.


## Candidate CPU construction receipt — 2026-10-16

The [build-only receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-candidate-build/qualification.json)
records successful configure, compile, install and one actual `diar_state` test
(1/1 passed) at Gen2 source `1bc12ce2291743a9d5b423db5fb1e00eeb2719ce`.
The frozen protocol SHA256 is
`27210ba8cdd8592ffa1871f13f83197f8bc76b01a7d4c38034309cc4704ddc34`.
Both review axes independently verified all 131 collected members, 15 installed
SDK paths and the runtime static archive. An independent archive plus 19-patch
replay reproduced all 3,425 materialized GGML compiler inputs. The complete
compiler database and phase logs are retained. The large source archive and
compiled artifacts remain local and hash-bound.

This is construction evidence only: no model was instantiated, no frontend or
patched finish ran, and neither inference nor production substitution is admitted.
The applied C++ source passes `git diff --check`; the literal unified patch's
last blank context row necessarily contains its single context marker space.
Keep that frozen patch payload unchanged.


## Fixed candidate finish regression, before neural execution

The separate `valid-mel-finish.protocol.json` admits one CPU helper invocation
only after review and commit. Its fixed scope is 28 stream sessions: four
calibration push schedules, one reset/replay, 21 silence boundary lengths, and
two nonzero-left-context calibration schedules. Four direct graph calls assemble
independent valid-Mel references. Internal stream forward counts are not
instrumented; the fixed inputs, schedules and geometry bound that work.

The same-runtime maximum absolute probability policy is **0.0001**, declared
before any candidate neural output. This applies only to candidate pipeline
versus independently assembled candidate graph probabilities; it does not change
an HF, DER, quantization or release criterion. Require 111 final rows on the
17872-sample calibration, `floor(N/160)` on each silence case, an actual preview,
unchanged stable prefixes, reset equivalence and unchanged second finish/late push.
The whole and uneven calibration cases also trigger actual front erasure before
finish by the pinned `compact_front` source rule. No probability-history
compaction or long-session qualification is implied.

The helper is already compiled without execution. Compiler sandboxing denied
model/audio access. Its 101912-byte binary SHA256 is
`e2c99b3e15c29f49484238ab6d5cb72f7649fa5228ad068e5bbf3141e82dfb2d`,
linked against the newly constructed SDK and the 235360-byte runtime archive
SHA256 `824d8f4771c907ddccbb98b62bb1af462c357d476b610980df1973f545a11cfb`.
The protocol binds compiler command, source, SDK libraries, archive, construction
receipt and exact calibration PCM/Mel/model identities. Available direct tensors
and per-push/final stream snapshots persist before validation, with started and
completed events; failed attempts remain intact. No candidate model or finish
has executed at this preregistration point. Production migration and all original
quality, platform, desktop and performance gates remain open.


## Executed candidate finish regression — 2026-10-16

The [candidate finish receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-finish/qualification.json)
now records one successful invocation at source
`0ec641aea9ec66a3549d2db1aafa2649febf09df`, under protocol SHA256
`fc7d7900b04fc7bb41ffea593dfedc1f1bbea38af92e40eed602945e99667eb6`.
All 28 fixed stream sessions and four explicit direct graph calls completed in
2.440346 seconds including admission/controller work. Each of the six
calibration pipeline/direct comparisons is **byte-identical**, maximum error
zero, within the preregistered 0.0001 policy. The calibration returns 111 rows.
All 21 silence boundaries return `floor(N/160)`, including zero rows for
0/1/159 samples, two rows for the public minimum 320 samples, and 112 rows for
17920 samples.

Saved per-push tensors establish an actual provisional result, unchanged stable
prefixes, final stable frontiers, unchanged repeated finish/late push and exact
reset replay. Direct first/full/final probabilities, preencoder embeddings,
FIFO and Mel inputs are retained. Both review axes independently verified all
387 collected files and recomputed the saved-array invariants without running
another model or helper. No audio, model, SDK or helper binary is in Git.

This closes the fixed **CPU/F32 candidate finish** regression only. The native
V3 correction is exercised rather than inferred from an output crop. It does
not qualify Q8, other backends/platforms, probability-history compaction, long
sessions or production Rust. At the receipt source, Rust still required `floor+1` and rejected this
candidate intentionally. Subsequent artifact/contract implementation is separate
from that fixed receipt and from model/platform/quality qualification.
No HF/original-NeMo model, AMI target or DER scoring ran; original quality and
release gates remain unchanged.

## Production migration seam and current inventory

The separate [runtime profile](../../tools/nemotron-diarization/runtime-valid-mel.lock.json)
is an implementation input, not a new artifact qualification. It retains the
original lock's upstream/model pins, binds the one compatibility patch and
patched pipeline, and adds the upstream-supported CMake definition
`NEMO_SPEECH_VERSION_METADATA=gen2-valid-mel-v1`. New builds must report exactly
`nemo-speech-asr 0.2.0+gen2-valid-mel-v1`. The currently qualified candidate SDK
reports the plain upstream version and cannot be silently relabelled: construct
a distinct tagged SDK, bind its real outputs, and qualify it separately.

The minimal migration keeps the existing trusted, explicit library-path API.
Change Rust's version admission to exact full-string equality for that profile
and require strict `floor(samples/160)` final counts. The compatibility check
runs **after library loading, before model creation**. It is a semantic guard,
not cryptographic verification of a library closure. Build and package receipts
must still verify the actual source patch and every relocated dependency; a
matching C shim alone cannot establish that its ASR core has the same contract.
Do not add a permissive alternative accepting either final count. A separate
public manifest-verification API would be broader work than this migration.

Inventory at source `f3ea770`:

- Runtime: `src/diarization/ffi.rs` currently strips all version metadata;
  `nemo.rs` requires the old extra frame. Keep cancellation, provisional bounds,
  retained/global accounting and resource checks. Current real-model tests in
  `tests/diarization_native.rs` expect 6001 rows and three rows for 320 samples;
  the negative ABI fixture must distinguish legacy/wrong metadata from a valid
  profile missing required symbols.
- Construction: retain `runtime.lock.json` as historical. Add explicit profile
  source/patch admission and tagged builds to Mac `build_sdk.py` and
  Linux/Windows `platform_native.py`; the latter also chooses the count-locked
  native helper and comparator. Windows LF compatibility and private naming
  remain separate recorded overlays. Existing frozen probe/comparator files
  must remain unchanged; add versioned valid-Mel tools for current qualification.
- Packages: Mac, Linux and Windows packagers currently bind the old lock.
  New package metadata and verification must bind the new profile, patch and
  frame contract while preserving closure/notice/CRT checks. Linux and Windows
  fresh-container qualifiers also import the old comparator, lock and installed
  helper identity; migrate these together with SDK construction. The central
  Windows workflow's immutable old SDK run cannot be reused as a candidate.
- Pnola: model-asset verification can retain the old unchanged model pins;
  runtime-package validation needs the new profile. Two real transcription
  tests in `overlay/src-tauri/src/transcription.rs` expect 6001 rows. Existing
  persisted results/receipts remain historical; new outputs expose the runtime
  version that produced them.
- Qualification: use new protocols and receipts for Q8, tagged CPU/Metal,
  all-platform wrapper/native equality, lifecycle, long sessions, coexistence,
  offline desktop deployment and the unchanged corpus gates. Preserve all
  original failures. This inventory authorizes no inference beyond the completed
  fixed CPU/F32 run and records no production migration as complete.


The first Rust migration change now requires the exact tagged version before
resolving model symbols and strict floor-valid final counts. It preserves the
existing conservative unfinished/provisional upper bound and all interval,
retained-range and resource validation. Model-free shared-library fixtures reject
plain upstream, arbitrary suffixes, an extra candidate suffix and a different ABI;
the exact profile reaches the missing-model-symbol check. Frame-admission tests
reject both missing rows and the historical extra row, including partial hops
and the one-hour endpoint. Current native integration expectations now use two
rows for 320 samples and 6000 for 60 seconds plus 17 samples.

Validation of this source change: 1030 library tests and 11 API/alignment/native
negative-evidence/cancellation tests pass; four library tests and nine
external-asset native integration tests are ignored. Clippy and formatting checks pass. The shared-library fixtures call only a
version function and never load a model. This is **not** tagged-SDK/Q8/public real-model qualification. Builders,
packages, current comparison tools, platform workflows and Pnola migration remain
pending. The old plain-version candidate is still inadmissible to current Rust;
construct and qualify a distinct tagged SDK before any real-model public run.


## Tagged CPU SDK construction preregistration

`prepare_valid_mel_source.py` and `build_valid_mel_sdk.py` now provide the next
construction step. The source preparer creates an independent local clone and
applies only the locked patch; it rejects existing source/evidence destinations.
The builder requires a fresh source build, SDK prefix and evidence directory,
all disjoint, and verifies source pins, sole diff and absence of untracked or
ignored compiler inputs. Shared `runtime_profiles.py` verifies the historical
lock, unchanged model/fixture pins, explicit patch provenance and exact semantic
profile. Seven mutation/admission tests pass; the model-free evidence-tool suite
now totals 121 tests.

The frozen `valid-mel-tagged-build.protocol.json` and external launcher admit
only a new macOS arm64 CPU construction at:

- source `/Users/victor/.cache/gen2-nemotron3-n0/NeMo-Speech-valid-mel-tagged-v1`;
- SDK `/Users/victor/.cache/gen2-nemotron3-n0/sdk-valid-mel-tagged-cpu-v1`;
- evidence `/Users/victor/.cache/gen2-nemotron3-n0/valid-mel-tagged-build-v1`.

Build phases repeat the independent GGML patch replay and before/after compiler
input audit, run the actual model-free state test, and query only the C ABI's
version function in a fresh process. Expected version is the exact identifier
required by current Rust. Network and model/audio roots are denied for source
preparation and build-phase children; the Python bookkeeping parent validates
source/metadata and performs no model work. It avoids nested seatbelt setup by
passing the strict profile to each build child. No model, frontend or finish
execution, inference admission, packaging, public Rust qualification or release
claim is included in this construction protocol.
