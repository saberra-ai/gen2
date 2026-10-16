# Nemotron 3 Diarization support in Gen2

Status: N0–N3 implemented and verified on macOS arm64 CPU; N4/N5 remain open. Development receipts do not establish a supported desktop release.
Prepared: 2026-10-14.
Updated: 2026-10-16.

## Outcome

Add local speaker diarization with `nvidia/Nemotron-3-Diarization`: load the
model once, process a recording or continuous PCM stream, and return a timeline
of anonymous speaker turns, including overlapping speech. Then use those turns
to add speaker attribution to Bark transcripts in Pnola.

Use NVIDIA's **NeMo-Speech.cpp native diarization C ABI** behind a Rust module.
The shipped inference path requires neither Python nor a custom ONNX/MLX port.
Use the officially supported Python reference as an independent qualification
path (N0 uses pinned Transformers with original FP32 weights). This backend is a
separate inference operation alongside transcription, generation and decisions;
a diarization GGUF is not interchangeable with a chat GGUF.

Deliver a macOS Apple Silicon CPU baseline first, then qualify Metal. Add Linux
and Windows CPU release claims only after their own native packaging and
real-model checks pass. CUDA, Vulkan and physical mobile support are follow-ups.

## Baseline and dependencies

The main checkout was inspected at `1bc9f5b`, with unrelated working-tree changes.
The newer Bark implementation lives in sibling `gen2-bark-onnx`, branch
`codex/bark-onnx`, originally inspected at `6a659ff`. N0 implementation uses
`896640d` in the isolated `codex/nemotron3-n0` worktree. Reconcile this plan with the actual
merged code before implementing public APIs; names and paths below are proposed.

- [Bark roadmap](bark-onnx-roadmap.md) supplies the reusable runtime admission,
  cancellation and worker pattern. Its initial proposal is older than the
  inspected implementation. Standalone native qualification can begin before
  Bark merges; public runtime integration must build on its reconciled code.
- [Bark Transcription](/Users/victor/workspace/gen2-bark-onnx/src/speech/worker.rs:91)
  now exposes full text, token IDs and approximate token intervals on an 80 ms
  grid. These can include silence or have zero length; validated word timing
  and speaker attribution remain additional alignment work.
- [Bark Audio](/Users/victor/workspace/gen2-bark-onnx/src/speech/audio.rs:5)
  enforces a 60-second recording limit. Do not use that type unchanged for
  long-running diarization or relax Bark's limit globally.
- The runtime already admits speech and decision workers under the same budget
  as chat. Integrate diarization into that budget; do not reuse Bark's fixed
  4 GiB reservation as a model-independent estimate.
- Existing Pio diarization and alignment code is useful prior art, not a
  dependency on all of `pio-core`. See the
  [options research](../research/speaker-diarization-options.md).

## Evidence and initial pins

The model supports up to eight speakers, 16 kHz mono input and streaming state
across chunks. Recommended input buffers include 1.04, 0.64 and 0.32 seconds;
these exclude compute time. Output has a nominal 10 ms frame cadence, which is
not a claim of 10 ms speaker-boundary accuracy. Weights use OpenMDW-1.1.
[NVIDIA model card][model]

Freeze these inspected artifacts for the first qualification run; revalidate
availability, metadata and upstream code at N0 rather than following `main`:

| Artifact | Initial pin |
| --- | --- |
| NeMo-Speech.cpp | `083861c241061562b8ec29eb06d7f6cdc3af8775` |
| Its llama.cpp submodule | `bd4f514db14d87fded667787a7a963bfbaa98e89` |
| Hugging Face revision | `f667ed73aee57d40cc39428eb768b4fd87a0a29e` |
| Model filename | `Nemotron-3-Diarization.q8_0.gguf` |
| Model size | `107012128` bytes, approximately 107 MB download |
| Model SHA-256 | `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1` |

The model size/hash are published in NVIDIA's model index and Hub artifact
metadata. Download size excludes native libraries, working buffers, audio and
stream state. Include runtime and third-party notices separately from model
terms. [Pinned model index][index] [Hub metadata][metadata]

The ABI uses opaque model/stream handles and size-prefixed structs. It exposes
create/destroy, stream open/push/finish/close, segments and retained frame
probabilities. Speakers are **one-based**. A model must outlive its streams;
individual streams are single-threaded. No native cancel callback or stable
timeline watermark is exposed. [Pinned header][header]

## Architecture and caller contract

Expose a focused `gen2::diarization` module and an additive runtime loader.
Keep FFI, model validation, native state, scheduling and result conversion
private. Avoid making callers depend on NVIDIA structs or native pointers.

Proposed surface, to finalize against the merged API:

- `Runtime::load_diarizer(bundle, options) -> Diarizer` loads a verified model
  and reserves memory. Loading is explicit and offline.
- `Diarizer::diarize(recording, options) -> Diarization` accepts bounded PCM
  and internally feeds one stateful stream; it does not reset at ASR cuts.
- `Diarizer::start_stream(options) -> DiarizationStream` accepts incremental
  normalized PCM with bounded admission and observable backpressure.
- `push`, `snapshot` and `finish` distinguish input acceptance, provisional
  results and successful completion. Final names and sync/async ergonomics are
  settled during N2; do not promise async execution of a blocking native call.
- Results include speaker turns, processed-audio position, model/runtime
  identity, effective preset/backend and timings. Normalize native seconds to
  one documented Gen2 timebase with checked conversion; clamp only proven
  rounding/padding spill at the recording boundary, not arbitrary bad values.

Core invariants:

1. Speaker IDs are scoped to a recording/session. Normalize native IDs once;
   expose the convention consistently. The capacity is eight speakers, not an
   option that guarantees detection of a ninth speaker or correct counting.
2. Preserve overlapping turns; silence can return no turns. Validate finite,
   ordered intervals and valid IDs without forcing a single-speaker timeline.
3. Every stream owns its own speaker cache and audio cursor. Separate sessions
   cannot share diarization state. A returning speaker across an ASR chunk
   boundary remains in the same diarization session.
4. Live snapshots are revisioned and provisional. Give each snapshot a session
   ID and monotonic sequence; consumers replace the prior snapshot instead of
   appending duplicates. Processed audio is not a finality watermark. Only
   successful `finish` marks the full result complete.
5. Reject pushes after finish/close in Rust even though the native API treats
   pushes after finish as silent no-ops. Define repeated finish as returning
   the cached final result, without calling native finish twice.
6. Cancellation/deadlines are checked between bounded native calls. Never free
   a stream, library or reservation while a synchronous push/finish is running.
   Hard interrupt deadlines are not part of the initial in-process contract.
7. Default to one active job per loaded diarizer. A live stream holds that slot;
   competing work receives `Busy` rather than creating unaccounted sessions.
   Bound both queue length and total queued PCM bytes.

First input contract: finite mono float32 PCM, explicitly 16 kHz, with a
qualified recording length and per-push sample cap. Extract only truly shared
normalization helpers from Bark; format decoding and microphone capture remain
host concerns. Preserve Bark's independently qualified input envelope.

## Delivery sequence

| Milestone | Concrete result | Exit gate | Dependency |
| --- | --- | --- | --- |
| N0 — freeze and qualify dependencies | Locked model/runtime, independent oracle, native build and coexistence receipt | Real audio runs upstream; selected packaging safely coexists with Gen2's existing ggml | None |
| N1 — native speaker timeline | Rust wrapper and example returning final speaker turns | Actual Q8 model runs repeatedly on Mac CPU; timings/turns match the matched native reference | N0 |
| N2 — runtime integration | Public diarizer handle, pinned provisioning, admission and lifecycle | Public-call tests prove bounded memory, cleanup, cancellation and mixed-model admission | N1 + reconciled Bark/runtime baseline |
| N3 — long recordings and live streams | Incremental PCM, revisioned snapshots, flush and long-session behavior | State survives ASR cuts and native probability compaction; no dropped tail or unbounded queue | N2 |
| N4 — Bark/Pnola speaker attribution | Speaker-labeled transcript, rename mapping and timed playback | Alignment and overlap are evaluated on real multi-speaker recordings | N3 + validated Bark alignment strategy |
| N5 — supported desktop release | CPU/Metal receipts, packaging, feature CI and docs | Every advertised OS/backend passes real-model, offline and regression checks | N3; N4 additionally required for speaker-attributed Pnola claim |

N3 establishes standalone Gen2 model support. N4 establishes the distinct
product claim of speaker-attributed transcription. Neither is complete when
the GGUF merely loads.

### Implementation checkpoint — 2026-10-15

N1 now runs through `gen2::diarization` on the owning CPU worker. The real
AMI fixture matches every native reference turn and probability within 1e-6,
including A–B–A state reset. The 20 ms minimum runs successfully. Unsupported
ABI versions and missing symbols fail before publishing a handle. Public
contracts and examples are documented in [the guide](../diarization.md).

N2 has pinned atomic provisioning (three concurrent processes, corrupt-entry
repair, abandoned staging and offline warm reuse verified), shared admission,
failed-load rollback, cancellation and shutdown. Actual Bark + chat + native
diarization + synthetic Laya smoke-graph tests pass in both load orders,
including chat eviction, rejected restore at the budget ceiling and successful
restore after speech shutdown. One hundred actual model reloads also passed,
including a full-queue cancellation and dropped consumers. The same budget/eviction/restore checks also passed in both orders with
the full 1.6 GiB English Laya checkpoint (4096 MiB reservation), alongside
real Bark, chat and diarization weights. This establishes native execution and
admission coexistence; separate existing Laya receipts cover checkpoint quality. The one-hour run peaked at 248889344 RSS bytes (237.36 MiB), below the
conservative 1024 MiB reservation. Both review axes found only the transferred
allocation-capacity bound and CI submodule initialization issues; both were
fixed and the capacity regression passed. The [development receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-15-n1-n3/receipt.json) captures executed tests and both soaks. A real process was killed with 88958883 bytes in its incomplete GGUF; three subsequent processes recovered the cache, and corrupt-entry repair/offline reuse passed.

N3 now has a two-block queue, returned PCM on saturation, separate finish and
cancel controls, session UUIDs/replacement revisions, coalesced five-second
segment snapshots, cached finish, and a one-hour/100000-turn hard safety cap.
Uneven transport including one-sample splits matches the recording result
exactly. A 3600-second repeated-AMI stream completed in 777.77 seconds (RTF 0.2161),
with 360001 frames, 1501 turns, 683 observed provisional revisions, and a final
retained probability range starting at frame 296627 (63374 rows). This crosses
the native compaction horizon without losing the full timeline. The separate 1800-second run completed in 350.53 seconds (RTF 0.1947),
peaked at 248201216 RSS bytes, and retained 62825 probability rows starting at
frame 117176. Seven real-model public API tests, three model-free API tests,
and eight Python evidence/tool regressions passed. The combined default +
ASR/Laya/diarization/Metal suite passed 1348 tests; 43 real-environment tests
remained ignored in that ordinary run, and are not counted as executed evidence.
The soak ran alongside development builds/tests and is not a clean-machine
release performance receipt. No N4 or N5 completion is claimed.

Initial N4 evidence uses the AMI manual 1.6.2 word annotations (CC-BY-4.0,
archive SHA-256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`)
for the same 60-second EN2002d window. Grouping whole Bark token intervals gives
108 exact matched words of 147 reference / 117 predicted normalized words;
chronologically merged WER is 27.21% (31 deletions, 8 substitutions, 1 insertion).
Matched start error is median 80 ms / p95 210 ms, while matched end error is
median 100 ms / p95 1150 ms. These diagnostic word-end intervals span silence
and do not qualify as precise word boundaries. Attribution must preserve this
uncertainty or add a validated alignment path; proportionally distributing text
is not justified. Reproduce with `tools/nemotron-diarization/qualify_alignment.py`.
This single clip is an investigation fixture, not the held-out attribution corpus.
Source: [AMI annotation downloads](https://groups.inf.ed.ac.uk/ami/download/).

### N0: freeze dependencies and resolve native coexistence

- [x] Add a Gen2-owned model/runtime lock with the pins above, artifact hashes,
  ABI/header digest, compiler/SDK versions, build flags and license inventory.
- [x] Build only the needed native components. Upstream documents CMake 3.26+,
  C++17, SentencePiece and, for relevant packages, Abseil. Start with the CPU
  diarization preset; explicitly disable unrelated CLI/server/TTS/NMT/capture
  components in the eventual SDK package. Record transitive dependencies rather
  than assuming standalone diarization has no ASR-linked libraries. [Build guide][build]
- [x] Qualify the stable `NeMoSpeech::Diarization` / `diar.h` interface using a
  real multi-speaker clip, silence and a short tail. Inspect the actual exported
  symbols and deployment dependencies of the pinned binaries. [SDK][sdk]
- [x] Establish two references: pinned native Q8 execution for wrapper parity,
  and a locked official Python original-weight run for quantization quality.
  Compare matched streaming presets and postprocessing; do not require bitwise
  equality between quantized ggml and full-precision PyTorch.
- [x] Prove coexistence with Gen2's existing llama.cpp/ggml: load in both orders,
  run chat and diarization, unload/reload, and inspect symbol/library resolution
  in release binaries. NVIDIA brings a separately pinned ggml and patch series;
  an independent dynamic library is not proof of symbol isolation. [Patches][patches]

Preferred packaging is an optional dynamically loaded C ABI library through
`libloading`, following Gen2's existing native-runtime pattern. Keep the
library alive as long as any native handle exists. No model/runtime downloads
during `cargo build`; package native artifacts explicitly for release.

If coexistence fails, resolve it before N2: use a demonstrated single compatible
ggml build or explicit symbol isolation. If that requires a disproportionate
fork, document an isolated native worker-process alternative and its IPC,
memory accounting and failure semantics. Do not silently fall back to another
process or change the chat vendor pin just to obtain a successful link.

N0 receipt must choose the deployable route. Stop dependent integration while
that decision is unresolved; the output is evidence, not a stub claiming support.

### N1: prove the native Rust path

- [x] Add a private FFI module with audited layouts, null/status checks and RAII
  cleanup. Initialize struct sizes and all override sentinels explicitly:
  `left_context_frames = 0` is a real override, whereas negative preserves the
  preset; `gpu = -1` selects CPU. Read native error text on the calling thread.
- [x] Verify `num_speakers == 8` and the expected V3 output cadence against the
  pinned model; reject wrong checkpoints, missing symbols and unsupported ABI
  versions with typed errors before publishing a ready handle.
- [x] Implement a normalized-PCM path that opens one stream, pushes bounded
  blocks, finishes once, copies final segments and closes the stream. Start
  with the V3 preset, selected explicitly; retain effective settings in receipts.
- [x] Add `examples/nemotron_diarize.rs`: load the verified bundle once and emit
  machine-readable turns or RTTM. Raw per-frame output is diagnostic opt-in,
  bounded by explicit output limits.
- [x] Compare every speaker turn and, where retained, frame probability against
  the pinned native reference from identical PCM. Set numerical tolerances
  from repeatability evidence. Test A–B–A sessions to catch state leakage.

Suggested implementation locations: `src/diarization/{mod,error,types,bundle,
ffi,nemo}.rs` and `tools/nemotron-diarization/`. These are proposed module
boundaries, not a requirement for one file per trivial helper.

### N2: integrate provisioning, scheduling and resource lifetime

- [x] Add an optional `backend-diar-nemo` feature and native library path option.
  Feature-only/default-off builds must not require model weights or a C++ SDK.
  Keep GPU selection independent of Gen2's existing `metal`/`cuda` features,
  which currently enable chat backends. Unavailable requested devices fail
  explicitly; record actual execution placement.
- [x] Add revision-pinned provisioning under `hf`, following staging → size/hash
  validation → atomic publish. Keep runtime binaries and model cache identities
  separate. Exercise offline load, interrupted/corrupt downloads and concurrent
  provisioning; no inference-time network requests.
- [x] Add runtime reservations for loaded weights, native workspaces, active
  stream state, bounded audio and copied outputs. Use measured conservative
  peak usage, not the 107 MB artifact size. Hold the reservation through native
  destruction and shutdown; cover failed load and registry insertion cleanup.
- [x] Keep native calls on one owning worker. Reuse existing cancellation and
  deadline conventions where compatible, without generalizing Bark's worker
  into an unneeded public plugin system. Dropping a caller stops admission;
  an explicit wait operation observes actual native teardown.
- [x] Make `Busy`, invalid input/state, bundle mismatch, unavailable native
  runtime/backend, resource limit, cancellation and deadline failures typed.
  Success must not hide truncated audio or partially completed output.
- [x] Verify mixed Bark + diarization + chat + Laya load/restore/unload ordering
  against the shared budget. Account for GPU/unified memory and worker-process
  memory if N0 selected process isolation.

### N3: qualify continuous streams and long recordings

- [x] Keep a single cache/FIFO for the recording; transport blocks and Bark's
  60-second windows are not session boundaries. Validate sample counts and
  monotonic time positions. Reject changing stream rate rather than silently
  restarting state.
- [x] Provide bounded `try_push` admission with no dropped audio on saturation.
  Reserve a control path for finish/cancel so a full data queue cannot deadlock
  teardown. Slow consumers may coalesce provisional snapshots, never input
  audio or the terminal event.
- [x] Define revisions, cancellation/failed-stream terminal states, and final
  tail flushing. Snapshots have no invented stable-prefix guarantee. Include
  client reconnection/replacement semantics in the Pnola transport contract.
- [x] Measure memory past the native probability compaction horizon, around
  20 minutes, on at least a 30-minute session and a one-hour soak. Native
  segments still grow with the recording: cap session duration, result count
  and snapshot copies, and establish the supported envelope from measurements.
- [x] Bound result polling cadence and account for copy cost. The current ABI
  copies the entire retained probability history rather than a requested range;
  polling after every tiny push can create quadratic copying before compaction.
  Prefer segment snapshots initially; add a ranged native accessor only if a
  measured use case needs incremental raw probabilities. [Implementation][native]
- [x] Initially use native default segmentation settings for long sessions.
  The pinned API finalizes older probabilities with default postprocessing;
  custom settings apply only to the retained tail. Do not expose apparently
  uniform custom thresholds until that inconsistency is fixed or the feature
  is explicitly bounded to pre-compaction sessions. [Implementation][native]
- [x] Test uneven and one-sample transport splits, short final blocks, long
  silence, rapid turns and returning speakers. Under the same effective
  stream geometry, transport splitting must not change the final result beyond
  measured numerical tolerance.

Use the stateful streaming path for recorded meetings, including a large-buffer
V3 preset where useful. Upstream's full-attention offline operation has a
positional limit of about 6.6 minutes; an “offline” preset still uses streaming
state. Do not route all uploaded recordings through stateless full attention.
[CLI semantics][cli]

### N4: align Bark text and integrate Pnola

Development checkpoint: a conservative `attribute_bark` API and Pnola
recorded-file integration are implemented and reviewed. Pnola now returns
independent text/timeline outcomes, additive session/revision/provenance data,
session-scoped rename maps and source-audio hashes. Native tests exercised real
Bark+Nemotron output and both directions of partial success. Public unit tests
cover whole-token word grouping, offsets, overlaps, transitions, gaps and
edited-text rejection. A child cancellation token prevents diarization cleanup
from cancelling sibling transcription; a real native stream-drop test passed.

Actual desktop UI execution on AMI completed both models, persisted a two-word
speaker name across reload, and sought to 33.610999 seconds. Editing text cleared
word labels while retaining 27 turns and the rename map. A layout correction
keeps the speaker panel and text editor at full width. Native screen inspection
was blocked by the locked Mac; WebView/DOM checks and rendered artifacts were
captured. The debug GUI run was much slower than CLI (77.32 seconds diarization
plus 51.43 seconds ASR for 60 seconds of audio, versus 8.51 + 5.14 seconds in the
native test). N5 must investigate background/locked-screen behavior and measure
a committed release build; GUI real-time performance is not established.

The current clip produces 91 single-speaker, 21 unknown and 5 overlap words.
Speaker-tagged chronological word error is 46.94% (global known-speaker mapping;
word or speaker mismatch costs one substitution, unknown/overlap never match a
person). Conservative concatenated per-speaker WER is 65.31%, with unknown and
overlap words treated as an unmapped insertion stream. Text WER remains 27.21%.
AMI words are human-transcribed, but their timestamps come from automatic forced
alignment; boundary metrics are comparisons with that alignment, not manually
timed word ground truth. Human-reviewed AMI transcriber segments are used for
primary DER in the newly frozen protocol.
These are single-clip investigations, not a held-out attribution acceptance gate.
The 800 ms/50%/20% rules are conservative uncertainty policy, not a calibrated
forced aligner. Broader timing/short-turn/overlap evaluation, quality improvement
and physical microphone/native visual qualification remain open. N4 is not complete.

The live capture path now tees normalized PCM into a bounded continuous native
speaker session beside the durable eight-second ASR sections. Four adapter
blocks plus the Gen2 queue bound transport; saturation and the one-hour speaker
cap preserve ASR and expose retained labels as incomplete. Active UI revisions
are provisional. Stop flushes and saves final turns; process recovery marks old
active labels interrupted because the native cache cannot be restored. A real
60-second fixture fed at 20 ms microphone cadence preserved every saved sample,
published native revisions under one UUID, and matched final recording turns.
This is capture-path evidence with recorded audio, not a physical microphone
receipt or a live word-attribution claim.
The concurrent 80.04-second live worker run also passed: 11 durable Bark
sections, complete export, 15 observed provisional revisions and a final
35-turn timeline matching an independent recording call within 1e-6 seconds.
In-memory parity is exact; the persisted boundary comparison allows only the
established native-reference tolerance (JSON introduced at most 3.55e-15 seconds
rounding). A review regression now proves that history listing returns an
existing live session's authoritative snapshot, while true disk recovery marks
its lost cache interrupted. Ordinary Pnola tests pass 14 cases; 28 frontend
tests, clippy and source checks pass. [N4 development evidence](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n4/receipt.json)
retains the results and original rounding diagnostic. Quality, physical capture,
native visual verification and N5 release qualification remain open.


- [ ] At the start of this milestone, choose the alignment path using real
  evidence. Preferred: retain validated Bark token/frame timing and group words,
  or use a compatible forced aligner. Alternative prototype: transcribe speaker
  turns with bounded context and explicit duplicate reconciliation. Measure
  short-turn word loss before choosing that route.
- [ ] Bark cleans speech; timestamps cannot be assigned reliably by proportional
  character offsets. Separate timing evidence from cleaned presentation text;
  leave attribution uncertain when text/audio alignment is ambiguous.
- [ ] Assign timed words by measured temporal overlap. Preserve competing
  speakers during overlap and expose unknown/ambiguous attribution when needed.
  Diarization alone does not separate mixed voices or recover both transcripts.
- [ ] Extend Pnola's response/storage schema additively with session speaker IDs,
  timed segments, completion/revision state and model identity. Old text-only
  history remains readable. Allow explicit speaker rename mappings per session.
- [ ] Show final speaker turns and seek-to-time playback first. Add provisional
  live labels only when capture/transport supplies audio incrementally. Bark
  remains completed-recording ASR until its own streaming feature is qualified;
  live diarization does not imply live Bark transcription.
- [ ] Define partial success: if diarization fails, preserve a successful text
  transcript and show attribution as unavailable. If ASR fails, retain a valid
  standalone timeline where the user has chosen to save it.

This milestone spans the separate Pnola project. Gen2 owns model execution and
speaker timeline contracts; Pnola owns capture, file conversion, history,
playback, names and user-facing failure recovery.

### N5: qualify and release only supported combinations

Pnola now stages the pinned model/card/license and qualified native package,
places native code in Frameworks for the app's selected signing identity, and
retains original package provenance under Resources. Actual native minimum 26.0
is propagated into the app declaration and executable. Model corruption fails
before replacing existing staged resources. The relocated 658.70 MiB release app
passed deep signatures, 94 individual signed-framework/closed-loader checks and
an 80.04-second real-audio offline pipeline: 11 durable ASR sections, 15 observed
provisional revisions and 35 final native turns matching a recording reference.
Networking, source models/checkouts/SDK/Homebrew and the original app locations
were denied; independent controls proved the denials. 31 frontend/build tests,
14 ordinary Rust tests and 14 Python tool tests pass. See the [app-bundle development receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-pnola-bundle/receipt.json).
This is same-host relocation and native CLI execution, not fresh-machine,
physical microphone/native GUI, public notarization or clean performance proof.
A subsequent spec review reproduced hidden ORT-notice/demo paths in an isolated
checkout. Preparation now requires explicit artifact paths and publishes the
complete asset set after successful staging. Both fixture and real-artifact
preparation pass with the original Bark/Pnola checkouts denied; all 127 file
contents match the earlier packaged resources. See the [portable-build fix receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-portable-build/receipt.json).
The other N5 and quality gates remain open.


Mac CPU packaging development now has an explicit offline relocation tool. It
verifies the pinned SDK build receipt, copies the complete 92-library closure
(about 11 MiB), gives every non-system dependency a private loader-relative name,
includes runtime/component notices, removes source runpaths and verifies ad-hoc
signatures. Minimum-deployment metadata requires macOS 26.0 for these artifacts;
the measured host is 26.3. A relocated copy ran real native-reference parity with
Homebrew, original SDK/package reads and networking denied. The full eight-test
native lifecycle suite also passed in that sandbox, including 100 actual reloads,
stream parity/cancellation, ABI rejection and offline warm provisioning. Loader
traces account for all 92 private libraries. Thirteen model-free tool tests pass.
The [CPU packaging development receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-cpu-package/receipt.json)
retains manifests, controls, policy and executed output. This is local
relocation evidence, not a fresh-machine, app bundle, Metal, notarization or
cross-platform release receipt. N5 remains open.


Metal development now builds the pinned `metal-diar` preset with embedded shaders
and a distinct private loader namespace (93 libraries). `LoadOptions::metal`
resolves the actual Metal GPU ordinal and records Apple M4 Pro placement; a CPU
package cannot silently satisfy that request. The public wrapper matches an
independent native Metal run. A relocated copy passed with networking, Homebrew,
both original SDKs/packages/model sources denied; loader traces account for all
93 private Metal libraries. This same-host development test used 245448704 bytes
maximum RSS, not a clean performance or long-session receipt. Metal native default
DER on the smoke clip is 28.08%, still above the proposed release target. See the
[Metal development receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-metal/receipt.json).

The relocated CPU package also passed actual chat/Bark/full-Laya/Nemotron load
orders and shared-budget eviction/restoration with source models and networking
denied. That admission test runs short silence through speech/diarization and a
text decision through Laya; it is not a new real-meeting quality or performance
receipt. See the [relocated mixed-model receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-relocated-mixed/receipt.json).

Metal additionally passed the existing eight real native lifecycle gates,
including100actual reloads, streaming transport parity, cancellation, deadlines,
ABI failures and warm-cache reuse. These are development functional receipts;
committed release long-session timings/independent long-reference comparisons
are now being collected. The examples accept explicit `--metal` selection. Feature CI passed all12
OS/feature combinations plus evidence-tools at4892de8. The first release soak
found an example allocation-cap bug before stream input; explicit bounded block
allocation fixes it. A60s actual release calibration now matches all26turns and
48008probability values exactly. The long comparator now requires the exported
Gen2 probability tail and rejects excess native padding; long gates await the
repaired runs. See the [stream-gate development receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-stream-gate/receipt.json).

The [corpus investigation](../research/nemotron-heldout-corpus.md) freezes eight
local tuning-held-out target slices from seven recordings before model output,
including2/4/8speaker references and paired close/distant AMI channels. Exact audio
hashes/reference extraction must be frozen before scoring. Training exposure is
explicit: six recordings are in the model's declared training inventory; ES2004a
has conditional official evaluation-split evidence. This set supports local
integration/quantization checks, not a broad unseen-recording claim. Stronger
unseen-data qualification still needs separately excluded or newer recordings.

Repaired committed Metal release streams now pass both30m and1h against an
independent C-header executable, with exact final turns and every retained
probability value, including after compaction. The [release stream receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-metal-release-stream/receipt.json)
records RTF, cold instance load, native push median/p95, peak RSS, revisions,
source/library/model/audio hashes and source/network-denial policy. This proves
long-stream parity and throughput on M4Pro; capture/UI end-to-end delay, broader
quality and other release combinations remain open.

Committed CPU release streams now also pass30m and1h against the independent
native executable:753/1501turns and502600/506992retained probability values
match exactly after compaction. RTF is0.19146/0.18871 and public peak RSS is
241238016/245252096bytes. Cold instance load and native push distributions are
reported, with ordinary OS/background and warm file-cache limits retained.
See the [CPU release-stream receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-cpu-release-stream/receipt.json).
These are repeated-fixture throughput/parity results, not capture/UI delay or
varied-meeting quality. No N5 completion is claimed.

The eight local tuning-held-out slices are now acquired and immutable before
inference:1896.992audio seconds with exact WAV/reference/source hashes, full
audio UEM and retained original annotations. AMI uses human-reviewed lexical
transcriber segments; its word times remain a separately named forced-alignment
diagnostic. ICSI includes substantial real digit reading. Vox auzru's26ms
annotation overrun is explicitly clipped to actual WAV length; no selected
recording, channel or crop changed. See the [corpus freeze receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-corpus-freeze/receipt.json).
The committed protocol pins all scoring/reference tools before model output.
Review repairs bind CPU/Metal public/native/FP32 output to the actual normalized
PCM digest and verify installed oracle package versions and Transformers commit.
Rebuilt release runners passed an excluded1.117s smoke calibration: native112
frames, FP32mask111frames, actual PCM identity and exact native/public parity.
See the [calibration receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-quality-calibration/receipt.json).
The13-job feature matrix passed at125f907. Frozen corpus scoring is in progress;
no held-out quality outcome is inferred from acquisition or calibration.

Linux/Windows CPU development jobs now build the pinned source SDK with an
explicit x86_64AVX2/FMA/F16C baseline, retain installed dependency discovery,
compare real public/native output and require all seven native lifecycle tests
to be compiled and executed. Windows public loading now searches the selected
absolute DLL directory for its imported siblings without changing process PATH;
a separate default-loader negative control exercises the original failure route.
Two actual Windows compiler-environment setup failures are retained as diagnostic
evidence, not qualification. Real SDK/public reruns are pending. These jobs do
not establish closed portable packages, fresh-machine offline operation or
mixed-model coexistence on those platforms.

- [ ] Run Mac CPU first; compare Metal against the same Q8 reference and real
  annotated audio. Record device placement/fallback rather than treating a
  successful Metal build as acceleration evidence.
- [ ] Qualify Linux and Windows CPU separately before advertising support.
  Include installed SDK dependency discovery, loader paths, clean-machine
  offline operation and coexistence with the existing Gen2 native backends.
- [ ] Add feature CI: no backend, diarization only, ASR + diarization, and chat +
  ASR + diarization. Run applicable existing formatting, clippy, tests and
  Bark/Laya/chat qualification gates after shared/runtime changes.
- [ ] Real-model qualification must fail clearly when its required assets are
  absent. Ordinary model-free unit tests can remain small; a skipped model
  test is not a release receipt.
- [ ] Publish examples, capabilities, supported input/session limits, optional
  runtime setup, offline provisioning, cancellation limits, provisional labels,
  known quality weaknesses and model/runtime notices. Update `api_spec.md` when
  the public contract is finalized.

## Verification and release criteria

Use public Gen2 calls for lifecycle and user-visible behavior; use the FFI seam
for ownership/status failures, and a pinned upstream executable for independent
native-output comparison. Stub-only tests do not establish model support.

| Area | Required evidence |
| --- | --- |
| Wrapper correctness | Final turns/probabilities match pinned native Q8 execution with identical PCM, preset and postprocessing; explained numerical tolerances |
| Quantization quality | DER against hand labels and original-checkpoint reference, with the same collar/overlap rules; record every dataset slice |
| State and streaming | Tail flush, silence, repeat finish, push-after-finish rejection, no inter-session cache leakage, returning speakers across transport/ASR boundaries |
| Lifecycle/resources | Queued/running cancellation, deadline expiry, dropped consumer, failed load, busy admission, shutdown/reload and mixed-model reservations |
| Long sessions | Real run beyond compaction; bounded PCM/probability/snapshot storage, documented segment growth and explicit duration/output caps |
| Attribution | Speaker-attributed word error, boundary errors and missing/duplicate words on Bark/Pnola recordings; overlap/unknown labeling reviewed |
| Deployment | Fresh-machine runtime resolution, license packaging, corrupt-cache behavior and network-disabled inference for every advertised platform/backend |

Freeze a licensed corpus covering two-, four- and eight-speaker audio, far-field
speech, noise, interruptions, overlap, short replies and silence. Keep a held-out
real meeting set distinct from tuning examples and synthetic regressions.

Proposed gates, to confirm with N0 measurements before implementation claims:
**zero unexplained wrapper mismatches**, no more than **2 absolute DER points**
of Q8 degradation versus the matched original-checkpoint reference on the frozen
corpus, and **warm real-time factor below 1.0** for a named supported live
configuration. Measure per-slice DER so an aggregate cannot hide a failed domain.
The N0 release proposal is DER at most 20% on each held-out target meeting slice,
with zero collar and overlap included, plus the two-point quantization bound.
The current single-clip smoke result does not meet that default-quality target;
it proves native compatibility, not readiness to enable the feature by default.

Report cold load, warm median/p95 push time, actual end-to-end label delay,
snapshot revisions, RTF, peak RSS, resident memory and a 100-session reload/soak
result. Separate input-buffer delay from compute and UI delay. Record CPU/GPU,
thread counts, OS, Gen2 SHA, runtime/library hashes, model lock digest, PCM/audio
hashes, effective preset, postprocessing and reference environment in
`benches/results/nemotron-diarization/<machine>/<run-id>/`.

## Reviewable work packages

1. N0 locks, native/reference runner, fixture manifest and coexistence receipt.
2. N1 audited FFI, PCM inference and standalone timeline example.
3. N2 pinned provisioning, runtime loader, reservations and lifecycle behavior.
4. N3 streaming transport, revised snapshots and long-session qualification.
5. N4 alignment qualification, then Pnola schema/UI integration.
6. N5 Mac CPU/Metal release receipts and docs; separate Linux/Windows packages.

Each package carries its own tests and evidence. Recheck the exact public
baseline before opening an implementation branch. No calendar estimate is
committed before the native coexistence and alignment investigations finish.

## Deferred scope

Physical iOS/Android deployment, CUDA/Vulkan qualification, a direct Rust/MLX or
ONNX model port, speaker enrollment across sessions, automatic real names,
source separation, more than eight speakers, full-precision/other quantizations,
and hard native-call interruption. None is implied by the first desktop release.

## N0 completion receipt — 2026-10-15

The five N0 items above are complete for the macOS CPU baseline. See the
[N0 implementation and reproduction commands](../../tools/nemotron-diarization/README.md)
and [measured receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-15-n0/receipt.json).

- Verified model and fixture bytes against the immutable lock; built only the
  selected CPU diarization SDK plus the upstream example and diagnostic shim.
  Toolchain, native exports and actual library dependencies are recorded.
- Ran the upstream C example on a real licensed AMI meeting, plus the direct
  public-C-ABI probe on the meeting, silence, a short tail and A–B–A sessions.
  All 25 native reference segments agree within upstream RTTM rounding.
- Ran an independent pinned Transformers/torch CPU FP32 oracle from the
  official original safetensors. This replaces the proposed heavyweight NeMo
  installation for N0; the actual environment is locked in `reference/uv.lock`.
- Both native load orders, model reloads, an isolated chat baseline and
  concurrent Gen2 chat/diarization calls passed with unchanged results. The
  selected route is an optional in-process, host-supplied dynamic SDK. macOS
  loader/namespace evidence supports this decision only for the tested stack.
- With explicitly matched low-latency geometry (chunk 9/right 4/FIFO 264/cache
  264/update 222), native default segmentation DER is 30.38% on this excerpt.
  Identical external thresholding gives Q8 33.38% and FP32 33.45%, with four
  speakers in each output. These are a smoke comparison, not a model ranking.
  The two-point Q8 degradation bound passes on this clip. The SDK's differently
  configured V3 default yields 25.22% DER and three speakers; it is a separate
  configuration, not evidence about quantization quality.

N0 also establishes two constraints for N1: retain the padded final raw frame
in diagnostics but clip presented intervals to actual audio duration; package
the complete native dependency closure, including host SentencePiece/Abseil
dependencies, rather than distributing only the C wrapper. N1 is the next
milestone; no public production diarization API is shipped by this change.

[model]: https://huggingface.co/nvidia/Nemotron-3-Diarization
[metadata]: https://huggingface.co/api/models/nvidia/Nemotron-3-Diarization/tree/f667ed73aee57d40cc39428eb768b4fd87a0a29e
[index]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/models/index.json
[header]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/include/nemo_speech/diar.h
[native]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/c_api.cpp
[sdk]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/docs/sdk.md
[build]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/docs/build.md
[patches]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/patches/README.md
[cli]: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/docs/cli.md
