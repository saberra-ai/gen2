# Nemotron 3 Diarization support in Gen2

Status: N0–N3 implemented and verified on macOS arm64 CPU; N4/N5 remain open. Development receipts do not establish a supported desktop release.
Current runtime migration: the tagged valid-Mel CPU v2 package passes
[60-second parity and seven lifecycle gates](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-macos-runtime/qualification.json).
The [30-minute and one-hour CPU qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-cpu-long/qualification.json)
now passes exact retained-tail parity after compaction, provisional revisions and
peak RSS below 249 MB on repeated smoke audio. The platform and debug Pnola
receipts below cover this profile; full application, original quality and
release criteria stay open.
The [tagged Metal SDK construction](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-construction/qualification.json)
passes source/shader provenance and model-free state/version checks. The
[fixed Metal runtime qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-runtime/qualification.json)
passes exact 6,000-row/26-turn parity, seven lifecycle gates including 100 reloads,
CPU-only SDK rejection, and sequential CPU rejection followed by Metal parity
in one process. Simultaneous residency and desktop application qualification
remain pending. The [Metal package construction and relocation](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-macos-package/qualification.json)
passes with all 93 private libraries.
The [first Metal long-session attempt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long-failure/qualification.json)
passes the 30-minute pair but fails overall after a disk-space/output-persistence
interruption during the one-hour native command. One-hour terminal tensors and
status are unavailable; its public run never started. That attempt remains failed.
The separate [guarded Metal continuation](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long/qualification.json)
passes two fresh one-hour streams and reuses the saved passing 30-minute pair.
Both durations have exact retained-probability/turn parity after compaction,
bounded revisions and peak process RSS below 261 MB. The reserve and storage
checks completed normally; no passing 30-minute neural run was repeated.

The [fresh Linux/Windows SDK protocol](../../tools/nemotron-diarization/valid-mel-platform-native.protocol.json)
is now frozen for tagged CPU construction and same-host native/public lifecycle.
Its [model-free source preflight](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-valid-mel-source-preflight/qualification.json)
passes independent 19-patch replay, 3,425 materialized files and all 98 saved CPU
compiler inputs. All 157 model-free tool tests passed at that source. The
[actual Linux candidate SDK](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-valid-mel-native/qualification.json)
now passes 40 commands, exact 6,000-row/26-turn parity and seven lifecycle gates.
Its compiler audit binds 3,425 files and all 97 actual compiler units; delivery
also preserves SDK bytes, modes and aliases. The
[actual Linux v2 construction](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-valid-mel-runtime-construction/qualification.json)
now passes eight-library closure/source mappings and source/notice provenance;
its archive restores with exact files, modes and aliases. The
[separate runtime protocol](../../tools/nemotron-diarization/valid-mel-linux-runtime.protocol.json)
freezes the exact SDK and constructed package for offline native/public/lifecycle
then actual CPU chat coexistence. The [actual Linux offline/chat qualification](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-valid-mel-offline-chat/qualification.json)
passes seven offline and three chat containers, exact 6,000-row/26-turn parity,
seven lifecycle gates, six unchanged diarization outputs across both load orders,
unchanged isolated chat text and overlapping API calls. Both load orders retain
720 observed private GGML bindings. The [actual Windows candidate SDK](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-valid-mel-native/qualification.json)
passes 43 commands, exact 6,000-row/27-turn parity and all seven lifecycle gates;
66 delivered SDK files and private PE/CRT provenance verify. Its fresh
[runtime protocol](../../tools/nemotron-diarization/valid-mel-windows-runtime.protocol.json)
binds five-DLL v2 construction, disposable central-VC setup and offline/chat
execution. The separate [actual Windows runtime receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-valid-mel-offline-chat/receiver-qualification.json)
now passes v2 five-private-DLL construction, publisher VC installation and fresh
System32 observation, offline native/public parity, all seven lifecycle tests,
both missing-dependency controls and CPU chat coexistence in nine disposable
Windows Server 2022 containers. All 6000 rows and 27 turns match exactly across
native/public and six chat comparisons. Receiver collection checks the actual
artifact digest, opaque runtime/helper archive bytes, frozen command specs and
source pins. Observed RTF around 1.06 is descriptive; Windows live performance,
physical desktop, Bark/Laya, quality, redistribution and release remain open.
The [Mac mixed-model protocol](../../tools/nemotron-diarization/valid-mel-mixed.protocol.json)
separately freezes CPU/Metal residency and admission smoke tests with actual
Bark/Laya/chat. It does not establish concurrent inference or real-speech quality.
The [actual mixed receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-mixed/qualification.json)
passes all four cases, complete runtime/provider image audits and initial/restored
29-layer Metal chat placement. The [Pnola adapter protocol](../../tools/nemotron-diarization/valid-mel-pnola.protocol.json)
now freezes ten recorded/partial-success/capture/live/shutdown cases against
explicit relocated artifacts. The [actual adapter qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-adapter/qualification.json)
passes all ten cases, six exact turn comparisons and complete library audits.
Both 80.04-second live runs retain 11 ASR sections, 152 words and 15 provisional
revisions. This is debug CLI/test evidence; signed-bundle resolution, GUI,
physical microphone and quality criteria remain open.
The [complete-app construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-app-construction/qualification.json)
now retains complete CPU/Metal ad-hoc build and static verification, with
94/95 physical framework entries representing 92/93 native components plus
the public entry copy and ORT provider. The original controller and first
Metal-only storage-admission failures remain failed; CPU compilation was
reused. The successful Metal-only continuation executes no model and uses a
separate construction storage policy. Five-GiB neural admission is unchanged.
The verifier checks exact framework siblings, signed payload mapping and
packaged model/license/Bark identities. These metadata checks do not establish
application inference, automatic resolution or GUI behavior.
The separate [complete-app runtime protocol](../../tools/nemotron-diarization/valid-mel-pnola-app.protocol.json)
now freezes five CLI cases against both exact app inventories: automatic
recorded and paced-live resolution per backend, then explicit Metal rejection
by the CPU package while preserving Bark. Source/SDK/build/Homebrew/network
reads are denied to each app; app and staged inputs are read-only. Actual
framework telemetry, recorded/native parity and saved compact-live timelines
are required. The [first attempt remains failed](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-app-trace-failure/receiver-qualification.json):
its CPU recording command returned0 with exact 6000-frame/27-turn parity and
117-word Bark text, but hardened signing suppressed the required environment
dyld trace. The remaining four commands never started. A model-free signed C
probe independently reproduces the suppression. A future versioned attempt
requires process-local image diagnostics while retaining hardened signing;
no full-app inference gate is promoted. GUI and microphone remain separate.
The [model-free callback proof](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-process-image-callback/qualification.json)
now verifies the actual Rust diagnostic module under hardened signing:
47 unique images, idempotent registration and a future load/unload are observed
while the environment trace remains suppressed. Only explicit verification
commands register this callback. Parsing rejects malformed reserved records;
the future qualifier requires callback history and the same complete closure.
The original app builds and failed neural attempt remain immutable; this
model-free proof does not qualify app inference.
The [instrumented CPU/Metal construction](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-instrumented-pnola-apps/qualification.json)
now passes all six build/clone/signature commands and complete static checks.
Both retain hardened signing and the exact original local entitlements;
315 Gen2 source pins and 107 actual Pnola source files are retained. Identical
model files were replaced with APFS clones to recover space, preserving every
app payload byte, inventory and signature. The separate
[v2 runtime protocol](../../tools/nemotron-diarization/valid-mel-pnola-app-v2.protocol.json)
freezes the same five cases on these newly built apps, requiring process-local
image history with no environment trace dependency. The
[actual complete-app qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-complete-pnola-apps/receiver-qualification.json)
now passes all five cases. Automatic recorded CPU/Metal resolution matches
6000 frames and 27/26 turns with zero turn error and exact Bark text. Both
80.04-second live runs retain 11 ASR sections, 152 words and 15 provisional
revisions, matching 35/34-turn saved timelines; independent recording parity
is asserted internally. No separate live probability tensor is delivered.
Callback histories match 462 native components and five ORT records across
the five processes, including the exact CPU-package Metal-rejection error
while preserving Bark. The original attempt remains failed. These hardened
same-host CLI results do not establish GUI, microphone, fresh-machine, quality,
uncontended performance, public signing/notarization or release qualification.
The separate current-profile corpus scorer preserves every valid row and all
original quality criteria. Its [frozen runner protocol](../../tools/nemotron-diarization/valid-mel-corpus.protocol.json)
now binds 33 tools, 258 unchanged compiled Rust sources and 306 staged entries.
It schedules 32 fresh native/public CPU/Metal processes across the same eight
slices, reusing only the exact eight original hash-locked CPU FP32 oracles.
Native children cannot read staged oracles or labels. Scoring retains all
quality failures and uses the unchanged original gates, with every valid row
included. The [completed current-profile corpus qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-frozen-corpus/receiver-qualification.json)
now retains all 32 successful processes and 16 byte-exact native/public pairs.
All Q8-versus-FP32 degradation bounds pass, but only 12 of 16 pairs meet every
quality gate. Both AMI channels fail the unchanged shared and production DER
thresholds on CPU and Metal. CPU production DER is 33.2146%/43.8407%; Metal is
33.2235%/43.8229%. This reuses the previously observed fixed corpus and exact
original FP32 arrays, so it is not a fresh held-out evaluation. All valid rows
are scored; historical failures remain unchanged and unwaived.
All 182 model-free tools tests pass locally;
[CI at the current 182-test source](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-182/qualification.json)
passes all 13 jobs, including fmt, clippy and tests across 12 OS/feature pairs.
This model-free CI does not execute app/model/corpus qualification. Quality,
GUI, physical microphone and the remaining release criteria stay open.
The separate [original-NeMo environment receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-environment/qualification.json)
passes source/wheel/installation checks and offline class imports without model
or frontend execution. A [one-forward excluded CPU calibration protocol](../../tools/nemotron-diarization/original-nemo-calibration.protocol.json)
now binds that environment and strict local checkpoint restore; it is a separate
numerical diagnostic, not a quality-gate replacement. The [actual short calibration](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-calibration/receiver-qualification.json)
now passes strict restore and one CPU forward, retaining all111valid rows.
Maximum probability difference from the saved official HF output is0.000250578;
frontend values differ and are retained. Original-NeMo corpus parity and all
original quality/release criteria remain unqualified. A separate
[two-crop original-NeMo protocol](../../tools/nemotron-diarization/original-nemo-ami.protocol.json)
now freezes one strict CPU forward per unchanged AMI channel to compare every
valid row and report the same reference-policy DER. The [actual original-NeMo AMI diagnostic](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-ami/receiver-qualification.json)
now completes both strict CPU calls. Original shared DER41.5071%/50.0045%
also fails20%, alongside cached HF41.4447%/50.0223%. All12,000valid rows are
compared; probabilities are unequal and raw discrepancies are retained.
This later official3.1path does not identify the checkpoint's training3.0source.
No old quality gate or release status changes. All185model-free tests pass.
The [185-test source CI receipt](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-185/qualification.json)
now passes all 13 jobs, including all 12 OS/feature combinations. Local model-free
tests now total 191. The separate [portable Mac construction plan](../research/nemotron-macos-fresh-machine.md)
freezes a fresh VM SDK/package build with an explicit ARM target and admits no
model execution. Its sandbox regression exposed an earlier
import-policy wildcard error, preserved in a separate correction receipt.
The original AMI quality failures, GUI/microphone, receiver runtime, performance
and public release gates remain open; main remains unchanged.
The [first portable Mac attempt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-construction-failure/qualification.json)
now preserves a validator failure after all 24 native commands passed on an M1
virtual machine. CTest changed its summary wording; terminal audits, version
query and packaging never ran. No SDK/runtime snapshot or model is admitted.
A separate frozen v2 construction retry retains the same criteria with a parser
for both observed single-test summary formats. Local model-free tests total 193.
The [actual portable v2 construction](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-construction-v2/qualification.json)
now passes on an M1 virtual machine: 98 explicit-target compiler units, exact
source checks, state/version and 92-library private packaging. Byte-only
collection checks the opaque GitHub artifact, exact restored inventories and
all original/private library sections. No model ran. A passing exact-artifact
receiver qualification is still required; no accuracy or release gate changes.
The [191-test source CI](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-191/qualification.json)
also passes all 13 jobs. The later [193-test parser source CI](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-193/qualification.json)
now also passes all 13 jobs. Local tests total198 with the receiver artifact
admission controls and the actual macOS child-loader regression.
The separate [M1 receiver protocol](../research/nemotron-macos-fresh-machine.md)
now freezes the exact construction artifact for offline native/public parity
and all seven lifecycle gates on a fresh VM. Its admission tests reject changed
artifact identity/bytes/members and unsafe destinations. The [first actual receiver attempt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-offline-runtime-failure/qualification.json)
failed before model admission: all helper builds and controls passed, then the
strict image auditor rejected the sandbox launcher included by misplaced
tracing. No inference or lifecycle case ran. A separate v2 protocol fixes
child-only tracing, retaining the unchanged package, cases and strict image
closure. Its model-free C regression fails before and passes after the fix;
the fresh v2 receiver has no completed passing qualification yet. All quality and release criteria remain
unchanged.

A separate [portable CPU complete-app candidate](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-construction/qualification.json)
also passes local model-free construction: 100 static/clone/signing commands,
unchanged main payload/models/ORT/hardened entitlements, and independent byte
comparisons of all 94 Frameworks against the exact portable package/ORT.
The 211-file receipt makes no app execution, fresh-machine, GUI/microphone,
performance, accuracy or public signing/notarization claim.
The [current receiver-source feature CI](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-198/qualification.json)
passes all 13 jobs at `7db170d`: all 12 OS/feature combinations and the evidence,
Pnola and desktop asset checks. Linux reports 198 tools tests, 195 passing and
three platform skips; all 198 pass locally on macOS. This does not replace the
separate real-model or desktop release gates.

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

A later fixed paired-AMI recording check uses the actual relocated signed CPU
app on ES2004a [300,420) for both close and distant microphones. Both complete
all three ASR parts with one continuous 120-second timeline; 39/36 turns exactly
match the saved qualified CPU results. Full-word speaker-tagged error is 49.63%
close / 55.88% distant, and conservative cpWER is 64.71% / 65.81%. Chronological
lexical WER is 33.82% / 48.90%: 81/108 deletions, 3/1 insertions and 8/24
substitutions. Matched-word end-time p95 error is 440/530 ms against automatic
forced-alignment word times; this timing subset
does not represent missing/substituted words. No uncertain or omitted word was
dropped from the word-error metrics. The initial part-count wrapper assertion
and a failed resume hashguard are retained; neither required repeated model
execution. See the [paired attribution receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n4-ami-paired/receipt.json).
This verifies functional multi-part app behavior while leaving transcript and
attribution quality open, and remains CLI/same-host evidence.

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

The actual signed Metal app initially produced recorded/live results but aborted
with a Metal residency-set assertion during process exit. A real native Engine
regression reproduced the asynchronous shutdown race while retained job handles
still existed. Pnola now waits for native worker destruction, fences late loader
publication and guards CLI early returns. The rebuilt `024a88f` app passes the
original one-second reproducer and 60-second recording with clean process exits;
all 26 recorded turns match the independent native reference exactly. Its
80.04-second paced live verification also exits cleanly, with 11 completed ASR
sections, 152 words, 15 provisional revisions and 34 persisted final Metal turns.
Requested Metal against a CPU-only SDK preserves complete ASR and explicitly
reports unavailable speakers. Exact native/model tree hashes match the failed
candidate. The original app failures and one verification-harness field error
are retained in the [shutdown regression receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-metal-app-shutdown/receipt.json).
These are same-host offline CLI checks; native GUI, physical microphone,
fresh-machine deployment and accuracy gates remain open. Earlier full-Laya/Bark/
chat mixed tests passed in both orders with a Metal diarizer, but recorded only a
Metal-enabled chat build, not chat placement; they do not establish both-GPU
coexistence. A separately frozen rerun now requires successful chat generation
and public placement metadata after initial loading and restoration in both
full-Laya/Bark/Metal-diarizer load orders: chat reports 29/29 layers, `MTL`, Apple
M4 Pro in all four checks. The diarizer independently reports Metal/device.
Shared-budget refusal, eviction/restoration, native shutdown and zero speech/
Laya/diarization reservations pass. The [placement receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-metal-mixed-placement/receipt.json)
records this loader-parameter/registry evidence; it is not per-kernel telemetry
or additional model-quality qualification.

The relocated CPU package also passed actual chat/Bark/full-Laya/Nemotron load
orders and shared-budget eviction/restoration with source models and networking
denied. That admission test runs short silence through speech/diarization and a
text decision through Laya; it is not a new real-meeting quality or performance
receipt. See the [relocated mixed-model receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-n5-relocated-mixed/receipt.json).

Metal additionally passed the existing eight real native lifecycle gates,
including100 actual reloads, streaming transport parity, cancellation, deadlines,
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
The frozen set is complete at `125f907`: all 16 CPU/Metal wrapper comparisons match
native turns after required terminal clipping and probabilities exactly; every
Q8-minus-FP32 bound is at most 2 percentage points.
Twelve of 16 slice/backend pairs pass all quality gates (all five Vox cases and
ICSI). Both AMI channels fail the 20% target on both backends: CPU production DER
is 33.2146% close-mic and 43.8407% distant-mic. ICSI production DER is 19.3362% CPU /
19.9881% Metal, close to the boundary. Failed slices remain in the unchanged
protocol. See the [full corpus receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frozen-corpus/receipt.json).
The [quality-gap investigation](../research/nemotron-quality-gaps.md) corrects an
earlier reference-policy recommendation: NVIDIA publishes AMI scores against
specific forced-alignment RTTMs, not the broader transcriber segments used here.
That mismatch materially affects missed-speech scoring, as quantified below;
it does not cure the original frozen failure. A separate published-reference
reproduction must retain these outcomes. That preregistered diagnostic is now
complete using saved outputs, with no inference or tuning: the exact pinned NeMo
continuous scorer gives CPU shared-threshold DER of 16.8135% close / 27.4897% distant
against published labels, compared with 41.4640% / 49.9537% against the original
transcriber labels. Production-default DER remains 27.6171% / 35.4388% against
published labels. Thus reference convention explains much of the score but does
not resolve the distant-mic or production-default quality deficit. All four
synthetic independent-scorer calibrations pass. See the [diagnostic receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-reference-diagnostic/collection.json).
The original frozen failures and release status are unchanged. The
[next-diagnostics source audit](../research/nemotron-quality-next-diagnostics.md)
confirms that the retained FP32 comparison already uses official HF/PyTorch CPU
execution at the recommended 9/4/0/264/264/222 geometry. It distinguishes the
legacy native postprocessor from that geometry and proposes saved-output error
localization, frontend/model numerical controls, one fixed session-prefix
contrast and a canonical context comparison on separate development data. These
were proposals at that source audit; neither a native F32 control nor
original-NeMo execution is claimed. The first proposed saved-output localization is now complete at
`5113a30`: all12 channel/backend/profile scores and four official synthetic
controls reproduce, with exhaustive additive bins using each full-UEM speaker
map. Distant CPU shared Q8 error is40.7240% within reference-overlap bins and
50.7493% within short-only reference-span bins, versus18.8983% in single-speaker
bins and12.6830% in long-only span bins. These use different conditional
denominators and do not exclude silence false alarms from the full total.
The distant remaining90s still has26.8872% shared Q8 DER, so the failure is not
confined to initial30s. Production false activity in reference silence is
9.037 close /8.190 distant speaker-seconds, versus2.439 /2.429 for shared Q8.
See the [11-file localization receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-ami-error-localization/qualification.json).
This diagnoses already-seen inputs, not a new release setting or passed gate.
The [CPU golden-control recipe](../research/nemotron-cpu-golden-control.md)
subsequently identifies the frontend bank, frame-mask and aligned V3 tensor
boundaries. Its first whole-centered frontend-only comparison is now complete
at `ec88564`, with no neural-model load/inference or checkpoint deserialization:
two synthetic controls pass before both unchanged AMI targets, four frontend/bank
combinations per input preserve all raw features/masks, and all12 comparisons
were independently recomputed from retained local arrays. Native and HF banks
differ in504 of32896 F32 values (maximum0.000117321). Using the same bank reduces
target mean feature error to approximately1.15–1.62e-6; close-mic retains one
value above the fixed0.001 absolute flag under each bank, maximum0.00112629.
The [42-file metadata receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frontend-control/qualification.json)
retains that outlier, all source/input/library/loader bindings and the numeric
limits; raw tensors remain outside Git. This neither selects a replacement bank
nor proves streaming/model/cache equivalence or a quality fix. The separate
F32-dominant artifact is now constructed at `23d2db7`: all363 original checkpoint
tensors are BF16; manual independent mapping/cast accounting verifies every one
of360 stored tensors (359F32 and oneF16 subpixel convolution), plus the three
intentional source exclusions. All stored values are finite and its native bank
matches the original Q8 bank exactly. The [14-file construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-construction/qualification.json)
retains the395298848-byte artifact hash, exact source/checkpoint/dependency pins,
logs and full tensor accounting; checkpoint/GGUF bytes remain outside Git.
This performs checkpoint deserialization/conversion, not neural inference, and
upcasting BF16 source values does not recover missing source precision. Native
F32 inference is subsequently complete at `03bc8a0` under its separately frozen
fixed-CPU protocol: the excluded1.117s calibration and both original AMI crops
run in three fresh native processes, with actual PCM/config/frame/range binding.
The [36-file numerical receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-execution/qualification.json)
retains all raw native vectors and nine independently recomputed comparisons.
F32/HF mean probability error is0.000104958close /0.000008018distant, versus
Q8/HF0.000344023 /0.000725374; maxima remain0.044496 /0.005974. The short
calibration retains0.240977maximum F32/HF difference near its end, so no universal
numerical-equality claim is made. Native terminal rows stay retained and separate
from the HF common valid prefix. This includes native neural inference, reuses
saved official HF/Q8 arrays and performs no DER scoring or gate promotion.
Original-NeMo, teacher-forced graph/cache and persistent-prefix controls remain
open. The separate [eight-file source-weight audit](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-hf-nemo-weight-identity/qualification.json)
now verifies all417 HF F32 tensors exactly against the pinned mapping/cast of
363 source BF16 tensors. Its source-backed addendum identifies four stored
native activity-head tensors unused by the pinned graph, preserving historical
construction accounting. This closes parameter identity only; frontend, runtime
arithmetic, framing and cache effects remain separate. Local model-free tests
now total108, including terminal/partial-tail/range and stack/conservation regressions.
A separately preregistered streaming-frontend/tail control at `5204386` now
executes five frontend-only cases under the shared retained native bank. Its
[44-file metadata receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tail-framing/qualification.json)
binds91 local files, all five actual CPU frontend commands and ten reconstructed
final stacks/masks. Both120s targets have48HF versus49native final Mel rows,
producing6HF versus7native encoder inputs; their common final Mel maxima are
1.90735e-6close /1.23978e-5distant. The short calibration has39HF versus40native
rows in five encoder inputs, with HF zero-padding the slot native fills. No
neural model/cache/projection ran; native scheduling is an explicitly
source-derived replay of observed frontend readiness. This establishes a framing
boundary without attributing probability errors or changing quality gates.
The separate [original-NeMo source audit](../research/nemotron-original-nemo-tail-reference.md)
finds that official V3 source `218d0b4` also crops to floor-valid Mel rows before
stacking, matching HF counts. This later official source declares3.1.0; the
checkpoint's3.0.0 label does not pin its training source. Actual original-NeMo
CPU environment, restore and numerical execution remain unqualified.
The [calibration-only teacher-forced graph control](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-calibration-graph-control/qualification.json)
now completes at `f8f7d71`: twoHF CPU forwards and four fresh native graph calls,
with identical39real final Mel rows and cloned HF FIFO. Native internal39-row
padding and explicit40th zero row give byte-identical full outputs; adding only
the computed terminal Mel row changes native probabilities by maximum0.241053343
at frame107/channel0. Native/HF valid39maximum is0.000422716 versus0.241040170
with that terminal row. All14saved-array probability/embedding reports were
independently recomputed. The prior loading-report serialization failure is
retained separately with zero forward/native calls. This supports a separately
qualified V3 valid-Mel implementation correction; no production patch, AMI-target
rerun, DER or quality-gate change occurred. Local evidence-tool tests total114. A separate
[CPU candidate construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-candidate-build/qualification.json)
now records a successful isolated build of the reviewed V3 valid-Mel finish patch,
including actual model-free state tests and independently replayed compiler inputs.
The subsequent [fixed CPU/F32 finish regression](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-finish/qualification.json)
passes all28stream sessions and21boundary lengths. Six calibration/direct graph
comparisons are byte-identical, with actual provisional/reset/stability checks.
Q8, production artifact/count-contract migration and all-platform/quality
requalification remain pending; no original quality gate is changed.
The first Rust migration step now admits only the exact tagged runtime identifier
and floor-valid final counts, with 1030 library and11targeted model-free tests
passing. The [tagged CPU construction receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tagged-valid-mel-construction/qualification.json)
retains a late Homebrew metadata failure and its separately reviewed read-only
recovery; compile, install, actual state-test and actual version-query phases
passed. The separate [tagged fixed-case qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tagged-valid-mel-qualification/qualification.json)
now passes both28-case F32/Q8 suites and independent C-API/public Rust parity on
320silent samples and17872calibration samples. All12native/direct comparisons,
the F32 rebuild comparison and both public/native probability arrays are
byte-identical; final row counts are2and111. Fifteen model-free preflights passed.
This is short CPU calibration/silence qualification only. Package/platform/Pnola,
full lifecycle/long-session/coexistence, Metal, AMI quality and original release
gates remain open. The subsequent [macOS v2 construction/relocation receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-macos-package/qualification.json)
binds92private libraries and all6SDK identities to the current patch/profile.
Relocated structural verification passes with original libraries denied; no
native library or model was loaded. That receipt covers structural verification only.
Pnola's updated staging overlay is applied with18model-free Rust tests passing
and5real-model tests still ignored. The [separate packaged CPU runtime run](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-macos-runtime/qualification.json)
now passes full 6000-row/27-turn native/public parity and all 7 lifecycle tests,
including 100 reloads. Nine runtime processes each load the exact 92-image private
closure; the missing-library control fails before model creation. An earlier
sandbox admission failure is retained with zero neural calls. Long-session,
coexistence, Metal, other platforms, Pnola and quality/release gates remain open.
The 13-job feature matrix also passed at `4974318`. Later source `cf64214`
passed12 jobs; its macOS mixed-feature job failed before compilation when GitHub
submodule fetch could not connect to port443. A failed-job rerun was denied by
the token's Actions permissions. Fresh source `ec88564` triggered a new full
matrix, and [all 13 jobs now pass](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix/qualification.json),
including the formerly blocked macOS job and102 evidence-tool tests. The original
fetch failure remains retained; no build gate is waived. The later
[c729cea matrix also passes all 13jobs](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-108/qualification.json),
including108evidence-tool tests. Subsequent calibration tools and114local tests
are outside that CI source. The subsequent [114-test matrix](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-114/qualification.json)
at `9e98003`, run38912826242, now passes all 13jobs. The later valid-Mel
candidate tools remain outside that CI source. The later [121-test matrix](../../benches/results/nemotron-diarization/cross-platform/2026-10-16-feature-matrix-121/qualification.json)
at `c15a0b9`, run38920681529, passes all 13jobs and covers strict tagged-runtime
Rust admission plus construction/recovery tools. The subsequent fixed-case
qualification tools are outside that CI source.

Linux/Windows CPU development jobs now build the pinned source SDK with an
explicit x86_64AVX2/FMA/F16C baseline, retain installed dependency discovery,
compare real public/native output and require all seven native lifecycle tests
to be compiled and executed. Windows public loading now searches the selected
absolute DLL directory for its imported siblings without changing process PATH;
a separate default-loader negative control exercises the original failure route.
Two actual Windows compiler-environment setup failures are retained as diagnostic
evidence, not qualification. Linux now passes actual public/native parity on
the60s smoke input:26turns and48008probability values match exactly, with actual
PCM identity checked. All seven required lifecycle tests execute and pass,
including 100 reloads; uploaded SDK file hashes were verified. See the
[Linux development receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-native-development/collection.json).
System SentencePiece remains an installed dependency. The Windows rerun at74c663e
passed compiler setup and dependency provisioning but failed upstream CMake's
first patch application; no model execution occurred. Its [failure receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-configure/collection.json)
is retained. That controlled Git-LF retry also failed before native compilation.
Captured patch bytes prove that upstream CMake's text-mode write emitted CRLF
against LF source: the exact failed patch reproduces the failure locally, while
normalizing only CRLF restores the pinned patch hash and applies successfully.
A guarded, build-only Windows compatibility overlay now materializes LF with
`file(CONFIGURE ... NEWLINE_STYLE UNIX)` and checks the generated bytes against
the unchanged normalized content before patching. It changes no native C++ or
patch payload. See the [investigation](../research/nemotron-windows-patch-lines.md)
and [captured failure receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-patch-materialization/collection.json).
That actual retry now applies all 19 patches and compiles native objects, but
fails during linking because CMake Ninja selects ambient MinGW for the intended
MSVC/vcpkg build (`c++.exe` rejects `-ignore:4221`). The LF correction is confirmed;
Windows model/native qualification remains incomplete. The [compiler-selection failure receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-compiler-selection/diagnostic.json)
retains the actual CMake cache and link diagnostics. A separate reviewed fix
binds CMake C/C++ explicitly to the absolute active MSVC `cl.exe`. That retry
successfully built and installed the SDK (65 delivered regular files verified),
then failed launching the standalone probe because its bare `cl.exe` was not
resolved from the child process environment. Absolute tool resolution fixed
launch; the next retry exposed global `/TP` parsing the import archive as C++.
The reviewed command now passes that archive after `/link`. Both failures and SDK byte checks are retained
in the [probe-build diagnostics](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-probe-build/collection.json).
The next actual MSVC retry builds the probe and Rust helpers and runs the locked
Q8 model successfully (27 native turns, 6,001 frames). It then stops at the
loader negative control: CPython rewrites error 126 without preserving
`winerror`. The control now reads the typed Win32 API directly and still
requires exactly error 126. Raw native outputs, 66 verified installed files,
and the failed control remain in the [native loader-control receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-loader-control/qualification.json).
The subsequent `4bd34aa` retry passes exact Windows native/public parity (27
turns, 6,001 frames, zero probability/boundary error), the explicit Win32 error
126 control, and all seven lifecycle gates, including 100 reloads. See the
[Windows development receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-development/qualification.json).
Observed CI RTF is about 1.20, so this does not qualify live performance.
Closed packaging, private DLL coexistence and fresh-machine operation remain open.
These jobs do not establish closed portable
packages, fresh-machine offline operation or mixed-model coexistence on those
platforms. The [Linux packaging investigation](../research/nemotron-linux-packaging.md)
sets out the next implementation gates: exact non-glibc dependency closure,
license/source provenance, relative ELF resolution, archive metadata, and a
clean offline container plus actual chat/GGML coexistence. This is a plan,
not a new portable-package or clean-machine result.

A portable [Windows development constructor](../../tools/nemotron-diarization/package_windows.py)
now preserves the qualified SDK's five private DLLs and eight unchanged VC
DLLs, installed notices and retained provenance. It checks actual SDK/PE/cache
identity and performs a bounded ZIP round trip into a Unicode path. All 80
model-free packaging/evidence regressions pass. The actual `abcef06`
[Windows construction receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-runtime-construction/qualification.json)
passes 25 Windows/portable tests and exact construction/restoration on Server
2022; the macOS receiver reverified 66 SDK files, all 13 DLLs and all 43 archive
entries. The 2,491,252-byte ZIP has SHA-256
`269c9e04522d662079b3b2c6063f7f2db5ee1f728c9bba15b6450755b0328bae`.
A subsequent gate now implements real native/public inference, seven lifecycle
tests, observed module paths/hashes and a missing-private-library control in
fresh offline Server Core containers. Its actual execution remains unverified;
Microsoft redistribution, desktop, coexistence, quality and performance gates
remain open.
The first `934d46f` [offline execution attempt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-offline-exe-inspection-control/qualification.json)
built both Gen2 release helpers, then stopped before any runtime container or
model execution because the PE inspector incorrectly required a DLL extension
for an executable's export identity. The checker now derives DLL/EXE export
identity from the COFF characteristics while retaining DLL-only dependencies.
Its new regression passes, all 81 tool tests pass and the qualified 13 DLL PE
reports remain unchanged. A retry needs separate actual evidence.

The subsequent `855a83c` [Windows offline receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-offline-container/qualification.json)
now passes all five fresh Server Core containers: native/public parity is exact
for 27 turns, 6,001 rows and 48,008 probability values; all seven lifecycle tests
including 100 reloads pass; removing private GGML-base returns native error126
with no probability output. Actual network-none/process/readonly mount controls
and managed cleanup pass. Stable native/public/lifecycle snapshots observe all
five private DLLs at the relocated Unicode path and only qualified packaged CRT
or System32 dependencies. Public/lifecycle VCRUNTIME140 starts from the test
helper directory; MSVCP140/VCRUNTIME140_1 load from the runtime directory. The
macOS receiver independently restores both opaque ZIPs (43 runtime/12 helper
entries), rehashes DLL/helpers and reconstructs all five exact frozen command
specifications. The runtime ZIP is 2,491,251 bytes, SHA-256
`89e7fb01b3d15a7d5c36527a4b1e7926dc8ccad360943d231c4c5d3547b79229`.
Observed RTF is 1.06057 public/1.06117 native; this is not Windows live-performance,
physical desktop, coexistence, rights or held-out-quality qualification.

The separate `9be6a4b` [cache-metadata receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-cache-metadata-recovery/qualification.json)
recovers all nine exact ABI-selected archives without installing or executing
cached packages. Receiver verification rehashes 304,920,284 archive bytes,
rechecks CONTROL/status, ABI/dependency and SPDX identities, all 45 metadata
files and four target SDK notices. All nine installed-list bodies were then
reconstructed with exact historical size/SHA matches. The original producer
record remains unchanged; source-recipe/archive closure and native linker
membership are separate open checks. All 85 model-free tool regressions pass.
The later [source-recipe receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-source-recipes/qualification.json)
now verifies six complete port directories/34 Git blobs, all 53 SPDX and 53 ABI
port-file checks, three source archive SHA512s (22,676,219 bytes) and retained
source notices. Full attribution remains open: the qualified SDK's vcpkg
SentencePiece root license omits Darts-clone's distinct BSD notice; esaxx scope
also needs review. Source identity does not establish final native linker-member
selection or complete release qualification.

The subsequent [supplemental-notice audit](../research/nemotron-windows-static-supplemental-notices.md)
identifies Darts-clone's runtime-source BSD notice and conservatively retains
both esaxx's MIT notice and the distinct sais-lite/Yuta Mori header notice.
The Windows constructor now adds those exact pinned texts and their source
selection provenance without changing the qualified native DLLs or installed
upstream notices. Local macOS construction, ZIP restoration and 88 model-free
tool tests pass; the [19-file local construction receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-supplemental-notice-construction/qualification.json)
retains the 47-entry archive identity, unchanged 13 DLLs/eight upstream notices
and exact supplemental selections. That local construction is not Windows execution or complete
license/release qualification. Actual Windows CPU chat coexistence is being
qualified separately using the normal OpenMP backend and fresh offline
containers for isolated chat and both load orders.

The first [chat path-audit control receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-chat-path-audit-control/qualification.json),
source `826f62f`/run `38893676099`, preserves the failed full gate. Native/public
parity and all seven lifecycle tests passed again; isolated chat and chat-first
execution produced six identical `Hello!` outputs, three exact native
diarization comparisons and 28.639463 seconds of actual API-call overlap.
The stage auditor then rejected the valid extended DOS `\\?\C:\...` spelling
reported by Rust's Win32 module snapshots. A strict C-drive spelling normalizer
fixes that observed issue without changing module locations, hashes or allowed
basenames; tests and all three recorded stage snapshots now pass. Diar-first
was not executed in that attempt, so full chat qualification remains false.
A new complete both-order run uses source `d47bdeb` and the supplemented runtime.
That [208-file Windows chat receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-chat-coexistence/qualification.json),
run `38896240605`, now passes. The delivered 47-entry runtime and both opaque
helper archives were independently restored/rehashed; all eight fresh container
control/command/cleanup chains, seven lifecycle tests, native/public comparison
and six mixed-model comparisons pass. Every comparison has 27 turns, 6,001 rows
and zero probability/turn error. All ten chat outputs remain `Hello!`; actual
chat/diarization API calls overlap by 31.815610 and 30.536961 seconds in the two
orders, with zero reservation after shutdown. Internal module stages show zero
private DLLs in isolated chat and all five exact private DLLs when mixed and
reloaded, alongside the exact declared CRT/OpenMP bytes. Native/public RTF is
1.05759/1.06111 on this CI host; Windows live/uncontended performance, desktop,
Bark/Laya coexistence, Microsoft distribution rights and general namespace
isolation remain unqualified.

The separate [central prerequisite investigation](../research/nemotron-windows-central-prerequisite.md)
has now acquired and hash-verified the fixed Microsoft 14.51.36247 installer,
24 static bundle payloads, its actual selected terms document and nine AMD64
CRT/OpenMP DLL identities. That static inspection alone does not prove Windows
signature trust, MSI destinations, setup or central-runtime execution; subsequent
actual receipts below cover those contracts. `package_windows.py --central-runtime`
constructs a distinct five-private-DLL format with a pinned publisher-hosted
prerequisite; it excludes Microsoft installer/runtime DLL payloads, retains the
source notices and preserves the app-local development contract. Local
construction/Unicode restoration and 90 model-free tool tests pass. The
[central construction receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-construction/qualification.json)
retains 39 archive entries/five unchanged DLLs; the
[static prerequisite receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static/qualification.json)
retains metadata and independent extraction checks. These are separate byte/data
results, not central-runtime deployment qualification.
The setup-only workflow now validates the actual Windows publisher signature
before any installer execution, provisions only a disposable process image,
requires exit 0/no pending restart, checks exact System32 DLL bytes/versions/
signatures and both registry views, then re-observes a fresh committed image.
The first actual [central setup receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-setup/qualification.json),
source `9c277fc`/run `38896477423`, now passes that setup-only gate: the installer
has a connected-host Valid Microsoft signature and returns 0; all nine targeted
DLLs were absent before setup and now match exact 14.51.36247 System32 payload
bytes/versions/signatures. Both registry views report the installed version,
no restart indicators are present, and a fresh offline Observe-only container
from the exact base-plus-one-layer image repeats the contracts. All 40 receipt
members and frozen inputs were independently reverified. This setup-only receipt
does not exercise Gen2/native/lifecycle/chat; no image or Microsoft binary bytes
are published or added to Git. Its 92 model-free tool regressions passed.
The subsequent same-job central model workflow now constructs the five-DLL
runtime, re-provisions the publisher prerequisite, builds regular CPU chat and
requires native/public/lifecycle/both-order comparisons with exact observed
System32 CRT/OpenMP bytes and no app-local copies. Its additional fresh-base
negative keeps all five private DLLs present and requires startup failure with
`STATUS_DLL_NOT_FOUND` and no output when the central prerequisite is absent.
The actual source `76952a9`/run `38898592596` now passes. Its
[265-file central runtime/chat receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-runtime-chat/qualification.json)
reverifies the delivered39-entry archive with exactly five private DLLs, the
three-/four-EXE test archives with no app-local CRT/OpenMP bytes, same-source
publisher setup and nine fresh offline model containers. Native/public output
matches all27 turns,6001 rows and48008 probabilities exactly; all seven lifecycle
tests including100 reloads pass. Both chat load orders retain six exact native
comparisons, ten `Hello!` chat observations, actual API overlap of29.473080 and
29.409753 seconds, and zero reservation after shutdown. Every observed CRT/OpenMP
module matches its declared publisher payload under System32; isolated chat
observes zero private DLLs and mixed/reloaded stages observe all five. The
missing-private control returns error126 without output; the fresh original-base
control retains all five private DLLs but fails with `STATUS_DLL_NOT_FOUND` and
no output. All11 model/setup container ID chains, mounts, network-none controls
and cleanup are retained. Native/public RTF0.84706/0.85235 is an observation on
this CI host, not a live/uncontended-performance claim. Microsoft distribution
rights, physical Windows desktop, Bark/Laya coexistence, universal namespace and
held-out-quality gates remain open. The separate AMI diagnostic raises the
model-free regression total to96 at that point. The frontend diagnostic
subsequently raises it to102; all 13 jobs in source `ec88564`
run `38903243271` now pass.

A guarded [Windows private-DLL build](../research/nemotron-windows-private-names.md)
now relinks the five owned DLLs through an external CMake naming hook, updates
probe/entry paths and rejects residual generic ordinary/delay/forwarded native
imports. A bounded PE inspector and malformed-input/naming regressions pass;
all fifteen previously delivered PE images independently agree with LLVM's
ordinary-import inspection. The workflow also captures actual redist byte
origin, versions/signers and vcpkg metadata without claiming distribution
qualification. The first private build linked all five DLLs and the probe, then stopped at an
overstrict installed-import-library assertion. Its [failed receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-private-import-library-control/qualification.json)
is retained; the checker now matches upstream's single installed public C ABI
import library. The actual `24ad927` [private Windows development receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-private-native-development/qualification.json)
now passes exact native/public parity (27 turns/6,001 frames), all seven lifecycle
tests including 100 reloads and the direct Win32 error126 control. All five
private DLL/export identities and import graphs pass, and all 66 delivered SDK
files were independently rehashed. Actual provenance records all eight CRT DLLs
as identical to the recorded redist source, with Valid/Microsoft signatures
observed on the build host; runtime file/product version is 14.44.35211.0 and
compiler file/product versions are 19.44.35229.0/14.44.35229.0. No accompanying
license candidates were found, so redistribution remains unqualified. Observed
RTF is 1.07498 public/1.05032 native, without a live-performance claim. The separate
[read-only Windows prerequisites](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-container-prerequisites/qualification.json)
confirm an available process-container engine and preloaded Server Core image.
No container/model payload ran in that prerequisite check. The subsequent
[Server Core controls receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-container-controls/qualification.json)
passes actual process isolation, network none, read-only Unicode input access
denial (Win32 error 5), writable results, absence of expected build tools and
managed-container cleanup. That controls test contains no Gen2/native/model
payload and keeps runtime qualification false. Offline packaging, licensing,
coexistence and desktop machine gates remain open.

The actual Linux SDK archive job now passes native/public parity and all seven
lifecycle tests, then preserves all 82 filesystem records, including seven
aliases and two executables. Delivered tar bytes and every header mode match;
a Mac extraction initially missed Linux symlink modes, then restoring those
modes produced an exact inventory match. The failed verifier comparison and a
standalone rechecker are retained in the [SDK archive receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-sdk-archive/qualification.json).
This is archive delivery/metadata evidence, not Linux runtime relocation. A
separate runtime packager now implements receipt/ISA checks, recursive ELF
closure, private native/SentencePiece identities, relative RPATHs, component
copyright/common-license texts and exact source-package acquisition. Actual
Linux construction now passes with eight libraries and exact GCC 14 and
SentencePiece source bundles. The delivered 110,126,341-byte archive matches
SHA-256 `05cf4bc046ea0de8472f2aca72533ecd95106e6bf093cd657ae94a9aab80864e`;
all 45 entries and both `.dsc` source inventories were rechecked on macOS.
See the [construction receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-runtime-construction/qualification.json).
The following first container attempts are retained as failures; the later
passing source-specific receipt below closes the offline container gate. Actual
chat/Bark/Laya coexistence remains a separate gate.
The first container attempt passed the isolation controls and initialized all
eight bundled libraries at the relocated path, including GCC. It then failed
opening the locked WAV: capability-free container root could not read the CI
user's 0600 fixture copies. The [retained container failure](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-container-permissions/qualification.json)
does not claim inference. The reviewed runner now uses the owning UID/GID and
checks exact input/output permissions before any model run. That retry now
passes offline native/public parity (26 turns, 6,001 frames, zero numerical
error), all seven lifecycle tests and the eight-library loader audit. The
missing-library process correctly fails, but Gen2 discarded the OS cause from
libloading's error chain, so the detailed negative assertion still fails. The
[retained diagnostic receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-container-loader-diagnostic/qualification.json)
keeps the overall gate false. A reviewed Rust fix preserves that cause and
requires the exact missing dependency in the assertion. The actual `c242201`
[offline container receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-offline-container/qualification.json)
now passes the complete gate in run 38878287549: relocated native/public parity
(26 turns, 6,001 frames, zero numerical error), all seven lifecycle tests including
100 reloads, all eight delivered libraries initialized and the exact missing-GGML
negative control. Each check used a fresh, network-disabled, read-only Ubuntu
24.04 container under the owning UID/GID. The delivered 110,126,343-byte archive
hashes to `83ef52fed709501d7aa57110cc74d125b1835dba83e48cc737fda40d6dd0515a`;
all 45 entries, corresponding-source bundles and frozen tool hashes were
independently rechecked on macOS, without executing Linux code there. This
closes this Ubuntu/glibc/ISA container scope only; physical-machine, other
baseline and quality gates remain open. The actual `3364fd7`
[Linux chat coexistence receipt](../../benches/results/nemotron-diarization/linux-x86_64/2026-10-16-chat-coexistence/qualification.json)
now passes run 38880349072, including the same-source offline prerequisite.
Both load orders return the isolated `Hello!` text at all ten chat observations.
All six diarization calls match the independent 26-turn/6,001-frame native
reference with zero probability error. Actual API calls overlap by 1.043754 and
1.509064 seconds, shutdown releases reservations, and both loader traces show
720 native GGML bindings confined to the delivered package. The CPU chat backend
retains OpenMP, with its test runtime tied to the exact delivered GCC source.
The receiver independently rechecked delivered runtime/source bytes and all
comparisons/traces; separate helper/model/OpenMP test-payload hashes are CI
observations because those test binaries/assets were not delivered for rehashing.
Bark/Laya coexistence, other baselines, physical machines and complete release
qualification remain open.

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
