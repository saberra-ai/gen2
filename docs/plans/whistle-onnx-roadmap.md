# Whistle ONNX as the second Gen2 speech adapter

Status: native CPU implementation complete; desktop qualification in progress.
Prepared and implemented: 2026-10-13. Implementation is based on merged Bark
commit `6a659ff`. Both models use `load_speech` and `transcribe_request`; existing
Bark calls remain supported. The implementation uses a common request and
result API, with supported controls exposed through `profile()`.

Local evidence: 140 FLEURS recordings across all seven languages match the
pinned JavaScript oracle in automatic and forced-language modes. The full
feature suite passed (1345 tests), as did the 250-record Bark regression,
Whistle lifecycle/cache tests and Bark/Whistle/Laya coexistence. Performance
and memory gates pass on Apple M4 Pro; three-platform CI remains in progress.
Streaming, timestamps and keyword biasing remain outside this initial slice.

## Implementation evidence (2026-10-13)

Bark was merged as `6a659ff`; Whistle implementation commits start at `80545f7`.
Both adapters expose the same source/load/request/result interface. Unsupported
language/token controls fail explicitly according to the loaded profile.

- 202 cases: 140 pinned FLEURS recordings (20 per language), 14 quiet/noisy
  derivatives and 48 WAV variants from English/Spanish recordings. Automatic
  and forced-language text, token IDs, detected/selected language and termination
  match the upstream oracle exactly. The 16 kHz public WAV path and optional
  libsoxr conversion passed; conversion retains its experimental designation.
- 19 independent stage probes (mel, fixed engram, encoder K/V, three decoder
  logits/recurrent/KV steps) had zero numeric difference on the named host;
  their acceptance tolerance is `atol=rtol=1e-4`.
- Lifecycle/queue/cancellation/deadlines/reload/runtime conflicts and offline
  integrity passed, as did online provisioning and corruption recovery.
- The existing 250-record Bark oracle regression passed. Bark + Whistle +
  chat + Laya stayed resident together, produced stable results and rejected
  an additional over-budget speech load. Shutdown released reservations.
- Full feature suite: 1345 passed before the byte-decoding regression was added;
  that regression passed separately. No-backend suite and ASR/Laya/HF Clippy pass.
- Apple M4 Pro, macOS 26.3: cold load 1.08 s, median RTF 0.008–0.021 for
  1/5/15/30/60 s inputs, native/Node median ratios 0.86–0.93, sampled peak RSS
  770 MiB. The 100-request soak grew 16 KiB after warmup. This supports the
  conservative 2 GiB floor; native allocator pages can remain after shutdown.

Receipts: [Apple M4 Pro run](../../benches/results/whistle/apple-m4-pro/2026-10-13/receipt.json).
The frozen corpus and host-native oracle are distinct from quality claims.
Held-out WER is 20.2% en, 21.0% de, 20.8% fr, 11.5% es, 19.5% it,
38.2% nl and 39.3% pl (ten recordings per language). Quiet/noisy and tagged
English names/numbers/negation metrics are separate receipt slices. These small
samples establish parity and expose upstream weaknesses; they do not establish
broad dictation or multilingual critical-content accuracy.

The initial Windows CI corpus encoding failure was fixed by explicit UTF-8.
macOS/Linux/Windows qualification and the PR's existing regression workflows
are still required before treating this as a qualified desktop release. W5
word details, biasing and streaming remain deferred.

## Outcome and prerequisite

Add native Rust transcription with `mrfakename/whistle-ONNX` through the same
speech module used by Bark. Applications select a speech model, load it once,
and submit recordings through a reusable transcriber. Keep frontend, tokenizer,
engram lookup, encoder sessions and decoder state inside the selected adapter.
Neither Node nor Python is required in the shipped inference path.

The prerequisite is the completed Bark M0–M4 slice: merged speech interface,
real-model desktop CPU gates on macOS/Linux/Windows, runtime initialization,
bounded workers, memory admission, offline provisioning and documented audio
limits. Preserve any experimental resampling designation. Bark acceleration,
mobile and MLX follow-ups do not block Whistle CPU work. See the
[Bark roadmap](bark-onnx-roadmap.md).

At implementation start, read the merged Bark code and freeze its SHA and
qualification receipts. The inspected implementation currently lives in
`/Users/victor/workspace/gen2-bark-onnx`, branch `codex/bark-onnx`, inspected at
`8277263`; the main checkout's Bark proposal is older. File locations and
interface names below are proposals to reconcile with the merged version.

## First release scope

- Desktop ONNX Runtime CPU, FP32 encoder and decoder, completed recordings.
- Finite mono 16 kHz float32 PCM first; WAV/downmixing/resampling through the
  qualified shared audio path once Whistle parity passes for those inputs.
- Automatic language identification and an explicit supported-language option.
  Qualify English, German, French, Spanish, Italian, Dutch and Polish separately.
- Greedy transcription with bounded generated tokens, caches and audio duration.
  Report the selected language, model identity, completion status and timings.
- Repeated requests, cancellation/deadlines, offline loading, and coexistence
  with Bark, chat and Laya within a shared memory budget.

Word timestamps/probabilities and keyword biasing are a later, optional slice
after text parity. Live streaming, long-audio windowing, microphone capture,
FP16, WebGPU/native GPU providers, MLX and physical mobile qualification are
separate follow-ups. Do not inherit the original Cactus engine's advertised
streaming capabilities into this ONNX adapter. [Reference pipeline][pipeline]

Whistle returns its own ASR transcript. Do not advertise Bark's disfluency
cleanup behavior for Whistle without independent quality evidence. Model
selection is explicit; never silently substitute one model for the other.

## Evidence that shapes the implementation

Pin initial qualification to Hugging Face revision
`14fc92cb09ad5c271fe9a4ec4be70274790050d2`. The published interface differs from
Bark throughout the inference pipeline. [Export model card][card]

| Contract | Whistle | Inspected Bark adapter |
| --- | --- | --- |
| Frontend | 80-bin normalized log-mel `[T,80]` | 128 bins `[1,128,T]` |
| Encoder outputs | Cross-attention K/V `[8,8,S,48]`, `[8,8,S,64]` | Encoded acoustic frames `[1,1024,T]` |
| Decoder | Autoregressive token/position steps with host engram features | Recurrent TDT token/duration decisions |
| State | `prev [8,2,608]`, growing self-attention K/V | Two recurrent `[2,1,640]` states |
| Vocabulary | 8,199 logits, including seven language IDs | 8,193 vocabulary logits plus five duration logits |

The Node reference loads `onnx/encoder.onnx`, `onnx/decoder.onnx`, their external
data files, and `whistle.pack`. The pack supplies the tokenizer, mel filterbank
and compressed engram tables, even when network weights are loaded directly
from the full ONNX files. Initially download the full FP32 graphs/data plus
the pack; defer rebuilding network weights from the smaller pack.
[Reference example][example] [Pack reader][pack]

The frontend includes percentile-based gain normalization, its own window
placement/FFT arithmetic, filterbank coefficients and per-band normalization.
Port the pinned pipeline rather than resizing Bark's filterbank. The host
decoder hashes token history and dequantizes engram rows before each graph
step. Integer overflow rules and floating-point rounding are part of parity.
[Frontend and engram implementation][pipeline]

The wrapper chooses a duration-dependent default token limit capped at 320,
and can return text when that limit is reached without EOS. It returns an
empty result for fewer than four mel frames. W0 below must record these
behaviors; Gen2 must define explicit completion and minimum-input semantics.
Do not claim that a nominal 60-second recording always fits the token cap.
[Decode loop][pipeline]

See the [compatibility research note](../research/whistle-onnx-compatibility.md)
for the full graph contract and source inventory. All measurements and gates
below are planned, not current compatibility or performance claims.

## Shared interface and adapter seam

Keep `gen2::speech` as the deep module callers use. Introduce a private adapter
seam now that two real implementations exist: Bark and Whistle. Each adapter
owns bundle validation, model-specific frontend/decoding, capability/profile
data, request validation and its memory estimate. The common worker owns
queueing, cancellation/deadlines, shutdown and reservation lifetime.

Use a closed internal enum or small private interface for the two adapters;
choose whichever is simpler in the merged code. A public third-party plugin
registry and a generic ONNX graph loader are outside this change.

Proposed additive caller surface:

- Keep `Runtime::load_transcriber(path, options)` as the existing Bark entrypoint
  and keep existing Bark `transcribe` calls working.
- Add explicit speech-source selection, for example
  `Runtime::load_speech(SpeechSource::WhistleOnnx(path), options)`, returning the
  same transcriber handle. Do not infer adapter identity solely from filenames.
- Keep basic `transcribe(audio, options)` for ordinary text. Add a request/detail
  interface for language choice, token limits and completion metadata. If basic
  transcription reaches the limit without EOS, return a typed failure; an
  opt-in detailed call may return clearly marked partial output.
- Expose supported languages, automatic identification, model-specific input
  limits, cleanup behavior and optional word details through profile/accessor
  methods. Preserve existing public struct construction where possible: do not
  add mandatory fields to existing options/results without an explicit version
  migration. Document any unavoidable source compatibility change.

Keep language-choice options distinct from detected language. An unsupported
language, a Whistle-only option submitted to Bark, or an unavailable capability
must fail clearly. Reconcile names and SemVer at W2 before publishing them.

Move Bark's fixed package identity and graph assumptions into its adapter.
Shared provisioning can handle staging/cache integrity, while adapter manifests
define permitted artifacts and hashes. Retain Bark's pinned allowlist; removing
its digest check does not constitute Whistle support.

Move memory floors/estimates and input limits behind the adapter profile.
Preserve Bark's current conservative minimum; measure Whistle independently
instead of retaining a universal 4 GiB floor or treating its 17 MB pack as RAM
usage. Bound engram-row caches as well as decoder K/V and queued audio. A
request exceeding the qualified reservation fails before avoidable allocation.

## Delivery sequence and exit gates

Each milestone produces a usable or independently verifiable result. No
dependent milestone proceeds on graph-load-only evidence.

| Milestone | Deliverable | Exit gate | Depends on |
| --- | --- | --- | --- |
| W0 — reference/package qualification | Pinned FP32 package, locked Node CPU oracle, graph inventory and real-audio traces | Reference runs reproducibly; chosen native ORT build runs the graphs; supported limits are recorded | Bark M0–M4 merged |
| W1 — native Whistle proof | Rust frontend, pack metadata/engram reader, tokenizer and autoregressive decoding on normalized PCM | Real audio matches oracle language, tokens and raw text; state/cache probes pass | W0 |
| W2 — second speech adapter | Shared worker/interface with explicit model selection and adapter-specific profiles | Existing Bark behavior remains equivalent; Whistle lifecycle and mixed admission pass through public calls | W1 |
| W3 — audio/provisioning workflow | Revision-pinned download, offline reuse and qualified audio formats | Corruption/recovery tests pass; advertised WAV/PCM inputs match reference; no inference-time network | W2 |
| W4 — desktop release | Documentation/example, multilingual quality and performance receipts, three-platform CPU CI | All supported OS/language and lifecycle gates pass with real weights | W3 |
| W5 — optional word details and biasing | Opt-in DTW timestamps, probabilities and keyword requests | Separate oracle comparisons, resource bounds and documented approximation limits pass | W4 |

### W0: freeze the oracle and package

1. Inventory the pinned files, sizes, SHA-256 hashes, graph external-data paths,
   opsets/domains, dtypes, dimensions and tokenizer/special/language IDs. Create
   a Gen2-owned revision lock; validate its provenance against the pinned Hub
   artifacts. Include Apache-2.0 attribution and inspect dependency notices.
2. Acquire the two FP32 graphs, both `.data` files, `whistle.pack` and reference
   JavaScript. Keep weights/audio in a cache or artifact store outside Git.
   Validate zero-length initial decoder caches and growing output dimensions.
3. Lock Node, `onnxruntime-node` and dependencies with integrity information.
   Reproduce the pinned JS CPU path, then qualify the native ORT version/build
   shared with Bark. Compare runtime versions if the Node package differs;
   record any transcript or token differences before choosing the oracle.
4. Export normalized PCM, gain decisions, mel features, encoder K/V, fixed-step
   engram features, decoder logits/states and token/language decisions. Capture
   both automatic and forced-language cases and EOS/token-limit termination.
5. Freeze licensed audio plus human transcripts covering all seven languages,
   accented characters/byte fallback, names, numbers, negation, quiet/noisy
   speech, rapid speech, silence and input-duration edges. Target at least ten
   real clips per language for the parity set; select a larger held-out quality
   corpus with a per-language size justified in the reference receipt.

Qualify the minimum and maximum audio durations with real inputs and resource
measurements. Start with the shared Bark audio envelope as a ceiling, not proof
of Whistle support. Record how frequently the wrapper's default token cap ends
otherwise valid recordings. Define a bounded opt-in limit if necessary, with
measured cache/reservation costs; never silently truncate successful output.

Suggested artifacts: `tools/whistle/{README.md,reference/package-lock.json,
qualify_reference.mjs,export_trace.mjs}`, a model lock, and
`tests/fixtures/whistle/reference.json`. Preserve the original reference
algorithm; instrumentation may observe it without replacing its decisions.

### W1: prove native inference before changing the shared worker

Implement a private Whistle module and a small native executable on mono 16 kHz
PCM. Load FP32 weights directly. Parse only the pack sections needed for
tokenization, filterbank and engram rows; bound header lengths, offsets, counts,
dequantized allocation sizes and index arithmetic.

Port gain normalization and the 80-band frontend with the reference arithmetic.
Port token decoding/byte fallback and tokenizer encoding required for later
biasing. Match JavaScript's unsigned 32-bit hash/wrapping multiplication and
engram dequantization against fixed history fixtures. Bound row caching or
precompute only a measured, admitted amount of data.

Run the encoder, initialize zero-length K/V and `prev`, process BOS and language
selection, and decode greedily until EOS or a declared limit. Match token
masking, tie behavior and position handling. Validate output ranks/lengths,
IDs, finite numbers and cache growth after every graph call. Reset all
utterance state on success, failure and cancellation. Cache immutable engram
rows only under a bounded policy; token history remains request-local.

Verify exact integer/language/token decisions and raw text against the oracle.
Use proposed `atol=rtol=1e-4` as an initial floating-point comparison, then set
stage-specific tolerances from W0 evidence. An unexplained token discrepancy
fails qualification even if aggregate error rates appear similar.

Suggested files: `src/speech/whistle/{mod,bundle,pack,frontend,tokenizer,
engram,onnx,decode}.rs`, `tests/whistle_native.rs`, and
`examples/whistle_transcribe.rs`. Keep the module private until W2.

### W2: integrate the two adapters without changing Bark inference

Snapshot Bark public-call results and its qualification corpus before extracting
the common worker seam. Route the existing Bark entrypoint to the Bark adapter,
then introduce explicit Whistle loading with the same lifecycle handle.

Keep common ORT initialization/native-library conflict detection. Never create
a second adapter-local initializer. Load both adapters in both orders; require
consistent errors for missing/incompatible/conflicting runtime paths and failed
loads without leaked reservations.

Add Whistle-specific profile/metadata and request validation. Preserve existing
Bark limits, options and cleanup claims. Define partial-output semantics in the
public interface, and include API documentation and migration notes in
`api_spec.md` when names/types are finalized.

Measure weights, encoder/cross-attention buffers, decoder cache at the maximum
supported token limit, frontend workspace, pack/engram cache and queued input.
Feed a conservative estimate into the existing shared admission budget. Keep
reservations until native sessions and in-flight work release; cancellation
between decoder calls must not imply interruption of an active encoder call.

Through public calls, verify A–B–A recordings, two concurrent requests, queue
overflow, expired deadlines, queued/running cancellation, shutdown, reload,
failed load and adapter selection. Run Bark + Whistle + chat + Laya within and
over budget, including restoration/eviction paths covered by the merged runtime.
Use fresh processes for independent qualification cases where native allocators
retain memory; do not relax admission ceilings to make coexistence pass.

### W3: provision and qualify audio inputs

Add Whistle's dedicated source/lock to shared provisioning. Download to staging,
verify all sizes/hashes and permitted graph external paths, then atomically
publish an immutable revision entry. Include the pack even though the FP32
network weight data is downloaded separately. Test interrupted downloads,
corrupt/truncated pack sections, missing external weights, wrong revision,
invalid tokenizer/table metadata, warm cache and offline reload.

Reuse shared decoding/downmixing/resampling; do not duplicate WAV parsing in the
adapter. Compare from identical normalized PCM first, then compare end-to-end
audio normalization and transcripts. Requalify Whistle across each advertised
rate, channel count and encoding. Respect original and post-resampling duration
bounds. If a format fails, keep it unadvertised or explicitly experimental with
measured limits; Bark's passing format tests alone cannot qualify Whistle.

### W4: qualify desktop CPU and document the result

Run the same pinned package and matched-reference settings on macOS, Linux and
Windows. Add Whistle-specific real-model CI plus mixed Bark/Whistle regression
coverage. Existing Bark qualification gates remain required after shared speech,
runtime or dependency changes.

| Evidence | Release gate |
| --- | --- |
| Native parity | Exact language, token IDs, raw text and termination reason for the frozen real-audio corpus under matched reference conditions |
| Quality | Human-reference WER/CER per language, quiet/noisy slices and critical-content cases; native quality no worse than matched oracle; document upstream weaknesses |
| Limits | Minimum/maximum duration, short/silent inputs, invalid language, malformed audio, token exhaustion and cache bounds behave as documented |
| Lifecycle | No cross-request decoder state, bounded queues/cache, actionable typed errors, cancellation/reload and zero reservation leaks |
| Regression | Bark retains its required corpus/lifecycle gates; mixed speech/chat/Laya admission and runtime reuse pass |
| Offline/package | Complete pinned cache works with network disabled; corrupted artifacts fail or recover predictably |

Measure cold load, warm inference and frontend/encoder/decoder/queue times for
1/5/15/30/60-second clips up to the qualified duration ceiling; include rapid
speech cases that stress token limits. Use warmups and at least five repetitions;
report median/p95, RTF, peak RSS, steady residency and 100-request soak behavior.
Compare Rust and Node from identical normalized PCM and record runtime/thread
settings. Proposed usefulness targets: warm RTF below 1.0 on a named desktop,
and native median inference within 1.25× the matched Node reference. Validate
these targets in W0; do not present them as existing measurements.

Record receipts under `benches/results/whistle/<machine>/<run-id>/`: code SHA,
model/lock digest, native runtime/library hash, oracle lock, corpus/audio hashes,
normalizer, OS/CPU, provider placement, language/token/raw transcripts,
termination reason, tensor comparisons, WER/CER, timings and memory. Report
pack size, full download size and residency separately.

Publish a runnable example and usage documentation for model selection,
provisioning, automatic/forced languages, input/output limits, incomplete output,
errors, cancellation, offline reuse and measured CPU behavior. Explicitly fail
real-model qualification if required assets are absent; skipped tests do not
establish compatibility. Default/no-backend, ASR-only and ASR+Laya feature builds
must pass their applicable existing checks.

### W5: qualify optional word details and keyword biasing

Port DTW alignment and word grouping with opt-in output types, bounded attention
storage and separate memory estimates. Compare word boundaries/probabilities
against the ONNX JS oracle, and independently annotated timing samples for
quality. Do not promise identical original-engine probabilities or timestamp
precision beyond the measured frame resolution.

Port the reference keyword tokenization, trie/continuation behavior and logit
boosts. Verify enabled and disabled paths, multilingual byte fallback, duplicate
and invalid keywords, bias resource limits and state reset. Document that the
export's biasing approximates Cactus behavior. No optional detail may alter the
default text result without an explicit request. [Model card][card]

## Suggested reviewable changes

1. W0 reference lock, package inventory and repeatable real-audio oracle tools.
2. Validated pack/tokenizer/engram and frontend with independently exported probes.
3. Full native Whistle PCM example and oracle parity tests (W1 useful slice).
4. Private two-adapter seam retaining Bark public behavior and corpus parity.
5. Explicit model selection, metadata/completion semantics and measured admission.
6. Pinned Whistle provisioning plus offline and qualified audio workflow.
7. Three-platform real-model/lifecycle/performance receipts and release docs.
8. Optional word details/biasing after the text-only release.

The first action after Bark completion is W0 on a new branch from the merged
qualified SHA. No work is automatically dispatched by this plan.

## Primary sources

[card]: https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md
[pipeline]: https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/whistle.js
[example]: https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/example.mjs
[pack]: https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/pack.js
