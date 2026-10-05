# Native Laya integration — experimental

`Runtime::load_decider(bundle_directory, LoadOptions)` returns a cloneable
`DecisionModel`. It supports ordered text/JSON/conversation requests, choice,
ordinal score and yes/no questions, calibrated probability distributions,
question microbatches, request batches, explicit checkpoint/language routing
and long-state scans. It does not generate text or execute actions.

Use `laya-dynamic` for host-supplied desktop libraries, or
`backend-laya-onnx` for an application-linked runtime. Default gen2 features are
unchanged. No Python or network access is used during native inference.
The optional binding is pinned to `ort = 2.0.0-rc.12`, API 24; use ONNX Runtime
1.24.4 on desktop (mobile framework versions are pinned in the host harness README). It requires Rust 1.88 or newer; local verification used Rust 1.95.0.

```toml
gen2 = { path = "../gen2", default-features = false, features = ["laya-dynamic", "tokio"] }
```

```rust,no_run
use gen2::{Runtime, decision::*};
let runtime = Runtime::builder().resident_memory_budget_mb(4096).build()?;
let model = runtime.load_decider("models/laya-english", LoadOptions {
    native_library: Some("runtime/onnxruntime.dll".into()),
    ..Default::default()
})?;
let answer = model.decide(
    DecisionRequest::text("Please refund the duplicate payment.")
        .question("refund", Question::yes_no("Is a refund requested?")),
    DecisionOptions::default(),
)?;
model.shutdown();
# Ok::<(), gen2::Error>(())
```

The budget above is an example host policy, not a universal hardware minimum.
The default runtime uses gen2's memory governor. Decision reservations cover
weights plus the declared peak workspace and remain held until native work
and session destruction finish. Chat load/restore admission includes them.
`Runtime::decision_models`, `decision_reserved_mb`, `stats`, and each model's
`status` expose ownership. The registry uses weak handles.

## Semantics and lifecycle

- `OrderedJson` preserves object insertion order and rejects duplicate keys.
  A JSON array or conversation truncates from the left when explicitly allowed;
  text and objects truncate from the right. JSON uses the pinned Python spacing
  and Unicode conventions. Questions and choices retain caller order.
- Overflow defaults to rejection, including instruction/option shortening.
  `AllowWithDiagnostics` reports dropped tokens; indistinguishable option spans,
  missing markers or a header that cannot fit always fail.
- `AnswerValue::Score` contains a probability-weighted ordinal expectation.
  `AnswerValue::YesNo` contains P(true), with false before true in the inputs.
  `confidence` preserves Laya's entropy statistic for choice/score;
  `answer_confidence` is max probability, optionally histogram calibrated.
  Language overrides bypass the base histogram. Clamped temperatures are listed
  in `calibration_diagnostics`. `act_probability` is diagnostic only.
- `DecisionRouter` routes explicitly by checkpoint, or by a caller-supplied
  language tag (English for `en`, multilingual otherwise). It does not guess
  language or silently download/substitute models.
- `decide_long` covers the complete serialized state using overlapping windows.
  Yes/no selects the highest P(true); choice/score select highest answer
  confidence, with first-window tie breaking. Returned token spans identify the
  selected window. Its confidence is **not** whole-document calibration.
  A window that grows beyond capacity when retokenized fails rather than losing
  evidence. Window count and stride are host-configurable.
- Queue admission is bounded and returns `decision_busy` when full. Request byte
  limits, token budgets, thread count and queue capacity are configurable.
  Deadlines/cancellation are cooperative: an executing native kernel can finish
  before buffers are freed. Async future drop requests cancellation.
- `unload()` is nonblocking and stops admission. Use it for suspension or memory
  pressure. `shutdown()` waits; `shutdown_async()` keeps the UI thread free.
  `Runtime::reload_decider` returns a new worker only after shutdown and verifies
  the original manifest digest. Final handle drop also stops the worker.

## Reproducible bundles

Model repository: `convaiinnovations/laya`, revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`.
Reference source: `NandhaKishorM/laya`, revision
`8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`.

Create a Python venv and install `requirements.txt` (use the CPU-only torch
index for the torch wheel). Check out the exact upstream revision. Then:

```sh
python tools/laya/export_bundle.py --source target/laya/upstream \
  --output target/laya/english --checkpoint english --reservation-mb 4096
```

Repeat with `multilingual` and `typed-decisions`. Output directories must be
new. Source checkpoint copies under `source/` are export-only and can be omitted
when packaging; package `manifest.json` and every entry in `files`.
Never mutate a bundle while a loaded model uses it.

The manifest lists graph, external tensor data, tokenizer, configuration,
license files, source revisions, exporter digest, versions, token IDs and shape
envelope. The loader hashes each file, rejects paths escaping the bundle, and
inspects graph signatures/external-data references before session creation.
Hashes detect corruption; obtain the manifest itself through a trusted channel.
Only FP32/opset-18 dense graphs using the pinned preprocessing/code contract are accepted. Reduced
precision, custom ONNX domains and graph-local functions need separate support.

Compatible fine-tunes are not restricted to a repository allowlist. A manifest
must identify a nonempty source and an immutable 40-character source commit or
64-character local content digest. All graph, tokenizer, integrity and memory
checks still apply. Export a local checkpoint containing `model.safetensors`,
`rl_agent_config.json`, `encoder/` and `tokenizer/` with:

```sh
python tools/laya/export_bundle.py --source target/laya/upstream \
  --checkpoint-dir /path/to/checkpoint --checkpoint english \
  --model-repository local/my-finetune --model-license /path/to/LICENSE \
  --output target/laya/my-finetune --reservation-mb 4096
```

The content digest binds weights, encoder metadata, tokenizer and agent config.
The owner's model license is included separately from Laya's code license.
The `checkpoint` family selects routing/calibration conventions, not model
identity. Requalify shapes, memory, parity and quality for changed checkpoints.
The original exporter used by archived model manifests is retained by digest
under `evidence/exporters/`.

`decide_batch` submits all states as one bounded worker job and shares native
microbatches across state boundaries. It preserves state/question order,
including empty states. `max_input_bytes` applies to the aggregate batch.
Tokio provides both `decide_async` and `decide_batch_async`; dropping either
future signals cancellation while the worker retains in-flight native buffers.

Questions accept `.with_option_order(vec![...])` (or JSON `option_order`).
Each presentation slot names a canonical option index; duplicate, missing or
out-of-range indices are rejected. Encoding uses presentation order and decoding
restores the original labels, ordinal levels and yes/no polarity. Unknown
question fields are rejected instead of being silently ignored. Independent
permutation fixtures cover all three question types and tokenizer families.

`model.capabilities()` exposes state/question forms, window/batch support,
calibration import versions, graph envelope and host budgets. `LoadOptions.execution`
selects CPU (default), DirectML/CUDA with a device index, CoreML or NNAPI.
Registration failure is always an error. CPU partition fallback for unsupported
accelerator nodes requires `allow_cpu_fallback: true`; the default is false.
Result metadata reports this configured policy, not measured node placement.
CUDA TF32, CoreML low-precision accumulation and NNAPI FP16 relaxation are
explicitly disabled. Accelerator parity and performance remain unqualified.

Dynamic hosts supply a runtime built with the selected provider. App-linked
hosts additionally enable `laya-directml`, `laya-cuda`, `laya-coreml` or
`laya-nnapi` as appropriate, and package all required native dependencies.
Queue and execution microseconds in each result cover the entire submitted job;
each state in a batch repeats these job-wide values.

To import calibration, add the payload to `files`, set `calibration`, and update
the file hashes. Imports require a v2 payload bound to the selected model,
subfolder and exact checkpoint configuration. `Calibration::from_json` also
allows pure parameter inspection without loading a model.

## Verification commands

```sh
cargo test --no-default-features --lib
cargo clippy --no-default-features --features laya-dynamic,tokio --all-targets -- -D warnings
python tools/laya/record_preprocessing.py --source target/laya/upstream/laya/common.py \
  --tokenizer target/laya/english/tokenizer --output target/laya/preprocessing.json
python tools/laya/record_inference.py --source target/laya/upstream \
  --bundle target/laya/english --fixture tests/fixtures/laya/preprocessing.json \
  --mode onnx --output target/laya/english-onnx-reference.json
```

Record `--mode eager` separately. Set `GEN2_ORT_LIBRARY`, `GEN2_LAYA_BUNDLE` and
`GEN2_LAYA_REFERENCE`, then run:

```sh
cargo test --no-default-features --features laya-dynamic --test laya_native native_matches \
  -- --ignored --nocapture --test-threads=1
python tools/laya/qualify_bundle.py --bundle target/laya/english \
  --output target/laya/english-windows-cpu.json
```

The synthetic fixture in `tests/fixtures/laya/smoke` only tests plumbing and
lifecycle; its manifest identifies it as `SYNTHETIC_TEST_ONLY`. It is not Laya.
Real-model references use the full encoder and decision heads. Padded option
slots never enter probability normalization; K=1 uses a masked second native
slot because the traced action head contains `topk(2)`.

## Current evidence and remaining release gates

### Full performance matrix

Build the runner in release mode, then run all checkpoints serially:

```sh
cargo build --release --no-default-features --features laya-dynamic --example laya_benchmark
python tools/laya/run_benchmarks.py --bundles target/laya --runtime /path/to/onnxruntime \
  --runner target/release/examples/laya_benchmark --output target/laya/benchmark-run
```

The runner measures 1/5/10/32 questions at short, medium and boundary lengths,
five warmups and 100 measured calls per case. Reports preserve every duration,
exact tensors, provider/runtime hashes, thread count, cold-load time, peak
process memory and post-shutdown reservation state. Each completed case is
flushed to JSONL. An interrupted report cannot pass the comparison gate.
Python uses the same library, graph, padding and microbatch shapes. Its baseline
starts with preencoded rows, so the comparison is conservative: Rust additionally
times preprocessing, decoding and worker admission. This is explicitly not a
full Python SDK benchmark. Reports retain failures of the provisional 15% p95
target rather than hiding slower cases. These CPU runs do not qualify thermal
behavior, acceleration or chat coexistence.

### Application-quality evaluation

Use the schema shown in `tests/fixtures/laya/quality-smoke.json` for a real
application dataset. Each sample has a unique ID, calibration/holdout split,
stable question group, request and canonical integer labels (including ordinal
level indices and false=0/true=1). Optional `heuristic` values use the same indices.
The committed fixture is **synthetic pipeline testing only**, not accuracy data.

```sh
python tools/laya/prepare_evaluation.py --dataset /private/labels.json \
  --bundle target/laya/english --output target/laya/evaluation-plan.json
target/release/examples/laya_benchmark target/laya/english /path/to/onnxruntime \
  target/laya/evaluation-plan.json target/laya/predictions.jsonl
python tools/laya/evaluate_quality.py --dataset /private/labels.json \
  --predictions target/laya/predictions.jsonl --output target/laya/quality-report.json
```

The evaluator rejects duplicate IDs, exact calibration/holdout evidence overlap,
changed question/label spaces, mismatched predictions, incomplete runs and invalid
probabilities. It reports per-type, per-group and option-count accuracy, score
MAE, multiclass Brier score, ten-bin ECE and risk/coverage with ties kept together.
Majority/prior baselines use calibration labels only. Missing heuristic labels
are reported explicitly. Application-specific acceptance thresholds, useful
labels and semantic/entity/time leakage review remain the dataset owner's input.
Prediction files contain input text and must stay in an appropriate private
directory. This evaluator does not fit or modify model calibration parameters.

### Recorded checks

On Windows x64, CPU, ONNX Runtime 1.24.4:

| Check | Result |
|---|---|
| Library, existing default features | 1180 passed, 17 ignored |
| Library, no default features | 996 passed, 3 ignored |
| Native feature + Tokio, full library suite | 1007 passed, 3 ignored |
| Native feature + Tokio, clippy | passed with warnings denied |
| Existing default features + Laya | compilation passed |
| English: Rust vs Python ONNX, eight cases | max probability difference 3.66e-8 |
| English: Rust vs eager FP32, eight cases | max probability difference 1.40e-6 |
| Multilingual: Rust vs Python ONNX, eight cases | max probability difference 3.54e-8 |
| Multilingual: Rust vs eager FP32, eight cases | max probability difference 1.02e-6 |
| Typed-decisions: Rust vs Python ONNX, eight cases | max probability difference 5.28e-8 |
| Typed-decisions: Rust vs eager FP32, eight cases | max probability difference 4.90e-7 |
| Native lifecycle + long-state + reload smoke | passed |
| English real long-state scan | six windows; selected spans and all three answer types match upstream |
| Mobile C boundary | Windows single/batch/long scans, budgets, deadlines, capability inspection and lifecycle pass; device runs pending |
| Android arm64 Rust/JNI/native dependency set | NDK r27d, API 24, ORT 1.24.3 linked; exports, dependency names and 16 KiB ELF alignment verified |
| Android debug host APKs | arm64 and x86_64 built/linted/signature-verified; native hashes, uncompressed 16 KiB alignment, synthetic assets and no permissions checked |
| Android emulator native smoke | API 35 x86_64: synthetic single/batch/window inference, deadlines and lifecycle pass; physical arm64 remains unverified |
| iOS arm64 device/simulator SDK package | iOS 15.1 floor, ORT 1.24.2, Xcode 26.6; Swift link, exports and XCFramework/app inspection pass |
| iOS simulator native smoke | iOS 26.5: single/batch/window inference, deadlines and lifecycle pass on the synthetic graph; physical devices and execution on iOS 15.1 remain unverified |
| macOS arm64 real models, all three families | eight-case ONNX/eager parity, cross-state batches, permutations and upstream long scans pass; largest differences 7.51e-8 vs ONNX / 1.47e-6 vs eager |
| Quality evaluation tools | six analytic/leakage tests and native synthetic pipeline passed; application holdout still required |
| Cross-state native batches, all three checkpoints | independent Python corpus passes, including empty states and heterogeneous rows |
| Local checkpoint exporter | English weights repackaged through local path; export and Rust inference passed with content-bound provenance (not a fine-tune quality test) |

Measured full envelopes were B=4/S=512/K=12 for English and B=4/S=1024/K=12
for the other checkpoints. Peak Python ONNX process memory was 1885.0, 1808.8
and 2843.7 MiB respectively, below each 4096 MiB reservation. These are host
measurements with two CPU threads and three runs per shape, not mobile latency
or thermal guarantees. Manifests, exact input corpus, eager/ONNX references and
shape reports are archived in `evidence/windows-x64/`.

This is an implementation preview. Domain-quality holdouts, sustained desktop performance, Windows arm64, signed
iOS installation and both mobile device/lifecycle tests
remain tracked gates. A successful compile or synthetic graph does not qualify
a real checkpoint on a mobile platform. See the approved scope in
`docs/plans/laya-support-design.html` and `examples/laya-mobile/README.md`.

The long-state API now checks the input byte budget and graph envelope before
tokenization, using a counting writer instead of allocating a duplicate JSON
buffer. Tokenization and every window now run as one bounded worker job, using
the worker's existing tokenizer. Queue saturation, unload and async-drop tests
verify that resources remain owned until the scan returns. Tokio exposes
`decide_long_async`; scan-level timings cover preprocessing and all windows.
The real English six-window reference still passes after this change.
Full performance runs must use the updated implementation. The earlier partial CPU run in
`target/laya/benchmarks-20261004` is preserved as superseded evidence, not a
completed performance gate.

The Android host app and build/inspection commands are documented in
`examples/laya-mobile/README.md`. APK and native-library evidence are separate:
a packaged debug app does not prove installation or inference on a device.
The CPU run under `target/laya/benchmarks-worker-scan-20261004` was interrupted
by a session reset and remains partial. It records
concurrent SDK/Gradle activity in its provenance; it is an active-host experiment,
not an isolated performance release qualification.

The desktop native smoke and Android packaging CI lanes passed at commit
`cfb3b37639e637124195994729666004df98c4ca`; see
[`evidence/ci/desktop-android.json`](evidence/ci/desktop-android.json). They cover
synthetic inference on Windows/Linux/macOS and Android native/APK packaging.
The separate Apple SDK workflow builds device/simulator libraries and links the
Swift host; Apple SDK inspection passes at iOS 15.1 (see `evidence/ios-arm64/sdk-build-evidence.json`). The app/simulator lane now passes native synthetic inference on iOS 26.5; see `evidence/ios-arm64/simulator-evidence.json`. Neither lane replaces real-model
or physical-device qualification.

### Real-model qualification on another desktop host

`qualify_native.py` records independent Python ONNX/eager references and runs
the Rust native comparisons against the committed preprocessing and permutation
corpora, then compares long scans with upstream. It records manifests, source
hashes, toolchain versions and tolerances. For an exported bundle:

```sh
python tools/laya/qualify_native.py --source target/laya/upstream \
  --bundle target/laya/english --output target/laya/qualification-english
```

The `Laya real-model macOS parity` workflow runs all three checkpoint families
on separate CPU hosts. It can be dispatched manually and runs when its driver
changes; routine CI remains weight-free. Its artifacts contain references and
logs, never model weights. These parity checks do not establish sustained
performance, accelerator behavior, mobile suitability or application quality.

The first full macOS arm64 CPU qualification passed for all three checkpoint
families at `be4f912`. Raw references, comparison logs and manifests are archived
in `evidence/macos-arm64/`; `evidence/ci/macos-real-models.json` links the jobs.
