# Calibration-only teacher-forced HF/native V3 graph control

2026-10-16. Proposed protocol only: no compilation, inference, frontend execution, model import or source mutation here. Scope is the excluded17872-sample/1.117s calibration, using native F32-dominant/F16-conv weights and pinned HF FP32. No AMI targets, DER scoring, threshold selection or production substitution. This controls graph inputs and the HF-provided FIFO, rather than repeating a free-running comparison.

## Fixed tensor boundary

[Native SortformerModel::run_chunk](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.h#L152-L183) accepts F32 mel as frame-major `[T,128]`, and separate F32 pre-transformer speaker-cache/FIFO arrays `[frames,512]`. These are projected feature-stack embeddings **before** RoPE encoder normalization/attention, not encoder hidden states or post-encoder512→192 projections. `run_chunk` copies its input arrays into a request and owns no AOSC state. It returns `chunk_embs`, coarse `preds` and full10ms `native_preds`; it neither emits final streaming slices nor advances a FIFO. The [native implementation](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.cpp#L320-L337) batches V3 requests only with equal mel/state lengths.

| Fixed call | HF/native mel input | State before call | Native `chunk_embs` | Full graph output | HF returned output |
| --- | --- | --- | --- | --- | --- |
| First9center+4right | `[104,128]` | cache0/FIFO0 | `[13,512]` |104×8probabilities / HF104×8logits |72×8logits |
| Final valid tail | `[39,128]` | cache0/FIFO9 | `[5,512]` after internal1-row zero pad |112×8probabilities / HF112×8logits |39×8logits |

All Python neural inputs are batch-one `[1,T,D]`, CPU float32, contiguous. Serialize raw arrays as little-endian F32 in C row order; record shape/bytes/hash before invoking either consumer. Native pointer order is128consecutive bins per mel frame and512consecutive embedding channels per FIFO frame. Native produces frame-major eight-speaker rows. The final full sequence has9cached+5new=14encoder tokens and112high-resolution rows. Compare final returned HF sigmoid probabilities to native rows **`[72:111]`**, retaining all112raw rows. The first returned comparison is native `[0:72]`.

Native exposes probabilities, not logits: compare `native_preds` with `sigmoid(HF full classifier logits)`. Do not inverse-logit native0/1values, compare sigmoid probabilities to raw logits, or compare its14-row coarse `preds` to112-row native output. The coarse prediction averages sigmoid probabilities over each8-row group; it is only an optional bookkeeping comparison. [Native output assembly](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.cpp#L443-L469).

## Capture HF without changing source or losing pre-update state

Use the original pinned [HF forward](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L731-L802). Supply the same already-frozen native-bank HF mel values to both runtimes, or separately preregister their creation; do not feed two independently recomputed frontends and call their inputs identical. For this graph control no frontend implementation belongs in the C++ helper.

Attach non-mutating [PyTorch2.8 module hooks](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook), always returningNone and cloning `detach().cpu().contiguous().clone()` tensors immediately:

- `model.model.audio_tower.embedder.register_forward_hook(...)`: capture the projected feature-stack outputs `[1,13,512]` first and `[1,5,512]` last. This is the exact counterpart of native `chunk_embs`, not raw stacked1024-vectors. A pre-hook can capture the104/39mel rows supplied to it.
- `model.model.register_forward_pre_hook(...,with_kwargs=True)`: capture `kwargs['inputs_embeds']`, `attention_mask` and `position_ids` **before** the base model runs. Expect first `[1,13,512]`, all-true13mask, positions0..12; final `[1,14,512]`, all-true14mask, positions0..13. Assert there is no raw `input_features` shortcut in this hook and that cached/current partitions are9/5 on the final call.
- `model.classifier.register_forward_hook(...)`: capture the full classifier output logits104/112rows before its caller updates cache and crops. The pinned [head](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L619-L630) is `Nemotron3DiarizationClassificationHead`, exposed as `model.classifier`; it applies ReLU→dense→ReLU→out_proj. Do not hook only the top-level returned logits and lose cached/padded rows.

First top-level call:104mel/all-true104mask, `num_lookahead_frames=4`, cacheNone. It emits72rows and updates the returned cache. Before the final call, clone active `cache.embeds[:,:num_cache_frames]`, `cache.fifo[:,:num_fifo_frames]`, relevant `cache.probs` active prefix, counts and flags. Assert cache0/FIFO9, uncompressed, and FIFO bytes exactly equal the first9captured embedder output rows. [HF cache get/update](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L95-L191) mutates preallocated buffers with `index_copy_`; views/references saved without `.clone()` can silently become post-update evidence. There is no compression or learned-silence sentinel insertion on this two-call calibration.

Final top-level call:39mel/all-true39mask, the existing HF cache, and **no `num_lookahead_frames` key**. The non-None cache selects streaming and default lookahead0. Its base-input pre-hook must show that the first9tokens equal the cloned pre-final FIFO and final5equal this call's captured embedder output. Capture returned39logit rows separately and verify they equal full classifier logits `[72:111]` exactly. Remove hook handles in `finally`. No hook may replace inputs/outputs or invoke an additional model pass.

Pass the cloned **HF** FIFO to native on the final call, regardless of native first-call output. This deliberately removes accumulated native state differences. Native recomputes its own current chunk's feature projection from the identical mel; comparing `chunk_embs` quantifies that remaining projection boundary. A run_chunk helper cannot directly override current chunk embeddings; that would require a lower graph/session interface and expand scope.

## Smallest native CPU helper and link caveat

[Runtime Params/BackendManager](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/runtime/ggml/runtime.h#L202-L244) and [constructor](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/runtime/ggml/backend.cpp#L40-L44) support this proposed construction:

```cpp
ggml_runtime::Params p{};
p.use_gpu = false;
p.n_threads = 4;
ggml_runtime::BackendManager bm(p); // constructor initializes backends
nemo_speech::asr::BatchingConfig b{};
b.enabled = false;
nemo_speech::asr::SortformerModel model(bm, locked_f32_gguf, b);
auto first = model.run_chunk(first_mel.data(), 104, nullptr, 0, nullptr, 0);
auto last = model.run_chunk(last_mel.data(), 39, nullptr, 0, hf_fifo.data(), 9);
```

Do not call `init_backends()` a second time. Model/session must be destroyed beforebm. Assert V3,128mel,512encoder,8speakers, output factor1/upsample8 before calls. No `DiarModel`, `DiarStream`, frontend or `AoscState` is required. Retain outputs without mutating any supplied buffer.

The SDK exports `SortformerModel` but its [internal runtime library](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/runtime/ggml/CMakeLists.txt#L8-L28) is hidden STATIC and not installed. Read-only `nm -gU` on the actual CPU ASR dylib confirms SortformerModel constructor/run_chunk exports and no BackendManager constructor export. Consequently `-lnemo_speech_asr` alone is insufficient for this helper, unlike the earlier frontend-only probe. Upstream CMake tests inherit the runtime static archive via PUBLIC target linkage.

**Preferred implementation route, coordinated with the root implementer:** compile only unchanged pinned `src/runtime/ggml/backend.cpp` and `src/runtime/ggml/logging.cpp` as qualification helper objects, with hidden visibility, linking the existing exact CPU ASR dylib plus matching SDK `ggml`, `ggml-base`, `ggml-cpu` libraries. `backend.cpp` includes `model_logging.h`, whose logging setup is inline; `logging.cpp` supplies `ggml_runtime::log_internal`. Pin both actual translation-unit bytes, every consumed header, compiler/version/flags, object/output hashes, linker map and loaded dependency paths. Do not add these objects to the SDK or compile another graph/session implementation. Preserve public imported SortformerModel linkage when applying per-object visibility flags; the actual link-only check must verify this. A first prototype failure from trying an uninstalled `-lnemo_speech_runtime_ggml` is separate evidence retained by the implementer; no successful helper build is claimed here.

A retained CPU static archive also exists,235304bytes, SHA256 `abc24b53048e8456616f9280cf5f0aad1ef7e32cfbe24139ef62ae815e8ba1db`, but its producer provenance would be an additional boundary. It is not the preferred build input. Do not search-link an arbitrary runtime archive or accidentally resolve another SDK/chat GGML. CPU Params disable GPU selection, but BackendManager also enumerates accelerator backends; bind the actual CPU-only SDK and retain backend/dependency observations rather than assuming the flag alone defines every loaded backend.

Needed include roots: native `src/asr/diar`, `src/asr`, `src/asr/encoder`, `src/runtime/ggml`, `src/common`, and pinned `llama.cpp/ggml/include`; include `sortformer_model.h`. Keep the matching CPU SDK lib directory as the sole qualification RPATH and use C++17/arm64, following the existing frontend build admission. The existing private production runtime package is unchanged. A separate helper process avoids introducing this internal C++ seam into Gen2's public API.

## Freeze comparisons and optional shape control in advance

Compare all first13/final5embedding rows, all first104/final112full probability rows and first72/final39returned slices. Retain scalar errors plus exact max location; no runtime parity tolerance is derived from these future outputs. Pin a meaningful diagnostic policy before execution, or report numerical results without claiming a newly invented pass. Preserve the F16 subpixel-convolution boundary and CPU arithmetic as remaining differences even with parameter and input identity.

The coordinated four-case plan adds a **preregistered** no-new-HF-call shape control feeds native the same39mel rows followed by one explicit zero feature row (`T=40`), using the same cloned FIFO. Both39internal-zero-pad and40explicit-zero inputs have identical5×1024stack values and should give the same full112probabilities; native's declared final crop length differs only outside this direct graph API. Do not substitute its computed native terminal mel row in this control. The fourth case uses the retained computed native terminal mel row as row40, keeping the first39HF rows and same HF FIFO unchanged. It is a deliberate fixed-content intervention, not identical graph inputs; bind its exact source tensor/hash and compare it separately against the same HF39full output. No third HF call is required for this intervention. Neither variant changes baseline framing or proves the extra row caused the original free-running error.

Source identities: native `sortformer_model.h` `30a3daf3396d628294457ab9e2c8b1d8ab63829ab28f68dc1e8e96f11c635893`, `runtime.h` `8b16754a48bc69a9d0f9d2bc29913e9cf4a7e9b5321bd118780093010c5c0f25`, `backend.cpp` `92ea19f833c9ba132c81cfe33693ab1090d436e25c2aaea7e6186315f5265394`, batching header `d5933c7db797a7722fbe8bdba445ef82093b38bdeb0da36afeb88c0b985c5857`. HF model source remains `6f7688dbcff3123216fa8860af73af934912683f2191e6b36adc4dd19a39fcad` at683b721. The earlier v3.0-tag correction is preserved: FeatureStacking exists there; the complete V3 Sortformer path is the unsupported boundary, not absence of that class.

## First execution admission failure and separate repair

Protocol `dc02371` was reviewed and committed before the first attempt. Three grouping tests passed; official HF weights loaded417tensors, then evidence serialization raised `TypeError: Object of type set is not JSON serializable`. The pinned Transformers loading-report API returns sets for missing/unexpected/mismatched keys. This happened before either HF forward or any native graph process started. No probability or numerical-control result exists. The [29-file failure receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-calibration-graph-admission-control/qualification.json) preserves the original source/protocol, commands, complete logs, raw input hashes and initial failed/successful probe compile records. Original protocol, tools and attempt prefix remain unchanged.

A distinct V2 controller/protocol repairs only public loading-report serialization: sets become deterministically sorted JSON arrays, while all diagnostics and raw nonempty-report rejection remain intact. Three serialization regressions join the three grouping tests before a fresh attempt; model weights, fixed Mel/FIFO policy, twoHF/fournative case counts and comparison definitions remain unchanged. Both reviews cleared this amendment before admission. This is an execution-adapter repair, not a numerical setting or quality-gate change.
