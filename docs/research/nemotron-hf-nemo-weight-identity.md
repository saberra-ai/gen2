# Pinned HF / NeMo source-weight identity audit

2026-10-16. **Actual result: all 417 official HF F32 tensors are byte-identical to the pinned converter's mapping and exact BF16→F32 cast of the original `.nemo` checkpoint.** This closes checkpoint-parameter identity as a confounder for the separate HF/native diagnostic. It does not establish frontend, arithmetic, streaming-state, output or quality equivalence. No neural model was instantiated or executed; no conversion, source mutation, installation, scoring, threshold search or frozen-protocol change occurred in this audit.

## Exact artifacts and method

The audit first verified the canonical [runtime lock](/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/runtime.lock.json), 4,917 bytes, SHA256 `e66e5651608636bbb04528fed7944ab8972a778d1141b36fef3fb391eefd0dcc`, then verified the locked HF artifact against its actual file. Both publisher artifacts are pinned to NVIDIA repository revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`; the [publisher inventory](https://huggingface.co/api/models/nvidia/Nemotron-3-Diarization/revision/f667ed73aee57d40cc39428eb768b4fd87a0a29e?blobs=true) identifies their separate file objects.

| Actual input | Bytes | SHA256 |
| --- | ---: | --- |
| [Original `.nemo`](https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.nemo) | 198,676,480 | `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d` |
| Archive member `model_weights.ckpt` | 198,666,820 | `9930e1719c85b35da083c557cdbd6503072c311ffe8bc9cbd3371d2609e21d0f` |
| [Official `model.safetensors`](https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/model.safetensors) | 396,954,592 | `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6` |
| [HF conversion source](https://github.com/huggingface/transformers/blob/683b721971417b3b2c64fba9001f525607547e5a/src/transformers/models/nemotron3_diarization/convert_nemotron3_diarization_to_hf.py) | 9,445 | `6a4de669f624642eb723d5361900f0e6c6c07d1d7153edf645e2fae0e0c71da1` |

Only `STATE_DICT_MAPPING`, `KEYS_TO_DROP`, `split_fused_qkv` and `convert_state_dict` were AST-extracted from that unchanged converter into a namespace containing `re`. The Transformers module was never imported. The checkpoint was deserialized with `torch.load(map_location="cpu", weights_only=True)`; HF tensors were read with `safetensors.safe_open(framework="pt", device="cpu")`. This is weight-data decoding, not inference. Actual environment: Torch 2.8.0, NumPy 2.2.6, safetensors 0.8.0, Transformers metadata 5.21.0.dev0 with installation commit `683b721971417b3b2c64fba9001f525607547e5a`.

The source contains 363 tensors, all BF16. The 31 fused attention QKV weights split along dimension 0 in query/key/value order, producing 93 HF weights (62 additional names). After eight exclusions, `363 + 62 − 8 = 417`. All target names, shapes and F32 dtypes match the HF safetensors inventory; every contiguous little-endian F32 byte matches the corresponding renamed/split source tensor cast to F32. The report retains all 363 raw-source hashes, every QKV source/row interval/target, and all 417 expected/actual byte hashes. No tolerance was used.

## Exact HF exclusions and native treatment

These are the eight actual source keys matched by HF's drop patterns. Shapes and BF16 source hashes are retained in the audit report, including keys omitted from both graphs.

| Source key | Shape | Pinned native treatment |
| --- | --- | --- |
| `preprocessor.featurizer.window` | `[400]` | Excluded; native frontend constructs its window |
| `preprocessor.featurizer.fb` | `[1,128,257]` | Separately stored as F32 `preprocessor.fb`, singleton removed |
| `sortformer_modules.hidden_to_spks.weight` | `[8,384]` | Excluded |
| `sortformer_modules.hidden_to_spks.bias` | `[8]` | Excluded |
| `sortformer_modules.activity_head.0.weight` | `[192]` | Stored as F32 `head.activity_head.0.weight`; unused by pinned graph |
| `sortformer_modules.activity_head.0.bias` | `[192]` | Stored as F32 `head.activity_head.0.bias`; unused by pinned graph |
| `sortformer_modules.activity_head.1.weight` | `[3,192]` | Stored as F32 `head.activity_head.1.weight`; unused by pinned graph |
| `sortformer_modules.activity_head.1.bias` | `[3]` | Stored as F32 `head.activity_head.1.bias`; unused by pinned graph |

The [native converter, lines 146–170](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/conversion/diarization.py#L146-L170) skips window/auxiliary speaker keys, emits the filterbank separately at lines 402–406, and generically remaps remaining `sortformer_modules.*` to `head.*`. Thus its loop reports four skips, including the separately emitted bank; there are only three source tensors absent from the resulting 360-tensor artifact. The existing [construction accounting](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-native-f32-construction/source-tensor-accounting.json) records all four activity-head destinations and exact cast-byte matches. This audit crosschecked those rows against its independently decoded source hashes; it did not reread or regenerate the native artifact.

**Stored does not imply used.** At pinned native revision `083861c241061562b8ec29eb06d7f6cdc3af8775`, [SortformerGraph](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.cpp#L118-L166) constructs, defines and binds only `head.first_hidden_to_hidden` and `head.single_hidden_to_spks`. Its [V3 forward path, lines 237–245](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/diar/sortformer_model.cpp#L237-L245) applies ReLU → first head → ReLU → speaker head → sigmoid. No `activity_head` reference exists in `src/asr/diar`. This is source-backed non-use by that graph; it does not claim an independent binary execution trace or change the historical construction report's generic “inference speaker head namespace” label.

The native source identities retained in the addendum are `conversion/diarization.py` SHA256 `b246c039dd56fea63500f105d77be17eafbc73f9c79fb6c8cd6e420101ed2317`, `sortformer_model.cpp` SHA256 `831d819e0a3986b7b6007987a67b4ccc5f2895d6d68247ca138ed4d9c3c3d2c9` and its header SHA256 `30a3daf3396d628294457ab9e2c8b1d8ab63829ab28f68dc1e8e96f11c635893`.

## Retained evidence and remaining boundary

The [eight-file metadata receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-hf-nemo-weight-identity/qualification.json) retains the full tensor-hash report, mapping/AST/script, source-backed unused-head addendum and collection checks. Publisher tensor bytes remain outside Git in [the cache directory](/Users/victor/.cache/gen2-nemotron3-n0/research/hf-nemo-weight-identity). `audit.py` SHA256 is `d879cb5a068732550fb1cb29b53e1156f4ca94413d0f1c2e8d43abc0730ce085`; `report.json` SHA256 is `2228086e215f947dfd7170f1f948c9da920b576e3b1a1223f78ff39a976dc79f`. The extracted AST digest is `383c14e90ea8a1f6e90d62bbd30979d82ce2b8ce0c4affe5298bca638d9cfe53`. `native-unused-head-audit.json` SHA256 `8a18d654f40529526697e0527322755ccf6916db12d3950b8b9e93823df9c415` separately pins the inspected native source and unchanged historical construction-accounting file.

This result permits interpreting a separate frozen-PCM HF/native comparison without assuming the two publisher checkpoints contain different mapped parameters. HF still reconstructs frontend buffers that it drops; native retains the source bank and changes convolution storage precision. Those boundaries, runtime arithmetic and state handling still require their own evidence. No original quality failure, proposed gate, model lock, scorer, production configuration or release status changes here.
