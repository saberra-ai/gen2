# Linux chat/diar coexistence fixture: immutable public metadata

Research only, 2026-10-16. No model bytes were downloaded and no build/inference was run. JSON metadata and small model-card/license text were read. Future Linux tests must freeze their chosen artifact and bind the actual local size/SHA256 to their own receipt; the Mac result is not Linux qualification.

## Exact Mac artifact located

The [Mac N0 receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-15-n0/receipt.json) records chat SHA256 `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`. The publisher's [immutable Hugging Face metadata](https://huggingface.co/api/models/unsloth/Qwen3-0.6B-GGUF/revision/50968a4468ef4233ed78cd7c3de230dd1d61a56b?blobs=true) identifies an exact matching artifact:

| Field | Frozen value |
| --- | --- |
| Repository/publisher | `unsloth/Qwen3-0.6B-GGUF` (Unsloth quantization; not Qwen's own GGUF repository) |
| Repository revision | `50968a4468ef4233ed78cd7c3de230dd1d61a56b` |
| Filename | `Qwen3-0.6B-Q4_K_M.gguf` |
| Bytes | `396705472` |
| LFS object SHA256 | `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a` |
| Git LFS pointer blob ID | `a09d43d97b8250d5de215fbebce5499cdbc1a448` |
| Access metadata at inspection | Public, `gated: false` |

Use the [commit-pinned artifact URL](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf). Matching the publisher's content hash and size provides a public acquisition route for the Mac fixture's exact bytes; it does not establish how that local Mac file was originally acquired. The pointer blob ID is not the weight-file SHA256.

The [pinned publisher model card](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/50968a4468ef4233ed78cd7c3de230dd1d61a56b/README.md) declares base model `Qwen/Qwen3-0.6B`, license `apache-2.0`, and links Qwen's license. Card bytes: `17082`; SHA256 `03994ad190e9e314ae0bff6f2cf780528f09ea70e0f22a267e287c8748a850f5`. The inspected quantization revision does **not** contain a standalone LICENSE file; preserve its card and the linked upstream license at an explicit pin. Qwen base-model revision `c1899de289a04d12100db370d81485cdf75e47ca` contains the [Apache-2.0 LICENSE](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE), `11343` bytes, SHA256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`. This is a frozen license-text source, not proof that Unsloth converted that exact base-weight revision.

## Optional official Qwen alternative

Exact Mac bytes are available, so a replacement is unnecessary. If an official-Qwen publisher is desired, [Qwen's pinned metadata](https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF/revision/23749fefcc72300e3a2ad315e1317431b06b590a?blobs=true) provides this separate candidate:

| Field | Frozen value |
| --- | --- |
| Repository/revision | `Qwen/Qwen3-0.6B-GGUF` / `23749fefcc72300e3a2ad315e1317431b06b590a` |
| Filename | `Qwen3-0.6B-Q8_0.gguf` |
| Bytes | `639446688` |
| LFS object SHA256 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Terms/access | Apache-2.0; public, ungated at inspection |

[Artifact](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf), [model card](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/README.md), [license](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/LICENSE). Its license bytes are `11544`, SHA256 `5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd`; card bytes `6236`, SHA256 `44adabae2153c3a7bb741b86fbffd48736bc95118831935fdfc4d3cc1854f2f5`. This Q8 artifact differs from the Mac Q4 fixture and needs its own Linux execution evidence.

For either choice, freeze model/prompt/token limits/seed/chat backend and loaded-library identities before execution. Record the actual acquired file hash/size, template/settings and chat output alongside the extracted diarization package identity, both load orders and lifecycle results. Do not infer Linux coexistence, output parity, runtime compatibility or performance from repository metadata or Mac results.
