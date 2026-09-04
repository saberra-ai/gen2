# Research decision 01 — Backend tiers

Status: DECIDED 2026-09-03 (research-decision pipeline, deep fork). Owner: Victor. Re-check: 2026-12-01 or on any trigger in §9.

## 1. FORK

| Field | Value |
|---|---|
| Decision | Which of gen2's 8 backends are Tier 1 (blessed: real-model conformance in CI, README headline, supported), Tier 2 (compiles + no-model conformance in CI, labelled experimental, real-model verification by hand), Tier 3 (removed, or parked behind a doc note). |
| Options | `backend-llamacpp`, `backend-external-api`, `backend-litertlm`, `backend-mistralrs`, `backend-mlx`, `backend-mlxcel`, `backend-onnx`, `backend-candle` |
| Stakes | Deep. Product-defining for the README headline and for what pio-app builds against. Reversible: features stay independent, code stays in git history. |
| Constraints | crates.io rejects git dependencies (today: `llama-cpp-2`, `llama-cpp-sys-2`, `mlx-rs`, `mlxcel`, `mlxcel-core` — 5 git lines, 3 upstreams). No paid GPU CI. Apple Silicon (macOS 26) is the primary dev machine; Windows RTX 3080 and Pi 5 are the other owned targets; iOS/Android are ambitions. |
| Success metric | Priority stack from the roadmap: DX/correctness > maintainability > publishability > performance > breadth. A Tier-1 backend must be provable in CI without a human or a paid runner. |
| Blast radius | README, CI matrix, `Cargo.toml` features, `conformance.rs` `NEVER_PRODUCED_A_TOKEN`, and pio-app's forwarded features (`src-tauri/Cargo.toml` lines 143-149). |
| Expiration | 90 days. Every upstream here ships monthly or faster. |

## 2. LANES

| Lane | Hit? | Where |
|---|---|---|
| Primary / official | Yes | crates.io API (versions, dates, features, reverse deps), GitHub API (stars, push dates, licenses, release dates), upstream READMEs, arXiv 2511.05502, Google LiteRT-LM release notes |
| Academic / evaluation | Yes (thin) | arXiv 2511.05502 (M2 Ultra runtime comparison); arXiv 2601.19139 lead only |
| Practitioner | Yes | reverse-dependency counts, Ollama's MLX switch, Tauri+LLM sample apps, mistral.rs/llama-cpp-rs/mlx-rs issue trackers, iPhone runtime benchmark (john-rocky) |
| Deployment | Yes | Apple Silicon MLX vs llama.cpp Metal numbers; iPhone 17 Pro burst vs sustained; Pi 5 CPU tok/s; Windows CUDA build failures |
| Critical / failure | Yes | mlx-rs #308 (macOS 26 Metal, open); llama-cpp-sys-2 #970 (Windows cmake); mistral.rs #807/#903/#935 (Metal gaps); ort rc.10-rc.13 breaking notes; docs.rs failed to build mistralrs 0.8.1; candle has no GitHub releases |
| User-context | Yes | gen2 `src/backend/{mod,conformance,caps}.rs`, README "Backends", `Cargo.toml`, `.github/workflows/ci.yml`; pio-app `src-tauri/Cargo.toml`, `pio-core/Cargo.toml`, `pio-daemon/Cargo.toml`, `scripts/check-rust-gates.sh`; memory notes on MLX/Tahoe |

Freshness swept (all queries 2026, all release dates pulled from APIs dated today). Snowball done (llama-cpp-rs pinned commit → crates.io release dates; mlxcel fork → lablup upstream → Kubesimplify/backend.ai coverage; mlx-rs fork → oxiglade upstream → issue #308). Lateral read done on the two synthetic "2026 benchmark" blogs (yage.ai, contracollective) against arXiv 2511.05502 and Ollama's own announcement. Note: `WebFetch` mis-rendered several GitHub release years as 2024; every date below was re-pulled from the GitHub or crates.io API and is 2026.

## 3. FORAGE — the facts that decide it (graded)

### Upstream health (A = API data pulled 2026-09-03)

| Upstream | Stars | Last push | crates.io latest | Reverse deps | Note |
|---|---:|---|---|---:|---|
| utilityai/llama-cpp-rs (`llama-cpp-2`) | 644 | 2026-09-03 | 0.1.156 (2026-09-02) | 76 | 12 releases since April. "Does not follow semver meaningfully" (README, A). Windows cmake failure #970 on 0.1.140 (C). |
| EricLBuehler/mistral.rs (`mistralrs`) | 7,652 | 2026-09-03 | **0.8.1 (2026-04-02)** | 25 | GitHub is at v0.9.2 (2026-08-20): **crates.io lags git by 5 months**. docs.rs failed to build 0.8.1 (C). Metal gaps: #807 "contiguous affine I64 not implemented", #903 Metal-vs-MLX perf tracking, #935 vision on Metal (C). |
| oxiglade/mlx-rs (`mlx-rs`, was oxideai) | 371 | 2026-09-02 | **0.25.3 (2025-12-16)** | 27 | Issue #308 "Metal support on macOS 26.2 not available" opened 2026-01-15, **still open** (B). The saberra-ai fork (rev fac86ba0, 2026-07-07) carries the mlx-c 0.6.0 / Metal 4 fix. Upstream pushes are dependabot + CI. |
| huggingface/candle | 20,991 | 2026-09-03 | 0.11.0 (2026-06-26) | 303 (`candle-transformers`) | Healthy. Has `quantized_qwen3`, `quantized_gemma3`, `gemma4/`, `qwen3_vl/` (B). No GitHub releases page at all (C). |
| pykeio/ort | 2,489 | 2026-09-02 | 2.0.0-rc.13 (2026-07-28) | 551 | Still rc; rc.11/12/13 each carried a breaking change (ndarray 0.17 + env rename; `api-24`; EPs behind cargo features) (A). gen2 pins `=rc.12`. |
| google-ai-edge/LiteRT-LM | 6,368 | 2026-09-04 | n/a (dlopen) | n/a | Monthly: v0.13.0 (06-02) … v0.16.0 (08-11), v0.16.1 (08-18) (A). **Versioned C API prebuilts for all platforms only since v0.16.0 — three weeks old.** 618 open issues. |
| lablup/mlxcel (`mlxcel`, `mlxcel-core`) | 385 | 2026-09-04 | **not on crates.io** | 0 | v0.5.2 (08-17), v0.6.0 (08-22), v0.7.0-beta.1 (09-04): weekly cadence. Server/CLI-first product; 80+ archs; ~119% of mlx-lm decode (C, vendor+Kubesimplify). saberra-ai fork rev 21306cf (2026-07-07) adds `set_metallib_path`, needed for app bundling; fork is 2 months and 3 upstream releases behind. |
| ggml-org/llama.cpp | 126,955 | 2026-09-03 | — | — | The reference. |
| ml-explore/mlx | 28,294 | 2026-09-04 | — | — | v0.31.x. |

### What gen2 has proven (A — `src/backend/conformance.rs`)

- `NEVER_PRODUCED_A_TOKEN = ["mlxcel", "onnx", "candle"]`.
- Proven: `llamacpp` (GGUF + all of `tests/live_inference.rs`), `mlx` (llama-3.2-3b-4bit on macOS 26.3, Metal Toolchain installed), `mistralrs` (GGUF, **CPU only**), `litertlm` (Qwen3-0.6B.litertlm, v0.16.0 macOS arm64 prebuilt, `PIO_TEST_CTX=4096`).
- `candle` is a declared stub: `upload_settings` → `Unimplemented` (358 LOC).
- `mlx` is 15,392 LOC, the largest module in the crate: an in-house model zoo (`model/{llama,gemma4,gemma4_fast,diffusion_gemma,eagle3,moe,vision}.rs`) plus Eagle-3 speculative decoding. `mlxcel` is 1,751 LOC and delegates the model zoo to lablup.
- `src/backend/mlxcel/mod.rs` records why it exists: the mlx-rs path "was measured ~8-11x slower than mlx-lm".
- `src/backend/mistralrs/` selects no device and `Cargo.toml` forwards no `mistralrs/metal` or `mistralrs/cuda` feature — so the backend is CPU-only in every gen2 build today.
- CI (`.github/workflows/ci.yml`): every backend compiles and passes the no-model half on ubuntu (mac for litertlm/metal); `backend-mlx` gets clippy only; iOS/Android cross-compile checks run for `backend-litertlm` only. **No job runs a real model.**

### What pio-app uses (A — pio-app tree, 2026-09-03)

- `src-tauri` default: `backend-llamacpp` (+ flock etc). `apple` bundle: `backend-llamacpp, backend-mlxcel, metal`. `ios`/`android`: `backend-llamacpp` + metal/vulkan. `desktop-all`: `backend-llamacpp, backend-onnx, backend-external-api`.
- `pio-core` default: `backend-external-api`. **`backend-onnx = []` in pio-core is an empty stub** used only to satisfy the "at least one backend" guard for the `vision`/`clip` features; ort is used there for vision/NER/diarization, never for chat.
- `pio-daemon` default: `backend-llamacpp`.
- cfg sites in pio-core: llamacpp 50, mlx 33, mlxcel 29, external-api 19, onnx 18, candle 2, executorch 2. Live tests exist for llamacpp (many) and mlx (`gemma4_multiturn_integration`, `flock_live_inference_integration`). None for mistralrs, onnx, candle.
- All Rust gates (`scripts/check-rust-gates.sh`, CI) run with `--no-default-features --features backend-llamacpp`.
- Memory (2026-06-20): the mlx-rs fork bump is what made Gemma-4 26B run on Tahoe (6/6 on the birds eval); the packaged Apple build stubs `resources/mlx/mlx.metallib` for mlxcel.

### Deployment reality

- **Apple Silicon.** MLX beats llama.cpp Metal on steady-state decode by 1.4-1.8x for <14B dense and ~3x on MoE; converges above ~27B (memory-bound); llama.cpp wins prefill and long context (30K+, MLX ~50% slower without flash attention) — yage.ai (C) laterally consistent with arXiv 2511.05502 (A: "MLX achieves the highest sustained generation throughput… llama.cpp is highly efficient for lightweight single-stream use"). Ollama switched its Apple engine to MLX on 2026-03-30, routing GGUF→llama.cpp and safetensors→MLX (C; the same routing rule gen2 uses).
- **Windows RTX 3080.** llama.cpp CUDA is the proven path in the crate and in pio-app; mistral.rs claims CUDA wins on paged-attention/MoE workloads on datacenter parts (GB10/B200/H100, vendor B) and one discussion (#612, C); no consumer-GPU head-to-head found. Both need a CUDA toolchain at build; neither can be proven in CI without a paid runner.
- **Pi 5.** llama.cpp CPU: 10-18 tok/s on 1-3B Q4, ~4-7 tok/s on 3B (C, multiple blogs agree). It is the only realistic option; mistral.rs CPU also works but drags candle+tokio+reqwest into the build.
- **iPhone 17 Pro, Gemma 4 E2B 4-bit** (john-rocky, C, reproducible harness on GitHub): burst LiteRT-LM 55-56 tok/s > MLX 47.5 > llama.cpp 37.8 > CoreML/ANE 33; sustained 10 min: LiteRT-LM 27, ANE 22, MLX 18 (GPU thermal collapse). Android S25 Ultra: LiteRT-LM beats llama.cpp CPU/GPU on prefill and decode, NPU +3x prefill (Google blog, A but vendor).
- **Publishing.** The `llama-cpp-2` git pin (43733d1, merged 2026-06-23, PR #1027 MTP) **predates** crates.io 0.1.151 (2026-07-06); 0.1.156 exposes `mtmd`, `llguidance`, `metal`, `cuda`, `vulkan`, `android-*`, `dynamic-link` features (A). So one of the three git upstreams is droppable today; the other two (mlx-rs fork, mlxcel) have no registry version at all.

## 4. RUBRIC

Weights follow the roadmap's priority stack. Scores 0-5. Evidence cited in §3.

| Criterion | Weight | What it measures |
|---|---:|---|
| Proven in gen2 | 25% | Decoded tokens through the public API (`conformance.rs`) |
| Upstream health | 20% | Release cadence, registry currency, open blockers |
| Publishable | 20% | Registry dependency exists and works on the owner's machine |
| Owner-app need | 15% | pio-app/daemon bundles and cfg sites today, owned devices |
| Perf on owned targets | 10% | Mac/Win/Pi/phone numbers above |
| Free CI provability | 10% | Real model, CPU/Metal on GitHub-hosted runners, no human |

| Backend | Proven | Upstream | Publish | Need | Perf | CI | **Score** |
|---|---:|---:|---:|---:|---:|---:|---:|
| llamacpp | 5 | 4 | 4 (git→registry is a one-line change) | 5 | 4 | 5 | **4.50** |
| external-api | 4 (mocked + pio-core default) | 5 | 5 | 5 | 3 (n/a) | 5 | **4.55** |
| litertlm | 5 | 4 | 5 | 2 | 4 | 4 | **4.15** |
| mistralrs | 4 (CPU only) | 4 | 4 (registry 5 months stale) | 1 | 2 (no accel forwarded) | 4 | **3.35** |
| candle | 0 (stub) | 5 | 5 | 1 | 2 | 4 | **2.75** |
| mlx (mlx-rs fork) | 5 | 2 | 1 | 3 | 2 (8-11x under mlx-lm) | 1 | **2.60** |
| mlxcel | 1 | 4 | 1 | 4 | 5 | 2 | **2.55** |
| onnx | 0 | 2 (as an LLM path; ort itself 5) | 4 (rc churn) | 2 (pio-app's is a stub) | 2 | 3 | **2.00** |

## 5. Critical alignment audit (code vs research)

Aligns:
- Format routing (GGUF→llama.cpp, safetensors→MLX, `.litertlm`→LiteRT-LM) is the same rule Ollama adopted in March; keep it.
- `conformance.rs` refusing to count a backend as working until it decodes a token is exactly the discipline the roadmap's anti-reference ("compiles, unproven") demands. It just is not wired to CI with a model.
- The `ios`/`android` bundles carrying LiteRT-LM additively is the right shape: the fastest phone runtime, zero link cost.

At risk:
- **mistralrs is CPU-only by construction** (no device choice, no accelerator feature forwarding). The README does not claim otherwise, but a consumer enabling it on a Mac will get CPU decode and blame gen2.
- **`llama-cpp-2` is pinned to git for no remaining reason** — the pinned commit is older than 0.1.151. Every day it stays, the crate is unpublishable for a reason that no longer exists.
- **Two MLX paths.** `backend-mlx` (15k LOC in-house model zoo, proven, slow, fork-only) and `backend-mlxcel` (1.7k LOC, unproven in gen2, fast, fork-only, not on crates.io). The crate cannot publish with either, and cannot afford to maintain both. pio-app's `apple` bundle already chose mlxcel.
- **ort rc churn.** rc.11, rc.12, rc.13 each broke something; gen2 pins rc.12 while pio-app's ort use (vision/NER) will want rc.13's EP gating eventually. As a chat backend it has never decoded a token and no ONNX LLM (onnxruntime-genai) binding exists in Rust.

Violates:
- The README lists eight features as peers. Three have never generated a token; one is a declared stub. That is the "backend matrix where most cells are 'compiles, unproven'" anti-reference, in the README's own table.

## 6. PREMORTEM (assume the call below is wrong)

1. *Expert annoyance:* "You demoted the only MLX path that works on macOS 26." True — `backend-mlx` is proven and `mlxcel` is not, in this repo. Mitigation: park, do not delete; the promotion gate for mlxcel is a decoded token in `conformance.rs`, and the fallback if that slips is to un-park mlx as Tier 2.
2. *No paper yet, practitioner favourite:* mlxcel (May 2026 first release, weekly cadence, 385 stars). Also rMLX (pure-Rust MLX server) surfaced as a lateral lead — not evaluated, noted in miss-risk.
3. *What practitioners ship:* Rust desktop apps ship on llama.cpp bindings (76 reverse deps, all Tauri samples) or mistral.rs (25); nobody found shipping on mlx-rs or ort for chat.
4. *Strongest counterexample:* MLX being 1.4-3x faster on the owner's primary machine. Performance is 4th in the stack, and the answer is "one MLX path, the fast one", not "no MLX path".
5. *Not searched:* MLC-LLM (paged KV, best long-context per arXiv 2511.05502) — no Rust binding, out of scope; ExecuTorch (pio-core has a 2-cfg stub) — not a gen2 backend; rMLX — lead only.
6. *Owner's stack changes the answer?* Yes, twice: pio-app's `apple` bundle already ships mlxcel (so demoting mlx follows the owner), and pio-core's `backend-onnx` is an empty stub (so removing gen2's onnx backend breaks nothing in pio-app).
7. *Wrong in 30 days if:* oxiglade/mlx-rs closes #308 and publishes 0.26 (mlx becomes publishable); lablup publishes `mlxcel-core` to crates.io (mlxcel becomes publishable); LiteRT-LM v0.17 changes the C ABI (litertlm conformance breaks — CI will say so); mistralrs 0.9 lands on crates.io (bump + forward accelerators).

## 7. THE CALL

**Tier 1 — blessed** (real-model conformance in CI, README headline, "supported"):

1. **`backend-llamacpp`** (default; `metal`/`cuda`/`vulkan`). Everywhere the owner runs: Mac, Windows, Pi 5, iOS, Android. Move the dependency from the git pin to `llama-cpp-2 = "=0.1.156"` (+ `llama-cpp-sys-2`), exact-pinned because upstream does not follow semver. CI: ubuntu CPU + macos-latest Metal against a small GGUF (Qwen3-0.6B Q4_K_M class, HF-cached).
2. **`backend-external-api`**. Zero toolchain, pio-core's default, mockito-provable. It is the "no C++ toolchain" first-run story the roadmap wants.
3. **`backend-litertlm`** — labelled *mobile*. Proven, dlopen (no link, no git dep), fastest burst and sustained runtime on iPhone, beats llama.cpp on Android CPU/GPU, monthly Google releases. CI: macos-latest with the v0.16.x arm64 prebuilt + Qwen3-0.6B.litertlm. Caveat stated in the README: the versioned C API is v0.16, three weeks old.

**Tier 2 — experimental** (compiles + no-model conformance in CI; real-model verification by hand, recorded in `conformance.rs`):

4. **`backend-mlxcel`** — *the* Mac fast path candidate. Gate to leave Tier 2: decode a token in `conformance.rs`, then either (a) lablup publishes `mlxcel-core`/`mlxcel` to crates.io with `set_metallib_path` upstreamed, or (b) it moves to a companion crate `gen2-mlxcel` that depends on gen2's public `Backend` trait and is not published to crates.io. Until then it cannot be in the published `Cargo.toml` at all.
5. **`backend-mistralrs`**. On crates.io, CPU-proven, unique for safetensors/UQFF off-Apple and CUDA paged attention. Required to stay: forward `mistralrs/metal` and `mistralrs/cuda` through gen2's `metal`/`cuda` features (today it is silently CPU-only), and say so in the README. Bump when 0.9.x reaches crates.io; do not chase git.

**Tier 3 — removed or parked:**

6. **`backend-mlx`** (mlx-rs fork) — **parked**, not deleted. Reason: fork-only forever (upstream #308 open eight months), 15k LOC in-house model zoo is the crate's largest maintenance liability, measured 8-11x under mlx-lm, mutually exclusive with mlxcel, and pio-app's packaged Apple build already chose mlxcel. Keep the module compiling under clippy-only CI (as now), drop it from the README table into a "Parked" note with the un-park rule: promote to Tier 2 if mlxcel has not decoded a token by the end of the wave, or if upstream mlx-rs becomes usable on macOS 26.
7. **`backend-onnx`** — **removed** from the crate (feature, module, `ort`/`ndarray` deps). Never decoded a token; no Rust ONNX-LLM path exists; rc churn; nothing in pio-app depends on gen2's onnx backend (pio-core's is an empty stub). Doc note explains what would bring it back (a `onnxruntime-genai` binding).
8. **`backend-candle`** — **removed**. A declared stub returning `Unimplemented` is the exact anti-reference; the "pure Rust / static binary" use case is served by mistral.rs, which is candle underneath.

Consequence for publishability: after (1), (7), (8), the published crate carries Tier 1 + mistralrs with zero git dependencies. The MLX family lives outside the registry until §9's triggers fire.

**Your call vs default:** LiteRT-LM in Tier 1 rather than Tier 2 is the one judgment the owner may reasonably reverse (pio-app ships no mobile build yet); everything else is the default I would take. Parking `backend-mlx` is the item with the most sunk investment — the un-park rule above is the safety net.

## 8. References to mirror

- **Product to mirror:** Ollama's Apple engine routing and messaging (GGUF→llama.cpp, safetensors→MLX; "preview" label for the new path) — the same shape as gen2's Tier 1 / Tier 2 split.
- **Implementation to inspect:** `pykeio/ort` rc.10 "alternative backends" API — a registry-published core that accepts out-of-tree backend crates. This is the pattern for a companion `gen2-mlxcel` crate. Also `mistral.rs` README feature/accelerator matrix for how to state per-feature GPU support honestly.
- **Standard to anchor on:** gen2's own `conformance.rs` `contract_with_a_model` — the promotion gate for every tier move is "decoded N>0 tokens through the public API, printed under `--nocapture`".
- **Negative example to avoid:** gen2's current README backends table (eight peers, three unproven); candle's release-less repo as a warning about relying on cadence you cannot see.

## 9. Re-check triggers

- oxiglade/mlx-rs closes #308 or publishes a crates.io release built on mlx-c ≥0.6.0 → re-score `backend-mlx` publishability.
- lablup/mlxcel publishes `mlxcel-core` to crates.io, or merges `set_metallib_path` → mlxcel Tier 1 candidacy (still gated on a decoded token in CI).
- `mistralrs` ≥0.9 on crates.io → bump, forward accelerators, re-run conformance on Metal.
- Any LiteRT-LM release ≥v0.17 → run the litertlm real-model job before merging anything else.
- pio-app ships an iOS or Android build → LiteRT-LM's Tier 1 status stops being a judgment call.
- Hard date: 2026-12-01.

## 10. Miss-risk

- No consumer-GPU (RTX 3080) head-to-head for mistral.rs vs llama.cpp CUDA was found; the CUDA ranking is inferred from datacenter numbers and from what is proven in the crate.
- mlxcel's "119% of mlx-lm" and "80+ architectures" are vendor/blog claims (C); gen2 has not measured it.
- The `llama-cpp-2` git→registry claim rests on dates (pinned commit 06-23 < 0.1.151 on 07-06); it is confirmed only when the build passes with the `mtmd,llguidance` features on 0.1.156.
- rMLX (pure-Rust MLX server) was surfaced late and not evaluated.
- pio-core's in-tree gen2 copy may have verified mlxcel separately (docs/plans/mlxcel-embedding-roadmap.md in pio-app was not read); this receipt grades only what this repo's `conformance.rs` records.
- Blogs dated "2026" in the Apple lane are partly synthetic content farms; every number used was cross-checked against the arXiv study or Ollama's announcement, but the exact tok/s figures should be treated as C-grade.

## Receipt

```
Claim: Tier 1 = llamacpp (default), external-api, litertlm (mobile). Tier 2 = mlxcel (Mac fast-path candidate), mistralrs (forward metal/cuda or it stays CPU-only). Tier 3 = mlx parked (un-park rule stated), onnx removed, candle removed. Move llama-cpp-2 from git pin to =0.1.156 so the published crate has zero git deps.
- Question: which of 8 backends to bless / label experimental / drop — deep (product-defining, reversible).
- Sources: crates.io + GitHub API data pulled 2026-09-03 (A); gen2 conformance.rs/Cargo.toml/ci.yml and pio-app Cargo files (A, user-context); arXiv 2511.05502 (A); LiteRT-LM releases v0.13-v0.16.1 (A); llama-cpp-rs README + #970 (A/C); mlx-rs #308 (B); mistral.rs releases + #807/#903/#935 (B/C); ort release notes rc.10-13 (A); Ollama MLX switch via yage.ai (C); iPhone runtime bench john-rocky (C, harness on GitHub); Pi 5 tok/s blogs (C); mlxcel coverage backend.ai/Kubesimplify (C).
- Coverage: primary · practitioner · deployment · failure · user-context all hit; freshness swept; snowball + lateral read done; premortem run (§6).
- Recommendation: the tier list above. Default I'd take: all of it. Your call: LiteRT-LM Tier 1 vs 2; parking backend-mlx despite its sunk investment.
- References to mirror: Ollama format routing + "preview" labelling; ort's alternative-backends API for a companion crate; gen2's own contract_with_a_model as the promotion gate. Rubric: §4.
- What's NOT proven: RTX 3080 mistral.rs vs llama.cpp; mlxcel perf in gen2; llama-cpp-2 0.1.156 build with mtmd+llguidance; rMLX; pio-app's separate mlxcel verification.
- Re-check trigger: §9 — mlx-rs #308 closure, mlxcel on crates.io, mistralrs 0.9 on crates.io, LiteRT-LM ≥v0.17, pio-app mobile ship, or 2026-12-01.
```

## Sources

- crates.io API: llama-cpp-2 https://crates.io/crates/llama-cpp-2 · mistralrs https://crates.io/crates/mistralrs · mlx-rs https://crates.io/crates/mlx-rs · ort https://crates.io/crates/ort · candle-core https://crates.io/crates/candle-core
- utilityai/llama-cpp-rs https://github.com/utilityai/llama-cpp-rs · Windows cmake failure https://github.com/utilityai/llama-cpp-rs/issues/970
- EricLBuehler/mistral.rs releases https://github.com/EricLBuehler/mistral.rs/releases · Metal I64 https://github.com/EricLBuehler/mistral.rs/issues/807 · Metal perf tracking https://github.com/EricLBuehler/mistral.rs/issues/903 · vision on Metal https://github.com/EricLBuehler/mistral.rs/issues/935 · CUDA vs llama.cpp discussion https://github.com/EricLBuehler/mistral.rs/discussions/612
- oxiglade/mlx-rs https://github.com/oxiglade/mlx-rs · macOS 26 Metal issue https://github.com/oxiglade/mlx-rs/issues/308
- huggingface/candle https://github.com/huggingface/candle (models dir listing via API)
- pykeio/ort releases https://github.com/pykeio/ort/releases · yanked non-rc discussion https://github.com/pykeio/ort/discussions/501
- google-ai-edge/LiteRT-LM releases https://github.com/google-ai-edge/LiteRT-LM/releases · v0.16.0 https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.16.0 · Google blog https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/
- lablup/mlxcel https://github.com/lablup/mlxcel · saberra-ai/mlxcel https://github.com/saberra-ai/mlxcel · Lablup announcement https://www.backend.ai/blog/2026-05-lablup-opensourced-mlxcel · Kubesimplify test https://blog.kubesimplify.com/mlxcel-rust-native-inference-engine-tested-on-m1-max
- arXiv 2511.05502 https://arxiv.org/abs/2511.05502
- Apple Silicon MLX vs llama.cpp + Ollama switch https://yage.ai/share/mlx-apple-silicon-en-20260331.html
- iPhone runtime benchmark https://dev.to/john-rocky/on-device-llm-on-iphone-which-runtime-is-fastest-mlx-vs-llamacpp-vs-litert-lm-vs-coreml-1b42 · harness https://github.com/john-rocky/apple-silicon-llm-bench
- Pi 5 https://tinyweights.dev/posts/run-llms-raspberry-pi-5/ · https://www.kunalganglani.com/blog/gemma-3-raspberry-pi-5-benchmark
- Tauri + local LLM samples https://github.com/danielbank/tauri-mistral-chat · https://github.com/dillondesilva/tauri-local-lm
- rMLX (lead only) https://github.com/Pushkinist/rMLX
