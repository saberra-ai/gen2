# 03 — The five-minute first run

Status: RECEIPT (research-decision, 2026-09-03). No source files modified.
Fork owner: roadmap slice "cargo add gen2 → a generated token in five minutes".

## FORK

What does a best-in-class five-minute first run look like for a
native-dependency Rust inference crate, and which pieces should gen2 adopt in
wave 1?

Stakes: this is the front door for the whole "go-to local inference" goal. A
first run that ends in a build error about the machine, or in a hunt for a
GGUF, loses the reader before any of the API's strengths are visible. Depth
matched: five sub-questions, five lanes, two live measurements.

Sub-questions:
1. Model acquisition — accept a Hugging Face reference in `Engine::load`? Which
   default model?
2. First build cost — prebuilt, sccache, pure-Rust default, or source build
   with perfect errors?
3. Hardware auto-detection — should `metal` / `cuda` be on without a feature?
4. The README top block — who does it best?
5. A CLI or example as the smoke test?

Constraints carried in from the roadmap: DX/correctness > maintainability >
publishability > performance; audience is pio-app first, Rust developers
second; crate name / positioning / publishing are human-only; "no step outside
the README".

## LANES and what each turned up

### User-context lane (this repo, pio-app) — grade A (read directly)

- `README.md` top block today: a git dependency, "the first build compiles a
  C++ toolchain", "On Apple silicon add `metal`", every example opens
  `/models/model.gguf` — a path the reader has to produce themselves.
- `examples/minimal.rs:6` — `std::env::args().nth(1).unwrap()`: run it without
  a path and it panics. The README's `cargo run --example minimal --features
  metal -- /path/model.gguf` is the only smoke test offered.
- `src/api/engine.rs:68` — `Engine::load(path: impl AsRef<Path>)` is
  `Engine::builder().model(path).build()`; `EngineBuilder::model` (line 621)
  stores a `PathBuf`. Nothing resolves anything that is not a local path.
- `Cargo.toml` — `llama-cpp-2` pinned to `utilityai/llama-cpp-rs@43733d1`
  (= crate version 0.1.151, llama.cpp vendored at ggml-org commit `9e3b928fd`,
  2026-06-07). `reqwest` 0.13 is already a dependency (optional, behind
  `backend-external-api`); `dirs` 6 and `tokio` are unconditional. No `hf-hub`.
- `src/zoo.rs` + `resources/models/zoo.json` — a model zoo keyed by canonical
  id → per-platform `{backend, source: "<hf repo>", file}` already exists
  (Gemma 4, Llama 3.2 1B, …) with thinking-toggle notes per family
  (zoo.rs:443–842). It has no downloader; it only names what to fetch.
- `tests/live_inference.rs:14` — the live suite's example model is
  `SmolLM2-360M-Instruct-Q4_K_M.gguf`; docs/EXTRACTION.md:18 confirms it was the
  model used to prove the llama.cpp path under `metal`.
- `.github/workflows/ci.yml:26,82` — CI installs `cmake libclang-dev` on Ubuntu
  and `brew install cmake` on macOS before building the default features.
  There is no Windows job.
- pio-app (the first audience): `pio-core/src/external/hugging_face.rs` already
  downloads GGUF/MLX/ONNX through `hf-hub = "0.5"` (`api::tokio::Api`, its
  `Progress` trait, `hf_hub::Cache::default().token()`), and notes that
  hf-hub 0.5 pins reqwest 0.12 while pio-core is on 0.13 so the two clients
  cannot share a type. `pio-core/src/model_tiers.rs:37` records the product
  decision "the Smart list starts at Qwen3-1.7B; nothing smaller".

### Primary lane — grade A unless noted

- **llama-cpp-rs at the pinned rev** (read from
  `~/.cargo/git/checkouts/llama-cpp-rs-274405c613038803/43733d1`):
  - `llama-cpp-2/Cargo.toml:58` has
    `[target.'cfg(all(target_os = "macos", any(target_arch = "aarch64", target_arch = "arm64")))'.dependencies] llama-cpp-sys-2 = { …, features = ["metal"] }`
    — Metal is turned on for Apple Silicon by the binding itself, with no
    feature from the consumer. `cargo tree -e features -i llama-cpp-sys-2`
    on gen2's *default* features shows `llama-cpp-sys-2 feature "metal"`
    enabled. The same table is present on upstream `main` (0.1.157).
  - `llama-cpp-sys-2/Cargo.toml:12` — `metal = []`, and `build.rs` has no
    `cfg!(feature = "metal")` branch at all (the only Metal lines are 714–718,
    which force it OFF on watchOS, and 1261–1264, which link the Metal
    frameworks on every Apple target). GGML's own CMake decides:
    `ggml/CMakeLists.txt` defaults `GGML_METAL` ON on Apple and
    `GGML_METAL_EMBED_LIBRARY` to `${GGML_METAL}` (line 242). The fresh build
    below confirms `GGML_METAL:BOOL=ON`, `GGML_METAL_EMBED_LIBRARY:BOOL=ON` in
    the CMake cache. Embedded library means the shaders are compiled from
    source at run time by the Metal framework — `xcrun metal` is never invoked
    (`ggml-metal/CMakeLists.txt:27–110`), so the macOS 26 "Metal Toolchain"
    component is *not* needed for llama.cpp (it is for `backend-mlx`, as the
    README already says).
  - **No prebuilt path.** `dynamic-link` only sets `build_shared_libs`;
    `system-ggml` finds a system GGML via `find_package` but llama itself is
    still compiled from the vendored sources. There is no
    `LLAMA_LIB_PATH`-style override. `bindgen` runs unconditionally, so
    libclang is a hard build requirement on every platform.
  - `build.rs:621` forwards every `CMAKE_*` environment variable into the
    CMake configure step, so `CMAKE_C_COMPILER_LAUNCHER=sccache
    CMAKE_CXX_COMPILER_LAUNCHER=sccache` reaches the llama.cpp build with no
    change to gen2 (sccache's documented CMake integration is exactly those
    two variables — mozilla/sccache README).
- **hf-hub** (crates.io API + docs.rs + repo README):
  - 1.0.0 released 2026-07-10 (rc.0 2026-04-30, rc.1 2026-05-07); repo
    `Cargo.toml` is already at 1.1.0. 0.5.0 (2026-02-19) is the last of the
    old `Api`/`ApiBuilder` line that pio-app uses.
  - 1.x API: `HFClient::builder().build()?.model("owner","name").download_file().filename("f.gguf").send().await? -> PathBuf`;
    `HFClientSync` behind the `blocking` feature (owns its own tokio runtime,
    safe to call inside another runtime). Download builder options:
    `revision`, `local_dir`, `force_download`, `local_files_only`, `progress`
    (a callback). Retries with exponential backoff (`retry_max_attempts`
    default 3), and an `HFError::RateLimited` variant.
  - Deps: reqwest 0.13 (matches gen2), tokio, hf-xet 1.6 (pure Rust, no
    build.rs), no `dirs`, no indicatif. Features: `blocking`, `rustls-tls`,
    `socks`. TLS is whatever reqwest's default is unless `rustls-tls` is set.
  - **Contradiction to verify at implementation (grade B):** the README says
    "hf-hub does not read environment variables" and that `HFClient::new()`
    caches under `.cache/huggingface/hub` *relative to the working
    directory*, while the docs.rs `HFClientBuilder` docs say `cache_dir`
    defaults to `HF_HUB_CACHE → $HF_HOME/hub → ~/.cache/huggingface/hub`,
    `token` falls back to `HF_TOKEN` and the `huggingface-cli login` token
    file, and `endpoint` honours `HF_ENDPOINT`. Either way gen2 should set
    `cache_dir` and `token` explicitly rather than trust the default.
- **Hugging Face Hub, live** (curl from this machine, 2026-09-03):
  - `GET https://huggingface.co/v2/unsloth/Qwen3-0.6B-GGUF/manifests/Q4_K_M`
    with `User-Agent: llama-cpp`, anonymous → JSON with
    `ggufFile.rfilename = "Qwen3-0.6B-Q4_K_M.gguf"`, `size = 396705472`, and
    the blob sha256. `manifests/latest` resolves to the same file. This is the
    endpoint llama.cpp's `-hf repo[:quant]` uses (ggml-org/llama.cpp PR
    #11195: no tag → try `Q4_K_M`, then any `Q4*`, then the first GGUF).
  - `HEAD …/resolve/main/Qwen3-0.6B-Q4_K_M.gguf` → 302 to the CDN,
    `accept-ranges: bytes`, `x-linked-size: 396705472`, and the rate-limit
    headers `ratelimit: "resolvers";r=2999;t=141` — anonymous quota is 3,000
    resolves per 5-minute window (HF rate-limits doc: anon 3,000 / free user
    5,000 / PRO 12,000 on the resolver bucket; 429 carries `RateLimit` with
    seconds-to-reset). One model download costs two or three resolver calls.
  - A 50 MB ranged GET ran at 17.2 MB/s here.
- **Cargo** (reference, "Specifying Dependencies"): a `[target.'cfg(…)']`
  table may name a dependency that is also in `[dependencies]` and add
  features to it — llama-cpp-2 itself relies on this — but
  `cfg(feature = …)` is not allowed in a target expression, and features
  themselves cannot be made platform-conditional. So "auto-metal" is only
  achievable by a cfg-gated dependency table (already done upstream), never by
  a default feature.
- **bindgen requirements page:** libclang ≥ 9; Debian/Ubuntu `apt install
  libclang-dev`; macOS via Xcode CLT or `brew install llvm`; Windows `winget
  install LLVM.LLVM` **and** `LIBCLANG_PATH=<LLVM>\bin`.
- **Model cards / repos (all public, none gated, all Apache-2.0):**
  | Repo | Q4_K_M | UD-Q4_K_XL | Q8_0 | Notes |
  | --- | --- | --- | --- | --- |
  | `unsloth/Qwen3-0.6B-GGUF` | **397 MB** | 405 MB | 639 MB | Qwen3 chat template with tools; thinking on by default (`/no_think` or `enable_thinking=false`); card: "do not use greedy decoding" in thinking mode; card's own run line is `llama serve -hf unsloth/Qwen3-0.6B-GGUF:UD-Q4_K_XL`. |
  | `unsloth/Qwen3-1.7B-GGUF` | 1.11 GB | 1.13 GB | 1.83 GB | pio-app's "smallest Smart" pick. |
  | `unsloth/Qwen3.5-0.8B-GGUF` | 533 MB | 559 MB | 812 MB | Feb 2026; vision (mmproj 205 MB); 262K ctx; thinking *off* by default at 0.8B–9B; card warns 0.8B "is prone to entering thinking loops"; template needed a 21-fix revision (QwenLM/Qwen3 #1831); llama.cpp's own README quickstart is `llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF`. |
  | `unsloth/Qwen3.5-2B-GGUF` | 1.28 GB | 1.34 GB | 2.01 GB | |
  | `unsloth/gemma-4-E2B-it-GGUF` | 3.11 GB | 3.18 GB | — | Apache-2.0 since Gemma 4; ollama's `gemma4:e2b` is 7.5 GB, default `gemma4` = e4b 9.6 GB. |
  | `unsloth/SmolLM3-3B-GGUF` | 1.92 GB | — | — | tools only via `xml_tools` template kwarg. |
  | Llama 3.2 1B (lmstudio-community mirror) | ~0.8 GB | — | — | Llama licence, Meta repos gated. |

### Practitioner lane — grade B (READMEs / docs read via fetch)

- **ollama** README quickstart is one line: `ollama run gemma4`. Model
  implicit, download implicit, hardware implicit. The bar.
- **mistral.rs**: `mistralrs run -m Qwen/Qwen3-4B` (CLI, prebuilt binaries via
  an install script — "standard acceleration needs no Rust or CUDA
  toolkit"); crate: `cargo add mistralrs` then
  `ModelBuilder::new("Qwen/Qwen3-4B").with_auto_isq(IsqBits::Four).build().await?`
  — HF fetch automatic; `metal`/`cuda` are explicit features; "builds pure
  Rust, no C compiler by default".
- **kalosm**: `cargo add kalosm --git … --features full`, then
  `Llama::new().await?` — zero-argument default. The default is
  `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/…Q4_K_M.gguf` (~4.9 GB),
  cached under `DATA_DIR/kalosm/cache`; named constructors include
  `qwen_3_0_6b_instruct` = `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf`
  and `qwen_3_1_7b_instruct`. `Llama::new()` is the DX gold standard; its
  default *size* is the anti-pattern.
- **llama-cpp-2** README: `git clone --recursive`, then `cargo run --release
  --bin simple -- --prompt "…" hf-model TheBloke/Llama-2-7B-GGUF
  llama-2-7b.Q4_K_M.gguf`. No `cargo add` path shown, 7B model, 2023 repo.
  Precedent for "repo + file" as two arguments.
- **ort**: `cargo add ort`; the default `download-binaries` feature fetches
  prebuilt ONNX Runtime at build time; `load-dynamic` + `ORT_DYLIB_PATH` and
  `ORT_LIB_LOCATION` as escape hatches. Zero toolchain, but pyke maintains a
  binary matrix — a cost gen2 cannot pay in wave 1.
- **candle**: `cargo add candle-core`; the LLM path is
  `cargo run --example quantized --release` in the candle repo, not a
  library quickstart.
- **llama.cpp**: `llama-cli -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0`;
  `hf.co/user/repo:QUANT` in ollama (default tag Q4_K_M, case-insensitive,
  or a filename after the colon). The `repo:quant` string is the convention
  users already type.
- r/LocalLLaMA / 2026 round-ups (grade C, aggregators): CPU-only starter
  picks are Qwen3.5 4B, Gemma 4 E4B, Phi-4-mini, SmolLM3 3B, Llama 3.2 3B;
  "for ≤2 GB RAM Llama 3.2 1B (~1.3 GB)". Nobody recommends a sub-1B model
  for *use*; everybody uses one for *smoke*.

### Deployment lane — grade A (measured here) / B (CI)

- **Cold build, this machine** (Apple M4 Pro, 14 cores, 64 GB, cmake 4.3.1,
  empty `CARGO_TARGET_DIR`, git checkouts already fetched):
  - `cargo build --example minimal --features metal` (default backend =
    llama.cpp + Metal): **53.5 s wall**, 302 s CPU, 296 crates, 2.0 GB
    target dir, llama.cpp out dir 137 MB.
  - `cargo build --no-default-features --features backend-mistralrs` (the
    pure-Rust alternative): **2 min 06 s wall**, 335 s CPU, 530 crates,
    4.8 GB target dir.
  - So the pure-Rust route is 2.4× slower to first build and 2.4× the disk,
    at the same CPU cost; what it buys is not needing cmake + libclang.
- **CI** (saberra-ai/gen2 run 33582977916, last green): ubuntu
  `backend-llamacpp` job 6 min 47 s end to end with rust-cache restored
  (clippy build 3 min 40 s, test build a further 2 min 30 s); macOS `metal`
  job 2 min 02 s total. Cold-cache Ubuntu will be longer; treat "3–5 minutes
  on a 4-core Linux box" as the planning number.
- **Download**: 397 MB at the measured 17 MB/s is 23 s; at a 50 Mbit/s home
  link (~6 MB/s) it is ~65 s; at 20 Mbit/s ~2.6 min. A 1.1 GB model at
  50 Mbit/s is ~3 min on its own; a 3.1 GB Gemma 4 E2B is 8+ min and blows
  the budget by itself.
- **Budget arithmetic** (laptop, decent link): build ~1–2 min + download
  ~0.5–1 min + model load and first token (< 5 s for 0.6B) fits in five
  minutes with margin **only** if the model is well under 1 GB.

### Failure lane — grade A/B

- **Windows**: llama-cpp-sys-2 0.1.140 failed to build (cmake exit 1,
  utilityai/llama-cpp-rs #970; workaround = pin 0.1.139 — gen2 pins a rev, so
  this class of regression is opt-in). MSVC mmap bug for models > 4 GB
  (#951) — a sub-1 GB default never hits it. bindgen needs LLVM +
  `LIBCLANG_PATH` (bindgen requirements; rust-bindgen #1276/#2360 are all
  "Unable to find libclang"). No Windows CI today, so nothing here is
  proven.
- **macOS 26**: llama.cpp is fine (embedded Metal library, verified above).
  The "xcrun: unable to find utility metal" reports (ggml-org #6946,
  LocalAI #2031) come from builds with `GGML_METAL_EMBED_LIBRARY=OFF`; gen2
  never sets that. `backend-mlx` still needs the Metal Toolchain component,
  and that must stay a documented exception, not a default.
- **Hugging Face**: anonymous 429s are a real failure mode (HF docs: "the
  number one reason users get rate limited" is no `HF_TOKEN`); a first-run
  download uses two or three resolver calls against a 3,000/5-min anonymous
  quota, so the risk is CI fan-out, not a developer's laptop. 403 for gated
  repos — none of the shortlisted defaults are gated. hf-hub 1.x parses
  `RateLimit` into `HFError::RateLimited`; gen2 must surface it as its own
  error text ("Hugging Face rate-limited this IP; set HF_TOKEN or retry in
  N s") rather than a generic HTTP error.
- **Offline / air-gapped**: the reference must degrade to "already in cache"
  (`local_files_only`) and the README must show the local-path form next to
  the `hf:` form.
- **Qwen3 thinking**: a 0.6B model with thinking on will spend the first
  hundred tokens inside `<think>`; the README already warns that the block
  lands in `latest_text()`. The first-run example must disable thinking
  (soft `/no_think` in the prompt is universal; `enable_thinking=false` is a
  template kwarg gen2 does not expose yet — `zoo.rs:701,842` already reasons
  about this per family). The model card also says thinking-mode greedy
  decoding degrades to repetition; the README's `.greedy()` examples are
  extraction/classification with grammars, where that is fine.
- **Qwen3.5 template**: 21 template fixes landed after release and llama.cpp
  #20837/#20409 track tool-call-in-thinking and `enable_thinking` bugs. gen2
  renders templates with its own minijinja, so a Qwen3.5 default would need
  the conformance suite extended before it can be trusted. Not wave 1.
- **hf-hub 1.x env-var contradiction** (README vs docs.rs) — see above.

## GRADE (source → grade → why)

| Source | Grade | Why |
| --- | --- | --- |
| Local checkout of llama-cpp-rs@43733d1, CMakeCache of a fresh build, `cargo tree` | A | primary, executed |
| Live HF manifest/resolve/rate-limit probes; measured build and download times | A | executed here today |
| gen2 sources, pio-app sources | A | read directly |
| ggml `CMakeLists.txt` (master), llama-cpp-rs `main` Cargo.toml/build.rs | A− | primary, fetched; matches the pinned rev |
| hf-hub crates.io API, docs.rs builder docs, repo Cargo.toml | A− | primary; README/docs disagree on env vars |
| llama.cpp PR #11195, HF hub docs (rate limits, gguf-llamacpp) | A− | primary |
| HF model repos (sizes, licence, gating) | A− | primary, fetched tree pages |
| ollama / mistral.rs / kalosm / ort / candle READMEs and docs.rs | B | practitioner, summarised by fetch |
| llama-cpp-rs issues #970 #951, bindgen issues, ggml #6946 | B | primary issues, single reports |
| gen2 CI job timings | B | warm cache, not cold |
| 2026 "best small model" round-ups, BFCL numbers for Qwen3-1.7B (7.8% multi-turn) | C | aggregators / one paper |

## RUBRIC

Criteria in priority order (from the intent contract): (1) first run works
on macOS + Linux with no step outside the README; (2) fits five minutes;
(3) correctness of the first output (no `<think>` soup, no grammar trap);
(4) maintainability (no new pipeline to babysit); (5) publishability (no git
deps, crates.io-clean); (6) performance.

### Q1 — model acquisition

| Option | (1) | (2) | (3) | (4) | (5) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Keep "bring your own GGUF" | ✗ step outside README | — | — | ✓ | ✓ | reject |
| `hf:` reference resolved inside `Engine::load`, via hf-hub 1.x, default-on feature | ✓ | ✓ | ✓ | ✓ small (one crate, pure Rust) | ✓ | **adopt** |
| Same, in-house on reqwest (manifest + resolve endpoints) | ✓ | ✓ | ✓ | ~150 LOC to own; no xet, no retry policy | ✓ | fallback if hf-hub misbehaves |
| Zoo id (`gen2::zoo::default()`) as the public front door | ✓ | ✓ | ✓ | zoo JSON becomes API | ✓ | keep zoo internal; it can *back* the `hf:` string later |

### Q2 — first build cost

| Option | Zero-toolchain? | Cold build here | Maintenance | Verdict |
| --- | --- | --- | --- | --- |
| Source build (today) + exact prerequisites + sccache hint | no (cmake + libclang + C++) | 53 s (M4 Pro), ~3–5 min Linux | none | **adopt** |
| Prebuilt libllama (ort-style download-binaries) | yes | seconds | a binary matrix per target × accelerator × llama.cpp bump, with no upstream support in llama-cpp-sys-2 | reject for wave 1 |
| Pure-Rust default (mistral.rs) | yes | 2 min 06 s, 4.8 GB | drags a second backend into the default path; mistral.rs is "claims only formats no other backend takes" today | reject as default; keep as opt-in |
| Pure-Rust default (candle) | yes | unmeasured | "compiles, not proven to generate" in this repo | reject |

### Q3 — hardware auto-detection

| Option | Works? | Verdict |
| --- | --- | --- |
| Cargo default feature that is platform-aware | impossible (features are not target-conditional) | — |
| cfg-gated dependency table adding `metal` on macOS | already done upstream in llama-cpp-2 0.1.151 (and ggml defaults Metal ON on Apple) | **nothing to build; fix the README** |
| `cuda` auto | needs the CUDA toolkit, cannot be detected at dependency-resolution time; every peer keeps it explicit | keep explicit |

### Q4 — README top block

Best-in-class = ollama's one line, transposed to a library: one dependency
line, one prerequisite line per OS, one `main` that names a model string and
prints a token, one `cargo run --example` that proves it. mistral.rs's
`ModelBuilder::new("Qwen/Qwen3-4B")` is the closest crate-shaped precedent;
kalosm's `Llama::new()` shows zero-argument is possible but its 4.9 GB
default shows why the size must be chosen deliberately.

### Q5 — smoke test

| Option | Cost | Verdict |
| --- | --- | --- |
| `cargo run --example hello` (no arguments, default `hf:` model, prints tokens + timing) | one file; already doc-tested build | **adopt** |
| `cargo install gen2-cli` | a second crate, a second publish gate, a CLI that competes with ollama/mistralrs — a stated non-goal | defer |

## PREMORTEM — how this call fails, and the guard for each

1. **hf-hub 1.x caches relative to CWD or ignores `HF_TOKEN`** (its README
   says so; docs.rs says otherwise). Guard: gen2 always passes an explicit
   `cache_dir` (HF hub layout so Python/llama.cpp users get cache hits) and
   an explicit token read from `HF_TOKEN`; a unit test asserts the resolved
   cache path is absolute and stable across CWDs.
2. **The default model is fine as a smoke test and useless for tool calling,
   so the second thing a reader tries fails.** Guard: the README names the
   default as a smoke model in the sentence that introduces it, and the
   `agent` section's first example uses `hf:unsloth/Qwen3-1.7B-GGUF` with a
   one-line reason. pio-app already draws this line at 1.7B.
3. **Thinking output on first run** ("why is the model talking to itself?").
   Guard: `examples/hello.rs` and the README `infer` example put `/no_think`
   in the prompt for Qwen3; a follow-up slice exposes `enable_thinking` as a
   template kwarg so the soft switch is not the only lever.
4. **Windows readers hit bindgen/libclang and there is no CI to have caught
   it.** Guard: the README's prerequisite table carries the Windows line
   (`winget install LLVM.LLVM Kitware.CMake` + `LIBCLANG_PATH`), and a
   `windows-latest` compile-only job is added when the Windows CI slice
   lands. Until then the README says "Windows: compile-checked only".
5. **A future llama-cpp-rs bump drops the cfg-gated `metal` table** and Apple
   builds silently go CPU-only. Guard: a CI assertion on macOS that
   `cargo tree -e features -i llama-cpp-sys-2` contains `feature "metal"`
   with default features; keep gen2's `metal` feature as an explicit alias
   so `--features metal` keeps working.
6. **HF rate-limits CI fan-out** (many jobs, one IP). Guard: CI sets
   `HF_TOKEN` from a secret and caches the HF hub dir; the error message on
   429 tells a human what to do.
7. **Bigger default creeps in** ("0.6B is embarrassing, make it 4B"). Guard:
   the receipt records the arithmetic: > 1 GB at 50 Mbit/s already costs
   3 minutes; the five-minute promise is the constraint, not taste.
8. **The `hf:` string grammar becomes ambiguous** (`repo:Q4_K_M` vs
   `repo:file.gguf` vs `repo:revision`). Guard: adopt llama.cpp/ollama's
   grammar exactly — after the colon is either a quant tag or a filename
   ending in `.gguf`; no revision syntax in wave 1; a typed `HfModel` struct
   is the escape hatch for anything else.
9. **hf-hub adds hf-xet + reqwest + tokio to every default build and cold
   build time slips.** Guard: measure the delta when the slice lands; if it
   is > 15 s cold, move to the in-house reqwest resolver (the endpoints are
   verified above) and keep hf-hub out.

## RECEIPT — the CALL

### README top block for gen2 (proposed; wording is the owner's to keep or change)

````markdown
```toml
[dependencies]
gen2 = "0.1"
```

```rust,no_run
use gen2::Engine;

fn main() -> Result<(), gen2::Error> {
    // Downloaded once into the Hugging Face cache, then loaded from there.
    let engine = Engine::load("hf:unsloth/Qwen3-0.6B-GGUF")?;
    let text = engine.infer("/no_think In one sentence, what is Rust?").max_tokens(48).text()?;
    println!("{text}");
    Ok(())
}
```

That is the whole first run. The default backend is llama.cpp, compiled from
source on the first build, so the machine needs a C++ toolchain, CMake and
libclang:

| | |
| --- | --- |
| macOS | `xcode-select --install && brew install cmake` — Metal is on; nothing to add |
| Debian/Ubuntu | `sudo apt install build-essential cmake libclang-dev` |
| Windows | `winget install Kitware.CMake LLVM.LLVM`, set `LIBCLANG_PATH` to LLVM's `bin`, use the MSVC toolchain |
| NVIDIA | add the `cuda` feature (needs the CUDA toolkit) |

The first build takes about a minute on an M-series laptop and a few on a
four-core Linux box; `CMAKE_C_COMPILER_LAUNCHER=sccache
CMAKE_CXX_COMPILER_LAUNCHER=sccache` makes every build after the first one
fast. The model is 397 MB and Apache-2.0. It is a smoke test, not a workhorse:
for tools and agents start at `hf:unsloth/Qwen3-1.7B-GGUF`, and any local
`.gguf` path works in the same place.

```sh
cargo run --example hello          # same program, prints tokens and the timings
```
````

### Default model

`hf:unsloth/Qwen3-0.6B-GGUF` → resolves to `Qwen3-0.6B-Q4_K_M.gguf`
(397 MB, Apache-2.0, not gated, Qwen3 tool template, already the family the
crate's `tojson` template fix was written for, kalosm's `qwen_3_0_6b_instruct`,
ollama's `qwen3:0.6b`). The README states in the same breath that
`Qwen3-1.7B` (1.11 GB) is the first size worth building on — the same line
pio-app draws. Qwen3.5-0.8B is the re-check candidate (llama.cpp's own
quickstart model, thinking off by default, vision) once gen2's template
renderer is proven against its template in the conformance suite.

### Acquisition API

- `Engine::load(s)` keeps `impl AsRef<Path>` (an `hf:` string is a valid
  path on every OS). `EngineBuilder::build` (and `load_model`) resolve a
  path whose string form starts with `hf:` (accept `hf://` too):
  `hf:<owner>/<repo>[:<QUANT>|:<file>.gguf]`, quant default `Q4_K_M` with
  llama.cpp's fallback order (any `Q4*`, then the first GGUF), resolved via
  `v2/<repo>/manifests/<tag>` with `User-Agent: gen2/<ver> llama-cpp`. Same
  grammar as `llama-cli -hf` and `ollama run hf.co/…`.
- Typed form for programs: `HfModel::new("owner/repo").quant("Q8_0")` /
  `.file("…gguf")` /  `.mmproj(…)`, accepted by `EngineBuilder::model_hf`.
  `EngineBuilder::on_download(|progress| …)` for UIs (pio-app needs this;
  hf-hub's download builder has the `progress` hook).
- Cache: the Hugging Face hub layout, `GEN2_MODEL_CACHE` → `HF_HUB_CACHE` →
  `$HF_HOME/hub` → `~/.cache/huggingface/hub`, set explicitly on the client.
  `HF_TOKEN` honoured; offline resolves from cache only and fails with a
  message that names the file it wanted.
- Errors: a dedicated `Error::Download { reference, cause }` with
  rate-limit and gated-repo text that says what to do.
- Dependency: `hf-hub = { version = "1", features = ["blocking"] }` behind a
  default-on `hf` feature (so `default-features = false` consumers such as
  the external-api build do not pay for it). If its cold-build delta is
  > 15 s or its env handling is wrong, replace with the in-house reqwest
  resolver (endpoints verified live above) — reqwest 0.13 is already in the
  tree.

### Build-cost strategy

Keep the source build. Do not build a prebuilt-binary pipeline in wave 1.
Concretely: (a) the prerequisite table above, per OS, in the README's first
screen; (b) the sccache hint (free: `build.rs` forwards `CMAKE_*`);
(c) drop "On Apple silicon add `metal`" — it is already on via
llama-cpp-2's cfg-gated table and ggml's default, and a macOS CI assertion
keeps it that way; keep the `metal` feature as an alias; (d) `cuda` stays an
explicit feature; (e) a `windows-latest` compile-only job in the CI slice.
Pure-Rust (mistral.rs) stays opt-in: 2.4× the cold build and disk for the
privilege of skipping cmake.

### Smoke test

`examples/hello.rs`: no arguments, `Engine::load("hf:unsloth/Qwen3-0.6B-GGUF")`,
`/no_think` prompt, streams tokens, prints download/load/first-token
timings and the resolved cache path. Fix `examples/minimal.rs:6` to fall back
to the same default instead of `unwrap()`. `cargo install gen2-cli` is
deferred (second crate, competes with ollama/mistralrs, non-goal).

### References to mirror

- `utilityai/llama-cpp-rs:llama-cpp-2/Cargo.toml:58` — the cfg-gated Metal
  table (the thing gen2 no longer needs to add).
- `ggml-org/llama.cpp:common/arg.cpp` + PR #11195 — `-hf repo[:tag]`
  grammar, `Q4_K_M` default, manifest endpoint.
- `huggingface/hf-hub:README.md` + docs.rs `HFClientBuilder` /
  `HFRepository::download_file` — the 1.x client shape and the `progress`
  hook.
- `floneum/floneum:models/kalosm-llama/src/source.rs` — named-constructor
  table (`qwen_3_0_6b_instruct` etc.) as the shape for a small curated list.
- `EricLBuehler/mistral.rs:mistralrs/src/lib.rs` crate docs —
  `ModelBuilder::new("Qwen/Qwen3-4B")` as the one-string precedent.
- `pykeio/ort` README / setup docs — the escape-hatch vocabulary
  (`load-dynamic`, `ORT_DYLIB_PATH`) to reuse *if* a prebuilt path is ever
  added for llama.cpp.
- `pio-app:pio-core/src/external/hugging_face.rs` — the consumer's existing
  downloader (hf-hub 0.5, progress, token from cache); gen2's API must let
  pio-app delete it.

### Miss-risk

Medium-low on the mechanism (every load-bearing claim was executed or read
from the pinned sources), medium on the model choice: Qwen3.5-0.8B may be
the better default within months, and BFCL-style tool-use numbers for
sub-1B models are thin (one paper, 1.7B at 7.8% multi-turn). The biggest
unverified piece is hf-hub 1.x's environment handling.

### Re-check trigger

Re-open this fork when any of: (a) llama-cpp-sys-2 gains a prebuilt /
`LLAMA_LIB_PATH` path or gen2 bumps its pin past 0.1.157; (b) gen2's
conformance suite passes Qwen3.5-0.8B's chat template (then the default
model flips); (c) hf-hub 1.x's env-var behaviour is confirmed wrong or its
cold-build delta exceeds 15 s (then the in-house resolver); (d) a Windows CI
job exists and fails on the README steps; (e) HF anonymous resolver quota
drops below ~500 per 5 min.
