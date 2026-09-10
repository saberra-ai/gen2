# gen2 → the go-to local inference library — roadmap

Status: ACTIVE (intent 2026-09-03; research + slices 2026-09-04)
Owner: Victor. Driven by an autonomous agent under the budget in "Autonomy budget".

## User intent

- Desired outcome: gen2 is the library a developer reaches for first when they
  want local inference. Wave 1 makes that *possible*: it is publishable to
  crates.io, `cargo add gen2` plus a `.gguf` works in five minutes, docs.rs is
  clean, there are honest benchmarks against llama.cpp on the same file, CI
  covers macOS/Linux/Windows, and pio-app consumes the crate instead of its
  in-tree copy.
- Audience: **Pio and Victor's own apps first** (pio-app, flock). The Rust
  ecosystem second. Other-language bindings are not in scope.
- Priority stack (user chose "publishable + best DX"; ordering below the first
  item is the agent's): DX and correctness > maintainability > publishability
  > measured performance > backend breadth > novelty.
- Taste / references: **`api_spec.md` (Victor, 2026-09-04, "Proposed") is the
  public-surface reference**: `Runtime`/`Model`/`Session`/`Turn`, semantic
  `EventStream`, first-order system prompt and tools, append-only history with an
  active projection, agent loops OUT of the core. The README's existing voice (plain
  sentences, every example doc-tested, costs stated honestly). Mirror: mistral.rs and
  llama-cpp-2 for the Rust surface, ollama's "one command and it talks" first
  run, `ort` for a native-dependency crate that still `cargo add`s cleanly.
- Anti-references: LangChain-style abstraction sprawl; a backend matrix where
  most cells are "compiles, unproven"; a first run that ends in a build error
  about the machine rather than about the model.
- Non-goals (wave 1): Python/Node/Swift bindings; an HTTP server mode; full
  feature parity with mistral.rs or ollama; adoption metrics as a target;
  audio/speech; NPU paths.
- Risk boundaries (⛔ human-only, user-stated): the crate name and public
  positioning (tagline, README voice changes); publishing to crates.io or
  tagging a release; spending money (paid CI runners, GPU hosts, domains).
- Explicitly **not** a human stop (user left it unselected): demoting or
  dropping a backend. The agent decides tiering from research.
- Good-enough bar: `cargo publish --dry-run` passes; a fresh clone follows the
  README to a generated token on macOS and Linux with no step outside the
  README; docs.rs builds warning-free; a benchmark table exists with the
  method stated; pio-app builds against the published-shape crate.
- Assumptions made: wave 1 is Rust-only; the 0.x version line is fine; the
  seven backends collapse to a tier list rather than all being proven;
  Windows CI is compile-and-unit-test only (no GPU); CUDA is compile-checked
  at most because GPU runners cost money.
- Human-only forks: ⛔ crate name "gen2" (free on crates.io as of 2026-09-04);
  ⛔ tagline/positioning; ⛔ publish/tag; ⛔ any paid runner; ⛔ whether the agent
  layer (`Agent`, `AgentRun`, approvals, tool execution, MCP lifecycle) is deleted
  or kept as an optional `agent` feature off the root (spec §26 allows either; the
  roadmap defaults to "optional feature, off the root" because pio-app still
  consumes it, and flags it).
- Research questions: see "Research calls".

## Research calls

Receipts (graded, cited): `docs/plans/research/0[1-5]-*.md`. Calls taken:

0. **Public API** (`api_spec.md`, user-authored, pulled 2026-09-04 after S1.2). The
   facade becomes `Runtime` → `Model` → `Turn` over a model-agnostic `Session`;
   `gen2::load(path)` is the one-liner; `Model::generate` replaces `Engine::infer`,
   `Model::turn(&mut session)` replaces `Engine::chat`; `GenerationOptions`,
   `Response`, semantic `Event` stream; remote OpenAI-compatible models via
   `runtime.openai()...connect()`; root namespace per §25 with `gen2::advanced` for
   local tuning, residency, hardware, grammar, and the backend plugin seam. The
   controller/backends stay below (§27). Wave 2 below implements it; the old Wave 2
   (first run) becomes Wave 3 and its README block uses the new API.

1. **Backend tiers** (01). Tier 1: `backend-llamacpp` (default), `backend-external-api`,
   `backend-litertlm` (mobile). Tier 2 experimental: `backend-mistralrs` (must forward
   `metal`/`cuda`; today silently CPU-only), MLX family. Tier 3: `backend-onnx` and
   `backend-candle` removed (never decoded a token; pio-core's `backend-onnx` is an
   empty stub). `backend-mlx` parked on the hybrid registry+git dependency.
   Re-check 2026-12-01 or mlx-rs #308 closing.
2. **Publish path** (02). `gen2` is free on crates.io. `llama-cpp-2 = "=0.1.151"` is the
   exact ancestor of the git pin and ships `mtmd`+`llguidance` (proven: dry-run exit 0
   on a scratch copy; 0.1.156 breaks `LlamaSampler::penalties`). `mlx-rs` becomes
   `{ version = "0.25.3", git, rev }`. mlxcel has no registry crate and cannot be an
   optional git dep of a published crate, so it moves to a workspace companion crate
   behind a public backend seam (git/path consumers only). docs.rs metadata:
   no-default-features + llamacpp/external-api/litertlm/tokio, x86_64 linux only.
   Superseded on the version by Victor's call to bump to the latest release (S1.1).
3. **Five-minute first run** (03). `Engine::load("hf:owner/repo[:QUANT|:file]")` via
   `hf-hub = "1"` behind a default-on `hf` feature; smoke model
   `hf:unsloth/Qwen3-0.6B-GGUF` (Q4_K_M, 397 MB, Apache-2.0); README says Qwen3-1.7B is
   the first size worth building on. Keep the source build (53 s cold on M4 Pro; no
   prebuilt path exists in llama-cpp-sys-2). Metal is ALREADY auto-enabled by
   llama-cpp-2 on macOS/aarch64: drop "add `metal`" from the README, keep the feature
   as an alias, assert it in CI. Add `examples/hello.rs` (zero-arg). No CLI crate.
4. **Benchmarks** (04). pp512/tg128 as llama-bench defines them, plus TTFT. Reference
   llama-bench built from the vendored llama.cpp sha and its `build_commit` asserted
   (today's parity bench compares against a brew build 553 commits behind: not
   publishable). 1 warmup, 5 reps, median±stddev from raw samples, greedy, batch=1,
   thermal bracket. Results JSON in `benches/results/<machine>/`, README table
   generated between markers, GPU-less `bench-freshness` CI job (drift, pin, >90 d).
   Remove the unsourced LiteRT-LM "1.7x" README sentence.
5. **pio-app switchover** (05). Zero merge risk: the in-tree copy is byte-identical to
   the split tip. 133 importing files; 40 paths now `pub(crate)`; `SystemTask` lost 8
   variants. Plan: gen2 `#[doc(hidden)] pub mod compat` → pio-app `gen2-crate` feature
   with path dep and a re-export shim → flip default → delete copy → git-tag dep.
   Three `RemoteDispatch` impls on the host side; gen2 needs no seam change.

## Done-when

- `cargo publish --dry-run` exits 0 on `main`, and CI's `publishable` job is a hard gate.
- The `api_spec.md` §28 walkthrough compiles and runs against the crate (doc tests
  plus live tests), and the root namespace matches §25.
- A fresh clone on macOS and Linux reaches a generated token by following only the
  README (`cargo run --example hello`), and a CI job proves it.
- `cargo doc` under the docs.rs metadata builds warning-free.
- The README benchmark table is generated from committed results JSON with the
  llama.cpp sha asserted, and a CI job fails when it rots.
- pio-app builds and passes its gate with the `gen2-crate` feature on by default.
- Publishing, tagging, and README positioning changes are handed to Victor (⛔).

## Environment ceiling

- Can verify here: default and Metal builds, unit/doc tests, live inference on real
  GGUFs (Qwen3-0.6B, Llama-3.2-3B, Gemma-4-E2B, an embedder, a reranker) on Apple
  Silicon under macOS 26; `cargo publish --dry-run`; GitHub CI (Linux + macOS runners,
  no GPU); pio-app builds and live tests on this Mac.
- Cannot verify here / honest ⬜: CUDA and Vulkan lanes (compile-check only); Windows
  beyond hosted-runner compile+unit tests; RTX 3080 benchmarks (SSH box exists,
  optional); iOS/Android on device; docs.rs itself (only its documented behaviour);
  mistral.rs GPU forwarding on NVIDIA.

## Autonomy budget + blast-radius stops

- Budget: this session, at most 20 slices, ≤8 commits per slice; one slice in flight
  at a time, raised to two on 2026-09-04 when Victor set the goal to `api_spec.md`
  (allowed only when the two touch disjoint files and the later one rebases); pre-flight disk (>40 GB free) before every build-heavy slice
  (`target/` was 79 GB on day one; `cargo clean` reclaimed 115 GB).
- Stop immediately for: `cargo publish` (not dry-run); creating a git tag; any paid
  runner/host/domain; changing the README's opening paragraph or tagline (positioning);
  pushing to pio-app `main` (PR only, that repo is shared and active); deleting data
  outside `target/`.
- Stop after: 3 same-cause gate failures on one slice; a second `--no-verify` for the
  same reason; a diff that no longer matches the slice.

## Integration policy

- gen2: direct commit to `main` and push after each slice's gate is green. Assumption
  (not user-stated): saberra-ai/gen2 is a solo, unprotected public repo whose 239
  commits were all landed this way; pushes are revertable. Victor can override.
- pio-app: worktree per slice, PR per slice, never push to `main`.

## Forks

- Engineering (resolved by the receipts above): tier list; dependency pins; hf-hub vs
  in-house resolver; benchmark protocol; switchover shim.
- Human ⛔: publish/tag (end of wave 1 and wave 3); README opening/tagline (S2.3 lands
  the structural README changes with the opening paragraph untouched and flags any
  positioning sentence for Victor); any spend.

## Slices

Status legend: ⬜ planned · ◐ in flight · ✅ done · ⛔ stopped · ↷ detour

### Wave 0 — green baseline
#### S0.1 Fix red `main`
Outcome: CI on `main` is green in every lane so later slices have a real gate.
Observable: `cargo clippy --all-targets -- -D warnings`, `cargo test`, and
`RUSTDOCFLAGS=-D warnings cargo doc --no-deps` all exit 0 locally; the next CI run is green.
Findings: `src/backend/llama/embedder.rs:596-609` has an orphaned doc block and a
duplicated `#[test]` (the Qwen3 integration test body was lost in 79da512); rustdoc has
two private intra-doc links to `session_rt::truncate` and a dead `crate::Capabilities`
link in `utilities/types.rs`; `utilities/acceptance_tests.rs:296` asserts a 600 ms
wall-clock bound that loaded runners trip (ONNX lane).
Reference to mirror: none needed.
Gate: the three commands above; CI green.
Blocked by: none. Forks: none. Honest ⬜: none.
Status: ✅ aefe0b8 (local gates green; CI run pending at push time)

### Wave 1 — publishable
#### S1.1 Dependencies off git
Outcome: every dependency the published crate needs resolves from crates.io.
Observable: `cargo publish --dry-run` fails only on `mlxcel` (or passes once S1.3 lands).
Work: `llama-cpp-2`/`llama-cpp-sys-2` → latest release (`=0.1.156`, 2026-09-02; the git
pin's llama.cpp is from 2026-06-07). Victor asked for the bump on 2026-09-04. Fix the
`LlamaSampler::penalties` signature change and anything else the build surfaces; fall
back to `=0.1.151` only if live inference regresses in a way the slice cannot fix.
`mlx-rs` → hybrid version+git.
Gate: `cargo test`; live_inference 22/22 under Metal; `cargo check --no-default-features
--features backend-external-api`. Blocked by: S0.1. Forks: none.
Honest ⬜: `backend-mlx` against upstream 0.25.3 not built (fork is what compiles).
Status: ✅ (see ledger)

#### S1.2 Backend tiering
Outcome: the crate carries only backends it stands behind, labelled by tier.
Observable: `backend-onnx` and `backend-candle` are gone from Cargo.toml, `src/backend`,
CI matrix, conformance and README; `backend-mistralrs` forwards `metal`/`cuda`; README
backend table is grouped Tier 1 / experimental / mobile.
Gate: `cargo test`; every remaining CI backend lane passes; conformance suite's
stale-list check passes. Blocked by: S1.1. Forks: none (user delegated tiering).
Honest ⬜: mistral.rs GPU forwarding unverified on NVIDIA.
Status: ✅ (see ledger)

#### S1.3 Public backend seam + `gen2-mlxcel` companion crate
Outcome: a consumer can register an out-of-tree backend, and mlxcel is one.
Observable: `crates/gen2-mlxcel` (workspace member, `publish = false`) builds with
`--features metal` on this Mac and a live test generates a token through
`Engine::builder().backend(...)`; the root crate has no `mlxcel` dependency.
Reference to mirror: `src/backend/traits.rs` (already `pub`), `mistralrs` device
plumbing for how a backend receives settings. Gate: root `cargo publish --dry-run`
exit 0; companion live test passes. Blocked by: S1.2. Forks: none.
Design (2026-09-04, from reading the facade): `Backend` is `!Send` by design, so a
consumer registers a factory, not an instance. `BackendPlugin { name, claims:
fn(&Path) -> bool, make: Box<dyn Fn() -> Box<dyn LocalBackend> + Send + Sync> }`
travels in `ControllerConfig.plugins`; `facade::detect_backend` asks plugins first
and the facade `Engine` gains a `Plugin(Box<dyn LocalBackend>)` variant. Public
surface for implementers: a curated `gen2::backend` (or `gen2::plugin`) module
re-exporting `Backend`, `LocalBackend`, `BackendSession`, `TokenPullerDyn`,
`HfTokenizer`, `ChatTemplate`, `load_chat_template`, `GrammarMatcher`, `SessionSpec`,
`messages_have_images`, `TokenEvent`, `ModelMeta`, `LoadRequest`, `HookBus`, and the
settings types (the ~20 paths `src/backend/mlxcel` reaches today). Root
`Cargo.toml` becomes a workspace root with `crates/gen2-mlxcel` (`publish = false`);
mlxcel's module moves there verbatim plus `pub fn plugin() -> BackendPlugin`. The
mlx/mlxcel link-conflict note stays in docs.
Honest ⬜: mlxcel throughput unmeasured; mlx+mlxcel still cannot be linked together.
Landed as designed, with three deviations: the seam lives under `gen2::advanced::plugin`
(api_spec §25 puts local/backend-specific controls under `gen2::advanced`, not the
root); `Engine::builder().backend(plugin)` and `ControllerConfig.plugins` carry
`Arc<BackendPlugin>`; and the "no backend selected" `compile_error!` is gone — a build
with no `backend-*` feature compiles (a plugin-only consumer needs none), and the first
load nothing claims fails with an error naming both ways out. CI's
`no-backend-is-a-compile-error` job became `no-backend-build-still-works`.
Status: ✅ (see ledger)

#### S1.4 Release readiness
Outcome: everything but the publish button is done.
Observable: `[package.metadata.docs.rs]` present and `cargo doc` with those exact
features is warning-free; CI `publishable` is a hard gate; CHANGELOG.md exists;
README install snippet says `gen2 = "0.1"`.
Gate: dry-run exit 0 on `main`; CI green. Blocked by: S1.3. Forks: ⛔ publish + tag (Victor may prefer to publish only after wave 2 so 0.1 ships the spec's API; the slice prepares either way).
Status: ✅ 2c0bd33 — ⛔ STOP: publish and tag are Victor's

### Wave 2 — the inference-first facade (`api_spec.md`)
Ceiling for the whole wave: every public example in the spec's §28 walkthrough
becomes a doc test or a live test on Qwen3-0.6B; the old `Engine` facade stays
until S2.6 retires it, so pio-app's switchover (wave 5) can target either.

#### S2.1 `Runtime`, `Model`, `gen2::load`, one-shot `generate`
Outcome: spec §4–§6 and §28.1–28.2 work over the existing controller.
Observable: `let model = gen2::load(path)?; model.generate("hello").text()?` is a
live test that prints a token; `Runtime::new()?.load(path)` returns a cloneable
`Model` with `info()`/`capabilities()` per §5; `runtime.openai().base_url(..)
.model(..).connect()?` produces a `Model` (mockito test, mirrors tests/external_openai.rs).
Reference to mirror: spec §4–§6, §25; `src/api/engine.rs` (what to wrap).
Gate: `cargo test`; live test; docs.rs-feature doc build. Blocked by: S1.4.
Forks: none. Honest ⬜: multi-model residency automation is S2.4.
Status: ✅ (see ledger)

#### S2.2 `Session`: first-order system prompt and tools, active projection, history
Outcome: spec §7–§9 (`Session::new().with_system(..).with_tools(..)`, `messages()`
active vs `history()` append-only, message ids, edit/remove/replace/fork,
`SessionRevision`).
Observable: unit tests for every §7 operation, each asserting the §29 invariants
(edits lossless; active vs record separate); `src/journal` is the substrate.
Reference to mirror: spec §7–§9, §29; `src/journal/*`.
Gate: `cargo test`. Blocked by: S2.1. Forks: none.
Status: ✅ (see ledger; `Message` enum → S2.6, `ToolChoice`/tool protocol → S2.3)

#### S2.3 `Turn`, `Response`, semantic `EventStream`, tools protocol, structured output
Outcome: spec §10–§16: `model.turn(&mut session).user(..).run()?`, `.stream()?`
yielding §15 events, `ToolDefinition`/`ToolSet`/`ToolChoice`/`ToolCall` with NO
execution in core, `.structured::<T>()` enforced by grammar where the backend can,
cancellation as a finish reason with partial output kept (§16).
Observable: §28.5 tool loop and §28.10 streaming walkthroughs pass as live tests on
a tool-capable GGUF; §28.11 structured output live test.
Reference to mirror: spec §10–§16, §28; existing `api/chat.rs`, grammar module.
Gate: `cargo test`; live tests. Blocked by: S2.2. Forks: none.
Status: ✅ (see ledger; `GenerationOptions::advanced(..)` grouping, stop sequences, grammar-forced `ToolChoice`, remote tool-call/reasoning parsing → ⬜ there)

#### S2.4 Model switching, cache identity, multiple resident models
Outcome: spec §17 and §23: two `Model`s from one `Runtime`, a session switched
between them mid-chat (§28.4), cache identity keyed by model/session/context
fingerprint, eviction and automatic restore (§4.2).
Observable: live test switching Qwen3-0.6B ↔ Llama-3.2-3B in one session.
Gate: `cargo test`; live test. Blocked by: S2.3. Forks: none.
Honest ⬜: accelerator contention scheduling beyond queueing.
Status: ✅ (see ledger; prefix reuse after a switch is rebuilt-not-appended by design until the backends render foreign assistant messages; budget = file-size estimate)

#### S2.5 Async surface
Outcome: spec §22: `run_async`, `stream_async` under the `tokio` feature with no
parallel object model. Gate: `cargo test --features tokio`; existing async lane.
Blocked by: S2.3. Status: ✅ (see ledger; the old `AsyncTurn`/`spawn_async` layer retires with the facade in S2.6)

#### S2.6 Root namespace and retirement of the old facade
Outcome: spec §25–§27: root exports exactly the §25 list; `gen2::advanced` holds
local tuning, residency, hardware, grammar, and the backend plugin seam from S1.3;
the agent layer moves behind an `agent` feature as `gen2::agent` (default per the
human fork above, flagged); `Engine`/`Chat`/`Inference` removed or kept as
`#[deprecated]` shims for one release; README examples rewritten to the new API.
Forks: ⛔ README opening/tagline (present the new text to Victor); ⛔ delete vs
feature-gate the agent layer. Gate: `cargo test`; `cargo doc`; every README
example doc-tested. Blocked by: S2.5.
Landed as: root = §25 + `load`/`Turn`/`Input`/`EventStream`/`schemars`; `gen2::api`
and `gen2::controller` crate-private; `gen2::legacy` = deprecated *type aliases*
(a `#[deprecated] pub use` is inert in rustc 1.98, aliases warn; the aliased
structs carry `#[allow(unnameable_types)]`); `gen2::agent` behind default-on
`agent` (⛔ **flagged for Victor: kept as a feature, not deleted** — pio-app
consumes it; `Agent::on(&model, &mut session)` is the non-deprecated entry, owned
agents still take a `legacy::Engine`); `gen2::advanced::{generation, runtime, fit,
wire, controller, utilities, plugin}`; `Runtime::builder().backend(plugin)`;
§21 `Runtime::load_embedder`/`load_reranker` → `Embedder`/`Reranker`.
Status: ✅ (see ledger; §24 error reshaping and §9 `enum Message` → ⬜ there)

### Wave 3 — five-minute first run
#### S3.1 `hf:` model references
Outcome: `Engine::load("hf:unsloth/Qwen3-0.6B-GGUF")` downloads to a cache and loads.
Observable: a live test with a temp cache dir downloads Q4_K_M and generates a token;
`:Q8_0` and `:file.gguf` forms resolve; `HF_TOKEN` honoured; progress callback fires.
Reference to mirror: ollama/llama.cpp `hf:` grammar; `hf-hub` crate API; receipt 03.
Gate: unit tests for the reference parser (offline) + live download test.
Blocked by: S2.6. Forks: none. Honest ⬜: HF rate limits under CI not observed.
Status: ✅ branch s3.1-hf (2026-09-06) — `gen2::hf` (`src/api/hf.rs`), `hf` feature
default-on over `hf-hub = "1"` (blocking); live: cold download+load 44.4 s, warm 0.2 s
offline. See ledger.

#### S3.2 Metal is the default on Apple Silicon
Outcome: nobody types `metal` to get the GPU.
Observable: a macOS CI job with default features loads a model and reports GPU offload.
Gate: that job green; README no longer says "add `metal`". Blocked by: S3.1.
Status: ✅ branch s3.first-run (2026-09-10) — proved in the binding's own manifest
and at run time: `ModelInfo::offload` reports `29/29 layers on the GPU (MTL, Apple
M4 Pro)` under DEFAULT features, and `metal_is_on_by_default_on_apple_silicon`
asserts it. See ledger.

#### S3.3 `hello` example, README first block, prerequisites
Outcome: the README's first screen is the whole five-minute path.
Observable: `examples/hello.rs` runs with no arguments; README has a per-OS
prerequisite line and an sccache hint; `examples/minimal.rs` no longer unwraps.
Gate: `cargo run --example hello` prints a token on this Mac; doc tests pass.
Forks: ⛔ opening paragraph/tagline untouched; any positioning sentence listed for
Victor in the ledger. Blocked by: S3.2.
Status: ✅ branch s3.first-run (2026-09-10) — `examples/hello.rs` zero-arg on
`hf:unsloth/Qwen3-0.6B-GGUF`; cold 35.3 s / warm 2.1 s on this Mac; README install
block, first screen and Hugging Face subsection rewritten. See ledger.

#### S3.4 Windows lane + fresh-clone job
Outcome: CI proves the README on three OSes.
Observable: `windows-latest` compiles and runs unit tests with default features; a
`first-run` job on macOS and Linux does `cargo run --example hello` with the HF cache
restored by `actions/cache`. Gate: CI green. Blocked by: S3.3.
Honest ⬜: Windows GPU; MSVC first-build time not optimised.
Status: ✅ branch s3.first-run (2026-09-10) — the Windows half landed 2026-09-06;
this half adds the `first-run` job (ubuntu + macOS, hard gate, model cached by
`actions/cache`, the macOS run doubles as S3.2's proof). ⬜ unobserved until the
push. See ledger.

### Wave 4 — honest benchmarks
#### S4.1 Benchmark harness and first results
Outcome: a benchmark table a skeptic can reproduce.
Observable: `benches/results/<machine>/<date>-<sha>.json` committed for this Mac,
produced by a harness that builds llama-bench from the vendored sha and asserts
`build_commit`; README table generated between markers by a `bench-table` bin.
Reference to mirror: `llama.cpp/tools/llama-bench`, `scripts/compare-llama-bench.py`,
mistral.rs release report shape. Gate: bin regenerates the table byte-identically.
Blocked by: S1.4 (may run alongside wave 2). Honest ⬜: RTX 3080 and Pi 5 rows.
Status: ✅ (see ledger). Deviations from receipt 04, forced by the code: the crates.io
`llama-cpp-sys-2` tarball ships a *subset* of llama.cpp (no `tools/`, no `.git`), so
the reference is built from upstream llama.cpp at the commit the binding repo's tag
names, after proving every vendored file byte-identical to it (1123 files); the
`bench-table` generator is a `harness = false` bench, not a `src/bin`, so it is neither
installed nor built by `cargo publish`; result files carry the model slug in the name
(one date + one sha would otherwise collide across models); tg128 is on the engine's
clock (`decode_tokens / avg_tps`), with the per-turn wrapper cost reported separately
in ms because a wall-clock rate charged context creation to decode (first smoke run
read 72 %, the real number is 93–95 %).

#### S4.2 Freshness gate
Outcome: the table cannot rot silently.
Observable: `bench-freshness` CI job fails on table drift, llama.cpp pin drift, or
age >90 days — `cargo bench --bench bench_table --no-default-features -- --check
--max-age-days 90` already does all three; the job wires it. (The unsourced LiteRT-LM
"1.7x" sentence was removed in S4.1.)
Gate: CI green. Blocked by: S4.1.
Status: ✅ (see ledger)

### Wave 5 — pio-app consumes the crate
#### S5.1 `compat` surface in gen2
Outcome: pio-app's 454 import leaves resolve against the crate.
Observable: `#[doc(hidden)] pub mod compat` re-exports the 40 `pub(crate)` paths listed
in receipt 05; `SystemTask` gap documented with the mapping pio-app must apply.
Gate: `cargo test`; a compile-only check in pio-app's worktree shows the error count
drop from ~204 to only `SystemTask`/`ControllerEvent` semantic sites. Blocked by: S2.6 (target the new facade where pio-app's use is inference; `compat` covers the rest).
Landed as: `#[doc(hidden)] pub mod compat` = the old module tree verbatim
(`controller{,::config}`, `engine{,::telemetry}`, `generation{,::events,reply_parts,spec,telemetry,thinking}`,
`backend{,::caps,facade,health,traits,common::{grammar,tokenizer,chat_template,sampler,
stop_matcher,output_filter,tool_calls,speculative},llama::{embedder,llama_config},
external_api,litertlm,mistralrs,mlx}`, `session_rt{,::compaction,media_util,prompt,spec,truncate}`,
`zoo`, `bundle{,::gguf,meta}`, `kv{,::store,types}`, `executor`, `residency{,_policy,_stats}`,
`router`, `media`, `hardware`, `memory`, `utilities`, `types{,::message,model,persona,execution_stats}`)
plus the old root re-export list, where `compat::Engine` is the **backend facade**
(`legacy::Engine` is the API one); `compat::system_task` rebuilds the 9 removed
`SystemTask` variants as `Custom` labels with `gen_spec()` carrying their old tuning;
`tests/compat_pio_paths.rs` names every inventoried path; inventory + residual-change
list in `docs/plans/research/05a-pio-app-import-map.md`.
Status: ✅ (see ledger; the semantic residue is S5.2's)

#### S5.2 pio-app `gen2-crate` feature (PR)
Outcome: pio-app builds against the standalone crate behind a feature.
Observable: worktree off origin/main, `gen2-crate` feature with path dep, shim
re-export, three `RemoteDispatch` impls, `HostInference` enum; gate per receipt 05
(filtered lib tests ×3, integration suites, live chat/tool/gemma4/flock tests).
Integration: PR, not merged by the agent. Blocked by: S5.1.
Honest ⬜: specta bindings drift only checked by diff.
Status: ⬜

#### S5.3 Flip default, delete in-tree copy (PR)
Outcome: pio-app has one gen2. Blocked by: S5.2 merged by Victor. Forks: ⛔ (merge).
Status: ⬜

## Heartbeat adapter

In-session continuous execution: the agent runs slices sequentially in this session,
dispatching build/verify to subagents in worktrees, and treats this doc as the ledger.
No TaskCreate available; no claim of unattended background execution beyond this
session. A ScheduleWakeup is the safety net if a background job goes quiet.

## Ledger

- 2026-09-04 S0.1 aefe0b8 · clippy/test/doc/fmt green locally; rerank test 10/10; docs 05899a5 · no detours · ⬜ CI confirmation
- 2026-09-04 S1.1 (this commit) · llama-cpp-2 =0.1.156 (llama.cpp b10405), penalties ported with -1→n_ctx; mlx-rs hybrid 0.25.3+git · unit 1015, live 22/22 Metal, clippy/doc/fmt/ext-api green · dry-run now fails only on mlxcel
- 2026-09-04 ↷ detour (this commit) · Linux memory probe read sysinfo.freeram (excludes page cache) so the residency governor denied helper loads after big builds; the mistral.rs CI lane had failed on it for 6 runs · now MemAvailable from /proc/meminfo, parser unit-tested · ⬜ host-memory dependence of the acceptance tests remains (governor is a global)
- 2026-09-04 S1.3 worktree-agent-a7a3921378b55dedc · `gen2::advanced::plugin` (BackendPlugin {name, claims, make} + 50 re-exported implementer types), facade `Engine::Plugin` variant asked before every built-in rule, `ControllerConfig.plugins`, `EngineBuilder::backend`; `compile_error!` guards removed (no-backend build is a run-time error at load); `src/backend/mlxcel` → `crates/gen2-mlxcel` (workspace member, `publish = false`, `pub fn plugin()`), root manifest has no mlxcel dep/feature · gate: `cargo publish --dry-run --allow-dirty` EXIT 0 (252 files, 5.7 MiB, verify build passed); `cargo test` 1025/0/16 ignored default + 1087/0/2 ext-api-only; plugin routing test green under default AND `--no-default-features --features backend-external-api` AND no features (3 tests, `advanced::`); clippy -D warnings default + no-features; doc -D warnings; rustfmt; check ext-api; `grep -rn mlxcel Cargo.toml src .github` → only the workspace `members` line and its comment in Cargo.toml (inherent to a workspace member) · companion: `cargo build -p gen2-mlxcel` EXIT 0 on macOS 26.5 (MLX C++ via cxx, Metal), 6 unit + 2 weightless + 1 doctest green, LIVE test green with qwen3-0.6b-4bit: mlxcel decoded its first token inside gen2 ("<think>\nOkay, the user wants me", 8-token cap) — off `NEVER_PRODUCED_A_TOKEN` · ⬜ CI confirmation; ⬜ mlxcel throughput unmeasured; ⬜ mlx+mlxcel link conflict now a doc rule, not a compile guard; ⬜ pio-app not yet switched to the companion (S4)
- 2026-09-04 S1.2 worktree-agent-a735c99fdb336ec65 · onnx+candle removed (1862 lines deleted, 685 added, net −1177; 1,525 LOC of backend code), mistralrs forwards metal/cuda (cargo tree proof, no Metal build), README tiered · gate: cargo test 1016/0/16 ignored default + 1014/0/4 mistralrs lane (CPU), clippy -D warnings, doc -D warnings, rustfmt, check ext-api / litertlm / llamacpp+litertlm, grep clean · ⬜ CI confirmation
- 2026-09-04 ↷ input: Victor pushed `api_spec.md` (660cca4) after S1.2; roadmap re-sliced: new wave 2 = the facade (S2.1–S2.6), first-run → wave 3, benchmarks → wave 4, pio-app → wave 5; S1.3 in flight, told to put the plugin seam under `gen2::advanced`
- 2026-09-04 goal set by Victor: "api_spec.md, ideally you get this done" → wave 2 is the priority; S2.1 dispatched in parallel with S1.3 (disjoint files, S2.1 rebases onto S1.3); S0.1 CI confirmed green at bdc6576 (first fully green main)
- 2026-09-04 S1.4 2c0bd33 · docs.rs metadata (no-default-features; llamacpp/external-api/litertlm/tokio; x86_64 linux) verified with a local doc build; CI `publishable` is a hard gate; CHANGELOG.md · README already said publishable (S1.3) · install snippet stays `git` until the crate exists on crates.io (⛔ publish) · dry-run exit 0 on main at 64c1f81 and 2c0bd33 · ⬜ CI confirmation
- 2026-09-04 S2.1 branch s2.1-facade (rebased onto S1.4 3b1b377) · `Runtime`/`Model`/`gen2::load`/`Generation`/`Response`/`Input`/`gen2::{model,input,output}` over the existing `Engine` (one engine per loaded model; one-shot = `Chat` on a throwaway `Session`); remote model name threaded as `LoadRequest.api_model`/`EngineBuilder::remote_model`; loaded context window added to the runtime snapshot (`Backend::context_window`); `types::model::Model` re-exported as `gen2::ModelRecord` so `gen2::Model` is the spec's · gate: cargo test 1053/0/16 ignored default (+18 fit, 24 external, 58 doc), clippy -D warnings, doc -D warnings (default + docs.rs feature set), rustfmt, ext-api lane 1115/0/2 + remote_runtime 4/4 (mockito: `"model":"m"` on the wire, key sent/omitted), live 24/24 Metal on Qwen3-0.6B (`gen2::load(..).generate(..).text()` → "hello"; clones on two threads; info: qwen3, ctx 40960; caps: tools/reasoning/structured_output) · ⬜ reasoning capability is an architecture allowlist (qwen3/gemma4) and `Response::reasoning()` splits `<think>`/Gemma thought scaffolds out of the token text (S2.3 makes it a semantic event) · ⬜ remote `tools`/`structured_output` report false: the external-api request carries no `tools`/`response_format` · ⬜ `FinishReason::Length` is inferred from `decode_tokens >= max_tokens` (backends report Eos for both); `ContentFilter`/`Error` never produced yet · ⬜ `InputPart::Audio` absent (no audio message chunk exists) · ⬜ §4.5 `hardware/residency/preload/evict/stats` → S2.4; `Model::turn`/`.structured()` → S2.3; root still exports the old `ModelInfo`/`ToolSet`/`Event`/`Turn` names (S2.6) · ⬜ `ControllerCmd::LoadModel` gained a field (external constructors must add `api_model`)
- 2026-09-04 ↷ detour (this commit) · CI `metal,tokio` lane on 33aa40f failed `a_follow_up_reaches_the_model_at_the_next_step`: a follow-up queued after spawn raced the loop's end and was dropped unless the finish was an interrupt. Loop now delivers any pending steer at the next step; `OwnedAgent::steering()` exists before `spawn`, and the test queues first · 6/6 under metal,tokio, full suite green
- 2026-09-06 S4.1 worktree-agent-aa3c4ea57501ce2c2 (measured 2026-09-04 at ba0ab9d) · `benches/build-llama-bench.sh` builds llama-bench from llama.cpp e79e4bf6 = b10405 (the commit `utilityai/llama-cpp-rs` tag 0.1.156 pins; 1123 registry-vendored files proven byte-identical; same cmake flags as the crate's build.rs, Metal), compiled `build_commit e79e4bf66` asserted by the harness on every row; `benches/backend_parity.rs` rewritten: pp512/tg128 `-r 5` + 1 warmup, `-o json` `samples_ts`, median ± sample stddev both sides, greedy, `-b 512 -ub 512 -t 4 -ngl 99 -fa auto -ctk/-ctv f16` mirroring gen2's defaults, TTFT from `first_token_us`, thermal bracket, machine/model/sha256 fingerprint, JSON to `benches/results/m4pro-14c-64g/`; `benches/bench_table.rs` (harness=false, `--write`/`--check`/`--max-age-days`) rewrites README between `bench:begin/end`, refuses pin drift and unknown schema · **Apple M4 Pro 64 GB, macOS 26.3, AC: Qwen3-0.6B Q4_K_M tg128 294.4±4.1 vs llama-bench 310.2±1.7 tok/s = 0.95; Llama-3.2-3B-Instruct Q4_K_M 94.0±0.8 vs 101.1±0.2 = 0.93**; pp512 0.95 both (includes template + session setup); TTFT 90.7 / 486.1 ms; per-turn wrapper cost 25.8 / 38.3 ms; thermal drop −1.2 % / 0.2 %; both `valid: true` · LiteRT-LM "1.7x" sentence removed · gate: cargo test 1112/0/19 ignored, clippy --all-targets -D warnings, rustdoc -D warnings, fmt, `cargo publish --dry-run --allow-dirty` EXIT 0 (260 files, 5.8 MiB; was 254), bench-table byte-identical twice, `--no-default-features -- --check --max-age-days 90` green · ⬜ a day-old `docker run … cargo build` container from another session was present for both runs (recorded in each JSON's `warnings`/`notes`; load avg ~3 on 14 cores) — rerun on a quiet machine before citing externally; ⬜ RTX 3080 and Pi 5 rows; ⬜ sustained thermal beyond the first/last bracket; ⬜ peak memory and depth sweeps; ⬜ pp512 is not the same work until gen2 has a token-in path; ⬜ CI confirmation (S4.2 wires the freshness job)
- 2026-09-06 S4.2 (this commit) · `bench-freshness` CI job runs `bench_table --check --max-age-days 90` with no default features (drift, pin, age); LiteRT-LM 1.7x sentence already removed in S4.1 · local check: "README.md is current (2 rows)" · ⬜ CI confirmation
- 2026-09-06 S2.5 branch s2.5-async · `Turn::run_async`/`stream_async`, `Generation::run_async`/`text_async`/`stream_async` and `gen2::event::AsyncEventStream` (= `gen2::AsyncEventStream`; `futures::Stream<Item = Result<Event>>` + `FusedStream`, `finish().await -> Response`, `canceller() -> Canceller` — the same type as sync) in `src/api/async_turn.rs` under `tokio`; no new option methods, same `Event`/`Response`/`FinishReason`/`Canceller` · mechanism: `EventStream<'a>` split into a `Send + 'static` `StreamCore` (token stream + assembler + queue + outcome) and a `SessionSlot<'a>` (`Borrowed(&'a mut Session)` | `Owned(Box<Session>)`); `Turn::begin()` does the non-blocking half (validate, commit, `engine.send`) on the caller's thread and hands the core to `tokio::task::spawn_blocking`, which pulls events into a bounded `tokio::sync::mpsc` (`EVENT_BUFFER = 64`; `blocking_send`, so an unpolled consumer parks the worker) and returns the core through the `JoinHandle`; the consumer then runs the same `StreamCore::complete(&mut Session)` the sync iterator runs (settle, record the reply, `opened`), so the four call shapes share one code path and the caller's `&mut Session` never leaves the future (`model.turn(&mut session).user("hello").run_async().await?` compiles as written in §22); dropping an `AsyncEventStream` mid-flight calls `Canceller::cancel()`, marks the session `opened = false` (the reply was never recorded, so the next turn rebuilds from the session's messages) and aborts the handle (a no-op for a running blocking task, which drains to the end on its own) · sync `Generation::stream()` added alongside (a one-shot stream owns its ephemeral session; `finish()` is `detached()`, `Drop` forgets the session in the engine) · gate: cargo test --features tokio 1122/0/16 ignored (+18 fit, 31 live-skip, 75 doc; 9 new async_turn unit tests: `run_async` == `run` (text/reasoning/calls/finish/message id/session messages/revision), `stream_async` yields the identical event vector to `stream` (reasoning + text + tool-call Start/ArgumentsDelta/End + Finished), one-shot `text_async`/`stream_async`/`stream` agree and forget the session, cancel from a `tokio::spawn`ed task → `Finished(Cancelled)` + partial recorded + `remove_message` ok, canceller-before-run → not run (0 pulls), drop mid-way → `end_session` observed, the later `Gate` never reached, session rebuilt on the next turn (`start_session` 2, the abandoned user message in the rebuilt transcript), bridge bounded: 300 tokens unpolled → backend reaches its hold, `rx.len()` == 64 and stays, then all 300 deltas in order; refusal leaves the session untouched; `Send` compile test over `Model`/`Response`/`Event`/`Canceller`/`AsyncEventStream`/`EventStream` and the `run_async`/`stream_async`/`text_async` futures) ; ext-api lane 1184/0/2 (+4 anthropic, 4 openai, 18 fit, remote_runtime 6/6, 75 doc); clippy --all-targets -D warnings under `tokio` and `backend-external-api,tokio` (one real hit fixed: `SessionSlot::Owned` boxed); rustdoc -D warnings docs.rs set; rustfmt; live 31/31 Metal on Qwen3-0.6B (36.6s; 29 existing + 2 new, `--features metal,tokio --test-threads=1`): `text_async` → "hello"; `run_async` → "The sky on a clear day is typically blue." (Stop, stats reported, message id == last active) then a blocking follow-up on the same session → "At night, the sky is usually dark blue or black." (4 messages); `stream_async` → 24 `TextDelta` + `Finished(Stop)`, `finish().await.text()` == the deltas; cancel from a spawned task after the first delta → 2 more events, `Finished(Cancelled)`, partial removed, next `run_async` → "Hello! I'm Pio Chat," · ⬜ the controller emits tokens best-effort (`emit_best_effort` → `try_send`) and sheds past its 512-slot event channel, so an idle reader — async or sync — loses tokens after 512 + 64; the bridge bounds the worker, not the backend (a real backpressure path needs the controller to park the puller on a full channel) · ⬜ a dropped `AsyncEventStream` is not joined: the worker drains after the stop on its own thread, and `Session` sees the stop only as `opened = false` (no partial reply recorded for a drop; `finish()` after `cancel()` is the §16.1 path) · ⬜ `Turn::structured` has no `structured_async` (`run_async` + `decode` is two lines; not added) · ⬜ `stream_async` is an `async fn` whose body never awaits — prep is non-blocking — kept `async` so §22's `turn.stream_async().await?` reads as written · ⬜ the scripted backend still cancels one token late (asserted `starts_with`; llama.cpp stops at once, live: 2 events after cancel) · ⬜ the old `OwnedChat::spawn_async`/`send_async`/`AsyncTurn`/`AsyncAgentRun` layer (`src/api/asynchronous.rs`, unbounded channel) is untouched and retires with the old facade in S2.6 · ⬜ the backpressure unit test checks "stays at 64" with a 50 ms sleep after the eventually-loop (the positive assertions are deterministic; the "never past its bound" one is a bounded-channel type guarantee restated) · ⬜ CI confirmation (`tokio` job + the `metal,tokio` backends lane skip the live tests without a model)
- 2026-09-06 S2.4 branch s2.4-switch · **switching is a normal turn** (§17, §28.4): `Session` binds per engine — `Engine` gained a process-unique `id`, and the session's process-local bookkeeping (`opened`, model generation, thinking, tools) became a `Binding` per engine, parked on switch and restored on return (`note_engine`), so `qwen.turn(&mut s)…; llama.turn(&mut s)…` recreates the model-specific runtime on whichever engine is invoked from the session's own transcript and nothing is converted (invariant 2) · **cache identity** (§17.1): `gen2::session::ContextFingerprint` — `Session::fingerprint()`, FNV-1a over system prompt, tool set (definitions+order), active message ids+content; recomputed in `log()`/`rebuild_projection()` exactly when the revision moves, O(active) per mutation, free to read — and a *prefix* digest recorded at settle (`record_prefix`: the first `sent_through` transcript entries + the offered tools' digest) that `Chat::begin` re-derives and compares before appending (`prefix_still_held`); `SessionRevision` stays the cheap signal, the digest is the proof, a mismatch re-prefills (invariant 7) · **found on the way, live**: llama.cpp's `append_messages` renders only non-assistant deltas (it assumes every assistant message in the tail is its own reply), so an appended qwen runtime after qwen→llama→qwen would never see llama's reply — `Session::tail_is_appendable` now rebuilds whenever the tail holds an assistant message the engine did not generate (`own_reply_follows` per binding); reuse happens when it doesn't (unit-tested: llama's reply removed → qwen appends, `start_session`=1/`append_messages`=1) · also found: the controller's ContinueChat-after-eviction rebuild did `transcript + new_messages` while the facade sent the *whole* transcript, so a rebuilt runtime saw the tail twice — now the facade sends exactly the held prefix (asserted `seen` == `["one","ok","two"]`) · **S2.3 ⬜ fixed**: `ControllerCmd::ContinueChat` carries `thinking` and `tools`, the rebuild pins both (scripted: `max_active_sessions(1)`, evicted session comes back with `tools_seen == ["read"]`, `thinking_seen == Off`) · **residency** (§4.2, §4.5, §1.2): `Loaded` remembers `estimated_mb` (`residency_policy::estimate_resident_mb_for_path_offloaded`), `resident`, `last_used`; `Runtime::{hardware, residency -> ResidencySnapshot{models: Vec<ModelResidency{id,name,local,resident,estimated_mb,idle_for}>}, preload, evict, stats -> RuntimeStats{models,resident_models,estimated_resident_mb,active_sessions,evictions}}`, documented as advanced, types re-exported at `gen2::advanced::runtime` (+`HardwareProfile`/`GpuBackend`); a runtime-wide `admission` mutex + ledger: `load`/restore call `make_room(extra, keep)` → `can_admit` (ledger + governor `inference_resident_mb` and `can_load_additional_model`; tests pin `resident_budget_mb`) → evict LRU resident local model until it fits or nothing is left (then the engine's own admission has the final say); `Model::ensure_resident` at the top of every `Turn::stream` (so `generate` too) reloads via `Engine::reload_model` (llama keeps its last `LoadRequest`); controller `UnloadModel` now retires every chat first (llama's bundle is only released by its last session) and moves the residency slot to `idle_unloaded_llm`, `ReloadModel` re-admits it; the fake backend's `reload_model` now survives an unload like llama.cpp's · **concurrency** (§23): `Runtime`/`Model`/`Session: Send + Sync` compile-asserted; two threads, two sessions, one model → both complete, `live_sessions == 2` · gate: cargo test 1131/0/16 ignored default (+18 fit, 33 integration, 69 doc; 18 new `api::switching_tests`), ext-api lane 1193/0/2 (+4/4/18/6 remote_runtime/69 doc), clippy --all-targets -D warnings default + `backend-external-api,tokio`, rustdoc -D warnings default + docs.rs set, rustfmt; **live 33/33 Metal, Qwen3-0.6B ↔ Llama-3.2-3B-Instruct Q4_K_M (`PIO_TEST_SECOND_MODEL`, README)**: §28.4 qwen "Hello Bob! How can I assist you today?" → llama "Bob" on one session; §17.2 qwen→llama("PINEAPPLE")→qwen recalls PINEAPPLE, cumulative prompt tokens 385/428/433, TTFT 74/405/78 ms (33.2 s for the suite); evict/restore: `residency()` both resident (384 MB est. each) → `evict(&qwen)` → qwen `resident: false` → `qwen.generate("…hi")` = "hi" → both resident, `stats{models:2,resident_models:2,estimated_resident_mb:768,active_sessions:1,evictions:0}`, `evict`+`preload(&llama)` round-trips; two threads "red"/"blue"; 29 existing live tests unchanged · ⬜ prefix reuse after a *switch* is not exercised live and by design rebuilds: the llama.cpp (and mlx) append path skips assistant messages, so reuse across models needs the backends to render foreign assistant turns as history and the facade to count its own reply as held (a backend-contract change; the mechanism — parked bindings + prefix digest — is in place and unit-proven) · ⬜ reuse is not provable live through public stats: llama.cpp reports `prompt_tokens` as the KV position at pull time (cumulative), identical for append and rebuild; TTFT is printed, not asserted · ⬜ budget heuristics: `estimated_mb` is the residency_policy file-size estimate (384 MB host overhead for a fully offloaded Metal model, which undercounts unified memory), and `can_admit` trusts the governor's `inference_resident_mb`; no measured RSS per model · ⬜ eviction is LRU by last turn only — no size-aware or pressure-driven choice, and the controller's own idle/pressure unloads are noticed only on the next restore (`is_model_loaded` probe) · ⬜ accelerator contention: N engines = N controller loops, each serialising its own turns; two models generating at once share the GPU with no scheduling beyond that · ⬜ `active_sessions` in `stats()` is a sum of controller snapshots (one round trip per engine); no aggregate token/throughput counters · ⬜ `Session::fingerprint` is process-local FNV over serde_json of each message (not a stable content address); prompt/template config enters the binding via thinking + offered tools, not the public fingerprint · ⬜ remote models report `resident: true`, `estimated_mb: 0`; evict/preload are no-ops for them
- 2026-09-06 S2.3 branch s2.3-turn · `Model::turn(&mut session) -> Turn` (src/api/turn.rs): staging `user`/`input`/`image`/`message`/`messages`, committed only when execution begins (a text-only model refusing an image, or a `ToolChoice` naming an unoffered tool, leaves the session and its revision untouched — unit-tested); no-new-message turns (§11.2); per-turn `max_tokens/temperature/seed/top_p/top_k/min_p/greedy/reasoning/tool_choice/options`; `system_override`/`tools_override` per turn only (reopens the conversation for the turn and closes it after; session `system`/`tools`/revision unchanged, no `SystemSet`/`ToolsSet` event — unit-tested against what the scripted backend saw); `run() -> Response`, `stream() -> EventStream`, `structured::<T>()`; `gen2::GenerationOptions` (§12, `#[non_exhaustive]`, builder, `reasoning: ThinkingMode`, maps onto `GenSpec` over the engine's defaults; expert knobs stay in `gen2::advanced`); `gen2::ToolChoice {Auto, None, Required, Tool(name)}` (`None` offers nothing, `Tool` offers only that definition, `Required`/`Tool` add a prompt instruction); `output::ToolCall::{id,name,arguments,parse_arguments}`; ids minted `call_<message id>_<n>` from the session's event-log counter (unique per session, stable across replay; provider ids kept); the reply is appended via `Session::push` as `assistant_structured(text, reasoning)` or `assistant_tool_calls` under those ids, `Response::message_id()` names it; §28.5 loop verbatim over the scripted engine (`push_tool_result(call.id(), ..)` → `turn().run()` → text) · `gen2::event::{Event, EventStream, Canceller}` (§15/§16): `Event::{TextDelta, ReasoningDelta, ToolCallStart, ToolCallArgumentsDelta, ToolCallEnd, Usage, Finished}`; reasoning synthesised by `ReplyStateMachine` over the union of the crate's `ChannelMarkers` (Qwen3/DeepSeek `<think>`, Gemma 4 thought channel) with the scaffold's own whitespace held back so deltas never carry what the message will not (S2.1's `split_reasoning` string-splitter deleted; `run()` = `stream()?.finish()`, one code path, so the two cannot disagree); a complete backend tool call becomes Start/ArgumentsDelta/End; `Canceller` (`Clone + Send`, `cancel()` from another thread) ends the stream `Finished(Cancelled)`, `finish()` keeps the partial text, the partial assistant message is recorded and `remove_message(id)` works (§16.1 as a unit test and live); `Turn::canceller()` before `run()` makes the turn not run at all (nothing staged) · `Generation` now runs a `Turn` on an ephemeral `Session` (§6), gained `tool_choice`/`reasoning`/`options`/`structured::<T>()`, `tools()` takes anything convertible to `ToolDefinition`s · structured output: grammar = `schema_for!(T)` when `capabilities().structured_output`, else the schema is asked for in the prompt and the reply parsed (Markdown fence stripped); reasoning forced `Off` unless set; a non-decoding reply is `Error::Extraction` with the raw text · `Session` gained process-local `thinking` bookkeeping (`note_thinking`: a different policy reopens, since backends pin it at start), `transcript_with_system`, `next_message_id`; `Chat` gained `pub(crate)` `system_override`/`without_tools`/`begin`/`settle` · fixes found live on Qwen3-0.6B: (1) the llama session ignored `SessionSpec.thinking` (family default only) and rendered every continuation delta and overflow probe with `enable_thinking: None`, so a Qwen3 `ContinueChat` fell back to the template's default and *thought* (the §28.5 continuation spent its whole budget in `<think>` and answered nothing; the old `a_named_chat_continues_across_turns` transcript shows the same `<think>` on turn 2) — now `On`/`Off` are honoured, `Auto` = family default, and the flag is pinned on the session and passed to every render (`src/backend/llama/session.rs`, `engine.rs`); (2) the controller's `FinalStats` forwarder was registered once with the first turn's channel, so every continuation's stats went to a dropped receiver (`usage` zero, `reported() == false`, `Length` undetectable) — `ContinueChat` now re-points it (`src/controller/commands.rs`; asserted live on the tool-loop continuation) · gate: cargo test 1113/0/16 ignored default (+18 fit, 69 doc; 26 new turn/event/generation/response unit tests incl. staging/commit, no-new-message, overrides-not-persisted, event synthesis from scripted tokens, mid-stream cancel, structured parse/typed error); ext-api lane 1175/0/2 (+4 anthropic, 4 openai, 18 fit, remote_runtime 6/6 — new: a remote `turn().stream()` yields `TextDelta`×3 + `Finished(Stop)` from a mockito SSE and records the reply; remote `structured()` parses without a grammar and reports prose as `extraction_failed`); clippy --all-targets -D warnings default + `backend-external-api,tokio`; rustdoc -D warnings default + docs.rs set; rustfmt; live 29/29 Metal on Qwen3-0.6B (33.3s; 24 existing + 5 new): §28.5 tool loop (`get_weather` → `ToolCall{call_1_0, city: Paris}` → `push_tool_result` → no-user-message turn → "The weather in Paris is currently 18°C with clear skies.", stats reported on the continuation), §28.10 streaming (24 `TextDelta` + `Finished(Stop)`, `finish().text()` == the deltas; with `reasoning(On)` 88 `ReasoningDelta`s and no `<think>` in the text), §28.11 structured (`Weather{city, sky: enum, temp_c}` greedy under grammar, one-shot and on a turn), cancellation (cancel after the first delta → `Finished(Cancelled)`, partial recorded, `remove_message(id)` ok, next turn runs), §18 dynamic system prompt (revision +1, no message moved; "Hello! How can I assist you today?" → "BANANA") · ⬜ `ToolChoice::Required`/`Tool` are enforced by prompt on local backends (no grammar-forced call; the finish reason says honestly whether a call came back) · ⬜ `GenerationOptions` has no stop-sequence field (`GenSpec` has no per-turn slot; engine `Settings.stopping` applies) and no `.advanced(..)` grouping — `gen2::advanced`/`GenSpec` remain the escape hatch · ⬜ remote: the external-api puller parses no `tool_calls`/`reasoning_content` deltas and puts no `tools`/`response_format` on the wire (inherited from S2.1), so remote `ToolCall` events and grammar never happen · ⬜ `FinishReason::ContentFilter`/`Error` still never produced; `Length` still inferred from `decode_tokens >= max_tokens` · ⬜ an assistant turn that both spoke and called tools is recorded as its calls only (`Message::assistant_tool_calls` has no text/reasoning slot; §9 `enum Message` is S2.6) · ⬜ the scripted backend cannot interrupt a pull, so a cancel lands one token late there (asserted `starts_with`); llama.cpp stops at once (live) · ⬜ the controller's ContinueChat-after-eviction rebuild passes `tools: None`/default thinking (pre-existing); the turn closes the session after a cancel so the next turn is a fresh start with tools, but an idle eviction between ordinary turns still hits it · ⬜ root still exports the old `Turn`/`Event`/`Canceller`/`ToolSet` names, so the new ones live at `gen2::turn::Turn`, `gen2::event::{Event, Canceller}`, `gen2::tool_defs::ToolSet` (S2.6) · ⬜ §13.1 `extract`/`classify` conveniences not re-based onto `Turn::structured` (still `Engine::extract`/`classify` over `Inference`) · ⬜ `Turn` has no `user_with_images` shorthand beyond `image()`/`input(Input)`; audio absent · ⬜ async `run_async`/`stream_async` → S2.5
- 2026-09-06 S2.2 branch s2.2-session · `Session` rewritten in place (src/api/session.rs): `SessionId`/`SessionRevision`/`MessageId` newtypes; `system: Option<String>` and `tools: ToolSet` first-order (`set_system`/`append_system`/`clear_system`, `set_tools`/`add_tool`/`remove_tool`, builders `with_system`/`with_tools`); append-only `events()` (`SessionEvent` §8 + `MessageRecorded` for records a `ContextReplaced` names, + `Forked {parent, at}` = provenance: a fork carries the parent's whole log and ids verbatim); immutable records with `all_messages() -> Vec<MessageRecord{id,message,active,replaced_by}>`; `messages()` is a cached projection; ops `push`/`push_user`/`push_user_with_images`/`push_tool_result` → `MessageId`, `replace_message`, `remove_message`, `replace_messages`, `restore_context`, `fork`, `fork_at`, `revision()`; `edit`/`clear`/`rollback_to` are `ContextReplaced` events over new records (never history loss; `clear` keeps system+tools); serde = `{id, events}` with replay (old flat `{id, messages}` still reads); the system prompt rejoins the message list only in crate-internal `Session::transcript()` (index 0) so the rendered prompt is byte-identical — proved by `entrypoint_tests::the_system_prompt_is_set_once_and_not_repeated_per_turn` (backend sees it exactly once) and live `a_named_chat_continues_across_turns`/`the_caller_owns_the_transcript`; new `gen2::tool_defs::{ToolDefinition, ToolSet}` (data-only, insertion-ordered, in-place replace, FNV fingerprint, ⇄ wire `ToolSpec`, `From<api::tools::ToolSpec>`), `gen2::session::{SessionId, SessionRevision, MessageId, SessionEvent, MessageRecord, ToolResult, SessionError}`, `Error::Session`; a `Chat` naming no tools offers `session.tools()` (fake-backend test); `Message::user_parts(InputPart…)`; tool-round invariant re-implemented at the mutation boundary (`check_rounds_over`, mirrors `journal::round_len`: an unanswered-anywhere call stands alone) — `remove_message`/`replace_messages`/`restore_context`/`fork_at` refuse a split (Err, session untouched), `edit` trims to a round boundary · fixes found on the way: `FunctionDefinition.arguments` gained a layer of escaping per serialize/deserialize cycle (now a fixed point, string-encoded objects read back as objects); `Chat::on_tool` results now carry their call id · gate: cargo test 1094/0/16 ignored default (+18 fit, 24 external, 60 doc); ext-api lane 1156/0/2 (+62 doc with tokio); clippy -D warnings default + ext-api,tokio; doc -D warnings default + docs.rs feature set; rustfmt; live 24/24 Metal on Qwen3-0.6B (14.3s); 37 session unit tests incl. §28.8/§28.9 walkthroughs and 8 round-invariant tests; 9 proptests over 21 op kinds (log prefix-immutable, records ⊇ projection, revision == context-affecting events and moves iff context changed, refusals change nothing, round invariant on every projection, replay identity) — the proptests caught 2 replay bugs (`replace_messages` records not in the log; `fork_at` not replayed) before commit · ⬜ `Message` stays the wire type (role string); the §9 `enum Message` is S2.6 · ⬜ `ToolChoice`/`ToolCall` on messages, `Model::turn` → S2.3 · ⬜ `src/journal` not reused: its unit is `JournalEntry`/`EntryId` and `Turn` groups by construction over a `Vec<Record>`, while a session addresses messages by id and needs a check at mutation time; same invariant, second implementation (unify when `Message` becomes the enum) · ⬜ an id-less (legacy) tool result is attributed by position and may be removed alone · ⬜ `set_system`/`set_tools` with an identical value is a no-op (no event, no bump) — a deliberate reading of "changing" · ⬜ `edit` cannot refuse (returns `()`), so it trims rather than errs · ⬜ session tools are rendered through the existing `Chat::tools` path with the one-shot `TOOL_PROMPT`; an agent run's registry still overrides them · ⬜ the `arguments` deserialize change (string → object) is proven by unit + mockito tests only, not against a live remote provider · ⬜ `all_messages()` allocates per call
- 2026-09-06 S3.4 (partial, this commit) · `windows` CI lane added (windows-latest, default features, `cargo test --lib`, advisory until green once) · fresh-clone job waits for S3.3's `hello` example · ⬜ Windows result unknown until the run
- 2026-09-06 S3.4 note · Windows lane: compiles and 1105 tests pass after the unix-gated test; 3 `mcp::tests::client` tests time out on the handshake with the python3 mock (mock uses only readline/stdout; no signals) · diagnostics step added to the lane · ⬜ cause unknown without a Windows machine (RTX box unreachable)
- 2026-09-06 README rewritten around the spec's API at Victor's request (opening paragraph included, which lifts that ⛔ for this edit): load/generate, Runtime+Session turn, tools loop, streaming+cancel, typed output, lossless sessions, hosted models; old facade and advanced layers in one section · 55 doc tests green · ⬜ async section still shows the old surface until S2.5 lands; model switching section waits for S2.4
- 2026-09-06 S3.1 branch s3.1-hf · `hf:owner/repo[:QUANT|:file.gguf]` accepted by `gen2::load`, `Runtime::load`, `Engine::load`/`EngineBuilder::model` (llama.cpp/ollama grammar; `hf://`, `hf.co/`, `https://huggingface.co/owner/repo[/resolve|blob/main/file.gguf]` as courtesy); new module `src/api/hf.rs` = `gen2::hf` (`HfModel { repo, quant, file }` + `.quant/.file/.mmproj/.without_mmproj/.cache_dir/.token/.on_progress(FnMut(Progress{file,bytes,total}))`, `resolve() -> HfResolved`, `download() -> HfDownload{model, mmproj, cache_hit}`, `HfError` (non_exhaustive, each variant says the fix) → `Error::Load`, `cache_dir()`, `cached_gguf_files()`, `choose_gguf()`/`choose_mmproj()`/`quant_of()` public and offline-testable); typed entry = `Engine::builder().hf(HfModel)` + `Runtime::load_hf(HfModel)`; `ModelSourceKind::HuggingFace { repo, file }` (enum drops `Copy`) · selection: file tagged Q4_K_M → smallest Q4* → smallest GGUF; `:TAG` case-insensitive whole-segment match else `QuantNotFound` listing the repo's tags; `mmproj*.gguf` and shards past 00001 never candidates; repo's `mmproj*.gguf` (F16 preferred) downloaded and wired as `EngineBuilder::mmproj` · cache: HF hub layout via `hf-hub = "1"` (`blocking`; verified it honours HF_TOKEN/HF_HUB_CACHE/HF_HOME/HF_ENDPOINT — the receipt's README/docs.rs contradiction resolves to docs.rs), dir = `GEN2_MODELS_DIR` → `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` → `dirs::cache_dir()/gen2/hf`, always absolute; cache-first resolution (exact tag/file only, never a fallback tier) then `local_files_only` per file, so a warm cache makes zero requests; `HF_TOKEN` passed explicitly · found on the way: hf-hub returns the tree listing's 404/401/403 as a generic `HFError::Http` (with `x-error-code`), mapped by code/status · gate: cargo test 1153/0/16 ignored default (+18 fit, 33 live-skip, 1 hf-live-skip, 58 doc; 22 new `api::hf::tests` — every grammar form, 16 rejected forms with their reason, selection order over a fake listing, tag extraction, cache-dir order over an explicit env, HF-layout cache scan through symlinks, warm-cache resolve/download with no client, cache-never-answers-a-fallback, projector ride-along, every `HfError` text + `Error::Load` mapping, plain-path-never-resolves), ext-api lane (hf off) 1216/0/2 incl. `FeatureDisabled` path; clippy --all-targets -D warnings default + `--no-default-features --features backend-external-api`; rustdoc -D warnings default + docs.rs set (`hf` added); rustfmt; `cargo publish --dry-run --allow-dirty` EXIT 0 (275 files, 6.3 MiB; hf-hub 1.0.0 from crates.io) · **LIVE (`GEN2_TEST_HF=1`, metal, temp `GEN2_MODELS_DIR`)**: `:Q8_0` → `Qwen3-0.6B-Q8_0.gguf`, `:Qwen3-0.6B-Q5_K_M.gguf` exact, `:Q9_9` → QuantNotFound listing 26 tags, missing repo → RepoNotFound; `Runtime::load_hf(hf:unsloth/Qwen3-0.6B-GGUF)` downloaded `Qwen3-0.6B-Q4_K_M.gguf` 396,705,472 bytes, **cold download+load 44.4 s** (59.0 s on the first run), 857 progress events with the exact total, `info().source == HuggingFace{repo,file}`, `generate("Reply with exactly one word: hello").max_tokens(8)` → "hello"; then `HF_ENDPOINT` pointed at a closed port: typed `download()` → `cache_hit`, hook fired 0×, `gen2::load` warm 0.2 s and generated, `load_hf` from cache, uncached `:Q6_K` → `Network` error not a hang · cold-build delta (M4 Pro, scratch target, `backend-external-api` vs `+hf`): 25 s → 46 s (+21 s, +62 rlibs, +0.6 GB) · ⬜ **that delta exceeds the receipt's 15 s guard** — the receipt's fallback is the in-house reqwest resolver (manifest + resolve endpoints); left as the owner's call since it trades hf-hub's cache/xet/retry for ~150 LOC · ⬜ selection is client-side over the tree listing rather than the `v2/…/manifests/<tag>` endpoint (hf-hub exposes no raw GET; same rules as llama.cpp's server-side match, but a repo the manifest would tag differently is not proven) · ⬜ HF rate limits under CI not observed; CI does not yet set `HF_TOKEN` or cache the models dir (S3.4's fresh-clone job) · ⬜ gated-repo path unit-tested via the error mapping only, not run against a real gated repo · ⬜ mmproj wired but not exercised live (Qwen3-0.6B has none); no vision repo tested · ⬜ sharded repos: the first shard is chosen but later shards are not downloaded · ⬜ `RateLimited.retry_after` is `None` for the listing path (hf-hub parses it only for mapped variants) · ⬜ `HfModel` holds an `Arc<Mutex<dyn FnMut>>` hook, so clones share one callback · ⬜ CI confirmation
- 2026-09-06 S2.6 branch s2.6-namespace (rebased onto S3.1 0bd84ab; conflicts in Cargo.toml features/docs.rs list, lib.rs module block, `EngineBuilder::build` "nothing to load" guard, CHANGELOG — `gen2::hf` kept at the root as S3.1 left it) · **root namespace = api_spec.md §25**: `gen2::{Runtime, Model, Session, Message, ToolDefinition, ToolSet, ToolChoice, GenerationOptions, Response, Event, Error, Result, load}` plus `Turn`, `Input`, `EventStream`, `AsyncEventStream` (tokio), `schemars`, and S3.1's `hf`; supporting modules `gen2::{model, session, input, output, event, tool_defs}` (`gen2::turn` gone; `model` also homes `Generation`, `RemoteModelBuilder`, `Embedder`, `Reranker`, `ThinkingMode`, `RerankResult`, `ModelSourceKind`); `gen2::api` and `gen2::controller` are `pub(crate)` · **`gen2::legacy`** (src/legacy.rs): `Engine`, `EngineBuilder`, `Chat`, `OwnedChat`, `Inference`, `Classify`, `Extract`, `Completion`, `TokenStream`, `Tokens`, token `Event`, `Finish`, `Budget`, `Struggle`, spawned `Turn`/`Canceller`/`Update`, `DEFAULT_TOOL_DEPTH`, `AsyncTurn`, fit `ModelInfo`/`Fit`/`FitVerdict` as `#[deprecated(since = "0.1.0")]` type aliases with the §27 replacement in every note (verified: `use gen2::legacy::Engine` warns in a consumer; `-D warnings` turns it into an error, so the live/external test files and the parity bench carry a file-level `#![allow(deprecated)]`) — mechanism chosen because `#[deprecated]` on a `pub use` does nothing in rustc 1.98 and deprecating the structs would need allows across the new facade that is built on them; the 18 aliased structs get `#[allow(unnameable_types)]` since the lint cannot see through an alias · **`gen2::agent`** (src/agent.rs, feature `agent`, default on, in the docs.rs list): `Agent` (+ new `Agent::on(&Model, &mut Session)`), `AgentConfig`, `AgentRun`, `AgentStep`, `ApprovalMode`, `Decision`, `Risk`, `Steering`, `OwnedAgent`, `AsyncAgentRun`, `DEFAULT_MAX_STEPS`, `SEARCH_TOOL`, executable `ToolSet`/`Tool`/`FunctionTool`/`AgentTool`/`Skill`/`SkillLibrary`/`ToolRegistry`/`ToolSearch`/`ToolSpec`/`ToolLoading`/`ExecutionPolicy`/`ToolContext`/`ToolOutput`/`ToolError`/`ToolConfigError`/`IntoTool`, un-deprecated `Completion`/`Finish`/`Budget`/`Struggle`/`Update`, `agent::mcp` (was `gen2::mcp`), `agent::journal` (was `gen2::journal`); with the feature off `src/api/{agent,agent_config,agent_spawned}.rs`, `api/tools/`, `mcp/`, `Error::Tools`, `Update::ToolResult`, `ToolResult: From<ToolOutput>`, `ToolDefinition: From<tools::ToolSpec>`, `Engine::agent`/`agent_owned` and the agent tests do not exist · **`gen2::advanced`**: `generation` (`GenSpec`, `ThinkingMode`, `Settings`+parts, `GrammarSpec`, `SpeculativeMode`/`Predictor`, `Capabilities`, `Degraded`, `LoadOutcome`, `BackendCaps`, `LatencyTier`), `runtime` (+`RuntimeBuilder`, memory governor, residency inventory/policy/stats), `fit`, `wire` (`MessageBody`/`Chunk`/`Content`, `FunctionDefinition`, `ToolSpec`, `ToolCall`, `Url`, `ModelRecord`, `ModelConfig`, `ModelMetadata`), `controller` (all of the old `gen2::controller` + `ExecError`, `ExecutionStats`, `MediaBoundary`, stream `ToolCall`), `utilities`, `plugin`; `RuntimeBuilder::backend(plugin)` (mlxcel docs/live test moved to it) · §21: `src/api/embed.rs` — `Runtime::load_embedder`/`load_reranker` (reranker-only engines now allowed) · `src/api/spec_walkthrough_tests.rs`: §28.1–§28.12 verbatim as `no_run` doc tests (12) · README: fit section, embeddings section, "The root, and what lives below it", features table, examples list, hf sentence; CHANGELOG "Changed"; examples `minimal`/`basic`/`structured`/`chat_app`/`embeddings`/`fit`/`tools`/`async_chat` rewritten to the new API, `agent`/`coding_agent`/`continuity` on `gen2::agent` with `required-features` · gate: cargo test 1154/0/16 ignored default (+18 fit, 73 doc, 33 live-skip), `--features tokio` 1163/0/16 (+79 doc), `--no-default-features --features backend-external-api,agent` 1217/0/2 (+4/4/18/6, 73 doc), agent-off `backend-external-api` check green and test 1062/0/2 (+65 doc); clippy `--all-targets -D warnings` default / tokio / agent-off ext-api all clean; rustdoc `-D warnings` default + docs.rs set (`…,tokio,hf,agent`); rustfmt; `cargo build --examples` (+tokio); `cargo publish --dry-run --allow-dirty` EXIT 0 (279 files, 6.3 MiB); `cargo check -p gen2-mlxcel --tests` green; **live 35/35 Metal** (28.7 s; Qwen3-0.6B + Llama-3.2-3B + Qwen3-Embedding), examples run live: `embeddings` on `Runtime::load_embedder` (768-d, cat/feline 0.741 vs cat/rust 0.225), `tools` (get_weather → "clear with a temperature of 18°C", 4-message transcript), `minimal`; pio-app path check compiles: `gen2::Message` → root, `gen2::controller::*` → `gen2::advanced::controller::*`, `gen2::GenSpec`/`Settings`/`ThinkingMode` → `gen2::advanced::generation`, `gen2::ExecError` → `gen2::advanced::controller::ExecError`, `SessionSpec` → `gen2::advanced::plugin::SessionSpec`, `MessageBody`… → `gen2::advanced::wire`, `Engine`/`Chat` → `gen2::legacy`, agent/MCP/journal → `gen2::agent` · ⬜ **§24 error model not reshaped**: `gen2::Error` keeps its variants (pio-app matches on them; the spec's categories carry different payloads); mapping for the future reshape: `ControllerGone`/`ModelNotLoaded` → Runtime; `Load`/`WontFit` → Load; `Unsupported` → Unsupported; `Tools`/`InvalidRequest` → InvalidInput; `Extraction` → Generation; `Generation{code}` → Context when `code` starts with `context`, else Generation (no Remote category exists today — external-api failures arrive as string codes under Generation/Load); `Session` → Session; `Exec(ExecError)`: `ModelNotLoaded`/`EmbedderNotLoaded` → Runtime, `FeatureUnsupported`/`UnsupportedArchitecture` → Unsupported, `InvalidArg` → InvalidInput, `InvalidModelFile`/`MmprojIncompatible`/`SettingsError`/`Io` → Load, `ContextOverflow` → Context, rest → Generation; cancellation stays a finish reason · ⬜ **§9 `enum Message` not this slice**: `Message` stays the role-based wire struct the backends render (pio-app: 64 `gen2::Message` uses); its parts are `advanced::wire`; `Message::tool_result`/`tool_result_for` already fit `push_tool_result` · ⬜ ⛔ **agent layer: kept as a default-on feature, not deleted — Victor's call** (spec §26 allows either); owned/spawned agents (`OwnedAgent`, `AgentConfig::agent_owned`, `examples/continuity.rs`) still take an `Arc<legacy::Engine>`, only the borrowed `Agent::on(&Model)` is engine-free · ⬜ `ThinkingMode` lives at `gen2::model::ThinkingMode` and keeps its name (spec §12 says `ReasoningMode`); `Generation`/`RemoteModelBuilder`/`Embedder`/`Reranker` in `gen2::model` and `RuntimeBuilder` in `advanced::runtime` are homes §25 does not list · ⬜ `Embedder`/`Reranker` are separate engines outside the residency ledger (no evict/preload; §21 "may participate"), and `load_embedder` has no family override (`legacy::EngineBuilder::embedder_kind`) · ⬜ the new surface has no explicit context-window knob (`legacy::EngineBuilder::context`): `RuntimeBuilder::settings(..).system.ctx_size` is honoured only where the fit preflight cannot read the file (LiteRT-LM, safetensors); a GGUF is always auto-sized · ⬜ `benches/backend_parity.rs` still measures `legacy::Engine::infer` (the committed results' path); not re-measured on `Model::generate` · ⬜ CI's backend-matrix lanes (`--no-default-features --features <backend>`, incl. `metal,tokio`) are now agent-off and hf-off, so the agent live tests run only under default features locally; only the `tokio` lane got `agent` added · ⬜ the `tests/live_inference.rs` first half, `tests/external_*.rs`, and the parity bench run on `legacy` under a file-level `#![allow(deprecated)]` until S5 retires the facade · ⬜ rustdoc for `legacy::*` is the alias pages (the structs are unreachable by path); verify docs.rs renders the aliased type's methods · ⬜ `gen2::hf` (S3.1) is a root module §25 does not list · ⬜ CI confirmation
- 2026-09-06 S3.4 (Windows half, this commit) · Windows lane green on c820cbf: 1108 unit tests; MCP timeouts were spawn contention (serialised run passed), handshake bound widened to 8 s on Windows; lane is now a hard gate · fresh-clone job still waits for S3.3
- 2026-09-10 ↷ detour · S2.6 broke two lanes its own gate never ran: `NoArgs` in the live tests is dead code without the `agent` feature (every backend matrix lane), and the separate `fuzz` workspace still imported `gen2::ModelInfo`/`gen2::ChannelMarkers` — the reply-parts scanner had lost every public path, now re-exported at `gen2::advanced::generation` · lesson: the gate must include `--no-default-features --features backend-llamacpp` and a `fuzz` check
- 2026-09-10 S3.2+S3.3+S3.4 branch s3.first-run (merged main 05cb1f1 mid-slice for the two lanes S2.6 broke) · **S3.2**: Metal needs no feature on Apple silicon — `llama-cpp-2 0.1.156`'s manifest has `[target.'cfg(all(target_os = "macos", any(target_arch = "aarch64", target_arch = "arm64")))'.dependencies.llama-cpp-sys-2] features = ["metal"]` (line 104-106), the sys crate's `metal = []` is a no-op marker its `build.rs` never reads, and ggml's own CMake defaults `GGML_METAL_DEFAULT ON` on Apple (`ggml/CMakeLists.txt:96,238`), so the crate's `metal` feature is an alias; new **`GpuOffload`** (`gen2::advanced::generation::GpuOffload`: `layers`/`total`/`backend`/`device`, `on_gpu()`, `Display`) computed the way llama.cpp computes its own `offloaded N/M layers to GPU` line (`min(n_gpu_layers, n_layer+1)` over `n_layer+1`, gated on `llama_supports_gpu_offload()`, device named from `list_llama_ggml_backend_devices()`) — the public C API exposes no per-tensor placement, so this mirrors the loader's inputs rather than parsing a log; surfaced as `Backend::gpu_offload` → `ControllerRuntimeSnapshot::loaded_model_offload` → **`Model::info().offload`**; `HardwareProfile::detect()` now returns `GpuBackend::Metal` for any llama.cpp build on macOS/aarch64 (it used to wait for the `metal` feature the binding had already turned on, so the profile disagreed with reality on a default build) · **S3.3**: `examples/hello.rs`, zero-arg, `gen2::load("hf:unsloth/Qwen3-0.6B-GGUF")` + `ThinkingMode::Off` + `max_tokens(128)`, prints the offload line, the answer, and TTFT/tok/s; an optional first argument overrides the reference; `examples/minimal.rs` returns a usage error instead of `.expect()`; every example's header lost its `--features metal`; README: per-OS prerequisites (macOS Xcode CLT + `brew install cmake`; Ubuntu/Debian `build-essential cmake libclang-dev`; Windows VS Build Tools + `winget install Kitware.CMake LLVM.LLVM` + `LIBCLANG_PATH` — matching what the CI lanes install), sccache hint (`CMAKE_C_COMPILER_LAUNCHER`/`CMAKE_CXX_COMPILER_LAUNCHER`, verified against `llama-cpp-sys-2/build.rs:666-671` which forwards every `CMAKE_*` env var into the cmake configure step), "The smallest useful thing" is now `cargo run --example hello` then the two-line `hf:` program, Qwen3-0.6B named as the smoke model with Qwen3-1.7B as the first size worth building on, Backends row says `metal` is an alias and `cuda`/`vulkan` are opt-in, live-test block drops `--features metal` · **S3.4 (fresh-clone half)**: `first-run` job on `ubuntu-latest` + `macos-latest`, no `Swatinem/rust-cache` (the first build's cost is part of the claim), README prerequisites installed verbatim, `actions/cache` on `GEN2_MODELS_DIR=${{ github.workspace }}/.gen2-models` keyed `gen2-models-hf-unsloth-Qwen3-0.6B-GGUF-Q4_K_M`, `cargo run --example hello` with default features and one retry after 30 s (the download is the flaky part; the retry logs a `::warning::`), then two greps: `tok/s` everywhere and `layers on the GPU \(MTL` on macOS — no `continue-on-error` · **LIVE (this Mac, M4 Pro)**: `cargo run --example hello` with a fresh `GEN2_MODELS_DIR`: **cold 35.3 s** (397 MB download + build-cached load + 54 tokens, TTFT 35 ms, 223.9 tok/s), **warm 2.1 s** (TTFT 37 ms, 234.2 tok/s), cache 378 MB · gate: cargo test 1154/0/16 ignored default (+18 fit, 34 live-skip, 1 hf-live-skip, 73 doc), `cargo test --doc` 73/0/3 ignored, clippy `--all-targets -D warnings` default + `--no-default-features --features backend-external-api` + `--no-default-features --features backend-llamacpp`, rustdoc `-D warnings` default + docs.rs set, rustfmt, `cargo build --examples`, `cargo publish --dry-run --allow-dirty` EXIT 0, `cd fuzz && cargo +nightly check`, ci.yml parses as YAML and no `run:` line ends in a colon, **live 36/36 Metal** with `--features metal,tokio` (the 35 that S2.6 left plus the new one) and `metal_is_on_by_default_on_apple_silicon` green on its own under DEFAULT features · ⬜ **the `first-run` job is unobserved until the push** — cache-key behaviour, the retry path, HF rate limits from a runner IP, and the real job duration (< 15 min is an estimate from this Mac's 35 s download plus a cold llama.cpp build, not a measurement) · ⬜ **no Windows first-run**: the `windows` lane still only runs `cargo test --lib`, so the README's Windows prerequisite line (VS Build Tools + LLVM + `LIBCLANG_PATH`) is sourced from bindgen's requirements page and the runner's preinstalled LLVM, never from a green `hello` on Windows · ⬜ `GpuOffload.layers` is what the loader asked for clamped to the model's layers, not a read-back of where tensors actually landed (llama.cpp exposes none); a partial offload forced by VRAM pressure inside ggml would still report the requested count · ⬜ the backend string is ggml's registry name, so Metal reads `"MTL"` (`GGML_METAL_NAME`), not `"Metal"` — the README, the example comment and the CI grep all say MTL · ⬜ only the llama.cpp backend implements `gpu_offload`; MLX, LiteRT-LM, mistral.rs and remote models report `None` · **found on the way**: `main` at 05cb1f1 does not pass `cargo doc -D warnings` — making the reply scanner public at `gen2::advanced::generation` turned four of `src/generation/reply_parts.rs`'s doc links into errors (three to the private `zoo::ModelFamily::channel_markers`, one to a nonexistent `ReplyShape`); 05cb1f1 plainified one of them and stopped. Fixed here as a doc-comment-only change, since the slice's own gate requires that lane · ⬜ the Linux `first-run` runner has no GPU, so its `hello` proves the download+build+decode path only · ⬜ CI confirmation
- 2026-09-10 S5.1 branch s5.1-compat (merged main 5d25291 first — the scanner is public at `advanced::generation` since, so `compat::generation` mirrors rather than widens it) · **`#[doc(hidden)] pub mod compat`** (src/compat.rs, 1 delimited `pub mod compat;` line in lib.rs): the module tree pio-app addresses, re-exported at the old relative positions, so the host's shim is `pub mod gen2 { pub use gen2_crate::compat::*; }` and its `crate::gen2::…`/`pio_core::gen2::…` paths stay untouched · **inventory measured, not estimated** (parser over `pio-core/src`, `pio-core/tests`, `src-tauri`, `pio-daemon`, `pio-bridge`, `pio-embed-seed` + 4 satellites, excluding the in-tree copy): **132 files, 281 `use` statements → 445 leaves (77 distinct), 283 inline mentions (83 distinct)**; receipt 05's "454 leaves / 715 inline" counted the in-tree copy's own self-references, which vanish with it; 1 leaf + 8 inline were false positives (pio-core's own `extract_gen2::{FactMemoryPipeline, LAST_K}`, `reflection_gen2::Gen2ReflectionModel`) · **resolve: 87 of 95 distinct paths = 437/445 use leaves (98.2 %) + 268/283 inline (94.7 %)**; **cannot: 8 paths / 15 mentions** — mlxcel ×4 paths/9 mentions (companion crate `gen2-mlxcel`; its `profile`/`ProfileMode`/`ProfileRun` do not exist there, `METALLIB_ENV` is private), `project_streaming_inference` (3, moves host-side), `InferenceHandle::Flock` (1, → `RemoteDispatch`), `zoo::tests::*` (2, crate-private unit tests named in doc comments) · classification: 1 path same (`gen2::Message`), 38 renamed by S2.6-or-since (327 leaves), 45 were `pub(crate)` (71 leaves), 3 are `Engine`-changed-meaning (11 leaves) — **every one of the 40 `pub(crate)` paths receipt 05 §2a listed is `pub` at its definition already**, so compat is re-exports only: no module was widened, and the only visibility changes are `external_api::puller` → `pub(crate)`, `LiteRtLmEngine`/`MistralRsEngine` → `pub` (all three so `unnameable_types` can see what the facade's enums wrap) · **`SystemTask`**: 29 host mentions / 11 files of the 9 removed variants (`Answer` 21, `TopicLabel` 2, `ContextualPrefix` 2, `Triples`/`QueryRewrite`/`EntityExtract`/`Contradiction` 1 each, `Stance`/`QueryUnderstand` 0) → `compat::system_task::{answer,triples,stance,entity_extract,topic_label,query_understand,contradiction,query_rewrite,contextual_prefix}()` + `gen_spec(&task)` reproducing the old per-task `GenSpec` table (unit-tested against all 9, and against the crate's own for the 4 kept variants + unknown labels); **not faked**: `system_infer`/`system_prompt` look the spec up by variant, so 11 of the 23 code sites must become `system_infer_with(.., gen_spec(&task))` (12 already pass their own spec); session ids lose the `sabra-` prefix (nothing parses it) and `SystemTask` is no longer `Copy` · gate: `cargo test` **1284/0/20 ignored** (incl. 3 new `compat::system_task` tests + `compat_pio_paths`); clippy `--all-targets -D warnings` green on **6 lanes** — default, `--no-default-features --features backend-llamacpp` (the S2.6-missed lane), `backend-external-api,agent`, `backend-external-api,tokio,agent`, `specta,tokio`, `backend-litertlm`, no-features, `backend-mistralrs`; rustdoc `-D warnings` default + docs.rs set (10 doc links plainified — compat made those modules publicly reachable, which is what surfaced them); rustfmt; `cd fuzz && cargo +nightly check` EXIT 0 (the other S2.6-missed lane); `cargo publish --dry-run --allow-dirty` EXIT 0 (282 files, 6.4 MiB); `cargo check --test compat_pio_paths` green under default / `backend-external-api,agent` / `specta,tokio` · **feature dependence**: only the 3 `backend::llama::*` paths (4 use + 6 inline) need a feature (`backend-llamacpp`); the other 84 are unconditional — `agent` is *not* needed by any compat path (`executor` is unconditional), and `--features specta` compiles, so the derives survive · ⬜ `backend-mlx` could not be verified: that lane does not compile on `main` (`src/backend/mlx/session.rs:421` missing `Message.tool_call_id`), pre-existing and independent of this slice; `compat::backend::mlx` is written but unbuilt · ⬜ **no compile was run against pio-app** — this slice is read-only on that checkout, so "437/445 resolve" is a path-resolution proof through `tests/compat_pio_paths.rs`, not a `cargo check -p pio-core --features gen2-crate`; method/field-level drift inside the 79 modified modules is still only findable by S5.2's build (receipt 05 §8 stands) · ⬜ the residual semantic list is S5.2's work: `LoadModel` constructors (27, `api_model` + `Result<LoadOutcome>`), `ContinueChat` constructors (15, `transcript`/`thinking`/`tools`), `ControllerEvent::Accepted` (27 files match, 1 exhaustive), `InferenceHandle` remote arms (34 sites) + `liveness()` (6) + `compute_provenance()` (3), `system_infer*` → `ExecError` (34), specta 45→71 types · ⬜ `compat` has no deprecation timer: it is `#[doc(hidden)]` and documented as shrinking, but nothing fails when a path stays · ⬜ CI confirmation
- 2026-09-10 ↷ detour · `backend-mlx` had not compiled since S2.2 added `Message.tool_call_id` (10 struct literals in `mlx/{engine,golden,session}.rs`); S5.1's compat re-export then surfaced 42 `unnameable_types` in the MLX model zoo (allowed on the three internal modules — layers and kernels are not API) and a missing `Default for Engine`. Lane green; `compat::backend::mlx` no longer claims the test-only `Session`/`TokenPuller`, which pio-app never imported
