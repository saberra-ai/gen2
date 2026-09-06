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
Status: ⬜

#### S2.5 Async surface
Outcome: spec §22: `run_async`, `stream_async` under the `tokio` feature with no
parallel object model. Gate: `cargo test --features tokio`; existing async lane.
Blocked by: S2.3. Status: ⬜

#### S2.6 Root namespace and retirement of the old facade
Outcome: spec §25–§27: root exports exactly the §25 list; `gen2::advanced` holds
local tuning, residency, hardware, grammar, and the backend plugin seam from S1.3;
the agent layer moves behind an `agent` feature as `gen2::agent` (default per the
human fork above, flagged); `Engine`/`Chat`/`Inference` removed or kept as
`#[deprecated]` shims for one release; README examples rewritten to the new API.
Forks: ⛔ README opening/tagline (present the new text to Victor); ⛔ delete vs
feature-gate the agent layer. Gate: `cargo test`; `cargo doc`; every README
example doc-tested. Blocked by: S2.5.
Status: ⬜

### Wave 3 — five-minute first run
#### S3.1 `hf:` model references
Outcome: `Engine::load("hf:unsloth/Qwen3-0.6B-GGUF")` downloads to a cache and loads.
Observable: a live test with a temp cache dir downloads Q4_K_M and generates a token;
`:Q8_0` and `:file.gguf` forms resolve; `HF_TOKEN` honoured; progress callback fires.
Reference to mirror: ollama/llama.cpp `hf:` grammar; `hf-hub` crate API; receipt 03.
Gate: unit tests for the reference parser (offline) + live download test.
Blocked by: S2.6. Forks: none. Honest ⬜: HF rate limits under CI not observed.
Status: ⬜

#### S3.2 Metal is the default on Apple Silicon
Outcome: nobody types `metal` to get the GPU.
Observable: a macOS CI job with default features loads a model and reports GPU offload.
Gate: that job green; README no longer says "add `metal`". Blocked by: S3.1.
Status: ⬜

#### S3.3 `hello` example, README first block, prerequisites
Outcome: the README's first screen is the whole five-minute path.
Observable: `examples/hello.rs` runs with no arguments; README has a per-OS
prerequisite line and an sccache hint; `examples/minimal.rs` no longer unwraps.
Gate: `cargo run --example hello` prints a token on this Mac; doc tests pass.
Forks: ⛔ opening paragraph/tagline untouched; any positioning sentence listed for
Victor in the ledger. Blocked by: S3.2.
Status: ⬜

#### S3.4 Windows lane + fresh-clone job
Outcome: CI proves the README on three OSes.
Observable: `windows-latest` compiles and runs unit tests with default features; a
`first-run` job on macOS and Linux does `cargo run --example hello` with the HF cache
restored by `actions/cache`. Gate: CI green. Blocked by: S3.3.
Honest ⬜: Windows GPU; MSVC first-build time not optimised.
Status: ⬜

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
Status: ⬜

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
- 2026-09-06 S2.3 branch s2.3-turn · `Model::turn(&mut session) -> Turn` (src/api/turn.rs): staging `user`/`input`/`image`/`message`/`messages`, committed only when execution begins (a text-only model refusing an image, or a `ToolChoice` naming an unoffered tool, leaves the session and its revision untouched — unit-tested); no-new-message turns (§11.2); per-turn `max_tokens/temperature/seed/top_p/top_k/min_p/greedy/reasoning/tool_choice/options`; `system_override`/`tools_override` per turn only (reopens the conversation for the turn and closes it after; session `system`/`tools`/revision unchanged, no `SystemSet`/`ToolsSet` event — unit-tested against what the scripted backend saw); `run() -> Response`, `stream() -> EventStream`, `structured::<T>()`; `gen2::GenerationOptions` (§12, `#[non_exhaustive]`, builder, `reasoning: ThinkingMode`, maps onto `GenSpec` over the engine's defaults; expert knobs stay in `gen2::advanced`); `gen2::ToolChoice {Auto, None, Required, Tool(name)}` (`None` offers nothing, `Tool` offers only that definition, `Required`/`Tool` add a prompt instruction); `output::ToolCall::{id,name,arguments,parse_arguments}`; ids minted `call_<message id>_<n>` from the session's event-log counter (unique per session, stable across replay; provider ids kept); the reply is appended via `Session::push` as `assistant_structured(text, reasoning)` or `assistant_tool_calls` under those ids, `Response::message_id()` names it; §28.5 loop verbatim over the scripted engine (`push_tool_result(call.id(), ..)` → `turn().run()` → text) · `gen2::event::{Event, EventStream, Canceller}` (§15/§16): `Event::{TextDelta, ReasoningDelta, ToolCallStart, ToolCallArgumentsDelta, ToolCallEnd, Usage, Finished}`; reasoning synthesised by `ReplyStateMachine` over the union of the crate's `ChannelMarkers` (Qwen3/DeepSeek `<think>`, Gemma 4 thought channel) with the scaffold's own whitespace held back so deltas never carry what the message will not (S2.1's `split_reasoning` string-splitter deleted; `run()` = `stream()?.finish()`, one code path, so the two cannot disagree); a complete backend tool call becomes Start/ArgumentsDelta/End; `Canceller` (`Clone + Send`, `cancel()` from another thread) ends the stream `Finished(Cancelled)`, `finish()` keeps the partial text, the partial assistant message is recorded and `remove_message(id)` works (§16.1 as a unit test and live); `Turn::canceller()` before `run()` makes the turn not run at all (nothing staged) · `Generation` now runs a `Turn` on an ephemeral `Session` (§6), gained `tool_choice`/`reasoning`/`options`/`structured::<T>()`, `tools()` takes anything convertible to `ToolDefinition`s · structured output: grammar = `schema_for!(T)` when `capabilities().structured_output`, else the schema is asked for in the prompt and the reply parsed (Markdown fence stripped); reasoning forced `Off` unless set; a non-decoding reply is `Error::Extraction` with the raw text · `Session` gained process-local `thinking` bookkeeping (`note_thinking`: a different policy reopens, since backends pin it at start), `transcript_with_system`, `next_message_id`; `Chat` gained `pub(crate)` `system_override`/`without_tools`/`begin`/`settle` · fixes found live on Qwen3-0.6B: (1) the llama session ignored `SessionSpec.thinking` (family default only) and rendered every continuation delta and overflow probe with `enable_thinking: None`, so a Qwen3 `ContinueChat` fell back to the template's default and *thought* (the §28.5 continuation spent its whole budget in `<think>` and answered nothing; the old `a_named_chat_continues_across_turns` transcript shows the same `<think>` on turn 2) — now `On`/`Off` are honoured, `Auto` = family default, and the flag is pinned on the session and passed to every render (`src/backend/llama/session.rs`, `engine.rs`); (2) the controller's `FinalStats` forwarder was registered once with the first turn's channel, so every continuation's stats went to a dropped receiver (`usage` zero, `reported() == false`, `Length` undetectable) — `ContinueChat` now re-points it (`src/controller/commands.rs`; asserted live on the tool-loop continuation) · gate: cargo test 1113/0/16 ignored default (+18 fit, 69 doc; 26 new turn/event/generation/response unit tests incl. staging/commit, no-new-message, overrides-not-persisted, event synthesis from scripted tokens, mid-stream cancel, structured parse/typed error); ext-api lane 1175/0/2 (+4 anthropic, 4 openai, 18 fit, remote_runtime 6/6 — new: a remote `turn().stream()` yields `TextDelta`×3 + `Finished(Stop)` from a mockito SSE and records the reply; remote `structured()` parses without a grammar and reports prose as `extraction_failed`); clippy --all-targets -D warnings default + `backend-external-api,tokio`; rustdoc -D warnings default + docs.rs set; rustfmt; live 29/29 Metal on Qwen3-0.6B (33.3s; 24 existing + 5 new): §28.5 tool loop (`get_weather` → `ToolCall{call_1_0, city: Paris}` → `push_tool_result` → no-user-message turn → "The weather in Paris is currently 18°C with clear skies.", stats reported on the continuation), §28.10 streaming (24 `TextDelta` + `Finished(Stop)`, `finish().text()` == the deltas; with `reasoning(On)` 88 `ReasoningDelta`s and no `<think>` in the text), §28.11 structured (`Weather{city, sky: enum, temp_c}` greedy under grammar, one-shot and on a turn), cancellation (cancel after the first delta → `Finished(Cancelled)`, partial recorded, `remove_message(id)` ok, next turn runs), §18 dynamic system prompt (revision +1, no message moved; "Hello! How can I assist you today?" → "BANANA") · ⬜ `ToolChoice::Required`/`Tool` are enforced by prompt on local backends (no grammar-forced call; the finish reason says honestly whether a call came back) · ⬜ `GenerationOptions` has no stop-sequence field (`GenSpec` has no per-turn slot; engine `Settings.stopping` applies) and no `.advanced(..)` grouping — `gen2::advanced`/`GenSpec` remain the escape hatch · ⬜ remote: the external-api puller parses no `tool_calls`/`reasoning_content` deltas and puts no `tools`/`response_format` on the wire (inherited from S2.1), so remote `ToolCall` events and grammar never happen · ⬜ `FinishReason::ContentFilter`/`Error` still never produced; `Length` still inferred from `decode_tokens >= max_tokens` · ⬜ an assistant turn that both spoke and called tools is recorded as its calls only (`Message::assistant_tool_calls` has no text/reasoning slot; §9 `enum Message` is S2.6) · ⬜ the scripted backend cannot interrupt a pull, so a cancel lands one token late there (asserted `starts_with`); llama.cpp stops at once (live) · ⬜ the controller's ContinueChat-after-eviction rebuild passes `tools: None`/default thinking (pre-existing); the turn closes the session after a cancel so the next turn is a fresh start with tools, but an idle eviction between ordinary turns still hits it · ⬜ root still exports the old `Turn`/`Event`/`Canceller`/`ToolSet` names, so the new ones live at `gen2::turn::Turn`, `gen2::event::{Event, Canceller}`, `gen2::tool_defs::ToolSet` (S2.6) · ⬜ §13.1 `extract`/`classify` conveniences not re-based onto `Turn::structured` (still `Engine::extract`/`classify` over `Inference`) · ⬜ `Turn` has no `user_with_images` shorthand beyond `image()`/`input(Input)`; audio absent · ⬜ async `run_async`/`stream_async` → S2.5
- 2026-09-06 S2.2 branch s2.2-session · `Session` rewritten in place (src/api/session.rs): `SessionId`/`SessionRevision`/`MessageId` newtypes; `system: Option<String>` and `tools: ToolSet` first-order (`set_system`/`append_system`/`clear_system`, `set_tools`/`add_tool`/`remove_tool`, builders `with_system`/`with_tools`); append-only `events()` (`SessionEvent` §8 + `MessageRecorded` for records a `ContextReplaced` names, + `Forked {parent, at}` = provenance: a fork carries the parent's whole log and ids verbatim); immutable records with `all_messages() -> Vec<MessageRecord{id,message,active,replaced_by}>`; `messages()` is a cached projection; ops `push`/`push_user`/`push_user_with_images`/`push_tool_result` → `MessageId`, `replace_message`, `remove_message`, `replace_messages`, `restore_context`, `fork`, `fork_at`, `revision()`; `edit`/`clear`/`rollback_to` are `ContextReplaced` events over new records (never history loss; `clear` keeps system+tools); serde = `{id, events}` with replay (old flat `{id, messages}` still reads); the system prompt rejoins the message list only in crate-internal `Session::transcript()` (index 0) so the rendered prompt is byte-identical — proved by `entrypoint_tests::the_system_prompt_is_set_once_and_not_repeated_per_turn` (backend sees it exactly once) and live `a_named_chat_continues_across_turns`/`the_caller_owns_the_transcript`; new `gen2::tool_defs::{ToolDefinition, ToolSet}` (data-only, insertion-ordered, in-place replace, FNV fingerprint, ⇄ wire `ToolSpec`, `From<api::tools::ToolSpec>`), `gen2::session::{SessionId, SessionRevision, MessageId, SessionEvent, MessageRecord, ToolResult, SessionError}`, `Error::Session`; a `Chat` naming no tools offers `session.tools()` (fake-backend test); `Message::user_parts(InputPart…)`; tool-round invariant re-implemented at the mutation boundary (`check_rounds_over`, mirrors `journal::round_len`: an unanswered-anywhere call stands alone) — `remove_message`/`replace_messages`/`restore_context`/`fork_at` refuse a split (Err, session untouched), `edit` trims to a round boundary · fixes found on the way: `FunctionDefinition.arguments` gained a layer of escaping per serialize/deserialize cycle (now a fixed point, string-encoded objects read back as objects); `Chat::on_tool` results now carry their call id · gate: cargo test 1094/0/16 ignored default (+18 fit, 24 external, 60 doc); ext-api lane 1156/0/2 (+62 doc with tokio); clippy -D warnings default + ext-api,tokio; doc -D warnings default + docs.rs feature set; rustfmt; live 24/24 Metal on Qwen3-0.6B (14.3s); 37 session unit tests incl. §28.8/§28.9 walkthroughs and 8 round-invariant tests; 9 proptests over 21 op kinds (log prefix-immutable, records ⊇ projection, revision == context-affecting events and moves iff context changed, refusals change nothing, round invariant on every projection, replay identity) — the proptests caught 2 replay bugs (`replace_messages` records not in the log; `fork_at` not replayed) before commit · ⬜ `Message` stays the wire type (role string); the §9 `enum Message` is S2.6 · ⬜ `ToolChoice`/`ToolCall` on messages, `Model::turn` → S2.3 · ⬜ `src/journal` not reused: its unit is `JournalEntry`/`EntryId` and `Turn` groups by construction over a `Vec<Record>`, while a session addresses messages by id and needs a check at mutation time; same invariant, second implementation (unify when `Message` becomes the enum) · ⬜ an id-less (legacy) tool result is attributed by position and may be removed alone · ⬜ `set_system`/`set_tools` with an identical value is a no-op (no event, no bump) — a deliberate reading of "changing" · ⬜ `edit` cannot refuse (returns `()`), so it trims rather than errs · ⬜ session tools are rendered through the existing `Chat::tools` path with the one-shot `TOOL_PROMPT`; an agent run's registry still overrides them · ⬜ the `arguments` deserialize change (string → object) is proven by unit + mockito tests only, not against a live remote provider · ⬜ `all_messages()` allocates per call
- 2026-09-06 S3.4 (partial, this commit) · `windows` CI lane added (windows-latest, default features, `cargo test --lib`, advisory until green once) · fresh-clone job waits for S3.3's `hello` example · ⬜ Windows result unknown until the run
- 2026-09-06 S3.4 note · Windows lane: compiles and 1105 tests pass after the unix-gated test; 3 `mcp::tests::client` tests time out on the handshake with the python3 mock (mock uses only readline/stdout; no signals) · diagnostics step added to the lane · ⬜ cause unknown without a Windows machine (RTX box unreachable)
- 2026-09-06 README rewritten around the spec's API at Victor's request (opening paragraph included, which lifts that ⛔ for this edit): load/generate, Runtime+Session turn, tools loop, streaming+cancel, typed output, lossless sessions, hosted models; old facade and advanced layers in one section · 55 doc tests green · ⬜ async section still shows the old surface until S2.5 lands; model switching section waits for S2.4
