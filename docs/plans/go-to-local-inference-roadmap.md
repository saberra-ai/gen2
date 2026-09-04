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
- Taste / references: the README's existing voice (plain sentences, every
  example doc-tested, costs stated honestly). Mirror: mistral.rs and
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
- Human-only forks: ⛔ crate name "gen2" (check availability; if taken the
  user picks); ⛔ tagline/positioning; ⛔ publish/tag; ⛔ any paid runner.
- Research questions: see "Research calls".

## Research calls

Receipts (graded, cited): `docs/plans/research/0[1-5]-*.md`. Calls taken:

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

- Budget: this session, at most 14 slices, ≤8 commits per slice, one slice in flight
  at a time; pre-flight disk (>40 GB free) before every build-heavy slice
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
Status: ⬜

#### S1.4 Release readiness
Outcome: everything but the publish button is done.
Observable: `[package.metadata.docs.rs]` present and `cargo doc` with those exact
features is warning-free; CI `publishable` is a hard gate; CHANGELOG.md exists;
README install snippet says `gen2 = "0.1"`.
Gate: dry-run exit 0 on `main`; CI green. Blocked by: S1.3. Forks: ⛔ publish + tag.
Status: ⬜

### Wave 2 — five-minute first run
#### S2.1 `hf:` model references
Outcome: `Engine::load("hf:unsloth/Qwen3-0.6B-GGUF")` downloads to a cache and loads.
Observable: a live test with a temp cache dir downloads Q4_K_M and generates a token;
`:Q8_0` and `:file.gguf` forms resolve; `HF_TOKEN` honoured; progress callback fires.
Reference to mirror: ollama/llama.cpp `hf:` grammar; `hf-hub` crate API; receipt 03.
Gate: unit tests for the reference parser (offline) + live download test.
Blocked by: S1.4. Forks: none. Honest ⬜: HF rate limits under CI not observed.
Status: ⬜

#### S2.2 Metal is the default on Apple Silicon
Outcome: nobody types `metal` to get the GPU.
Observable: a macOS CI job with default features loads a model and reports GPU offload.
Gate: that job green; README no longer says "add `metal`". Blocked by: S2.1.
Status: ⬜

#### S2.3 `hello` example, README first block, prerequisites
Outcome: the README's first screen is the whole five-minute path.
Observable: `examples/hello.rs` runs with no arguments; README has a per-OS
prerequisite line and an sccache hint; `examples/minimal.rs` no longer unwraps.
Gate: `cargo run --example hello` prints a token on this Mac; doc tests pass.
Forks: ⛔ opening paragraph/tagline untouched; any positioning sentence listed for
Victor in the ledger. Blocked by: S2.2.
Status: ⬜

#### S2.4 Windows lane + fresh-clone job
Outcome: CI proves the README on three OSes.
Observable: `windows-latest` compiles and runs unit tests with default features; a
`first-run` job on macOS and Linux does `cargo run --example hello` with the HF cache
restored by `actions/cache`. Gate: CI green. Blocked by: S2.3.
Honest ⬜: Windows GPU; MSVC first-build time not optimised.
Status: ⬜

### Wave 3 — honest benchmarks
#### S3.1 Benchmark harness and first results
Outcome: a benchmark table a skeptic can reproduce.
Observable: `benches/results/<machine>/<date>-<sha>.json` committed for this Mac,
produced by a harness that builds llama-bench from the vendored sha and asserts
`build_commit`; README table generated between markers by a `bench-table` bin.
Reference to mirror: `llama.cpp/tools/llama-bench`, `scripts/compare-llama-bench.py`,
mistral.rs release report shape. Gate: bin regenerates the table byte-identically.
Blocked by: S1.4. Honest ⬜: RTX 3080 and Pi 5 rows.
Status: ⬜

#### S3.2 Freshness gate
Outcome: the table cannot rot silently.
Observable: `bench-freshness` CI job fails on table drift, llama.cpp pin drift, or
age >90 days; the unsourced LiteRT-LM "1.7x" sentence is removed or sourced.
Gate: CI green. Blocked by: S3.1.
Status: ⬜

### Wave 4 — pio-app consumes the crate
#### S4.1 `compat` surface in gen2
Outcome: pio-app's 454 import leaves resolve against the crate.
Observable: `#[doc(hidden)] pub mod compat` re-exports the 40 `pub(crate)` paths listed
in receipt 05; `SystemTask` gap documented with the mapping pio-app must apply.
Gate: `cargo test`; a compile-only check in pio-app's worktree shows the error count
drop from ~204 to only `SystemTask`/`ControllerEvent` semantic sites. Blocked by: S1.4.
Status: ⬜

#### S4.2 pio-app `gen2-crate` feature (PR)
Outcome: pio-app builds against the standalone crate behind a feature.
Observable: worktree off origin/main, `gen2-crate` feature with path dep, shim
re-export, three `RemoteDispatch` impls, `HostInference` enum; gate per receipt 05
(filtered lib tests ×3, integration suites, live chat/tool/gemma4/flock tests).
Integration: PR, not merged by the agent. Blocked by: S4.1.
Honest ⬜: specta bindings drift only checked by diff.
Status: ⬜

#### S4.3 Flip default, delete in-tree copy (PR)
Outcome: pio-app has one gen2. Blocked by: S4.2 merged by Victor. Forks: ⛔ (merge).
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
- 2026-09-04 S1.2 worktree-agent-a735c99fdb336ec65 · onnx+candle removed (1862 lines deleted, 685 added, net −1177; 1,525 LOC of backend code), mistralrs forwards metal/cuda (cargo tree proof, no Metal build), README tiered · gate: cargo test 1016/0/16 ignored default + 1014/0/4 mistralrs lane (CPU), clippy -D warnings, doc -D warnings, rustfmt, check ext-api / litertlm / llamacpp+litertlm, grep clean · ⬜ CI confirmation
