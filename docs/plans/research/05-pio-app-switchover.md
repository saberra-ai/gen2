# 05 — pio-app switchover to the standalone `gen2` crate

**Lane:** user-context (no web; every claim below is from the two checkouts).
**Date:** 2026-09-03. **Read-only** on both repos.
**Repos:** `/Users/victor/workspace/gen2` @ `79da512` (239 commits, 79 since extraction) ·
`/Users/victor/workspace/pio-app` @ `64f220da` on `main` (231 commits since extraction commit `5367cf0f`).

## The fork

In what order, and behind what seams, should pio-app switch from its in-tree
`pio-core/src/gen2` to the standalone crate — path dependency first, versioned
dependency later — with the least risk?

## CALL (summary)

**Swap behind a shim, not a rewrite.** Keep `pio_core::gen2::…` as the path every
host file imports (715 `crate::gen2` + 188 `pio_core::gen2` occurrences across
133 files) and make that module a re-export of the extern crate behind a
`gen2-crate` cargo feature. This works because of three facts found in the
code:

1. **There is no merge risk.** pio-app's in-tree copy is byte-identical to the
   tree gen2 was split from (`diff -rq` against gen2 `5e5cadc^` = split tip
   `8ee96da` returns nothing; `git log 5367cf0f..HEAD -- pio-core/src/gen2` is
   empty; last in-tree gen2 commit is `a947439e`, 2026-07-29, *before* the
   split). All 100 differing entries are gen2-side progress. **Both-changed
   list: empty.**
2. **Native pins are identical** on both sides (llama-cpp-2/-sys-2
   `43733d1…` + `mtmd,llguidance`; mlx-rs `fac86ba0…`; mlxcel/-core
   `21306cf1…`; ort `=2.0.0-rc.12`; specta `=2.0.0-rc.24`; tokio `1.44.2`;
   tokenizers 0.23; llguidance 1.7; minijinja 2.12). So the in-tree copy and
   the crate can coexist in one build during the feature-flag window without a
   duplicate `links` error.
3. **The command surface is a strict superset.** gen2's `ControllerCmd` has all
   20 of pio-app's variants (+6 new: `GetCapabilities UnloadModel ReloadModel
   LoadReranker Rerank GetUtilityStatus`) with identical `StartChat` /
   `ContinueChat` / `SystemInfer` / `LoadModel` field lists; every host `match`
   on `ControllerCmd` has a wildcard arm (11 files checked). The flock
   projection can move host-side verbatim.

What is *not* free: gen2 made 12 modules `pub(crate)` and pio-app imports 40
distinct paths from them (121 broken leaf imports + 83 mechanical root
rewrites), `SystemTask` lost 8 variants (30 host sites), `ControllerEvent`
gained `Accepted`, and `InferenceHandle` lost three host-typed arms plus
`liveness()` / `compute_provenance()` / `project_streaming_inference`.

Ordered plan, blast-radius stops, test gate and re-check trigger are at the end.

---

## 1. Drift inventory

Method: `git archive 5e5cadc^` (gen2's pre-move split tip) → `diff -rq` against
`pio-app/pio-core/src/gen2` → **exit 0, no output**. So the classification is
determined entirely by gen2's history since `5e5cadc` (`git diff --name-status
5e5cadc..HEAD -- src`: 129 files, +31,589 / −1,117).

| Category | Count | Entries |
| --- | ---: | --- |
| gen2-only progress — modified | 79 | every `Files … differ` line of `diff -rq` (backend/* 47, controller/* 10, engine/* 4, generation/* 3, session_rt/* 5, kv/* 2, bundle/* 2, residency*/router/zoo/executor 6) |
| gen2-only progress — added | 16 | `src/{api,journal,mcp,memory,types,utilities,test_support}/`, `src/{hardware,task_util,lib}.rs`, `backend/{conformance.rs,litertlm/,mistralrs/}`, `backend/external_api/tests/`, `controller/tests/`, `session_rt/compaction.rs` |
| pio-app-only (not progress) | 5 | `mod.rs` (replaced by `lib.rs`), `README.md`, `plan.md`, `milestones.md` (docs → gen2/docs already), `backend/executorch/` (stub; gen2 deleted it, `D src/backend/executorch/mod.rs`) |
| pio-app-only progress (commits after `5367cf0f`) | **0** | — |
| **both changed** | **0** | — |

Working-tree check in pio-app: 2 dirty entries (`pio-core/proptest-regressions/runners/chat_runtime.txt`, untracked `docs/plans/flock-extraction-research.md`), neither under `gen2/`. Three stashes exist; none touched (read-only).

Consequence: no three-way merge is needed. The switchover is purely
"adapt the host to gen2's current API".

## 2. Import surface

**133 host files** contain `gen2::` (the prompt's "91" undercounts; the extra
are `pio-core/tests/*` live suites, pio-daemon, and the satellite Tauri
crates). By location:

| Location | Files |
| --- | ---: |
| `pio-core/src/app/**` | 32 |
| `pio-core/tests/*.rs` (integration/live) | 21 |
| `src-tauri/src/**` (+1 test) | 17 |
| `pio-daemon/src/**` (+9 tests) | 16 |
| `pio-core/src/p2p/**` | 11 |
| `pio-core/src/runners/**` | 7 |
| `pio-core/src/{loops,birds}` | 6 |
| `pio-core/src/{engine,error,events,hardware,research,web_search,store,sabra,flock,compute,bin,diagnostics,studio,types,bench_support}` | 17 |
| satellites: `pio-birds-tauri` 3, `pio-regulate-tauri` 1, `pio-nola-tauri` 1, `pio-base-app-tauri` 1, `pio-bridge` 1, `pio-embed-seed` 1 | 8 |

283 `use …gen2::…` statements expand to **454 leaf paths**. Classified
against `src/lib.rs` (`pub mod api, controller, mcp, journal`; everything
else `pub(crate)` + a root re-export list):

| First segment | OK as-is | Rewrite to root re-export | Broken (no public path) | Files hit |
| --- | ---: | ---: | ---: | ---: |
| `controller::*` | 196 | 0 | 0 (but see semantic breaks) | — |
| `generation::*` | 0 | 43 (`GenSpec ThinkingMode MediaBoundary ToolCall`) | 37 | 36 |
| `engine::*` | 0 | 25 (`Settings ExecError SamplingSettings ExecutionStats`) | 16 | 26 |
| `backend::*` | 0 | 15 (`GrammarSpec LatencyTier BackendCaps Speculative*`) | 26 | 20 |
| `session_rt::*` | 0 | 0 | 15 | 10 |
| `zoo::*` | 0 | 0 | 9 | 5 |
| `bundle::*` | 0 | 0 | 7 | 2 |
| `executor::*` / `kv::*` | 0 | 0 | 3 / 2 | 1 / 1 |
| root `Message MessageBody MessageContent ExecutionStats Settings Residency*` | 54 | 0 | 0 | — |
| root `Engine EmbedLoadRequest LoadRequest effective_context_budget` | 0 | 0 | 6 | 6 |
| **Total** | **250** | **83** | **121** | ~70 distinct |

**Naive `use gen2::` swap: ≈204 `use`-level errors** (83 rename + 121
unresolved) before counting the 715 inline `crate::gen2::…` path tokens, which
follow the same split (e.g. `gen2::backend::traits::TokenPullerDyn` ×13,
`gen2::bundle::ModelMeta` ×8, `gen2::backend::common::tokenizer::HfTokenizer`
×6, `gen2::backend::common::chat_template::ChatTemplate` ×5,
`gen2::session_rt::media_util::messages_have_images` ×4).

### 2a. The 40 unexported paths (all still *exist* in gen2, just not `pub`)

Verified by definition grep in gen2 `src/`:

- `engine::` `LoadRequest`(15) `EmbedLoadRequest`(2+2) `Engine`(1) `Capabilities` `validate_model_file` `read_gguf_file_type` `telemetry::HookBus`
- `generation::` `TokenEvent`(13) `ChannelMarkers`(6) `ReplyStateMachine`(4) `StreamEmission`(3) `TurnTelemetry Termination ReplyShape CacheState`(2 each) `TelemetrySnapshot global_aggregator ttft_bucket_upper_bounds_us` `reply_parts` `thinking::ThinkingMode`
- `backend::` `Engine`(9, the *facade*) `traits::{Backend(5) BackendSession SessionTokenizer TokenPullerDyn}` `mlxcel::{MlxcelEngine(5) ProfileMode ProfileRun worker::METALLIB_ENV}` `llama::{Engine, embedder::{EmbedderKind LlamaEmbedder}, llama_config::ModelConfig}` `mlx::{Engine, puller::TokenPuller, tokenizer::HfTokenizer}` `onnx::Engine` `external_api::Engine` `common::{tokenizer::HfTokenizer, grammar::GrammarMatcher, stop_matcher::StopMatcher, output_filter::OutputFilter, sampler::{Sampler XtcParams DryParams}, tool_calls::ToolCallTally, parse_hf_model_metadata, load_chat_template, default_llama3_template, compute_hf_model_meta}` `{TokenPuller SessionId SessionHealth ModelBundle}`
- `session_rt::` `SessionSpec`(14) `Session WarmStart ColdStart` `prompt::{merge_prompts generation_reserve build_meta_prompt}` `media_util::validate_image_path` `truncate`
- `zoo::` `ModelZoo`(3) `ModelZooEntry ModelFamily PlatformBundle detect_ram_mb current_platform_id select_for_device`
- `bundle::` `ModelMeta`(8) `gguf::{GgufMetadata parse_gguf_metadata build_model_metadata detect_format_from_path fit_context kv_bytes_per_token trim_optional estimate_ram_bytes}`
- `kv::` `KvSaveSpec KvLoadSpec` `store::{kv_dir keepwarm_enabled candidate_for_chat}`
- `executor::` `StreamingToolExecutor ConcurrencyGuard BoxedOperation`
- root: `read_gguf_architecture validate_model_architecture validate_model_file effective_context_budget ContextBudget default_context_budget_for_tier estimate_resident_mb_for_path` (all in pio-app's `mod.rs` re-export list; none in gen2's `lib.rs`)

Two are `pub(crate)` on *both* sides and only compile today because the copy
lives inside pio-core: `backend::common::chat_template::ChatTemplate` (5 uses)
and `session_rt::media_util::messages_have_images` (4). Expect more of this
class — only a compile finds them (see miss-risk).

### 2b. Semantic breaks inside "OK" paths

| Break | Host sites | Evidence |
| --- | ---: | --- |
| `gen2::Engine` now means `api::Engine`; pio-app's means the backend facade (`detect_backend_for_path`, `available_backends`, `new`) | 2 `use` + ~6 inline | gen2 `src/lib.rs` `pub use api::Engine`; facade still at `src/backend/facade.rs:320-382` |
| `SystemTask` variants removed: `Answer`(22) `TopicLabel`(2) `ContextualPrefix`(2) `Triples QueryRewrite EntityExtract Contradiction`(1 each); gen2 keeps `Title Suggestions Summary Compact` + `Custom(Cow<'static,str>)` | 30 sites / 11 files (`app/sabra_inference.rs` 9, `runners/agent_eval.rs` 4, `app/capture/ask.rs` 3, `app/base_chat.rs` 3, …) | gen2 `controller/config.rs`; `system_task_spec` has no per-task `GenSpec` for the removed ones — host must carry its own map |
| `ControllerEvent::Accepted { … }` added (emitted before the first token on StartChat/ContinueChat) | 25 host files match on `ControllerEvent`; 1 without a wildcard arm; every event loop must tolerate a new first event | gen2 `controller/commands.rs:760,834` |
| `system_infer*` return `ExecError` not `PioError` | 34 call sites | `From<ExecError> for PioError` exists at `pio-core/src/error.rs:491`, so `?` still compiles; only explicit `map_err` sites need touch |
| `InferenceHandle::{Remote,Flock,RegisteredFlockGateway}` arms, `liveness()`, `compute_provenance()`, `project_streaming_inference` removed | 18 variant-match sites + 1 + 2 + 1 | §3 |
| `ControllerCmd` +6 variants | 0 (all 11 matching files have wildcard arms) | — |
| specta derives: 45 in-tree → 71 in gen2 | TS bindings may gain types | `scripts/check-specta-compat.sh` is the gate |

## 3. The remote-dispatch seam

gen2 `src/controller/mod.rs:536`:

```rust
pub trait RemoteDispatch: Send + Sync + 'static {
    fn send(&self, cmd: ControllerCmd) -> Result<(), String>;
    fn label(&self) -> &str { "remote" }
    fn config(&self) -> &ControllerConfig { &DEFAULT_REMOTE_CONFIG }
    fn warm_model(&self, model_dir: PathBuf) {}
}
pub enum InferenceHandle { Local(ControllerHandle), Remote(Arc<dyn RemoteDispatch>) }
// + InferenceHandle::remote(impl RemoteDispatch), placement() -> Placement::{Local, Remote(&str)}
```

pio-app's three arms map onto it cleanly — each already has
`send(&self, ControllerCmd) -> Result<(), String>`:

| pio-app arm | Host type (`pio-core/src/p2p/…`) | `send` | Extra the host relies on |
| --- | --- | --- | --- |
| `Remote` | `client.rs::ResilientRemoteHandle` | `:723` | `liveness()` `:680`, `is_connected()` `:700`, `remote_node_id()` `:685` |
| `Flock` | `flock/handle.rs::FlockHandle` | `:383` + `dispatch_inference_with_failover` `:469` | `liveness()` `:787`, `flock_id()`; **the projection** `project_streaming_inference` (gen2 controller, `pub(crate)`, `cfg(flock)`) |
| `RegisteredFlockGateway` | `flock/registered_inference.rs::RegisteredFlockInferenceHandle` | `:57` | none |

What pio-app implements (all host-side; **no gen2 change required**):

1. `impl RemoteDispatch for ResilientRemoteHandle { send; label = "remote" }`,
   `… for FlockHandle { send = project→failover, else single-shot; label = "flock_peer" }`,
   `… for RegisteredFlockInferenceHandle { send; label = "flock_gateway" }`.
   `config()` default = `ControllerConfig::default()` — exactly what pio-app's
   `DEFAULT_CONFIG` arms return today; `warm_model` default no-op matches.
2. Move `project_streaming_inference` (pio-app `controller/mod.rs:968-1030`)
   into `p2p/flock/handle.rs` verbatim: it only destructures pub enum-variant
   fields, all of which gen2 still has with the same names. Its single host
   caller is `p2p/flock/fit_route.rs:1281` (a unit test that names it "THE
   SEAM") — retarget the path.
3. A host-owned typed handle so the 18 variant-matching sites keep working
   (`pio-core/src/engine.rs` :87-88, :1364-1367, :1551, :1664, :1676, :1736,
   :1745-1750; `app/common/compute.rs:20-28`; `src-tauri/src/api/p2p/{peers.rs:28,
   device_link.rs:94, client.rs:27,68,91,103,141}`; `pio-daemon/src/session_runtime.rs:747`).
   Recommended shape: `enum HostInference { Local(ControllerHandle),
   Remote(Arc<ResilientRemoteHandle>), Flock(Arc<FlockHandle>),
   Gateway(Arc<RegisteredFlockInferenceHandle>) }` stored where `gen2_ctrl` is
   today, with `fn as_gen2(&self) -> InferenceHandle` (Local passes through;
   others `InferenceHandle::remote(arc.clone())`). `Placement::Remote(label)`
   alone cannot recover `flock_id()`/`is_connected()`, and `Arc<dyn
   RemoteDispatch>` has no `as_any`, so the typed enum is the least-change route.
4. `liveness()` → method on `HostInference` (`Local`/`Gateway` = `Alive`).
   `compute_provenance()` → host fn (gen2 deleted `provenance.rs`;
   `ComputeProvenance` lives in host `compute::escalation`). Callers:
   `runners/agent_loop.rs:715`, `pio-daemon/src/session_runtime.rs:769`,
   `src-tauri/src/api/p2p/device_link.rs:95`.

Seam tests already on the host side (all need `--features flock`):
`pio-core/tests/{flock_live_inference_integration (10 run + 11 ignored),
registered_flock_client_integration (9), flock_server_readiness_integration (3),
p2p_integration (4)}`, `p2p/flock/fit_route.rs` seam test, `p2p/flock/handle.rs`
liveness tests (`:1389,:1395`).

gen2 still declares `p2p-client` / `flock` features that "will not build
standalone" (EXTRACTION.md §seam) — delete them from gen2's `Cargo.toml` as
part of step 0; they are dead now that the trait exists.

## 4. Feature flags to keep

What pio-app consumers actually enable (grep of every `Cargo.toml` + scripts):

| Feature | Enabled by | Keep in gen2? |
| --- | --- | --- |
| `backend-llamacpp` | src-tauri default, pio-daemon default, every `apple/ios/android/desktop-all/app-*` bundle, CI gates (`check-rust-gates.sh` uses `--no-default-features --features backend-llamacpp`), `install-claude-code-gate.sh` | **yes** (gen2 default) |
| `metal` / `cuda` / `vulkan` / `native` | bundles; `win-run-cuda-daemon.bat` | **yes** |
| `backend-mlxcel` | pio-daemon `apple`, src-tauri `apple` (S7 metallib bundling), 4 captest rows | **yes**; keep the mlx/mlxcel `compile_error` exclusion |
| `backend-mlx` | 2 captest rows (`mlx-vision*`), `fetch-bench-models.sh` | **yes** (verified generating on macOS 26.3 per gen2 Cargo.toml note) |
| `backend-onnx` | `vision`, `clip`, `image-gen` use it as the "≥1 backend" stub; `desktop-all` | **yes**, but note pio-core's is `[]` (ort is unconditional there) while gen2's pulls `dep:ort,dep:ndarray` — same pin, unifies |
| `backend-external-api` | pio-core default, `pio-bridge`, `app-ios-remote`, `desktop-all` | **yes** |
| `specta` | src-tauri + all satellites | **yes** (same `=2.0.0-rc.24`) |
| `tokio` | (gen2 API gate; host is async everywhere) | **yes**, on |
| `backend-candle` | no consumer enables it (`clip` uses candle directly) | optional; not load-bearing |
| `backend-executorch` | pio-core declares `[]`; no consumer | drop; gen2 already deleted the stub |
| `backend-litertlm`, `backend-mistralrs` | no pio-app consumer today | gen2-only; irrelevant to switchover |
| `p2p-client`, `flock` (gen2 side) | — | **delete from gen2** |

pio-core must forward: `backend-X = ["gen2_crate/backend-X"]`, `metal =
["gen2_crate/metal", …whisper bits…]`, `specta = ["dep:specta",
"gen2_crate/specta"]`, and the "≥1 inference backend" compile guard moves to
gen2's side (pio-core's satellites satisfy it via `backend-onnx` today — keep
that forwarding so `pio-nola-tauri` et al. are untouched).

## 5. Test surface (the gate)

- **Unit:** 1,024 `#[test]`/`#[tokio::test]` inside the 133 importing files.
  Because `cargo test -p pio-core --lib` is ~30 % flaky in `store::*` (owner
  memory), gate on the modules that import gen2:
  `cargo test -p pio-core --lib --features flock -- runners:: app::chat::
  app::capture:: app::models:: birds:: loops:: p2p::flock:: p2p::client
  compute:: engine:: sabra::memory:: studio::` — and require **3 consecutive
  green runs**, not one.
- **Integration (run by default, no `#[ignore]`):** `pio-core/tests/`
  `gen2_residency_integration` (1), `tokenizer_template_contract` (7),
  `mcp_agent_loop_roundtrip` (3); with `--features flock`:
  `flock_server_readiness_integration` (3), `registered_flock_client_integration`
  (9), `p2p_integration` (4), `flock_live_inference_integration` (10 non-live).
  `pio-daemon/tests/parity_test` (2). `pio-bridge/tests/common/mod.rs` helper.
- **Live (`#[ignore]`, need a model):** `pio-core/tests/{chat_turn_live,
  tool_call_stream_live, gemma4_multiturn_integration, agentic_chat_live,
  contextual_prefix_live, ctx_fit_live, jtbd_coding_live, kv_keepwarm_bench,
  studio_generation_live, summary_live_integration, zoo_multiturn_matrix,
  chaos}`, `src-tauri/tests/gen2_integration` (2), `pio-daemon/tests/{remote_bird_live,
  remote_code_run_live}`. Minimum live gate for the flip: `chat_turn_live`,
  `tool_call_stream_live`, `gemma4_multiturn_integration`, `contextual_prefix_live`
  (exercises the removed `SystemTask::ContextualPrefix`), and
  `flock_live_inference_integration --include-ignored` (the seam).
- **Captests (`scripts/verify-capabilities.sh`, ADR-0036 no-silent-skip):**
  rows `vlm-llama`, `embed-qwen3`, `mlx-vision`, `mlx-vision-multi` name test
  paths *inside* gen2 (`gen2::backend::llama::engine::tests::captest_vlm_caption`,
  `gen2::backend::llama::embedder::tests::captest_qwen3_embedding`,
  `gen2::backend::mlx::vision_parity::*`). After the swap those tests live in
  the gen2 crate: either gen2 keeps them and the rows get a 6th-column target
  (`-p gen2 --lib`, path dep must be a workspace member or use
  `--manifest-path`), or pio-app re-homes them. `mlxcel-stream/throughput/
  metallib/pio-code-gonogo-mlxcel` are `runners::agent_eval` host tests and stay.
- **CI:** `ci.yml` via `scripts/check-rust-gates.sh all` (fmt, clippy
  `--workspace --no-default-features --features backend-llamacpp -D warnings`,
  strict, specta, store, core, daemon, shell `cargo check -p pio
  --no-default-features --features backend-llamacpp`);
  `capability-verify.yml` compiles captests per feature surface
  (`cargo test -p pio-core --no-default-features --features $f --lib --no-run`)
  — its matrix must exercise both `gen2-crate` on/off during step 1–2.
- **gen2's own gate** (crate side): `cargo test` (521+ unit), `tests/live_inference.rs`
  (real SmolLM2 GGUF under `metal`), `backend/conformance.rs`, `.github/workflows/ci.yml`.

## 6. The plan (ordered)

**Step 0 — gen2 side, prerequisite (small, additive):**
Add `#[doc(hidden)] pub mod compat` mirroring the old module tree
(`compat::{engine, generation, backend::{traits, caps, common::{…}, llama, mlx,
mlxcel, onnx, external_api, facade}, session_rt::{prompt, media_util},
zoo, bundle::gguf, kv::store, executor}`) that `pub use`s the §2a list, with
`ChatTemplate` and `messages_have_images` promoted from `pub(crate)`. Delete the
dead `p2p-client`/`flock` features. Tag it `v0.1.0-compat`. This keeps the
narrow public API the crate chose (EXTRACTION.md §pio-app) while making the
host swap mechanical; the module is explicitly the "shrink over time" list.

**Step 1 — pio-app, path dep behind a feature (in-tree copy still default):**
`pio-core/Cargo.toml`: `gen2_crate = { package = "gen2", path = "../../gen2",
optional = true, default-features = false, features = ["tokio"] }` (renamed key
avoids `gen2` module/crate name ambiguity in `lib.rs`; workspace-external path
deps are fine — root `Cargo.toml` has no `[patch]` for it). Feature
`gen2-crate = ["dep:gen2_crate"]` plus per-backend forwards (§4).
`lib.rs`: `#[cfg(not(feature="gen2-crate"))] pub mod gen2;` /
`#[cfg(feature="gen2-crate")] pub mod gen2 { pub use gen2_crate::*;
pub use gen2_crate::compat::{engine, generation, backend, session_rt, zoo,
bundle, kv, executor}; pub use gen2_crate::compat::backend::Engine; … }` — the
last line preserves pio-app's `gen2::Engine` = backend facade meaning.
Host edits in this step (all `cfg`-neutral, so they compile against both copies):
SystemTask 8 removed variants → `SystemTask::Custom("answer")` etc. with a
host-side `GenSpec` map (30 sites, 11 files); `ControllerEvent::Accepted` arm in
the one exhaustive match + ignore-by-default elsewhere; `HostInference` enum +
three `RemoteDispatch` impls + moved `project_streaming_inference` (§3);
`compute_provenance`/`liveness` hoisted. Gate: §5 with **both** feature states
(`--features gen2-crate` and without) green; specta bindings diff empty.

**Step 2 — flip the default** (`default = [..,"gen2-crate"]`), keep the
in-tree copy compiling for one CI cycle via the capability-verify matrix
(`--no-default-features --features backend-llamacpp` = old copy). Live gate ×1
on Mac (`apple` bundle, mlxcel) + daemon CUDA box (`win-run-cuda-daemon.bat`).

**Step 3 — delete `pio-core/src/gen2/`** (move `milestones.md/plan.md/README.md`
notes to gen2/docs — already there), remove the feature, re-point the four
captest rows, drop `backend-executorch`. This is the point of no return for
parallel branches: any open branch touching `pio-core/src/gen2/` (none exist
on `main` today; check `git branch -a --contains` of any such commit first)
must land or rebase before this step.

**Step 4 — versioned dependency.** Not crates.io: gen2 depends on *git* revs
(`llama-cpp-2`, `mlx-rs`, `mlxcel`) and crates.io rejects git dependencies, so
"publish" means a tagged git dep: `gen2_crate = { package = "gen2", git =
"https://github.com/saberra-ai/gen2", tag = "v0.1.0" }`. Move to crates.io
only when the three git deps are released or vendored.

## 7. Blast-radius stops (stop and decide, do not push through)

1. **Compat surface refused or partial.** If gen2 will not expose the §2a list,
   step 1 becomes a rewrite of ≥70 host files (36 generation + 26 engine + 20
   backend importers) — a different project. Stop at step 0.
2. **Pin divergence.** Any difference in `llama-cpp-2`/`mlx-rs`/`mlxcel`/`ort`/
   `specta` between `gen2/Cargo.toml` and `pio-core/Cargo.toml` → cargo `links`
   duplicate or type mismatch across the two copies. Align pins before step 1.
3. **`SystemTask::Answer` parity.** 22 sites route the main answer path;
   gen2's `Custom` has no per-task `GenSpec`. If `contextual_prefix_live` /
   `ctx_fit_live` outputs change, stop before step 2.
4. **specta / TS bindings drift** (`check-specta-compat.sh`, bindings diff):
   frontend-visible types changing is a product change, not a refactor.
5. **Packaged Mac app**: `backend::mlxcel::worker::METALLIB_ENV` (S7 metallib
   bundling) must be in `compat`; verify `mlxcel-metallib` captest RUN before
   step 2.
6. **First-event ordering.** Any host runtime that treats the first
   `ControllerEvent` as `Token`/`Accepted`-unaware (SSE in
   `pio-daemon/src/sse.rs`, `runners/chat_runtime.rs`) — an integration test
   must show `Accepted` is swallowed, not surfaced as text.

## 8. Miss-risk (what this pass could not see)

- **No compile was run.** Error counts are static from 454 `use` leaves; the
  715 inline `crate::gen2::…` tokens, method/field-level changes in the 79
  modified files, and `pub(crate)`-across-pio-core items (found 2, expect more)
  are only counted by `cargo check -p pio-core --features gen2-crate`. Treat
  "≈204" as a floor.
- **Same-name, different-behaviour.** gen2 changed semantics under unchanged
  signatures (auto_context sizing `33a791f`, `.seed()` `b32d482`, truncation
  drops whole tool rounds `7a9875d`, residency-as-cache `6713459`, atomic KV
  writes `7097b70`, `LoadOutcome` from `load_model` `3cbd5e6`). Only the live
  gates catch these; the unit gate will be green either way.
- **Parallel-active main.** 231 commits since extraction touched none of
  `pio-core/src/gen2/` but do touch the importers; the shim minimizes conflict
  surface but step 1's 11-file `SystemTask` edit will collide with anything in
  `app/sabra_inference.rs` / `runners/agent_eval.rs`. Land it as its own small
  commit first.
- **Flaky core suite** means a red gate is not a signal; use the filtered
  module set and 3× rule.

## 9. Re-check trigger

Re-run this inventory (`git archive 5e5cadc^ | tar -x` → `diff -rq` against
`pio-app/pio-core/src/gen2`; `git log 5367cf0f..HEAD -- pio-core/src/gen2`) if
**any** of: a commit lands under `pio-app/pio-core/src/gen2/` (today: zero);
a native pin changes on either side; gen2 removes `compat` or renames a
controller item; pio-app's importer count moves past ~150 before step 1 lands;
or gen2's `SystemTask`/`ControllerEvent` change again. The drift-zero finding
is what makes the shim plan cheap — it expires the moment either condition breaks.
