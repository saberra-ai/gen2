# 02 — Fastest sound path to `cargo publish --dry-run` green

Research-decision receipt. Pipeline: FORK → LANES → FORAGE → GRADE → RUBRIC → PREMORTEM → RECEIPT.
Searched 2026-09-03. No source files were modified; every local proof ran on a scratch copy of the
repo under `/private/tmp/claude-502/…/scratchpad/simA-0.1.151` (logs: `dryrun-151.log`,
`check-151.log`, `check-156.log`, `doc-151.log`).

## The CALL (summary)

| Dependency | Call | Mechanism |
|---|---|---|
| `llama-cpp-2` / `llama-cpp-sys-2` | **(1) use the crates.io release** | `version = "=0.1.151"`, drop `git`/`rev`. The pinned rev `43733d1` is an *ancestor* of the `0.1.151` tag with the identical version number and identical `llama.cpp` submodule; the release is a strict superset. `=` is mandatory: `0.1.156` breaks gen2 (`LlamaSampler::penalties` grew a 5th argument). |
| `mlx-rs` (saberra fork) | **(4) hybrid "multiple locations"** now, **(5) upstream** as follow-up | `{ version = "0.25.3", git = …, rev = "fac86ba0…" }`. Git copy is used locally and by path/git consumers (pio-app); crates.io consumers get upstream `0.25.3`. Fork delta is build-level only (iOS cross-compile, mlx-c 0.6.0, `MACOSX_DEPLOYMENT_TARGET=26.0`). Document a `[patch.crates-io]` for macOS-26 consumers; open two upstream PRs. |
| `mlxcel` / `mlxcel-core` | **exclude from the published crate** | Neither crate exists on crates.io (upstream lablup is unpublished at v0.7.0-beta.1; fork is 0.3.3 + 2 commits, 483 behind). crates.io rejects unknown dependencies and cargo refuses to package a version-less one, so there is no manifest form that keeps it. Remove `backend-mlxcel` (feature, 2 deps, `src/backend/mlxcel/`, 31 cfg sites in 4 files) from `main`; tag the last commit; fund a "public backend seam → `gen2-mlxcel` companion crate" slice if the Mac fast path is still wanted. |
| Name | **`gen2` is free — take it** | crates.io API returns 404 for `gen2`, `gen-2`, `gen2-rs`, `pio-gen2`, `gen2-core`, `saberra-gen2`. Do **not** reserve the variants with empty crates (policy forbids name-reservation crates). |
| docs.rs | **add `[package.metadata.docs.rs]`** | `no-default-features = true`, `features = ["backend-llamacpp","backend-external-api","backend-litertlm","tokio"]`, `targets = ["x86_64-unknown-linux-gnu"]`. Verified locally with `cargo doc --no-deps`. Never enable `backend-mistralrs` (its own docs.rs build fails on Linux), `backend-onnx` (ort downloads binaries; docs.rs blocks network) or `backend-mlx` (mlx-sys clones MLX at build time) there. |
| Size / `include` | **no action needed** | Simulated `.crate`: 255 files, 5.6 MiB raw, **885.7 KiB compressed** vs the 10 MB limit. `fuzz/` (sub-package) and gitignored `resources/models` are auto-excluded. |

**Proof:** with exactly those manifest changes, `cargo publish --dry-run --allow-dirty` on the scratch copy
ran packaging → verification build → `Uploading gen2 v0.1.0` → `warning: aborting upload due to dry run`, exit 0
(`dryrun-151.log`). `cargo check` against crates.io `0.1.151` with `backend-llamacpp,backend-external-api,backend-litertlm,tokio`: 0 errors in 1m46s (`check-151.log`).

---

## 1. FORK

| Field | Value |
|---|---|
| Decision | Fastest *sound* path to a green `cargo publish --dry-run` for `gen2` 0.1.0, and is the name available |
| Options (per git dep) | (1) a crates.io release already has what we need · (2) publish the fork under a saberra-prefixed name · (3) vendor · (4) keep the feature, move the dependency to a `[patch]`/git override · (5) upstream the fork's patches |
| Stakes | Medium — wave-1 gate; reversible (a manifest change, no data). Ran the full pipeline anyway because two of the three answers turned out to hinge on facts that were easy to get wrong (version floating; "optional git dep" myth). |
| Constraints | MIT crate; default feature compiles llama.cpp; MLX backends are macOS/iOS-only; pio-app consumes gen2 by path, not registry |
| Success metric | `cargo publish --dry-run` exit 0 on CI's `publishable` job; docs.rs build succeeds; crates.io consumer of default features gets the same llama.cpp build the git consumer gets |
| Blast radius | Public crate name + a `=` version pin that every downstream user of `llama-cpp-2` must match |
| Expiration | Re-check on the next `llama-cpp-2` bump, on any `mlx-rs` release > 0.25.3, or when lablup publishes `mlxcel` |

## 2. LANES

| Lane | Hit? | Where |
|---|---|---|
| Primary / official | yes | Cargo Book (publishing, specifying-dependencies "Multiple locations", manifest include/exclude, overriding-dependencies); docs.rs `/about/metadata`, `/about/builds`; RFC 3463 (crates.io policy); crates.io server source (`publish.rs`, `canon_crate_name` migration); crates.io API (names, versions, features, sizes) |
| Practitioner ecosystem | yes | how `llama-cpp-2`, `ort`, `mistralrs`, `mlx-rs` package native code and configure docs.rs; the prefixed-fork crates that exist (`shimmy-llama-cpp-2`, `babichjacob-llama-cpp-sys-2`, `llama-cpp-4`, `safemlx*`) |
| Deployment reality | yes | crates-build-env package list (cmake/clang/libclang present, network blocked); docs.rs sandbox limits; local builds on Apple silicon |
| Critical / failure | yes | `mistralrs` 0.8.1 docs.rs failure log; llama-cpp-rs API drift inside 0.1.x (reproduced); upstream `mlx-rs` issues #308/#327 on macOS 26; llama-cpp-rs Windows cmake issues |
| Academic / evaluation | N/A | packaging policy, no evaluation literature applies |
| User-context | yes | `Cargo.toml`, `Cargo.lock`, `.github/workflows/ci.yml` `publishable` job, README line 610, `src/backend/` seam, `git ls-files` sizes |

Freshness swept (crates.io versions as of 2026-09-02; upstream repos as of 2026-09-01/02). Snowball: lockfile → git rev → tag compare → submodule sha → crates.io tarball listing. Lateral read: docs.rs "success" for `llama-cpp-2` was cross-checked against the artifact size and the build-env package list rather than trusting the badge. Adversarial: searched for docs.rs failures of C++ crates, feature-missing-on-crates.io reports, and squatting policy.

## 3. FACTS (graded)

Grades: **A** primary/official · **B** reproducible artifact (repo, tarball, local run with log) · **C** practitioner signal · **D** weak.

### 3.1 Name availability and policy
- **`gen2` is free.** `GET https://crates.io/api/v1/crates/gen2` → HTTP 404 `{"errors":[{"detail":"crate \`gen2\` does not exist"}]}`. Same 404 for `gen-2`, `gen2-rs`, `pio-gen2`, `gen2-core`, `saberra-gen2`, and for `mlxcel`, `mlxcel-core`, `saberra-mlx-rs`. **A** (registry API, UA-identified, 2026-09-03).
- **Name canonicalisation:** crates.io uniqueness uses `canon_crate_name(text) = replace(lower($1), '-', '_')` ([migration](https://github.com/rust-lang/crates.io/blob/main/migrations/20150319224700_dumped_migration_93/up.sql)). So `gen-2` ≡ `gen_2`, but `gen2` is a distinct name. **A**.
- **Squatting:** crates.io may delete a crate that "exists only to reserve a name for a prolonged period of time (often called 'name squatting') without having any genuine functionality, purpose, or significant development activity"; ownership is first-come-first-served and transfers go through the current owner ([RFC 3463](https://rust-lang.github.io/rfcs/3463-crates-io-policy-update.html), which is the text now at [crates.io/policies](https://crates.io/policies)). **A**. Implication: publish `gen2` with real content; do not park `gen-2`/`pio-gen2` as placeholders.
- **Build-time native builds / downloads:** neither RFC 3463 nor the Cargo Book restricts what a build script may compile or download. Practitioner reality: `llama-cpp-sys-2` compiles vendored llama.cpp (1.21 M downloads), `ort` downloads ONNX Runtime binaries by default (`download-binaries` in its default features; 17.4 M downloads), `mlx-sys` runs cmake `FetchContent` to clone `ml-explore/mlx` at configure time ([mlx-c CMakeLists v0.2.0](https://raw.githubusercontent.com/ml-explore/mlx-c/v0.2.0/CMakeLists.txt), 238 k downloads). All live on crates.io. **A** (absence in policy) + **B** (artifacts).
- **Size limit:** "crates.io currently has a 10MB size limit on the `.crate` file" ([Cargo Book: Publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)). The server enforces a per-crate `max_upload_size` ("max upload size is: {max_length}", [publish.rs](https://raw.githubusercontent.com/rust-lang/crates.io/main/src/controllers/krate/publish.rs)) that the team raises on request via help@crates.io ([crates.io#40](https://github.com/rust-lang/crates.io/issues/40), **C**). **A/C**.

### 3.2 Git dependencies — the rules
- "crates.io does not allow packages to be published with dependencies on code published outside of crates.io itself (dev-dependencies are ignored)." Multiple locations: "The `git` or `path` dependency will be used locally (in which case the `version` is checked against the local copy), and when published to a registry like crates.io, it will use the registry version." ([Cargo Book: Specifying Dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations)). **A**.
- **Optional does not exempt (option 4 as literally posed is impossible).** Local `cargo package --no-verify` on the real manifest: `all dependencies must have a version requirement specified when packaging. dependency \`llama-cpp-2\` does not specify a version` — `llama-cpp-2` is `optional = true`. **B** (reproduced). Server side, a dependency whose name is unknown to the registry is rejected with `no known crate named \`{}\`` and a `registry = …` dependency with "Cross-registry dependencies are not permitted on crates.io" ([publish.rs](https://raw.githubusercontent.com/rust-lang/crates.io/main/src/controllers/krate/publish.rs)). **A**.
- **`[patch]`** is stripped from the picture for dependents: "Cargo only looks at the patch settings in the `Cargo.toml` manifest at the root of the workspace. Patch settings defined in dependencies will be ignored." ([Cargo Book: Overriding Dependencies](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html)). So a `[patch]` in gen2 helps gen2's own builds only; a crates.io consumer must repeat it in *their* root manifest. **A**.
- The normalised manifest inside the simulated `.crate` shows the rewrite: `[dependencies.llama-cpp-2] version = "=0.1.151"` and `[target.'cfg(any(target_os = "macos", target_os = "ios"))'.dependencies.mlx-rs] version = "0.25.3"` with no `git` key anywhere. **B**.

### 3.3 `llama-cpp-2` / `llama-cpp-sys-2`
- **Published features:** every crates.io release since `0.1.134` (2026-02-17) ships both `mtmd` and `llguidance` (`mtmd` since `0.1.131`, 2026-01-02). `0.1.151` was published 2026-07-06; latest is `0.1.156` (2026-09-02). ([crates.io versions API](https://crates.io/api/v1/crates/llama-cpp-2/versions)). **A**.
- **Which llama.cpp:** the `0.1.151` tag (`7f0a0d95`) and the pinned rev `43733d1` both point `llama-cpp-sys-2/llama.cpp` at `9e3b928fd8c9…` = ggml-org/llama.cpp 2026-06-07 "common : relax sampler name matching (#23744)". `0.1.156` moves it to `e79e4bf6…`. **B** (GitHub contents API).
- **Pin vs tag:** `compare/0.1.151...43733d1` → `behind_by: 10, ahead_by: 0`: the pin is an ancestor of the tag. The 10 commits: model-load progress callback (`llama-cpp-2/src/model/params.rs` +47), Vulkan build fix on `x86_64-pc-windows-gnu` (`build.rs` +12/−6), `cc`/`anyhow` bumps, CI. Additive only. **B**.
- **The tarball is complete:** `llama-cpp-sys-2-0.1.151.crate` from static.crates.io is 3.3 MB, 1070 files, includes 52 `llama.cpp/tools/mtmd/*` files plus `common/`, `ggml/`, `src/` and the CMakeLists (`include` list in its [Cargo.toml](https://raw.githubusercontent.com/utilityai/llama-cpp-rs/0.1.151/llama-cpp-sys-2/Cargo.toml), `links = "llama"`). **B**.
- **Compiles:** scratch copy with `llama-cpp-2 = { version = "=0.1.151", features = ["mtmd","llguidance"] }`, `llama-cpp-sys-2 = "=0.1.151"`: `cargo check --no-default-features --features backend-llamacpp,backend-external-api,backend-litertlm,tokio` → `Finished` in 1m46s, 0 errors. **B** (`check-151.log`).
- **`0.1.156` does NOT compile gen2:** `error[E0061]: this function takes 5 arguments but 4 arguments were supplied` at `src/backend/llama/session.rs:312` and `:598` (`LlamaSampler::penalties`). llama-cpp-rs breaks its API inside 0.1.x patch releases (see also 0.1.152's `SeqState` rework). A caret `"0.1.151"` would let a consumer's resolver pick `0.1.156` and fail. **B** (`check-156.log`).
- `llama-cpp-2 0.1.151` depends on `llguidance = "1.7.5"` / `toktrie = "1.7.5"`; gen2's `llguidance = "1.7"` resolves to the same crate, so the cross-backend grammar contract in `Cargo.toml` comments holds. **B**.
- docs.rs: `llama-cpp-2` sets `[package.metadata.docs.rs] features = ["sampler"]` (default features on: `openmp`, `android-shared-stdcxx`, `common`) and its latest build **succeeded** (0.1.156, rustc 1.100.0-nightly 2026-09-01, 1m12s, 5.3 GB artifact) ([builds](https://docs.rs/crate/llama-cpp-2/latest/builds)). The visible log tail shows only the 0.40 s rustdoc step; the 5.3 GB artifact and the sandbox package list (`cmake`, `clang`, `libclang-dev`, `build-essential` in [crates-build-env packages.txt](https://raw.githubusercontent.com/rust-lang/crates-build-env/master/linux/packages.txt)) are the evidence that the vendored llama.cpp compiled there without network. **B/C**.

### 3.4 `mlx-rs` (saberra fork)
- Upstream renamed: `oxideai/mlx-rs` → `oxiglade/mlx-rs`; crates.io `mlx-rs` max is `0.25.3` (2025-12-16; no release since), `mlx-sys` `0.2.0` (2025-06-28, 170 KB, vendors mlx-c 0.2.0 which `FetchContent`s MLX `v0.25.1` at build time). **A/B**.
- Fork = upstream + 5 commits (`compare/oxiglade:main...fac86ba0`: ahead 5, behind 14): iOS cross-compile ×3 (Apr 2026), mlx-c 0.6.0 / MLX 0.31.2 bump (Jun 2026), `MACOSX_DEPLOYMENT_TARGET=26.0` export (Jul 2026). Files: `mlx-sys/build.rs` +120/−45, submodule bump, and mechanical `mlx_fft_norm` signature updates in `fft/*.rs`, `ops/quantization.rs`. No new public API → the fork's crate API is upstream 0.25.3's, so `version = "0.25.3"` is honest. Workspace version at the fork rev is `0.25.3`, which is what cargo checks against the git copy; `cargo package` accepted it. **B**.
- Upstream state: `oxiglade/mlx-rs` main already tracks mlx-c 7 commits past `v0.6.0` and has a macOS-26 `clang_rt` link workaround, but has **no** deployment-target export and **no** iOS handling; issue [#308 "Metal support on MacOS 26.2 not available"](https://github.com/oxiglade/mlx-rs/issues/308) is open (2026-01-15), [#327](https://github.com/oxiglade/mlx-rs/issues/327) (metallib path baked at compile time) closed. No PR from saberra exists upstream. **B**.
- Consequence for a crates.io consumer who enables `backend-mlx`: they get upstream `0.25.3` → MLX 0.25.1 → the pre-fix behaviour on macOS 26 (memory: "Gen2 MLX broken on macOS 26" before the fork). They must add to their root manifest: `[patch.crates-io] mlx-rs = { git = "https://github.com/saberra-ai/mlx-rs", rev = "fac86ba092ba457d055f8d96d8a7a86dc8dee343" }`. Patching `mlx-rs` alone suffices because `mlx-sys`/`mlx-macros`/`mlx-internal-macros` are `path` deps inside that git workspace and resolve from the same checkout. **A** (patch semantics) + inference.
- Option (2) cost: a renamed fork means four crates (`saberra-mlx-sys`, `saberra-mlx-macros`, `saberra-mlx-internal-macros`, `saberra-mlx-rs`) with proc-macro crate-path rewrites, plus a `links` key collision (`mlx-sys` links MLX; two `-sys` crates linking the same library cannot coexist in one graph). The existing prefixed forks show the pattern works but rots: `shimmy-llama-cpp-2` stuck at 0.1.123 (2025-10), `babichjacob-llama-cpp-sys-2` at 0.1.85 (2024-12). **B/C**.

### 3.5 `mlxcel` / `mlxcel-core`
- Not on crates.io under any name (404 for both). Upstream `lablup/mlxcel` is at `v0.7.0-beta.1`, unpublished, manifest already carries description/keywords/categories (publish-ready-looking) but no crates.io issue/PR exists. Fork `saberra-ai/mlxcel` at `21306cf` = 0.3.3 + 2 commits ("expose `set_metallib_path` for embedded bundling", "make `vision_config` optional"), 483 commits behind; GitHub code search for `set_metallib_path` in upstream: 0 hits → not upstreamed. **B**.
- Structure: `mlxcel` is a workspace root with 4 `[[bin]]`s + lib and `path` deps on `mlxcel-core`, `mlxcel-surgery`, `mlxcel-xla`; `mlxcel-core/build.rs` builds MLX via cmake from `../mlx-cpp/CMakeLists.txt` (outside its own package root — unpackagable as laid out), which `FetchContent`s `ml-explore/mlx` at a pinned sha and applies source patches. Publishing a saberra-prefixed pair would mean restructuring and then owning a 483-commits-behind fork of an "experimental inference engine" on the registry. **B**.
- In gen2: `src/backend/mlxcel/` is 1751 LOC, 31 `cfg(feature = "backend-mlxcel")` sites across 4 files, imports 15 crate-private paths (`backend::traits`, `engine::ExecError`, `session_rt::media_util`, …); there is no public backend-registration API (`facade.rs` selects backends by `cfg`). So a companion crate cannot be built today without first making the seam public. Off by default, absent from every platform bundle, mutually exclusive with `backend-mlx`. **B** (repo).

### 3.6 docs.rs
- Config keys: `features`, `all-features`, `no-default-features`, `default-target` (default `x86_64-unknown-linux-gnu`), `targets` (default: `x86_64-unknown-linux-gnu, aarch64-apple-darwin, x86_64-pc-windows-msvc, aarch64-unknown-linux-gnu, i686-pc-windows-msvc`), `rustdoc-args`, `cargo-args` ([docs.rs metadata](https://docs.rs/about/metadata)). **A**.
- Sandbox: 6.44 GB RAM, 15 min rustdoc limit, **network blocked** ("Network access will not be enabled for any crate"), read-only FS except `OUT_DIR`, `DOCS_RS` env var set, max 10 targets ([docs.rs builds](https://docs.rs/about/builds)). **A**.
- Failure example: `mistralrs 0.8.1` sets `[package.metadata.docs.rs] no-default-features = true` and **still fails** on docs.rs: `objc2 only works on Apple platforms. Pass --target aarch64-apple-darwin` ([build 3095014](https://docs.rs/crate/mistralrs/0.8.1/builds/3095014)). **B**.
- `ort` lists an explicit feature set for docs.rs that omits `download-binaries`, restricts `targets`, and passes `rustdoc-args = ["--cfg", "docsrs"]` ([ort Cargo.toml](https://raw.githubusercontent.com/pykeio/ort/main/Cargo.toml)). gen2's non-Windows `ort` dep uses `default-features = true` (→ `download-binaries`) → would fail on docs.rs. **A/B**.
- Local proof: `cargo doc --no-deps --no-default-features --features backend-llamacpp,backend-external-api,backend-litertlm,tokio` on the scratch copy → `Generated …/doc/gen2/index.html`, exit 0, 28 warnings incl. two `unresolved link to \`crate::Capabilities\``. **B** (`doc-151.log`).

### 3.7 Package contents / size
- `cargo package --list`: 255 files; auto-excluded: `fuzz/` (sub-package with its own `Cargo.toml`), `target/`, gitignored `resources/models` ([Cargo Book: manifest exclude/include](https://doc.rust-lang.org/cargo/reference/manifest.html#the-exclude-and-include-fields): sub-packages and `target` are always excluded; VCS-ignored files are excluded unless `include` is set). Included: `src/`, `tests/` (incl. `tests/fixtures/gguf/deeply_nested_arrays.gguf` at 2.3 MB — the largest tracked file), `benches/`, `examples/`, `docs/` (6 files), `test_plan.md`, `proptest-regressions/`, `README.md`, `LICENSE`. **A/B**.
- Simulated `.crate`: `Packaged 255 files, 5.6MiB (885.7KiB compressed)` — 8.6 % of the limit. No `include`/`exclude` needed. Optional trim: `exclude = ["/docs/plans", "/test_plan.md"]` keeps planning docs out of the tarball (this file will otherwise ship in it). **B**.

## 4. RUBRIC

| Criterion | Weight | (1) crates.io release | (2) prefixed fork on crates.io | (3) vendor | (4) version+git hybrid / exclude | (5) upstream |
|---|---:|---|---|---|---|---|
| Gate goes green now | 35 % | llama: yes (proven) | yes, after publishing 2–4 crates | yes if < 10 MB (MLX tree is not) | yes (proven) | no — weeks |
| Registry consumer parity with git consumer | 25 % | llama: identical (same submodule sha) | identical | identical | mlx: **degraded** on macOS 26 unless they `[patch]`; mlxcel: feature absent | identical once released |
| Maintenance burden | 20 % | none | own N crates that rot (shimmy/babichjacob precedent) | rebase a C++ tree per bump | one line + README note | two PRs, then nothing |
| Policy / size risk | 10 % | none | `links` collision for `-sys` forks | 10 MB | none | none |
| Reversibility | 10 % | trivial | crate names are forever | trivial | trivial | trivial |

Per-dependency winner: llama → (1); mlx-rs → (4) now + (5) queued; mlxcel → exclude (the only sound form of (4)).

**References to mirror**
- Manifest pattern for a C++-building crate on docs.rs: [`llama-cpp-2/Cargo.toml`](https://github.com/utilityai/llama-cpp-rs/blob/0.1.151/llama-cpp-2/Cargo.toml) (`[package.metadata.docs.rs] features = [...]`) and [`ort/Cargo.toml`](https://github.com/pykeio/ort/blob/main/Cargo.toml) (explicit feature list, `targets`, `--cfg docsrs`).
- Git-locally/registry-when-published: [Cargo Book "Multiple locations"](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations).
- Negative example: [`mistralrs 0.8.1` on docs.rs](https://docs.rs/crate/mistralrs/0.8.1/builds/3095014) — `no-default-features` alone is not a docs.rs strategy; only enable features you have built on `x86_64-unknown-linux-gnu`.
- Negative example: prefixed forks `shimmy-llama-cpp-2` / `babichjacob-llama-cpp-sys-2` — they work, then freeze.

## 5. Critical alignment audit (repo vs findings)

- **Aligns:** CI's `publishable` job already runs the exact gate and installs `cmake libclang-dev`, which matches the docs.rs sandbox contents. The `Cargo.toml` comment on `llguidance` ("same crate llama_cpp_2's sampler uses") is true at 0.1.151 (llguidance 1.7.5).
- **At risk:** `publishable` is `continue-on-error: true`, so a red gate never blocks; once green it should become a hard gate. README line 610 ("three dependencies resolve only to git") and the install snippet at lines 9/22 (`gen2 = { git = … }`) go stale on publish. README feature table rows 576–577 mention `backend-mlxcel`.
- **Would violate:** the CI comment says "crates.io does not accept a package whose dependencies resolve only to git repositories, and three of ours currently do" — accurate, but the fix is not "add versions": for `mlxcel` there is no version to add, which is why the call is exclusion rather than a manifest tweak.
- **Where the literature was off vs the repo:** the fork question as posed assumed an optional dependency might be exempt; it is not (reproduced). It also assumed the git pin might need a newer release than any published; the opposite is true — the pin is *older* than the same-numbered release.

## 6. PREMORTEM

1. *"You pinned `=0.1.151`; a downstream that also depends on `llama-cpp-2` at a different `=` cannot build."* True and accepted: llama-cpp-rs breaks API across 0.1.x, so co-dependants must agree on a version regardless. Mitigation: bump on a schedule, re-run the 0.1.15x check, and say so in README.
2. *"docs.rs succeeded for `llama-cpp-2` but gen2 pulls tokio/full, minijinja, image, reqwest… will it hit 15 min / 6.4 GB?"* The 15-minute limit is rustdoc time, not dependency compile time; local `cargo doc` of gen2 completes in well under a minute after deps. Residual risk: docs.rs's slower CPU on the llama.cpp compile. Recovery: docs.rs rebuilds can be requested, and the feature list can be shrunk to `backend-external-api,backend-litertlm,tokio` (also verified compiling) if the first build fails.
3. *"A crates.io consumer enables `backend-mlx` on macOS 26 and gets 'Unable to load kernel'."* Known and documented; the `[patch.crates-io]` recipe is the remedy until upstream releases. Not the default path (`apple` bundle is llama+metal).
4. *"Deleting `backend-mlxcel` loses the Mac fast path pio-app uses."* pio-app builds its own in-tree gen2 copy (memory: both can drift), so the registry cut does not remove it there; and the durable path (public backend seam → `gen2-mlxcel`) is a named slice. Tag `mlxcel-last` before removal.
5. *"Did I verify upstream `mlx-rs 0.25.3` even compiles gen2's `backend-mlx`?"* No — `cargo publish --dry-run` verifies default features only, and the fork's Rust API is upstream's plus internal signature changes, so compile parity is inferred, not proven. Listed under miss-risk.
6. *"Is the `gen2` name too generic / trademark-adjacent?"* Free on crates.io, no reserved-name hit, and the crate has genuine content; the policy only bites empty reservations.
7. *"What changes in 30 days?"* A new `llama-cpp-2` (weekly cadence) with a new llama.cpp; an `mlx-rs` release (none in 9 months, so unlikely); lablup publishing `mlxcel` (would reopen the companion-crate question).

## 7. Ordered steps to a green dry-run

1. `Cargo.toml`: `llama-cpp-2 = { version = "=0.1.151", features = ["mtmd", "llguidance"], optional = true }`, `llama-cpp-sys-2 = { version = "=0.1.151", optional = true }`; delete the `git`/`rev` keys. `cargo update -p llama-cpp-2 -p llama-cpp-sys-2`. Expect `Cargo.lock` source to flip to `registry+https://github.com/rust-lang/crates.io-index`, same version.
2. `Cargo.toml`: `mlx-rs = { version = "0.25.3", git = "https://github.com/saberra-ai/mlx-rs", rev = "fac86ba092ba457d055f8d96d8a7a86dc8dee343", optional = true }` (add `version`, keep the rest). Update the comment block to say why both are present.
3. Remove `backend-mlxcel`: the feature line, `mlxcel`/`mlxcel-core` deps and their comment block, `src/backend/mlxcel/`, the `cfg` sites (`grep -rn 'feature = "backend-mlxcel"' src` → 31 sites in 4 files incl. the `compile_error!` guard and the no-backend list in `src/backend/mod.rs`), README rows 576–577. `git tag mlxcel-last` on the commit before. Open the slice "public backend seam so `gen2-mlxcel` can live as a `publish = false` companion crate".
4. Add
   ```toml
   [package.metadata.docs.rs]
   no-default-features = true
   features = ["backend-llamacpp", "backend-external-api", "backend-litertlm", "tokio"]
   targets = ["x86_64-unknown-linux-gnu"]
   ```
   (Add `rustdoc-args = ["--cfg", "docsrs"]` only together with `#![cfg_attr(docsrs, feature(doc_cfg))]` feature badges — optional polish.)
5. Optional: `exclude = ["/docs/plans", "/test_plan.md"]`.
6. Locally: `cargo publish --dry-run --allow-dirty` → expect `Packaged ~255 files … (~886KiB compressed)`, verification build, `aborting upload due to dry run`. Then `cargo doc --no-deps --no-default-features --features backend-llamacpp,backend-external-api,backend-litertlm,tokio` and fix the two `crate::Capabilities` intra-doc links. (`cargo install cargo-docs-rs && cargo docs-rs` reproduces the docs.rs flags more faithfully.)
7. CI: drop `continue-on-error: true` from `publishable` and rewrite its comment; README: line 610 paragraph, install snippets → `gen2 = "0.1"` / `gen2 = { version = "0.1", default-features = false, features = ["backend-external-api"] }`; add the `backend-mlx` on macOS 26 `[patch.crates-io]` recipe and the `=0.1.151` note.
8. Publish: `cargo login` with a token scoped `publish-new` + `publish-update`; `cargo publish`; then `cargo owner --add github:saberra-ai:<team>` so ownership is not a single account. Watch `https://docs.rs/crate/gen2/0.1.0/builds`.
9. Follow-ups: two upstream PRs to `oxiglade/mlx-rs` (`MACOSX_DEPLOYMENT_TARGET` export citing their #308; iOS cross-compile) so the `git` key on `mlx-rs` can be deleted at their next release.

## 8. RECEIPT

```
Claim: Publish `gen2` from crates.io-only dependencies by (a) taking llama-cpp-2 =0.1.151 from the registry,
       (b) the version+git hybrid for mlx-rs with a documented [patch] and upstream PRs queued,
       (c) cutting backend-mlxcel from the published crate; the name `gen2` is free; no include/exclude
       needed (886 KiB); add a docs.rs metadata block with an explicit, Linux-verified feature list.
- Question: fastest sound path to `cargo publish --dry-run` green + name availability (medium stakes; full
  pipeline run because two premises of the fork were false).
- Sources: Cargo Book publishing / specifying-dependencies / manifest / overriding-dependencies (A);
  docs.rs about/metadata + about/builds (A); RFC 3463 (A); crates.io publish.rs + canon_crate_name migration (A);
  crates.io API for names/versions/features/sizes (A); GitHub compares + contents for llama-cpp-rs, saberra/oxiglade
  mlx-rs, saberra/lablup mlxcel (B); static.crates.io tarball of llama-cpp-sys-2 0.1.151 (B); crates-build-env
  packages.txt (B); docs.rs build pages for llama-cpp-2 0.1.156 and mistralrs 0.8.1 (B); local scratch-copy
  runs: cargo package, cargo check ×2, cargo publish --dry-run, cargo doc (B, logs in scratchpad);
  forum/search summaries on squatting and size-limit raises (C).
- Coverage: primary · practitioner · deployment · failure · user-context hit; academic N/A. Freshness swept
  (2026-09-02 crate versions). Snowball (lock → rev → tag → submodule → tarball) and lateral read (docs.rs
  badge vs artifact size vs sandbox packages) done. Premortem run (7 items above).
- Recommendation: default I'll take — steps 1–8 as written. Your call — (i) whether to fund the public
  backend seam so mlxcel returns as a companion crate, or let pio-app's in-tree copy be its only home;
  (ii) whether to trim docs/plans from the tarball.
- References to mirror: llama-cpp-2 + ort docs.rs metadata; Cargo "Multiple locations"; negative: mistralrs 0.8.1
  docs.rs failure, stale prefixed forks. Rubric: §4.
- What's NOT proven (miss-risk): upstream mlx-rs 0.25.3 compiling gen2's backend-mlx (not built);
  docs.rs wall-clock for gen2's full dependency compile (only llama-cpp-2's own build observed);
  the crates.io server-side checks (dry-run stops before upload; unknown-dependency and size checks are
  server-side and were reasoned from source, not exercised); any llama-cpp-2 release after 2026-09-02.
- Re-check trigger: next llama-cpp-2 bump (re-run the 0.1.15x check before changing the `=` pin);
  any mlx-rs release > 0.25.3 (drop the git key); lablup publishing mlxcel or merging set_metallib_path
  (reopen the companion-crate slice); 2026-10-03 at the latest.
```
