# 04 — Benchmark method: what gen2 publishes, and how it is kept from rotting

Research-decision receipt. Pipeline: FORK → LANES → FORAGE → GRADE → RUBRIC → PREMORTEM → RECEIPT.
Researched 2026-09-03. Not a source change; a decision for wave 1 of the README benchmark table.

## FORK

| Field | Value |
|---|---|
| Decision | The metrics, protocol, file format and rot control behind a public "gen2 overhead vs raw llama.cpp" table (and, later, a cross-backend table) |
| Options | (a) publish `backend_parity` output as-is; (b) llama-bench-shaped protocol (pp512/tg128, same llama.cpp *build* on both sides, JSON artifacts, generated table, CI freshness gate); (c) criterion/hyperfine-style harness; (d) no table until CI has a GPU |
| Stakes | **Deep** — trust. A wrong number in a README is the thing r/LocalLLaMA screenshots. Reversible in code, not in reputation |
| Constraints | Rust; llama.cpp vendored at a pinned `llama-cpp-rs` rev; owner's machines only (M4 Pro / other Apple Silicon on macOS 26, RTX 3080 Windows box, Pi 5); GitHub CI has no GPU and paid runners need a human decision; batch=1 embeddable runtime |
| Success metric | A reader can reproduce any row from the committed JSON + the README's stated commands, and the table can never silently disagree with what was measured |
| Blast radius | The table format and result-file schema harden once external people cite them |
| Expiration | Re-check when the llama.cpp pin moves, when llama-bench's JSON schema changes, or by 2026-12-01 |

## User-context (what already exists — read first)

- `benches/backend_parity.rs` already runs gen2 and `llama-bench` on the same GGUF and prints a decode ratio, with median-of-5, a 2 s settle, a ±15 % noise ceiling and a "faster than the reference means you compared different work" guard. Good instincts; three things make it unpublishable as-is (see Alignment audit).
- `benches/hot_paths.rs` is explicitly *not a gate* and deliberately avoids criterion ("pulls a large tree for something that runs by hand"). That stance is consistent with criterion's own FAQ and is kept.
- `src/backend/conformance.rs` establishes the house style for honesty: skip *loudly*, and a checked list (`NEVER_PRODUCED_A_TOKEN`) that fails when it goes stale. The rot control below is the same idea applied to numbers.
- `Cargo.toml` pins `llama-cpp-rs` at `43733d1de03e8a7d582f5732e9836f2387997c4d`; that checkout vendors llama.cpp submodule `9e3b928fd8c9d14dbf15a8768b9fdd7e5c721d66` = tag **b9553** (2026-06-07). The `llama-bench` on this machine's `PATH` is Homebrew `llama.cpp 0.3.0`, `version: 9000 (1a03cf47f)` — **553 builds older than the one gen2 links**. Today's parity ratio therefore compares two different llama.cpp builds, which the bench file's own doc-comment admits ("says more about the two builds than about the wrapper").
- CI (`.github/workflows/ci.yml`) already has a weekly `schedule` and a `macos-latest` job; there is no bench job and no README-generation step. README has no numbers today except "roughly 1.7x the CPU's decode rate" for LiteRT-LM GPU (unsourced, no commit, no machine — the kind of claim this decision replaces).

## LANES and FORAGE (what was searched; reddit.com is blocked to this agent — see miss-risk)

| Lane | Hit | Notes |
|---|---|---|
| Primary / official | llama-bench README; `scripts/compare-llama-bench.py`; `scripts/compare-commits.sh`; llama.cpp discussion #4167 (Apple Silicon table); mlx-lm `benchmark.py` + `BENCHMARKS.md`; Apple ML Research M5/MLX post; vLLM `bench serve` docs; criterion FAQ; hyperfine JSON | |
| Academic | arXiv 2511.05502 (MLX / MLC-LLM / Ollama / llama.cpp / MPS on M2 Ultra) | Metrics vocabulary only; abstract states no parity protocol |
| Practitioner | mistral.rs README "Benchmarks" + `releases/v0.8.2/report.md`; mistral.rs discussion #612; llama-cpp-python discussion #1831; llm-tracker "Benchmarking LLM Speed" + "Cheat-sheet for hardware reviewers"; hiesch.eu 29-GGUF post; famstack "isolating variables"; Ollama MLX blog; TanStack/charts benchmark rot control; github-action-benchmark | |
| Deployment | Apple thermal behaviour (`pmset -g therm`, `powermetrics`), NVIDIA clock locking (`nvidia-smi -pm 1 -lgc`) incl. RTX 3080 caveat, macOS `phys_footprint` vs RSS, MLX `get_peak_memory` under-reporting (mlx #3896) | |
| Failure | llama.cpp #21494 (M3 Ultra tg128 regression report with no stddev/bisect); Ollama's NVFP4-vs-Q4_K_M comparison; llama.cpp #6233 (continuous perf testing, went stale); famstack's list of misleading cross-format comparisons | |
| User-context | the two bench files, conformance.rs, Cargo.toml pin, ci.yml, README "Backends" | |

Snowball: #4167 → `--delay 30` protocol; llama-bench README → JSON schema → `compare-llama-bench.py` key fields; mistral.rs README → v0.8.2 report → `--iterations 3 --warmup 1` + host-metadata table. Lateral read: Ollama's MLX numbers checked against its own blog (different quant on each side, single date, no repetitions) and against the HN thread (no methodology critique there — the critique lives in practitioner posts). Adversarial: `regression`, `unfair`, `not apples to apples`, `under-reports`, `stale`.

## GRADE — the claims this decision rests on

| # | Claim | Grade | Source |
|---|---|---|---|
| 1 | llama-bench defaults: `-p 512 -n 128 -r 5`, warmup on by default (`--no-warmup` to skip), `--delay N` seconds between tests, `-d` depth, `-o json|jsonl|md|csv|sql`; results are *averaged* over `-r` and reported as avg t/s ± stddev; JSON rows carry `build_commit`, `build_number`, `cpu_info`, `gpu_info`, `backends`, `model_filename`, `model_type`, `model_size`, `model_n_params`, `n_batch`, `n_ubatch`, `n_threads`, `n_gpu_layers`, `flash_attn`, `type_k/v`, `n_prompt`, `n_gen`, `n_depth`, `test_time`, `avg_ns`, `stddev_ns`, `avg_ts`, `stddev_ts`, `samples_ns`, `samples_ts` | A | [tools/llama-bench/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) |
| 2 | llama.cpp's own cross-commit comparison keys rows on ~27 identity fields (cpu/gpu/backends/model/batch/threads/…) and prints `compare ÷ baseline` speedup; `compare-commits.sh` builds both commits and runs `llama-bench -o sql` into one sqlite | A | [scripts/compare-llama-bench.py](https://github.com/ggml-org/llama.cpp/blob/master/scripts/compare-llama-bench.py), [scripts/compare-commits.sh](https://github.com/ggml-org/llama.cpp/blob/master/scripts/compare-commits.sh) |
| 3 | The canonical Apple Silicon table is run at a *named commit* with `-p 512 -n 128 -ngl 99 --delay 30`, rows carry the build commit, and the maintainers attribute run-to-run inconsistency to laptop cooling | A | [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |
| 4 | `mlx_lm.benchmark`: `-p/--prompt-tokens` (512), `-g/--generation-tokens` (1024), `-b`, `-n/--num-trials` (5), `--delay`; one untimed warmup; prompt is *random token ids* (`mx.random.randint(0, vocab_size, …)`); prints per-trial and **average** `prompt_tps`, `generation_tps`, `peak_memory` — no stddev, no TTFT | A | [mlx_lm/benchmark.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) |
| 5 | MLX-LM's published table uses `mlx_lm.benchmark --model … -p 2048 -g 128` on a 64 GB M4 Max, macOS 26.1, with MLX and mlx-lm versions recorded; trial count / thermal not stated | A | [mlx_lm/BENCHMARKS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md) |
| 6 | Apple reports TTFT (s) and generation tok/s at prompt 4096 / gen 128 via `mlx_lm.generate`, M4 vs M5, no repetitions stated | A | [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |
| 7 | mistral.rs publishes "mean speedup vs llama.cpp" tables with `--iterations 3 --warmup 1`, full command lines, HF revisions, driver/CUDA/rustc versions and a per-host metadata block; llama.cpp side is `llama-bench -ngl 99 -fa 1`; compared quants are UQFF q8 vs GGUF Q8_0 | B | [mistral.rs README#benchmarks](https://github.com/EricLBuehler/mistral.rs/blob/master/README.md), [releases/v0.8.2/report.md](https://github.com/EricLBuehler/mistral.rs/blob/master/releases/v0.8.2/report.md) |
| 8 | mistral.rs's earlier CUDA claim was absolute decode t/s per GPU on "Mistral 7B Q4_K_M" both sides, batch 1, `llama-cli … -n 1024` as the reference | C | [mistral.rs discussion #612](https://github.com/EricLBuehler/mistral.rs/discussions/612) |
| 9 | llama-cpp-python "overhead" (22 vs 25 t/s, 80 % GPU load) was a wrapper *default* (`logits_all=True`), not the binding; measured through an HTTP UI against `llama-cli` — i.e. different work on each side | C | [llama-cpp-python #1831](https://github.com/abetlen/llama-cpp-python/discussions/1831) |
| 10 | vLLM's serving vocabulary: TTFT, TPOT (mean per-output-token, first token excluded), ITL, E2EL | A | [vllm bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/) |
| 11 | Hardware-reviewer consensus: pp512 = compute-bound context replay, tg128 = bandwidth-bound single-user decode; record exact build number, quant, flags; test with/without `-fa`; run headless; Llama-2-7B Q4_0 is the cross-site baseline | C | [llm-tracker cheat-sheet](https://llm-tracker.info/howto/LLM-Inference-Benchmarking-Cheat%E2%80%91Sheet-for-Hardware-Reviewers), [Benchmarking LLM Speed](https://llm-tracker.info/howto/Benchmarking-LLM-Speed) |
| 12 | criterion's FAQ: don't rely on benchmark numbers from cloud CI; use a baseline/compare flow or instruction counting instead | A | [criterion FAQ](https://bheisler.github.io/criterion.rs/book/faq.html) |
| 13 | hyperfine JSON exports `mean, stddev, median, min, max, times[]`, with `--warmup N`; it is a whole-process timer (good for cold load, useless for tok/s) | B | [sharkdp/hyperfine](https://github.com/sharkdp/hyperfine) |
| 14 | Laptop Apple Silicon throttles under sustained inference on the order of 5–15 min (Air/Pro-non-Max worst; Max/Ultra desktops often not at all); `pmset -g therm` exposes `CPU_Speed_Limit`; GPU frequency needs `sudo powermetrics --samplers gpu_power` | C/D | search-lane consensus (multiple blogs), [pmset via node_exporter #2031](https://github.com/prometheus/node_exporter/issues/2031) |
| 15 | NVIDIA: `nvidia-smi -pm 1` then `-lgc <base>` reduces variance; on a GeForce RTX 3080 the lock silently fails to take unless persistence mode is on | C | [NVIDIA forum](https://forums.developer.nvidia.com/t/nvidia-3080-gpu-clock-frequency-lock-issue-with-nvidia-smi-520-56-06-in-ubuntu-20-04/297027), [Microway](https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/) |
| 16 | macOS: `resident_size`/max RSS and `phys_footprint` differ by multiples on Apple Silicon; MLX's `get_peak_memory()` under-reports Metal-resident memory (46 GB logged vs 110 GB footprint) | C | [mlx #3896](https://github.com/ml-explore/mlx/issues/3896), [javacpp #516](https://github.com/bytedeco/javacpp/issues/516) |
| 17 | Negative example: Ollama's MLX launch compares Ollama 0.19/MLX **NVFP4** against Ollama 0.18/llama.cpp **Q4_K_M** on one date with no repetitions or prompt length stated | A (for what it says) / D (as evidence) | [ollama.com/blog/mlx](https://ollama.com/blog/mlx) |
| 18 | Negative example: a tg128 regression report with no stddev, no exact command, no bisect, no thermal note — and no response | C | [llama.cpp #21494](https://github.com/ggml-org/llama.cpp/issues/21494) |
| 19 | Cross-format pitfalls named by a practitioner who isolated variables: model-version confusion (`-Instruct-2507` vs not), bf16 on M1/M2 vs fp16 GGUF, "same engine, different wrapper, 37 % slower" (Ollama vs raw llama.cpp), uniform 4-bit vs K-quant quality | C | [famstack.dev](https://famstack.dev/guides/mlx-vs-gguf-part-2-isolating-variables/) |
| 20 | Rot control in the wild: committed results JSON + generated markdown; docs check rejects a cell whose recorded source revision ≠ last commit that touched core source; docs-only commits do not stale it; CI uploads a candidate baseline instead of overwriting the tracked one | B | [TanStack/charts benchmarks/comparison](https://github.com/TanStack/charts/blob/main/benchmarks/comparison/README.md) |
| 21 | llama.cpp's own continuous server benchmarking (k6 + Prometheus on a self-hosted T4, PR comments) went stale and was closed | C | [llama.cpp #6233](https://github.com/ggml-org/llama.cpp/issues/6233) |

## The five sub-questions, answered

### (1) Metrics
- **pp512 / tg128 exactly as llama-bench defines them** (`-p 512 -n 128`): 512 prompt tokens processed in batches; 128 tokens generated one at a time at batch 1; t/s = tokens ÷ elapsed per test. These are what #4167, llm-tracker, and every hardware review use — publish anything else and nobody can line it up. (Claims 1, 3, 11.)
- **TTFT** (ms, prompt 512): gen2-only, measured from `send()` to first `Token` event with the engine's own `first_token_us`. llama-bench has no TTFT; do not fabricate one from pp512. Report it because the embeddable story is "how long until the host can paint" (Apple and vLLM both headline it — claims 6, 10).
- **Peak memory: wave 2, not wave 1.** RSS on macOS is not comparable to Metal-resident memory, MLX's counter under-reports, and llama-bench does not emit one (claim 16). Wave 1 reports `model_size` from llama-bench's JSON (file bytes) and nothing else about memory. When added, it is `phys_footprint` (macOS) / `ru_maxrss` (Linux) / working-set peak (Windows), labelled by method.
- Peers: MLX-LM publishes prompt tps / generation tps / peak memory (averages only); Apple publishes TTFT + gen tok/s; mistral.rs publishes prefill/decode t/s per (prompt-len, depth) plus **mean speedup vs llama.cpp**; vLLM publishes TTFT/TPOT/ITL under concurrency, which does not apply to a batch-1 runtime and is explicitly out of scope.

### (2) Protocol
- **One llama.cpp build on both sides.** Build `llama-bench` from the *vendored* submodule (`~/.cargo/git/checkouts/llama-cpp-rs-*/<rev>/llama-cpp-sys-2/llama.cpp`, today b9553) with the same backend flag (`-DGGML_METAL=ON` / `-DGGML_CUDA=ON`), and assert `build_commit` in its JSON starts with the pinned sha. Refuse to write a result file otherwise. This is the single largest change from today, where brew's b9000 is the reference (Alignment audit).
- **Same work.** `-p 512 -n 128 -ngl 99 -fa <same as gen2>`, `n_batch`/`n_ubatch`/`n_threads`/`type_k`/`type_v` copied from what gen2's load actually used and echoed into the result so the two rows share llama.cpp's own identity key (claim 2). gen2 side: 512 *tokens* of prompt, greedy, 128 tokens out, `n_ctx` ≥ 1024 on both.
- **Warmup 1, repetitions 5, `--delay`.** Matches llama-bench defaults and #4167; mistral.rs uses 3+1 for a much bigger sweep. Use llama-bench's `--delay 5` between its tests and a ≥5 s settle between sides (the bench file already learned this the hard way; #4167 uses 30 s for laptops — use 30 on any machine with a battery).
- **Report median and stddev from the raw samples, both sides identically.** Read `samples_ts` from `-o json` rather than scraping the markdown; compute the same statistic for gen2's five samples. Publish `median ± stddev (n=5)`. The bench file's median-vs-mean mismatch goes away.
- **Thermal sentinel, not a thermal opinion.** Bracket the run: reference tg128 first *and* last; if the last median is >5 % below the first, the run is written with `"valid": false, "reason": "throttled"` and does not enter the table. Record `pmset -g therm` (`CPU_Speed_Limit`) before and after on macOS, `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` on the 3080; lid open, on mains, display idle. On the 3080: `nvidia-smi -pm 1` then `-lgc <base clock>` (claim 15).
- **Batch = 1 only, one model class per table row.** No concurrency numbers; this is an embeddable runtime. Do not put a MoE and a dense model in the same headline sentence.
- **Pin the artifact, not just the name.** Record the GGUF's SHA-256 and byte size and the HF repo/revision if known (hiesch and #21494 are both weakened by not doing this).

### (3) Overhead framing
- Headline is **`gen2 ÷ llama-bench` for tg128** on the identical llama.cpp build, as a percentage, with both absolute medians beside it so the ratio cannot hide a slow machine. This is llama.cpp's own `compare-llama-bench.py` shape (speedup = compare ÷ baseline) and mistral.rs's "mean speedup vs llama.cpp" shape (claims 2, 7).
- pp512 ratio is **reported, labelled "includes chat template + session setup"**, and not a target until gen2 has a template-free token-in path (none found in the public API today). The bench file already says this; the table must too.
- State in one line what gen2 adds that llama-bench does not run: the sampler (greedy argmax), the token puller/channel scan, session bookkeeping. llama-bench's decode loop feeds random tokens with no sampler, so a ratio of exactly 1.00 is not the ceiling of "correct", it is a red flag (the bench file's `≥1.05 → check your settings` guard stays).
- How peers do it: llama-cpp-python does not publish an overhead figure at all and its famous "overhead" turned out to be a default flag (claim 9) — so publish gen2's *defaults* in the row (`n_batch`, `fa`, `kv type`) to pre-empt that. mistral.rs publishes ratio tables backed by a per-release report of commands, revisions and host metadata (claim 7) — mirror the report, not just the table.

### (4) Rot control
- Runs are by hand on the owner's machines (CI has no trustworthy GPU — claim 12, and llama.cpp's own hosted attempt rotted — claim 21).
- Every run writes `benches/results/<machine-id>/<YYYY-MM-DD>-<gen2-short-sha>.json` and it is committed. `<machine-id>` is a stable slug (`m4pro-20c-64g`, `rtx3080-win`, `pi5-8g`), not a hostname.
- The README table lives between `<!-- bench:begin -->` / `<!-- bench:end -->` markers and is **generated** from those files by a small binary (`cargo run --bin bench-table`, std-only; same no-criterion stance as `hot_paths.rs`). Humans never edit the table.
- A CI job `bench-freshness` (ubuntu, no GPU, seconds) fails when any of these is true:
  1. regenerating the table produces different bytes than the README (table edited by hand, or JSON added without regeneration);
  2. any result marked `valid: true` in the table has `llama_cpp_commit` ≠ the submodule sha resolved from the `llama-cpp-rs` rev in `Cargo.toml` (pin moved without a rerun) — the generator resolves this by reading `Cargo.toml` and a committed `benches/results/llama-cpp-pin.txt` that the run script writes; the check is that the two agree;
  3. the newest valid result for the primary machine is older than **90 days** (quarterly rerun cadence; the weekly `schedule` already exists to host this and cost nothing);
  4. a result's `schema_version` is unknown to the generator.
- Wave 2 (TanStack pattern, claim 20): stale a cell when `src/backend/llama/**`, `src/session_rt/**`, `src/generation/**` or the pin changed since the result's `gen2_commit`; docs-only commits do not stale. Not wave 1 because it needs `git` in the generator and a path list that will be argued about.
- Tools: **`llama-bench -o json`** is the reference and the schema donor; **criterion is not adopted** (own FAQ says CI numbers are noise; the repo already declined the dependency); **hyperfine only for cold model-load wall time** if that row is ever added (claim 13); **`mlx_lm.benchmark -p 512 -g 128 -n 5`** is the MLX-side reference for the cross-backend table, with the caveat that it feeds random token ids and reports averages only, so gen2's MLX row must be tokens-in and reported as median from its own samples.

### (5) What r/LocalLLaMA / HN will call out
Each is a row in the rubric below, so the table cannot ship with the flaw:
- different llama.cpp builds on the two sides (today's state);
- no build commit, no `-fa`/`-ngl`/batch flags, no GGUF hash (hiesch, #21494);
- mean without stddev, or a single run (#21494, MLX BENCHMARKS.md);
- laptop numbers with no thermal statement (#4167's own caveat);
- different quants across backends presented as an engine comparison (Ollama NVFP4 vs Q4_K_M; MLX 4-bit g64 vs GGUF Q4_K_M — claim 17, 19);
- a MoE model chosen because it flatters (Qwen3-35B-A3B in both Ollama's and #21494's tables);
- "faster than the reference" without an explanation (the bench file already refuses this);
- comparing through an HTTP UI to a CLI (llama-cpp-python #1831);
- a wrapper hiding a non-default setting (`logits_all`) — publish defaults in the row.

## RUBRIC — what a published row must satisfy (re-testable by the generator)

| Criterion | Weight | Check | Evidence |
|---|---:|---|---|
| Same llama.cpp build both sides | 25 % | `llama_bench.build_commit` prefix-matches pinned submodule sha; generator refuses otherwise | claims 1, 2, 3; alignment audit |
| Same work both sides | 15 % | `n_prompt=512, n_gen=128, n_gpu_layers, flash_attn, n_batch, n_ubatch, type_k, type_v, n_threads` identical in both halves of the JSON | claim 2 (llama.cpp's key fields) |
| Spread reported and bounded | 15 % | `n≥5`, `median` and `stddev` from raw samples on both sides; `stddev/median ≤ 0.10` or row marked invalid | claims 1, 3, 18 |
| Thermal validity | 10 % | first/last reference tg128 within 5 %; `therm_before/after` recorded; battery machines used `--delay 30` | claims 3, 14 |
| Artifact pinned | 10 % | GGUF sha256 + bytes + (repo, revision) present | claims 11, 19 |
| Machine fingerprint | 10 % | chip, core counts, RAM, OS version, GPU driver (CUDA) / Metal toolchain, gen2 commit, rustc | claim 7 (mistral.rs host block), claim 5 |
| Overhead framed as ratio with absolutes | 5 % | table shows `gen2 t/s`, `llama-bench t/s`, `ratio %` | claims 2, 7 |
| No cross-quant comparison unlabelled | 5 % | cross-backend rows carry `quant` per side and a footnote when they differ | claims 17, 19 |
| Freshness | 5 % | newest valid row ≤ 90 days; table bytes == generated | claims 12, 20, 21 |

References to mirror
- Product/protocol: `ggml-org/llama.cpp:tools/llama-bench/README.md` (flags, JSON schema, warmup/averaging semantics) and discussion #4167 (`-p 512 -n 128 -ngl 99 --delay 30` at a named commit).
- Implementation: `ggml-org/llama.cpp:scripts/compare-llama-bench.py` (identity-key matching, compare÷baseline) and `scripts/compare-commits.sh` (build both sides from source, `-o sql`).
- Report shape: `EricLBuehler/mistral.rs:releases/v0.8.2/report.md` (commands, revisions, "Versions and Commits" table, per-host metadata block) — this is what `benches/results/README.md` should look like.
- Rot control: `TanStack/charts:benchmarks/comparison/README.md` (committed JSON + generated md + stale-cell rejection).
- Cross-backend reference tool: `ml-explore/mlx-lm:mlx_lm/benchmark.py` (`-p -g -n --delay`, random-token prompt, warmup 1).
- Negative examples to avoid: `ollama.com/blog/mlx` (different quants, one date, no n); `ggml-org/llama.cpp#21494` (no stddev, no command, no bisect).

## Alignment audit — `benches/backend_parity.rs` against the research

Aligns: same file, greedy, median not mean, settle between sides, spread ceiling, "faster than reference → suspect", refuses to run silently without a comparison, no criterion.

At risk / would violate:
1. **Reference build mismatch** — uses whatever `llama-bench` is on `PATH` (brew b9000) against vendored b9553. Violates rubric row 1. Fix: locate the vendored submodule via `CARGO_HOME` + the rev from `Cargo.toml`, build `llama-bench` into `target/llama-bench-<sha>/`, or accept `GEN2_LLAMA_BENCH=/path` and *verify* `build_commit` from `-o json`.
2. **Non-standard sizes** — `-p 128 -n 64`. Nobody can line that up with pp512/tg128. Fix: 512/128.
3. **Markdown scraping** — parses the `md` table and reconstructs a fake two-point spread from `± stddev`. Fix: `-o json`, use `samples_ts`, compute the same median/stddev on both sides.
4. **Prompt is text through the chat template** — `"token ".repeat(128)` via `.user()`; prompt token count is whatever the template produces. Rate is computed from the engine's own count so it is not *wrong*, but it is not the same work as llama-bench's 512 raw tokens. Wave 1: label pp as informational (already done in the doc-comment; must reach the README). Wave 2: token-in path.
5. **No artifact written** — prints only. Everything in §(4) depends on a JSON result with the fingerprint fields.
6. **No thermal bracket** — one 2 s settle. Add the first/last reference bracket and `pmset -g therm` capture.
7. **Ratio flags are hard-coded prose** (`≥0.9 → "not in the way"`). Keep for the console; the README shows numbers, not verdicts.

## PREMORTEM

1. *Expert annoyance:* "you compared against a llama-bench built with different cmake flags." → the run script records `backends` and `gpu_info` from llama-bench JSON and gen2 records its own feature set; generator refuses rows where `backends` disagrees with gen2's feature (`metal`/`cuda`).
2. *No paper, practitioner favourite:* the `-d`/depth sweep (tg at 4k–16k context) is what mistral.rs and #21494 now report and what users feel in long chats. Wave 1 stays at depth 0 to match #4167; wave 2 adds `-d 4096`.
3. *What people actually use:* llama-bench, `mlx_lm.benchmark`, `mistralrs bench`, `ollama run --verbose`. Only the first two are references here; `mistralrs bench` becomes the reference for gen2's mistral.rs backend when that row is added.
4. *Strongest counterexample:* if gen2's ratio is ~1.00 the whole table looks like vanity. Answer: the table exists to keep it there; the freshness gate is the point, the number is not.
5. *Not searched:* r/LocalLLaMA directly (blocked); Windows-specific llama-bench CUDA quirks; Pi 5 CPU numbers (llama.cpp has Pi tables, not fetched). Pi 5 is a CPU row and needs `-t` pinned to 4.
6. *User's stack changes the answer:* yes — macOS 26 broke the in-tree MLX path once (memory note), so the cross-backend MLX row must record the Metal Toolchain version and MLX commit, or its rot is invisible.
7. *Wrong in 30 days:* llama-bench's JSON field list changed recently (`use_direct_io`, `fit_target`, `no_host` are new); the generator must tolerate unknown fields and only *require* the identity + samples subset.

## RECEIPT

**Claim:** Wave 1 publishes a llama-bench-shaped overhead table — pp512/tg128 medians ± stddev from five repetitions after one warmup, gen2 versus a `llama-bench` built from the *same vendored llama.cpp commit*, headline `tg128 ratio` with absolutes beside it, produced from committed JSON result files carrying a machine fingerprint, artifact hash and thermal bracket, rendered into the README by a generator, and guarded by a GPU-less CI job that fails on table drift, pin drift, or >90-day age. Criterion and hosted-runner benchmarking are rejected; peak memory and depth sweeps are wave 2.

- **Question:** which methodology makes a public gen2-vs-llama.cpp table trustworthy and non-rotting — deep (trust, public, hard to un-publish).
- **Sources:** graded table above; A-grade anchors are llama-bench README, `compare-llama-bench.py`, discussion #4167, `mlx_lm/benchmark.py`, criterion FAQ; B: mistral.rs v0.8.2 report, TanStack rot control, hyperfine; C: llm-tracker, famstack, NVIDIA forum, mlx #3896; negative examples Ollama MLX post, llama.cpp #21494.
- **Coverage:** primary · academic (thin) · practitioner · deployment · failure · user-context all hit; freshness swept (2026 sources incl. Ollama-MLX March 2026, llama.cpp #21494, mistral.rs v0.8.2); snowball + lateral read done; premortem run. Reddit not reachable.
- **Recommendation — default I'll take:** everything in §(1)–(4) marked wave 1. Concretely: `benches/backend_parity.rs` grows to `-p 512 -n 128 -r 5`, `-o json` on the reference, build-or-verify the reference from the vendored sha, write `benches/results/<machine>/<date>-<sha>.json` (`schema_version`, `gen2_commit`, `llama_cpp_rs_rev`, `llama_cpp_commit`, `machine{chip,cores,ram,os,gpu,driver,rustc}`, `model{path_basename,sha256,bytes,type,quant,n_params}`, `settings{n_ctx,n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,type_k,type_v}`, `runs{gen2:{pp:{samples,median,stddev},tg:{…},ttft_ms:{…}}, reference:{pp,tg}}`, `thermal{therm_before,therm_after,ref_tg_first,ref_tg_last}`, `valid`, `reason`); a `bench-table` bin; README markers; `bench-freshness` job in `ci.yml`.
  **Your call:** (a) 90-day age vs 60; (b) whether wave 1 includes the RTX 3080 row (needs a Windows build of llama-bench from the same sha — extra day) or ships Mac-only; (c) whether the LiteRT-LM "1.7x" sentence is removed until it has a result file (recommended: remove).
- **References to mirror:** listed under Rubric. Rubric: nine criteria above, all machine-checkable by the generator.
- **What's NOT proven (miss-risk):** r/LocalLLaMA methodology threads (blocked) — the "dishonesty" list is assembled from GitHub issues, practitioner posts and vendor blogs instead; llama-bench prompt/decode token semantics (random-token fill, no sampler) are from memory of the source, not re-read this session; whether gen2's llama puller sets `first_token_us` for TTFT was not verified in this session (`grep` found setters only in external-api/onnx/mlx pullers); Windows CUDA build-from-submodule not attempted; Pi 5 not researched.
- **Re-check trigger:** the `llama-cpp-rs` rev in `Cargo.toml` changes (CI will fail the table on purpose); llama-bench JSON schema changes (generator parse failure); mlx-lm changes `benchmark.py` output or adds stddev; or 2026-12-01, whichever first.
