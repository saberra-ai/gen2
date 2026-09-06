# Benchmark results

Every number in the README's benchmark table comes from a JSON file in this
directory, and every file was written by `benches/backend_parity.rs` on a
machine its author owns. Nothing here is edited by hand. The design and the
sources behind it: `docs/plans/research/04-benchmark-method.md`.

## What is measured

Two sides, the same GGUF, the same llama.cpp commit, the same machine, the
same minute:

| Side | What runs |
| --- | --- |
| **llama-bench** | llama.cpp's own benchmark, built from the exact commit `llama-cpp-sys-2` vendors (`benches/build-llama-bench.sh`). No sampler, random token ids, the ceiling for the code gen2 links. |
| **gen2** | The public API: `Engine::builder().model(path).context(2048).build()`, then `engine.infer(prompt).max_tokens(128).greedy().run()` — controller, session runtime, chat template, token puller, reply scanner, greedy sampler included. |

Three numbers per row, as llama-bench defines the first two:

- **pp512** — 512 prompt tokens processed in batches, tokens ÷ seconds. Through
  gen2 the prompt is text through the chat template (calibrated so the
  engine's own count of what it prefilled is 512 ± 2 tokens), and the time
  includes template rendering, tokenisation and session setup. That is why the
  README labels it "includes chat template + session setup" and does not
  treat it as a target.
- **tg128** — 128 tokens generated one at a time at batch 1, tokens ÷ seconds.
  gen2's rate is on the engine's own clock, `(decode_tokens − 1) ÷
  (decode_tokens ÷ avg_tps − first_token)`: `avg_tps` spans prefill start to
  the final stats, so this is the same span llama-bench times, and the first
  token is charged to prefill. **The headline ratio is `gen2 tg128 ÷
  llama-bench tg128`.** Two companions live in the JSON only: `tg_wall`, the
  same tokens over the caller's wall clock after the first token (context
  creation and teardown included — what a host application sees), and
  `overhead_ms`, `run()` wall time minus the engine's span — what a turn costs
  around the decode loop, in milliseconds rather than smeared into a rate.
- **TTFT** — gen2 only: prefill start to first token, from the engine's
  `ExecutionStats::first_token_us`. llama-bench has no such number and none is
  derived for it.

## Protocol

- One untimed warmup, five measured repetitions on each side (llama-bench
  `-r 5` with its default warmup; the harness mirrors it).
- **Median ± sample standard deviation (n−1)** from the raw per-repetition
  samples, computed identically on both sides (`samples_ts` from
  `llama-bench -o json`; the harness's own five samples for gen2). Not the mean,
  which one disturbed run on a shared GPU can drag anywhere.
- Same work: `-p 512 -n 128 -b 512 -ub 512 -t 4 -ngl 99 -fa auto -ctk f16
  -ctv f16`. Those are gen2's effective defaults — `DEFAULT_BATCH_SIZE = 512`
  for `n_batch`/`n_ubatch` (`src/backend/llama/session.rs`), `n_threads` left
  at llama.cpp's context default of 4 (`threads: None` in
  `src/backend/llama/llama_config.rs`), flash attention `AUTO`, every layer
  offloaded on a GPU build, f16 KV. gen2 additionally `mlock`s the weights.
- Greedy decoding on gen2 (`temperature 0`, `LlamaSampler::greedy()`), batch 1.
  A ratio ≥ 1.05 is printed as a warning: llama-bench runs no sampler, so gen2
  being faster means the two sides were not given the same work.
- `--delay 5` between llama-bench's tests on mains power, `--delay 30` on a
  machine with a battery; a 5 s settle between the two sides.
- **Thermal bracket**: llama-bench tg128 runs first and again last. If the last
  median is more than 5 % below the first, the file is written with
  `"valid": false, "reason": "throttled…"` and the table ignores it.
  `pmset -g therm` is recorded before and after on macOS.
- **Spread ceiling**: if either side's tg128 stddev exceeds 10 % of its median,
  the file is `valid: false` ("noisy…").
- Concurrent compilers or benchmarks on the machine are recorded in
  `warnings` (`ps` scan), not acted on.

## Reproducing a row

```sh
benches/build-llama-bench.sh                       # once per llama-cpp-sys-2 bump; asserts the source
PIO_TEST_MODEL=/path/to/model.gguf cargo bench --bench backend_parity
cargo bench --bench bench_table -- --write         # regenerate the README table
```

`build-llama-bench.sh` resolves the llama.cpp commit from the
`utilityai/llama-cpp-rs` tag matching the pinned crate version, proves every
file the crate vendors is byte-identical to that commit (the crate ships a
subset of llama.cpp without `tools/`, so the reference cannot be built from
the registry copy itself), configures with the same cmake options the crate's
`build.rs` uses (`GGML_NATIVE=OFF`, `GGML_OPENMP=ON`, `GGML_BLAS=OFF` and
`GGML_METAL=ON` on macOS, static, no curl), builds `llama-bench` into
`target/llama-bench/`, and writes `llama-cpp-pin.txt` here. The harness reads
the pin, checks it against `Cargo.toml`, and checks every `build_commit`
llama-bench reports against it before recording anything. `GEN2_BENCH_CUDA=1`
adds `GGML_CUDA=ON` for the NVIDIA row.

Useful knobs: `GEN2_LLAMA_BENCH=/path` (a reference built elsewhere; still
verified), `GEN2_BENCH_MACHINE=slug` (result directory name), `GEN2_BENCH_NOTES`
(free text into the file), `GEN2_BENCH_NO_WRITE=1` (measure, print, write
nothing).

## Files

`<machine-slug>/<YYYY-MM-DD>-<gen2-sha7>-<model-slug>.json`

- `machine-slug` is derived from the chip, core count and RAM
  (`m4pro-14c-64g`), not the hostname, so a machine keeps its identity across
  renames. Override with `GEN2_BENCH_MACHINE`.
- `gen2-sha7` is `HEAD` when the run happened. The commit that adds the file
  is necessarily a later one, and `gen2.dirty` records whether the tree had
  uncommitted changes (the result file itself is untracked at that point and is
  not counted).
- The model slug is in the name because one date and one commit produce one
  file per model.

`llama-cpp-pin.txt` — written by the build script; read by the harness and the
generator. It must agree with the `llama-cpp-sys-2` version in `Cargo.toml`, and
every `valid` result must carry its `llama_cpp_commit`, or the generator refuses
to render. Moving the pin without rerunning is therefore a build failure, not a
stale table.

## Schema (`schema_version: 1`)

| Key | Contents |
| --- | --- |
| `schema_version` | `1`. The generator rejects versions it does not know. |
| `valid`, `reason`, `warnings`, `notes` | Whether the row may enter the table and why not; non-fatal observations (ratio ≥ 1.05, concurrent load); operator notes from `GEN2_BENCH_NOTES`. |
| `date`, `started_at`, `finished_at` | UTC. |
| `protocol` | `n_prompt`, `n_gen`, `warmup`, `repetitions`, `sampling`, `batch`, `statistic`, `delay_s`, `settle_s`, `thermal_drop_limit`, `spread_limit`. |
| `gen2` | `commit`, `short`, `dirty`, `version`, `features` (`backend-llamacpp`, `metal`, `cuda`, …), `rustc`, `profile`. |
| `llama_cpp` | `commit` (full sha the pin names), `build_commit` (what the binary reported), `build_number`, `llama_cpp_sys_2` (crate version), `backends` (from llama-bench, e.g. `Metal,BLAS,CPU`), `reference_binary`. |
| `machine` | `slug`, `label`, `chip`, `cpu_cores`, `perf_cores`, `eff_cores`, `ram_bytes`, `os`, `cpu_info` and `gpu_info` (as llama-bench saw them), `power_source`, `battery`, `therm_before`, `therm_after`. |
| `model` | `path`, `file`, `name`, `quant`, `bytes`, `sha256`, `type` / `n_params` / `size_reported` (from llama-bench), `hf_repo` (null when the file was not fetched by gen2). |
| `settings` | The identity fields both sides share: `n_batch`, `n_ubatch`, `n_threads`, `n_gpu_layers`, `flash_attn`, `type_k`, `type_v`, `use_mmap`; plus gen2's `n_ctx_gen2`, `use_mlock_gen2`, the calibrated `gen2_prompt_tokens`, per-repetition `gen2_decode_tokens` and `gen2_finish`. |
| `runs.gen2` | `pp`, `tg`, `ttft_ms` — each `{samples_ts, median, stddev, n}` — and `wall_s` per repetition. |
| `runs.reference` | `pp`, `tg` in the same shape; `command` (the exact llama-bench invocation); `raw` (llama-bench's own JSON rows, untouched, so any field it reports is available later). |
| `ratio` | `tg`, `pp`, and the pp caveat. |
| `thermal` | `ref_tg_first`, `ref_tg_last`, `ref_tg_last_command`, `drop`, `therm_before`, `therm_after`. |

The generator requires only `schema_version`, `valid`, `date`, `gen2.short`,
`machine.{slug,label}`, `model.{name,quant}`, `llama_cpp.{commit,build_number}`
and the `median`/`stddev` of `runs.*.{pp,tg}` and `runs.gen2.ttft_ms`; it
tolerates any extra field, which is how a future llama-bench column or a peak
memory number gets added without a schema bump.

## Not measured (yet)

- Peak memory: macOS RSS is not comparable to Metal-resident memory, and
  llama-bench does not emit one. When added it will be `phys_footprint` (macOS)
  / `ru_maxrss` (Linux) / working-set peak (Windows), labelled by method.
- Depth (`-d 4096`) — long-context decode. Rows are at depth 0.
- Sustained thermal behaviour beyond the first/last bracket.
- RTX 3080 (Windows, CUDA) and Raspberry Pi 5 (CPU, `-t 4`) rows: machines
  exist, no run yet.
- A token-in path for gen2 that would make pp512 the same work on both sides.
