# In-process DFlash2

## Scope and acceptance

Integrate DFlash2 into Gen2's llama.cpp backend, without an HTTP bridge, and
compare it to the local llama-server. Preserve the public model/session APIs:
validate drafts, support thinking off, preserve greedy output with drafting
disabled, handle cancellation, and permit multiple turns. Leave unrelated
Bark/ONNX and build jobs running during measurements.

## Usage

Initialize the pinned native dependency before building:

```sh
git submodule update --init --recursive
cargo build --no-default-features --features llama-dflash2,metal
```

On this macOS 26.3 machine, builds also use `MACOSX_DEPLOYMENT_TARGET=26.3`.

```rust,no_run
use gen2::{Runtime, advanced::generation::{DFlash2Config, Settings, ThinkingMode}};

let mut settings = Settings::default();
settings.system.dflash2 = Some(DFlash2Config {
    model_path: "/path/to/Qwen3.8-27B-DFlash2-Q8_0.gguf".into(),
    max_tokens: 5,
});
settings.system.kv_cache_type = Some("q8_0".into());
let runtime = Runtime::builder().settings(settings).build()?;
let model = runtime.load("/path/to/Underdog-Saluki-27B-1.0-IQ2-mix-MTP.gguf")?;
let reply = model.generate("Explain an LRU cache.").greedy()
    .reasoning(ThinkingMode::Off).max_tokens(128).run()?;
# Ok::<(), gen2::Error>(())
```

The configured draft is loaded with the target. Unload before changing it;
per-session changes are rejected. Draft and target are owned together and
each session owns its own target and draft contexts. `ExecutionStats` reports
`spec_drafted` and `spec_accepted` through `Response.stats.execution()`.

This first integration speculates only for greedy generation without a grammar.
Nonzero temperature, grammar constraints, or advanced `SpeculativeMode::Off`
use ordinary target decoding; they still maintain draft feature state. Thinking
off is independent of speculation. The trained block and available context
bound drafting. Batch capacity must fit the anchor plus all proposals;
smaller explicit batches are rejected before entering native code.
Persistent KV snapshots are explicitly unsupported with a
draft configured: saving just the target would lose the paired draft state.
Multimodal generation is also rejected for this experimental path.

Dimension/vocabulary compatibility and DFlash2 selector/block metadata are
checked. This cannot prove a draft was trained for a particular fine-tune;
choose the appropriate target/draft family. This 27B draft is not for Woof 4B.
Admission estimates include draft weights and its per-session context; estimates
remain heuristics, not measured memory bounds.

## Native dependency

Rust bindings are vendored from `llama-cpp-{2,sys-2}` 0.1.156. Their default
native snapshot did not support DFlash2. The native submodule is pinned to
`847f447c310a3784d623a953e6f0bb07f04c0f27`, matching the local server pilot.
Both ordinary and speculative llama.cpp builds use this native revision.
The unmodified native helper implements feature extraction, convolution and
selector drafting. A small binding adapter supplies its DFlash2 initializer
and adapts changed native batch/JSON/MTMD helper APIs.

The caller removes the temporary noise block before verification, injects
target features, accepts the matching greedy prefix and rolls both caches
back. Dropping a partially consumed block removes un-emitted verified tokens;
a native failure discards the context rather than reusing damaged state.

## Verification and benchmark

```sh
GEN2_DFLASH2_TARGET=/path/to/target.gguf \
GEN2_DFLASH2_DRAFT=/path/to/draft.gguf \
cargo test --no-default-features --features llama-dflash2,metal \
  --test dflash2_public -- --ignored --test-threads=1

cargo run --release --no-default-features --features llama-dflash2,metal \
  --example dflash2_bench -- /path/to/target.gguf /path/to/draft.gguf 2 128
# Replace the draft path with '-' for ordinary decoding.
```

The example emits machine-readable `BENCH_JSON` records. Fresh sessions,
8192 context, 512 batch, four CPU threads, GPU offload, flash attention,
Q8 target KV and F16 draft KV, greedy sampling, seed 42, and 128 output tokens are used for both
paths. There is one first request and one warmup before measured trials.
Decode TPS is `(tokens - 1) / (last visible delta - first visible delta)`;
end-to-end TPS includes session creation/prefill and final cleanup. Visible
TTFT includes context creation and prefill. Load is warm-filesystem weight
loading for Gen2, but process-start-to-ready (including native warmup) for
the server, so those load times are not interchangeable.

Counterbalanced runs compare Gen2 ordinary, Gen2 DFlash2, server ordinary,
and server DFlash2 using identical prompts and model files. Output parity
must hold across measured configurations. Concurrent unrelated jobs mean
the result is a local comparison under contention, not a clean maximum-TPS
or universal speedup claim.

For the equivalent server, use the same native revision and a fresh process:

```sh
llama-server -m /path/to/target.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -fa on -c 8192 -np 1 -b 512 -ub 512 -t 4 \
  -ctk q8_0 -ctv q8_0 --cache-ram 0 \
  --spec-type draft-dflash -md /path/to/draft.gguf -ngld 99 \
  --spec-draft-n-max 5
# Omit the second model and speculation flags for the ordinary control.
```

Send the example's LRU-cache prompt through `/completion`, with the model's
chat template ending in a closed, empty thinking block; use temperature 0,
seed 42, `n_predict: 128`, `cache_prompt: false`, and streaming. Count generated
tokens from native timings rather than HTTP chunks. Repeat one first request,
one warmup and two measured requests per process; then reverse configuration
order for a second round. The resulting four measured samples per configuration
all have 48 prompt tokens and 128 output tokens.

## Local results — 2026-10-13 UTC

Machine: Mac mini M4 Pro, 20 GPU cores, 64 GiB, macOS 26.3. Target:
`Underdog-Saluki-27B-1.0-IQ2-mix-MTP.gguf` (8,349,689,920 bytes), with
`Qwen3.8-27B-DFlash2-Q8_0.gguf` (2,056,414,816 bytes) as the draft. Both
implementations use native revision `847f447`. Four measured samples per row;
all medians below except load, which lists the two process/model-load samples.
GB means decimal GB. RSS is the median of two process peak-RSS measurements,
not a bound on total system memory or a direct GPU allocation measurement.

| Configuration | Decode TPS | End-to-end TPS | Visible TTFT | Load range | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gen2 ordinary | 13.98 | 13.12 | 650 ms | 0.68–0.73 s | 9.54 GB |
| Server ordinary | 13.15 | 12.40 | 658 ms | 0.94–1.04 s | 8.72 GB |
| Gen2 DFlash2 | 21.04 | 18.25 | 977 ms | 1.22–3.59 s | 14.83 GB |
| Server DFlash2 | 22.20 | 19.90 | 706 ms | 1.65–4.77 s | 11.78 GB |

Gen2's median decode improvement is 1.51× over its ordinary control and its
DFlash2 median is 5.2% below the server. Measured Gen2 DFlash2 decode ranges
from 20.49 to 22.90 TPS; server DFlash2 ranges from 19.47 to 24.55 TPS. These
observed peaks are not uncontended maximum throughput. All 16 measured outputs
are byte-for-byte equal, thinking is off, and both speculative configurations
accept 344 of 820 proposed tokens (41.95%).

An earlier Gen2 run reached only 15.43 median TPS. A focused probe limiting
native output allocation to the anchor plus speculative block reached 21.61
TPS, motivating the output-limit binding and final measurement. Target Q8 KV,
draft F16 KV and rollback sequence capacity are also aligned with the server.
The final integration still has higher TTFT and about 3.05 GB extra peak RSS
versus the server; memory optimization remains future work.

Machine-readable summaries and per-trial measurements are saved in
[`dflash2-m4-pro-20261013.json`](../benchmarks/dflash2-m4-pro-20261013.json).
The complete local raw report and logs remain under
`../underdog-benchmark/results/embedded-compare-20261013T033109Z-*` relative to
the repository root. Load ranges use the different definitions described
above; do not interpret them as an apples-to-apples cold-start comparison.

## Qualification

Tests run against an isolated snapshot of the staged DFlash2 changes because
other work in the shared checkout was still in progress. The native/default
bundle with `llama-dflash2,metal` passed 1,208 unit tests, all enabled integration
tests, and 74 doctests. All nine public DFlash2 tests, including the five
model-dependent tests, passed. The additional ignored compatibility-API test
passed: invalid merged session batch settings are rejected before native context
creation, and changing the loaded draft requires unloading first.
The final live rerun also passed all nine tests. The ordinary native build,
without the `llama-dflash2` feature, passed all three portable validation tests,
including explicit rejection of otherwise-valid draft configuration.

The expanded native/default plus `backend-external-api` run passed 1,300 unit
tests but failed `external_anthropic`'s system-prompt assertion. The same
native/default plus external configuration at the pre-integration baseline
`faeb3604190496d661cdf050976707e49623ce00` reproduces that failure; the pure
external baseline passes. This existing feature-dependent failure is not fixed
by this integration.

### Standards

No unresolved findings. Review-driven fixes cover paired-state error cleanup,
warm-prefill invalidation and batch/session validation; public regression tests
cover cancellation, reuse, context compaction and thinking off.

### Spec

No unresolved findings against the original request and confirmed public-API
requirements. Greedy drafting remains experimental and intentionally does not
claim compatibility with arbitrary target fine-tunes.
