# Exact local text requests

`Turn::json_schema::<T>()` prepares a schema without discarding the response.
The caller retains cancellation, terminal reason and statistics, and decodes and
validates the returned text explicitly. Existing `structured::<T>()` remains a
convenience decoder with its previous behavior.

```rust,ignore
let mut conversation = gen2::Session::new().with_system("Use only supplied evidence.");
let turn = model.turn(&mut conversation)
    .user(evidence)
    .max_tokens(1024)
    .json_schema::<Summary>()?
    .full_context();
let cancellation = turn.canceller();
let response = turn.run()?;
// Reject cancellation, non-Stop terminal reasons, invalid JSON and unsupported claims.
let budget = response.stats().request_context().expect("enforced native budget");
assert!(budget.prompt_tokens + budget.output_tokens <= budget.capacity_tokens);
```

`full_context()` currently supports the local llama.cpp text path. It rebuilds
the entire supplied conversation each turn and reserves the explicit positive
output budget. The backend renders system/meta/tool/thinking inputs once,
tokenizes that exact prompt, queries the allocated native context and rejects
overflow before prefill. It bypasses conversational truncation and cache restore.
The same frozen tokens are then prefilled. Completed accounting includes
`RequestContext { prompt_tokens, output_tokens, capacity_tokens }`; the model's
training context reported by `Model::info()` is a different value.

A requested grammar is compiled before strict prefill and reused for generation.
Strict matcher errors fail closed. A strict pull cannot be activated on a
legacy-prepared session or change its frozen sampling settings. Native decode
also checks capacity. Partial output is never proof of a completed artifact;
callers must check terminal reason/cancellation and perform their own validation.

Strict requests do not read/write keepwarm KV files and do not use warm append,
media or speculative decoding. Low-level PauseChat/ResumeChat commands have no
reply channel: on strict requests they are refused without changing the live
stream; use cancellation and a fresh request instead. Direct strict session
pause/resume similarly leave state unchanged; a second pull is rejected.
Cancellation cannot interrupt synchronous model loading/preparation and is not a
durable job implementation.

Strict prompt text, source messages, source/output content tokens and grammar
error payloads are excluded from the affected diagnostic logs. Counts, timings
and terminal control-token IDs remain diagnostic metadata. This is not a blanket guarantee
about host logging, output hooks, other backends or non-strict requests. Parsed
chat templates now own their source through a shared environment; their final
handle releases the allocation rather than leaking on repeated cold turns.

The context check occurs during execution before inference; a separate read-only
prepare/preview API is not provided yet. Optimized strict warm/restore/resume
requires independent accounting and qualification. ASR and default non-strict
request behavior are unchanged.
