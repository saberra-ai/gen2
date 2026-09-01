# Test plan

gen2 is a stateful runtime with interchangeable execution backends. The thing
worth proving is not line coverage: it is that the runtime honours its
contracts at the seams, and that every backend honours the same one.

The organising idea: make a fake backend behave like a psychopath — fail
during load, block forever, poison a session, emit tokens after a terminal
event, vanish mid-stream — and prove gen2 still obeys its public contract. The
real neural network is then only needed to prove each concrete backend
implements that same contract.

## Tiers

| Tier | What | When |
| --- | --- | --- |
| T0 | unit, property, fake backend, hermetic HTTP fixtures | every commit |
| T1 | feature matrix, clippy, docs, examples | every PR |
| T2 | one tiny real GGUF | every PR |
| T3 | Metal / mlxcel / ONNX / Candle parity | nightly |
| T4 | fuzzing, sanitizers, soak | nightly |
| T5 | benchmarks | scheduled |

T0 must finish well under a minute.

## 1. A deterministic FakeBackend — the centre of the suite

`Backend` / `BackendSession` / `TokenPullerDyn` is already a clean boundary.
A programmable fake against it turns behaviour that currently needs a real
GGUF into microseconds.

The fake must be able to misbehave on purpose: fail a load, fail
`start_session`, fail a pull at token N, poison a session, block until a
barrier, report arbitrary capabilities, and record every call it received.

The controller needs an injection seam — `ControllerState::new` builds
`Engine::new()` today, and `Backend` is not `Send`, so the seam takes a
`Send` factory that constructs the engine on the loop thread while the test
keeps an `Arc` handle to the script.

## 2. Controller invariants, not controller methods

Organise around what must never happen:

    terminal → token                NEVER
    terminal → terminal             NEVER
    paused → token                  NEVER
    failed start → resident session  NEVER
    shutdown → surviving session     NEVER

with one reusable trace assertion so the same contract applies to every
backend.

Target: the full state machine, not the handful of pure transitions currently
covered.

## 3. Session and KV semantics, property-tested

Keep the existing example-based tests; add generated operation sequences
(push / open / fork / edit / serialize / note_tools / note_model / clear) and
assert the invariants that must hold across any ordering.

## 4. Truncation as an algorithm

Property territory: output fits the window, order is preserved, a system
message stays first, more context never drops more messages, a larger
generation reserve never drops fewer, and it always terminates.

## 5. A backend conformance suite

The crate's claim is one API over many backends, so hold every backend to one
contract, and to the capability contracts it opts into (embeddings, images,
KV snapshots).

## 6. External API: a large hermetic suite

`backend-external-api` is the default feature and `mockito` is already a dev
dependency, yet the wire format is untested. Request shape (URL, auth headers,
system-prompt placement, sampling fields, stop sequences) and SSE parsing
(blank lines, `event:` fields, malformed JSON, unknown fields, role-only
chunks, finish reasons, EOF without `[DONE]`, HTTP 400/401/429/500) are all
model-free and should run in seconds.

## 7. Agent contracts, on scripted model behaviour

Live agent tests are probabilistic by construction — the concurrency test has
to accept the model deciding to make one call. That is fine as a smoke test
and insufficient as the correctness test. Script the tool calls instead, and
test dispatch, budgets, approval, steering, parallelism policy, and that a
final-answer grammar constrains only the final answer.

## 8. Tool search: a fixed corpus regression

Retrieval quality has no test that fails when BM25 tokenisation or RRF
weighting silently degrades. A pinned corpus of tools and queries, asserting
the target appears in the top k, is that test.

## 9. Fuzz the parsers

`bundle/gguf.rs` parses attacker-controlled lengths and nested arrays;
`generation/reply_parts.rs` transforms model output. Neither may panic, hang,
or allocate unboundedly on arbitrary bytes. Same for the SSE and tool-call
parsers and the KV codec.

## 10. KV persistence and corruption

`decode(encode(x)) == x`, plus rejection of truncated blobs, wrong magic,
version mismatch, checksum failure, and fingerprint mismatch.

## 11. Fit and hardware on synthetic inputs

No test should ask how much RAM the CI runner happens to have. Build
`HardwareProfile` values directly and assert monotonicity: more RAM never
un-fits a fitting model, more context never improves fit.

## 12. The model zoo is executable configuration

Validate it like code: unique ids, known backends, positive memory floors,
parseable URLs, no contradictory platform combinations.

## 13. Public API tests act like a third party

`tests/` must not reach for `pub(crate)`. That is what catches an essential
type going private, which `#![warn(unnameable_types)]` already cares about.

## 14. The feature graph is part of the contract

Some combinations are intentionally invalid (`backend-mlx` + `backend-mlxcel`,
no backend at all). CI builds the valid ones and asserts the invalid ones fail
to compile.

## 15. Live tests split per backend

`live_inference.rs` is gated on `backend-llamacpp`, so all model-backed proof
currently exercises one backend. Split it, and have each implement the same
stable subset. Assert engine properties, never model intelligence.

## 16. A tiny pinned model for CI

One small GGUF, pinned by URL and SHA256, cached. A PR pipeline should not
download five 5 GB models.

## 17. Soak and leak

Runtimes fail from lifecycle bugs more than algorithm bugs. Repeated engine
create/drop, session churn, pause/resume storms, receiver disconnects, model
reloads — asserting threads and resident sessions return to baseline.

## 18. Performance is a separate suite

Benchmarks tracked over time, never a PR gate. Two of them: `hot_paths`
for the per-turn work that does not involve a model, and
`backend_parity` for the question nothing else asks — what the wrapper
costs against llama.cpp's own `llama-bench` on the same file.

That one found a third of throughput being spent idle: the controller
waited `tick_idle` for a command before every token pull, so 2ms landed
on each 4ms token. 73% of the reference before, ~82% after. The rest is
work a streaming API does and `llama-bench` does not — detokenising to a
`String`, a channel send per token, scanning for reply parts.

## Build order

1. ~~FakeBackend and the controller state machine~~ — done
2. ~~Hermetic OpenAI / Anthropic protocol suite~~ — done
3. ~~Feature matrix in CI~~ — done
4. ~~Fuzz GGUF, reply parts, KV~~ — done (SSE covered by hermetic
   fixtures rather than a fuzz target; the parser is `pub(crate)`, and
   widening the public API to reach it is a design decision, not a
   test's to force)
5. ~~Backend conformance suite~~ — done. Splitting the live tests per
   backend is what remains, and it is blocked on having models for the
   other backends rather than on the suite
6. ~~Property, soak, and performance work~~ — done

## Where it got to

857 unit tests and 41 integration tests, green, clippy and rustdoc
clean, and the whole non-live suite runs in about two seconds.

Nine bugs, found by the tests rather than by reading:

- A hostile GGUF file could abort the host process two ways — an
  unbounded allocation from an attacker-declared string length, and
  unbounded recursion on nested arrays. Both reachable from the public
  `ModelInfo::read`, and neither catchable by a `Result`.
- A completed answer from an external provider was reported to the
  caller as cancelled by the user, on the chunk shape llama.cpp's
  server, vLLM and Together all produce.
- `ApprovalMode::AskOnRisky` asked about every tool, because no tool
  could declare `Risk` despite it being exported and documented.
- `auto_tune_ctx` under-charged KV fourfold on a header without
  architecture metadata, handing back a context that OOMs at load.
- `fits` pointed callers at a smaller context that was also refused.
- A header without layer counts priced context at nothing, so a
  four-million-token window fit on any machine.
- Header-derived dimensions overflowed, wrapping in release into an
  estimate that passes an admission check it should fail.
- A truncated GGUF parsed as valid.
- The KV blob header sat outside its own integrity check.
- Tool-search ties broke on registration order, so adding one tool
  could push another out of reach.

## Still not proven

- mlxcel, ONNX and Candle satisfy the half of the conformance contract
  that needs no weights, and none of the three has been shown to
  generate a single token. The suite says so out loud on every run
  rather than skipping quietly, and `unverified_backends` fails if that
  list goes stale. Point `PIO_TEST_ONNX_MODEL` or `PIO_TEST_CANDLE_MODEL`
  at a model and the generating half runs.
- MLX is verified as of 2026-09-01: it builds on macOS 26.3 with the
  Metal Toolchain installed, and decodes to its cap against an MLX
  safetensors bundle. The note in Cargo.toml claiming its build was
  broken was stale by four months, which nothing would have caught,
  because nothing tried.
- No sanitizer run.
- The KV store's writes are not atomic. Digests turn a torn write into a
  cache miss rather than corruption, but a crash mid-write leaves a dead
  file until the budget sweep reclaims it.
- The eviction chaos test needs macOS file flags, so its invariant is
  unguarded on Linux CI.
