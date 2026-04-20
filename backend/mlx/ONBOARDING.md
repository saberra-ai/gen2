# Onboarding a new model to the MLX backend

Lessons from bringing up Gemma 4 (E2B, 26B MoE, 31B dense+k_eq_v). Read this
before adding a new model family to `backend/mlx/`.

The order matters: each layer assumes the one above is correct. If the model
produces garbage, start from the top of this list — a broken tokenizer makes
every downstream check misleading.

## 1. Tokenizer & special tokens (`common/tokenizer.rs`)

**Check BOS resolution.** `HfTokenizer::from_dir` resolves BOS by name from
the added-vocab. If the model ships BOS under a name we don't know yet
(Gemma: `<bos>`, Llama 3: `<|begin_of_text|>`, Llama 2: `<s>`), add it to the
candidate list. Missing BOS produces **catastrophic first-token collapse**
(e.g. the model gets a different embedding at position 0 and top-1 logits
are garbage across the whole forward pass).

**Check EOS / stop set.** Models declare `eos_token_id` in
`generation_config.json` as an int OR an array (e.g. Gemma 31B lists
`[1, 106, 50]` — `<eos>`, `<turn|>`, `<|tool_response>`). The parser in
`from_dir` merges these with the hardcoded chat-EOT name list. If the model
runs to `max_tokens` on every prompt, one of its stops is missing from your
`stop_ids()` set — dump them, grep tokenizer.json, add by name or bump the
generation_config parser.

**Decode specials for template expansion.** `decode()` strips specials by
default (good for UI text), but `{{ bos_token }}` / `{{ eos_token }}` in the
chat template need the literal string. Use `decode_keep_specials()` when
feeding the template renderer — this was the Gemma 4 BOS bug.

## 2. Chat template (`backend/common/chat_template.rs` + `session.rs`)

**Does the template have `enable_thinking` branches?** Gemma 4 does; omitting
it causes the template to append a trailing `<|channel>thought\n<channel|>`
that the model wasn't trained on. Match what mlx-lm's `apply_chat_template`
produces by default:

```rust
chat_template.apply(messages, None, Some(true)) // enable_thinking=true for Gemma 4
```

**Compare the rendered prompt byte-for-byte against mlx-lm's reference.**
```python
tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
```
If the rendered strings differ, that's the bug. Check for: missing BOS,
extra channel blocks, missing system-message default, trailing newline
differences.

**Verify tokenization.** Even with the right string, check the token IDs
match. Special tokens like `<|think|>` must tokenize to a single ID; if they
split into multiple tokens, the tokenizer's `added_tokens` lookup isn't
hitting. Our tokenizer uses HF's `tokenizers` crate which generally gets
this right — but verify once per model.

## 3. Config deserialization (`model/mod.rs::ModelConfig`)

**Nested RoPE parameters.** Recent Gemma variants ship config as:
```json
"rope_parameters": {
  "full_attention":    {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"},
  "sliding_attention": {"rope_theta": 10000.0,   "rope_type": "default"}
}
```
Flat fields like `rope_theta` / `rope_local_base_freq` may be absent or
stale. `loader.rs::build_gemma4_model` parses `rope_parameters` into the
flat fields after deserialization — any new model with this shape needs the
same treatment.

**Architecture flags to check for new variants:**
- `attention_k_eq_v: bool` — full-attn layers ship without `v_proj` (V = K
  pre-norm). Gemma 31B / 26B.
- `num_global_key_value_heads: Option<usize>` — different KV-head count on
  full-attn layers vs sliding.
- `hidden_size_per_layer_input: Option<usize>` — Per-Layer Embeddings
  (E-series only). `0` or `None` = disabled; do NOT run the PLE branch.
- `enable_moe_block: bool` + `num_experts` / `top_k_experts` /
  `moe_intermediate_size` — MoE variants.
- `final_logit_softcapping: Option<f32>` — applied after lm_head.

## 4. Weight loading (`loader.rs`)

**N-D quantized weights.** `detect_bits` / `detect_group_size` must accept
3D tensors (MoE experts are `[n_experts, out, in_packed]`). Our versions do,
but if you see a 4D+ variant, extend them.

**Try multiple weight naming conventions.** Gemma 4 26B's experts are under
`experts.switch_glu.gate_proj` (+ `.weight`/`.scales`/`.biases`), not the
flat `experts.gate_proj` the handoff predicted. Always try the
mlx-lm-sanitized form first, fall back to alternatives. When loading fails
silently (Weight stays zero-initialized), output is garbage with no panic —
always log which tensors loaded successfully.

**Use `load_weight` (quantized-aware) for all projections.** Direct
`Weight::plain(w.clone())` won't handle the `.weight + .scales + .biases`
triple for quantized models. The 26B MoE path hit this early.

## 5. Attention (`model/gemma4.rs` or a new per-family file)

**Per-head dimensions differ by layer type.** Gemma 4 sliding layers use
`head_dim=256`; full-attn layers use `global_head_dim=512`. Wire
`layer.attention.head_dim` from the layer-type map, not a global constant.

**q_norm, k_norm, v_norm.** Gemma 4 has all three per-head. Note: `v_norm`
is RMSNorm **without** a learnable scale — use `RmsNormNoScale`. q_norm /
k_norm have learnable weight of size `head_dim` (per-head, not per-hidden).

**Attention scale.** Gemma 4 uses `scale=1.0` (magnitude controlled by
q_norm/k_norm), not the conventional `1/√head_dim`. Check the reference
impl.

**RoPE application order.** Q: project → reshape to `[B, L, H, D]` → q_norm
→ transpose to `[B, H, L, D]` → RoPE. K same path. V: project/clone →
reshape → v_norm (NoScale) → transpose. **No RoPE on V.**

**Proportional RoPE pairing.** If the config uses
`rope_type: "proportional"` and `partial_rotary_factor < 1.0`, the rotation
pair stride is `full_head_dim/2`, NOT `rotated_dim/2`. Build the cos/sin
tables at full width with zero angles past `rotated_dim/2` so
`rope.forward` can be called on the full-width tensor directly — don't
split-and-concat (that gives the wrong pairs).

**k_eq_v path.** When `use_k_eq_v=true` on a full-attn layer:
- Skip loading `v_proj` entirely.
- In forward: `values = k.clone()` before k_norm is applied.
- k gets `k_norm`, v gets `v_norm` (still NoScale). They diverge because
  they came from the same raw projection but went through different norms.

## 6. Forward pass plumbing (`model/mod.rs`)

**Explicit offset.** `Model::forward` takes `offset: usize`. Pass the
session's `cur_pos` — do NOT derive from `cache[0].shape[2]`. Sliding-
attention layers cap their cache at `sliding_window`, so inferring offset
from cache length stalls at the window boundary while full-attn RoPE still
needs the true position. This bug caused 26B/31B to degrade after the first
window crossing.

**forward_all for speculative decoding.** Returns per-position logits (not
just the last). Required for the puller's n-gram speculative path. If your
model can't cheaply produce full-sequence logits, return `None` — the
puller falls back to single-token decode.

## 7. MoE (`model/moe.rs`)

**Router normalization order matters.** `mx.fast.rms_norm(x, scale *
root_size)` is **not** the same as `rms_norm(x * root_size, scale)` — RMS
normalizes by magnitude, so scaling the input first cancels out. Apply
`root_size` AFTER the norm.

**`gather_qmm` for batched expert dispatch.** Available in mlx-rs 0.25 as
`mlx_rs::ops::gather_qmm`. Use it for the prefill path; single-token decode
can stay simple. Handles quantized weights directly — no need to dequant
all experts up front.

## 8. Sampler (`common/sampler.rs`, `puller.rs`)

**Defaults that matter.** `top_p=0.9` is the industry default; leaving it
unset caused repetition traps on Gemma 4 with the default temperature.

**Repetition penalty + token-loop detector.** Infrastructure is in
`CommonSampler`. Off by default — turn on per-model if needed. Rep penalty
at 1.05-1.1 is safe; higher breaks late-turn quality.

## 9. Debugging playbook when output is garbage

Start from the top — each step's validity is a precondition for the next.

**Step 1 — Reference the model via mlx-lm.** Install mlx-lm in a venv, run
the same model with the same prompt at `temp=0` (greedy), save output.
This is your ground truth. If mlx-lm loops too, the model itself has that
failure mode (you'll need force-stop heuristics). If mlx-lm stops cleanly,
your inference has a bug.

**Step 2 — Compare prompts byte-for-byte.** Dump what you send to the
model. mlx-lm's `tokenizer.apply_chat_template(..., tokenize=False,
add_generation_prompt=True)` is the reference string. Missing BOS, extra
channel blocks, trailing newline drift — all will diverge output.

**Step 3 — Compare tokenizations.** Same string can tokenize differently
if special tokens aren't registered. Dump token IDs from both sides.

**Step 4 — Compare outputs token-by-token at greedy.** Greedy (`temp=0`)
makes inference deterministic given same weights and same input tokens. If
your first token differs from mlx-lm's, your forward pass has a bug —
nothing downstream (sampling, stop handling) can fix that. Dump logits
if possible and diff the argmax.

**Step 5 — Bisect within the forward pass.** Suspects in order:
1. Embedding lookup (gets wrong row for an id).
2. Attention (wrong scale, wrong pairing, wrong GQA).
3. Norms (offset, scale order).
4. LM head (softcap, tied vs untied embeddings).

**Step 6 — Watch for long-context bugs.** Short prompts may work while
long prompts fail. Classic causes: sliding-window cache eviction
interacting with RoPE offset, KV cache memory fragmentation, numerical
precision drift. Our explicit-offset fix is the canonical example.

## 10. Regression test

Add a 10-turn multi-turn integration test (see
`tests/gemma4_multiturn_integration.rs`) gated on `TEST_MLX_MODEL_DIR`.
Use `max_tokens=600` when `enable_thinking` is on (reasoning channel
eats a few hundred tokens before the final answer). Greedy (`temp=0`)
for reproducibility; tests should assert turn-1 non-empty + cross-turn
recall (turn 10 references turn 1's neighborhood).

Live server-node tests live in `flock_live_inference_integration.rs` —
mirror that pattern for P2P coverage.
