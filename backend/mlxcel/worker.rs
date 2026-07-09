//! Dedicated MLX worker thread — the thread-confinement boundary for `!Send`
//! MLX state.
//!
//! MLX's `LoadedModel` and the C++ generator state are **not** `Send` (they
//! hold raw MLX-C++ handles bound to the thread that created them). mlxcel's own
//! server confines each model to one OS thread and crosses request/response over
//! channels — see `src/server/audio_worker.rs` and `src/server/model_worker.rs`.
//! We mirror that: a single [`ModelWorker`] thread owns
//! `(LoadedModel, MlxcelTokenizer)` for the process, and every model touch
//! (load / generate) is enqueued as a [`Command`] and executed there.
//!
//! ## Why the `unsafe impl Send` is sound
//! [`Command`] carries closures / model outputs that are `!Send` in general, but
//! each command is *constructed* on a caller thread and *only ever executed* on
//! the single worker thread — it is moved across the channel but never touched
//! concurrently and never run anywhere but the worker. The MLX handles inside
//! are created on the worker and stay on the worker; the caller only ever holds
//! `mpsc::Sender<Command>` and receives back plain `Result`/token tuples, none
//! of which alias MLX state. This is the same single-owner-thread invariant that
//! makes mlxcel's audio/model workers sound.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};

use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel::{LoadedModel, MlxInferenceSession, SamplingConfig};
use mlxcel_core::generate::LanguageModel;

use crate::gen2::backend::common::grammar::GrammarSpec;
use crate::gen2::backend::common::tokenizer::HfTokenizer;
use crate::gen2::engine::ExecError;

/// One decoded token pushed from the worker's `on_token` callback to a puller.
pub(crate) struct DecodedToken {
    pub id: u32,
    pub text: String,
}

/// A streaming-generation request handed to the worker.
pub(crate) struct GenRequest {
    /// Raw prompt string. Tokenization happens on the worker because the
    /// tokenizer is `!Send` and lives there.
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    /// Grammar to constrain output to, if any. `Some(_)` diverts this
    /// generation off the fast `generate_streaming` path onto the manual
    /// masked decode loop ([`super::grammar::run_grammar_generation`]) — see
    /// the module docs for why the fast path can't carry a per-step mask.
    pub grammar: Option<GrammarSpec>,
    /// Set by the session's `stop()` to halt mlxcel's decode loop mid-stream.
    pub stop: Arc<AtomicBool>,
    /// Bounded channel the worker pushes decoded tokens onto; the puller drains
    /// it. Bounded so a slow consumer applies backpressure to the decode loop
    /// instead of letting an unbounded queue grow without limit.
    pub tokens_tx: SyncSender<DecodedToken>,
    /// One-shot reply: `Ok(prompt_token_count)` once generation *starts*
    /// (prompt tokenized, generation about to run), or an error if tokenization
    /// / setup failed. The token stream itself flows over `tokens_tx`.
    pub started_tx: Sender<Result<usize, ExecError>>,
}

/// Commands the worker thread executes. See the module-level soundness note for
/// why moving these across the channel is safe despite the `!Send` MLX state
/// they touch — they only ever *run* on the worker.
enum Command {
    /// Load a model directory, replacing any currently-loaded model.
    Load {
        model_dir: PathBuf,
        reply: Sender<Result<LoadInfo, ExecError>>,
    },
    /// Drop the loaded model (free MLX memory).
    Unload { reply: Sender<()> },
    /// Run one streaming generation on the loaded model. Boxed because
    /// `GenRequest` (which carries the prompt string, a `SamplingConfig`, and an
    /// optional `GrammarSpec` holding a JSON-schema `Value`) is far larger than
    /// the other variants — boxing keeps `Command` small on the channel.
    Generate(Box<GenRequest>),
    /// Terminate the worker loop.
    Shutdown,
    /// PROFILE-ONLY (S5 perf verify): run one `generate_streaming` pass with a
    /// selectable `on_token` mode and report the elapsed time + generated-token
    /// count on the worker thread (where the `!Send` model + tokenizer live).
    /// Used exclusively by the `captest_mlxcel_decode_profile` captest to settle
    /// whether the decode callback is pipeline-overlapped. Not on any user path,
    /// hence `#[cfg(test)]` — it never compiles into a production binary.
    #[cfg(test)]
    Profile {
        prompt: String,
        max_tokens: usize,
        mode: ProfileMode,
        reply: Sender<Result<ProfileRun, ExecError>>,
    },
}

/// Which `on_token` body the profile pass runs — the three-mode A/B/C ladder.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProfileMode {
    /// (A) no-op: `on_token` just returns `true`. The pure forward+sample
    /// ceiling — no callback CPU work at all.
    NoOp,
    /// (B) id-only: send the raw id down a channel, NO text decode. Isolates the
    /// channel/send cost.
    IdOnly,
    /// (C) decode+send: the production behavior — decode this id → text, then
    /// send. Full callback cost including per-token UTF-8 decode.
    DecodeSend,
}

/// Result of one profile pass. `text` is only populated for `DecodeSend` (the
/// concatenated streamed text, for the byte-identical parity guard).
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct ProfileRun {
    pub tokens: usize,
    pub elapsed_s: f64,
    pub text: String,
}

/// Cheap, `Send` facts about a freshly-loaded model, returned to the engine.
#[derive(Clone, Debug)]
pub(crate) struct LoadInfo {
    pub num_layers: usize,
    pub n_ctx: usize,
    pub architecture: Option<String>,
    /// Raw Jinja chat template (`tokenizer_config.json` `chat_template` or the
    /// `chat_template.jinja` sidecar). `None` when the model ships neither —
    /// `build_prompt` then falls back to the naive role-tagged concat with a
    /// loud warn. Mirrors `mlx::engine::build_bundle_from_dir`
    /// (`pio-core/src/gen2/backend/mlx/engine.rs:170`).
    pub chat_template: Option<String>,
    /// BOS string, decoded from the bos id KEEPING specials so `{{ bos_token }}`
    /// expands to the literal `<bos>`/`<s>`. Mirrors
    /// `pio-core/src/gen2/backend/mlx/engine.rs:178-181`.
    pub bos_str: Option<String>,
    /// EOS string, decoded from the eos id keeping specials. Mirrors
    /// `pio-core/src/gen2/backend/mlx/engine.rs:182-185`.
    pub eos_str: Option<String>,
}

/// SAFETY: see the module-level soundness note. `Command` moves to the single
/// worker thread and is executed only there; the MLX handles it may carry are
/// created on, and never leave, that thread.
unsafe impl Send for Command {}

/// Handle the engine holds. Cloneable senders route commands to the one worker.
pub(crate) struct ModelWorker {
    tx: Sender<Command>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for ModelWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelWorker").finish_non_exhaustive()
    }
}

impl ModelWorker {
    /// Spawn the worker thread. It initializes the MLX runtime once, then loops
    /// on the command channel until shutdown.
    pub(crate) fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<Command>();
        let join = std::thread::Builder::new()
            .name("pio-mlxcel-worker".into())
            .spawn(move || worker_loop(rx))
            .expect("spawn mlxcel worker thread");
        Self {
            tx,
            join: Some(join),
        }
    }

    /// Load a model, blocking until the worker reports the result.
    pub(crate) fn load(&self, model_dir: PathBuf) -> Result<LoadInfo, ExecError> {
        let (reply, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Load { model_dir, reply })
            .map_err(|_| worker_gone())?;
        reply_rx.recv().map_err(|_| worker_gone())?
    }

    /// Unload the model, blocking until done.
    pub(crate) fn unload(&self) {
        let (reply, reply_rx) = mpsc::channel();
        if self.tx.send(Command::Unload { reply }).is_ok() {
            let _ = reply_rx.recv();
        }
    }

    /// Enqueue a generation and block for the "started" reply (prompt token
    /// count). Tokens continue to stream asynchronously over `req.tokens_tx`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_generation_blocking(
        &self,
        prompt: String,
        max_tokens: usize,
        sampling: SamplingConfig,
        grammar: Option<GrammarSpec>,
        stop: Arc<AtomicBool>,
        tokens_tx: SyncSender<DecodedToken>,
    ) -> Result<usize, ExecError> {
        let (started_tx, started_rx) = mpsc::channel();
        let req = GenRequest {
            prompt,
            max_tokens,
            sampling,
            grammar,
            stop,
            tokens_tx,
            started_tx,
        };
        self.tx
            .send(Command::Generate(Box::new(req)))
            .map_err(|_| worker_gone())?;
        started_rx.recv().map_err(|_| worker_gone())?
    }

    /// PROFILE-ONLY (S5): run one `generate_streaming` pass in the given mode and
    /// block for the timing result. See [`Command::Profile`]. Not on any user
    /// path — only the decode-profile captest calls this.
    #[cfg(test)]
    pub(crate) fn profile_blocking(
        &self,
        prompt: String,
        max_tokens: usize,
        mode: ProfileMode,
    ) -> Result<ProfileRun, ExecError> {
        let (reply, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Profile {
                prompt,
                max_tokens,
                mode,
                reply,
            })
            .map_err(|_| worker_gone())?;
        reply_rx.recv().map_err(|_| worker_gone())?
    }
}

impl Drop for ModelWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn worker_gone() -> ExecError {
    ExecError::Other(anyhow::anyhow!("mlxcel worker thread is gone"))
}

/// The env var a host sets to the absolute path of a bundled `mlx.metallib`.
/// A packaged macOS `.app` resolves its Tauri resource path and exports this
/// before the gen2 controller (and hence this worker thread) starts. Unset in
/// dev and daemon-from-source, where MLX's compile-time `METAL_PATH` still
/// resolves to the build dir.
pub(crate) const METALLIB_ENV: &str = "PIO_MLX_METALLIB";

/// Point MLX at a host-bundled `mlx.metallib` if `PIO_MLX_METALLIB` names an
/// existing file. Calls the cxx binding
/// [`set_metallib_path`](mlxcel_core::set_metallib_path) (MLX's only runtime
/// override — checked first in its device library loader, no env-var
/// equivalent inside MLX). Must run before the first MLX op. Unset or
/// missing → no-op, so MLX falls back to its baked `METAL_PATH` (dev) or a
/// colocated search. Apple-only: MLX Metal only exists there.
#[cfg(target_vendor = "apple")]
fn apply_bundled_metallib() {
    let Some(raw) = std::env::var_os(METALLIB_ENV) else {
        return;
    };
    if raw.is_empty() {
        return;
    }
    let path = std::path::PathBuf::from(&raw);
    if path.is_file() {
        let path_str = path.to_string_lossy();
        mlxcel_core::set_metallib_path(&path_str);
        tracing::info!(
            path = %path_str,
            "mlxcel: applied bundled MLX metallib override ({METALLIB_ENV})"
        );
    } else {
        tracing::warn!(
            path = ?raw,
            "mlxcel: {METALLIB_ENV} set but is not an existing file — ignoring; \
             MLX will fall back to its baked METAL_PATH / colocated search"
        );
    }
}

/// Non-Apple builds have no MLX Metal backend; nothing to override.
#[cfg(not(target_vendor = "apple"))]
fn apply_bundled_metallib() {}

/// The worker thread body. Owns the `!Send` model + tokenizer; nothing MLX ever
/// leaves this stack frame.
fn worker_loop(rx: Receiver<Command>) {
    // ORDERING SEAM: if a host (a packaged macOS `.app`) bundled `mlx.metallib`
    // and pointed us at it, override MLX's metallib search path BEFORE any MLX
    // op runs — including `initialize_runtime` below. This is the ONLY runtime
    // hook (there is no env var inside MLX itself); a packaged app has no
    // compile-time `METAL_PATH` (that was the build dir). Must precede runtime
    // init. Dev / daemon-from-source leave it unset → MLX uses its baked path.
    apply_bundled_metallib();

    // Initialize the MLX runtime exactly once on this thread, matching mlxcel's
    // production binaries (bench_decode.rs / serve). `apply_metal_ops_per_buffer_default`
    // sets the hardware-gated MLX_MAX_OPS_PER_BUFFER; `initialize_runtime` resolves
    // the device (Metal GPU on Apple Silicon) and wired-memory limits.
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();
    let _runtime = mlxcel::initialize_runtime();

    let mut loaded: Option<LoadedState> = None;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::Load { model_dir, reply } => {
                // Own the reply value before moving `state` into `loaded` — cloning
                // the info on success, propagating the owned error on failure (a
                // borrowed `&ExecError` can't cross the reply channel).
                let info = match load_on_worker(&model_dir) {
                    Ok(state) => {
                        let info = state.info.clone();
                        loaded = Some(state);
                        Ok(info)
                    }
                    Err(e) => Err(e),
                };
                let _ = reply.send(info);
            }
            Command::Unload { reply } => {
                loaded = None;
                let _ = reply.send(());
            }
            Command::Generate(req) => {
                run_generation(loaded.as_ref(), *req);
            }
            #[cfg(test)]
            Command::Profile {
                prompt,
                max_tokens,
                mode,
                reply,
            } => {
                let _ = reply.send(run_profile(loaded.as_ref(), &prompt, max_tokens, mode));
            }
            Command::Shutdown => break,
        }
    }
}

/// The `!Send` model + tokenizer, plus cached facts.
struct LoadedState {
    model: LoadedModel,
    tokenizer: MlxcelTokenizer,
    info: LoadInfo,
    /// EOS ids read from the model dir, merged into every SamplingConfig so the
    /// stream terminates cleanly (mirrors bench_decode.rs::sampling_config).
    eos_token_ids: Vec<i32>,
    /// pio-core's own tokenizer for this model dir, used ONLY to drive the
    /// grammar matcher on the manual masked-decode path. `None` when the model
    /// dir ships no readable `tokenizer.json` — grammar generations then fail
    /// loud with a clear error rather than silently falling back to unmasked
    /// output. Loading it is cheap (parse `tokenizer.json`) so we do it at
    /// model-load time; text-only generations simply never touch it.
    hf_tok: Option<HfTokenizer>,
}

fn load_on_worker(model_dir: &Path) -> Result<LoadedState, ExecError> {
    mlxcel_core::clear_memory_cache();
    let (model, tokenizer) = mlxcel::load_model(model_dir)
        .map_err(|e| ExecError::InvalidModelFile(format!("mlxcel load_model failed: {e}")))?;

    let num_layers = model.num_layers();
    let n_ctx = mlxcel::read_model_context_window(model_dir).unwrap_or(4096);
    let architecture = read_architecture(model_dir);
    let eos_token_ids = mlxcel::read_eos_token_ids(model_dir);

    // pio-core's own tokenizer for the grammar path. Non-fatal if absent: a
    // model with no `tokenizer.json` simply can't do grammar-constrained
    // decode, and that's surfaced (fail-loud) at the point of use.
    let hf_tok = match HfTokenizer::from_dir(model_dir) {
        Ok(t) => Some(t),
        Err(e) => {
            tracing::warn!(
                "mlxcel: pio-core tokenizer unavailable ({e}); grammar-constrained \
                 decode will error for this model"
            );
            None
        }
    };

    // Load the model's REAL Jinja chat template + derive bos/eos strings the
    // same way the mlx backend does (`build_bundle_from_dir`,
    // `pio-core/src/gen2/backend/mlx/engine.rs:170-185`). Decode the bos/eos ids
    // KEEPING specials so `{{ bos_token }}` in the template expands to the
    // literal `<bos>` — `skip_special=true` would strip it and Gemma 4 would see
    // a different position-0 embedding, producing catastrophic step-1 logits
    // (the documented mlx failure mode). The pio-core `HfTokenizer` (`hf_tok`)
    // supplies bos_id/eos_id/decode_keep_specials; when it's absent we simply
    // can't derive the strings — the template still renders, minus BOS expansion.
    let chat_template = crate::gen2::backend::common::load_chat_template(model_dir);
    let (bos_str, eos_str) = match hf_tok.as_ref() {
        Some(tok) => {
            let bos = tok
                .bos_id()
                .and_then(|id| tok.decode_keep_specials(&[id]).ok());
            let eos = tok
                .eos_id()
                .and_then(|id| tok.decode_keep_specials(&[id]).ok());
            (bos, eos)
        }
        None => (None, None),
    };

    Ok(LoadedState {
        model,
        tokenizer,
        info: LoadInfo {
            num_layers,
            n_ctx,
            architecture,
            chat_template,
            bos_str,
            eos_str,
        },
        eos_token_ids,
        hf_tok,
    })
}

/// Best-effort `model_type` from the bundle's `config.json` for
/// `bundle_architecture()` (drives Gemma-4 channel markers in the mapper).
fn read_architecture(model_dir: &Path) -> Option<String> {
    let cfg = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&cfg).ok()?;
    json.get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Run one streaming generation. Tokenizes the prompt (needs the worker-local
/// tokenizer), then drives the FAST-path `generate_streaming`. The `on_token`
/// callback decodes each id→text with the same worker-local tokenizer and
/// pushes it down the puller's channel; it returns `false` to halt the decode
/// loop when the session's `stop` flag is set or the consumer has hung up.
fn run_generation(loaded: Option<&LoadedState>, req: GenRequest) {
    let GenRequest {
        prompt,
        max_tokens,
        mut sampling,
        grammar,
        stop,
        tokens_tx,
        started_tx,
    } = req;

    let Some(state) = loaded else {
        let _ = started_tx.send(Err(ExecError::ModelNotLoaded));
        return;
    };

    // Tokenize on the worker (the tokenizer is `!Send` and lives here).
    let prompt_ids_u32 = match state.tokenizer.encode(&prompt, true) {
        Ok(ids) => ids,
        Err(e) => {
            let _ = started_tx.send(Err(ExecError::Other(anyhow::anyhow!(
                "mlxcel tokenizer encode failed: {e}"
            ))));
            return;
        }
    };
    if prompt_ids_u32.is_empty() {
        let _ = started_tx.send(Err(ExecError::InvalidArg(
            "empty prompt after tokenization",
        )));
        return;
    }
    let prompt_ids_i32: Vec<i32> = prompt_ids_u32.iter().map(|&t| t as i32).collect();
    let prompt_len = prompt_ids_i32.len();

    // Merge the model's EOS ids into the sampling config so the stream stops on
    // end-of-turn (mirrors bench_decode.rs::sampling_config → stop_token_ids).
    if sampling.stop_token_ids.is_empty() {
        sampling.stop_token_ids = state.eos_token_ids.clone();
    }

    // GRAMMAR PATH (S4) — a grammar spec diverts off the fast path onto the
    // manual masked decode loop. Validate the pio-core tokenizer up front so a
    // missing one fails loud into `started_tx` (never a silent unmasked stream).
    if let Some(spec) = grammar {
        let Some(hf_tok) = state.hf_tok.as_ref() else {
            let _ = started_tx.send(Err(ExecError::Other(anyhow::anyhow!(
                "grammar-constrained decode requires a tokenizer.json in the model \
                 dir; this model shipped none"
            ))));
            return;
        };
        // Signal "started"; from here, grammar-masked tokens stream.
        if started_tx.send(Ok(prompt_len)).is_err() {
            return;
        }
        let res = super::grammar::run_grammar_generation(
            &state.model,
            &state.tokenizer,
            hf_tok,
            spec,
            &prompt_ids_i32,
            max_tokens,
            &state.eos_token_ids,
            &stop,
            &tokens_tx,
        );
        if let Err(e) = res {
            // A grammar failure (stuck matcher, parser error) after the stream
            // started can't be reported via `started_tx` (already consumed); log
            // it. The dropped `tokens_tx` still closes the stream cleanly, and
            // the emitted-so-far text is what the puller saw.
            tracing::error!("mlxcel grammar generation failed: {e}");
        }
        return;
    }

    // Signal "started" with the prompt-token count. From here, tokens stream.
    if started_tx.send(Ok(prompt_len)).is_err() {
        // Caller hung up before we even started — nothing to stream to.
        return;
    }

    // FAST PATH — pipelined streaming decode. `MlxInferenceSession` wraps
    // `CxxGenerator`; `generate_streaming` delegates verbatim, preserving the
    // lookahead pipeline. See docs/plans/mlxcel-embedding-roadmap.md (S2).
    let mut session = MlxInferenceSession::new(state.model.num_layers());

    let tokenizer = &state.tokenizer;
    let on_token = |id: i32| -> bool {
        // Stop requested by the session (user hit stop) → halt the loop.
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        // Decode this single id → text. `decode` on a 1-element slice is how the
        // server's streaming path renders per-token text (skip_special=true so
        // control tokens don't leak into visible text).
        let text = tokenizer.decode(&[id as u32], true).unwrap_or_default();
        // Push to the puller. A `SyncSender` send fails only if the receiver was
        // dropped (puller gone / consumer done) — treat that as "stop".
        tokens_tx
            .send(DecodedToken {
                id: id as u32,
                text,
            })
            .is_ok()
    };

    let _generated = session.generate_streaming(
        &state.model,
        &prompt_ids_i32,
        max_tokens,
        &sampling,
        on_token,
    );
    // Dropping `tokens_tx` here (end of scope) closes the channel → the puller's
    // `recv()` returns `Err`, which it maps to end-of-stream (`Eos` then `None`).
}

/// PROFILE-ONLY (S5 perf verify): run one greedy `generate_streaming` pass with
/// the mode-selected `on_token` body and time it on the worker thread. Mirrors
/// [`run_generation`]'s fast path exactly (same greedy SamplingConfig, same
/// tokenization, same session) so the A/B/C deltas isolate ONLY the callback
/// body — not any other difference. Never on a user path.
#[cfg(test)]
fn run_profile(
    loaded: Option<&LoadedState>,
    prompt: &str,
    max_tokens: usize,
    mode: ProfileMode,
) -> Result<ProfileRun, ExecError> {
    let state = loaded.ok_or(ExecError::ModelNotLoaded)?;

    let prompt_ids_u32 = state
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| ExecError::Other(anyhow::anyhow!("mlxcel tokenizer encode failed: {e}")))?;
    if prompt_ids_u32.is_empty() {
        return Err(ExecError::InvalidArg("empty prompt after tokenization"));
    }
    let prompt_ids_i32: Vec<i32> = prompt_ids_u32.iter().map(|&t| t as i32).collect();

    // Greedy, EOS-merged — identical to the fast text path so the pass is
    // FAST-pipeline eligible (temp 0.0 → mlxcel async GPU-argmax decode).
    let sampling = SamplingConfig {
        temperature: 0.0,
        stop_token_ids: state.eos_token_ids.clone(),
        ..SamplingConfig::default()
    };

    let mut session = MlxInferenceSession::new(state.model.num_layers());
    let tokenizer = &state.tokenizer;

    // For (B) id-only, a bounded channel + a drain thread so `send` has a live
    // receiver (matches the production shape: bounded SyncSender to a consumer).
    // We drain on THIS worker thread would deadlock, so spawn a scratch drainer.
    let (id_tx, id_rx) = mpsc::sync_channel::<i32>(256);
    let drainer = std::thread::spawn(move || {
        let mut n = 0usize;
        while id_rx.recv().is_ok() {
            n += 1;
        }
        n
    });

    let mut text = String::new();
    let mut count = 0usize;
    let t = std::time::Instant::now();
    match mode {
        ProfileMode::NoOp => {
            let _ = session.generate_streaming(
                &state.model,
                &prompt_ids_i32,
                max_tokens,
                &sampling,
                |_id: i32| -> bool {
                    count += 1;
                    true
                },
            );
        }
        ProfileMode::IdOnly => {
            let _ = session.generate_streaming(
                &state.model,
                &prompt_ids_i32,
                max_tokens,
                &sampling,
                |id: i32| -> bool {
                    count += 1;
                    id_tx.send(id).is_ok()
                },
            );
        }
        ProfileMode::DecodeSend => {
            // The EXACT production callback body: decode then send text down a
            // bounded channel. We reuse the same id channel but send the decoded
            // text length as the payload isn't needed — decode cost is what we
            // measure, and the send must be a real bounded send to a consumer.
            let (txt_tx, txt_rx) = mpsc::sync_channel::<String>(256);
            let txt_drainer = std::thread::spawn(move || {
                let mut acc = String::new();
                while let Ok(s) = txt_rx.recv() {
                    acc.push_str(&s);
                }
                acc
            });
            let _ = session.generate_streaming(
                &state.model,
                &prompt_ids_i32,
                max_tokens,
                &sampling,
                |id: i32| -> bool {
                    count += 1;
                    let s = tokenizer.decode(&[id as u32], true).unwrap_or_default();
                    txt_tx.send(s).is_ok()
                },
            );
            drop(txt_tx);
            text = txt_drainer.join().unwrap_or_default();
        }
    }
    let elapsed_s = t.elapsed().as_secs_f64().max(1e-6);

    // Tear the scratch id-channel down (NoOp/DecodeSend never used it, but the
    // drainer thread must be joined either way).
    drop(id_tx);
    let _ = drainer.join();

    Ok(ProfileRun {
        tokens: count,
        elapsed_s,
        text,
    })
}
