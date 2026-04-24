//! Per-turn inference telemetry.
//!
//! Phase 0 of the inference-robustness plan: measure before you fix. Every
//! generated turn emits one [`TurnTelemetry`] record summarising how it ran.
//! Consumers (OAI route today; observability snapshot in Phase 5) use this
//! to answer:
//!
//! - Was this turn a cache hit, a cold start, or a near-miss?
//! - How did generation terminate — natural EOS, max-tokens, loop detector?
//! - What did the reply actually look like — how many thinking vs. visible
//!   tokens, any special-token leakage, did it round-trip back to text?
//!
//! The record is intentionally flat and additive. It is NOT part of any
//! hot-path control flow — emit-and-forget at turn completion.

use serde::{Deserialize, Serialize};

/// Result of looking the turn's prefix up in whatever session cache the
/// transport layer owns (today: [`OaiSessionCache`] in pio-daemon).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum CacheState {
    /// Prefix hash matched; controller took the continuation path and
    /// only prefilled the new user delta.
    Hit,
    /// No cached session for this prefix — cold start with a fresh
    /// session id. Expected on first turn of a conversation; a rising
    /// miss rate on turn >=2 is the smoking gun for template-lossy
    /// replay.
    Miss,
    /// (Reserved.) Prefix matched every prior turn's hash except one.
    /// Populating this requires fuzzy-matching in the cache, which we
    /// don't do today — included so Phase 5 aggregation can count it
    /// when the cache grows that capability.
    NearMiss,
    /// Transport has no prefix-hash cache — chat_id is the cache key
    /// and KV reuse happens implicitly inside the controller (today:
    /// Tauri). These turns should NOT count against hit/miss rate
    /// because the whole notion of a prefix-hash lookup doesn't apply.
    /// Separate bucket so dashboards can show "native-routed turns"
    /// without polluting the OAI cache SLO.
    Native,
}

impl CacheState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::NearMiss => "near_miss",
            Self::Native => "native",
        }
    }
}

/// How the turn's generation loop terminated. A turn ending on anything
/// other than `Eot` is a risk factor for next-turn degeneration: the
/// model's last sampled token isn't an end-of-turn marker, so the
/// subsequent prompt's turn boundary is fuzzy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    /// Natural end-of-turn token. The KV state after this is clean.
    Eot,
    /// `max_tokens` was reached before the model emitted an EOT.
    MaxTokens,
    /// Loop detector or other sampler guard aborted generation.
    LoopDetector,
    /// Client disconnected, user pressed stop, or host requested halt.
    Stopped,
    /// Backend error (OOM, shape mismatch, tokenizer failure).
    Error,
}

impl Termination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eot => "eot",
            Self::MaxTokens => "max_tokens",
            Self::LoopDetector => "loop_detector",
            Self::Stopped => "stopped",
            Self::Error => "error",
        }
    }

    /// True when the turn ended naturally. Use this to gate aggregate
    /// metrics that only make sense for clean turns.
    pub fn is_clean(self) -> bool {
        matches!(self, Self::Eot)
    }
}

/// Shape of the emitted reply. Populated by the reasoning-channel state
/// machine (Phase 2a). When unavailable (no channel detected or backend
/// doesn't report), the counts are zero and `round_trips_to_text` is
/// `None` (not `false` — the check wasn't run).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ReplyShape {
    /// Tokens emitted inside a reasoning channel (`<|channel>thought`,
    /// `<think>`, etc.). Zero for non-thinking models.
    #[serde(default)]
    pub thinking_tokens: u32,
    /// Tokens emitted outside any reasoning channel — what the user
    /// eventually reads.
    #[serde(default)]
    pub content_tokens: u32,
    /// Count of special tokens (role markers, channel markers, tool
    /// markers) that appeared in the sampled stream. Non-zero on a
    /// visible-text emission indicates the template leaked a special
    /// token the state machine didn't recognise.
    #[serde(default)]
    pub special_token_count: u32,
    /// `Some(true)` when the emitted text, run back through the chat
    /// template, produces the same token IDs the model sampled.
    /// `Some(false)` is the bright-red indicator for template-lossy
    /// replay — the single most actionable signal in this struct.
    /// `None` when the round-trip check was skipped (sampled, always
    /// skipped, or sampling rate was low enough that this turn wasn't
    /// selected).
    #[serde(default)]
    pub round_trips_to_text: Option<bool>,
}

/// One record per completed turn. Emit at turn-end (`Eos` / `Stopped` /
/// `Error`) from the transport layer that owns session-cache state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct TurnTelemetry {
    /// Opaque identifier for the session this turn ran on. Used for
    /// correlation in logs — not security-sensitive.
    pub session_id: String,
    /// 1-indexed position of this turn within the session's history,
    /// counting only user→assistant round trips. `1` = first turn.
    pub turn_index: u32,
    /// Fingerprint of the model that generated the turn. Pair with
    /// [`ModelMeta::tokenizer_digest`] for offline slicing by model.
    #[serde(default)]
    pub model_id: Option<String>,
    /// How the session-cache lookup resolved for this turn.
    pub cache_state: CacheState,
    /// How the generation loop terminated.
    pub termination: Termination,
    /// Shape of the emitted reply.
    #[serde(default)]
    pub reply_shape: ReplyShape,
    /// Wall time from request-start to the first emitted token, in
    /// microseconds. Mirrors [`ExecutionStats::first_token_us`] so
    /// aggregators don't need to cross-reference.
    #[serde(default)]
    pub first_token_us: u64,
    /// Total decode tokens for the turn (thinking + content + any
    /// specials). Mirrors [`ExecutionStats::decode_tokens`].
    #[serde(default)]
    pub decode_tokens: u32,
}

impl TurnTelemetry {
    /// Emit this record via the `tracing` facade at `info!` level
    /// under the target `pio_inference_telemetry`. Downstream log
    /// shippers filter on that target to isolate turn records.
    /// Also updates the process-global [`TelemetryAggregator`] so
    /// Phase 5 aggregate stats stay current.
    pub fn emit(&self) {
        tracing::info!(
            target: "pio_inference_telemetry",
            session_id = %self.session_id,
            turn_index = self.turn_index,
            model_id = self.model_id.as_deref().unwrap_or(""),
            cache_state = self.cache_state.as_str(),
            termination = self.termination.as_str(),
            thinking_tokens = self.reply_shape.thinking_tokens,
            content_tokens = self.reply_shape.content_tokens,
            special_token_count = self.reply_shape.special_token_count,
            round_trips_to_text = ?self.reply_shape.round_trips_to_text,
            first_token_us = self.first_token_us,
            decode_tokens = self.decode_tokens,
            "turn complete",
        );
        global_aggregator().record(self);
    }
}

// ── Phase 5: aggregator ────────────────────────────────────────────────────

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed-bucket latency histogram. Upper bounds (inclusive) in
/// microseconds; the final bucket is the implicit "overflow" (>10s).
/// Chosen to cover the realistic TTFT range for local inference:
/// 10ms is near-instant (warm KV cache hits), 1s is a full prefill on
/// a cold session, 10s is near-timeout. Bucket count is kept small so
/// `percentile()` scans stay in cache.
///
/// The bucket layout is deliberately static (no HDRHistogram crate dep).
/// Going dynamic would be premature — these ~11 buckets are enough to
/// resolve p50, p90, p99 at the granularity that moves product
/// decisions (sub-50ms vs. sub-500ms vs. sub-5s).
const TTFT_BUCKET_UPPER_US: &[u64] = &[
    1_000,      // 1ms
    10_000,     // 10ms
    25_000,     // 25ms
    50_000,     // 50ms
    100_000,    // 100ms
    250_000,    // 250ms
    500_000,    // 500ms
    1_000_000,  // 1s
    2_500_000,  // 2.5s
    5_000_000,  // 5s
    10_000_000, // 10s
];
// Plus one overflow bucket for samples > 10s. Total = len() + 1.
const TTFT_BUCKET_COUNT: usize = TTFT_BUCKET_UPPER_US.len() + 1;

/// Process-global counters fed by every [`TurnTelemetry::emit`].
///
/// Exposes the Phase-5 SLO metrics without requiring a separate
/// observability pipeline:
///
/// - cache hit / miss rate (Phase-0 signal: how often multi-turn reuse
///   is actually saving work)
/// - termination breakdown (EOT vs max_tokens vs loop-detector vs
///   stopped vs error — non-EOT rate on >turn-1 is the precursor to
///   degeneration)
/// - "poison" turns: replies whose state machine detected special-token
///   leakage — the bright-red indicator that template fidelity broke
/// - TTFT histogram: p50 / p90 / p99 resolved over a fixed-bucket set.
///   Mean + max are still exposed for compatibility but the percentiles
///   are the ones to alert on — the tail is where the real regressions
///   live.
///
/// All counters are `Relaxed` atomics — ordering across counters
/// doesn't matter because the aggregator is read, not synchronised
/// against.
pub struct TelemetryAggregator {
    turns_total: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_near_misses: AtomicU64,
    /// Turns that bypassed the prefix-hash cache entirely (Tauri path,
    /// native chat_id routing). Counted separately so they don't
    /// pollute the OAI `cache_hit_rate` SLO — the ratio that dashboard
    /// viewers care about is `hits / (hits + misses + near_misses)`,
    /// not `hits / turns_total`.
    cache_native: AtomicU64,
    term_eot: AtomicU64,
    term_max_tokens: AtomicU64,
    term_loop: AtomicU64,
    term_stopped: AtomicU64,
    term_error: AtomicU64,
    poison_turns: AtomicU64,
    first_token_us_sum: AtomicU64,
    first_token_us_max: AtomicU64,
    first_token_us_samples: AtomicU64,
    /// Per-bucket counts for the TTFT histogram. Same order as
    /// [`TTFT_BUCKET_UPPER_US`]; final slot is the >10s overflow.
    /// Kept as owned `AtomicU64` array so no `Mutex` on the hot path.
    ttft_buckets: [AtomicU64; TTFT_BUCKET_COUNT],
}

impl Default for TelemetryAggregator {
    fn default() -> Self {
        Self {
            turns_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_near_misses: AtomicU64::new(0),
            cache_native: AtomicU64::new(0),
            term_eot: AtomicU64::new(0),
            term_max_tokens: AtomicU64::new(0),
            term_loop: AtomicU64::new(0),
            term_stopped: AtomicU64::new(0),
            term_error: AtomicU64::new(0),
            poison_turns: AtomicU64::new(0),
            first_token_us_sum: AtomicU64::new(0),
            first_token_us_max: AtomicU64::new(0),
            first_token_us_samples: AtomicU64::new(0),
            ttft_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl TelemetryAggregator {
    pub fn record(&self, t: &TurnTelemetry) {
        self.turns_total.fetch_add(1, Ordering::Relaxed);
        match t.cache_state {
            CacheState::Hit => &self.cache_hits,
            CacheState::Miss => &self.cache_misses,
            CacheState::NearMiss => &self.cache_near_misses,
            CacheState::Native => &self.cache_native,
        }
        .fetch_add(1, Ordering::Relaxed);
        match t.termination {
            Termination::Eot => &self.term_eot,
            Termination::MaxTokens => &self.term_max_tokens,
            Termination::LoopDetector => &self.term_loop,
            Termination::Stopped => &self.term_stopped,
            Termination::Error => &self.term_error,
        }
        .fetch_add(1, Ordering::Relaxed);
        if t.reply_shape.special_token_count > 0 {
            self.poison_turns.fetch_add(1, Ordering::Relaxed);
        }
        if t.first_token_us > 0 {
            self.first_token_us_sum
                .fetch_add(t.first_token_us, Ordering::Relaxed);
            self.first_token_us_samples.fetch_add(1, Ordering::Relaxed);
            let prev_max = self.first_token_us_max.load(Ordering::Relaxed);
            if t.first_token_us > prev_max {
                // Relaxed CAS; losing a race with another thread here
                // is fine — the next turn picks up the new max.
                let _ = self.first_token_us_max.compare_exchange_weak(
                    prev_max,
                    t.first_token_us,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
            let bucket = ttft_bucket_index(t.first_token_us);
            self.ttft_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        let turns_total = self.turns_total.load(Ordering::Relaxed);
        let samples = self.first_token_us_samples.load(Ordering::Relaxed);
        let sum = self.first_token_us_sum.load(Ordering::Relaxed);
        let mut ttft_buckets = [0u64; TTFT_BUCKET_COUNT];
        for (i, b) in self.ttft_buckets.iter().enumerate() {
            ttft_buckets[i] = b.load(Ordering::Relaxed);
        }
        TelemetrySnapshot {
            turns_total,
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_near_misses: self.cache_near_misses.load(Ordering::Relaxed),
            cache_native: self.cache_native.load(Ordering::Relaxed),
            term_eot: self.term_eot.load(Ordering::Relaxed),
            term_max_tokens: self.term_max_tokens.load(Ordering::Relaxed),
            term_loop: self.term_loop.load(Ordering::Relaxed),
            term_stopped: self.term_stopped.load(Ordering::Relaxed),
            term_error: self.term_error.load(Ordering::Relaxed),
            poison_turns: self.poison_turns.load(Ordering::Relaxed),
            first_token_us_mean: if samples > 0 { sum / samples } else { 0 },
            first_token_us_max: self.first_token_us_max.load(Ordering::Relaxed),
            first_token_us_buckets: ttft_buckets.to_vec(),
        }
    }
}

/// Find the histogram bucket for `value_us`. Bucket layout is defined
/// by [`TTFT_BUCKET_UPPER_US`] — return the index of the first bucket
/// whose upper bound is `>= value_us`, or the overflow bucket (the
/// last one) when value exceeds every upper bound.
fn ttft_bucket_index(value_us: u64) -> usize {
    for (i, &upper) in TTFT_BUCKET_UPPER_US.iter().enumerate() {
        if value_us <= upper {
            return i;
        }
    }
    TTFT_BUCKET_UPPER_US.len() // overflow slot
}

/// Process-global aggregator. Lives for the lifetime of the process
/// and is thread-safe via atomic counters.
pub fn global_aggregator() -> &'static TelemetryAggregator {
    static AGG: OnceLock<TelemetryAggregator> = OnceLock::new();
    AGG.get_or_init(TelemetryAggregator::default)
}

/// Read-only snapshot of the aggregator. Consumers (Tauri's observability
/// snapshot, daemon's `/debug/metrics`) serialize this struct. Cheap:
/// one atomic load per counter.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct TelemetrySnapshot {
    pub turns_total: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_near_misses: u64,
    /// Turns routed through a transport without prefix-hash caching
    /// (Tauri). Intentionally excluded from [`Self::cache_hit_rate`]
    /// — the rate is about the OAI session cache's effectiveness,
    /// and counting Tauri as "miss" would make it look broken.
    #[serde(default)]
    pub cache_native: u64,
    pub term_eot: u64,
    pub term_max_tokens: u64,
    pub term_loop: u64,
    pub term_stopped: u64,
    pub term_error: u64,
    /// Turns whose reply contained a special-token leak (state machine
    /// detected an unrecognised `<|...>` literal in visible content).
    pub poison_turns: u64,
    /// Mean time-to-first-token across samples with `first_token_us > 0`.
    pub first_token_us_mean: u64,
    pub first_token_us_max: u64,
    /// TTFT histogram — bucket counts in the same order as the
    /// internal `TTFT_BUCKET_UPPER_US` thresholds, plus one trailing
    /// overflow slot for samples > 10s. `percentile_us(p)` resolves
    /// a fractional percentile from these buckets. Callers that need
    /// both the thresholds and counts can call
    /// [`ttft_bucket_upper_bounds_us`].
    #[serde(default)]
    pub first_token_us_buckets: Vec<u64>,
}

impl TelemetrySnapshot {
    /// Cache hit rate — `None` when no turns have been recorded yet
    /// (`0/0` would imply "perfect" otherwise).
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let total = self.cache_hits + self.cache_misses + self.cache_near_misses;
        if total == 0 {
            None
        } else {
            Some(self.cache_hits as f64 / total as f64)
        }
    }

    /// Fraction of turns that terminated on something other than a
    /// natural end-of-turn token. Non-EOT is a risk factor for next-turn
    /// degeneration; watching this rate by model is Phase 5's main
    /// actionable alert.
    pub fn non_eot_rate(&self) -> Option<f64> {
        if self.turns_total == 0 {
            None
        } else {
            Some((self.turns_total - self.term_eot) as f64 / self.turns_total as f64)
        }
    }

    /// Fraction of turns flagged as poisoned. Should be near zero; a
    /// rising rate means either a new model is emitting unrecognised
    /// specials or the chat template changed upstream.
    pub fn poison_rate(&self) -> Option<f64> {
        if self.turns_total == 0 {
            None
        } else {
            Some(self.poison_turns as f64 / self.turns_total as f64)
        }
    }

    /// Resolve a TTFT percentile (e.g. `0.5` = p50, `0.99` = p99) in
    /// microseconds from the histogram. Returns `None` when no samples
    /// have been recorded. The result is the upper bound of the bucket
    /// containing the `p`-th cumulative sample — exact within the
    /// chosen bucket width. For the overflow bucket (>10s), returns
    /// `u64::MAX` so dashboards render it as "tail exceeded 10s".
    ///
    /// Contract: `p` is clamped to `[0.0, 1.0]`. p=0.0 returns the
    /// smallest populated bucket's upper bound; p=1.0 returns the
    /// largest populated bucket's upper bound.
    pub fn percentile_us(&self, p: f64) -> Option<u64> {
        if self.first_token_us_buckets.is_empty() {
            return None;
        }
        let total: u64 = self.first_token_us_buckets.iter().sum();
        if total == 0 {
            return None;
        }
        let p = p.clamp(0.0, 1.0);
        // Use ceil semantics so p50 of [1,1,1,1] lands on the 2nd sample,
        // matching the "at-least p fraction of samples <= X" reading.
        let target = ((total as f64) * p).ceil().max(1.0) as u64;
        let mut running: u64 = 0;
        for (i, &count) in self.first_token_us_buckets.iter().enumerate() {
            running += count;
            if running >= target {
                return Some(bucket_upper_bound_us(i));
            }
        }
        // Shouldn't reach here because target <= total.
        Some(u64::MAX)
    }

    /// Convenience wrappers. Callers that plot multiple percentiles
    /// should call [`percentile_us`] directly to avoid recomputing the
    /// total repeatedly.
    pub fn ttft_p50_us(&self) -> Option<u64> {
        self.percentile_us(0.50)
    }
    pub fn ttft_p90_us(&self) -> Option<u64> {
        self.percentile_us(0.90)
    }
    pub fn ttft_p99_us(&self) -> Option<u64> {
        self.percentile_us(0.99)
    }
}

/// Upper bound (µs) for the histogram bucket at index `i`. The final
/// slot ("overflow") returns `u64::MAX` since we don't bound the
/// long-tail explicitly — callers that want a numeric cap should pick
/// one appropriate for their dashboard.
pub fn bucket_upper_bound_us(i: usize) -> u64 {
    TTFT_BUCKET_UPPER_US.get(i).copied().unwrap_or(u64::MAX)
}

/// Upper bounds (µs) for the fixed TTFT histogram, in ascending order.
/// Length equals `first_token_us_buckets.len() - 1` (the overflow
/// bucket has no finite upper bound).
pub fn ttft_bucket_upper_bounds_us() -> &'static [u64] {
    TTFT_BUCKET_UPPER_US
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn termination_is_clean_only_for_eot() {
        assert!(Termination::Eot.is_clean());
        assert!(!Termination::MaxTokens.is_clean());
        assert!(!Termination::LoopDetector.is_clean());
        assert!(!Termination::Stopped.is_clean());
        assert!(!Termination::Error.is_clean());
    }

    #[test]
    fn cache_state_strings_are_stable() {
        // Log aggregators depend on these exact strings — don't rename
        // without updating downstream dashboards.
        assert_eq!(CacheState::Hit.as_str(), "hit");
        assert_eq!(CacheState::Miss.as_str(), "miss");
        assert_eq!(CacheState::NearMiss.as_str(), "near_miss");
    }

    #[test]
    fn termination_strings_are_stable() {
        assert_eq!(Termination::Eot.as_str(), "eot");
        assert_eq!(Termination::MaxTokens.as_str(), "max_tokens");
        assert_eq!(Termination::LoopDetector.as_str(), "loop_detector");
        assert_eq!(Termination::Stopped.as_str(), "stopped");
        assert_eq!(Termination::Error.as_str(), "error");
    }

    #[test]
    fn reply_shape_default_is_all_zero_and_unknown() {
        let s = ReplyShape::default();
        assert_eq!(s.thinking_tokens, 0);
        assert_eq!(s.content_tokens, 0);
        assert_eq!(s.special_token_count, 0);
        assert_eq!(s.round_trips_to_text, None);
    }

    #[test]
    fn serde_roundtrip_preserves_every_field() {
        let t = TurnTelemetry {
            session_id: "oai-compat-abc".into(),
            turn_index: 7,
            model_id: Some("gemma-4-e2b".into()),
            cache_state: CacheState::Hit,
            termination: Termination::MaxTokens,
            reply_shape: ReplyShape {
                thinking_tokens: 200,
                content_tokens: 50,
                special_token_count: 2,
                round_trips_to_text: Some(false),
            },
            first_token_us: 42_000,
            decode_tokens: 252,
        };
        let js = serde_json::to_string(&t).expect("serialize");
        let back: TurnTelemetry = serde_json::from_str(&js).expect("deserialize");
        assert_eq!(back.session_id, t.session_id);
        assert_eq!(back.turn_index, 7);
        assert_eq!(back.cache_state, CacheState::Hit);
        assert_eq!(back.termination, Termination::MaxTokens);
        assert_eq!(back.reply_shape.thinking_tokens, 200);
        assert_eq!(back.reply_shape.round_trips_to_text, Some(false));
    }

    // ── Aggregator ──────────────────────────────────────────────────

    fn sample_telemetry(
        term: Termination,
        cache: CacheState,
        first_token_us: u64,
    ) -> TurnTelemetry {
        TurnTelemetry {
            session_id: "test".into(),
            turn_index: 1,
            model_id: None,
            cache_state: cache,
            termination: term,
            reply_shape: ReplyShape::default(),
            first_token_us,
            decode_tokens: 0,
        }
    }

    #[test]
    fn aggregator_counts_cache_states() {
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 100));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 200));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Miss, 150));
        let s = agg.snapshot();
        assert_eq!(s.turns_total, 3);
        assert_eq!(s.cache_hits, 2);
        assert_eq!(s.cache_misses, 1);
        assert_eq!(s.cache_hit_rate(), Some(2.0 / 3.0));
    }

    #[test]
    fn native_cache_state_excluded_from_hit_rate() {
        // The Tauri path reports CacheState::Native because it has
        // no prefix-hash cache. These turns must NOT show up in the
        // OAI cache_hit_rate SLO — they'd always count as "misses"
        // and drag the dashboard number down to zero.
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 100));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Native, 50));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Native, 50));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Native, 50));
        let s = agg.snapshot();
        assert_eq!(s.turns_total, 4);
        assert_eq!(s.cache_hits, 1);
        assert_eq!(s.cache_native, 3);
        // hit / (hit + miss + near_miss) = 1/1 = 1.0 — natives excluded.
        assert_eq!(s.cache_hit_rate(), Some(1.0));
    }

    #[test]
    fn cache_state_strings_include_native() {
        assert_eq!(CacheState::Native.as_str(), "native");
    }

    #[test]
    fn aggregator_tracks_termination_breakdown() {
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 0));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 0));
        agg.record(&sample_telemetry(
            Termination::MaxTokens,
            CacheState::Miss,
            0,
        ));
        agg.record(&sample_telemetry(Termination::Error, CacheState::Miss, 0));
        let s = agg.snapshot();
        assert_eq!(s.term_eot, 2);
        assert_eq!(s.term_max_tokens, 1);
        assert_eq!(s.term_error, 1);
        // Non-EOT rate = (1 + 1) / 4 = 0.5
        assert_eq!(s.non_eot_rate(), Some(0.5));
    }

    #[test]
    fn aggregator_tracks_ttft_mean_and_max() {
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 100));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 200));
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 300));
        // Zero TTFT samples must not skew the mean.
        agg.record(&sample_telemetry(Termination::Error, CacheState::Miss, 0));
        let s = agg.snapshot();
        assert_eq!(s.first_token_us_mean, 200);
        assert_eq!(s.first_token_us_max, 300);
    }

    // ── TTFT histogram percentiles ──────────────────────────────────

    #[test]
    fn bucket_index_maps_values_to_correct_bucket() {
        // Boundary values go to the bucket whose upper bound matches.
        assert_eq!(ttft_bucket_index(1_000), 0); // 1ms bucket
        assert_eq!(ttft_bucket_index(999), 0); // sub-1ms also falls in first
        assert_eq!(ttft_bucket_index(1_001), 1); // 10ms bucket
        assert_eq!(ttft_bucket_index(100_000), 4); // 100ms bucket
        assert_eq!(ttft_bucket_index(10_000_000), 10); // 10s bucket
        assert_eq!(ttft_bucket_index(15_000_000), 11); // overflow
    }

    #[test]
    fn percentiles_resolve_over_buckets() {
        let agg = TelemetryAggregator::default();
        // 100 samples spread across buckets: 50 @ 10ms, 40 @ 100ms, 9 @ 1s, 1 @ 5s.
        for _ in 0..50 {
            agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 5_000));
        }
        for _ in 0..40 {
            agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 75_000));
        }
        for _ in 0..9 {
            agg.record(&sample_telemetry(
                Termination::Eot,
                CacheState::Hit,
                900_000,
            ));
        }
        for _ in 0..1 {
            agg.record(&sample_telemetry(
                Termination::Eot,
                CacheState::Hit,
                4_000_000,
            ));
        }
        let s = agg.snapshot();

        // p50 = 50th sample → first half all in the 10ms bucket.
        assert_eq!(s.ttft_p50_us(), Some(10_000));
        // p90 = 90th sample → cumulative 50 (10ms) + 40 (100ms) = 90.
        // First bucket that reaches 90 is the 100ms bucket.
        assert_eq!(s.ttft_p90_us(), Some(100_000));
        // p99 = 99th sample → cumulative 50+40+9 = 99. 1s bucket wins.
        assert_eq!(s.ttft_p99_us(), Some(1_000_000));
    }

    #[test]
    fn percentile_none_on_empty_histogram() {
        let agg = TelemetryAggregator::default();
        let s = agg.snapshot();
        assert_eq!(s.ttft_p50_us(), None);
        assert_eq!(s.ttft_p90_us(), None);
        assert_eq!(s.ttft_p99_us(), None);
    }

    #[test]
    fn percentile_clamps_inputs_to_valid_range() {
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(Termination::Eot, CacheState::Hit, 5_000));
        let s = agg.snapshot();
        // p > 1.0 and p < 0.0 are clamped. Both end up valid.
        assert_eq!(s.percentile_us(1.5), Some(10_000));
        assert_eq!(s.percentile_us(-0.5), Some(10_000));
    }

    #[test]
    fn overflow_bucket_returns_max_upper_bound() {
        let agg = TelemetryAggregator::default();
        agg.record(&sample_telemetry(
            Termination::Eot,
            CacheState::Hit,
            20_000_000,
        ));
        let s = agg.snapshot();
        assert_eq!(s.ttft_p50_us(), Some(u64::MAX));
    }

    #[test]
    fn ttft_bucket_upper_bounds_exposes_the_thresholds() {
        // Dashboard integrations need the thresholds to render the
        // histogram correctly. Lock in the public contract.
        let bounds = ttft_bucket_upper_bounds_us();
        assert_eq!(bounds.first(), Some(&1_000));
        assert_eq!(bounds.last(), Some(&10_000_000));
    }

    #[test]
    fn aggregator_flags_poison_turns_when_specials_leak() {
        let agg = TelemetryAggregator::default();
        let mut clean = sample_telemetry(Termination::Eot, CacheState::Hit, 100);
        clean.reply_shape.special_token_count = 0;
        agg.record(&clean);
        let mut poisoned = sample_telemetry(Termination::Eot, CacheState::Hit, 100);
        poisoned.reply_shape.special_token_count = 3;
        agg.record(&poisoned);
        let s = agg.snapshot();
        assert_eq!(s.poison_turns, 1);
        assert_eq!(s.turns_total, 2);
        assert_eq!(s.poison_rate(), Some(0.5));
    }

    #[test]
    fn snapshot_rates_are_none_on_empty_aggregator() {
        let agg = TelemetryAggregator::default();
        let s = agg.snapshot();
        assert_eq!(s.cache_hit_rate(), None);
        assert_eq!(s.non_eot_rate(), None);
        assert_eq!(s.poison_rate(), None);
    }

    #[test]
    fn serde_wire_format_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&CacheState::NearMiss).unwrap(),
            "\"near_miss\""
        );
        assert_eq!(
            serde_json::to_string(&Termination::MaxTokens).unwrap(),
            "\"max_tokens\""
        );
        assert_eq!(
            serde_json::to_string(&Termination::LoopDetector).unwrap(),
            "\"loop_detector\""
        );
    }
}
