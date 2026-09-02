//! Deciding when a persistent agent should think.
//!
//! A heartbeat is a *stimulus*, not a message. The distinction matters more
//! than it sounds: an agent woken every minute for a week that recorded each
//! wake as `user: heartbeat` would have a transcript of ten thousand identical
//! turns and no room for anything else. Here a wake is a reason to run, the
//! journal records that it happened, and the self-prompt that goes with it is
//! ephemeral — prepended to one inference and never written down as
//! conversation.
//!
//! # The invariants
//!
//! An autonomous process that runs indefinitely gets these wrong slowly, so
//! they are stated once and enforced in one place:
//!
//! - **Never concurrent.** A wake while the agent is working does not start a
//!   second run.
//! - **Coalescing.** Three heartbeats missed while busy are one wake
//!   afterwards, not three queued self-prompts.
//! - **A floor.** Heartbeats closer together than [`MIN_INTERVAL`] are a
//!   configuration mistake, and on a local runtime an expensive one.
//! - **Boredom.** An agent that wakes, finds nothing, and does nothing must
//!   eventually stop waking, or it burns a battery flat rediscovering that
//!   there is nothing to do.

use std::time::{Duration, Instant};

use super::entry::WakeReason;

/// The closest together heartbeats may be.
///
/// Not arbitrary: a wake is a full inference — prefill and decode — and on a
/// local runtime it contends with whatever the user is doing for the same
/// accelerator. A sub-minute heartbeat means the model is essentially never
/// idle, and the foreground pays for it in latency.
pub const MIN_INTERVAL: Duration = Duration::from_secs(60);

/// How often to wake, and when to give up.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    interval: Duration,
    /// Consecutive wakes that changed nothing before heartbeats stop.
    ///
    /// `None` never stops. That is a legitimate choice for an agent watching
    /// for external change, and the wrong default for one working through a
    /// task it has finished.
    idle_after: Option<u32>,
}

impl Heartbeat {
    /// Wake every `interval`, clamped to [`MIN_INTERVAL`].
    ///
    /// Clamped rather than rejected: a caller asking for ten seconds wants
    /// "often", and failing their build over it helps nobody. What it must not
    /// do is silently comply.
    pub fn every(interval: Duration) -> Self {
        Self {
            interval: interval.max(MIN_INTERVAL),
            idle_after: Some(5),
        }
    }

    /// Stop after this many consecutive wakes that changed nothing.
    pub fn idle_after(mut self, wakes: u32) -> Self {
        self.idle_after = Some(wakes);
        self
    }

    /// Keep waking forever, however little happens.
    pub fn never_bored(mut self) -> Self {
        self.idle_after = None;
        self
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }
}

/// Why a scheduler declined to wake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Declined {
    /// Not time yet.
    TooSoon,
    /// The agent is already working. One wake is now pending; further
    /// heartbeats before it runs fold into that one.
    Busy,
    /// Nothing has happened for long enough that heartbeats have stopped.
    /// Only an external wake restarts them.
    Bored,
}

/// Decides whether a heartbeat should actually run.
///
/// Deliberately holds no clock of its own and spawns nothing: it is a state
/// machine the caller ticks. That makes every invariant above testable without
/// waiting a minute, and lets the caller decide what "now" means — including
/// deferring a wake that would fight the foreground for the accelerator.
#[derive(Debug)]
pub struct WakeScheduler {
    heartbeat: Heartbeat,
    last_wake: Option<Instant>,
    /// A wake asked for while busy, waiting for the agent to be free.
    pending: Option<WakeReason>,
    busy: bool,
    /// Consecutive heartbeat wakes that produced nothing.
    idle_wakes: u32,
}

impl WakeScheduler {
    pub fn new(heartbeat: Heartbeat) -> Self {
        Self {
            heartbeat,
            last_wake: None,
            pending: None,
            busy: false,
            idle_wakes: 0,
        }
    }

    /// The agent started working.
    pub fn began_running(&mut self) {
        self.busy = true;
    }

    /// The agent stopped working.
    ///
    /// `did_something` is what boredom counts: a wake that appended nothing to
    /// the journal did not accomplish anything, however much the model said
    /// while deciding that.
    pub fn finished_running(&mut self, did_something: bool) {
        self.busy = false;
        if did_something {
            self.idle_wakes = 0;
        }
    }

    /// Something outside asked for attention.
    ///
    /// Always honoured eventually — external input is the one thing that must
    /// not be dropped for being too frequent, and it clears boredom, since the
    /// world has evidently changed.
    pub fn request(&mut self, reason: WakeReason) {
        self.idle_wakes = 0;
        self.pending = Some(reason);
    }

    /// Ask what to do at `now`.
    ///
    /// `Ok(reason)` means run. Anything else says why not.
    pub fn poll(&mut self, now: Instant) -> Result<WakeReason, Declined> {
        // A run in flight beats everything. A pending wake stays pending; it
        // does not become a second concurrent run.
        if self.busy {
            if self.pending.is_none() && self.due(now) {
                // Coalescing happens here: the heartbeat that fired while busy
                // becomes *the* pending wake, and the next three fold into it
                // because `pending` is already set.
                self.pending = Some(WakeReason::Heartbeat);
                self.last_wake = Some(now);
            }
            return Err(Declined::Busy);
        }

        // Anything explicitly requested runs next, ahead of the timer.
        if let Some(reason) = self.pending.take() {
            self.last_wake = Some(now);
            return Ok(reason);
        }

        if self
            .heartbeat
            .idle_after
            .is_some_and(|limit| self.idle_wakes >= limit)
        {
            return Err(Declined::Bored);
        }
        if !self.due(now) {
            return Err(Declined::TooSoon);
        }

        self.last_wake = Some(now);
        // Counted optimistically and cleared by `finished_running(true)`. The
        // alternative — counting on completion — would let a long run mask
        // that nothing is happening.
        self.idle_wakes += 1;
        Ok(WakeReason::Heartbeat)
    }

    /// Whether heartbeats have stopped for want of anything to do.
    pub fn is_bored(&self) -> bool {
        self.heartbeat
            .idle_after
            .is_some_and(|limit| self.idle_wakes >= limit)
    }

    fn due(&self, now: Instant) -> bool {
        match self.last_wake {
            None => true,
            Some(last) => now.duration_since(last) >= self.heartbeat.interval,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> WakeScheduler {
        WakeScheduler::new(Heartbeat::every(MIN_INTERVAL))
    }

    fn later(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    #[test]
    fn a_sub_minute_interval_is_raised_to_the_floor() {
        // Silently complying would put a full inference on the accelerator
        // every ten seconds, and the foreground would pay for it.
        let heartbeat = Heartbeat::every(Duration::from_secs(10));
        assert_eq!(heartbeat.interval(), MIN_INTERVAL);
    }

    #[test]
    fn the_first_poll_wakes_immediately() {
        assert_eq!(scheduler().poll(Instant::now()), Ok(WakeReason::Heartbeat));
    }

    #[test]
    fn a_second_poll_before_the_interval_declines() {
        let start = Instant::now();
        let mut s = scheduler();
        s.poll(start).unwrap();
        assert_eq!(s.poll(later(start, 30)), Err(Declined::TooSoon));
        assert_eq!(s.poll(later(start, 60)), Ok(WakeReason::Heartbeat));
    }

    /// The invariant that matters most for something running indefinitely.
    #[test]
    fn a_wake_while_busy_never_starts_a_second_run() {
        let start = Instant::now();
        let mut s = scheduler();
        s.poll(start).unwrap();
        s.began_running();
        for minute in 1..5 {
            assert_eq!(s.poll(later(start, minute * 60)), Err(Declined::Busy));
        }
    }

    /// Three missed heartbeats are one wake, not three.
    #[test]
    fn heartbeats_missed_while_busy_coalesce_into_a_single_wake() {
        let start = Instant::now();
        let mut s = scheduler();
        s.poll(start).unwrap();
        s.began_running();
        for minute in 1..4 {
            let _ = s.poll(later(start, minute * 60));
        }
        s.finished_running(true);

        assert_eq!(
            s.poll(later(start, 240)),
            Ok(WakeReason::Heartbeat),
            "the missed heartbeats should produce one wake"
        );
        assert_eq!(
            s.poll(later(start, 241)),
            Err(Declined::TooSoon),
            "and only one — three accumulated self-prompts is the failure mode"
        );
    }

    /// A wake owed from being busy runs immediately, not on the next tick.
    #[test]
    fn a_pending_wake_runs_as_soon_as_the_agent_is_free() {
        let start = Instant::now();
        let mut s = scheduler();
        s.poll(start).unwrap();
        s.began_running();
        let _ = s.poll(later(start, 60));
        s.finished_running(true);
        assert_eq!(s.poll(later(start, 61)), Ok(WakeReason::Heartbeat));
    }

    #[test]
    fn external_input_is_honoured_ahead_of_the_timer() {
        let start = Instant::now();
        let mut s = scheduler();
        s.poll(start).unwrap();
        s.request(WakeReason::User);
        assert_eq!(
            s.poll(later(start, 1)),
            Ok(WakeReason::User),
            "a person should not wait out the heartbeat interval"
        );
    }

    /// An agent with nothing to do stops waking.
    #[test]
    fn repeated_wakes_that_change_nothing_eventually_stop() {
        let start = Instant::now();
        let mut s = WakeScheduler::new(Heartbeat::every(MIN_INTERVAL).idle_after(3));
        for minute in 0..3 {
            assert_eq!(s.poll(later(start, minute * 60)), Ok(WakeReason::Heartbeat));
            s.began_running();
            s.finished_running(false);
        }
        assert_eq!(s.poll(later(start, 180)), Err(Declined::Bored));
        assert!(s.is_bored());
    }

    /// ...and doing something resets it.
    #[test]
    fn a_wake_that_accomplishes_something_resets_the_boredom_count() {
        let start = Instant::now();
        let mut s = WakeScheduler::new(Heartbeat::every(MIN_INTERVAL).idle_after(2));
        s.poll(start).unwrap();
        s.began_running();
        s.finished_running(false);
        s.poll(later(start, 60)).unwrap();
        s.began_running();
        s.finished_running(true); // did something
        assert_eq!(s.poll(later(start, 120)), Ok(WakeReason::Heartbeat));
    }

    /// External input rouses a bored agent — the world changed.
    #[test]
    fn an_external_wake_brings_a_bored_agent_back() {
        let start = Instant::now();
        let mut s = WakeScheduler::new(Heartbeat::every(MIN_INTERVAL).idle_after(1));
        s.poll(start).unwrap();
        s.began_running();
        s.finished_running(false);
        assert_eq!(s.poll(later(start, 60)), Err(Declined::Bored));

        s.request(WakeReason::External("a file changed".into()));
        assert!(matches!(
            s.poll(later(start, 61)),
            Ok(WakeReason::External(_))
        ));
        assert!(!s.is_bored(), "and heartbeats resume");
    }

    #[test]
    fn never_bored_keeps_waking_regardless() {
        let start = Instant::now();
        let mut s = WakeScheduler::new(Heartbeat::every(MIN_INTERVAL).never_bored());
        for minute in 0..20 {
            let outcome = s.poll(later(start, minute * 60));
            assert_eq!(outcome, Ok(WakeReason::Heartbeat), "at minute {minute}");
            s.began_running();
            s.finished_running(false);
        }
    }
}
