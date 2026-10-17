//! Non-reserving inspection through the same planner and admission predicates.
use super::{Error, Result, Runtime};
use crate::{
    memory::{MemoryGovernor, MemorySnapshot},
    residency_policy::ModelMemoryPlan,
};
use std::path::Path;

/// Individual inference charges at the observation, in MiB.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdmissionLedger {
    pub resident_models_mb: u64,
    pub speech_reserved_mb: u64,
    pub decision_reserved_mb: u64,
    pub diarization_reserved_mb: u64,
    pub working_reserved_mb: u64,
    /// None means the sum overflowed; inspection must refuse.
    pub total_mb: Option<u64>,
}

impl AdmissionLedger {
    pub(super) fn available_mb(&self, governor: &MemoryGovernor) -> u64 {
        self.total_mb.map_or(0, |total| {
            crate::residency_policy::available_for_model_mb(governor).min(
                governor
                    .budgets()
                    .inference_resident_mb
                    .saturating_sub(total),
            )
        })
    }
}

/// Every independent guard is evaluated, including simultaneous failures.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdmissionChecks {
    pub projected_ledger_mb: Option<u64>,
    pub projected_process_mb: Option<u64>,
    pub quota_allows: bool,
    pub process_allows: bool,
    pub pressure_allows: bool,
    pub overflowed: bool,
}

impl AdmissionChecks {
    pub fn allows(&self) -> bool {
        !self.overflowed && self.quota_allows && self.process_allows && self.pressure_allows
    }
}

/// A local GGUF memory-policy observation, never an admission reservation.
///
/// No model is evicted or loaded. A later normal load may evict eligible models
/// and always repeats admission. Fit establishes neither backend compatibility
/// nor a measured native allocation bound. All memory quantities are MiB.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LocalAdmissionReport {
    pub plan: ModelMemoryPlan,
    pub snapshot: MemorySnapshot,
    pub ledger: AdmissionLedger,
    pub remaining_quota_mb: u64,
    pub remaining_process_mb: u64,
    pub checks: AdmissionChecks,
    pub fits_without_eviction: bool,
}

impl Runtime {
    /// Inspect a local GGUF using this Runtime's actual settings and policy.
    ///
    /// Reads headers only. Does not resolve Hub references, instantiate an
    /// Engine, evict, reserve memory or load weights. The final plan is retained
    /// even when denied; explicit contexts never shrink. Unspecified contexts
    /// use normal planning rules. No projector is inferred for this text load.
    /// The snapshot can become stale immediately and is not permission to load.
    pub fn inspect_local_admission(&self, path: impl AsRef<Path>) -> Result<LocalAdmissionReport> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::InvalidRequest(
                "inspection requires a local GGUF file".into(),
            ));
        }
        let _admission = self
            .inner
            .admission
            .lock()
            .map_err(|_| Error::Load("admission lock poisoned".into()))?;
        let governor = self.inner.config.memory_governor();
        let ledger = self.admission_ledger()?;
        let remaining_quota_mb = ledger.total_mb.map_or(0, |total| {
            governor
                .budgets()
                .inference_resident_mb
                .saturating_sub(total)
        });
        let remaining_process_mb = governor
            .budgets()
            .process_soft_limit_mb
            .saturating_sub(governor.snapshot().estimated_process_mb)
            .saturating_sub(1);
        let plan = crate::residency_policy::inspect_gguf_plan(
            path,
            &self.inner.settings,
            self.inner.config.max_active_chats,
            ledger.available_mb(&governor),
        )
        .map_err(Error::Load)?
        .ok_or_else(|| Error::Load("inspection requires supported GGUF geometry".into()))?;
        let checks = Self::check_admission(&ledger, plan.total_mb, &governor);
        Ok(LocalAdmissionReport {
            fits_without_eviction: checks.allows(),
            plan,
            snapshot: governor.snapshot().clone(),
            ledger,
            remaining_quota_mb,
            remaining_process_mb,
            checks,
        })
    }

    pub(super) fn admission_ledger(&self) -> Result<AdmissionLedger> {
        let resident = self
            .inner
            .models
            .lock()
            .map_err(|_| Error::Load("model registry lock poisoned".into()))?
            .values()
            .filter(|model| model.source.is_local() && model.is_resident())
            .try_fold(0u64, |sum, model| sum.checked_add(model.estimated_mb));
        let speech = self.speech_reserved_mb();
        let decision = self.decision_reserved_mb();
        let diarization = self.diarization_reserved_mb();
        let working = self.working_memory_reserved_mb();
        let total_mb = resident.and_then(|resident| {
            [speech, decision, diarization, working]
                .into_iter()
                .try_fold(resident, |sum, value| sum.checked_add(value))
        });
        Ok(AdmissionLedger {
            resident_models_mb: resident.unwrap_or(u64::MAX),
            speech_reserved_mb: speech,
            decision_reserved_mb: decision,
            diarization_reserved_mb: diarization,
            working_reserved_mb: working,
            total_mb,
        })
    }

    pub(super) fn check_admission(
        ledger: &AdmissionLedger,
        extra_mb: u64,
        governor: &MemoryGovernor,
    ) -> AdmissionChecks {
        let projected = ledger
            .total_mb
            .and_then(|total| total.checked_add(extra_mb));
        let process = governor.model_load_conditions(extra_mb);
        AdmissionChecks {
            projected_ledger_mb: projected,
            projected_process_mb: process.projected_process_mb,
            quota_allows: projected
                .is_some_and(|value| value <= governor.budgets().inference_resident_mb),
            process_allows: process.process_allows,
            pressure_allows: process.pressure_allows,
            overflowed: projected.is_none() || process.projected_process_mb.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced::{BackendPlugin, generation::Settings};
    use crate::memory::{MemoryPolicyInput, MemoryPressureLevel};
    use crate::test_support::local_gguf::hybrid_header;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn fixture(
        quota: u64,
        soft: u64,
        rss: u64,
        pressure: MemoryPressureLevel,
        context: Option<u32>,
    ) -> (Runtime, Arc<AtomicUsize>) {
        let mut snapshot = MemorySnapshot::new(
            &MemoryPolicyInput {
                total_memory_mb: 8192,
                available_memory_mb: 6000,
                is_mobile: true,
            },
            rss,
        );
        snapshot.budgets.inference_resident_mb = quota;
        snapshot.budgets.process_soft_limit_mb = soft;
        snapshot.pressure = pressure;
        let starts = Arc::new(AtomicUsize::new(0));
        let observed = starts.clone();
        let mut settings = Settings::default();
        settings.system.ctx_size = context;
        settings.system.gpu_layers = Some(0);
        let mut builder = Runtime::builder()
            .settings(settings)
            .max_active_sessions(1)
            .backend(BackendPlugin {
                name: "inspection-must-not-start",
                claims: |_| true,
                make: Box::new(move || {
                    observed.fetch_add(1, Ordering::SeqCst);
                    panic!("inspection constructed a backend");
                }),
            });
        builder.config.memory_snapshot_override = Some(snapshot);
        (builder.build().unwrap(), starts)
    }

    #[test]
    fn all_policy_failures_are_visible_without_starting_a_backend() {
        use MemoryPressureLevel::{Normal, Severe};
        let directory = tempfile::tempdir().unwrap();
        let path = hybrid_header(directory.path());
        for (quota, soft, pressure, expected) in [
            (1209, 5000, Normal, (false, true, true)),
            (5000, 1310, Normal, (true, false, true)),
            (5000, 5000, Severe, (true, true, false)),
            (1209, 1310, Severe, (false, false, false)),
            (1210, 1311, Normal, (true, true, true)),
        ] {
            let (runtime, starts) = fixture(quota, soft, 100, pressure, Some(1024));
            let before = runtime.stats();
            for _ in 0..2 {
                let report = runtime.inspect_local_admission(&path).unwrap();
                let checks = &report.checks;
                assert_eq!(
                    (
                        checks.quota_allows,
                        checks.process_allows,
                        checks.pressure_allows
                    ),
                    expected
                );
                assert_eq!(report.plan.total_mb, 1210);
                assert_eq!(checks.projected_ledger_mb, Some(1210));
                assert_eq!(checks.projected_process_mb, Some(1310));
                assert!(!checks.overflowed);
                assert_eq!(
                    report.fits_without_eviction,
                    expected.0 && expected.1 && expected.2
                );
                assert_eq!(runtime.stats(), before);
                assert_eq!(starts.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn resident_and_all_auxiliary_charges_are_counted_without_eviction() {
        let directory = tempfile::tempdir().unwrap();
        let path = hybrid_header(directory.path());
        let (runtime, starts) = fixture(2000, 5000, 100, MemoryPressureLevel::Normal, Some(1024));
        let model = runtime.scripted_in(crate::test_support::Script::new(), 200);
        let working = runtime.reserve_working_memory(144).unwrap();
        // Synthetic metadata for native worker classes; no workers or weights.
        runtime
            .inner
            .speech_reserved_mb
            .store(384, Ordering::SeqCst);
        runtime
            .inner
            .decision_reserved_mb
            .store(64, Ordering::SeqCst);
        runtime
            .inner
            .diarization_reserved_mb
            .store(32, Ordering::SeqCst);
        let before = runtime.stats();
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert_eq!(report.ledger.resident_models_mb, 200);
        assert_eq!(report.ledger.speech_reserved_mb, 384);
        assert_eq!(report.ledger.decision_reserved_mb, 64);
        assert_eq!(report.ledger.diarization_reserved_mb, 32);
        assert_eq!(report.ledger.working_reserved_mb, 144);
        assert_eq!(report.ledger.total_mb, Some(824));
        assert_eq!(report.checks.projected_ledger_mb, Some(2034));
        assert_eq!(report.remaining_quota_mb, 1176);
        assert!(!report.fits_without_eviction);
        assert!(runtime.residency().is_resident(model.id()));
        assert_eq!(runtime.stats(), before);
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        runtime.inner.speech_reserved_mb.store(0, Ordering::SeqCst);
        runtime
            .inner
            .decision_reserved_mb
            .store(0, Ordering::SeqCst);
        runtime
            .inner
            .diarization_reserved_mb
            .store(0, Ordering::SeqCst);
        drop(working);
        assert_eq!(runtime.working_memory_reserved_mb(), 0);
    }

    #[test]
    fn ledger_and_process_overflow_refuse_with_all_other_checks_visible() {
        let directory = tempfile::tempdir().unwrap();
        let path = hybrid_header(directory.path());
        let (runtime, _) = fixture(5000, 5000, 100, MemoryPressureLevel::Normal, Some(1024));
        runtime
            .inner
            .working_memory_reserved_mb
            .store(u64::MAX, Ordering::SeqCst);
        runtime.inner.speech_reserved_mb.store(1, Ordering::SeqCst);
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert_eq!(report.ledger.total_mb, None);
        assert!(report.checks.overflowed);
        assert!(!report.checks.quota_allows);
        assert!(report.checks.process_allows && report.checks.pressure_allows);
        assert!(!report.fits_without_eviction);
        runtime.inner.speech_reserved_mb.store(0, Ordering::SeqCst);
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert_eq!(report.ledger.total_mb, Some(u64::MAX));
        assert_eq!(report.checks.projected_ledger_mb, None);
        assert!(report.checks.overflowed && !report.fits_without_eviction);
        let (runtime, _) = fixture(
            5000,
            u64::MAX,
            u64::MAX,
            MemoryPressureLevel::Normal,
            Some(1024),
        );
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert!(report.checks.quota_allows && report.checks.pressure_allows);
        assert_eq!(report.checks.projected_process_mb, None);
        assert!(report.checks.overflowed && !report.fits_without_eviction);
    }

    #[test]
    fn unspecified_context_selection_and_explicit_refusal_match_normal_loading() {
        let directory = tempfile::tempdir().unwrap();
        let path = hybrid_header(directory.path());
        let (runtime, starts) = fixture(1230, 5000, 100, MemoryPressureLevel::Normal, None);
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert_eq!(report.plan.context_tokens, 2048);
        assert_eq!(report.plan.total_mb, 1222);
        assert!(report.plan.context_reduced && report.fits_without_eviction);
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        let (runtime, starts) = fixture(1230, 5000, 100, MemoryPressureLevel::Normal, Some(4096));
        let report = runtime.inspect_local_admission(&path).unwrap();
        assert_eq!(report.plan.context_tokens, 4096);
        assert_eq!(report.plan.total_mb, 1246);
        assert!(!report.plan.context_reduced && !report.fits_without_eviction);
        assert!(
            runtime
                .load(&path)
                .unwrap_err()
                .to_string()
                .contains("model admission denied")
        );
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        assert!(runtime.models().is_empty());
    }

    #[test]
    fn malformed_and_nonlocal_requests_never_construct_a_backend() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("broken.gguf");
        std::fs::write(&path, b"not GGUF").unwrap();
        let (runtime, starts) = fixture(5000, 5000, 100, MemoryPressureLevel::Normal, Some(1024));
        let before = runtime.stats();
        assert!(runtime.inspect_local_admission(&path).is_err());
        assert!(runtime.inspect_local_admission(directory.path()).is_err());
        assert!(
            runtime
                .inspect_local_admission("hf:does-not-resolve/model")
                .is_err()
        );
        assert_eq!(runtime.stats(), before);
        assert_eq!(starts.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn poisoned_model_registry_is_an_error_not_an_empty_ledger() {
        let directory = tempfile::tempdir().unwrap();
        let path = hybrid_header(directory.path());
        let runtime = Runtime::builder()
            .resident_memory_budget_mb(0)
            .build()
            .unwrap();
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = runtime.inner.models.lock().unwrap();
            panic!("synthetic registry failure");
        }));
        assert!(poisoned.is_err());
        assert!(
            runtime.inspect_local_admission(&path).is_err(),
            "inspection must not report a decision from an unreadable registry"
        );
    }
}
