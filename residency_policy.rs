use crate::diagnostics::MachineMemoryTier;

/// Residency sizing helpers and context-budget policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_context_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyPolicy {
    pub helper_idle_timeout_secs: u64,
    pub llm_swap_requires_unload: bool,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            helper_idle_timeout_secs: 300,
            llm_swap_requires_unload: true,
        }
    }
}

pub fn default_context_budget_for_tier(tier: MachineMemoryTier) -> ContextBudget {
    let max_context_tokens = match tier {
        MachineMemoryTier::MobileConstrained => 4_096,
        MachineMemoryTier::DesktopConstrained => 8_192,
        MachineMemoryTier::DesktopMainstream => 16_384,
        MachineMemoryTier::DesktopPower => 24_576,
        MachineMemoryTier::Workstation => 32_768,
    };
    ContextBudget { max_context_tokens }
}

/// Estimate resident memory for a local runtime path.
///
/// For file-backed runtimes this uses on-disk bytes as an upper-bound proxy and
/// floors tiny fixtures to a realistic minimum so tests stay meaningful.
pub fn estimate_resident_mb_for_path(path: &std::path::Path) -> u64 {
    const MIN_ESTIMATE_MB: u64 = 256;
    let file_mb = std::fs::metadata(path)
        .ok()
        .map(|md| (md.len().saturating_add(1024 * 1024 - 1)) / (1024 * 1024))
        .unwrap_or(0);
    file_mb.max(MIN_ESTIMATE_MB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_budget_grows_with_tier() {
        let mobile = default_context_budget_for_tier(MachineMemoryTier::MobileConstrained);
        let desk = default_context_budget_for_tier(MachineMemoryTier::DesktopMainstream);
        let work = default_context_budget_for_tier(MachineMemoryTier::Workstation);
        assert!(mobile.max_context_tokens < desk.max_context_tokens);
        assert!(desk.max_context_tokens < work.max_context_tokens);
    }

    #[test]
    fn residency_policy_defaults_to_helper_eviction_and_llm_swap_unload() {
        let policy = ResidencyPolicy::default();
        assert!(policy.helper_idle_timeout_secs > 0);
        assert!(policy.llm_swap_requires_unload);
    }
}
