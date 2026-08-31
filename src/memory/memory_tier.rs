//! Machine memory tier enum.
//!
//! A tier is determined once at startup from hardware properties
//! and drives all downstream budget calculations. It is intentionally
//! coarse — five tiers cover the realistic Pio deployment surface.

use serde::Serialize;

/// Coarse classification of the host machine's memory capacity.
///
/// Ordered from smallest to largest so that `<` / `>` comparisons work
/// naturally (e.g. `tier <= DesktopConstrained` catches both mobile and
/// constrained desktop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum MachineMemoryTier {
    /// Mobile platform (iOS / Android). Always assigned when `is_mobile` is
    /// true regardless of reported RAM.
    MobileConstrained,
    /// Non-mobile host with < 8 GiB total RAM.
    DesktopConstrained,
    /// 8–16 GiB — the typical mainstream laptop/desktop.
    DesktopMainstream,
    /// 16–32 GiB — developer machine or recent high-end laptop.
    DesktopPower,
    /// 32 GiB+ — workstation, build server, or high-memory Mac.
    Workstation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering() {
        assert!(MachineMemoryTier::MobileConstrained < MachineMemoryTier::DesktopConstrained);
        assert!(MachineMemoryTier::DesktopConstrained < MachineMemoryTier::DesktopMainstream);
        assert!(MachineMemoryTier::DesktopMainstream < MachineMemoryTier::DesktopPower);
        assert!(MachineMemoryTier::DesktopPower < MachineMemoryTier::Workstation);
    }
}
