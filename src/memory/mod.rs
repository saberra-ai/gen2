//! Memory governance — how much of the machine a resident model may take.
//!
//! Moved from `pio-core::diagnostics` when gen2 became its own crate. The
//! residency policy is the only real consumer: it needs the machine's memory
//! tier, the current pressure level, and the budget those imply before it will
//! admit another runtime. `pio-core`'s memory *reporting* stayed behind — it
//! reads this, not the other way around.

pub mod memory_governor;
pub mod memory_policy;
pub mod memory_pressure;
pub mod memory_snapshot;
pub mod memory_tier;
pub mod runtime_memory;

pub use memory_governor::MemoryGovernor;
pub use memory_policy::{
    MemoryBudgets, MemoryPolicyInput, base_budgets_for_tier, detect_machine_tier, effective_budgets,
};
pub use memory_pressure::{MemoryPressureLevel, classify_pressure};
pub use memory_snapshot::MemorySnapshot;
pub use memory_tier::MachineMemoryTier;
pub use runtime_memory::{
    current_memory_governor, current_memory_policy_input, current_memory_snapshot,
};
