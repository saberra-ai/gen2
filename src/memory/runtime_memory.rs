//! Best-effort runtime memory snapshot helpers.
//!
//! Production shells use this module to derive a live [`MemoryGovernor`]
//! instead of hard-coding static budgets. The probes are intentionally
//! conservative: if a platform-specific value is unavailable we fall back
//! to a safe approximation rather than failing closed.

use super::{MemoryGovernor, MemoryPolicyInput, MemorySnapshot};
use crate::hardware::HardwareProfile;

pub fn current_memory_policy_input() -> MemoryPolicyInput {
    let hw = HardwareProfile::detect();
    let total_memory_mb = bytes_to_mb(hw.total_ram_bytes);
    let available_memory_mb = detect_available_memory_mb().min(total_memory_mb);

    MemoryPolicyInput {
        total_memory_mb,
        available_memory_mb,
        is_mobile: cfg!(any(target_os = "android", target_os = "ios")),
    }
}

pub fn current_memory_snapshot() -> MemorySnapshot {
    let input = current_memory_policy_input();
    MemorySnapshot::new(&input, detect_process_memory_mb())
}

pub fn current_memory_governor() -> MemoryGovernor {
    MemoryGovernor::new(current_memory_snapshot())
}

fn bytes_to_mb(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_available_memory_mb() -> u64 {
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    unsafe { libc::sysinfo(&mut info) };
    bytes_to_mb(info.freeram as u64 * info.mem_unit as u64)
}

#[cfg(target_os = "macos")]
fn detect_available_memory_mb() -> u64 {
    // macOS free-memory accounting is not exposed by the existing hardware
    // helpers; use total RAM as an upper bound until a finer probe is added.
    bytes_to_mb(HardwareProfile::detect().total_ram_bytes)
}

#[cfg(target_os = "windows")]
fn detect_available_memory_mb() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    unsafe { GlobalMemoryStatusEx(&mut status) };
    bytes_to_mb(status.ullAvailPhys)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows"
)))]
fn detect_available_memory_mb() -> u64 {
    bytes_to_mb(HardwareProfile::detect().total_ram_bytes)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_process_memory_mb() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc == 0 {
        // Linux reports ru_maxrss in KiB.
        (usage.ru_maxrss as u64) / 1024
    } else {
        0
    }
}

#[cfg(target_os = "macos")]
fn detect_process_memory_mb() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc == 0 {
        // macOS reports ru_maxrss in bytes.
        bytes_to_mb(usage.ru_maxrss as u64)
    } else {
        0
    }
}

#[cfg(target_os = "windows")]
fn detect_process_memory_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        )
    };
    if ok != 0 {
        bytes_to_mb(counters.WorkingSetSize as u64)
    } else {
        0
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows"
)))]
fn detect_process_memory_mb() -> u64 {
    0
}
