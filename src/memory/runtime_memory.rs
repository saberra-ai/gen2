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

/// Linux and Android: `MemAvailable` from `/proc/meminfo`, which is what
/// the kernel says a new allocation can take without swapping. `sysinfo`'s
/// `freeram` is the wrong number here: it excludes the page cache, and after
/// a large build or a long read the cache is most of RAM. On that reading a
/// 16 GB box reports a few hundred MB "free", and the residency governor
/// refuses to load a 100 MB helper. `MemAvailable` has been in `meminfo`
/// since Linux 3.14; if it is missing the fallback counts free plus buffer
/// pages, which at least does not ignore the cache entirely.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_available_memory_mb() -> u64 {
    if let Some(mb) = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| meminfo_available_mb(&text))
    {
        return mb;
    }
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    unsafe { libc::sysinfo(&mut info) };
    let unit = info.mem_unit as u64;
    bytes_to_mb((info.freeram as u64 + info.bufferram as u64) * unit)
}

/// `MemAvailable:   12345678 kB` → MB. Any platform can parse the text, so
/// the test for it runs everywhere, not only on the runner that needs it.
fn meminfo_available_mb(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb / 1024)
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

#[cfg(test)]
mod tests {
    use super::meminfo_available_mb;

    #[test]
    fn reads_mem_available_not_mem_free() {
        let text = "MemTotal:       16303620 kB\nMemFree:          412872 kB\n\
                    MemAvailable:   13115460 kB\nBuffers:          254000 kB\n";
        assert_eq!(meminfo_available_mb(text), Some(13115460 / 1024));
    }

    #[test]
    fn a_kernel_without_mem_available_yields_nothing() {
        let text = "MemTotal:       16303620 kB\nMemFree:          412872 kB\n";
        assert_eq!(meminfo_available_mb(text), None);
    }
}
