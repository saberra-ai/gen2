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
    current_memory_snapshot_with_policy(Default::default())
}

pub fn current_memory_snapshot_with_policy(policy: super::DesktopMemoryPolicy) -> MemorySnapshot {
    let input = current_memory_policy_input();
    MemorySnapshot::with_desktop_policy(
        &input,
        detect_process_memory_mb(),
        detect_available_commit_mb(),
        policy,
    )
}

#[cfg(target_os = "windows")]
fn detect_available_commit_mb() -> Option<u64> {
    windows_memory_status()
        .map(|status| bytes_to_mb(status.ullAvailPageFile.min(status.ullAvailVirtual)))
}

#[cfg(not(target_os = "windows"))]
fn detect_available_commit_mb() -> Option<u64> {
    None
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
    windows_memory_status().map_or(0, |status| bytes_to_mb(status.ullAvailPhys))
}

#[cfg(target_os = "windows")]
fn windows_memory_status() -> Option<windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX>
{
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
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
    // Admission needs live RSS: ru_maxrss never falls after a model unloads.
    // statm's resident count is in host pages (not necessarily 4 KiB).
    if let Some(pages) = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|text| text.split_whitespace().nth(1)?.parse::<u64>().ok())
    {
        // SAFETY: sysconf reads a process-independent system property.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size > 0 {
            return pages.saturating_mul(page_size as u64).div_ceil(1024 * 1024);
        }
    }
    // A missing procfs keeps the previous conservative peak-RSS fallback.
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
    // SAFETY: proc_pidinfo fills the correctly sized writable structure for
    // our own PID. A short/error result is not inspected; it falls back below.
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    let read = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            (&mut info as *mut libc::proc_taskinfo).cast(),
            size,
        )
    };
    if read == size {
        return info.pti_resident_size.div_ceil(1024 * 1024);
    }
    // Preserve the conservative fallback if the live probe is unavailable.
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

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    #[ignore = "touches 128 MiB; run alone in a fresh process"]
    fn snapshot_tracks_released_pages_not_lifetime_peak() {
        const SIZE: usize = 128 * 1024 * 1024;
        struct Mapping(*mut libc::c_void);
        impl Drop for Mapping {
            fn drop(&mut self) {
                // SAFETY: exactly the live mapping returned below, owned once.
                assert_eq!(unsafe { libc::munmap(self.0, SIZE) }, 0);
            }
        }
        let before = super::current_memory_snapshot().estimated_process_mb;
        // SAFETY: anonymous private mapping, no requested address or file.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(pointer, libc::MAP_FAILED);
        let mapping = Mapping(pointer);
        for offset in (0..SIZE).step_by(4096) {
            // SAFETY: every offset is inside the writable mapping. Volatile
            // stores make these pages resident even in an optimized build.
            unsafe { pointer.cast::<u8>().add(offset).write_volatile(1) };
        }
        let during = super::current_memory_snapshot().estimated_process_mb;
        assert!(
            during >= before + 112,
            "pages must be resident: {before} -> {during}"
        );
        drop(mapping);
        let after = super::current_memory_snapshot().estimated_process_mb;
        assert!(
            after + 96 < during,
            "released pages must leave the snapshot: {during} -> {after}"
        );
    }

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
