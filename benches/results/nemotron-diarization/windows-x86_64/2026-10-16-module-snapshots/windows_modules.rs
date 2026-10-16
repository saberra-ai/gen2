//! Current-process module evidence for qualification diagnostics only.
#![cfg(windows)]

use std::{ffi::OsString, io, os::windows::ffi::OsStringExt, path::PathBuf};

type Handle = *mut std::ffi::c_void;
const MAX_MODULES: usize = 4096;
const MAX_PATH_UNITS: usize = 32768;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> Handle;
    fn K32EnumProcessModules(
        process: Handle,
        modules: *mut Handle,
        capacity_bytes: u32,
        needed_bytes: *mut u32,
    ) -> i32;
    fn GetModuleFileNameW(module: Handle, path: *mut u16, capacity_units: u32) -> u32;
}

fn snapshot() -> io::Result<Vec<(usize, PathBuf)>> {
    let mut modules = vec![std::ptr::null_mut(); MAX_MODULES];
    let mut needed = 0;
    // SAFETY: the pseudo-handle refers to this process; the writable array and
    // DWORD are valid for the exact advertised byte capacities. These module
    // handles are borrowed snapshots and must never be closed/freed.
    let success = unsafe {
        K32EnumProcessModules(
            GetCurrentProcess(),
            modules.as_mut_ptr(),
            (modules.len() * std::mem::size_of::<Handle>()) as u32,
            &mut needed,
        )
    };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    let width = std::mem::size_of::<Handle>();
    if needed == 0
        || !(needed as usize).is_multiple_of(width)
        || needed as usize > modules.len() * width
    {
        return Err(io::Error::other(
            "invalid or oversized Windows module snapshot",
        ));
    }
    modules.truncate(needed as usize / width);
    let mut result = Vec::with_capacity(modules.len());
    for module in modules {
        if module.is_null() {
            return Err(io::Error::other("null Windows module handle"));
        }
        let mut path = vec![0u16; MAX_PATH_UNITS];
        // SAFETY: module came from the current-process snapshot and path is a
        // live writable WCHAR buffer. A concurrent unload fails/retries below.
        let length = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) };
        if length == 0 {
            return Err(io::Error::last_os_error());
        }
        if length as usize >= path.len() {
            return Err(io::Error::other("truncated Windows module filename"));
        }
        let path = PathBuf::from(OsString::from_wide(&path[..length as usize]));
        if !path.is_absolute() || path.to_str().is_none() {
            return Err(io::Error::other(
                "non-absolute or non-Unicode Windows module filename",
            ));
        }
        result.push((module as usize, path));
    }
    result.sort();
    if result.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(io::Error::other("duplicate Windows module snapshot handle"));
    }
    Ok(result)
}

/// Require two consecutive equal handle/path snapshots; bound loader races.
pub fn stable_paths() -> io::Result<Vec<PathBuf>> {
    let mut previous = None;
    let mut last_error = io::Error::other("Windows module list did not stabilize");
    for _ in 0..5 {
        match snapshot() {
            Ok(current) => {
                if previous.as_ref() == Some(&current) {
                    return Ok(current.into_iter().map(|(_, path)| path).collect());
                }
                previous = Some(current);
            }
            Err(error) => {
                last_error = error;
                previous = None;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_executable_is_in_stable_snapshot() {
        let paths = stable_paths().unwrap();
        let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
        assert!(
            paths
                .iter()
                .any(|path| path.canonicalize().unwrap() == executable)
        );
        assert!(paths.iter().any(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .eq_ignore_ascii_case("kernel32.dll")
        }));
        println!("actual modules: {paths:?}");
    }

    #[test]
    fn unicode_relocated_executable_keeps_its_actual_module_path() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("gen2 module ü {} {nonce}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let executable = root.join("module evidence ü.exe");
        std::fs::copy(std::env::current_exe().unwrap(), &executable).unwrap();
        let prefix = module_path!().split_once("::").unwrap().1;
        let filter = format!("{prefix}::current_executable_is_in_stable_snapshot");
        let result = std::process::Command::new(&executable)
            .args(["--exact", &filter, "--nocapture"])
            .output();
        std::fs::remove_dir_all(root).unwrap();
        let result = result.unwrap();
        let output = String::from_utf8(result.stdout).unwrap();
        println!("{output}");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            output.contains("1 passed"),
            "relocated module test did not execute"
        );
    }
}
