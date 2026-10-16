//! Process-local loader history for explicit verification commands only.
//! Hardened signing removes DYLD_PRINT_LIBRARIES, so observe dyld directly.

#[cfg(target_os = "macos")]
mod macos {
    use std::{
        ffi::{CStr, c_char, c_int, c_void},
        io::Write,
        sync::Once,
    };

    #[repr(C)]
    struct DlInfo {
        filename: *const c_char,
        base: *mut c_void,
        symbol: *const c_char,
        address: *mut c_void,
    }

    unsafe extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
        fn _dyld_register_func_for_add_image(callback: extern "C" fn(*const c_void, isize));
    }

    extern "C" fn image_added(header: *const c_void, _slide: isize) {
        let mut info = DlInfo {
            filename: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol: std::ptr::null(),
            address: std::ptr::null_mut(),
        };
        // SAFETY: dyld supplies a mapped Mach-O header for the callback lifetime.
        // dladdr is thread safe; its filename remains valid while this image is
        // loaded. Copy it during this callback, before dyld can unload the image.
        let name = unsafe {
            if dladdr(header, &mut info) == 0 || info.filename.is_null() {
                None
            } else {
                Some(CStr::from_ptr(info.filename).to_bytes())
            }
        };
        let Some(name) = name.filter(|name| !name.is_empty() && name.len() <= 1024 * 1024) else {
            let _ = std::io::stderr().write_all(b"PNOLA_LOADED_IMAGE_ERROR\n");
            return;
        };
        // Hex makes arbitrary path bytes unambiguous, without line injection or
        // dependencies that could themselves load images from this callback.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut line = b"PNOLA_LOADED_IMAGE_HEX ".to_vec();
        for byte in name {
            line.push(HEX[(byte >> 4) as usize]);
            line.push(HEX[(byte & 15) as usize]);
        }
        line.push(b'\n');
        let _ = std::io::stderr().write_all(&line);
    }

    pub fn start() {
        static START: Once = Once::new();
        // SAFETY: process-lifetime C callback with the SDK-declared signature.
        // Registration reports all current images and every later addition,
        // including images subsequently unloaded on a rejected model load.
        START.call_once(|| unsafe { _dyld_register_func_for_add_image(image_added) });
    }
}

pub fn start() {
    #[cfg(target_os = "macos")]
    macos::start();
}
