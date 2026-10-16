#[path="/Users/victor/workspace/gen2-nemotron3-n0/integrations/pnola/overlay/src-tauri/src/verification_images.rs"]
mod verification_images;
unsafe extern "C" { fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void; fn dlclose(handle: *mut std::ffi::c_void) -> i32; }
fn main() { verification_images::start(); verification_images::start(); eprintln!("BEFORE_DLOPEN"); let handle = unsafe { dlopen(c"/usr/lib/libiconv.2.dylib".as_ptr(), 2) }; assert!(!handle.is_null()); assert_eq!(unsafe { dlclose(handle) }, 0); eprintln!("AFTER_DLCLOSE"); println!("DYLD_PRINT_LIBRARIES={}", std::env::var("DYLD_PRINT_LIBRARIES").unwrap_or("<absent>".into())); }
