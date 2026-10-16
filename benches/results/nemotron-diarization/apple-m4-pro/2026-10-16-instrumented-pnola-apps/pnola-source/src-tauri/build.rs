fn main() {
    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{}/Info.plist",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    tauri_build::build();
}
