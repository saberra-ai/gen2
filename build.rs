fn main() {
    #[cfg(feature = "asr-resample-soxr")]
    {
        println!("cargo:rerun-if-changed=assets/bark/soxr");
        // Match Python-SoXR 1.1.0's bundled libsoxr configuration, including
        // the SIMD/PFFFT HQ engine. Scalar FFT4G changes greedy word decisions
        // on a measured Windows fixture despite very small sample errors.
        let destination = cmake::Config::new("assets/bark/soxr")
            .profile("Release")
            .define("BUILD_TESTS", "OFF")
            .define("BUILD_EXAMPLES", "OFF")
            .define("WITH_OPENMP", "OFF")
            .define("WITH_LSR_BINDINGS", "OFF")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .build();
        println!(
            "cargo:rustc-link-search=native={}/lib",
            destination.display()
        );
        println!("cargo:rustc-link-lib=static=soxr");
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
            println!("cargo:rustc-link-lib=m");
        }
    }
}
