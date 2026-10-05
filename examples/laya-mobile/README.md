# Laya mobile host harness

This small C ABI exposes the same Rust runtime as desktop, with JSON requests
and owned JSON replies. It is test scaffolding, not a claim of mobile device
qualification. Windows compile/clippy and Android arm64 native link checks have
passed. Android device execution and iOS SDK/device runs are still required.

Functions are declared in `gen2_laya.h`. Every reply has `ok` and either `value`
or `error`; release the returned string once with `gen2_laya_free`. Handles are
integer registry IDs, so closing one never leaves an FFI pointer dangling.
In-flight calls hold their own cloned model handle. Suspension/reload epochs
prevent an older resume from overwriting a later suspension or close.

Load JSON:

```json
{"bundle":"/app/private/laya-english","native_library":null,"resident_budget_mb":4096,"queue_capacity":8,"intra_threads":2,"max_input_bytes":16777216}
```

The returned value contains `handle`. Request JSON:

```json
{"state":{"kind":"text","value":"Please refund the duplicate payment."},"questions":[["refund",{"type":"yes_no","instructions":"Is a refund requested?","false_label":"false","true_label":"true","false_description":null,"true_description":null}]]}
```

Run open, inference, resume and close on a background queue. The suspend call is
nonblocking and may be used in a memory-pressure/background callback. It stops
admission immediately; an active kernel still owns buffers until completion.
Resume waits for shutdown and verifies the original bundle digest before load.

`gen2_laya_invoke(handle, json)` (Java `LayaNative.invoke`, Swift
`LayaSmoke.invoke`) exposes the remaining decision operations:

| `operation` | Payload | Result value |
|---|---|---|
| `describe` | No payload | Bundle info, capabilities and current status |
| `decide` | `request`, optional `options` | One decision result |
| `batch` | `requests` array, optional `options` | Results in original state order |
| `long` | `request`, optional `scan` and `options` | Selected answers, window results and scan diagnostics |

`options` accepts `language`, `timeout_ms` and `encode` (the Rust `max_len`,
`head_max_len`, `overflow` fields). `scan` accepts `window_tokens`,
`stride_tokens` and `max_windows`. Defaults match the Rust API. For example:

```json
{"operation":"batch","requests":[],"options":{"timeout_ms":30000}}
```

A relative deadline covers admission, queueing and execution after parsing.
Timeout and suspension are cooperative: the worker retains buffers and memory
reservations until native execution returns. Unknown operation/option fields
are rejected. The original `gen2_laya_decide` remains a default-options shortcut.
Load configuration also exposes the Rust execution-provider policy.

## iOS

The Apple SDK CI lane downloads the full ONNX Runtime **1.24.2** XCFramework
published in Microsoft's [Swift package manifest](https://github.com/microsoft/onnxruntime-swift-package-manager/blob/main/Package.swift).
The archive SHA-256 is `f7100a992d2a8135168c8afd831e6a58b465349101982aa58b3e11d36e600b54`;
its device and simulator slices use ORT API 24, matching the Rust binding.
This patch version is separate from the desktop and Android runtimes and still
requires iOS parity qualification. A reduced operator build needs its own
graph-coverage testing.

On a macOS build host with Xcode, both Rust iOS targets and the active toolchain's
`llvm-tools-preview` component installed, extract that archive and run:

```sh
bash tools/laya/build_ios.sh /absolute/path/onnxruntime.xcframework
```

The script sets `ORT_IOS_XCFWK_PATH`, builds arm64 device and simulator libraries,
links `LayaSmoke.swift` against the C ABI and runtime for both targets, and packages
`target/laya/ios/build/Gen2Laya.xcframework`. The build floor is iOS 15.1.
`inspect_ios.py` checks slices, exported C symbols and linked host binaries, then
records hashes and toolchain versions. Choose a fresh output directory as the
second argument for repeat builds. The CI job uploads these products and evidence.
This is SDK/link qualification; it does not execute a model on a simulator or device.

Add the Gen2 XCFramework and matching ONNX framework to the host app. Include
`gen2_laya.h` in its bridging header, or import the packaged `Gen2Laya` C module.
Link the C++ runtime and Foundation/CoreML/Accelerate frameworks as in the script;
sign/embed dynamic frameworks through Xcode. Bundles live in the app bundle or
app-private storage and are immutable while loaded.

## Android

The local Android link check uses the published ONNX Runtime **1.24.3** AAR,
NDK **r27d (27.3.13750724)**, Rust **1.95.0**, arm64-v8a and Android API **24**.
Maven did not publish a 1.24.4 Android AAR at inspection time; the Windows
runtime remains 1.24.4. Both use ORT API 24. The AAR declares minSdkVersion 24;
this is a build floor, not a claim of device/model qualification.

Build or obtain the matching ONNX Runtime for the same NDK/API/ABI as the application. Set
`ORT_LIB_LOCATION` to the directory containing `libonnxruntime.so` and
`ORT_PREFER_DYNAMIC_LINK=1`; set Cargo's target linker and the C/C++ compiler
environment to the chosen NDK toolchain. Then:

```sh
cargo build -p gen2-laya-mobile --release --target aarch64-linux-android
```

Place `libgen2_laya_mobile.so`, `libonnxruntime.so` and required C++ runtime
dependencies in `jniLibs/arm64-v8a`. Build `android/laya_jni.cpp` with the NDK
against that library and the provided header; load `gen2_laya_jni` from Java.
`android/LayaNative.java` passes UTF-8 byte arrays across JNI, preserving astral
Unicode rather than treating UTF-8 as JNI modified UTF-8. Copy the entire bundle
to an app-private directory before opening it. Do not pass an APK asset URI as
a filesystem path. The host selects and records its supported Android API floor.

## Android host app

### Build and run the Android host app

The `android/` Gradle project packages the inspected arm64 native libraries in
an offline debug APK. It uses AGP 8.9.2, Gradle 8.11.1, compile/target SDK 35
and min SDK 24. The local build used Microsoft OpenJDK 21.0.12.1; AGP 8.9
requires at least JDK 17 ([Google's compatibility table](https://developer.android.com/build/releases/agp-8-9-0-release-notes)).

After `build_android.ps1` produces `target/laya/android/jniLibs`, install SDK
packages `platforms;android-35` and `build-tools;35.0.0`, then run:

```powershell
tools/laya/build_android_apk.ps1 -Sdk <sdk-directory> -JavaHome <jdk-directory> -GradleHome <gradle-8.11.1-directory> -AllowDependencyDownload
```

The download flag is needed for the first Gradle dependency resolution; omit it
for subsequent offline builds. Caches and the local debug signing key stay under
`target/laya/android`. This key is only for the test harness. The script builds,
lints and verifies the signature of
`examples/laya-mobile/android/app/build/outputs/apk/debug/app-debug.apk`.
`inspect_android_apk.py` checks the APK's library hashes, uncompressed 16 KiB
alignment, SDK metadata, synthetic fixture contents and absence of permissions.
The actual local package evidence is in
`tools/laya/evidence/android-arm64/apk-evidence.json`. The CI lane now builds and
inspects the APK too; that workflow has not been run remotely in this session.

On an attached, authorized arm64 device:

```sh
adb install -r examples/laya-mobile/android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n example.gen2.laya/example.gen2.MainActivity
```

Select **smoke** and tap **Run inference and lifecycle checks**. The synthetic
graph is bundled only to test runtime plumbing. The app checks capability
inspection, Unicode input, single/batch/long inference, rejection while suspended, stable output after resume, and rejection
after close. Loading, inference, resume and close use a background executor;
background/memory-pressure events signal nonblocking suspension.

For real models, copy only `manifest.json` and its listed `files` into the app's
private `files/bundles/english`, `multilingual` or `typed-decisions` directory
using `adb run-as example.gen2.laya` while the app is stopped. Preserve relative
paths and licenses; omit the export-only `source/` directory. Then select that
family in the app. These short yes/no smoke checks do not replace the full
reference corpus, shape matrix or sustained device qualification.

Reports are private JSON files and can be collected with:

```sh
adb exec-out run-as example.gen2.laya cat files/reports/smoke-smoke.json
```

Replace the first `smoke` in the filename with the selected real family. The
app requests no network or storage permissions and excludes its files from
backup/transfer. `adb devices -l` found no attached devices in this workspace;
installation and runtime UI/lifecycle behavior remain unverified.

### Device qualification evidence

For **each checkpoint**, archive OS/SDK, architecture, device/RAM, bundle hash,
native runtime hash/version/provider, cold-load time, peak resident memory,
latency distribution and sustained thermal behavior. Run the real reference
corpus, not just the synthetic fixture. Exercise offline launch, Unicode/JSON/
conversation inputs, full declared shapes, simultaneous chat admission,
background during inference, pressure during inference, repeated resume,
overlapping suspend/resume/close, process termination and cold restart.

Return to the design review if the intended device floor cannot run the graph
within its host budget. FP32 is the current qualified export path; do not
silently introduce INT8 or route to HTTP to make a device test pass.

## Reproduce the Windows-to-Android link check

Download [NDK r27d](https://github.com/android/ndk/releases/tag/r27d) and the
[Android 1.24.3 AAR](https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.3/).
Verify the publishers' checksums, extract both under an ignored build directory,
and install Rust's `aarch64-linux-android` target. Then run:

```powershell
tools/laya/build_android.ps1 -Ndk target/laya/android/android-ndk-r27d -Ort target/laya/android/ort-1.24.3
```

The script defaults to offline dependency resolution; use
`-AllowDependencyDownload` for the initial Cargo fetch. It builds the Rust and
JNI libraries with 16 KiB LOAD alignment and stages their native dependencies.
`tools/laya/inspect_android.py` verifies architecture, alignment, exported ABI
symbols and dependency names. An absolute development path in `DT_NEEDED` is a
failure. Committed link evidence is under `tools/laya/evidence/android-arm64/`.
The CI lane repeats compilation/link inspection without claiming device runs.
