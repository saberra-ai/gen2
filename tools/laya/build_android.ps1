param(
    [Parameter(Mandatory=$true)][string]$Ndk,
    [Parameter(Mandatory=$true)][string]$Ort,
    [int]$Api = 24,
    [string]$RustToolchain = '1.95.0',
    [switch]$AllowDependencyDownload
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ndkRoot = (Resolve-Path -LiteralPath $Ndk).Path
$ortRoot = (Resolve-Path -LiteralPath $Ort).Path
$bin = Join-Path $ndkRoot 'toolchains/llvm/prebuilt/windows-x86_64/bin'
$cc = Join-Path $bin "aarch64-linux-android$Api-clang.cmd"
$cxx = Join-Path $bin "aarch64-linux-android$Api-clang++.cmd"
$native = Join-Path $ortRoot 'jni/arm64-v8a'
if (!(Test-Path -LiteralPath $cc) -or !(Test-Path -LiteralPath (Join-Path $native 'libonnxruntime.so'))) {
    throw 'Expected a Windows NDK and an extracted arm64-v8a ONNX Runtime AAR'
}
Set-Location -LiteralPath $workspace
$env:CC_aarch64_linux_android = $cc
$env:CXX_aarch64_linux_android = $cxx
$env:AR_aarch64_linux_android = Join-Path $bin 'llvm-ar.exe'
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $cc
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS = '-C link-arg=-Wl,-z,max-page-size=16384'
$env:ORT_LIB_LOCATION = $native
$env:ORT_PREFER_DYNAMIC_LINK = '1'
# Bound compilation concurrency while keeping the native runtime's thread policy independent.
$cargoArguments = @("+$RustToolchain", 'build', '--locked', '-j', '2', '-p', 'gen2-laya-mobile', '--release', '--target', 'aarch64-linux-android')
if (!$AllowDependencyDownload) { $cargoArguments += '--offline' }
& cargo @cargoArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$rustLibrary = Join-Path $workspace 'target/aarch64-linux-android/release/libgen2_laya_mobile.so'
$output = Join-Path $workspace 'target/laya/android/jniLibs/arm64-v8a'
New-Item -ItemType Directory -Force -Path $output | Out-Null
# Compile/link JNI with the same API, ABI and 16 KiB page alignment as Rust.
$jniArguments = @('-std=c++17', '-O2', '-fPIC', '-shared', '-static-libstdc++', '-Wl,-z,max-page-size=16384',
    'examples/laya-mobile/android/laya_jni.cpp', '-L', (Split-Path -Parent $rustLibrary), '-lgen2_laya_mobile',
    '-o', (Join-Path $output 'libgen2_laya_jni.so'))
& $cxx @jniArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item -LiteralPath $rustLibrary -Destination (Join-Path $output 'libgen2_laya_mobile.so') -Force
Copy-Item -LiteralPath (Join-Path $native 'libonnxruntime.so') -Destination (Join-Path $output 'libonnxruntime.so') -Force
& (Join-Path $bin 'llvm-readelf.exe') -h -l -d (Join-Path $output 'libgen2_laya_mobile.so')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Native Android libraries built in $output; physical-device qualification is still required."
