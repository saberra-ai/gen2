#!/usr/bin/env bash
# Run on macOS with Xcode and the two Rust iOS targets installed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export ORT_IOS_XCFWK_PATH="$(cd "${1:?Pass the ONNX Runtime XCFramework directory}" && pwd)"
OUT="${2:-$ROOT/target/laya/ios/build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
test ! -e "$OUT/Gen2Laya.xcframework" || { echo 'Choose a fresh output directory.' >&2; exit 1; }
export IPHONEOS_DEPLOYMENT_TARGET=15.1
# The full ORT Apple framework includes CoreML even when CPU inference is selected.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-framework -C link-arg=CoreML -C link-arg=-framework -C link-arg=Accelerate -C link-arg=-framework -C link-arg=Foundation -C link-arg=-lc++"
mkdir -p "$OUT/headers"
cp examples/laya-mobile/gen2_laya.h "$OUT/headers/"
printf 'module Gen2Laya { header "gen2_laya.h" export * }\n' > "$OUT/headers/module.modulemap"
for TARGET in aarch64-apple-ios aarch64-apple-ios-sim; do
  if [[ "$TARGET" == aarch64-apple-ios ]]; then
    SDK=iphoneos
    SWIFT_TARGET=arm64-apple-ios15.1
    SLICE=ios-arm64
  else
    SDK=iphonesimulator
    SWIFT_TARGET=arm64-apple-ios15.1-simulator
    SLICE=ios-arm64_x86_64-simulator
    if [[ ! -d "$ORT_IOS_XCFWK_PATH/$SLICE" ]]; then SLICE=ios-arm64-simulator; fi
  fi
  test -d "$ORT_IOS_XCFWK_PATH/$SLICE/onnxruntime.framework"
  SDKROOT="$(xcrun --sdk "$SDK" --show-sdk-path)" cargo build --locked --release -p gen2-laya-mobile --target "$TARGET"
  LIB="$ROOT/target/$TARGET/release/libgen2_laya_mobile.a"
  # This final link proves that Swift, the C ABI, Rust and ORT resolve together.
  # It does not claim execution on a simulator or a physical device.
  xcrun --sdk "$SDK" swiftc -target "$SWIFT_TARGET" \
    -sdk "$(xcrun --sdk "$SDK" --show-sdk-path)" \
    -import-objc-header examples/laya-mobile/gen2_laya.h \
    -emit-library examples/laya-mobile/LayaSmoke.swift "$LIB" \
    -F "$ORT_IOS_XCFWK_PATH/$SLICE" -framework onnxruntime \
    -framework Foundation -framework CoreML -framework Accelerate \
    -lc++ -lz -liconv -o "$OUT/$TARGET-host.dylib"
  APP="$OUT/$TARGET/Gen2Laya.app"
  mkdir -p "$APP"
  xcrun --sdk "$SDK" swiftc -parse-as-library -target "$SWIFT_TARGET" \
    -sdk "$(xcrun --sdk "$SDK" --show-sdk-path)" \
    -import-objc-header examples/laya-mobile/gen2_laya.h \
    examples/laya-mobile/LayaSmoke.swift examples/laya-mobile/ios/LayaSmokeApp.swift "$LIB" \
    -F "$ORT_IOS_XCFWK_PATH/$SLICE" -framework onnxruntime \
    -framework Foundation -framework CoreML -framework Accelerate -framework SwiftUI -framework UIKit \
    -lc++ -lz -liconv -o "$APP/Gen2Laya"
  cp -R tests/fixtures/laya/smoke "$APP/smoke"
  python3 - "$APP" "$SDK" <<'PY'
import pathlib, plistlib, sys
app, sdk = pathlib.Path(sys.argv[1]), sys.argv[2]
info = dict(CFBundleIdentifier='ai.saberra.gen2.laya.smoke', CFBundleName='Gen2 Laya',
            CFBundleExecutable='Gen2Laya', CFBundlePackageType='APPL', CFBundleVersion='1',
            CFBundleShortVersionString='1.0', MinimumOSVersion='15.1',
            CFBundleSupportedPlatforms=['iPhoneOS' if sdk == 'iphoneos' else 'iPhoneSimulator'],
            UIDeviceFamily=[1, 2], UILaunchScreen={}, UIFileSharingEnabled=True,
            UIApplicationSceneManifest=dict(UIApplicationSupportsMultipleScenes=False),
            LSSupportsOpeningDocumentsInPlace=True,
            UISupportedInterfaceOrientations=['UIInterfaceOrientationPortrait', 'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight'])
(app / 'Info.plist').write_bytes(plistlib.dumps(info))
PY
  # Simulator signing is ad hoc. A device app requires the owner's signing identity.
  if [[ "$SDK" == iphonesimulator ]]; then codesign --force --sign - "$APP"; fi
done
xcodebuild -create-xcframework \
  -library "$ROOT/target/aarch64-apple-ios/release/libgen2_laya_mobile.a" -headers "$OUT/headers" \
  -library "$ROOT/target/aarch64-apple-ios-sim/release/libgen2_laya_mobile.a" -headers "$OUT/headers" \
  -output "$OUT/Gen2Laya.xcframework"
python3 tools/laya/inspect_ios.py --output "$OUT" --runtime "$ORT_IOS_XCFWK_PATH"
