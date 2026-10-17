# Local VM host prerequisites: no guest or OS image

The [source-only VM research](nemotron-local-vm-receiver.md) supports a small
local Apple Virtualization.framework runner without VNC or a restricted
networking entitlement. Its next prerequisite is actual local artifact execution
and unprivileged disk-image access. This stage does not create or boot a VM and
cannot qualify a fresh-machine Pnola run.

The [frozen protocol](../../tools/nemotron-diarization/macos-vm-host.protocol.json)
binds nine tool/source/notice files and 16 external compiler, utility, selected
SDK/header and primary-source records. Require current descriptors before local
imports, exact committed Git blobs and protocol, then recheck all descriptors
before each admitted child. Cleanup remains available if an input changed. The compiler is the actual local Swift 6.3.3 with Apple SDK
26.5; explicitly target arm64/macOS26.0 and use VZ APIs available since macOS11.
This records selected tool/SDK inputs, not complete SDK-source reproducibility.
Apple's sample notice is retained with the adapted API code.

After review and commit, the controller will:

1. Verify actual offline/weight-format sandbox controls. Use task-private HOME,
   TMPDIR and module cache with no inherited Apple credentials. Deny reads/writes
   to existing cache directories, private apps, SSH and user Keychain, and writes
   to the checkout.
2. Compile only the small synchronous host probe and ad-hoc sign that new local
   artifact with runtime hardening and exactly `com.apple.security.virtualization`.
   Inspect its actual strict signature, entitlement, arm64 architecture and
   minimum macOS26.0. No host identity, profile, Keychain or setting is changed.
3. Execute only `VZVirtualMachine.isSupported` and configuration limit queries.
   Check proposed four-CPU/eight-GiB geometry against measured host bounds. No
   VM instance, restore loader, network attachment, audio, UI or listener is
   created. A supported result and accepted executable do not prove image
   compatibility or entitlement-mediated VM creation/boot.
4. Create one fresh 256 MiB GPT/APFS disk image with no auto-attach or Spotlight.
   Attach without mounting; select its whole device only from the exact image
   path's GUID entity. Bridge its partition to the APFS container and require a
   single physical store and single named volume before explicitly mounting
   under the fresh task output. Never guess a disk number or select the first
   disk/volume with a matching label.
5. On the verified actual task mount, create one dummy file, fsync, read it and
   check mode600. Record the observed ownership policy without overriding it.
   Re-query the exact image before detach and require its attachment and mount
   to be absent afterward. Failures remain failures; there is no privilege,
   password, GUI or silent-retry fallback.

Every child has a 300-second process-group deadline. Preserve five-GiB start,
one-GiB running and 256-MiB physical-reserve floors. Cleanup bypasses storage
floors and frozen-input admission so low space or source drift cannot prevent
detach, while retaining its deadline. Capture cleanup output in memory before
best-effort diagnostic writes; a diagnostic failure prevents a passing gate,
but cannot gate exact-image detach. The physical reserve is freed before cleanup.
Cover spawn, PID persistence and wait with one kill/reap handler. Global
disk-service inventory stays local; only the exact task-image
selection and raw-log descriptors should be retained in public evidence. No
other image is mounted or detached. No inference runs alongside compilation.

Six model-free tests reject wrong/ambiguous whole devices, foreign/shared APFS
physical stores and volumes, unsupported/spoofed host geometry and accidental
guest-execution claims, and verify kill/reap after PID-write failure, cleanup
capture despite diagnostic failure and group termination on cleanup timeout.
The complete model-free tool suite passes 286 tests in
11.028 seconds. At preregistration the controller has not been invoked: no helper
has been built/signed/executed and no image has been created or mounted by it.

Success would establish only these host prerequisites. Access to this blank
user-created volume does not establish permission to patch a restored root-owned
macOS Data volume. Restore-image acquisition/hash/support, real VM installation,
headless boot, guest provisioning/SSH, cleanup and fresh-guest complete-app
offline checks remain required. GUI/microphone, other platforms/performance,
public signing, original 20% DER and deferred human-word review remain open.

The [actual first attempt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-entitlements-format-failure/collection-qualification.json)
at `7a5f3da930fd6d1e516d035bd7b7820c6b72e7b3` remains failed. Offline controls,
Swift compilation, ad-hoc signing and strict signature inspection return zero,
but default `codesign --display --entitlements -` emits an abstract dictionary,
not an XML plist. Parsing stops before helper execution or any image operation.
All eight actual children return zero; both exact-image cleanup inventories
confirm absence, no mount remains, and the reserve is released.

The 38-member receipt retains the original protocol, all source copies, command
and failure logs, filtered exact-image inventories, binary descriptor and passing
six/286-test logs. Unrelated global disk inventory remains local by hash. A
separate read-only `codesign --display --xml --entitlements -` control on the
same artifact extracts exactly the one declared virtualization entitlement; no
helper executes or changes. The actual local codesign manual explicitly defines
abstract output as the default and `--xml` as its XML-format selector. This
diagnoses the format mismatch without qualifying host support, a mount or a VM.
