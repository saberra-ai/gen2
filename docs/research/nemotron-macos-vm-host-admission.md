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

The separate [v2 protocol](../../tools/nemotron-diarization/macos-vm-host-v2.protocol.json)
binds that exact failed receipt and the read-only XML control, with ten source
and 24 external descriptors. The new controller differs only in its committed
protocol filename and explicit `--xml` display request. The original controller,
protocol, failed output and all evidence remain unchanged. Use a fresh v2 output;
all signing, host-support/geometry, blank-image identity/mount/write/cleanup,
offline, storage and no-guest gates remain unchanged. Review and commit this
retry before invoking it; no v2 controller or helper has run at preregistration.
The full 286-test model-free suite also passes for this retry in 10.117 seconds.

The [actual v2 attempt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-partition-failure/collection-qualification.json)
at `0d585141b56546464028f7f4c79f85fc9cdf31f1` now proves helper execution,
host virtualization support and valid four-CPU/eight-GiB proposed geometry.
Measured host bounds are 1–64 CPUs and 4 MiB–64 GiB memory. Its exact one-key
XML entitlement, strict ad-hoc hardened signature, arm64 and minimum26.0 verify.

The attempt remains failed: the fresh 256-MiB image creates and attaches, but
its actual unique physical partition is `s1`, while the checker queries fixed
`s2`. That read-only query returns not found. No APFS-list, mount or write starts.
Cleanup re-identifies the exact task image, detaches it successfully and confirms
absence; the reserve is released. The 56-member receipt retains all raw command
and failure evidence, source copies and filtered owned-image mappings. The
detached image/helper/global inventories stay local by hash. This proves neither
blank-volume write permission nor VM creation/boot or fresh-machine operation.

The separate [v3 protocol](../../tools/nemotron-diarization/macos-vm-host-v3.protocol.json)
binds the actual v2 failure and exact owned-image mapping, with 13 source and
30 external descriptors. The new controller derives the sole physical child
partition from that current image observation, then requires the APFS container's
sole store to equal it. Synthetic container/volume devices are excluded from
physical-child selection. No partition index is assumed. Multiple physical
children, wrong images, foreign stores and ambiguous volumes reject before mount.

Two new regressions cover the actual single-`s1` layout, a single-`s2` layout and
the unsafe mappings above; the unchanged earlier cleanup/host checks remain.
All 288 model-free tests pass in 11.670 seconds. Original v1/v2 files and failed
receipts remain unchanged. All other signing, no-guest, offline, storage,
cleanup and deadline gates remain unchanged. Review and commit before actual
v3 invocation; no v3 helper or image operation has run at preregistration.

The [actual v3 attempt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-mounted-key-failure/collection-qualification.json)
at `a8700acec1a714adbb24c56b23ebfd41aeb2c174` passes all 22 children, including
the observed physical-store/container/volume bridge and explicit mount. The
reported mountpoint, device and APFS filesystem agree. It remains failed:
`diskutil info -plist` omits the assumed `Mounted` key, so validation raises
before `Path.is_mount()` or the dummy write. Cleanup detaches the exact image,
confirms absence and releases the reserve. The 67-member receipt preserves all
evidence and current/historical sources. Reported mount success is not a passed
filesystem stat/write, root-owned Data-volume or VM/fresh-machine gate.

The separate [v4 protocol](../../tools/nemotron-diarization/macos-vm-host-v4.protocol.json)
binds that failed attempt and its actual mounted-info bytes, with 16 source and
36 external descriptors. Before writing, the new controller requires actual
`Path.is_mount()` and distinct mount/parent `st_dev`, plus exact reported device,
mountpoint, filesystem, container and sole physical store. A writable volume is
required. An absent undocumented `Mounted` key is not used as a mount signal; a
present contradictory value still rejects. Record the actual
`GlobalPermissionsEnabled` observation without changing it.

The new regression accepts the observed plist shape only with the OS mount
evidence, and rejects false/same-device mounts, wrong paths/devices/stores/
containers, read-only volume and contradictory metadata. All 289 model-free tests
pass in 11.213 seconds. Earlier source/protocol/failure records remain unchanged.
All no-guest, signing, offline, own-image, storage, cleanup and deadline criteria
remain required. Review and commit before actual v4 invocation; no v4 helper or
image operation has run at preregistration.

The [actual v4 prerequisite receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-prerequisites/collection-qualification.json)
now passes at `5a9dcbd97bfc29ffe4e6b2113d3b13b24cb6f2be`. The controller was
observed terminal with exit 0 before independent saved-data collection. All
22 children return zero, including offline controls, helper build/signature/
execution and exact-image create/attach/mount/detach. Actual OS mount evidence
has distinct device IDs, and the exact APFS device/container/observed-store
metadata agrees. The 52-byte dummy file passes fsync/read/mode600 checks; its
payload also remains independently observable in the detached image bytes.
`GlobalPermissionsEnabled=false` is recorded without an ownership override.

The exact image and task mount are absent after detach; the physical reserve is
released and all storage floors pass. The 72-member receipt binds 16 current/
historical source files, 36 external records, actual commands/metadata, filtered
owned-image observations, OS-stat/write results, collectors/preparation and the
passing 289-test suite. The executable, 256-MiB blank image and unrelated global
disk inventories remain local with recorded hashes; module caches also stay
local. Collection executes no
helper/compiler/signing/image/VM/model operation. All earlier attempts remain
failed with their original evidence.

This establishes the local helper and blank-volume prerequisites only. It does
not establish permission to patch a restored root-owned Data volume, support for
the pinned restore image, actual VM creation/boot, guest provisioning/SSH or
fresh-guest complete-app operation. No OS restore body has been acquired and no
guest or private app has run in this stage. Those checks, real GUI/microphone,
quality/deferred human review, other platforms and public release remain open.
