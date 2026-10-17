# Pinned restore image and no-boot VM seed

Status on 2026-10-17: actual no-boot qualification passes at
`4c9eb3612defd9c1ab6a8dfc982d489d23bb084b`; see the
[47-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-restore-no-boot/collection-qualification.json).
The actual
[host/API and blank-APFS prerequisites](nemotron-macos-vm-host-admission.md)
pass. This next stage acquires the exact public Apple restore image and validates
a fresh configuration before any VM instance, installer or guest operation.

The frozen [protocol](../../tools/nemotron-diarization/macos-vm-restore.protocol.json)
binds 12 current/committed tool sources, 62 external compiler/tool/selected SDK
header/metadata descriptors and the complete passing host receipt. It uses the
named physical M4 Pro. This is selected-input provenance, not a full SDK closure.
Both review axes cleared, and the source/protocol was committed before
actual invocation. All 297 model-free tests pass; eight targeted tests cover
download integrity/deadlines and incompatible or altered seeds.

The frozen protocol order is:

1. Admit 140GiB free space and physically allocate a 256MiB diagnostic reserve.
   Build the new helper offline with SDK26.5, Swift language mode5, two compiler
   jobs, warnings as errors and arm64/macOS26.0 target. Apply a new local ad-hoc
   hardened signature and verify exactly the one unrestricted virtualization
   entitlement, architecture, deployment and dependencies before downloading.
2. Acquire `UniversalMac_26.3_25D125_Restore.ipsw` from its exact frozen public
   Apple CDN URL, 18,877,622,182 bytes, SHA256
   `7d22296ca6ab46f255bed8f5fc80845edab14fb781f845d7cae9427a22970b53`.
   The acquisition worker has an isolated environment with no proxy/cookies,
   rejects redirects, compression, wrong size/hash, existing targets and retries,
   retains partial failures, and publishes with a no-clobber hard link only after
   full integrity verification. Its checked deadline is 7,200 seconds, individual
   reads at most 30 seconds; the parent kills/reaps the worker after 7,230 seconds
   even if DNS/TLS stalls. Both parent and worker check the 120GiB running floor.
3. Run the helper offline to load the local image, require actual support,
   OS26.3.0/build25D125 and image minima within fixed 4CPU/8GiB. Create a fresh
   private seed with model, machine identifier, matching auxiliary storage and a
   sparse RAW100GiB disk; call only `configuration.validate()`. Persist a random
   locally administered unicast MAC with NAT configuration, one framebuffer and
   keyboard/pointing devices, zero audio/shared directories/window/custom listener.
4. Independently verify all four owned regular files, modes600/single links and
   directory700. Hash only the bounded metadata files. The disk must have exactly
   107,374,182,400 logical bytes and zero allocated blocks after exclusive creation
   and `ftruncate`; do not read/hash the entire sparse disk. Release the reserve
   and retain terminal status, raw logs, image metadata, seed descriptors and
   immutable input checks. Keep the OS body, seed binaries/disk, executable and
   module cache local; copy no proprietary OS body into the repository.

All helper/build/inspection children deny networking and model/private reads;
the exact public Apple acquisition worker is the sole network exception. A
configured NAT device is not a booted network. No host sudo, credentials,
certificate/profile/Keychain/security-policy/settings changes, private source
edits or authentication flow are part of this stage.

The seed remains immutable for a later separately reviewed clone/install phase.
The [lifecycle audit](nemotron-local-vm-receiver.md) requires installer progress
cancellation after start and a terminal callback, and forbids VM pause/stop during
installation. Owner termination alone cannot prove bounded guest teardown.
This stage cannot qualify installation, boot, provisioning/SSH, a fresh-machine
app, real GUI/microphone/playback/IPC, accuracy/human gold, other platforms or
public signing/release. Those original gates remain open.

The actual controller was observed terminal with exit0 before saved-data-only
collection, which also exited0. All 12 commands return zero. Acquisition takes
800.403 seconds; the full 18,877,622,182-byte body matches the pinned SHA256,
independently recomputed during collection. Actual image requirements are 2CPU
and 4GiB; the selected 4CPU/8GiB configuration validates with the supported model.
The exact four-file seed is owned, directory700/files600/single-link. Its raw disk
has 100GiB logical capacity and zero allocated blocks; the three small metadata
files are fully hashed. The auxiliary storage is 33,570,816 logical bytes with
16,384 allocated bytes. No whole sparse-disk hash is claimed.

All 813 parent storage observations stay above the 120GiB running floor; minimum
free space is 162,454,814,720 bytes. The physically allocated 256MiB reserve is
released. Collection verifies all 71 prior host receipt members, 12 current and
historical tool sources, 62 external descriptors, helper/signature/entitlement
metadata, image/seed results and raw logs. It performs no helper/compiler/network/
image-mount/VM/model call. The OS body, all seed binary files/disk, helper and
module cache stay local. This closes image acquisition/support and no-boot
configuration validation only; installation/boot/provisioning and every original
fresh-machine/app/quality/release gate remain open.

Feature CI for the preregistered source also passes
[all 13 jobs](https://github.com/saberra-ai/gen2/actions/runs/39015483440).
Those jobs cover feature compatibility and model-free tools, not guest execution.
