# Paired clone and install-only VM owner

Status on 2026-10-17: actual install-only qualification passes at
`550905e469c95e1373f75f330f68d32acfebf2bc`; see the
[56-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-install-stopped/collection-qualification.json).
The actual
[pinned-image/no-boot stage](nemotron-macos-vm-restore.md) passes; normal
guest boot/provisioning and fresh-machine complete-app qualification remain open.
This is the next step toward those original gates, with unchanged quality,
human-word, desktop, platform and public-release requirements.

The [frozen protocol](../../tools/nemotron-diarization/macos-vm-install.protocol.json)
binds 19 current/committed tool sources, 75 external tool/selected SDK/prerequisite
descriptors and all 47 prior receipt members. Metadata-only preparation performs
no compiler, clone, helper, VM, installer or mount call. Both reviews and a commit
must precede actual invocation. Ten targeted controls and all 307 model-free tests
pass; tests do not assert runtime cancellation or installation. Both review axes
cleared before the actual committed invocation described below.

The controller verifies the full local Apple image and unchanged original
four-file seed. It admits 140GiB initial free space, maintains an 80GiB running
floor and physically allocates a 256MiB diagnostic reserve. First it builds the
new owner offline with SDK26.5, Swift language mode5, arm64/minimum26.0, two jobs
and warnings as errors. It applies a new local ad-hoc hardened signature and
verifies exact one-key virtualization entitlement, signature, architecture,
deployment and dependencies. No identity, certificate/profile, host credentials,
Keychain/security/settings change or host sudo is part of this work.

Before cloning, a Foundation-only offline control verifies single and multiple
independent underlying NSError branches, including a shared error reference.
Diagnostic writes must succeed synchronously before VM creation and before the
installer invocation; failure cannot start guest execution and wait for a timer
to notice. Full error graphs retain both standard underlying-error keys.

Under the new output's exclusive `owner.lock`, it enumerates exactly the paired
hardware model, machine ID, auxiliary storage and RAW disk. `fclonefileat` uses
verified source/destination descriptors, same-volume checks, a fresh basename and
no copy fallback. The clone must be an owned regular private single-link file
with distinct inode and exact logical length. Source identities remain unchanged;
all small metadata bytes are fully hashed. The initial cloned RAW100GiB disk
must still have zero allocated blocks. Clone provenance is an OS operation, not
a whole-disk SHA256 or measurement of physical shared allocation. The source
image and original seed remain untouched and are checked again after execution.

The owner reconstructs the exact supported platform/identity/auxiliary/disk and
persisted locally administered MAC, with 4CPU/8GiB, NAT, one framebuffer, virtual
keyboard/pointing, no audio/shares/window/custom listener. It validates the
configuration, creates one VM on the main serial queue, requires stopped state,
retains its weak delegate strongly, and invokes the installer once. **Installation
executes guest code**; the owner never calls a subsequent normal guest `start`.

Build/sign/inspection children deny networking and model/private reads. Before
VM creation, controls also check that the setup policy denies model reads and
write-opening the original seed. The installer policy explicitly permits setup
networking. A local IPSW does not establish an offline installer, and Apple-managed
networking/personalization and temporary files are not claimed to be completely
observed or confined by the owner sandbox. There is no endpoint allowlist or
silent network-policy retry. No private app/model enters the guest at this stage.

The owner enforces a 1,800-second operation deadline and 180-second cleanup
allowance. Before install starts, cancellation never cancels progress. While
installing, signals/deadline request `progress.cancel()` and await the installer
callback; VM pause/stop/start during installation is forbidden. Only after the
terminal callback may Running/Paused plus `canStop` lead to a separately logged
destructive stop. Its callback and actual stopped state are required. Transient
states get a bounded asynchronous wait, never a guessed transition.

The parent waits up to 2,010 seconds, covering operation, cleanup and a 30-second
allowance. On deadline/storage/diagnostic error it frees the reserve, requests
SIGTERM cancellation and allows another 180 seconds before kill/reap if needed.
The maximum declared parent owner wait is 2,190 seconds plus process-reap
scheduling. A killed/exited owner cannot prove framework teardown. A missing,
failed, cancelled or unconfirmed result quarantines the disk: no mount, clone,
reuse or automatic replacement VM follows. Lock release is not teardown proof.

Admission requires ordered finite JSONL events, exactly one successful installer
callback, observed stopped final state, no cancellation/error/log failure,
successful normal owner exit, unchanged inputs and the exact owned paired
installed files. Platform IDs stay byte-exact; auxiliary storage is fully hashed;
the installed RAW disk is stat-only with bounded logical/allocation geometry.
The new lock must be reused by every cooperating later phase. Apple errors10006/
10007 remain failures, with their underlying chains retained and no host-update,
credential or security-policy fallback.

Image, paired seed binaries/disk, executable and module cache stay local. No
complete SDK/OS/runtime closure is claimed. Even a passing result establishes
only an installed stopped seed; normal boot, disposable guest account/SSH,
provisioning, private app execution and every original release gate remain open.
The [primary-source lifecycle/clone audit](nemotron-local-vm-receiver.md) explains
the API contracts and the explicit uncertainty retained by this design.

The actual controller and independent saved-data collector were each observed
terminal with exit0. All 12 children return zero, including both sandbox controls,
the build/signature/architecture/deployment checks, Foundation error-graph
control and installer owner. All four `fclonefileat` calls return0 without copy
fallback; source identities remain unchanged and the cloned seed passes exact
content/ownership checks before installation. The owner records 65 ordered events.

The installer callback succeeds after 169.773970 seconds while state is Running.
Only then does the owner request the declared destructive stop; its completion
succeeds at 169.936335 seconds with observed Stopped state. The final uncancelled,
error-free report and successful owner exit close this stage's stopped-seed gate.
This is not a graceful guest-shutdown observation, normal boot or an operational
fresh-machine app result. No guest account, SSH, offline patch, host mount or
private app/model call occurs.

The installed RAW disk remains 107,374,182,400 logical bytes and reports
22,080,176,128 allocated bytes; this is stat/provenance evidence, not a full-disk
hash or exact physical CoW sharing measurement. The 132-byte model and 68-byte
machine identifier retain their original hashes. Auxiliary storage is fully
hashed at 33,579,164 bytes. All four files remain owned regular single-link600,
under directory700. Original no-boot seed and the full pinned image are unchanged;
collection independently rehashes the image without reading the sparse disks.

All 186 parent storage observations stay above the 80GiB running floor; minimum
free space is 138,644,246,528 bytes. The 256MiB physical reserve and exclusive
lock are released after the terminal owner wait. Collection verifies 46 prior
receipt members, 19 current/historical source pins and 75 external descriptors.
The image/seed binaries/disks, helper and module cache remain local. Collection
performs no new clone/helper/VM/network/model/mount call. Runtime cancellation
and killed-owner behavior were not exercised by this successful attempt;
their unconfirmed-cleanup limits remain explicit. Normal boot, restored Data
permissions/provisioning, authenticated SSH, complete app and all original
quality/human/desktop/platform/public-release gates remain open.
