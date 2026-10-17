# Paired clone and install-only VM owner

Status on 2026-10-17: prepared, not yet invoked. The actual
[pinned-image/no-boot stage](nemotron-macos-vm-restore.md) passes; installation,
guest boot/provisioning and fresh-machine complete-app qualification remain open.
This is the next step toward those original gates, with unchanged quality,
human-word, desktop, platform and public-release requirements.

The [frozen protocol](../../tools/nemotron-diarization/macos-vm-install.protocol.json)
binds 19 current/committed tool sources, 75 external tool/selected SDK/prerequisite
descriptors and all 47 prior receipt members. Metadata-only preparation performs
no compiler, clone, helper, VM, installer or mount call. Both reviews and a commit
must precede actual invocation. Ten targeted controls and all 307 model-free tests
pass; no runtime cancellation or successful installation is asserted by tests.

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
