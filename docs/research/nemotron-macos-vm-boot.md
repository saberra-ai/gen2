# Normal boot and stopped read-only disk discovery

Status on 2026-10-17: prepared, not executed. This follows the passing
[installation/stopped-seed receipt](nemotron-macos-vm-install.md). It advances the
fresh-machine prerequisite; it supplies no guest readiness, provisioning or app
result. Human annotation remains documented and deferred at the user's request.
All original quality, desktop, platform and public-release gates remain open.

The [protocol](../../tools/nemotron-diarization/macos-vm-boot.protocol.json) binds
24 current/committed sources and 84 external descriptors, including the prior
installation receipt, source seed identities and local Apple command manuals.
Preparation only reads bounded metadata and stats the disk; it performs no
clone, compiler, helper, VM, attachment or mount call. Two-axis review and a
commit precede actual invocation. Five targeted model-free checks cover the
full running interval, callback/stop/exit evidence and foreign/ambiguous layout
rejection. These tests do not establish actual boot or cancellation behavior.

The controller reacquires the installed source's exclusive lock and holds it
with a fresh output lock through clone, VM execution and disk cleanup. All four
paired files use enforced descriptor-anchored `fclonefileat`, with no copy
fallback. The original installed seed must retain its preregistered identities;
metadata is fully hashed, the 100GiB RAW disk is stat-only. It admits 120GiB
initial free space, maintains 80GiB while running and physically reserves 256MiB
for diagnostics. No original disk or private checkout is modified.

A new local owner builds offline with the pinned SDK, Swift mode5, arm64/min26,
two jobs and warnings as errors. Its new ad-hoc hardened signature has exactly
the unrestricted virtualization entitlement. No host sudo, certificate,
credentials, Keychain or security/settings changes are involved. Offline and
setup controls check denied model reads and original-seed write opening.
Only the VM owner explicitly permits NAT/setup networking. Apple-managed traffic
and temporary files are not claimed to be completely confined or observed.

One headless VM reconstructs the cloned hardware model, identifier, auxiliary
storage, disk and persisted MAC, with 4CPU/8GiB, one framebuffer, keyboard and
pointing device. It has no audio, shares, window or custom listener. Synchronous
diagnostic failures prevent VM creation/start. A successful normal-start callback
and 120 seconds of observed Running state qualify only the public lifecycle.
They do not prove macOS/Setup Assistant readiness, an account or a working app.

The owner has a 300-second operation deadline and 90-second cleanup allowance.
After the run interval or cancellation, it awaits any active start callback,
requests graceful stop when supported and allows 30 seconds before a declared
destructive stop in Running/Paused with `canStop`. Completion and actual stopped
state are mandatory. Without destructive stop, the guest-stop delegate is
required. Parent supervision allows 330 seconds, then requests SIGTERM with
90 seconds of cleanup before kill/reap failure. A missing, cancelled, failed or
unconfirmed owner quarantines the clone: no attachment, mount or replacement.
This successful path will not qualify cancellation or killed-owner cleanup.
The grace interval starts when a successful graceful request returns; a late
request receives the full allowance within the bounded cleanup deadline. An
offline compiled control checks this late-request case before any VM call.

Only after qualified stop and successful owner exit may `hdiutil` attach that
exact cloned RAW image with `CRawDiskImage`, `-readonly -nomount -nobrowse`.
Discovery selects the exact own-image GUID whole device and its observed
physical children, without assuming `s2`. Scoped `diskutil info`, `list` and
APFS queries must bind each container to its sole observed physical store;
every queried volume comes from that container's listing and has no mount point.

The local Apple manuals document `apfs listVolumeGroups -plist container` but
do not publish a stable dictionary schema. This stage retains exact scoped raw
group metadata for review; it does not invent keys, choose the first Data volume
or admit a System/Data pair. `system_data_group_verified` stays false, even if
both roles are observed. Filesystem mounting, root writes, ownership changes,
guest provisioning, SSH and app/model transfer belong to later work.

Exact own-image detach and verified absence run independently of source,
storage and diagnostic admission failures. The reserve is freed before cleanup
diagnostics, and both locks remain held until cleanup ends. The controller ignores
SIGTERM/SIGINT until its bounded detach sequence ends. Global image inventory
stays local; only exact own-image filtered rows may enter Git. Paired
seed binaries/disk, executable and module cache also stay local. No whole-disk
hash or complete OS/runtime closure is claimed. Unknown teardown remains a
failure, never a reason to infer safety from process exit alone.
