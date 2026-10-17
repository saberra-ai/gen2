# Normal boot and stopped read-only disk discovery

Status on 2026-10-17: actual normal-start/stopped lifecycle and read-only discovery
pass at `8c76eb8ad9764f134ea74432d54df8a72a46bc91`; see the
[100-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-boot-stopped-readonly/collection-qualification.json).
This follows the passing [installation/stopped-seed receipt](nemotron-macos-vm-install.md).
It advances the fresh-machine prerequisite; it supplies no guest readiness,
provisioning or app result. Human annotation remains documented and deferred at the user's request.
All original quality, desktop, platform and public-release gates remain open.

The [protocol](../../tools/nemotron-diarization/macos-vm-boot.protocol.json) binds
24 current/committed sources and 84 external descriptors, including the prior
installation receipt, source seed identities and local Apple command manuals.
Preparation only reads bounded metadata and stats the disk; it performs no
clone, compiler, helper, VM, attachment or mount call. Two-axis review and a
commit precede actual invocation. Five targeted model-free checks cover the
full running interval, callback/stop/exit evidence and foreign/ambiguous layout
rejection. These tests do not establish actual boot or cancellation behavior.
Both review axes cleared before committed invocation. All 312 model-free tests
pass; the actual results below qualify only their observed successful path.
The [feature run](https://github.com/saberra-ai/gen2/actions/runs/39019003838)
at `8c76eb8a` now completes successfully in all 13 jobs.

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

The actual controller and independent saved-data collector each return terminal
exit0. All 32 commands succeed, including the offline build/signature controls,
Foundation error graph, five shutdown-timing checks, VM owner and scoped disk
queries/cleanup. All four enforced clone calls succeed without fallback and
preserve the original installed seed's frozen identities and bounded metadata.

Normal start succeeds at 0.222143 seconds. The owner observes Running for
120.799614 seconds and requests graceful shutdown. No guest-stop delegate arrives;
after 30.999066 seconds from successful request return, the declared destructive
fallback runs. Its completion succeeds at 152.189704 seconds in Stopped state.
The uncancelled, error-free 24-event terminal report and normal owner exit qualify
the boot lifecycle. **This is not a successful graceful guest shutdown.**

Only then does the exact clone attach read-only with no mount. The observed
whole device is `/dev/disk12`, with physical children `s1`, `s2` and `s3`. Actual
metadata binds `disk12s2` to sole-store APFS container `disk15`; its six observed
volumes carry System, Preboot, Recovery, Update, Data and VM roles. These names
are this attempt's observations, never identifiers for later selection. Every
queried whole/partition/volume reports no mount point. Exact image detach succeeds,
the final inventory has no matching image and both locks release afterward.

The retained raw volume-group plist now exposes one container's `VolumeGroups`
array, with `APFSVolumeGroupUUID` and each member's `DeviceIdentifier`, `DiskUUID`
and singular `Role`. This is actual schema evidence for a later strict validator;
this stage deliberately leaves `system_data_group_verified` false. No filesystem
mount, account/SSH/ownership change, guest patch or private app/model call occurs.

The booted disk is still 107,374,182,400 logical bytes, with 22,359,080,960
reported allocated bytes; it is never read or hashed as a whole. The platform
model/identifier hashes remain exact, and auxiliary storage is fully hashed.
The raw seed helper's inherited `content_evidence` text describes the earlier
installation prerequisite. The collection summary explicitly corrects its scope:
this attempt clones an already-installed source and performs normal boot/stop;
it does not invoke an installer after that clone. Raw records remain unchanged.
All 185 parent storage checks pass, with minimum free 136,417,222,656 bytes;
the physical 256MiB reserve is released. Collection independently checks all
55 prior receipt members, 24 current/historical sources, 84 external descriptors,
raw-to-filtered image inventories and raw-to-recorded scoped metadata. The four
global inventory logs remain local with only their hashes and exact own-image
records retained publicly. Collection makes no new VM/helper/device/network/model
call. Guest readiness/provisioning, fresh-machine complete app and every original
quality/human/GUI/microphone/platform/public-release gate remain open.
