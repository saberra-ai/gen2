# Strict volume group and stopped-guest Data access

Status on 2026-10-17: actual strict pairing and mounts pass, but bounded Data
access **fails** at `ec80cca155b7d214c42922f0fc88d3182f9658ed`; see the
[104-member failed-access receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-data-failure/collection-qualification.json).
The passing
[boot/stopped/read-only discovery](nemotron-macos-vm-boot.md) provides actual raw
volume-group schema. This next stage validates the System/Data pair and tests
bounded access on another disposable paired clone. It supplies no account,
authenticated SSH, guest readiness, private app/model or release result. Human
timing review remains documented and deferred; all original release gates remain.

The [protocol](../../tools/nemotron-diarization/macos-vm-data.protocol.json) freezes
29 current/committed sources and 93 external descriptors, including the prior
100-member receipt and actual stopped booted-seed identities. Preparation only
reads bounded metadata and stats the sparse disk. Two-axis review and a commit
must precede actual invocation. Eight targeted tests exercise actual saved Apple
plists, ambiguous/foreign/UUID/mount rejection, descriptor escape rejection,
exclusive probe writes/removal and bounded regular-file reading. They do not
establish actual guest-volume mounting or ownership.
All 320 model-free tests pass. Both review axes cleared the current protocol
before committed invocation; this is source review, not actual access evidence.

The controller reacquires the source's exclusive lock and holds it with a new
output lock across four enforced `fclonefileat` calls, attachment, mounts, probe
and teardown. It never uses a copy fallback or reads/hashes the 100GiB RAW disk.
All four paired files stay owned regular single-link600 under directory700.
Original source identities and bounded metadata hashes must remain unchanged.
It requires 120GiB initial/80GiB running free space and a physically allocated
256MiB diagnostic reserve. Children run with offline/model/private-read denials
and original-seed/repository write denials; actual negative controls run first.

Only the fresh clone attaches as explicit RAW/readwrite/nomount/nobrowse. Actual
own-image GUID whole and physical children determine scoped container queries;
there is no fixed disk number, `s2`, name or first-Data selection. Before mounting,
one image-wide unambiguous group must contain exactly one System and one Data
member. Container UUID, group UUID, per-member group `DiskUUID`, listing
`APFSVolumeUUID`, device-info `DiskUUID`/`VolumeUUID`/`APFSVolumeGroupID`, exact
roles and sole physical-store membership must agree. Locked/encrypted members,
duplicate or extra groups and existing mounts fail admission.

Private existing mountpoint directories receive DataRW and SystemRO through
`diskutil mount [readOnly] nobrowse -mountOptions noowners -mountPoint ...`.
Actual device/UUID/group/store/path, APFS filesystem, distinct mounted `st_dev`,
OS read-only flags and `WritableVolume` must match. Global permissions remain
false. Mounting may run minimal fsck or write metadata to the disposable clone;
this phase is not read-only. Host sudo, ownership-database commands, credentials,
certificate/Keychain/security changes and VM/network/listener operations are
excluded. See the [primary-source access audit](nemotron-macos-vm-data-access-audit.md).

The probe anchors no-follow descriptors to validated mounts, rejecting symlink
or filesystem escapes. A bounded System `SystemVersion.plist` supplies the actual
guest version/build, matched to the pinned restore's numeric components/build;
two-component versions acquire patch0 for comparison only. It never substitutes
the host version, guesses Setup Assistant metadata or copies Lume's 26.5.2 values.

It observes seven existing Data directories used by later provisioning: setup
marker/db, dslocal users/groups, launchd, preferences, homes and private/etc.
Within each it exclusively creates one task-named non-plist probe, writes a tiny
fixed payload, fsyncs, reads back, sets600, removes it and fsyncs the directory.
The parent independently checks absence. Existing root/admin record files are
opened for writing without truncation or writes, with unchanged metadata; their
contents are never read. No existing directory is created or chmodded, and no
account/home/plist/Setup Assistant/sudoers/SSH setting is provisioned. A failed
or killed probe may leave its task file on the quarantined disposable clone.

Only the expected bounded public Ed25519 host-key file may be read if present;
its wire format is checked and private keys are never opened. Absence is an
observation, not permission to trust a later scanned key. Apparent noowners
UID/GID is not proof of stored guest ownership. That flag stays false, with
future authenticated guest-side repair/verification required before payload use.

Cleanup frees the reserve first and defers SIGTERM/SIGINT until bounded cleanup
ends. Exact own-image entities plus UUID/store checks bind selected volumes
before normal unmount; actual absence is checked afterward. A query/unmount
failure cannot skip the exact current own-image detach attempt. Final attachment
and mount absence, source preservation and both lock releases are mandatory.
There is no force, silent retry or automatic reuse after failure. A failed access
or cleanup attempt quarantines the clone. Raw global image inventories and seed
binaries/disk stay local; only scoped records and global-log hashes enter Git.

The actual controller was observed terminal with exit1; saved-data collection
returns exit0 while preserving `data_access_passed=false` and
`clone_quarantined=true`. Of 32 children, 31 return zero and only `data-probe`
fails. All four enforced clones preserve the original stopped booted seed's
identities and bounded metadata. The actual UUID bridge qualifies the unique
System/Data pair in sole-store container `disk15` on observed `disk12s2`.

Both private mounts succeed: Data is writable with filesystem device16777270,
System is read-only with device16777267, and their host parent device is16777233.
Exact device/UUID/group/store/path, OS access flags and APFS metadata agree.
Global permissions remain false; no ownership-database operation occurs. These
are this attempt's observations, never future device-selection constants.

The probe saves the actual 600-byte SystemVersion plist, SHA256
`aaecfe6f3b00111a7d23b868013798eba4d09c886fe1b3324388f8290d32f3d1`.
It identifies guest26.3/build25D125, matching the restored numeric components.
Traversal then fails with `PermissionError(13, 'Permission denied')` opening
`users` during metadata inspection of
`private/var/db/dslocal/nodes/Default/users`. Only the preceding db directory's
metadata is observed. **No probe file, existing account write-open or account
content read occurs.** No existing guest file or directory mode is changed.
The public-host-key step is never reached: the partial report's initial
`present=false` field does not establish actual key absence.

Both normal unmounts and their no-MountPoint checks succeed, followed by exact
image detach and verified absence. Source preservation, both lock releases and
reserve release pass. All 24 parent storage observations stay above the running
floor; minimum free is131,509,043,200bytes. The failed clone remains quarantined
and is not mounted/read/reused by collection. Collection replays all99 prior
receipt members, 29 current/copied/historical sources and93 external descriptors,
plus raw group/mount/unmount/filtered image records. All seven global image
inventory logs stay local with hashes only. No VM, network, provisioning,
credentials, private app/model or host sudo/security/settings call occurs.

Next, use another fresh paired clone to localize the failed entry's mode/ACL and
access policy through an already accessible parent descriptor. The current
failure does not establish that the missing permission is an execute bit. The
pinned patcher explicitly adds owner execute (`current | 0o100`) to Default/users
directories before provisioning; that is source evidence for a separately
declared guest-only bootstrap, not authorization to infer its success here.
Inspect first, then freeze any exact bounded mode change and restoration before
another access attempt. Do not remount this failed clone, escalate on the host,
or change ownership settings to force a pass. Stored guest UID/GID, provisioning,
authenticated SSH and complete-app admission remain unproved.
The [permission diagnostic API audit](nemotron-macos-vm-data-permissions.md)
identifies parent-relative metadata inspection, public API limits and a narrowly
conditional future guest-only change. It performs no new experiment or repair.

Even a passing complete access result would establish strict group validation and stopped-volume
access only. Guest provisioning/readiness/SSH, complete fresh-machine Pnola,
GUI/microphone/playback/IPC, quality/human timing, other platform performance and
public signing/release/main promotion remain open.
