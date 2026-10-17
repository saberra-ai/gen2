# Strict volume group and stopped-guest Data access

Status on 2026-10-17: prepared, not executed. The passing
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

Even a passing result would establish strict group validation and stopped-volume
access only. Guest provisioning/readiness/SSH, complete fresh-machine Pnola,
GUI/microphone/playback/IPC, quality/human timing, other platform performance and
public signing/release/main promotion remain open.
