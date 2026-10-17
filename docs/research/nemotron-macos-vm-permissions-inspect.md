# Parent-relative stopped-guest permission inspection

Status on 2026-10-17: executed at committed source `af3055c82a9c6e988c39642a26097243a2390c16`;
inspection fails before users metadata, with partial parent metadata retained. The
[failed Data access receipt](nemotron-macos-vm-data.md) validates the pair and
mounts but stops at users-directory EACCES before writes. This diagnostic uses a
fresh paired clone of the original qualified stopped booted seed, not that failed
clone. It records mode/flags/ACL metadata before opening entries. It does not
repair permissions, establish write access or create a prepared guest.

The [protocol](../../tools/nemotron-diarization/macos-vm-permissions-inspect.protocol.json)
binds32 current/committed sources and102 external descriptors, including both
prior receipts and selected Xcode26.5 ABI headers/manual. Preparation is bounded
metadata/stat only; no clone, query control, attachment, mount or probe executes.
Two-axis review and a commit must precede invocation. Six targeted tests cover
canonical/null security references, distinct empty/NOACL layouts, truncation and
offset hazards, denied-open metadata retention, symlink confinement and partial
failure descriptor cleanup. Tests do not establish actual guest permissions.
All326 model-free tests pass; both review axes cleared the current source before
committed invocation. Actual metadata/ABI observations are recorded below.

## Actual result

The [96-member failure receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-permissions-failure/collection-qualification.json)
preserves all27 actual command results:26 return zero; only the permission probe
returns1. Four enforced clones preserve the original stopped booted seed. The
strict System/Data UUID pairing and selected DataRO mount pass, as does the
task-owned host ABI control. No VM or installer runs in this attempt.

Parent-relative stat and ordinary directory open succeed for
`private/var/db/dslocal/nodes/Default`. Its observed mode is `0600`, flags0,
inode17958, filesystem device16777269. Both child lookups (`users` and `groups`)
then fail at stat with EACCES13, before their own metadata/security query or open.
The parent has no search bit; this is evidence supporting a search-permission
hypothesis, not an actual repair result or proof that adding a bit suffices.
The mounted noowners UID502/GID20 does not establish stored guest ownership.

All11 accessible directory security queries and the host control return the
canonical12-byte `12/8/0` packed response, retained with raw bytes and SHA256.
The ABI shape is recognized, but ACL policy qualification stays false: this
response alone cannot distinguish absent from unsupported ACL metadata. The
users/groups ACLs remain unreached. No directory enumeration, account content,
permission change or provisioning occurs.

Normal unmount, no-MountPoint observation, exact image detach/absence, original
seed preservation and both exclusive-lock releases pass. All22 storage checks
stay above the80GiB running floor; minimum observed free space is131442155520B.
The physically allocated256MiB reserve is released. The failed clone remains
quarantined and must not be reused. Six raw global image inventories stay local;
only their hashes and scoped own-image records enter the receipt.

The saved-data collector rechecks32 current/copied/historical sources,102 external
descriptors,99 prior boot members and103 failed-access members without another
attachment, mount, clone or native query. The inherited pre-attach disk evidence
label refers to earlier installation; no inspected-seed success is claimed here.
All326 model-free tests pass. Inspection, Data access, stored ownership, prepared
guest, readiness, fresh-machine app and release flags remain false.

A future fresh-clone experiment may test a separately declared temporary owner
search-bit change on the observed parent, with exact identity checks and mode
restoration on every exit. It must first qualify that operation in the selected
runtime and inspect child metadata before considering any child change. This
receipt authorizes no mode/ACL/flag fallback and does not satisfy DataRW access.

## Frozen protocol

The controller keeps source/new exclusive locks through four enforced paired
`fclonefileat` operations, attachment/mount/probe and exact teardown. Source
identities and bounded metadata stay unchanged; the100GiB RAW disk is stat-only.
It admits120GiB initial/80GiB running free space and a physical256MiB reserve.
Offline/model/private-read and source/repository-write denials apply to all
children; actual negative controls run first. A fresh task-owned host directory
provides a bounded `getattrlistat` ABI control before any guest attachment.

Only the fresh RAW clone attaches read-only/unmounted/nobrowse. Actual whole,
physical-store, container, System/Data group and member UUIDs must agree through
the existing strict validator. Only selected Data mounts read-only/noowners at
a private path; System remains unmounted. Device/UUID/group/store/path/APFS,
OS readonly flags, nonwritable volume and distinct actual filesystem device are
checked. There is no VM, compiler/signing, network, SSH, host privilege/security
setting or ownership-database operation in this stage.

The probe anchors its root descriptor to that verified DataRO mount. It visits
only literal prefixes of six known provisioning-directory targets, with shared
ancestors inspected once. Before each ordinary no-follow directory open, it
stats the leaf relative to an already accessible parent, records full mode/type,
inode/device, links/flags/size/times and apparent UID/GID, rejects symlinks and
filesystem crossings, queries bounded security metadata and rechecks identity.
It never enumerates a directory or reads account records, host keys or files.
Denied/missing entries remain observations; descendants whose parent cannot open
are marked unreached. Unexpected failures retain partial rows and close descriptors.

The selected native API request is a24-byte attrlist containing only extended
security, `FSOPT_NOFOLLOW`, and a zeroed64KiB buffer with explicit return/errno.
All offsets/lengths, including zero-length references, are bounded before slicing.
Raw bytes/base64/digest and classification are retained. Recognized native
filesec layouts distinguish count0 from the NOACL sentinel; unknown or unsupported
responses never become an empty-ACL claim. All ACL policy-qualification flags
remain false. See the [primary-source ABI audit](nemotron-macos-acl-abi.md) and
[permission API limits](nemotron-macos-vm-data-permissions.md).

Diagnostic success requires reaching actual parent-relative users metadata and
a zero-exit probe with zero permission changes, plus validated mount, source
preservation and clean teardown. It does not require users to open or the guest
ACL query to succeed; those outcomes remain explicit. Even on diagnostic success,
`data_access_passed`, `prepared_guest`, stored ownership, provisioning/readiness,
fresh-machine app and release flags stay false. This cannot replace the failed
DataRW access requirement or qualify a future permission change.

Cleanup releases reserve space before diagnostics and defers controller signals
through bounded normal unmount/exact detach. Current own-image membership plus
UUID/store binding precedes unmount; an unmount/query failure cannot skip the
exact-image detach attempt. Actual mount/image absence and both lock releases
are required. Raw global image inventories and paired binaries/disks stay local;
only scoped metadata and global-log hashes enter Git. There is no force, silent
retry or failed-clone reuse. Any later narrowly declared guest-only change needs
separate admission and actual restoration/ownership verification.

The original quality/human-word, guest provisioning/authenticated SSH/fresh-app,
GUI/microphone/playback/IPC, platform performance and public-signing/release/main
gates remain open. Human timing annotation/review remains deferred at the user's
request; no reviewer or reference is invented here.
