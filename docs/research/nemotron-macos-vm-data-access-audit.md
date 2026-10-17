# Stopped macOS clone: offline Data access audit

2026-10-17; research only. The [normal-boot/read-only discovery receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-boot-stopped-readonly/qualification.json), source `8c76eb8ad9764f134ea74432d54df8a72a46bc91`, passes boot/discovery and detach but explicitly leaves `system_data_group_verified`, `guest_os_ready_qualified` and `fresh_machine_qualified` false. No helper, VM, attachment, mount, SSH, guest/private payload or host setting was operated in this investigation.

## Proposed mount contract

Use a new enforced paired-state clone; retain the seed and prior receipt. Under the exclusive lock, revalidate the actual System/Data UUID group and physical-store bridge before any mount. These are documented command forms, **not executed commands**:

```text
/usr/bin/hdiutil attach -imagekey diskimage-class=CRawDiskImage -readwrite -nomount -nobrowse -plist CLONE_IMAGE
/usr/sbin/diskutil mount nobrowse -mountOptions noowners -mountPoint DATA_MOUNT OBSERVED_DATA_DEVICE
/usr/sbin/diskutil mount readOnly nobrowse -mountOptions noowners -mountPoint SYSTEM_MOUNT OBSERVED_SYSTEM_DEVICE
```

Mountpoint directories must already exist. There is no documented `rw` positional token: omit `readOnly` for Data, then verify its actual RW state. Re-query exact device/UUID/container/mountpoint identity, distinct mounted `st_dev`, filesystem type and OS read-only flags; independently require System RO. `nobrowse` is a visibility recommendation, not access control. Mount may invoke a minimal filesystem check, so this phase can change **disposable clone metadata** before any provisioning. Command success alone is insufficient. [Local diskutil manual](/usr/share/man/man8/diskutil.8), [raw-image/attachment semantics](/usr/share/man/man1/hdiutil.1).

`noowners` affects apparent ownership; it neither removes mode/ACL restrictions nor proves persisted guest UID/GID correctness. New disk-image files can acquire the attaching user's ownership. Do not use host ownership-database commands, host sudo, authentication dialogs or security changes. A permission/TCC failure is retained failure. Guest-side ownership repair/verification remains required. [Ownership semantics](/usr/share/man/man1/hdiutil.1), [mount options](/usr/share/man/man8/mount.8).

## Meaningful bounded probe

An access-policy proposal, not proof of actual path existence: inspect only the following existing Data-relative directories, all directly used by [pinned Lume patcher](https://github.com/trycua/cua/blob/3350f504014c85416009a90fcb339fe556ce02fc/libs/lume/src/Unattended/MacOSOfflineSetupPatcher.swift). Record missing/denied paths rather than silently creating them or changing permissions.

| Data-relative directory | Provisioning dependency |
| --- | --- |
| `private/var/db/dslocal/nodes/Default/users` | Local account record |
| `private/var/db/dslocal/nodes/Default/groups` | Admin/staff membership records |
| `private/var/db` | Setup marker |
| `private/var/db/com.apple.xpc.launchd` | SSH disabled-state plist/marker |
| `private/etc` | Guest autologin file; SSH directory metadata |
| `Library/Preferences` | Setup Assistant, loginwindow and power plists |
| `Users` | Future account home |

For each existing directory, use descriptor-relative, no-follow traversal bound to the verified Data mount. A uniquely named non-plist probe created with `O_EXCL`, mode0600 and a fixed tiny payload can establish create/write/fsync/readback/unlink; verify removal and close descriptors. A Data-root-only probe establishes much less. Do not enumerate account contents, edit existing plists, create an account/home, or chmod existing directories in this phase. This tests directory access, not eventual in-place plist-update success or correct guest ownership.

Use Data-relative `private/...`, not `/etc` or `/var` symlinks that may resolve outside the mount. Reject symlink escapes and filesystem crossings. Do not assume a privately mounted System volume recreates the booted guest's merged/firmlink namespace: Apple distinguishes System content from writable `/Library`, `/Users` and `/private`, and says normal boot uses a System snapshot. [Apple APFS layout](https://support.apple.com/guide/security/role-of-apple-file-system-seca6147599e/web).

Read bounded `SYSTEM_MOUNT/System/Library/CoreServices/SystemVersion.plist`, the path documented by [sw_vers(1)](/usr/share/man/man1/sw_vers.1); never substitute the host's version. Treat `System/Library/CoreServices/Setup Assistant.app/Contents/Resources/SetupAssistantInfo.plist` as an **unverified candidate**, not a published Apple path/schema contract. Record absent, renamed or unreadable metadata honestly; do not derive Setup Assistant settings from guessed fields. Lume hardcodes `LastSeenBuddyBuildVersion=25F84` and `LastSeenCloudProductVersion=26.5.2`; neither is validated for installed26.3/25D125 by its source. This audit does not prescribe replacement values.

At `DATA_MOUNT/private/etc/ssh`, bound existence/type checks to expected `ssh_host_*_key.pub` names; only public-key contents may be retained. Never read private-key files. Absence is expected to remain possible before SSH activation and does not justify trusting `ssh-keyscan` later. [Pinned orchestration](https://github.com/trycua/cua/blob/3350f504014c85416009a90fcb339fe556ce02fc/libs/lume/src/Unattended/UnattendedInstaller.swift).

## Cleanup and scope

Close files, remove only exact recorded probes, unmount each exact owned volume normally, then detach its exact image. Bound every command and retain failures; manual unmount can take up to a minute. Forced cleanup, if separately declared, is a recorded failure disposition, not clean-unmount success. Verify mount and attachment absence before reuse; unlike [Lume's best-effort detach](https://github.com/trycua/cua/blob/3350f504014c85416009a90fcb339fe556ce02fc/libs/lume/src/Utils/DiskImageSession.swift), never clear ownership state merely because a detach was attempted. Unprivileged restored-volume mounting/writing remains unproved; passing this future probe would still not establish provisioning, authenticated SSH, app execution or fresh-machine/release qualification.

Manual SHA256s: `diskutil.8` `49cb89da31420099bfb871e889cc72bde69606ca3a8abac1b4a4f2a87f84889e`; `hdiutil.1` `ad8a8a5b2bb3b35b68adbdabb26a46fc7bd52bbdcb5a20054603968f098b8a85`.
