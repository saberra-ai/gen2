"""Freeze metadata only for strict group selection and stopped-clone Data access."""
from pathlib import Path
import hashlib
import json
import os
import sys
from uuid import uuid4

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(tools))
from macos_vm_seed_clone import file_identity
from macos_vm_install_metrics import installed_seed_metrics


def descriptor(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


prior = json.loads((tools / 'macos-vm-boot.protocol.json').read_text())
for name, want in prior['tools'].items():
    assert descriptor(tools / name) == want
for name, want in prior['external_files'].items():
    assert descriptor(name) == want
receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-boot-stopped-readonly'
manifest = json.loads((receipt / 'collection.json').read_text())
for name, want in manifest['files'].items():
    assert descriptor(receipt / name) == want
assert json.loads((receipt / 'collection-qualification.json').read_text())['boot_discovery_passed'] is True
origin = cache / 'macos-vm-boot-local-v1'
assert (origin / 'qualification.json').read_bytes() == (receipt / 'qualification.json').read_bytes()
pq = json.loads((origin / 'qualification.json').read_text())
assert pq['boot_discovery_passed'] is True and pq['boot_owner_qualified'] is True
assert pq['owner_result']['cleanup_confirmed'] is True and pq['owner_result']['final_state'] == 0
assert pq['installed_disk_quarantined'] is False and pq['owned_image_detached'] is True
assert pq['exclusive_lock_released'] is True and pq['source_exclusive_lock_released'] is True
assert installed_seed_metrics(origin / 'vm-seed', pq['booted_seed'], uid=os.getuid()) == pq['booted_seed']
seed_identity = {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}
restore = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-restore-no-boot/qualification.json'
image = json.loads(restore.read_text())['image']
external = dict(prior['external_files'])
records = [Path(__file__), receipt / 'collection.json', receipt / 'collection-qualification.json', receipt / 'qualification.json',
    origin / 'qualification.json', origin / 'protocol.json', origin / 'owner.lock', restore,
    Path('/usr/share/man/man8/mount.8'), Path('/usr/share/man/man1/sw_vers.1')]
external.update({str(path): descriptor(path) for path in records})
sources = dict(prior['tools'])
sources.update({name: descriptor(tools / name) for name in ['qualify_macos_vm_data.py', 'macos_vm_data_metrics.py',
    'macos_vm_data_probe.py', 'test_macos_vm_data_metrics.py', 'test_macos_vm_data_probe.py']})
denied = set(prior['denied_paths']) | {str(path) for path in cache.iterdir() if path.is_dir()}
denied.discard(str(origin))
old = tools / 'macos-vm-data.protocol.json'
probe_name = json.loads(old.read_text())['probe']['probe_name'] if old.exists() else 'gen2-data-probe-' + uuid4().hex
protocol = {'schema_version': 1,
    'scope': 'Fresh paired clone of a qualified stopped booted seed; strict System/Data UUID group validation, private DataRW/SystemRO mounts, bounded access probes and exact unmount/detach. No VM, network, guest account/SSH/provisioning, private app/model or release qualification.',
    'output_parent': str(cache), 'output_name': 'macos-vm-data-local-v1', 'expected_cpu': 'Apple M4 Pro',
    'tools': sources, 'external_files': external,
    'boot_prerequisite': {'receipt': str(receipt), 'local_root': str(origin),
        'source_commit': '8c76eb8ad9764f134ea74432d54df8a72a46bc91', 'seed_identity': seed_identity},
    'denied_paths': sorted(denied),
    'storage': {'start_minimum': 120 * 1024**3, 'running_minimum': 80 * 1024**3, 'reserve_bytes': 256 * 1024**2},
    'command_timeout_seconds': 120,
    'probe': {'probe_name': probe_name, 'product_version_components': image['os_version'], 'product_build': image['build_version']},
    'limits': [
        'Review and commit current source/protocol before invocation. Passing100-member boot/stopped/read-only receipt and unchanged paired seed required; preserve all earlier outcomes. One fresh output/no retries, physical M4Pro only, source/new exclusive locks held through clone/mount/probe/cleanup.',
        'Four descriptor-anchored same-volume fclonefileat(CLONE_NOFOLLOW) calls, no fallback, exact paired600 owned regular single-link distinct-inode files under700. Original booted seed identities and bounded metadata hashes unchanged. Never read/hash100GiB disk; record stat/OSclone provenance only, inherited helper installer label applies to earlier prerequisite, not this invocation.',
        'Initial120GiB/running80GiB/physical256MiB reserve. All child operations deny networking, models/private reads, repository and original stopped-root writes. Actual negative controls precede clone. No VM/installer/helper build/sign, host sudo/credential/certificate/profile/Keychain/security/settings/ownership-database changes, VNC/window/listener or private app/model.',
        'Attach only exact fresh cloned RAW via hdiutil CRawDiskImage/readwrite/nomount/nobrowse/plist. Observe exact own-image GUID whole and physical children; discover scoped containers with sole matching physical store, query actual volume IDs. Reject existing mounts, ambiguous/foreign/duplicate metadata. No fixed disk number/partition index/name/firstData selection.',
        'Use observed raw group schema from prior receipt. Exactly one image-wide System/Data group, one System and Data member, matching containerUUID/groupUUID, group DiskUUID/list APFSVolumeUUID/info DiskUUID+VolumeUUID/APFSVolumeGroupID, physical-store bridge, exact roles and unlocked/unencrypted members. Reject extra groups, duplicates and disagreements; validation uses UUIDs/devices rather than labels.',
        'Privately mount selected Data read-write and selected System readOnly with diskutil mount nobrowse -mountOptions noowners -mountPoint owned700path. Observe exact device/UUID/group/container/store/path/APFS and distinct actual filesystem devices; OS readonly flags and WritableVolume must match DataRW/SystemRO; GlobalPermissionsEnabled remainsfalse. DiskArbitration may run minimal fsck/metadata writes on disposable clone. No host enableOwnership/disableOwnership.',
        'Probe child anchors O_DIRECTORY/O_NOFOLLOW descriptors to independently validated mounts, rejects symlink/filesystem escapes, bounds SystemVersion.plist to1MiB and requires pinned numeric OS components/build (two-component productversion normalized with patch0). No host version fallback. No unverified SetupAssistantInfo schema or hardcoded Lume26.5.2 values.',
        'Observe seven existing Data provisioning directories (db, dslocal users/groups, xpc.launchd, Preferences, Users, private/etc) without creating/chmodding existing paths. In each create one task-unique non-plist O_EXCL/NOFOLLOW mode600 tiny probe; write/fsync/readback/chmod600/unlink/fsync. Parent independently verifies absence. Open existing root.plist/admin.plist O_WRONLY/NOFOLLOW without truncate/write and require metadata unchanged; never read account content. No account/home/SetupAssistant/sudoers/SSH/settings provisioning.',
        'Optionally read only expected Data/private/etc/ssh/ssh_host_ed25519_key.pub bounded8192B and validate public-key wire format; absence remains an observation, never trust-on-first-use or permission to read private key. No SSH/network call. noowners apparent UID/GID is not stored guest ownership proof; flag remainsfalse and future guest-side repair/verification required before private payloads.',
        'Free reserve before bounded cleanup, ignore controller SIGTERM/INT during cleanup, never let admission/persist errors veto teardown. Bind selected members to current exact own-image entities and UUID/store before normal unmount, verify no MountPoint/is_mount; continue exact current own-image detach even if a member query/unmount fails. Verify attachment/mount absence, preserve failures, no force or silent retry. Locks release only afterward. Any probe/mount/cleanup failure quarantines clone, never a reusable prepared guest.',
        'Keep paired binaries/disk and raw global image inventories local; publish scoped own-image JSON and global-log hashes only. No complete OS/runtime closure or correct guest UID/GID proof. Passing bounded access establishes strict pair and stopped-volume access only, not guest readiness/provisioning/authenticatedSSH/fresh-machine complete app, GUI/microphone/IPC/playback, original quality/human timing, other platform performance or public-signing/release/main promotion.']}
assert not (cache / protocol['output_name']).exists()
old.write_text(json.dumps(protocol, indent=2) + '\n')
print(descriptor(old), 'tools', len(sources), 'external', len(external))
