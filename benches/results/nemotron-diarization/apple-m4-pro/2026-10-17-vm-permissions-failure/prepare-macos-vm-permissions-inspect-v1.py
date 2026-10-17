"""Metadata-only freeze for fresh-clone parent-relative mode/ACL inspection."""
from pathlib import Path
import hashlib
import json
import os
import sys

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


prior = json.loads((tools / 'macos-vm-data.protocol.json').read_text())
for name, want in prior['tools'].items():
    assert descriptor(tools / name) == want
for name, want in prior['external_files'].items():
    assert descriptor(name) == want
receipt = Path(prior['boot_prerequisite']['receipt'])
manifest = json.loads((receipt / 'collection.json').read_text())
for name, want in manifest['files'].items():
    assert descriptor(receipt / name) == want
origin = Path(prior['boot_prerequisite']['local_root'])
pq = json.loads((receipt / 'qualification.json').read_text())
assert (origin / 'qualification.json').read_bytes() == (receipt / 'qualification.json').read_bytes()
assert pq['boot_discovery_passed'] is True and pq['boot_owner_qualified'] is True
assert pq['owned_image_detached'] is True and pq['owner_result']['cleanup_confirmed'] is True
assert pq['owner_result']['final_state'] == 0 and pq['exclusive_lock_released'] is True
assert installed_seed_metrics(origin / 'vm-seed', pq['booted_seed'], uid=os.getuid()) == pq['booted_seed']
seed_identity = {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()}
failed_receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-data-failure'
failed_manifest = json.loads((failed_receipt / 'collection.json').read_text())
for name, want in failed_manifest['files'].items():
    assert descriptor(failed_receipt / name) == want
failed = json.loads((failed_receipt / 'qualification.json').read_text())
assert failed['data_access_passed'] is False and failed['clone_quarantined'] is True and failed['owned_image_detached'] is True
assert failed['data_probe']['writes'] == [] and failed['data_probe']['existing_write_opens'] == []
assert failed['data_probe']['step'] == 'directory metadata: private/var/db/dslocal/nodes/Default/users'
sdk = Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk')
external = dict(prior['external_files'])
records = [Path(__file__), failed_receipt / 'collection.json', failed_receipt / 'qualification.json',
    failed_receipt / 'collection-qualification.json', cache / 'macos-vm-data-local-v1/qualification.json',
    cache / 'macos-vm-data-local-v1/protocol.json', sdk / 'usr/include/sys/attr.h', sdk / 'usr/include/sys/kauth.h',
    sdk / 'usr/share/man/man2/getattrlist.2']
external.update({str(path): descriptor(path) for path in records})
sources = dict(prior['tools'])
sources.update({name: descriptor(tools / name) for name in ['qualify_macos_vm_permissions_inspect.py',
    'macos_vm_permissions_probe.py', 'test_macos_vm_permissions_probe.py']})
denied = set(prior['denied_paths']) | {str(path) for path in cache.iterdir() if path.is_dir()}
denied.discard(str(origin))
protocol = {'schema_version': 1,
    'scope': 'Metadata-only parent-relative inspection of known guest directories on a fresh paired clone and DataRO mount. No permission repair, account content, enumeration, provisioning, VM/network/SSH/private app/model or release qualification.',
    'output_parent': str(cache), 'output_name': 'macos-vm-permissions-inspect-local-v1', 'expected_cpu': 'Apple M4 Pro',
    'tools': sources, 'external_files': external,
    'boot_prerequisite': {**prior['boot_prerequisite'], 'seed_identity': seed_identity},
    'failed_access_prerequisite': {'receipt': str(failed_receipt), 'source_commit': 'ec80cca155b7d214c42922f0fc88d3182f9658ed'},
    'denied_paths': sorted(denied),
    'storage': {'start_minimum': 120 * 1024**3, 'running_minimum': 80 * 1024**3, 'reserve_bytes': 256 * 1024**2},
    'command_timeout_seconds': 120,
    'limits': [
        'Reviewed committed source/protocol required before invocation. Preserve passing100-member boot receipt and104-member failed-access receipt; do not touch/remount/read the failed clone. One fresh output/pairedclone from qualified stopped booted source, physical M4Pro only, no retry. Source/new exclusive locks held through clone/attachment/mount/metadata/cleanup.',
        'Exactly four descriptor-anchored fclonefileat(CLONE_NOFOLLOW) calls with no fallback, fresh600 owned single-link regular distinct-inode destinations under700. Original stopped seed identities and bounded metadata remain unchanged. Never read/hash100GiB disk; OSclone and stat provenance only. Source remains operationally frozen, not protected from its owner/privileged processes.',
        'Initial120GiB/running80GiB/physical256MiB reserve. All children offline/model/private-read denied and original/repositorywrite denied; actual negative controls precede clone. New fresh host scratchdir only for a bounded getattrlistat ABI control before any guest attachment. No VM/build/sign, hostsudo/credentials/Keychain/security/settings/ownershipDB/listeners, guestaccount/SSH/privateapp/model or permission repair.',
        'Actual control and guest queries request only ATTR_CMN_EXTENDED_SECURITY using pinned Xcode26.5 attrlist24B/bitmapcount5/common0x00400000/FSOPT_NOFOLLOW1, nativeDarwinarm64/c_ulong8, 65536B zeroed buffer, explicit errno. Validate uint32length and relative int32offset/uint32length including zero-length references, bounds/alignment/truncation; retain rawbytes/base64/hash plus status/errno. Host scratch success qualifies syscall path only, never guestACL support.',
        'Native kauth_filesec recognized only for magic0x012cc16d and exact44+24*N layout with N<=128 (NOACL0xffffffff useszeroentries). Count0 is distinct fromNOACL. Canonical one-attribute zero length is12/offset8/length0; null0/0 retained as unknown/unparsed, never recognized or an access verdict. All ACL policy qualification flags remainfalse. Unsupported/denied/unparsed metadata is not emptyACL or permission to change modes.',
        'Attach only exact freshRAW readonly/nomount/nobrowse/plist, verifywriteablefalse. Discover ownGUIDwhole/physicalchildren/scopedcontainers and strict uniqueSystem/Data group with physicalstoreUUID/containerUUID/groupUUID/allvolumeUUIDs/roles agreeing. No fixed disk/index/name/firstData selection. Mount only selected Data readOnly/nobrowse/noowners at privatepoint; validate currentUUID/group/store/path/APFS/WritableVolumefalse/OSreadonly/distinctstdev/permissionsfalse. System remainsunmounted.',
        'Probe verifies its openedrootfd against independentlyvalidatedactualDataRO. Literal known provisioning directory prefixes only, at mostsix target paths/sharedancestors; no directoryenumeration or account/file/hostkeycontents. Before every open: parent-relative stat NOFOLLOW captures fullmode/type/inode/device/links/flags/size/times/apparentUIDGID, rejectsymlink/crossfilesystem; query boundedsecurity metadata, re-stat sameidentity, then attempt normal readonlydirectoryopen and comparefstat. Preserve EACCES/ENOENT as observations and markunreacheddescendants explicitly; unexpectedexceptions retainpartialrows andcloseallfds.',
        'Inspection PASS requires actual users-entry stat reached and completeprobezeroexit with zero permissionchanges and all account/enumeration/access/provisioning/ownershipclaimsfalse, plus strictpair/mount/sourcepreservation andteardown. It does not require a successful usersopen or supported ACL; those remain explicit rawoutcomes. data_access_passed andprepared_guest remainfalse evenifinspectionpasses. Missing usersmetadata isfailedinspection, never guesswhy.',
        'Free reserve before cleanup, ignoreSIGTERM/INT during boundedcleanup; admission/persist failures cannot veto teardown. Bindcurrentownimageentities+UUID/store before normalunmount, verify noMountPoint/is_mount, continue exactcurrentown-image detach evenif memberquery/unmountfails, verifyabsence, releasebothlocksafterward. No force/silent retry. Failurequarantinesclone; do not inferguestreadiness from metadata or successfulteardown.',
        'Keep paired binaries/disk and globalHDIinventorieslocal. Publicreceipt contains only scopedmetadata/rawACL buffers/generatedscratchcontrol and globalinventoryhashes/ownfilteredrecords. No fullOS/kernel/runtimeclosure claim. No chmod/chown/ACL/flags repair, storedguestownership, authenticatedSSH, freshmachinecompleteapp, GUI/mic/playback/IPC, originalquality/humantiming, platformperformance or publicsigning/release/main promotion. Any later boundedguestchange needs separate declaredadmission andactualverification.']}
assert not (cache / protocol['output_name']).exists()
target = tools / 'macos-vm-permissions-inspect.protocol.json'
target.write_text(json.dumps(protocol, indent=2) + '\n')
print(descriptor(target), 'tools', len(sources), 'external', len(external))
