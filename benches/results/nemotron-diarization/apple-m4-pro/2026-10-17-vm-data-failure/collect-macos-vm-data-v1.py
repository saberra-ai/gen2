"""Retain terminal stopped-volume access evidence; saved-data reads only."""
from pathlib import Path
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'macos-vm-data-local-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
passed = q['data_access_passed'] is True
source = q['source_commit']
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/macos-vm-data.protocol.json'], cwd=repo,
    env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'})
for name, want in p['tools'].items():
    assert descriptor(tools / name) == want
for name, want in p['external_files'].items():
    assert descriptor(name) == want
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_vm_seed_clone import file_identity
from macos_vm_install_metrics import installed_seed_metrics
from macos_vm_data_metrics import system_data_pair, mounted_member
from macos_vm_host_metrics import attached_image_device
from qualify_macos_vm_host_v4 import policy
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: want for name, want in p['tools'].items()})
prior = Path(p['boot_prerequisite']['receipt'])
manifest = read(prior / 'collection.json')
for name, want in manifest['files'].items():
    assert descriptor(prior / name) == want
pq = read(prior / 'qualification.json')
assert pq['boot_discovery_passed'] is True and pq['source_commit'] == p['boot_prerequisite']['source_commit']
assert q['prior_boot_receipt_members_verified'] == len(manifest['files'])
origin = Path(p['boot_prerequisite']['local_root'])
source_preservation = {'unchanged': False}
try:
    assert installed_seed_metrics(origin / 'vm-seed', pq['booted_seed'], uid=os.getuid()) == pq['booted_seed']
    assert {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()} == q['original_seed_identity'] == p['boot_prerequisite']['seed_identity']
    source_preservation['unchanged'] = True
except (OSError, ValueError, AssertionError) as error:
    source_preservation['error'] = repr(error)
    if passed:
        raise
for flag in ['stored_guest_ownership_verified', 'fresh_machine_qualified', 'release_qualified',
        'guest_os_ready_qualified', 'virtual_machine_created', 'guest_boot_started', 'provisioning_performed',
        'ssh_or_private_app_or_models_executed', 'networking_permitted']:
    assert q[flag] is False
assert not (root / 'diagnostic-reserve.bin').exists()
storage = q.get('storage')
if storage:
    assert storage['released'] is True
    assert storage['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
offline = policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n'
sandbox_status = 'absent'
if (root / 'offline.sb').is_file():
    sandbox_status = 'verified' if (root / 'offline.sb').read_text() == offline else 'changed'
if passed:
    assert sandbox_status == 'verified'
assert all('returncode' in command or 'startup_error' in command for command in q['commands'])
image = root / 'vm-seed/disk.img'
raw_inventories = {}
inventory_names = {command['name'] + '.stdout.log' for command in q['commands']
    if command['argv'][-3:] == ['/usr/bin/hdiutil', 'info', '-plist']}
for name in inventory_names:
    raw_path = root / name
    if raw_path.is_file():
        raw_inventories[name] = descriptor(raw_path)
for path in root.glob('*.own-image.json'):
    prefix = path.name.removesuffix('.own-image.json')
    raw_path = root / (prefix + '.stdout.log')
    assert raw_path.name in inventory_names
    inventory = plistlib.loads(raw_path.read_bytes())
    expected = {'image': str(image), 'device': attached_image_device(inventory, image),
        'matching_images': [row for row in inventory['images'] if row.get('image-path') == str(image)]}
    assert read(path) == expected
if 'data_probe' in q:
    assert read(root / 'data-probe.stdout.log') == q['data_probe']
    assert q['data_probe']['stored_guest_ownership_verified'] is False and q['data_probe']['provisioning_performed'] is False
    if 'system_version_file' in q['data_probe']:
        assert descriptor(root / 'guest-SystemVersion.plist') == q['data_probe']['system_version_file']
partial_verification = {'system_data_pair': False, 'mounts': {}, 'unmounts': {}, 'attachment_absence': False}
if q['system_data_group_verified'] is True:
    assert read(root / 'before-image.own-image.json')['device'] is None
    own = read(root / 'attached-image.own-image.json')
    assert own['device'] == q['layout']['whole']
    assert len(own['matching_images']) == 1 and own['matching_images'][0]['writeable'] is True
    observed_partitions = [row['dev-entry'] for row in own['matching_images'][0]['system-entities']
        if re.fullmatch(re.escape(own['device']) + r's[1-9][0-9]*', row.get('dev-entry', ''))]
    assert observed_partitions == [row['device'] for row in q['layout']['partitions']]
    observed_pairs = []
    partitions = q['layout']['partitions']
    for index, row in enumerate(partitions):
        assert row['info'] == plistlib.loads((root / f'partition-{index}-info.stdout.log').read_bytes())
        assert row['info']['DeviceIdentifier'] == row['device'][5:] and row['info']['DeviceNode'] == row['device']
        assert row['info']['ParentWholeDisk'] == own['device'][5:] and not row['info'].get('MountPoint')
    for row in q['layout']['containers']:
        index = next(index for index, physical in enumerate(partitions) if physical['device'] == row['physical_store'])
        assert row['listing'] == plistlib.loads((root / f'container-{index}-list.stdout.log').read_bytes())
        assert row['groups'] == plistlib.loads((root / f'container-{index}-groups.stdout.log').read_bytes())
        for volume_index, volume in enumerate(row['listing']['Containers'][0]['Volumes']):
            assert row['infos'][volume['DeviceIdentifier']] == plistlib.loads((root / f'volume-{index}-{volume_index}-info.stdout.log').read_bytes())
        if any(role in {'System', 'Data'} for volume in row['listing']['Containers'][0]['Volumes'] for role in volume['Roles']):
            observed_pairs.append(system_data_pair(row['listing'], row['groups'], row['device'], row['physical_store'], row['infos'], partitions[index]['info']))
    assert observed_pairs == [q['system_data_pair']]
    partial_verification['system_data_pair'] = True
    for role, record in q.get('mounts', {}).items():
        pair = q['system_data_pair']
        assert record['info'] == plistlib.loads((root / ('mounted-' + role + '-info.stdout.log')).read_bytes())
        assert record['validated'] == mounted_member(record['info'], root / (role + '-mount'), pair[role], pair, record['status'], readonly=role == 'system')
        partial_verification['mounts'][role] = True
        unmounted_path = root / ('unmounted-' + role + '-info.stdout.log')
        if unmounted_path.is_file() and unmounted_path.stat().st_size:
            unmounted = plistlib.loads(unmounted_path.read_bytes())
            assert unmounted['DeviceNode'] == pair[role]['device'] and unmounted['VolumeUUID'] == pair[role]['uuid']
            assert not unmounted.get('MountPoint')
            partial_verification['unmounts'][role] = True
if q['owned_image_detached'] is True:
    assert read(root / 'after-image.own-image.json')['device'] is None
    partial_verification['attachment_absence'] = True
if passed:
    assert not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors', 'source_preservation_error', 'post_detach_seed_error'])
    assert q['clone_quarantined'] is False and q['system_data_group_verified'] is True
    assert q['exclusive_lock_released'] is True and q['source_exclusive_lock_released'] is True
    assert q['owned_image_detached'] is True and q['original_seed_unchanged_after_cleanup'] is True
    assert all(command['returncode'] == 0 for command in q['commands'])
    assert storage and storage['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
    assert set(q['clones']) == set(pq['booted_seed'])
    for name, clone in q['clones'].items():
        assert clone['operation'] == 'fclonefileat' and clone['returncode'] == 0 and clone['fallback_copy'] is False
        assert clone['source_before'] == clone['source_after'] == q['original_seed_identity'][name]
        assert clone['destination']['inode'] != clone['source_before']['inode']
        assert clone['destination']['bytes'] == clone['source_before']['bytes']
        assert all(q['cloned_seed_before_attach'][name][key] == pq['booted_seed'][name][key]
            for key in ['bytes', 'sha256', 'mode', 'uid', 'links'])
    accessed = installed_seed_metrics(root / 'vm-seed', pq['booted_seed'], uid=os.getuid())
    accessed['disk.img']['content_evidence'] = 'OS clone of qualified stopped booted seed, then readwrite filesystem access and confirmed unmount/detach; no installer or VM in this attempt, no whole-disk hash.'
    assert accessed == q['accessed_seed']
    for name in accessed:
        if name != 'disk.img':
            assert all(accessed[name][key] == pq['booted_seed'][name][key] for key in ['bytes', 'sha256', 'mode', 'uid', 'links'])
    assert partial_verification['system_data_pair'] is True
    assert partial_verification['mounts'] == partial_verification['unmounts'] == {'data': True, 'system': True}
    probe = q['data_probe']
    assert probe['access_passed'] is True and probe['step'] == 'completed'
    version = plistlib.loads((root / 'guest-SystemVersion.plist').read_bytes())
    assert version['ProductVersion'] == probe['product_version'] and version['ProductBuildVersion'] == probe['product_build'] == p['probe']['product_build']
    components = [int(part) for part in probe['product_version'].split('.')]
    assert components + [0] * (3 - len(components)) == p['probe']['product_version_components']
    directories = {'private/var/db', 'private/var/db/dslocal/nodes/Default/users',
        'private/var/db/dslocal/nodes/Default/groups', 'private/var/db/com.apple.xpc.launchd',
        'Library/Preferences', 'Users', 'private/etc'}
    assert len(probe['writes']) == 7 and {row['directory'] for row in probe['writes']} == set(probe['directory_metadata']) == directories
    payload = b'Gen2 disposable guest-volume access control; no account or app.\n'
    for row in probe['writes']:
        assert row['probe_removed'] is True and row['probe']['mode'] == 0o600
        assert row['probe']['device'] == row['parent']['device'] == q['mounts']['data']['status']['stat_device']
        assert row['payload'] == {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    assert len(probe['existing_write_opens']) == 2
    assert {row['path'] for row in probe['existing_write_opens']} == {'private/var/db/dslocal/nodes/Default/users/root.plist', 'private/var/db/dslocal/nodes/Default/groups/admin.plist'}
    assert all(row['write_open_without_write'] is True for row in probe['existing_write_opens'])
    assert q['probe_removal_independently_verified'] is True
    assert read(root / 'after-image.own-image.json')['device'] is None
else:
    assert q['clone_quarantined'] is True
    assert any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors', 'source_preservation_error', 'post_detach_seed_error'])
    # Never mount or read a failed clone in collection; only saved files and original seed metadata.
test_log = cache / 'platform-ci/model-free-320-macos-vm-data-local.log'
assert 'Ran 320 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
out = repo / ('benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-data-' + ('access' if passed else 'failure'))
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb', '.plist'} and path.name not in inventory_names:
        shutil.copy2(path, out / path.name)
for path in [Path(__file__), cache / 'prepare-macos-vm-data-v1.py', root.with_suffix('.controller.log'), test_log]:
    shutil.copy2(path, out / path.name)
(out / 'sources').mkdir()
for name in p['tools']:
    shutil.copy2(tools / name, out / 'sources' / name)
summary = {'schema_version': 1, 'source_commit': source, 'data_access_passed': passed,
    'system_data_group_verified': q['system_data_group_verified'], 'stored_guest_ownership_verified': False,
    'fresh_machine_qualified': False, 'release_qualified': False, 'guest_os_ready_qualified': False, 'provisioning_performed': False,
    'actual_commands': len(q['commands']), 'zero_status_commands': sum(command.get('returncode') == 0 for command in q['commands']),
    'original_booted_seed_identity_unchanged': source_preservation['unchanged'],
    'source_preservation': source_preservation, 'sandbox_status': sandbox_status,
    'partial_phase_verification': partial_verification, 'prior_boot_members_verified': len(manifest['files']),
    'current_historical_source_pins_verified': len(p['tools']), 'external_descriptors_verified': len(p['external_files']),
    'model_free_tests_passed': 320, 'clone_quarantined': q['clone_quarantined'], 'owned_image_detached': q['owned_image_detached'],
    'system_data_pair': q.get('system_data_pair'), 'data_probe': q.get('data_probe'), 'accessed_seed': q.get('accessed_seed'),
    'public_host_key_check_reached': q.get('data_probe', {}).get('step') in {'optional public host key', 'completed'},
    'public_host_key_default_limit': 'A false present field in a partial probe report is an initial placeholder unless the public-host-key step was reached; it does not establish actual key absence.',
    'global_image_inventory_retained_local_only': raw_inventories,
    'inherited_pre_attach_disk_evidence_label_limit': 'The helper label in cloned_seed_before_attach describes the earlier installed-seed prerequisite, not an installer after this clone. This attempt has no VM or installer; successful current disk provenance is recorded separately in accessed_seed.',
    'physical_reserve_released': not (root / 'diagnostic-reserve.bin').exists(),
    'new_helper_vm_network_model_mount_clone_calls_during_collection': 0,
    'limits': ['Strict group and bounded stopped-volume access only; no guest account/SSH/provisioning, OS readiness or private complete-app/model execution.',
        'Noowners observedUID/GID is not storedguestownershipproof. Future authenticated guest repair/verification remains required before payload use.',
        'Raw100GiB disk stat/OSclone provenance only; no whole-disk read/hash. Mount/fsck/probes may change disposable clone metadata; original booted seed preserved.',
        'Failure quarantines clone; no silent retry/force/automatic reuse. No new mount/helper/device/network/VM operation during saved-data collection.',
        'Human timing deferred. Quality, GUI/microphone/playback/IPC, fresh-machine app, otherplatform performance and publicsigning/release/main promotion remain open.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members', source)
