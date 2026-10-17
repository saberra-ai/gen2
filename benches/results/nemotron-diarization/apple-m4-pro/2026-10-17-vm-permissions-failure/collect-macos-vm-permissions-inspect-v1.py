"""Collect terminal read-only permission inspection from saved data only."""
from pathlib import Path
import base64
import hashlib
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'macos-vm-permissions-inspect-local-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
passed = q['inspection_passed'] is True
source = q['source_commit']
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/macos-vm-permissions-inspect.protocol.json'], cwd=repo,
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
from macos_vm_permissions_probe import CAPACITY, TARGETS, packed_security
from qualify_macos_vm_host_v4 import policy
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: want for name, want in p['tools'].items()})
prior = Path(p['boot_prerequisite']['receipt'])
manifest = read(prior / 'collection.json')
for name, want in manifest['files'].items():
    assert descriptor(prior / name) == want
pq = read(prior / 'qualification.json')
assert pq['boot_discovery_passed'] is True and pq['source_commit'] == p['boot_prerequisite']['source_commit']
assert q['prior_boot_receipt_members_verified'] == len(manifest['files'])
failure = Path(p['failed_access_prerequisite']['receipt'])
failed_manifest = read(failure / 'collection.json')
for name, want in failed_manifest['files'].items():
    assert descriptor(failure / name) == want
failed = read(failure / 'qualification.json')
assert failed['data_access_passed'] is False and failed['clone_quarantined'] is True
assert failed['source_commit'] == p['failed_access_prerequisite']['source_commit']
assert q['prior_failed_access_receipt_members_verified'] == len(failed_manifest['files'])
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
for flag in ['data_access_passed', 'prepared_guest', 'stored_guest_ownership_verified', 'fresh_machine_qualified',
        'release_qualified', 'guest_os_ready_qualified', 'virtual_machine_created', 'guest_boot_started',
        'provisioning_performed', 'ssh_or_private_app_or_models_executed', 'networking_permitted']:
    assert q[flag] is False
assert not (root / 'diagnostic-reserve.bin').exists()
storage = q.get('storage')
if storage:
    assert storage['released'] is True and storage['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
offline = policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n'
sandbox_status = 'absent'
if (root / 'offline.sb').is_file():
    sandbox_status = 'verified' if (root / 'offline.sb').read_text() == offline else 'changed'
if passed:
    assert sandbox_status == 'verified'
assert all('returncode' in row or 'startup_error' in row for row in q['commands'])
inventory_names = {row['name'] + '.stdout.log' for row in q['commands']
    if row['argv'][-3:] == ['/usr/bin/hdiutil', 'info', '-plist']}
inventories = {name: descriptor(root / name) for name in inventory_names if (root / name).is_file()}
image = root / 'vm-seed/disk.img'
for path in root.glob('*.own-image.json'):
    raw_name = path.name.removesuffix('.own-image.json') + '.stdout.log'
    assert raw_name in inventory_names
    inventory = plistlib.loads((root / raw_name).read_bytes())
    expected = {'image': str(image), 'device': attached_image_device(inventory, image),
        'matching_images': [row for row in inventory['images'] if row.get('image-path') == str(image)]}
    assert read(path) == expected


def verify_security(value):
    assert value['acl_policy_qualified'] is False
    if value['status'] == 'query_error':
        assert value['returncode'] != 0 and type(value['errno']) is int
        return
    assert value['status'] in {'ok', 'unparsed'} and value['returncode'] == 0
    raw = base64.b64decode(value['raw_base64'], validate=True)
    assert len(raw) == value['bytes'] <= CAPACITY and hashlib.sha256(raw).hexdigest() == value['sha256']
    try:
        decoded = packed_security(raw)
    except ValueError:
        assert value['status'] == 'unparsed'
    else:
        assert value['status'] == ('ok' if decoded['abi_shape_qualified'] else 'unparsed')
        assert all(value[key] == expected for key, expected in decoded.items())


if 'acl_abi_control' in q:
    assert read(root / 'acl-abi-control.stdout.log') == q['acl_abi_control']
    verify_security(q['acl_abi_control']['result'])
if 'permissions_probe' in q:
    assert read(root / 'permissions-probe.stdout.log') == q['permissions_probe']
    for row in q['permissions_probe'].get('nodes', {}).values():
        if 'security' in row:
            verify_security(row['security'])
if passed:
    assert not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors', 'source_preservation_error', 'post_detach_seed_error'])
    assert q['clone_quarantined'] is False and q['system_data_group_verified'] is True
    assert q['exclusive_lock_released'] is True and q['source_exclusive_lock_released'] is True
    assert q['owned_image_detached'] is True and q['original_seed_unchanged_after_cleanup'] is True
    assert q['users_metadata_acquired'] is True and all(row['returncode'] == 0 for row in q['commands'])
    assert storage and storage['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
    control = q['acl_abi_control']
    assert control['attrlist_bytes'] == 24 and control['capacity'] == CAPACITY and control['result']['status'] == 'ok'
    assert set(q['clones']) == set(pq['booted_seed'])
    for name, clone in q['clones'].items():
        assert clone['operation'] == 'fclonefileat' and clone['returncode'] == 0 and clone['fallback_copy'] is False
        assert clone['source_before'] == clone['source_after'] == q['original_seed_identity'][name]
        assert clone['destination']['inode'] != clone['source_before']['inode']
        assert clone['destination']['bytes'] == clone['source_before']['bytes']
        assert all(q['cloned_seed_before_attach'][name][key] == pq['booted_seed'][name][key]
            for key in ['bytes', 'sha256', 'mode', 'uid', 'links'])
    inspected = installed_seed_metrics(root / 'vm-seed', pq['booted_seed'], uid=os.getuid())
    inspected['disk.img']['content_evidence'] = 'OS clone of qualified stopped booted seed, then readonly metadata inspection and confirmed unmount/detach; no installer, VM or permission change in this attempt, no whole-disk hash.'
    assert inspected == q['inspected_seed']
    for name in inspected:
        if name != 'disk.img':
            assert all(inspected[name][key] == pq['booted_seed'][name][key] for key in ['bytes', 'sha256', 'mode', 'uid', 'links'])
    assert read(root / 'before-image.own-image.json')['device'] is None
    own = read(root / 'attached-image.own-image.json')
    assert own['device'] == q['layout']['whole'] and len(own['matching_images']) == 1
    assert own['matching_images'][0]['writeable'] is False
    partitions = [row['dev-entry'] for row in own['matching_images'][0]['system-entities']
        if re.fullmatch(re.escape(own['device']) + r's[1-9][0-9]*', row.get('dev-entry', ''))]
    assert partitions == [row['device'] for row in q['layout']['partitions']]
    for index, row in enumerate(q['layout']['partitions']):
        assert row['info'] == plistlib.loads((root / f'partition-{index}-info.stdout.log').read_bytes())
        assert row['info']['ParentWholeDisk'] == own['device'][5:] and not row['info'].get('MountPoint')
    pairs = []
    for row in q['layout']['containers']:
        index = partitions.index(row['physical_store'])
        assert row['listing'] == plistlib.loads((root / f'container-{index}-list.stdout.log').read_bytes())
        assert row['groups'] == plistlib.loads((root / f'container-{index}-groups.stdout.log').read_bytes())
        for volume_index, volume in enumerate(row['listing']['Containers'][0]['Volumes']):
            assert row['infos'][volume['DeviceIdentifier']] == plistlib.loads((root / f'volume-{index}-{volume_index}-info.stdout.log').read_bytes())
        if any(role in {'System', 'Data'} for volume in row['listing']['Containers'][0]['Volumes'] for role in volume['Roles']):
            pairs.append(system_data_pair(row['listing'], row['groups'], row['device'], row['physical_store'], row['infos'], q['layout']['partitions'][index]['info']))
    assert pairs == [q['system_data_pair']]
    pair = pairs[0]
    assert set(q['mounts']) == {'data'}
    mounted = q['mounts']['data']
    assert mounted['info'] == plistlib.loads((root / 'mounted-data-info.stdout.log').read_bytes())
    assert mounted['validated'] == mounted_member(mounted['info'], root / 'data-mount', pair['data'], pair, mounted['status'], readonly=True)
    unmounted = plistlib.loads((root / 'unmounted-data-info.stdout.log').read_bytes())
    assert unmounted['DeviceNode'] == pair['data']['device'] and unmounted['VolumeUUID'] == pair['data']['uuid']
    assert not unmounted.get('MountPoint') and read(root / 'after-image.own-image.json')['device'] is None
    report = q['permissions_probe']
    assert report['schema_version'] == 1 and report['inspection_passed'] is True and report['permission_changes'] == 0
    assert all(report[name] is False for name in ['directory_contents_read', 'account_content_read', 'data_access_passed', 'provisioning_performed', 'stored_guest_ownership_verified'])
    expected_paths = {'/'.join(path.split('/')[:index]) for path in TARGETS for index in range(1, len(path.split('/')) + 1)}
    assert set(report['nodes']) == expected_paths
    for row in report['nodes'].values():
        if 'stat' in row:
            info = row['stat']
            assert info['device'] == mounted['status']['stat_device'] and stat.S_ISDIR(info['full_mode'])
            assert info['permissions'] == stat.S_IMODE(info['full_mode']) and info['file_type'] == stat.S_IFDIR
    assert 'stat' in report['nodes']['private/var/db/dslocal/nodes/Default/users']
else:
    assert q['clone_quarantined'] is True
    assert any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors', 'unmount_errors', 'source_preservation_error', 'post_detach_seed_error'])
    # Saved records only; never mount/read a failed clone here.
test_log = cache / 'platform-ci/model-free-326-macos-vm-permissions-inspect-local.log'
assert 'Ran 326 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
out = repo / ('benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-permissions-' + ('inspected' if passed else 'failure'))
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'} and path.name not in inventory_names:
        shutil.copy2(path, out / path.name)
for path in [Path(__file__), cache / 'prepare-macos-vm-permissions-inspect-v1.py', root.with_suffix('.controller.log'), test_log]:
    shutil.copy2(path, out / path.name)
(out / 'sources').mkdir()
for name in p['tools']:
    shutil.copy2(tools / name, out / 'sources' / name)
summary = {'schema_version': 1, 'source_commit': source, 'inspection_passed': passed,
    'data_access_passed': False, 'prepared_guest': False, 'stored_guest_ownership_verified': False,
    'fresh_machine_qualified': False, 'release_qualified': False, 'guest_os_ready_qualified': False,
    'provisioning_performed': False, 'permission_changes': 0,
    'actual_commands': len(q['commands']), 'zero_status_commands': sum(row.get('returncode') == 0 for row in q['commands']),
    'source_preservation': source_preservation, 'sandbox_status': sandbox_status,
    'prior_boot_members_verified': len(manifest['files']), 'prior_failed_access_members_verified': len(failed_manifest['files']),
    'current_historical_source_pins_verified': len(p['tools']), 'external_descriptors_verified': len(p['external_files']),
    'model_free_tests_passed': 326, 'clone_quarantined': q['clone_quarantined'], 'owned_image_detached': q['owned_image_detached'],
    'permissions_probe': q.get('permissions_probe'), 'global_image_inventory_retained_local_only': inventories,
    'physical_reserve_released': not (root / 'diagnostic-reserve.bin').exists(),
    'inherited_pre_attach_disk_evidence_label_limit': 'The inherited clone-helper installer label describes the earlier prerequisite; this attempt has no VM/installer. Current readonly-inspection provenance is separate in inspected_seed.',
    'new_helper_vm_network_model_mount_clone_calls_during_collection': 0,
    'limits': ['Read-only metadata diagnosis only. Earlier DataRW access remains failed; no repair, access acceptance, stored ownership, prepared guest or provisioning is established.',
        'Zero-length/unparsed/denied security metadata is not ACL absence or policy. Apparent UID/GID is not persisted guest ownership. No host keys, account content or directory enumeration.',
        'No whole100GiB disk read/hash. Paired disk/binaries and rawglobalimage inventories stay local; scopedrawACLmetadata is retained without an access-policy conclusion.',
        'No failureclone reuse or automatic privilege/permission/security/ownershipDB fallback. Guest readiness/SSH/freshapp, originalquality/humantiming, GUI/microphone/IPC/playback, platformperformance and publicrelease/main gates remain open.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members', source)
