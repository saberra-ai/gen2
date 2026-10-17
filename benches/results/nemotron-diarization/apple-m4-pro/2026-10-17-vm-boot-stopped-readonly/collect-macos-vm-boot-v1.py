"""Collect a terminal boot/discovery attempt from saved data only; no device calls."""
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
root = cache / 'macos-vm-boot-local-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
source = q['source_commit']
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/macos-vm-boot.protocol.json'], cwd=repo)
for name, want in p['tools'].items():
    assert descriptor(tools / name) == want
for name, want in p['external_files'].items():
    assert descriptor(name) == want
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_vm_seed_clone import file_identity
from macos_vm_install_metrics import installed_seed_metrics
from macos_vm_boot_metrics import boot_metrics, observed_partitions, scoped_container
from macos_vm_host_metrics import attached_image_device
from qualify_macos_vm_host_v4 import policy
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: want for name, want in p['tools'].items()})
prior = Path(p['installed_prerequisite']['receipt'])
manifest = read(prior / 'collection.json')
for name, want in manifest['files'].items():
    assert descriptor(prior / name) == want
pq = read(prior / 'qualification.json')
assert pq['installation_passed'] is True and pq['source_commit'] == p['installed_prerequisite']['source_commit']
assert q['prior_install_receipt_members_verified'] == len(manifest['files'])
origin = Path(p['installed_prerequisite']['local_root'])
assert installed_seed_metrics(origin / 'vm-seed', pq['installed_seed'], uid=os.getuid()) == pq['installed_seed']
assert {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()} == q['original_seed_identity'] == p['installed_prerequisite']['seed_identity']
for flag in ['fresh_machine_qualified', 'release_qualified', 'installation_started',
        'private_app_or_models_executed', 'guest_os_ready_qualified', 'system_data_group_verified']:
    assert q[flag] is False
assert q['setup_networking_permitted'] is True
assert q['storage']['released'] is True and not (root / 'diagnostic-reserve.bin').exists()
assert q['storage']['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
offline = policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n'
assert (root / 'offline.sb').read_text() == offline
assert (root / 'setup.sb').read_text() == offline.replace('(deny network*)\n', '')
assert all('returncode' in command for command in q['commands'])
passed = q['boot_discovery_passed'] is True
image = root / 'vm-seed/disk.img'
if passed:
    assert not any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors'])
    assert q['installed_disk_quarantined'] is False
    assert q['exclusive_lock_released'] is True and q['source_exclusive_lock_released'] is True
    assert q['owned_image_detached'] is True and q['scoped_metadata_acquired'] is True
    assert q['virtual_machine_created'] is True and q['normal_guest_boot_started'] is True
    names = [command['name'] for command in q['commands']]
    prefix = ['offline-controls', 'setup-controls', 'build', 'sign', 'signature-verify', 'signature-metadata',
        'entitlements', 'architecture', 'deployment', 'dependencies', 'error-control', 'shutdown-control', 'boot-owner',
        'before-image', 'attach-readonly', 'attached-image', 'whole-list', 'whole-info']
    assert names[:len(prefix)] == prefix
    assert names[-3:] == ['cleanup-image', 'detach-image', 'after-image']
    assert all(command['returncode'] == 0 for command in q['commands'])
    assert read(root / 'owner-result.json') == q['owner_result']
    events = [json.loads(line) for line in (root / 'boot-owner.stdout.log').read_text().splitlines()]
    assert boot_metrics(q['owner_result'], q['commands'][names.index('boot-owner')], events) == q['owner_result']
    assert installed_seed_metrics(root / 'vm-seed', pq['installed_seed'], uid=os.getuid()) == q['booted_seed']
    assert set(q['clones']) == set(pq['installed_seed'])
    for name, clone in q['clones'].items():
        assert clone['operation'] == 'fclonefileat' and clone['returncode'] == 0 and clone['fallback_copy'] is False
        assert clone['source_before'] == clone['source_after'] == q['original_seed_identity'][name]
        assert clone['destination']['inode'] != clone['source_before']['inode']
        assert clone['destination']['bytes'] == clone['source_before']['bytes']
        assert all(q['cloned_seed_before_boot'][name][key] == pq['installed_seed'][name][key]
            for key in ['bytes', 'sha256', 'mode', 'uid', 'links'])
    assert descriptor(root / 'vm-boot-owner') == q['owner_executable']
    assert plistlib.loads((root / 'entitlements.stdout.log').read_bytes()) == q['entitlements'] == {'com.apple.security.virtualization': True}
    signature = (root / 'signature-metadata.stderr.log').read_text()
    assert 'Signature=adhoc' in signature and re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature)
    assert (root / 'architecture.stdout.log').read_bytes().strip() == b'arm64'
    assert re.search(r'\bminos 26\.0\s', (root / 'deployment.stdout.log').read_text())
    assert read(root / 'error-control.stdout.log') == q['error_graph_control']
    assert read(root / 'shutdown-control.stdout.log') == q['shutdown_timing_control'] == {'checks': 5, 'passed': True}
    assert q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
    for name in ['before-image', 'attached-image', 'cleanup-image', 'after-image']:
        inventory = plistlib.loads((root / (name + '.stdout.log')).read_bytes())
        expected = {'image': str(image), 'device': attached_image_device(inventory, image),
            'matching_images': [row for row in inventory['images'] if row.get('image-path') == str(image)]}
        assert read(root / (name + '.own-image.json')) == expected
    assert read(root / 'before-image.own-image.json')['device'] is None
    own = read(root / 'attached-image.own-image.json')
    assert own['image'] == str(image) and own['device'] == q['layout']['whole']
    partitions = observed_partitions({'images': own['matching_images']}, image, own['device'])
    assert partitions == [row['device'] for row in q['layout']['partitions']]
    whole_info = plistlib.loads((root / 'whole-info.stdout.log').read_bytes())
    assert whole_info['DeviceIdentifier'] == own['device'][5:] and whole_info['DeviceNode'] == own['device']
    assert not whole_info.get('MountPoint')
    for index, row in enumerate(q['layout']['partitions']):
        assert row['info'] == plistlib.loads((root / f'partition-{index}-info.stdout.log').read_bytes())
        assert row['info']['DeviceIdentifier'] == row['device'][5:]
        assert row['info']['DeviceNode'] == row['device']
        assert row['info']['ParentWholeDisk'] == own['device'][5:] and not row['info'].get('MountPoint')
    roles = set()
    for row in q['layout']['containers']:
        index = partitions.index(row['physical_store'])
        assert q['layout']['partitions'][index]['info']['APFSContainerReference'] == row['device']
        listing = plistlib.loads((root / f'container-{index}-list.stdout.log').read_bytes())
        container = scoped_container(listing, row['device'], row['physical_store'])
        assert descriptor(root / f'container-{index}-groups.stdout.log') == row['volume_groups_raw']
        plistlib.loads((root / f'container-{index}-groups.stdout.log').read_bytes())
        assert len(row['volumes']) == len(container['Volumes'])
        for volume_index, (actual, observed) in enumerate(zip(row['volumes'], container['Volumes'])):
            assert actual['info'] == plistlib.loads((root / f'volume-{index}-{volume_index}-info.stdout.log').read_bytes())
            assert actual['device'] == '/dev/' + observed['DeviceIdentifier']
            assert actual['roles'] == observed['Roles'] and not actual['info'].get('MountPoint')
            assert actual['info']['DeviceIdentifier'] == observed['DeviceIdentifier']
            assert actual['info']['DeviceNode'] == actual['device']
            assert actual['info'].get('APFSContainerReference', row['device']) == row['device']
            roles.update(actual['roles'])
    assert {'System', 'Data'} <= roles
    assert read(root / 'after-image.own-image.json')['device'] is None
else:
    assert q['installed_disk_quarantined'] is True
    assert any(key in q for key in ['error', 'cleanup_error', 'cleanup_metadata_errors'])
    # Never read or inspect the contents of a failed/unknown owner's cloned disk.
    if (root / 'owner-result.json').is_file():
        assert read(root / 'owner-result.json') == q['owner_result']
test_log = cache / 'platform-ci/model-free-312-macos-vm-boot-local.log'
assert 'Ran 312 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
out = repo / ('benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-boot-' + ('stopped-readonly' if passed else 'failure'))
assert not out.exists()
out.mkdir()
raw_inventory = {name + '.stdout.log' for name in ['before-image', 'attached-image', 'cleanup-image', 'after-image']}
omitted = {}
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        if path.name in raw_inventory:
            omitted[path.name] = descriptor(path)
        else:
            shutil.copy2(path, out / path.name)
for path in [Path(__file__), cache / 'prepare-macos-vm-boot-v1.py', root.with_suffix('.controller.log'), test_log]:
    shutil.copy2(path, out / path.name)
(out / 'sources').mkdir()
for name in p['tools']:
    shutil.copy2(tools / name, out / 'sources' / name)
summary = {'schema_version': 1, 'source_commit': source, 'boot_discovery_passed': passed,
    'fresh_machine_qualified': False, 'release_qualified': False, 'guest_os_ready_qualified': False,
    'system_data_group_verified': False,
    'actual_commands': len(q['commands']), 'zero_status_commands': sum(command['returncode'] == 0 for command in q['commands']),
    'original_installed_seed_identity_unchanged': True,
    'prior_install_members_verified': len(manifest['files']), 'current_historical_source_pins_verified': len(p['tools']),
    'external_descriptors_verified': len(p['external_files']), 'model_free_tests_passed': 312,
    'setup_networking_permitted': True, 'installed_disk_quarantined': q['installed_disk_quarantined'],
    'owner_result': q.get('owner_result'), 'booted_seed': q.get('booted_seed'),
    'inherited_disk_evidence_label_limit': 'Raw booted_seed.disk.img.content_evidence comes from the installed-seed helper and describes the earlier installation prerequisite. This attempt cloned that already-installed source, then successfully normal-started, observed Running, stopped and exited; no installer callback occurred after this clone. The inherited label is not evidence of an installer invocation in this attempt.',
    'owned_image_detached': q['owned_image_detached'], 'physical_reserve_released': True,
    'global_image_inventory_retained_local_only': omitted,
    'new_helper_vm_network_model_mount_clone_calls_during_collection': 0,
    'limits': ['Normal start plus120s Running and stopped owner exit proves public lifecycle, not macOS readiness or app operation.',
        'Read-only no-mount metadata acquisition does not qualify System/Data group membership, ownership or guest provisioning.',
        'No whole100GiB disk hash/read; paired disk/metadata binaries/helper/modulecache remain local. Source stats plus bounded metadata hashes and OSclone provenance only.',
        'No cancellation/killed-owner actual qualification. Unknown/failed owner disks remain quarantined. No automatic replacement or inference from exit alone.',
        'Human timing review deferred; accuracy, GUI/microphone/IPC/playback, fresh-machine complete app, platforms and public release/main promotion remain open.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members', source)
