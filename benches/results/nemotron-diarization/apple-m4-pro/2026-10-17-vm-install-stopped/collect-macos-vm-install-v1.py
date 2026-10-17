"""Retain terminal installer-owner evidence; saved-data reads only, no VM calls."""
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
root = cache / 'macos-vm-install-local-v1'
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
    'git', 'show', source + ':tools/nemotron-diarization/macos-vm-install.protocol.json'], cwd=repo)
for name, want in p['tools'].items():
    assert descriptor(tools / name) == want
for name, want in p['external_files'].items():
    assert descriptor(name) == want
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_vm_restore_metrics import seed_metrics
from macos_vm_seed_clone import file_identity
from macos_vm_install_metrics import owner_metrics, installed_seed_metrics
from qualify_macos_vm_host_v4 import policy
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: want for name, want in p['tools'].items()})
prior = Path(p['restore_prerequisite']['receipt'])
manifest = read(prior / 'collection.json')
for name, want in manifest['files'].items():
    assert descriptor(prior / name) == want
pq = read(prior / 'qualification.json')
assert pq['no_boot_restore_passed'] is True and pq['source_commit'] == p['restore_prerequisite']['source_commit']
assert q['prior_restore_receipt_members_verified'] == len(manifest['files'])
origin = Path(p['restore_prerequisite']['local_root'])
assert seed_metrics(origin / 'vm-seed', pq['prepared'], pq['image'], uid=os.getuid()) == pq['seed']
assert {path.name: file_identity(path.lstat()) for path in (origin / 'vm-seed').iterdir()} == q['original_seed_identity']
assert q['fresh_machine_qualified'] is False and q['release_qualified'] is False
assert q['normal_guest_boot_started'] is False and q['private_app_or_models_executed'] is False
assert q['setup_networking_permitted'] is True
assert q['storage']['released'] is True and not (root / 'diagnostic-reserve.bin').exists()
assert q['storage']['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
offline = policy(p['denied_paths']) + '(deny file-write* (subpath ' + json.dumps(str(origin)) + '))\n'
assert (root / 'offline.sb').read_text() == offline
assert (root / 'setup.sb').read_text() == offline.replace('(deny network*)\n', '')
assert all('returncode' in command for command in q['commands'])
passed = q['installation_passed'] is True
if passed:
    assert 'error' not in q and q['installed_disk_quarantined'] is False and q['exclusive_lock_released'] is True
    names = ['offline-controls', 'setup-controls', 'build', 'sign', 'signature-verify', 'signature-metadata',
        'entitlements', 'architecture', 'deployment', 'dependencies', 'error-control', 'install-owner']
    assert [command['name'] for command in q['commands']] == names
    assert all(command['returncode'] == 0 for command in q['commands'])
    assert read(root / 'owner-result.json') == q['owner_result']
    events = [json.loads(line) for line in (root / 'install-owner.stdout.log').read_text().splitlines()]
    assert owner_metrics(q['owner_result'], q['commands'][-1], events) == q['owner_result']
    assert installed_seed_metrics(root / 'vm-seed', pq['seed'], uid=os.getuid()) == q['installed_seed']
    assert set(q['clones']) == set(pq['seed'])
    for name, clone in q['clones'].items():
        assert clone['operation'] == 'fclonefileat' and clone['returncode'] == 0 and clone['fallback_copy'] is False
        assert clone['source_before'] == clone['source_after'] == q['original_seed_identity'][name]
        assert clone['destination']['inode'] != clone['source_before']['inode']
        assert clone['destination']['bytes'] == clone['source_before']['bytes']
    assert descriptor(root / 'vm-install-owner') == q['owner_executable']
    assert plistlib.loads((root / 'entitlements.stdout.log').read_bytes()) == q['entitlements'] == {'com.apple.security.virtualization': True}
    signature = (root / 'signature-metadata.stderr.log').read_text()
    assert 'Signature=adhoc' in signature and re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature)
    assert (root / 'architecture.stdout.log').read_bytes().strip() == b'arm64'
    assert re.search(r'\bminos 26\.0\s', (root / 'deployment.stdout.log').read_text())
    assert read(root / 'error-control.stdout.log') == q['error_graph_control']
    assert q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
else:
    assert q['installed_disk_quarantined'] is True and 'error' in q
    # A failed/killed owner's disk stays unread and unmounted. Inspect saved report only.
    if (root / 'owner-result.json').is_file():
        assert read(root / 'owner-result.json') == q['owner_result']
image_pin = {key: p['restore_image'][key] for key in ['bytes', 'sha256']}
assert descriptor(origin / p['restore_image']['name']) == image_pin
test_log = cache / 'platform-ci/model-free-307-macos-vm-install-local.log'
assert 'Ran 307 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
out = repo / ('benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-install-' + ('stopped' if passed else 'failure'))
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
for path in [Path(__file__), cache / 'prepare-macos-vm-install-v1.py', root.with_suffix('.controller.log'),
        test_log, cache / 'macos-vm-install-tests-local.log']:
    shutil.copy2(path, out / path.name)
(out / 'sources').mkdir()
for name in p['tools']:
    shutil.copy2(tools / name, out / 'sources' / name)
summary = {'schema_version': 1, 'source_commit': source, 'installation_passed': passed,
    'fresh_machine_qualified': False, 'release_qualified': False,
    'actual_commands': len(q['commands']), 'zero_status_commands': sum(command['returncode'] == 0 for command in q['commands']),
    'original_image_independently_hashed': image_pin, 'original_seed_identity_unchanged': True,
    'prior_restore_members_verified': len(manifest['files']), 'current_historical_source_pins_verified': len(p['tools']),
    'external_descriptors_verified': len(p['external_files']), 'model_free_tests_passed': 307,
    'setup_networking_permitted': True, 'installed_disk_quarantined': q['installed_disk_quarantined'],
    'owner_result': q.get('owner_result'), 'installed_seed': q.get('installed_seed'),
    'physical_reserve_released': True, 'new_helper_vm_network_model_mount_clone_calls_during_collection': 0,
    'limits': ['Installation executes guest code; no subsequent normal guest boot/account/SSH/provisioning or private complete-app/model qualification.',
        'Only successful callback plus observed stopped state, supported post-callback stop if needed and successful owner exit can admit the installed seed.',
        'Owner kill/exit and lock release are not independent framework-teardown proof. Any failed/unconfirmed disk remains quarantined, never mounted/read/cloned in collection.',
        'Image/seed binaries and sparse disk/helper/modulecache stay local. Disk stat and OSclone/installer provenance only; never a full100GiB hash.',
        'Setup networking is permitted, not offline installation or a complete endpoint/Apple-helper sandbox claim. GUI/microphone/IPC/playback, quality/human gold, platform and public release gates remain open.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members', source)
