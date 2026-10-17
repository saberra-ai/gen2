"""Saved-data-only independent collection of pinned IPSW/no-boot seed evidence."""
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
root = cache / 'macos-vm-restore-local-v1'
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
assert q['no_boot_restore_passed'] is True and 'error' not in q
assert all(q[name] is False for name in ['fresh_machine_qualified', 'release_qualified',
    'virtual_machine_created', 'installation_started', 'guest_booted', 'models_or_applications_executed'])
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/macos-vm-restore.protocol.json'], cwd=repo)
for name, want in p['tools'].items():
    assert descriptor(tools / name) == want
for name, want in p['external_files'].items():
    assert descriptor(name) == want
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_vm_restore_metrics import image_metrics, seed_metrics
from qualify_macos_vm_host_v4 import policy
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: want for name, want in p['tools'].items()})
prior = Path(p['host_prerequisite']['root'])
manifest = read(prior / 'collection.json')
for name, want in manifest['files'].items():
    assert descriptor(prior / name) == want
assert q['prior_host_receipt_members_verified'] == len(manifest['files'])
assert read(prior / 'qualification.json')['source_commit'] == p['host_prerequisite']['source_commit']
assert read(prior / 'collection-qualification.json')['host_prerequisites_passed'] is True
names = ['offline-controls', 'build', 'sign', 'signature-verify', 'signature-metadata', 'entitlements',
    'architecture', 'deployment', 'dependencies', 'download', 'inspect', 'prepare']
assert [command['name'] for command in q['commands']] == names
assert all(command['returncode'] == 0 and not command.get('forced_process_group_termination') for command in q['commands'])
for command in q['commands']:
    assert (command['argv'][0] == p['python']) if command['name'] == 'download' else (
        command['argv'][:3] == ['/usr/bin/sandbox-exec', '-f', str(root / 'offline.sb')])
assert q['download'] == read(root / 'download.json')
assert q['download']['complete'] is True and 'error' not in q['download']
image = root / p['restore_image']['name']
expected = {key: p['restore_image'][key] for key in ['bytes', 'sha256']}
assert q['download']['actual'] == q['download']['expected'] == expected
# Full independent hash of the acquired body; deliberately never hash the 100GiB sparse disk.
assert descriptor(image) == expected
assert image.stat().st_nlink == 1 and not image.with_name(image.name + '.partial').exists()
assert q['download']['bytes_received'] == q['download']['bytes_written'] == expected['bytes']
assert q['download']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert image_metrics(read(root / 'inspect.stdout.log'), operation='inspect') == q['image']
assert read(root / 'prepare.stdout.log') == q['prepared']
assert seed_metrics(root / 'vm-seed', q['prepared'], q['image'], uid=os.getuid()) == q['seed']
assert descriptor(root / 'vm-restore-probe') == q['probe_executable']
assert plistlib.loads((root / 'entitlements.stdout.log').read_bytes()) == q['entitlements'] == {'com.apple.security.virtualization': True}
signature = (root / 'signature-metadata.stderr.log').read_text()
assert 'Signature=adhoc' in signature and re.search(r'flags=0x[0-9a-f]+\([^)]*runtime', signature)
assert (root / 'architecture.stdout.log').read_bytes().strip() == b'arm64'
assert re.search(r'\bminos 26\.0\s', (root / 'deployment.stdout.log').read_text())
assert (root / 'offline.sb').read_text() == policy(p['denied_paths'])
assert q['storage']['released'] is True and not (root / 'diagnostic-reserve.bin').exists()
assert q['storage']['reserve_reported_allocated_bytes'] == p['storage']['reserve_bytes']
assert q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
test_log = cache / 'platform-ci/model-free-297-macos-vm-restore-local.log'
assert 'Ran 297 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-restore-no-boot'
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
for path in [Path(__file__), cache / 'prepare-macos-vm-restore-v1.py',
        root.with_suffix('.controller.log'), test_log, cache / 'macos-restore-tests-local.log']:
    shutil.copy2(path, out / path.name)
(out / 'sources').mkdir()
for name in p['tools']:
    shutil.copy2(tools / name, out / 'sources' / name)
summary = {'schema_version': 1, 'source_commit': source, 'no_boot_restore_passed': True,
    'fresh_machine_qualified': False, 'release_qualified': False, 'actual_commands': len(names),
    'zero_status_commands': len(names), 'complete_apple_restore_body_independently_hashed': expected,
    'actual_image_support_version_model_requirements_verified': True,
    'actual_no_boot_configuration_validation_passed': True, 'seed': q['seed'],
    'geometry': {key: q['prepared'][key] for key in ['cpu_count', 'memory_bytes', 'disk_logical_bytes', 'network_attachment']},
    'one_entitlement_adhoc_hardened_helper': q['probe_executable'], 'prior_host_receipt_members_verified': len(manifest['files']),
    'current_historical_source_pins_verified': len(p['tools']), 'external_descriptors_verified': len(p['external_files']),
    'model_free_tests_passed': 297, 'reserve_released': True,
    'new_compiler_helper_network_model_mount_vm_calls_during_collection': 0,
    'limits': ['No VM instance, installer/installation, boot, guest provisioning/SSH, app/model, GUI/microphone or fresh-machine qualification.',
        'No complete SDK closure or module-cache hash. Selected source/tool/header inputs only. No public signing/release or quality/human-gold acceptance.',
        'OS body, seed binaries and sparse disk, executable and module cache stay local. Sparse disk stat/provenance only; never read/hash whole disk.',
        'Prepared NAT is not running networking or proof of offline installation. Immutable seed requires separately reviewed clone/install lifecycle.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members', source)
