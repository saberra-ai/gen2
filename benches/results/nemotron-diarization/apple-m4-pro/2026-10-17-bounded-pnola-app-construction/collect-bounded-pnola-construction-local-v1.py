"""Replay complete bounded app construction using metadata inspection only."""
from pathlib import Path
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-pnola-construction-local-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
source = 'edf7b898b00df89d868761bfc80487d7c3bdf097'
assert q['source_commit'] == source and q['app_construction_passed'] is True
assert q['deliveries_admitted'] is True and q['admit_only'] is False and q['release_qualified'] is False
assert q['native_version_queries_or_neural_calls'] == 0 and q['models_loaded_for_inference'] is False and 'error' not in q
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-pnola-construction.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
for name, value in pins.items():
    assert descriptor(repo / name) == value
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from package_macos import verify
from sdk_archive import inventory
from verify_bounded_pnola_app import verify_app

verify_git_sources(repo, source, pins)
vendor = repo / 'vendor/llama-cpp-sys-2/llama.cpp'
assert subprocess.check_output(['git', 'rev-parse', source + ':vendor/llama-cpp-sys-2/llama.cpp'], cwd=repo, text=True).strip() == p['vendor_revision']
verify_git_sources(vendor, p['vendor_revision'], p['vendor_sources'])
for n, v in p['vendor_sources'].items():
    assert descriptor(vendor / n) == v
private = Path(p['private_pnola'])
for name, value in p['private_source_files'].items():
    assert descriptor(private / name) == value
assert inventory(private / 'node_modules') == p['node_modules_inventory']
for name, value in p['asset_directories'].items():
    assert inventory(Path(name)) == value
for name, value in p['asset_files'].items():
    assert descriptor(Path(name)) == value
for value in p['build_tools'].values():
    assert descriptor(Path(value['path'])) == value['file']
for role, flag in [('long', 'bounded_30m_1h_passed'), ('mixed', 'bounded_cpu_mixed_residency_passed')]:
    value = p['prior_qualifications'][role]
    assert descriptor(Path(value['path'])) == value['file']
    prior_q = read(value['path'])
    assert prior_q[flag] is True and prior_q['source_commit'] == value['source_commit'] and prior_q['storage']['released']
expected = ['node-dependencies-clone', 'overlay-check', 'overlay-apply', 'cpu-build', 'cpu-clone', 'cpu-signature']
assert [c['name'] for c in q['commands']] == expected
assert all(c['returncode'] == 0 and c['inference_admitted'] is False for c in q['commands'])
assert all(c['argv'][:2] == ['/usr/bin/sandbox-exec', '-f'] for c in q['commands'])
assert all(not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes') for c in q['commands'])
for c in q['commands']:
    env = c['environment']
    assert all(env[n] == v for n, v in p['build_environment'].items())
    assert not any(n.startswith('APPLE_') for n in env)
    assert env['PNOLA_SIGNING_IDENTITY'] == '-'
    assert env['CARGO_TARGET_DIR'] == str(root / 'cargo-target')
    assert env['GEN2_ROOT'] == str(repo) and env['GEN2_NEMO_PACKAGE'] == str(root / 'native-package')
producer = root / 'producer'
members = {str(path.relative_to(producer)): descriptor(path) for path in producer.rglob('*') if path.is_file()}
assert members == p['producer']['files'] and len(members) == 200
delivery = cache / 'bounded-macos-construction-ci-v1'
assert descriptor(delivery / 'artifact.zip') == p['producer']['zip']
api = q['producer_artifact']
assert api == read(delivery / 'artifacts.json')['artifacts'][0]
assert api['id'] == p['producer']['id'] and api['name'] == p['producer']['name']
assert api['workflow_run']['id'] == p['producer']['run_id'] and api['workflow_run']['head_sha'] == p['producer']['source_commit']
assert api['size_in_bytes'] == p['producer']['zip']['bytes'] and api['digest'] == 'sha256:' + p['producer']['zip']['sha256']
with zipfile.ZipFile(delivery / 'artifact.zip') as bundle:
    assert len(bundle.infolist()) == len(members) and set(bundle.namelist()) == set(members)
    assert all(bundle.read(n) == (producer / n).read_bytes() for n in members)
construction = read(producer / 'qualification.json')
assert construction['construction_and_negative_c_api_controls_passed'] is True
assert construction['valid_models_loaded'] == construction['frontend_or_neural_calls'] == 0
archive = read(producer / 'pooled-cpu-runtime-files.json')
assert {k: v for k, v in archive.items() if k != 'files'} == construction['package_archive']
assert inventory(root / 'native-package') == archive['files']
manifest = verify(root / 'native-package')
assert manifest['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2' and manifest['backend'] == 'cpu'
snapshot = root / 'pnola-source'
rendered = read(root / 'rendered-source-files.json')
assert set(rendered) == set(p['private_source_files'])
assert {n: descriptor(snapshot / n) for n in rendered} == rendered
assert {n for n in rendered if rendered[n] != p['private_source_files'][n]} == set(p['expected_overlay_changes'])
for name in ['scripts/speaker-assets.mjs', 'tests/desktop-assets.test.mjs']:
    assert (snapshot / name).read_bytes() == (repo / 'integrations/pnola/overlay' / name).read_bytes()
cargo = (repo / 'integrations/pnola/overlay/src-tauri/Cargo.toml').read_bytes().replace(
    b'"@GEN2_ROOT@"', json.dumps(Path(os.path.relpath(repo, snapshot / 'src-tauri')).as_posix()).encode())
assert (snapshot / 'src-tauri/Cargo.toml').read_bytes() == cargo
node_after = inventory(snapshot / 'node_modules')
assert all(node_after.get(n) == v for n, v in p['node_modules_inventory'].items())
assert read(root / 'host.json')['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
assert set(q['apps']) == {'cpu'}
app = Path(q['apps']['cpu']['path'])
assert app == root / 'cpu/Pnola.app'
assert inventory(app) == q['apps']['cpu']['files']
assert verify_app(app) == read(root / 'cpu-static-verification.json')
assert descriptor(root / 'cpu-static-verification.json') == q['apps']['cpu']['static_verification']
assert verify(app / 'Contents/Resources/nemo') == manifest
assert q['apps']['cpu']['ad_hoc_hardened_signature_observed'] is True and q['apps']['cpu']['local_entitlements_unchanged'] is True
signing = (root / 'cpu-signing-details.log').read_text()
assert 'Signature=adhoc' in signing and re.search(r'flags=0x[0-9a-f]+\(.*runtime.*\)', signing)
assert plistlib.loads((root / 'cpu-entitlements.plist').read_bytes()) == plistlib.loads((snapshot / 'src-tauri/Entitlements.local.plist').read_bytes())
assert inventory(app) == q['apps']['cpu']['files']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert not (root / 'diagnostic-reserve.bin').exists()
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-pnola-app-construction'
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb', '.plist'}:
        shutil.copy2(path, out / path.name)
shutil.copy2(__file__, out / Path(__file__).name)
summary = {'schema_version': 1, 'source_commit': source, 'app_construction_passed': True,
    'historical_source_pins_verified': len(pins), 'vendor_source_pins_verified': len(p['vendor_sources']),
    'private_source_files_unchanged': len(p['private_source_files']),
    'original_npm_entries_unchanged': len(p['node_modules_inventory']),
    'all_original_assets_and_earlier_apps_unchanged': True,
    'isolated_source_delta': p['expected_overlay_changes'], 'six_commands_returned_zero': True,
    'complete_producer_delivery_and_package_archive_verified': True,
    'ad_hoc_hardened_signature_and_original_entitlements_verified': True,
    'static_frameworks_verified': len(read(root / 'cpu-static-verification.json')['frameworks']),
    'app_entries_verified': len(q['apps']['cpu']['files']),
    'new_app_sdk_or_model_calls_during_collection': 0, 'release_qualified': False, 'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.name: descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, len(files) + 1, 'members', summary)
