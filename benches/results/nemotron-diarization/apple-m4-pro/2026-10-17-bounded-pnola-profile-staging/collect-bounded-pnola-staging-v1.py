"""Retain actual static Pnola staging evidence without app/SDK/model calls."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
run = cache / 'bounded-pnola-staging-test-v1'
staged = cache / 'bounded-pnola-staging-local-v1/.desktop-assets'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(cache / 'bounded-pnola-staging-inputs-v1.json'), read(run / 'qualification.json')
source = '00b067f68472de8f033690bae999e9734bc5af79'
assert p['source_commit'] == q['source_commit'] == source
for name, expected in p['sources'].items():
    assert descriptor(repo / name) == expected
    raw = subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo)
    assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from sdk_archive import inventory

assert q['actual_staging_passed'] is True and q['returncode'] == 0 and q['release_qualified'] is False
assert q['app_or_native_sdk_calls'] == q['model_inference_calls'] == 0
assert q['source_inputs'] == descriptor(cache / 'bounded-pnola-staging-inputs-v1.json')
assert q['launcher'] == descriptor(cache / 'stage-bounded-pnola-local-v1.mjs')
assert q['node_binary'] == descriptor(Path('/opt/homebrew/bin/node').resolve())
assert inventory(staged / 'nemo') == q['staged_package_inventory'] == p['source_package']
assert inventory(cache / 'bounded-mixed-local-v1/work/nemo') == p['source_package']
assert inventory(cache / 'bounded-mixed-local-v1/work/models/nemotron') == p['source_models']
assert inventory(staged / 'nemotron') == q['staged_model_inventory']
private = Path('/Users/victor/workspace/pnola')
assert {n: descriptor(private / n) for n in p['private_pnola_source_files_before']} == p['private_pnola_source_files_before']
assert q['private_pnola_source_files_unchanged'] == len(p['private_pnola_source_files_before']) == 107
lock = read(repo / 'tools/nemotron-diarization/runtime-valid-mel-pool-cpu.lock.json')
license_root = repo / 'tools/nemotron-diarization/notices/openmdw-1.1'
assert descriptor(staged / 'nemotron/LICENSE.OpenMDW-1.1') == descriptor(license_root / 'LICENSE.OpenMDW-1.1')
assert read(staged / 'nemotron/bundle.json') == {'model': lock['model'], 'runtime_profile': lock['profile'],
    'frame_contract': lock['runtime']['frame_contract'], 'files': [e for e in lock['artifacts'] if e['role'] in {'model', 'notice'}],
    'license': read(license_root / 'sources.json')}
for entry in lock['artifacts']:
    if entry['role'] in {'model', 'notice'}:
        assert descriptor(staged / 'nemotron' / entry['name']) == {'bytes': entry['size'], 'sha256': entry['sha256']}
assert set(path.name for path in (staged / 'nemotron').iterdir()) == {
    'Nemotron-3-Diarization.q8_0.gguf', 'README.md', 'LICENSE.OpenMDW-1.1', 'bundle.json'}
assert (run / 'staging.stdout.log').read_text().strip() == 'Actual bounded Pnola staging and bundle/lock provenance passed.'
assert not (run / 'staging.stderr.log').read_bytes()
assert 'Unsupported native runtime profile: gen2-valid-mel-pool-cpu-v2' in (cache / 'bounded-pnola-profile-red.log').read_text()
assert 'overlay checksum mismatch' in (cache / 'bounded-pnola-overlay-red.log').read_text()
assert 'pass 15' in (cache / 'bounded-pnola-profile-green.log').read_text()
assert 'fail 0' in (cache / 'bounded-pnola-profile-green.log').read_text()
assert 'Ran 3 tests' in (cache / 'bounded-pnola-overlay-green.log').read_text()
assert (cache / 'bounded-pnola-overlay-green.log').read_text().rstrip().endswith('OK')
assert read(cache / 'bounded-pnola-overlay-private-check.log') == {'valid': True, 'files_to_update': 2}
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-pnola-profile-staging'
assert not out.exists()
out.mkdir()
files = [run / 'qualification.json', cache / 'bounded-pnola-staging-inputs-v1.json',
    staged.parent / 'result.json', run / 'staging.stdout.log', run / 'staging.stderr.log', run / 'static-staging.sb',
    cache / 'stage-bounded-pnola-local-v1.mjs', cache / 'run-bounded-pnola-staging-v1.py', Path(__file__),
    *[cache / name for name in ['bounded-pnola-profile-red.log', 'bounded-pnola-profile-green.log',
        'bounded-pnola-profile-real-lock.log', 'bounded-pnola-overlay-red.log', 'bounded-pnola-overlay-green.log',
        'bounded-pnola-overlay-private-check.log']]]
for path in files:
    shutil.copy2(path, out / path.name)
record = {'schema_version': 1, 'source_commit': source, 'actual_staging_passed': True,
    'runtime_profile': lock['profile'], 'historical_and_current_sources_verified': len(p['sources']),
    'staged_package_entries': len(q['staged_package_inventory']), 'staged_model_entries': len(q['staged_model_inventory']),
    'private_pnola_source_files_unchanged': 107, 'node_tests_passed': 15, 'overlay_tests_passed': 3,
    'model_payloads_retained_in_git': False, 'new_app_or_native_or_model_calls_during_collection': 0,
    'release_qualified': False, 'limits': q['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
members = {path.name: descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': members}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in members.items())
print(out, len(members) + 1, 'members', record)
