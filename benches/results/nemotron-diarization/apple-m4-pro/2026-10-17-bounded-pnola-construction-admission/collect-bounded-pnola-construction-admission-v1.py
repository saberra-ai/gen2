"""Retain artifact-only bounded app admission; no app/SDK/model execution."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-pnola-construction-admission-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
source = q['source_commit']
assert source == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert q['admit_only'] is True and q['deliveries_admitted'] is True
assert q['app_construction_passed'] is False and q['release_qualified'] is False
assert q['commands'] == [] and q['apps'] == {} and 'error' not in q
assert q['native_version_queries_or_neural_calls'] == 0 and q['models_loaded_for_inference'] is False
assert not any((root / n).exists() for n in ['host.json', 'native-package', 'pnola-source', 'tmp', 'cargo-target', 'cpu', 'diagnostic-reserve.bin'])
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-pnola-construction.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
for name, expected in pins.items():
    assert descriptor(repo / name) == expected
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
verify_git_sources(repo, source, pins)
verify_git_sources(repo / 'vendor/llama-cpp-sys-2/llama.cpp', p['vendor_revision'], p['vendor_sources'])
directory = root / 'producer'
members = {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()}
assert members == p['producer']['files'] and len(members) == 200
prior = cache / 'bounded-macos-construction-ci-v1'
assert descriptor(prior / 'artifact.zip') == p['producer']['zip']
api = q['producer_artifact']
assert api == read(prior / 'artifacts.json')['artifacts'][0]
assert api['id'] == p['producer']['id'] and api['name'] == p['producer']['name']
assert api['workflow_run']['id'] == p['producer']['run_id']
assert api['workflow_run']['head_sha'] == p['producer']['source_commit']
assert api['size_in_bytes'] == p['producer']['zip']['bytes'] and api['digest'] == 'sha256:' + p['producer']['zip']['sha256']
with zipfile.ZipFile(prior / 'artifact.zip') as bundle:
    assert len(bundle.infolist()) == len(members) and set(bundle.namelist()) == set(members)
    assert all(bundle.read(name) == (directory / name).read_bytes() for name in members)
construction = read(directory / 'qualification.json')
assert construction['construction_and_negative_c_api_controls_passed'] is True
assert construction['valid_models_loaded'] == construction['frontend_or_neural_calls'] == 0
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
for log, count in [('bounded-pnola-construction-tests-local.log', 2), ('platform-ci/model-free-277-bounded-pnola-construction-local.log', 277)]:
    text = (cache / log).read_text()
    assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-pnola-construction-admission'
assert not out.exists()
out.mkdir()
for path in [root / 'protocol.json', root / 'qualification.json', Path(__file__),
             cache / 'bounded-pnola-construction-admission-v1.controller.log',
             cache / 'bounded-pnola-construction-tests-local.log', cache / 'platform-ci/model-free-277-bounded-pnola-construction-local.log',
             cache / 'bounded-mixed-inputs-v1/producer-api.json']:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'source_commit': source, 'complete_producer_admitted': True,
    'delivery_members_verified': 200, 'historical_source_pins_verified': len(pins),
    'vendor_source_pins_verified': len(p['vendor_sources']), 'commands_started': 0,
    'package_or_source_or_dependencies_restored': False, 'app_build_or_signing_started': False,
    'native_version_queries_or_neural_calls': 0, 'model_free_tests_passed': 277,
    'storage_reserve_released': True, 'release_qualified': False,
    'limits': ['Actual full producer/source-only admission. No host probe, SDK restoration, private-source snapshot, npm clone, build, signing, app/SDK/model execution or release qualification.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.name: descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, len(files) + 1, 'members')
