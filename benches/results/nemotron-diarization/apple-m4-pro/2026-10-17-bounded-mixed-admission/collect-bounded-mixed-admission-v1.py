"""Retain actual bounded mixed delivery admission, without native execution."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-mixed-admission-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


q, p = read(root / 'qualification.json'), read(root / 'protocol.json')
source = q['source_commit']
assert source == '29294bd1b078447fedb11cf695f66b8d8ddd6c12'
assert q['admit_only'] is True and q['deliveries_admitted'] is True
assert q['bounded_cpu_mixed_residency_passed'] is False and q['release_qualified'] is False
assert q['commands'] == q['neural_commands_started'] == q['cases'] == []
assert q['loaded_images'] == {} and 'error' not in q
assert not any((root / name).exists() for name in [
    'work', 'home', 'tmp', 'public-helpers', 'results', 'cargo-target',
    'long-prerequisite.json', 'build-inputs.json', 'diagnostic-reserve.bin'])
assert all(key not in q for key in ['helper_archive', 'long_prerequisite'])
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-mixed.protocol.json'], cwd=repo)
for key, folder, original in [
    ('producer', 'producer', 'bounded-macos-construction-ci-v1'),
    ('public', 'public-delivery', 'bounded-public-ci-v1')]:
    directory = root / folder
    members = {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()}
    assert members == p[key]['files']
    prior = cache / original
    assert descriptor(prior / 'artifact.zip') == p[key]['zip']
    api = q[key + '_artifact']
    assert api == read(prior / 'artifacts.json')['artifacts'][0]
    assert api['id'] == p[key]['id'] and api['name'] == p[key]['name']
    assert api['workflow_run']['id'] == p[key]['run_id']
    assert api['workflow_run']['head_sha'] == p[key]['source_commit']
    assert api['size_in_bytes'] == p[key]['zip']['bytes']
    assert api['digest'] == 'sha256:' + p[key]['zip']['sha256']
    with zipfile.ZipFile(prior / 'artifact.zip') as bundle:
        assert len(bundle.infolist()) == len(members) and set(bundle.namelist()) == set(members)
        assert all(bundle.read(name) == (directory / name).read_bytes() for name in members)
construction = read(root / 'producer/qualification.json')
public = read(root / 'public-delivery/qualification.json')
assert construction['construction_and_negative_c_api_controls_passed'] is True
assert construction['valid_models_loaded'] == construction['frontend_or_neural_calls'] == 0
assert public['producer_artifact'] == q['producer_artifact']
assert public['native_reference_artifact'] == p['native_origin']
assert public['bounded_public_runtime_passed'] is True and public['public_realtime_gate_passed'] is True
assert public['lifecycle_tests_passed'] == p['lifecycle_tests']
assert all(c['returncode'] == c['expected_returncode'] for c in public['commands'])
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
for log, count in [('bounded-mixed-protocol-local.log', 1), ('platform-ci/model-free-275-bounded-mixed-local.log', 275)]:
    text = (cache / log).read_text()
    assert f'Ran {count} test' in text and text.rstrip().endswith('OK')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-mixed-admission'
assert not out.exists()
out.mkdir()
for path in [root / 'protocol.json', root / 'qualification.json',
             cache / 'bounded-mixed-admission-v1.controller.log', Path(__file__),
             cache / 'bounded-mixed-protocol-local.log', cache / 'platform-ci/model-free-275-bounded-mixed-local.log',
             *sorted((cache / 'bounded-mixed-inputs-v1').iterdir())]:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'source_commit': source,
    'both_full_deliveries_admitted': True, 'members_verified': {'producer': 200, 'public': 60},
    'package_or_helpers_restored': False, 'assets_copied': 0,
    'helper_or_model_execution': 0, 'commands_started': 0,
    'complete_api_identity_and_qualified_origin_contracts_verified': True,
    'storage_reserve_released': True, 'model_free_python_tests_passed': 275,
    'release_qualified': False,
    'limits': ['Actual artifact/source-only local preflight. No long-record copy, host/package/helper restoration, asset copying, build, probe or model execution. Actual mixed residency and all remaining quality/app/release gates stay pending.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members')
