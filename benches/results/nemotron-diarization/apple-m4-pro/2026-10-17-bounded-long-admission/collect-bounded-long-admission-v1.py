from pathlib import Path
import hashlib
import json
import shutil
import subprocess

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-long-admission-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


q, p = read(root / 'qualification.json'), read(root / 'protocol.json')
source = q['source_commit']
assert source == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert q['admit_only'] is True and q['deliveries_admitted'] is True
assert q['bounded_30m_1h_passed'] is False and q['release_qualified'] is False
assert q['commands'] == q['neural_commands_started'] == []
assert q['loaded_images'] == q['comparisons'] == {} and 'error' not in q
assert not any((root / name).exists() for name in [
    'work', 'home', 'tmp', 'native-helpers', 'public-helpers', 'results', 'diagnostic-reserve.bin'])
assert all(key not in q for key in ['acquisitions', 'helper_archive'])
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-long.protocol.json'], cwd=repo)
for key, folder, original in [
    ('producer', 'producer', 'bounded-macos-construction-ci-v1'),
    ('native', 'native-delivery', 'bounded-native-ci-v2'),
    ('public', 'public-delivery', 'bounded-public-ci-v1')]:
    directory = root / folder
    assert {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()} == p[key]['files']
    assert descriptor(cache / original / 'artifact.zip') == p[key]['zip']
    api = q[key + '_artifact']
    assert api == read(cache / original / 'artifacts.json')['artifacts'][0]
    assert api['id'] == p[key]['id'] and api['name'] == p[key]['name']
    assert api['workflow_run']['id'] == p[key]['run_id']
    assert api['workflow_run']['head_sha'] == p[key]['source_commit']
    assert api['digest'] == 'sha256:' + p[key]['zip']['sha256']
native = read(root / 'native-delivery/qualification.json')
public = read(root / 'public-delivery/qualification.json')
assert native['deliveries']['producer'] == public['producer_artifact'] == q['producer_artifact']
assert public['native_reference_artifact'] == q['native_artifact']
assert native['native_parity_passed'] is True and native['realtime_gate_passed'] is True
assert public['bounded_public_runtime_passed'] is True and public['public_realtime_gate_passed'] is True
assert public['lifecycle_tests_passed'] == p['lifecycle_tests']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
for log, count in [('bounded-long-tests-local.log', 6), ('platform-ci/model-free-274-bounded-long-local.log', 274)]:
    text = (cache / log).read_text()
    assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-long-admission'
assert not out.exists()
out.mkdir()
for path in [root / 'protocol.json', root / 'qualification.json',
             cache / 'bounded-long-admission-v1.controller.log', Path(__file__),
             cache / 'bounded-long-tests-local.log', cache / 'platform-ci/model-free-274-bounded-long-local.log',
             *sorted((cache / 'bounded-long-inputs-v1').iterdir())]:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'source_commit': source,
           'all_three_full_deliveries_admitted': True,
           'members_verified': {'producer': 200, 'native': 44, 'public': 60},
           'package_or_helpers_restored': False, 'assets_copied': 0,
           'helper_or_model_execution': 0, 'commands_started': 0,
           'all_original_api_identity_and_qualified_origin_contracts_verified': True,
           'storage_reserve_released': True, 'model_free_python_tests_passed': 274,
           'release_qualified': False,
           'limits': ['Actual artifact/source-only local preflight. No package/helper restoration, asset copying, version probe or neural execution. Four fresh bounded long streams and all remaining roadmap quality/app/release gates stay pending.']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {str(path.relative_to(out)): descriptor(path) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members')
