"""Replay four saved bounded streams without executing a helper or model."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'bounded-long-local-v1'
source = 'ab090d275fa5cd9c833d748a3682d43b426ea52c'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
assert q['source_commit'] == source and q['bounded_30m_1h_passed'] is True
assert q['deliveries_admitted'] is True and q['admit_only'] is False
assert q['release_qualified'] is False and 'error' not in q
assert descriptor(root / 'protocol.json') == q['protocol'] == {
    'bytes': 115261, 'sha256': '1b6a863c31fe2201315e0cf054313d172f78239ddfc412b774e40f63734a7c6d'}
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-long.protocol.json'], cwd=repo)
for key, base in [('tools', tools), ('repository_files', repo)]:
    for name, expected in p[key].items():
        assert descriptor(base / name) == expected, name
sys.path.insert(0, str(tools))
from bounded_long_metrics import audit_case
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files, verify_package_provenance
from sdk_archive import inventory, restore_archive

pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
old = read(tools / 'valid-mel-cpu-long.protocol.json')
for key in ['acceptance', 'cases', 'command_timeout_seconds']:
    assert p[key] == old[key]
host = read(root / 'host.json')
assert host['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
expected = ['model-free-controls', 'runtime-controls', 'version-only',
            '30m-native', '30m-public', '60m-native', '60m-public']
assert [c['name'] for c in q['commands']] == expected
assert all(c['returncode'] == 0 and c['inference_admitted'] == (c['name'] in expected[3:]) for c in q['commands'])
assert all(c['argv'][:2] == ['/usr/bin/sandbox-exec', '-f'] for c in q['commands'])
assert all(not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes') for c in q['commands'])
assert q['neural_commands_started'] == expected[3:]
protected = read(root / 'protected-inputs.json')
for key, directory in [('package', 'work/nemo'), ('binaries', 'work/bin'), ('assets', 'work/assets'),
                       ('producer', 'producer'), ('native_delivery', 'native-delivery'),
                       ('public_delivery', 'public-delivery'), ('native_helpers', 'native-helpers'),
                       ('public_helpers', 'public-helpers')]:
    assert inventory(root / directory) == protected[key], key
for key, folder, original in [
    ('producer', 'producer', 'bounded-macos-construction-ci-v1'),
    ('native', 'native-delivery', 'bounded-native-ci-v2'),
    ('public', 'public-delivery', 'bounded-public-ci-v1')]:
    directory = root / folder
    before = {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()}
    assert before == p[key]['files']
    prior = cache / original
    assert descriptor(prior / 'artifact.zip') == p[key]['zip']
    api = q[key + '_artifact']
    assert api == read(prior / 'artifacts.json')['artifacts'][0]
    assert api['id'] == p[key]['id'] and api['name'] == p[key]['name']
    assert api['workflow_run']['id'] == p[key]['run_id']
    assert api['workflow_run']['head_sha'] == p[key]['source_commit']
    assert api['size_in_bytes'] == p[key]['zip']['bytes'] and api['digest'] == 'sha256:' + p[key]['zip']['sha256']
    with zipfile.ZipFile(prior / 'artifact.zip') as bundle:
        assert len(bundle.infolist()) == len(before) and set(bundle.namelist()) == set(before)
        assert all(bundle.read(name) == (directory / name).read_bytes() for name in before)
construction = read(root / 'producer/qualification.json')
native_q = read(root / 'native-delivery/qualification.json')
public_q = read(root / 'public-delivery/qualification.json')
assert construction['construction_and_negative_c_api_controls_passed'] is True
assert construction['valid_models_loaded'] == construction['frontend_or_neural_calls'] == 0
assert native_q['native_parity_passed'] is True and native_q['realtime_gate_passed'] is True
assert public_q['bounded_public_runtime_passed'] is True and public_q['public_realtime_gate_passed'] is True
assert public_q['lifecycle_tests_passed'] == p['lifecycle_tests']
assert native_q['deliveries']['producer'] == public_q['producer_artifact'] == q['producer_artifact']
assert public_q['native_reference_artifact'] == q['native_artifact']
helpers = read(root / 'bounded-long-helpers-files.json')
assert inventory(root / 'work/bin') == helpers['files']
assert {k: v for k, v in helpers.items() if k != 'files'} == q['helper_archive']
assert not (root / 'independent-helper-restore').exists()
restore_archive(root / helpers['archive'], helpers, root / 'independent-helper-restore')
assert inventory(root / 'independent-helper-restore') == helpers['files']
assert descriptor(root / 'work/bin/native-stream') == descriptor(root / 'native-helpers/native')
for name in ['public-stream', 'load-probe']:
    assert descriptor(root / 'work/bin' / name) == descriptor(root / 'public-helpers' / name)
for directory, name, summary in [
    ('native-delivery', 'bounded-native-helpers-files.json', native_q['helper_archive']),
    ('public-delivery', 'pooled-public-helpers-files.json', public_q['helper_archive'])]:
    archive = read(root / directory / name)
    assert {k: v for k, v in archive.items() if k != 'files'} == summary
    actual_helpers = root / ('native-helpers' if directory == 'native-delivery' else 'public-helpers')
    assert inventory(actual_helpers) == archive['files']
package = root / 'work/nemo'
package_archive = read(root / 'producer/pooled-cpu-runtime-files.json')
assert {k: v for k, v in package_archive.items() if k != 'files'} == construction['package_archive']
assert inventory(package) == package_archive['files']
manifest = read(package / 'package.json')
verify_files(package, manifest['files'])
verify_package_provenance(package, manifest)
assert read(root / 'package.json') == manifest
assert manifest['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2'
assert manifest['runtime_reported_version'] == 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2'
assert set(q['loaded_images']) == set(expected[2:])
for name, original in q['loaded_images'].items():
    executable = root / 'work/bin' / ('load-probe' if name == 'version-only' else
        'native-stream' if name.endswith('-native') else 'public-stream')
    assert audit_loaded_libraries((root / (name + '.stderr.log')).read_text(), package, manifest, executable) == original
    assert original['private_library_count'] == 92
assets = {entry['name']: {'bytes': entry['size'], 'sha256': entry['sha256']} for entry in p['assets']}
assert q['acquisitions'] == assets
assert all(descriptor(root / 'work/assets' / name) == value for name, value in assets.items())
results_before = {path.name: descriptor(path) for path in (root / 'results').iterdir() if path.is_file()}
assert not any(path.is_symlink() for path in (root / 'results').rglob('*'))
expected_results = {'prior-output-control.txt'}
for case in p['cases']:
    for suffix in ['-native.json', '-native.json.f32', '-public.json', '-public.f32']:
        expected_results.add(case['name'] + suffix)
assert set(results_before) == expected_results
comparisons, usage = {}, {}
for case in p['cases']:
    name = case['name']
    native_path = root / 'results' / (name + '-native.json')
    public_path = root / 'results' / (name + '-public.json')
    native, public = read(native_path), read(public_path)
    replay = audit_case(public, native, Path(str(native_path) + '.f32').read_bytes(),
        (root / 'results' / (name + '-public.f32')).read_bytes(),
        (root / (name + '-native.stderr.log')).read_text(),
        (root / (name + '-public.stderr.log')).read_text(), case)
    assert replay == q['comparisons'][name]
    assert replay['native_frame_count'] == case['final_rows']
    assert replay['retained_first_frame'] > 0 and 60000 <= replay['retained_rows'] <= 130000
    comparisons[name] = replay
    for kind in ['native', 'public']:
        key = name + '-' + kind
        text = (root / (key + '.stderr.log')).read_text()
        times = re.findall(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$', text, re.M)
        assert len(times) == 1
        wall, user, system = map(float, times[0])
        usage[key] = {'real_seconds': wall, 'user_seconds': user, 'system_seconds': system,
                      'peak_rss_bytes': replay['peak_rss_bytes'][kind]}
assert results_before == {path.name: descriptor(path) for path in (root / 'results').iterdir() if path.is_file()}
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert not (root / 'diagnostic-reserve.bin').exists()
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-long'
assert not out.exists()
out.mkdir()


def retain(path, relative):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    assert descriptor(path) == descriptor(destination)


for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        retain(path, path.name)
for path in (root / 'results').rglob('*'):
    if path.is_file():
        retain(path, path.relative_to(root))
retain(cache / 'platform-ci/model-free-274-bounded-long-local.log', 'model-free-274.log')
retain(cache / 'bounded-long-tests-local.log', 'metrics-protocol-preflight.log')
retain(Path(__file__), Path(__file__).name)
record = {'schema_version': 1, 'source_commit': source, 'bounded_30m_1h_passed': True,
    'historical_source_pins_verified': len(pins), 'four_fresh_neural_processes': expected[3:],
    'five_strict_private_library_audits': 92, 'protected_input_inventories_unchanged': True,
    'qualified_helper_payloads_independently_restored_and_rehashed': True,
    'all_three_opaque_artifact_deliveries_verified': True,
    'all_raw_and_json_result_bytes_bound_before_and_after': results_before,
    'new_collection_native_helper_or_model_calls': 0,
    'comparisons': comparisons, 'time_l': usage, 'release_qualified': False, 'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, 'members', len(files) + 1, 'comparisons', comparisons)
