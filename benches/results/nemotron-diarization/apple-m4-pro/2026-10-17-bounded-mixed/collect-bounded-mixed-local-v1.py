"""Replay saved bounded mixed evidence; never execute a helper or model."""
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
root = cache / 'bounded-mixed-local-v1'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


p, q = read(root / 'protocol.json'), read(root / 'qualification.json')
source = q['source_commit']
assert q['bounded_cpu_mixed_residency_passed'] is True
assert q['deliveries_admitted'] is True and q['admit_only'] is False
assert q['release_qualified'] is False and 'error' not in q
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-mixed.protocol.json'], cwd=repo)
for group, base in [('tools', tools), ('repository_files', repo),
                    ('vendor_sources', repo / 'vendor/llama-cpp-sys-2/llama.cpp')]:
    for name, expected in p[group].items():
        assert descriptor(base / name) == expected, name
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_auxiliary_loaded_libraries import audit_with_auxiliary
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files, verify_package_provenance
from sdk_archive import inventory, restore_archive

pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
vendor = repo / 'vendor/llama-cpp-sys-2/llama.cpp'
assert p['vendor_revision'] == subprocess.check_output(
    ['git', 'rev-parse', source + ':vendor/llama-cpp-sys-2/llama.cpp'], cwd=repo, text=True).strip()
verify_git_sources(vendor, p['vendor_revision'], p['vendor_sources'])
original = read(tools / 'valid-mel-mixed.protocol.json')
assert p['tests'] == [case['test'] for case in original['cases'] if case['backend'] == 'cpu']
assert p['compiled_test_names'] == original['compiled_test_names']
assert p['model_and_provider_files'] == {n: v for n, v in original['files'].items() if n.startswith(('models/', 'providers/'))}
assert p['command_timeout_seconds'] == 900
expected = ['compile-controls', 'mel-build-input-readable', 'rustc-version', 'mixed-helper-build',
            'runtime-controls', 'premodel-controls', 'version-only', 'list', *p['tests']]
assert [c['name'] for c in q['commands']] == expected
assert all(c['returncode'] == 0 and c['inference_admitted'] == (c['name'] in p['tests']) for c in q['commands'])
assert all(c['argv'][:2] == ['/usr/bin/sandbox-exec', '-f'] for c in q['commands'])
assert all(not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes') for c in q['commands'])
assert q['neural_commands_started'] == p['tests']
assert read(root / 'host.json')['cpu'] == p['expected_cpu'] == 'Apple M4 Pro'
for key, folder, original_delivery in [
    ('producer', 'producer', 'bounded-macos-construction-ci-v1'),
    ('public', 'public-delivery', 'bounded-public-ci-v1')]:
    directory = root / folder
    members = {str(path.relative_to(directory)): descriptor(path) for path in directory.rglob('*') if path.is_file()}
    assert members == p[key]['files']
    prior = cache / original_delivery
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
public_q = read(root / 'public-delivery/qualification.json')
assert construction['construction_and_negative_c_api_controls_passed'] is True
assert construction['valid_models_loaded'] == construction['frontend_or_neural_calls'] == 0
assert public_q['bounded_public_runtime_passed'] is True and public_q['public_realtime_gate_passed'] is True
assert public_q['lifecycle_tests_passed'] == p['lifecycle_tests']
assert public_q['producer_artifact'] == q['producer_artifact']
assert public_q['native_reference_artifact'] == p['native_origin']
assert all(c['returncode'] == c['expected_returncode'] for c in public_q['commands'])
long_path = cache / 'bounded-long-local-v1/qualification.json'
retained_long = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-long'
assert descriptor(long_path) == descriptor(root / 'long-prerequisite.json') == p['long_prerequisite']['qualification']
assert long_path.read_bytes() == (root / 'long-prerequisite.json').read_bytes() == (retained_long / 'qualification.json').read_bytes()
assert q['long_prerequisite'] == {'file': p['long_prerequisite']['qualification'], 'source_commit': p['long_prerequisite']['source_commit']}
long_q = read(long_path)
assert long_q['bounded_30m_1h_passed'] is True and long_q['source_commit'] == p['long_prerequisite']['source_commit']
assert long_q['protocol'] == p['long_prerequisite']['protocol'] and long_q['storage']['released']
assert all(c['returncode'] == 0 for c in long_q['commands'])
assert read(retained_long / 'collection-qualification.json')['bounded_30m_1h_passed'] is True
inputs = read(root / 'build-inputs.json')
assert inputs['repository_sources'] == p['repository_files'] and inputs['vendor_sources'] == p['vendor_sources']
assert inputs['vendor_revision'] == p['vendor_revision']
assert inputs['model_or_frontend_execution_during_build'] is False
for key, directory in [('work', 'work'), ('producer', 'producer'), ('public_delivery', 'public-delivery'), ('public_helpers', 'public-helpers')]:
    assert inventory(root / directory) == inputs['protected_inputs'][key], key
for name, value in p['model_and_provider_files'].items():
    assert descriptor(root / 'work' / name) == value
for value in p['compiler_tools'].values():
    assert descriptor(value['path']) == value['file']
env = inputs['compiler_environment']
assert env['CMAKE'] == p['compiler_tools']['cmake']['path'] and env['CMAKE_BUILD_PARALLEL_LEVEL'] == '2'
assert env['RUSTFLAGS'] == '-C target-cpu=generic -C link-arg=-mmacosx-version-min=26.0'
package = root / 'work/nemo'
manifest = read(root / 'package.json')
assert manifest == read(package / 'package.json')
verify_files(package, manifest['files'])
verify_package_provenance(package, manifest)
assert manifest['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2' and manifest['backend'] == 'cpu'
package_files = read(root / 'producer/pooled-cpu-runtime-files.json')
assert {k: v for k, v in package_files.items() if k != 'files'} == construction['package_archive']
assert inventory(package) == package_files['files']
helpers = read(root / 'bounded-mixed-helpers-files.json')
assert {k: v for k, v in helpers.items() if k != 'files'} == q['helper_archive']
assert inventory(root / 'work/bin') == helpers['files']
assert not (root / 'independent-helper-restore').exists()
restore_archive(root / helpers['archive'], helpers, root / 'independent-helper-restore')
assert inventory(root / 'independent-helper-restore') == helpers['files']
public_helpers = read(root / 'public-delivery/pooled-public-helpers-files.json')
assert {k: v for k, v in public_helpers.items() if k != 'files'} == public_q['helper_archive']
assert inventory(root / 'public-helpers') == public_helpers['files']
assert descriptor(root / 'public-helpers/load-probe') == descriptor(root / 'work/bin/load-probe')
assert set(q['loaded_images']) == {'version-only', 'list', *p['tests']}
assert audit_loaded_libraries((root / 'version-only.stderr.log').read_text(), package, manifest,
    root / 'work/bin/load-probe') == q['loaded_images']['version-only']
assert q['loaded_images']['version-only']['private_library_count'] == 92
assert audit_loaded_libraries((root / 'list.stderr.log').read_text(), package, manifest,
    root / 'work/bin/mixed-native-tests', False) == q['loaded_images']['list']
assert q['loaded_images']['list']['private_library_count'] == 0
assert [row['name'] for row in q['cases']] == p['tests']
for row in q['cases']:
    name = row['name']
    assert row['passed'] is True and row['diarization_backend'] == 'cpu'
    text = (root / (name + '.stdout.log')).read_text()
    assert re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', text, re.M)
    assert '1 passed; 0 failed; 0 ignored;' in text
    placement = {}
    for line in text.splitlines():
        if line.startswith('chat-placement '):
            phase, value = line.removeprefix('chat-placement ').split(': ', 1)
            assert phase not in placement
            placement[phase] = json.loads(value)
    assert placement == row['chat_placement'] and set(placement) == {'initial', 'restored'}
    assert all(v['layers'] > 0 and v['backend'] == 'MTL' and v['device'] == 'Apple M4 Pro' for v in placement.values())
    assert audit_with_auxiliary((root / (name + '.stderr.log')).read_text(), package, manifest,
        root / 'work/bin/mixed-native-tests', {str(root / 'work/providers/libonnxruntime.dylib'):
        p['model_and_provider_files']['providers/libonnxruntime.dylib']}) == q['loaded_images'][name]
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
assert not (root / 'diagnostic-reserve.bin').exists()
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-mixed'
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
shutil.copy2(__file__, out / Path(__file__).name)
summary = {'schema_version': 1, 'source_commit': source,
    'bounded_cpu_mixed_residency_passed': True, 'release_qualified': False,
    'historical_source_pins_verified': len(pins), 'vendor_source_pins_verified': len(p['vendor_sources']),
    'actual_commands': len(expected), 'neural_load_order_cases': p['tests'], 'cases': q['cases'],
    'all_protected_inputs_unchanged': True, 'helper_archive_independently_restored': True,
    'loaded_images_replayed': list(q['loaded_images']), 'complete_deliveries_and_origins_verified': True,
    'completed_long_qualification_digest_verified': p['long_prerequisite']['qualification'],
    'new_model_calls_during_collection': 0,
    'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, len(files) + 1, 'members')
