"""Independently replay delivered public evidence; execute no native code."""
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
root = cache / 'bounded-public-ci-v1'
source = '9ddefaf9c17806730cb2e12f1106c08102331b43'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


ci = read(root / 'run.json')
artifacts = read(root / 'artifacts.json')['artifacts']
assert ci['databaseId'] == 39006112891 and ci['headSha'] == source
assert ci['status'] == 'completed' and ci['conclusion'] == 'success'
assert len(artifacts) == 1
api = artifacts[0]
assert api['name'] == 'nemotron-macos-bounded-public'
assert api['workflow_run']['id'] == ci['databaseId'] and api['workflow_run']['head_sha'] == source
opaque = descriptor(root / 'artifact.zip')
assert opaque['bytes'] == api['size_in_bytes'] and api['digest'] == 'sha256:' + opaque['sha256']
expected_commands = ['compile-controls', 'rustc-version', 'load-helper-build', 'lifecycle-build',
    'public-helper-build', 'runtime-controls', 'lifecycle-controls', 'lifecycle-reference-readable',
    'version-only', 'missing-library', 'lifecycle-list', 'cpu-rejects-metal', 'public']
expected_commands += ['lifecycle-' + str(i) for i in range(7)]
expected_members = {'protocol.json', 'qualification.json', 'producer-artifact.json', 'package.json', 'host.json', 'acquisitions.json',
    'helper-load-commands.json', 'build-inputs.json', 'pooled-public-helpers-files.json',
    'pooled-public-helpers.tar.gz', 'compile.sb', 'runtime.sb', 'lifecycle.sb', 'premodel.sb',
    'helper-source/load-probe.cpp', 'results/prior-output-control.txt', 'results/public.json',
    'results/public.f32', 'reference/ami.json', 'reference/ami.json.f32'}
expected_members.update(name + suffix for name in expected_commands for suffix in ['.stdout.log', '.stderr.log'])
# The exact expected set, not arbitrary successful-job contents, is admitted.
with zipfile.ZipFile(root / 'artifact.zip') as bundle:
    entries = bundle.infolist()
    assert len(entries) == len(expected_members) and {m.filename for m in entries} == expected_members
    assert sum(m.file_size for m in entries) < 32 * 2**20
    members = {m.filename: {'bytes': m.file_size, 'sha256': hashlib.sha256(bundle.read(m)).hexdigest()} for m in entries}
    protocol_bytes = bundle.read('protocol.json')
p = json.loads(protocol_bytes)
assert protocol_bytes == subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-public.protocol.json'], cwd=repo)
for key, base in [('tools', tools), ('repository_files', repo)]:
    for name, expected in p[key].items():
        assert descriptor(base / name) == expected, name
sys.path.insert(0, str(tools))
from compare_stream_bounded import compare
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify_files, verify_package_provenance
from sdk_archive import inventory, restore_archive

pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
delivery = {'id': api['id'], 'name': api['name'], 'run_id': ci['databaseId'],
            'source_commit': source, 'zip': opaque, 'files': members}
(root / 'download-identity.json').write_text(json.dumps(api, indent=2) + '\n')
(root / 'delivery.json').write_text(json.dumps(delivery, indent=2) + '\n')
e = root / 'evidence'
admit_delivery(root / 'artifact.zip', root / 'download-identity.json', delivery, e)
before = {str(path.relative_to(e)): descriptor(path) for path in e.rglob('*') if path.is_file()}
assert before == members
q = read(e / 'qualification.json')
assert read(e / 'producer-artifact.json') == q['producer_artifact']
assert q['source_commit'] == source and q['admit_only'] is False and q['deliveries_admitted'] is True
assert q['bounded_public_runtime_passed'] is True and q['public_realtime_gate_passed'] is True
assert q['release_qualified'] is False and q['model_execution_admitted'] is True and 'error' not in q
assert descriptor(e / 'protocol.json') == q['protocol']
for key, field, folder in [('producer', 'producer_artifact', 'bounded-macos-construction-ci-v1'),
                            ('native_reference', 'native_reference_artifact', 'bounded-native-ci-v2')]:
    prior = cache / folder
    assert descriptor(prior / 'artifact.zip') == p[key]['zip']
    assert {str(path.relative_to(prior / 'evidence')): descriptor(path) for path in (prior / 'evidence').rglob('*') if path.is_file()} == p[key]['files']
    assert q[field] == read(prior / 'artifacts.json')['artifacts'][0]
    assert q[field]['id'] == p[key]['id'] and q[field]['name'] == p[key]['name']
    assert q[field]['workflow_run']['id'] == p[key]['run_id']
    assert q[field]['workflow_run']['head_sha'] == p[key]['source_commit']
native_q = read(cache / 'bounded-native-ci-v2/evidence/qualification.json')
assert native_q['native_parity_passed'] is True and native_q['realtime_gate_passed'] is True
assert native_q['deliveries']['producer'] == q['producer_artifact']
assert [c['name'] for c in q['commands']] == expected_commands
assert all(c['returncode'] == c['expected_returncode'] ==
           (1 if c['name'] in ['missing-library', 'cpu-rejects-metal'] else 0) for c in q['commands'])
assert all(not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes') for c in q['commands'])
assert all(c['argv'][:2] == ['/usr/bin/sandbox-exec', '-f'] for c in q['commands'])
assert q['inference_commands_started'] == ['cpu-rejects-metal', 'public', 'lifecycle-0',
    'lifecycle-1', 'lifecycle-2', 'lifecycle-3', 'lifecycle-4', 'lifecycle-6']
assert q['lifecycle_tests_passed'] == p['lifecycle_tests'] and len(p['lifecycle_tests']) == 7
for i, name in enumerate(p['lifecycle_tests']):
    log = (e / ('lifecycle-' + str(i) + '.stdout.log')).read_text()
    assert re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', log, re.MULTILINE)
    assert '1 passed; 0 failed; 0 ignored;' in log
build = read(e / 'build-inputs.json')
assert build['source_pins'] == p['repository_files'] and build['model_or_frontend_execution_during_build'] is False
assert descriptor(e / 'helper-source/load-probe.cpp') == p['tools']['macos_runtime_load_probe.cpp']
assert (e / 'helper-source/load-probe.cpp').read_bytes() == (tools / 'macos_runtime_load_probe.cpp').read_bytes()
helpers = root / 'restored-helpers'
helper_manifest = read(e / 'pooled-public-helpers-files.json')
assert {k: v for k, v in helper_manifest.items() if k != 'files'} == q['helper_archive']
restore_archive(e / helper_manifest['archive'], helper_manifest, helpers)
assert inventory(helpers) == helper_manifest['files'] == build['compiled_helpers']
assert {path.name for path in helpers.iterdir()} == {'public-stream', 'native-lifecycle', 'load-probe'}
for name, row in read(e / 'helper-load-commands.json').items():
    assert row['architectures'] == 'arm64' and descriptor(helpers / name) == row['binary']
package = cache / 'bounded-macos-construction-ci-v1/restored-pooled-cpu-runtime'
manifest = read(package / 'package.json')
verify_files(package, manifest['files'])
verify_package_provenance(package, manifest)
assert read(e / 'package.json') == manifest
full = absent = 0
assert set(q['loaded_images']) == {'version-only', 'lifecycle-list', 'cpu-rejects-metal', 'public'} | {'lifecycle-' + str(i) for i in range(7)}
remote = Path(q['commands'][0]['cwd'])
remote_package = remote / 'work/nemo'
for name, original in q['loaded_images'].items():
    executable = helpers / ('load-probe' if name == 'version-only' else
        'public-stream' if name in ['cpu-rejects-metal', 'public'] else 'native-lifecycle')
    trace = (e / (name + '.stderr.log')).read_text().replace(original['executable'], str(executable)).replace(str(remote_package), str(package))
    entire = name not in ['lifecycle-list', 'lifecycle-5']
    replay = audit_loaded_libraries(trace, package, manifest, executable, entire)
    assert replay['private_library_count'] == (92 if entire else 0)
    assert json.loads(json.dumps(replay).replace(str(executable), original['executable']).replace(str(package), str(remote_package))) == original
    full += entire
    absent += not entire
assert (full, absent) == (9, 2)
native_path = cache / 'bounded-native-ci-v2/evidence/results/native-60s.json'
native = read(native_path)
public = read(e / 'results/public.json')
native_raw = Path(str(native_path) + '.f32').read_bytes()
actual_raw = (e / 'results/public.f32').read_bytes()
assert len(native_raw) == len(actual_raw) == 192000
replay = compare(public, native, native_raw, actual_raw)
assert replay == q['comparison'] and replay['retained_rows'] == 6000
assert replay['probabilities_bit_exact'] and replay['turns'] == 27 and replay['maximum_turn_error_seconds'] == 0
assert replay['gen2_rtf'] < p['require_warm_rtf_below'] == 1.0
assert public['input_pcm_f32le_sha256'] == native['input_pcm_f32le_sha256'] == p['pcm_sha256']
assert public['input_repeats'] == native['input_repeats'] == 1
assert (e / 'reference/ami.json.f32').read_bytes() == native_raw
assert read(e / 'reference/ami.json') == {'segments': [
    {'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in native['turns']]}
assert read(e / 'host.json')['cpu'] == 'Apple M1 (Virtual)'
assert read(e / 'host.json')['cpu_counts'] == p['expected_cpu_counts']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= p['storage']['running_minimum']
assert q['storage']['reserve_reported_allocated_bytes'] >= p['storage']['reserve_bytes']
acquisitions = {entry['name']: {'bytes': entry['size'], 'sha256': entry['sha256']} for entry in p['assets']}
assert read(e / 'acquisitions.json') == acquisitions
assert q['acquisition_attempts'] == [{'name': entry['name'], 'expected': entry, 'verified': acquisitions[entry['name']]} for entry in p['assets']]
trace = (e / 'public.stderr.log').read_text()
usage = re.search(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$', trace, re.M)
wall, user, system = map(float, usage.groups())
rss_values = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', trace, re.M)
assert len(rss_values) == 1 and int(rss_values[0]) > 0
peak_rss_bytes = int(rss_values[0])
assert before == {str(path.relative_to(e)): descriptor(path) for path in e.rglob('*') if path.is_file()}
out = repo / 'benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-public'
assert not out.exists()
out.mkdir()
for path in e.rglob('*'):
    if path.is_file() and path.suffix != '.f32' and path.name != helper_manifest['archive']:
        destination = out / path.relative_to(e)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
for name in ['delivery.json', 'download-identity.json', 'run.json']:
    shutil.copy2(root / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'source_commit': source, 'ci_url': ci['url'], 'artifact': api, 'opaque_zip': opaque,
    'bounded_public_runtime_passed': True, 'public_realtime_gate_passed': True,
    'delivered_members_verified': len(members), 'all_saved_payloads_bound_before_and_after': before,
    'historical_source_pins_verified': len(pins), 'probabilities_compared': 48000,
    'all_probabilities_bit_exact': True, 'turns_compared': 27, 'all_raw_turns_exact': True,
    'seven_real_lifecycle_tests_passed': p['lifecycle_tests'], 'actual_model_reloads_in_existing_lifecycle_test': 100,
    'strict_private_library_audits': {'full92': full, 'no_private_runtime': absent},
    'helper_payloads_independently_restored_and_rehashed': True,
    'new_collection_native_helper_or_model_calls': 0,
    'acquired_asset_bytes_are_recorded_descriptors_not_delivered_payloads': True,
    'source_text_bytes_verified_modes_only_recorded': True,
    'warm_rtf': replay['gen2_rtf'], 'public_process_peak_rss_bytes': peak_rss_bytes,
    'time_l': {'real_seconds': wall, 'user_seconds': user, 'system_seconds': system},
    'release_qualified': False, 'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, 'members', len(files) + 1, 'comparison', replay)
