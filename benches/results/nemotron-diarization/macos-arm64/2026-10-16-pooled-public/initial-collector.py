"""Independently admit delivered M1 public evidence; never execute helpers."""
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
sys.path.insert(0, str(tools))
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'pooled-public-ci-v1'
source = 'f72176c51f39b595b8ac9ea243a46a9390c24f13'
from build_valid_mel_candidate import descriptor
from compare_stream_pooled import compare
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from sdk_archive import inventory, restore_archive
from valid_mel_long_metrics import peak_rss

api = json.loads((root / 'download-identity.json').read_text())
ci = json.loads((root / 'run.json').read_text())
opaque = {'bytes': 2057780, 'sha256': '8173d94a54dff4ead347182e1648ef5c36db8f70600265b0ddb4c4202e7a94d8'}
assert ci['status'] == 'completed' and ci['conclusion'] == 'success' and ci['headSha'] == source
assert ci['url'].endswith('/38987912935')
assert descriptor(root / 'artifact.zip') == opaque
with zipfile.ZipFile(root / 'artifact.zip') as archive:
    assert len(archive.infolist()) == len(set(archive.namelist())) == 60
    members = {m.filename: {'bytes': m.file_size, 'sha256': hashlib.sha256(archive.read(m)).hexdigest()}
               for m in archive.infolist()}
delivery = {'id': 12083733968, 'name': 'nemotron-macos-pooled-public', 'run_id': 38987912935,
            'source_commit': source, 'zip': opaque, 'files': members}
(root / 'delivery.json').write_text(json.dumps(delivery, indent=2) + '\n')
e = root / 'evidence'
admit_delivery(root / 'artifact.zip', root / 'download-identity.json', delivery, e)
p = json.loads((e / 'protocol.json').read_text())
q = json.loads((e / 'qualification.json').read_text())
assert q['source_commit'] == source and q['pooled_public_runtime_passed'] is True
assert q['release_qualified'] is False and q['model_execution_admitted'] is True
assert descriptor(e / 'protocol.json') == q['protocol']
assert (e / 'protocol.json').read_bytes() == subprocess.check_output(
    ['git', 'show', source + ':tools/nemotron-diarization/macos-pooled-public.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
build = json.loads((e / 'build-inputs.json').read_text())
assert build['source_pins'] == pins and build['model_or_frontend_execution_during_build'] is False
assert q['producer_artifact']['id'] == p['producer']['id'] == 12082417995
assert q['native_reference_artifact']['id'] == p['native']['id'] == 12083481818
expected = ['compile-controls', 'rustc-version', 'load-helper-build', 'lifecycle-build',
            'public-helper-build', 'runtime-controls', 'lifecycle-controls',
            'lifecycle-reference-readable', 'version-only', 'missing-library',
            'lifecycle-list', 'cpu-rejects-metal', 'public'] + ['lifecycle-' + str(i) for i in range(7)]
assert [c['name'] for c in q['commands']] == expected
assert all(c['returncode'] == c['expected_returncode'] ==
           (1 if c['name'] in ['missing-library', 'cpu-rejects-metal'] else 0) for c in q['commands'])
assert q['inference_commands_started'] == ['cpu-rejects-metal', 'public', 'lifecycle-0',
    'lifecycle-1', 'lifecycle-2', 'lifecycle-3', 'lifecycle-4', 'lifecycle-6']
assert q['lifecycle_tests_passed'] == p['lifecycle_tests'] and len(p['lifecycle_tests']) == 7
for i, name in enumerate(p['lifecycle_tests']):
    log = (e / ('lifecycle-' + str(i) + '.stdout.log')).read_text()
    assert re.search(r'^test ' + re.escape(name) + r' \.\.\. ok$', log, re.MULTILINE)
    assert '1 passed; 0 failed; 0 ignored;' in log
helpers = root / 'restored-helpers'
helper_manifest = json.loads((e / 'pooled-public-helpers-files.json').read_text())
assert {k: v for k, v in helper_manifest.items() if k != 'files'} == q['helper_archive']
restore_archive(e / helper_manifest['archive'], helper_manifest, helpers)
assert inventory(helpers) == helper_manifest['files'] == build['compiled_helpers']
package = cache / 'pooled-macos-construction-ci-v1/restored-pooled-cpu-runtime'
manifest = verify(package)
assert json.loads((e / 'package.json').read_text()) == manifest
full = absent = 0
for name, original in q['loaded_images'].items():
    executable = helpers / ('load-probe' if name == 'version-only' else
        'public-stream' if name in ['cpu-rejects-metal', 'public'] else 'native-lifecycle')
    remote_package = Path('/Users/runner/work/_temp/pooled-public/work/nemo')
    trace = (e / (name + '.stderr.log')).read_text().replace(original['executable'], str(executable)).replace(
        str(remote_package), str(package))
    entire = name not in ['lifecycle-list', 'lifecycle-5']
    replay = audit_loaded_libraries(trace, package, manifest, executable, entire)
    assert replay['private_library_count'] == (92 if entire else 0)
    assert json.loads(json.dumps(replay).replace(str(executable), original['executable']).replace(
        str(package), str(remote_package))) == original
    full += entire
    absent += not entire
assert (full, absent) == (9, 2)
native_path = cache / 'pooled-native-ci-v1/evidence/results/native-60s.json'
native = json.loads(native_path.read_text())
public = json.loads((e / 'results/public.json').read_text())
replay = compare(public, native, Path(str(native_path) + '.f32').read_bytes(), (e / 'results/public.f32').read_bytes())
assert replay == q['comparison'] and replay['retained_rows'] == 6000
assert replay['probabilities_bit_exact'] and replay['turns'] == 27 and replay['maximum_turn_error_seconds'] == 0
assert public['input_pcm_f32le_sha256'] == native['input_pcm_f32le_sha256'] == p['pcm_sha256']
assert public['input_repeats'] == native['input_repeats'] == 1
assert (e / 'reference/ami.json.f32').read_bytes() == Path(str(native_path) + '.f32').read_bytes()
assert json.loads((e / 'reference/ami.json').read_text()) == {'segments': [
    {'start': t['start_seconds'], 'end': t['end_seconds'], 'speaker': t['speaker']} for t in native['turns']]}
assert json.loads((e / 'host.json').read_text())['cpu'] == 'Apple M1 (Virtual)'
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes'] >= 1024 ** 3
out = repo / 'benches/results/nemotron-diarization/macos-arm64/2026-10-16-pooled-public'
assert not out.exists()
out.mkdir()
for path in e.rglob('*'):
    if path.is_file() and path.name != helper_manifest['archive']:
        destination = out / path.relative_to(e)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        assert descriptor(path) == descriptor(destination)
for name in ['delivery.json', 'download-identity.json', 'run.json']:
    shutil.copy2(root / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'source_commit': source, 'ci_url': ci['url'],
    'pooled_public_runtime_passed': True, 'delivered_members_verified': 60,
    'historical_source_pins_verified': len(pins), 'probabilities_compared': 48000,
    'all_probabilities_bit_exact': True, 'turns_compared': 27, 'all_raw_turns_exact': True,
    'seven_real_lifecycle_tests_passed': p['lifecycle_tests'], 'actual_model_reloads_in_existing_lifecycle_test': 100,
    'strict_private_library_audits': {'full92': full, 'no_private_runtime': absent},
    'helper_payloads_independently_restored_and_rehashed': True,
    'warm_rtf': replay['gen2_rtf'], 'public_process_peak_rss_bytes': peak_rss((e / 'public.stderr.log').read_text()),
    'release_qualified': False, 'limits': [
        'Actual fresh M1 virtual CPU public Rust qualification. All20commands passed expectedstatuses; all7unchanged lifecycle tests passed including100reloads. No new inference during collection.',
        'All6000frames/48000probabilities and27rawturns exact against independently admitted immutable CABI delivery. Ninefull92-library audits andtwozero-runtime audits independently replayed with allhelperpayloads restored/rehashed.',
        'Shared-VM warmRTF2.1098284 exceeds1; this does not pass the real-time performance gate or establish a matched speedup.',
        'No new long/mixed/Metal/otherOS/quality/word-attribution/app/GUI/physicalmicrophone/freshcompleteapp/signing/notary/release gate or defaultpromotion. Opaquehelperarchive stays outsideGit with complete descriptors.']}
(out / 'collection-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
assert all(descriptor(out / n) == v for n, v in files.items())
print(out, 'members', len(files) + 1, 'comparison', replay)
