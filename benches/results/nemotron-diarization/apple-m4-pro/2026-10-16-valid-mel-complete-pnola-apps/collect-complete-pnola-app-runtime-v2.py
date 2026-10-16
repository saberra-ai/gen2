"""Replay saved complete-app outputs, actual loader histories and frozen controls."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'; sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor
from qualify_valid_mel_pnola_app import live_matches, recorded_matches
from sdk_archive import inventory
from verify_valid_mel_pnola_app import audit_app_images

base = cache / 'valid-mel-pnola-complete-app-runtime-v2'
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-complete-pnola-apps'
assert not out.exists(); read = lambda path: json.loads(path.read_text())
p = read(base / 'protocol.json'); r = read(base / 'qualification.json'); commands = read(base / 'commands.json')
assert r['automatic_packaged_resolution_gates_passed'] and not r['release_qualified'] and not r['microphone_used'] and not r['artifact_environment_overrides_used']
assert descriptor(base / 'protocol.json') == r['protocol'] == descriptor(tools / 'valid-mel-pnola-app-v2.protocol.json')
assert len(commands) == len(r['cases']) == len(p['cases']) == 5 and all(c['returncode'] == 0 for c in commands)
assert r['storage']['released'] and r['storage']['reserve_reported_allocated_bytes'] == 256 * 1024**2
assert r['storage']['start_minimum_bytes'] == 5 * 1024**3 and r['storage']['running_minimum_bytes'] == 1024**3
for key, root, prefix in [('tools', tools, 'tools/nemotron-diarization/'), ('repository_files', repo, '')]:
    for name, expected in p[key].items():
        historical = subprocess.check_output(['git', 'show', r['source_commit'] + ':' + prefix + name], cwd=repo)
        assert descriptor(root / name) == {'bytes': len(historical), 'sha256': hashlib.sha256(historical).hexdigest()} == expected, name
for name, expected in p['preparation_records'].items():
    assert descriptor(Path(name)) == expected, name
work = Path(p['work_path'])
for name, expected in p['files'].items():
    assert descriptor(work / name) == expected
protected = [str(work), *[app['path'] for app in p['apps'].values()]]
policy = '(version 1)\n(allow default)\n(deny network*)\n(deny file-write* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in protected) + ')\n' + '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
assert (base / 'runtime.sb').read_text() == policy
comparisons = []; image_counts = []; build = cache / 'valid-mel-pnola-instrumented-app-build-v1'
for case, command, completed in zip(p['cases'], commands, r['cases']):
    name = case['name']; backend = case['backend']; app = Path(p['apps'][backend]['path'])
    assert inventory(app) == p['apps'][backend]['inventory']
    executable = app / 'Contents/MacOS/pnola-desktop'; result_path = base / (name + '.result.json')
    expected_env = {'PATH': '/usr/bin:/bin', 'HOME': str(base / 'home'), 'TMPDIR': str(base / 'tmp') + '/', 'LC_ALL': 'C'}
    if case.get('unsupported_metal'):
        expected_env['GEN2_NEMO_EXECUTION'] = 'metal'
    arguments = ['--verify-live-speakers' if case['kind'] == 'live' else '--verify-bundle', str(work / 'ami.wav'), str(result_path)]
    assert command['name'] == completed['name'] == name and command['cwd'] == str(base) and command['environment'] == expected_env
    assert command['argv'] == ['/usr/bin/sandbox-exec', '-f', str(base / 'runtime.sb'), str(executable), *arguments]
    actual = audit_app_images((base / (name + '.stderr.log')).read_text(), app, read(build / (backend + '-static-verification.json')))
    assert actual == read(base / (name + '.images.json')) and actual['process_local_add_image_history_observed']
    assert actual['logical_runtime_library_count'] == (92 if backend == 'cpu' else 93) and actual['ort_provider_count'] == 1
    image_counts.append(actual['logical_runtime_library_count'])
    value = read(result_path)
    if case['kind'] == 'live':
        comparison = live_matches(value, read(work / (backend + '-live.json')), backend)
    else:
        comparison = recorded_matches(value, read(work / (backend + '.json')), p['bark_text'], backend, p['wav_sha256'], unavailable=case.get('unsupported_metal', False))
    assert comparison == completed['comparison'] and descriptor(result_path) == completed['result']
    assert descriptor(base / (name + '.images.json')) == completed['images'] and completed['passed']
    comparisons.append({'name': name, **comparison})
assert read(base / 'cases.json') == r['cases']
data_reclone = read(cache / 'identical-app-data-reclone-v1.json')
assert data_reclone['both_app_inventories_and_deep_signatures_unchanged'] and len(data_reclone['files']) == 8
for row in data_reclone['files']:
    assert row['replaced'] and descriptor(Path(row['source'])) == descriptor(Path(row['target'])) == row['payload']
out.mkdir()
for path in base.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
for name, expected in p['tools'].items():
    target = out / 'execution-tools' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(tools / name, target)
for name in ['identical-app-data-reclone-v1.json', 'reclone-identical-app-data-v1.py', 'valid-mel-pnola-complete-app-runtime-v2.launch.log']:
    shutil.copy2(cache / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'automatic_packaged_resolution_gates_passed': True, 'release_qualified': False,
          'source_commit': r['source_commit'], 'protocol': r['protocol'], 'source_pins': {'tools': len(p['tools']), 'repository_files': len(p['repository_files'])},
          'scope': 'Actual instrumented hardened same-host complete CPU/Metal app CLI resolution, recorded/live recovery/export and explicit CPU-package Metal rejection; all five neural processes completed.',
          'comparisons': comparisons, 'logical_native_image_counts': image_counts, 'ort_image_records': 5,
          'process_local_add_image_history_required_and_reparsed': True, 'no_artifact_environment_overrides': True,
          'neural_processes': 5, 'new_app_or_neural_or_native_version_calls_during_collection': 0,
          'limits': ['The original first attempt remains failed; these newly built binaries include verification-only loader instrumentation and retain hardened signing/exact local entitlements.',
                     'Recorded full timelines match hash-locked independent native outputs. Live compact timelines match prior debug outputs, plus source-pinned internal recording-API parity assertions; separate8004-row live tensors are not delivered or inferred.',
                     'All inputs/apps are read-only under source/SDK/build/Homebrew/network denials. Actual image history is complete for these five calls, including unloaded negative-load components, not every future workload.',
                     'No GUI, physical microphone, fresh-machine, quality, clean latency/performance, Developer ID/notarization or supported desktop release qualification.']}
record['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'receiver-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'members': len(record['files']) + 1, 'image_counts': image_counts, 'comparisons': comparisons}))
