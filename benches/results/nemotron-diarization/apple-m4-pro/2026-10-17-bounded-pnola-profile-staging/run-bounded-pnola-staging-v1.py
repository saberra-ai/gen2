"""One sandboxed actual Pnola asset-staging check, without app/SDK execution."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert source == '00b067f68472de8f033690bae999e9734bc5af79'
p = read(tools / 'macos-bounded-mixed.protocol.json')
sources = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
for name in ['integrations/pnola/overlay/scripts/speaker-assets.mjs',
             'integrations/pnola/overlay/tests/desktop-assets.test.mjs',
             'integrations/pnola/manifest.json', 'integrations/pnola/apply.py', 'integrations/pnola/test_apply.py']:
    sources[name] = descriptor(repo / name)
for name, value in sources.items():
    assert descriptor(repo / name) == value
    assert descriptor(repo / name) == {
        'bytes': len(subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo)),
        'sha256': hashlib.sha256(subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo)).hexdigest()}
sys.path.insert(0, str(tools))
from sdk_archive import inventory

package = cache / 'bounded-mixed-local-v1/work/nemo'
models = cache / 'bounded-mixed-local-v1/work/models/nemotron'
private = Path('/Users/victor/workspace/pnola')
private_names = read(cache / 'pooled-pnola-complete-app-build-v1/protocol.json')['pnola_source_files']
private_before = {n: descriptor(private / n) for n in private_names}
package_before, models_before = inventory(package), inventory(models)
assert read(cache / 'bounded-mixed-local-v1/qualification.json')['bounded_cpu_mixed_residency_passed'] is True
expected = {'source_commit': source, 'sources': sources,
    'source_package': package_before, 'source_models': models_before,
    'private_pnola_source_files_before': private_before,
    'scope': __doc__}
input_path = cache / 'bounded-pnola-staging-inputs-v1.json'
assert not input_path.exists()
input_path.write_text(json.dumps(expected, indent=2) + '\n')
run_dir = cache / 'bounded-pnola-staging-test-v1'
assert not run_dir.exists()
run_dir.mkdir()
for name in ['home', 'tmp']:
    (run_dir / name).mkdir()
policy = run_dir / 'static-staging.sb'
policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write* ' +
    ' '.join('(subpath ' + json.dumps(str(path)) + ')' for path in [repo, package, models, private]) + ')\n')
environment = {'PATH': '/usr/bin:/bin', 'HOME': str(run_dir / 'home'),
    'TMPDIR': str(run_dir / 'tmp') + '/', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}
launcher = cache / 'stage-bounded-pnola-local-v1.mjs'
command = ['/usr/bin/sandbox-exec', '-f', str(policy), '/opt/homebrew/bin/node', str(launcher)]
with (run_dir / 'staging.stdout.log').open('xb') as stdout, (run_dir / 'staging.stderr.log').open('xb') as stderr:
    result = subprocess.run(command, cwd=repo, env=environment, stdout=stdout, stderr=stderr, timeout=120)
assert result.returncode == 0
destination = cache / 'bounded-pnola-staging-local-v1'
staged = destination / '.desktop-assets'
assert read(destination / 'result.json')['passed'] is True
assert inventory(staged / 'nemo') == package_before
lock = read(tools / 'runtime-valid-mel-pool-cpu.lock.json')
for entry in lock['artifacts']:
    if entry['role'] in {'model', 'notice'}:
        assert descriptor(staged / 'nemotron' / entry['name']) == {'bytes': entry['size'], 'sha256': entry['sha256']}
assert inventory(package) == package_before and inventory(models) == models_before
assert {n: descriptor(private / n) for n in private_names} == private_before
assert all(descriptor(repo / n) == v for n, v in sources.items())
record = {'schema_version': 1, 'source_commit': source, 'actual_staging_passed': True,
    'command': command, 'environment': environment, 'returncode': result.returncode,
    'source_inputs': descriptor(input_path), 'launcher': descriptor(launcher),
    'node_binary': descriptor(Path('/opt/homebrew/bin/node').resolve()),
    'source_package_and_models_unchanged': True, 'private_pnola_source_files_unchanged': len(private_before),
    'staged_package_inventory': inventory(staged / 'nemo'), 'staged_model_inventory': inventory(staged / 'nemotron'),
    'app_or_native_sdk_calls': 0, 'model_inference_calls': 0, 'release_qualified': False,
    'limits': ['Actual isolated package/model/card/license staging and static provenance only. No app build, signing, launch, inference, GUI, microphone, fresh-machine or release qualification.']}
(run_dir / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print('Actual staging passed; private sources unchanged:', len(private_before))
