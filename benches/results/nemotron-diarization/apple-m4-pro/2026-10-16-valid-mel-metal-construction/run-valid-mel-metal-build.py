"""One committed model-free Metal construction; preserve failures, no retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
protocol_path = tools / 'valid-mel-metal-build.protocol.json'
p = json.loads(protocol_path.read_text())

def descriptor(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

assert descriptor(Path(__file__)) == p['external_launcher']
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/valid-mel-metal-build.protocol.json'], cwd=repo) == protocol_path.read_bytes()
for name, expected in p['tools'].items():
    data = subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == expected == descriptor(tools / name)
for name, expected in p['preparation_files'].items():
    assert descriptor(Path(p['preparation_path']) / name) == expected
source, prefix, output = (Path(p[name]) for name in ['source_path', 'prefix_path', 'output_path'])
assert not prefix.exists() and not output.exists() and not (source / 'build').exists()
env = {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'GIT_LFS_SKIP_SMUDGE': '1'}
output.mkdir()
(output / 'protocol.json').write_bytes(protocol_path.read_bytes())
commands = []

def run(name, argv):
    record = {'phase': name, 'argv': argv, 'environment': env, 'source_commit': commit}
    commands.append(record)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    started = time.monotonic()
    with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
        result = subprocess.run(argv, cwd=repo, env=env, stdout=stdout, stderr=stderr)
    record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(json.dumps({'phase': name, 'returncode': result.returncode}), flush=True)
    result.check_returncode()

sandbox = output / 'preflight.sb'
sandbox.write_text('(version 1)\n(allow default)\n(deny network*)\n')
python = str(cache / 'evidence-tools-env/bin/python')
for case in p['preflights']:
    run(case['name'], ['/usr/bin/sandbox-exec', '-f', str(sandbox), python, '-m', 'unittest', 'discover', '-s', str(tools), '-p', case['pattern']])
    assert 'Ran ' + str(case['tests']) + ' tests' in (output / (case['name'] + '.stderr.log')).read_text()
run('construction', [python, str(tools / 'build_valid_mel_metal_sdk.py'), '--source', str(source), '--prefix', str(prefix), '--output', str(output / 'build'), '--jobs', '4'])
for name, expected in p['tools'].items():
    assert descriptor(tools / name) == expected
record = json.loads((output / 'build/build.json').read_text())
assert record['source_commit'] == commit
assert record['backend'] == 'metal' and record['metal_embedded_shader_text_bundles_verified'] == 20
assert record['model_free_state_test_completed'] and record['version_function_only_executed']
assert not record['model_instantiated_or_inference_performed']
print(json.dumps({'constructed': True, 'neural_calls': 0, 'source_commit': commit}), flush=True)
