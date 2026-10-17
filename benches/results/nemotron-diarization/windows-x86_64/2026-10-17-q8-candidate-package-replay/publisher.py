"""Retain the actual failed-SDK package rejection and source/test proof."""
import json
from pathlib import Path
import shutil
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor

actual = cache / 'windows-q8-candidate-ci-v2/package-payload-replay-v2.json'
record = json.loads(actual.read_text())
assert record['strict_package_admission_rejected'] is True
assert record['payload_replay']['payload_replay_passed'] is True
assert record['payload_replay']['realtime_gate_passed'] is False
assert record['payload_replay']['production_candidate_sdk_gates_passed'] is False
assert record['package_constructed'] is False and record['release_qualified'] is False
assert record['new_compiler_sdk_model_calls'] == 0
assert all(descriptor(Path(name)) == value for name, value in record['inputs'].items())
parent = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate'
parent_files = json.loads((parent / 'collection.json').read_text())['files']
assert {p.relative_to(parent).as_posix() for p in parent.rglob('*') if p.is_file()} == set(parent_files) | {'collection.json'}
assert all(descriptor(parent / name) == value for name, value in parent_files.items())
for suffix, phrase in [('full', 'Ran 539 tests'), ('pnola', 'Ran 3 tests')]:
    text = (cache / ('q8-candidate-payloads-' + suffix + '-suite-v1.stderr.log')).read_text()
    assert phrase in text and '\nOK' in text and 'FAILED' not in text
files = {'actual-replay.json': actual, 'publisher.py': Path(__file__),
    'run-q8-candidate-package-payload-replay-v2.py': cache / 'run-q8-candidate-package-payload-replay-v2.py'}
for name in ('candidate_windows_compiler_receipt.py', 'candidate_windows_payloads.py',
             'candidate_windows_sdk_receipt.py', 'test_candidate_windows_payloads.py', 'test_candidate_windows_sdk_receipt.py'):
    files['sources/' + name] = tools / name
for kind in ('full', 'pnola'):
    for stream in ('stdout', 'stderr'):
        name = 'q8-candidate-payloads-' + kind + '-suite-v1.' + stream + '.log'
        files['tests/' + name] = cache / name
inputs = {str(path): descriptor(path) for path in files.values()}
dest = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate-package-replay'
assert not dest.exists()
dest.mkdir()
for name, path in files.items():
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert descriptor(target) == inputs[str(path)]
proof = {**record, 'publication_inputs': inputs,
    'parent_collection': {'path': parent.relative_to(repo).as_posix(), **descriptor(parent / 'collection.json')},
    'tool_tests': {'total': 539, 'skipped_windows_only': 7, 'passed': True},
    'pnola_tests': {'total': 3, 'passed': True},
    'limits': 'Actual failed SDK payload replay and strict package rejection only. No passing producer, package construction, performance, desktop or release qualification is established.'}
assert all(descriptor(Path(name)) == value for name, value in inputs.items())
with (dest / 'collection-qualification.json').open('x') as stream:
    stream.write(json.dumps(proof, indent=2) + '\n')
manifest = {path.relative_to(dest).as_posix(): descriptor(path) for path in dest.rglob('*') if path.is_file()}
with (dest / 'collection.json').open('x') as stream:
    stream.write(json.dumps({'schema_version': 1, 'files': manifest}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'files': len(manifest) + 1, 'actual_package_rejection': True}))
