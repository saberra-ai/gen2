"""Publish static inspection with its controls and unchanged timing-result parent."""
import hashlib
import json
from pathlib import Path
import shutil

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'windows-q8-pair-ci-v1'


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


result_path = root / 'loop-inspection-v1.json'
result = json.loads(result_path.read_text())
assert result['source_commit'] == '259966ceca113f50d7d172fa63269b0bcb17b37a' and result['run_id'] == 39066832336
assert result['new_native_calls'] == 0 and result['production_qualified'] is False
assert set(result['functions']) == {'control', 'pair'}
assert all(descriptor(Path(name)) == value for name, value in result['inputs'].items())
pins_path = cache / 'q8-pair-loop-inspection-pre-execution-v1.json'
pins = json.loads(pins_path.read_text())
assert all(descriptor(cache / name) == value for name, value in pins.items())
assert pins['inspect-q8-pair-loops-v1.py'] == result['inputs'][str(cache / 'inspect-q8-pair-loops-v1.py')]
control_path = cache / 'q8-pair-loop-controls-v1.json'
controls = json.loads(control_path.read_text())
assert controls['tests_run'] == 5 and controls['passed'] is True and controls['new_compiler_native_model_calls'] == 0
parent = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair'
parent_manifest_path = parent / 'collection.json'
parent_raw = parent_manifest_path.read_bytes()
parent_identity = {'bytes': len(parent_raw), 'sha256': hashlib.sha256(parent_raw).hexdigest()}
manifest = json.loads(parent_raw)['files']
parent_inputs = {str(parent / name): value for name, value in manifest.items()}
parent_inputs[str(parent_manifest_path)] = parent_identity
def verify_parent():
    assert not parent.is_symlink()
    assert {path.relative_to(parent).as_posix() for path in parent.rglob('*') if path.is_file()} == set(manifest) | {'collection.json'}
    assert all(not path.is_symlink() for path in parent.rglob('*'))
    assert all(descriptor(Path(name)) == value for name, value in parent_inputs.items())
verify_parent()
qualification = parent / 'collection-qualification.json'
q = json.loads(qualification.read_text())
assert q['source_commit'] == result['source_commit'] and q['run_id'] == result['run_id']
assert q['diagnostic_completed'] is True and q['full_phase_vectors_verified'] == 1056
assert q['production_sdk_qualified'] is False and q['model_rtf_qualified'] is False
files = {'inspection.json': result_path, 'inspector.py': cache / 'inspect-q8-pair-loops-v1.py',
    'controls.py': cache / 'test-q8-pair-loop-inspection-v1.py', 'controls.json': control_path,
    'pre-execution-pins.json': pins_path, 'publisher.py': Path(__file__)}
inputs = {str(path): descriptor(path) for path in files.values()}
dest = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair-loop-inspection'
assert not dest.exists()
dest.mkdir()
for name, path in files.items():
    shutil.copyfile(path, dest / name)
    assert descriptor(dest / name) == inputs[str(path)]
proof = {'source_commit': result['source_commit'], 'run_id': result['run_id'],
    'parent_collection': {'path': parent_manifest_path.relative_to(repo).as_posix(), **parent_identity},
    'parent_qualification': manifest['collection-qualification.json'], 'parent_inputs': parent_inputs, 'inputs': inputs,
    'functions': result['functions'], 'original_timing_comparison': q['timing_comparison'],
    'new_native_calls': 0, 'production_qualified': False, 'release_qualified': False, 'limits': result['limits']}
assert all(descriptor(Path(name)) == value for name, value in inputs.items())
assert all(descriptor(Path(name)) == value for name, value in result['inputs'].items())
verify_parent()
with (dest / 'collection-qualification.json').open('x') as stream:
    stream.write(json.dumps(proof, indent=2) + '\n')
payloads = {path.name: descriptor(path) for path in dest.iterdir()}
with (dest / 'collection.json').open('x') as stream:
    stream.write(json.dumps({'schema_version': 1, 'files': payloads}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'files': len(payloads) + 1}))
