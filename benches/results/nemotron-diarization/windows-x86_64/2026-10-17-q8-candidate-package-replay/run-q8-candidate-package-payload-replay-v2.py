"""Replay the actual failed SDK and prove strict package admission rejects it."""
import json
from pathlib import Path
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'windows-q8-candidate-ci-v2'
delivery = root / 'delivery-admission-v1/delivered'
tools = repo / 'tools/nemotron-diarization'
sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor
from candidate_windows_payloads import replay_candidate_delivery, verify_candidate_delivery
from windows_session_timing import file_inventory

inputs = {str(tools / name): descriptor(tools / name) for name in (
    'candidate_windows_compiler_receipt.py', 'candidate_windows_payloads.py', 'candidate_windows_sdk_receipt.py')}
inputs[str(Path(__file__))] = descriptor(Path(__file__))
before = file_inventory(delivery)
admission = json.loads((root / 'delivery-admission-v1/qualification.json').read_text())
assert before == json.loads((root / 'delivery-admission-v1/artifact-members.json').read_text())
assert descriptor(root / 'artifact.zip') == admission['archive']
assert admission['source_commit'] == 'fee2177adcd7b75b488c67bea5a593ea17df65de'
assert admission['run_id'] == 39063644421 and admission['job_id'] == 117486629205
result = replay_candidate_delivery(delivery)
assert result['payload_replay_passed'] is True and result['functional_lifecycle_gates_passed'] is True
assert result['realtime_gate_passed'] is False and result['production_candidate_sdk_gates_passed'] is False
try:
    verify_candidate_delivery(delivery)
except ValueError as error:
    rejection = str(error)
    assert rejection == 'completed exact Windows Q8 SDK/public/lifecycle qualification required'
else:
    raise AssertionError('actual failed SDK must not enter package construction')
assert file_inventory(delivery) == before
assert all(descriptor(Path(name)) == value for name, value in inputs.items())
report = {'schema_version': 1, 'run_id': admission['run_id'], 'job_id': admission['job_id'],
    'artifact_id': admission['artifact_id'], 'archive': admission['archive'], 'inputs': inputs,
    'payload_replay': result, 'strict_package_admission_rejected': True, 'rejection': rejection,
    'new_compiler_sdk_model_calls': 0, 'package_constructed': False, 'release_qualified': False}
with (root / 'package-payload-replay-v2.json').open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
