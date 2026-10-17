"""Correct one stale test count in fresh receivers; V1 remains uninvoked; V2 preparation stopped on an exact whitespace mismatch before any artifact invocation."""
import hashlib
import json
from pathlib import Path

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')


def desc(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


record = json.loads((cache / 'windows-q8-scale-receivers-pre-execution-v1.json').read_text())
for name, expected in record['files'].items(): assert desc(cache / name) == expected
assert record['receivers_invoked'] is False and record['artifact_consumed'] is False
names = [name for name in record['files'] if name.endswith('.py') and name != 'prepare-windows-q8-scale-receivers-v1.py']
assert len(names) == 6
renames = {name: name.replace('-v1.py', '-v3.py') for name in names}
renames.update({name: name.replace('-v1.', '-v3.') for name in [
    'prepare-windows-q8-scale-receivers-v1.py', 'windows-q8-scale-receivers-preparation-v1.json',
    'windows-q8-scale-receivers-pre-execution-v1.json', 'q8-scale-receiver-controls-local-v1.json',
    'q8-scale-receiver-controls-local-v1.stdout.log', 'q8-scale-receiver-controls-local-v1.stderr.log']})
outputs = {}
for old in names:
    raw = (cache / old).read_text()
    for before, after in renames.items(): raw = raw.replace(before, after)
    if old == 'admit-windows-q8-scale-delivery-v1.py':
        assert raw.count("'actual_scale_control_tests':6") == 1
        raw = raw.replace("'actual_scale_control_tests':6", "'actual_scale_control_tests':7", 1)
    compile(raw, renames[old], 'exec')
    with (cache / renames[old]).open('x') as stream: stream.write(raw)
    outputs[renames[old]] = desc(cache / renames[old])
report = {'parents': {name: record['files'][name] for name in names},
    'parent_preexecution': desc(cache / 'windows-q8-scale-receivers-pre-execution-v1.json'),
    'outputs': outputs, 'preparer': desc(Path(__file__)), 'receivers_invoked': False,
    'artifact_consumed': False, 'source_commit': record['source_commit'],
    'run_id': record['run_id'], 'job_id': record['job_id'],
    'change': 'Only actual_scale_control_tests6-to-7 plus versioned source/control/preparation filename bindings. Data root and stage output names stay V1. V1 receivers never invoked.'}
with (cache / 'windows-q8-scale-receivers-preparation-v3.json').open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
print(json.dumps({'receivers_prepared': len(outputs), 'artifact_consumed': False}))
