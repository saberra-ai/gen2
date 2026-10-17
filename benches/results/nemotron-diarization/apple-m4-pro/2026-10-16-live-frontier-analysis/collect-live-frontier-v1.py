"""Recompute retained observer statistics only; no app/model/native runtime calls."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
root = cache / 'live-frontier-analysis-v1'
source = '87c542b9937b71632c1ce5debf21456c03f53aa7'
descriptor = lambda data: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
read = lambda path: json.loads(path.read_text())
p = read(root / 'protocol.json')
r = read(root / 'analysis.json')
environment = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1')
blob = lambda name: subprocess.check_output(['git', 'show', source + ':' + name], cwd=repo, env=environment)
assert (root / 'protocol.json').read_bytes() == blob('tools/nemotron-diarization/live-frontier-analysis.protocol.json')
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['inputs'])
for name, expected in pins.items():
    assert descriptor((repo / name).read_bytes()) == descriptor(blob(name)) == expected
sys.path.insert(0, str(tools))
from live_frontier_metrics import frontier_observations
assert r['source_commit'] == source and r['analysis_completed'] is True
assert r['release_qualified'] is False and r['new_model_or_native_calls'] == 0
assert list(r['cases']) == [c['name'] for c in p['cases']]
for case in p['cases']:
    value = read(repo / case['result'])
    receiver = read(repo / case['receipt'])
    construction = read(repo / case['construction'])
    assert receiver['release_qualified'] is False and receiver[case['success_field']] is True
    assert receiver['source_commit'] == case['source_commit']
    assert construction['pnola_source_files']['src-tauri/src/live.rs'] == p['observer_source']
    assert value['passed'] is True and value['microphone_used'] is False and value['speaker_qualification'] is True
    assert value['duration_seconds'] == 80.04 and value['sections'] == 11 and value['queued'] == 0
    assert value['speakers']['status'] == 'complete'
    assert value['speakers']['timeline']['runtime_version'] == case['runtime_version']
    assert value['speakers']['timeline']['backend'] == case['backend']
    assert frontier_observations(value['provisional_revisions'], 80.04) == r['cases'][case['name']]
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-live-frontier-analysis'
assert not out.exists()
out.mkdir()
for name in ['protocol.json', 'analysis.json']:
    shutil.copy2(root / name, out / name)
for path in [Path(__file__), cache / 'live-frontier-analysis-v1.controller.log',
             cache / 'platform-ci/model-free-242-live-frontier-local.log']:
    shutil.copy2(path, out / path.name)
summary = {'schema_version': 1, 'source_commit': source, 'analysis_completed': True,
           'historical_and_current_input_source_pins_verified': len(pins),
           'all_three_complete_retained_observer_traces_replayed': True,
           'retained_observations': {name: len(value['observations']) for name, value in r['cases'].items()},
           'new_model_app_or_native_runtime_calls': 0,
           'actual_end_to_end_label_latency_measured': False, 'release_qualified': False,
           'limits': p['limits']}
(out / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
files = {path.name: descriptor(path.read_bytes()) for path in out.iterdir() if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, len(files) + 1, 'members')
