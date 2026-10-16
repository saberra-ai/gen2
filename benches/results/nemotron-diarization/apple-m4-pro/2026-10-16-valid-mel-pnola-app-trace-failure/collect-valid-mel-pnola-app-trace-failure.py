"""Retain actual complete-app trace failure and model-free hardened-runtime diagnosis."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from qualify_valid_mel_pnola_app import recorded_matches

base = cache / 'valid-mel-pnola-complete-app-runtime-v1'
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-app-trace-failure'
assert not out.exists(); read = lambda p: json.loads(p.read_text())
r = read(base / 'qualification.json'); p = read(base / 'protocol.json'); commands = read(base / 'commands.json')
assert not r['automatic_packaged_resolution_gates_passed'] and r['cases'] == [] and r['error'] == "ValueError('missing app dyld telemetry')"
assert r['source_commit'] == '19718b52c0435cb6b6bb4f35660a2a2ef52d9097'
assert len(commands) == 1 and commands[0]['name'] == 'cpu-recorded' and commands[0]['returncode'] == 0
assert not (base / 'cpu-recorded.stderr.log').read_bytes()
assert r['storage']['released'] and r['storage']['reserve_reported_allocated_bytes'] == 256 * 1024**2
assert descriptor(base / 'protocol.json') == r['protocol'] == descriptor(repo / 'tools/nemotron-diarization/valid-mel-pnola-app.protocol.json')
result = read(base / 'cpu-recorded.result.json'); work = Path(p['work_path'])
comparison = recorded_matches(result, read(work / 'cpu.json'), p['bark_text'], 'cpu', p['wav_sha256'])
out.mkdir()
for path in base.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
shutil.copy2(cache / 'valid-mel-pnola-complete-app-runtime-v1.launch.log', out / 'launch.log')
for name, expected in p['tools'].items():
    source = repo / 'tools/nemotron-diarization' / name; assert descriptor(source) == expected
    dest = out / 'execution-tools' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
diagnosis = cache / 'hardened-dyld-diagnostics-v1'; diagnosed = read(diagnosis / 'result.json')
assert diagnosed['reproduced'] and diagnosed['app_or_native_model_calls'] == 0
dest = out / 'model-free-diagnosis'; dest.mkdir()
for path in diagnosis.iterdir():
    if path.is_file() and path.suffix in {'.c', '.json', '.log'}:
        shutil.copy2(path, dest / path.name)
shutil.copy2(cache / 'reproduce-hardened-dyld-v1.py', out / 'reproduce-hardened-dyld-v1.py')
for name in ['cleanup-stale-owned-incremental-app-v1.py', 'owned-incremental-app-cleanup-v1.json']:
    shutil.copy2(cache / name, out / name)
signature = subprocess.run(['/usr/bin/codesign', '--display', '--verbose=4', p['apps']['cpu']['path']], capture_output=True, check=True)
(out / 'actual-cpu-app-signature.log').write_bytes(signature.stdout + signature.stderr)
assert b'(adhoc,runtime)' in signature.stderr
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'automatic_packaged_resolution_gates_passed': False, 'release_qualified': False,
          'scope': 'Actual first complete CPU app recording returned0 with correct output; qualification fails because hardened signing suppressed required environment dyld telemetry.',
          'source_commit': r['source_commit'], 'protocol': r['protocol'], 'neural_child_processes_started': 1,
          'remaining_four_neural_cases_started': False, 'partial_recorded_turn_comparison': comparison,
          'first_command_elapsed_seconds': commands[0]['elapsed_seconds'], 'recorded_words': len(result['attribution']['words']),
          'recorded_text_matches_frozen_bark': True, 'actual_dyld_telemetry_observed': False,
          'hardened_suppression_reproduced_without_models': True,
          'diagnostic_probe_binaries': {name: descriptor(diagnosis / name) for name in ['plain', 'hardened']},
          'new_native_or_neural_calls_during_collection': 0,
          'limits': ['CPU partial numerical/ASR output is retained, but it cannot establish actual framework closure or pass the original case.',
                     'Original five-case attempt stays failed. A future versioned protocol must obtain process image diagnostics without weakening release signing.',
                     'No CPU live, Metal app, explicit-Metal negative, GUI, microphone, quality, performance or release qualification from this attempt.']}
record['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'receiver-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'members': len(record['files']) + 1, 'passed': False, 'partial_comparison': comparison}))
