"""Replay the saved paced public trial without any additional model execution."""
from pathlib import Path
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import wave

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'paced-public-runtime-v1'
source = '1bc54c85d694fd3a15983dc7d6bc2554578ae441'
read = lambda path: json.loads(Path(path).read_text())

def desc(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

p = read(root / 'protocol.json')
for name, expected in p['tools'].items():
    assert desc(tools / name) == expected, name
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/paced-public-runtime.protocol.json'], cwd=repo)
sys.path.insert(0, str(tools))
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from qualify_paced_public_stream import compare_reference
from paced_stream_metrics import paced_timing
from sdk_archive import inventory

verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: expected for name, expected in p['tools'].items()})
work = Path(p['work_path'])
assert inventory(work) == p['work_inventory']
for path, expected in p['prerequisite_records'].items():
    assert desc(path) == expected, path
build = read(work / 'baseline/construction.json')
bp = read(work / 'baseline/build-protocol.json')
assert build['constructed'] is True and build['model_calls'] == 0
assert desc(work / 'bin/paced-public') == build['binary']
verify_git_sources(repo, build['source_commit'], bp['repository_files'])
for name, expected in bp['repository_files'].items():
    assert desc(repo / name) == expected, name
assert all(bp['repository_files'][name] == expected for name, expected in build['used_gen2_source_files'].items())
with wave.open(str(work / 'assets/live-80s.wav'), 'rb') as audio:
    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) == (1,2,16000,1280640)
    pcm = audio.readframes(1280640)
normalized = b''.join(struct.pack('<f', sample[0] / 32768.) for sample in struct.iter_unpack('<h', pcm))
assert hashlib.sha256(normalized).hexdigest() == p['pcm_f32le_sha256']

q = read(root / 'qualification.json')
assert q['source_commit'] == source and q['trial_completed'] is True
assert q['physical_microphone_used'] is q['gui_used'] is q['release_qualified'] is False
assert [row['name'] for row in q['commands']] == p['command_order']
assert [row['returncode'] for row in q['commands']] == [0,0,0,1,0]
assert all(row['returncode'] == row['expected_returncode'] for row in q['commands'])
assert q['neural_commands_started'] == ['paced-public']
assert q['storage']['released'] is True
manifest = read(work / 'nemo/package.json')
assert set(q['loaded_images']) == {'version-only','error-before-model','paced-public'}
for name, expected in q['loaded_images'].items():
    executable = work / 'bin' / ('load-probe' if name == 'version-only' else 'paced-public')
    actual = audit_loaded_libraries((root / (name + '.stderr.log')).read_text(), work / 'nemo', manifest, executable,
        require_runtime=name != 'error-before-model')
    assert actual == expected
    assert actual['private_library_count'] == (0 if name == 'error-before-model' else 92)
results = root / 'results'
outputs_before = {path.name: desc(path) for path in results.iterdir() if path.is_file()}
assert outputs_before == q['output_files']
assert set(outputs_before) == {'prior-output.txt','error-before-model.json','paced-public.json','paced-public.json.f32'}
err = read(results / 'error-before-model.json')
assert err == q['pre_model_error_control'] and err['complete'] is False
assert 'Operation not permitted (os error 1)' in err['error']
assert err['pushes'] == err['observations'] == [] and err['checkpoints'] == {}
public = read(results / 'paced-public.json')
native = read(work / 'baseline/native-80s.json')
assert public['input_pcm_f32le_sha256'] == native['input_pcm_f32le_sha256'] == p['pcm_f32le_sha256']
assert public['input_repeats'] == 1 and native['audio_samples'] == 1280640
comparison = compare_reference(public, native, (work / 'baseline/native-80s.json.f32').read_bytes(),
    (results / 'paced-public.json.f32').read_bytes())
assert comparison == q['comparison']
timing = paced_timing(public)
assert timing == q['timing']
assert outputs_before == {path.name: desc(path) for path in results.iterdir() if path.is_file()}
assert inventory(work) == p['work_inventory']

output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-paced-public-runtime'
assert not output.exists()
output.mkdir()
for path in [root / 'protocol.json',root / 'qualification.json',root / 'runtime.sb',root / 'premodel.sb',
             *root.glob('*.stdout.log'),*root.glob('*.stderr.log'),*results.glob('*.json'),
             Path(__file__),cache / 'paced-public-runtime-v1.controller.log']:
    shutil.copy2(path, output / path.name)
summary = {'schema_version':1,'source_commit':source,'trial_completed':True,
    'current_tools_checked_before_import':len(p['tools']), 'staged_inventory_entries':len(p['work_inventory']),
    'historical_and_current_construction_source_files':len(bp['repository_files']),
    'used_gen2_dependencies_covered':len(build['used_gen2_source_files']),
    'original_neural_processes':1,'collection_model_calls':0,'private_images_replayed':[92,0,92],
    'input_pcm_independently_normalized':True,'input_pcm_f32le_sha256':p['pcm_f32le_sha256'],
    'comparison_replayed':comparison,'timing_replayed':timing,'outside_git_outputs_bound_before_and_after':outputs_before,
    'physical_microphone_used':False,'gui_used':False,'original_quality_gate_status_changed':False,
    'release_qualified':False,'limits':p['limits']}
(output / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files = {path.name:desc(path) for path in output.iterdir() if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(output, len(files)+1, 'members')
print(json.dumps({'comparison':comparison,'timing':timing},indent=2))
