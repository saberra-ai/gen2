"""Replay every fixed development cell without another neural call."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'context-development-diagnostic-v2'
read = lambda path: json.loads(path.read_text())
def desc(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
p = read(root / 'protocol.json')
source = sys.argv[1]
for name, expected in p['tools'].items():
    assert desc(tools / name) == expected
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git', 'show', source + ':tools/nemotron-diarization/context-development-diagnostic-v2.protocol.json'], cwd=repo)
sys.path.insert(0, str(tools))
import numpy as np
from ami_prefix_metrics import load_readonly_scorer
from compare import segments
from compare_stream_valid_mel import compare
from context_development_metrics import production_turns
from diagnose_ami_reference import calibrate_engine, official_der
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from score import der
from sdk_archive import inventory
verify_git_sources(repo, source, {'tools/nemotron-diarization/' + name: expected for name, expected in p['tools'].items()})
work = Path(p['work_path'])
assert inventory(work) == p['work_inventory']
for path, expected in p['prerequisite_records'].items():
    assert desc(path) == expected
old_build = read(work / 'provenance/public-build.json')
verify_git_sources(repo, old_build['source_commit'], old_build['rust_source_files'])
assert desc(work / 'bin/public-stream') == {key:old_build['inputs']['bin/public-stream'][key] for key in ['bytes','sha256']}
q = read(root / 'qualification.json')
assert q['source_commit'] == source and q['diagnostic_completed'] is True
assert q['training_exposed_development'] is True and q['original_gate_status_changed'] is q['release_qualified'] is False
assert q['fresh_fp32_calls'] == 0 and len(q['neural_commands_started']) == 6
assert [row['name'] for row in q['commands']] == p['command_order']
assert all(row['returncode'] == row['expected_returncode'] for row in q['commands'])
assert q['storage']['released'] is True
manifest = read(work / 'nemo/package.json')
for name, saved in q['loaded_images'].items():
    executable = work / 'bin' / ('load-probe' if name == 'version-only' else 'public-stream' if name.endswith('-public-low') else 'native-context')
    actual = audit_loaded_libraries((root / (name + '.stderr.log')).read_text(), work / 'nemo', manifest, executable)
    assert actual == saved and actual['private_library_count'] == 92
assert len(q['loaded_images']) == 8
engine = load_readonly_scorer(work / 'scoring/md_eval.py', p['independent_scorer'])
assert calibrate_engine(engine) == read(root / 'scorer-calibrations.json')
refs = {name:read(work / 'references' / (name + '.json')) for name in ['transcriber','published-ntt-mfa']}
results = root / 'results'
arrays = {path.name: desc(path) for path in results.glob('*.f32')}
expected_arrays = {channel + suffix for channel in p['channels'] for suffix in
                   ['-low_latency.json.f32', '-large_context.json.f32', '-public-low.f32']}
assert set(arrays) == expected_arrays
for channel in p['channels']:
    for profile in p['geometries']:
        name = channel + '-' + profile
        value = read(results / (name + '.json'))
        raw = (results / (name + '.json.f32')).read_bytes()
        assert value['geometry'] == p['geometries'][profile]
        assert value['audio_samples'] == 4800000 and value['input_repeats'] == 1
        assert value['input_pcm_f32le_sha256'] == p['pcm_f32le_sha256'][channel]
        assert value['native_frame_count'] == value['retained_rows'] == 30000 and value['retained_first_frame'] == 0
        assert len(raw) == 30000 * 8 * 4
        probabilities = np.frombuffer(raw,dtype='<f4').reshape(30000,8)
        assert np.isfinite(probabilities).all() and ((0<=probabilities)&(probabilities<=1)).all()
        hyp = {'shared':segments(probabilities),'production':production_turns(value['turns'])}
        scores = {policy:{mode:{'frame':der(ref,h,300.),'official':official_der(engine,ref,h,300.)}
                         for mode,h in hyp.items()} for policy,ref in refs.items()}
        assert scores == q['scores'][name]
        assert q['native_metrics'][name] == {'rtf':value['elapsed_micros']/1e6/300.,
            'cold_load_seconds':value['cold_native_load_micros']/1e6,'push_median_seconds':value['native_push_median_micros']/1e6,
            'push_p95_seconds':value['native_push_p95_micros']/1e6,'finish_seconds':value['finish_micros']/1e6}
    public = read(results / (channel + '-public-low.json'))
    native = read(results / (channel + '-low_latency.json'))
    native_raw = (results / (channel + '-low_latency.json.f32')).read_bytes()
    public_raw = (results / (channel + '-public-low.f32')).read_bytes()
    assert native_raw == public_raw
    assert public['input_pcm_f32le_sha256'] == p['pcm_f32le_sha256'][channel] and public['input_repeats'] == 1
    assert compare(public,native,native_raw,public_raw) == q['comparisons'][channel]
differences = {}
assert arrays == {path.name: desc(path) for path in results.glob('*.f32')}
for channel in p['channels']:
    low, large = q['scores'][channel + '-low_latency'], q['scores'][channel + '-large_context']
    differences[channel] = {policy:{mode:{scorer:{key:large[policy][mode][scorer][key]-low[policy][mode][scorer][key]
        for key in ['der','missed_speaker_seconds','false_alarm_speaker_seconds','confusion_speaker_seconds']}
        for scorer in ['frame','official']} for mode in ['shared','production']} for policy in refs}
output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-development-diagnostic'
assert not output.exists()
output.mkdir()
for path in [root / 'protocol.json',root / 'qualification.json',root / 'scorer-calibrations.json',
             *root.glob('*.stdout.log'),*root.glob('*.stderr.log'),*results.glob('*.json'),
             Path(__file__),cache / 'context-development-diagnostic-v2.controller.log',
             cache / 'context-development-bounds-regression.log',cache / 'platform-ci/model-free-245-context-execution-local.log']:
    shutil.copy2(path,output / path.name)
summary = {'schema_version':1,'source_commit':source,'diagnostic_completed':True,
    'current_tool_pins_checked_before_import':len(p['tools']),'staged_input_inventory_entries':len(p['work_inventory']),
    'historical_public_rust_source_blobs_verified':len(old_build['rust_source_files']),
    'original_neural_processes':6,'new_collection_neural_calls':0,'strict_runtime_image_audits_replayed':8,
    'native_public_exact_rows_per_channel':30000,'all4native_cells_replayed':True,
    'all32scores_replayed_exactly':True,'scorer_controls_replayed':4,'large_minus_low':differences,
    'outside_git_probability_arrays':arrays,
    'arrays_and_audio_outside_git':True,'large_context_public_rust_parity_evaluated':False,
    'matched_fp32_q8_bound_evaluated':False,'original_gate_status_changed':False,'release_qualified':False,'limits':p['limits']}
(output / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files={path.name:desc(path) for path in output.iterdir() if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(output,len(files)+1,'members')
