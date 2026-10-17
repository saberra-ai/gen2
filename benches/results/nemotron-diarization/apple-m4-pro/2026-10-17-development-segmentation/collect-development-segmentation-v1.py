"""Independently replay saved operation ablations; never call a model/helper."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'development-segmentation-v1'
source = 'fa6fe6b3875ea4ccaa5b37f35bc3429c4b16d267'
read = lambda path: json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p = read(root / 'protocol.json')
for name,expected in p['tools'].items():
    assert desc(tools / name) == expected, name
assert (root / 'protocol.json').read_bytes() == subprocess.check_output([
    'git','show',source+':tools/nemotron-diarization/development-segmentation.protocol.json'],cwd=repo)
before = desc(root / 'qualification.json')
for path,expected in p['inputs'].items():
    assert desc(path) == expected, path
sys.path.insert(0,str(tools))
import numpy as np
from ami_prefix_metrics import load_readonly_scorer
from diagnose_ami_reference import calibrate_engine, official_der
from frozen_git_sources import verify_git_sources
from score import der
from segmentation_ablation import CONTRASTS, compare_native_turns, native_segments

verify_git_sources(repo,source,{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()})
verify_git_sources(repo,source,{str(Path(name).relative_to(repo)):value for name,value in p['inputs'].items()
                               if Path(name).is_relative_to(repo)})
assert CONTRASTS == p['contrasts']
q = read(root / 'qualification.json')
assert q['source_commit'] == source and q['diagnostic_completed'] is True and q['model_calls'] == 0
assert q['training_exposed_development'] is True
assert q['original_gate_status_changed'] is q['production_policy_selected'] is q['release_qualified'] is False
assert q['meeting'] == 'ES2011a' and q['scoring_uem_seconds'] == [0,300]
prior = read(p['prior_qualification'])
prior_collection = read(p['prior_collection'])
manifest = read(Path(p['prior_collection']).parent / 'collection.json')
work = Path(p['prior_work'])
results = Path(p['prior_results'])
engine = load_readonly_scorer(work / 'scoring/md_eval.py',p['independent_scorer'])
assert calibrate_engine(engine) == q['calibrations']
refs = {name:read(work / 'references' / (name+'.json')) for name in p['references']}
expected_cells = {channel+'-'+profile for channel in p['channels'] for profile in p['profiles']}
assert set(q['hypotheses']) == set(q['scores']) == set(q['baseline_parity']) == set(q['deltas']) == expected_cells
scores_replayed = 0
baseline_turns = 0
for name in sorted(expected_cells):
    native = read(results / (name+'.json'))
    assert desc(results / (name+'.json')) == manifest['files'][name+'.json']
    array = results / (name+'.json.f32')
    assert desc(array) == prior_collection['outside_git_probability_arrays'][array.name]
    values = np.frombuffer(array.read_bytes(),dtype='<f4').reshape(30000,8)
    actual = {mode:native_segments(values,config) for mode,config in CONTRASTS.items()}
    assert actual == q['hypotheses'][name]
    assert compare_native_turns(actual['native_default'],native['turns']) == q['baseline_parity'][name]
    baseline_turns += len(native['turns'])
    assert set(q['scores'][name]) == set(q['deltas'][name]) == set(refs)
    for policy,reference in refs.items():
        scores = {mode:{'frame':der(reference,hypothesis,300.),'official':official_der(engine,reference,hypothesis,300.)}
                  for mode,hypothesis in actual.items()}
        assert scores == q['scores'][name][policy]
        assert scores['native_default'] == prior['scores'][name][policy]['production']
        deltas = {mode:{scorer:{metric:value[scorer][metric]-scores['native_default'][scorer][metric]
                    for metric in ['der','missed_speaker_seconds','false_alarm_speaker_seconds','confusion_speaker_seconds']}
                  for scorer in ['frame','official']} for mode,value in scores.items()}
        assert deltas == q['deltas'][name][policy]
        scores_replayed += len(scores)*2
assert before == desc(root / 'qualification.json')
for path,expected in p['inputs'].items():
    assert desc(path) == expected, path
output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-development-segmentation'
assert not output.exists()
output.mkdir()
for path in [root / 'protocol.json',root / 'qualification.json',Path(__file__),
             cache / 'development-segmentation-v1.controller.log',cache / 'prepare-development-segmentation-v1.py',
             cache / 'segmentation-native-order-red.log',cache / 'segmentation-native-order-green.log',
             cache / 'platform-ci/model-free-251-segmentation-local.log']:
    shutil.copy2(path,output / path.name)
summary = {'schema_version':1,'source_commit':source,'diagnostic_completed':True,'new_model_calls':0,
           'current_tools_checked_before_import':len(p['tools']),'inputs_checked_before_and_after':p['inputs'],
           'saved_qualification_checked_before_and_after':before,
           'native_baseline_turns_exact':baseline_turns,'fixed_development_cells':4,'contrasts_per_cell':5,
           'references_per_cell':2,'scorers_per_contrast':2,'score_reports_replayed_exactly':scores_replayed,
           'scorer_controls_replayed':4,'all_result_turns_and_deltas_replayed':True,
           'training_exposed_development':True,'original_gate_status_changed':False,
           'production_policy_selected':False,'release_qualified':False,'limits':p['limits']}
(output / 'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files = {path.name:desc(path) for path in output.iterdir() if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(output,len(files)+1,'members',scores_replayed,'score reports',baseline_turns,'baseline turns')
