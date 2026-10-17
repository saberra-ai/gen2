from pathlib import Path
import hashlib
import json
import struct

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
root = cache / 'context-development-diagnostic-v2'
old = json.loads((root / 'protocol.json').read_text())
work = Path(old['work_path'])
read = lambda path: json.loads(Path(path).read_text())
def desc(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
f32 = lambda v: struct.unpack('<f',struct.pack('<f',v))[0]
default = {'onset':f32(.641),'offset':f32(.561),'pad_onset':.229,'pad_offset':.079,
           'min_duration_on':.511,'min_duration_off':.296}
contrasts = {'native_default':default,
    'without_edge_padding':{**default,'pad_onset':0.,'pad_offset':0.},
    'without_gap_fill':{**default,'min_duration_off':0.},
    'without_short_segment_filter':{**default,'min_duration_on':0.},
    'shared_thresholds_only':{**default,'onset':.5,'offset':.5}}
qualification = root / 'qualification.json'
collection = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-development-diagnostic/collection-qualification.json'
inputs = [qualification,root / 'protocol.json',collection,
    collection.parent / 'collection.json',work / 'scoring/md_eval.py',
    *(work / 'references' / (name+'.json') for name in ['transcriber','published-ntt-mfa'])]
for channel in old['channels']:
    for profile in old['geometries']:
        inputs.extend(root / 'results' / (channel+'-'+profile+suffix) for suffix in ['.json','.json.f32'])
native = cache / 'NeMo-Speech-valid-mel-v1/src/asr/diar'
inputs.extend(native / name for name in ['diar_pipeline.cpp','diar_pipeline.h','sortformer_model.h'])
p = {'schema_version':1,'meeting':'ES2011a','crop_seconds':[0,300],
    'channels':old['channels'],'profiles':list(old['geometries']),
    'references':['transcriber','published-ntt-mfa'],'contrasts':contrasts,
    'prior_qualification':str(qualification),'prior_collection':str(collection),
    'prior_work':str(work),'prior_results':str(root / 'results'),
    'independent_scorer':old['independent_scorer'],
    'output_path':str(cache / 'development-segmentation-v1'),
    'tools':{name:desc(tools / name) for name in ['diagnose_development_segmentation.py',
        'segmentation_ablation.py','test_segmentation_ablation.py','ami_prefix_metrics.py',
        'diagnose_ami_reference.py','compare.py','score.py','artifacts.py','frozen_git_sources.py']},
    'inputs':{str(path):desc(path) for path in inputs},
    'limits':[
        'Saved-output operation ablations on four fixed ES2011a 300-second development cells only. Zero model/native calls, no audio acquisition, no new inference or threshold sweep. Every baseline turn/boundary/speaker and previous frame/official score must reproduce before reporting contrasts.',
        'Remove one default operation per contrast, or change only the documented onset/offset pair to 0.5 while retaining all smoothing. Report every cell, reference policy, contrast and scorer, full error components and deltas. Operation interactions preclude monotonic or additive causal claims. No winner or production policy selected.',
        'Preserve original failed ES2004 references/gates. This training-exposed development meeting has weak speaker-B coverage; MFA lexical labels are automatic and transcriber occupancy has a distinct scope. No human word gold, held-out generalization, new wrapper behavior, changed-policy streaming compaction, acoustic label timing, GUI/microphone, throughput or release claim.',
        'Bind every imported local tool before import and all input JSON/raw/source/scorer descriptors before and after replay; verify committed protocol and historical tool blobs before scoring. Preserve exact native float32 threshold and cadence arithmetic, strict boundary comparisons and pad/merge/filter order. Equal-start ordering alone is canonicalized.']}
target = tools / 'development-segmentation.protocol.json'
assert not target.exists()
target.write_text(json.dumps(p,indent=2)+'\n')
print(desc(target),len(p['tools']),'tools',len(p['inputs']),'inputs; no output scored')
