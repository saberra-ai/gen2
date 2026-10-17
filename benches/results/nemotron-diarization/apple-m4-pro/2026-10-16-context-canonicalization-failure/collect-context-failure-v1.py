"""Preserve V1's two completed native cells and public pre-model failure."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
root=cache/'context-development-diagnostic-v1'
read=lambda path:json.loads(path.read_text())
def desc(path):
    data=Path(path).read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
p=read(root/'protocol.json');q=read(root/'qualification.json')
source='aaa97b587852439e5cb932e7c7aedf498c2deeae'
assert q['source_commit']==source and q['diagnostic_completed'] is False
for name,identity in p['tools'].items():assert desc(tools/name)==identity
assert (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/context-development-diagnostic.protocol.json'],cwd=repo)
sys.path.insert(0,str(tools))
import numpy as np
from ami_prefix_metrics import load_readonly_scorer
from compare import segments
from context_development_metrics import production_turns
from diagnose_ami_reference import calibrate_engine,official_der
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from score import der
from sdk_archive import inventory
verify_git_sources(repo,source,{'tools/nemotron-diarization/'+name:identity for name,identity in p['tools'].items()})
work=Path(p['work_path']);assert inventory(work)==p['work_inventory']
assert [row['name'] for row in q['commands']]==p['command_order'][:7]
assert [row['returncode'] for row in q['commands']]==[0,0,0,1,0,0,1]
assert q['neural_commands_started']==p['command_order'][4:7]
assert q['storage']['released'] is True
manifest=read(work/'nemo/package.json')
for name,saved in q['loaded_images'].items():
    exe=work/'bin'/('load-probe' if name=='version-only' else 'native-context')
    assert audit_loaded_libraries((root/(name+'.stderr.log')).read_text(),work/'nemo',manifest,exe)==saved
assert len(q['loaded_images'])==4
failed=(root/'Mix-Headset-public-low.stderr.log').read_text()
assert 'Diarization(InvalidBundle("Operation not permitted (os error 1)"))' in failed
zero=audit_loaded_libraries(failed,work/'nemo',manifest,work/'bin/public-stream',require_runtime=False)
assert zero['private_library_count']==0
engine=load_readonly_scorer(work/'scoring/md_eval.py',p['independent_scorer'])
assert calibrate_engine(engine)==read(root/'scorer-calibrations.json')
refs={name:read(work/'references'/(name+'.json')) for name in ['transcriber','published-ntt-mfa']}
for profile in p['geometries']:
    name='Mix-Headset-'+profile;value=read(root/'results'/(name+'.json'))
    values=np.frombuffer((root/'results'/(name+'.json.f32')).read_bytes(),dtype='<f4').reshape(30000,8)
    assert value['input_pcm_f32le_sha256']==p['pcm_f32le_sha256']['Mix-Headset']
    assert value['geometry']==p['geometries'][profile] and np.isfinite(values).all()
    hypotheses={'shared':segments(values),'production':production_turns(value['turns'])}
    scores={policy:{mode:{'frame':der(reference,hyp,300.),'official':official_der(engine,reference,hyp,300.)}
        for mode,hyp in hypotheses.items()} for policy,reference in refs.items()}
    assert scores==q['scores'][name]
diagnosis=read(cache/'context-cache-policy-diagnosis-v1.json');assert diagnosis['model_calls']==0
assert diagnosis['records']['original']['stdout']['directory']['success'] is False
assert diagnosis['records']['data_only']['stdout']['directory']['success'] is True
output=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-context-canonicalization-failure'
assert not output.exists();output.mkdir()
for path in [root/'protocol.json',root/'qualification.json',root/'scorer-calibrations.json',
    *root.glob('*.stdout.log'),*root.glob('*.stderr.log'),*root.glob('*.sb'),* (root/'results').glob('*.json'),
    Path(__file__),cache/'context-development-diagnostic-v1.controller.log',cache/'context-cache-policy-diagnosis-v1.json',
    cache/'diagnose-context-cache-policy-v1.py',cache/'context-cache-metadata-control.sb']:
    shutil.copy2(path,output/path.name)
summary={'schema_version':1,'source_commit':source,'diagnostic_completed':False,
    'inference_commands_admitted':3,'completed_native_neural_processes':2,'public_failed_before_native_loading':True,
    'failed_public_private_images':zero,'strict92image_audits_replayed':4,'all16_partial_scores_replayed':True,
    'original_staged_inputs_unchanged':True,'new_collection_model_calls':0,'model_free_libc_diagnosis_verified':True,
    'arrays':{path.name:desc(path) for path in (root/'results').glob('*.f32')},
    'release_qualified':False,'original_gate_status_changed':False,
    'limits':['Partialfirstchannelresults only; no completed publicparity or distantcell. Failurestop respected, reserve released. OriginalV1outputs/protocol/tools unchanged. SeparatepreregisteredV2required; no automaticretry or qualitypromotion.']}
(output/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files={path.name:desc(path) for path in output.iterdir() if path.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(output,len(files)+1,'members')
