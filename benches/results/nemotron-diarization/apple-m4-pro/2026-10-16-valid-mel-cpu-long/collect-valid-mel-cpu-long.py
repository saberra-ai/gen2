"""Replay saved CPU long-session metrics and loader traces; never execute native code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
src=cache/'valid-mel-cpu-long-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-cpu-long'
def read(p):return json.loads(p.read_text())
def rec(p):
    d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
p=read(src/'protocol.json');r=read(src/'result.json');commit=r['source_commit']
assert commit==subprocess.check_output(['git','rev-parse','1f62fd5'],cwd=repo,text=True).strip()
assert rec(src/'protocol.json')['sha256']==r['protocol_sha256']=='126b18708d8be2b4bbaf7f20ca56e75cef0bc71ef814bd4f0b5c6eb942611cb6'
assert subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/valid-mel-cpu-long.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
    for name,expected in records.items():
        d=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo)
        assert {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}==expected==rec(root/name)
for name,expected in p['inputs'].items():assert rec(Path(name))==expected
work=Path(p['work_path']);actual={str(q.relative_to(work)):{**rec(q),'symlink_target':str(q.readlink())if q.is_symlink()else None}for q in work.rglob('*')if q.is_file()}
assert actual==p['work_files']
commands=read(src/'commands.json')
assert [c['name']for c in commands]==['30m-native','30m-public','60m-native','60m-public']
assert all(c['returncode']==0 and not c.get('execution_error') and not c.get('forced_process_group_termination') for c in commands)
for c in commands:
    assert c['argv'][:7]==['/usr/bin/time','-l','/usr/bin/sandbox-exec','-f',str(src/'runtime.sb'),'/usr/bin/env','DYLD_PRINT_LIBRARIES=1']
outer=read(Path(str(src)+'.qualification.command.json'));assert outer['returncode']==0
for case in p['preflights']:
    prefix=Path(str(src)+'.'+case['name'])
    assert read(Path(str(prefix)+'.command.json'))['returncode']==0
    log=Path(str(prefix)+'.stderr.log').read_text()
    assert 'Ran '+str(case['tests'])+' tests' in log and '\nOK\n' in log
sys.path.insert(0,str(tools))
from valid_mel_long_metrics import audit_case
from macos_loaded_libraries import audit_loaded_libraries
manifest=read(work/'nemo/package.json');loaded={};comparisons={}
for case in p['cases']:
    name=case['name'];native=read(src/(name+'-native.json'));public=read(src/(name+'-public.json'))
    comparison=audit_case(public,native,(src/(name+'-native.json.f32')).read_bytes(),(src/(name+'-public.f32')).read_bytes(),(src/(name+'-native.stderr.log')).read_text(),(src/(name+'-public.stderr.log')).read_text(),case)
    assert comparison==read(src/(name+'-comparison.json'))==r['comparisons'][name]
    comparisons[name]=comparison
for c in commands:
    name=c['name'];loaded[name]=audit_loaded_libraries((src/(name+'.stderr.log')).read_text(),work/'nemo',manifest,c['argv'][7])
assert loaded==r['loaded_image_audits']==read(src/'loaded-libraries.json')
assert len(loaded)==4 and sum(v['private_library_count']for v in loaded.values())==368
assert r['fixed_cpu_30m_1h_compaction_parity_passed'] and r['neural_inference_performed'] and not r['release_qualified']
assert not out.exists();shutil.copytree(src,out)
for q in src.parent.glob(src.name+'.*'):
    if q.is_file():shutil.copy2(q,out/('outer'+q.name.removeprefix(src.name)))
for name in ['run-valid-mel-cpu-long.py','collect-valid-mel-cpu-long.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':commit,'repository_head_at_completion':r['repository_head_at_completion'],'protocol_sha256':r['protocol_sha256'],'fixed_cpu_30m_1h_compaction_parity_passed':True,'comparisons':comparisons,'runtime_private_images_per_process':92,'actual_runtime_processes_audited':4,'audited_private_image_records':368,'model_free_preflight_tests_passed':30,'neural_stream_runs':4,'qualification_elapsed_seconds':outer['elapsed_seconds'],'neural_inference_performed':True,'original_quality_gate_changed':False,'release_qualified':False,'limits':p['limits']+['Collection repeats only saved-array, metadata and dyld-trace audits. No native version, helper, model or inference rerun.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(q.relative_to(out)):rec(q)for q in sorted(out.rglob('*'))if q.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'members':len(files),'comparisons':comparisons,'audited_private_image_records':368}))
