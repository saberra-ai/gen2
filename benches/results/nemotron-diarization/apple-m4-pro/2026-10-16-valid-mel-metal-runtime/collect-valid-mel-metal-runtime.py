"""Replay saved Metal package qualification arrays/traces; never execute native code."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
src=cache/'valid-mel-metal-package-qualification-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-runtime'
def read(p):return json.loads(p.read_text())
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
p=read(src/'protocol.json');r=read(src/'result.json');commit=r['source_commit']
assert commit==subprocess.check_output(['git','rev-parse','54f61b3'],cwd=repo,text=True).strip()
assert rec(src/'protocol.json')['sha256']==r['protocol_sha256']=='eff1c4e161da9ab63fc32a038cc65841ebbc9d625edbbd988ea44d883d40a267'
assert subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/valid-mel-metal-package.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
 for name,expected in records.items():
  data=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo)
  assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==rec(root/name)==expected
for name,expected in p['inputs'].items():assert rec(Path(name))==expected
for name,expected in p['tree_files'].items():
 root=Path(name);assert {str(q.relative_to(root)):{**rec(q),'symlink_target':str(q.readlink())if q.is_symlink()else None}for q in root.rglob('*')if q.is_file()}==expected
commands=read(src/'commands.json')
assert [c['name']for c in commands]==['version-only','missing-library','native','public']+['lifecycle-'+str(i)for i in range(7)]+['cpu-metal-unavailable','combined-cpu-metal']
for c in commands:
 assert c['returncode']==(1 if c['name']in ['missing-library','cpu-metal-unavailable']else 0)
 assert not c.get('timed_out')and not c.get('forced_process_group_termination')and c['timeout_seconds']==180
 assert c['argv'][:5]==['/usr/bin/sandbox-exec','-f',str(src/'runtime.sb'),'/usr/bin/env','DYLD_PRINT_LIBRARIES=1']
outer=read(Path(str(src)+'.qualification.command.json'));assert outer['returncode']==0
for case in p['preflights']:
 prefix=str(src)+'.'+case['name'];assert read(Path(prefix+'.command.json'))['returncode']==0
 log=Path(prefix+'.stderr.log').read_text();assert 'Ran '+str(case['tests'])+' tests'in log and '\nOK\n'in log
sys.path.insert(0,str(tools))
from qualify_valid_mel_macos_package import compare_complete_60s
from macos_loaded_libraries import audit_loaded_libraries
from macos_multiple_loaded_libraries import audit_multiple_loaded_libraries
work=Path(p['work_path']);package=work/'nemo';manifest=read(package/'package.json');cpu=Path(p['cpu_package_path']);cpu_manifest=read(cpu/'package.json')
native=read(src/'native.json');public=read(src/'public.json');raw=(src/'native.json.f32').read_bytes();actual=(src/'public.f32').read_bytes()
comparison=compare_complete_60s(public,native,raw,actual);assert comparison==r['comparison']==read(src/'comparison.json')
assert public['backend']=='metal' and public['device']==p['expected_metal_device']=='Apple M4 Pro' and native['requested_gpu_index']==0
assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']==p['pcm_sha256']
loaded={}
for c in commands:
 name=c['name']
 if name=='missing-library':continue
 trace=(src/(name+'.stderr.log')).read_text()
 if name=='combined-cpu-metal':loaded[name]=audit_multiple_loaded_libraries(trace,{'cpu':(cpu,cpu_manifest),'metal':(package,manifest)},c['argv'][5])
 else:
  selected,selected_manifest=(cpu,cpu_manifest)if name=='cpu-metal-unavailable'else(package,manifest)
  loaded[name]=audit_loaded_libraries(trace,selected,selected_manifest,c['argv'][5],name!='lifecycle-5')
assert loaded==r['loaded_image_audits']==read(src/'loaded-libraries.json')and len(loaded)==12
records=sum(value['private_library_count']if 'private_library_count'in value else sum(v['private_library_count']for v in value.values())for value in loaded.values())
assert records==1114
for index,test in enumerate(p['lifecycle_tests']):
 text=(src/('lifecycle-'+str(index)+'.stdout.log')).read_text()
 assert re.search(r'^test '+re.escape(test)+r' \.\.\. ok$',text,re.MULTILINE)and '1 passed; 0 failed; 0 ignored;'in text
text=(src/'combined-cpu-metal.stdout.log').read_text();assert re.search(r'^test '+re.escape(p['combined_metal_test'])+r' \.\.\. ok$',text,re.MULTILINE)and '1 passed; 0 failed; 0 ignored;'in text
assert (src/'version-only.stdout.log').read_text().strip()==p['runtime_version']
assert not (src/'missing-library.stdout.log').stat().st_size and 'Library not loaded:'in(src/'missing-library.stderr.log').read_text()and Path(p['missing_library_path']).name in(src/'missing-library.stderr.log').read_text()
assert not (src/'cpu-metal-unavailable.stdout.log').stat().st_size and not(src/'unexpected-cpu.f32').exists()and 'requested Metal device is unavailable'in(src/'cpu-metal-unavailable.stderr.log').read_text()
assert r['fixed_packaged_metal_parity_and_lifecycle_passed']and r['explicit_metal_cpu_sdk_rejected_without_fallback']and r['combined_cpu_rejection_and_metal_parity_passed']and not r['release_qualified']
assert not out.exists();out.mkdir()
for q in src.rglob('*'):
 if q.is_file()and not q.is_relative_to(src/'tmp'):
  dest=out/q.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dest)
for q in src.parent.glob(src.name+'.*'):
 if q.is_file():shutil.copy2(q,out/('outer'+q.name.removeprefix(src.name)))
for root in [work.parent,cache/'valid-mel-macos-load-probe-v1']:
 target=out/'preparation'/root.name;target.mkdir(parents=True)
 for q in root.iterdir():
  if q.is_file()and q.name!='load-probe':shutil.copy2(q,target/q.name)
for name in ['run-valid-mel-metal-package.py','freeze-valid-mel-metal-package.py','collect-valid-mel-metal-runtime.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':commit,'protocol_sha256':r['protocol_sha256'],'fixed_packaged_metal_parity_and_lifecycle_passed':True,'comparison':comparison,'all6000rows_native_public_byte_identical':raw==actual,'metal_private_images_per_process':93,'cpu_private_images_in_negative_process':92,'combined_cpu_metal_image_records':185,'audited_private_image_records':records,'audited_process_traces':12,'warm_provisioning_native_images':0,'missing_private_dependency_failed_before_model_creation':True,'explicit_metal_cpu_sdk_rejected_without_fallback':True,'combined_cpu_rejection_and_metal_parity_passed':True,'lifecycle_tests_passed':p['lifecycle_tests'],'combined_test_passed':p['combined_metal_test'],'actual100reload_test_passed':True,'model_free_preflight_tests_passed':31,'qualification_elapsed_seconds':outer['elapsed_seconds'],'neural_inference_performed':True,'original_quality_gate_changed':False,'release_qualified':False,'limits':p['limits']+['Collection uses saved arrays, source/input hashes, original dyld traces and test outputs only. No native query/load, helper, model or inference rerun.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(q.relative_to(out)):rec(q)for q in sorted(out.rglob('*'))if q.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files),'comparison':comparison,'private_image_records':records,'lifecycle_tests':7,'combined_test':True}))
