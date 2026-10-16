"""Replay saved guarded1h pair plus prior completed30m evidence; no native calls."""
import hashlib,json
from pathlib import Path
import shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
src=cache/'valid-mel-metal-long-v2';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long'
def read(p):return json.loads(p.read_text())
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
p=read(src/'protocol.json');r=read(src/'result.json');commit=r['source_commit']
assert commit==subprocess.check_output(['git','rev-parse','6265017'],cwd=repo,text=True).strip()
assert rec(src/'protocol.json')['sha256']==r['protocol_sha256']=='f0efc694866c5fc2c948c153ab4f25bd9d70738d2c5283460944798682e68538'
assert subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/valid-mel-metal-long-v2.protocol.json'],cwd=repo)==(src/'protocol.json').read_bytes()
for records,root,prefix in [(p['tools'],tools,'tools/nemotron-diarization/'),(p['rust_sources'],repo,'')]:
 for name,expected in records.items():
  d=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo)
  assert {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}==rec(root/name)==expected
for name,expected in p['inputs'].items():assert rec(Path(name))==expected
work=Path(p['work_path']);assert {str(q.relative_to(work)):{**rec(q),'symlink_target':str(q.readlink())if q.is_symlink()else None}for q in work.rglob('*')if q.is_file()}==p['work_files']
commands=read(src/'commands.json');assert [c['name']for c in commands]==['60m-native','60m-public']
assert all(c['returncode']==0 and not c.get('execution_error')and not c.get('forced_process_group_termination')and not c.get('disk_pressure_abort_free_bytes')for c in commands)
assert all(c['disk_free_bytes_at_start']>=5*1024**3 for c in commands)
outer=read(Path(str(src)+'.qualification.command.json'));assert outer['returncode']==0
for case in p['preflights']:
 prefix=str(src)+'.'+case['name'];assert read(Path(prefix+'.command.json'))['returncode']==0
 text=Path(prefix+'.stderr.log').read_text();assert 'Ran '+str(case['tests'])+' tests'in text and '\nOK\n'in text
sys.path.insert(0,str(tools))
from valid_mel_long_metrics import audit_case
from macos_loaded_libraries import audit_loaded_libraries
prior_path=Path(p['prior_failed_attempt_path']);prior=read(prior_path);assert prior['completed_30m_pair_passed']and not prior['one_hour_qualified']and not prior['fixed_metal_30m_1h_compaction_parity_passed']
old_protocol=read(prior_path.parent/'protocol.json')
comparisons={}
for root,case,name in [(prior_path.parent,old_protocol['cases'][0],'30m'),(src,p['cases'][0],'60m')]:
 public=read(root/(name+'-public.json'));native=read(root/(name+'-native.json'))
 assert native['requested_gpu_index']==0
 comparison=audit_case(public,native,(root/(name+'-native.json.f32')).read_bytes(),(root/(name+'-public.f32')).read_bytes(),(root/(name+'-native.stderr.log')).read_text(),(root/(name+'-public.stderr.log')).read_text(),case,backend='metal',device='Apple M4 Pro')
 assert comparison==read(root/(name+'-comparison.json'))==r['comparisons'][name]
 comparisons[name]=comparison
manifest=read(work/'nemo/package.json');loaded={c['name']:audit_loaded_libraries((src/(c['name']+'.stderr.log')).read_text(),work/'nemo',manifest,c['argv'][7])for c in commands}
assert loaded==r['loaded_image_audits']==read(src/'loaded-libraries.json')and sum(v['private_library_count']for v in loaded.values())==186
storage=r['storage_guard'];assert storage['reserve_reported_allocated_bytes']==storage['reserve_requested_bytes']==256*1024**2
assert storage['minimum_observed_free_bytes']>=1024**3 and storage['released'] and storage['checks']>=2
assert not(src/'diagnostic-reserve.bin').exists()and not(src/'failure.json').exists()and not list(src.glob('*.next'))
assert r['fixed_metal_30m_1h_compaction_parity_passed']and r['successful_streams_this_attempt']==2 and not r['release_qualified']
assert not out.exists();shutil.copytree(src,out)
for q in src.parent.glob(src.name+'.*'):
 if q.is_file():shutil.copy2(q,out/('outer'+q.name.removeprefix(src.name)))
for name in ['run-valid-mel-metal-long-v2.py','freeze-valid-mel-metal-long-v2.py','collect-valid-mel-metal-long-v2.py']:shutil.copy2(cache/name,out/name)
shutil.copy2(prior_path,out/'prior-failed-attempt-qualification.json')
q={'schema_version':1,'source_commit':commit,'protocol_sha256':r['protocol_sha256'],'fixed_metal_30m_1h_compaction_parity_passed':True,'successful_30m_pair_from_prior_attempt':True,'new_one_hour_pair_passed':True,'successful_native_public_streams_across_attempts':4,'new_streams_this_attempt':2,'failed_prior_native_command_repeated_as_new_complete_run':True,'prior_failed_attempt_preserved':True,'prior_failure_receipt':str(prior_path.relative_to(repo)),'prior_failure_qualification':rec(prior_path),'comparisons':comparisons,'actual_runtime_processes_audited_this_attempt':2,'private_images_per_process':93,'actual_private_image_records_this_attempt':186,'successful_runtime_processes_across_attempts':4,'successful_private_image_records_across_attempts':372,'model_free_preflight_tests_passed':35,'qualification_elapsed_seconds':outer['elapsed_seconds'],'storage_guard':storage,'neural_inference_performed':True,'original_quality_gate_changed':False,'release_qualified':False,'limits':p['limits']+['The30m pair remains in the preserved failed-attempt receipt. This receipt reruns only saved-array audits for it and adds two successful new1h streams. V1itself remains failed and its missing terminal status/tensors are not repaired or inferred. Collection executes no native query/helper/build/model/inference.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={str(q.relative_to(out)):rec(q)for q in sorted(out.rglob('*'))if q.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files),'comparisons':comparisons,'new_native_calls_in_collection':0,'new_streams_qualified':2,'prior_30m_pair_reused':True,'storage_checks':storage['checks']}))
