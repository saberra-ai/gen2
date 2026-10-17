"""Collect saved candidate construction/control bytes, never execute native code."""
import hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory
from package_macos import verify_files,verify_package_provenance
from macos_loaded_libraries import audit_loaded_libraries
commit='eeae443e18b03715c6fa1ebf577a0f69de51bedd';m1=cache/'pooled-macos-construction-ci-v1';e=m1/'evidence';api=json.loads((m1/'download-identity.json').read_text());ci=json.loads((m1/'run.json').read_text());delivery=json.loads((m1/'delivery.json').read_text());q=json.loads((e/'qualification.json').read_text());p=json.loads((e/'protocol.json').read_text())
assert ci['id']==38984207454 and ci['head_sha']==commit and ci['status']=='completed' and ci['conclusion']=='success'
assert api['id']==12082417995 and api['workflow_run']['id']==ci['id'] and api['workflow_run']['head_sha']==commit
assert descriptor(m1/'artifact.zip')=={'bytes':17618400,'sha256':'9f2833d05cbabc4d3272af67af8f5b192e3f88d418976b8ebade9f7bf9e5479c'}==delivery['zip']
assert api['digest']=='sha256:'+delivery['zip']['sha256'] and api['size_in_bytes']==17618400
with zipfile.ZipFile(m1/'artifact.zip') as z:
 assert len(z.infolist())==len({x.filename for x in z.infolist()})==196
 assert {str(f.relative_to(e)) for f in e.rglob('*') if f.is_file()}==set(delivery['files'])
 for f in z.infolist():assert (e/f.filename).read_bytes()==z.read(f) and descriptor(e/f.filename)==delivery['files'][f.filename]
assert q['source_commit']==commit and q['construction_and_negative_c_api_controls_passed'] and q['valid_models_loaded']==q['frontend_or_neural_calls']==0 and q['release_qualified'] is False
assert [c['name'] for c in q['commands']]==['sandbox-controls','prepare','build','package','failed-load-probe-build','failed-loads'] and all(c['returncode']==0 for c in q['commands'])
assert (e/'protocol.json').read_bytes()==subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/macos-pooled-construction.protocol.json'],cwd=repo)
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for f,d in p[group].items():
  b=subprocess.check_output(['git','show',commit+':'+prefix+f],cwd=repo);assert d=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
for stem,count in [('pooled-cpu-sdk',84),('pooled-cpu-runtime',123),('construction-controls',2)]:
 manifest=json.loads((e/(stem+'-files.json')).read_text());assert inventory(m1/('restored-'+stem))==manifest['files'] and len(manifest['files'])==count
 assert descriptor(e/manifest['archive'])=={'bytes':manifest['archive_bytes'],'sha256':manifest['archive_sha256']}
package=m1/'restored-pooled-cpu-runtime';binary=m1/'restored-construction-controls/failed-load-probe';manifest=json.loads((package/'package.json').read_text());verify_files(package,manifest['files']);verify_package_provenance(package,manifest);assert descriptor(binary)==q['binary']
remote_exe=Path(q['loaded_images']['executable']);remote_package=Path(next(iter(q['loaded_images']['private_libraries']))).parent.parent
pattern=r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)'
def remap(m):
 f=m[2]
 if f.startswith(('/usr/lib/','/System/Library/')):return m[0]
 if f==str(remote_exe):return m[1]+str(binary)
 rel=Path(f).relative_to(remote_package/'lib');assert str(rel) in manifest['libraries'];return m[1]+str(package/'lib'/rel)
audit=audit_loaded_libraries(re.sub(pattern,remap,(e/'failed-loads.stderr.log').read_text()),package,manifest,binary);assert json.loads(json.dumps(audit).replace(str(package),str(remote_package)).replace(str(binary),str(remote_exe)))==q['loaded_images'] and audit['private_library_count']==92
r=q['failed_loads'];assert r['failed_model_construction_attempts']==101 and len(r['threads_after_each_failure'])==101 and set(r['threads_after_each_failure'])=={1,2}
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=1024**3
build=json.loads((e/'build/build.json').read_text());assert build['runtime_profile']=='gen2-valid-mel-pool-v1' and build['runtime_reported_version']=='nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1' and len(build['actual_compiler_sources'])==98
# Retain metadata/logs, and descriptors for all opaque payloads outsideGit.
def retain_files(source,dest):
 for f in source.rglob('*'):
  rel=f.relative_to(source)
  if f.is_file() and f.suffix in {'.json','.log','.sb','.patch','.txt','.cpp'} and 'work' not in rel.parts:
   target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
def finish(out,record):
 shutil.copy2(__file__,out/Path(__file__).name);(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n');files={str(f.relative_to(out)):descriptor(f) for f in out.rglob('*') if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(str(out),len(files)+1)
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-16-pooled-cpu-construction-v1';assert not out.exists();out.mkdir();retain_files(e,out)
for name in ['delivery.json','download-identity.json','run.json']:shutil.copy2(m1/name,out/name)
shutil.copy2(package/'package.json',out/'restored-runtime-package.json')
finish(out,{'schema_version':1,'source_commit':commit,'ci_url':ci['html_url'],'opaque_artifact':api,'delivered_members_independently_verified':196,'all_39_historical_source_pins_verified':True,'restored_inventory_entries':{'sdk':84,'runtime':123,'controls':2},'helper_binary_independently_rehashed':True,'private_libraries_rehashed_and_trace_replayed':92,'actual_compiler_units':98,'failed_construction_attempts':101,'observed_immediate_thread_counts':[1,2],'construction_and_negative_c_api_controls_passed':True,'valid_models_loaded':0,'frontend_or_neural_calls':0,'release_qualified':False,'limits':['Original immediate-snapshot control passed on this M1 VM. The separate M4 immediate-snapshot failure remains failed. Later V2 observes bounded settling; it does not retroactively relabel either result.','No candidate model inference, publicRust admission, numerical/quality/long/mixed/app/release qualification. SDK/runtime and helper opaque archives remain outsideGit and are bound by delivered manifests.']})
# Local construction's initial failed thread snapshot and its diagnosis remainfailed.
local=cache/'pooled-cpu-construction-local-v1';failure=json.loads((local/'evidence/qualification.json').read_text());assert not failure['construction_and_negative_c_api_controls_passed'] and failure['commands'][-1]['name']=='failed-loads' and failure['commands'][-1]['returncode']==1
assert (local/'evidence/failed-loads.stderr.log').read_text().count('gguf_init_from_file: failed to open')==26
control=cache/'pooled-cleanup-final-control-v3';v=json.loads((control/'qualification.json').read_text());sdk=local/'sdk';pkg=local/'runtime';assert inventory(sdk)==v['protected_inputs']['sdk'] and inventory(pkg)==v['protected_inputs']['package']
assert v['source']==descriptor(tools/'native_failed_load_probe.cpp') and [c['returncode'] for c in v['commands']]==[0,0,0,1]
assert json.loads((control/'failed-loads.stdout.log').read_text())==v['result']
fault_text=(control/'fault-observations.stderr.log').read_text();assert 'forced observation diagnostic' in fault_text and not (control/'fault-observations.stdout.log').read_bytes()
lines=[line for line in fault_text.splitlines() if line.startswith('failed_load_thread_observations:')];assert len(lines)==1
for label,next_label in [('immediate=',' settled='),('settled=',' requested_wait_ms='),('requested_wait_ms=',None)]:
 part=lines[0].split(label,1)[1]
 if next_label:part=part.split(next_label,1)[0]
 assert len([int(value) for value in part.split(',') if value])==5
cpp=(tools/'native_failed_load_probe.cpp').read_text();needle='            waits.push_back(requested_wait);';assert cpp.count(needle)==1
assert (control/'fault.cpp').read_text()==cpp.replace(needle,needle+'\n            if (index == 4) throw std::runtime_error("forced observation diagnostic");')
assert descriptor(control/'fault.cpp')==v['fault_injection']['source']
local_manifest=json.loads((pkg/'package.json').read_text());verify_files(pkg,local_manifest['files']);verify_package_provenance(pkg,local_manifest)
for name,record in [('failed-loads',v['loaded_images']),('fault-observations',v['fault_injection']['loaded_images'])]:
 executable=Path(record['executable']);expected_binary=v['binary'] if name=='failed-loads' else v['fault_injection']['binary'];assert descriptor(executable)==expected_binary
 assert audit_loaded_libraries((control/(name+'.stderr.log')).read_text(),pkg,local_manifest,executable)==record and record['private_library_count']==92
diag=(cache/'pooled-cleanup-diagnostic-v1/diagnostic.stderr.log').read_text();assert '[DEBUG-pool-cleanup] failure_index=6 baseline=1 count=4 history=1,2,3,3,3,3,4,' in diag
assert [(int(a),int(b)) for a,b in re.findall(r'\[DEBUG-pool-cleanup\] quiet_millis=(\d+) count=(\d+)',diag)]==[(10*i,1) for i in range(1,11)]
assert '[DEBUG-pool-cleanup] thread=0 status=0 name= run_state=1' in diag
r=v['result'];assert r['failed_model_construction_attempts']==101 and len(r['immediate_thread_counts'])==len(r['threads_after_each_failure'])==len(r['requested_settling_wait_milliseconds'])==101 and set(r['threads_after_each_failure'])=={1} and set(r['requested_settling_wait_milliseconds'])=={10} and v['fault_injection']['all_three_partial_histories_retained']
assert r['thread_growth_allowance']==2 and r['maximum_requested_settling_wait_milliseconds']==100
newp=json.loads((tools/'macos-pooled-construction-v2.protocol.json').read_text())
for name,d in newp['tools'].items():assert descriptor(tools/name)==d
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-cpu-cleanup';assert not out.exists();out.mkdir();retain_files(local/'evidence',out/'initial-construction-failure')
for label,name in [('failure-diagnostic','pooled-cleanup-diagnostic-v1'),('settled-continuation','pooled-cleanup-settled-control-v2'),('final-control','pooled-cleanup-final-control-v3')]:retain_files(cache/name,out/label)
for name in ['run-pooled-cleanup-diagnostic-v1.py','run-pooled-cleanup-final-control-v3.py']:shutil.copy2(cache/name,out/name)
shutil.copy2(tools/'macos-pooled-construction-v2.protocol.json',out/'v2-protocol.json')
finish(out,{'schema_version':1,'sdk_build_source_commit':commit,'initial_full_sdk_and_package_build_passed':True,'initial_full_construction_control_result':'failed; immediate task_threads growth guard after26missing-model attempts','initial_failure_relabelled':False,'failure_diagnostic':{'baseline':1,'immediate':4,'observation_index':6,'after10milliseconds':1,'through100milliseconds':1,'persistent_worker_leak_established':False},'corrected_final_control':{'failed_model_attempts':101,'immediate_thread_counts':sorted(set(r['immediate_thread_counts'])),'settled_thread_counts':[1],'requested_wait_ms':[10],'growth_allowance':2,'maximum_requested_wait_ms':100,'five_observation_fault_diagnostic_retains_all_three_partial_histories':True},'same_original_sdk_and_package_inventories_verified_unchanged':True,'final_control_source':v['source'],'v2_protocol':descriptor(tools/'macos-pooled-construction-v2.protocol.json'),'valid_models_loaded':0,'frontend_or_neural_calls':0,'release_qualified':False,'limits':['No SDK/ownership/runtime behavior changed to correct the observation guard. Final helper source/protocol are byte-bound independently of the earlierSDKbuild commit.','The immediate4-to-quiet1 observations are consistent with transient kernel-thread visibility after worker teardown, rather than persistent growth. Both immediate and settled observations are preserved; the unchanged warmup+2bound is checked after10..100msrequested sleep.','The first failed construction remains failed; subsequent controls reuse its successfullybuilt SDK/package. This is no neural/numerical/publicRust/long/quality/app/release qualification.']})
