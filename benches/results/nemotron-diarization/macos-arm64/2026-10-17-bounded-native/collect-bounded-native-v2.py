"""Replay saved C ABI evidence; execute no native helper or model."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys,zipfile
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'bounded-native-ci-v2';saved=root/'evidence';source='05c5ac1535a38e76b258ca57749bea820a7dc14f';read=lambda path:json.loads(Path(path).read_text())
def desc(path):
 raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p=read(saved/'protocol.json')
for names,base in [(p['tools'],tools),(p['repository_files'],repo)]:
 for name,expected in names.items():assert desc(base/name)==expected,name
sys.path.insert(0,str(tools))
from bounded_cpu_metrics import VERSION,derive_helper,compare_bounded_history,validate_failed_loads
from cpu_thread_metrics import used_dependencies
from frozen_git_sources import verify_git_sources
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files,verify_package_provenance
from sdk_archive import inventory,restore_archive
verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
assert (saved/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-bounded-native-v2.protocol.json'],cwd=repo)
ci,api=read(root/'run.json'),read(root/'artifacts.json')['artifacts'];assert len(api)==1;a=api[0]
assert ci['databaseId']==39004916708 and ci['headSha']==source and ci['status']=='completed' and ci['conclusion']=='success'
assert a['id']==12090213937 and a['name']=='nemotron-macos-bounded-native' and a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==source
assert desc(root/'artifact.zip')=={'bytes':403280,'sha256':'b2c02a333862370620a710f7dd2ace7e9c149f6588fdbf3749ed7db64af28b53'}
assert a['size_in_bytes']==403280 and a['digest']=='sha256:'+desc(root/'artifact.zip')['sha256']
before={str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
with zipfile.ZipFile(root/'artifact.zip') as archive:
 assert len(archive.infolist())==len(before)==44 and {m.filename for m in archive.infolist()}==set(before)
 for name in before:assert archive.read(name)==(saved/name).read_bytes()
assert not any(path.is_symlink() for path in saved.rglob('*'))
q=read(saved/'qualification.json');assert q['source_commit']==source and q['admit_only'] is False and q['deliveries_admitted'] is True
assert q['native_parity_passed'] is True and q['realtime_gate_passed'] is True and q['release_qualified'] is False and 'error' not in q
assert [c['name'] for c in q['commands']]==p['preparation_order']+p['neural_order'] and len(q['commands'])==10
assert q['neural_commands_started']==p['neural_order']==['native-12s','native-60s']
for row in q['commands']:
 assert row['returncode']==0 and row['inference_admitted']==(row['name'] in p['neural_order']) and row['argv'][:2]==['/usr/bin/sandbox-exec','-f']
 assert not row.get('forced_process_group_termination') and not row.get('disk_pressure_abort_free_bytes')
remote=Path(q['commands'][0]['cwd'])
producer=cache/'bounded-macos-construction-ci-v1';reference=cache/'pooled-native-ci-v1'
for key,prior in [('producer',producer),('native_reference',reference)]:
 assert desc(prior/'artifact.zip')==p[key]['zip']
 assert {str(path.relative_to(prior/'evidence')):desc(path) for path in (prior/'evidence').rglob('*') if path.is_file()}==p[key]['files']
 api=q['deliveries'][key];assert api['id']==p[key]['id'] and api['name']==p[key]['name'] and api['workflow_run']['id']==p[key]['run_id']
 assert api['workflow_run']['head_sha']==p[key]['source_commit'] and api['digest']=='sha256:'+p[key]['zip']['sha256']
sdk,package=producer/'restored-pooled-cpu-sdk',producer/'restored-pooled-cpu-runtime'
manifest=read(saved/'package.json');assert manifest==read(package/'package.json') and manifest['runtime_reported_version']==VERSION
verify_files(package,manifest['files']);verify_package_provenance(package,manifest)
helpers=root/'replayed-helpers';archive=read(saved/'bounded-native-helpers-files.json')
assert {k:v for k,v in archive.items() if k!='files'}==q['helper_archive'] and not helpers.exists()
restore_archive(saved/archive['archive'],archive,helpers)
protected=read(saved/'protected-inputs.json')
replayed={str(remote/'producer'):producer/'evidence',str(remote/'native_reference'):reference/'evidence',str(remote/'sdk'):sdk,
          str(remote/'work/nemo'):package,str(remote/'work/bin'):helpers}
assert set(protected)==set(replayed)|{str(remote/'helper-source'),str(remote/'work/assets')}
for name,path in replayed.items():assert inventory(path)==protected[name],name
source_root=saved/'helper-source';expected_sources=protected[str(remote/'helper-source')]
actual_sources=inventory(source_root)
assert set(actual_sources)==set(expected_sources)
for name,row in expected_sources.items():
 assert {k:v for k,v in row.items() if k!='mode'}=={k:v for k,v in actual_sources[name].items() if k!='mode'}
assert (source_root/'native.cpp').read_text()==derive_helper((tools/'native_stream_probe_valid_mel.cpp').read_text())
assert desc(source_root/'native.cpp')==p['derived_helper'] and desc(source_root/'failed.cpp')==p['tools']['native_failed_load_probe.cpp']
assert {path.relative_to(source_root).as_posix() for path in source_root.rglob('*') if path.is_file()}==set(p['helper_inputs'])|{'native.cpp','failed.cpp'}
for name,entry in p['helper_inputs'].items():
 assert desc(source_root/name)==q['helper_acquisitions'][name]=={'bytes':entry['size'],'sha256':entry['sha256']}
used=set().union(*(used_dependencies((saved/(name+'.d')).read_text()) for name in ['native','sha','failed']))
assert used==set(q['used_compiler_inputs']) and len(used)==8
for path,expected in q['used_compiler_inputs'].items():
 path=Path(path)
 if path.is_relative_to(remote/'sdk'):local=sdk/path.relative_to(remote/'sdk')
 else:
  assert path.is_relative_to(remote/'helper-source');local=source_root/path.relative_to(remote/'helper-source')
 assert desc(local)==expected
assert validate_failed_loads(read(saved/'failed-loads.stdout.log'))==q['failed_load_control']
for name in ['failed-loads',*p['neural_order']]:
 executable=helpers/('failed' if name=='failed-loads' else 'native');remote_executable=remote/'work/bin'/executable.name
 def remap(match):
  path=match[2]
  if path.startswith(('/usr/lib/','/System/Library/')):return match[0]
  if path==str(remote_executable):return match[1]+str(executable)
  relative=Path(path).relative_to(remote/'work/nemo/lib');assert relative.as_posix() in set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib'}
  return match[1]+str(package/'lib'/relative)
 trace=(saved/(name+'.stderr.log')).read_text()
 audited=audit_loaded_libraries(re.sub(r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)',remap,trace),package,manifest,executable)
 assert json.loads(json.dumps(audited).replace(str(executable),str(remote_executable)).replace(str(package),str(remote/'work/nemo')))==q['loaded_images'][name]
 assert audited['private_library_count']==92
assert q['acquisitions']==[{'asset':entry['name'],'bytes':entry['size'],'sha256':entry['sha256']} for entry in p['assets']]
assets=protected[str(remote/'work/assets')];assert set(assets)=={entry['name'] for entry in p['assets']}|{'.','crop-12s.wav'}
for entry in p['assets']:assert {key:assets[entry['name']][key] for key in ['bytes','sha256']}=={'bytes':entry['size'],'sha256':entry['sha256']}
assert assets['crop-12s.wav']['bytes']==384044
trials=[]
for name in p['neural_order']:
 value=read(saved/'results'/(name+'.json'));raw=(saved/'results'/(name+'.json.f32')).read_bytes();expected=reference/'evidence/results'/(name+'.json')
 result=compare_bounded_history(value,raw,read(expected),Path(str(expected)+'.f32').read_bytes());assert result==q['comparisons'][name] and result['warm_rtf']<p['require_warm_rtf_below']==1.0
 trace=(saved/(name+'.stderr.log')).read_text();usage=re.search(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$',trace,re.M)
 wall,user,system=map(float,usage.groups());rss=int(re.search(r'(\d+)\s+maximum resident set size',trace)[1])
 trials.append({'name':name,**result,'time_l':{'real_seconds':wall,'user_seconds':user,'system_seconds':system,'peak_rss_bytes':rss}})
assert q['result_files']=={path.name:desc(path) for path in (saved/'results').iterdir()}
host=read(saved/'host.json');assert host['cpu']==p['expected_receiver_cpu'] and host['cpu_counts']==p['expected_cpu_counts']
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and q['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
assert before=={str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-native';assert not out.exists();out.mkdir()
for name in before:
 if name.endswith(('.f32','.tar.gz')):continue
 target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(saved/name,target)
for path in [root/'run.json',root/'artifacts.json',Path(__file__),cache/'extract-bounded-native-v2.py']:
 shutil.copy2(path,out/path.name)
summary={'schema_version':1,'source_commit':source,'ci_url':ci['url'],'artifact':a,'opaque_zip':desc(root/'artifact.zip'),
 'delivered_members_verified':44,'all_saved_payloads_bound_before_and_after':before,'new_collection_native_helper_or_model_calls':0,
 'native_parity_passed':True,'realtime_gate_passed':True,'strict_private_image_audits_replayed':[92,92,92],
 'all8compiler_dependencies_are_public_sdk_headers_or_admitted_helpers':True,'source_text_bytes_and_topology_verified_modes_only_recorded':True,
 'opaque_helper_and_sdk_runtime_archive_modes_independently_verified':True,'acquired_asset_bytes_are_recorded_descriptors_not_delivered_payloads':True,
 'trials':trials,'host':host,'release_qualified':False,'limits':p['limits']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');entries={str(path.relative_to(out)):desc(path) for path in out.rglob('*') if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(out,len(entries)+1,'members');print(json.dumps(trials,indent=2))
