"""Replay saved ABBA experiment bytes; no compilation/native/model execution."""
import hashlib,json,math,re,shutil,stat,struct,subprocess,sys,zipfile
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import compare_probabilities,SELF_TEST,ORDER,model_free_policy
from package_macos import verify_files,verify_package_provenance
from sdk_archive import inventory
commit='643377e8917778a1be0361db8f5e2f63071bc075';root=cache/'pool-comparison-ci-v1';source=root/'evidence'
a=json.loads((root/'download-identity.json').read_text());ci=json.loads((root/'run.json').read_text());delivery=json.loads((root/'delivery.json').read_text())
assert ci['id']==38982552508 and ci['head_sha']==commit and ci['status']=='completed' and ci['conclusion']=='success'
assert a['id']==12082035831 and a['name']=='nemotron-macos-cpu-pools' and a['workflow_run']['id']==ci['id'] and a['workflow_run']['head_sha']==commit
assert descriptor(root/'artifact.zip')=={'bytes':338699,'sha256':'3062077c8e5bca7e6e8683544f6923d0360335b43579ec01c5956df0e5d0faae'}==delivery['zip']
assert a['size_in_bytes']==338699 and a['digest']=='sha256:'+delivery['zip']['sha256']
with zipfile.ZipFile(root/'artifact.zip') as z:
 members=z.infolist();assert len(members)==len({m.filename for m in members})==78
 assert {str(f.relative_to(source)) for f in source.rglob('*') if f.is_file()}=={m.filename for m in members}
 for m in members:
  path=Path(m.filename);assert not path.is_absolute() and '..' not in path.parts and not m.is_dir() and stat.S_IFMT(m.external_attr>>16) in (0,stat.S_IFREG)
  assert not (source/path).is_symlink() and (source/path).read_bytes()==z.read(m) and descriptor(source/path)==delivery['files'][m.filename]
p=json.loads((source/'protocol.json').read_text());q=json.loads((source/'qualification.json').read_text())
assert (source/'protocol.json').read_bytes()==subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/macos-cpu-pools.protocol.json'],cwd=repo)
assert q['source_commit']==commit and q['protocol']==descriptor(source/'protocol.json') and q['comparison_completed'] and q['release_qualified'] is False and 'error' not in q
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for name,d in p[group].items():
  raw=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo);assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==d==descriptor(repo/(prefix+name))
assert len(p['tools'])==15 and len(p['native_inputs'])==111
native=cache/'NeMo-Speech-valid-mel-tagged-v1'
for name,d in p['native_inputs'].items():
 raw=subprocess.check_output(['git','show',p['native_revision']+':'+name],cwd=native);assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==d
for name,dirname in [('construction','portable-macos-construction-ci-v2'),('reference','portable-cpu-profile-ci-v1')]:
 saved=cache/dirname;assert descriptor(saved/'artifact.zip')==p[name]['zip']
 assert {str(f.relative_to(saved/'evidence')):descriptor(f) for f in (saved/'evidence').rglob('*') if f.is_file()}==p[name]['files']
 assert q['deliveries'][name]['id']==p[name]['id'] and q['deliveries'][name]['workflow_run']['head_sha']==p[name]['source_commit']
construction=cache/'portable-macos-construction-ci-v2';package=construction/'restored-portable-cpu-runtime';manifest=json.loads((source/'package.json').read_text());assert manifest==json.loads((package/'package.json').read_text())
verify_files(package,manifest['files']);verify_package_provenance(package,manifest)
for kind,count in [('sdk',84),('runtime',123)]:
 files=json.loads((construction/('evidence/portable-cpu-'+kind+'-files.json')).read_text())['files'];assert len(files)==count and inventory(construction/('restored-portable-cpu-'+kind))==files
audit=json.loads((construction/'evidence/build/llama-materialized-audit.json').read_text());assert q['ggml_headers']=={n:d for n,d in audit['files'].items() if n.startswith('ggml/include/')} and len(q['ggml_headers'])==22
names=['compile-controls',*[f'headers-patch-{i}' for i in range(19)],'probe-compile','backend-compile','logging-compile','link','runtime-controls','model-free-controls','pool-self-test','1-disposable','2-persistent','3-persistent','4-disposable']
assert [c['name'] for c in q['commands']]==names and len(names)==31
for c in q['commands']:
 assert c['returncode']==0 and not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes')
 assert c['argv'][:2]==['/usr/bin/sandbox-exec','-f'] and 'DYLD_PRINT_LIBRARIES' not in c['environment']
remote=Path(q['commands'][0]['cwd']);remote_executable=remote/'work/bin/native-pool-probe';remote_package=remote/'work/nemo'
assert (source/'model-free.sb').read_text()==model_free_policy((source/'runtime.sb').read_text())
assert json.loads((source/'pool-self-test.stdout.log').read_text())==SELF_TEST
assert q['inference_commands_started']==['1-disposable','2-persistent','3-persistent','4-disposable']
expected_private={str(remote_package/'lib'/n):{'sha256':manifest['files']['lib/'+n]['sha256'],'reported_path':str(remote_package/'lib'/n)} for n in manifest['libraries']}
assert len(expected_private)==92
# No M1 helper binary was delivered. Recheck exact trace paths plus all92private
# library payloads; the helper payload hash remains an explicitly CI-only observation.
for name in ['pool-self-test',*q['inference_commands_started']]:
 trace=(source/(name+'.stderr.log')).read_text();reported=set(re.findall(r'^dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)',trace,re.M))
 assert len(set(re.findall(r'^dyld\[(\d+)\]: <[A-Fa-f0-9-]+> /',trace,re.M)))==1
 assert {n for n in reported if not n.startswith(('/usr/lib/','/System/Library/'))}==set(expected_private)|{str(remote_executable)}
 assert q['loaded_images'][name]=={'executable':str(remote_executable),'executable_sha256':q['binary']['sha256'],'unique_reported_images':len(reported),'private_libraries':expected_private,'private_library_count':92,'entire_runtime_required':True}
loads=json.loads((source/'helper-load-commands.json').read_text());assert loads['binary']==q['binary'] and loads['architectures']=='arm64'
assert [r['expected'] for r in q['acquisition_attempts']]==p['assets']
for row in q['acquisition_attempts']:assert row['verified']=={'bytes':row['expected']['size'],'sha256':row['expected']['sha256']}
assert q['pcm']=={'bytes':768000,'sha256':p['pcm_f32le_sha256']}
reference=(cache/'portable-cpu-profile-ci-v1/evidence/native.json.f32').read_bytes();rows=[]
for name,mode in zip(q['inference_commands_started'],ORDER):
 n=json.loads((source/'results'/(name+'.json')).read_text());raw=(source/'results'/(name+'.json.f32')).read_bytes();comparison=compare_probabilities(raw,reference)
 assert comparison['bit_exact'] and n['audio_samples']==192000 and n['frames']==1200 and n['channels']==8 and n['retained_first_frame']==0 and n['threads']==4
 assert n['persistent_pool']==(mode=='persistent') and n['geometry']==p['geometry'] and n['input_pcm_f32le_sha256']==p['pcm_f32le_sha256']
 assert q['comparisons'][name]=={**comparison,'warm_rtf':n['elapsed_micros']/12e6}
 trace=(source/(name+'.stderr.log')).read_text();times=re.findall(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$',trace,re.M);assert len(times)==1
 wall,user,system=map(float,times[0]);rss=re.findall(r'^\s*(\d+)  maximum resident set size$',trace,re.M);assert len(rss)==1
 rows.append({'name':name,'persistent_pool':n['persistent_pool'],'warm_rtf':n['elapsed_micros']/12e6,'warm_seconds':n['elapsed_micros']/1e6,'model_constructor_seconds_excluding_backend_and_pool_setup':n['load_micros']/1e6,'whole_process':{'real_seconds':wall,'user_seconds':user,'system_seconds':system,'maximum_rss_bytes':int(rss[0])},'full_history_parity':comparison})
means={mode:{'warm_rtf':sum(r['warm_rtf'] for r in rows if r['persistent_pool']==(mode=='persistent'))/2,'whole_process_real_seconds':sum(r['whole_process']['real_seconds'] for r in rows if r['persistent_pool']==(mode=='persistent'))/2} for mode in ['disposable','persistent']}
ratio=means['disposable']['warm_rtf']/means['persistent']['warm_rtf'];assert 3.14<ratio<3.16
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and q['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
assert q['storage']['start_minimum_bytes']==p['storage']['start_minimum'] and q['storage']['running_minimum_bytes']==p['storage']['running_minimum']
host=json.loads((source/'host.json').read_text());assert host['cpu']==p['expected_receiver_cpu'] and host['cpu_counts']=='hw.ncpu: 3\nhw.logicalcpu: 3\nhw.physicalcpu: 3\n'
record={'schema_version':1,'source_commit':commit,'ci_url':ci['html_url'],'artifact':a,'opaque_zip':descriptor(root/'artifact.zip'),'delivered_members':78,'experiment_completed':True,'release_qualified':False,'helper_binary_delivered':False,'helper_payload_hash_is_ci_observation':True,'acquired_audio_and_model_are_saved_descriptors_not_delivered_payloads':True,'exact_private_paths_and_92_library_payload_hashes_rechecked_per_case':True,'rows':rows,'means':means,'ratio_of_disposable_to_persistent_mean_warm_rtf':ratio,'ratio_of_disposable_to_persistent_mean_whole_process_time':means['disposable']['whole_process_real_seconds']/means['persistent']['whole_process_real_seconds'],'all_four_38400_byte_histories_bit_exact_with_qualified_C_ABI':True,'live_below_one_gate_passed':False,'host':host,'load_after':q['load_after'],'storage':q['storage'],'scope':p['scope'],'limits':p['limits']+['Helper executable was not delivered. Its hash is a CI observation; independent collection verifies all delivered bytes and private-library trace closure, not the absent helper payload.','This is measured diagnostic evidence for integrating a persistent pool under a new runtime identity and requalifying it, not an already deployed optimization.']}
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-16-cpu-pool-comparison';assert not out.exists();shutil.copytree(source,out)
for name in ['download-identity.json','run.json','delivery.json']:shutil.copy2(root/name,out/name)
shutil.copy2(__file__,out/Path(__file__).name);(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n')
files={str(f.relative_to(out)):descriptor(f) for f in out.rglob('*') if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'members':len(files)+1,'warm_speedup_ratio':ratio,'all_bit_exact':True}))
