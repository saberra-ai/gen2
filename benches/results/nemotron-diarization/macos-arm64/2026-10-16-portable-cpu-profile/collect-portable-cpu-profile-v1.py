"""Replay saved diagnostic bytes only; no native/profiler/model execution."""
import hashlib,json,math,re,shutil,stat,struct,subprocess,sys,zipfile
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files, verify_package_provenance
from sdk_archive import inventory
commit='0770aaaa62ddd50e5bccc1de9a25cb86ae3f53ae'
ci=json.loads((cache/'platform-ci/cpu-profile-38979777083.final.json').read_text())
assert ci['databaseId']==38979777083 and ci['headSha']==commit and ci['status']=='completed' and ci['conclusion']=='success'
api=json.loads((cache/'platform-ci/cpu-profile-38979777083.artifacts.json').read_text())['artifacts']
assert len(api)==1
a=api[0]
root=cache/'portable-cpu-profile-ci-v1';source=root/'evidence';opaque=root/'artifact.zip'
assert a['id']==12080083616 and a['name']=='nemotron-macos-cpu-profile' and a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==commit
assert a['size_in_bytes']==opaque.stat().st_size and a['digest']=='sha256:'+descriptor(opaque)['sha256']
with zipfile.ZipFile(opaque) as z:
 members=z.infolist()
 assert len(members)==len({m.filename for m in members})==19
 for m in members:
  f=Path(m.filename)
  assert not f.is_absolute() and '..' not in f.parts and not m.is_dir() and stat.S_IFMT(m.external_attr>>16) in (0,stat.S_IFREG)
  assert (source/f).read_bytes()==z.read(m)
 assert {str(f.relative_to(source)) for f in source.rglob('*') if f.is_file()}=={m.filename for m in members}
 assert not any(f.is_symlink() for f in source.rglob('*'))
p=json.loads((source/'protocol.json').read_text());r=json.loads((source/'qualification.json').read_text())
assert (source/'protocol.json').read_bytes()==subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/macos-portable-cpu-profile.protocol.json'],cwd=repo)
assert r['protocol']==descriptor(source/'protocol.json') and r['source_commit']==commit and r['profile_completed'] and r['model_execution_admitted'] and r['release_qualified'] is False
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
 for name,d in p[group].items():
  raw=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo)
  assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==d==descriptor(repo/(prefix+name))
for name,relative in [('construction','portable-macos-construction-ci-v2'),('receiver','portable-macos-runtime-ci-v2')]:
 saved=cache/relative
 assert descriptor(saved/'artifact.zip')==p[name]['zip']
 assert r['deliveries'][name]['id']==p[name]['id'] and r['deliveries'][name]['workflow_run']['head_sha']==p[name]['source_commit']
 assert {str(f.relative_to(saved/'evidence')):descriptor(f) for f in (saved/'evidence').rglob('*') if f.is_file()}==p[name]['files']
helpers=cache/'portable-macos-runtime-ci-v2/restored-helpers'
package=cache/'portable-macos-construction-ci-v2/restored-portable-cpu-runtime'
assert inventory(helpers)==p['helpers']
manifest=json.loads((source/'package.json').read_text())
assert manifest==json.loads((package/'package.json').read_text())
verify_files(package,manifest['files'])
verify_package_provenance(package,manifest)
assert [row['expected'] for row in r['acquisition_attempts']]==p['assets']
for row in r['acquisition_attempts']:
 assert row['verified']=={'bytes':row['expected']['size'],'sha256':row['expected']['sha256']}
names=['compile-control','sampling-control','runtime-controls','native']
assert [c['name'] for c in r['commands']]==names
for c in r['commands']:
 assert c['returncode']==0 and not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes')
 assert 'DYLD_PRINT_LIBRARIES' not in c['environment'] and c['argv'][:2]==['/usr/bin/sandbox-exec','-f']
remote=Path(r['commands'][0]['cwd'])
assert all(c['cwd']==str(remote) for c in r['commands'])
assert (source/'control.c').read_text()=='#include <unistd.h>\nint main(void) { sleep(10); return 0; }\n'
for name in ['sampling-control','native']:
 s=r['sampling'][name];text=(source/(name+'.sample.txt')).read_text();trace=(source/(name+'.stderr.log')).read_text()
 assert s['returncode']==0 and 'error' not in s and descriptor(source/(name+'.sample.txt'))==s['report']
 assert s['argv']==['/usr/bin/sample',str(s['pid']),str(1 if name=='sampling-control' else 5),'10','-mayDie','-file',str(remote/(name+'.sample.txt'))]
 assert re.search(r'^Process:\s+.*\['+str(s['pid'])+r'\]',text,re.M)
 exe=Path(s['executable']);assert exe==remote/('work/control' if name=='sampling-control' else 'work/bin/native-stream')
 assert set(re.findall(r'^dyld\[(\d+)\]: <[A-Fa-f0-9-]+> /',trace,re.M))=={str(s['pid'])}
 assert re.search(r'^dyld\['+str(s['pid'])+r'\]: <'+re.escape(s['target_dyld_uuid'])+r'> '+re.escape(str(exe))+r'$',trace,re.M)
 assert s['target_binary_image'] in text and '<'+s['target_dyld_uuid']+'>' in s['target_binary_image']
# Control payload was not delivered: its hash/ps path are producer observations.
control=r['control_images'];assert control['private_library_count']==0 and control['private_libraries']=={} and not control['entire_runtime_required']
control_trace=(source/'sampling-control.stderr.log').read_text()
images=set(re.findall(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)',control_trace))
assert {x for x in images if not x.startswith(('/usr/lib/','/System/Library/'))}=={str(remote/'work/control')}
trace=(source/'native.stderr.log').read_text()
remote_package=remote/'work/nemo';remote_helper=remote/'work/bin/native-stream'
pattern=r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)'
def remap(m):
 path=m[2]
 if path.startswith(('/usr/lib/','/System/Library/')):return m[0]
 if path==str(remote_helper):return m[1]+str(helpers/'native-stream')
 rel=Path(path).relative_to(remote_package/'lib')
 assert rel.as_posix() in set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib'}
 return m[1]+str(package/'lib'/rel)
audited=audit_loaded_libraries(re.sub(pattern,remap,trace),package,manifest,helpers/'native-stream')
normalized=json.loads(json.dumps(audited).replace(str(package),str(remote_package)).replace(str(helpers/'native-stream'),str(remote_helper)))
assert normalized==r['loaded_images'] and audited['private_library_count']==92
n=json.loads((source/'native.json').read_text());raw=(source/'native.json.f32').read_bytes()
assert n['audio_samples']==192000 and n['native_frame_count']==n['retained_rows']==1200 and n['retained_first_frame']==0 and len(raw)==38400
assert n['input_pcm_f32le_sha256']==p['pcm_f32le_sha256'] and n['geometry']==p['geometry'] and n['runtime_version']==p['runtime_version']
assert all(math.isfinite(v[0]) and 0<=v[0]<=1 for v in struct.iter_unpack('<f',raw))
assert r['native_rtf']==n['elapsed_micros']/1e6/12
host=json.loads((source/'host.json').read_text());assert host['cpu']==p['expected_receiver_cpu'] and host['cpu_counts']=='hw.ncpu: 3\nhw.logicalcpu: 3\nhw.physicalcpu: 3\n'
usage=re.search(r'^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$',trace,re.M)
assert usage
wall,user,system=map(float,usage.groups())
assert (wall,user,system)==(70.32,193.45,1.86)
assert r['storage']['released'] and r['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum']
assert r['storage']['start_minimum_bytes']==p['storage']['start_minimum'] and r['storage']['running_minimum_bytes']==p['storage']['running_minimum']
assert r['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
record={'schema_version':1,'diagnostic_completed':True,'release_qualified':False,'source_commit':commit,'ci_url':ci['url'],'artifact':a,'opaque_zip':descriptor(opaque),'delivered_members':19,'frozen_sources':21,'native_helper_and_package_bytes_independently_rehashed':True,'control_binary_delivered':False,'control_payload_hash_and_live_ps_identity_are_ci_observations':True,'acquisitions_are_saved_descriptors_not_delivered_payloads':True,'strict_private_images':92,'native_frames':1200,'native_rtf':r['native_rtf'],'time_l':{'real_seconds':wall,'user_seconds':user,'system_seconds':system,'cpu_seconds_per_wall_second':(user+system)/wall},'host':host,'scope':p['scope'],'limits':p['limits'],'storage':r['storage']}
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-16-portable-cpu-profile'
assert not out.exists();out.mkdir(parents=True)
for f in source.iterdir():shutil.copy2(f,out/f.name)
for suffix in ['final.json','artifacts.json']:shutil.copy2(cache/('platform-ci/cpu-profile-38979777083.'+suffix),out/('ci-'+suffix))
shutil.copy2(__file__,out/Path(__file__).name)
(out/'collection-qualification.json').write_text(json.dumps(record,indent=2)+'\n')
files={f.name:descriptor(f) for f in out.iterdir() if f.is_file()}
(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'receipt_members':len(files)+1,'output':str(out)}))
