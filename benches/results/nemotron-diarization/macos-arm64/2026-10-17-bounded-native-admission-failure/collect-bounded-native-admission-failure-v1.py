from pathlib import Path,PurePosixPath
import hashlib,json,shutil,stat,subprocess,zipfile
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'bounded-native-ci-v1';read=lambda path:json.loads(Path(path).read_text())
def desc(path):
 raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
ci=read(root/'run.json');a=read(root/'artifacts.json')['artifacts'];assert len(a)==1;a=a[0]
source='f7c661a85d76f7c2993a008d3a6f6c5991973723'
assert ci['databaseId']==39004520414 and ci['headSha']==source and ci['status']=='completed' and ci['conclusion']=='failure'
assert a['id']==12090805567 and a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==source
assert desc(root/'artifact.zip')=={'bytes':16418,'sha256':'4d785428b738c361be81a5be174dc42d50466234d7a86f8a9883e4fc6c2fc024'}
assert a['size_in_bytes']==16418 and a['digest']=='sha256:'+desc(root/'artifact.zip')['sha256']
saved=root/'evidence';assert not saved.exists()
with zipfile.ZipFile(root/'artifact.zip') as archive:
 members=archive.infolist();assert len(members)==2 and {m.filename for m in members}=={'protocol.json','qualification.json'}
 for m in members:
  p=PurePosixPath(m.filename);assert not p.is_absolute() and '..' not in p.parts and not m.is_dir() and stat.S_IFMT(m.external_attr>>16) in {0,stat.S_IFREG}
 saved.mkdir()
 for m in members:(saved/m.filename).write_bytes(archive.read(m))
q=read(saved/'qualification.json');p=read(saved/'protocol.json')
assert (saved/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-bounded-native.protocol.json'],cwd=repo)
assert q['source_commit']==source and q['commands']==q['neural_commands_started']==[] and q['deliveries']=={}
assert q['native_parity_passed'] is False and q['realtime_gate_passed'] is False and q['release_qualified'] is False
assert q['error']=="KeyError('name')" and 'name' not in p['producer'] and 'run_id' not in p['producer']
assert q['storage']['released'] is True and q['storage']['reserve_reported_allocated_bytes']==268435456
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-native-admission-failure';assert not out.exists();out.mkdir()
for path in [*saved.iterdir(),root/'run.json',root/'artifacts.json',root/'failed-job.log',Path(__file__)]:shutil.copy2(path,out/path.name)
summary={'schema_version':1,'source_commit':source,'ci_url':ci['url'],'artifact':a,'opaque_zip':desc(root/'artifact.zip'),'delivered_members_verified':2,
 'result':'failed; protocol producer omitted required name and run_id','commands_started':0,'model_calls':0,'helpers_compiled_or_executed':0,
 'source_sdk_or_reference_payloads_admitted':False,'storage_reserve_released':True,'failure_relabelled':False,'release_qualified':False}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');files={str(path.relative_to(out)):desc(path) for path in out.iterdir() if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(out,len(files)+1,'members')
