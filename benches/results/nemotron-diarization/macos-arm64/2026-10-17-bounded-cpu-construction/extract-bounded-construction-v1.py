from pathlib import Path,PurePosixPath
import hashlib,json,stat,zipfile
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/bounded-macos-construction-ci-v1')
ci=json.loads((root/'run.json').read_text());api=json.loads((root/'artifacts.json').read_text())['artifacts']
assert ci['databaseId']==39003880853 and ci['headSha']=='aa06657316e16d996886e55b93bc1bc25251b620'
assert ci['status']=='completed' and ci['conclusion']=='success' and len(api)==1
a=api[0];raw=(root/'artifact.zip').read_bytes()
assert a['id']==12089814731 and a['name']=='nemotron-macos-bounded-construction'
assert a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==ci['headSha']
assert len(raw)==a['size_in_bytes']==17622146
assert a['digest']=='sha256:'+hashlib.sha256(raw).hexdigest()=='sha256:16afdf64e55028a6ba087ec50effadb2e30c199a2a2fd36c1885fc0becc89570'
old=json.loads(Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/macos-cpu-threads.protocol.json').read_text())
expected=set(old['producer']['files'])|{'preparation/'+str(i)+suffix for i in [8,9] for suffix in ['.stdout.log','.stderr.log']}
destination=root/'evidence';assert not destination.exists()
with zipfile.ZipFile(root/'artifact.zip') as archive:
 members=archive.infolist()
 print('members',len(members),'expected',len(expected),'extra',sorted({m.filename for m in members}-expected),'missing',sorted(expected-{m.filename for m in members}))
 assert len(members)==len(expected)==200 and {m.filename for m in members}==expected
 assert sum(m.file_size for m in members)<256*2**20
 for member in members:
  path=PurePosixPath(member.filename)
  assert not path.is_absolute() and '..' not in path.parts and path.as_posix()==member.filename
  assert not member.is_dir() and stat.S_IFMT(member.external_attr>>16) in {0,stat.S_IFREG}
 destination.mkdir()
 for member in members:
  target=destination/member.filename;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as output:output.write(archive.read(member))
q=json.loads((destination/'qualification.json').read_text())
print({k:q[k] for k in ['source_commit','construction_and_negative_c_api_controls_passed','valid_models_loaded','frontend_or_neural_calls']})
print([(c['name'],c['returncode']) for c in q['commands']])
