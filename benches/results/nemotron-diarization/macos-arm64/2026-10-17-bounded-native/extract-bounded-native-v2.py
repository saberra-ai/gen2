from pathlib import Path,PurePosixPath
import hashlib,json,stat,zipfile
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/bounded-native-ci-v2');ci=json.loads((root/'run.json').read_text());a=json.loads((root/'artifacts.json').read_text())['artifacts'];assert len(a)==1;a=a[0]
assert ci['databaseId']==39004916708 and ci['headSha']=='05c5ac1535a38e76b258ca57749bea820a7dc14f' and ci['status']=='completed' and ci['conclusion']=='success'
assert a['id']==12090213937 and a['name']=='nemotron-macos-bounded-native' and a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==ci['headSha']
raw=(root/'artifact.zip').read_bytes();assert len(raw)==a['size_in_bytes']==403280
assert a['digest']=='sha256:'+hashlib.sha256(raw).hexdigest()=='sha256:b2c02a333862370620a710f7dd2ace7e9c149f6588fdbf3749ed7db64af28b53'
p=json.loads(Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/macos-bounded-native-v2.protocol.json').read_text())
expected={'protocol.json','qualification.json','package.json','host.json','protected-inputs.json','bounded-native-helpers-files.json','bounded-native-helpers.tar.gz',
 'compile.sb','runtime.sb','premodel.sb','native.d','sha.d','failed.d','results/prior.txt'}
expected.update(name+suffix for name in p['preparation_order']+p['neural_order'] for suffix in ['.stdout.log','.stderr.log'])
expected.update('results/'+name+'.json'+suffix for name in p['neural_order'] for suffix in ['','.f32'])
expected.update('helper-source/'+name for name in [*p['helper_inputs'],'native.cpp','failed.cpp'])
saved=root/'evidence';assert not saved.exists()
with zipfile.ZipFile(root/'artifact.zip') as archive:
 members=archive.infolist();assert len(members)==len(expected)==44 and {m.filename for m in members}==expected
 assert sum(m.file_size for m in members)<16*2**20
 for m in members:
  path=PurePosixPath(m.filename);assert not path.is_absolute() and '..' not in path.parts and path.as_posix()==m.filename and not m.is_dir() and stat.S_IFMT(m.external_attr>>16) in {0,stat.S_IFREG}
 saved.mkdir()
 for m in members:
  target=saved/m.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(m))
q=json.loads((saved/'qualification.json').read_text());print('verified',len(expected),'members');print(json.dumps({k:q[k] for k in ['native_parity_passed','realtime_gate_passed','neural_commands_started','comparisons']},indent=2))
