from pathlib import Path,PurePosixPath
import hashlib
import json
import stat
import zipfile

root = Path('/Users/victor/.cache/gen2-nemotron3-n0/pooled-cpu-profile-ci-v1')
ci = json.loads((root / 'run.json').read_text())
api = json.loads((root / 'artifacts.json').read_text())['artifacts']
assert ci['databaseId'] == 39001644761 and ci['headSha'] == '761c9ca1613d09fa68eb4c0fad957131a775edd2'
assert ci['status'] == 'completed' and ci['conclusion'] == 'success' and len(api) == 1
a = api[0]
assert a['id'] == 12089168327 and a['name'] == 'nemotron-macos-pooled-profile'
assert a['workflow_run']['id'] == ci['databaseId'] and a['workflow_run']['head_sha'] == ci['headSha']
raw = (root / 'artifact.zip').read_bytes()
assert len(raw) == a['size_in_bytes'] == 200631
assert a['digest'] == 'sha256:'+hashlib.sha256(raw).hexdigest() == 'sha256:fc2022f82804c6eba1355e646d57105b8b7d4d26fff428177afa377e53c93fa7'
expected = {'protocol.json','qualification.json','package.json','host.json','protected-before.json',
            'profile-helpers-files.json','profile-helpers.tar.gz','runtime.sb','premodel.sb','control-build.sb','control.c',
            'sampling-control.sample.txt','native.sample.txt','results/prior.txt','results/native.json','results/native.json.f32','work/control'}
expected.update(name+suffix for name in ['compile-control','sampling-control','runtime-controls','native']
                for suffix in ['.stdout.log','.stderr.log'])
destination = root / 'evidence'
assert not destination.exists()
with zipfile.ZipFile(root / 'artifact.zip') as archive:
    members = archive.infolist()
    assert len(members) == len(expected) == 25 and {m.filename for m in members} == expected
    assert sum(m.file_size for m in members) <= 16*2**20
    destination.mkdir()
    for member in members:
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == member.filename
        assert not member.is_dir() and stat.S_IFMT(member.external_attr>>16) in {0,stat.S_IFREG}
        target = destination / member.filename
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output:
            output.write(archive.read(member))
print(destination,len(expected),'verified members')
q = json.loads((destination / 'qualification.json').read_text())
print(json.dumps({key:q[key] for key in ['profile_completed','native_rtf','comparison','neural_commands_started','load_after']},indent=2))
print((destination / 'host.json').read_text())
