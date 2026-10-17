"""Admit the immutable thread-contrast artifact; execute no delivered code."""
from pathlib import Path,PurePosixPath
import hashlib
import json
import stat
import zipfile
root = Path('/Users/victor/.cache/gen2-nemotron3-n0/cpu-threads-ci-v1')
ci = json.loads((root / 'run.json').read_text())
api = json.loads((root / 'artifacts.json').read_text())['artifacts']
assert ci['databaseId'] == 39003266888 and ci['headSha'] == '2995f0e0b25be69c6b9d4c5f05dc53b3cf22fa26'
assert ci['status'] == 'completed' and ci['conclusion'] == 'success' and len(api) == 1
item = api[0]
assert item['name'] == 'nemotron-macos-cpu-threads' and not item['expired']
assert item['workflow_run']['id'] == ci['databaseId'] and item['workflow_run']['head_sha'] == ci['headSha']
raw = (root / 'artifact.zip').read_bytes()
assert len(raw) == item['size_in_bytes'] and item['digest'] == 'sha256:'+hashlib.sha256(raw).hexdigest()
protocol = json.loads(Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/macos-cpu-threads.protocol.json').read_text())
expected = {'protocol.json','qualification.json','package.json','host.json','protected-inputs.json',
    'thread-helpers-files.json','thread-helpers.tar.gz','runtime.sb','premodel.sb','compile.sb','probe.d','backend.d','logging.d','results/prior.txt'}
expected.update(name+suffix for name in protocol['preparation_order']+protocol['neural_order'] for suffix in ['.stdout.log','.stderr.log'])
expected.update('results/'+name+'.json'+suffix for name in protocol['neural_order'] for suffix in ['','.f32'])
destination = root / 'evidence'
assert not destination.exists()
with zipfile.ZipFile(root / 'artifact.zip') as archive:
    members = archive.infolist()
    assert len(members) == len(expected) == 86 and {m.filename for m in members} == expected
    assert sum(m.file_size for m in members) < 16*2**20
    destination.mkdir()
    for member in members:
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == member.filename
        assert not member.is_dir() and stat.S_IFMT(member.external_attr>>16) in {0,stat.S_IFREG}
        target = destination / member.filename
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output: output.write(archive.read(member))
print(destination,len(expected),'verified members')
q = json.loads((destination / 'qualification.json').read_text())
print(json.dumps({key:q[key] for key in ['preparation_completed','comparison_completed','comparisons','neural_commands_started','load_after']},indent=2))
