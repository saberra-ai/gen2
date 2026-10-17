"""Bounded saved artifact extraction only; never execute delivered binaries."""
from pathlib import Path,PurePosixPath
import zipfile,hashlib,json,stat,sys
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/bounded-windows-native-ci-v1');archive=root/'artifact.zip';out=root/'unpacked'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='49ae76bfa000287e6e0831618fdd0aed776bbb6b6134e0a2ea55f194880aad16'
assert not out.exists();out.mkdir();seen=set();total=0
with zipfile.ZipFile(archive) as z:
 assert len(z.infolist())<=512
 for info in z.infolist():
  name=info.filename;parts=PurePosixPath(name).parts
  assert parts and not name.startswith('/') and all(part not in {'.','..'} and ':' not in part and '\\' not in part for part in parts)
  assert name.casefold() not in seen;seen.add(name.casefold())
  assert stat.S_IFMT(info.external_attr>>16) in {0,stat.S_IFREG,stat.S_IFDIR}
  assert not info.flag_bits&1 and info.file_size<=64*1024**2
  total+=info.file_size;assert total<=128*1024**2
  target=out.joinpath(*parts)
  if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with z.open(info) as src,target.open('xb') as dst:
   size=0
   while block:=src.read(1024**2):
    size+=len(block);assert size<=info.file_size;dst.write(block)
  assert size==info.file_size
sys.path.insert(0,'/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization')
from platform_bounded_metrics import compare_complete_60s
from windows_archive import byte_inventory
q=json.loads((out/'nemo-bounded-evidence/receipt.json').read_text());p=out/'nemo-bounded-evidence'
comparison=compare_complete_60s(json.loads((p/'public.json').read_text()),json.loads((p/'native-ami.json').read_text()),(p/'native-ami.json.f32').read_bytes(),(p/'public.f32').read_bytes())
assert comparison==json.loads((p/'comparison.json').read_text())
files=byte_inventory(out/'nemo-bounded-sdk')
assert {n:r for n,r in files.items() if r['type']=='file'}=={n.replace('\\','/'):{'type':'file',**r} for n,r in q['installed_files'].items()}
print(json.dumps({'source_commit':q['source_commit'],'flags':{n:q[n] for n in ['development_native_gates_passed','functional_lifecycle_gates_passed','realtime_gate_passed']},'error':q['error'],'comparison':comparison,'cpu':json.loads((p/'windows-provenance.json').read_text(encoding='utf-8-sig'))['cpu'],'commands':len(q['commands']),'all_command_returns_zero':all(row['returncode']==0 for row in q['commands']),'lifecycle_count':len(q['completed_lifecycle_tests']),'sdk_files':len(q['installed_files'])},indent=2))
