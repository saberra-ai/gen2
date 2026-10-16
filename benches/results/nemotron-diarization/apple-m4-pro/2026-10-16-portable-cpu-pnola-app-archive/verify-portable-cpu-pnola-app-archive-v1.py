"""Read the whole delivered tar stream and compare exact original inventory; never extract or execute."""
import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/portable-cpu-pnola-app-delivery-v1')
p=json.loads((root/'protocol.json').read_text());q=json.loads((root/'qualification.json').read_text());m=json.loads((root/'portable-cpu-pnola-app-files.json').read_text());assert q['archive_construction_passed'] and m['files']==p['inventory']
def rec(path):
 h=hashlib.sha256();size=0
 with path.open('rb') as stream:
  while chunk:=stream.read(1024**2):h.update(chunk);size+=len(chunk)
 return {'bytes':size,'sha256':h.hexdigest()}
archive=root/m['archive'];assert rec(archive)==q['archive_descriptor']=={'bytes':m['archive_bytes'],'sha256':m['archive_sha256']}
seen=set();file_count=0
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  archive_name=member.name
  assert archive_name=='sdk' or archive_name.startswith('sdk/')
  name='.' if archive_name=='sdk' else archive_name[4:];path=Path(name)
  assert archive_name==('sdk' if name=='.' else 'sdk/'+name) and path.as_posix()==name
  assert name not in seen and name in m['files'] and not path.is_absolute() and '..' not in path.parts
  seen.add(name);expected=m['files'][name];assert member.mode==expected['mode']
  if member.isdir():assert expected['type']=='directory'
  elif member.issym():assert expected['type']=='symlink' and member.linkname==expected['target']
  else:
   assert member.isfile() and expected['type']=='file' and member.size==expected['bytes']
   h=hashlib.sha256();stream=tar.extractfile(member);size=0
   while chunk:=stream.read(1024**2):h.update(chunk);size+=len(chunk)
   assert size==expected['bytes'] and h.hexdigest()==expected['sha256'];file_count+=1
assert seen==set(m['files'])
r={'schema_version':1,'all_archive_entries_match_original_inventory':True,'entries':len(seen),'regular_files':file_count,'archive':rec(archive),'archive_construction_protocol':rec(root/'protocol.json'),'verifier':rec(Path(__file__)),'scope':__doc__,'models_or_app_executed':False,'extracted':False,'release_qualified':False}
output=root/'archive-byte-verification.json';assert not output.exists();output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'entries':len(seen),'regular_files':file_count,'passed':True}))
