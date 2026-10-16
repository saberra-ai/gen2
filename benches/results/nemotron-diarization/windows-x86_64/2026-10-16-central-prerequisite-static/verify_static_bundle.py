"""Static inspection only: does not execute the installer or any extracted payload."""
import hashlib,json,pathlib,shutil,sys,xml.etree.ElementTree as E
P=pathlib.Path(__file__).parent
b=(P/'VC_redist.x64.exe').read_bytes();assert len(b)==18731856 and hashlib.sha256(b).hexdigest()=='843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c'
root=E.fromstring((P/'ux/0').read_bytes());ns={'b':'http://schemas.microsoft.com/wix/2008/Burn'}
container=root.find('b:Container',ns);n=int(container.attrib['FileSize']);expected=container.attrib['Hash'].lower()
positions=[];start=0
while (o:=b.find(b'MSCF\0\0\0\0',start))>=0:
 if hashlib.sha1(b[o:o+n]).hexdigest()==expected:positions.append(o)
 start=o+1
assert len(positions)==1;offset=positions[0];cab=b[offset:offset+n];(P/'attached-container.cab').write_bytes(cab)
if '--carve-only' in sys.argv:
 print('Verified attached cabinet',offset,n,expected)
 sys.exit(0)
rows=[]
for tag,parent,folder in [('b:Payload',root.find('b:UX',ns),'ux'),('b:Payload',root,'attached')]:
 for x in parent.findall(tag,ns):
  d=x.attrib;src=P/folder/d['SourcePath'];raw=src.read_bytes();assert len(raw)==int(d['FileSize']);assert hashlib.sha1(raw).hexdigest()==d['Hash'].lower()
  parts=pathlib.PureWindowsPath(d['FilePath']).parts;assert not pathlib.PureWindowsPath(d['FilePath']).is_absolute() and '..' not in parts
  target=P/'static-layout'/pathlib.Path(*parts);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
  rows.append({'id':d['Id'],'archive_source':folder+'/'+d['SourcePath'],'logical_path':d['FilePath'],'layout_path':str(target.relative_to(P)),'bytes':len(raw),'sha1_matches_burn_manifest':True,'sha256':hashlib.sha256(raw).hexdigest()})
tools=[]
for path in ['/opt/homebrew/Cellar/p7zip/17.06/bin/7z','/opt/homebrew/Cellar/p7zip/17.06/lib/p7zip/7z','/opt/homebrew/Cellar/p7zip/17.06/lib/p7zip/7z.so']:
 p=pathlib.Path(path)
 if p.is_file():raw=p.read_bytes();tools.append({'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
meta={'installer_executed':False,'installer_authenticode_verified':False,'inner_payload_signatures_verified':False,'installer_sha256':hashlib.sha256(b).hexdigest(),'ux_manifest_sha256':hashlib.sha256((P/'ux/0').read_bytes()).hexdigest(),'attached_container':{'offset':offset,'bytes':n,'sha1':expected,'sha256':hashlib.sha256(cab).hexdigest()},'payloads':rows,'tool_files':tools,'scope':'Static container extraction and exact size/SHA1 comparison to hash-bound Burn manifest. No DLL file/version mapping, Windows trust/signature verification, install or runtime qualification.'}
cert=P/'pe-data/CERTIFICATE'
if cert.exists():
 raw=cert.read_bytes();meta['static_certificate_region']={'path':'pe-data/CERTIFICATE','offset':18721664,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'interpretation':'Presence only; no Authenticode cryptographic or Windows trust validation.'}
(P/'static-inspection.json').write_text(json.dumps(meta,indent=2)+'\n');print('Verified',len(rows),'payloads; cabinet at',offset,'; tool files',tools)
