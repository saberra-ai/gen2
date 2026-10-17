"""Freeze explicit XML retry and prior failed receipt; no helper/OS/image operation."""
from pathlib import Path
import hashlib,json
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb')as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
original=tools/'qualify_macos_vm_host.py';target=tools/'qualify_macos_vm_host_v2.py';assert not target.exists()
a=original.read_text();old_protocol="source + ':tools/nemotron-diarization/macos-vm-host.protocol.json'";new_protocol="source + ':tools/nemotron-diarization/macos-vm-host-v2.protocol.json'";old_args="['/usr/bin/codesign', '--display', '--entitlements', '-', executable]";new_args="['/usr/bin/codesign', '--display', '--xml', '--entitlements', '-', executable]";assert a.count(old_protocol)==a.count(old_args)==1;target.write_text(a.replace(old_protocol,new_protocol).replace(old_args,new_args))
p=json.loads((tools/'macos-vm-host.protocol.json').read_text())
for n,want in p['tools'].items():assert descriptor(tools/n)==want
for n,want in p['external_files'].items():assert descriptor(n)==want
prior=cache/'macos-vm-host-admission-local-v1';q=json.loads((prior/'qualification.json').read_text());assert q['host_prerequisites_passed']is False and q['error']=="InvalidFileException('Invalid file')" and q['owned_image_detached']is True and q['storage']['released']
receipt=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-entitlements-format-failure';collection=json.loads((receipt/'collection.json').read_text())
for n,want in collection['files'].items():assert descriptor(receipt/n)==want
assert (prior/'qualification.json').read_bytes()==(receipt/'qualification.json').read_bytes()
p['tools'][target.name]=descriptor(target);p['output_name']='macos-vm-host-admission-local-v2';assert not(cache/p['output_name']).exists()
records=[Path(__file__),prior/'qualification.json',prior/'protocol.json',receipt/'collection.json',receipt/'collection-qualification.json',cache/'vm-host-entitlements-xml-control-v1.json',cache/'vm-host-entitlements-xml-control-v1.stdout.log',cache/'vm-host-entitlements-xml-control-v1.stderr.log']
p['external_files'].update({str(n):descriptor(n)for n in records});p['denied_paths']=sorted(set(p['denied_paths'])|{str(n)for n in cache.iterdir()if n.is_dir()})
p['limits'].insert(0,'V2 binds the actual failed7a5f3da receipt and read-only XML control. Preserve original controller/protocol/attempt. The only controller differences are this committed protocol filename and explicit --xml for codesign display; all helper/geometry/signing/offline/own-disk/cleanup/storage/no-guest gates remain unchanged. Use a distinct fresh output; no silent retry or claim that original attempt passed.')
f=tools/'macos-vm-host-v2.protocol.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n');print(descriptor(f),'tools',len(p['tools']),'external',len(p['external_files']))
