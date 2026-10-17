"""Freeze observed-image partition selection; no helper/OS/image operation."""
from pathlib import Path
import hashlib,json,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb')as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
original=tools/'qualify_macos_vm_host_v2.py';target=tools/'qualify_macos_vm_host_v3.py';assert not target.exists()
a=original.read_text();changes=[
("source + ':tools/nemotron-diarization/macos-vm-host-v2.protocol.json'", "source + ':tools/nemotron-diarization/macos-vm-host-v3.protocol.json'"),
('from macos_vm_host_metrics import attached_image_device, apfs_probe_volume, probe_metrics', 'from macos_vm_host_v3_metrics import attached_image_device, apfs_image_partition, apfs_probe_volume, probe_metrics'),
('def own_attachment(name, *, cleanup=False):','def own_attachment(name, *, cleanup=False, details=False):'),
('        return device\n','        return (device, value) if details else device\n'),
("whole = own_attachment('attached-image')", "whole, observed = own_attachment('attached-image', details=True)"),
("        partition = plistlib.loads(run('partition-info', ['/usr/sbin/diskutil', 'info', '-plist', whole + 's2']))", "        physical_store = apfs_image_partition(observed, image, whole)\n        partition = plistlib.loads(run('partition-info', ['/usr/sbin/diskutil', 'info', '-plist', physical_store]))"),
('volume = apfs_probe_volume(listing, whole, container)', 'volume = apfs_probe_volume(listing, whole, container, physical_store)'),
("q['blank_volume'] = {'whole_device': whole, 'container': container, 'volume': volume,", "q['blank_volume'] = {'whole_device': whole, 'physical_store': physical_store, 'container': container, 'volume': volume,")]
for old,new in changes:assert a.count(old)==1,old;a=a.replace(old,new)
target.write_text(a)
p=json.loads((tools/'macos-vm-host-v2.protocol.json').read_text())
for n,want in p['tools'].items():assert descriptor(tools/n)==want
for n,want in p['external_files'].items():assert descriptor(n)==want
prior=cache/'macos-vm-host-admission-local-v2';q=json.loads((prior/'qualification.json').read_text());assert q['host_prerequisites_passed']is False and q['error']=="ValueError('host admission child failed: partition-info')" and q['owned_image_detached']is True and q['storage']['released']
receipt=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-partition-failure';collection=json.loads((receipt/'collection.json').read_text())
for n,want in collection['files'].items():assert descriptor(receipt/n)==want
assert (prior/'qualification.json').read_bytes()==(receipt/'qualification.json').read_bytes()
sys.path.insert(0,str(tools));from macos_vm_host_v3_metrics import apfs_image_partition
owned=json.loads((prior/'attached-image.own-image.json').read_text());assert apfs_image_partition({'images':owned['matching_images']},Path(owned['image']),owned['device'])==owned['device']+'s1'
for n in [target.name,'macos_vm_host_v3_metrics.py','test_macos_vm_host_v3_metrics.py']:p['tools'][n]=descriptor(tools/n)
p['output_name']='macos-vm-host-admission-local-v3';assert not(cache/p['output_name']).exists()
records=[Path(__file__),prior/'qualification.json',prior/'protocol.json',prior/'attached-image.own-image.json',receipt/'collection.json',receipt/'collection-qualification.json']
p['external_files'].update({str(n):descriptor(n)for n in records});p['denied_paths']=sorted(set(p['denied_paths'])|{str(n)for n in cache.iterdir()if n.is_dir()})
p['limits'].insert(0,'V3 binds failed V2 actual host support and unique-s1 image mapping. Preserve V1/V2 originals. Derive exactly one physical child partition from the current exact-image observation; do not assume s1 or s2. Require the APFS container sole physical store equal that observed owned partition before mounting its sole named volume. All other no-guest/signing/offline/cleanup/storage/300s gates remain unchanged. Distinct fresh V3 output; prior failures stay failed.')
f=tools/'macos-vm-host-v3.protocol.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n');print(descriptor(f),'tools',len(p['tools']),'external',len(p['external_files']))
