"""Freeze real mount/device proof instead of an absent plist key; no host probe."""
from pathlib import Path
import hashlib,json
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb')as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
original=tools/'qualify_macos_vm_host_v3.py';target=tools/'qualify_macos_vm_host_v4.py';assert not target.exists()
a=original.read_text();changes=[
("source + ':tools/nemotron-diarization/macos-vm-host-v3.protocol.json'", "source + ':tools/nemotron-diarization/macos-vm-host-v4.protocol.json'"),
('from macos_vm_host_v3_metrics import attached_image_device, apfs_image_partition, apfs_probe_volume, probe_metrics', 'from macos_vm_host_v4_metrics import attached_image_device, apfs_image_partition, apfs_probe_volume, probe_metrics, mounted_volume_matches'),
("        if info['MountPoint'] != str(mount) or info['Mounted'] is not True or not mount.is_mount():\n            raise ValueError('actual exact task mount required before write')", "        mount_status = {'is_mount': mount.is_mount(), 'stat_device': mount.stat().st_dev,\n                        'parent_stat_device': mount.parent.stat().st_dev}\n        mount_report = mounted_volume_matches(info, mount, volume, container, physical_store, mount_status)"),
("'owners_enabled': info.get('OwnersEnabled'), 'filesystem': info.get('FilesystemType')", "'global_permissions_enabled': info['GlobalPermissionsEnabled'], 'filesystem': info['FilesystemType'],\n            'mount_evidence': mount_report")]
for old,new in changes:assert a.count(old)==1,old;a=a.replace(old,new)
target.write_text(a)
p=json.loads((tools/'macos-vm-host-v3.protocol.json').read_text())
for n,want in p['tools'].items():assert descriptor(tools/n)==want
for n,want in p['external_files'].items():assert descriptor(n)==want
prior=cache/'macos-vm-host-admission-local-v3';q=json.loads((prior/'qualification.json').read_text());assert q['host_prerequisites_passed']is False and q['error']=="KeyError('Mounted')" and q['owned_image_detached']is True and q['storage']['released']
receipt=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-mounted-key-failure';collection=json.loads((receipt/'collection.json').read_text())
for n,want in collection['files'].items():assert descriptor(receipt/n)==want
assert (prior/'qualification.json').read_bytes()==(receipt/'qualification.json').read_bytes()
for n in [target.name,'macos_vm_host_v4_metrics.py','test_macos_vm_host_v4_metrics.py']:p['tools'][n]=descriptor(tools/n)
p['output_name']='macos-vm-host-admission-local-v4';assert not(cache/p['output_name']).exists()
records=[Path(__file__),prior/'qualification.json',prior/'protocol.json',prior/'mounted-info.stdout.log',receipt/'collection.json',receipt/'collection-qualification.json']
p['external_files'].update({str(n):descriptor(n)for n in records});p['denied_paths']=sorted(set(p['denied_paths'])|{str(n)for n in cache.iterdir()if n.is_dir()})
p['limits'].insert(0,'V4 binds actual V3 mount metadata and the absent-Mounted-key failure. Preserve all prior failures. Before dummy write require actual OS Path.is_mount and distinct mount/parent stat devices, exact mounted DeviceNode/DeviceIdentifier/path/filesystem/container/sole physical store and writable volume. Record actual GlobalPermissionsEnabled without changing it. An absent undocumented Mounted field is not a mount signal; a present contradictory field still rejects. All other gates stay unchanged; distinct fresh V4 output.')
f=tools/'macos-vm-host-v4.protocol.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n');print(descriptor(f),'tools',len(p['tools']),'external',len(p['external_files']))
