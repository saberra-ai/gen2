"""Retain reported mount and unsupported Mounted-key failure; saved data only."""
from pathlib import Path
import hashlib,json,plistlib,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'macos-vm-host-admission-local-v3'
read=lambda p:json.loads(Path(p).read_text())
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb')as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
p,q=read(root/'protocol.json'),read(root/'qualification.json');source=q['source_commit'];assert source==subprocess.check_output(['git','rev-parse','a8700ac'],cwd=repo,text=True).strip()
assert q['host_prerequisites_passed'] is False and q['fresh_machine_qualified']is False and q['release_qualified']is False
assert q['error']=="KeyError('Mounted')" and 'cleanup_error'not in q and 'cleanup_metadata_errors'not in q
assert all(q[n]is False for n in ['restore_image_downloaded','virtual_machine_created','guest_booted','models_or_applications_executed'])
assert descriptor(root/'protocol.json')==q['protocol'] and(root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-vm-host-v3.protocol.json'],cwd=repo)
for n,want in p['tools'].items():assert descriptor(tools/n)==want
for n,want in p['external_files'].items():assert descriptor(n)==want
sys.path.insert(0,str(tools));from frozen_git_sources import verify_git_sources
from macos_vm_host_v3_metrics import attached_image_device,probe_metrics,apfs_image_partition,apfs_probe_volume
verify_git_sources(repo,source,{'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()})
expected=['offline-controls','build','sign','signature-verify','signature-metadata','entitlements','architecture','deployment','dependencies','host-probe','before-image','create-image','attach-image','attached-image','partition-info','apfs-list','before-mount','mount-volume','mounted-info','cleanup-image','detach-image','after-image']
assert[c['name']for c in q['commands']]==expected
assert all(c['returncode']==0 for c in q['commands'])
assert probe_metrics(read(root/'host-probe.stdout.log'))==q['probe'] and q['probe']['virtualization_supported']is True
assert plistlib.loads((root/'entitlements.stdout.log').read_bytes())==q['entitlements']=={'com.apple.security.virtualization':True}
assert descriptor(root/'vm-host-probe')==q['probe_executable'] and descriptor(root/'blank-apfs.dmg')==q['detached_image']
assert q['detached_image']['bytes']==256*1024**2 and not(root/'mount').is_mount()
assert q['owned_image_detached']is True and q['storage']['released'] and q['storage']['reserve_reported_allocated_bytes']==256*1024**2
assert q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and not(root/'diagnostic-reserve.bin').exists()
assert 'blank_volume' not in q
attached=plistlib.loads((root/'attached-image.stdout.log').read_bytes());whole=attached_image_device(attached,root/'blank-apfs.dmg');assert whole==read(root/'attached-image.own-image.json')['device']
entities=read(root/'attached-image.own-image.json')['matching_images'][0]['system-entities'];physical=[v['dev-entry']for v in entities if v['dev-entry'].startswith(whole+'s')]
assert physical==[whole+'s1']
partition=plistlib.loads((root/'partition-info.stdout.log').read_bytes());container=partition['APFSContainerReference'];observed=apfs_image_partition(attached,root/'blank-apfs.dmg',whole);assert observed==physical[0]
listing=plistlib.loads((root/'apfs-list.stdout.log').read_bytes());volume=apfs_probe_volume(listing,whole,container,observed)
mounted=plistlib.loads((root/'mounted-info.stdout.log').read_bytes());assert 'Mounted' not in mounted and mounted['MountPoint']==str(root/'mount') and mounted['DeviceNode']==volume and mounted['FilesystemType']=='apfs'
assert attached_image_device(plistlib.loads((root/'cleanup-image.stdout.log').read_bytes()),root/'blank-apfs.dmg')==whole
for n in ['before-image','after-image']:
 assert attached_image_device(plistlib.loads((root/(n+'.stdout.log')).read_bytes()),root/'blank-apfs.dmg')is None
for log,count in [('macos-vm-host-tests-local.log',6),('platform-ci/model-free-288-macos-vm-host-v3-local.log',288)]:
 text=(cache/log).read_text();assert f'Ran {count} tests'in text and text.rstrip().endswith('OK')
excluded={n+'.stdout.log'for n in ['before-image','attached-image','before-mount','cleanup-image','after-image']}
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-mounted-key-failure';assert not out.exists();out.mkdir()
for f in root.iterdir():
 if f.is_file()and f.name not in excluded and f.suffix in {'.json','.log','.sb'}:shutil.copy2(f,out/f.name)
for f in [Path(__file__),root.with_suffix('.controller.log'),cache/'prepare-macos-vm-host-v3.py',cache/'macos-vm-host-tests-local.log',cache/'platform-ci/model-free-288-macos-vm-host-v3-local.log']:shutil.copy2(f,out/f.name)
(out/'sources').mkdir()
for n in p['tools']:shutil.copy2(tools/n,out/'sources'/n)
summary={'schema_version':1,'source_commit':source,'host_prerequisites_passed':False,'fresh_machine_qualified':False,'release_qualified':False,
'actual_commands':22,'zero_status_commands':22,'failure':'mounted-info has no Mounted boolean; checker fails before filesystem stat/write control',
'actual_virtualization_support_query_passed':True,'four_cpu_eight_gib_within_host_limits':True,'one_entitlement_adhoc_hardened_helper_executed':True,
'blank_image_create_attach_and_mount_commands_passed':True,'observed_store_container_volume_bridge_verified':True,'mountpoint_and_device_reported':True,
'filesystem_stat_or_write_control_passed':False,'exact_image_detached':True,'storage_reserve_released':True,
'actual_physical_partition':observed,'actual_container':container,'actual_volume':volume,'mounted_global_permissions_enabled':mounted['GlobalPermissionsEnabled'],
'helper':q['probe_executable'],'detached_blank_image':q['detached_image'],
'current_historical_source_pins_verified':len(p['tools']),'external_descriptors_verified':len(p['external_files']),'model_free_tests_passed':288,
'new_helper_or_model_or_vm_or_image_calls_during_collection':0,'local_only_global_inventory_stdout':{n:descriptor(root/n)for n in sorted(excluded)},
'limits':['Original V3 remains failed. Actual mount command/metadata and safe image bridge verify, but absent Mounted key stops before independent stat or file write.',
'No blank-volume write permission, root-owned Data permission, OS acquisition/restore, VM creation/boot, guest/app/model, GUI/microphone, fresh-machine or release qualification.',
'Exact owned image is detached and mount absent. Global inventory, image/executable remain local by hash.']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');files={str(f.relative_to(out)):descriptor(f)for f in out.rglob('*')if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members',source)
