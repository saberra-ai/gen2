"""Retain actual host support and failed partition query; no tool/VM/image calls."""
from pathlib import Path
import hashlib,json,plistlib,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'macos-vm-host-admission-local-v2'
read=lambda p:json.loads(Path(p).read_text())
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb')as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
p,q=read(root/'protocol.json'),read(root/'qualification.json');source=q['source_commit'];assert source==subprocess.check_output(['git','rev-parse','0d58514'],cwd=repo,text=True).strip()
assert q['host_prerequisites_passed'] is False and q['fresh_machine_qualified']is False and q['release_qualified']is False
assert q['error']=="ValueError('host admission child failed: partition-info')" and 'cleanup_error'not in q and 'cleanup_metadata_errors'not in q
assert all(q[n]is False for n in ['restore_image_downloaded','virtual_machine_created','guest_booted','models_or_applications_executed'])
assert descriptor(root/'protocol.json')==q['protocol'] and(root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-vm-host-v2.protocol.json'],cwd=repo)
for n,want in p['tools'].items():assert descriptor(tools/n)==want
for n,want in p['external_files'].items():assert descriptor(n)==want
sys.path.insert(0,str(tools));from frozen_git_sources import verify_git_sources
from macos_vm_host_metrics import attached_image_device,probe_metrics
verify_git_sources(repo,source,{'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()})
expected=['offline-controls','build','sign','signature-verify','signature-metadata','entitlements','architecture','deployment','dependencies','host-probe','before-image','create-image','attach-image','attached-image','partition-info','cleanup-image','detach-image','after-image']
assert[c['name']for c in q['commands']]==expected
assert all(c['returncode']==(1 if c['name']=='partition-info'else 0)for c in q['commands'])
assert probe_metrics(read(root/'host-probe.stdout.log'))==q['probe'] and q['probe']['virtualization_supported']is True
assert plistlib.loads((root/'entitlements.stdout.log').read_bytes())==q['entitlements']=={'com.apple.security.virtualization':True}
assert descriptor(root/'vm-host-probe')==q['probe_executable'] and descriptor(root/'blank-apfs.dmg')==q['detached_image']
assert q['detached_image']['bytes']==256*1024**2 and not(root/'mount').is_mount()
assert q['owned_image_detached']is True and q['storage']['released'] and q['storage']['reserve_reported_allocated_bytes']==256*1024**2
assert q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and not(root/'diagnostic-reserve.bin').exists()
assert not any((root/n).exists() for n in ['apfs-list.stdout.log','mount-volume.stdout.log','mounted-info.stdout.log'])
attached=plistlib.loads((root/'attached-image.stdout.log').read_bytes());whole=attached_image_device(attached,root/'blank-apfs.dmg');assert whole==read(root/'attached-image.own-image.json')['device']
entities=read(root/'attached-image.own-image.json')['matching_images'][0]['system-entities'];physical=[v['dev-entry']for v in entities if v['dev-entry'].startswith(whole+'s')]
assert physical==[whole+'s1']
failure=plistlib.loads((root/'partition-info.stdout.log').read_bytes());assert failure['Error']is True and failure['ExitCode']==1 and failure['ErrorMessage']=='Could not find disk: '+whole+'s2'
assert attached_image_device(plistlib.loads((root/'cleanup-image.stdout.log').read_bytes()),root/'blank-apfs.dmg')==whole
for n in ['before-image','after-image']:
 assert attached_image_device(plistlib.loads((root/(n+'.stdout.log')).read_bytes()),root/'blank-apfs.dmg')is None
for log,count in [('macos-vm-host-tests-local.log',6),('platform-ci/model-free-286-macos-vm-host-v2-local.log',286)]:
 text=(cache/log).read_text();assert f'Ran {count} tests'in text and text.rstrip().endswith('OK')
excluded={n+'.stdout.log'for n in ['before-image','attached-image','cleanup-image','after-image']}
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-partition-failure';assert not out.exists();out.mkdir()
for f in root.iterdir():
 if f.is_file()and f.name not in excluded and f.suffix in {'.json','.log','.sb'}:shutil.copy2(f,out/f.name)
for f in [Path(__file__),root.with_suffix('.controller.log'),cache/'prepare-macos-vm-host-v2.py',cache/'macos-vm-host-tests-local.log',cache/'platform-ci/model-free-286-macos-vm-host-v2-local.log']:shutil.copy2(f,out/f.name)
(out/'sources').mkdir()
for n in p['tools']:shutil.copy2(tools/n,out/'sources'/n)
summary={'schema_version':1,'source_commit':source,'host_prerequisites_passed':False,'fresh_machine_qualified':False,'release_qualified':False,
'actual_commands':18,'zero_status_commands':17,'failed_command':'partition-info','failure':'fixed s2 assumption differs from actual single physical s1 partition',
'actual_virtualization_support_query_passed':True,'four_cpu_eight_gib_within_host_limits':True,'one_entitlement_adhoc_hardened_helper_executed':True,
'blank_image_created_and_attached':True,'blank_image_mounted_or_written':False,'exact_image_detached':True,'storage_reserve_released':True,
'actual_physical_partition':physical[0],'attempted_partition_query':whole+'s2','helper':q['probe_executable'],'detached_blank_image':q['detached_image'],
'current_historical_source_pins_verified':len(p['tools']),'external_descriptors_verified':len(p['external_files']),'model_free_tests_passed':286,
'new_helper_or_model_or_vm_or_image_calls_during_collection':0,'local_only_global_inventory_stdout':{n:descriptor(root/n)for n in sorted(excluded)},
'limits':['Original V2 remains failed despite actual helper/API/signature/geometry and image create/attach/detach successes.',
'No mount/write test, root-owned Data permission, OS acquisition/restore, VM creation/boot, guest/app/model, GUI/microphone, fresh-machine or release qualification. Private payloads stay local.',
'Actual image mapping shows s1; never change a failed result to passing or assume a physical partition number. Global inventory and blank image/executable remain local by hash.']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');files={str(f.relative_to(out)):descriptor(f)for f in out.rglob('*')if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members',source)
