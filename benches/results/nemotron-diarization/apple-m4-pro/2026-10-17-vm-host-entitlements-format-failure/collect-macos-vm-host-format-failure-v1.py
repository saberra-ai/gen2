"""Retain no-guest host admission format failure; saved data/hash reads only."""
from pathlib import Path
import hashlib,json,plistlib,shutil,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'macos-vm-host-admission-local-v1'
read=lambda p:json.loads(Path(p).read_text())
def descriptor(path):
 d=hashlib.sha256();size=0
 with Path(path).open('rb') as stream:
  while raw:=stream.read(1024**2):d.update(raw);size+=len(raw)
 return {'bytes':size,'sha256':d.hexdigest()}
p,q=read(root/'protocol.json'),read(root/'qualification.json');source=q['source_commit'];assert source=='7a5f3da930fd6d1e516d035bd7b7820c6b72e7b3'
assert q['host_prerequisites_passed'] is False and q['fresh_machine_qualified'] is False and q['release_qualified'] is False
assert q['error']=="InvalidFileException('Invalid file')" and 'cleanup_error'not in q and 'cleanup_metadata_errors'not in q
assert all(q[n]is False for n in ['restore_image_downloaded','virtual_machine_created','guest_booted','models_or_applications_executed'])
assert descriptor(root/'protocol.json')==q['protocol'] and (root/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-vm-host.protocol.json'],cwd=repo)
for name,want in p['tools'].items():assert descriptor(tools/name)==want
for name,want in p['external_files'].items():assert descriptor(name)==want
sys.path.insert(0,str(tools));from frozen_git_sources import verify_git_sources
from macos_vm_host_metrics import attached_image_device
verify_git_sources(repo,source,{'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()})
assert [c['name']for c in q['commands']]==['offline-controls','build','sign','signature-verify','signature-metadata','entitlements','cleanup-image','after-image']
assert all(c['returncode']==0 for c in q['commands'])
assert not any((root/n).exists() for n in ['blank-apfs.dmg','host-probe.stdout.log','create-image.stdout.log','diagnostic-reserve.bin'])
assert not (root/'mount').is_mount() and q['owned_image_detached']is True
assert q['storage']['released'] and q['storage']['reserve_reported_allocated_bytes']==256*1024**2
assert q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum']
assert (root/'entitlements.stdout.log').read_bytes()==b'[Dict]\n\t[Key] com.apple.security.virtualization\n\t[Value]\n\t\t[Bool] true\n'
try:plistlib.loads((root/'entitlements.stdout.log').read_bytes())
except plistlib.InvalidFileException:pass
else:raise ValueError('original failure not reproduced')
assert 'Signature=adhoc' in (root/'signature-metadata.stderr.log').read_text()
assert 'flags=0x10002(adhoc,runtime)' in (root/'signature-metadata.stderr.log').read_text()
for n in ['cleanup-image','after-image']:
 assert attached_image_device(plistlib.loads((root/(n+'.stdout.log')).read_bytes()),root/'blank-apfs.dmg')is None
 assert read(root/(n+'.own-image.json'))=={'image':str(root/'blank-apfs.dmg'),'device':None,'matching_images':[]}
control=read(cache/'vm-host-entitlements-xml-control-v1.json');assert control['original_abstract_output_reproduces_plist_failure']is True
for n,want in control['files'].items():assert descriptor(n)==want
assert plistlib.loads((cache/'vm-host-entitlements-xml-control-v1.stdout.log').read_bytes())=={'com.apple.security.virtualization':True}
for log,count in [('macos-vm-host-tests-local.log',6),('platform-ci/model-free-286-macos-vm-host-local.log',286)]:
 text=(cache/log).read_text();assert f'Ran {count} tests'in text and text.rstrip().endswith('OK')
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-host-entitlements-format-failure';assert not out.exists();out.mkdir()
excluded={'cleanup-image.stdout.log','after-image.stdout.log'}
for f in root.iterdir():
 if f.is_file() and f.name not in excluded and f.suffix in {'.json','.log','.sb'}:shutil.copy2(f,out/f.name)
for f in [Path(__file__),root.with_suffix('.controller.log'),cache/'prepare-macos-vm-host-v1.py',cache/'macos-vm-host-tests-local.log',cache/'platform-ci/model-free-286-macos-vm-host-local.log',cache/'vm-host-entitlements-xml-control-v1.json',cache/'vm-host-entitlements-xml-control-v1.stdout.log',cache/'vm-host-entitlements-xml-control-v1.stderr.log']:shutil.copy2(f,out/f.name)
(out/'sources').mkdir()
for n in p['tools']:shutil.copy2(tools/n,out/'sources'/n)
summary={'schema_version':1,'source_commit':source,'host_prerequisites_passed':False,'fresh_machine_qualified':False,'release_qualified':False,
'actual_commands':8,'all_child_returncodes_zero':True,'failure':'codesign abstract entitlement output is not a plist; parser stopped before helper execution or image creation',
'helper_built_and_strict_signature_verified':True,'helper_executable':descriptor(root/'vm-host-probe'),'helper_execution_started':False,
'blank_image_created':False,'image_attached_or_mounted':False,'cleanup_exact_image_absent':True,'storage_reserve_released':True,
'current_and_historical_source_pins_verified':len(p['tools']),'external_descriptors_verified':len(p['external_files']),'model_free_tests_passed':286,
'format_control_uses_read_only_codesign_xml':True,'new_helper_or_model_or_vm_calls_during_collection':0,
'local_only_global_inventory_stdout':{n:descriptor(root/n)for n in sorted(excluded)},
'limits':['Failed attempt retained. No host API result, OS image acquisition, VM creation/boot, APFS creation/mount/write, guest/app/model or fresh-machine qualification.',
'Only a new local ad-hoc helper artifact was built/signed; no host signing identity/Keychain/security change.',
'Read-only XML-format control extracts exactly the one declared entitlement without executing or altering the helper. No gate was waived.']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n');files={str(f.relative_to(out)):descriptor(f)for f in out.rglob('*')if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(out,len(files)+1,'members')
