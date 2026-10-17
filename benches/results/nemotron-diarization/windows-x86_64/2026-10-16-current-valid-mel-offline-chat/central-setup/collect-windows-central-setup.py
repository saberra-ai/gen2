"""Reparse actual Windows prerequisite setup/state evidence without executing it."""
import hashlib
import io
import tarfile
import tempfile
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
base, ci_path, output = map(Path,sys.argv[1:])
ci=json.loads(ci_path.read_text(encoding='utf-8-sig'));source=ci['headSha']
source_temp=tempfile.TemporaryDirectory();source_root=Path(source_temp.name)
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',source,'tools/nemotron-diarization'],cwd=repo))) as bundle:
 bundle.extractall(source_root,filter='data')
sys.path.insert(0,str(source_root/'tools/nemotron-diarization'))
from windows_archive import digest
from windows_central_state import audit_state, require_setup_success, require_signature

cache=Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci')
read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
receipt=read(base/'receipt.json');frozen=read(base/'frozen-inputs.json')
assert ci['conclusion']=='success' and ci['headSha']==receipt['source_commit']==frozen['source_commit']==source
assert receipt['setup_and_fresh_image_state_passed'] is True
assert all(receipt[key] is False for key in ['central_runtime_qualified','release_qualified','redistribution_qualified'])
assert digest(base/'frozen-inputs.json')==receipt['frozen_inputs_sha256']
for name,expected in frozen['tools'].items():
 data=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)
 assert hashlib.sha256(data).hexdigest()==expected==digest(source_root/'tools/nemotron-diarization'/name)
minimum_path=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static/minimum-x64-cab-inspection.json'
assert digest(minimum_path)==frozen['minimum_static_receipt_sha256']
minimum=read(minimum_path);lock=read(repo/'tools/nemotron-diarization/windows-vc-runtime.lock.json')
expected={row['runtime_name']:{key:row[key]for key in ['bytes','sha256']}for row in minimum['selected_dlls']}
installer=repo.parent.parent/'.cache'
installer=Path('/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/VC_redist.x64.exe')
assert installer.stat().st_size==lock['installer']['size'] and digest(installer)==lock['installer']['sha256']==receipt['installer_sha256']
signature=read(base/'installer-host-signature.json');require_signature(signature)
assert digest(base/'installer-host-signature.json')==frozen['installer_host_signature_sha256']
assert {key:signature[key]for key in ['bytes','sha256']}=={'bytes':lock['installer']['size'],'sha256':lock['installer']['sha256']}
inputs={'.':{'type':'directory'},lock['installer']['name']:{'type':'file','bytes':installer.stat().st_size,'sha256':digest(installer)}}
for filename,tool in [('setup.ps1','windows_central_setup.ps1'),('controls.ps1','windows_smoke_controls.ps1')]:
 raw=subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+tool],cwd=repo)
 inputs[filename]={'type':'file','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
raw=b'Gen2 offline read-only input control';inputs['control.txt']={'type':'file','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
raw=(json.dumps({'files':expected,'installer':lock['installer'],'version':lock['version']},indent=2)+'\n').replace('\n','\r\n').encode()
inputs['expected.json']={'type':'file','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
assert frozen['inputs']==inputs
base_image,=read(base/'base-image-inspect.stdout.log');image,=read(base/'provisioned-image-inspect.stdout.log')
assert base_image==frozen['base_image'] and base_image['Id']==receipt['base_image_id']
image_lock=read(repo/'tools/nemotron-diarization/windows-container.lock.json')
assert base_image['Id']==image_lock['image_id'] and image_lock['repository']+'@'+image_lock['digest'] in base_image['RepoDigests']
assert image['Id']==receipt['provisioned_image_id']==(base/'freeze-provisioned-image.stdout.log').read_text(encoding='utf-8-sig').strip()
assert image['RootFS']['Layers'][:-1]==base_image['RootFS']['Layers'] and image['Architecture']=='amd64' and image['Os']=='windows'
commands=read(base/'commands.json');assert all(row['returncode']==0 for row in commands)
states={}
for name,mode,image_id in [('install','Install',base_image['Id']),('fresh-state','Observe',image['Id'])]:
 state=read(base/name/'state.json');states[name]=state
 assert state['completed'] is True and not state.get('error') and state['mode']==mode
 if name=='install':
  assert state['installer_executed'] is True
  require_setup_success(state,expected,lock['version'])
 else:
  assert state['installer_executed'] is False
  audit_state(state['before'],expected,lock['version']);audit_state(state['after'],expected,lock['version'])
 control=read(base/name/'controls.json')
 assert control['input_read'] and control['input_write_denied'] and control['input_write_win32_error']==5
 assert control['input_sha256']==inputs['control.txt']['sha256'] and control['process_architecture']=='AMD64'
 assert not control['operational_non_loopback_interfaces'] and control['development_tools_absent_from_path'] and control['visual_studio_directory_absent']
 inspected,=read(base/(name+'-inspect.stdout.log'));created=(base/(name+'-create.stdout.log')).read_text(encoding='utf-8-sig').strip()
 assert inspected['Id']==created and len(created)==64 and inspected['Image']==image_id
 assert inspected['State']['ExitCode']==0 and not inspected['State']['Running']
 assert inspected['HostConfig']['Isolation']=='process' and inspected['HostConfig']['NetworkMode']=='none'
 mounts={row['Destination'].lower():row for row in inspected['Mounts']}
 assert set(mounts)=={'c:\\inputs','c:\\results'} and mounts['c:\\inputs']['RW'] is False and mounts['c:\\results']['RW'] is True
 assert PureWindowsPath(mounts['c:\\inputs']['Source'])==PureWindowsPath('D:/a/_temp/windows-valid-mel-runtime/central-setup/setup inputs ü')
 assert PureWindowsPath(mounts['c:\\results']['Source'])==PureWindowsPath('D:/a/_temp/windows-valid-mel-runtime/central-setup')/name
 for suffix in ['start','inspect','remove']:
  row,=[row for row in commands if row['name']==name+'-'+suffix];assert row['argv'][-1]==created
assert receipt['actual_central_files']==states['fresh-state']['after']['files']
assert digest(base/'install/state.json')==receipt['installed_state_sha256'] and digest(base/'fresh-state/state.json')==receipt['fresh_state_sha256']
assert states['install']['installer_exit_code']==receipt['actual_installer_exit_code']==0
assert all(row['present'] is False for row in states['install']['before']['files'])
output.mkdir(parents=True,exist_ok=False)
for path in base.iterdir():
 if path.is_file():shutil.copy2(path,output/path.name)
 elif path.is_dir():shutil.copytree(path,output/path.name)
for suffix in ['json','log']:shutil.copy2(ci_path.with_suffix('.'+suffix),output/('ci.'+suffix))
shutil.copytree(base.parent/'prerequisites',output/'prerequisites')
shutil.copy2(__file__,output/'collect-windows-central-setup.py')
qualification={'schema_version':1,'source_commit':source,'ci_url':ci['url'],'setup_and_fresh_image_state_passed':True,
 'installer_host_signature_reparsed':True,'exact_publisher_installer_and_all_input_hashes_recomputed':True,
 'installer_exit_code':0,'all_nine_targeted_dlls_absent_before_setup':True,
 'installed_and_fresh_nine_amd64_dll_contracts_reparsed':True,'both_registry_views_and_no_pending_restart_reparsed':True,
 'both_managed_container_ids_mounts_network_cleanup_reparsed':True,'provisioned_image_exact_base_plus_one_layer':True,
 'central_runtime_qualified':False,'release_qualified':False,'redistribution_qualified':False,
 'base_image_id':base_image['Id'],'provisioned_image_id':image['Id'],
 'limits':receipt['limits']+['macOS receiver rehashed fixed installer/frozen inputs and reparsed CI metadata; actual Windows signatures/setup/System32 hashes/immutability/registry remain recorded CI observations.',
 'No Gen2/native/public/lifecycle/chat calls performed for this central image. Image and Microsoft DLL/installer bytes are not included in this Git receipt.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
files={path.relative_to(output).as_posix():{'bytes':path.stat().st_size,'sha256':digest(path)}for path in output.rglob('*')if path.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'installer_exit_code':0,'central_runtime_qualified':False}))
