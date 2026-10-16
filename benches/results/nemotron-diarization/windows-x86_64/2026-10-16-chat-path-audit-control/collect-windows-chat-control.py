"""Preserve an actual failed path-audit gate; do not promote partial chat runs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from compare_stream import compare
from qualify_linux_package import require_lifecycle
from windows_archive import digest, restore_archive
from package_windows import verify
from windows_runtime_audit import loaded_modules

cache=Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci')
base=cache/'windows-chat-38893676099'; package=cache/'windows-chat-package-38893676099'
output=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-chat-path-audit-control'
read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
ci=read(Path(str(base)+'-ci.json'));receipt=read(base/'receipt.json');frozen=read(base/'frozen-inputs.json');chat=read(base/'chat-frozen-inputs.json')
source='826f62f4a801b4d3fd605cb5d9f7f4a6ac7f8801'
assert ci['conclusion']=='failure' and ci['headSha']==receipt['source_commit']==frozen['source_commit']==chat['source_commit']==source
assert receipt['offline_container_gates_passed'] is True and receipt['chat_diarization_gates_passed'] is False
assert receipt['error']=='unsafe, duplicate or incomplete observed Windows module' and not (base/'diar-first').exists()
for name,expected in {**frozen['tools'],**chat['tools']}.items():
 assert hashlib.sha256(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+name],cwd=repo)).hexdigest()==expected
assert digest(base/'frozen-inputs.json')==receipt['frozen_inputs_sha256'] and digest(base/'chat-frozen-inputs.json')==receipt['chat_frozen_inputs_sha256']
archive=read(package/'development-windows-runtime-files.json')
helper_manifest=read(base/'development-windows-chat-helpers-files.json')
assert helper_manifest==chat['chat_helper_archive']
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp)
 runtime=restore_archive(package/archive['archive'],archive,root/'runtime ü');manifest=verify(runtime)
 assert digest(runtime/'package.json')==receipt['package_manifest_sha256']
 helpers=restore_archive(base/helper_manifest['archive'],helper_manifest,root/'helpers ü')
 for name,row in helper_manifest['files'].items():
  assert chat['files']['bin' if name=='.' else 'bin/'+name]==row
  if name in frozen['helpers']:
   assert {'bytes':(helpers/name).stat().st_size,'sha256':digest(helpers/name)}==frozen['helpers'][name]
legacy=types.ModuleType('legacy_path_audit')
exec(compile(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/windows_runtime_audit.py'],cwd=repo),'producer_windows_runtime_audit.py','exec'),legacy.__dict__)
native=read(base/'native/results/native.json');raw=(base/'native/results/native.json.f32').read_bytes()
assert compare(read(base/'public/results/public.json'),native,raw,(base/'public/results/public.f32').read_bytes())==receipt['native_public_comparison']
require_lifecycle((base/'lifecycle-list/results/child.stdout.log').read_text())
require_lifecycle((base/'lifecycle/results/child.stdout.log').read_text(),completed=True)
commands=read(base/'commands.json');assert all(row['returncode']==0 for row in commands)
for name in ['native','public','lifecycle-list','lifecycle','missing-private-library','isolated','chat-first']:
 inspected,=read(base/(name+'-inspect.stdout.log')); created=(base/(name+'-create.stdout.log')).read_text(encoding='utf-8-sig').strip()
 assert inspected['Id']==created and len(created)==64 and inspected['State']['ExitCode']==0 and not inspected['State']['Running']
 assert inspected['HostConfig']['Isolation']=='process' and inspected['HostConfig']['NetworkMode']=='none'
 removed=[row for row in commands if row['name']==name+'-remove'];assert len(removed)==1 and removed[0]['argv'][-1]==created
 assert read(base/name/'results/controls.json')['input_write_win32_error']==5
 assert read(base/name/'results/child.json')['completed'] is True
isolated=read(base/'isolated/results/isolated/summary.json');mixed=read(base/'chat-first/results/chat-first/summary.json')
assert isolated['fingerprints']==mixed['fingerprints']==chat['fingerprints']
assert isolated['generation']==mixed['generation']=={'max_tokens':8,'temperature':0.0,'seed':0}
assert isolated['settings']==mixed['settings'] and mixed['prompt']==isolated['prompt']=='Reply with the single word hello.'
assert len(isolated['chat_outputs'])==1 and len(mixed['chat_outputs'])==5
text=isolated['chat_outputs']['isolated']['text'];assert text
for value in [*isolated['chat_outputs'].values(),*mixed['chat_outputs'].values()]:
 assert value['text']==text and value['offload']['layers']==0
comparisons={}
for stage in ['first','concurrent','reloaded']:
 path=base/'chat-first/results/chat-first'/(stage+'.json');public=read(path)
 assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']
 comparisons[stage]=compare(public,native,raw,path.with_suffix('.f32').read_bytes())
a,b=mixed['concurrent_intervals']['chat_micros'],mixed['concurrent_intervals']['diarization_micros']
assert max(a[0],b[0])<min(a[1],b[1]) and a==mixed['chat_outputs']['concurrent']['api_call_micros']
assert b==read(base/'chat-first/results/chat-first/concurrent.json')['api_call_micros'] and mixed['reservation_after_shutdown_mb']==0
helper={key:read(base/'chat-helper-pe.json')[key]for key in ['bytes','sha256']}
openmp={key:read(base/'chat-openmp-provenance.json')[key]for key in ['bytes','sha256']}
audits={}
for order,stages in [('isolated',['isolated']),('chat-first',['both-loaded','reloaded'])]:
 for stage in stages:
  rows=read(base/order/'results'/order/(stage+'.modules.json'))
  observation={'stable_snapshots':1,'modules':rows}
  audits[order+'/'+stage]=loaded_modules(observation,manifest['libraries'],'chat-coexistence.exe',helper,private_required=order!='isolated',openmp=openmp)
  if order=='chat-first':
   try:legacy.loaded_modules(observation,manifest['libraries'],'chat-coexistence.exe',helper,private_required=True,openmp=openmp)
   except ValueError as error:assert str(error)==receipt['error']
   else:raise AssertionError('original auditor did not reproduce the recorded rejection')
output.mkdir(parents=True,exist_ok=False)
for path in base.iterdir():
 if path.is_file() and path.suffix!='.zip':shutil.copy2(path,output/path.name)
 elif path.is_dir():shutil.copytree(path,output/path.name)
(output/'package').mkdir()
for name in ['receipt.json','development-windows-runtime-files.json']:shutil.copy2(package/name,output/'package'/name)
shutil.copy2(package/'runtime/package.json',output/'package/package.json')
for suffix in ['json','log']:shutil.copy2(Path(str(base)+'-ci.'+suffix),output/('ci.'+suffix))
shutil.copy2(__file__,output/'collect-windows-chat-control.py')
qualification={'schema_version':1,'source_commit':source,'ci_url':ci['url'],'full_windows_chat_gate_passed':False,
 'release_qualified':False,'redistribution_qualified':False,'diar_first_executed':False,
 'actual_failure':'Original stage auditor rejected actual extended DOS C: module path spelling after successful chat-first helper exit.',
 'offline_native_public_lifecycle_prerequisites_reparsed':True,'all_seven_managed_container_ids_and_cleanup_reparsed':True,
 'partial_chat_observations':6,'partial_chat_text':text,'partial_chat_first_native_comparisons_recomputed':comparisons,
 'partial_api_call_overlap_micros':min(a[1],b[1])-max(a[0],b[0]),
 'repaired_auditor_observed_private_counts':{name:len(value['private_native'])for name,value in audits.items()},
 'repair_scope':'Receiver diagnosis only; new both-order actual Windows run remains required.',
 'repair_tool_sha256':digest(repo/'tools/nemotron-diarization/windows_runtime_audit.py'),
 'limits':receipt['limits']+['Opaque runtime/helper archives rehashed/restored on macOS; downloaded CI partial observations are not a successful full chat qualification.']}
(output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
files={path.relative_to(output).as_posix():{'bytes':path.stat().st_size,'sha256':digest(path)}for path in output.rglob('*')if path.is_file()}
(output/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'evidence_files':len(files),'full_gate_passed':False,'partial_comparisons':len(comparisons)}))
