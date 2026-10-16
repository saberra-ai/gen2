"""Receiver-only verification; Windows execution occurred only in CI."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path,PureWindowsPath
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');sys.path.insert(0,str(repo/'tools/nemotron-diarization'))
from compare_stream import compare
from qualify_linux_package import require_lifecycle
from windows_naming import NAMES,verify_private_outputs
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
base=Path(sys.argv[1]);out=Path(sys.argv[2]);evidence=base/'nemo-evidence';sdk=base/'nemo-sdk'
ci=json.loads(Path(str(base)+'-ci.json').read_text());r=json.loads((evidence/'receipt.json').read_text())
assert ci['conclusion']=='success' and ci['headSha']==r['source_commit'] and r['development_native_gates_passed'] is True and r['release_qualified'] is False
for name,key in [('platform_native.py','runner_sha256'),('runtime.lock.json','lock_sha256'),('windows_loader_control.py','windows_loader_control_sha256'),('windows_provenance.ps1','windows_provenance_script_sha256')]:
 source=subprocess.check_output(['git','show',ci['headSha']+':tools/nemotron-diarization/'+name],cwd=repo)
 assert hashlib.sha256(source).hexdigest()==r[key],name
naming=r['windows_private_names'];assert naming['names']==NAMES and naming['generic_imports_absent'] is True
for name,key in [('windows_private_dlls.cmake','hook_sha256'),('windows_naming.py','controller_sha256'),('windows_pe.py','inspector_sha256')]:
 source=subprocess.check_output(['git','show',ci['headSha']+':tools/nemotron-diarization/'+name],cwd=repo)
 assert hashlib.sha256(source).hexdigest()==naming[key],name
actual={p.relative_to(sdk).as_posix() for p in sdk.rglob('*') if p.is_file()}
expected=set()
for name,record in r['installed_files'].items():
 relative=PureWindowsPath(name);assert not relative.is_absolute() and '..' not in relative.parts
 p=sdk.joinpath(*relative.parts);expected.add(p.relative_to(sdk).as_posix())
 assert not p.is_symlink() and sha(p)==record['sha256'] and p.stat().st_size==record['bytes'],name
assert actual==expected
assert sha(evidence/'windows-private-pe.json')==naming['pe_report_sha256']
with tempfile.TemporaryDirectory() as temporary:
 verify_private_outputs(sdk,Path(temporary))
 assert json.loads((Path(temporary)/'windows-private-pe.json').read_text())==json.loads((evidence/'windows-private-pe.json').read_text())
old=json.loads(Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci/windows-pe-4bd34aa.json').read_text());pe=json.loads((evidence/'windows-private-pe.json').read_text())
for original,private in NAMES.items():
 for field in ['exports','export_count','forwarders']:assert pe[private][field]==old[original][field],(private,field)
public=json.loads((evidence/'public.json').read_text());native=json.loads((evidence/'native-ami.json').read_text())
comparison=compare(public,native,(evidence/'native-ami.json.f32').read_bytes(),(evidence/'public.f32').read_bytes())
assert comparison==json.loads((evidence/'comparison.json').read_text())
assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256'] and public['input_repeats']==native['input_repeats']==1
require_lifecycle((evidence/'native-lifecycle-list.stdout.log').read_text());require_lifecycle((evidence/'native-lifecycle.stdout.log').read_text(),completed=True)
control=json.loads((evidence/'windows-default-loader-control.stdout.log').read_text());assert control['passed'] and control['native_win32_error']==126 and control['load_flags']==0 and not control['default_loader_resolves_siblings']
provenance=json.loads((evidence/'windows-provenance.json').read_text(encoding='utf-8-sig'))
assert provenance['redistribution_qualified'] is False and provenance['cmake_cache']['sha256']==sha(evidence/'cmake-cache.log')
assert len(provenance['redistributables'])==8
for item in provenance['redistributables']:
 path=sdk/'bin'/item['name'];source=item['source'];installed=item['installed']
 assert sha(path)==source['sha256']==installed['sha256'] and path.stat().st_size==source['bytes']==installed['bytes']
 assert PureWindowsPath(source['path']).is_relative_to(PureWindowsPath(provenance['redist_root'])/'x64')
 assert item['signature']['status']=='Valid' and 'O=Microsoft Corporation' in item['signature']['signer']['subject']
status=next(x for x in provenance['vcpkg_metadata'] if PureWindowsPath(x['path']).name=='windows-vcpkg-status.log')
assert sha(evidence/'windows-vcpkg-status.log')==status['sha256']
job=next(x['databaseId'] for x in ci['jobs'] if x['name']=='native (windows-2022)')
with (base/'github-job.log').open('wb') as target:subprocess.run(['gh','api',f'repos/saberra-ai/gen2/actions/jobs/{job}/logs'],stdout=target,check=True)
out.mkdir(parents=True,exist_ok=False)
for path in evidence.iterdir():
 if path.is_file() and path.suffix in {'.json','.log','.f32'}:shutil.copy2(path,out/path.name)
shutil.copytree(evidence/'reference',out/'reference')
shutil.copytree(sdk/'share/licenses/nemo-speech',out/'sdk-notices')
shutil.copy2(Path(str(base)+'-ci.json'),out/'ci.json');shutil.copy2(base/'github-job.log',out/'github-job.log')
qualification={'schema_version':1,'release_qualified':False,'development_native_gates_passed':True,'private_static_dll_names_verified':True,
 'receiver_harness_note':'Initial receiver used an incorrect loader-control log basename and stopped before collection; corrected to the retained windows-default-loader-control.stdout.log without any native/model rerun.',
 'source_commit':r['source_commit'],'run_url':ci['url'],'ci_job_id':job,'sdk_regular_files_verified':len(actual),
 'native_public_comparison_recomputed':comparison,'actual_pcm_sha256':public['input_pcm_f32le_sha256'],'default_loader_control':control,
 'all_seven_lifecycle_passes_reparsed':True,'all_five_native_export_sets_unchanged_from_4bd34aa':True,
 'redist_provenance':{'eight_installed_dll_bytes_match_recorded_source':True,'observed_authenticode_status':'All8Valid/Microsoft on connected Windows build host; not independently revalidated on macOS.',
 'actual_runtime_file_product_version':sorted({x['source']['product_version'] for x in provenance['redistributables']}),'actual_compiler_file_version':provenance['compiler']['file_version'],
 'actual_compiler_product_version':provenance['compiler']['product_version'],'actual_redist_license_candidates':provenance['redist_license_candidates'],'redistribution_qualified':False},
 'limits':['Same-host Windows Server2022 development SDK only; no closed runtime package, offline relocation, desktop machine or chat/Bark/Laya coexistence qualification.',
 'Static private DLL imports/exports do not by themselves establish dynamic-load or mixed-model isolation.',
 'No applicable Microsoft distribution terms/entitlement or complete vcpkg source provenance established; only retained status bytes were independently rehashed among vcpkg metadata.',
 'Observed CI RTF exceeds1; no live/uncontended performance or held-out accuracy claim.',
 'Actual Windows execution occurred only in the named CI job. macOS receiver rehashed66SDKfiles, notices and outputs, reparsed PE/lifecycle/comparison evidence; source/signature observations remain CI evidence.']}
(out/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n');shutil.copy2(Path(__file__),out/'collect-windows-private.py')
files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
(out/'collection.json').write_text(json.dumps({'schema_version':1,'release_qualified':False,'files':files},indent=2)+'\n')
print('PASS Windows private collection',len(files),'members;',len(actual),'SDK files')
