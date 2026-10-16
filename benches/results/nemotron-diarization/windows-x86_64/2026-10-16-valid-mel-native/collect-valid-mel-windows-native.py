"""Verify saved Windows SDK bytes and CI outputs on macOS; execute no SDK code."""
import json,shutil,subprocess,sys,zipfile
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0/valid-mel-platform-ci-v1');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from platform_sdk_receipt import validate_native_receipt
from platform_valid_mel_native import compare_complete_60s
from package_windows import qualified_sdk
from runtime_profiles import load_profile
source=cache/'windows/nemo-valid-mel-evidence';raw=cache/'windows';target=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-valid-mel-native'
assert not target.exists()
r=json.loads((source/'receipt.json').read_text());validate_native_receipt(r)
run=json.loads((cache/'completed-run.json').read_text());job=json.loads((cache/'windows-job.json').read_text());artifact=next(a for a in json.loads((cache/'completed-artifacts.json').read_text())['artifacts']if a['id']==12065431951)
assert run['conclusion']==job['conclusion']=='success' and run['id']==job['run_id']==38940008466 and job['id']==117104967749
assert run['head_sha']==r['source_commit']==artifact['workflow_run']['head_sha']=='8fbef3e8a46a034b8e0ffa8f096409db683a923e'
archive=cache/'windows-artifact.zip';assert 'sha256:'+descriptor(archive)['sha256']==artifact['digest'] and not artifact['expired']
with zipfile.ZipFile(archive)as z:
 for item in z.infolist():
  if not item.is_dir():assert (raw/item.filename).read_bytes()==z.read(item)
build,provenance,records=qualified_sdk(raw/'nemo-valid-mel-sdk',source,load_profile()[1]);assert build==r
p=json.loads((source/'protocol.json').read_text());assert (source/'protocol.json').read_bytes()==(tools/'valid-mel-platform-native.protocol.json').read_bytes()
a=json.loads((source/'platform-llama-materialized-audit.json').read_text());saved=json.loads((repo/'benches/results/nemotron-diarization/cross-platform/2026-10-16-valid-mel-source-preflight/platform-llama-materialized-audit.json').read_text())
assert a['files']==saved['files'] and a['patch_series']==saved['patch_series'] and len(a['files'])==3425 and len(a['patch_series'])==19
assert len(a['actual_compiler_sources'])==27 and len(a['all_compiler_sources'])==97 and descriptor(source/'platform-llama-materialized-audit.json')==r['compiler_audit']
commands=json.loads((source/'commands.json').read_text());assert commands==r['commands'] and all(c['returncode']==0 for c in commands)
public=json.loads((source/'public.json').read_text());native=json.loads((source/'native-ami.json').read_text());assert public['input_pcm_f32le_sha256']==native['input_pcm_f32le_sha256']==p['cases']['actual_pcm_sha256']
comparison=compare_complete_60s(public,native,(source/'native-ami.json.f32').read_bytes(),(source/'public.f32').read_bytes());assert comparison==json.loads((source/'comparison.json').read_text())
assert (source/'native-ami.json.f32').read_bytes()==(source/'public.f32').read_bytes()
assert json.loads((source/'reference/ami.json').read_text())=={'segments':[{'start':t['start_seconds'],'end':t['end_seconds'],'speaker':t['speaker']}for t in native['turns']]}
assert (source/'reference/ami.json.f32').read_bytes()==(source/'native-ami.json.f32').read_bytes()
for index,test in enumerate(p['required_lifecycle_tests']):
 text=(source/('native-lifecycle-'+str(index)+'.stdout.log')).read_text();assert 'test '+test+' ... ok' in text and '1 passed; 0 failed; 0 ignored;' in text
target.mkdir(parents=True);omitted={}
for path in sorted(source.rglob('*')):
 if not path.is_file():continue
 rel=path.relative_to(source)
 if path.suffix in {'.json','.log','.f32','.patch','.txt'}:
  dest=target/'raw'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
 else:omitted[str(rel)]=descriptor(path)
for name in ['completed-run.json','windows-job.json','windows-live-log-attempt-v1.log','completed-artifacts.json']:shutil.copy2(cache/name,target/name)
for name,expected in p['tools'].items():
 assert descriptor(tools/name)==expected
 dest=target/'execution-tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,dest)
shutil.copy2(Path(__file__),target/Path(__file__).name)
record={'schema_version':1,'development_native_gates_passed':True,'release_qualified':False,'scope':'Fresh tagged Windows Server 2022 AMD64 AVX2/FMA/F16C source SDK and same-host native/public lifecycle only. Saved evidence collection; no neural rerun.','producer_source_commit':r['source_commit'],'collection_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'run_id':run['id'],'job_id':job['id'],'artifact_id':artifact['id'],'artifact_zip':descriptor(archive),'artifact_zip_verified_against_github_digest':True,'protocol_sha256':r['protocol_sha256'],'native_receipt':descriptor(source/'receipt.json'),'commands_passed':len(commands),'state_test_completed':True,'completed_lifecycle_tests':r['completed_lifecycle_tests'],'native_public_comparison':comparison,'independent_saved_cpu_source_replay_match':True,'ggml_patches':19,'materialized_files':3425,'actual_materialized_compiler_units':27,'all_compiler_units':97,'installed_sdk_real_files':len(r['installed_files']),'private_dlls':r['windows_private_names']['private_dlls_verified'],'retained_crt_provenance_observations_verified':8,'installed_sdk_pe_and_bytes_reverified':True,'new_native_or_neural_calls_during_collection':0,'omitted_raw_payloads':omitted,'limits':['Same-host SDK only. Package construction, offline relocation and chat/Bark/Laya coexistence remain separate.','Retained Authenticode observations and byte identity do not establish Microsoft redistribution rights.','Observed RTF is descriptive; no heldout quality, uncontended live performance, physical desktop or release claim.'],'files':{str(path.relative_to(target)):descriptor(path)for path in sorted(target.rglob('*'))if path.is_file()}}
(target/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(target),'members':len(record['files'])+1,'receipt':record['native_receipt'],'commands':len(commands),'turns':comparison['turns']}))
