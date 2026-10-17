"""Verify delivered construction payloads without loading native code."""
from pathlib import Path
import hashlib,json,re,shutil,struct,subprocess,sys,zipfile
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'bounded-macos-construction-ci-v1';saved=root/'evidence'
source='aa06657316e16d996886e55b93bc1bc25251b620'
read=lambda path:json.loads(Path(path).read_text())
def desc(path):
 raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p=read(saved/'protocol.json')
for group,base in [('tools',tools),('repository_files',repo)]:
 for name,expected in p[group].items():assert desc(base/name)==expected,name
sys.path.insert(0,str(tools))
from ctest_output import require_single_test_pass
from frozen_git_sources import verify_git_sources
from macos_cpu_architecture import audit_portable_commands
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files,verify_package_provenance
from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes,profile_provenance
from sdk_archive import inventory,restore_archive
from build_valid_mel_candidate import source_inventory
verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
assert (saved/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-bounded-construction.protocol.json'],cwd=repo)
ci,api=read(root/'run.json'),read(root/'artifacts.json')['artifacts'];assert len(api)==1;a=api[0]
assert ci['databaseId']==39003880853 and ci['headSha']==source and ci['status']=='completed' and ci['conclusion']=='success'
assert a['id']==12089814731 and a['workflow_run']['id']==ci['databaseId'] and a['workflow_run']['head_sha']==source
assert desc(root/'artifact.zip')=={'bytes':17622146,'sha256':'16afdf64e55028a6ba087ec50effadb2e30c199a2a2fd36c1885fc0becc89570'}
assert a['digest']=='sha256:'+desc(root/'artifact.zip')['sha256'] and a['size_in_bytes']==17622146
before={str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
with zipfile.ZipFile(root/'artifact.zip') as archive:
 assert len(archive.infolist())==len(before)==200 and {m.filename for m in archive.infolist()}==set(before)
 for name in before:assert archive.read(name)==(saved/name).read_bytes()
assert not any(path.is_symlink() for path in saved.rglob('*'))
q=read(saved/'qualification.json');assert q['source_commit']==source and q['construction_and_negative_c_api_controls_passed'] is True
assert q['valid_models_loaded']==q['frontend_or_neural_calls']==0 and q['release_qualified'] is False and 'error' not in q
assert q['protocol']==desc(saved/'protocol.json')
assert [c['name'] for c in q['commands']]==['sandbox-controls','prepare','build','package','failed-load-probe-build','failed-loads']
assert all(c['returncode']==0 and not c.get('forced_process_group_termination') for c in q['commands'])
lock_path,lock=load_profile(BOUNDED_PROFILE);assert profile_provenance(lock_path,lock)==p['runtime_provenance']
assert (saved/'build/runtime-profile.json').read_bytes()==lock_path.read_bytes()
assert (saved/'build/source.patch').read_bytes()==compatibility_patch_bytes(lock)
build=read(saved/'build/build.json')
assert build['runtime_profile']==BOUNDED_PROFILE and build['runtime_reported_version']==lock['runtime']['reported_version']
assert build['model_free_state_test_completed'] and build['version_function_only_executed']
assert not build['model_instantiated_or_inference_performed'] and not build['artifact_admitted_for_inference'] and not build['release_qualified']
assert all(c['returncode']==0 for c in build['commands'])
assert (saved/'build/version-only.stdout.log').read_text().strip()==lock['runtime']['reported_version']
require_single_test_pass((saved/'build/model-free-state-test.stdout.log').read_text(),'diar_state')
architecture=audit_portable_commands(read(saved/'build/compile_commands.json'));assert architecture==build['cpu_architecture']
materialized=read(saved/'build/llama-materialized-audit.json')
prior=cache/'pooled-macos-construction-ci-v1/evidence/build/llama-materialized-audit.json';old=read(prior)
assert materialized['exact_independent_replay_match'] is True and len(materialized['patch_series'])==19
assert len(materialized['files'])==3425 and materialized['files']==old['files'] and materialized['patch_series']==old['patch_series']
assert desc(saved/'build/pristine-llama.tar')==materialized['archive']==old['archive']
local_ggml=cache/'pooled-cpu-construction-local-v1/source/build/cpu-diar/_deps/llama.cpp'
assert source_inventory(local_ggml)==materialized['files']
compiler=read(saved/'build/compiler-inputs.json');assert compiler==build['actual_compiler_sources'] and len(compiler)==98
remote_ggml=Path(materialized['compiled_tree_path']);remote_source=remote_ggml.parents[3]
pristine=cache/'NeMo-Speech.cpp'
for name,expected in compiler.items():
 path=Path(name)
 if path.is_relative_to(remote_ggml):assert desc(local_ggml/path.relative_to(remote_ggml))==expected,name
 else:
  relative=path.relative_to(remote_source).as_posix()
  if relative in lock['runtime']['source_sha256']:
   assert expected['sha256']==lock['runtime']['source_sha256'][relative],name
   patch_entries=[item for item in lock['runtime']['compatibility_patches'] if relative in item['source_sha256_after']]
   if patch_entries:assert patch_entries[-1]['source_sha256_after'][relative]==expected['sha256']
  else:
   raw=subprocess.check_output(['git','show',lock['runtime']['revision']+':'+relative],cwd=pristine)
   assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected,name
restored={}
for stem,key in [('pooled-cpu-sdk','sdk_archive'),('pooled-cpu-runtime','package_archive'),('construction-controls','controls_archive')]:
 manifest=read(saved/(stem+'-files.json'));assert {k:v for k,v in manifest.items() if k!='files'}==q[key]
 destination=root/('restored-'+stem);assert not destination.exists()
 restore_archive(saved/manifest['archive'],manifest,destination);assert inventory(destination)==manifest['files'];restored[key]=destination
sdk,package,binary=restored['sdk_archive'],restored['package_archive'],restored['controls_archive']/'failed-load-probe'
manifest=read(package/'package.json');verify_files(package,manifest['files']);verify_package_provenance(package,manifest)
assert (package/'provenance/build.json').read_bytes()==(saved/'build/build.json').read_bytes() and desc(binary)==q['binary']
def sections(path):
 raw=path.read_bytes();magic,cpu,subcpu,kind,count,size,flags,reserved=struct.unpack_from('<8I',raw)
 assert magic==0xfeedfacf and cpu==0x100000c and kind==6
 end=32+size;assert end<=len(raw);cursor=32;result={}
 for _ in range(count):
  command,length=struct.unpack_from('<II',raw,cursor);assert length>=8 and cursor+length<=end
  if command==0x19:
   segment=struct.unpack_from('<II16sQQQQiiII',raw,cursor);n=segment[-2];assert length==72+n*80
   for index in range(n):
    row=struct.unpack_from('<16s16sQQIIIIIIII',raw,cursor+72+index*80)
    key=(row[1].rstrip(b'\0').decode(),row[0].rstrip(b'\0').decode());assert key not in result
    section_size,offset,section_flags=row[3],row[4],row[8];zero=section_flags&0xff in {1,0xc,0x12}
    assert zero or offset>=end and offset+section_size<=len(raw)
    result[key]=(row[2:],None if zero else hashlib.sha256(raw[offset:offset+section_size]).hexdigest())
  cursor+=length
 assert cursor==end and result;return result
assert {path.name for path in (saved/'source-libraries').iterdir()}==set(manifest['libraries'])
for name,item in manifest['libraries'].items():
 original=saved/'source-libraries'/name
 assert desc(original)['sha256']==item['source_sha256'] and sections(original)==sections(package/'lib'/name)
 if item['sdk_library'] is not None:assert desc(original)==desc(sdk/'lib'/item['sdk_library'])
remote_executable=Path(q['loaded_images']['executable']);remote_package=Path(next(iter(q['loaded_images']['private_libraries']))).parent.parent
def remap(match):
 path=match[2]
 if path.startswith(('/usr/lib/','/System/Library/')):return match[0]
 if path==str(remote_executable):return match[1]+str(binary)
 relative=Path(path).relative_to(remote_package/'lib');assert relative.as_posix() in manifest['libraries'];return match[1]+str(package/'lib'/relative)
trace=(saved/'failed-loads.stderr.log').read_text()
audited=audit_loaded_libraries(re.sub(r'(dyld\[\d+\]: <[A-Fa-f0-9-]+> )(/[^\r\n]+)',remap,trace),package,manifest,binary)
assert json.loads(json.dumps(audited).replace(str(binary),str(remote_executable)).replace(str(package),str(remote_package)))==q['loaded_images']
assert audited['private_library_count']==92
control=q['failed_loads'];assert control==read(saved/'failed-loads.stdout.log')
assert control['failed_model_construction_attempts']==101 and control['valid_models_loaded']==control['frontend_or_neural_calls']==0
assert control['thread_growth_allowance']==2 and control['maximum_requested_settling_wait_milliseconds']==100
counts,immediate,waits=[control[key] for key in ['threads_after_each_failure','immediate_thread_counts','requested_settling_wait_milliseconds']]
assert len(counts)==len(immediate)==len(waits)==101 and min(counts+immediate)>0 and max(counts)<=counts[0]+2
assert all(w in range(10,101,10) for w in waits)
assert q['storage']['released'] and q['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum'] and q['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
assert before=={str(path.relative_to(saved)):desc(path) for path in saved.rglob('*') if path.is_file()}
out=repo/'benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-cpu-construction';assert not out.exists();out.mkdir()
for name in before:
 if name.startswith('source-libraries/') or name.endswith(('.tar','.tar.gz')):continue
 target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(saved/name,target)
for path in [root/'run.json',root/'artifacts.json',Path(__file__),cache/'extract-bounded-construction-v1.py',cache/'platform-ci/model-free-258-bounded-cpu-local.log']:
 shutil.copy2(path,out/path.name)
shutil.copy2(package/'package.json',out/'restored-package.json')
summary={'schema_version':1,'source_commit':source,'ci_url':ci['url'],'artifact':a,'opaque_zip':desc(root/'artifact.zip'),
 'all200delivered_members_verified':True,'all_saved_payloads_bound_before_and_after':before,'model_calls':0,'collection_native_library_or_helper_calls':0,
 'construction_and_negative_c_api_controls_passed':True,'private_libraries_rehashed_and_audit_replayed':92,'compiler_units_checked':98,
 'ggml_files_compared_to_independently_qualified_inventory':3425,'same19patches_and_pristine_archive':True,
 'sdk_runtime_controls_archives_restored_with_exact_inventories':{key:len(inventory(path)) for key,path in restored.items()},
 'all_original_and_private_library_sections_match':True,'failed_construction_attempts':101,'immediate_thread_counts':sorted(set(immediate)),
 'settled_thread_counts':sorted(set(counts)),'requested_settling_wait_milliseconds':sorted(set(waits)),
 'host':read(saved/'host.json'),'runtime_profile':BOUNDED_PROFILE,'release_qualified':False,'limits':p['limits']}
(out/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
entries={str(path.relative_to(out)):desc(path) for path in out.rglob('*') if path.is_file()};(out/'collection.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(out,len(entries)+1,'members');print(json.dumps({key:summary[key] for key in ['immediate_thread_counts','settled_thread_counts','sdk_runtime_controls_archives_restored_with_exact_inventories']},indent=2))
