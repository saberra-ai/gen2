"""Admit saved production SDK/public/lifecycle evidence without native execution."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path,PureWindowsPath
from unittest.mock import patch
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-q8-candidate-ci-v2';admitted=root/'delivery-admission-v1';delivered=admitted/'delivered'
source='fee2177adcd7b75b488c67bea5a593ea17df65de';run_id,job_id=39063644421,117486629205

def descriptor(path):
 value,size=hashlib.sha256(),0
 with path.open('rb')as stream:
  while block:=stream.read(1024**2):size+=len(block);value.update(block)
 return {'bytes':size,'sha256':value.hexdigest()}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def admit_execution_commands(commands, ci_root, ci_repo, public_bindings, order, lifecycle):
 public,=[PureWindowsPath(name)for name in public_bindings
          if PureWindowsPath(name).name=='nemotron_valid_mel_stream.exe'
          and not PureWindowsPath(name).is_relative_to(ci_root/'evidence')]
 assets=ci_root/'assets';evidence=ci_root/'evidence'
 probe=ci_root/'sdk/bin/gen2_native_stream_probe_valid_mel.exe'
 library=ci_root/'sdk/bin/gen2_nemo_windows_nemo_speech_asr_c.dll'
 expected=[]
 for index,kind in enumerate(order):
  phase=f'case-{index}-{kind}';target=evidence/(phase+'.json')
  if kind=='native':argv=[probe,assets/'Nemotron-3-Diarization.q8_0.gguf',assets/'ami_en2002d_2132.wav','1','-1',target]
  else:argv=[public,assets,library,assets/'ami_en2002d_2132.wav','1',PureWindowsPath(str(target)+'.f32')]
  expected.append((phase,list(map(str,argv))))
 cargo=['cargo','test','--locked','--release','--no-default-features','--features','backend-diar-nemo','--test','diarization_native','--']
 expected.append(('native-lifecycle-list',cargo+['--ignored','--list']))
 for index,test in enumerate(lifecycle):
  expected.append(('native-lifecycle-'+str(index),cargo+['--ignored','--exact',test,'--test-threads=1']))
 actual=[row for row in commands if row['name'].startswith(('case-','native-lifecycle'))]
 assert [row['name']for row in actual]==[name for name,argv in expected]
 for row,(name,argv)in zip(actual,expected):
  assert row['command']==argv and PureWindowsPath(row['cwd'])==ci_repo
 return {'history_commands':len(order),'lifecycle_commands':len(lifecycle)+1}

identity=descriptor(Path(__file__));admission=read(admitted/'qualification.json');supervision=read(root/'supervision-replay-v1.json')
assert admission['source_commit']==source and admission['run_id']==run_id and admission['job_id']==job_id
assert descriptor(root/'artifact.zip')==admission['archive']
assert descriptor(cache/'admit-windows-q8-candidate-delivery-v2.py')==admission['collector']
assert descriptor(admitted/'artifact-members.json')==admission['artifact_members']
assert supervision['source_commit']==source and supervision['run_id']==run_id and supervision['artifact_id']==admission['artifact_id']
assert supervision['archive']==admission['archive'] and supervision['delivery_admission']==descriptor(admitted/'qualification.json')
assert descriptor(cache/'replay-windows-q8-candidate-supervision-v1.py')==supervision['collector']
assert supervision['all_observed_commands_cleanup_admitted']is True
assert all(type(row['returncode'])is int and row['returncode']==0 and row['error']is None and row['observed_exit_preserved']is True for row in supervision['commands'])
bound_paths={**{name:root/name for name in admission['metadata']},'delivery-admission.json':admitted/'qualification.json','artifact-members.json':admitted/'artifact-members.json','supervision-replay.json':root/'supervision-replay-v1.json','delivery-helper.py':cache/'admit-windows-q8-candidate-delivery-v2.py','supervision-helper.py':cache/'replay-windows-q8-candidate-supervision-v1.py'}
bound={name:descriptor(path)for name,path in bound_paths.items()}
assert all(descriptor(root/name)==value for name,value in admission['metadata'].items())
p=read(delivered/'protocol.json');assert len(p['tools'])==105 and len(p['repository_files'])==818
assert descriptor(delivered/'protocol.json')=={'bytes':190779,'sha256':'8d22329d34a58422ffde3fc2067ea3ebfa2e9928d3845e16fb07f91e58ec18ff'}
env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
expected={**{'tools/nemotron-diarization/'+n:v for n,v in p['tools'].items()},**p['repository_files']}
sys.path.insert(0,str(tools))
# Imports execute only local source whose current bytes and recorded commit are re-admitted.
for name,value in expected.items():assert descriptor(repo/name)==value
from frozen_git_sources import verify_git_sources
verify_git_sources(repo,source,expected)
assert (delivered/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/windows-q8-candidate.protocol.json'],cwd=repo,env=env)
from build_valid_mel_candidate import source_inventory
from ctest_output import require_single_test_pass
from pooled_runtime import BOUNDED_PROFILE,load_profile,profile_provenance,compatibility_patch_bytes
from platform_bounded_metrics import derive_native_probe
from windows_compiler_fp_metrics import baseline
from windows_q8_candidate import ORDER,LIFECYCLE,require_host,require_uninstrumented
from windows_q8_candidate_sources import load_recipe,RECIPE_ID
from windows_q8_candidate_metrics import admit_candidate_audit,compare_candidate_pair,both_pairs_realtime
import windows_q8_candidate_metrics as candidate_metrics
from windows_q8_candidate_controls import LIBRARY_NAMES
from windows_q8_candidate_verify import verify_completed_source
from windows_q8_model_metrics import graph_vectors
from windows_q8_model_sources import PATH,q8_patch,Q8_PATCH
from windows_naming import verify_private_outputs
from windows_session_timing import file_inventory
members=read(admitted/'artifact-members.json');assert file_inventory(delivered)==members
q=read(delivered/'qualification.json');build=q['build'];evidence=delivered/'evidence';sdk=delivered/'sdk'
assert q['source_commit']==source and q['protocol']==descriptor(delivered/'protocol.json')
assert q['order']==p['order']==ORDER and q['limits']==p['limits']
assert q['qualification_completed']is True and q['completed_lifecycle_tests']==p['required_lifecycle_tests']==LIFECYCLE
assert all(q[k]is False for k in('release_qualified','package_qualified'))
assert q['commands']==read(evidence/'commands.json') and len(q['commands'])==supervision['commands_verified']
commands=q['commands'];assert 0<len(commands)<=128
assert commands[0]['name']=='clone-native';ci_native=PureWindowsPath(commands[0]['command'][-1]);ci_root=ci_native.parent;ci_evidence=ci_root/'evidence';ci_build=ci_native/'build/gen2-cpu-diar'
ci_repo=PureWindowsPath(commands[0]['cwd']);assert ci_native.name=='source'
_,lock=load_profile(BOUNDED_PROFILE);recipe_path,recipe=load_recipe()
assert q['build_recipe']==recipe and q['build_recipe_id']==RECIPE_ID
assert descriptor(evidence/'build-recipe.json')==descriptor(recipe_path)
assert read(evidence/'build-recipe.json')==recipe
for k,v in profile_provenance(tools/'runtime-valid-mel-pool-cpu.lock.json',lock).items():assert q[k]==v
assert read(evidence/'runtime-profile.json')==lock
assert q['assets']=={item['name']:{'bytes':item['size'],'sha256':item['sha256']}for item in lock['artifacts']if item['role']!='reference'}
assert q['host']==read(evidence/'receiver-host.stdout.log');require_host(q['host'])
assert build['reported_version']==lock['runtime']['reported_version']
assert (evidence/'version-only.stdout.log').read_text().strip()==build['reported_version']
assert file_inventory(sdk)==build['installed_files']
require_single_test_pass((evidence/'model-free-state-test.stdout.log').read_text(),'diar_state')
assert (evidence/'bounded-inference.patch').read_bytes()==compatibility_patch_bytes(lock)
for item in lock['runtime']['compatibility_patches']:
 assert descriptor(evidence/Path(item['path']).name)=={k:item[k]for k in('bytes','sha256')}
assert (evidence/Q8_PATCH).read_bytes()==q8_patch()
original,derived=tools/'native_stream_probe_valid_mel.cpp',evidence/'native_stream_probe_bounded.cpp'
assert descriptor(original)==build['native_probe_source']['original'] and descriptor(derived)==build['native_probe_source']['derived']
assert derived.read_bytes()==derive_native_probe(original.read_bytes())
proof_root=cache/'windows-q8-candidate-source-admission-v7';proof_path=proof_root/'qualification.json';proof=read(proof_path);proof_identity=descriptor(proof_path)
assert proof['completed']is True and proof['compiled_sdk_or_model']is False and len(proof['commands'])==11
assert all(row['returncode']==0 and row['process_tree_reaped']is True for row in proof['commands'])
assert all(descriptor(Path(n))==v for n,v in proof['inputs'].items())
native=proof_root/'source';materialized=native/'build/gen2-cpu-diar/_deps/llama.cpp'
verify_completed_source(native,native/'build/gen2-cpu-diar',lock,proof_root/'evidence/bounded-inference.patch',True)
assert source_inventory(materialized)==proof['materialized_files'] and len(proof['materialized_files'])==3425
assert proof['diagnostic_worker_telemetry']is False
actual_audit=read(evidence/'platform-llama-materialized-audit.json')
assert actual_audit['files']==proof['materialized_files'] and admit_candidate_audit(actual_audit)==build['exact_source']
assert descriptor(evidence/'platform-pristine-llama.tar')==actual_audit['archive']
assert descriptor(evidence/'compile_commands.json')==actual_audit['compile_database']
database=read(evidence/'compile_commands.json');audit_inputs={PureWindowsPath(n):v for n,v in actual_audit['all_compiler_sources'].items()}
assert {PureWindowsPath(row['file'])for row in database}==set(audit_inputs) and len(database)==97
for name,value in audit_inputs.items():
 assert name.is_relative_to(ci_native) and descriptor(native/name.relative_to(ci_native).as_posix())==value
work=root/'result-replay-v2';work.mkdir();pe=work/'pe';pe.mkdir()
verify_private_outputs(sdk,pe);assert read(pe/'windows-private-pe.json')==read(evidence/'windows-private-pe.json')
admission_record=read(evidence/'compiler-admission.json');assert admission_record==build['compiler_admission']
compiler,linker=admission_record['compiler'],admission_record['linker'];aliases=admission_record['compiler_path_aliases']
assert admission_record['compiler_binary']==recipe['compiler']['binary']==baseline('clang-cl')[2]['compiler_binary']
assert admission_record['linker_binary']==recipe['compiler']['linker_binary']==baseline('clang-cl')[2]['linker_binary']
assert recipe['compiler']['probe_compiler_binary']==baseline('msvc')[2]['compiler_binary']
assert aliases and all(v==admission_record['compiler_binary']for v in aliases.values())
compiler_identities={PureWindowsPath(compiler):admission_record['compiler_binary'],PureWindowsPath(linker):admission_record['linker_binary']}
with patch.object(candidate_metrics,'descriptor',side_effect=lambda path:compiler_identities[PureWindowsPath(str(path))]):
 compiler_check=candidate_metrics.compiler_evidence((evidence/'CMakeCache.txt').read_text(),database,{lang:(evidence/('compiler-'+lang+'.cmake')).read_text()for lang in('C','CXX')},compiler,'clang-cl',aliases)
compiler_check.update(compiler_binary=admission_record['compiler_binary'],compiler_path_aliases=aliases,linker_binary=admission_record['linker_binary'])
assert compiler_check==admission_record
kernel=build['kernel'];units=kernel['translation_units'];assert len(units)==97
kernel_bindings={};objects=set()
for unit,row in zip(units,database):
 assert unit['command']==row and PureWindowsPath(row['directory'])==ci_build
 source_path,obj=PureWindowsPath(row['file']),PureWindowsPath(row['output'])
 if not obj.is_absolute():obj=ci_build/obj
 assert obj.is_relative_to(ci_build) and obj not in objects;objects.add(obj)
 retained='compiler-objects/'+obj.relative_to(ci_build).as_posix()
 assert unit['retained_object']==retained and descriptor(evidence/retained)==unit['object']
 assert unit['source']==audit_inputs[source_path]
 kernel_bindings.update({str(source_path):unit['source'],str(obj):unit['object'],str(ci_evidence/retained):unit['object']})
sg,=[row for row in database if PureWindowsPath(row['file'])==ci_build/'_deps/llama.cpp'/PATH]
assert kernel['compiler_command']==sg
sg_unit,=[unit for unit in units if unit['command']==sg]
assert descriptor(evidence/'kernel.obj')==sg_unit['object']
dump,=[row for row in commands if row['name']=='kernel-disassembly']
assert dump['command'][1:]==['/nologo','/disasm','/symbols',str(ci_evidence/'kernel.obj')]
assert PureWindowsPath(dump['command'][0]).name.lower()=='dumpbin.exe'
recorded_dumpbin_identity=kernel['bindings'][str(PureWindowsPath(dump['command'][0]))]
kernel_bindings[str(PureWindowsPath(dump['command'][0]))]=recorded_dumpbin_identity
for name in('kernel.obj','kernel-disassembly.stdout.log','kernel-disassembly.log'):kernel_bindings[str(ci_evidence/name)]=descriptor(evidence/name)
assert kernel_bindings==kernel['bindings'] and 'llamafile_sgemm'in(evidence/'kernel-disassembly.stdout.log').read_text()
controls=build['model_free_controls'];libraries=controls['retained_import_libraries']
assert controls['graph_input_payloads_fully_retained']is True and q['graph_input_payloads_fully_retained']is True
assert len(libraries)==3 and {PureWindowsPath(n).name for n in libraries}==LIBRARY_NAMES
for name,row in libraries.items():
 path=PureWindowsPath(name);assert path.is_relative_to(ci_build)
 assert row['retained_path']=='graph-inputs/'+path.name
 assert descriptor(evidence/row['retained_path'])=={k:row[k]for k in('bytes','sha256')}
 assert controls['bindings'][name]==descriptor(evidence/row['retained_path'])
control_bindings={}
for name,value in controls['bindings'].items():
 path=PureWindowsPath(name)
 if path.is_relative_to(ci_evidence):actual=evidence/path.relative_to(ci_evidence).as_posix()
 elif path==ci_repo/'tools/nemotron-diarization/ggml_matrix_profile_control.cpp':actual=tools/'ggml_matrix_profile_control.cpp'
 else:
  assert name in libraries;actual=evidence/libraries[name]['retained_path']
 assert descriptor(actual)==value;control_bindings[name]=value
assert len(control_bindings)==50 and controls['vectors']==36 and [c['threads']for c in controls['cases']]==[1,2,4]
graph_reference=None;graph_count=0
for case in controls['cases']:
 threads=case['threads'];phase=f'matrix-control-{threads}'
 paths=[evidence/(phase+f'.{shape}.{repeat}.f32')for shape in range(4)for repeat in range(3)]
 graph_reference=graph_vectors((evidence/(phase+'.stdout.log')).read_text(),threads,[path.read_bytes()for path in paths],graph_reference)
 outputs={str(ci_evidence/path.name):descriptor(path)for path in [*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]}
 assert case['outputs_and_logs']==outputs and all(control_bindings[n]==v for n,v in outputs.items())
 call,=[row for row in commands if row['name']==phase]
 assert call['command']==[str(ci_evidence/'matrix-profile-control.exe'),str(threads),str(ci_evidence/phase)]
 require_uninstrumented((evidence/(phase+'.log')).read_text());graph_count+=12
public_bindings=build['public_executables'];assert len(public_bindings)==4
for name,value in public_bindings.items():
 path=PureWindowsPath(name);actual=evidence/'public-binaries'/path.name
 assert descriptor(actual)==value
 assert path.is_relative_to(ci_evidence/'public-binaries') or path.parts[-3:-1]in(('release','examples'),('release','deps'))
assert len({PureWindowsPath(n).name for n in public_bindings})==2
retained_bindings={str(ci_evidence/n):v for n,v in file_inventory(evidence).items()if n!='commands.json'}
assert retained_bindings==q['retained_output_bindings']
execution_admission=admit_execution_commands(commands,ci_root,ci_repo,public_bindings,ORDER,LIFECYCLE)
cases=q['cases'];assert len(cases)==4 and [c['kind']for c in cases]==ORDER
comparisons=[];case_summary=[]
for index,case in enumerate(cases):
 kind=ORDER[index];phase=f'case-{index}-{kind}';target=evidence/(phase+'.json');raw=Path(str(target)+'.f32')
 assert type(case['index'])is int and case['index']==index and case['command_started']is True
 assert case['result_path']==target.name and case['probability_path']==raw.name
 assert case['result']==read(target) and case['result_file']==descriptor(target) and case['probabilities']==descriptor(raw)
 assert descriptor(raw)['bytes']==192000
 call,=[row for row in commands if row['name']==phase]
 assert call['command'][-1]==str(ci_evidence/(target.name if kind=='native'else raw.name))
 if kind=='public':assert read(evidence/(phase+'.stdout.log'))==case['result']
 require_uninstrumented((evidence/(phase+'.log')).read_text())
 case_summary.append({'index':index,'kind':kind,'elapsed_micros':case['result']['elapsed_micros'],'rtf':case['result']['elapsed_micros']/60e6,'turns':len(case['result']['turns']),'probabilities':descriptor(raw)})
 if index in(1,3):
  first,second=cases[index-1:index+1];n,u=(first,second)if first['kind']=='native'else(second,first)
  comparisons.append(compare_candidate_pair(u['result'],n['result'],(evidence/n['probability_path']).read_bytes(),(evidence/u['probability_path']).read_bytes()))
assert comparisons==q['comparisons']
realtime=both_pairs_realtime(comparisons);assert q['realtime_gate_passed']is realtime
reference=read(evidence/'reference/ami.json')
assert reference=={'segments':[{'start':t['start_seconds'],'end':t['end_seconds'],'speaker':t['speaker']}for t in cases[0]['result']['turns']]}
assert (evidence/'reference/ami.json.f32').read_bytes()==(evidence/cases[0]['probability_path']).read_bytes()
listed={line.removesuffix(': test')for line in(evidence/'native-lifecycle-list.stdout.log').read_text().splitlines()if line.endswith(': test')}
assert set(LIFECYCLE)<=listed
for index,test in enumerate(LIFECYCLE):
 phase='native-lifecycle-'+str(index);stdout=(evidence/(phase+'.stdout.log')).read_text()
 call,=[row for row in commands if row['name']==phase]
 assert call['command'][-3:]==['--exact',test,'--test-threads=1']
 assert '--ignored'in call['command']
 assert re.search(r'^test '+re.escape(test)+r' \.\.\. ok$',stdout,re.MULTILINE)and'1 passed; 0 failed; 0 ignored;'in stdout
 require_uninstrumented((evidence/(phase+'.log')).read_text())
assert q['functional_lifecycle_gates_passed']is True
assert q['development_native_gates_passed']is realtime
run=read(root/'producer-run.json');assert(run['conclusion']=='success')is realtime
assert ('error'not in q)is realtime
assert file_inventory(delivered)==members and descriptor(root/'artifact.zip')==admission['archive']
assert descriptor(proof_path)==proof_identity and all(descriptor(Path(n))==v for n,v in proof['inputs'].items())
assert all(descriptor(path)==bound[name]for name,path in bound_paths.items()) and descriptor(Path(__file__))==identity
result={'schema_version':1,'collection_completed':True,'source_commit':source,'run_id':run_id,'job_id':job_id,'artifact_id':admission['artifact_id'],'archive':admission['archive'],'protocol':descriptor(delivered/'protocol.json'),'collector':identity,'delivery_admission':descriptor(admitted/'qualification.json'),'supervision_replay':descriptor(root/'supervision-replay-v1.json'),'receiver_inputs':bound,'independent_source_only_proof':{'path':str(proof_path),**proof_identity},'tools_verified':105,'repository_files_verified':818,'artifact_members_verified':len(members),'supervised_zero_exit_and_reaped_commands':len(commands),'host':q['host'],'sdk_files':len(build['installed_files']),'ggml_files':3425,'compiler_inputs':97,'retained_compiler_objects_verified':97,'graph_vectors_verified':graph_count,'retained_graph_import_libraries_verified':3,'graph_input_payloads_fully_rehashed':True,'all_preregistered_evidence_retained':True,'full_sequence_bit_exact':True,'cases':case_summary,'comparisons':comparisons,'completed_lifecycle_tests':LIFECYCLE,'execution_command_admission':execution_admission,'functional_lifecycle_gates_passed':True,'realtime_gate_passed':realtime,'production_candidate_sdk_gates_passed':realtime,'release_qualified':False,'package_qualified':False,'new_compiler_sdk_model_calls_during_collection':0,'limits':['Source SDK and same-host public/lifecycle qualification only; package/offline/coexistence/desktop, accuracy and deferred human word-time/release gates remain.','Compiler/linker/MSVC aliases and execution are CI observations; receiver rehashes retained source/object/library/SDK/public binaries without executing Windows tools.','Independent source-only reconstruction admits bounded inference, all3425 materialized files and97 actual compiler sources.','Earlier failed5percent diagnostic prediction remains failed; allfour original strict RTFs are evaluated independently.']}
with(root/'result-replay-v2.json').open('x')as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'completed':True,'commands':len(commands),'graphs':graph_count,'objects':97,'graph_libraries':3,'cases':case_summary,'realtime_passed':realtime,'lifecycle_tests':len(LIFECYCLE)}))
