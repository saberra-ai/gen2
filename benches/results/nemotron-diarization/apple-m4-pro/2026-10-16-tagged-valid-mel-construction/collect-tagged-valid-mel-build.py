"""Collect failed tagged build and separately successful metadata recovery; no execution."""
import hashlib,json,pathlib,shutil,subprocess,sys
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');failed=cache/'valid-mel-tagged-build-v1';recovery=cache/'valid-mel-tagged-metadata-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-tagged-valid-mel-construction'
def read(p):return json.loads(p.read_text())
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def git_blob(commit,name):return subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/'+name],cwd=repo)
p=read(recovery/'protocol.json');r=read(recovery/'recovered-build.json');old=read(failed/'protocol.json');outer=read(pathlib.Path(str(recovery)+'.command.json'));source=outer['source_commit'];construction=p['construction_commit'];assert source=='0cafd2944cebcea5340a68f8968f06c22d5f2814'and construction=='d5a05c43efa62336610c0ec654928a13b5841abb'
assert git_blob(source,'valid-mel-tagged-metadata.protocol.json')==(recovery/'protocol.json').read_bytes()
assert git_blob(construction,'valid-mel-tagged-build.protocol.json')==(failed/'protocol.json').read_bytes()
for protocol,commit in [(p,source),(old,construction)]:
 for name,expected in protocol['tools'].items():
  blob=git_blob(commit,name);assert {'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}==expected
for name,expected in p['inputs'].items():assert rec(pathlib.Path(name))==expected
assert rec(cache/'run-tagged-metadata-recovery.py')==p['external_launcher']
assert rec(cache/'run-tagged-valid-mel-build.py')==old['external_launcher']
assert read(failed/'prepare.command.json')['returncode']==0 and read(failed/'build.command.json')['returncode']==1
assert outer['returncode']==0 and outer['argv']==[str(cache/'evidence-tools-env/bin/python'),str(tools/'recover_valid_mel_build_metadata.py')]
assert 'Error: $HOME must be set to run brew.'in(failed/'build.stderr.log').read_text()
assert [c['phase']for c in r['commands']]==['configure','llama-archive']+['llama-patch-'+str(i)for i in range(19)]+['build','install','model-free-state-test','version-only']
assert all(c['returncode']==0 for c in r['commands'])and r['commands']==read(failed/'build/commands.json')
assert r['metadata_commands']==read(recovery/'commands.json')and [c['name']for c in r['metadata_commands']]==['cmake','ninja','clang++','xcode-sdk','brew-packages']and all(c['returncode']==0 for c in r['metadata_commands'])
assert r['metadata_commands'][-1]['environment']['HOME']==p['actual_user_home']==str(pathlib.Path.home())and r['metadata_commands'][-1]['environment']['HOMEBREW_NO_AUTO_UPDATE']=='1'
assert r['toolchain']['packages']==(recovery/'brew-packages.stdout.log').read_text().strip()
assert set(x.split()[0]for x in r['toolchain']['packages'].splitlines())=={'sentencepiece','abseil','protobuf'}
for base in [failed,recovery]:
 expected='(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(x)+')'for x in p['denied_paths'])+')\n';assert (base/'offline.sb').read_text()==expected
assert (failed/'build/offline.sb').read_bytes()==(failed/'offline.sb').read_bytes()
for command in r['commands']:assert command['argv'][:3]==['/usr/bin/sandbox-exec','-f',str(failed/'build/offline.sb')]
for command in r['metadata_commands']:assert command['argv'][:3]==['/usr/bin/sandbox-exec','-f',str(recovery/'offline.sb')]
assert '100% tests passed, 0 tests failed out of 1'in(failed/'build/model-free-state-test.stdout.log').read_text()
assert r['runtime_reported_version']==(failed/'build/version-only.stdout.log').read_text().strip()=='nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
sys.path.insert(0,str(tools));from build_valid_mel_candidate import source_inventory
native=pathlib.Path(p['native_source_path']);sdk=pathlib.Path(p['sdk_path']);a=read(failed/'build/llama-materialized-audit.json');assert a['exact_independent_replay_match']is True and source_inventory(pathlib.Path(a['compiled_tree_path']))==a['files']and len(a['files'])==3425
assert r['installed_library_paths']==p['sdk_library_paths']and len(r['installed_library_paths'])==15 and len(r['libraries'])==6
for name,expected in r['installed_library_paths'].items():
 path=sdk/'lib'/name;assert rec(path)=={k:expected[k]for k in ['bytes','sha256']}and(str(path.readlink())if path.is_symlink()else None)==expected['symlink_target']
for name,value in r['libraries'].items():
 path=sdk/'lib'/name;assert value['dependencies']==subprocess.check_output(['/usr/bin/otool','-L',str(path)],text=True)and value['exports']==subprocess.check_output(['/usr/bin/nm','-gU',str(path)],text=True)
archive=r['runtime_archive'];assert rec(pathlib.Path(archive['path']))=={k:archive[k]for k in ['bytes','sha256']}
for name in ['CMakeCache.txt','compile_commands.json']:assert (recovery/name).read_bytes()==(native/'build/cpu-diar'/name).read_bytes()
assert r['metadata_recovery_only']is True and r['construction_outer_returncode']==1 and r['new_compilation_or_tests_or_version_query']is False
for flag in ['model_instantiated_or_inference_performed','artifact_admitted_for_inference','production_qualified','release_qualified']:assert r[flag]is False
assert not out.exists();out.mkdir()
shutil.copytree(failed,out/'failed-construction',ignore=shutil.ignore_patterns('pristine-llama.tar'))
shutil.copytree(recovery,out/'metadata-recovery')
for suffix in ['command.json','stdout.log','stderr.log']:shutil.copy2(pathlib.Path(str(recovery)+'.'+suffix),out/('metadata-outer.'+suffix))
for name in ['run-tagged-valid-mel-build.py','run-tagged-metadata-recovery.py','collect-tagged-valid-mel-build.py']:shutil.copy2(cache/name,out/name)
(out/'construction-tools').mkdir()
for name in old['tools']:
 target=out/'construction-tools'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(git_blob(construction,name))
q={'schema_version':1,'construction_commit':construction,'metadata_recovery_commit':source,'original_outer_returncode':1,'original_failure':'late Homebrew metadata capture required HOME; all25 construction/state/version phases succeeded','metadata_recovery_returncode':0,'tagged_cpu_sdk_construction_metadata_complete':True,'source_preparation_completed':True,'all_25_construction_phase_commands_succeeded':True,'model_free_state_test_completed':True,'actual_saved_runtime_version':r['runtime_reported_version'],'sdk_library_paths':15,'sdk_real_libraries':6,'materialized_compiler_inputs':3425,'runtime_static_archive_bound':True,'new_metadata_queries':5,'new_compile_test_version_or_neural_calls_in_recovery':0,'original_attempt_and_sdk_unchanged':True,'model_instantiated_or_inference_performed':False,'artifact_admitted_for_inference':False,'production_qualified':False,'release_qualified':False,'limits':p['limitations']+['The builder source atd5failed after successfulnative construction. The distinct0cafdmetadata-onlyprotocol completed its provenance without rerunning compilation, state tests, CAPIversion or neural work. Historicalfailure remains visible.','This qualifies construction metadata only. TaggedSDK Q8/F32/publicRust, otherbackend/platforms, package/offline/coexistence/lifecycle/longsession/quality/desktop qualification remain open.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');files={x.relative_to(out).as_posix():rec(x)for x in out.rglob('*')if x.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'sdk_paths':15,'compiler_inputs':3425,'metadata_queries':5,'new_neural_calls':0}))
