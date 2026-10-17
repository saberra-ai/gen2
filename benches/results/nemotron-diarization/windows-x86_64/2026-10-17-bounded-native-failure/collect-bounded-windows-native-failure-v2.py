"""Replay saved Windows evidence only; no SDK/helper/model/build/workflow execution."""
import hashlib,json,os,shutil,subprocess,sys,tarfile,zipfile
from pathlib import Path,PureWindowsPath
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');root=cache/'bounded-windows-native-ci-v1';evidence=root/'unpacked/nemo-bounded-evidence';sdk=root/'unpacked/nemo-bounded-sdk'
# All replay source admission below is stdlib-only and precedes local imports.
def pre_descriptor(path):
    size=0;digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):size+=len(block);digest.update(block)
    return {'bytes':size,'sha256':digest.hexdigest()}
pre_env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
producer_source='dfdede8e74709a0c89be01d53aad92d346b50b99'
collector_source='6cc8c4b9737aed4ffc941b470e3e67e117f3e10d'
pre_protocol_bytes=(evidence/'protocol.json').read_bytes()
assert pre_protocol_bytes==subprocess.check_output(['git','show',producer_source+':tools/nemotron-diarization/bounded-windows-native.protocol.json'],cwd=repo,env=pre_env)
pre_protocol=json.loads(pre_protocol_bytes)
for name,want in pre_protocol['tools'].items():assert pre_descriptor(tools/name)==want,name
collector_extra_sources={}
for name in ['bounded_platform_sdk_receipt.py','windows_archive.py']:
    raw=subprocess.check_output(['git','show',collector_source+':tools/nemotron-diarization/'+name],cwd=repo,env=pre_env)
    want={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    assert pre_descriptor(tools/name)==want,name
    collector_extra_sources[name]=want
assert pre_descriptor(root/'artifact.zip')=={'bytes':15481445,'sha256':'49ae76bfa000287e6e0831618fdd0aed776bbb6b6134e0a2ea55f194880aad16'}
# Bind every unpacked input to the actual verified ZIP, not a mutable prior summary.
artifact_members={}
with zipfile.ZipFile(root/'artifact.zip') as archive:
    assert len(archive.infolist())==191
    for info in archive.infolist():
        assert not info.is_dir() and not info.filename.startswith('/') and '..' not in Path(info.filename).parts
        size=0;digest=hashlib.sha256()
        with archive.open(info) as stream:
            while block:=stream.read(1024**2):size+=len(block);digest.update(block)
        assert size==info.file_size
        want={'bytes':size,'sha256':digest.hexdigest()}
        assert pre_descriptor(root/'unpacked'/info.filename)==want,info.filename
        artifact_members[info.filename]=want
work=root/'collection-v2';work.mkdir()
sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor,source_inventory
from frozen_git_sources import verify_git_sources
from platform_bounded_metrics import compare_complete_60s,derive_native_probe
from pooled_runtime import BOUNDED_PROFILE,load_profile,profile_provenance,compatibility_patch_bytes
from windows_archive import byte_inventory
from windows_naming import verify_private_outputs
from bounded_platform_sdk_receipt import validate_native_receipt
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
source='dfdede8e74709a0c89be01d53aad92d346b50b99';protocol=read(evidence/'protocol.json');q=read(evidence/'receipt.json');run=read(root/'producer-run.json')
assert run['id']==39022551571 and run['head_sha']==source and run['conclusion']=='failure' and run['repository']['full_name']=='saberra-ai/gen2'
artifact_meta=read(cache/'bounded-windows-native-artifacts-v1.json')['artifacts'];artifact,=[v for v in artifact_meta if v['id']==12097637508]
assert descriptor(root/'artifact.zip')=={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'].removeprefix('sha256:')}
assert (evidence/'protocol.json').read_bytes()==(tools/'bounded-windows-native.protocol.json').read_bytes()
assert q['source_commit']==source and q['protocol_sha256']==descriptor(evidence/'protocol.json')['sha256']
pins={**{'tools/nemotron-diarization/'+n:r for n,r in protocol['tools'].items()},**protocol['repository_files'],'tools/nemotron-diarization/bounded-windows-native.protocol.json':descriptor(evidence/'protocol.json')}
verify_git_sources(repo,source,pins)
for key,base in [('tools',tools),('repository_files',repo)]:
 for name,want in protocol[key].items():assert descriptor(base/name)==want,name
lockpath,lock=load_profile(BOUNDED_PROFILE)
for n,v in profile_provenance(lockpath,lock).items():assert q[n]==v,n
assert q['actual_reported_version']==lock['runtime']['reported_version']
assert q['development_native_gates_passed'] is False and q['functional_lifecycle_gates_passed'] is True and q['realtime_gate_passed'] is False
assert q['error']=='bounded Windows native and public RTF must each be below one'
assert q['completed_lifecycle_tests']==protocol['required_lifecycle_tests'] and len(q['commands'])==47 and all(type(c['returncode'])is int and c['returncode']==0 for c in q['commands'])
for i,name in enumerate(protocol['required_lifecycle_tests']):
 text=(evidence/f'native-lifecycle-{i}.stdout.log').read_text()
 assert 'test '+name+' ... ok' in text and '1 passed; 0 failed; 0 ignored;' in text
assert (evidence/'bounded-inference.patch').read_bytes()==compatibility_patch_bytes(lock)
for patch in lock['runtime']['compatibility_patches']:
 assert descriptor(evidence/Path(patch['path']).name)=={k:patch[k] for k in ['bytes','sha256']}
original=tools/'native_stream_probe_valid_mel.cpp';derived=evidence/'native_stream_probe_bounded.cpp'
assert descriptor(original)==q['native_probe_source']['original'] and descriptor(derived)==q['native_probe_source']['derived']
assert derived.read_bytes()==derive_native_probe(original.read_bytes())
native,public=read(evidence/'native-ami.json'),read(evidence/'public.json')
comparison=compare_complete_60s(public,native,(evidence/'native-ami.json.f32').read_bytes(),(evidence/'public.f32').read_bytes())
assert comparison==read(evidence/'comparison.json') and comparison['realtime_gate_passed'] is False
assert all(v['input_pcm_f32le_sha256']==protocol['cases']['actual_pcm_sha256'] and v['input_repeats']==1 for v in [native,public])
sdk_inventory=byte_inventory(sdk)
assert {n:r for n,r in sdk_inventory.items() if r['type']=='file'}=={n.replace('\\','/'):{'type':'file',**r} for n,r in q['installed_files'].items()}
peout=work/'pe-replay';peout.mkdir();verify_private_outputs(sdk,peout)
assert read(peout/'windows-private-pe.json')==read(evidence/'windows-private-pe.json')
audit=read(evidence/'platform-llama-materialized-audit.json')
assert descriptor(evidence/'platform-llama-materialized-audit.json')==q['compiler_audit'] and audit['exact_independent_replay_match'] is True
assert descriptor(evidence/'platform-pristine-llama.tar')==audit['archive'] and audit['base_revision']==lock['runtime']['llama_revision']
pristine=cache/'bounded-windows-source-admission-v1/source'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=pristine,text=True).strip()==lock['runtime']['revision']
replay=work/'ggml-replay';replay.mkdir();commands=[]
with tarfile.open(evidence/'platform-pristine-llama.tar') as archive:archive.extractall(replay,filter='data')
env={'PATH':'/usr/bin:/bin','LC_ALL':'C','GIT_CEILING_DIRECTORIES':str(replay.parent),'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1','GIT_CONFIG_COUNT':'2','GIT_CONFIG_KEY_0':'core.autocrlf','GIT_CONFIG_VALUE_0':'false','GIT_CONFIG_KEY_1':'core.eol','GIT_CONFIG_VALUE_1':'lf'}
for i,item in enumerate(audit['patch_series']):
 path=pristine/'patches'/item['name'];assert descriptor(path)=={k:item[k] for k in ['bytes','sha256']}
 patch=work/('ggml-replay-'+str(i)+'.patch');patch.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
 row={'argv':['/usr/bin/git','apply','--whitespace=nowarn',str(patch)],'cwd':str(replay)};commands.append(row)
 with (work/('ggml-replay-'+str(i)+'.stdout.log')).open('xb') as stdout,(work/('ggml-replay-'+str(i)+'.stderr.log')).open('xb') as stderr:r=subprocess.run(row['argv'],cwd=replay,env=env,stdout=stdout,stderr=stderr)
 row['returncode']=r.returncode;r.check_returncode()
assert len(commands)==19 and source_inventory(replay)==audit['files']
assert descriptor(evidence/'compile_commands.json')==audit['compile_database']
materialized=PureWindowsPath(audit['compiled_tree_path']);native_root=materialized.parents[3]
entries=read(evidence/'compile_commands.json')
assert {PureWindowsPath(r['file']) for r in entries}=={PureWindowsPath(n) for n in audit['all_compiler_sources']}
for name,want in audit['actual_compiler_sources'].items():assert audit['files'][name]==want
for name,want in audit['all_compiler_sources'].items():
 path=PureWindowsPath(name)
 if path.is_relative_to(materialized):assert audit['files'][path.relative_to(materialized).as_posix()]==want
 else:
  assert path.is_relative_to(native_root)
  assert descriptor(pristine/path.relative_to(native_root).as_posix())==want
try:validate_native_receipt(q)
except ValueError as error:assert str(error)=='completed bounded Windows SDK qualification required'
else:raise AssertionError('failed producer was admitted as qualified SDK')
dest=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-bounded-native-failure';dest.mkdir()
for path in sorted(evidence.iterdir()):
 if path.is_file() and path.suffix.lower() in {'.json','.log','.patch','.cpp','.f32'}:shutil.copy2(path,dest/path.name)
for path in [root/'producer-run.json',cache/'bounded-windows-native-artifacts-v1.json',cache/'bounded-windows-native-run-v1-failed.log',cache/'extract-bounded-windows-native-v1.py',Path(__file__)]:shutil.copy2(path,dest/path.name)
for name in protocol['tools']:
 target=dest/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,target)
for name in collector_extra_sources:
 target=dest/'collector-sources'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(tools/name,target)
(dest/'artifact-members.json').write_text(json.dumps(artifact_members,indent=2)+'\n')
summary={'collector_extra_source_commit':collector_source,'collector_extra_sources_admitted_before_import':collector_extra_sources,'all_producer_tool_sources_admitted_before_import':True,'all_unpacked_inputs_rebound_to_verified_zip':True,'raw_probability_files':{name:descriptor(evidence/name) for name in ['native-ami.json.f32','public.f32']},'schema_version':1,'source_commit':source,'run_id':39022551571,'artifact_id':12097637508,'artifact':descriptor(root/'artifact.zip'),'actual_windows_functional_lifecycle_passed':True,'performance_gate_passed':False,'development_native_gates_passed':False,'package_admission_rejected_failed_producer':True,'release_qualified':False,'comparison':comparison,'commands':47,'all_commands_zero':True,'lifecycle_tests':q['completed_lifecycle_tests'],'producer_tools_verified':len(protocol['tools']),'producer_repository_files_verified':len(protocol['repository_files']),'sdk_files_verified':len(q['installed_files']),'sdk_inventory':sdk_inventory,'ggml_replay_patches':19,'materialized_files_verified':len(audit['files']),'actual_compiler_sources_verified':len(audit['all_compiler_sources']),'cpu':[{k:row[k] for k in ['Name','Manufacturer','Architecture','NumberOfCores','NumberOfLogicalProcessors']} for row in read(evidence/'windows-provenance.json')['cpu']],'collection_git_apply_commands':commands,'new_native_or_neural_calls_during_collection':0,'limits':['Native/public complete history, exact tag, source/ISA/PE/MSVC controls and seven lifecycle cases pass; both observed RTFs fail the fixed below-one gate. No timing tolerance, averaging, retry or qualification promotion.','CIM reports2cores/4logicalprocessors; effective GGML worker count and scheduling cause were not instrumented. A smaller worker budget is only a candidate to measure, not an established fix.','Downloaded ZIP, installed binaries and57MB pristine archive remain local. Git retains scoped evidence, hashes and raw probabilities only; no delivered code execution during collection.','No actual bounded package/negative-load Windows observer/receiver or fresh desktop qualification, quality/human timing or public release/main closure.']}
(dest/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files={str(p.relative_to(dest)):descriptor(p) for p in sorted(dest.rglob('*')) if p.is_file()}
(dest/'collection.json').write_text(json.dumps({'schema_version':1,'files':files},indent=2)+'\n')
print(dest,len(files)+1,'members')
