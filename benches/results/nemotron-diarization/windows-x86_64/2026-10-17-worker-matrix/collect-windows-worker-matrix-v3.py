"""Replay completed saved diagnostic evidence; never execute delivered code/models."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
root=cache/'windows-worker-diagnostic-ci-v2'
source='1745243b221368855381ea36879ccdc1bb731528'
run_id=39025306014

def descriptor(path):
    size=0
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):
            size+=len(block)
            value.update(block)
    return {'bytes':size,'sha256':value.hexdigest()}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def inventory(path):
    result={}
    for item in sorted(path.rglob('*')):
        assert not item.is_symlink()
        if item.is_file():
            result[item.relative_to(path).as_posix()]=descriptor(item)
    return result

env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}

def git_bytes(name):
    return subprocess.check_output(['git','show',source+':'+name],cwd=repo,env=env)

metadata_descriptors={name:descriptor(root/name) for name in ['producer-run.json','artifacts.json']}
collector_descriptor=descriptor(Path(__file__))
run=read(root/'producer-run.json')
assert run['id']==run_id and run['head_sha']==source and run['conclusion']=='success'
assert run['repository']['full_name']=='saberra-ai/gen2'
artifact,=read(root/'artifacts.json')['artifacts']
assert artifact['workflow_run']['id']==run_id and artifact['workflow_run']['head_sha']==source
assert descriptor(root/'artifact.zip')=={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'].removeprefix('sha256:')}
members={}
with zipfile.ZipFile(root/'artifact.zip') as archive:
    for info in archive.infolist():
        path=Path(info.filename)
        assert not info.is_dir() and not path.is_absolute() and '..' not in path.parts
        assert info.filename not in members
        assert (info.external_attr >> 16) & 0o170000 != 0o120000
        size=0
        value=hashlib.sha256()
        with archive.open(info) as stream:
            while block:=stream.read(1024**2):
                size+=len(block)
                value.update(block)
        want={'bytes':size,'sha256':value.hexdigest()}
        assert size==info.file_size and descriptor(root/'unpacked'/path)==want
        members[info.filename]=want
assert inventory(root/'unpacked')==members
p=read(root/'unpacked/protocol.json')
assert (root/'unpacked/protocol.json').read_bytes()==git_bytes('tools/nemotron-diarization/windows-worker-diagnostic-v2.protocol.json')
# Admit the entire evaluator import closure with stdlib before local imports.
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
    for name,want in p[group].items():
        raw=git_bytes(prefix+name)
        assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==want
        assert descriptor(repo/(prefix+name))==want
sys.path.insert(0,str(tools))
from build_valid_mel_candidate import source_inventory
from ctest_output import require_single_test_pass
from pooled_runtime import BOUNDED_PROFILE,load_profile,compatibility_patch_bytes
from windows_naming import verify_private_outputs
from windows_worker_metrics import ORDER,metadata,version,derive_probe,telemetry,compare_native
from windows_worker_sources_v2 import verify_completed_source

q=read(root/'unpacked/qualification.json')
assert q['source_commit']==source and q['protocol']==descriptor(root/'unpacked/protocol.json')
assert q['diagnostic_completed'] is True and q['order']==ORDER==p['order']
assert all(q[key] is False for key in ['release_qualified','package_qualified','development_native_gates_passed'])
assert q['source_checkout_child_environment']=={'GIT_LFS_SKIP_SMUDGE':'1'}
assert len(q['cases'])==4 and set(q['builds'])==set(q['commands'])=={'1','2','4'}
_,lock=load_profile(BOUNDED_PROFILE)
assert q['assets']=={item['name']:{'bytes':item['size'],'sha256':item['sha256']} for item in lock['artifacts'] if item['role']!='reference'}
work=root/'collection-replay-v2'
work.mkdir()
replay=work/'ggml'
replay.mkdir()
commands=[]
# Construct the independently expected pristine archive from the pinned local
# Git object, rather than accepting only a self-reported delivered archive hash.
pristine=cache/'NeMo-Speech.cpp'
expected_archive=work/'independent-pristine-llama.tar'
subprocess.run(['git','archive','--format=tar','-o',str(expected_archive),lock['runtime']['llama_revision'],
    '--','.',':(exclude)models',':(exclude)docs',':(exclude)media'],
    cwd=pristine/'llama.cpp',env=env,check=True)
archive_identity=descriptor(expected_archive)
with tarfile.open(expected_archive) as bundle:
    bundle.extractall(replay,filter='data')
first_audit=read(root/'unpacked/evidence-4/platform-llama-materialized-audit.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=pristine,env=env,text=True).strip()==lock['runtime']['revision']
assert (pristine/'patches/series').read_bytes()==subprocess.check_output(['git','show',lock['runtime']['revision']+':patches/series'],cwd=pristine,env=env)
canonical_series=[line.split('#',1)[0].strip() for line in (pristine/'patches/series').read_text().splitlines()]
canonical_series=[name for name in canonical_series if name]
assert len(canonical_series)==19 and canonical_series==[item['name'] for item in first_audit['patch_series']]
for index,item in enumerate(first_audit['patch_series']):
    patch_source=pristine/'patches'/item['name']
    assert descriptor(patch_source)=={key:item[key] for key in ['bytes','sha256']}
    assert patch_source.read_bytes()==subprocess.check_output(['git','show',lock['runtime']['revision']+':patches/'+item['name']],cwd=pristine,env=env)
    patch=work/('patch-'+str(index)+'.diff')
    patch.write_bytes(patch_source.read_bytes().replace(b'\r\n',b'\n'))
    argv=['/usr/bin/git','apply','--whitespace=nowarn',str(patch)]
    row={'argv':argv,'cwd':str(replay)}
    commands.append(row)
    with (work/('patch-'+str(index)+'.stdout.log')).open('xb') as out,(work/('patch-'+str(index)+'.stderr.log')).open('xb') as err:
        result=subprocess.run(argv,cwd=replay,env={**env,'GIT_CEILING_DIRECTORIES':str(replay.parent)},stdout=out,stderr=err)
    row['returncode']=result.returncode
    result.check_returncode()
assert len(commands)==19
replayed=source_inventory(replay)
common={'clone-native','checkout-native','clone-llama','checkout-llama','cmake-version','git-line-ending-config',
        'sdk-build','sdk-install','compiler-llama-archive','model-free-state-test','native-probe-build',
        'windows-provenance','loader-dependencies','version-only'}
common|={name+str(index) for index in range(3) for name in ['inference-patch-check-','inference-patch-']}
common|={'compiler-llama-patch-'+str(index) for index in range(19)}
assert len(common)==39
build_summaries={}
for cap in (4,2,1):
    evidence=root/('unpacked/evidence-'+str(cap))
    sdk=root/('unpacked/sdk-'+str(cap))
    build=q['builds'][str(cap)]
    assert build['reported_version']==version(cap)
    assert inventory(sdk)=={name.replace('\\','/'):value for name,value in build['installed_files'].items()}
    actual_commands=q['commands'][str(cap)]
    assert actual_commands==build['commands']==read(evidence/'commands.json')
    assert all(type(c['returncode']) is int and c['returncode']==0 for c in actual_commands)
    expected=common|{'case-'+str(index)+'-cap-'+str(cap) for index,value in enumerate(ORDER) if value==cap}
    names=[row['name'] for row in actual_commands]
    assert len(names)==len(set(names)) and set(names)==expected
    assert read(evidence/'upstream-patch-diagnostic.json') is not None
    assert (evidence/'bounded-inference.patch').read_bytes()==compatibility_patch_bytes(lock)
    for patch in lock['runtime']['compatibility_patches']:
        assert descriptor(evidence/Path(patch['path']).name)=={key:patch[key] for key in ['bytes','sha256']}
    require_single_test_pass((evidence/'model-free-state-test.stdout.log').read_text(),'diar_state')
    assert (evidence/'version-only.stdout.log').read_text().strip()==version(cap)
    original=tools/'native_stream_probe_valid_mel.cpp'
    derived=evidence/'native_stream_probe_bounded.cpp'
    assert descriptor(original)==build['native_probe_source']['original']
    assert descriptor(derived)==build['native_probe_source']['derived']
    assert derived.read_bytes()==derive_probe(original.read_bytes(),cap)
    pe=work/('pe-'+str(cap))
    pe.mkdir()
    verify_private_outputs(sdk,pe)
    assert read(pe/'windows-private-pe.json')==read(evidence/'windows-private-pe.json')
    audit=read(evidence/'platform-llama-materialized-audit.json')
    assert descriptor(evidence/'platform-llama-materialized-audit.json')==build['compiler_audit']
    assert audit['exact_independent_replay_match'] is True
    assert audit['base_revision']==lock['runtime']['llama_revision']
    assert descriptor(evidence/'platform-pristine-llama.tar')==audit['archive']==archive_identity
    assert audit['files']==replayed and audit['patch_series']==first_audit['patch_series']
    assert descriptor(evidence/'compile_commands.json')==audit['compile_database']
    local=cache/'windows-worker-source-admission-v1'/('source-'+str(cap))
    verify_completed_source(local,local/'build/gen2-cpu-diar',lock,
        cache/'windows-worker-source-admission-v1'/('evidence-'+str(cap))/'bounded-inference.patch',True,
        p['variants'][str(cap)]['source_sha256'])
    for name,want in p['variants'][str(cap)]['source_sha256'].items():
        assert descriptor(local/name)['sha256']==want
    materialized=PureWindowsPath(audit['compiled_tree_path'])
    native_root=materialized.parents[3]
    database=read(evidence/'compile_commands.json')
    assert {PureWindowsPath(row['file']) for row in database}=={PureWindowsPath(n) for n in audit['all_compiler_sources']}
    for name,want in audit['actual_compiler_sources'].items():
        assert replayed[name]==want
    for name,want in audit['all_compiler_sources'].items():
        path=PureWindowsPath(name)
        if path.is_relative_to(materialized):
            assert replayed[path.relative_to(materialized).as_posix()]==want
        else:
            assert path.is_relative_to(native_root)
            assert descriptor(local/path.relative_to(native_root).as_posix())==want
    cache_values=dict(line.split('=',1) for line in (evidence/'CMakeCache.txt').read_text().splitlines()
                      if '=' in line and not line.startswith(('#','//')))
    assert cache_values['NEMO_SPEECH_VERSION_METADATA:STRING']==metadata(cap)
    for key,value in {'GGML_NATIVE':'OFF','GGML_AVX':'ON','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON',
                      'GGML_AVX512':'OFF','GGML_AVX_VNNI':'OFF','GGML_OPENMP':'OFF',
                      'GGML_BACKEND_DL':'OFF','GGML_METAL':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF',
                      'NEMO_SPEECH_BUILD_ASR':'OFF','NEMO_SPEECH_BUILD_TTS':'OFF'}.items():
        cache_type='UNINITIALIZED' if key in ('GGML_FMA','GGML_F16C') else 'BOOL'
        assert cache_values[key+':'+cache_type]==value,(key,value)
    cpu_entries=[row for row in database if 'ggml-cpu' in PureWindowsPath(row['file']).parts]
    assert cpu_entries
    for row in cpu_entries:
        assert {'-DGGML_AVX2','-DGGML_FMA','-DGGML_F16C','/arch:AVX2'} <= set(row['command'].split())
    build_summaries[str(cap)]={'sdk_files':len(build['installed_files']),'commands':len(actual_commands),
        'materialized_ggml_files':len(replayed),'actual_compiler_sources':len(audit['all_compiler_sources']),
        'cpu':read(evidence/'windows-provenance.json')['cpu']}
baseline=baseline_raw=None
cases=[]
for index,cap in enumerate(ORDER):
    evidence=root/('unpacked/evidence-'+str(cap))
    phase='case-'+str(index)+'-cap-'+str(cap)
    saved=q['cases'][index]
    target=evidence/(phase+'.json')
    result=read(target)
    raw=Path(str(target)+'.f32').read_bytes()
    assert saved['index']==index and saved['cap']==cap and saved['command_started'] is True
    assert result==saved['result'] and descriptor(target)==saved['result_file']
    assert descriptor(Path(str(target)+'.f32'))==saved['probabilities']
    workers=telemetry((evidence/(phase+'.log')).read_text(),cap)
    assert workers==saved['workers'] and workers['effective_threads']==cap
    if baseline is None:
        baseline,baseline_raw=result,raw
    comparison=compare_native(baseline,result,baseline_raw,raw,4,cap)
    assert comparison==saved['comparison']
    cases.append({'index':index,'cap':cap,'workers':workers,'comparison':comparison,
                  'raw_probabilities':descriptor(Path(str(target)+'.f32')),
                  'native_push_median_micros':result['native_push_median_micros'],
                  'native_push_p95_micros':result['native_push_p95_micros'],
                  'native_push_max_micros':result['native_push_max_micros'],
                  'cold_native_load_micros':result['cold_native_load_micros'],
                  'finish_micros':result['finish_micros']})
assert inventory(root/'unpacked')==members
for name,want in metadata_descriptors.items():
    assert descriptor(root/name)==want
assert descriptor(Path(__file__))==collector_descriptor
dest=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix'
dest.mkdir()
for cap in (4,2,1):
    evidence=root/('unpacked/evidence-'+str(cap))
    for path in sorted(evidence.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.json','.log','.patch','.cpp','.diff','.f32','.txt'}:
            target=dest/('evidence-'+str(cap))/path.relative_to(evidence)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
for path in [root/'producer-run.json',root/'artifacts.json',root/'unpacked/protocol.json',root/'unpacked/qualification.json',Path(__file__)]:
    shutil.copy2(path,dest/path.name)
for name in p['tools']:
    target=dest/'sources'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(git_bytes('tools/nemotron-diarization/'+name))
(dest/'artifact-members.json').write_text(json.dumps(members,indent=2)+'\n')
summary={'schema_version':1,'source_commit':source,'run_id':run_id,'artifact_id':artifact['id'],
    'artifact':descriptor(root/'artifact.zip'),'artifact_members_rebound':len(members),
    'all_evaluator_sources_admitted_before_local_import':True,
    'tools_verified':len(p['tools']),'repository_files_verified':len(p['repository_files']),
    'diagnostic_completed':True,'release_qualified':False,'package_qualified':False,
    'public_or_lifecycle_qualified':False,'builds':build_summaries,'cases':cases,
    'independent_pristine_archive':archive_identity,'ggml_replay_commands':commands,
    'new_compiler_helper_binary_or_model_calls_during_collection':0,
    'canonical_upstream_patch_order_verified':True,
    'msvc_avx2_fma_f16c_compiler_defines_verified':True,
    'fma_f16c_cache_types':'UNINITIALIZED CLI values; effective definitions verified in every GGML CPU compiler command.',
    'copied_artifact_evidence_rebound_before_manifest':True,
    'limits':['All observations are native diagnostic timings; none is a supported live/public/package qualification.',
              'Every case retains complete6000x8 probabilities and final turns with unchanged1e-6 tolerance.',
              'No timing averaging/retry/promotion. Configured persistent-pool count is not an OS thread census.',
              'Downloaded archives and native binaries remain local; Git retains hashes, scoped logs/sources and full raw arrays.']}
# Bind the actual copies, not only the earlier replay inputs, before publication.
for item in dest.rglob('*'):
    if item.is_file() and item.relative_to(dest).as_posix() in members:
        assert descriptor(item)==members[item.relative_to(dest).as_posix()]
for name,want in metadata_descriptors.items():
    assert descriptor(dest/name)==want
assert descriptor(dest/Path(__file__).name)==collector_descriptor
assert inventory(root/'unpacked')==members
(dest/'collection-qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
files=inventory(dest)
(dest/'collection.json').write_text(json.dumps({'schema_version':1,'files':files},indent=2)+'\n')
print(json.dumps({'collection':str(dest),'members':len(files)+1,'cases':cases}))
