"""Replay saved Q8 tile source/build/vector/timing evidence; no native calls."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PureWindowsPath
import subprocess
import sys

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-q8-worker-timing-ci-v1')
work=root/'delivery-admission-v1'
delivered=work/'delivered'
source='7fadd1497f58f2d287dc44e90c1337283074ad5e'

def desc(path):
    h=hashlib.sha256(); size=0
    with path.open('rb') as stream:
        while block:=stream.read(1024**2): size+=len(block); h.update(block)
    return {'bytes':size,'sha256':h.hexdigest()}

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

identity=desc(Path(__file__))
admission=read(work/'qualification.json')
admission_identity=desc(work/'qualification.json')
assert admission['source_commit']==source and admission['run_id']==39056905799 and admission['job_id']==117466626836
assert desc(root/'artifact.zip')==admission['archive']
assert desc(root.parent/'admit-windows-q8-worker-timing-delivery-v1.py')==admission['collector']
assert all(desc(root/name)==value for name,value in admission['metadata'].items())
assert desc(work/'artifact-members.json')==admission['artifact_members']
p=read(delivered/'protocol.json')
env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
assert (delivered/'protocol.json').read_bytes()==subprocess.check_output(['git','show',source+':tools/nemotron-diarization/windows-q8-worker-timing.protocol.json'],cwd=repo,env=env)
for group,prefix in [('tools','tools/nemotron-diarization/'),('repository_files','')]:
    for name,expected in p[group].items():
        raw=subprocess.check_output(['git','show',source+':'+prefix+name],cwd=repo,env=env)
        assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected
        assert desc(repo/(prefix+name))==expected
sys.path.insert(0,str(tools))
def file_inventory(root):
    result, folded, count, total = {}, set(), 0, 0
    assert root.is_dir() and not root.is_symlink()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            count += 1
            assert count <= 10000 and relative.casefold() not in folded
            folded.add(relative.casefold())
            mode = path.lstat().st_mode
            assert not stat.S_ISLNK(mode)
            if stat.S_ISDIR(mode):
                continue
            assert stat.S_ISREG(mode)
            value = desc(path)
            assert 0 <= value['bytes'] <= 256 * 1024**2
            total += value['bytes']
            assert total <= 1024**3
            result[relative] = value
    return result



from q8_mx4_geometry import MODEL_RESULT, admit_geometry
from windows_q8_worker_timing_metrics import ORDER, SHAPES, GUARD_CODES, admit_guard, admit_half_table, validate_output, timing_evidence
from q8_mx4_source import PATH, ORIGINAL, transform
from windows_compiler_fp_metrics import FLAG, baseline, cache_value
from windows_compiler_metrics import command_compiler

members=read(work/'artifact-members.json')
assert file_inventory(delivered)==members
q=read(delivered/'qualification.json')
supervision=read(root/'supervision-replay-v1.json')
assert supervision['source_commit']==source and supervision['run_id']==39056905799
assert supervision['archive']==admission['archive'] and supervision['delivery_admission']==admission_identity
assert supervision['collector']==desc(root.parent/'replay-windows-q8-worker-timing-supervision-v1.py')
assert supervision['commands_verified']==len(q['commands'])
assert q['source_commit']==source and q['protocol']==desc(delivered/'protocol.json')
assert q['order']==ORDER and q['shapes']==p['shapes'] and q['limits']==p['limits']
assert q['kernel_order']==p['kernel_order']==['control','mx4','mx4','control']
assert q['new_sdk_or_model_calls']==0
assert all(q[name] is False for name in ['release_qualified','package_qualified','public_or_lifecycle_qualified','sdk_or_model_qualified'])
prior_root=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno'
audit=read(prior_root/'evidence-control/platform-llama-materialized-audit.json')
canonical_observed = file_inventory(delivered/'canonical') if (delivered/'canonical').is_dir() else {}
canonical_complete = canonical_observed == audit['files'] and len(audit['files']) == 3425
archive_matches = (delivered/'pristine.tar').is_file() and desc(delivered/'pristine.tar') == audit['archive']
evidence=delivered/'evidence'
original_matches = (delivered/'canonical'/PATH).is_file() and (delivered/'canonical'/PATH).read_bytes() == (tools/'fixtures/q8-tile-sgemm.cpp').read_bytes()
candidate_matches = ((evidence/'sgemm-mx4.cpp').is_file()
                     and (evidence/'sgemm-mx4.cpp').read_bytes() == transform((tools/'fixtures/q8-tile-sgemm.cpp').read_bytes())
                     and desc(evidence/'sgemm-mx4.cpp') == p['candidate_source'])
assert len(audit['patch_series']) == 19
patches_matching = []
for index,item in enumerate(audit['patch_series']):
    path = evidence/('patch-'+str(index)+'-read.stdout.log')
    patches_matching.append(path.is_file() and desc(path) == {key:item[key]for key in ['bytes','sha256']})
source_checks = {'canonical_complete_and_exact': canonical_complete, 'pristine_archive_matches': archive_matches,
                 'original_sgemm_matches': original_matches, 'candidate_sgemm_matches': candidate_matches,
                 'original_patch_reads_matching': patches_matching, 'observed_canonical_files': len(canonical_observed)}
if q['diagnostic_completed'] is True:
    assert canonical_complete and archive_matches and original_matches and candidate_matches and all(patches_matching)
result={'schema_version':1,'source_commit':source,'run_id':39056905799,'job_id':117466626836,
        'artifact_id':admission['artifact_id'],'archive':admission['archive'],'collector':identity,
        'delivery_admission':admission_identity,'supervision_replay':desc(root/'supervision-replay-v1.json'),
        'diagnostic_completed':q['diagnostic_completed'],'canonical_files_verified':3425 if canonical_complete else 0,
        'source_checks':source_checks,
        'new_compiler_sdk_model_calls':0,'release_qualified':False,'model_rtf_qualified':False}
if q['diagnostic_completed'] is True:
    assert 'error' not in q
    geometry=admit_geometry(read(repo/MODEL_RESULT/'collection-qualification.json'))
    assert q['model_geometry_admission']==p['model_geometry_admission']==geometry
    result['model_geometry_admission']=geometry
    assert all(row['returncode']==row['expected_returncode'] and 'error' not in row for row in q['commands'])
    assert len(q['commands'])==95 and supervision['all_observed_commands_cleanup_admitted'] is True
    assert q['host']['avx2'] is True and q['host']['fma'] is True
    assert sum(row['NumberOfLogicalProcessors']for row in q['host']['cpu'])>=4
    build=q['build']
    assert build['canonical_files']==audit['files'] and build['patch_series']==audit['patch_series']
    assert build['archive']==audit['archive'] and build['candidate']==p['candidate_source']
    project_inputs={PureWindowsPath(n).name:v for n,v in build['project_inputs'].items()}
    assert len(build['project_inputs'])==len(project_inputs)==4
    assert project_inputs=={'pristine.tar':audit['archive'],'sgemm-mx4.cpp':p['candidate_source'],
        'CMakeLists.txt':p['tools']['q8_worker_timing_benchmark.cmake'],'q8_worker_timing_benchmark.cpp':p['tools']['q8_worker_timing_benchmark.cpp']}
    assert desc(evidence/'q8_tile_benchmark.exe')==build['binary']
    assert desc(evidence/'compile_commands.json')==build['compile_database']
    assert desc(evidence/'binary-dependencies.stdout.log')==build['dependencies']
    assert not re.search(r'libomp|libgomp|vcomp\d*\.dll',(evidence/'binary-dependencies.stdout.log').read_text(),re.I)
    assert not re.search(r'libomp|libgomp|vcomp\d*\.lib|[-/]f?openmp|\[-Wexceptions\]|non-throwing exception specification',(evidence/'build.stdout.log').read_text(),re.I)
    cache=(evidence/'CMakeCache.txt').read_text()
    for name in ['GGML_CPU','GGML_OPENMP','GGML_BACKEND_DL','BUILD_SHARED_LIBS']:
        assert re.findall(r'^'+name+r':BOOL=([^\r\n]*)\r?$',cache,re.M)==['OFF']
    prior=baseline('clang-cl')
    for name,expected in [('compiler',prior[2]['compiler_binary']),('linker',prior[2]['linker_binary'])]:
        assert {key:q['toolchain'][name][key]for key in ['bytes','sha256']}==expected
    cache=(evidence/'CMakeCache.txt').read_text()
    for language in ('C','CXX'):
        name='CMAKE_'+language+'_FLAGS'; flags=cache_value(cache,name)
        assert flags.count(FLAG)==1 and cache_value(cache,name+'_INIT')==[FLAG]
        flags.remove(FLAG)
        assert flags==cache_value(prior[0],name)
        assert cache_value(cache,name+'_RELEASE')==cache_value(prior[0],name+'_RELEASE')
    units=build['kernel_translation_units']
    assert set(units)=={'control','mx4'}
    canonical=PureWindowsPath(units['control']['source']).parents[4]
    candidate=PureWindowsPath(units['mx4']['source'])
    compiled={PureWindowsPath(name):value for name,value in build['compiler_inputs'].items()}
    objects={PureWindowsPath(name):value for name,value in build['objects'].items()}
    database=read(evidence/'compile_commands.json')
    assert len(database)==len(compiled)==len(objects)>2
    seen=set()
    for row in database:
        path=PureWindowsPath(row['file']); assert path not in seen;seen.add(path)
        if path.is_relative_to(canonical): expected=audit['files'][path.relative_to(canonical).as_posix()]
        elif path==candidate: expected=p['candidate_source']
        else:
            assert path.name=='q8_worker_timing_benchmark.cpp'
            expected=p['tools']['q8_worker_timing_benchmark.cpp']
        assert compiled[path]==expected
        assert command_compiler(row['command']) in build['compiler_aliases']
        assert build['compiler_aliases'][command_compiler(row['command'])]==prior[2]['compiler_binary']
        assert row['command'].split().count(FLAG)==1
        assert not any(token in row['command']for token in ['/fp:fast','-ffast-math','-fno-math-errno','/GL','-flto','/openmp','-fopenmp'])
        obj=PureWindowsPath(row['output'])
        if not obj.is_absolute():obj=PureWindowsPath(row['directory'])/obj
        assert obj in objects
        for variant,unit in units.items():
            if path==PureWindowsPath(unit['source']):
                assert row['command']==unit['command'] and row['command'].count('/arch:AVX2')==1
                assert 'llamafile_sgemm='+unit['entry'] in row['command']
                if variant=='mx4':assert row['command'].count('GEN2_Q8_MX4_EXPECTED=1')==1
                assert desc(evidence/('kernel-'+variant+'.obj'))==unit['object']==objects[obj]
                assert unit['entry'] in (evidence/('disasm-'+variant+'.stdout.log')).read_text()
    assert seen==set(compiled)
    assert len(q['cases'])==12
    reference=None;reference_half=None;cases=[]
    for index,case in enumerate(q['cases']):
        workers=(1,2,4)[index//4]; variant=ORDER[index%4];phase=f'case-{workers}-{index%4}'
        assert case['workers']==workers and case['index']==index%4 and case['telemetry_mode']==variant
        assert case['command_started'] is True and case['outputs_bit_exact'] is True
        controls=[]
        for mode,expected in GUARD_CODES.items():
            name=phase+'-guard-'+mode
            control_row,=[row for row in q['commands']if row['name']==name]
            assert control_row['expected_returncode']==expected and control_row['command'][1:]==['--support-control',mode]
            controls.append(admit_guard(mode,control_row['returncode'],(evidence/(name+'.stdout.log')).read_bytes(),(evidence/(name+'.log')).read_bytes()))
        assert controls==case['support_controls']
        half_raw=(evidence/(phase+'.half.f32')).read_bytes()
        assert admit_half_table(half_raw)==case['half_table']
        if reference_half is None:reference_half=half_raw
        assert half_raw==reference_half
        row,=[row for row in q['commands']if row['name']==phase]
        assert row['command'][1:3]==[variant,str(workers)]
        outputs=[(evidence/(phase+'.'+str(shape)+'.'+str(position)+'.f32')).read_bytes()for shape in range(len(SHAPES))for position in range(4)]
        observed=validate_output((evidence/(phase+'.stdout.log')).read_text(),variant,workers,outputs)
        assert observed==case['observed']
        if reference is None:reference=outputs
        assert outputs==reference
        cases.append(case)
    timing=timing_evidence(cases)
    assert q['timing_comparison']==timing and q['all_1056_outputs_bit_exact'] is True
    assert q['all_12_half_tables_bit_exact'] is True and q['all_36_support_rejections_verified'] is True
    result.update(all_1056_outputs_bit_exact=True,all_12_half_tables_bit_exact=True,all_36_support_rejections_verified=True,timing_comparison=timing,
                  compiler_inputs_verified=len(compiled),host=q['host'],commands_verified=len(q['commands']))
else:
    assert q.get('error')
    result.update(error=q['error'],commands=q['commands'],scope='Uncompleted kernel experiment; no numerical or timing acceptance.')
assert desc(Path(__file__))==identity
assert desc(work/'qualification.json')==admission_identity
assert all(desc(root/name)==value for name,value in admission['metadata'].items())
assert desc(root/'artifact.zip')==admission['archive'] and file_inventory(delivered)==members
with(root/'collection-replay-v1.json').open('x')as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
