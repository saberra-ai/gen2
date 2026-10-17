"""Saved-only admission of a failed macOS matrix observer control; no native/compiler/model calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import struct
import sys

cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
build=cache/'ggml-matrix-path-control-local-v1';run=cache/'ggml-matrix-path-execution-local-v2'
dest=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-control-failure'

def desc(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def read(path):return json.loads(path.read_text())
def inventory(root):
    assert root.is_dir() and not root.is_symlink()
    result={}
    for path in root.rglob('*'):
        mode=path.lstat().st_mode
        if stat.S_ISDIR(mode):continue
        assert stat.S_ISREG(mode) and not path.is_symlink()
        result[path.relative_to(root).as_posix()]=desc(path)
    return result

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

identity=desc(Path(__file__));q=read(run/'qualification.json');inputs=read(build/'inputs.json');admission=read(run/'admission.json')
assert q['completed']is False and q['production_qualified']is False and q['error']=='AssertionError()'
assert q['admission']==desc(run/'admission.json') and admission['original_inputs']==desc(build/'inputs.json')
assert admission['controller']==q['controller']==desc(cache/'run-matrix-path-execution-local-v2.py')
fixed=admission['fixed_inputs'].copy()
for field in ('audit_logs','compiler_files','compiler_sources','objects','compilers','libraries'):fixed.update(q['build'][field])
fixed.update({str(run/'admission.json'):q['admission'],str(build/'build/matrix-control'):q['build']['binary'],
 str(build/'build/compile_commands.json'):q['build']['compiler_database'],str(build/'build/CMakeCache.txt'):q['build']['cache']})
assert all(desc(Path(path))==value for path,value in fixed.items())
assert q['prior_failed_attempt']==admission['prior_failed_attempt']
failed=read(build/'qualification.json')
assert failed['completed']is False and failed['cases']==[] and len(failed['commands'])==2
assert failed['error']=="FileNotFoundError(2, 'No such file or directory')"
assert failed['controller']==desc(cache/'run-matrix-path-control-local-v1.py')
assert all(c['returncode']==0 and c['process_tree_reaped']is True and 'error'not in c for c in failed['commands']+q['commands'])
assert len({c['name']for c in q['commands']})==len(q['commands'])==12
assert [c['name']for c in q['commands']]==[f'dependencies-{i}'for i in range(10)]+['workers-1-case-0','workers-1-case-1']
assert q['build']['completed']is True and len(q['cases'])==1
assert inventory(build/'source')==inputs['instrumented_files']==admission['instrumented_files']
assert len(inputs['canonical_files'])==len(inputs['instrumented_files'])==3425
assert [name for name,value in inputs['canonical_files'].items()if inputs['instrumented_files'].get(name)!=value]==['ggml/src/ggml-cpu/ggml-cpu.c']
for path,value in inputs['subject_inputs'].items():assert desc(Path(path))==value
for name in inputs['local_import_closure']+['ggml_matrix_profile_control.cpp','ggml_matrix_profile_control.cmake','fixtures/matrix-paths-ggml-cpu.c']:
    assert desc(run/'subjects'/name)==desc(build/'subjects'/name)==inputs['subject_inputs'][str(tools/name)]
assert desc(run/'subjects/spec.md')==desc(build/'subjects/spec.md')==inputs['subject_inputs'][str(repo/'docs/research/nemotron-matrix-path-profile.md')]
for name,row in q['build']['library_aliases'].items():
    path=Path(name);assert {'resolved':str(path.resolve(strict=True)),'link':os.readlink(path)if path.is_symlink()else None}==row
    assert path.resolve(strict=True).is_relative_to(build/'build')
database=read(build/'build/compile_commands.json');sources={};objects={};compilers={}
for row in database:
    source=Path(row['file']).resolve(strict=True);assert source==build/'control.cpp' or source.is_relative_to(build/'source')
    assert str(source)not in sources;sources[str(source)]=desc(source)
    argv=shlex.split(row['command']);assert argv.count('-o')==1 and not any('GGML_USE_LLAMAFILE'in arg for arg in argv)
    obj=Path(argv[argv.index('-o')+1]);obj=obj if obj.is_absolute()else Path(row['directory'])/obj
    obj=obj.resolve(strict=True);assert obj.is_relative_to(build/'build')
    field=Path(row['output']);field=field if field.is_absolute()else build/'build'/field
    assert obj==field.resolve(strict=True);objects[str(obj)]=desc(obj)
    compiler=Path(argv[0]).resolve(strict=True);compilers[str(compiler)]=desc(compiler)
assert sources==q['build']['compiler_sources'] and objects==q['build']['objects'] and compilers==q['build']['compilers']
assert len(sources)==len(objects)==len(database)==27
assert 'GGML_LLAMAFILE:BOOL=OFF' in (build/'build/CMakeCache.txt').read_text().splitlines()
# Freeze the two actual graph process outputs before any local decoder import.
observed=inventory(run);fixed.update({str(run/name):value for name,value in observed.items()})
sys.path.insert(0,str(tools))
from ggml_matrix_profile import summarize
from ggml_matrix_profile_source import transform,PATH
assert transform((tools/'fixtures/matrix-paths-ggml-cpu.c').read_bytes())==(build/'source'/PATH).read_bytes()
case,=q['cases'];assert (case['workers'],case['index'],case['enabled'])==(1,0,False)
previous={};all_outputs={};profiles=[]
for index,enabled in enumerate((False,True)):
    phase=f'workers-1-case-{index}'
    profile=summarize((run/(phase+'.stderr.log')).read_text(),enabled,expected_graphs=12,expected_threads=1,expected_matrices=1);profiles.append(profile)
    stdout=[f'matrix-control threads=1 shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'for shape in range(4)for repeat in range(3)]
    assert (run/(phase+'.stdout.log')).read_text().splitlines()==stdout
    command,=[c for c in q['commands']if c['name']==phase]
    assert command['argv']==[str(build/'build/matrix-control'),'1',str(run/phase)]
    if not enabled:assert profile==case['profile']
    else:
        assert len({g['pool']for g in profile['graphs']})==1
        for graph_index,graph in enumerate(profile['graphs']):
            shape=graph_index//3;worker,=graph['workers'];matrix,=worker['matrices']
            assert worker['fused']==1 and matrix['route']==5
            assert matrix['types']==[8 if shape%2 else 0,0,0] and matrix['dot']==(8 if shape%2 else 0)
            assert matrix['a']==[64,64,1,1] and matrix['b']==[64,32 if shape<2 else 1,1,1]
            assert matrix['first']==matrix['second']==[0,0,0,0] and matrix['conversion'][0]==shape%2
            assert matrix['chunks'][0]==(8 if shape<2 else 1)
    for shape in range(4):
        for repeat in range(3):
            name=phase+f'.{shape}.{repeat}.f32';path=run/name;raw=path.read_bytes()
            if not enabled:assert desc(path)==case['outputs'][name]
            assert len(raw)==(8192 if shape<2 else 256)
            values=[v for v,in struct.iter_unpack('<f',raw)]
            assert all(math.isfinite(v)for v in values) and any(v!=0 for v in values)
            if shape not in previous:previous[shape]=raw
            assert raw==previous[shape];all_outputs[name]=path
assert len(all_outputs)==24 and set(all_outputs)==set(p.name for p in run.glob('*.f32'))
files={'build-inputs.json':build/'inputs.json','build-attempt-qualification.json':build/'qualification.json',
 'build-controller.py':cache/'run-matrix-path-control-local-v1.py','execution-admission.json':run/'admission.json',
 'execution-qualification.json':run/'qualification.json','execution-controller.py':cache/'run-matrix-path-execution-local-v2.py',
 'control.cpp':build/'control.cpp','CMakeLists.txt':build/'CMakeLists.txt','CMakeCache.txt':build/'build/CMakeCache.txt',
 'compile_commands.json':build/'build/compile_commands.json'}
for row in failed['commands']:
    for suffix in ('.stdout.log','.stderr.log'):files['build-'+row['name']+suffix]=build/(row['name']+suffix)
for row in q['commands']:
    for suffix in ('.stdout.log','.stderr.log'):files[row['name']+suffix]=run/(row['name']+suffix)
for label,prefix in [('build','matrix-path-control-local-v1.controller'),('execution','matrix-path-execution-local-v2.controller')]:
    for suffix in ('.stdout.log','.stderr.log'):files[label+'-controller'+suffix]=cache/(prefix+suffix)
for name in q['build']['compiler_files']:files['compiler/'+Path(name).name]=Path(name)
for name in inventory(run/'subjects'):files['subjects/'+name]=run/'subjects'/name
files.update(all_outputs)
bindings={name:desc(path)for name,path in files.items()}
assert not dest.exists();dest.mkdir()
for name,path in files.items():
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert desc(target)==bindings[name]
shutil.copyfile(Path(__file__),dest/'collector.py');assert desc(dest/'collector.py')==identity
assert all(desc(Path(path))==value for path,value in fixed.items())
assert inventory(run)==observed
assert all(desc(path)==desc(dest/name)==bindings[name]for name,path in files.items())
summary={'collection_completed':True,'control_completed':False,'scope':__doc__,'collector':identity,'build_attempts':1,
 'preserved_post_build_admission_failures':1,'preserved_route_prediction_failures':1,'rebuilds':0,'native_graph_processes':2,
 'graph_evaluations':24,'profiled_graph_evaluations':12,'profiled_worker_matrix_records':12,
 'canonical_source_files':3425,'compiler_inputs':27,'raw_outputs':24,'raw_output_bytes_by_shape':[8192,8192,256,256],
 'all_24_outputs_bit_exact_per_shape':True,'planned_144_output_control_passed':False,
 'actual_routes_all_vec_dot':True,'sgemm_calls_observed':0,'ggml_llamafile':'OFF',
 'instrumented_cpu_source':inputs['instrumented_files'][PATH],'compiler_and_runtime_binaries_retained_in_cache':True,
 'probability_or_model_calls':0,'windows_or_model_qualified':False,'production_qualified':False,
 'limits':['Only two macOS arm64 standalone GGML CPU processes ran, at one worker: off then on.',
 'V1 built successfully then failed object-path admission before graphs; V2 admits that same binary and performs its first graph runs.',
 'V2 stopped after its first enabled process: actual routes were [5,5,5,5], contradicting expected [2,4,5,5].',
 'The actual cache has GGML_LLAMAFILE=OFF and no compile command defines GGML_USE_LLAMAFILE; this run cannot validate either SGEMM stage.',
 'Saved-only admission verifies all 24 raw outputs and complete enabled trace after failure, without native retries or claiming the original controller completed.',
 'Worker walls include scheduling, internal waits and observer effects; no release performance, Windows, full-model or quality qualification.']}
save(dest/'saved-profiles.json',profiles);save(dest/'collection-qualification.json',summary);save(dest/'collection.json',{'files':inventory(dest)})
print(json.dumps({'receipt':str(dest),'files':len(inventory(dest)),'raw_outputs':24,'control_completed':False}))
