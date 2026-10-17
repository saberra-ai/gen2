"""Admit saved batched-emission macOS observer controls; no native/compiler/model calls."""
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
build=run=cache/'ggml-matrix-batch-control-local-v1'
dest=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-batch-control'

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

identity=desc(Path(__file__));q=read(run/'qualification.json');inputs=read(build/'inputs.json')
assert q['completed'] is True and q['production_qualified'] is False and 'error'not in q
assert q['all_144_outputs_bit_exact_per_shape'] is True
assert q['build']['inputs']==desc(build/'inputs.json')
assert q['controller']==desc(cache/'run-matrix-batch-control-local-v1.py')
fixed=q['final_bound_inputs'];outputs=q['final_output_bindings']
assert all(desc(Path(path))==value for path,value in {**fixed,**outputs}.items())
assert all(c['returncode']==0 and c['process_tree_reaped']is True and 'error'not in c for c in q['commands'])
assert len({c['name']for c in q['commands']})==len(q['commands'])
assert q['build']['completed'] is True
assert inventory(build/'source')==inputs['instrumented_files']
assert len(inputs['canonical_files'])==len(inputs['instrumented_files'])==3425
changed=[name for name,value in inputs['canonical_files'].items()if inputs['instrumented_files'].get(name)!=value]
assert changed==['ggml/src/ggml-cpu/ggml-cpu.c']
for path,value in inputs['subject_inputs'].items():assert desc(Path(path))==value
for name in inputs['local_import_closure']+['ggml_matrix_profile_control.cpp','ggml_matrix_profile_sgemm_control.cmake','fixtures/matrix-paths-ggml-cpu.c']:
    assert desc(run/'subjects'/name)==desc(build/'subjects'/name)==inputs['subject_inputs'][str(tools/name)]
assert desc(run/'subjects/spec.md')==desc(build/'subjects/spec.md')==inputs['subject_inputs'][str(repo/'docs/research/nemotron-matrix-batch-control.md')]
for name,row in q['build']['library_aliases'].items():
    path=Path(name);assert {'resolved':str(path.resolve(strict=True)),'link':os.readlink(path)if path.is_symlink()else None}==row
    assert path.resolve(strict=True).is_relative_to(build/'build')
database=read(build/'build/compile_commands.json');sources={};objects={};compilers={}
for row in database:
    source=Path(row['file']).resolve(strict=True);assert source==build/'control.cpp' or source.is_relative_to(build/'source')
    assert str(source)not in sources;sources[str(source)]=desc(source)
    argv=shlex.split(row['command']);assert argv.count('-o')==1
    obj=Path(argv[argv.index('-o')+1]);obj=obj if obj.is_absolute()else Path(row['directory'])/obj
    obj=obj.resolve(strict=True);assert obj.is_relative_to(build/'build')
    field=Path(row['output']);field=field if field.is_absolute()else build/'build'/field
    assert obj==field.resolve(strict=True);objects[str(obj)]=desc(obj)
    compiler=Path(argv[0]).resolve(strict=True);compilers[str(compiler)]=desc(compiler)
assert sources==q['build']['compiler_sources'] and objects==q['build']['objects'] and compilers==q['build']['compilers']
assert len(sources)==len(objects)==len(database)
cache_lines=(build/'build/CMakeCache.txt').read_text().splitlines()
expected_cache={'GGML_LLAMAFILE:BOOL':'ON','GGML_NATIVE:BOOL':'OFF','GGML_METAL:BOOL':'OFF',
 'GGML_ACCELERATE:BOOL':'OFF','GGML_BLAS:BOOL':'OFF','GGML_OPENMP:BOOL':'OFF','GGML_BACKEND_DL:BOOL':'OFF',
 'HAVE_DOTPROD:INTERNAL':'1','HAVE_SVE:INTERNAL':'','HAVE_MATMUL_INT8:INTERNAL':''}
assert q['sgemm_construction']['cache']==expected_cache
for key,value in expected_cache.items():assert [line for line in cache_lines if line.startswith(key+'=')]==[key+'='+value]
sgemm_inputs={}
for name in ('ggml/src/ggml-cpu/ggml-cpu.c','ggml/src/ggml-cpu/llamafile/sgemm.cpp'):
    row,=[r for r in database if Path(r['file']).resolve(strict=True)==build/'source'/name]
    argv=shlex.split(row['command']);assert '-DGGML_USE_LLAMAFILE' in argv and not any(a.startswith('-UGGML_USE_LLAMAFILE')for a in argv)
    sgemm_inputs[name]=row
assert sgemm_inputs==q['sgemm_construction']['compiler_inputs']
# Only import the already admitted parser/transform closure.
sys.path.insert(0,str(tools))
from ggml_matrix_profile import summarize
from ggml_matrix_batch_profile_source import transform,PATH
assert transform((tools/'fixtures/matrix-paths-ggml-cpu.c').read_bytes())==(build/'source'/PATH).read_bytes()
expected=[(w,i,e)for w in (1,2,4)for i,e in enumerate((False,True,True,False))]
assert [(c['workers'],c['index'],c['enabled'])for c in q['cases']]==expected
previous={}
for shape in range(4):
    path=repo/f'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-sgemm-control/workers-1-case-0.{shape}.0.f32'
    assert desc(path)==q['baseline_output_subjects'][str(shape)]==inputs['subject_inputs'][str(path)]
    previous[shape]=path.read_bytes()
all_outputs={};profiled_records=0
for case in q['cases']:
    workers,index,enabled=case['workers'],case['index'],case['enabled'];phase=f'workers-{workers}-case-{index}'
    profile=summarize((run/(phase+'.stderr.log')).read_text(),enabled,expected_graphs=12,expected_threads=workers,expected_matrices=1)
    assert profile==case['profile']
    stdout=[f'matrix-control threads={workers} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'for shape in range(4)for repeat in range(3)]
    assert (run/(phase+'.stdout.log')).read_text().splitlines()==stdout
    command,=[c for c in q['commands']if c['name']==phase]
    assert command['argv']==[str(build/'build/matrix-control'),str(workers),str(run/phase)]
    if enabled:
        assert len({g['pool']for g in profile['graphs']})==1
        for graph_index,graph in enumerate(profile['graphs']):
            shape=graph_index//3
            for worker in graph['workers']:
                matrix,=worker['matrices'];profiled_records+=1
                assert worker['fused']==1 and matrix['route']==[2,4,5,5][shape]
                assert matrix['types']==[8 if shape%2 else 0,0,0]
                assert matrix['a']==[64,64,1,1] and matrix['b']==[64,32 if shape<2 else 1,1,1]
                assert matrix['first'][:3]==([1,1,0]if shape==0 else[1,0,1])
                assert matrix['second'][:3]==([1,1,0]if shape==1 else[1,0,1]if shape==3 else[0,0,0])
                assert matrix['conversion'][0]==shape%2
    assert len(case['outputs'])==12
    for shape in range(4):
        for repeat in range(3):
            name=phase+f'.{shape}.{repeat}.f32';path=run/name;raw=path.read_bytes()
            assert desc(path)==case['outputs'][name]==outputs[str(path)]
            assert len(raw)==(8192 if shape<2 else 256)
            values=[v for v,in struct.iter_unpack('<f',raw)]
            assert all(math.isfinite(v)for v in values) and any(v!=0 for v in values)
            if shape not in previous:previous[shape]=raw
            assert raw==previous[shape];all_outputs[name]=path
assert len(all_outputs)==144 and profiled_records==168
assert set(outputs)=={str(p)for p in all_outputs.values()}|{str(run/(f'workers-{w}-case-{i}'+suffix))for w,i,_ in expected for suffix in ('.stdout.log','.stderr.log')}
files={'inputs.json':build/'inputs.json','qualification.json':build/'qualification.json',
 'controller.py':cache/'run-matrix-batch-control-local-v1.py','control.cpp':build/'control.cpp',
 'CMakeLists.txt':build/'CMakeLists.txt','CMakeCache.txt':build/'build/CMakeCache.txt',
 'compile_commands.json':build/'build/compile_commands.json'}
assert len(q['commands'])==24 and [row['name']for row in q['commands']]==['configure','build']+[f'dependencies-{i}'for i in range(10)]+[f'workers-{w}-case-{i}'for w,i,_ in expected]
for row in q['commands']:
    for suffix in ('.stdout.log','.stderr.log'):files[row['name']+suffix]=run/(row['name']+suffix)
for suffix in ('.stdout.log','.stderr.log'):files['controller'+suffix]=cache/('matrix-batch-control-local-v1.controller'+suffix)
for name in q['build']['compiler_files']:files['compiler/'+Path(name).name]=Path(name)
for name in inventory(run/'subjects'):files['subjects/'+name]=run/'subjects'/name
files.update(all_outputs)
bindings={name:desc(path)for name,path in files.items()}
assert not dest.exists();dest.mkdir()
for name,path in files.items():
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert desc(target)==bindings[name]
shutil.copyfile(Path(__file__),dest/'collector.py');assert desc(dest/'collector.py')==identity
assert all(desc(Path(path))==value for path,value in {**fixed,**outputs}.items())
assert all(desc(path)==desc(dest/name)==bindings[name]for name,path in files.items())
summary={'completed':True,'scope':__doc__,'collector':identity,'build_attempts':1,'prior_failed_control_remains_preserved':True,
    'native_configure_build_and_dependency_commands':12,'native_graph_processes':12,'graph_evaluations':144,'profiled_graph_evaluations':72,
    'profiled_worker_matrix_records':profiled_records,'canonical_source_files':3425,'compiler_inputs':len(sources),
    'raw_outputs':144,'raw_output_bytes_by_shape':[8192,8192,256,256],'all_outputs_bit_exact_per_shape':True,'all_outputs_bit_exact_to_previous_observer_control':True,
    'instrumented_cpu_source':inputs['instrumented_files'][PATH],'compiler_and_runtime_binaries_retained_in_cache':True,
    'sgemm_construction':q['sgemm_construction'],'probability_or_model_calls':0,'windows_or_model_qualified':False,'production_qualified':False,
    'limits':['Only macOS arm64 standalone GGML CPU graphs were executed.',
    'LLAMAFILE explicitly enabled; earlier per-row observer results and failures remain separate historical evidence.',
    'Worker wall regions overlap and include scheduling, internal waits and observer effects; no pure arithmetic or release performance claim.',
    'Windows full-model paths, observer overhead, numerical parity and original quality/performance/package gates remain open.']}
save(dest/'collection-qualification.json',summary);save(dest/'collection.json',{'files':inventory(dest)})
print(json.dumps({'receipt':str(dest),'files':len(inventory(dest)),'profiled_worker_matrix_records':profiled_records,'all_144_outputs_bit_exact_per_shape':True}))
