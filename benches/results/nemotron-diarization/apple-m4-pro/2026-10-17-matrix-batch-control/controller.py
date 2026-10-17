"""Fresh batched-emission macOS observer control; no SDK/model/Windows qualification."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time

repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
root=Path('/Users/victor/.cache/gen2-nemotron3-n0/ggml-matrix-batch-control-local-v1')
canonical=Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-compiler-fp-ci-v1/compiler-fp-replay-v1/ggml')
audit_path=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp/evidence-clang-cl/platform-llama-materialized-audit.json'

def desc(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def inventory(path):
    result={}
    for file in path.rglob('*'):
        mode=file.lstat().st_mode
        if stat.S_ISDIR(mode):continue
        assert stat.S_ISREG(mode) and not file.is_symlink()
        result[file.relative_to(path).as_posix()]=desc(file)
    return result

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def group_alive(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False

assert platform.system()=='Darwin' and platform.machine()=='arm64'
assert not root.exists() and not root.is_symlink();root.mkdir()
q={'completed':False,'scope':__doc__,'controller':desc(Path(__file__)),'host':dict(platform.uname()._asdict()),'commands':[],'cases':[],'production_qualified':False}
qpath=root/'qualification.json'
def persist():save(qpath,q)

def run(name,argv,timeout=180,env=None):
    row={'name':name,'argv':list(map(str,argv)),'owned_process_group':True,'process_tree_reaped':False}
    q['commands'].append(row);persist();child=None
    if env is None:env={k:v for k,v in os.environ.items() if k.upper() not in ('CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CL','_CL_') and not k.upper().startswith('DYLD_')}
    with(root/(name+'.stdout.log')).open('xb')as out,(root/(name+'.stderr.log')).open('xb')as err:
        try:
            child=subprocess.Popen(row['argv'],cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
            row['pid']=child.pid
            try:row['returncode']=child.wait(timeout=timeout)
            finally:
                if group_alive(child.pid):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                    if group_alive(child.pid):os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=5)
                for _ in range(100):
                    if not group_alive(child.pid):break
                    time.sleep(.01)
                row['process_tree_reaped']=not group_alive(child.pid)
            if not row['process_tree_reaped'] or row['returncode']!=0:raise RuntimeError('owned command failure: '+name)
        except BaseException as error:row['error']=repr(error);raise
        finally:persist()
    print('PASS '+name,flush=True)

persist()
try:
    original=json.loads(audit_path.read_text())['files'];assert len(original)==3425
    assert audit_path.read_bytes()==subprocess.check_output(['git','show','5e636c3cc834a0bdd3cb044c114bf5ed838477d1:'+audit_path.relative_to(repo).as_posix()],cwd=repo)
    assert inventory(canonical)==original
    pending=['ggml_matrix_batch_profile_source.py','ggml_matrix_profile.py','test_ggml_matrix_batch_profile.py'];modules=set()
    while pending:
        name=pending.pop()
        if name in modules:continue
        modules.add(name)
        for node in ast.walk(ast.parse((tools/name).read_bytes())):
            imports=[node.module]if isinstance(node,ast.ImportFrom)else[item.name for item in node.names]if isinstance(node,ast.Import)else[]
            for module in imports:
                local=(module or '').split('.')[0]+'.py'
                if(tools/local).is_file():pending.append(local)
    subjects={str(tools/name):desc(tools/name)for name in modules|{'ggml_matrix_profile_control.cpp','ggml_matrix_profile_sgemm_control.cmake','fixtures/matrix-paths-ggml-cpu.c'}}
    subjects[str(audit_path)]=desc(audit_path)
    spec=repo/'docs/research/nemotron-matrix-batch-control.md';subjects[str(spec)]=desc(spec)
    subjects[str(Path(__file__))]=q['controller']
    for name in modules-{'ggml_matrix_batch_profile_source.py','test_ggml_matrix_batch_profile.py'}:
        assert (tools/name).read_bytes()==subprocess.check_output(['git','show','4f8e3e8ea9bbf873a460b3a6048df7a39517e932:tools/nemotron-diarization/'+name],cwd=repo)
    baseline={}
    for shape in range(4):
        path=repo/f'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-sgemm-control/workers-1-case-0.{shape}.0.f32'
        assert path.read_bytes()==subprocess.check_output(['git','show','4f8e3e8ea9bbf873a460b3a6048df7a39517e932:'+path.relative_to(repo).as_posix()],cwd=repo)
        baseline[shape]=path.read_bytes();subjects[str(path)]=desc(path)
    q['baseline_output_subjects']={str(k):desc(repo/f'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-sgemm-control/workers-1-case-0.{k}.0.f32')for k in range(4)}
    snapshot=root/'subjects';snapshot.mkdir()
    for name in modules|{'ggml_matrix_profile_control.cpp','ggml_matrix_profile_sgemm_control.cmake','fixtures/matrix-paths-ggml-cpu.c'}:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tools/name,target)
    shutil.copyfile(spec,snapshot/'spec.md')
    sys.path.insert(0,str(tools))
    from ggml_matrix_batch_profile_source import transform,PATH
    from ggml_matrix_profile import environment,summarize
    source=root/'source';shutil.copytree(canonical,source)
    assert(source/PATH).read_bytes()==(tools/'fixtures/matrix-paths-ggml-cpu.c').read_bytes()
    (source/PATH).write_bytes(transform((source/PATH).read_bytes()))
    assert desc(source/PATH)=={'bytes':143331,'sha256':'2d0ea19dcc51f53ffc376f51ebd53aa5c575adeead0b29cf4bf1993a9ac58000'}
    expected=original.copy();expected[PATH]=desc(source/PATH)
    assert inventory(source)==expected
    shutil.copyfile(tools/'ggml_matrix_profile_control.cpp',root/'control.cpp')
    shutil.copyfile(tools/'ggml_matrix_profile_sgemm_control.cmake',root/'CMakeLists.txt')
    fixed={**subjects,str(root/'control.cpp'):desc(root/'control.cpp'),str(root/'CMakeLists.txt'):desc(root/'CMakeLists.txt')}
    cmake=Path(shutil.which('cmake')).resolve(strict=True);fixed[str(cmake)]=desc(cmake)
    inputs={'subject_inputs':subjects,'local_import_closure':sorted(modules),'canonical_files':original,'instrumented_files':expected,'audit':desc(audit_path),'fixed_inputs':fixed}
    save(root/'inputs.json',inputs);inputs_identity=desc(root/'inputs.json')
    def unchanged():
        assert desc(root/'inputs.json')==inputs_identity
        assert inventory(source)==expected
        assert all(desc(Path(name))==value for name,value in fixed.items())
    unchanged()
    run('configure',[cmake,'-S',root,'-B',root/'build','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
    run('build',[cmake,'--build',root/'build','--parallel','4','--verbose'])
    database=json.loads((root/'build/compile_commands.json').read_text())
    compiler_sources={};objects={};compilers={}
    for row in database:
        file=Path(row['file']).resolve(strict=True);assert file==root/'control.cpp' or file.is_relative_to(source)
        assert str(file)not in compiler_sources
        compiler_sources[str(file)]=desc(file)
        argv=shlex.split(row['command']);assert argv.count('-o')==1
        obj=Path(argv[argv.index('-o')+1]);obj=obj if obj.is_absolute()else Path(row['directory'])/obj
        obj=obj.resolve(strict=True);assert obj.is_relative_to(root/'build')
        output=Path(row['output']);output=output if output.is_absolute()else root/'build'/output
        assert output.resolve(strict=True)==obj;objects[str(obj)]=desc(obj)
        compiler=Path(argv[0]).resolve(strict=True);compilers[str(compiler)]=desc(compiler)
    assert len(compiler_sources)==len(objects)==len(database)
    cache_lines=(root/'build/CMakeCache.txt').read_text().splitlines()
    expected_cache={'GGML_LLAMAFILE:BOOL':'ON','GGML_NATIVE:BOOL':'OFF','GGML_METAL:BOOL':'OFF',
        'GGML_ACCELERATE:BOOL':'OFF','GGML_BLAS:BOOL':'OFF','GGML_OPENMP:BOOL':'OFF','GGML_BACKEND_DL:BOOL':'OFF',
        'HAVE_DOTPROD:INTERNAL':'1','HAVE_SVE:INTERNAL':'','HAVE_MATMUL_INT8:INTERNAL':''}
    for key,value in expected_cache.items():assert [line for line in cache_lines if line.startswith(key+'=')]==[key+'='+value]
    sgemm_inputs={}
    for name in ('ggml/src/ggml-cpu/ggml-cpu.c','ggml/src/ggml-cpu/llamafile/sgemm.cpp'):
        row,=[row for row in database if Path(row['file']).resolve(strict=True)==source/name]
        argv=shlex.split(row['command'])
        assert '-DGGML_USE_LLAMAFILE' in argv and not any(arg.startswith('-UGGML_USE_LLAMAFILE') for arg in argv)
        sgemm_inputs[name]=row
    q['sgemm_construction']={'cache':expected_cache,'compiler_inputs':sgemm_inputs};persist()
    binary=root/'build/matrix-control';binary_identity=desc(binary)
    libraries={str(path):desc(path)for path in (root/'build').rglob('*.dylib')}
    assert libraries
    aliases={name:{'resolved':str(Path(name).resolve(strict=True)),'link':os.readlink(name) if Path(name).is_symlink() else None}for name in libraries}
    assert all(Path(row['resolved']).is_relative_to(root/'build')for row in aliases.values())
    for index,path in enumerate([binary,*[Path(name) for name in libraries]]):run('dependencies-'+str(index),['/usr/bin/otool','-L',path])
    fixed.update({**compiler_sources,**objects,**compilers,**libraries,str(binary):binary_identity,
                  str(root/'build/compile_commands.json'):desc(root/'build/compile_commands.json'),str(root/'build/CMakeCache.txt'):desc(root/'build/CMakeCache.txt')})
    audit_logs={str(root/(row['name']+suffix)):desc(root/(row['name']+suffix))for row in q['commands']for suffix in ('.stdout.log','.stderr.log')}
    compiler_files={str(path):desc(path)for language in ('C','CXX') for path in (root/'build').glob('CMakeFiles/*/CMake'+language+'Compiler.cmake')}
    assert len(compiler_files)==2
    fixed.update({**audit_logs,**compiler_files})
    q['build']={'audit_logs':audit_logs,'compiler_files':compiler_files,'compiler_sources':compiler_sources,'objects':objects,'compilers':compilers,'libraries':libraries,'library_aliases':aliases,'binary':binary_identity,'inputs':inputs_identity,
        'compiler_database':desc(root/'build/compile_commands.json'),'cache':desc(root/'build/CMakeCache.txt'),'completed':True};persist()
    previous=baseline.copy();output_bindings={}
    for workers in (1,2,4):
        for index,enabled in enumerate((False,True,True,False)):
            unchanged();assert all(desc(Path(name))==value for name,value in output_bindings.items())
            assert aliases=={name:{'resolved':str(Path(name).resolve(strict=True)),'link':os.readlink(name) if Path(name).is_symlink() else None}for name in libraries}
            phase=f'workers-{workers}-case-{index}';prefix=root/phase
            child_environment={name:value for name,value in os.environ.items() if not name.upper().startswith('DYLD_')}
            run(phase,[binary,str(workers),prefix],env=environment(child_environment,enabled))
            stdout=root/(phase+'.stdout.log');stderr=root/(phase+'.stderr.log')
            lines=[f'matrix-control threads={workers} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'for shape in range(4)for repeat in range(3)]
            assert stdout.read_text().splitlines()==lines
            profile=summarize(stderr.read_text(),enabled,expected_graphs=12,expected_threads=workers,expected_matrices=1)
            if enabled:
                assert len({g['pool']for g in profile['graphs']})==1
                for graph_index,graph in enumerate(profile['graphs']):
                    shape=graph_index//3
                    for worker in graph['workers']:
                        assert worker['fused']==1
                        matrix,=worker['matrices']
                        assert matrix['route']==[2,4,5,5][shape]
                        assert matrix['types']==[8 if shape%2 else 0,0,0]
                        assert matrix['a']==[64,64,1,1] and matrix['b']==[64,32 if shape<2 else 1,1,1]
                        assert matrix['first'][:3]==([1,1,0]if shape==0 else[1,0,1])
                        assert matrix['second'][:3]==([1,1,0]if shape==1 else[1,0,1]if shape==3 else[0,0,0])
                        assert matrix['conversion'][0]==shape%2
            outputs={}
            for shape in range(4):
                for repeat in range(3):
                    path=Path(str(prefix)+f'.{shape}.{repeat}.f32');raw=path.read_bytes()
                    assert len(raw)==(8192 if shape<2 else 256)
                    values=[value for value,in struct.iter_unpack('<f',raw)]
                    assert all(math.isfinite(v)for v in values) and any(v!=0 for v in values)
                    if shape not in previous:previous[shape]=raw
                    assert raw==previous[shape]
                    outputs[path.name]=desc(path);output_bindings[str(path)]=desc(path)
            output_bindings.update({str(stdout):desc(stdout),str(stderr):desc(stderr)})
            q['cases'].append({'workers':workers,'index':index,'enabled':enabled,'profile':profile,'outputs':outputs});persist()
    unchanged();assert all(desc(Path(name))==value for name,value in output_bindings.items())
    assert aliases=={name:{'resolved':str(Path(name).resolve(strict=True)),'link':os.readlink(name) if Path(name).is_symlink() else None}for name in libraries}
    q['final_bound_inputs']=fixed.copy();q['final_output_bindings']=output_bindings.copy()
    q['all_144_outputs_bit_exact_per_shape']=True;q['completed']=True
except BaseException as error:q['error']=repr(error);raise
finally:persist()
print(json.dumps({'completed':q['completed'],'cases':len(q['cases']),'compiler_inputs':len(q.get('build',{}).get('compiler_sources',{}))}))
