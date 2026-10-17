"""Actual Windows model-free Q8 tile control; no SDK or model qualification."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import time

from build_valid_mel_candidate import descriptor, source_inventory
from frozen_git_sources import verify_git_sources
from platform_native import capture, msvc_environment, msvc_tool, pinned_git_environment
from profile_windows_sessions import HOST_SCRIPT
from qualification_storage import atomic_json
from q8_mx4_source import PATH, ORIGINAL, transform
from windows_compiler_fp_metrics import FLAG, baseline, cache_value
from windows_compiler_metrics import admit_compiler_paths, command_compiler, compiler_environment
from windows_job_v2 import run_in_job
from windows_q8_worker_timing_metrics import ORDER, SHAPES, PRIMARY_SHAPES, REPETITIONS, GUARD_CODES, admit_guard, admit_half_table, validate_output, timing_evidence

from q8_mx4_geometry import MODEL_RESULT, admit_geometry

REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno'


def field(text, name):
    values = re.findall(r'^' + re.escape(name) + r':(?:STRING|FILEPATH|BOOL|INTERNAL|UNINITIALIZED)=([^\r\n]*)\r?$', text, re.M)
    if len(values) != 1:
        raise ValueError('one CMake value required: ' + name)
    return values[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 required')
    tools = Path(__file__).resolve().parent
    repo, source, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (output.exists() or output.is_symlink() or output.is_relative_to(repo) or output.is_relative_to(source)
            or source.is_relative_to(output)):
        parser.error('fresh disjoint output required')
    protocol_path = tools / 'windows-q8-worker-timing.protocol.json'
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER or p['shapes'] != list(map(list, SHAPES)):
            raise ValueError('tile protocol changed')
        if (p['schema_version'] != 4 or p['kernel_order'] != ['control','mx4','mx4','control'] or p['guard_codes'] != GUARD_CODES
                or p['half_table_entries'] != 65536 or p['half_table_bytes'] != 262144
                or p['finite_half_entries'] != 63488 or p['primary_shapes'] != PRIMARY_SHAPES
                or p['timed_calls_per_shape'] != REPETITIONS):
            raise ValueError('corrected support/table protocol required')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('tile frozen input changed: ' + name)

    frozen()
    verify_git_sources(repo, commit, {**{'tools/nemotron-diarization/' + name:value for name,value in p['tools'].items()}, **p['repository_files']})
    if subprocess.check_output(['git','show',commit + ':tools/nemotron-diarization/' + protocol_path.name],
            cwd=repo,env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}) != raw_protocol:
        raise ValueError('committed tile protocol required')
    output.mkdir()
    evidence = output / 'evidence'
    evidence.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    q = {'schema_version':1,'source_commit':commit,'protocol':descriptor(protocol_path),'diagnostic_completed':False,
        'release_qualified':False,'package_qualified':False,'public_or_lifecycle_qualified':False,
        'sdk_or_model_qualified':False,'new_sdk_or_model_calls':0,'kernel_order':p['kernel_order'],'order':ORDER,'shapes':p['shapes'],'cases':[],'limits':p['limits']}
    commands = []
    persist = lambda: atomic_json(output / 'qualification.json', q)
    environment = {}

    def run(name, command, cwd=repo, timeout=600, overrides=None, expected_returncode=0):
        frozen()
        row = {'name':name,'command':list(map(str,command)),'cwd':str(cwd),
            'process_supervision':'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup',
            'process_tree_reaped':False,'expected_returncode':expected_returncode}
        commands.append(row)
        atomic_json(evidence / 'commands.json', commands)
        print('START ' + name, flush=True)
        started=time.monotonic()
        try:
            with (evidence/(name+'.stdout.log')).open('xb') as out, (evidence/(name+'.log')).open('xb') as err:
                result=run_in_job(row['command'],cwd=cwd,env={**environment,**(overrides or {})},stdout=out,stderr=err,timeout=timeout)
            row.update(returncode=result.returncode,process_tree_reaped=result.process_tree_reaped,job_observation=result.job_observation)
            if result.returncode != expected_returncode:
                raise ValueError('unexpected actual command exit: ' + str(result.returncode) + '; expected ' + str(expected_returncode))
        except BaseException as error:
            row.update(error=repr(error),process_tree_reaped=getattr(error,'process_tree_reaped',row['process_tree_reaped']),
                job_observation=getattr(error,'job_observation',row.get('job_observation')))
            raise
        finally:
            row['elapsed_seconds']=time.monotonic()-started
            atomic_json(evidence/'commands.json',commands)
        print('PASS '+name,flush=True)
        return result.returncode

    persist()
    try:
        actual_geometry=admit_geometry(json.loads((repo/MODEL_RESULT/'collection-qualification.json').read_text()))
        if actual_geometry!=p['model_geometry_admission']:
            raise ValueError('exact captured model geometry proof required before construction')
        q['model_geometry_admission']=actual_geometry
        persist()
        environment=pinned_git_environment(compiler_environment(msvc_environment(dict(os.environ))))
        compiler=msvc_tool(environment,'clang-cl.exe')
        linker=msvc_tool(environment,'lld-link.exe')
        archive_tool=msvc_tool(environment,'lib.exe')
        dumpbin=msvc_tool(environment,'dumpbin.exe')
        cmake=msvc_tool(environment,'cmake.exe')
        try:
            ninja=msvc_tool(environment,'ninja.exe')
        except ValueError:
            variables={name.upper():value for name,value in environment.items()}
            ninja=(Path(variables['VSINSTALLDIR'])/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe').resolve(strict=True)
        q['toolchain']={name:{'path':str(path),**descriptor(path)} for name,path in
            [('compiler',compiler),('linker',linker),('archive_tool',archive_tool),('dumpbin',dumpbin),('cmake',cmake),('ninja',ninja)]}
        prior=baseline('clang-cl')[2]
        if descriptor(compiler)!=prior['compiler_binary'] or descriptor(linker)!=prior['linker_binary']:
            raise ValueError('original exact Clang and linker required')
        run('receiver-host',['pwsh','-NoProfile','-NonInteractive','-Command',HOST_SCRIPT])
        q['host']=json.loads((evidence/'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        if (q['host']['avx2'] is not True or q['host']['fma'] is not True
                or sum(cpu['NumberOfLogicalProcessors'] for cpu in q['host']['cpu'])<4):
            raise ValueError('actual AVX2/FMA four-logical-processor host required')
        reference=repo/REFERENCE
        manifest=json.loads((reference/'collection.json').read_text())['files']
        audit_name='evidence-control/platform-llama-materialized-audit.json'
        if descriptor(reference/audit_name)!=manifest[audit_name]:
            raise ValueError('original canonical source audit changed')
        original_audit=json.loads((reference/audit_name).read_text())
        run('native-revision',['git','-C',source,'rev-parse','HEAD'])
        run('llama-revision',['git','-C',source/'llama.cpp','rev-parse','HEAD'])
        for name,expected in [('native',p['native_revision']),('llama',p['llama_revision'])]:
            if (evidence/(name+'-revision.stdout.log')).read_text().strip()!=expected:
                raise ValueError('pinned '+name+' source required')
        archive=output/'pristine.tar'
        run('pristine-archive',['git','-C',source/'llama.cpp','archive','--format=tar','-o',archive,p['llama_revision'],
            '--','.',':(exclude)models',':(exclude)docs',':(exclude)media'])
        if descriptor(archive)!=original_audit['archive']:
            raise ValueError('original pristine archive required')
        canonical=output/'canonical'
        canonical.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(canonical,filter='data')
        for index,item in enumerate(original_audit['patch_series']):
            name='patch-'+str(index)
            run(name+'-read',['git','-C',source,'show',p['native_revision']+':patches/'+item['name']])
            patch=evidence/(name+'-read.stdout.log')
            if descriptor(patch)!={key:item[key] for key in ('bytes','sha256')}:
                raise ValueError('original patch bytes required')
            run(name+'-apply',['git','apply','--whitespace=nowarn',patch],canonical,
                overrides={'GIT_CEILING_DIRECTORIES':str(output)})
        if len(original_audit['patch_series'])!=19 or source_inventory(canonical)!=original_audit['files']:
            raise ValueError('complete original nineteen-patch source required')
        if descriptor(canonical/PATH)!=ORIGINAL or (canonical/PATH).read_bytes()!=(tools/'fixtures/q8-tile-sgemm.cpp').read_bytes():
            raise ValueError('original SGEMM fixture and canonical source differ')
        candidate=evidence/'sgemm-mx4.cpp'
        candidate.write_bytes(transform((canonical/PATH).read_bytes()))
        if descriptor(candidate)!=p['candidate_source'] or ORIGINAL!=p['original_source']:
            raise ValueError('exact preregistered tile source delta required')
        import difflib
        (evidence/'tile-source.diff').write_text(''.join(difflib.unified_diff((canonical/PATH).read_text().splitlines(True),
            candidate.read_text().splitlines(True),fromfile=PATH,tofile='candidate/'+PATH)))
        project=output/'project'
        project.mkdir()
        shutil.copyfile(tools/'q8_worker_timing_benchmark.cmake',project/'CMakeLists.txt')
        build=output/'build'
        inputs={str(path):descriptor(path) for path in [archive,candidate,project/'CMakeLists.txt',tools/'q8_worker_timing_benchmark.cpp']}
        command=[cmake,'-S',project,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_C_COMPILER='+str(compiler),'-DCMAKE_CXX_COMPILER='+str(compiler),'-DCMAKE_LINKER='+str(linker),
            '-DCMAKE_AR='+str(archive_tool),'-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_FLAGS_INIT:STRING='+FLAG,'-DCMAKE_CXX_FLAGS_INIT:STRING='+FLAG,
            '-DGGML_SOURCE:PATH='+str(canonical),'-DCANDIDATE_SOURCE:FILEPATH='+str(candidate),
            '-DBENCHMARK_SOURCE:FILEPATH='+str(tools/'q8_worker_timing_benchmark.cpp'),'-DGGML_CCACHE=OFF']
        run('configure',command)
        for name in ['CMakeCache.txt','compile_commands.json']:
            shutil.copyfile(build/name,evidence/name)
        cache=(evidence/'CMakeCache.txt').read_text()
        old_cache=baseline('clang-cl')[0]
        for language in ('C','CXX'):
            flag='CMAKE_'+language+'_FLAGS'
            values=cache_value(cache,flag)
            if values.count(FLAG)!=1 or cache_value(cache,flag+'_INIT')!=[FLAG]:
                raise ValueError('one original contraction initializer required')
            values.remove(FLAG)
            if values!=cache_value(old_cache,flag) or cache_value(cache,flag+'_RELEASE')!=cache_value(old_cache,flag+'_RELEASE'):
                raise ValueError('original platform/Release flags required')
            file,=build.glob('CMakeFiles/*/CMake'+language+'Compiler.cmake')
            shutil.copyfile(file,evidence/('compiler-'+language+'.cmake'))
        for name in ['GGML_CPU','GGML_OPENMP','GGML_BACKEND_DL','BUILD_SHARED_LIBS']:
            if field(cache,name)!='OFF': raise ValueError('static base-only GGML control required')
        if Path(field(cache,'CMAKE_LINKER')).resolve(strict=True)!=linker:
            raise ValueError('selected original linker required')
        run('build',[cmake,'--build',build,'--parallel','2','--verbose'],timeout=1800)
        build_log=(evidence/'build.stdout.log').read_text()
        if re.search(r'libomp|libgomp|vcomp\d*\.lib|[-/]f?openmp|\[-Wexceptions\]|non-throwing exception specification', build_log, re.I):
            raise ValueError('OpenMP or unsupported exception-contract evidence in corrected build')
        database=json.loads((evidence/'compile_commands.json').read_text())
        aliases=admit_compiler_paths(compiler,{command_compiler(row['command']) for row in database})
        compiled,objects={},{}
        kernels={}
        for row in database:
            file=Path(row['file']).resolve(strict=True)
            if file.is_relative_to(canonical): expected=original_audit['files'][file.relative_to(canonical).as_posix()]
            elif str(file) in inputs: expected=inputs[str(file)]
            else: raise ValueError('unexpected compiled source')
            if descriptor(file)!=expected or str(file) in compiled: raise ValueError('exact unique compiler inputs required')
            if row['command'].split().count(FLAG)!=1: raise ValueError('all compiler commands must retain contraction off')
            if any(token in row['command'] for token in ['/fp:fast','-ffast-math','-fno-math-errno','/GL','-flto','/openmp','-fopenmp']):
                raise ValueError('unrelated optimization policy changed')
            compiled[str(file)]=expected
            obj=Path(row['output'])
            if not obj.is_absolute(): obj=build/obj
            obj=obj.resolve(strict=True)
            if not obj.is_relative_to(build): raise ValueError('private compiler output required')
            objects[str(obj)]=descriptor(obj)
            for variant,path in [('control',canonical/PATH),('mx4',candidate)]:
                if file==path:
                    expected_symbol='gen2_q8_tile_'+('control' if variant=='control' else 'candidate')+'_sgemm'
                    if row['command'].count('/arch:AVX2')!=1 or ('llamafile_sgemm='+expected_symbol) not in row['command']:
                        raise ValueError('original AVX2 and separate entry aliases required')
                    if variant=='mx4' and row['command'].count('GEN2_Q8_MX4_EXPECTED=1')!=1:
                        raise ValueError('candidate compilation must enforce actual Windows AVX2/F16C')
                    target=evidence/('kernel-'+variant+'.obj')
                    shutil.copyfile(obj,target)
                    run('disasm-'+variant,[dumpbin,'/nologo','/disasm','/symbols',target])
                    if expected_symbol not in (evidence/('disasm-'+variant+'.stdout.log')).read_text():
                        raise ValueError('actual separate kernel entry evidence required')
                    kernels[variant]={'source':str(file),'object':descriptor(target),'entry':expected_symbol,'command':row['command']}
        if set(kernels)!={'control','mx4'}: raise ValueError('both actual kernel translation units required')
        binary=build/'q8_tile_benchmark.exe'
        target=evidence/'q8_tile_benchmark.exe'
        shutil.copyfile(binary,target)
        run('binary-dependencies',[dumpbin,'/nologo','/dependents',target])
        dependencies=(evidence/'binary-dependencies.stdout.log').read_text()
        if re.search(r'libomp|libgomp|vcomp\d*\.dll',dependencies,re.I):
            raise ValueError('OpenMP dependency in standalone corrected control')
        q['build']={'compiler_inputs':compiled,'objects':objects,'kernel_translation_units':kernels,
            'compiler_aliases':aliases,'binary':descriptor(target),'project_inputs':inputs,
            'canonical_files':original_audit['files'],'patch_series':original_audit['patch_series'],'archive':descriptor(archive),
            'candidate':descriptor(candidate),'dependencies':descriptor(evidence/'binary-dependencies.stdout.log'),'compile_database':descriptor(evidence/'compile_commands.json')}
        bound={str(path):descriptor(path) for path in evidence.iterdir() if path.is_file() and path.name!='commands.json'}
        case_bindings={}
        def unchanged():
            frozen()
            if source_inventory(canonical)!=original_audit['files'] or any(descriptor(Path(name))!=value for name,value in inputs.items()):
                raise ValueError('kernel source changed')
            if any(descriptor(Path(name))!=value for name,value in {**bound,**case_bindings}.items()) or any(descriptor(Path(name))!=value for name,value in objects.items()):
                raise ValueError('built evidence changed')
            for name,item in q['toolchain'].items():
                if descriptor(Path(item['path']))!={key:item[key] for key in ('bytes','sha256')}:
                    raise ValueError('build tool changed')
            if admit_compiler_paths(compiler,aliases)!=aliases: raise ValueError('compiler aliases changed')
        reference_outputs=None
        reference_half=None
        for workers in (1,2,4):
            for index,variant in enumerate(ORDER):
                unchanged()
                phase=f'case-{workers}-{index}'
                case={'workers':workers,'index':index,'telemetry_mode':variant,'command_started':False,'support_controls':[]}
                q['cases'].append(case)
                persist()
                for mode,expected in GUARD_CODES.items():
                    unchanged()
                    name=phase+'-guard-'+mode
                    status=run(name,[target,'--support-control',mode],expected_returncode=expected)
                    control=admit_guard(mode,status,(evidence/(name+'.stdout.log')).read_bytes(),(evidence/(name+'.log')).read_bytes())
                    case['support_controls'].append(control)
                    case_bindings.update({str(path):descriptor(path) for path in [evidence/(name+'.stdout.log'),evidence/(name+'.log')]})
                    persist()
                unchanged()
                case['command_started']=True
                persist()
                run(phase,[target,variant,str(workers),evidence/phase])
                half_path=Path(str(evidence/phase)+'.half.f32')
                half_raw=half_path.read_bytes()
                case['half_table']=admit_half_table(half_raw)
                if reference_half is None: reference_half=half_raw
                if reference_half!=half_raw: raise ValueError('all twelve complete half tables must match')
                paths=[Path(str(evidence/phase)+'.'+str(i)+'.'+str(position)+'.f32') for i in range(len(SHAPES)) for position in range(4)]
                outputs=[path.read_bytes() for path in paths]
                case['observed']=validate_output((evidence/(phase+'.stdout.log')).read_text(),variant,workers,outputs)
                case_bindings.update({str(path):descriptor(path) for path in [half_path,*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]})
                persist()
                if reference_outputs is None: reference_outputs=outputs
                if outputs!=reference_outputs: raise ValueError('all worker/variant outputs must be byte-identical')
                case['outputs_bit_exact']=True
                persist()
        unchanged()
        q['timing_comparison']=timing_evidence(q['cases'])
        q['all_1056_outputs_bit_exact']=True
        q['all_12_half_tables_bit_exact']=True
        q['all_36_support_rejections_verified']=True
        q['diagnostic_completed']=True
    except BaseException as error:
        q['error']=repr(error)
        raise
    finally:
        q['commands']=commands
        persist()


if __name__=='__main__': main()
