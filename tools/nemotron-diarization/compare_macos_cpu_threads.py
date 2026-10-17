"""Four-versus-three CPU threads, unchanged owned-pool runtime, fixed ABBA order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
import wave


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','producer-zip','producer-metadata','native-reference-zip','native-reference-metadata','native-source','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo,upstream,output = tools.parents[1],args.native_source.resolve(strict=True),args.output.resolve()
    deliveries = [args.producer_zip,args.producer_metadata,args.native_reference_zip,args.native_reference_metadata]
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or output.exists() or args.output.is_symlink()
            or output.is_relative_to(repo) or upstream.is_relative_to(output)
            or any(path.resolve().is_relative_to(output) for path in deliveries)):
        parser.error('fresh disjoint Apple Silicon diagnostic output required')
    protocol_raw = args.protocol.read_bytes()
    p = json.loads(protocol_raw)

    def frozen():
        if args.protocol.read_bytes() != protocol_raw:
            raise ValueError('thread contrast protocol changed')
        for names,base in [(p['tools'],tools),(p['repository_files'],repo),(p['native_inputs'],upstream)]:
            for name,expected in names.items():
                if descriptor(base / name) != expected:
                    raise ValueError('thread contrast source changed: '+name)
    frozen()
    from artifacts import acquire
    from compare_macos_cpu_pools import materialize_headers
    from cpu_thread_metrics import ORDER,compare_thread_trial,used_dependencies,validate_self_test,validate_failure_control
    from frozen_git_sources import verify_git_sources
    from macos_child_environment import sandboxed_command
    from macos_loaded_libraries import audit_loaded_libraries
    from macos_portable_delivery import admit_delivery
    from package_macos import verify
    from qualification_storage import StorageGuard,atomic_json
    from sdk_archive import inventory,restore_archive,write_archive

    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
    if (protocol_raw != subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-cpu-threads.protocol.json'],cwd=repo)
            or p['order'] != ORDER):
        raise ValueError('fixed committed thread contrast required')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_raw)
    r = {'schema_version':1,'source_commit':source,'prepare_only':args.prepare_only,'preparation_completed':False,
         'comparison_completed':False,'commands':[],'neural_commands_started':[],'loaded_images':{},'comparisons':{},
         'release_qualified':False,'limits':p['limits']}
    guard = None
    protected = {}
    persist = lambda:atomic_json(output / 'qualification.json',r)
    env = {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(output / 'home'),'TMPDIR':str(output / 'tmp')+'/','LC_ALL':'C'}

    def unchanged():
        frozen()
        for path,expected in protected.items():
            if inventory(Path(path)) != expected:
                raise ValueError('thread contrast protected inputs changed: '+path)

    def run(name,argv,sandbox,*,cwd=output,inference=False,expected=0):
        unchanged()
        guard.before_model()
        actual,environment = sandboxed_command(argv,sandbox,env)
        row = {'name':name,'argv':actual,'cwd':str(cwd),'environment':environment,
               'inference_admitted':inference,'expected_returncode':expected}
        r['commands'].append(row)
        if inference: r['neural_commands_started'].append(name)
        persist()
        started = time.monotonic()
        with (output / (name+'.stdout.log')).open('xb') as stdout,(output / (name+'.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(actual,cwd=cwd,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            row['returncode'] = guard.wait(process,row,p['command_timeout_seconds'])
        row['elapsed_seconds'] = time.monotonic()-started
        persist()
        if row['returncode'] != expected: raise ValueError('thread contrast command failed: '+name)

    try:
        guard = StorageGuard(output,**p['storage'])
        for name in ['home','tmp','work','results']: (output / name).mkdir()
        r['deliveries'] = {}
        for name in ['producer','native_reference']:
            r['deliveries'][name] = admit_delivery(getattr(args,name+'_zip'),getattr(args,name+'_metadata'),p[name],output / name)
        producer,reference = output / 'producer',output / 'native_reference'
        pq,nq = [json.loads((path / 'qualification.json').read_text()) for path in [producer,reference]]
        if (pq['construction_and_negative_c_api_controls_passed'] is not True or pq['valid_models_loaded'] != 0
                or pq['frontend_or_neural_calls'] != 0 or pq['source_commit'] != p['producer']['source_commit']
                or nq['native_parity_passed'] is not True or nq['source_commit'] != p['native_reference']['source_commit']
                or nq['neural_commands_started'] != ['native-12s','native-60s']
                or any(row['returncode'] != 0 for row in nq['commands'])):
            raise ValueError('complete qualified owned-pool deliveries required')
        origin = nq['deliveries']['construction']
        if (origin['id'] != p['producer']['id'] or origin['digest'] != 'sha256:'+p['producer']['zip']['sha256']
                or origin['workflow_run']['head_sha'] != p['producer']['source_commit']):
            raise ValueError('native reference construction differs')
        sdk,package = output / 'sdk',output / 'work/nemo'
        for stem,key,destination in [('pooled-cpu-sdk','sdk_archive',sdk),('pooled-cpu-runtime','package_archive',package)]:
            files = json.loads((producer / (stem+'-files.json')).read_text())
            if {k:v for k,v in files.items() if k != 'files'} != pq[key]:
                raise ValueError('qualified archive declaration differs')
            restore_archive(producer / files['archive'],files,destination)
        manifest = verify(package)
        if manifest['runtime_reported_version'] != p['runtime_version'] or manifest['backend'] != 'cpu':
            raise ValueError('exact owned-pool CPU package required')
        atomic_json(output / 'package.json',manifest)
        for path in [producer,reference,sdk,package]: protected[str(path)] = inventory(path)
        host = {'cpu':subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                'cpu_counts':subprocess.check_output(['/usr/sbin/sysctl','hw.ncpu','hw.logicalcpu','hw.physicalcpu'],text=True),
                'platform':platform.platform(),'python':sys.version,'load_before':os.getloadavg()}
        atomic_json(output / 'host.json',host)
        if not args.prepare_only and (host['cpu'] != p['expected_receiver_cpu'] or host['cpu_counts'] != p['expected_cpu_counts']):
            raise ValueError('neural contrast requires the fixed three-CPU M1 receiver')
        assets,binary = output / 'work/assets',output / 'work/bin'
        assets.mkdir(); binary.mkdir()
        quote = lambda path:json.dumps(str(Path(path).resolve()),ensure_ascii=False)
        build_policy = output / 'compile.sb'
        build_policy.write_text('(version 1)(allow default)(deny network*)\n'
            +''.join('(deny file-read* (subpath '+quote(path)+'))\n' for path in [assets,reference])
            +''.join('(deny file-write* (subpath '+quote(path)+'))\n' for path in [repo,upstream,producer,reference,sdk,package])
            +'(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
        control_code = 'import json,sys\nfor path,mode in json.loads(sys.argv[1]):\n try:open(path,mode).close()\n except PermissionError:continue\n raise RuntimeError("protected access allowed: "+path)\nprint("protected accesses denied")\n'
        run('compile-controls',['/usr/bin/python3','-B','-c',control_code,json.dumps([
            [str(reference / 'results/native-12s.json.f32'),'rb'],[str(sdk / 'include/nemo_speech/asr.h'),'wb']])],build_policy)
        native = output / 'helper-native'
        native.mkdir()
        for name in p['native_inputs']:
            target = native / name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(upstream / name,target)
        run('owned-pool-header-patch',['/usr/bin/git','-c','core.hooksPath=/dev/null','apply','--whitespace=nowarn',
            tools / 'patches/persistent-cpu-pool.patch'],build_policy,cwd=native)
        for name,expected in p['helper_native_expected'].items():
            if descriptor(native / name) != expected: raise ValueError('owned-pool helper source differs: '+name)
        protected[str(native)] = inventory(native)
        headers = output / 'ggml-headers'
        audit = json.loads((producer / 'build/llama-materialized-audit.json').read_text())
        def apply_patch(index,path):
            run('ggml-header-patch-'+str(index),['/usr/bin/git','-c','core.hooksPath=/dev/null','apply',
                '--include=ggml/include/*','--allow-empty','--whitespace=nowarn',path],build_policy,cwd=headers)
        r['ggml_headers'] = materialize_headers(producer / 'build/pristine-llama.tar',native,audit,headers,apply_patch)
        protected[str(headers)] = inventory(headers)
        substitutions = {'@native@':str(native),'@headers@':str(headers),'@sdk@':str(sdk),'@tools@':str(tools),'@output@':str(output)}
        for command in p['compile_commands']:
            argv = command['argv']
            for key,value in substitutions.items(): argv = [arg.replace(key,value) for arg in argv]
            run(command['name'],argv,build_policy)
        allowed = {str(tools / 'native_thread_probe.cpp'):descriptor(tools / 'native_thread_probe.cpp')}
        for path in [native,headers,sdk]:
            allowed.update({str(path / name):{'bytes':row['bytes'],'sha256':row['sha256']} for name,row in inventory(path).items() if row['type'] == 'file'})
        used = set().union(*(used_dependencies((output / (name+'.d')).read_text()) for name in ['probe','backend','logging']))
        if not used <= set(allowed) or any(descriptor(path) != allowed[path] for path in used):
            raise ValueError('compiler used an unadmitted source/header')
        r['used_compiler_inputs'] = {path:allowed[path] for path in sorted(used)}
        executable = binary / 'native-thread-probe'
        run('link',['/usr/bin/clang++','-arch','arm64','-mmacosx-version-min=26.0',
            *[output / (name+'.o') for name in ['probe','backend','logging']],
            *[package / 'lib' / name for name in p['link_libraries']],'-Wl,-rpath,@loader_path/../nemo/lib','-o',executable],build_policy)
        r['binary'] = descriptor(executable)
        r['compiler'] = subprocess.check_output(['/usr/bin/clang++','--version'],text=True)
        r['helper_archive'] = write_archive(binary,output,'thread-helpers')
        protected[str(binary)] = inventory(binary)
        denied = [repo,upstream,native,headers,sdk,producer,reference,'/opt/homebrew',*[path.resolve() for path in deliveries]]
        runtime = output / 'runtime.sb'
        runtime.write_text('(version 1)(allow default)(deny network*)(deny file-write*)\n'
            +'(allow file-write* '+' '.join('(subpath '+quote(output / name)+')' for name in ['results','home','tmp'])+' (literal "/dev/null"))\n'
            +'(deny file-read-data (subpath '+quote(output / 'results')+'))\n'
            +''.join('(deny file-read* (subpath '+quote(path)+'))\n' for path in denied))
        premodel = output / 'premodel.sb'
        premodel.write_text(runtime.read_text()+'(deny file-read* (subpath '+quote(assets)+'))\n'
            +'(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
        marker = output / 'results/prior.txt'; marker.write_text('not a model')
        run('runtime-controls',['/usr/bin/python3','-B','-c',control_code,json.dumps([
            [str(repo / 'Cargo.toml'),'rb'],[str(reference / 'results/native-12s.json.f32'),'rb'],[str(marker),'rb'],[str(executable),'wb']])],runtime)
        run('lifetime-failure-control',['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',executable,'--self-test-failure-control'],premodel,expected=1)
        r['failure_control'] = validate_failure_control(json.loads((output / 'lifetime-failure-control.stdout.log').read_text()))
        if 'injected lifetime evidence control' not in (output / 'lifetime-failure-control.stderr.log').read_text():
            raise ValueError('expected lifetime failure marker missing')
        r['loaded_images']['lifetime-failure-control'] = audit_loaded_libraries((output / 'lifetime-failure-control.stderr.log').read_text(),package,manifest,executable)
        run('lifetime-self-test',['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',executable,'--self-test'],premodel)
        r['self_test'] = validate_self_test(json.loads((output / 'lifetime-self-test.stdout.log').read_text()))
        r['loaded_images']['lifetime-self-test'] = audit_loaded_libraries((output / 'lifetime-self-test.stderr.log').read_text(),package,manifest,executable)
        atomic_json(output / 'protected-inputs.json',protected)
        unchanged()
        if [row['name'] for row in r['commands']] != p['preparation_order']:
            raise ValueError('fixed model-free preparation sequence differs')
        r['preparation_completed'] = True
        if args.prepare_only: return
        r['acquisitions'] = []
        for entry in p['assets']:
            guard.before_model()
            acquire(entry,assets)
            r['acquisitions'].append({'asset':entry['name'],**descriptor(assets / entry['name'])})
            persist()
        with wave.open(str(assets / p['source_wav']),'rb') as audio:
            if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getnframes()) != (1,2,16000,960000):
                raise ValueError('source WAV geometry differs')
            pcm = b''.join(struct.pack('<f',v[0]/32768.) for v in struct.iter_unpack('<h',audio.readframes(192000)))
        pcm_path = assets / 'input.f32'; pcm_path.write_bytes(pcm)
        if descriptor(pcm_path)['sha256'] != p['pcm_f32le_sha256']: raise ValueError('actual normalized PCM differs')
        protected[str(assets)] = inventory(assets)
        atomic_json(output / 'protected-inputs.json',protected)
        expected_path = reference / 'results/native-12s.json'
        expected = json.loads(expected_path.read_text())
        golden = Path(str(expected_path)+'.f32').read_bytes()
        for index,threads in enumerate(ORDER,1):
            name = str(index)+'-'+str(threads)
            target = output / 'results' / (name+'.json')
            run(name,['/usr/bin/time','-l','/usr/bin/env','DYLD_PRINT_LIBRARIES=1',executable,str(threads),
                assets / p['model_name'],pcm_path,target],runtime,inference=True)
            r['loaded_images'][name] = audit_loaded_libraries((output / (name+'.stderr.log')).read_text(),package,manifest,executable)
            value = json.loads(target.read_text())
            r['comparisons'][name] = compare_thread_trial(value,Path(str(target)+'.f32').read_bytes(),expected,golden,threads)
            persist()
        unchanged()
        r['result_files'] = {path.name:descriptor(path) for path in (output / 'results').iterdir()}
        r['load_after'] = os.getloadavg()
        if ([row['name'] for row in r['commands']] != p['preparation_order']+p['neural_order']
                or r['neural_commands_started'] != p['neural_order']):
            raise ValueError('fixed ABBA process sequence differs')
        r['comparison_completed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__': main()
