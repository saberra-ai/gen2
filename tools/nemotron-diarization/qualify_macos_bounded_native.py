"""Fixed 12/60-second public C ABI qualification of the bounded CPU candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import wave


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','producer-zip','producer-metadata','native-reference-zip','native-reference-metadata','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--admit-only',action='store_true')
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo,output = tools.parents[1],args.output.resolve()
    deliveries = [args.producer_zip,args.producer_metadata,args.native_reference_zip,args.native_reference_metadata]
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or output.exists() or args.output.is_symlink()
            or output.is_relative_to(repo) or any(path.resolve().is_relative_to(output) for path in deliveries)):
        parser.error('fresh disjoint Apple Silicon qualification output required')
    raw_protocol = args.protocol.read_bytes()
    p = json.loads(raw_protocol)

    def frozen():
        if args.protocol.read_bytes() != raw_protocol: raise ValueError('bounded native protocol changed')
        for names,base in [(p['tools'],tools),(p['repository_files'],repo)]:
            for name,expected in names.items():
                if descriptor(base / name) != expected: raise ValueError('bounded native source changed: '+name)
    frozen()
    from artifacts import acquire
    from bounded_cpu_metrics import VERSION,derive_helper,compare_bounded_history,validate_failed_loads
    from cpu_thread_metrics import used_dependencies
    from frozen_git_sources import verify_git_sources
    from macos_child_environment import sandboxed_command
    from macos_loaded_libraries import audit_loaded_libraries
    from macos_portable_delivery import admit_delivery
    from package_macos import verify
    from pooled_runtime import BOUNDED_PROFILE,load_profile,profile_provenance
    from qualification_storage import StorageGuard,atomic_json
    from sdk_archive import inventory,restore_archive,write_archive

    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},**p['repository_files']})
    if raw_protocol != subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-bounded-native-v2.protocol.json'],cwd=repo):
        raise ValueError('committed bounded native protocol required')
    lock_path,lock = load_profile(BOUNDED_PROFILE)
    if profile_provenance(lock_path,lock) != p['runtime_provenance']: raise ValueError('bounded runtime provenance differs')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(raw_protocol)
    r = {'schema_version':1,'source_commit':source,'admit_only':args.admit_only,'deliveries_admitted':False,'native_parity_passed':False,'realtime_gate_passed':False,
         'commands':[],'neural_commands_started':[],'loaded_images':{},'comparisons':{},'release_qualified':False,'limits':p['limits']}
    persist = lambda:atomic_json(output / 'qualification.json',r)
    protected,guard = {},None
    env = {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(output / 'home'),'TMPDIR':str(output / 'tmp')+'/','LC_ALL':'C'}

    def unchanged():
        frozen()
        for path,expected in protected.items():
            if inventory(Path(path)) != expected: raise ValueError('bounded native input changed: '+path)

    def run(name,argv,sandbox,*,inference=False):
        unchanged()
        guard.before_model()
        actual,environment = sandboxed_command(argv,sandbox,env)
        row = {'name':name,'argv':actual,'environment':environment,'cwd':str(output),'inference_admitted':inference}
        r['commands'].append(row)
        if inference: r['neural_commands_started'].append(name)
        persist()
        started = time.monotonic()
        with (output / (name+'.stdout.log')).open('xb') as stdout,(output / (name+'.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(actual,cwd=output,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            row['returncode'] = guard.wait(process,row,p['command_timeout_seconds'])
        row['elapsed_seconds'] = time.monotonic()-started
        persist()
        if row['returncode'] != 0: raise ValueError('bounded native command failed: '+name)

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
            raise ValueError('qualified construction and pooled reference deliveries required')
        origin = nq['deliveries']['construction']
        if (origin['id'] != p['reference_producer']['id'] or origin['workflow_run']['head_sha'] != p['reference_producer']['source_commit']
                or origin['digest'] != 'sha256:'+p['reference_producer']['zip']['sha256']):
            raise ValueError('reference construction identity differs')
        r['deliveries_admitted'] = True
        if args.admit_only: return
        sdk,package = output / 'sdk',output / 'work/nemo'
        for stem,key,destination in [('pooled-cpu-sdk','sdk_archive',sdk),('pooled-cpu-runtime','package_archive',package)]:
            files = json.loads((producer / (stem+'-files.json')).read_text())
            if {k:v for k,v in files.items() if k != 'files'} != pq[key]: raise ValueError('producer archive summary differs')
            restore_archive(producer / files['archive'],files,destination)
        manifest = verify(package)
        if manifest['runtime_profile'] != BOUNDED_PROFILE or manifest['runtime_reported_version'] != VERSION or manifest['backend'] != 'cpu':
            raise ValueError('exact bounded CPU package required')
        atomic_json(output / 'package.json',manifest)
        host = {'cpu':subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                'cpu_counts':subprocess.check_output(['/usr/sbin/sysctl','hw.ncpu','hw.logicalcpu','hw.physicalcpu'],text=True),
                'platform':platform.platform(),'python':sys.version,'load_before':os.getloadavg()}
        atomic_json(output / 'host.json',host)
        if host['cpu'] != p['expected_receiver_cpu'] or host['cpu_counts'] != p['expected_cpu_counts']:
            raise ValueError('fixed three-CPU M1 receiver required')
        for path in [producer,reference,sdk,package]: protected[str(path)] = inventory(path)
        helpers,assets,binaries = output / 'helper-source',output / 'work/assets',output / 'work/bin'
        for path in [helpers,assets,binaries]: path.mkdir()
        r['helper_acquisitions'] = {}
        for name,entry in p['helper_inputs'].items():
            guard.before_model()
            target = helpers / name
            target.parent.mkdir(parents=True,exist_ok=True)
            acquire({**entry,'name':target.name},target.parent)
            r['helper_acquisitions'][name] = descriptor(target)
        native_source = helpers / 'native.cpp'
        native_source.write_text(derive_helper((tools / 'native_stream_probe_valid_mel.cpp').read_text()))
        if descriptor(native_source) != p['derived_helper']: raise ValueError('derived helper differs')
        shutil.copy2(tools / 'native_failed_load_probe.cpp',helpers / 'failed.cpp')
        protected[str(helpers)] = inventory(helpers)
        quote = lambda path:json.dumps(str(Path(path).resolve()),ensure_ascii=False)
        compile_policy = output / 'compile.sb'
        compile_policy.write_text('(version 1)(allow default)(deny network*)\n'
            +''.join('(deny file-read* (subpath '+quote(path)+'))\n' for path in [assets,reference])
            +''.join('(deny file-write* (subpath '+quote(path)+'))\n' for path in [repo,helpers,producer,reference,sdk,package])
            +'(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
        control_code = 'import json,sys\nfor path,mode in json.loads(sys.argv[1]):\n try:open(path,mode).close()\n except PermissionError:continue\n raise RuntimeError("protected access allowed: "+path)\nprint("protected accesses denied")\n'
        run('compile-controls',['/usr/bin/python3','-B','-c',control_code,json.dumps([
            [str(reference / 'results/native-12s.json.f32'),'rb'],[str(sdk / 'include/nemo_speech/diar.h'),'wb']])],compile_policy)
        flags = ['-std=c++17','-O2','-Wall','-Wextra','-Werror','-arch','arm64','-march=armv8.2-a+dotprod+fp16','-mmacosx-version-min=26.0']
        for name,path in [('native',native_source),('sha',helpers / 'sha256/sha256.c'),('failed',helpers / 'failed.cpp')]:
            run(name+'-compile',['/usr/bin/clang++',*flags,'-x','c++','-I'+str(sdk / 'include'),'-I'+str(helpers),
                '-MMD','-MF',output / (name+'.d'),'-c',path,'-o',output / (name+'.o')],compile_policy)
        allowed = {str(base / name):{'bytes':row['bytes'],'sha256':row['sha256']}
                   for base in [helpers,sdk] for name,row in inventory(base).items() if row['type'] == 'file'}
        used = set().union(*(used_dependencies((output / (name+'.d')).read_text()) for name in ['native','sha','failed']))
        if not used <= set(allowed) or any(descriptor(path) != allowed[path] for path in used):
            raise ValueError('unadmitted native compiler dependency')
        r['used_compiler_inputs'] = {path:allowed[path] for path in sorted(used)}
        for name,objects in [('native',['native','sha']),('failed',['failed'])]:
            run('link-'+name,['/usr/bin/clang++','-arch','arm64','-mmacosx-version-min=26.0',
                *[output / (obj+'.o') for obj in objects],'-L'+str(package / 'lib'),'-lnemo_speech_asr_c',
                '-Wl,-rpath,@loader_path/../nemo/lib','-o',binaries / name],compile_policy)
        r['helper_archive'] = write_archive(binaries,output,'bounded-native-helpers')
        protected[str(binaries)] = inventory(binaries)
        r['compiler'] = subprocess.check_output(['/usr/bin/clang++','--version'],text=True)
        denied = [repo,helpers,sdk,producer,reference,'/opt/homebrew',*[path.resolve() for path in deliveries]]
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
            [str(repo / 'Cargo.toml'),'rb'],[str(reference / 'results/native-60s.json.f32'),'rb'],
            [str(marker),'rb'],[str(binaries / 'native'),'wb']])],runtime)
        missing = output / 'work/missing-model'
        if missing.exists() or missing.is_symlink(): raise ValueError('negative model path must be absent')
        failed = binaries / 'failed'
        run('failed-loads',['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',failed,VERSION,missing],premodel)
        r['failed_load_control'] = validate_failed_loads(json.loads((output / 'failed-loads.stdout.log').read_text()))
        r['loaded_images']['failed-loads'] = audit_loaded_libraries((output / 'failed-loads.stderr.log').read_text(),package,manifest,failed)
        if [row['name'] for row in r['commands']] != p['preparation_order']: raise ValueError('fixed preparation sequence differs')
        r['acquisitions'] = []
        for entry in p['assets']:
            guard.before_model()
            acquire(entry,assets)
            r['acquisitions'].append({'asset':entry['name'],**descriptor(assets / entry['name'])})
            persist()
        with wave.open(str(assets / p['source_wav']),'rb') as audio:
            if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getnframes()) != (1,2,16000,960000):
                raise ValueError('source WAV geometry differs')
            first = audio.readframes(192000)
        with wave.open(str(assets / 'crop-12s.wav'),'wb') as audio:
            audio.setparams((1,2,16000,0,'NONE','not compressed')); audio.writeframes(first)
        protected[str(assets)] = inventory(assets)
        atomic_json(output / 'protected-inputs.json',protected)
        for seconds,wav in [(12,'crop-12s.wav'),(60,p['source_wav'])]:
            name = 'native-'+str(seconds)+'s'
            executable,target = binaries / 'native',output / 'results' / (name+'.json')
            run(name,['/usr/bin/time','-l','/usr/bin/env','DYLD_PRINT_LIBRARIES=1',executable,
                assets / p['model_name'],assets / wav,'1','-1',target],runtime,inference=True)
            r['loaded_images'][name] = audit_loaded_libraries((output / (name+'.stderr.log')).read_text(),package,manifest,executable)
            value = json.loads(target.read_text())
            expected = reference / 'results' / (name+'.json')
            r['comparisons'][name] = compare_bounded_history(value,Path(str(target)+'.f32').read_bytes(),
                json.loads(expected.read_text()),Path(str(expected)+'.f32').read_bytes())
            persist()
            if r['comparisons'][name]['warm_rtf'] >= p['require_warm_rtf_below']:
                raise ValueError('bounded M1 C ABI still misses realtime: '+name)
        unchanged()
        if ([row['name'] for row in r['commands']] != p['preparation_order']+p['neural_order']
                or r['neural_commands_started'] != p['neural_order']): raise ValueError('fixed two-process sequence differs')
        r['result_files'] = {path.name:descriptor(path) for path in (output / 'results').iterdir()}
        r['load_after'] = os.getloadavg()
        r['native_parity_passed'] = r['realtime_gate_passed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__': main()
